# -*- coding: utf-8 -*-
"""
RWTH48 V4.2 | 冻结高覆盖率 RefAh Event 并重提 HI
=================================================

目的
----
在 V4.1 Event coverage scan 已确认覆盖率后，只对最终冻结的高覆盖率 Event 节点重提 HI，
不重新读取原始 ZIP、不重新清洗、不重新构造 SOH。

冻结主 Event：
    RefAhFraction = 0.05 / 0.09 / 0.13

理由：
- 0.05：早期节点，覆盖率接近 100%
- 0.09：中期节点，覆盖率接近 100%
- 0.13：晚期节点；相比 0.14 只少约 1% reference Ah，但覆盖率显著更稳
- RWTH-028：整只排除（BOL/SOH 轨迹异常）
- DeltaV：V4.1 已证实没有任何节点通过正式 coverage gate，因此不再提取

重要：
- 直接复用 V4 保存的 clean_timeseries_parts
- 直接动态加载原 V4 脚本中的 prefix_features / event_cut_index_refah，确保 HI 定义完全一致
- SOH 仅在 HI 提取完成后 merge，绝不参与 prefix/Event 构造
- 输出全量 Event 表 + Label-ready 严格训练表 + coverage/acceptance audit

推荐运行：
    python "src/02_build_refah_features.py" --n-jobs 4

也可显式指定：
    python "src/02_build_refah_features.py" ^
      --v4-root "workspace/RWTH48_全流程输出_V4" ^
      --v4-script "src/01_preprocess_soh_labels.py" ^
      --n-jobs 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. 冻结配置
# =============================================================================

DEFAULT_BASE = (Path(__file__).resolve().parents[1] / "workspace")
DEFAULT_V4_ROOT_CANDIDATES = (
    DEFAULT_BASE / "RWTH48_全流程输出_V4",
    DEFAULT_BASE / "RWTH48_全流程输出",
)
DEFAULT_OUTPUT_NAME = "RWTH48_V4_2_冻结RefAh_Event_HI"

FROZEN_REFAH_NODES: Tuple[float, ...] = (0.05, 0.09, 0.13)
EXCLUDED_BATTERIES = {"RWTH-028"}

# Acceptance gate：这里故意比 V4.1 的最低 PASS 更稳健
MIN_GLOBAL_MODEL_COVERAGE = 0.97
MIN_GLOBAL_PHYSICAL_COVERAGE = 0.97
MIN_BATTERY_COVERAGE = 0.95
MIN_BATTERY_RATIO = 0.95
MIN_WORST_SOH_BIN_COVERAGE = 0.95

SOH_BINS = [-np.inf, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, np.inf]
SOH_BIN_LABELS = ["<0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00", ">=1.00"]

META_COLS = {
    "Battery", "GlobalCycle", "Scope", "NodeType", "NodeValue", "ReferenceCapacity_Ah",
    "SOH_Target_SameCycle", "SOH_IsMeasuredCheckUp", "SOH_LabelUsable", "SOH_LabelSource",
    "EventReachable", "PrefixFeatureValid", "MissingFeatureCount",
}


# =============================================================================
# 1. 基础工具
# =============================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    comp = "gzip" if str(path).lower().endswith(".gz") else None
    df.to_csv(path, index=False, encoding="utf-8-sig", compression=comp)


def save_json(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def normalize_battery(x: object) -> str:
    s = str(x).strip()
    if s.startswith("RWTH-"):
        try:
            return f"RWTH-{int(s.split('-')[-1]):03d}"
        except Exception:
            return s
    if s.isdigit():
        return f"RWTH-{int(s):03d}"
    return s


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RWTH48 V4.2 frozen RefAh Event HI extractor")
    p.add_argument("--v4-root", type=str, default="", help="V4 输出根目录；留空自动探测。")
    p.add_argument("--v4-script", type=str, default="", help="原 V4 完整脚本；留空自动搜索。")
    p.add_argument("--output-root", type=str, default="", help="V4.2 输出目录；留空放在 Rawdata1 下。")
    p.add_argument("--n-jobs", type=int, default=4, help="并行电池数。Windows 推荐 2~4。")
    p.add_argument("--force", action="store_true", help="忽略已有 per-battery 完成文件重新提取。")
    p.add_argument("--only-battery", type=str, default="", help="只跑一个电池，例如 RWTH-002 或 002。")
    if argv is not None:
        return p.parse_args(list(argv))
    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        args, _ = p.parse_known_args()
        return args
    return p.parse_args()


def detect_v4_root(user_value: str) -> Path:
    if user_value:
        p = Path(user_value)
        if not p.exists():
            raise FileNotFoundError(f"V4输出目录不存在：{p}")
        return p
    for p in DEFAULT_V4_ROOT_CANDIDATES:
        if p.exists() and (p / "02_batteries").exists():
            return p
    raise FileNotFoundError(
        "未自动找到 V4 输出目录。请用 --v4-root 显式指定。\n"
        + "\n".join(str(x) for x in DEFAULT_V4_ROOT_CANDIDATES)
    )


def detect_v4_script(user_value: str) -> Path:
    if user_value:
        p = Path(user_value)
        if not p.exists():
            raise FileNotFoundError(f"V4脚本不存在：{p}")
        return p

    p = Path(__file__).resolve().with_name("01_preprocess_soh_labels.py")
    if not p.is_file():
        raise FileNotFoundError(f"Missing preprocessing script: {p}; use --v4-script")
    return p


def load_v4_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("rwth48_v4_source", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 V4 模块：{script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    required = [
        "prefix_features",
        "event_cut_index_refah",
        "collapse_duplicate_timestamps",
        "get_discharge_cycle",
        "MIN_PREFIX_ROWS",
        "MIN_DCH_DURATION_S",
    ]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f"V4脚本缺少预期对象：{missing}")
    return mod


def find_global_file(v4_root: Path, filename: str) -> Path:
    candidates = [v4_root / "03_global" / filename, v4_root / filename]
    for p in candidates:
        if p.exists():
            return p
    hits = list(v4_root.rglob(filename))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"V4输出中找不到：{filename}")


def find_battery_dir(v4_root: Path, battery: str) -> Optional[Path]:
    base = v4_root / "02_batteries"
    if not base.exists():
        return None
    variants = [
        base / f"Cell_{battery}",
        base / battery,
        base / f"Cell_{battery.split('-')[-1]}",
    ]
    for p in variants:
        if p.exists():
            return p
    # 宽松匹配
    tail = battery.split("-")[-1]
    for p in base.iterdir():
        if p.is_dir() and tail in p.name:
            return p
    return None


# =============================================================================
# 2. 输入元数据
# =============================================================================

def load_quality_and_labels(v4_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    qpath = find_global_file(v4_root, "00_battery_quality_summary.csv")
    lpath = find_global_file(v4_root, "02_all_cycle_soh_labels.csv.gz")
    q = pd.read_csv(qpath, low_memory=False)
    labels = pd.read_csv(lpath, low_memory=False, compression="infer")

    if "Battery" not in q or "Battery" not in labels:
        raise RuntimeError("quality/labels 缺少 Battery")
    q["Battery"] = q["Battery"].map(normalize_battery)
    labels["Battery"] = labels["Battery"].map(normalize_battery)
    labels["GlobalCycle"] = pd.to_numeric(labels["GlobalCycle"], errors="coerce")
    labels = labels[labels["GlobalCycle"].notna()].copy()
    labels["GlobalCycle"] = labels["GlobalCycle"].astype(int)

    # 整只排除异常电池
    q = q[~q["Battery"].isin(EXCLUDED_BATTERIES)].copy()
    labels = labels[~labels["Battery"].isin(EXCLUDED_BATTERIES)].copy()

    # 若质量门存在，则只使用 PASS 电池
    if "QualityGatePASS" in q.columns:
        q = q[pd.to_numeric(q["QualityGatePASS"], errors="coerce").fillna(0).astype(int).eq(1)].copy()

    if q["Battery"].duplicated().any():
        raise RuntimeError("quality summary 中 Battery 重复")
    if labels.duplicated(["Battery", "GlobalCycle"]).any():
        dup = labels[labels.duplicated(["Battery", "GlobalCycle"], keep=False)].head(20)
        raise RuntimeError(f"SOH labels 存在重复 Battery+GlobalCycle：\n{dup}")

    return q.reset_index(drop=True), labels.reset_index(drop=True)


# =============================================================================
# 3. 单电池冻结 Event HI 提取
# =============================================================================

def read_clean_parts_exact(bdir: Path) -> pd.DataFrame:
    pdir = bdir / "clean_timeseries_parts"
    parts = sorted(pdir.glob("*_TBA_Zyk_clean.csv.gz")) if pdir.exists() else []
    if not parts:
        raise FileNotFoundError(f"缺少 clean_timeseries_parts：{bdir}")

    needed = [
        "Timestamp", "GlobalCycle", "State",
        "Voltage_V_Clean", "Current_A_Clean", "Temperature_C_Clean",
        "Voltage_V", "Current_A", "Temperature_C",
        "AnyHardInvalidFlag", "AnySpikeFlag",
    ]
    arr = []
    for p in parts:
        header = pd.read_csv(p, nrows=0, compression="infer").columns.tolist()
        usecols = [c for c in needed if c in header]
        required = {"Timestamp", "GlobalCycle", "State"}
        if not required.issubset(set(usecols)):
            raise RuntimeError(f"{p} 缺少必要列；现有={header}")
        d = pd.read_csv(p, usecols=usecols, compression="infer", low_memory=False)
        arr.append(d)
    d = pd.concat(arr, ignore_index=True)
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
    d["GlobalCycle"] = pd.to_numeric(d["GlobalCycle"], errors="coerce")
    d = d[d["Timestamp"].notna() & d["GlobalCycle"].notna()].copy()
    d["GlobalCycle"] = d["GlobalCycle"].astype(int)
    d = d.sort_values(["GlobalCycle", "Timestamp"], kind="mergesort").reset_index(drop=True)
    return d


def process_battery(
    battery: str,
    ref_capacity_ah: float,
    v4_root: Path,
    out_root: Path,
    v4,
    force: bool,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    bsave = ensure_dir(out_root / "01_batteries" / battery)
    done = bsave / "99_COMPLETE.json"
    out_csv = bsave / "01_frozen_refah_event_features.csv.gz"
    cov_csv = bsave / "02_frozen_refah_coverage.csv"

    if done.exists() and out_csv.exists() and cov_csv.exists() and not force:
        try:
            meta = json.loads(done.read_text(encoding="utf-8"))
            meta["Status"] = "SKIP_COMPLETE"
            return meta
        except Exception:
            pass

    bdir = find_battery_dir(v4_root, battery)
    if bdir is None:
        meta = {"Battery": battery, "Status": "FAIL_NO_BATTERY_DIR", "Elapsed_s": time.perf_counter() - t0}
        save_json(meta, done)
        return meta

    try:
        d = read_clean_parts_exact(bdir)
        cycles = sorted(pd.to_numeric(d["GlobalCycle"], errors="coerce").dropna().astype(int).unique())
        event_rows: List[Dict[str, object]] = []
        coverage_rows: List[Dict[str, object]] = []
        valid_dch_cycles = 0

        for cyc in cycles:
            g = v4.get_discharge_cycle(d, int(cyc))
            if len(g) < int(v4.MIN_PREFIX_ROWS):
                continue
            t = pd.to_datetime(g["Timestamp"], errors="coerce")
            dur = (t.max() - t.min()).total_seconds()
            if not np.isfinite(dur) or dur < float(v4.MIN_DCH_DURATION_S):
                continue
            valid_dch_cycles += 1

            p_collapsed = v4.collapse_duplicate_timestamps(g)
            for node in FROZEN_REFAH_NODES:
                idx = v4.event_cut_index_refah(g, float(ref_capacity_ah), float(node))
                reachable = int(idx is not None)
                coverage_rows.append({
                    "Battery": battery,
                    "GlobalCycle": int(cyc),
                    "EventFamily": "RefAhFraction",
                    "Node": float(node),
                    "Reachable": reachable,
                })
                if idx is None:
                    continue
                pref = p_collapsed.iloc[: int(idx) + 1].copy()
                feat = v4.prefix_features(pref, float(ref_capacity_ah))
                if not feat:
                    continue
                event_rows.append({
                    "Battery": battery,
                    "GlobalCycle": int(cyc),
                    "Scope": "Discharge_Event_RefAhFraction_FrozenV4p2",
                    "NodeType": "RefAhFraction",
                    "NodeValue": float(node),
                    "ReferenceCapacity_Ah": float(ref_capacity_ah),
                    **feat,
                })

        ev = pd.DataFrame(event_rows)
        cov = pd.DataFrame(coverage_rows)
        save_csv(ev, out_csv)
        save_csv(cov, cov_csv)

        meta = {
            "Battery": battery,
            "Status": "PASS",
            "ReferenceCapacity_Ah": float(ref_capacity_ah),
            "ValidDischargeCycles": int(valid_dch_cycles),
            "EventRows": int(len(ev)),
            "CoverageRows": int(len(cov)),
            "Elapsed_s": time.perf_counter() - t0,
        }
        save_json(meta, done)
        return meta
    except Exception as exc:
        meta = {
            "Battery": battery,
            "Status": "FAIL",
            "Error": repr(exc),
            "Elapsed_s": time.perf_counter() - t0,
        }
        save_json(meta, done)
        return meta


# =============================================================================
# 4. 汇总 + 标签后置 merge + 审计
# =============================================================================

def collect_per_battery(out_root: Path, batteries: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evs, covs, metas = [], [], []
    for b in batteries:
        bdir = out_root / "01_batteries" / b
        ep = bdir / "01_frozen_refah_event_features.csv.gz"
        cp = bdir / "02_frozen_refah_coverage.csv"
        jp = bdir / "99_COMPLETE.json"
        if ep.exists():
            evs.append(pd.read_csv(ep, low_memory=False, compression="infer"))
        if cp.exists():
            covs.append(pd.read_csv(cp, low_memory=False))
        if jp.exists():
            try:
                metas.append(json.loads(jp.read_text(encoding="utf-8")))
            except Exception:
                pass
    return (
        pd.concat(evs, ignore_index=True) if evs else pd.DataFrame(),
        pd.concat(covs, ignore_index=True) if covs else pd.DataFrame(),
        pd.DataFrame(metas),
    )


def merge_labels_after_features(event: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if event.empty:
        return event
    label_cols = [
        c for c in [
            "Battery", "GlobalCycle", "SOH_Target_SameCycle", "SOH_IsMeasuredCheckUp",
            "SOH_LabelUsable", "SOH_LabelSource"
        ] if c in labels.columns
    ]
    out = event.merge(labels[label_cols], on=["Battery", "GlobalCycle"], how="left", validate="many_to_one")
    return out


def add_missing_feature_count(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if df.empty:
        return df, []
    feature_cols = [
        c for c in df.columns
        if c not in META_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    out = df.copy()
    out["MissingFeatureCount"] = out[feature_cols].isna().sum(axis=1) if feature_cols else 0
    return out, feature_cols


def coverage_audit(
    cov: pd.DataFrame,
    labels: pd.DataFrame,
    valid_cycle_keys: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cov.empty:
        raise RuntimeError("coverage为空")

    # Physical denominator：本次实际进入 coverage 的合法 DCH cycle
    phys_keys = cov[["Battery", "GlobalCycle"]].drop_duplicates()

    lab = labels.copy()
    if "SOH_LabelUsable" in lab.columns:
        usable = pd.to_numeric(lab["SOH_LabelUsable"], errors="coerce").fillna(0).astype(int).eq(1)
    else:
        usable = lab["SOH_Target_SameCycle"].notna()
    model_keys = lab[usable & lab["SOH_Target_SameCycle"].notna()][["Battery", "GlobalCycle", "SOH_Target_SameCycle"]].copy()
    model_keys = model_keys.merge(phys_keys, on=["Battery", "GlobalCycle"], how="inner")

    rows = []
    bat_rows = []
    soh_rows = []
    for node in FROZEN_REFAH_NODES:
        q = cov[np.isclose(pd.to_numeric(cov["Node"], errors="coerce"), node)].copy()
        q["Reachable"] = pd.to_numeric(q["Reachable"], errors="coerce").fillna(0).astype(int)

        # physical
        p = phys_keys.merge(q[["Battery", "GlobalCycle", "Reachable"]], on=["Battery", "GlobalCycle"], how="left")
        p["Reachable"] = p["Reachable"].fillna(0).astype(int)
        pcov = float(p["Reachable"].mean()) if len(p) else np.nan

        # model
        m = model_keys.merge(q[["Battery", "GlobalCycle", "Reachable"]], on=["Battery", "GlobalCycle"], how="left")
        m["Reachable"] = m["Reachable"].fillna(0).astype(int)
        mcov = float(m["Reachable"].mean()) if len(m) else np.nan

        per_b = m.groupby("Battery")["Reachable"].agg(["size", "sum", "mean"]).reset_index()
        per_b.columns = ["Battery", "EligibleCycles", "ReachableCycles", "Coverage"]
        per_b["Node"] = node
        bat_rows.append(per_b)

        b_ratio = float((per_b["Coverage"] >= MIN_BATTERY_COVERAGE).mean()) if len(per_b) else np.nan
        b_min = float(per_b["Coverage"].min()) if len(per_b) else np.nan

        mm = m.copy()
        mm["SOH_Bin"] = pd.cut(
            pd.to_numeric(mm["SOH_Target_SameCycle"], errors="coerce"),
            bins=SOH_BINS, labels=SOH_BIN_LABELS, right=False,
        )
        sg = mm.groupby("SOH_Bin", observed=False)["Reachable"].agg(["size", "sum", "mean"]).reset_index()
        sg.columns = ["SOH_Bin", "EligibleCycles", "ReachableCycles", "Coverage"]
        sg["Node"] = node
        soh_rows.append(sg)
        finite_cov = pd.to_numeric(sg.loc[sg["EligibleCycles"] > 0, "Coverage"], errors="coerce").dropna()
        worst_soh = float(finite_cov.min()) if len(finite_cov) else np.nan

        pass_gate = int(
            np.isfinite(mcov) and mcov >= MIN_GLOBAL_MODEL_COVERAGE
            and np.isfinite(pcov) and pcov >= MIN_GLOBAL_PHYSICAL_COVERAGE
            and np.isfinite(b_ratio) and b_ratio >= MIN_BATTERY_RATIO
            and np.isfinite(b_min) and b_min >= MIN_BATTERY_COVERAGE
            and np.isfinite(worst_soh) and worst_soh >= MIN_WORST_SOH_BIN_COVERAGE
        )
        rows.append({
            "EventFamily": "RefAhFraction",
            "Node": node,
            "PhysicalEligibleCycles": len(p),
            "PhysicalReachableCycles": int(p["Reachable"].sum()),
            "PhysicalCoverage": pcov,
            "ModelEligibleCycles": len(m),
            "ModelReachableCycles": int(m["Reachable"].sum()),
            "ModelCoverage": mcov,
            "BatteryCount": int(per_b["Battery"].nunique()),
            "BatteryCoverage_Min": b_min,
            f"BatteryRatio_CoverageGE{int(MIN_BATTERY_COVERAGE*100)}": b_ratio,
            "WorstSOHBinCoverage": worst_soh,
            "FinalAcceptancePASS": pass_gate,
        })

    return pd.DataFrame(rows), pd.concat(bat_rows, ignore_index=True), pd.concat(soh_rows, ignore_index=True)


def feature_registry_from_event(event: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        s = pd.to_numeric(event[c], errors="coerce")
        rows.append({
            "Feature": c,
            "Rows": len(event),
            "Missing": int(s.isna().sum()),
            "MissingRate": float(s.isna().mean()),
            "UniqueNonNA": int(s.nunique(dropna=True)),
            "ConstantFeature": int(s.nunique(dropna=True) <= 1),
            "CausalPrefixLegal": 1,
            "Note": "Exact V4 prefix_features; SOH merged only after feature extraction.",
        })
    return pd.DataFrame(rows).sort_values(["ConstantFeature", "MissingRate", "Feature"], ascending=[False, False, True])


# =============================================================================
# 5. main
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    t_all = time.perf_counter()

    v4_root = detect_v4_root(args.v4_root)
    v4_script = detect_v4_script(args.v4_script)
    out_root = ensure_dir(Path(args.output_root) if args.output_root else DEFAULT_BASE / DEFAULT_OUTPUT_NAME)
    v4 = load_v4_module(v4_script)

    q, labels = load_quality_and_labels(v4_root)
    if "ReferenceCapacity_Ah" not in q.columns:
        raise RuntimeError("00_battery_quality_summary.csv 缺少 ReferenceCapacity_Ah")

    batteries = sorted(q["Battery"].astype(str).unique())
    if args.only_battery:
        only = normalize_battery(args.only_battery)
        batteries = [b for b in batteries if b == only]
        if not batteries:
            raise RuntimeError(f"指定电池不在可用列表：{only}")

    print("=" * 110)
    print("RWTH48 V4.2 | 冻结高覆盖率 RefAh Event HI")
    print("V4 ROOT   :", v4_root)
    print("V4 SCRIPT :", v4_script)
    print("OUTPUT    :", out_root)
    print("BATTERIES :", len(batteries), "| excluded:", sorted(EXCLUDED_BATTERIES))
    print("NODES     :", FROZEN_REFAH_NODES)
    print("N_JOBS    :", max(1, int(args.n_jobs)))
    print("=" * 110)

    frozen_def = {
        "Task": "RWTH48 V4.2 Frozen RefAh Event HI",
        "EventFamily": "RefAhFraction",
        "FrozenNodes": list(FROZEN_REFAH_NODES),
        "ExcludedBatteries": sorted(EXCLUDED_BATTERIES),
        "DeltaV": "DROPPED after V4.1 coverage scan",
        "FeatureDefinitionSource": str(v4_script),
        "Causality": "Frozen BOL/early reference capacity only; no current-cycle final capacity/duration; SOH merged after HI extraction.",
        "Acceptance": {
            "MinGlobalModelCoverage": MIN_GLOBAL_MODEL_COVERAGE,
            "MinGlobalPhysicalCoverage": MIN_GLOBAL_PHYSICAL_COVERAGE,
            "MinBatteryCoverage": MIN_BATTERY_COVERAGE,
            "MinBatteryRatio": MIN_BATTERY_RATIO,
            "MinWorstSOHBinCoverage": MIN_WORST_SOH_BIN_COVERAGE,
        },
    }
    save_json(frozen_def, out_root / "00_frozen_event_definition.json")

    ref_map = q.set_index("Battery")["ReferenceCapacity_Ah"].to_dict()
    statuses = []
    nj = max(1, int(args.n_jobs))

    if nj == 1:
        for i, b in enumerate(batteries, 1):
            print(f"[{i}/{len(batteries)}] {b}")
            st = process_battery(b, float(ref_map[b]), v4_root, out_root, v4, args.force)
            statuses.append(st)
            print("   ", st.get("Status"), "EventRows=", st.get("EventRows", ""), "min=", round(float(st.get("Elapsed_s", 0))/60, 2))
    else:
        # ThreadPool：避免 Windows ProcessPool 对动态 import 模块的 pickle 问题；每个线程处理不同电池。
        with ThreadPoolExecutor(max_workers=nj) as ex:
            futs = {
                ex.submit(process_battery, b, float(ref_map[b]), v4_root, out_root, v4, args.force): b
                for b in batteries
            }
            done_n = 0
            for fut in as_completed(futs):
                b = futs[fut]
                try:
                    st = fut.result()
                except Exception as exc:
                    st = {"Battery": b, "Status": "FAIL_FUTURE", "Error": repr(exc)}
                statuses.append(st)
                done_n += 1
                print(f"[{done_n}/{len(batteries)}] {b}: {st.get('Status')} | EventRows={st.get('EventRows','')}")

    status_df = pd.DataFrame(statuses)
    save_csv(status_df, out_root / "02_battery_processing_status.csv")

    bad = status_df[~status_df["Status"].astype(str).isin(["PASS", "SKIP_COMPLETE"])] if not status_df.empty else status_df
    if not bad.empty:
        print("\n[ERROR] 存在失败电池：")
        print(bad.to_string(index=False))
        raise RuntimeError("V4.2 有电池提取失败；已写 02_battery_processing_status.csv")

    event_raw, coverage, meta_collected = collect_per_battery(out_root, batteries)
    if event_raw.empty or coverage.empty:
        raise RuntimeError("汇总后 event/coverage 为空")

    # feature 唯一性审计：同 Battery+Cycle+Node 不允许重复
    key = ["Battery", "GlobalCycle", "NodeValue"]
    dup = event_raw.duplicated(key, keep=False)
    if dup.any():
        save_csv(event_raw.loc[dup].sort_values(key), out_root / "98_DUPLICATE_EVENT_KEYS.csv")
        raise RuntimeError("发现重复 Battery+GlobalCycle+NodeValue；已输出 98_DUPLICATE_EVENT_KEYS.csv")

    # SOH 标签只在此处后置 merge
    event = merge_labels_after_features(event_raw, labels)
    event, feature_cols = add_missing_feature_count(event)

    # Label-ready：真正可用于监督建模的目标样本；不因为少数可选 HI NaN 而整行删除
    usable = pd.to_numeric(event.get("SOH_LabelUsable", 0), errors="coerce").fillna(0).astype(int).eq(1)
    target_ok = pd.to_numeric(event.get("SOH_Target_SameCycle"), errors="coerce").notna()
    label_ready = event[usable & target_ok].copy()

    # coverage denominator 必须使用本次真实合法 DCH cycles
    valid_keys = coverage[["Battery", "GlobalCycle"]].drop_duplicates()
    cov_summary, cov_battery, cov_soh = coverage_audit(coverage, labels, valid_keys)

    registry = feature_registry_from_event(label_ready, feature_cols)

    # 与 V4.1 scan 预期值进行轻量交叉验证（若能找到）
    compare_rows = []
    scan_candidates = [
        v4_root / "04_Event_Candidate_Coverage_Scan" / "03_candidate_global_summary.csv",
        DEFAULT_BASE / "04_Event_Candidate_Coverage_Scan" / "03_candidate_global_summary.csv",
    ]
    scan_path = next((p for p in scan_candidates if p.exists()), None)
    if scan_path is not None:
        scan = pd.read_csv(scan_path, low_memory=False)
        scan = scan[scan["EventFamily"].astype(str).eq("RefAhFraction")].copy()
        for r in cov_summary.itertuples(index=False):
            qscan = scan[np.isclose(pd.to_numeric(scan["Node"], errors="coerce"), float(r.Node))]
            if qscan.empty:
                continue
            sr = qscan.iloc[0]
            old_model = float(sr.get("GlobalCoverage", np.nan))
            old_phys = float(sr.get("PhysicalCoverage", np.nan))
            compare_rows.append({
                "Node": float(r.Node),
                "V4p1_ModelCoverage": old_model,
                "V4p2_ModelCoverage": float(r.ModelCoverage),
                "AbsDiff_Model": abs(old_model - float(r.ModelCoverage)) if np.isfinite(old_model) else np.nan,
                "V4p1_PhysicalCoverage": old_phys,
                "V4p2_PhysicalCoverage": float(r.PhysicalCoverage),
                "AbsDiff_Physical": abs(old_phys - float(r.PhysicalCoverage)) if np.isfinite(old_phys) else np.nan,
            })

    # 全局输出
    save_csv(event, out_root / "03_ALL_FROZEN_REFAH_EVENT_WITH_LABELS.csv.gz")
    save_csv(label_ready, out_root / "04_ALL_FROZEN_REFAH_EVENT_LABEL_READY.csv.gz")
    save_csv(coverage, out_root / "05_ALL_FROZEN_REFAH_EVENT_COVERAGE.csv.gz")
    save_csv(cov_summary, out_root / "06_frozen_event_coverage_summary.csv")
    save_csv(cov_battery, out_root / "07_frozen_event_battery_coverage.csv")
    save_csv(cov_soh, out_root / "08_frozen_event_soh_bin_coverage.csv")
    save_csv(registry, out_root / "09_frozen_event_feature_registry.csv")
    save_csv(pd.DataFrame(compare_rows), out_root / "10_V4p1_V4p2_coverage_crosscheck.csv")

    # 最终 acceptance gate
    all_nodes_pass = int(len(cov_summary) == len(FROZEN_REFAH_NODES) and cov_summary["FinalAcceptancePASS"].eq(1).all())
    expected_battery_count = len(batteries)
    actual_battery_count = int(label_ready["Battery"].nunique()) if not label_ready.empty else 0
    target_missing = int(event["SOH_Target_SameCycle"].isna().sum()) if "SOH_Target_SameCycle" in event else len(event)
    excluded_present = int(event["Battery"].isin(EXCLUDED_BATTERIES).any()) if not event.empty else 0

    acceptance = pd.DataFrame([
        {"Check": "FrozenNodeCount", "Observed": int(cov_summary.shape[0]), "Expected": len(FROZEN_REFAH_NODES), "PASS": int(cov_summary.shape[0] == len(FROZEN_REFAH_NODES))},
        {"Check": "AllFrozenNodesCoverageGate", "Observed": all_nodes_pass, "Expected": 1, "PASS": all_nodes_pass},
        {"Check": "BatteryCount", "Observed": actual_battery_count, "Expected": expected_battery_count, "PASS": int(actual_battery_count == expected_battery_count)},
        {"Check": "RWTH028Excluded", "Observed": excluded_present, "Expected": 0, "PASS": int(excluded_present == 0)},
        {"Check": "DuplicateEventKey", "Observed": int(dup.sum()), "Expected": 0, "PASS": int(dup.sum() == 0)},
        {"Check": "LabelReadyRows", "Observed": int(len(label_ready)), "Expected": ">0", "PASS": int(len(label_ready) > 0)},
    ])
    save_csv(acceptance, out_root / "11_final_acceptance_gate.csv")

    final_pass = int(acceptance["PASS"].eq(1).all())
    complete = {
        "Task": "RWTH48 V4.2 Frozen RefAh Event HI",
        "Status": "PASS" if final_pass else "FAIL",
        "V4Root": str(v4_root),
        "V4Script": str(v4_script),
        "OutputRoot": str(out_root),
        "ExcludedBatteries": sorted(EXCLUDED_BATTERIES),
        "BatteryCount": actual_battery_count,
        "FrozenNodes": list(FROZEN_REFAH_NODES),
        "EventRowsAll": int(len(event)),
        "EventRowsLabelReady": int(len(label_ready)),
        "FeatureCount": int(len(feature_cols)),
        "RowsWithAnyFeatureMissing_LabelReady": int((label_ready["MissingFeatureCount"] > 0).sum()) if "MissingFeatureCount" in label_ready else 0,
        "FinalAcceptancePASS": final_pass,
        "Elapsed_min": (time.perf_counter() - t_all) / 60.0,
        "Next": "Run leakage-safe feature screening / redundancy pruning separately by frozen node, then LOBO ML/BiLSTM.",
    }
    save_json(complete, out_root / "99_RWTH48_V4_2_COMPLETE.json")

    print("\n" + "=" * 110)
    print("RWTH48 V4.2 COMPLETE |", complete["Status"])
    print("Frozen RefAh nodes :", FROZEN_REFAH_NODES)
    print("Batteries          :", actual_battery_count)
    print("Event rows all     :", len(event))
    print("Label-ready rows   :", len(label_ready))
    print("Feature count      :", len(feature_cols))
    print("Coverage:")
    print(cov_summary.to_string(index=False))
    print("Final gate         :", final_pass)
    print("Output             :", out_root)
    print("=" * 110)

    if not final_pass:
        raise RuntimeError("V4.2 final acceptance gate FAILED，请检查 11_final_acceptance_gate.csv")


if __name__ == "__main__":
    main()

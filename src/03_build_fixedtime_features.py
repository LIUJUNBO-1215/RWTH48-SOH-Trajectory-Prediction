# -*- coding: utf-8 -*-
"""
RWTH48 V4.2 | 双平台冻结：FixedTime + RefAh Event HI
=====================================================

目标
----
把 RWTH48 后续建模正式冻结成两条并行 prefix 平台：

A. FixedTime 主平台（直接复用 V4 已提取结果，不重复计算）
   60 / 120 / 180 / 300 / 450 / 600 / 900 / 1200 s

B. RefAh Event 物理状态对齐平台（从 V4 clean_timeseries_parts 重提）
   RefAhFraction = 0.05 / 0.09 / 0.13

统一规则
--------
1) RWTH-028 整只排除；
2) 只保留 V4 QualityGatePASS=1 的电池；
3) SOH 只作为监督标签，在 prefix HI 构造完成后 merge；
4) Event 使用冻结 BOL/reference capacity，不使用当前 cycle 最终容量；
5) FixedTime 与 Event 均复用原 V4 的 HI 定义；
6) 输出 ALL 与 LABEL_READY 两套数据；LABEL_READY 仅要求标签可用，特征 NaN 不在这里全局删除，
   后续必须在 LOBO 折内做缺失处理/筛选，避免数据泄露；
7) 后续模型按“单节点”分别评估，外层 Leave-One-Battery-Out，禁止随机 row split。

推荐运行
--------
python "src/03_build_fixedtime_features.py" --n-jobs 4

也可显式指定：
python "src/03_build_fixedtime_features.py" ^
  --v4-root "workspace/RWTH48_全流程输出_V4" ^
  --v4-script "src/01_preprocess_soh_labels.py" ^
  --n-jobs 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
DEFAULT_OUTPUT_NAME = "RWTH48_V4_2_双平台冻结_FixedTime_Event_HI"

FROZEN_FIXEDTIME_NODES_S: Tuple[int, ...] = (60, 120, 180, 300, 450, 600, 900, 1200)
FROZEN_REFAH_NODES: Tuple[float, ...] = (0.05, 0.09, 0.13)
EXCLUDED_BATTERIES = {"RWTH-028"}

# 双平台最低验收线。正式节点应明显高于它，而不是刚好擦线。
MIN_PHYSICAL_COVERAGE = 0.95
MIN_MODEL_COVERAGE = 0.95
MIN_BATTERY_MEDIAN_COVERAGE = 0.95
MIN_WORST_SOH_BIN_COVERAGE = 0.85

SOH_BINS = [-np.inf, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, np.inf]
SOH_BIN_LABELS = ["<0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00", ">=1.00"]

LABEL_COLS = [
    "Battery", "GlobalCycle", "SOH_Target_SameCycle", "SOH_IsMeasuredCheckUp",
    "SOH_LabelUsable", "SOH_LabelSource",
]

NON_FEATURE_COLS = {
    "Battery", "GlobalCycle", "Scope", "Platform", "NodeType", "NodeValue", "PlatformNode",
    "ReferenceCapacity_Ah", "SOH_Target_SameCycle", "SOH_IsMeasuredCheckUp",
    "SOH_LabelUsable", "SOH_LabelSource", "EventReachable", "PrefixFeatureValid",
    "MissingFeatureCount",
}


# =============================================================================
# 1. 基础工具
# =============================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    compression = "gzip" if str(path).lower().endswith(".gz") else None
    df.to_csv(path, index=False, encoding="utf-8-sig", compression=compression)


def save_json(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def normalize_battery(x: object) -> str:
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return s
    if s.upper().startswith("RWTH-"):
        tail = s.split("-")[-1]
        try:
            return f"RWTH-{int(tail):03d}"
        except Exception:
            return s
    if s.isdigit():
        return f"RWTH-{int(s):03d}"
    return s


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RWTH48 V4.2 dual-platform frozen FixedTime + Event HI")
    p.add_argument("--v4-root", type=str, default="", help="V4输出根目录；留空自动探测。")
    p.add_argument("--v4-script", type=str, default="", help="原V4脚本；留空自动搜索。")
    p.add_argument("--output-root", type=str, default="", help="本脚本输出目录；留空自动创建。")
    p.add_argument("--n-jobs", type=int, default=4, help="Event 按电池并行数，Windows建议2~4。")
    p.add_argument("--force", action="store_true", help="忽略Event per-battery缓存，重新提取。")
    p.add_argument("--only-battery", type=str, default="", help="调试：仅提一个电池Event，例如002。")
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
        "未自动找到V4输出目录，请用 --v4-root 指定。候选：\n" +
        "\n".join(str(x) for x in DEFAULT_V4_ROOT_CANDIDATES)
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
    # dataclass 在类装饰阶段会通过 cls.__module__ 回查 sys.modules。
    # 使用 importlib 动态加载时，module_from_spec() 不会自动完成该注册，
    # 因此必须在 exec_module() 之前手动放入 sys.modules。
    module_name = "rwth48_v4_source"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载V4模块：{script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # 加载失败时移除半初始化模块，避免后续重试受到污染。
        sys.modules.pop(module_name, None)
        raise
    required = [
        "prefix_features", "event_cut_index_refah", "collapse_duplicate_timestamps",
        "get_discharge_cycle", "MIN_PREFIX_ROWS", "MIN_DCH_DURATION_S",
    ]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f"V4脚本缺少对象：{missing}")
    return mod


def find_global_file(v4_root: Path, filename: str, required: bool = True) -> Optional[Path]:
    candidates = [v4_root / "03_global" / filename, v4_root / filename]
    for p in candidates:
        if p.exists():
            return p
    hits = list(v4_root.rglob(filename))
    if hits:
        return hits[0]
    if required:
        raise FileNotFoundError(f"V4输出中找不到：{filename}")
    return None


def find_battery_dir(v4_root: Path, battery: str) -> Optional[Path]:
    base = v4_root / "02_batteries"
    if not base.exists():
        return None
    tail = battery.split("-")[-1]
    variants = [base / f"Cell_{battery}", base / battery, base / f"Cell_{tail}"]
    for p in variants:
        if p.exists():
            return p
    for p in base.iterdir():
        if p.is_dir() and tail in p.name:
            return p
    return None


# =============================================================================
# 2. 质量、电池、SOH 标签
# =============================================================================

def load_quality_and_labels(v4_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    q = pd.read_csv(find_global_file(v4_root, "00_battery_quality_summary.csv"), low_memory=False)
    labels = pd.read_csv(find_global_file(v4_root, "02_all_cycle_soh_labels.csv.gz"), low_memory=False)

    if "Battery" not in q.columns or "Battery" not in labels.columns:
        raise RuntimeError("quality/labels 缺少 Battery")

    q["Battery"] = q["Battery"].map(normalize_battery)
    labels["Battery"] = labels["Battery"].map(normalize_battery)
    labels["GlobalCycle"] = pd.to_numeric(labels["GlobalCycle"], errors="coerce")
    labels = labels[labels["GlobalCycle"].notna()].copy()
    labels["GlobalCycle"] = labels["GlobalCycle"].astype(int)

    q = q[~q["Battery"].isin(EXCLUDED_BATTERIES)].copy()
    labels = labels[~labels["Battery"].isin(EXCLUDED_BATTERIES)].copy()

    if "QualityGatePASS" in q.columns:
        mask = pd.to_numeric(q["QualityGatePASS"], errors="coerce").fillna(0).astype(int).eq(1)
        q = q[mask].copy()

    if q["Battery"].duplicated().any():
        raise RuntimeError("quality summary 中 Battery 重复")
    if labels.duplicated(["Battery", "GlobalCycle"]).any():
        dup = labels[labels.duplicated(["Battery", "GlobalCycle"], keep=False)].head(10)
        raise RuntimeError(f"SOH labels 主键重复：\n{dup}")

    batteries = sorted(q["Battery"].dropna().astype(str).unique().tolist())
    labels = labels[labels["Battery"].isin(batteries)].copy()
    return q, labels, batteries


def label_ready_mask(df: pd.DataFrame) -> pd.Series:
    usable = pd.to_numeric(df.get("SOH_LabelUsable", 0), errors="coerce").fillna(0).astype(int).eq(1)
    y = pd.to_numeric(df.get("SOH_Target_SameCycle", np.nan), errors="coerce")
    return usable & y.notna() & np.isfinite(y)


# =============================================================================
# 3. FixedTime：直接复用 V4 正式结果
# =============================================================================

def load_frozen_fixedtime(v4_root: Path, batteries: List[str]) -> pd.DataFrame:
    path = find_global_file(v4_root, "04_ALL_FIXEDTIME_MODEL_READY.csv.gz")
    d = pd.read_csv(path, low_memory=False)
    required = {"Battery", "GlobalCycle", "NodeValue"}
    if not required.issubset(d.columns):
        raise RuntimeError(f"FixedTime表缺列：{sorted(required - set(d.columns))}")

    d["Battery"] = d["Battery"].map(normalize_battery)
    d["GlobalCycle"] = pd.to_numeric(d["GlobalCycle"], errors="coerce")
    d["NodeValue"] = pd.to_numeric(d["NodeValue"], errors="coerce")
    d = d[d["GlobalCycle"].notna() & d["Battery"].isin(batteries)].copy()
    d["GlobalCycle"] = d["GlobalCycle"].astype(int)

    # 仅保留冻结时间节点
    nodes = np.array(FROZEN_FIXEDTIME_NODES_S, dtype=float)
    d = d[d["NodeValue"].apply(lambda x: np.any(np.isclose(float(x), nodes, atol=1e-9)))].copy()

    d["Platform"] = "FixedTime"
    d["Scope"] = "Discharge_FixedTime"
    d["NodeType"] = "Time_s"
    d["PlatformNode"] = d["NodeValue"].map(lambda x: f"FixedTime_{int(round(float(x)))}s")
    return d.reset_index(drop=True)


# =============================================================================
# 4. Event：只重提冻结节点 0.05 / 0.09 / 0.13
# =============================================================================

def read_clean_timeseries_parts(bdir: Path) -> pd.DataFrame:
    cdir = bdir / "clean_timeseries_parts"
    if not cdir.exists():
        raise FileNotFoundError(f"缺少 clean_timeseries_parts：{cdir}")

    parts = sorted(cdir.glob("*.csv.gz")) + sorted(cdir.glob("*.csv"))
    if not parts:
        raise FileNotFoundError(f"clean_timeseries_parts 为空：{cdir}")

    wanted = [
        "Timestamp", "GlobalCycle", "LocalCycle", "Step", "State", "Procedure", "AhStep",
        "Voltage_V", "Current_A", "Temperature_C",
        "Voltage_V_Clean", "Current_A_Clean", "Temperature_C_Clean",
        "AnyHardInvalidFlag", "AnySpikeFlag",
    ]
    frames: List[pd.DataFrame] = []
    for p in parts:
        header = pd.read_csv(p, nrows=0).columns.tolist()
        usecols = [c for c in wanted if c in header]
        if not {"Timestamp", "GlobalCycle", "State"}.issubset(usecols):
            continue
        x = pd.read_csv(p, usecols=usecols, low_memory=False)
        if not x.empty:
            frames.append(x)

    if not frames:
        raise RuntimeError(f"未从 {cdir} 读到有效clean时序")

    d = pd.concat(frames, ignore_index=True)
    d["GlobalCycle"] = pd.to_numeric(d["GlobalCycle"], errors="coerce")
    d = d[d["GlobalCycle"].notna()].copy()
    d["GlobalCycle"] = d["GlobalCycle"].astype(int)
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
    d = d[d["Timestamp"].notna()].copy()
    d = d.sort_values(["GlobalCycle", "Timestamp"], kind="mergesort").reset_index(drop=True)
    return d


def extract_one_battery_event(
    battery: str,
    ref_capacity_ah: float,
    bdir: Path,
    out_bdir: Path,
    v4mod,
    force: bool,
) -> Dict[str, object]:
    """
    V4.2 FIX2:
    - 不再对每个cycle调用 get_discharge_cycle(d, cyc) 扫描整张电池时序表；
      改为一次 groupby("GlobalCycle") 后逐组处理，避免 O(N_cycle * N_row) 重复扫描。
    - 每个cycle只 collapse_duplicate_timestamps 一次，并一次性计算累计放电Ah，
      供 0.05/0.09/0.13 三个冻结Event共用。
    - prefix_features 仍直接调用原V4函数，HI定义完全不变。
    """
    ensure_dir(out_bdir)
    f_event = out_bdir / "event_features.csv.gz"
    f_cov = out_bdir / "event_coverage.csv"
    f_done = out_bdir / "COMPLETE.json"

    if (not force) and f_done.exists() and f_event.exists() and f_cov.exists():
        return {
            "Battery": battery, "Status": "CACHED",
            "EventPath": str(f_event), "CoveragePath": str(f_cov),
        }

    t0 = time.perf_counter()
    d = read_clean_timeseries_parts(bdir)
    event_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []

    # 关键优化：一次groupby，之后不再为每个cycle扫描整张d。
    grouped = d.groupby("GlobalCycle", sort=True, observed=True)
    total_cycles = int(grouped.ngroups)
    eligible_cycles = 0
    processed_cycles = 0

    for cyc, raw_g in grouped:
        processed_cycles += 1

        # 精确复现 V4 get_discharge_cycle 在“已知当前cycle分组”条件下的逻辑。
        g = raw_g[raw_g["State"].eq("DCH")].copy()
        if g.empty:
            continue
        I0 = pd.to_numeric(g.get("Current_A_Clean", g.get("Current_A")), errors="coerce")
        active = I0 < float(v4mod.DISCHARGE_CURRENT_THRESHOLD_A)
        if active.any():
            first_idx = int(np.flatnonzero(active.to_numpy())[0])
            g = g.iloc[first_idx:].copy()
        g = g.sort_values("Timestamp", kind="mergesort")

        if len(g) < int(v4mod.MIN_PREFIX_ROWS):
            continue
        t = pd.to_datetime(g["Timestamp"], errors="coerce")
        dur = (t.max() - t.min()).total_seconds()
        if not np.isfinite(dur) or dur < float(v4mod.MIN_DCH_DURATION_S):
            continue

        eligible_cycles += 1
        p = v4mod.collapse_duplicate_timestamps(g)
        if len(p) < int(v4mod.MIN_PREFIX_ROWS):
            continue

        # 一次性计算累计放电Ah；数学定义与原V4 event_cut_index_refah完全一致。
        tp = pd.to_datetime(p["Timestamp"], errors="coerce")
        ts = (tp - tp.min()).dt.total_seconds().to_numpy(float)
        Ip = pd.to_numeric(
            p.get("Current_A_Clean", p.get("Current_A")), errors="coerce"
        ).to_numpy(float)
        dt = np.diff(ts)
        cur = np.maximum(-Ip, 0.0)
        inc = np.where(
            (dt > 0) & np.isfinite(dt),
            0.5 * (cur[:-1] + cur[1:]) * dt / 3600.0,
            0.0,
        )
        cum_ah = np.r_[0.0, np.cumsum(np.nan_to_num(inc, nan=0.0))]

        for frac in FROZEN_REFAH_NODES:
            target = float(ref_capacity_ah) * float(frac)
            hit = np.flatnonzero(cum_ah >= target)
            idx = int(hit[0]) if len(hit) else None
            reachable = int(idx is not None)

            coverage_rows.append({
                "Battery": battery,
                "GlobalCycle": int(cyc),
                "Platform": "Event",
                "Scope": "Event_RefAhFraction",
                "NodeType": "RefAhFraction",
                "NodeValue": float(frac),
                "PlatformNode": f"Event_RefAh_{frac:.2f}",
                "Reachable": reachable,
            })
            if idx is None:
                continue

            pref = p.iloc[:idx + 1].copy()
            feats = v4mod.prefix_features(pref, float(ref_capacity_ah))
            if not feats:
                continue

            event_rows.append({
                "Battery": battery,
                "GlobalCycle": int(cyc),
                "Platform": "Event",
                "Scope": "Discharge_Event_RefAhFraction",
                "NodeType": "RefAhFraction",
                "NodeValue": float(frac),
                "PlatformNode": f"Event_RefAh_{frac:.2f}",
                "ReferenceCapacity_Ah": float(ref_capacity_ah),
                "EventReachable": 1,
                "PrefixFeatureValid": 1,
                **feats,
            })

        # 心跳：单只电池较大时也能看到程序仍在工作。
        if processed_cycles % 500 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"    [{battery}] cycles {processed_cycles}/{total_cycles} "
                f"eligible={eligible_cycles} elapsed={elapsed/60:.1f} min",
                flush=True,
            )

    event = pd.DataFrame(event_rows)
    coverage = pd.DataFrame(coverage_rows)
    save_csv(event, f_event)
    save_csv(coverage, f_cov)

    info = {
        "Battery": battery,
        "Status": "PASS",
        "EligibleCycles": int(eligible_cycles),
        "EventRows": int(len(event)),
        "CoverageRows": int(len(coverage)),
        "Elapsed_s": float(time.perf_counter() - t0),
        "EventPath": str(f_event),
        "CoveragePath": str(f_cov),
    }
    save_json(info, f_done)
    return info


def run_event_extraction(
    v4_root: Path,
    output_root: Path,
    q: pd.DataFrame,
    batteries: List[str],
    v4mod,
    n_jobs: int,
    force: bool,
    only_battery: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = batteries
    if only_battery:
        b = normalize_battery(only_battery)
        if b in EXCLUDED_BATTERIES:
            raise RuntimeError(f"{b} 已冻结排除，不允许作为正式输入。")
        if b not in batteries:
            raise RuntimeError(f"{b} 不在质量通过电池列表。")
        selected = [b]

    qmap = q.set_index("Battery")
    jobs = []
    for bat in selected:
        if bat not in qmap.index:
            continue
        ref = pd.to_numeric(pd.Series([qmap.loc[bat, "ReferenceCapacity_Ah"]]), errors="coerce").iloc[0]
        if not np.isfinite(ref) or ref <= 0:
            raise RuntimeError(f"{bat} ReferenceCapacity_Ah 无效：{ref}")
        bdir = find_battery_dir(v4_root, bat)
        if bdir is None:
            raise FileNotFoundError(f"找不到 {bat} V4电池目录")
        out_bdir = output_root / "01_event_by_battery" / bat
        jobs.append((bat, float(ref), bdir, out_bdir))

    results: List[Dict[str, object]] = []
    workers = max(1, int(n_jobs))
    print(f"Event jobs: {len(jobs)} batteries | workers={workers}", flush=True)
    if workers == 1:
        for k, args in enumerate(jobs, 1):
            print(f"[Event {k}/{len(jobs)}] {args[0]}")
            results.append(extract_one_battery_event(*args, v4mod, force))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(extract_one_battery_event, *args, v4mod, force): args[0]
                for args in jobs
            }
            done = 0
            for fut in as_completed(futs):
                done += 1
                bat = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    r = {"Battery": bat, "Status": "ERROR", "Error": repr(exc)}
                results.append(r)
                print(f"[Event {done}/{len(jobs)}] {bat}: {r.get('Status')}")

    report = pd.DataFrame(results)
    bad = report[~report["Status"].isin(["PASS", "CACHED"])] if not report.empty else report
    if not bad.empty:
        save_csv(report, output_root / "02_event_extraction_status.csv")
        raise RuntimeError(f"Event提取有失败电池：\n{bad[['Battery','Status']].to_string(index=False)}")

    event_frames: List[pd.DataFrame] = []
    cov_frames: List[pd.DataFrame] = []
    for r in results:
        ep = Path(str(r["EventPath"]))
        cp = Path(str(r["CoveragePath"]))
        if ep.exists():
            event_frames.append(pd.read_csv(ep, low_memory=False))
        if cp.exists():
            cov_frames.append(pd.read_csv(cp, low_memory=False))

    event = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    coverage = pd.concat(cov_frames, ignore_index=True) if cov_frames else pd.DataFrame()
    save_csv(report, output_root / "02_event_extraction_status.csv")
    return event, coverage, report


# =============================================================================
# 5. Coverage 审计：FixedTime + Event 统一
# =============================================================================

def load_fixedtime_coverage(v4_root: Path, batteries: List[str]) -> pd.DataFrame:
    p = find_global_file(v4_root, "06_ALL_prefix_coverage.csv.gz", required=False)
    if p is None:
        p = find_global_file(v4_root, "06_ALL_prefix_coverage.csv", required=False)
    if p is None:
        return pd.DataFrame()

    c = pd.read_csv(p, low_memory=False)
    if "Battery" not in c.columns:
        return pd.DataFrame()
    c["Battery"] = c["Battery"].map(normalize_battery)
    c = c[c["Battery"].isin(batteries)].copy()

    scope_col = c.get("Scope", pd.Series("", index=c.index)).astype(str)
    c = c[scope_col.eq("FixedTime")].copy()
    if c.empty:
        return c

    c["NodeValue"] = pd.to_numeric(c.get("Node", c.get("NodeValue")), errors="coerce")
    c["GlobalCycle"] = pd.to_numeric(c["GlobalCycle"], errors="coerce")
    c = c[c["GlobalCycle"].notna() & c["NodeValue"].notna()].copy()
    c["GlobalCycle"] = c["GlobalCycle"].astype(int)
    nodes = np.array(FROZEN_FIXEDTIME_NODES_S, dtype=float)
    c = c[c["NodeValue"].apply(lambda x: np.any(np.isclose(float(x), nodes, atol=1e-9)))].copy()
    c["Reachable"] = pd.to_numeric(c.get("Reachable", 0), errors="coerce").fillna(0).astype(int)
    c["Platform"] = "FixedTime"
    c["NodeType"] = "Time_s"
    c["PlatformNode"] = c["NodeValue"].map(lambda x: f"FixedTime_{int(round(float(x)))}s")
    return c[["Battery", "GlobalCycle", "Platform", "NodeType", "NodeValue", "PlatformNode", "Reachable"]].copy()


def build_coverage_audits(
    fixed_cov: pd.DataFrame,
    event_cov: pd.DataFrame,
    labels: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    covs = []
    for x in (fixed_cov, event_cov):
        if x is not None and not x.empty:
            y = x.copy()
            y["Battery"] = y["Battery"].map(normalize_battery)
            y["GlobalCycle"] = pd.to_numeric(y["GlobalCycle"], errors="coerce")
            y = y[y["GlobalCycle"].notna()].copy()
            y["GlobalCycle"] = y["GlobalCycle"].astype(int)
            y["Reachable"] = pd.to_numeric(y["Reachable"], errors="coerce").fillna(0).astype(int)
            covs.append(y)
    if not covs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cov = pd.concat(covs, ignore_index=True)
    lab = labels[[c for c in LABEL_COLS if c in labels.columns]].copy()
    cov = cov.merge(lab, on=["Battery", "GlobalCycle"], how="left", validate="many_to_one")
    cov["ModelEligible"] = label_ready_mask(cov).astype(int)
    y = pd.to_numeric(cov.get("SOH_Target_SameCycle"), errors="coerce")
    cov["SOHBin"] = pd.cut(y, bins=SOH_BINS, labels=SOH_BIN_LABELS, right=False)

    global_rows = []
    battery_rows = []
    soh_rows = []

    group_cols = ["Platform", "NodeType", "NodeValue", "PlatformNode"]
    for key, g in cov.groupby(group_cols, dropna=False):
        platform, ntype, nvalue, pnode = key
        physical_n = len(g)
        physical_reach = int(g["Reachable"].sum())
        gm = g[g["ModelEligible"].eq(1)].copy()
        model_n = len(gm)
        model_reach = int(gm["Reachable"].sum())

        bvals = []
        for bat, gb in gm.groupby("Battery"):
            bcov = float(gb["Reachable"].mean()) if len(gb) else np.nan
            bvals.append(bcov)
            battery_rows.append({
                "Platform": platform, "NodeType": ntype, "NodeValue": nvalue,
                "PlatformNode": pnode, "Battery": bat,
                "ModelEligibleCycles": int(len(gb)),
                "ReachableCycles": int(gb["Reachable"].sum()),
                "Coverage": bcov,
            })

        bin_covs = []
        for sb, gs in gm.groupby("SOHBin", observed=False):
            if len(gs) == 0:
                continue
            scov = float(gs["Reachable"].mean())
            bin_covs.append(scov)
            soh_rows.append({
                "Platform": platform, "NodeType": ntype, "NodeValue": nvalue,
                "PlatformNode": pnode, "SOHBin": str(sb),
                "ModelEligibleCycles": int(len(gs)),
                "ReachableCycles": int(gs["Reachable"].sum()),
                "Coverage": scov,
            })

        physical_cov = physical_reach / physical_n if physical_n else np.nan
        model_cov = model_reach / model_n if model_n else np.nan
        bmedian = float(np.nanmedian(bvals)) if bvals else np.nan
        bmin = float(np.nanmin(bvals)) if bvals else np.nan
        worst_soh = float(np.nanmin(bin_covs)) if bin_covs else np.nan

        passed = int(
            np.isfinite(physical_cov) and physical_cov >= MIN_PHYSICAL_COVERAGE
            and np.isfinite(model_cov) and model_cov >= MIN_MODEL_COVERAGE
            and np.isfinite(bmedian) and bmedian >= MIN_BATTERY_MEDIAN_COVERAGE
            and np.isfinite(worst_soh) and worst_soh >= MIN_WORST_SOH_BIN_COVERAGE
        )

        global_rows.append({
            "Platform": platform, "NodeType": ntype, "NodeValue": nvalue,
            "PlatformNode": pnode,
            "PhysicalEligibleCycles": int(physical_n),
            "PhysicalReachableCycles": int(physical_reach),
            "PhysicalCoverage": physical_cov,
            "ModelEligibleCycles": int(model_n),
            "ModelReachableCycles": int(model_reach),
            "ModelCoverage": model_cov,
            "BatteryCoverageMedian": bmedian,
            "BatteryCoverageMin": bmin,
            "WorstSOHBinCoverage": worst_soh,
            "PASS": passed,
        })

    global_df = pd.DataFrame(global_rows).sort_values(["Platform", "NodeValue"]).reset_index(drop=True)
    battery_df = pd.DataFrame(battery_rows)
    soh_df = pd.DataFrame(soh_rows)
    return global_df, battery_df, soh_df


# =============================================================================
# 6. 特征注册与建模 manifest
# =============================================================================

def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def add_missing_feature_count(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    x = df.copy()
    feats = numeric_feature_columns(x)
    if feats:
        x["MissingFeatureCount"] = x[feats].isna().sum(axis=1).astype(int)
    else:
        x["MissingFeatureCount"] = 0
    return x, feats


def feature_registry(fixed: pd.DataFrame, event: pd.DataFrame) -> pd.DataFrame:
    union = sorted(set(numeric_feature_columns(fixed)) | set(numeric_feature_columns(event)))
    rows = []
    for c in union:
        vals = []
        if c in fixed.columns:
            vals.append(pd.to_numeric(fixed[c], errors="coerce"))
        if c in event.columns:
            vals.append(pd.to_numeric(event[c], errors="coerce"))
        z = pd.concat(vals, ignore_index=True) if vals else pd.Series(dtype=float)
        finite = z[np.isfinite(z)]
        nunique = int(finite.nunique()) if len(finite) else 0
        rows.append({
            "Feature": c,
            "AvailableInFixedTime": int(c in fixed.columns),
            "AvailableInEvent": int(c in event.columns),
            "OverallNonMissingRate": float(z.notna().mean()) if len(z) else np.nan,
            "OverallUniqueFiniteValues": nunique,
            "ConstantOrEmpty": int(nunique <= 1),
            "CausalPrefixLegal": 1,
            "Note": "Feature comes from observed prefix only; final selection/imputation must be fitted within each LOBO training fold.",
        })
    return pd.DataFrame(rows)


def modeling_manifest(all_ready: pd.DataFrame, coverage_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (platform, ntype, nvalue, pnode), g in all_ready.groupby(
        ["Platform", "NodeType", "NodeValue", "PlatformNode"], dropna=False
    ):
        covrow = coverage_summary[coverage_summary["PlatformNode"].eq(pnode)] if not coverage_summary.empty else pd.DataFrame()
        rows.append({
            "Platform": platform,
            "NodeType": ntype,
            "NodeValue": nvalue,
            "PlatformNode": pnode,
            "Rows": int(len(g)),
            "Batteries": int(g["Battery"].nunique()),
            "Cycles": int(g[["Battery", "GlobalCycle"]].drop_duplicates().shape[0]),
            "SOH_Min": float(pd.to_numeric(g["SOH_Target_SameCycle"], errors="coerce").min()),
            "SOH_Max": float(pd.to_numeric(g["SOH_Target_SameCycle"], errors="coerce").max()),
            "PhysicalCoverage": float(covrow["PhysicalCoverage"].iloc[0]) if len(covrow) else np.nan,
            "ModelCoverage": float(covrow["ModelCoverage"].iloc[0]) if len(covrow) else np.nan,
            "CoveragePASS": int(covrow["PASS"].iloc[0]) if len(covrow) else 0,
            "RecommendedOuterSplit": "Leave-One-Battery-Out",
            "DoNotUse": "random row split",
            "ModelingRule": "Train/evaluate this node as an independent prediction task; any imputation/scaling/feature selection is fit on training batteries only.",
        })
    return pd.DataFrame(rows).sort_values(["Platform", "NodeValue"]).reset_index(drop=True)


# =============================================================================
# 7. 主程序
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    t0 = time.perf_counter()

    v4_root = detect_v4_root(args.v4_root)
    v4_script = detect_v4_script(args.v4_script)
    output_root = ensure_dir(
        Path(args.output_root) if args.output_root else (DEFAULT_BASE / DEFAULT_OUTPUT_NAME)
    )

    print("=" * 110)
    print("RWTH48 V4.2 | 双平台冻结 FixedTime + RefAh Event")
    print("V4 ROOT :", v4_root)
    print("V4 SCRIPT:", v4_script)
    print("OUTPUT   :", output_root)
    print("FixedTime:", FROZEN_FIXEDTIME_NODES_S)
    print("Event    :", FROZEN_REFAH_NODES)
    print("Excluded :", sorted(EXCLUDED_BATTERIES))
    print("=" * 110)

    v4mod = load_v4_module(v4_script)
    q, labels, batteries = load_quality_and_labels(v4_root)

    # inclusion / exclusion audit
    inclusion = q[[c for c in ["Battery", "QualityGatePASS", "ReferenceCapacity_Ah", "RecordedGlobalCycles", "UsableSOHLabelCycles"] if c in q.columns]].copy()
    inclusion["Included"] = 1
    exclusion_rows = [{"Battery": b, "Included": 0, "Reason": "Frozen exclusion: abnormal BOL/SOH trajectory"} for b in sorted(EXCLUDED_BATTERIES)]
    save_csv(inclusion, output_root / "00_battery_inclusion.csv")
    save_csv(pd.DataFrame(exclusion_rows), output_root / "00_battery_exclusion.csv")

    # A. FixedTime: reuse existing V4 output
    print("\n[A] Loading frozen FixedTime from V4 ...")
    fixed_all = load_frozen_fixedtime(v4_root, batteries)
    fixed_all, fixed_feats = add_missing_feature_count(fixed_all)
    fixed_ready = fixed_all[label_ready_mask(fixed_all)].copy().reset_index(drop=True)

    # B. Event: re-extract only frozen nodes
    print("\n[B] Extracting frozen RefAh Event from clean timeseries ...")
    event_all, event_cov, event_status = run_event_extraction(
        v4_root=v4_root,
        output_root=output_root,
        q=q,
        batteries=batteries,
        v4mod=v4mod,
        n_jobs=args.n_jobs,
        force=args.force,
        only_battery=args.only_battery,
    )

    if event_all.empty:
        raise RuntimeError("Event重提结果为空")

    event_all["Battery"] = event_all["Battery"].map(normalize_battery)
    event_all["GlobalCycle"] = pd.to_numeric(event_all["GlobalCycle"], errors="coerce").astype(int)
    lab_merge = labels[[c for c in LABEL_COLS if c in labels.columns]].copy()
    event_all = event_all.merge(
        lab_merge,
        on=["Battery", "GlobalCycle"],
        how="left",
        validate="many_to_one",
    )
    event_all, event_feats = add_missing_feature_count(event_all)
    event_ready = event_all[label_ready_mask(event_all)].copy().reset_index(drop=True)

    # 固定时间 coverage
    fixed_cov = load_fixedtime_coverage(v4_root, batteries)
    coverage_summary, battery_coverage, soh_coverage = build_coverage_audits(
        fixed_cov=fixed_cov,
        event_cov=event_cov,
        labels=labels,
    )

    # 若旧V4没有 coverage 文件，仍能继续，但 acceptance 会标明缺失
    if fixed_cov.empty:
        print("WARNING: 未找到 V4 FixedTime coverage 明细；FixedTime coverage 审计无法重算，但特征数据仍会保留。")

    # 双平台合并，仅用于统一管理。正式建模仍按单节点拆开。
    common_cols = sorted(set(fixed_ready.columns) | set(event_ready.columns))
    fixed_u = fixed_ready.reindex(columns=common_cols)
    event_u = event_ready.reindex(columns=common_cols)
    all_ready = pd.concat([fixed_u, event_u], ignore_index=True)
    all_ready = all_ready.sort_values(["Platform", "NodeValue", "Battery", "GlobalCycle"], kind="mergesort").reset_index(drop=True)

    # 特征注册 / manifest
    freg = feature_registry(fixed_ready, event_ready)
    manifest = modeling_manifest(all_ready, coverage_summary)

    # 验收
    acceptance_rows = []
    expected_nodes = [f"FixedTime_{x}s" for x in FROZEN_FIXEDTIME_NODES_S] + [f"Event_RefAh_{x:.2f}" for x in FROZEN_REFAH_NODES]
    for pnode in expected_nodes:
        m = manifest[manifest["PlatformNode"].eq(pnode)]
        if m.empty:
            acceptance_rows.append({"PlatformNode": pnode, "PASS": 0, "Reason": "Missing model-ready node"})
            continue
        covpass = int(m["CoveragePASS"].iloc[0]) if not np.isnan(pd.to_numeric(m["CoveragePASS"].iloc[0], errors="coerce")) else 0
        batteries_n = int(m["Batteries"].iloc[0])
        pass_ = int(covpass == 1 and batteries_n == len(batteries))
        reason = "PASS" if pass_ else f"CoveragePASS={covpass}; Batteries={batteries_n}/{len(batteries)}"
        acceptance_rows.append({"PlatformNode": pnode, "PASS": pass_, "Reason": reason})
    acceptance = pd.DataFrame(acceptance_rows)

    # excluded/unusable summary
    excluded_summary = pd.DataFrame([
        {
            "Scope": "Battery",
            "Excluded": "RWTH-028",
            "Reason": "abnormal BOL/SOH trajectory",
        },
        {
            "Scope": "FixedTime rows",
            "Excluded": int(len(fixed_all) - len(fixed_ready)),
            "Reason": "SOH_LabelUsable != 1 or target NaN; no feature-based global deletion",
        },
        {
            "Scope": "Event rows",
            "Excluded": int(len(event_all) - len(event_ready)),
            "Reason": "SOH_LabelUsable != 1 or target NaN; no feature-based global deletion",
        },
    ])

    # 输出
    save_csv(fixed_all, output_root / "03_FIXEDTIME_FROZEN_ALL.csv.gz")
    save_csv(fixed_ready, output_root / "04_FIXEDTIME_FROZEN_LABEL_READY.csv.gz")
    save_csv(event_all, output_root / "05_EVENT_REFAH_FROZEN_ALL.csv.gz")
    save_csv(event_ready, output_root / "06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz")
    save_csv(all_ready, output_root / "07_ALL_PREFIX_DUAL_PLATFORM_LABEL_READY.csv.gz")
    save_csv(coverage_summary, output_root / "08_node_coverage_summary.csv")
    save_csv(battery_coverage, output_root / "09_node_battery_coverage.csv")
    save_csv(soh_coverage, output_root / "10_node_soh_bin_coverage.csv")
    save_csv(manifest, output_root / "11_node_modeling_manifest.csv")
    save_csv(freg, output_root / "12_feature_registry_dual_platform.csv")
    save_csv(acceptance, output_root / "13_final_acceptance_gate.csv")
    save_csv(excluded_summary, output_root / "14_excluded_rows_summary.csv")

    config = {
        "V4Root": str(v4_root),
        "V4Script": str(v4_script),
        "OutputRoot": str(output_root),
        "IncludedBatteryCount": len(batteries),
        "ExcludedBatteries": sorted(EXCLUDED_BATTERIES),
        "FrozenFixedTimeNodes_s": list(FROZEN_FIXEDTIME_NODES_S),
        "FrozenRefAhEventNodes": list(FROZEN_REFAH_NODES),
        "FixedTimeStrategy": "reuse V4 extracted HI",
        "EventStrategy": "re-extract from V4 clean_timeseries_parts using original V4 functions",
        "StrictLabelRule": "SOH_LabelUsable==1 and finite SOH_Target_SameCycle",
        "FeatureMissingRule": "do not globally delete/impute by full data; handle within each LOBO training fold",
        "OuterValidation": "Leave-One-Battery-Out",
        "ForbiddenSplit": "random row split",
    }
    save_json(config, output_root / "00_run_config.json")

    final_pass = int((acceptance["PASS"] == 1).all()) if len(acceptance) else 0
    complete = {
        "Task": "RWTH48 V4.2 dual-platform frozen FixedTime + Event HI",
        "Status": "PASS" if final_pass else "CHECK_REQUIRED",
        "IncludedBatteryCount": len(batteries),
        "FixedTimeAllRows": int(len(fixed_all)),
        "FixedTimeLabelReadyRows": int(len(fixed_ready)),
        "EventAllRows": int(len(event_all)),
        "EventLabelReadyRows": int(len(event_ready)),
        "DualPlatformLabelReadyRows": int(len(all_ready)),
        "NodeCount": int(manifest["PlatformNode"].nunique()) if not manifest.empty else 0,
        "AcceptancePassNodes": int(acceptance["PASS"].sum()) if not acceptance.empty else 0,
        "AcceptanceTotalNodes": int(len(acceptance)),
        "Elapsed_s": float(time.perf_counter() - t0),
        "NextStep": "LOBO fold-wise feature redundancy removal + feature selection, separately for each frozen node.",
    }
    save_json(complete, output_root / "99_RWTH48_V4_2_COMPLETE.json")

    print("\n" + "=" * 110)
    print("RWTH48 V4.2 双平台冻结完成")
    print(f"Batteries: {len(batteries)}")
    print(f"FixedTime label-ready rows: {len(fixed_ready):,}")
    print(f"Event label-ready rows    : {len(event_ready):,}")
    print(f"Total dual-platform rows  : {len(all_ready):,}")
    print("核心输出：")
    print("  04_FIXEDTIME_FROZEN_LABEL_READY.csv.gz")
    print("  06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz")
    print("  07_ALL_PREFIX_DUAL_PLATFORM_LABEL_READY.csv.gz")
    print("  08_node_coverage_summary.csv")
    print("  11_node_modeling_manifest.csv")
    print("  12_feature_registry_dual_platform.csv")
    print("  13_final_acceptance_gate.csv")
    print("  99_RWTH48_V4_2_COMPLETE.json")
    print("FINAL STATUS:", complete["Status"])
    print("=" * 110)


if __name__ == "__main__":
    main()

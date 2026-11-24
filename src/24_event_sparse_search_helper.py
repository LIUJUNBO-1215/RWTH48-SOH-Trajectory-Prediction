# -*- coding: utf-8 -*-
r"""
RWTH48 V4.8E LITE
Event Top2 × J/L matched-span Pareto 消融（RF/XGBoost）+ 47-cell Outer LOBO
===============================================================================

目的
----
在 V4.7E 已完成 Event HI 数量消融后，冻结树模型最优 Event Top2：
    1) V_TimeWeightedMean
    2) P_Last

仅改变历史诊断调度 J/L，并将历史跨度尽量固定在约 40 cycles：
    J05_L10 : J=5,  L=10, span=45, 10 diagnoses  [V4.7E baseline]
    J08_L06 : J=8,  L=6,  span=40,  6 diagnoses
    J10_L05 : J=10, L=5,  span=40,  5 diagnoses
    J13_L04 : J=13, L=4,  span=39,  4 diagnoses
    J20_L03 : J=20, L=3,  span=40,  3 diagnoses
    J40_L02 : J=40, L=2,  span=40,  2 diagnoses

冻结：
    Event checkpoints : RefAh 0.05 / 0.09 / 0.13
    Models            : RF / XGBoost
    XGBoost backend   : auto CUDA/CPU；默认优先 CUDA，失败时 auto 模式回退 CPU
    Horizons          : 10 / 50 / 100
    Outer LOBO        : 46 source / 1 held-out
    X                 : HI only
    No Cycle / historical SOH / future HI / calibration in X
    Exact J-spaced history and exact t+H target in one StrictSegmentKey
    No nearest-cycle substitution

主公平比较：以上 6 个 matched-span 配置，按
    Battery + OriginCycle + TargetCycle
做 exact common-cohort intersection。

V4.7E baseline 复用：
若存在 V4.7E/06_ALL_EVENT_HICOUNT_OOF_predictions.csv.gz，代码验证 Top2、key、SOH truth
与本脚本重建的 J=5,L=10 cohort 完全一致后直接复用，避免重复训练。

本脚本独立于正在运行的 V4.8C；复用 V4.8B 的冻结训练/评估函数以及 V4.7E 的 Event 读取函数；因此代码目录中需要：
    25_fixedtime_sparse_search_helper.py
    05_rank_hi_topk_ablation.py

默认新电脑根目录：
    workspace
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# 不再屏蔽 GPU。XGBoost 后端由 --xgb-device 控制；auto 默认优先 CUDA。


# =============================================================================
# 0. Frozen settings
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR = Path(__file__).resolve().parent

STRICT_DEFAULT = (
    BASE / "RWTH48_V4_5C_FIX_最小冲击共同扰动处理" /
    "06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz"
)
V47E_DEFAULT = BASE / "RWTH48_V4_7E_Event_HI数量消融_SourceOnly_LOBO_LITE"
V47E_OOF_DEFAULT = V47E_DEFAULT / "06_ALL_EVENT_HICOUNT_OOF_predictions.csv.gz"
V47E_MANIFEST_DEFAULT = V47E_DEFAULT / "05_selected_event_HI_manifest.csv"
V47E_COMPLETE_DEFAULT = V47E_DEFAULT / "99_RWTH48_V4_7E_COMPLETE.json"
OUTPUT_DEFAULT = BASE / "RWTH48_V4_8E_Event_Top2_RF_XGB_JL_Pareto_LOBO_LITE_ACCEL"

TARGET = "CyclingSOH_Charge_Canonical"
SEGMENT = "StrictSegmentKey"
EVENT_NODES = (
    (0.05, "Event_RefAh_0.05", "E005"),
    (0.09, "Event_RefAh_0.09", "E009"),
    (0.13, "Event_RefAh_0.13", "E013"),
)
EVENT_TOP2 = (
    "V_TimeWeightedMean",
    "P_Last",
)
EVENT_TOP2_COLUMNS = tuple(
    f"{prefix}__{hi}"
    for _, _, prefix in EVENT_NODES
    for hi in EVENT_TOP2
)

JL_CONFIGS: Dict[str, Tuple[int, int, str]] = {
    "J05_L10": (5, 10, "baseline_span45"),
    "J08_L06": (8, 6, "matched_span40"),
    "J10_L05": (10, 5, "matched_span40"),
    "J13_L04": (13, 4, "matched_span39"),
    "J20_L03": (20, 3, "matched_span40"),
    "J40_L02": (40, 2, "matched_span40"),
}
PRIMARY_CONFIGS = ("J05_L10", "J08_L06", "J10_L05", "J13_L04", "J20_L03", "J40_L02")
BASELINE_CONFIG = "J05_L10"
DEFAULT_HORIZONS = (10, 50, 100)
ALL_MODELS = ("RF", "XGBoost")
FAST_GZIP = {"method": "gzip", "compresslevel": 1}
KEY = ["Horizon", "Battery", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly"]


# =============================================================================
# 1. Utilities / module reuse
# =============================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: object, p: Path) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_battery(x: object) -> str:
    s = str(x).strip().upper().replace("_", "-")
    if s.startswith("RWTH-"):
        try:
            return f"RWTH-{int(s.split('-')[-1]):03d}"
        except Exception:
            return s
    try:
        return f"RWTH-{int(float(s)):03d}"
    except Exception:
        return s


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_script(code_dir: Path, exact_name: str, glob_pat: str) -> Path:
    candidates = [
        code_dir / exact_name,
        Path(__file__).resolve().parent / exact_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    for root in (code_dir, Path(__file__).resolve().parent):
        hits = sorted(root.glob(glob_pat)) if root.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"缺少 helper：{exact_name}\n请放入代码目录：{code_dir}"
    )


def resolve_event_file(base: Path, explicit: str, v47e) -> Path:
    return v47e.resolve_event_file(base, explicit)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWTH48 V4.8E Event Top2 RF/XGBoost J/L matched-span Pareto ablation (accelerator-aware)"
    )
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--code-dir", type=Path, default=CODE_DIR)
    p.add_argument("--strict-input", type=Path, default=STRICT_DEFAULT)
    p.add_argument("--event-file", type=str, default="")
    p.add_argument("--v47e-oof", type=Path, default=V47E_OOF_DEFAULT)
    p.add_argument("--v47e-manifest", type=Path, default=V47E_MANIFEST_DEFAULT)
    p.add_argument("--v47e-complete", type=Path, default=V47E_COMPLETE_DEFAULT)
    p.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    p.add_argument("--configs", nargs="+", choices=list(JL_CONFIGS), default=list(JL_CONFIGS))
    p.add_argument("--models", nargs="+", choices=list(ALL_MODELS), default=list(ALL_MODELS))
    p.add_argument("--threads-model", type=int, default=4)
    p.add_argument(
        "--xgb-device", choices=["auto", "cuda", "cpu"], default="auto",
        help="XGBoost执行设备：auto=优先CUDA且失败回退CPU；cuda=强制CUDA；cpu=强制CPU。",
    )
    p.add_argument("--xgb-gpu-id", type=int, default=0, help="XGBoost CUDA设备编号，默认0。")
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--force-folds", action="store_true")
    p.add_argument("--retrain-baseline", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args()



def configure_xgboost_backend(v48b, requested: str, gpu_id: int, threads: int) -> Dict[str, object]:
    """Configure the helper's frozen XGBoost parameters without hiding CUDA.

    Scientific rule:
    - If CUDA is selected, all J/L configurations (including J05_L10 baseline) must be
      trained with the same CUDA backend; the old V4.7E CPU XGBoost baseline is therefore
      not reused.
    - auto probes a tiny model. If CUDA is unavailable, it falls back to the frozen CPU
      hist backend rather than failing the entire long job.
    """
    info: Dict[str, object] = {
        "Requested": str(requested),
        "Selected": "cpu",
        "GPU_ID": int(gpu_id),
        "XGBoostAvailable": False,
        "XGBoostVersion": "",
        "Probe": "NOT_RUN",
        "FallbackReason": "",
    }
    try:
        import xgboost as xgb
        from xgboost import XGBRegressor
    except Exception as exc:
        raise ImportError("未安装 xgboost，无法运行 V4.8E。") from exc

    info["XGBoostAvailable"] = True
    info["XGBoostVersion"] = str(getattr(xgb, "__version__", "unknown"))
    try:
        major = int(str(info["XGBoostVersion"]).split(".")[0])
    except Exception:
        major = 2

    base = dict(v48b.XGB_PARAMS)
    base["n_jobs"] = max(1, int(threads))

    def cpu_params() -> Dict[str, object]:
        p = dict(base)
        p["tree_method"] = "hist"
        p.pop("gpu_id", None)
        p.pop("device", None)
        # XGBoost >=2 supports device='cpu'; omitting device is more backward compatible.
        return p

    def cuda_params() -> Dict[str, object]:
        p = dict(base)
        if major >= 2:
            p["tree_method"] = "hist"
            p["device"] = f"cuda:{int(gpu_id)}"
            p.pop("gpu_id", None)
        else:
            p["tree_method"] = "gpu_hist"
            p["gpu_id"] = int(gpu_id)
            p.pop("device", None)
        return p

    if requested == "cpu":
        v48b.XGB_PARAMS = cpu_params()
        info["Selected"] = "cpu"
        info["Probe"] = "SKIPPED_FORCED_CPU"
        return info

    # Tiny CUDA probe. This does not use any research samples and therefore cannot leak
    # held-out battery information or affect fitted research models.
    Xp = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    yp = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float32)
    try:
        pp = cuda_params()
        probe = XGBRegressor(
            n_estimators=2,
            max_depth=2,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=20260821,
            **{k: v for k, v in pp.items() if k not in {"n_estimators", "max_depth", "learning_rate", "objective", "random_state"}},
        )
        probe.fit(Xp, yp, verbose=False)
        _ = probe.predict(Xp)
        # XGBoost >=2 may silently fall back to CPU when no visible GPU exists.
        # Inspect the fitted booster config so auto mode never mislabels CPU as CUDA.
        if major >= 2:
            cfg = json.loads(probe.get_booster().save_config())
            actual_device = str(cfg.get("learner", {}).get("generic_param", {}).get("device", ""))
            info["ActualProbeDevice"] = actual_device
            if not actual_device.lower().startswith("cuda"):
                raise RuntimeError(f"XGBoost requested CUDA but fitted probe reports device={actual_device!r}")
        del probe
        gc.collect()
        v48b.XGB_PARAMS = cuda_params()
        info["Selected"] = "cuda"
        info["Probe"] = "CUDA_PASS"
        return info
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError(
                "--xgb-device cuda 已强制要求 CUDA，但 XGBoost CUDA probe 失败。\n"
                f"XGBoost={info['XGBoostVersion']} GPU={gpu_id}\n原因：{repr(exc)}"
            ) from exc
        v48b.XGB_PARAMS = cpu_params()
        info["Selected"] = "cpu"
        info["Probe"] = "CUDA_FAIL_CPU_FALLBACK"
        info["FallbackReason"] = repr(exc)
        return info


# =============================================================================
# 2. V4.7E Top2 baseline loading / audits
# =============================================================================

def validate_v47e_manifest(path: Path, horizons: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return pd.DataFrame([{
            "Check": "V4.7E Top2 manifest", "Observed": "MISSING",
            "Expected": "optional but recommended", "PASS": 1,
        }])
    d = pd.read_csv(path, low_memory=False)
    req = {"Horizon", "TestBattery", "HI_Count", "SelectedHI"}
    if not req.issubset(d.columns):
        return pd.DataFrame([{
            "Check": "V4.7E Top2 manifest columns",
            "Observed": str(sorted(d.columns)), "Expected": str(sorted(req)), "PASS": 0,
        }])
    q = d[pd.to_numeric(d["HI_Count"], errors="coerce").eq(2)].copy()
    q = q[pd.to_numeric(q["Horizon"], errors="coerce").isin([int(h) for h in horizons])]
    expected = "|".join(EVENT_TOP2)
    uniq = sorted(q["SelectedHI"].dropna().astype(str).unique().tolist())
    ok = len(q) > 0 and all(x == expected for x in uniq)
    return pd.DataFrame([{
        "Check": "V4.7E Top2 manifest frozen order",
        "Observed": f"rows={len(q)} unique={uniq[:5]}",
        "Expected": expected,
        "PASS": int(ok),
    }])


def load_v47e_top2(path: Path, models: Sequence[str], horizons: Sequence[int],
                    batteries: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path, compression="infer", low_memory=False)
    req = {*KEY, "SOH_True", "HI_Set", "HI_Count", *[f"Pred_{m}" for m in models]}
    missing = sorted(req - set(d.columns))
    if missing:
        raise RuntimeError("V4.7E OOF缺列：" + str(missing))
    d["Battery"] = d["Battery"].map(normalize_battery)
    for c in ("Horizon", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly", "HI_Count"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[
        d["HI_Set"].astype(str).eq("Top2")
        & d["HI_Count"].eq(2)
        & d["Horizon"].isin([int(x) for x in horizons])
        & d["Battery"].isin(list(batteries))
    ].copy()
    for c in ("Horizon", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly", "HI_Count"):
        d[c] = d[c].astype(int)
    d["SOH_True"] = pd.to_numeric(d["SOH_True"], errors="coerce")
    for m in models:
        d[f"Pred_{m}"] = pd.to_numeric(d[f"Pred_{m}"], errors="coerce")

    if "SelectedHI" in d.columns:
        expected = "|".join(EVENT_TOP2)
        bad = sorted(set(d.loc[d["SelectedHI"].notna(), "SelectedHI"].astype(str)) - {expected})
        if bad:
            raise RuntimeError(
                "V4.7E Top2不是冻结的 Event Top2 顺序，拒绝复用：" + str(bad[:5])
            )
    if d.duplicated(KEY).any():
        raise RuntimeError("V4.7E Top2 OOF存在重复 key")

    J, L, _ = JL_CONFIGS[BASELINE_CONFIG]
    d["JL_Config"] = BASELINE_CONFIG
    d["J"] = J
    d["L"] = L
    d["HistorySpan_cycles"] = (L - 1) * J
    d["DiagnosticCount"] = L
    d["HistoryScalarInputs"] = L * len(EVENT_TOP2_COLUMNS)
    d["Source"] = "reused_V4.7E_EventTop2"
    keep = [
        "JL_Config", "J", "L", "HistorySpan_cycles", "DiagnosticCount", "HistoryScalarInputs",
        *KEY, "SOH_True", *[f"Pred_{m}" for m in models], "Source"
    ]
    return d[keep].sort_values(KEY).reset_index(drop=True)


def audit_baseline_against_bundle(v48b, baseline: pd.DataFrame, bundle, H: int,
                                  batteries: Sequence[str]) -> Dict[str, object]:
    if baseline.empty:
        return {"Horizon": int(H), "ReusedV47E": 0, "PASS": 0, "Reason": "OOF unavailable"}
    a = v48b.bundle_keys(bundle, H)
    a = a[a["Battery"].isin(list(batteries))].copy()
    b = baseline[baseline["Horizon"].eq(int(H))][KEY + ["SOH_True"]].copy()
    m = a.merge(b, on=KEY, how="outer", suffixes=("_rebuilt", "_v47e"), indicator=True)
    both = m[m["_merge"].eq("both")]
    maxdiff = float(np.nanmax(np.abs(
        both["SOH_True_rebuilt"].to_numpy(float) - both["SOH_True_v47e"].to_numpy(float)
    ))) if len(both) else np.nan
    left = int((m["_merge"] == "left_only").sum())
    right = int((m["_merge"] == "right_only").sum())
    ok = left == 0 and right == 0 and np.isfinite(maxdiff) and maxdiff <= 1e-12
    return {
        "Horizon": int(H), "ReusedV47E": 1,
        "RebuiltRows": int(len(a)), "V47ERows": int(len(b)),
        "RebuiltOnly": left, "V47EOnly": right,
        "MaxAbsSOHDiff": maxdiff, "PASS": int(ok),
        "Reason": "PASS" if ok else "KEY_OR_TARGET_MISMATCH",
    }


# =============================================================================
# 2C. Event-specific gain / Pareto helpers
# =============================================================================

def event_gain(primary_metrics: pd.DataFrame) -> pd.DataFrame:
    """Gain table with the correct Event Top2 baseline budget (60 scalars, 10 diagnoses)."""
    if primary_metrics.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    bJ, bL, _ = JL_CONFIGS[BASELINE_CONFIG]
    baseline_scalars = bL * len(EVENT_TOP2_COLUMNS)
    for (H, model), g in primary_metrics.groupby(["Horizon", "Model"], sort=True):
        b = g[g["JL_Config"].eq(BASELINE_CONFIG)]
        if b.empty:
            continue
        br = float(b.iloc[0]["RMSE"]); bm = float(b.iloc[0]["MAE"])
        for _, r in g.iterrows():
            rr = float(r["RMSE"]); rm = float(r["MAE"])
            diag = int(r["DiagnosticCount"]); scalars = int(r["HistoryScalarInputs"]); J = int(r["J"])
            rows.append({
                "Horizon": int(H), "Model": str(model), "JL_Config": str(r["JL_Config"]),
                "J": J, "L": int(r["L"]),
                "HistorySpan_cycles": int(r["HistorySpan_cycles"]),
                "DiagnosticCount": diag, "HistoryScalarInputs": scalars,
                "CommonWindows": int(r["CommonWindows"]),
                "RMSE": rr, "Baseline_RMSE": br,
                "RMSE_Improvement_vs_J05_L10_pct": (br - rr) / br * 100.0 if br > 0 else np.nan,
                "RMSE_Loss_vs_J05_L10_pct": (rr - br) / br * 100.0 if br > 0 else np.nan,
                "MAE": rm, "Baseline_MAE": bm,
                "MAE_Improvement_vs_J05_L10_pct": (bm - rm) / bm * 100.0 if bm > 0 else np.nan,
                "DiagnosticCountReduction_pct": (1.0 - diag / float(bL)) * 100.0,
                "HistoryScalarReduction_pct": (1.0 - scalars / float(baseline_scalars)) * 100.0,
                "NominalDiagnosticFrequencyReduction_pct": (1.0 - bJ / float(J)) * 100.0,
                "AtOrBelowFixedTime40ScalarBudget": int(scalars <= 40),
                "AtOrBelowFixedTime5DiagnosisBudget": int(diag <= 5),
            })
    return pd.DataFrame(rows)


def pareto_frontier(gain: pd.DataFrame) -> pd.DataFrame:
    """Mark nondominated RMSE-vs-input-budget points on the exact common cohort."""
    if gain.empty:
        return pd.DataFrame()
    out = []
    for (H, model), g in gain.groupby(["Horizon", "Model"], sort=True):
        g = g.copy().reset_index(drop=True)
        flags = []
        for i, r in g.iterrows():
            dominated = False
            for j, q in g.iterrows():
                if i == j:
                    continue
                no_worse = (float(q["RMSE"]) <= float(r["RMSE"]) + 1e-15 and
                            int(q["HistoryScalarInputs"]) <= int(r["HistoryScalarInputs"]))
                strict = (float(q["RMSE"]) < float(r["RMSE"]) - 1e-15 or
                          int(q["HistoryScalarInputs"]) < int(r["HistoryScalarInputs"]))
                if no_worse and strict:
                    dominated = True
                    break
            flags.append(int(not dominated))
        g["ParetoEfficient_RMSE_vs_Scalars"] = flags
        for tol in (1, 2, 5, 10):
            g[f"Within{tol}pctBaselineRMSE"] = (g["RMSE_Loss_vs_J05_L10_pct"] <= float(tol)).astype(int)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def sparse_recommendations(pareto: pd.DataFrame) -> pd.DataFrame:
    if pareto.empty:
        return pd.DataFrame()
    rows = []
    for (H, model), g in pareto.groupby(["Horizon", "Model"], sort=True):
        best = g.sort_values(["RMSE", "HistoryScalarInputs"]).iloc[0]
        row: Dict[str, object] = {
            "Horizon": int(H), "Model": str(model),
            "BestRMSE_Config": str(best["JL_Config"]),
            "BestRMSE": float(best["RMSE"]),
            "BestRMSE_Diagnostics": int(best["DiagnosticCount"]),
            "BestRMSE_Scalars": int(best["HistoryScalarInputs"]),
        }
        for tol in (1, 2, 5, 10):
            q = g[g[f"Within{tol}pctBaselineRMSE"].eq(1)].copy()
            if q.empty:
                row[f"MinDiagWithin{tol}pct_Config"] = "NONE"
                row[f"MinDiagWithin{tol}pct_Diagnostics"] = np.nan
                row[f"MinDiagWithin{tol}pct_Scalars"] = np.nan
                row[f"MinDiagWithin{tol}pct_RMSE"] = np.nan
            else:
                z = q.sort_values(["DiagnosticCount", "RMSE", "HistoryScalarInputs"]).iloc[0]
                row[f"MinDiagWithin{tol}pct_Config"] = str(z["JL_Config"])
                row[f"MinDiagWithin{tol}pct_Diagnostics"] = int(z["DiagnosticCount"])
                row[f"MinDiagWithin{tol}pct_Scalars"] = int(z["HistoryScalarInputs"])
                row[f"MinDiagWithin{tol}pct_RMSE"] = float(z["RMSE"])
        q40 = g[g["HistoryScalarInputs"].le(40)].copy()
        if len(q40):
            z = q40.sort_values(["RMSE", "HistoryScalarInputs"]).iloc[0]
            row["BestAtOrBelow40Scalars_Config"] = str(z["JL_Config"])
            row["BestAtOrBelow40Scalars_RMSE"] = float(z["RMSE"])
            row["BestAtOrBelow40Scalars_Scalars"] = int(z["HistoryScalarInputs"])
            row["BestAtOrBelow40Scalars_Diagnostics"] = int(z["DiagnosticCount"])
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 3. Main
# =============================================================================

def main() -> None:
    args = parse_args()
    t_all = time.perf_counter()
    horizons = sorted(set(int(x) for x in args.horizons))
    configs = list(dict.fromkeys(args.configs))
    models = list(dict.fromkeys(args.models))
    if any(h < 1 for h in horizons) or not configs or not models:
        raise ValueError("horizons/configs/models 参数非法")

    out = ensure_dir(args.output_root)
    cache_root = ensure_dir(out / "Cache_Folds")

    v48b_path = resolve_script(
        args.code_dir,
        "25_fixedtime_sparse_search_helper.py",
        "*V4_8B*JL*LOBO*py",
    )
    v47e_path = resolve_script(
        args.code_dir,
        "05_rank_hi_topk_ablation.py",
        "*V4_7E*Event*HI*LOBO*py",
    )
    v48b = load_module(v48b_path, "rwth48_v48b_helper_for_v48e")
    v47e = load_module(v47e_path, "rwth48_v47e_helper_for_v48e")

    # Critical: reuse V4.8B training/window logic but replace its representation globals only.
    v48b.TOP2_HI = EVENT_TOP2
    v48b.TOP2_COLUMNS = EVENT_TOP2_COLUMNS
    v48b.JL_CONFIGS = JL_CONFIGS
    v48b.PRIMARY_CONFIGS = PRIMARY_CONFIGS
    v48b.BASELINE_CONFIG = BASELINE_CONFIG

    xgb_info = configure_xgboost_backend(
        v48b, args.xgb_device, args.xgb_gpu_id, args.threads_model
    ) if "XGBoost" in models else {
        "Requested": args.xgb_device, "Selected": "NOT_REQUESTED",
        "GPU_ID": int(args.xgb_gpu_id), "XGBoostAvailable": False,
        "XGBoostVersion": "", "Probe": "NOT_REQUESTED", "FallbackReason": "",
    }

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(args.threads_model))))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(max(1, int(args.threads_model))))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    v48b.set_seed(20260821)
    if args.audit_only:
        # Input/window audit must not require TensorFlow merely because deep models are in the formal model list.
        tf_info = {"TensorFlowRequiredForAudit": False, "RequestedModels": models}
    else:
        tf_info = v48b.configure_tensorflow_if_needed(models)

    event_file = resolve_event_file(args.base, args.event_file, v47e)
    config_json = {
        "Task": "RWTH48 V4.8E Event Top2 RF/XGBoost J/L matched-span Pareto ablation (accelerator-aware)",
        "StrictInput": str(args.strict_input),
        "EventFile": str(event_file),
        "V47EOOF": str(args.v47e_oof),
        "Output": str(out),
        "EventCheckpoints": [x[1] for x in EVENT_NODES],
        "FixedEventTop2": list(EVENT_TOP2),
        "JLConfigs": {k: {"J": v[0], "L": v[1], "Role": v[2]} for k, v in JL_CONFIGS.items()},
        "ConfigsRequested": configs,
        "PrimaryMatchedSpanConfigs": list(PRIMARY_CONFIGS),
        "Horizons": horizons,
        "Models": models,
        "CPUOnly": False,
        "AcceleratorPolicy": "XGBoost auto CUDA/CPU; RF remains CPU by algorithm",
        "XGBoostBackend": xgb_info,
        "ThreadsModel": int(args.threads_model),
        "HelperV48B": str(v48b_path),
        "HelperV47E": str(v47e_path),
    }
    save_json(config_json, out / "00_config.json")

    print("=" * 118)
    print("RWTH48 V4.8E | Event Top2 × J/L matched-span Pareto | accelerator-aware")
    print("STRICT   :", args.strict_input)
    print("EVENT    :", event_file)
    print("V4.7E OOF:", args.v47e_oof)
    print("OUTPUT   :", out)
    print("TOP2     :", EVENT_TOP2)
    print("CONFIGS  :", configs)
    print("HORIZONS :", horizons)
    print("MODELS   :", models)
    print("XGB backend:", xgb_info)
    print("TF info    :", tf_info)
    print("=" * 118, flush=True)

    # A. frozen inputs
    strict = v48b.load_strict_target(args.strict_input)
    header = pd.read_csv(event_file, nrows=0).columns.tolist()
    missing_hi = [x for x in EVENT_TOP2 if x not in header]
    if missing_hi:
        raise RuntimeError("V4.2 Event 文件缺少冻结 Event Top2：" + str(missing_hi))
    wide, node_audit = v47e.load_event_wide(event_file, EVENT_TOP2)
    node_audit.to_csv(out / "01_event_node_audit.csv", index=False, encoding="utf-8-sig")

    manifest_audit = validate_v47e_manifest(args.v47e_manifest, horizons)
    manifest_audit.to_csv(out / "01b_v47e_top2_manifest_audit.csv", index=False, encoding="utf-8-sig")
    if (manifest_audit["PASS"].astype(int) == 0).any():
        raise RuntimeError("V4.7E Top2 manifest 与冻结 Event Top2 不一致：\n" + manifest_audit.to_string(index=False))

    merged = strict[["Battery", "GlobalCycle", SEGMENT, TARGET]].merge(
        wide[["Battery", "GlobalCycle", *EVENT_TOP2_COLUMNS]],
        on=["Battery", "GlobalCycle"], how="inner", validate="one_to_one",
    )
    batteries = sorted(strict["Battery"].unique())
    if args.only_battery:
        b = normalize_battery(args.only_battery)
        if b not in batteries:
            raise ValueError(f"--only-battery不存在：{b}")
        outer_bats = [b]
    else:
        outer_bats = batteries

    compat_rows = [
        {"Check": "47 strict batteries", "Observed": len(batteries), "Expected": 47, "PASS": int(len(batteries) == 47)},
        {"Check": "3 Event checkpoints", "Observed": len(node_audit), "Expected": 3, "PASS": int(len(node_audit) == 3)},
        {"Check": "Event Top2 columns", "Observed": len(EVENT_TOP2_COLUMNS), "Expected": 6, "PASS": int(len(EVENT_TOP2_COLUMNS) == 6)},
        {"Check": "Strict/Event common cycles", "Observed": len(merged), "Expected": ">0", "PASS": int(len(merged) > 0)},
    ]
    compat = pd.DataFrame(compat_rows)
    compat.to_csv(out / "01_input_compatibility_audit.csv", index=False, encoding="utf-8-sig")
    if not bool((compat["PASS"].astype(int) == 1).all()):
        raise RuntimeError("输入兼容性审计失败：\n" + compat.to_string(index=False))

    # B. exact windows
    bundles = {}
    inv_rows = []
    for cfg in configs:
        J, L, role = JL_CONFIGS[cfg]
        for H in horizons:
            bun = v48b.build_sparse_history_windows(merged, J, L, H)
            bundles[(cfg, H)] = bun
            vc = pd.Series(bun.battery).value_counts() if len(bun.y) else pd.Series(dtype=int)
            inv_rows.append({
                "JL_Config": cfg, "Role": role, "J": J, "L": L,
                "HistorySpan_cycles": (L - 1) * J,
                "DiagnosticCount": L,
                "EventHI_Count": len(EVENT_TOP2),
                "EventCheckpoints": len(EVENT_NODES),
                "HistoryScalarInputs": L * len(EVENT_TOP2_COLUMNS),
                "Horizon": int(H), "TotalWindows": int(len(bun.y)),
                "BatteriesWithWindows": int(len(vc)),
                "MinWindowsPerBattery": int(vc.min()) if len(vc) else 0,
                "MedianWindowsPerBattery": float(vc.median()) if len(vc) else 0.0,
                "MaxWindowsPerBattery": int(vc.max()) if len(vc) else 0,
                "Status": "AVAILABLE" if len(bun.y) else "NO_WINDOWS",
            })
    availability = pd.DataFrame(inv_rows)
    availability.to_csv(out / "02_JL_availability_inventory.csv", index=False, encoding="utf-8-sig")

    # V4.7E baseline reuse and exact audit
    baseline = pd.DataFrame()
    baseline_audit_rows = []
    cuda_xgb_selected = ("XGBoost" in models and xgb_info.get("Selected") == "cuda")
    can_reuse = (
        BASELINE_CONFIG in configs
        and not args.retrain_baseline
        and args.v47e_oof.exists()
        and not cuda_xgb_selected
    )
    if can_reuse:
        baseline = load_v47e_top2(args.v47e_oof, models, horizons, outer_bats)
        for H in horizons:
            bun = bundles.get((BASELINE_CONFIG, H))
            if bun is not None and len(bun.y):
                baseline_audit_rows.append(audit_baseline_against_bundle(v48b, baseline, bun, H, outer_bats))
        ba = pd.DataFrame(baseline_audit_rows)
        ba.to_csv(out / "02b_V47E_Top2_exact_baseline_audit.csv", index=False, encoding="utf-8-sig")
        if len(ba) and not bool((ba["PASS"].astype(int) == 1).all()):
            raise RuntimeError("V4.7E Top2 baseline 与重建 J05_L10 cohort 不一致：\n" + ba.to_string(index=False))
        print("    V4.7E Event Top2 baseline: exact PASS, will reuse.", flush=True)
    else:
        reason = "CUDA XGBoost selected -> retrain baseline with same backend" if cuda_xgb_selected else "baseline unavailable/retrain requested"
        print(f"    V4.7E Event Top2 baseline not reused: {reason}.", flush=True)

    # Pre-training common cohort inventory
    primary_pre = []
    if set(PRIMARY_CONFIGS).issubset(set(configs)):
        for H in horizons:
            sets = []
            for cfg in PRIMARY_CONFIGS:
                q = v48b.bundle_keys(bundles[(cfg, H)], H)
                sets.append(set(map(tuple, q[KEY].to_numpy(object))))
            common = set.intersection(*sets) if sets else set()
            primary_pre.append({
                "Horizon": int(H), "Configs": "|".join(PRIMARY_CONFIGS),
                "CommonWindows_PreTraining": int(len(common)),
                "CommonBatteries_PreTraining": int(len(set(x[1] for x in common))) if common else 0,
            })
    pd.DataFrame(primary_pre).to_csv(out / "03_primary_common_cohort_inventory.csv", index=False, encoding="utf-8-sig")

    if args.audit_only:
        print("\n--audit-only PASS")
        print(availability.to_string(index=False))
        if primary_pre:
            print("\nPrimary exact common cohort:")
            print(pd.DataFrame(primary_pre).to_string(index=False))
        return

    # C. LOBO
    pred_parts: List[pd.DataFrame] = []
    runtime_rows: List[Dict[str, object]] = []
    if can_reuse and not baseline.empty:
        pred_parts.append(baseline)

    for cfg in configs:
        J, L, role = JL_CONFIGS[cfg]
        for H in horizons:
            bun = bundles[(cfg, H)]
            if len(bun.y) == 0:
                print(f"  {cfg} | H={H}: NO WINDOWS -> skip", flush=True)
                continue
            if cfg == BASELINE_CONFIG and can_reuse:
                print(f"  {cfg} | H={H}: reused V4.7E Event Top2 OOF", flush=True)
                continue
            available_bats = sorted(set(bun.battery.tolist()))
            for fi, test_bat in enumerate(outer_bats, start=1):
                if test_bat not in available_bats:
                    continue
                fold_dir = ensure_dir(cache_root / cfg / f"H{H:03d}" / test_bat)
                pc = fold_dir / "predictions.csv.gz"
                mc = fold_dir / "metrics.csv"
                print(
                    f"  {cfg} (J={J},L={L},span={(L-1)*J}) | H={H:>3} | "
                    f"fold {fi:02d}/{len(outer_bats):02d} | held-out={test_bat}", flush=True,
                )
                t0 = time.perf_counter()
                p, m = v48b.run_fold(bun, cfg, H, test_bat, models, args, pc, mc)
                if not p.empty:
                    p["Source"] = "trained_V4.8E_EventTop2"
                    pred_parts.append(p)
                    runtime_rows.append({
                        "JL_Config": cfg, "J": J, "L": L, "Horizon": H,
                        "TestBattery": test_bat, "FoldWall_s": float(time.perf_counter() - t0),
                        "PredictionCache": str(pc),
                    })
                    msg = " | ".join(f"{r.Model}:RMSE={float(r.RMSE):.6f}" for r in m.itertuples(index=False))
                    print("      " + msg, flush=True)

    if not pred_parts:
        raise RuntimeError("没有任何 OOF 预测")
    oof = pd.concat(pred_parts, ignore_index=True, sort=False)
    oof = oof[oof["JL_Config"].isin(configs) & oof["Horizon"].isin(horizons)].copy()
    oof["Battery"] = oof["Battery"].map(normalize_battery)
    oof = oof.sort_values(["JL_Config", "Horizon", "Battery", "OriginCycle_KeyOnly"]).reset_index(drop=True)
    if oof.duplicated(["JL_Config", *KEY]).any():
        raise RuntimeError("OOF 存在重复 JL_Config + exact key")

    # D. exact common-cohort metrics
    fold_metrics = v48b.melt_fold_metrics_from_oof(oof, models)
    own = v48b.own_cohort_pooled(oof, models)
    pairwise = v48b.pairwise_common_metrics(oof, configs, horizons, models) if BASELINE_CONFIG in configs else pd.DataFrame()
    primary_inv, primary_met, primary_pred = v48b.primary_common_outputs(oof, horizons, models)
    gain = event_gain(primary_met)
    pareto = pareto_frontier(gain)
    recommendations = sparse_recommendations(pareto)
    efficiency = pareto.sort_values(
        ["Horizon", "Model", "Within5pctBaselineRMSE", "DiagnosticCount", "RMSE"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True) if not pareto.empty else pd.DataFrame()
    runtime = pd.DataFrame(runtime_rows)

    # Add explicit Event metadata to outputs.
    for df in (fold_metrics, own, pairwise, primary_met, gain, efficiency, pareto, recommendations):
        if not df.empty:
            df["Platform"] = "Event_RefAh"
            df["FixedEventHI_Count"] = len(EVENT_TOP2)
            df["EventCheckpoints"] = len(EVENT_NODES)

    fold_metrics.to_csv(out / "04_fold_model_metrics.csv", index=False, encoding="utf-8-sig")
    own.to_csv(out / "05_own_cohort_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(out / "06_pairwise_common_cohort_metrics.csv", index=False, encoding="utf-8-sig")
    primary_met.to_csv(out / "07_matched_span_common_metrics.csv", index=False, encoding="utf-8-sig")
    gain.to_csv(out / "08_gain_vs_J05_L10_exact_common.csv", index=False, encoding="utf-8-sig")
    efficiency.to_csv(out / "09_measurement_efficiency_summary.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(out / "10_pareto_frontier.csv", index=False, encoding="utf-8-sig")
    recommendations.to_csv(out / "11_recommended_sparse_configs.csv", index=False, encoding="utf-8-sig")
    runtime.to_csv(out / "12_runtime_summary.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(out / "13_ALL_OOF_EventTop2_JL_Pareto.csv.gz", index=False, compression=FAST_GZIP)
    if not primary_pred.empty:
        primary_pred.to_csv(out / "15_matched_span_common_predictions.csv.gz", index=False, compression=FAST_GZIP)

    # Acceptance gate
    gate_rows: List[Dict[str, object]] = []
    def gate(check: str, ok: bool, detail: object) -> None:
        gate_rows.append({"Check": check, "PASS": int(bool(ok)), "Detail": str(detail)})

    gate("47 strict batteries", len(batteries) == 47, len(batteries))
    gate("3 Event checkpoints", len(node_audit) == 3, len(node_audit))
    gate("Event Top2 fixed", EVENT_TOP2 == ("V_TimeWeightedMean", "P_Last"), EVENT_TOP2)
    gate("No duplicate config OOF keys", not oof.duplicated(["JL_Config", *KEY]).any(), len(oof))
    finite_ok = True; detail = []
    for m in models:
        c = f"Pred_{m}"
        if c not in oof.columns:
            finite_ok = False; detail.append(f"{m}:missing")
        else:
            bad = int((~np.isfinite(pd.to_numeric(oof[c], errors="coerce").to_numpy(float))).sum())
            finite_ok = finite_ok and bad == 0
            detail.append(f"{m}:bad={bad}")
    gate("All requested predictions finite", finite_ok, "; ".join(detail))
    if set(PRIMARY_CONFIGS).issubset(set(configs)):
        ok_primary = len(primary_inv) == len(horizons) and bool((primary_inv["PASS"].astype(int) == 1).all())
        gate("Primary matched-span exact common cohort", ok_primary, primary_inv.to_dict("records"))
    if can_reuse:
        ba = pd.DataFrame(baseline_audit_rows)
        gate("Reused V4.7E Top2 exact compatibility", len(ba) == len(horizons) and bool((ba["PASS"].astype(int) == 1).all()), ba.to_dict("records"))
    if not args.only_battery:
        base_rows = oof[oof["JL_Config"].eq(BASELINE_CONFIG)] if BASELINE_CONFIG in configs else pd.DataFrame()
        gate("Formal baseline covers 47 held-out batteries", BASELINE_CONFIG not in configs or base_rows["Battery"].nunique() == 47,
             base_rows["Battery"].nunique() if len(base_rows) else 0)

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(out / "14_acceptance_gate.csv", index=False, encoding="utf-8-sig")

    if not args.skip_figures and not primary_met.empty:
        try:
            v48b.make_figures(primary_met, out)
        except Exception as exc:
            print("[WARN] figure generation skipped:", repr(exc), flush=True)

    status = "PASS" if (not args.only_battery and gate_df["PASS"].eq(1).all()) else "SMOKE_PASS" if gate_df["PASS"].eq(1).all() else "CHECK"
    best_rows = []
    if not primary_met.empty:
        for (H, model), g in primary_met.groupby(["Horizon", "Model"], sort=True):
            z = g.sort_values("RMSE").iloc[0]
            best_rows.append({
                "Horizon": int(H), "Model": str(model), "BestJL": str(z["JL_Config"]),
                "J": int(z["J"]), "L": int(z["L"]),
                "RMSE": float(z["RMSE"]), "MAE": float(z["MAE"]),
                "CommonWindows": int(z.get("CommonWindows", z.get("N", 0))),
            })

    complete = {
        "Task": "RWTH48 V4.8E Event Top2 RF/XGBoost J/L matched-span Pareto ablation (accelerator-aware)",
        "Status": status,
        "Batteries": len(batteries),
        "OuterTestBatteriesRun": len(outer_bats),
        "EventTop2": list(EVENT_TOP2),
        "EventCheckpoints": [x[1] for x in EVENT_NODES],
        "ConfigsRequested": configs,
        "PrimaryMatchedSpanConfigs": list(PRIMARY_CONFIGS),
        "Horizons": horizons,
        "Models": models,
        "V47ETop2BaselineReused": bool(can_reuse),
        "OnlyHIAsModelInput": True,
        "CycleAsFeature": False,
        "HistoricalSOHAsFeature": False,
        "PeriodicSOHCalibration": False,
        "NearestCycleSubstitution": False,
        "BestPrimaryMatchedSpanByHorizonModel": best_rows,
        "SparseRecommendations": recommendations.to_dict("records") if not recommendations.empty else [],
        "OOFRows": int(len(oof)),
        "Elapsed_s": float(time.perf_counter() - t_all),
        "RuntimeBackendAudit": {"TensorFlow": tf_info, "XGBoost": xgb_info},
        "Important": [
            "Only J and L change; Event RefAh checkpoints and Event Top2 remain frozen.",
            "J/L are scheduling variables only and never enter model X.",
            "Primary comparison uses one exact common Battery/Origin/Target cohort across all six approximately matched-span J/L configurations.",
            "No nearest-cycle substitution is allowed.",
            "V4.7E J05_L10 Event Top2 OOF is reused only after exact key/SOH truth checks and only when the current XGBoost backend is CPU-compatible; CUDA XGBoost forces same-backend baseline retraining.",
            "Event Top2 has 6 scalar values per diagnosis; J10_L05 uses 30 historical scalars, below the FixedTime Top2 J10_L05 budget of 40.",
            "Pareto/recommendation tables are computed only on the exact common cohort shared by all requested matched-span configurations.",
            "The script does not mask CUDA. XGBoost auto mode prefers CUDA and falls back to CPU only if the CUDA probe fails; --xgb-device cuda can force CUDA.",
        ],
    }
    save_json(complete, out / "99_RWTH48_V4_8E_COMPLETE.json")

    print("\n" + "=" * 118)
    print("RWTH48 V4.8E COMPLETE")
    print("Status  :", status)
    print("OOF rows:", len(oof))
    print("Output  :", out)
    print("Elapsed :", f"{complete['Elapsed_s']/3600:.2f} h")
    print("=" * 118)


if __name__ == "__main__":
    main()

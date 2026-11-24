# -*- coding: utf-8 -*-
r"""
RWTH48 V4.7E | Event HI 数量消融（独立于 FixedTime V4.7）
=========================================================

研究目的
--------
对 RWTH48 的 RefAh Event 平台独立回答：
    “使用多少种 Event HI，才能获得稳定的跨电池未来 SOH 预测能力？”

本脚本与 FixedTime V4.7 独立：
- Event 输入来自 V4.2 冻结的 RefAh 事件节点：0.05 / 0.09 / 0.13；
- 未来 SOH / StrictSegmentKey 来自 V4.5C-FIX；
- 每个 outer LOBO fold 内只用 46 source batteries 对 Event HI 做未来任务相关排名；
- held-out battery 不参与 ranking / imputation / scaling / fitting；
- J=5, L=10, H=10/50/100 默认与 FixedTime 主线一致，便于后续公平比较；
- Cycle 只作 exact J/H 调度键，不进入 X；
- 不使用历史 SOH、RPT SOH、capacity、future HI。

Event 表示
---------
每个历史诊断 cycle 使用三个累计物理事件节点：
    RefAh 0.05 / 0.09 / 0.13
对每个候选 base HI 形成：
    [E005__HI, E009__HI, E013__HI]
然后 L=10 个 exact J-spaced 历史 cycle 构成序列。

候选 HI
-------
直接从 V4.2 Event 冻结表头推断，并复用 V4.3 的“设计级合法性”排除规则：
- 元数据 / 标签 / QC shortcut 不作为模型输入；
- SampleCount / ObservedHardInvalidRate / ObservedSpikeRate 禁用；
- PartialThroughput_Wh（与 P_AreaAbs_Wh 结构等价）禁用；
- V4.2 已确认全局恒定的 3 个电压 band ratio 禁用。

注意：本任务不读取 V4.3 的全数据 consensus 作为同轮特征集，避免 same-round LOBO 泄漏。

默认 TopK：1 / 2 / 3 / 5 / 8。
可用 --subset-sizes 修改。

建议并行运行方式（同一 RTX 2070）：
1) 本脚本先只跑 CPU 树模型：
   python "src/05_rank_hi_topk_ablation.py" --models RF XGBoost
2) 同时运行 Disjoint+Hierarchical 深度模型任务；
3) Hierarchical 任务结束后，再补：
   python "src/05_rank_hi_topk_ablation.py" --models LSTM BiLSTM
模型 cache 按 model 独立保存，第二次会保留第一次结果并重新聚合。
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. Paths / frozen settings
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR = Path(__file__).resolve().parent

STRICT_DEFAULT = (
    BASE
    / "RWTH48_V4_5C_FIX_最小冲击共同扰动处理"
    / "06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz"
)

OUTPUT_DEFAULT = BASE / "RWTH48_V4_7E_Event_HI数量消融_SourceOnly_LOBO_LITE"

EVENT_NODES = (
    (0.05, "Event_RefAh_0.05", "E005"),
    (0.09, "Event_RefAh_0.09", "E009"),
    (0.13, "Event_RefAh_0.13", "E013"),
)

TARGET = "CyclingSOH_Charge_Canonical"
SEGMENT = "StrictSegmentKey"
J_DEFAULT = 5
L_DEFAULT = 10
H_DEFAULT = (10, 50, 100)
TOPK_DEFAULT = (1, 2, 3, 5, 8)
MODELS_ALL = ("RF", "XGBoost", "LSTM", "BiLSTM")
SEED = 20260821
FAST_GZIP = {"method": "gzip", "compresslevel": 1}

# Same design-level exclusions used by V4.3.
NON_FEATURE_COLS = {
    "Battery", "GlobalCycle", "Scope", "Platform", "NodeType", "NodeValue", "PlatformNode",
    "ReferenceCapacity_Ah", "ReferenceCapacity_Ah_QCOnly",
    "SOH_Target_SameCycle", "SOH_IsMeasuredCheckUp", "SOH_LabelUsable", "SOH_LabelSource",
    "EventReachable", "PrefixFeatureValid", "MissingFeatureCount", "Reachable",
    "SourcePart", SEGMENT, TARGET,
}
DESIGN_FORBIDDEN = {"SampleCount", "ObservedHardInvalidRate", "ObservedSpikeRate"}
STRUCTURAL_REDUNDANT_DROP = {"PartialThroughput_Wh"}
GLOBAL_CONSTANT_DROP = {
    "VBand_3p0_3p2_Ratio",
    "VBand_3p2_3p4_Ratio",
    "VBand_4p0_4p2_Ratio",
}


# =============================================================================
# 1. Generic helpers
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


def resolve_v47_helper(code_dir: Path) -> Path:
    exact = code_dir / "19_fixedtime_ranking_helper.py"
    if exact.exists():
        return exact
    matches = sorted(code_dir.glob("*V4_7*HI数量消融*SourceOnly*LOBO*py"))
    if not matches:
        raise FileNotFoundError(
            "缺少 FixedTime V4.7 helper 脚本。请把 19_fixedtime_ranking_helper.py "
            f"放入：{code_dir}"
        )
    return matches[0]


def resolve_event_file(base: Path, explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    preferred = (
        base
        / "RWTH48_V4_2_双平台冻结_FixedTime_Event_HI"
        / "06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz"
    )
    if preferred.exists():
        return preferred

    hits = sorted(base.glob("RWTH48_V4_2*/06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        "Event HI 消融缺少 V4.2 Event 冻结数据。\n"
        "请从旧电脑额外复制文件夹：RWTH48_V4_2_双平台冻结_FixedTime_Event_HI\n"
        "其中至少需要：06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz\n"
        f"当前搜索根目录：{base}"
    )


def infer_event_candidates(event_file: Path) -> Tuple[List[str], pd.DataFrame]:
    cols = pd.read_csv(event_file, nrows=0).columns.tolist()
    rows: List[Dict[str, object]] = []
    keep: List[str] = []

    for c in cols:
        enabled = 1
        reason = "CANDIDATE"
        if c in NON_FEATURE_COLS:
            enabled, reason = 0, "METADATA_OR_LABEL"
        elif c in DESIGN_FORBIDDEN:
            enabled, reason = 0, "DESIGN_FORBIDDEN_QC_PROXY"
        elif c in STRUCTURAL_REDUNDANT_DROP:
            enabled, reason = 0, "STRUCTURAL_REDUNDANCY"
        elif c in GLOBAL_CONSTANT_DROP:
            enabled, reason = 0, "V4P2_GLOBAL_CONSTANT"
        elif c.startswith("SOH_") or c.startswith("Pred_"):
            enabled, reason = 0, "TARGET_OR_PREDICTION_LIKE"
        elif "Capacity" in c and c not in {"PartialCapacity_Ah"}:
            # Capacity-like absolute/reference quantities are not admitted by default.
            enabled, reason = 0, "CAPACITY_OR_REFERENCE_LIKE"

        if enabled:
            keep.append(c)
        rows.append({"Feature": c, "Enabled": enabled, "Reason": reason})

    if not keep:
        raise RuntimeError("Event 文件未推断出任何合法候选 HI")
    return keep, pd.DataFrame(rows)


def load_event_wide(event_file: Path, candidates: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usecols = ["Battery", "GlobalCycle", "PlatformNode", *candidates]
    d = pd.read_csv(event_file, usecols=lambda c: c in set(usecols), compression="infer", low_memory=False)

    required = {"Battery", "GlobalCycle", "PlatformNode"}
    if not required.issubset(d.columns):
        raise RuntimeError(f"Event 文件缺关键列：{sorted(required - set(d.columns))}")

    d["Battery"] = d["Battery"].map(normalize_battery)
    d["GlobalCycle"] = pd.to_numeric(d["GlobalCycle"], errors="coerce")
    d = d[d["GlobalCycle"].notna()].copy()
    d["GlobalCycle"] = d["GlobalCycle"].astype(int)
    d = d[d["PlatformNode"].isin([x[1] for x in EVENT_NODES])].copy()

    for c in candidates:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = np.nan

    if d.duplicated(["Battery", "GlobalCycle", "PlatformNode"]).any():
        dup = d[d.duplicated(["Battery", "GlobalCycle", "PlatformNode"], keep=False)]
        raise RuntimeError(
            "Event 冻结表存在重复 Battery+GlobalCycle+PlatformNode：\n"
            + dup[["Battery", "GlobalCycle", "PlatformNode"]].head(20).to_string(index=False)
        )

    parts = []
    node_audit = []
    for value, node, prefix in EVENT_NODES:
        g = d[d["PlatformNode"].eq(node)][["Battery", "GlobalCycle", *candidates]].copy()
        rename = {c: f"{prefix}__{c}" for c in candidates}
        g.rename(columns=rename, inplace=True)
        parts.append(g)
        node_audit.append({
            "NodeValue": value,
            "PlatformNode": node,
            "Rows": len(g),
            "Batteries": g["Battery"].nunique(),
            "Cycles": g[["Battery", "GlobalCycle"]].drop_duplicates().shape[0],
        })

    wide = parts[0]
    for g in parts[1:]:
        wide = wide.merge(g, on=["Battery", "GlobalCycle"], how="inner", validate="one_to_one")

    if wide.empty:
        raise RuntimeError("三个 Event checkpoint 没有共同 cycle")

    return wide.sort_values(["Battery", "GlobalCycle"]).reset_index(drop=True), pd.DataFrame(node_audit)


def selected_indices(all_hi_columns: Sequence[str], selected_hi: Sequence[str]) -> List[int]:
    wanted = set(selected_hi)
    return [i for i, c in enumerate(all_hi_columns) if c.split("__", 1)[1] in wanted]


def source_rank(
    helper,
    bundle,
    base_hi: Sequence[str],
    all_hi_columns: Sequence[str],
    source_bats: Sequence[str],
    horizon: int,
    test_battery: str,
    min_availability: float,
) -> pd.DataFrame:
    idx = helper.balanced_indices(bundle.battery, source_bats, helper.RANK_PER_SOURCE_BATTERY)
    if len(idx) == 0:
        raise RuntimeError("source ranking 无窗口")
    X = bundle.X[idx]
    y = bundle.y[idx]

    rows = []
    order = {h: i for i, h in enumerate(base_hi)}
    for hi in base_hi:
        obs = [i for i, c in enumerate(all_hi_columns) if c.endswith("__" + hi)]
        Z = X[:, :, obs].reshape(len(X), -1)
        availability = float(np.isfinite(Z).mean()) if Z.size else 0.0
        vals = np.asarray([helper.abs_spearman_column(Z[:, j], y) for j in range(Z.shape[1])], dtype=float)
        finite = vals[np.isfinite(vals)]
        rows.append({
            "Horizon": int(horizon),
            "TestBattery": test_battery,
            "HI": hi,
            "Availability": availability,
            "Eligible": int(availability >= min_availability and len(finite) > 0),
            "MeanAbsSpearman": float(np.mean(finite)) if len(finite) else 0.0,
            "MedianAbsSpearman": float(np.median(finite)) if len(finite) else 0.0,
            "MaxAbsSpearman": float(np.max(finite)) if len(finite) else 0.0,
            "ValidPositions": int(len(finite)),
            "_order": order[hi],
        })

    r = pd.DataFrame(rows)
    r = r.sort_values(
        ["Eligible", "MeanAbsSpearman", "MedianAbsSpearman", "MaxAbsSpearman", "_order"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    r["Rank"] = np.arange(1, len(r) + 1)
    r.drop(columns="_order", inplace=True)
    return r


def model_cache_paths(cache_root: Path, h: int, bat: str, k: int, model: str) -> Tuple[Path, Path]:
    stem = f"H{h:03d}__{bat}__Top{k:02d}__{model}"
    return cache_root / f"{stem}.pred.csv.gz", cache_root / f"{stem}.metric.csv"


def run_one_model(
    helper,
    bundle,
    test_battery: str,
    horizon: int,
    selected_hi: Sequence[str],
    all_hi_columns: Sequence[str],
    model: str,
    args,
    pred_cache: Path,
    metric_cache: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if (
        not args.force_folds
        and pred_cache.exists() and pred_cache.stat().st_size > 0
        and metric_cache.exists() and metric_cache.stat().st_size > 0
    ):
        try:
            p = pd.read_csv(pred_cache, compression="infer")
            m = pd.read_csv(metric_cache)
            if f"Pred_{model}" in p.columns and len(m) == 1 and str(m.iloc[0]["Model"]) == model:
                return p, m
        except Exception:
            pass

    all_bats = sorted(set(bundle.battery.tolist()))
    source_bats = [b for b in all_bats if b != test_battery]
    test_idx = np.flatnonzero(bundle.battery == test_battery)
    if len(test_idx) == 0:
        raise RuntimeError(f"{test_battery} H={horizon}: 无测试窗口")

    obs_idx = selected_indices(all_hi_columns, selected_hi)
    if len(obs_idx) != len(EVENT_NODES) * len(selected_hi):
        raise RuntimeError(f"Event selected HI 维度异常：{selected_hi}, obs={len(obs_idx)}")

    Xsub = bundle.X[:, :, obs_idx]
    deep_idx = helper.balanced_indices(bundle.battery, source_bats, helper.DEEP_PER_SOURCE_BATTERY)
    rf_idx = helper.balanced_indices(bundle.battery, source_bats, helper.RF_PER_SOURCE_BATTERY)
    xgb_idx = helper.balanced_indices(bundle.battery, source_bats, helper.XGB_PER_SOURCE_BATTERY)

    # One source-only scaler policy shared across all models, matching V4.7.
    scaler = helper.fit_scaler(Xsub[deep_idx])
    Xte = scaler.transform(Xsub[test_idx])
    yte = bundle.y[test_idx]

    if model == "RF":
        Xtr = scaler.transform(Xsub[rf_idx]).reshape(len(rf_idx), -1)
        pred, runtime = helper.train_predict_rf(Xtr, bundle.y[rf_idx], Xte.reshape(len(Xte), -1), args.threads_model)
        ntrain = len(rf_idx)
    elif model == "XGBoost":
        Xtr = scaler.transform(Xsub[xgb_idx]).reshape(len(xgb_idx), -1)
        pred, runtime = helper.train_predict_xgb(Xtr, bundle.y[xgb_idx], Xte.reshape(len(Xte), -1), args.threads_model)
        ntrain = len(xgb_idx)
    elif model in {"LSTM", "BiLSTM"}:
        Xtr = scaler.transform(Xsub[deep_idx])
        seed = SEED + horizon * 1000 + len(selected_hi) * 100 + (0 if model == "LSTM" else 1)
        pred, runtime = helper.train_predict_deep(model, Xtr, bundle.y[deep_idx], Xte, args.deep_epochs, args.deep_batch, seed)
        ntrain = len(deep_idx)
    else:
        raise ValueError(model)

    selected_text = "|".join(selected_hi)
    p = pd.DataFrame({
        "Horizon": int(horizon),
        "HI_Set": f"Top{len(selected_hi)}",
        "HI_Count": int(len(selected_hi)),
        "SelectedHI": selected_text,
        "TestBattery": test_battery,
        "Battery": bundle.battery[test_idx],
        "OriginCycle_KeyOnly": bundle.origin_cycle[test_idx],
        "TargetCycle_KeyOnly": bundle.target_cycle[test_idx],
        "StrictSegmentKey_QCOnly": bundle.segment[test_idx],
        "SOH_True": yte,
        f"Pred_{model}": pred,
    })
    m = pd.DataFrame([{
        "Horizon": int(horizon),
        "HI_Set": f"Top{len(selected_hi)}",
        "HI_Count": int(len(selected_hi)),
        "SelectedHI": selected_text,
        "TestBattery": test_battery,
        "Model": model,
        "TrainSourceBatteries": len(source_bats),
        "TrainWindows": int(ntrain),
        "TestWindows": len(test_idx),
        "InputFeaturesPerObservation": len(obs_idx),
        "HistoryScalarInputs": len(obs_idx) * bundle.X.shape[1],
        "Runtime_s": float(runtime),
        **helper.calc_metrics(yte, pred),
    }])

    ensure_dir(pred_cache.parent)
    p.to_csv(pred_cache, index=False, compression=FAST_GZIP)
    m.to_csv(metric_cache, index=False, encoding="utf-8-sig")
    del Xsub, Xte
    gc.collect()
    return p, m


def aggregate_prediction_caches(cache_root: Path) -> pd.DataFrame:
    files = sorted(cache_root.glob("*.pred.csv.gz"))
    if not files:
        return pd.DataFrame()

    key = [
        "Horizon", "HI_Set", "HI_Count", "SelectedHI", "TestBattery", "Battery",
        "OriginCycle_KeyOnly", "TargetCycle_KeyOnly", "StrictSegmentKey_QCOnly", "SOH_True",
    ]
    groups: Dict[Tuple[int, str, int, str, str], pd.DataFrame] = {}
    for f in files:
        d = pd.read_csv(f, compression="infer", low_memory=False)
        if d.empty:
            continue
        gk = (
            int(d["Horizon"].iloc[0]), str(d["HI_Set"].iloc[0]), int(d["HI_Count"].iloc[0]),
            str(d["SelectedHI"].iloc[0]), str(d["TestBattery"].iloc[0]),
        )
        pred_cols = [c for c in d.columns if c.startswith("Pred_")]
        q = d[key + pred_cols].copy()
        if gk not in groups:
            groups[gk] = q
        else:
            groups[gk] = groups[gk].merge(q, on=key, how="outer", validate="one_to_one")

    return pd.concat(groups.values(), ignore_index=True) if groups else pd.DataFrame()


def pooled_metrics(helper, oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pred_cols = [c for c in oof.columns if c.startswith("Pred_")]
    for (h, hs, k), g in oof.groupby(["Horizon", "HI_Set", "HI_Count"], sort=True):
        for c in pred_cols:
            model = c.replace("Pred_", "")
            if not g[c].notna().any():
                continue
            rows.append({
                "Horizon": int(h), "HI_Set": str(hs), "HI_Count": int(k),
                "EventCheckpoints": len(EVENT_NODES),
                "InputFeaturesPerObservation": len(EVENT_NODES) * int(k),
                "HistoryScalarInputs": len(EVENT_NODES) * int(k) * L_DEFAULT,
                "Model": model, "Eval": "POOLED_OOF",
                **helper.calc_metrics(g["SOH_True"].to_numpy(float), g[c].to_numpy(float)),
            })
    return pd.DataFrame(rows)


def macro_metrics(helper, oof: pd.DataFrame) -> pd.DataFrame:
    fold_rows = []
    pred_cols = [c for c in oof.columns if c.startswith("Pred_")]
    for (h, hs, k, b), g in oof.groupby(["Horizon", "HI_Set", "HI_Count", "Battery"], sort=True):
        for c in pred_cols:
            if not g[c].notna().any():
                continue
            fold_rows.append({
                "Horizon": int(h), "HI_Set": str(hs), "HI_Count": int(k),
                "TestBattery": str(b), "Model": c.replace("Pred_", ""),
                **helper.calc_metrics(g["SOH_True"].to_numpy(float), g[c].to_numpy(float)),
            })
    fd = pd.DataFrame(fold_rows)
    if fd.empty:
        return fd
    rows = []
    for (h, hs, k, m), g in fd.groupby(["Horizon", "HI_Set", "HI_Count", "Model"], sort=True):
        rows.append({
            "Horizon": int(h), "HI_Set": hs, "HI_Count": int(k), "Model": m,
            "Folds": g["TestBattery"].nunique(),
            "MacroMean_RMSE": float(g["RMSE"].mean()),
            "MacroMedian_RMSE": float(g["RMSE"].median()),
            "MacroMean_MAE": float(g["MAE"].mean()),
            "MacroMean_R2": float(g["R2"].mean()),
            "WorstFold_RMSE": float(g["RMSE"].max()),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RWTH48 V4.7E Event source-only HI-count ablation")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--strict-input", type=Path, default=STRICT_DEFAULT)
    p.add_argument("--event-file", type=str, default="")
    p.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    p.add_argument("--horizons", nargs="+", type=int, default=list(H_DEFAULT))
    p.add_argument("--history-interval", type=int, default=J_DEFAULT)
    p.add_argument("--history-observations", type=int, default=L_DEFAULT)
    p.add_argument("--subset-sizes", nargs="+", type=int, default=list(TOPK_DEFAULT))
    p.add_argument("--models", nargs="+", choices=list(MODELS_ALL), default=list(MODELS_ALL))
    p.add_argument("--min-source-availability", type=float, default=0.80)
    p.add_argument("--threads-model", type=int, default=4)
    p.add_argument("--deep-epochs", type=int, default=12)
    p.add_argument("--deep-batch", type=int, default=256)
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--force-folds", action="store_true")
    return p.parse_args()


# =============================================================================
# 2. Main
# =============================================================================

def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    base = Path(args.base)
    out = ensure_dir(Path(args.output_root))
    cache = ensure_dir(out / "Cache_ModelByFold")

    helper_path = resolve_v47_helper(Path(__file__).resolve().parent)
    helper = load_module(helper_path, "rwth_v47_helper_event")

    event_file = resolve_event_file(base, args.event_file)
    strict_file = Path(args.strict_input)
    if not strict_file.exists():
        raise FileNotFoundError(strict_file)

    horizons = sorted(set(int(x) for x in args.horizons))
    subset_sizes = sorted(set(int(x) for x in args.subset_sizes if int(x) >= 1))
    J = int(args.history_interval)
    L = int(args.history_observations)
    if J < 1 or L < 2:
        raise ValueError("J>=1 且 L>=2")
    if not (0 < args.min_source_availability <= 1):
        raise ValueError("--min-source-availability 必须在 (0,1]")

    print("=" * 118)
    print("RWTH48 V4.7E | EVENT HI-count ablation | source-only LOBO")
    print("BASE     :", base)
    print("EVENT    :", event_file)
    print("STRICT   :", strict_file)
    print("OUTPUT   :", out)
    print("J / L    :", J, "/", L)
    print("H        :", horizons)
    print("TopK     :", subset_sizes)
    print("Models   :", args.models)
    print("=" * 118, flush=True)

    candidates, registry = infer_event_candidates(event_file)
    registry.to_csv(out / "01_event_candidate_registry.csv", index=False, encoding="utf-8-sig")

    wide, node_audit = load_event_wide(event_file, candidates)
    node_audit.to_csv(out / "02_event_node_audit.csv", index=False, encoding="utf-8-sig")

    # Monkey-patch the already-tested V4.7 window/model helper to Event 3-checkpoint schema.
    helper.CHECKPOINTS_S = tuple(x[0] for x in EVENT_NODES)
    helper.COMPACT_HI = tuple(candidates)
    all_hi_columns = tuple(
        f"{prefix}__{f}" for _, _, prefix in EVENT_NODES for f in candidates
    )
    helper.ALL_HI_COLUMNS = all_hi_columns
    helper.HISTORY_INTERVAL_J = J
    helper.HISTORY_OBSERVATIONS_L = L

    strict = helper.load_strict_target(strict_file)
    batteries = sorted(strict["Battery"].unique())
    if len(batteries) != 47:
        raise RuntimeError(f"预期47节 strict 电池，实际{len(batteries)}")

    merged = strict[["Battery", "GlobalCycle", SEGMENT, TARGET]].merge(
        wide[["Battery", "GlobalCycle", *all_hi_columns]],
        on=["Battery", "GlobalCycle"], how="inner", validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError("V4.2 Event HI 与 V4.5C-FIX strict target 无交集")

    input_audit = pd.DataFrame([{
        "StrictBatteries": len(batteries),
        "EventWideBatteries": wide["Battery"].nunique(),
        "EventWideCycles": len(wide),
        "StrictEventMergedCycles": len(merged),
        "CandidateBaseHI": len(candidates),
        "EventCheckpoints": len(EVENT_NODES),
        "RawFeaturesPerObservation": len(all_hi_columns),
    }])
    input_audit.to_csv(out / "00_input_audit.csv", index=False, encoding="utf-8-sig")

    save_json({
        "Task": "RWTH48 V4.7E Event HI-count ablation",
        "EventFile": str(event_file),
        "StrictInput": str(strict_file),
        "EventNodes": [x[0] for x in EVENT_NODES],
        "CandidatePoolSize": len(candidates),
        "J": J, "L": L, "Horizons": horizons, "SubsetSizes": subset_sizes,
        "ModelsRequestedThisRun": list(args.models),
        "Ranking": "source-only future-SOH absolute Spearman; availability-gated",
        "HeldOutUsedForRanking": False,
        "HistoricalSOHAsFeature": False,
        "CycleAsFeature": False,
        "FutureHI": False,
    }, out / "00_config.json")

    bundles = {}
    inventory = []
    for H in horizons:
        b = helper.build_sparse_history_windows(merged, J, L, H)
        bundles[H] = b
        inventory.append({
            "Horizon": H, "Windows": len(b.y), "Batteries": len(set(b.battery.tolist())),
            "OriginMin": int(np.min(b.origin_cycle)) if len(b.y) else np.nan,
            "OriginMax": int(np.max(b.origin_cycle)) if len(b.y) else np.nan,
        })
    inv = pd.DataFrame(inventory)
    inv.to_csv(out / "03_window_inventory.csv", index=False, encoding="utf-8-sig")
    if (inv["Windows"] <= 0).any():
        raise RuntimeError("至少一个 H 没有 Event sparse-history 窗口")

    if args.audit_only:
        print("\n--audit-only PASS--")
        print(input_audit.to_string(index=False))
        print(inv.to_string(index=False))
        return

    # TensorFlow is configured only when this run actually requests deep models.
    helper.configure_tensorflow_if_needed(args.models)

    outer_bats = [normalize_battery(args.only_battery)] if args.only_battery else batteries
    ranking_rows = []
    manifest_rows = []

    for H in horizons:
        bundle = bundles[H]
        available_bats = sorted(set(bundle.battery.tolist()))
        for fi, test_bat in enumerate(outer_bats, start=1):
            if test_bat not in available_bats:
                print(f"[SKIP] H={H} {test_bat}: no windows", flush=True)
                continue
            source_bats = [b for b in available_bats if b != test_bat]
            rank = source_rank(
                helper, bundle, candidates, all_hi_columns, source_bats,
                H, test_bat, float(args.min_source_availability),
            )
            ranking_rows.extend(rank.to_dict("records"))
            eligible_order = rank.loc[rank["Eligible"].eq(1), "HI"].astype(str).tolist()
            if len(eligible_order) < max(subset_sizes):
                raise RuntimeError(
                    f"H={H} {test_bat}: eligible Event HI={len(eligible_order)} < max TopK={max(subset_sizes)}. "
                    "可降低 --subset-sizes 或 --min-source-availability。"
                )

            print(
                f"[H={H:3d} fold {fi:02d}/{len(outer_bats)}] {test_bat} | "
                f"Top8={eligible_order[:8]}", flush=True,
            )

            for k in subset_sizes:
                selected = eligible_order[:k]
                manifest_rows.append({
                    "Horizon": H, "TestBattery": test_bat, "HI_Count": k,
                    "SelectedHI": "|".join(selected),
                })
                for model in args.models:
                    pc, mc = model_cache_paths(cache, H, test_bat, k, model)
                    _, mm = run_one_model(
                        helper, bundle, test_bat, H, selected, all_hi_columns,
                        model, args, pc, mc,
                    )
                    r = mm.iloc[0]
                    print(
                        f"    Top{k:<2d} {model:<8s} RMSE={float(r.RMSE):.6f} "
                        f"MAE={float(r.MAE):.6f} time={float(r.Runtime_s):.1f}s",
                        flush=True,
                    )

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(out / "04_source_only_event_HI_ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_rows).to_csv(out / "05_selected_event_HI_manifest.csv", index=False, encoding="utf-8-sig")

    oof = aggregate_prediction_caches(cache)
    if oof.empty:
        raise RuntimeError("模型 cache 为空")
    oof.to_csv(out / "06_ALL_EVENT_HICOUNT_OOF_predictions.csv.gz", index=False, compression=FAST_GZIP)

    pooled = pooled_metrics(helper, oof)
    macro = macro_metrics(helper, oof)
    pooled.to_csv(out / "07_pooled_OOF_metrics.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(out / "08_macro_LOBO_metrics.csv", index=False, encoding="utf-8-sig")

    metric_files = sorted(cache.glob("*.metric.csv"))
    metric_df = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True) if metric_files else pd.DataFrame()
    metric_df.to_csv(out / "09_fold_metrics_all_available_models.csv", index=False, encoding="utf-8-sig")

    # Best and smallest within 5% of best, per H/model.
    best_rows = []
    if not pooled.empty:
        for (H, model), g in pooled.groupby(["Horizon", "Model"], sort=True):
            g = g.sort_values(["RMSE", "HI_Count"])
            best = g.iloc[0]
            lim = float(best.RMSE) * 1.05
            within = g[g["RMSE"] <= lim].sort_values("HI_Count")
            small = within.iloc[0]
            best_rows.append({
                "Horizon": int(H), "Model": model,
                "BestHI_Count": int(best.HI_Count), "BestRMSE": float(best.RMSE),
                "SmallestWithin5pctBest": int(small.HI_Count),
                "SmallestWithin5pct_RMSE": float(small.RMSE),
            })
    pd.DataFrame(best_rows).to_csv(out / "10_best_event_HI_count_summary.csv", index=False, encoding="utf-8-sig")

    # Selection frequency by rank.
    freq_rows = []
    if not ranking_df.empty:
        n_rankings = ranking_df[["Horizon", "TestBattery"]].drop_duplicates().shape[0]
        for hi, g in ranking_df.groupby("HI"):
            row = {"HI": hi, "Rankings": n_rankings}
            for k in subset_sizes:
                row[f"SelectedInTop{k}"] = int((g["Rank"] <= k).sum())
                row[f"SelectedInTop{k}_Rate"] = float((g["Rank"] <= k).mean())
            row["MeanRank"] = float(g["Rank"].mean())
            row["MeanAvailability"] = float(g["Availability"].mean())
            freq_rows.append(row)
    pd.DataFrame(freq_rows).sort_values(["MeanRank", "HI"] if freq_rows else ["HI"]).to_csv(
        out / "11_event_HI_selection_frequency.csv", index=False, encoding="utf-8-sig"
    )

    available_models = sorted(c.replace("Pred_", "") for c in oof.columns if c.startswith("Pred_"))
    expected_folds = len(outer_bats) * len(horizons) * len(subset_sizes) * len(available_models)
    observed_metric_rows = len(metric_df)
    full_models = set(MODELS_ALL).issubset(set(available_models)) and not args.only_battery
    gate = pd.DataFrame([
        {"Check": "47 strict batteries", "Observed": len(batteries), "Expected": 47, "PASS": int(len(batteries) == 47)},
        {"Check": "3 Event checkpoints", "Observed": len(EVENT_NODES), "Expected": 3, "PASS": 1},
        {"Check": "No empty horizon", "Observed": int((inv["Windows"] > 0).sum()), "Expected": len(horizons), "PASS": int((inv["Windows"] > 0).all())},
        {"Check": "Model metric caches", "Observed": observed_metric_rows, "Expected": expected_folds, "PASS": int(observed_metric_rows >= expected_folds)},
        {"Check": "All four models available", "Observed": ",".join(available_models), "Expected": ",".join(MODELS_ALL), "PASS": int(set(MODELS_ALL).issubset(set(available_models)))},
    ])
    gate.to_csv(out / "12_acceptance_gate.csv", index=False, encoding="utf-8-sig")

    status = "PASS" if full_models and gate["PASS"].eq(1).all() else "PARTIAL_OR_CHECK"
    complete = {
        "Task": "RWTH48 V4.7E Event HI-count ablation",
        "Status": status,
        "EventPlatform": "RefAh 0.05/0.09/0.13",
        "CandidatePoolSize": len(candidates),
        "J": J, "L": L, "Horizons": horizons, "SubsetSizes": subset_sizes,
        "AvailableModels": available_models,
        "OuterLOBO": "46 source / 1 held-out; source-only ranking",
        "HistoricalSOHAsFeature": False,
        "CycleAsFeature": False,
        "FutureHI": False,
        "Elapsed_s": float(time.perf_counter() - t0),
        "Important": [
            "Event HI selection is independent of FixedTime V4.7 Top2.",
            "Held-out battery never participates in HI ranking or preprocessing.",
            "The V4.3 all-data consensus is not reused for same-round feature selection.",
            "CPU and deep model runs can be split; per-model caches are merged on rerun.",
        ],
    }
    save_json(complete, out / "99_RWTH48_V4_7E_COMPLETE.json")

    print("\n" + "=" * 118)
    print("RWTH48 V4.7E COMPLETE")
    print("Status          :", status)
    print("Candidate HI    :", len(candidates))
    print("Models available:", available_models)
    print("Output          :", out)
    print("Elapsed         :", f"{complete['Elapsed_s']/3600:.2f} h")
    print("=" * 118)


if __name__ == "__main__":
    main()

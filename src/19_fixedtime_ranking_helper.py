# -*- coding: utf-8 -*-
r"""
RWTH48 V4.7 LITE
HI 数量消融：Source-only Top1 / Top2 / Top3 vs frozen Full5
HI-Only Sparse-History Future SOH Prognosis + 47-cell Outer LOBO
================================================================================

研究目的
--------
V4.6A 已冻结：
    4 cumulative checkpoints: 0-30 / 0-60 / 0-90 / 0-120 s
    5 compact causal HIs per checkpoint
    J=5 cycles, L=10 history observations
    H=10/50/100
    RF / XGBoost / LSTM / BiLSTM
    47-cell outer LOBO
    no cycle index / historical SOH / RPT SOH / capacity / future HI in X

V4.7 只回答一个问题：
    “需要多少种 HI，才能保留 Full5 的跨电池未来 SOH 预测能力？”

严格控制变量：
    - 不重新提取 HI；直接读取 V4.6A/03_prefix_HI_all_cycles.csv.gz
    - 不改变 checkpoint、J、L、H、模型超参数、训练样本上限、LOBO 划分
    - Full5 直接复用 V4.6A/08_ALL_OOF_predictions.csv.gz，不重复训练
    - 只训练 Top1 / Top2 / Top3
    - TopK 不是用全体 47 节电池先筛选；每个 outer fold 内只用 46 source batteries 排名
    - held-out battery 完全不参与 HI ranking / imputation / scaling / fitting
    - 四模型在同一 fold/H 下共用同一份 HI ranking，避免模型特异性筛选混入“数量消融”

Source-only ranking
-------------------
默认 ranking_mode=source_spearman：
    对每个 held-out battery × horizon：
      1) 从 46 source batteries 中按电池均衡抽取历史窗口；
      2) 对每一种 base HI，汇总其 L × 4 checkpoint 的所有位置；
      3) 计算每个位置与未来 SOH target 的 absolute Spearman correlation；
      4) 以 MeanAbsSpearman 作为该 base HI 的 source-only ranking score；
      5) 形成 nested Top1 -> Top2 -> Top3。

因此：
    Top1 / Top2 / Top3 可能随 held-out battery 与 H 改变。
    这是 nested feature selection，且严格无 held-out leakage。

输入维度
--------
    Top1: 4 checkpoint × 1 HI = 4 features / observation
          L=10 -> 40 scalar HI values / sample
    Top2: 8 / observation  -> 80 scalar values / sample
    Top3: 12 / observation -> 120 scalar values / sample
    Full5:20 / observation -> 200 scalar values / sample

运行
----
新电脑正式：
    python "src/19_fixedtime_ranking_helper.py"

单折 smoke：
    python "src/19_fixedtime_ranking_helper.py" --only-battery RWTH-002 --horizons 10

只跑传统模型：
    python "src/19_fixedtime_ranking_helper.py" --only-battery RWTH-002 --horizons 10 --models RF XGBoost

重新训练已有 subset cache：
    python "src/19_fixedtime_ranking_helper.py" --force-folds

只做输入、窗口、Full5 key 一致性审计：
    python "src/19_fixedtime_ranking_helper.py" --audit-only

重要输出
--------
    00_config.json
    01_input_compatibility_audit.csv
    02_window_inventory.csv
    03_source_only_HI_ranking_by_fold.csv
    04_selected_HI_manifest_by_fold.csv
    05_fold_model_metrics.csv
    06_pooled_OOF_metrics.csv
    07_macro_LOBO_metrics.csv
    08_HI_count_gain_vs_Full5.csv
    09_HI_selection_frequency.csv
    10_input_efficiency_summary.csv
    11_runtime_summary.csv
    12_acceptance_gate.csv
    13_ALL_OOF_HI_count_ablation.csv.gz
    Figures/
    99_RWTH48_V4_7_COMPLETE.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# =============================================================================
# 0. Frozen project configuration
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")

DEFAULT_V46A_ROOT = (
    BASE
    / "RWTH48_V4_6A_HIOnly_120sMultiCheckpoint_LOBO_LITE"
)

DEFAULT_STRICT_INPUT = (
    BASE
    / "RWTH48_V4_5C_FIX_最小冲击共同扰动处理"
    / "06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz"
)

DEFAULT_HI_INPUT = (
    DEFAULT_V46A_ROOT
    / "03_prefix_HI_all_cycles.csv.gz"
)

DEFAULT_BASELINE_OOF = (
    DEFAULT_V46A_ROOT
    / "08_ALL_OOF_predictions.csv.gz"
)

DEFAULT_V46A_COMPLETE = (
    DEFAULT_V46A_ROOT
    / "99_RWTH48_V4_6A_COMPLETE.json"
)

DEFAULT_OUTPUT = (
    BASE
    / "RWTH48_V4_7_HI数量消融_SourceOnlyRanking_LOBO_LITE"
)

TARGET = "CyclingSOH_Charge_Canonical"
SEGMENT = "StrictSegmentKey"
EXCLUDED_BATTERIES = {"RWTH-028"}

CHECKPOINTS_S = (30, 60, 90, 120)

COMPACT_HI = (
    "V_TimeWeightedMean",
    "I_Slope_A_per_s",
    "P_Std",
    "dVdt_Std",
    "I_Entropy",
)

ALL_HI_COLUMNS = tuple(
    f"T{sec:03d}__{f}"
    for sec in CHECKPOINTS_S
    for f in COMPACT_HI
)

HISTORY_INTERVAL_J = 5
HISTORY_OBSERVATIONS_L = 10
DEFAULT_HORIZONS = (10, 50, 100)
DEFAULT_SUBSET_SIZES = (1, 2, 3)

# Exactly follow V4.6A lightweight source-balanced training caps.
RF_PER_SOURCE_BATTERY = 60
XGB_PER_SOURCE_BATTERY = 80
DEEP_PER_SOURCE_BATTERY = 80
RANK_PER_SOURCE_BATTERY = 80

RF_PARAMS = dict(
    n_estimators=120,
    max_depth=16,
    min_samples_leaf=2,
    max_features=0.75,
    random_state=20260820,
)

XGB_PARAMS = dict(
    n_estimators=180,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.90,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    min_child_weight=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=20260820,
)

DEEP_EPOCHS = 12
DEEP_BATCH = 256
DEEP_LR = 1e-3

RANDOM_SEED = 20260820
FAST_GZIP = {"method": "gzip", "compresslevel": 1}

ALL_MODELS = (
    "RF",
    "XGBoost",
    "LSTM",
    "BiLSTM",
)

OOF_KEY = [
    "Horizon",
    "Battery",
    "OriginCycle_KeyOnly",
    "TargetCycle_KeyOnly",
]


# =============================================================================
# 1. CLI / utility
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "RWTH48 V4.7 source-only nested HI-count ablation: "
            "Top1/Top2/Top3 vs frozen V4.6A Full5."
        )
    )

    p.add_argument("--strict-input", type=Path, default=DEFAULT_STRICT_INPUT)
    p.add_argument("--hi-input", type=Path, default=DEFAULT_HI_INPUT)
    p.add_argument("--baseline-oof", type=Path, default=DEFAULT_BASELINE_OOF)
    p.add_argument("--v46a-complete", type=Path, default=DEFAULT_V46A_COMPLETE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)

    p.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
    )
    p.add_argument("--history-interval", type=int, default=HISTORY_INTERVAL_J)
    p.add_argument("--history-observations", type=int, default=HISTORY_OBSERVATIONS_L)
    p.add_argument(
        "--subset-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SUBSET_SIZES),
        help="Train these HI counts. Full5 is reused from V4.6A and should not be listed.",
    )
    p.add_argument(
        "--ranking-mode",
        choices=["source_spearman", "fixed"],
        default="source_spearman",
    )
    p.add_argument(
        "--fixed-order",
        nargs="+",
        default=list(COMPACT_HI),
        help="Used only when --ranking-mode fixed.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(ALL_MODELS),
        default=list(ALL_MODELS),
    )
    p.add_argument("--threads-model", type=int, default=4)
    p.add_argument("--deep-epochs", type=int, default=DEEP_EPOCHS)
    p.add_argument("--deep-batch", type=int, default=DEEP_BATCH)
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--force-folds", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--skip-figures", action="store_true")

    return p.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: object, p: Path) -> None:
    ensure_dir(p.parent)
    p.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def normalize_battery(x: object) -> str:
    s = str(x).strip()
    if s.upper().startswith("RWTH-"):
        try:
            return f"RWTH-{int(s.split('-')[-1]):03d}"
        except Exception:
            return s
    try:
        return f"RWTH-{int(s):03d}"
    except Exception:
        return s


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def safe_mean(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.mean()) if x.notna().any() else np.nan


def safe_median(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.median()) if x.notna().any() else np.nan


def safe_max(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.max()) if x.notna().any() else np.nan


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]

    if len(yt) == 0:
        return dict(
            N=0,
            MAE=np.nan,
            RMSE=np.nan,
            R2=np.nan,
            MAPE_pct=np.nan,
            Bias=np.nan,
        )

    e = yp - yt

    return dict(
        N=int(len(yt)),
        MAE=float(np.mean(np.abs(e))),
        RMSE=float(np.sqrt(np.mean(e ** 2))),
        R2=(
            float(r2_score(yt, yp))
            if len(yt) >= 2 and np.std(yt) > 1e-12
            else np.nan
        ),
        MAPE_pct=float(
            np.mean(np.abs(e) / np.maximum(np.abs(yt), 1e-9)) * 100.0
        ),
        Bias=float(np.mean(e)),
    )


def configure_tensorflow_if_needed(models: Sequence[str]) -> Dict[str, object]:
    info: Dict[str, object] = {
        "TensorFlowRequired": any(m in {"LSTM", "BiLSTM"} for m in models),
        "TensorFlowAvailable": False,
        "TensorFlowVersion": "",
        "GPUDevices": [],
    }

    if not info["TensorFlowRequired"]:
        return info

    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError(
            "请求了 LSTM/BiLSTM，但当前 Python 环境无法 import TensorFlow。"
        ) from exc

    info["TensorFlowAvailable"] = True
    info["TensorFlowVersion"] = str(tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    info["GPUDevices"] = [str(x) for x in gpus]

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    return info


# =============================================================================
# 2. Frozen inputs
# =============================================================================


def load_strict_target(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"strict input不存在：{path}")

    d = pd.read_csv(path, compression="infer", low_memory=False)

    required = {"Battery", "GlobalCycle", TARGET}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"strict input缺列：{missing}")

    d["Battery"] = d["Battery"].map(normalize_battery)
    d["GlobalCycle"] = pd.to_numeric(d["GlobalCycle"], errors="coerce")
    d[TARGET] = pd.to_numeric(d[TARGET], errors="coerce")

    d = d[
        d["GlobalCycle"].notna()
        & d[TARGET].notna()
        & np.isfinite(d[TARGET])
    ].copy()
    d["GlobalCycle"] = d["GlobalCycle"].astype(int)

    if "SourcePart" not in d.columns:
        d["SourcePart"] = ""

    if SEGMENT not in d.columns:
        d = d.sort_values(["Battery", "GlobalCycle"]).reset_index(drop=True)
        pb = d["Battery"].shift()
        pc = d["GlobalCycle"].shift()
        ps = d["SourcePart"].shift()
        br = (
            d["Battery"].ne(pb)
            | (d["GlobalCycle"] - pd.to_numeric(pc, errors="coerce")).ne(1)
        )
        if d["SourcePart"].astype(str).str.len().gt(0).any():
            br = br | d["SourcePart"].astype(str).ne(ps.astype(str))
        d["_seg"] = br.groupby(d["Battery"]).cumsum().astype(int)
        d[SEGMENT] = (
            d["Battery"].astype(str)
            + "__AUTO"
            + d["_seg"].astype(str).str.zfill(4)
        )
        d.drop(columns="_seg", inplace=True)

    if d.duplicated(["Battery", "GlobalCycle"]).any():
        raise RuntimeError("strict target存在重复 Battery+GlobalCycle")

    d = d[~d["Battery"].isin(EXCLUDED_BATTERIES)].copy()

    return d.sort_values(["Battery", "GlobalCycle"]).reset_index(drop=True)


def load_prefix_hi(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"V4.6A prefix HI不存在：{path}")

    d = pd.read_csv(path, compression="infer", low_memory=False)
    required = {"Battery", "GlobalCycle", *ALL_HI_COLUMNS}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(
            "prefix HI缺冻结列：\n" + "\n".join(missing)
        )

    d["Battery"] = d["Battery"].map(normalize_battery)
    d["GlobalCycle"] = pd.to_numeric(d["GlobalCycle"], errors="coerce")
    d = d[d["GlobalCycle"].notna()].copy()
    d["GlobalCycle"] = d["GlobalCycle"].astype(int)

    for c in ALL_HI_COLUMNS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d[~d["Battery"].isin(EXCLUDED_BATTERIES)].copy()

    if d.duplicated(["Battery", "GlobalCycle"]).any():
        raise RuntimeError("prefix HI存在重复 Battery+GlobalCycle")

    return d.sort_values(["Battery", "GlobalCycle"]).reset_index(drop=True)


def load_v46a_complete(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"V4.6A COMPLETE.json不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_oof(
    path: Path,
    horizons: Sequence[int],
    models: Sequence[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"V4.6A baseline OOF不存在：{path}")

    d = pd.read_csv(path, compression="infer", low_memory=False)

    required = {
        *OOF_KEY,
        "SOH_True",
        *[f"Pred_{m}" for m in models],
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError("baseline OOF缺列：\n" + "\n".join(missing))

    d["Battery"] = d["Battery"].map(normalize_battery)

    for c in ("Horizon", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d[
        d["Horizon"].isin([int(x) for x in horizons])
        & ~d["Battery"].isin(EXCLUDED_BATTERIES)
    ].copy()

    for c in ("Horizon", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly"):
        if d[c].isna().any():
            raise RuntimeError(f"baseline OOF的{c}存在无法解析值")
        d[c] = d[c].astype(int)

    for c in ("SOH_True", *[f"Pred_{m}" for m in models]):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if d.duplicated(OOF_KEY).any():
        raise RuntimeError("baseline OOF存在重复 key")

    return d.sort_values(OOF_KEY).reset_index(drop=True)


def validate_v46a_configuration(
    complete: Dict[str, object],
    J: int,
    L: int,
    horizons: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(check: str, observed: object, expected: object, passed: bool) -> None:
        rows.append(
            dict(
                Check=check,
                Observed=observed,
                Expected=expected,
                PASS=int(bool(passed)),
            )
        )

    add("V4.6A status", complete.get("Status"), "PASS", complete.get("Status") == "PASS")
    add(
        "Only HI as model input",
        complete.get("OnlyHIAsModelInput"),
        True,
        complete.get("OnlyHIAsModelInput") is True,
    )
    add(
        "GlobalCycle as feature",
        complete.get("GlobalCycleAsFeature"),
        False,
        complete.get("GlobalCycleAsFeature") is False,
    )
    add(
        "Historical SOH as feature",
        complete.get("HistoricalSOHAsFeature"),
        False,
        complete.get("HistoricalSOHAsFeature") is False,
    )
    add(
        "Periodic SOH calibration",
        complete.get("PeriodicSOHCalibration"),
        False,
        complete.get("PeriodicSOHCalibration") is False,
    )
    add(
        "History interval J",
        int(complete.get("HistoryInterval_J", -1)),
        J,
        int(complete.get("HistoryInterval_J", -1)) == J,
    )
    add(
        "History observations L",
        int(complete.get("HistoryObservations_L", -1)),
        L,
        int(complete.get("HistoryObservations_L", -1)) == L,
    )

    observed_h = sorted(int(x) for x in complete.get("Horizons", []))
    requested_h = sorted(int(x) for x in horizons)
    add(
        "Requested horizons are available in V4.6A",
        observed_h,
        requested_h,
        set(requested_h).issubset(set(observed_h)),
    )

    observed_cp = [int(x) for x in complete.get("Checkpoints_s", [])]
    add(
        "Checkpoints",
        observed_cp,
        list(CHECKPOINTS_S),
        observed_cp == list(CHECKPOINTS_S),
    )

    observed_hi = list(complete.get("CompactHI", []))
    add(
        "Compact HI",
        observed_hi,
        list(COMPACT_HI),
        observed_hi == list(COMPACT_HI),
    )

    observed_models = set(str(x) for x in complete.get("Models", []))
    add(
        "Baseline models include all requested models",
        sorted(observed_models),
        sorted(ALL_MODELS),
        set(ALL_MODELS).issubset(observed_models),
    )

    return pd.DataFrame(rows)


# =============================================================================
# 3. Exact sparse-history windows
# =============================================================================


@dataclass
class WindowBundle:
    X: np.ndarray
    y: np.ndarray
    battery: np.ndarray
    origin_cycle: np.ndarray
    target_cycle: np.ndarray
    segment: np.ndarray


def build_sparse_history_windows(
    merged: pd.DataFrame,
    J: int,
    L: int,
    H: int,
) -> WindowBundle:
    Xs: List[np.ndarray] = []
    ys: List[float] = []
    bs: List[str] = []
    origins: List[int] = []
    targets: List[int] = []
    segs: List[str] = []

    history_offsets = np.array(
        [-J * k for k in range(L - 1, -1, -1)],
        dtype=int,
    )

    for (battery, seg), g in merged.groupby(["Battery", SEGMENT], sort=False):
        g = g.sort_values("GlobalCycle").copy()

        cycle_map = {
            int(c): i
            for i, c in enumerate(g["GlobalCycle"].to_numpy(int))
        }
        cycles = g["GlobalCycle"].to_numpy(int)
        Xmat = g[list(ALL_HI_COLUMNS)].to_numpy(dtype=np.float32)
        yvec = g[TARGET].to_numpy(float)

        for origin in cycles:
            target = int(origin + H)
            if target not in cycle_map:
                continue

            hcycles = origin + history_offsets
            if not all(int(c) in cycle_map for c in hcycles):
                continue

            idx_hist = [cycle_map[int(c)] for c in hcycles]
            idx_target = cycle_map[target]

            x = Xmat[idx_hist, :]
            y = yvec[idx_target]

            if not np.isfinite(y):
                continue

            Xs.append(x)
            ys.append(float(y))
            bs.append(str(battery))
            origins.append(int(origin))
            targets.append(target)
            segs.append(str(seg))

    if not Xs:
        return WindowBundle(
            X=np.empty((0, L, len(ALL_HI_COLUMNS)), dtype=np.float32),
            y=np.empty(0, dtype=float),
            battery=np.empty(0, dtype=object),
            origin_cycle=np.empty(0, dtype=int),
            target_cycle=np.empty(0, dtype=int),
            segment=np.empty(0, dtype=object),
        )

    return WindowBundle(
        X=np.stack(Xs).astype(np.float32, copy=False),
        y=np.asarray(ys, dtype=float),
        battery=np.asarray(bs, dtype=object),
        origin_cycle=np.asarray(origins, dtype=int),
        target_cycle=np.asarray(targets, dtype=int),
        segment=np.asarray(segs, dtype=object),
    )


def bundle_key_frame(bundle: WindowBundle, horizon: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Horizon": int(horizon),
            "Battery": bundle.battery,
            "OriginCycle_KeyOnly": bundle.origin_cycle,
            "TargetCycle_KeyOnly": bundle.target_cycle,
            "SOH_True_Rebuilt": bundle.y,
        }
    )


def compare_baseline_keys(
    bundle: WindowBundle,
    baseline_h: pd.DataFrame,
    horizon: int,
    batteries_to_check: Sequence[str],
) -> Dict[str, object]:
    rebuilt = bundle_key_frame(bundle, horizon)
    rebuilt = rebuilt[rebuilt["Battery"].isin(list(batteries_to_check))].copy()
    base = baseline_h[baseline_h["Battery"].isin(list(batteries_to_check))].copy()

    if rebuilt.duplicated(OOF_KEY).any():
        raise RuntimeError(f"H={horizon} rebuilt windows存在重复 key")

    j = rebuilt.merge(
        base[OOF_KEY + ["SOH_True"]],
        on=OOF_KEY,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    both = j["_merge"].eq("both")
    target_diff = np.abs(
        pd.to_numeric(j.loc[both, "SOH_True_Rebuilt"], errors="coerce").to_numpy(float)
        - pd.to_numeric(j.loc[both, "SOH_True"], errors="coerce").to_numpy(float)
    )

    max_diff = float(np.nanmax(target_diff)) if len(target_diff) else np.nan

    return {
        "Horizon": int(horizon),
        "RebuiltRows": int(len(rebuilt)),
        "BaselineRows": int(len(base)),
        "MatchedRows": int(both.sum()),
        "RebuiltOnly": int(j["_merge"].eq("left_only").sum()),
        "BaselineOnly": int(j["_merge"].eq("right_only").sum()),
        "MaxAbsTargetDiff": max_diff,
        "PASS": int(
            len(rebuilt) == len(base)
            and both.all()
            and (not np.isfinite(max_diff) or max_diff <= 1e-12)
        ),
    }


# =============================================================================
# 4. Source-balanced sampling + source-only ranking
# =============================================================================


def balanced_indices(
    battery: np.ndarray,
    source_batteries: Sequence[str],
    max_per_battery: int,
) -> np.ndarray:
    idx: List[int] = []

    for b in source_batteries:
        ix = np.flatnonzero(battery == b)
        if len(ix) <= max_per_battery:
            pick = ix
        else:
            pos = np.linspace(0, len(ix) - 1, max_per_battery)
            pick = ix[np.round(pos).astype(int)]
        idx.extend(pick.tolist())

    return np.asarray(sorted(set(idx)), dtype=int)


def observation_indices_for_hi(base_hi: str) -> List[int]:
    return [
        i
        for i, c in enumerate(ALL_HI_COLUMNS)
        if c.endswith("__" + base_hi)
    ]


def selected_observation_indices(selected_hi: Sequence[str]) -> List[int]:
    selected = set(selected_hi)
    return [
        i
        for i, c in enumerate(ALL_HI_COLUMNS)
        if c.split("__", 1)[1] in selected
    ]


def abs_spearman_column(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 20:
        return np.nan

    xx = np.asarray(x[m], dtype=float)
    yy = np.asarray(y[m], dtype=float)

    if np.nanstd(xx) < 1e-12 or np.nanstd(yy) < 1e-12:
        return 0.0

    # pandas rank is available without an extra direct scipy dependency.
    rx = pd.Series(xx).rank(method="average").to_numpy(float)
    ry = pd.Series(yy).rank(method="average").to_numpy(float)

    r = np.corrcoef(rx, ry)[0, 1]
    return float(abs(r)) if np.isfinite(r) else 0.0


def rank_hi_source_only(
    bundle: WindowBundle,
    source_batteries: Sequence[str],
    horizon: int,
    test_battery: str,
    ranking_mode: str,
    fixed_order: Sequence[str],
) -> pd.DataFrame:
    if ranking_mode == "fixed":
        order = [x for x in fixed_order if x in COMPACT_HI]
        if set(order) != set(COMPACT_HI) or len(order) != len(COMPACT_HI):
            raise ValueError(
                "--fixed-order 必须恰好包含 5 个冻结 HI，且不得重复/遗漏。"
            )
        return pd.DataFrame(
            [
                {
                    "Horizon": int(horizon),
                    "TestBattery": test_battery,
                    "HI": hi,
                    "MeanAbsSpearman": np.nan,
                    "MedianAbsSpearman": np.nan,
                    "MaxAbsSpearman": np.nan,
                    "ValidPositions": np.nan,
                    "RankingMode": "fixed",
                    "Rank": rank,
                }
                for rank, hi in enumerate(order, start=1)
            ]
        )

    rank_idx = balanced_indices(
        bundle.battery,
        source_batteries,
        RANK_PER_SOURCE_BATTERY,
    )
    if len(rank_idx) == 0:
        raise RuntimeError("source-only ranking窗口为空")

    X = bundle.X[rank_idx]
    y = bundle.y[rank_idx]
    rows: List[Dict[str, object]] = []

    for hi in COMPACT_HI:
        obs_idx = observation_indices_for_hi(hi)
        # [N, L, 4] -> every observed time position is a source-only univariate association.
        Z = X[:, :, obs_idx].reshape(len(X), -1)
        vals = np.asarray(
            [abs_spearman_column(Z[:, j], y) for j in range(Z.shape[1])],
            dtype=float,
        )
        finite = vals[np.isfinite(vals)]

        rows.append(
            {
                "Horizon": int(horizon),
                "TestBattery": test_battery,
                "HI": hi,
                "MeanAbsSpearman": float(np.mean(finite)) if len(finite) else 0.0,
                "MedianAbsSpearman": float(np.median(finite)) if len(finite) else 0.0,
                "MaxAbsSpearman": float(np.max(finite)) if len(finite) else 0.0,
                "ValidPositions": int(len(finite)),
                "RankingMode": "source_spearman",
            }
        )

    r = pd.DataFrame(rows)
    original_order = {h: i for i, h in enumerate(COMPACT_HI)}
    r["_order"] = r["HI"].map(original_order)
    r = r.sort_values(
        ["MeanAbsSpearman", "MedianAbsSpearman", "MaxAbsSpearman", "_order"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    r["Rank"] = np.arange(1, len(r) + 1)
    r.drop(columns="_order", inplace=True)
    return r


# =============================================================================
# 5. Source-only imputation / scaling
# =============================================================================


@dataclass
class SeqScaler:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = np.asarray(X, dtype=np.float32).copy()
        for j in range(z.shape[2]):
            a = z[:, :, j]
            bad = ~np.isfinite(a)
            if bad.any():
                a[bad] = self.median[j]
            z[:, :, j] = (a - self.mean[j]) / self.std[j]
        return z


def fit_scaler(X_source: np.ndarray) -> SeqScaler:
    flat = np.asarray(X_source, dtype=np.float64).reshape(-1, X_source.shape[2])
    med = np.nanmedian(flat, axis=0)

    # If an entire selected position is NaN in source batteries, use 0 as a
    # neutral source-only fallback. This is reported through the normal model metrics;
    # no held-out value is consulted.
    med[~np.isfinite(med)] = 0.0

    filled = flat.copy()
    for j in range(filled.shape[1]):
        bad = ~np.isfinite(filled[:, j])
        filled[bad, j] = med[j]

    mean = np.mean(filled, axis=0)
    std = np.std(filled, axis=0)
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0

    return SeqScaler(
        median=med.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


# =============================================================================
# 6. Models — frozen V4.6A hyperparameters
# =============================================================================


def build_deep_model(kind: str, L: int, F: int):
    import tensorflow as tf

    tf.keras.backend.clear_session()

    inp = tf.keras.Input(shape=(L, F), name="HI_history")

    if kind == "LSTM":
        x = tf.keras.layers.LSTM(
            32,
            activation="tanh",
            recurrent_activation="sigmoid",
        )(inp)
    elif kind == "BiLSTM":
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                24,
                activation="tanh",
                recurrent_activation="sigmoid",
            )
        )(inp)
    else:
        raise ValueError(kind)

    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    out = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inp, out, name=kind)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=DEEP_LR),
        loss="mse",
    )
    return model


def train_predict_deep(
    kind: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    epochs: int,
    batch: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    import tensorflow as tf

    set_seed(seed)

    ym = float(np.mean(ytr))
    ys = float(np.std(ytr))
    if not np.isfinite(ys) or ys < 1e-8:
        ys = 1.0

    yz = ((ytr - ym) / ys).astype(np.float32)
    model = build_deep_model(kind, Xtr.shape[1], Xtr.shape[2])

    t0 = time.perf_counter()
    model.fit(
        Xtr,
        yz,
        epochs=int(epochs),
        batch_size=int(batch),
        shuffle=True,
        verbose=0,
    )
    predz = model.predict(
        Xte,
        batch_size=max(256, int(batch)),
        verbose=0,
    ).reshape(-1)
    elapsed = time.perf_counter() - t0

    pred = predz.astype(float) * ys + ym

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return pred, float(elapsed)


def train_predict_rf(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    threads: int,
) -> Tuple[np.ndarray, float]:
    params = dict(RF_PARAMS)
    params["n_jobs"] = max(1, int(threads))
    model = RandomForestRegressor(**params)

    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    elapsed = time.perf_counter() - t0

    del model
    gc.collect()
    return np.asarray(pred, dtype=float), float(elapsed)


def train_predict_xgb(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    threads: int,
) -> Tuple[np.ndarray, float]:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise ImportError("未安装 xgboost。") from exc

    params = dict(XGB_PARAMS)
    params["n_jobs"] = max(1, int(threads))
    model = XGBRegressor(**params)

    t0 = time.perf_counter()
    model.fit(Xtr, ytr, verbose=False)
    pred = model.predict(Xte)
    elapsed = time.perf_counter() - t0

    del model
    gc.collect()
    return np.asarray(pred, dtype=float), float(elapsed)


# =============================================================================
# 7. One subset fold
# =============================================================================


def cache_matches_selection(
    pred_path: Path,
    metric_path: Path,
    expected_selected: Sequence[str],
    expected_models: Sequence[str],
) -> bool:
    if not (
        pred_path.exists()
        and metric_path.exists()
        and pred_path.stat().st_size > 0
        and metric_path.stat().st_size > 0
    ):
        return False

    try:
        p = pd.read_csv(pred_path, compression="infer", nrows=3)
        m = pd.read_csv(metric_path, nrows=10)
    except Exception:
        return False

    expected = "|".join(expected_selected)
    if "SelectedHI" not in p.columns or p["SelectedHI"].astype(str).nunique() != 1:
        return False
    if str(p["SelectedHI"].iloc[0]) != expected:
        return False

    for model in expected_models:
        if f"Pred_{model}" not in p.columns:
            return False
    if "Model" not in m.columns or not set(expected_models).issubset(set(m["Model"].astype(str))):
        return False

    return True


def run_subset_fold(
    bundle: WindowBundle,
    test_battery: str,
    horizon: int,
    selected_hi: Sequence[str],
    hi_count: int,
    models: Sequence[str],
    args: argparse.Namespace,
    fold_cache: Path,
    metric_cache: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if (
        not args.force_folds
        and cache_matches_selection(
            fold_cache,
            metric_cache,
            selected_hi,
            models,
        )
    ):
        return (
            pd.read_csv(fold_cache, compression="infer", low_memory=False),
            pd.read_csv(metric_cache, low_memory=False),
        )

    all_bats = sorted(set(bundle.battery.tolist()))
    source_bats = [b for b in all_bats if b != test_battery]
    test_idx = np.flatnonzero(bundle.battery == test_battery)

    if len(test_idx) == 0:
        raise RuntimeError(f"{test_battery} | H={horizon} 无测试窗口")

    obs_idx = selected_observation_indices(selected_hi)
    if len(obs_idx) != len(CHECKPOINTS_S) * hi_count:
        raise RuntimeError(
            f"HI选择维度错误：K={hi_count}, selected={selected_hi}, obs_dim={len(obs_idx)}"
        )

    Xsub = bundle.X[:, :, obs_idx]

    deep_idx = balanced_indices(
        bundle.battery,
        source_bats,
        DEEP_PER_SOURCE_BATTERY,
    )
    rf_idx = balanced_indices(
        bundle.battery,
        source_bats,
        RF_PER_SOURCE_BATTERY,
    )
    xgb_idx = balanced_indices(
        bundle.battery,
        source_bats,
        XGB_PER_SOURCE_BATTERY,
    )

    scaler = fit_scaler(Xsub[deep_idx])
    Xte = scaler.transform(Xsub[test_idx])
    Xtflat = Xte.reshape(len(Xte), -1)

    selected_text = "|".join(selected_hi)

    pred_df = pd.DataFrame(
        {
            "Horizon": int(horizon),
            "HI_Set": f"Top{hi_count}",
            "HI_Count": int(hi_count),
            "InputFeaturesPerObservation": int(len(obs_idx)),
            "HistoryScalarInputs": int(len(obs_idx) * Xsub.shape[1]),
            "SelectedHI": selected_text,
            "TestBattery": test_battery,
            "Battery": bundle.battery[test_idx],
            "OriginCycle_KeyOnly": bundle.origin_cycle[test_idx],
            "TargetCycle_KeyOnly": bundle.target_cycle[test_idx],
            "StrictSegmentKey_QCOnly": bundle.segment[test_idx],
            "SOH_True": bundle.y[test_idx],
        }
    )

    metric_rows: List[Dict[str, object]] = []

    for model_name in models:
        if model_name == "RF":
            Xtr = scaler.transform(Xsub[rf_idx]).reshape(len(rf_idx), -1)
            pred, runtime = train_predict_rf(
                Xtr,
                bundle.y[rf_idx],
                Xtflat,
                args.threads_model,
            )
            ntrain = len(rf_idx)
            del Xtr

        elif model_name == "XGBoost":
            Xtr = scaler.transform(Xsub[xgb_idx]).reshape(len(xgb_idx), -1)
            pred, runtime = train_predict_xgb(
                Xtr,
                bundle.y[xgb_idx],
                Xtflat,
                args.threads_model,
            )
            ntrain = len(xgb_idx)
            del Xtr

        elif model_name in {"LSTM", "BiLSTM"}:
            Xtr = scaler.transform(Xsub[deep_idx])
            seed = (
                RANDOM_SEED
                + int(horizon) * 1000
                + int(hi_count) * 100
                + (0 if model_name == "LSTM" else 1)
            )
            pred, runtime = train_predict_deep(
                model_name,
                Xtr,
                bundle.y[deep_idx],
                Xte,
                args.deep_epochs,
                args.deep_batch,
                seed,
            )
            ntrain = len(deep_idx)
            del Xtr

        else:
            raise ValueError(model_name)

        pred_df[f"Pred_{model_name}"] = pred

        metric_rows.append(
            {
                "Horizon": int(horizon),
                "HI_Set": f"Top{hi_count}",
                "HI_Count": int(hi_count),
                "SelectedHI": selected_text,
                "TestBattery": test_battery,
                "Model": model_name,
                "TrainSourceBatteries": int(len(source_bats)),
                "TrainWindows": int(ntrain),
                "TestWindows": int(len(test_idx)),
                "InputFeaturesPerObservation": int(len(obs_idx)),
                "HistoryScalarInputs": int(len(obs_idx) * Xsub.shape[1]),
                "Runtime_s": float(runtime),
                **calc_metrics(bundle.y[test_idx], pred),
            }
        )

        gc.collect()

    metric_df = pd.DataFrame(metric_rows)

    ensure_dir(fold_cache.parent)
    pred_df.to_csv(
        fold_cache,
        index=False,
        compression=FAST_GZIP,
    )
    metric_df.to_csv(
        metric_cache,
        index=False,
        encoding="utf-8-sig",
    )

    del Xsub, Xte, Xtflat
    gc.collect()

    return pred_df, metric_df


# =============================================================================
# 8. Full5 baseline conversion / exact matching
# =============================================================================


def baseline_full5_rows(
    baseline: pd.DataFrame,
    models: Sequence[str],
    batteries: Sequence[str],
) -> pd.DataFrame:
    d = baseline[baseline["Battery"].isin(list(batteries))].copy()
    d["HI_Set"] = "Full5"
    d["HI_Count"] = 5
    d["InputFeaturesPerObservation"] = 20
    d["HistoryScalarInputs"] = 20 * HISTORY_OBSERVATIONS_L
    d["SelectedHI"] = "|".join(COMPACT_HI)

    keep = [
        "Horizon",
        "HI_Set",
        "HI_Count",
        "InputFeaturesPerObservation",
        "HistoryScalarInputs",
        "SelectedHI",
        "Battery",
        "OriginCycle_KeyOnly",
        "TargetCycle_KeyOnly",
        "SOH_True",
        *[f"Pred_{m}" for m in models],
    ]
    if "TestBattery" in d.columns:
        d["TestBattery"] = d["TestBattery"].map(normalize_battery)
    else:
        d["TestBattery"] = d["Battery"]
    keep.insert(6, "TestBattery")

    if "StrictSegmentKey_QCOnly" in d.columns:
        keep.insert(10, "StrictSegmentKey_QCOnly")

    return d[keep].copy()


def validate_subset_vs_baseline(
    subset: pd.DataFrame,
    full5: pd.DataFrame,
    label: str,
) -> Dict[str, object]:
    keys = OOF_KEY
    a = subset[keys + ["SOH_True"]].copy()
    b = full5[keys + ["SOH_True"]].copy()

    if a.duplicated(keys).any() or b.duplicated(keys).any():
        return {
            "Check": f"{label} exact cohort",
            "PASS": 0,
            "Detail": "duplicate keys",
        }

    j = a.merge(
        b,
        on=keys,
        how="outer",
        suffixes=("_Subset", "_Full5"),
        indicator=True,
        validate="one_to_one",
    )

    both = j["_merge"].eq("both")
    diff = np.abs(
        pd.to_numeric(j.loc[both, "SOH_True_Subset"], errors="coerce").to_numpy(float)
        - pd.to_numeric(j.loc[both, "SOH_True_Full5"], errors="coerce").to_numpy(float)
    )
    maxdiff = float(np.nanmax(diff)) if len(diff) else np.nan

    passed = (
        len(a) == len(b)
        and both.all()
        and (not np.isfinite(maxdiff) or maxdiff <= 1e-12)
    )

    return {
        "Check": f"{label} exact cohort",
        "PASS": int(passed),
        "Detail": (
            f"subset={len(a)}, full5={len(b)}, matched={int(both.sum())}, "
            f"subset_only={int(j['_merge'].eq('left_only').sum())}, "
            f"full5_only={int(j['_merge'].eq('right_only').sum())}, "
            f"max_target_diff={maxdiff}"
        ),
    }


# =============================================================================
# 9. Aggregation
# =============================================================================


def fold_metrics_from_oof(
    oof: pd.DataFrame,
    models: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (h, hi_set, hi_count, b), g in oof.groupby(
        ["Horizon", "HI_Set", "HI_Count", "Battery"],
        sort=True,
    ):
        for m in models:
            c = f"Pred_{m}"
            if c not in g.columns:
                continue
            rows.append(
                {
                    "Horizon": int(h),
                    "HI_Set": str(hi_set),
                    "HI_Count": int(hi_count),
                    "TestBattery": str(b),
                    "Model": m,
                    "TestWindows": int(len(g)),
                    **calc_metrics(
                        g["SOH_True"].to_numpy(float),
                        g[c].to_numpy(float),
                    ),
                }
            )

    return pd.DataFrame(rows)


def pooled_metrics(oof: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (h, hi_set, hi_count), g in oof.groupby(
        ["Horizon", "HI_Set", "HI_Count"],
        sort=True,
    ):
        for m in models:
            c = f"Pred_{m}"
            if c not in g.columns:
                continue
            rows.append(
                {
                    "Horizon": int(h),
                    "HI_Set": str(hi_set),
                    "HI_Count": int(hi_count),
                    "InputFeaturesPerObservation": int(4 * int(hi_count)),
                    "HistoryScalarInputs": int(4 * int(hi_count) * HISTORY_OBSERVATIONS_L),
                    "Model": m,
                    "Eval": "POOLED_OOF",
                    **calc_metrics(
                        g["SOH_True"].to_numpy(float),
                        g[c].to_numpy(float),
                    ),
                }
            )

    return pd.DataFrame(rows)


def macro_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (h, hi_set, hi_count, model), g in fold_metrics.groupby(
        ["Horizon", "HI_Set", "HI_Count", "Model"],
        sort=True,
    ):
        row: Dict[str, object] = {
            "Horizon": int(h),
            "HI_Set": str(hi_set),
            "HI_Count": int(hi_count),
            "Model": str(model),
            "Folds": int(g["TestBattery"].nunique()),
        }
        for c in ("MAE", "RMSE", "R2", "MAPE_pct", "Bias"):
            row[f"MacroMean_{c}"] = safe_mean(g[c])
            row[f"MacroMedian_{c}"] = safe_median(g[c])
        row["WorstFold_RMSE"] = safe_max(g["RMSE"])
        row["NegativeR2Folds"] = int(
            (pd.to_numeric(g["R2"], errors="coerce") < 0).sum()
        )
        rows.append(row)

    return pd.DataFrame(rows)


def gain_vs_full5(pooled: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    p = pooled.set_index(["Horizon", "HI_Set", "Model"])

    for r in pooled.itertuples(index=False):
        if str(r.HI_Set) == "Full5":
            continue
        full_key = (int(r.Horizon), "Full5", str(r.Model))
        if full_key not in p.index:
            continue

        full = p.loc[full_key]
        rmse = float(r.RMSE)
        mae = float(r.MAE)
        rmse_full = float(full["RMSE"])
        mae_full = float(full["MAE"])

        rows.append(
            {
                "Horizon": int(r.Horizon),
                "Model": str(r.Model),
                "HI_Set": str(r.HI_Set),
                "HI_Count": int(r.HI_Count),
                "Full5_HI_Count": 5,
                "InputReduction_pct": (1.0 - int(r.HI_Count) / 5.0) * 100.0,
                "RMSE_Subset": rmse,
                "RMSE_Full5": rmse_full,
                "RMSE_Increase_vs_Full5_pct": ((rmse - rmse_full) / rmse_full * 100.0),
                "RMSE_Improvement_vs_Full5_pct": ((rmse_full - rmse) / rmse_full * 100.0),
                "MAE_Subset": mae,
                "MAE_Full5": mae_full,
                "MAE_Increase_vs_Full5_pct": ((mae - mae_full) / mae_full * 100.0),
                "AccuracyRetention_RMSE_pct": (rmse_full / rmse * 100.0) if rmse > 0 else np.nan,
                "BeatsFull5_RMSE": int(rmse < rmse_full),
            }
        )

    return pd.DataFrame(rows)


def selection_frequency(ranking: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for horizon_group, g in list(ranking.groupby("Horizon", sort=True)) + [("ALL", ranking)]:
        nfolds = int(g["TestBattery"].nunique())
        for hi in COMPACT_HI:
            q = g[g["HI"].eq(hi)]
            row: Dict[str, object] = {
                "Horizon": horizon_group,
                "HI": hi,
                "Folds": nfolds,
                "MeanRank": safe_mean(q["Rank"]),
                "MedianRank": safe_median(q["Rank"]),
                "MeanSourceScore": safe_mean(q["MeanAbsSpearman"]),
            }
            for k in (1, 2, 3):
                count = int((pd.to_numeric(q["Rank"], errors="coerce") <= k).sum())
                row[f"SelectedInTop{k}_Folds"] = count
                row[f"SelectedInTop{k}_Rate"] = count / max(nfolds, 1)
            rows.append(row)

    return pd.DataFrame(rows)


def input_efficiency_summary(
    pooled: pd.DataFrame,
    gain: pd.DataFrame,
) -> pd.DataFrame:
    base = pooled.copy()
    base["InputReduction_pct"] = (1.0 - base["HI_Count"] / 5.0) * 100.0

    if gain.empty:
        base["RMSE_Increase_vs_Full5_pct"] = np.nan
        base.loc[base["HI_Set"].eq("Full5"), "RMSE_Increase_vs_Full5_pct"] = 0.0
        return base

    keep = gain[
        [
            "Horizon",
            "Model",
            "HI_Set",
            "RMSE_Increase_vs_Full5_pct",
            "AccuracyRetention_RMSE_pct",
        ]
    ].copy()
    out = base.merge(keep, on=["Horizon", "Model", "HI_Set"], how="left")
    out.loc[out["HI_Set"].eq("Full5"), "RMSE_Increase_vs_Full5_pct"] = 0.0
    out.loc[out["HI_Set"].eq("Full5"), "AccuracyRetention_RMSE_pct"] = 100.0
    return out


# =============================================================================
# 10. Figures
# =============================================================================


def make_figures(pooled: pd.DataFrame, gain: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = ensure_dir(outdir / "Figures")

    for h, g in pooled.groupby("Horizon", sort=True):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for model, q in g.groupby("Model", sort=True):
            q = q.sort_values("HI_Count")
            ax.plot(
                q["HI_Count"],
                q["RMSE"],
                marker="o",
                label=str(model),
            )
        ax.set_xlabel("Number of base HIs")
        ax.set_ylabel("Pooled OOF RMSE")
        ax.set_title(f"RWTH48 HI-count ablation | H={int(h)}")
        ax.set_xticks([1, 2, 3, 5])
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / f"01_RMSE_vs_HI_Count_H{int(h):03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if not gain.empty:
        for h, g in gain.groupby("Horizon", sort=True):
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            for model, q in g.groupby("Model", sort=True):
                q = q.sort_values("HI_Count")
                ax.plot(
                    q["HI_Count"],
                    q["RMSE_Increase_vs_Full5_pct"],
                    marker="o",
                    label=str(model),
                )
            ax.axhline(0.0, linewidth=1.0)
            ax.set_xlabel("Number of base HIs")
            ax.set_ylabel("RMSE increase vs Full5 (%)")
            ax.set_title(f"Cost of HI reduction | H={int(h)}")
            ax.set_xticks([1, 2, 3])
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figdir / f"02_RMSE_Cost_vs_Full5_H{int(h):03d}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


# =============================================================================
# 11. Main
# =============================================================================


def main() -> None:
    args = parse_args()
    t_all = time.perf_counter()

    J = int(args.history_interval)
    L = int(args.history_observations)
    horizons = sorted(set(int(x) for x in args.horizons))
    subset_sizes = sorted(set(int(x) for x in args.subset_sizes))
    models = list(dict.fromkeys(str(x) for x in args.models))

    if J < 1:
        raise ValueError("history-interval必须>=1")
    if L < 2:
        raise ValueError("history-observations必须>=2")
    if any(h < 1 for h in horizons):
        raise ValueError("horizon必须>=1")
    if any(k < 1 or k >= 5 for k in subset_sizes):
        raise ValueError("subset-sizes只能取1~4；V4.7正式建议1 2 3，Full5由V4.6A复用。")
    if not models:
        raise ValueError("models不能为空")

    out = ensure_dir(args.output_root)
    cache_root = ensure_dir(out / "Cache_Folds")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(args.threads_model))))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(max(1, int(args.threads_model))))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

    set_seed(RANDOM_SEED)
    tf_info = configure_tensorflow_if_needed(models)

    config = {
        "Task": "RWTH48 V4.7 source-only nested HI-count ablation",
        "StrictInput": str(args.strict_input),
        "HIInput": str(args.hi_input),
        "BaselineOOF": str(args.baseline_oof),
        "V46AComplete": str(args.v46a_complete),
        "Output": str(out),
        "Checkpoints_s": list(CHECKPOINTS_S),
        "CheckpointDefinition": "cumulative 0-T prefixes",
        "FullHI": list(COMPACT_HI),
        "SubsetSizesTrained": subset_sizes,
        "Full5Reused": True,
        "RankingMode": args.ranking_mode,
        "RankingUsesHeldOutBattery": False,
        "RankingUsesSourceTargetLabels": args.ranking_mode == "source_spearman",
        "RankingPerSourceBatteryCap": RANK_PER_SOURCE_BATTERY,
        "HistoryInterval_J": J,
        "HistoryObservations_L": L,
        "HistorySpan_cycles": (L - 1) * J,
        "Horizons": horizons,
        "Models": models,
        "OnlyHIAsModelInput": True,
        "GlobalCycleAsFeature": False,
        "HistoricalSOHAsFeature": False,
        "PeriodicSOHCalibration": False,
        "FutureHI": False,
        "NearestCycleSubstitution": False,
        "OuterLOBO": "46 source batteries train / 1 whole held-out battery test; rotate over 47.",
        "TrainingCapsPerSourceBattery": {
            "RF": RF_PER_SOURCE_BATTERY,
            "XGBoost": XGB_PER_SOURCE_BATTERY,
            "LSTM": DEEP_PER_SOURCE_BATTERY,
            "BiLSTM": DEEP_PER_SOURCE_BATTERY,
        },
        "RFParams": RF_PARAMS,
        "XGBParams": XGB_PARAMS,
        "DeepEpochs": int(args.deep_epochs),
        "DeepBatch": int(args.deep_batch),
        "TensorFlow": tf_info,
        "Important": [
            "V4.7 does not re-extract HI; it consumes frozen V4.6A cumulative-prefix HI.",
            "Full5 predictions are reused from V4.6A after exact key/target equivalence checks.",
            "TopK selection is performed inside each outer fold using source batteries only.",
            "All four models share the same fold-specific HI ranking.",
            "No held-out battery participates in ranking, imputation, scaling or model fitting.",
            "No cycle index, historical SOH, RPT SOH, capacity or future HI enters X.",
        ],
    }
    save_json(config, out / "00_config.json")

    print("=" * 118)
    print("RWTH48 V4.7 | Source-only HI-count Ablation | Top1/Top2/Top3 vs Full5")
    print("STRICT       :", args.strict_input)
    print("HI           :", args.hi_input)
    print("V4.6A OOF    :", args.baseline_oof)
    print("OUTPUT       :", out)
    print("J / L        :", J, "/", L)
    print("HORIZONS     :", horizons)
    print("SUBSET SIZES :", subset_sizes, "+ Full5(reused)")
    print("RANKING      :", args.ranking_mode, "| SOURCE ONLY")
    print("MODELS       :", models)
    print("TF/GPU       :", tf_info)
    print("=" * 118, flush=True)

    # -------------------------------------------------------------------------
    # A. Inputs / compatibility
    # -------------------------------------------------------------------------
    print("[A] Loading frozen V4.5C-FIX / V4.6A inputs ...", flush=True)

    strict = load_strict_target(args.strict_input)
    hi = load_prefix_hi(args.hi_input)
    complete = load_v46a_complete(args.v46a_complete)
    baseline = load_baseline_oof(args.baseline_oof, horizons, models)

    compat = validate_v46a_configuration(complete, J, L, horizons)

    batteries = sorted(strict["Battery"].unique())
    hi_batteries = sorted(hi["Battery"].unique())

    compat_extra = pd.DataFrame(
        [
            {
                "Check": "Strict batteries after RWTH-028 exclusion",
                "Observed": len(batteries),
                "Expected": 47,
                "PASS": int(len(batteries) == 47),
            },
            {
                "Check": "HI covers all strict batteries",
                "Observed": len(set(batteries) & set(hi_batteries)),
                "Expected": len(batteries),
                "PASS": int(set(batteries).issubset(set(hi_batteries))),
            },
        ]
    )
    compat = pd.concat([compat, compat_extra], ignore_index=True)
    compat.to_csv(out / "01_input_compatibility_audit.csv", index=False, encoding="utf-8-sig")

    if not bool((compat["PASS"].astype(int) == 1).all()):
        raise RuntimeError(
            "V4.7输入与V4.6A冻结配置不兼容：\n"
            + compat.to_string(index=False)
        )

    merged = strict[["Battery", "GlobalCycle", SEGMENT, TARGET]].merge(
        hi[["Battery", "GlobalCycle", *ALL_HI_COLUMNS]],
        on=["Battery", "GlobalCycle"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError("strict target 与 V4.6A prefix HI 无交集")

    if args.only_battery:
        only_battery = normalize_battery(args.only_battery)
        if only_battery not in batteries:
            raise ValueError(f"--only-battery不存在：{only_battery}")
        outer_test_batteries = [only_battery]
    else:
        outer_test_batteries = batteries

    # -------------------------------------------------------------------------
    # B. Build exact windows + audit Full5 keys before any training
    # -------------------------------------------------------------------------
    print("[B] Rebuilding exact V4.6A windows and auditing frozen Full5 OOF ...", flush=True)

    bundles: Dict[int, WindowBundle] = {}
    window_rows: List[Dict[str, object]] = []
    key_audit_rows: List[Dict[str, object]] = []

    for h in horizons:
        bundle = build_sparse_history_windows(merged, J, L, h)
        if len(bundle.y) == 0:
            raise RuntimeError(f"H={h} 无合法窗口")
        bundles[h] = bundle

        counts = pd.Series(bundle.battery).value_counts().sort_index()
        window_rows.append(
            {
                "Horizon": int(h),
                "TotalWindows": int(len(bundle.y)),
                "BatteriesWithWindows": int(len(counts)),
                "MinWindowsPerBattery": int(counts.min()),
                "MedianWindowsPerBattery": float(counts.median()),
                "MaxWindowsPerBattery": int(counts.max()),
            }
        )

        audit = compare_baseline_keys(
            bundle,
            baseline[baseline["Horizon"].eq(int(h))],
            h,
            outer_test_batteries,
        )
        key_audit_rows.append(audit)

    window_inventory = pd.DataFrame(window_rows)
    window_inventory.to_csv(out / "02_window_inventory.csv", index=False, encoding="utf-8-sig")

    key_audit = pd.DataFrame(key_audit_rows)
    key_audit.to_csv(out / "02b_Full5_exact_key_audit.csv", index=False, encoding="utf-8-sig")

    if not bool((key_audit["PASS"].astype(int) == 1).all()):
        raise RuntimeError(
            "重建窗口与V4.6A frozen Full5 OOF不完全一致；拒绝做数量消融。\n"
            + key_audit.to_string(index=False)
        )

    if args.audit_only:
        print("\n--audit-only PASS：输入、窗口与 Full5 OOF 完全兼容。", flush=True)
        return

    # -------------------------------------------------------------------------
    # C. Ranking + subset training
    # -------------------------------------------------------------------------
    print("[C] Source-only ranking + TopK LOBO training ...", flush=True)

    ranking_parts: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []
    pred_parts: List[pd.DataFrame] = []
    runtime_rows: List[Dict[str, object]] = []

    for h in horizons:
        bundle = bundles[h]
        all_bundle_bats = sorted(set(bundle.battery.tolist()))

        for fold_i, test_battery in enumerate(outer_test_batteries, start=1):
            if test_battery not in all_bundle_bats:
                raise RuntimeError(f"H={h} | {test_battery} 无任何合法窗口")

            source_bats = [b for b in all_bundle_bats if b != test_battery]

            print(
                f"  H={h:>3} | fold {fold_i:02d}/{len(outer_test_batteries):02d} | "
                f"held-out={test_battery} | ranking ...",
                flush=True,
            )

            ranking = rank_hi_source_only(
                bundle,
                source_bats,
                h,
                test_battery,
                args.ranking_mode,
                args.fixed_order,
            )
            ranking_parts.append(ranking)
            order = ranking.sort_values("Rank")["HI"].astype(str).tolist()

            print("      source-only order:", " > ".join(order), flush=True)

            for k in subset_sizes:
                selected = order[:k]
                manifest_rows.append(
                    {
                        "Horizon": int(h),
                        "TestBattery": test_battery,
                        "HI_Set": f"Top{k}",
                        "HI_Count": int(k),
                        "SelectedHI": "|".join(selected),
                        "InputFeaturesPerObservation": int(4 * k),
                        "HistoryScalarInputs": int(4 * k * L),
                        "RankingMode": args.ranking_mode,
                    }
                )

                fold_dir = ensure_dir(
                    cache_root
                    / f"H{int(h):03d}"
                    / test_battery
                )
                pred_cache = fold_dir / f"Top{k}_predictions.csv.gz"
                metric_cache = fold_dir / f"Top{k}_metrics.csv"

                t_fold = time.perf_counter()
                pred_df, metric_df = run_subset_fold(
                    bundle=bundle,
                    test_battery=test_battery,
                    horizon=h,
                    selected_hi=selected,
                    hi_count=k,
                    models=models,
                    args=args,
                    fold_cache=pred_cache,
                    metric_cache=metric_cache,
                )
                pred_parts.append(pred_df)

                runtime_rows.append(
                    {
                        "Horizon": int(h),
                        "TestBattery": test_battery,
                        "HI_Set": f"Top{k}",
                        "HI_Count": int(k),
                        "SelectedHI": "|".join(selected),
                        "FoldWall_s": float(time.perf_counter() - t_fold),
                        "CachePrediction": str(pred_cache),
                    }
                )

                msg = []
                for rr in metric_df.itertuples(index=False):
                    if str(rr.Model) in models:
                        msg.append(f"{rr.Model}:RMSE={float(rr.RMSE):.6f}")
                print(f"      Top{k} [{', '.join(selected)}] | " + " | ".join(msg), flush=True)

    ranking_df = pd.concat(ranking_parts, ignore_index=True)
    manifest_df = pd.DataFrame(manifest_rows)

    ranking_df.to_csv(
        out / "03_source_only_HI_ranking_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest_df.to_csv(
        out / "04_selected_HI_manifest_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    subset_oof = pd.concat(pred_parts, ignore_index=True)

    # Frozen Full5 rows are filtered to exactly the folds/horizons run now.
    full5 = baseline_full5_rows(
        baseline,
        models,
        outer_test_batteries,
    )

    # -------------------------------------------------------------------------
    # D. Exact common cohort gate for every TopK
    # -------------------------------------------------------------------------
    cohort_gate_rows: List[Dict[str, object]] = []
    for hi_set, g in subset_oof.groupby("HI_Set", sort=True):
        cohort_gate_rows.append(
            validate_subset_vs_baseline(g, full5, str(hi_set))
        )

    if not all(int(r["PASS"]) == 1 for r in cohort_gate_rows):
        raise RuntimeError(
            "TopK 与 Full5 测试 cohort 不一致；拒绝比较。\n"
            + pd.DataFrame(cohort_gate_rows).to_string(index=False)
        )

    oof = pd.concat([subset_oof, full5], ignore_index=True, sort=False)
    oof = oof.sort_values(
        ["Horizon", "HI_Count", "Battery", "OriginCycle_KeyOnly"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # E. Aggregate
    # -------------------------------------------------------------------------
    print("[D] Aggregating fair common-cohort ablation results ...", flush=True)

    fold_metrics = fold_metrics_from_oof(oof, models)
    pooled = pooled_metrics(oof, models)
    macro = macro_metrics(fold_metrics)
    gain = gain_vs_full5(pooled)
    freq = selection_frequency(ranking_df)
    efficiency = input_efficiency_summary(pooled, gain)
    runtime_df = pd.DataFrame(runtime_rows)

    fold_metrics.to_csv(out / "05_fold_model_metrics.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(out / "06_pooled_OOF_metrics.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(out / "07_macro_LOBO_metrics.csv", index=False, encoding="utf-8-sig")
    gain.to_csv(out / "08_HI_count_gain_vs_Full5.csv", index=False, encoding="utf-8-sig")
    freq.to_csv(out / "09_HI_selection_frequency.csv", index=False, encoding="utf-8-sig")
    efficiency.to_csv(out / "10_input_efficiency_summary.csv", index=False, encoding="utf-8-sig")
    runtime_df.to_csv(out / "11_runtime_summary.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(
        out / "13_ALL_OOF_HI_count_ablation.csv.gz",
        index=False,
        compression=FAST_GZIP,
    )

    # -------------------------------------------------------------------------
    # F. Acceptance gate
    # -------------------------------------------------------------------------
    gate_rows: List[Dict[str, object]] = []

    def gate(check: str, passed: bool, detail: object) -> None:
        gate_rows.append(
            {
                "Check": check,
                "PASS": int(bool(passed)),
                "Detail": str(detail),
            }
        )

    gate("Input compatibility", bool((compat["PASS"].astype(int) == 1).all()), "see 01_input_compatibility_audit.csv")
    gate("47 strict batteries", len(batteries) == 47, len(batteries))
    gate("Full5 exact rebuilt keys", bool((key_audit["PASS"].astype(int) == 1).all()), "see 02b_Full5_exact_key_audit.csv")
    gate("TopK exact common cohort", all(int(r["PASS"]) == 1 for r in cohort_gate_rows), cohort_gate_rows)

    expected_fold_count = len(outer_test_batteries) * len(horizons) * len(subset_sizes)
    observed_manifest = len(manifest_df)
    gate("All requested TopK folds selected", observed_manifest == expected_fold_count, f"observed={observed_manifest}, expected={expected_fold_count}")

    expected_sets = {f"Top{k}" for k in subset_sizes} | {"Full5"}
    observed_sets = set(oof["HI_Set"].astype(str).unique())
    gate("All HI sets present", expected_sets.issubset(observed_sets), f"observed={sorted(observed_sets)}")

    duplicate = oof.duplicated([*OOF_KEY, "HI_Set"]).any()
    gate("No duplicate OOF key within HI set", not duplicate, f"duplicate={duplicate}")

    finite_pred_ok = True
    finite_detail = []
    for m in models:
        c = f"Pred_{m}"
        s = pd.to_numeric(oof[c], errors="coerce")
        bad = int((~np.isfinite(s.to_numpy(float))).sum())
        finite_detail.append(f"{m}:bad={bad}")
        finite_pred_ok = finite_pred_ok and bad == 0
    gate("All requested OOF predictions finite", finite_pred_ok, "; ".join(finite_detail))

    count_ok = True
    for r in manifest_df.itertuples(index=False):
        selected = str(r.SelectedHI).split("|") if str(r.SelectedHI) else []
        if len(selected) != int(r.HI_Count) or len(set(selected)) != int(r.HI_Count):
            count_ok = False
            break
    gate("Selected HI count equals requested K", count_ok, "nested source-only TopK")

    if not args.only_battery:
        expected_outer = 47
        observed_outer = int(oof["Battery"].nunique())
        gate("Formal run covers all 47 held-out batteries", observed_outer == expected_outer, observed_outer)
    else:
        gate("Smoke run held-out battery", int(oof["Battery"].nunique()) == 1, sorted(oof["Battery"].unique()))

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(out / "12_acceptance_gate.csv", index=False, encoding="utf-8-sig")
    gate_pass = bool((gate_df["PASS"].astype(int) == 1).all())

    if not args.skip_figures:
        make_figures(pooled, gain, out)

    # -------------------------------------------------------------------------
    # G. Completion summary
    # -------------------------------------------------------------------------
    elapsed = float(time.perf_counter() - t_all)

    best_reduced: List[Dict[str, object]] = []
    reduced = pooled[pooled["HI_Count"].lt(5)].copy()
    for (h, model), g in reduced.groupby(["Horizon", "Model"], sort=True):
        r = g.sort_values(["RMSE", "HI_Count", "MAE"]).iloc[0]
        best_reduced.append(
            {
                "Horizon": int(h),
                "Model": str(model),
                "BestReducedHISet": str(r["HI_Set"]),
                "HI_Count": int(r["HI_Count"]),
                "RMSE": float(r["RMSE"]),
                "MAE": float(r["MAE"]),
                "R2": float(r["R2"]),
            }
        )

    # A pragmatic paper-facing recommendation: smallest K within 5% RMSE of Full5.
    smallest_within_5pct: List[Dict[str, object]] = []
    for (h, model), g in pooled.groupby(["Horizon", "Model"], sort=True):
        full = g[g["HI_Set"].eq("Full5")]
        if full.empty:
            continue
        full_rmse = float(full.iloc[0]["RMSE"])
        q = g[g["HI_Count"].lt(5)].sort_values("HI_Count")
        eligible = q[q["RMSE"].le(full_rmse * 1.05)]
        if len(eligible):
            r = eligible.iloc[0]
            smallest_within_5pct.append(
                {
                    "Horizon": int(h),
                    "Model": str(model),
                    "SmallestHIWithin5pctRMSE": int(r["HI_Count"]),
                    "HI_Set": str(r["HI_Set"]),
                    "RMSE": float(r["RMSE"]),
                    "Full5_RMSE": full_rmse,
                }
            )
        else:
            smallest_within_5pct.append(
                {
                    "Horizon": int(h),
                    "Model": str(model),
                    "SmallestHIWithin5pctRMSE": None,
                    "HI_Set": "NONE",
                    "RMSE": None,
                    "Full5_RMSE": full_rmse,
                }
            )

    complete_out = {
        "Task": "RWTH48 V4.7 source-only nested HI-count ablation",
        "Status": "PASS" if gate_pass else "CHECK",
        "Batteries": 47,
        "OuterTestBatteriesRun": int(len(outer_test_batteries)),
        "HistoryInterval_J": J,
        "HistoryObservations_L": L,
        "HistorySpan_cycles": (L - 1) * J,
        "Horizons": horizons,
        "SubsetSizesTrained": subset_sizes,
        "Full5ReusedFromV46A": True,
        "RankingMode": args.ranking_mode,
        "RankingSourceOnly": True,
        "Models": models,
        "OnlyHIAsModelInput": True,
        "GlobalCycleAsFeature": False,
        "HistoricalSOHAsFeature": False,
        "PeriodicSOHCalibration": False,
        "BestReducedByHorizonModel": best_reduced,
        "SmallestHIWithin5pctFull5RMSE": smallest_within_5pct,
        "OOFRows": int(len(oof)),
        "Elapsed_s": elapsed,
        "TensorFlow": tf_info,
        "Important": [
            "Full5 is a frozen V4.6A OOF baseline and is not retrained.",
            "Top1/Top2/Top3 are selected independently inside each outer fold using source batteries only.",
            "The same source-only ranking is shared by RF/XGBoost/LSTM/BiLSTM within each fold/horizon.",
            "All HI-count comparisons use exactly the same test Battery/Origin/Target/SOH cohort.",
            "Input scalar budget is 40/80/120/200 for Top1/Top2/Top3/Full5 when L=10.",
        ],
    }
    save_json(complete_out, out / "99_RWTH48_V4_7_COMPLETE.json")

    print("\n" + "=" * 118)
    print("RWTH48 V4.7 COMPLETE")
    print("Status        :", complete_out["Status"])
    print("Outer folds   :", len(outer_test_batteries))
    print("Horizons      :", horizons)
    print("TopK trained  :", subset_sizes)
    print("Full5         : reused V4.6A OOF")
    print("OOF rows      :", f"{len(oof):,}")
    print("Elapsed       :", f"{elapsed / 3600.0:.2f} h")
    print("Output        :", out)
    print("=" * 118)

    print("\nPooled OOF RMSE:")
    show = pooled[["Horizon", "Model", "HI_Set", "HI_Count", "RMSE", "MAE", "R2"]].copy()
    print(show.sort_values(["Horizon", "Model", "HI_Count"]).to_string(index=False))

    if not gate_pass:
        print("\nWARNING: acceptance gate contains CHECK/FAIL rows; inspect 12_acceptance_gate.csv")


if __name__ == "__main__":
    main()

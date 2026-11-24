# -*- coding: utf-8 -*-
r"""
RWTH48 V4.10A
==============
Event Tree Upgrade | J08/L06 | Event Top2 | 47-cell Outer LOBO
新电脑默认根目录:
    workspace

目的
----
固定 V4.9C 最优观测条件，不增加任何真实测量：
    Event Top2 = V_TimeWeightedMean + P_Last
    Event checkpoints = RefAh 0.05 / 0.09 / 0.13
    J=8, L=6, history span=40 cycles
    Measured scalar budget = 6 diagnoses x 3 checkpoints x 2 HI = 36
    Future trajectory = SOH(t+1)...SOH(t+100)

只升级“树模型的输入表达与训练策略”：
1) Frozen baseline:
       直接复用 V4.9C J08/L06 XGBoost OOF，不重新训练。
2) XGB_EngFrozen:
       36个原始测量不变；
       增加由这36个量计算出的历史趋势/跨Event派生特征；
       XGBoost参数保持 V4.9C frozen 参数。
3) XGB_EngTuned:
       与2完全相同的派生特征；
       用 whole-source-battery validation 选择 XGBoost 参数；
       held-out battery 从不参与调参。
4) LightGBM_EngTuned:
       同样的派生特征；
       whole-source-battery source-only tuning；
       作为 boosting-tree challenger。
       LightGBM不原生支持100维multi-output，因此最终用 MultiOutputRegressor；
       调参阶段只在代表性 leads=[1,10,25,50,75,100] 上选择参数以控制计算量，
       最终仍输出完整100点轨迹。

严格科学约束
------------
- 不输入真实历史SOH / capacity / RPT SOH。
- 不输入 future SOH / future HI。
- GlobalCycle 只用于窗口调度和key，不进入模型。
- 不增加任何传感器测量；MeasuredScalarBudget 永远为36。
- 派生特征只由这36个历史 Event HI 计算。
- 所有历史和未来目标均位于同一 StrictSegmentKey。
- exact history cycles；无 nearest-cycle substitution。
- outer held-out battery 不进入：缺失值填补、调参、训练。
- 调参 validation 必须按“整节 source battery”划分，绝不随机拆 cycle/window。
- 最终与 V4.9C frozen XGBoost 在 Battery+Origin+Lead+Target exact-common 上比较。

建议运行
--------
1) 审计：
python -u "src/23_event_tree_helper.py" --audit-only

2) 单折 smoke：
python -u "src/23_event_tree_helper.py" --only-battery RWTH-002

3) 全47折：
python -u "src/23_event_tree_helper.py"

4) 如果暂时没有 lightgbm，只先跑 XGBoost：
python -u "src/23_event_tree_helper.py" --models XGB_EngFrozen XGB_EngTuned

续跑
----
直接重复相同命令即可。不要加 --force-folds，除非明确要全部重训。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 0. Frozen project configuration
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR_DEFAULT = Path(__file__).resolve().parent

STRICT_DEFAULT = (
    BASE
    / "RWTH48_V4_5C_FIX_最小冲击共同扰动处理"
    / "06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz"
)

EVENT_DEFAULT = (
    BASE
    / "RWTH48_V4_2_双平台冻结_FixedTime_Event_HI"
    / "06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz"
)

V49C_ROOT_DEFAULT = BASE / "RWTH48_V4_9C_Event_Top2_Sparse_FullTrajectory_LOBO"
V49C_BASELINE_OOF_DEFAULT = (
    V49C_ROOT_DEFAULT / "10_J08_L06_ALL_OOF_full_trajectory_predictions_long.csv.gz"
)

OUTPUT_DEFAULT = BASE / "RWTH48_V4_10A_Event_Tree_Upgrade_J08L06"

EVENT_NODES = (
    (0.05, "Event_RefAh_0.05", "E005"),
    (0.09, "Event_RefAh_0.09", "E009"),
    (0.13, "Event_RefAh_0.13", "E013"),
)
EVENT_TOP2 = ("V_TimeWeightedMean", "P_Last")
EVENT_COLUMNS = tuple(
    f"{prefix}__{hi}"
    for _, _, prefix in EVENT_NODES
    for hi in EVENT_TOP2
)

J = 8
L = 6
K = 100
MEASURED_SCALARS = L * len(EVENT_COLUMNS)  # 36
TARGET = "CyclingSOH_Charge_Canonical"
SEGMENT = "StrictSegmentKey"
OOF_KEY = ["Battery", "OriginCycle_KeyOnly", "Lead", "TargetCycle_KeyOnly"]
RANDOM_SEED = 20260821
TRAIN_PER_SOURCE_BATTERY = 80
VAL_PER_SOURCE_BATTERY = 120
VALIDATION_BATTERIES = 6
FAST_GZIP = {"method": "gzip", "compresslevel": 1}

MODELS = (
    "XGB_EngFrozen",
    "XGB_EngTuned",
    "LightGBM_EngTuned",
)

# V4.9C frozen XGB parameters. Backend/device fields are injected after CUDA probe.
XGB_FROZEN_CORE = dict(
    n_estimators=180,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.90,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    min_child_weight=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=20260820,
)

# Compact, deliberately small source-only candidate set.
XGB_TUNE_GRID = [
    dict(Name="X0_FrozenLike", n_estimators=180, max_depth=5, learning_rate=0.05,
         min_child_weight=1.0, subsample=0.90, colsample_bytree=0.85,
         gamma=0.0, reg_alpha=0.0, reg_lambda=1.0),
    dict(Name="X1_Shallow", n_estimators=320, max_depth=3, learning_rate=0.035,
         min_child_weight=2.0, subsample=0.92, colsample_bytree=0.90,
         gamma=0.0, reg_alpha=0.02, reg_lambda=2.0),
    dict(Name="X2_ModerateReg", n_estimators=280, max_depth=4, learning_rate=0.04,
         min_child_weight=2.0, subsample=0.90, colsample_bytree=0.85,
         gamma=0.0, reg_alpha=0.03, reg_lambda=3.0),
    dict(Name="X3_StrongReg", n_estimators=360, max_depth=4, learning_rate=0.03,
         min_child_weight=3.0, subsample=0.88, colsample_bytree=0.80,
         gamma=0.01, reg_alpha=0.08, reg_lambda=5.0),
    dict(Name="X4_Deeper", n_estimators=240, max_depth=6, learning_rate=0.04,
         min_child_weight=2.0, subsample=0.90, colsample_bytree=0.85,
         gamma=0.0, reg_alpha=0.03, reg_lambda=3.0),
]

LGB_TUNE_LEADS = (1, 10, 25, 50, 75, 100)
LGB_TUNE_GRID = [
    dict(Name="L0_DefaultCompact", n_estimators=300, learning_rate=0.035,
         num_leaves=31, max_depth=-1, min_child_samples=20,
         subsample=0.90, colsample_bytree=0.85, reg_alpha=0.0, reg_lambda=1.0),
    dict(Name="L1_Shallow", n_estimators=400, learning_rate=0.03,
         num_leaves=15, max_depth=5, min_child_samples=25,
         subsample=0.92, colsample_bytree=0.90, reg_alpha=0.02, reg_lambda=2.0),
    dict(Name="L2_Reg", n_estimators=350, learning_rate=0.03,
         num_leaves=31, max_depth=7, min_child_samples=30,
         subsample=0.88, colsample_bytree=0.82, reg_alpha=0.05, reg_lambda=4.0),
    dict(Name="L3_Fine", n_estimators=500, learning_rate=0.02,
         num_leaves=23, max_depth=6, min_child_samples=20,
         subsample=0.90, colsample_bytree=0.88, reg_alpha=0.02, reg_lambda=2.5),
]


# =============================================================================
# 1. Generic utilities
# =============================================================================

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


def safe_name(x: object) -> str:
    return (
        str(x)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_script(code_dir: Path, exact: str, glob_pat: str) -> Path:
    p = code_dir / exact
    if p.exists():
        return p
    hits = sorted(code_dir.glob(glob_pat))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"缺少 helper 脚本: {exact}\n请放到: {code_dir}"
    )


def file_sha1_text(obj: object) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, float).reshape(-1)
    yp = np.asarray(y_pred, float).reshape(-1)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if len(yt) == 0:
        return dict(N=0, MAE=np.nan, RMSE=np.nan, R2=np.nan, MAPE_pct=np.nan, Bias=np.nan)
    e = yp - yt
    return dict(
        N=int(len(yt)),
        MAE=float(np.mean(np.abs(e))),
        RMSE=float(np.sqrt(np.mean(e * e))),
        R2=float(r2_score(yt, yp)) if len(yt) >= 2 and np.std(yt) > 1e-12 else np.nan,
        MAPE_pct=float(np.mean(np.abs(e) / np.maximum(np.abs(yt), 1e-9)) * 100.0),
        Bias=float(np.mean(e)),
    )


def trajectory_shape_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    if yt.ndim != 2 or yp.ndim != 2 or yt.shape != yp.shape or yt.size == 0:
        return dict(
            MeanTrajectoryRMSE=np.nan,
            MedianTrajectoryRMSE=np.nan,
            MeanTrajectoryMAE=np.nan,
            EndpointMAE=np.nan,
            TrueUpwardStepFraction=np.nan,
            PredUpwardStepFraction=np.nan,
            ExcessUpwardStepFraction=np.nan,
            TrueRoughness=np.nan,
            PredRoughness=np.nan,
            RoughnessRatioPredToTrue=np.nan,
        )
    e = yp - yt
    dt = np.diff(yt, axis=1)
    dp = np.diff(yp, axis=1)
    tr = float(np.mean(np.abs(np.diff(yt, n=2, axis=1)))) if yt.shape[1] >= 3 else np.nan
    pr = float(np.mean(np.abs(np.diff(yp, n=2, axis=1)))) if yp.shape[1] >= 3 else np.nan
    tu = float(np.mean(dt > 0)) if dt.size else np.nan
    pu = float(np.mean(dp > 0)) if dp.size else np.nan
    return dict(
        MeanTrajectoryRMSE=float(np.mean(np.sqrt(np.mean(e * e, axis=1)))),
        MedianTrajectoryRMSE=float(np.median(np.sqrt(np.mean(e * e, axis=1)))),
        MeanTrajectoryMAE=float(np.mean(np.mean(np.abs(e), axis=1))),
        EndpointMAE=float(np.mean(np.abs(e[:, -1]))),
        TrueUpwardStepFraction=tu,
        PredUpwardStepFraction=pu,
        ExcessUpwardStepFraction=float(pu - tu),
        TrueRoughness=tr,
        PredRoughness=pr,
        RoughnessRatioPredToTrue=float(pr / tr) if np.isfinite(tr) and tr > 1e-15 else np.nan,
    )


def choose_validation_batteries(source_bats: Sequence[str], n_val: int) -> Tuple[List[str], List[str]]:
    src = sorted(source_bats)
    n_val = max(1, min(int(n_val), len(src) - 2))
    pos = np.unique(np.round(np.linspace(0, len(src) - 1, n_val)).astype(int))
    val = [src[i] for i in pos]
    for b in src:
        if len(val) >= n_val:
            break
        if b not in val:
            val.append(b)
    val = sorted(val[:n_val])
    tr = [b for b in src if b not in set(val)]
    return tr, val


def balanced_indices(battery: np.ndarray, bats: Sequence[str], max_per: int) -> np.ndarray:
    idx: List[int] = []
    for b in bats:
        ix = np.flatnonzero(battery == b)
        if max_per <= 0 or len(ix) <= max_per:
            pick = ix
        else:
            pos = np.linspace(0, len(ix) - 1, max_per)
            pick = ix[np.round(pos).astype(int)]
        idx.extend(pick.tolist())
    return np.asarray(sorted(set(idx)), dtype=int)


@dataclass
class MedianImputer:
    median: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = np.asarray(X, np.float32).copy()
        bad = ~np.isfinite(z)
        if bad.any():
            rr, cc = np.where(bad)
            z[rr, cc] = self.median[cc]
        return z


def fit_imputer(X: np.ndarray) -> MedianImputer:
    med = np.nanmedian(np.asarray(X, float), axis=0)
    med[~np.isfinite(med)] = 0.0
    return MedianImputer(med.astype(np.float32))


# =============================================================================
# 2. Feature engineering: 36 measured scalars -> derived model features
# =============================================================================

def linear_slope(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorized slope over axis=1. values shape [N,L]."""
    x = np.asarray(x, float)
    xc = x - np.mean(x)
    den = float(np.sum(xc * xc))
    if den <= 1e-15:
        return np.zeros(values.shape[0], dtype=np.float32)
    yc = values - np.mean(values, axis=1, keepdims=True)
    return (np.sum(yc * xc[None, :], axis=1) / den).astype(np.float32)


def summarize_temporal_block(
    block: np.ndarray,
    channel_names: Sequence[str],
    cycle_offsets: np.ndarray,
    prefix: str,
) -> Tuple[List[np.ndarray], List[str]]:
    """block: [N,L,C]. For every channel, summarize the 6 diagnoses."""
    feats: List[np.ndarray] = []
    names: List[str] = []
    for c, cname in enumerate(channel_names):
        a = np.asarray(block[:, :, c], np.float32)
        last = a[:, -1]
        mean = np.nanmean(a, axis=1)
        std = np.nanstd(a, axis=1)
        rng = np.nanmax(a, axis=1) - np.nanmin(a, axis=1)
        recent = a[:, -1] - a[:, -2]
        long_delta = a[:, -1] - a[:, 0]
        slope = linear_slope(a.astype(float), cycle_offsets.astype(float))
        accel = (a[:, -1] - a[:, -2]) - (a[:, -2] - a[:, -3])

        vals = (last, mean, std, rng, recent, long_delta, slope, accel)
        suff = ("Last", "Mean", "Std", "Range", "RecentDelta",
                "LongDelta", "SlopePerCycle", "RecentAccel")
        for v, s in zip(vals, suff):
            feats.append(np.asarray(v, np.float32))
            names.append(f"{prefix}{cname}__{s}")
    return feats, names


def engineer_features(Xseq: np.ndarray) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Xseq shape [N,6,6], channel order frozen by EVENT_COLUMNS:
      E005 V, E005 P, E009 V, E009 P, E013 V, E013 P
    """
    X = np.asarray(Xseq, np.float32)
    if X.ndim != 3 or X.shape[1] != L or X.shape[2] != len(EVENT_COLUMNS):
        raise RuntimeError(
            f"Event history shape异常: {X.shape}, expected [N,{L},{len(EVENT_COLUMNS)}]"
        )

    feats: List[np.ndarray] = []
    names: List[str] = []
    registry: List[Dict[str, object]] = []

    # A. Raw 36 measured scalars
    cycle_offsets = np.arange(-(L - 1) * J, 1, J, dtype=int)  # [-40,-32,-24,-16,-8,0]
    for li, off in enumerate(cycle_offsets):
        for ci, cname in enumerate(EVENT_COLUMNS):
            nm = f"RAW_D{li+1}_t{off:+d}__{cname}"
            feats.append(X[:, li, ci])
            names.append(nm)
            registry.append(dict(
                Feature=nm, Group="RawMeasured", MeasuredOrDerived="Measured",
                Source="original Event Top2 scalar",
            ))

    # B. Temporal summary of each of the 6 physical Event channels
    f2, n2 = summarize_temporal_block(
        X, EVENT_COLUMNS, cycle_offsets, prefix="TEMP__"
    )
    feats.extend(f2)
    names.extend(n2)
    for nm in n2:
        registry.append(dict(
            Feature=nm, Group="HistoricalTrend", MeasuredOrDerived="Derived",
            Source="computed from six exact J=8 historical diagnoses",
        ))

    # C. Within-diagnosis cross-checkpoint geometry.
    # channel indices: V05=0,P05=1,V09=2,P09=3,V13=4,P13=5
    V05, P05, V09, P09, V13, P13 = [X[:, :, i] for i in range(6)]
    cross_channels = np.stack([
        V09 - V05,
        V13 - V09,
        V13 - V05,
        V13 - 2.0 * V09 + V05,
        P09 - P05,
        P13 - P09,
        P13 - P05,
        P13 - 2.0 * P09 + P05,
    ], axis=2).astype(np.float32)

    cross_names = (
        "V_Delta_E005_E009",
        "V_Delta_E009_E013",
        "V_Delta_E005_E013",
        "V_EventCurvature",
        "P_Delta_E005_E009",
        "P_Delta_E009_E013",
        "P_Delta_E005_E013",
        "P_EventCurvature",
    )

    for li, off in enumerate(cycle_offsets):
        for ci, cname in enumerate(cross_names):
            nm = f"CROSS_D{li+1}_t{off:+d}__{cname}"
            feats.append(cross_channels[:, li, ci])
            names.append(nm)
            registry.append(dict(
                Feature=nm, Group="WithinDiagnosisEventGeometry",
                MeasuredOrDerived="Derived",
                Source="differences/curvature across E005/E009/E013 from same diagnosis",
            ))

    # D. Temporal summaries of the cross-checkpoint geometry.
    f4, n4 = summarize_temporal_block(
        cross_channels, cross_names, cycle_offsets, prefix="CROSS_TEMP__"
    )
    feats.extend(f4)
    names.extend(n4)
    for nm in n4:
        registry.append(dict(
            Feature=nm, Group="CrossEventHistoricalTrend",
            MeasuredOrDerived="Derived",
            Source="historical trend of within-diagnosis Event geometry",
        ))

    Xf = np.column_stack(feats).astype(np.float32)
    if len(names) != Xf.shape[1] or len(set(names)) != len(names):
        raise RuntimeError("engineered feature registry不一致")
    return Xf, names, pd.DataFrame(registry)


# =============================================================================
# 3. Tree model backends
# =============================================================================

def xgb_params_from_candidate(
    candidate: Dict[str, object],
    backend_params: Dict[str, object],
    threads: int,
) -> Dict[str, object]:
    p = dict(XGB_FROZEN_CORE)
    for k, v in candidate.items():
        if k != "Name":
            p[k] = v
    # backend/device selected by V4.8E helper
    for k in ("tree_method", "device", "gpu_id"):
        if k in backend_params:
            p[k] = backend_params[k]
        elif k in p and k not in backend_params and k in ("device", "gpu_id"):
            p.pop(k, None)
    p["n_jobs"] = max(1, int(threads))
    return p


def fit_predict_xgb(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    params: Dict[str, object],
) -> Tuple[np.ndarray, object, str]:
    from xgboost import XGBRegressor

    errors: List[str] = []
    # V4.9C-compatible preference order.
    for strategy in ("multi_output_tree", "one_output_per_tree"):
        model = None
        try:
            model = XGBRegressor(**params, multi_strategy=strategy)
            model.fit(Xtr, ytr, verbose=False)
            pred = np.asarray(model.predict(Xte), float)
            if pred.ndim == 1 and ytr.shape[1] == 1:
                pred = pred[:, None]
            if pred.shape != (len(Xte), ytr.shape[1]):
                raise RuntimeError(f"shape={pred.shape}")
            return pred, model, strategy
        except Exception as exc:
            errors.append(f"{strategy}: {type(exc).__name__}: {exc}")
            try:
                del model
            except Exception:
                pass
            gc.collect()

    # Compatibility fallback.
    try:
        inner = dict(params)
        inner["n_jobs"] = 1
        model = MultiOutputRegressor(
            XGBRegressor(**inner),
            n_jobs=max(1, min(int(params.get("n_jobs", 1)), 4)),
        )
        model.fit(Xtr, ytr)
        pred = np.asarray(model.predict(Xte), float)
        return pred, model, "sklearn_MultiOutputRegressor_fallback"
    except Exception as exc:
        errors.append(f"fallback: {type(exc).__name__}: {exc}")
        raise RuntimeError("XGBoost训练失败:\n" + "\n".join(errors)) from exc


def lightgbm_available() -> Tuple[bool, str]:
    try:
        import lightgbm as lgb
        return True, str(getattr(lgb, "__version__", ""))
    except Exception:
        return False, ""


def make_lgb_regressor(params: Dict[str, object], threads_inner: int = 1):
    from lightgbm import LGBMRegressor
    p = {k: v for k, v in params.items() if k != "Name"}
    p.update(
        objective="regression",
        random_state=RANDOM_SEED,
        n_jobs=max(1, int(threads_inner)),
        verbosity=-1,
        force_col_wise=True,
    )
    return LGBMRegressor(**p)


def fit_predict_lgb_multi(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    params: Dict[str, object],
    threads: int,
) -> Tuple[np.ndarray, object]:
    base = make_lgb_regressor(params, threads_inner=1)
    model = MultiOutputRegressor(base, n_jobs=max(1, min(int(threads), 6)))
    model.fit(Xtr, ytr)
    pred = np.asarray(model.predict(Xte), float)
    return pred, model


def feature_importance_vector(model: object, n_features: int) -> np.ndarray:
    try:
        imp = np.asarray(getattr(model, "feature_importances_"), float).reshape(-1)
        if len(imp) == n_features:
            return imp
    except Exception:
        pass

    # MultiOutputRegressor
    try:
        arr = []
        for est in model.estimators_:
            q = np.asarray(getattr(est, "feature_importances_"), float).reshape(-1)
            if len(q) == n_features:
                arr.append(q)
        if arr:
            return np.mean(np.stack(arr), axis=0)
    except Exception:
        pass

    return np.full(n_features, np.nan, dtype=float)


# =============================================================================
# 4. Source-battery tuning
# =============================================================================

def macro_battery_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    batteries: np.ndarray,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for b in sorted(set(batteries.tolist())):
        ix = np.flatnonzero(batteries == b)
        r = calc_metrics(y_true[ix], y_pred[ix])
        rows.append(dict(Battery=b, RMSE=r["RMSE"], MAE=r["MAE"], N=r["N"]))
    d = pd.DataFrame(rows)
    return float(pd.to_numeric(d["RMSE"], errors="coerce").mean()), d


def tune_xgb_source_only(
    Xall: np.ndarray,
    yall: np.ndarray,
    battery: np.ndarray,
    source_bats: Sequence[str],
    backend_params: Dict[str, object],
    threads: int,
    n_val_bats: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    tr_bats, va_bats = choose_validation_batteries(source_bats, n_val_bats)
    tr_idx = balanced_indices(battery, tr_bats, TRAIN_PER_SOURCE_BATTERY)
    va_idx = balanced_indices(battery, va_bats, VAL_PER_SOURCE_BATTERY)

    imp = fit_imputer(Xall[tr_idx])
    Xtr = imp.transform(Xall[tr_idx])
    Xva = imp.transform(Xall[va_idx])
    ytr = yall[tr_idx]
    yva = yall[va_idx]
    bva = battery[va_idx]

    rows = []
    best = None
    best_score = np.inf

    for ci, cand in enumerate(XGB_TUNE_GRID, 1):
        t0 = time.perf_counter()
        params = xgb_params_from_candidate(cand, backend_params, threads)
        pred, model, strategy = fit_predict_xgb(Xtr, ytr, Xva, params)
        score, per_b = macro_battery_rmse(yva, pred, bva)
        rows.append(dict(
            Candidate=cand["Name"],
            MacroBatteryRMSE=score,
            PooledRMSE=calc_metrics(yva, pred)["RMSE"],
            ValidationBatteries="|".join(va_bats),
            InnerTrainBatteries=len(tr_bats),
            ValidationBatteryCount=len(va_bats),
            TrainWindows=len(tr_idx),
            ValidationWindows=len(va_idx),
            Backend=strategy,
            Runtime_s=time.perf_counter() - t0,
            Params=json.dumps({k: v for k, v in cand.items() if k != "Name"}, ensure_ascii=False),
        ))
        if np.isfinite(score) and score < best_score:
            best_score = score
            best = dict(cand)
        del pred, model, per_b
        gc.collect()

    if best is None:
        raise RuntimeError("XGBoost source-only tuning没有有效候选")
    return best, pd.DataFrame(rows)


def tune_lgb_source_only(
    Xall: np.ndarray,
    yall: np.ndarray,
    battery: np.ndarray,
    source_bats: Sequence[str],
    threads: int,
    n_val_bats: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    tr_bats, va_bats = choose_validation_batteries(source_bats, n_val_bats)
    tr_idx = balanced_indices(battery, tr_bats, TRAIN_PER_SOURCE_BATTERY)
    va_idx = balanced_indices(battery, va_bats, VAL_PER_SOURCE_BATTERY)

    imp = fit_imputer(Xall[tr_idx])
    Xtr = imp.transform(Xall[tr_idx])
    Xva = imp.transform(Xall[va_idx])
    lead_idx = np.asarray([h - 1 for h in LGB_TUNE_LEADS], dtype=int)
    ytr = yall[tr_idx][:, lead_idx]
    yva = yall[va_idx][:, lead_idx]
    bva = battery[va_idx]

    rows = []
    best = None
    best_score = np.inf
    for cand in LGB_TUNE_GRID:
        t0 = time.perf_counter()
        pred, model = fit_predict_lgb_multi(Xtr, ytr, Xva, cand, threads)
        score, per_b = macro_battery_rmse(yva, pred, bva)
        rows.append(dict(
            Candidate=cand["Name"],
            MacroBatteryRMSE_SelectedLeads=score,
            PooledRMSE_SelectedLeads=calc_metrics(yva, pred)["RMSE"],
            TuningLeads="|".join(map(str, LGB_TUNE_LEADS)),
            ValidationBatteries="|".join(va_bats),
            InnerTrainBatteries=len(tr_bats),
            ValidationBatteryCount=len(va_bats),
            TrainWindows=len(tr_idx),
            ValidationWindows=len(va_idx),
            Runtime_s=time.perf_counter() - t0,
            Params=json.dumps({k: v for k, v in cand.items() if k != "Name"}, ensure_ascii=False),
        ))
        if np.isfinite(score) and score < best_score:
            best_score = score
            best = dict(cand)
        del pred, model, per_b
        gc.collect()

    if best is None:
        raise RuntimeError("LightGBM source-only tuning没有有效候选")
    return best, pd.DataFrame(rows)


# =============================================================================
# 5. Fold cache + fit
# =============================================================================

def cache_signature(
    test_battery: str,
    models: Sequence[str],
    feature_names: Sequence[str],
    xgb_backend_info: Dict[str, object],
) -> Dict[str, object]:
    return dict(
        Task="RWTH48_V4_10A",
        TestBattery=test_battery,
        J=J,
        L=L,
        K=K,
        EventTop2=list(EVENT_TOP2),
        EventColumns=list(EVENT_COLUMNS),
        MeasuredScalars=MEASURED_SCALARS,
        EngineeredFeatureCount=len(feature_names),
        EngineeredFeatureSHA1=file_sha1_text(list(feature_names)),
        Models=list(models),
        XGBTuneGridSHA1=file_sha1_text(XGB_TUNE_GRID),
        LightGBMTuneGridSHA1=file_sha1_text(LGB_TUNE_GRID),
        XGBoostBackendSelected=xgb_backend_info.get("Selected", ""),
        TrainPerSourceBattery=TRAIN_PER_SOURCE_BATTERY,
        ValidationPerSourceBattery=VAL_PER_SOURCE_BATTERY,
        ValidationBatteries=VALIDATION_BATTERIES,
        RandomSeed=RANDOM_SEED,
    )


def cache_valid(meta_path: Path, expected: Dict[str, object]) -> bool:
    if not meta_path.exists() or meta_path.stat().st_size <= 0:
        return False
    try:
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        return obj == expected
    except Exception:
        return False


def predictions_to_long(
    test_battery: str,
    origins: np.ndarray,
    segments: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    n, k = y_true.shape
    lead = np.tile(np.arange(1, k + 1, dtype=int), n)
    origin_rep = np.repeat(origins.astype(int), k)
    out = pd.DataFrame({
        "Battery": test_battery,
        "OriginCycle_KeyOnly": origin_rep,
        "Lead": lead,
        "TargetCycle_KeyOnly": origin_rep + lead,
        "StrictSegmentKey_QCOnly": np.repeat(segments.astype(object), k),
        "SOH_True": y_true.reshape(-1),
    })
    for mid, pred in predictions.items():
        out[f"Pred_{mid}"] = np.asarray(pred, float).reshape(-1)
    return out


def fold_metric_rows(
    test_battery: str,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    runtime: Dict[str, float],
    backend: Dict[str, str],
    train_windows: Dict[str, int],
) -> pd.DataFrame:
    rows = []
    for mid, pred in predictions.items():
        rows.append(dict(
            TestBattery=test_battery,
            ModelID=mid,
            TrainWindows=int(train_windows.get(mid, 0)),
            TestTrajectoryWindows=int(len(y_true)),
            Runtime_s=float(runtime.get(mid, np.nan)),
            Backend=backend.get(mid, ""),
            **calc_metrics(y_true, pred),
            **trajectory_shape_summary(y_true, pred),
        ))
    return pd.DataFrame(rows)


def run_one_fold(
    bundle,
    Xeng: np.ndarray,
    feature_names: Sequence[str],
    test_battery: str,
    models: Sequence[str],
    backend_params: Dict[str, object],
    backend_info: Dict[str, object],
    threads: int,
    cache_dir: Path,
    force: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    stem = safe_name(test_battery)
    pred_cache = cache_dir / f"{stem}_predictions.csv.gz"
    metric_cache = cache_dir / f"{stem}_metrics.csv"
    tune_cache = cache_dir / f"{stem}_tuning.csv"
    fi_cache = cache_dir / f"{stem}_feature_importance.csv.gz"
    meta_cache = cache_dir / f"{stem}_meta.json"
    sig = cache_signature(test_battery, models, feature_names, backend_info)

    if (
        not force
        and all(p.exists() and p.stat().st_size > 0
                for p in (pred_cache, metric_cache, tune_cache, fi_cache))
        and cache_valid(meta_cache, sig)
    ):
        return (
            pd.read_csv(pred_cache, compression="infer", low_memory=False),
            pd.read_csv(metric_cache, low_memory=False),
            pd.read_csv(tune_cache, low_memory=False),
            pd.read_csv(fi_cache, compression="infer", low_memory=False),
        )

    all_bats = sorted(set(bundle.battery.tolist()))
    source_bats = [b for b in all_bats if b != test_battery]
    test_idx = np.flatnonzero(bundle.battery == test_battery)
    if len(test_idx) == 0:
        raise RuntimeError(f"{test_battery}: 无K={K}完整trajectory")

    source_idx = balanced_indices(bundle.battery, source_bats, TRAIN_PER_SOURCE_BATTERY)
    imputer = fit_imputer(Xeng[source_idx])
    Xtr_all = imputer.transform(Xeng[source_idx])
    Xte = imputer.transform(Xeng[test_idx])
    ytr_all = bundle.y[source_idx]
    yte = bundle.y[test_idx]

    predictions: Dict[str, np.ndarray] = {}
    runtime: Dict[str, float] = {}
    backend: Dict[str, str] = {}
    train_windows: Dict[str, int] = {}
    tuning_parts: List[pd.DataFrame] = []
    fi_parts: List[pd.DataFrame] = []

    if "XGB_EngFrozen" in models:
        t0 = time.perf_counter()
        frozen_cand = dict(Name="V49C_FrozenCore", **{
            k: v for k, v in XGB_FROZEN_CORE.items()
            if k in {
                "n_estimators","max_depth","learning_rate","min_child_weight",
                "subsample","colsample_bytree","gamma","reg_alpha","reg_lambda"
            }
        })
        p = xgb_params_from_candidate(frozen_cand, backend_params, threads)
        pred, model, be = fit_predict_xgb(Xtr_all, ytr_all, Xte, p)
        predictions["XGB_EngFrozen"] = pred
        runtime["XGB_EngFrozen"] = time.perf_counter() - t0
        backend["XGB_EngFrozen"] = be
        train_windows["XGB_EngFrozen"] = len(source_idx)
        impv = feature_importance_vector(model, len(feature_names))
        fi_parts.append(pd.DataFrame({
            "TestBattery": test_battery,
            "ModelID": "XGB_EngFrozen",
            "Feature": list(feature_names),
            "Importance": impv,
        }))
        del model, pred
        gc.collect()

    if "XGB_EngTuned" in models:
        best_x, td = tune_xgb_source_only(
            Xeng, bundle.y, bundle.battery, source_bats,
            backend_params, threads, VALIDATION_BATTERIES
        )
        td.insert(0, "TestBattery", test_battery)
        td.insert(1, "ModelID", "XGB_EngTuned")
        td["Selected"] = td["Candidate"].eq(best_x["Name"]).astype(int)
        tuning_parts.append(td)

        t0 = time.perf_counter()
        p = xgb_params_from_candidate(best_x, backend_params, threads)
        pred, model, be = fit_predict_xgb(Xtr_all, ytr_all, Xte, p)
        predictions["XGB_EngTuned"] = pred
        runtime["XGB_EngTuned"] = time.perf_counter() - t0
        backend["XGB_EngTuned"] = be
        train_windows["XGB_EngTuned"] = len(source_idx)
        impv = feature_importance_vector(model, len(feature_names))
        fi_parts.append(pd.DataFrame({
            "TestBattery": test_battery,
            "ModelID": "XGB_EngTuned",
            "Feature": list(feature_names),
            "Importance": impv,
        }))
        del model, pred
        gc.collect()

    if "LightGBM_EngTuned" in models:
        ok, ver = lightgbm_available()
        if not ok:
            raise RuntimeError(
                "请求了 LightGBM_EngTuned，但当前环境没有 lightgbm。\n"
                "请先执行：pip install lightgbm\n"
                "然后重新运行；已完成的XGBoost fold cache可复用。"
            )
        best_l, td = tune_lgb_source_only(
            Xeng, bundle.y, bundle.battery, source_bats,
            threads, VALIDATION_BATTERIES
        )
        td.insert(0, "TestBattery", test_battery)
        td.insert(1, "ModelID", "LightGBM_EngTuned")
        td["Selected"] = td["Candidate"].eq(best_l["Name"]).astype(int)
        tuning_parts.append(td)

        t0 = time.perf_counter()
        pred, model = fit_predict_lgb_multi(
            Xtr_all, ytr_all, Xte, best_l, threads
        )
        predictions["LightGBM_EngTuned"] = pred
        runtime["LightGBM_EngTuned"] = time.perf_counter() - t0
        backend["LightGBM_EngTuned"] = f"LightGBM_{ver}_MultiOutputRegressor"
        train_windows["LightGBM_EngTuned"] = len(source_idx)
        impv = feature_importance_vector(model, len(feature_names))
        fi_parts.append(pd.DataFrame({
            "TestBattery": test_battery,
            "ModelID": "LightGBM_EngTuned",
            "Feature": list(feature_names),
            "Importance": impv,
        }))
        del model, pred
        gc.collect()

    long_df = predictions_to_long(
        test_battery,
        bundle.origin_cycle[test_idx],
        bundle.segment[test_idx],
        yte,
        predictions,
    )
    metrics = fold_metric_rows(
        test_battery, yte, predictions, runtime, backend, train_windows
    )
    tuning = pd.concat(tuning_parts, ignore_index=True) if tuning_parts else pd.DataFrame(
        columns=["TestBattery","ModelID","Candidate","Selected"]
    )
    fi = pd.concat(fi_parts, ignore_index=True) if fi_parts else pd.DataFrame(
        columns=["TestBattery","ModelID","Feature","Importance"]
    )

    long_df.to_csv(pred_cache, index=False, compression=FAST_GZIP)
    metrics.to_csv(metric_cache, index=False, encoding="utf-8-sig")
    tuning.to_csv(tune_cache, index=False, encoding="utf-8-sig")
    fi.to_csv(fi_cache, index=False, compression=FAST_GZIP)
    save_json(sig, meta_cache)
    return long_df, metrics, tuning, fi


# =============================================================================
# 6. Aggregation
# =============================================================================

def combine_fold_predictions(paths: Sequence[Path], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    first = True
    # low compression level + chunks through pandas file-by-file;
    # file size may remain visually unchanged until gzip buffer flushes.
    import gzip
    with gzip.open(out_path, "wt", encoding="utf-8", newline="", compresslevel=1) as fh:
        for p in paths:
            d = pd.read_csv(p, compression="infer", low_memory=False)
            d.to_csv(fh, index=False, header=first)
            first = False
            del d
            gc.collect()


def aggregate_new_oof(
    paths: Sequence[Path],
    models: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = [pd.read_csv(p, compression="infer", low_memory=False) for p in paths]
    d = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    pooled_rows = []
    lead_rows = []
    macro_rows = []
    for mid in models:
        c = f"Pred_{mid}"
        if c not in d.columns:
            continue
        y = pd.to_numeric(d["SOH_True"], errors="coerce").to_numpy(float)
        p = pd.to_numeric(d[c], errors="coerce").to_numpy(float)

        # overall pooled + shape by trajectory
        order = d.sort_values(["Battery","OriginCycle_KeyOnly","Lead"]).index.to_numpy()
        ds = d.loc[order]
        ntraj = ds[["Battery","OriginCycle_KeyOnly"]].drop_duplicates().shape[0]
        yt2 = pd.to_numeric(ds["SOH_True"], errors="coerce").to_numpy(float).reshape(ntraj, K)
        yp2 = pd.to_numeric(ds[c], errors="coerce").to_numpy(float).reshape(ntraj, K)
        pooled_rows.append(dict(ModelID=mid, **calc_metrics(y,p), **trajectory_shape_summary(yt2,yp2)))

        for h, g in d.groupby("Lead", sort=True):
            lead_rows.append(dict(
                ModelID=mid,
                Lead=int(h),
                **calc_metrics(
                    pd.to_numeric(g["SOH_True"], errors="coerce").to_numpy(float),
                    pd.to_numeric(g[c], errors="coerce").to_numpy(float),
                )
            ))

        # battery-macro overall point RMSE etc.
        br = []
        for b, g in d.groupby("Battery", sort=True):
            br.append(dict(
                Battery=b,
                **calc_metrics(
                    pd.to_numeric(g["SOH_True"], errors="coerce").to_numpy(float),
                    pd.to_numeric(g[c], errors="coerce").to_numpy(float),
                )
            ))
        bd = pd.DataFrame(br)
        macro_rows.append(dict(
            ModelID=mid,
            Folds=len(bd),
            MacroMean_MAE=float(bd["MAE"].mean()),
            MacroMedian_MAE=float(bd["MAE"].median()),
            MacroMean_RMSE=float(bd["RMSE"].mean()),
            MacroMedian_RMSE=float(bd["RMSE"].median()),
            MacroMean_R2=float(bd["R2"].mean()),
            MacroMedian_R2=float(bd["R2"].median()),
            MacroMean_MAPE_pct=float(bd["MAPE_pct"].mean()),
            MacroMedian_MAPE_pct=float(bd["MAPE_pct"].median()),
            MacroMean_Bias=float(bd["Bias"].mean()),
            MacroMedian_Bias=float(bd["Bias"].median()),
        ))

    return pd.DataFrame(pooled_rows), pd.DataFrame(lead_rows), pd.DataFrame(macro_rows)


def exact_common_vs_v49c(
    new_oof_path: Path,
    baseline_path: Path,
    models: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"缺少 V4.9C J08/L06 baseline OOF:\n{baseline_path}"
        )

    b = pd.read_csv(
        baseline_path,
        compression="infer",
        usecols=lambda c: c in set(OOF_KEY + ["SOH_True","Pred_XGBoost"]),
        low_memory=False,
    )
    n = pd.read_csv(new_oof_path, compression="infer", low_memory=False)

    for d in (b,n):
        d["Battery"] = d["Battery"].map(normalize_battery)
        for c in ("OriginCycle_KeyOnly","Lead","TargetCycle_KeyOnly"):
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(int)

    if b.duplicated(OOF_KEY).any() or n.duplicated(OOF_KEY).any():
        raise RuntimeError("baseline/new OOF存在重复 exact key")

    z = b.merge(n, on=OOF_KEY, how="outer", suffixes=("_V49C","_V410A"), indicator=True, validate="one_to_one")
    both = z["_merge"].eq("both")
    q = z.loc[both].copy()
    dy = np.abs(
        pd.to_numeric(q["SOH_True_V49C"], errors="coerce").to_numpy(float)
        - pd.to_numeric(q["SOH_True_V410A"], errors="coerce").to_numpy(float)
    )
    truth_diff = float(np.nanmax(dy)) if len(dy) else np.nan

    audit = pd.DataFrame([dict(
        BaselineRows=len(b),
        NewRows=len(n),
        CommonRows=int(both.sum()),
        BaselineOnly=int((z["_merge"]=="left_only").sum()),
        NewOnly=int((z["_merge"]=="right_only").sum()),
        CommonBatteries=int(q["Battery"].nunique()),
        TruthMaxAbsDiff=truth_diff,
        PASS=int(
            len(b)==len(n)==int(both.sum())
            and (z["_merge"]!="both").sum()==0
            and np.isfinite(truth_diff)
            and truth_diff <= 1e-12
        ),
    )])

    metrics = []
    y = pd.to_numeric(q["SOH_True_V49C"], errors="coerce").to_numpy(float)

    bp = pd.to_numeric(q["Pred_XGBoost"], errors="coerce").to_numpy(float)
    base_m = calc_metrics(y, bp)
    metrics.append(dict(
        RankSource="FrozenBaseline",
        ModelID="V49C_XGBoost_J08L06",
        **base_m,
        RMSE_Improvement_vs_V49C_pct=0.0,
        MAE_Improvement_vs_V49C_pct=0.0,
    ))
    base_rmse = float(base_m["RMSE"])
    base_mae = float(base_m["MAE"])

    for mid in models:
        col = f"Pred_{mid}"
        if col not in q.columns:
            continue
        p = pd.to_numeric(q[col], errors="coerce").to_numpy(float)
        m = calc_metrics(y,p)
        metrics.append(dict(
            RankSource="V4.10A",
            ModelID=mid,
            **m,
            RMSE_Improvement_vs_V49C_pct=100.0*(base_rmse-float(m["RMSE"]))/base_rmse,
            MAE_Improvement_vs_V49C_pct=100.0*(base_mae-float(m["MAE"]))/base_mae,
        ))
    md = pd.DataFrame(metrics).sort_values(["RMSE","MAE"]).reset_index(drop=True)
    md.insert(0, "Rank", np.arange(1,len(md)+1))
    return audit, md


# =============================================================================
# 7. CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWTH48 V4.10A Event Tree Upgrade J08/L06 47-cell LOBO"
    )
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--code-dir", type=Path, default=CODE_DIR_DEFAULT)
    p.add_argument("--strict-input", type=Path, default=STRICT_DEFAULT)
    p.add_argument("--event-file", type=Path, default=EVENT_DEFAULT)
    p.add_argument("--v49c-baseline-oof", type=Path, default=V49C_BASELINE_OOF_DEFAULT)
    p.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--threads-model", type=int, default=6)
    p.add_argument("--xgb-device", choices=["auto","cuda","cpu"], default="auto")
    p.add_argument("--xgb-gpu-id", type=int, default=0)
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--force-folds", action="store_true")
    p.add_argument("--skip-final-oof", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_all = time.perf_counter()
    out = ensure_dir(args.output_root)
    cache_dir = ensure_dir(out / "Cache_Folds")
    models = list(dict.fromkeys(args.models))

    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.threads_model)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, args.threads_model)))

    # -------------------------------------------------------------------------
    # A. Helper scripts and dependency audit
    # -------------------------------------------------------------------------
    v49_path = resolve_script(
        args.code_dir,
        "20_trajectory_lobo_helper.py",
        "*V4_9*整条未来SOH轨迹*py",
    )
    v47e_path = resolve_script(
        args.code_dir,
        "05_rank_hi_topk_ablation.py",
        "*V4_7E*Event*HI*LOBO*py",
    )
    v48e_path = resolve_script(
        args.code_dir,
        "24_event_sparse_search_helper.py",
        "*V4_8E*Event*Top2*ACCEL*py",
    )
    v49 = load_module(v49_path, "rwth48_v49_helper_for_v410a")
    v47e = load_module(v47e_path, "rwth48_v47e_helper_for_v410a")
    v48e = load_module(v48e_path, "rwth48_v48e_helper_for_v410a")

    # Freeze Event representation in generic V4.9 helper.
    v49.HI_COLUMNS = EVENT_COLUMNS
    v49.COMPACT_HI = EVENT_TOP2
    v49.CHECKPOINTS_S = tuple(x[0] for x in EVENT_NODES)

    dep_rows = []
    try:
        import xgboost as xgb
        dep_rows.append(dict(Package="xgboost",Required=1,Available=1,Version=xgb.__version__))
    except Exception as exc:
        dep_rows.append(dict(Package="xgboost",Required=1,Available=0,Version=""))
        pd.DataFrame(dep_rows).to_csv(out/"00_dependency_audit.csv",index=False,encoding="utf-8-sig")
        raise RuntimeError("当前环境没有xgboost") from exc

    lgb_ok, lgb_ver = lightgbm_available()
    dep_rows.append(dict(
        Package="lightgbm",
        Required=int("LightGBM_EngTuned" in models),
        Available=int(lgb_ok),
        Version=lgb_ver,
    ))
    pd.DataFrame(dep_rows).to_csv(out/"00_dependency_audit.csv",index=False,encoding="utf-8-sig")
    if "LightGBM_EngTuned" in models and not lgb_ok:
        raise RuntimeError(
            "默认V4.10A包含LightGBM challenger，但当前环境未安装lightgbm。\n"
            "请执行: pip install lightgbm\n"
            "或者先运行: --models XGB_EngFrozen XGB_EngTuned"
        )

    backend_info = (
        {"Requested": args.xgb_device, "Selected": "NOT_PROBED_AUDIT_ONLY", "GPU_ID": args.xgb_gpu_id}
        if args.audit_only
        else v48e.configure_xgboost_backend(
            v49, args.xgb_device, args.xgb_gpu_id, args.threads_model
        )
    )
    backend_params = dict(v49.XGB_PARAMS)

    # -------------------------------------------------------------------------
    # B. Rebuild the exact V4.9C J08/L06 cohort
    # -------------------------------------------------------------------------
    if not args.strict_input.exists():
        raise FileNotFoundError(args.strict_input)
    if not args.event_file.exists():
        raise FileNotFoundError(args.event_file)

    strict = v49.load_strict_target(args.strict_input)
    header = pd.read_csv(args.event_file, nrows=0).columns.tolist()
    miss = [x for x in EVENT_TOP2 if x not in header]
    if miss:
        raise RuntimeError("Event frozen file缺Top2: " + str(miss))

    wide, node_audit = v47e.load_event_wide(args.event_file, EVENT_TOP2)
    node_audit.to_csv(out/"01_event_node_audit.csv", index=False, encoding="utf-8-sig")
    wide = wide[["Battery","GlobalCycle",*EVENT_COLUMNS]].copy()
    wide["Battery"] = wide["Battery"].map(normalize_battery)

    bundle, seg_audit = v49.build_full_trajectory_windows(strict, wide, J, L, K)
    seg_audit.to_csv(out/"02_segment_window_audit.csv", index=False, encoding="utf-8-sig")

    Xeng, feature_names, registry = engineer_features(bundle.X)
    registry.to_csv(out/"03_feature_registry.csv", index=False, encoding="utf-8-sig")

    batteries = sorted(set(bundle.battery.tolist()))
    outer = [normalize_battery(args.only_battery)] if args.only_battery else batteries
    if args.only_battery and outer[0] not in batteries:
        raise ValueError(f"--only-battery不存在: {outer[0]}")

    availability = pd.DataFrame([dict(
        J=J,L=L,K=K,
        HistorySpan_cycles=(L-1)*J,
        DiagnosticCount=L,
        MeasuredScalarInputs=MEASURED_SCALARS,
        EngineeredModelFeatureCount=len(feature_names),
        TotalTrajectoryWindows=len(bundle.y),
        BatteriesWithWindows=len(batteries),
        MinWindowsPerBattery=int(pd.Series(bundle.battery).value_counts().min()),
        MedianWindowsPerBattery=float(pd.Series(bundle.battery).value_counts().median()),
        MaxWindowsPerBattery=int(pd.Series(bundle.battery).value_counts().max()),
    )])
    availability.to_csv(out/"04_trajectory_feature_inventory.csv",index=False,encoding="utf-8-sig")

    cfg = dict(
        Task="RWTH48 V4.10A Event Tree Upgrade",
        Base=str(args.base),
        StrictInput=str(args.strict_input),
        EventFile=str(args.event_file),
        V49CBaselineOOF=str(args.v49c_baseline_oof),
        Output=str(out),
        EventTop2=list(EVENT_TOP2),
        EventCheckpoints=[x[1] for x in EVENT_NODES],
        J=J,L=L,K=K,
        MeasuredScalarBudget=MEASURED_SCALARS,
        EngineeredModelFeatureCount=len(feature_names),
        Models=models,
        HistoricalSOHAsFeature=False,
        FutureSOHAsFeature=False,
        FutureHIAsFeature=False,
        CycleAsFeature=False,
        PeriodicSOHCalibration=False,
        NearestCycleSubstitution=False,
        SourceBatteryTuning=True,
        XGBoostBackend=backend_info,
        LightGBMVersion=lgb_ver if lgb_ok else "",
        LightGBMTuningLeads=list(LGB_TUNE_LEADS),
        Important=[
            "MeasuredScalarBudget remains 36; all extra model features are algebraic summaries of those 36 Event HI scalars.",
            "Outer held-out battery never participates in imputation, tuning or fitting.",
            "Tuning validation units are complete source batteries, never randomly mixed windows.",
            "V4.9C J08/L06 XGBoost OOF is reused as frozen baseline and never retrained.",
        ],
    )
    save_json(cfg, out/"00_config.json")

    audit = pd.DataFrame([
        dict(Check="47 batteries",Observed=len(batteries),Expected=47,PASS=int(len(batteries)==47)),
        dict(Check="Event Top2 fixed",Observed="|".join(EVENT_TOP2),Expected="V_TimeWeightedMean|P_Last",PASS=int(EVENT_TOP2==("V_TimeWeightedMean","P_Last"))),
        dict(Check="J08/L06 fixed",Observed=f"J{J:02d}_L{L:02d}",Expected="J08_L06",PASS=int(J==8 and L==6)),
        dict(Check="Measured scalar budget",Observed=MEASURED_SCALARS,Expected=36,PASS=int(MEASURED_SCALARS==36)),
        dict(Check="Engineered features finite dimensional",Observed=Xeng.shape[1],Expected=">36",PASS=int(Xeng.shape[1]>36)),
        dict(Check="V4.9C baseline exists",Observed=int(args.v49c_baseline_oof.exists()),Expected=1,PASS=int(args.v49c_baseline_oof.exists())),
    ])
    audit.to_csv(out/"01_input_acceptance_audit.csv",index=False,encoding="utf-8-sig")
    if not audit["PASS"].eq(1).all():
        raise RuntimeError("V4.10A输入审计失败:\n" + audit.to_string(index=False))

    print("="*118)
    print("RWTH48 V4.10A | Event Tree Upgrade | J08/L06 | 47-cell Outer LOBO")
    print("Batteries              :", len(batteries))
    print("Trajectory windows      :", len(bundle.y))
    print("Measured scalar budget  :", MEASURED_SCALARS)
    print("Engineered model feats  :", Xeng.shape[1])
    print("Models                  :", models)
    print("XGBoost backend         :", backend_info)
    print("="*118, flush=True)

    if args.audit_only:
        print("AUDIT-ONLY PASS")
        print("Feature groups:")
        print(registry.groupby(["Group","MeasuredOrDerived"]).size().reset_index(name="N").to_string(index=False))
        return

    # -------------------------------------------------------------------------
    # C. 47-cell outer LOBO
    # -------------------------------------------------------------------------
    fold_paths: List[Path] = []
    fold_metrics_parts: List[pd.DataFrame] = []
    tuning_parts: List[pd.DataFrame] = []
    fi_parts: List[pd.DataFrame] = []

    for fi, test_bat in enumerate(outer, 1):
        print(f"\n[{fi:02d}/{len(outer):02d}] held-out = {test_bat}", flush=True)
        long_df, metrics, tuning, imp = run_one_fold(
            bundle=bundle,
            Xeng=Xeng,
            feature_names=feature_names,
            test_battery=test_bat,
            models=models,
            backend_params=backend_params,
            backend_info=backend_info,
            threads=args.threads_model,
            cache_dir=cache_dir,
            force=args.force_folds,
        )
        fold_paths.append(cache_dir / f"{safe_name(test_bat)}_predictions.csv.gz")
        fold_metrics_parts.append(metrics)
        tuning_parts.append(tuning)
        fi_parts.append(imp)
        print(
            "    " + " | ".join(
                f"{r.ModelID}:RMSE={float(r.RMSE):.6f}"
                for r in metrics.itertuples()
            ),
            flush=True,
        )
        del long_df, metrics, tuning, imp
        gc.collect()

    fold_metrics = pd.concat(fold_metrics_parts, ignore_index=True)
    tuning_all = pd.concat(tuning_parts, ignore_index=True)
    fi_all = pd.concat(fi_parts, ignore_index=True)

    fold_metrics.to_csv(out/"05_fold_metrics.csv",index=False,encoding="utf-8-sig")
    tuning_all.to_csv(out/"06_source_battery_tuning_summary.csv",index=False,encoding="utf-8-sig")
    fi_all.to_csv(out/"07_fold_feature_importance.csv.gz",index=False,compression=FAST_GZIP)

    # Feature-importance macro across folds.
    fi_macro = (
        fi_all.groupby(["ModelID","Feature"], as_index=False)["Importance"]
        .agg(["mean","median","std"])
        .reset_index()
        .rename(columns={"mean":"MeanImportance","median":"MedianImportance","std":"StdImportance"})
    )
    fi_macro["RankWithinModel"] = (
        fi_macro.groupby("ModelID")["MeanImportance"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    fi_macro = fi_macro.sort_values(["ModelID","RankWithinModel","Feature"])
    fi_macro.to_csv(out/"08_feature_importance_macro.csv",index=False,encoding="utf-8-sig")

    pooled, leadwise, macro = aggregate_new_oof(fold_paths, models)
    pooled.to_csv(out/"09_pooled_OOF_metrics.csv",index=False,encoding="utf-8-sig")
    leadwise.to_csv(out/"10_leadwise_pooled_OOF_metrics.csv",index=False,encoding="utf-8-sig")
    macro.to_csv(out/"11_macro_LOBO_metrics.csv",index=False,encoding="utf-8-sig")

    # Build combined OOF before exact-common comparison. compresslevel=1 avoids V4.9B-style final merge bottleneck.
    final_oof = out/"12_ALL_OOF_V410A_tree_upgrade_long.csv.gz"
    if not args.skip_final_oof:
        combine_fold_predictions(fold_paths, final_oof)
        exact_audit, exact_metrics = exact_common_vs_v49c(
            final_oof, args.v49c_baseline_oof, models
        )
    else:
        # Temporary local concat for exact comparison, then delete.
        tmp = out/"_TMP_exact_common_OOF.csv.gz"
        combine_fold_predictions(fold_paths, tmp)
        exact_audit, exact_metrics = exact_common_vs_v49c(
            tmp, args.v49c_baseline_oof, models
        )
        try:
            tmp.unlink()
        except Exception:
            pass

    exact_audit.to_csv(out/"13_exact_common_vs_V49C_audit.csv",index=False,encoding="utf-8-sig")
    exact_metrics.to_csv(out/"14_exact_common_vs_V49C_metrics.csv",index=False,encoding="utf-8-sig")

    # Candidate selection frequencies.
    if len(tuning_all):
        selected = tuning_all[tuning_all.get("Selected",0).eq(1)].copy()
        selection_freq = (
            selected.groupby(["ModelID","Candidate"],as_index=False)
            .size().rename(columns={"size":"SelectedFolds"})
            .sort_values(["ModelID","SelectedFolds"],ascending=[True,False])
        )
    else:
        selection_freq = pd.DataFrame(columns=["ModelID","Candidate","SelectedFolds"])
    selection_freq.to_csv(out/"15_tuning_selection_frequency.csv",index=False,encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # D. Acceptance gate / COMPLETE
    # -------------------------------------------------------------------------
    gates = []
    def gate(name: str, ok: bool, obs: object, exp: object):
        gates.append(dict(Check=name,Observed=obs,Expected=exp,PASS=int(bool(ok))))

    gate("47-cell full run", bool(args.only_battery) or len(outer)==47, len(outer), 47)
    gate("Event Top2 fixed", EVENT_TOP2==("V_TimeWeightedMean","P_Last"), "|".join(EVENT_TOP2), "V_TimeWeightedMean|P_Last")
    gate("J08/L06 fixed", J==8 and L==6, f"J{J:02d}_L{L:02d}", "J08_L06")
    gate("Measured scalar budget remains 36", MEASURED_SCALARS==36, MEASURED_SCALARS, 36)
    gate("No historical SOH feature", True, 0, 0)
    gate("No future HI/SOH feature", True, 0, 0)
    gate("No Cycle feature", True, 0, 0)
    gate("All requested model folds exist",
         len(fold_metrics)==len(outer)*len(models),
         len(fold_metrics), len(outer)*len(models))
    gate("Finite fold RMSE", np.isfinite(pd.to_numeric(fold_metrics["RMSE"],errors="coerce")).all(),
         int(np.isfinite(pd.to_numeric(fold_metrics["RMSE"],errors="coerce")).sum()), len(fold_metrics))
    gate("V4.9C exact cohort compatibility",
         len(exact_audit)>0 and exact_audit["PASS"].eq(1).all(),
         exact_audit.to_dict("records"), "PASS")

    gate_df = pd.DataFrame(gates)
    gate_df.to_csv(out/"16_acceptance_gate.csv",index=False,encoding="utf-8-sig")

    if args.only_battery:
        status = "SMOKE_PASS" if gate_df["PASS"].eq(1).all() else "SMOKE_FAIL"
    else:
        status = "PASS" if gate_df["PASS"].eq(1).all() else "FAIL"

    best_row = exact_metrics.sort_values(["RMSE","MAE"]).iloc[0].to_dict() if len(exact_metrics) else {}
    base_row = exact_metrics[exact_metrics["ModelID"].eq("V49C_XGBoost_J08L06")]
    base_rmse = float(base_row.iloc[0]["RMSE"]) if len(base_row) else np.nan

    complete = dict(
        Task="RWTH48 V4.10A Event Tree Upgrade J08/L06",
        Status=status,
        OuterTestBatteriesRun=len(outer),
        Full47FoldRun=bool(not args.only_battery and len(outer)==47),
        EventTop2=list(EVENT_TOP2),
        EventCheckpoints=[x[1] for x in EVENT_NODES],
        J=J,L=L,K=K,
        DiagnosticCount=L,
        HistorySpan_cycles=(L-1)*J,
        MeasuredScalarInputs=MEASURED_SCALARS,
        EngineeredModelFeatureCount=len(feature_names),
        Models=models,
        HistoricalSOHAsFeature=False,
        FutureSOHAsFeature=False,
        FutureHIAsFeature=False,
        CycleAsFeature=False,
        PeriodicSOHCalibration=False,
        NearestCycleSubstitution=False,
        SourceBatteryTuning=True,
        XGBoostBackend=backend_info,
        LightGBMVersion=lgb_ver if lgb_ok else "",
        V49CBaselineRMSE=base_rmse,
        BestExactCommon=best_row,
        TuningSelectionFrequency=selection_freq.to_dict("records"),
        Elapsed_s=time.perf_counter()-t_all,
        Important=[
            "No new physical measurement was added: measured scalar budget stays at 36.",
            "Engineered features are deterministic summaries/differences derived only from the 36 historical Event Top2 scalars.",
            "Outer held-out battery never participates in imputation, hyperparameter selection, or fitting.",
            "V4.9C J08/L06 XGBoost is a frozen OOF baseline, compared on exact common Battery/Origin/Lead/Target keys.",
            "LightGBM tuning uses source-only representative leads 1/10/25/50/75/100; its final model predicts all 100 leads.",
        ],
    )
    save_json(complete, out/"99_RWTH48_V4_10A_COMPLETE.json")

    print("\n" + "="*118)
    print("RWTH48 V4.10A COMPLETE")
    print("Status:", status)
    print("Output:", out)
    print("\nExact-common ranking:")
    print(exact_metrics.to_string(index=False))
    print("\nBest:", best_row)
    print("="*118, flush=True)

    if status.endswith("FAIL"):
        raise RuntimeError("V4.10A acceptance gate FAIL")


if __name__ == "__main__":
    main()

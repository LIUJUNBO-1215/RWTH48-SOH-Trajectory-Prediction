# -*- coding: utf-8 -*-
r"""
RWTH48 V4.9 LITE
HI-Only Sparse-History -> Full Future SOH Trajectory Prognosis
47-cell Outer Leave-One-Battery-Out
RF / XGBoost / LSTM / BiLSTM multi-output
================================================================================

研究任务
--------
将 V4.6A 的单点预测
    X(t-45,...,t) -> SOH(t+H), H in {10,50,100}
改为一次性直接预测完整未来轨迹
    X(t-45,...,t) -> [SOH(t+1), SOH(t+2), ..., SOH(t+K)]
默认 K=100。

输入信息预算保持不变：
    - 仅使用 V4.6A 冻结的 5 个 causal HI；
    - 每个历史 diagnostic cycle 使用 30/60/90/120 s 四个累计 prefix；
    - 每个 observation = 4 checkpoints x 5 HI = 20 维；
    - J=5, L=10，输入 X.shape=[N,10,20]；
    - GlobalCycle 只用于调度，不进入模型；
    - 不输入历史 SOH / RPT SOH / capacity / future HI / full-cycle information。

严格未来轨迹定义：
    对 origin=t，target 必须为同一个 StrictSegmentKey 内 exact 连续：
        t+1, t+2, ..., t+K
    不插值、不 nearest-cycle substitution。

关键修正：
    future target cycle 只需要 strict SOH 标签；
    不要求 future cycle 同时存在 HI。
    HI 只在历史窗口 t-(L-1)J,...,t 上要求存在。

模型：
    RF       : sklearn RandomForestRegressor 原生 multi-output
    XGBoost  : 优先使用 native multi_output_tree；失败则回退 one_output_per_tree，
               再失败则回退 sklearn MultiOutputRegressor
    LSTM     : history encoder + Dense(K) direct multi-output
    BiLSTM   : bidirectional history encoder + Dense(K) direct multi-output

输出评价：
    1) 每折所有未来点 pooled metrics
    2) lead=1..K 的 pooled OOF metrics
    3) lead=1..K 的 LOBO macro metrics
    4) 每条 trajectory 的 RMSE/MAE/endpoint error
    5) trajectory shape diagnostics：upward-step fraction / roughness
    6) 与 V4.6A H=10/50/100 单点模型在 common cohort 上公平比较
    7) 全部 OOF trajectory long-format prediction

正式运行：
    python "src/20_trajectory_lobo_helper.py"

单折 smoke：
    python "src/20_trajectory_lobo_helper.py" --only-battery RWTH-002 --future-length 100 --models RF LSTM

只跑深度模型：
    python "src/20_trajectory_lobo_helper.py" --models LSTM BiLSTM

续跑：
    直接重复同一命令即可；不要加 --force-folds。
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 0. Frozen project configuration
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")

DEFAULT_STRICT_INPUT = (
    BASE
    / "RWTH48_V4_5C_FIX_最小冲击共同扰动处理"
    / "06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz"
)

DEFAULT_V46A_ROOT = (
    BASE
    / "RWTH48_V4_6A_HIOnly_120sMultiCheckpoint_LOBO_LITE"
)

DEFAULT_HI_INPUT = (
    DEFAULT_V46A_ROOT
    / "03_prefix_HI_all_cycles.csv.gz"
)

DEFAULT_V46A_COMPLETE = (
    DEFAULT_V46A_ROOT
    / "99_RWTH48_V4_6A_COMPLETE.json"
)

DEFAULT_SINGLEPOINT_OOF = (
    DEFAULT_V46A_ROOT
    / "08_ALL_OOF_predictions.csv.gz"
)

DEFAULT_OUTPUT = (
    BASE
    / "RWTH48_V4_9_FullTrajectory_HIOnly_LOBO_LITE"
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
HI_COLUMNS = tuple(
    f"T{sec:03d}__{f}"
    for sec in CHECKPOINTS_S
    for f in COMPACT_HI
)

HISTORY_INTERVAL_J = 5
HISTORY_OBSERVATIONS_L = 10
FUTURE_LENGTH_K = 100

RF_PER_SOURCE_BATTERY = 60
XGB_PER_SOURCE_BATTERY = 80
DEEP_PER_SOURCE_BATTERY = 80

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
DROPOUT = 0.10
RANDOM_SEED = 20260820
FAST_GZIP = {"method": "gzip", "compresslevel": 1}

MODELS_ALL = ("RF", "XGBoost", "LSTM", "BiLSTM")


# =============================================================================
# 1. CLI / utilities
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "RWTH48 V4.9 HI-only full future SOH trajectory prognosis "
            "with 47-cell outer LOBO."
        )
    )
    p.add_argument("--strict-input", type=Path, default=DEFAULT_STRICT_INPUT)
    p.add_argument("--hi-input", type=Path, default=DEFAULT_HI_INPUT)
    p.add_argument("--v46a-complete", type=Path, default=DEFAULT_V46A_COMPLETE)
    p.add_argument("--singlepoint-oof", type=Path, default=DEFAULT_SINGLEPOINT_OOF)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--future-length", type=int, default=FUTURE_LENGTH_K)
    p.add_argument("--history-interval", type=int, default=HISTORY_INTERVAL_J)
    p.add_argument("--history-observations", type=int, default=HISTORY_OBSERVATIONS_L)
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS_ALL),
        choices=list(MODELS_ALL),
    )
    p.add_argument("--threads-model", type=int, default=4)
    p.add_argument("--deep-epochs", type=int, default=DEEP_EPOCHS)
    p.add_argument("--deep-batch", type=int, default=DEEP_BATCH)
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--force-folds", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-singlepoint-compare", action="store_true")
    p.add_argument("--skip-final-oof", action="store_true")
    p.add_argument(
        "--allow-v46a-config-mismatch",
        action="store_true",
        help="仅用于明确知道自己在做新消融时；默认严格检查 V4.6A J/L/HI 配置。",
    )
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


def safe_name(x: object) -> str:
    return (
        str(x)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if len(yt) == 0:
        return dict(N=0, MAE=np.nan, RMSE=np.nan, R2=np.nan, MAPE_pct=np.nan, Bias=np.nan)
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


def safe_mean(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.mean()) if x.notna().any() else np.nan


def safe_median(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.median()) if x.notna().any() else np.nan


def safe_max(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.max()) if x.notna().any() else np.nan


# =============================================================================
# 2. Frozen input loaders
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
        dup = d[d.duplicated(["Battery", "GlobalCycle"], keep=False)].head(20)
        raise RuntimeError(
            "strict target存在重复 Battery+GlobalCycle：\n" + dup.to_string(index=False)
        )

    d = d[~d["Battery"].isin(EXCLUDED_BATTERIES)].copy()
    return d.sort_values(["Battery", "GlobalCycle"]).reset_index(drop=True)


def load_prefix_hi(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"V4.6A prefix HI不存在：{path}")
    d = pd.read_csv(path, compression="infer", low_memory=False)
    required = {"Battery", "GlobalCycle", *HI_COLUMNS}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"prefix HI缺冻结列：{missing}")

    d["Battery"] = d["Battery"].map(normalize_battery)
    d["GlobalCycle"] = pd.to_numeric(d["GlobalCycle"], errors="coerce")
    d = d[d["GlobalCycle"].notna()].copy()
    d["GlobalCycle"] = d["GlobalCycle"].astype(int)
    d = d[~d["Battery"].isin(EXCLUDED_BATTERIES)].copy()
    for c in HI_COLUMNS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if d.duplicated(["Battery", "GlobalCycle"]).any():
        raise RuntimeError("prefix HI存在重复 Battery+GlobalCycle")
    return d[["Battery", "GlobalCycle", *HI_COLUMNS]].sort_values(
        ["Battery", "GlobalCycle"]
    ).reset_index(drop=True)


def validate_v46a_complete(
    path: Path,
    J: int,
    L: int,
    allow_mismatch: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        rows.append({
            "Check": "V46ACompleteExists",
            "Expected": True,
            "Observed": False,
            "PASS": 0,
        })
        if not allow_mismatch:
            raise FileNotFoundError(f"V4.6A COMPLETE不存在：{path}")
        return pd.DataFrame(rows)

    obj = json.loads(path.read_text(encoding="utf-8"))
    checks = [
        ("Status", "PASS", obj.get("Status")),
        ("HistoryInterval_J", int(J), int(obj.get("HistoryInterval_J", -999))),
        ("HistoryObservations_L", int(L), int(obj.get("HistoryObservations_L", -999))),
        ("Checkpoints_s", list(CHECKPOINTS_S), obj.get("Checkpoints_s")),
        ("CompactHI", list(COMPACT_HI), obj.get("CompactHI")),
        ("OnlyHIAsModelInput", True, bool(obj.get("OnlyHIAsModelInput", False))),
        ("HistoricalSOHAsFeature", False, bool(obj.get("HistoricalSOHAsFeature", True))),
    ]
    for name, expected, observed in checks:
        ok = expected == observed
        rows.append({
            "Check": name,
            "Expected": json.dumps(expected, ensure_ascii=False),
            "Observed": json.dumps(observed, ensure_ascii=False),
            "PASS": int(ok),
        })

    audit = pd.DataFrame(rows)
    if (audit["PASS"] == 0).any() and not allow_mismatch:
        raise RuntimeError(
            "V4.6A冻结配置与V4.9不一致；拒绝继续。\n" + audit.to_string(index=False)
        )
    return audit


def build_input_audit(strict: pd.DataFrame, hi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in sorted(strict["Battery"].unique()):
        s = strict[strict["Battery"].eq(b)]
        h = hi[hi["Battery"].eq(b)]
        strict_cycles = set(s["GlobalCycle"].astype(int))
        hi_cycles = set(h["GlobalCycle"].astype(int))
        rows.append({
            "Battery": b,
            "StrictCycles": len(strict_cycles),
            "HICycles": len(hi_cycles),
            "StrictWithHI": len(strict_cycles & hi_cycles),
            "StrictHICoverage_pct": (
                100.0 * len(strict_cycles & hi_cycles) / len(strict_cycles)
                if strict_cycles else np.nan
            ),
            "StrictSegments": int(s[SEGMENT].nunique()),
            "HIAllFiniteRows": int(
                np.isfinite(h[list(HI_COLUMNS)].to_numpy(float)).all(axis=1).sum()
            ) if len(h) else 0,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 3. Strict full-trajectory window construction
# =============================================================================

@dataclass
class TrajectoryBundle:
    X: np.ndarray                 # [N,L,F]
    y: np.ndarray                 # [N,K]
    battery: np.ndarray
    origin_cycle: np.ndarray
    segment: np.ndarray

    @property
    def K(self) -> int:
        return int(self.y.shape[1]) if self.y.ndim == 2 else 0

    @property
    def L(self) -> int:
        return int(self.X.shape[1]) if self.X.ndim == 3 else 0

    @property
    def F(self) -> int:
        return int(self.X.shape[2]) if self.X.ndim == 3 else 0


def build_full_trajectory_windows(
    strict: pd.DataFrame,
    hi: pd.DataFrame,
    J: int,
    L: int,
    K: int,
) -> Tuple[TrajectoryBundle, pd.DataFrame]:
    """
    未来 target 仅从 strict SOH 读取，不要求未来 cycle 有 HI。
    HI 只在 exact history cycles 上要求存在。
    """
    history_offsets = np.array(
        [-J * k for k in range(L - 1, -1, -1)], dtype=int
    )

    hi_by_battery: Dict[str, Dict[int, np.ndarray]] = {}
    for b, g in hi.groupby("Battery", sort=False):
        hi_by_battery[str(b)] = {
            int(r.GlobalCycle): np.asarray(
                [getattr(r, c) for c in HI_COLUMNS], dtype=np.float32
            )
            for r in g.itertuples(index=False)
        }

    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    Bs: List[str] = []
    Origins: List[int] = []
    Segs: List[str] = []
    inv_rows: List[Dict[str, object]] = []

    for (battery, seg), g in strict.groupby(["Battery", SEGMENT], sort=False):
        battery = str(battery)
        g = g.sort_values("GlobalCycle")
        cycles = g["GlobalCycle"].to_numpy(int)
        yvals = g[TARGET].to_numpy(float)
        ymap = {int(c): float(y) for c, y in zip(cycles, yvals)}
        hmap = hi_by_battery.get(battery, {})

        accepted = 0
        reject_hist = 0
        reject_future = 0
        reject_y = 0

        for origin in cycles:
            hcycles = origin + history_offsets
            if not all(int(c) in hmap for c in hcycles):
                reject_hist += 1
                continue

            fcycles = origin + np.arange(1, K + 1, dtype=int)
            if not all(int(c) in ymap for c in fcycles):
                reject_future += 1
                continue

            x = np.stack([hmap[int(c)] for c in hcycles]).astype(np.float32, copy=False)
            y = np.asarray([ymap[int(c)] for c in fcycles], dtype=float)
            if not np.isfinite(y).all():
                reject_y += 1
                continue

            Xs.append(x)
            Ys.append(y)
            Bs.append(battery)
            Origins.append(int(origin))
            Segs.append(str(seg))
            accepted += 1

        inv_rows.append({
            "Battery": battery,
            "StrictSegmentKey": str(seg),
            "SegmentCycles": int(len(cycles)),
            "AcceptedTrajectoryWindows": int(accepted),
            "RejectedMissingHistoryHI": int(reject_hist),
            "RejectedIncompleteFutureTrajectory": int(reject_future),
            "RejectedNonFiniteFutureSOH": int(reject_y),
            "HistorySpan_cycles": int((L - 1) * J),
            "FutureLength_K": int(K),
        })

    if not Xs:
        bundle = TrajectoryBundle(
            X=np.empty((0, L, len(HI_COLUMNS)), dtype=np.float32),
            y=np.empty((0, K), dtype=float),
            battery=np.empty(0, dtype=object),
            origin_cycle=np.empty(0, dtype=int),
            segment=np.empty(0, dtype=object),
        )
    else:
        bundle = TrajectoryBundle(
            X=np.stack(Xs).astype(np.float32, copy=False),
            y=np.stack(Ys).astype(float, copy=False),
            battery=np.asarray(Bs, dtype=object),
            origin_cycle=np.asarray(Origins, dtype=int),
            segment=np.asarray(Segs, dtype=object),
        )

    return bundle, pd.DataFrame(inv_rows)


# =============================================================================
# 4. Source-only scaling / balancing
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


# =============================================================================
# 5. Multi-output models
# =============================================================================

def build_deep_model(kind: str, L: int, F: int, K: int):
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

    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    out = tf.keras.layers.Dense(K, activation="linear", name="SOH_future_trajectory")(x)
    model = tf.keras.Model(inp, out, name=f"{kind}_DirectTrajectory")
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

    # Per-lead source-only target standardization.
    ym = np.mean(ytr, axis=0).astype(float)
    ys = np.std(ytr, axis=0).astype(float)
    ys[~np.isfinite(ys) | (ys < 1e-8)] = 1.0
    yz = ((ytr - ym[None, :]) / ys[None, :]).astype(np.float32)

    model = build_deep_model(kind, Xtr.shape[1], Xtr.shape[2], ytr.shape[1])
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
    )
    pred = predz.astype(float) * ys[None, :] + ym[None, :]
    elapsed = time.perf_counter() - t0

    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return np.asarray(pred, dtype=float), float(elapsed)


def train_predict_rf(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    threads: int,
) -> Tuple[np.ndarray, float, str]:
    params = dict(RF_PARAMS)
    params["n_jobs"] = max(1, int(threads))
    model = RandomForestRegressor(**params)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    elapsed = time.perf_counter() - t0
    del model
    gc.collect()
    return np.asarray(pred, dtype=float), float(elapsed), "native_multioutput"


def train_predict_xgb(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    threads: int,
) -> Tuple[np.ndarray, float, str]:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError("未安装xgboost。") from exc

    base = dict(XGB_PARAMS)
    base["n_jobs"] = max(1, int(threads))

    errors: List[str] = []
    for strategy in ("multi_output_tree", "one_output_per_tree"):
        try:
            model = XGBRegressor(**base, multi_strategy=strategy)
            t0 = time.perf_counter()
            model.fit(Xtr, ytr, verbose=False)
            pred = model.predict(Xte)
            elapsed = time.perf_counter() - t0
            pred = np.asarray(pred, dtype=float)
            if pred.ndim == 1 and ytr.shape[1] == 1:
                pred = pred[:, None]
            if pred.shape != (len(Xte), ytr.shape[1]):
                raise RuntimeError(
                    f"XGBoost {strategy}返回形状{pred.shape}，预期{(len(Xte), ytr.shape[1])}"
                )
            del model
            gc.collect()
            return pred, float(elapsed), strategy
        except Exception as exc:
            errors.append(f"{strategy}: {type(exc).__name__}: {exc}")
            try:
                del model
            except Exception:
                pass
            gc.collect()

    # Compatibility fallback for older xgboost.
    try:
        from sklearn.multioutput import MultiOutputRegressor
        inner = dict(base)
        # Keep per-target training lightweight in this emergency fallback.
        inner["n_jobs"] = 1
        model = MultiOutputRegressor(
            XGBRegressor(**inner),
            n_jobs=max(1, min(int(threads), 4)),
        )
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        pred = np.asarray(model.predict(Xte), dtype=float)
        elapsed = time.perf_counter() - t0
        del model
        gc.collect()
        return pred, float(elapsed), "sklearn_MultiOutputRegressor_fallback"
    except Exception as exc:
        errors.append(f"fallback: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "XGBoost multi-output训练失败：\n" + "\n".join(errors)
        ) from exc


# =============================================================================
# 6. Fold metrics / trajectory diagnostics
# =============================================================================

def trajectory_shape_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim != 2 or yp.ndim != 2 or yt.shape != yp.shape or yt.size == 0:
        return {
            "MeanTrajectoryRMSE": np.nan,
            "MedianTrajectoryRMSE": np.nan,
            "MeanTrajectoryMAE": np.nan,
            "EndpointMAE": np.nan,
            "TrueUpwardStepFraction": np.nan,
            "PredUpwardStepFraction": np.nan,
            "ExcessUpwardStepFraction": np.nan,
            "TrueRoughness": np.nan,
            "PredRoughness": np.nan,
        }

    err = yp - yt
    sample_rmse = np.sqrt(np.mean(err ** 2, axis=1))
    sample_mae = np.mean(np.abs(err), axis=1)
    dt = np.diff(yt, axis=1)
    dp = np.diff(yp, axis=1)
    true_up = float(np.mean(dt > 0)) if dt.size else np.nan
    pred_up = float(np.mean(dp > 0)) if dp.size else np.nan
    if yt.shape[1] >= 3:
        true_rough = float(np.mean(np.abs(np.diff(yt, n=2, axis=1))))
        pred_rough = float(np.mean(np.abs(np.diff(yp, n=2, axis=1))))
    else:
        true_rough = np.nan
        pred_rough = np.nan

    return {
        "MeanTrajectoryRMSE": float(np.mean(sample_rmse)),
        "MedianTrajectoryRMSE": float(np.median(sample_rmse)),
        "MeanTrajectoryMAE": float(np.mean(sample_mae)),
        "EndpointMAE": float(np.mean(np.abs(err[:, -1]))),
        "TrueUpwardStepFraction": true_up,
        "PredUpwardStepFraction": pred_up,
        "ExcessUpwardStepFraction": (
            float(pred_up - true_up)
            if np.isfinite(pred_up) and np.isfinite(true_up)
            else np.nan
        ),
        "TrueRoughness": true_rough,
        "PredRoughness": pred_rough,
    }


def leadwise_metrics_for_fold(
    test_battery: str,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    K = y_true.shape[1]
    for model, pred in predictions.items():
        for lead in range(1, K + 1):
            rows.append({
                "TestBattery": test_battery,
                "Model": model,
                "Lead": int(lead),
                **calc_metrics(y_true[:, lead - 1], pred[:, lead - 1]),
            })
    return pd.DataFrame(rows)


def overall_fold_metrics(
    test_battery: str,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    runtimes: Dict[str, float],
    train_counts: Dict[str, int],
    source_bats: Sequence[str],
    backend: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for model, pred in predictions.items():
        rows.append({
            "TestBattery": test_battery,
            "Model": model,
            "TrainSourceBatteries": int(len(source_bats)),
            "TrainWindows": int(train_counts.get(model, 0)),
            "TestTrajectoryWindows": int(len(y_true)),
            "FuturePointsPerTrajectory": int(y_true.shape[1]),
            "Runtime_s": float(runtimes.get(model, np.nan)),
            "Backend": backend.get(model, ""),
            **calc_metrics(y_true, pred),
            **trajectory_shape_summary(y_true, pred),
        })
    return pd.DataFrame(rows)


def predictions_to_long(
    test_battery: str,
    origins: np.ndarray,
    segments: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    n, K = y_true.shape
    lead = np.tile(np.arange(1, K + 1, dtype=int), n)
    origin_rep = np.repeat(origins.astype(int), K)
    out = pd.DataFrame({
        "TestBattery": test_battery,
        "Battery": test_battery,
        "OriginCycle_KeyOnly": origin_rep,
        "Lead": lead,
        "TargetCycle_KeyOnly": origin_rep + lead,
        "StrictSegmentKey_QCOnly": np.repeat(segments.astype(object), K),
        "SOH_True": y_true.reshape(-1),
    })
    for model, pred in predictions.items():
        out[f"Pred_{model}"] = np.asarray(pred, dtype=float).reshape(-1)
    return out


# =============================================================================
# 7. One LOBO fold + cache
# =============================================================================

def cache_signature(J: int, L: int, K: int, models: Sequence[str], args: argparse.Namespace) -> Dict[str, object]:
    return {
        "J": int(J),
        "L": int(L),
        "K": int(K),
        "HI_COLUMNS": list(HI_COLUMNS),
        "Models": list(models),
        "DeepEpochs": int(args.deep_epochs),
        "DeepBatch": int(args.deep_batch),
        "RF_PARAMS": RF_PARAMS,
        "XGB_PARAMS": XGB_PARAMS,
    }


def cache_valid(meta_path: Path, expected: Dict[str, object]) -> bool:
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        return False
    try:
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        return obj == expected
    except Exception:
        return False


def run_one_fold(
    bundle: TrajectoryBundle,
    test_battery: str,
    models: Sequence[str],
    J: int,
    L: int,
    K: int,
    args: argparse.Namespace,
    cache_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stem = safe_name(test_battery)
    pred_cache = cache_dir / f"{stem}_trajectory_predictions.csv.gz"
    overall_cache = cache_dir / f"{stem}_overall_metrics.csv"
    lead_cache = cache_dir / f"{stem}_leadwise_metrics.csv.gz"
    meta_cache = cache_dir / f"{stem}_cache_meta.json"
    sig = cache_signature(J, L, K, models, args)

    if (
        not args.force_folds
        and pred_cache.exists()
        and overall_cache.exists()
        and lead_cache.exists()
        and all(p.stat().st_size > 0 for p in (pred_cache, overall_cache, lead_cache))
        and cache_valid(meta_cache, sig)
    ):
        return (
            pd.read_csv(pred_cache, compression="infer", low_memory=False),
            pd.read_csv(overall_cache, low_memory=False),
            pd.read_csv(lead_cache, compression="infer", low_memory=False),
        )

    all_bats = sorted(set(bundle.battery.tolist()))
    source_bats = [b for b in all_bats if b != test_battery]
    test_idx = np.flatnonzero(bundle.battery == test_battery)
    if len(test_idx) == 0:
        raise RuntimeError(f"{test_battery} 无完整 K={K} trajectory测试窗口")

    deep_idx = balanced_indices(bundle.battery, source_bats, DEEP_PER_SOURCE_BATTERY)
    rf_idx = balanced_indices(bundle.battery, source_bats, RF_PER_SOURCE_BATTERY)
    xgb_idx = balanced_indices(bundle.battery, source_bats, XGB_PER_SOURCE_BATTERY)
    if len(deep_idx) == 0:
        raise RuntimeError("source trajectory训练窗口为空")

    # Source-only input scaler; held-out battery never participates.
    scaler = fit_scaler(bundle.X[deep_idx])
    Xte = scaler.transform(bundle.X[test_idx])
    yte = bundle.y[test_idx]

    predictions: Dict[str, np.ndarray] = {}
    runtimes: Dict[str, float] = {}
    train_counts: Dict[str, int] = {}
    backend: Dict[str, str] = {}

    Xtflat = Xte.reshape(len(Xte), -1)

    if "RF" in models:
        Xtr = scaler.transform(bundle.X[rf_idx]).reshape(len(rf_idx), -1)
        pred, runtime, be = train_predict_rf(
            Xtr, bundle.y[rf_idx], Xtflat, args.threads_model
        )
        predictions["RF"] = pred
        runtimes["RF"] = runtime
        train_counts["RF"] = len(rf_idx)
        backend["RF"] = be
        del Xtr
        gc.collect()

    if "XGBoost" in models:
        Xtr = scaler.transform(bundle.X[xgb_idx]).reshape(len(xgb_idx), -1)
        pred, runtime, be = train_predict_xgb(
            Xtr, bundle.y[xgb_idx], Xtflat, args.threads_model
        )
        predictions["XGBoost"] = pred
        runtimes["XGBoost"] = runtime
        train_counts["XGBoost"] = len(xgb_idx)
        backend["XGBoost"] = be
        del Xtr
        gc.collect()

    if any(m in models for m in ("LSTM", "BiLSTM")):
        Xdeep = scaler.transform(bundle.X[deep_idx])
        ydeep = bundle.y[deep_idx]
        for k, model in enumerate(("LSTM", "BiLSTM")):
            if model not in models:
                continue
            pred, runtime = train_predict_deep(
                model,
                Xdeep,
                ydeep,
                Xte,
                args.deep_epochs,
                args.deep_batch,
                RANDOM_SEED + k + int(test_battery.split("-")[-1]) * 1000,
            )
            predictions[model] = pred
            runtimes[model] = runtime
            train_counts[model] = len(deep_idx)
            backend[model] = "keras_direct_multioutput_DenseK"
        del Xdeep
        gc.collect()

    for model, pred in predictions.items():
        if pred.shape != yte.shape:
            raise RuntimeError(
                f"{test_battery} {model} prediction shape={pred.shape}, y={yte.shape}"
            )

    overall = overall_fold_metrics(
        test_battery,
        yte,
        predictions,
        runtimes,
        train_counts,
        source_bats,
        backend,
    )
    leadwise = leadwise_metrics_for_fold(test_battery, yte, predictions)
    long_df = predictions_to_long(
        test_battery,
        bundle.origin_cycle[test_idx],
        bundle.segment[test_idx],
        yte,
        predictions,
    )

    ensure_dir(cache_dir)
    long_df.to_csv(pred_cache, index=False, compression=FAST_GZIP)
    overall.to_csv(overall_cache, index=False, encoding="utf-8-sig")
    leadwise.to_csv(lead_cache, index=False, compression=FAST_GZIP)
    save_json(sig, meta_cache)

    del Xte, Xtflat
    gc.collect()
    return long_df, overall, leadwise


# =============================================================================
# 8. Exact pooled metrics from streaming fold outputs
# =============================================================================

@dataclass
class MetricAccumulator:
    n: int = 0
    sum_abs: float = 0.0
    sum_sqerr: float = 0.0
    sum_err: float = 0.0
    sum_ape: float = 0.0
    sum_y: float = 0.0
    sum_y2: float = 0.0

    def update(self, y: np.ndarray, p: np.ndarray) -> None:
        yt = np.asarray(y, dtype=float).reshape(-1)
        yp = np.asarray(p, dtype=float).reshape(-1)
        m = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[m]
        yp = yp[m]
        if len(yt) == 0:
            return
        e = yp - yt
        self.n += int(len(yt))
        self.sum_abs += float(np.sum(np.abs(e)))
        self.sum_sqerr += float(np.sum(e ** 2))
        self.sum_err += float(np.sum(e))
        self.sum_ape += float(np.sum(np.abs(e) / np.maximum(np.abs(yt), 1e-9)))
        self.sum_y += float(np.sum(yt))
        self.sum_y2 += float(np.sum(yt ** 2))

    def metrics(self) -> Dict[str, float]:
        if self.n == 0:
            return dict(N=0, MAE=np.nan, RMSE=np.nan, R2=np.nan, MAPE_pct=np.nan, Bias=np.nan)
        sst = self.sum_y2 - (self.sum_y ** 2) / self.n
        r2 = 1.0 - self.sum_sqerr / sst if sst > 1e-18 else np.nan
        return dict(
            N=int(self.n),
            MAE=float(self.sum_abs / self.n),
            RMSE=float(math.sqrt(self.sum_sqerr / self.n)),
            R2=float(r2) if np.isfinite(r2) else np.nan,
            MAPE_pct=float(self.sum_ape / self.n * 100.0),
            Bias=float(self.sum_err / self.n),
        )


def aggregate_from_fold_predictions(
    fold_pred_paths: Sequence[Path],
    models: Sequence[str],
    K: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lead_acc: Dict[Tuple[str, int], MetricAccumulator] = {
        (m, lead): MetricAccumulator()
        for m in models
        for lead in range(1, K + 1)
    }
    all_acc: Dict[str, MetricAccumulator] = {m: MetricAccumulator() for m in models}

    for p in fold_pred_paths:
        d = pd.read_csv(p, compression="infer", low_memory=False)
        y = pd.to_numeric(d["SOH_True"], errors="coerce").to_numpy(float)
        lead_arr = pd.to_numeric(d["Lead"], errors="coerce").to_numpy(int)
        for m in models:
            col = f"Pred_{m}"
            if col not in d.columns:
                continue
            pred = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
            all_acc[m].update(y, pred)
            for lead in range(1, K + 1):
                mask = lead_arr == lead
                if mask.any():
                    lead_acc[(m, lead)].update(y[mask], pred[mask])
        del d
        gc.collect()

    pooled_lead_rows = []
    for m in models:
        for lead in range(1, K + 1):
            pooled_lead_rows.append({
                "Model": m,
                "Lead": int(lead),
                **lead_acc[(m, lead)].metrics(),
            })

    overall_rows = [
        {"Model": m, **all_acc[m].metrics()}
        for m in models
    ]
    return pd.DataFrame(pooled_lead_rows), pd.DataFrame(overall_rows)


def macro_lead_metrics(fold_lead: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, lead), g in fold_lead.groupby(["Model", "Lead"], sort=True):
        row = {
            "Model": str(model),
            "Lead": int(lead),
            "Folds": int(g["TestBattery"].nunique()),
        }
        for c in ("MAE", "RMSE", "R2", "MAPE_pct", "Bias"):
            row[f"MacroMean_{c}"] = safe_mean(g[c])
            row[f"MacroMedian_{c}"] = safe_median(g[c])
        row["WorstFold_RMSE"] = safe_max(g["RMSE"])
        row["NegativeR2Folds"] = int((pd.to_numeric(g["R2"], errors="coerce") < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def overall_macro_metrics(fold_overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in fold_overall.groupby("Model", sort=True):
        row = {
            "Model": str(model),
            "Folds": int(g["TestBattery"].nunique()),
        }
        cols = [
            "MAE", "RMSE", "R2", "MAPE_pct", "Bias",
            "MeanTrajectoryRMSE", "MedianTrajectoryRMSE", "MeanTrajectoryMAE",
            "EndpointMAE", "TrueUpwardStepFraction", "PredUpwardStepFraction",
            "ExcessUpwardStepFraction", "TrueRoughness", "PredRoughness",
        ]
        for c in cols:
            if c in g.columns:
                row[f"MacroMean_{c}"] = safe_mean(g[c])
                row[f"MacroMedian_{c}"] = safe_median(g[c])
        rows.append(row)
    return pd.DataFrame(rows)


def stream_combine_fold_predictions(paths: Sequence[Path], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as fh:
        first = True
        for p in paths:
            d = pd.read_csv(p, compression="infer", low_memory=False)
            d.to_csv(fh, index=False, header=first)
            first = False
            del d
            gc.collect()


# =============================================================================
# 9. Fair comparison to V4.6A single-point OOF on common test cohort
# =============================================================================

def load_singlepoint_oof(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    d = pd.read_csv(path, compression="infer", low_memory=False)
    required = {
        "Horizon", "Battery", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly", "SOH_True"
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"V4.6A OOF缺列：{missing}")
    d["Battery"] = d["Battery"].map(normalize_battery)
    for c in ("Horizon", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["SOH_True"] = pd.to_numeric(d["SOH_True"], errors="coerce")
    d = d.dropna(subset=["Horizon", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly", "SOH_True"]).copy()
    for c in ("Horizon", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly"):
        d[c] = d[c].astype(int)
    return d


def compare_singlepoint_common(
    fold_pred_paths: Sequence[Path],
    singlepoint_oof: pd.DataFrame,
    models: Sequence[str],
    K: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    available_h = sorted(
        h for h in singlepoint_oof["Horizon"].unique().tolist()
        if 1 <= int(h) <= K
    )
    if not available_h:
        return pd.DataFrame(), pd.DataFrame()

    traj_parts = []
    for p in fold_pred_paths:
        d = pd.read_csv(p, compression="infer", low_memory=False)
        q = d[d["Lead"].isin(available_h)].copy()
        if len(q):
            traj_parts.append(q)
    if not traj_parts:
        return pd.DataFrame(), pd.DataFrame()
    traj = pd.concat(traj_parts, ignore_index=True)
    del traj_parts

    keys_tr = ["Battery", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly", "Lead"]
    keys_sp = ["Battery", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly", "Horizon"]
    if traj.duplicated(keys_tr).any():
        raise RuntimeError("trajectory OOF common-comparison key重复")
    if singlepoint_oof.duplicated(keys_sp).any():
        raise RuntimeError("V4.6A OOF common-comparison key重复")

    sp = singlepoint_oof.rename(columns={
        "Horizon": "Lead",
        "SOH_True": "SOH_True_SP",
        **{f"Pred_{m}": f"Pred_SP_{m}" for m in models if f"Pred_{m}" in singlepoint_oof.columns},
    })
    keep_sp = [
        "Battery", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly", "Lead", "SOH_True_SP",
        *[f"Pred_SP_{m}" for m in models if f"Pred_SP_{m}" in sp.columns],
    ]
    merged = traj.merge(
        sp[keep_sp],
        on=["Battery", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly", "Lead"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) == 0:
        return pd.DataFrame(), pd.DataFrame()

    diff = np.abs(
        pd.to_numeric(merged["SOH_True"], errors="coerce")
        - pd.to_numeric(merged["SOH_True_SP"], errors="coerce")
    )
    if np.nanmax(diff.to_numpy(float)) > 1e-12:
        raise RuntimeError(
            f"trajectory与V4.6A common cohort SOH_True不一致，max diff={np.nanmax(diff):.3e}"
        )

    rows = []
    for lead, g in merged.groupby("Lead", sort=True):
        y = g["SOH_True"].to_numpy(float)
        for m in models:
            ct = f"Pred_{m}"
            cs = f"Pred_SP_{m}"
            if ct not in g.columns or cs not in g.columns:
                continue
            mt = calc_metrics(y, g[ct].to_numpy(float))
            ms = calc_metrics(y, g[cs].to_numpy(float))
            rows.append({
                "Lead": int(lead),
                "Model": m,
                "CommonWindows": int(len(g)),
                "Trajectory_RMSE": mt["RMSE"],
                "SinglePoint_RMSE": ms["RMSE"],
                "Trajectory_MAE": mt["MAE"],
                "SinglePoint_MAE": ms["MAE"],
                "Trajectory_R2": mt["R2"],
                "SinglePoint_R2": ms["R2"],
                "TrajectoryVsSinglePoint_RMSE_Improvement_pct": (
                    (ms["RMSE"] - mt["RMSE"]) / ms["RMSE"] * 100.0
                    if np.isfinite(ms["RMSE"]) and ms["RMSE"] > 0
                    else np.nan
                ),
            })
    return pd.DataFrame(rows), merged


# =============================================================================
# 10. Figures
# =============================================================================

def make_figures(
    pooled_lead: pd.DataFrame,
    fold_pred_paths: Sequence[Path],
    models: Sequence[str],
    outdir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = ensure_dir(outdir / "Figures")

    for metric in ("RMSE", "MAE", "R2"):
        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        for model, g in pooled_lead.groupby("Model", sort=False):
            ax.plot(g["Lead"], g[metric], label=model, linewidth=1.8)
        ax.set_xlabel("Future lead (cycles)")
        ax.set_ylabel(metric)
        ax.set_title(f"Full-trajectory OOF {metric} vs future lead")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / f"01_leadwise_{metric}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # Representative trajectories: earliest cached origin from up to 4 held-out cells.
    selected = list(fold_pred_paths[:4])
    for p in selected:
        d = pd.read_csv(p, compression="infer", low_memory=False)
        if d.empty:
            continue
        battery = str(d["Battery"].iloc[0])
        origin = int(d["OriginCycle_KeyOnly"].min())
        q = d[d["OriginCycle_KeyOnly"].eq(origin)].sort_values("Lead")
        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        ax.plot(q["TargetCycle_KeyOnly"], q["SOH_True"], label="True SOH", linewidth=2.4)
        for m in models:
            c = f"Pred_{m}"
            if c in q.columns:
                ax.plot(q["TargetCycle_KeyOnly"], q[c], label=m, linewidth=1.5)
        ax.set_xlabel("Target cycle")
        ax.set_ylabel("SOH")
        ax.set_title(f"{battery} | origin={origin} | full future trajectory")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figdir / f"02_example_{safe_name(battery)}_origin_{origin}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)
        del d, q


# =============================================================================
# 11. Main
# =============================================================================

def main() -> None:
    args = parse_args()
    t_all = time.perf_counter()

    J = int(args.history_interval)
    L = int(args.history_observations)
    K = int(args.future_length)
    models = list(dict.fromkeys(args.models))

    if J < 1:
        raise ValueError("--history-interval必须>=1")
    if L < 2:
        raise ValueError("--history-observations必须>=2")
    if K < 2:
        raise ValueError("--future-length必须>=2；整轨迹任务不建议K=1")
    if not models:
        raise ValueError("至少选择一个模型")

    out = ensure_dir(args.output_root)
    cache_dir = ensure_dir(out / "Cache_Folds")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(args.threads_model))))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(max(1, int(args.threads_model))))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

    # Dependency preflight: fail before building large windows if a requested backend is absent.
    dependency_rows = []
    if "XGBoost" in models:
        try:
            import xgboost as _xgb
            dependency_rows.append({"Package": "xgboost", "RequiredBy": "XGBoost", "Available": 1, "Version": getattr(_xgb, "__version__", "")})
        except Exception as exc:
            dependency_rows.append({"Package": "xgboost", "RequiredBy": "XGBoost", "Available": 0, "Version": ""})
            pd.DataFrame(dependency_rows).to_csv(out / "00_dependency_audit.csv", index=False, encoding="utf-8-sig")
            raise RuntimeError("请求了XGBoost，但当前Python环境未安装xgboost。") from exc
    if any(m in models for m in ("LSTM", "BiLSTM")):
        try:
            import tensorflow as _tf
            dependency_rows.append({"Package": "tensorflow", "RequiredBy": "LSTM/BiLSTM", "Available": 1, "Version": getattr(_tf, "__version__", "")})
        except Exception as exc:
            dependency_rows.append({"Package": "tensorflow", "RequiredBy": "LSTM/BiLSTM", "Available": 0, "Version": ""})
            pd.DataFrame(dependency_rows).to_csv(out / "00_dependency_audit.csv", index=False, encoding="utf-8-sig")
            raise RuntimeError(
                "请求了LSTM/BiLSTM，但当前Python环境未安装TensorFlow。"
                "请切回已经成功运行V4.6A/V4.6B的环境，或临时用 --models RF XGBoost。"
            ) from exc
    if not dependency_rows:
        dependency_rows.append({"Package": "sklearn", "RequiredBy": "RF", "Available": 1, "Version": ""})
    pd.DataFrame(dependency_rows).to_csv(out / "00_dependency_audit.csv", index=False, encoding="utf-8-sig")

    set_seed(RANDOM_SEED)

    config = {
        "Task": "RWTH48 V4.9 HI-only full future SOH trajectory prognosis",
        "StrictInput": str(args.strict_input),
        "HIInput": str(args.hi_input),
        "V46AComplete": str(args.v46a_complete),
        "SinglePointOOF": str(args.singlepoint_oof),
        "Output": str(out),
        "Checkpoints_s": list(CHECKPOINTS_S),
        "CheckpointDefinition": "cumulative 0-T prefixes",
        "CompactHI": list(COMPACT_HI),
        "HIColumnsPerObservation": len(HI_COLUMNS),
        "HistoryInterval_J": J,
        "HistoryObservations_L": L,
        "HistorySpan_cycles": (L - 1) * J,
        "FutureTrajectory": f"SOH(t+1)...SOH(t+{K})",
        "FutureLength_K": K,
        "FutureStep_cycles": 1,
        "InputShape": [L, len(HI_COLUMNS)],
        "OutputShape": [K],
        "Models": models,
        "OnlyHIAsModelInput": True,
        "GlobalCycleAsFeature": False,
        "HistoricalSOHAsFeature": False,
        "FutureHIAsFeature": False,
        "FutureHIAccessRequired": False,
        "PeriodicSOHCalibration": False,
        "NearestCycleSubstitution": False,
        "ExactContinuousFutureTargets": True,
        "StrictSameSegment": True,
        "OuterValidation": "46 source batteries train / 1 whole battery held out; rotate over 47.",
        "TrainingCapsPerSourceBattery": {
            "RF": RF_PER_SOURCE_BATTERY,
            "XGBoost": XGB_PER_SOURCE_BATTERY,
            "LSTM": DEEP_PER_SOURCE_BATTERY,
            "BiLSTM": DEEP_PER_SOURCE_BATTERY,
        },
        "DeepEpochs": int(args.deep_epochs),
        "DeepBatch": int(args.deep_batch),
        "Important": [
            "The target is a complete direct multi-output trajectory, not three independent H points.",
            "Future targets require strict SOH labels only; future HI existence is never required.",
            "HI is required only at exact J-spaced historical diagnostic cycles.",
            "No historical SOH, future HI, capacity, cycle index, RPT SOH or calendar time enters X.",
            "All target cycles t+1..t+K must exist exactly inside the same strict segment.",
            "Held-out battery is excluded from imputation, scaling and fitting.",
            "Raw predicted trajectories are evaluated without monotonic post-processing.",
        ],
    }
    save_json(config, out / "00_config.json")

    print("=" * 118)
    print("RWTH48 V4.9 | HI-only FULL FUTURE SOH TRAJECTORY | 47-cell Outer LOBO")
    print("STRICT       :", args.strict_input)
    print("HI           :", args.hi_input)
    print("OUTPUT       :", out)
    print("INPUT SHAPE  :", f"[N,{L},{len(HI_COLUMNS)}]")
    print("TARGET       :", f"SOH(t+1)...SOH(t+{K})")
    print("J / L / K    :", J, "/", L, "/", K)
    print("MODELS       :", models)
    print("NO SOH INPUT | NO FUTURE HI | NO CYCLE FEATURE")
    print("=" * 118, flush=True)

    # A. frozen configuration audit
    print("[A] Validating V4.6A frozen configuration ...", flush=True)
    cfg_audit = validate_v46a_complete(
        args.v46a_complete, J, L, args.allow_v46a_config_mismatch
    )
    cfg_audit.to_csv(out / "01_v46a_configuration_audit.csv", index=False, encoding="utf-8-sig")

    # B. inputs
    print("[B] Loading strict SOH labels and frozen prefix HI ...", flush=True)
    strict = load_strict_target(args.strict_input)
    hi = load_prefix_hi(args.hi_input)
    batteries = sorted(strict["Battery"].unique())
    if len(batteries) != 47:
        raise RuntimeError(f"预期47节有效电池（排除RWTH-028），实际{len(batteries)}：{batteries}")

    hi_bats = sorted(set(hi["Battery"].unique()) & set(batteries))
    missing_hi_bats = sorted(set(batteries) - set(hi_bats))
    if missing_hi_bats:
        raise RuntimeError(f"以下strict电池完全没有HI：{missing_hi_bats}")

    input_audit = build_input_audit(strict, hi)
    input_audit.to_csv(out / "02_input_HI_coverage_audit.csv", index=False, encoding="utf-8-sig")

    # C. windows
    print(
        f"[C] Building exact sparse-history -> continuous t+1...t+{K} trajectory windows ...",
        flush=True,
    )
    tw = time.perf_counter()
    bundle, window_inventory = build_full_trajectory_windows(strict, hi, J, L, K)
    window_build_s = time.perf_counter() - tw
    window_inventory.to_csv(
        out / "03_trajectory_window_inventory_by_segment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if len(bundle.y) == 0:
        raise RuntimeError(f"J={J}, L={L}, K={K} 无合法完整未来轨迹窗口")

    counts = pd.Series(bundle.battery).value_counts().sort_index()
    window_battery_summary = pd.DataFrame({
        "Battery": counts.index,
        "TrajectoryWindows": counts.values,
    })
    window_battery_summary.to_csv(
        out / "04_trajectory_window_inventory_by_battery.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_window_bats = sorted(set(batteries) - set(counts.index))
    if missing_window_bats:
        raise RuntimeError(
            f"以下电池没有 K={K} 完整轨迹测试窗口：{missing_window_bats}\n"
            "若确实是strict segment长度不足，可先用 --future-length 50 做消融；"
            "主任务仍建议K=100。"
        )

    print(
        f"    windows={len(bundle.y):,} | cells={len(counts)} | "
        f"per-cell min/median/max={counts.min()}/{counts.median():.0f}/{counts.max()} | "
        f"build={window_build_s:.1f}s",
        flush=True,
    )

    if args.only_battery:
        ob = normalize_battery(args.only_battery)
        if ob not in batteries:
            raise RuntimeError(f"--only-battery不存在：{ob}")
        outer_test_batteries = [ob]
    else:
        outer_test_batteries = batteries

    # D. LOBO
    print("[D] Running multi-output LOBO folds ...", flush=True)
    fold_overall_parts: List[pd.DataFrame] = []
    fold_lead_parts: List[pd.DataFrame] = []
    fold_pred_paths: List[Path] = []
    runtime_rows = []

    for fi, test_battery in enumerate(outer_test_batteries, 1):
        print(
            f"  [fold {fi:02d}/{len(outer_test_batteries)}] held-out={test_battery}",
            flush=True,
        )
        tfold = time.perf_counter()
        long_df, overall_df, lead_df = run_one_fold(
            bundle, test_battery, models, J, L, K, args, cache_dir
        )
        fold_overall_parts.append(overall_df)
        fold_lead_parts.append(lead_df)
        pred_path = cache_dir / f"{safe_name(test_battery)}_trajectory_predictions.csv.gz"
        fold_pred_paths.append(pred_path)
        runtime_rows.append({
            "TestBattery": test_battery,
            "FoldRuntime_s": float(time.perf_counter() - tfold),
            "TestTrajectoryWindows": int(long_df["OriginCycle_KeyOnly"].nunique()),
            "LongPredictionRows": int(len(long_df)),
        })
        print(
            "      " + " | ".join(
                f"{r.Model}: pointRMSE={r.RMSE:.6f}, trajRMSE={r.MeanTrajectoryRMSE:.6f}"
                for r in overall_df.itertuples()
            ),
            flush=True,
        )
        del long_df
        gc.collect()

    fold_overall = pd.concat(fold_overall_parts, ignore_index=True)
    fold_lead = pd.concat(fold_lead_parts, ignore_index=True)
    runtime_df = pd.DataFrame(runtime_rows)

    fold_overall.to_csv(out / "05_fold_overall_trajectory_metrics.csv", index=False, encoding="utf-8-sig")
    fold_lead.to_csv(out / "06_fold_leadwise_metrics.csv.gz", index=False, compression=FAST_GZIP)
    runtime_df.to_csv(out / "07_fold_runtime.csv", index=False, encoding="utf-8-sig")

    # E. pooled / macro
    print("[E] Aggregating exact pooled lead-wise OOF metrics ...", flush=True)
    pooled_lead, pooled_overall = aggregate_from_fold_predictions(fold_pred_paths, models, K)
    macro_lead = macro_lead_metrics(fold_lead)
    macro_overall = overall_macro_metrics(fold_overall)

    pooled_lead.to_csv(out / "08_leadwise_pooled_OOF_metrics.csv", index=False, encoding="utf-8-sig")
    macro_lead.to_csv(out / "09_leadwise_macro_LOBO_metrics.csv", index=False, encoding="utf-8-sig")
    pooled_overall.to_csv(out / "10_overall_pointwise_pooled_OOF_metrics.csv", index=False, encoding="utf-8-sig")
    macro_overall.to_csv(out / "11_overall_trajectory_macro_metrics.csv", index=False, encoding="utf-8-sig")

    # F. fair comparison with V4.6A single-point models
    compare_df = pd.DataFrame()
    if not args.skip_singlepoint_compare and args.singlepoint_oof.exists():
        print("[F] Comparing joint trajectory vs V4.6A single-point models on exact common cohort ...", flush=True)
        sp = load_singlepoint_oof(args.singlepoint_oof)
        if args.only_battery:
            sp = sp[sp["Battery"].eq(normalize_battery(args.only_battery))].copy()
        compare_df, common_rows = compare_singlepoint_common(
            fold_pred_paths, sp, models, K
        )
        compare_df.to_csv(
            out / "12_singlepoint_vs_trajectory_common_cohort_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if len(common_rows):
            # Small: only leads present in V4.6A, usually 10/50/100.
            common_rows.to_csv(
                out / "13_singlepoint_vs_trajectory_common_rows.csv.gz",
                index=False,
                compression=FAST_GZIP,
            )
        del sp, common_rows
        gc.collect()
    else:
        print("[F] Single-point common comparison skipped/unavailable.", flush=True)

    # G. combined OOF long
    final_oof = out / "14_ALL_OOF_full_trajectory_predictions_long.csv.gz"
    if not args.skip_final_oof:
        print("[G] Streaming fold predictions into final long-format OOF file ...", flush=True)
        stream_combine_fold_predictions(fold_pred_paths, final_oof)

    # H. figures
    if not args.skip_figures:
        print("[H] Making trajectory figures ...", flush=True)
        make_figures(pooled_lead, fold_pred_paths, models, out)

    # I. acceptance gate
    gate_rows = []
    gate_rows.append({
        "Gate": "47StrictBatteries",
        "Observed": len(batteries),
        "Expected": 47,
        "PASS": int(len(batteries) == 47),
    })
    gate_rows.append({
        "Gate": "AllStrictBatteriesHaveTrajectoryWindows",
        "Observed": len(counts),
        "Expected": 47,
        "PASS": int(len(counts) == 47),
    })
    gate_rows.append({
        "Gate": "FullTrajectoryLength",
        "Observed": K,
        "Expected": K,
        "PASS": int(bundle.y.shape[1] == K),
    })
    gate_rows.append({
        "Gate": "NoFutureHIRequiredByWindowBuilder",
        "Observed": True,
        "Expected": True,
        "PASS": 1,
    })
    gate_rows.append({
        "Gate": "NoHistoricalSOHAsInput",
        "Observed": False,
        "Expected": False,
        "PASS": 1,
    })
    gate_rows.append({
        "Gate": "RequestedOuterFoldsCompleted",
        "Observed": int(fold_overall["TestBattery"].nunique()),
        "Expected": len(outer_test_batteries),
        "PASS": int(fold_overall["TestBattery"].nunique() == len(outer_test_batteries)),
    })
    gate_rows.append({
        "Gate": "AllRequestedModelsPresent",
        "Observed": sorted(fold_overall["Model"].unique().tolist()),
        "Expected": sorted(models),
        "PASS": int(set(fold_overall["Model"]) == set(models)),
    })
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(out / "15_final_acceptance_gate.csv", index=False, encoding="utf-8-sig")
    gate_pass = bool((gate["PASS"] == 1).all())

    # J. summary / completion
    best_overall = None
    if len(pooled_overall):
        r = pooled_overall.sort_values(["RMSE", "MAE"]).iloc[0]
        best_overall = {
            "Model": str(r["Model"]),
            "RMSE": float(r["RMSE"]),
            "MAE": float(r["MAE"]),
            "R2": float(r["R2"]),
        }

    selected_leads = sorted(set([1, 10, 50, 100, K]))
    selected_leads = [h for h in selected_leads if 1 <= h <= K]
    lead_best = []
    for lead in selected_leads:
        g = pooled_lead[pooled_lead["Lead"].eq(lead)]
        if len(g):
            r = g.sort_values(["RMSE", "MAE"]).iloc[0]
            lead_best.append({
                "Lead": int(lead),
                "BestModel": str(r["Model"]),
                "RMSE": float(r["RMSE"]),
                "MAE": float(r["MAE"]),
                "R2": float(r["R2"]),
            })

    elapsed = float(time.perf_counter() - t_all)
    complete = {
        "Task": "RWTH48 V4.9 HI-only full future SOH trajectory prognosis",
        "Status": "PASS" if gate_pass else "CHECK",
        "Batteries": 47,
        "OuterTestBatteriesRun": int(len(outer_test_batteries)),
        "InputShape": [L, len(HI_COLUMNS)],
        "FutureLength_K": K,
        "Target": f"SOH(t+1)...SOH(t+{K})",
        "TrajectoryWindows": int(len(bundle.y)),
        "HistoryInterval_J": J,
        "HistoryObservations_L": L,
        "HistorySpan_cycles": int((L - 1) * J),
        "Models": models,
        "OnlyHIAsModelInput": True,
        "HistoricalSOHAsFeature": False,
        "FutureHIAsFeature": False,
        "FutureHIAccessRequired": False,
        "CycleAsFeature": False,
        "StrictContinuousTargets": True,
        "BestOverallPointwise": best_overall,
        "BestAtSelectedLeads": lead_best,
        "SinglePointCommonComparisonRows": int(len(compare_df)),
        "FinalOOFPath": str(final_oof) if not args.skip_final_oof else None,
        "Elapsed_s": elapsed,
        "Important": [
            "This is direct multi-output trajectory prognosis, not recursive use of predicted SOH.",
            "No predicted/true SOH is fed back into the input sequence.",
            "Future t+1..t+K SOH labels are outputs only.",
            "Future HI existence is not used to decide target availability.",
            "All historical HI observations and all future SOH labels stay inside one strict segment.",
            "V4.6A single-point comparison is evaluated on exact common test cohorts at matching leads.",
        ],
    }
    save_json(complete, out / "99_RWTH48_V4_9_COMPLETE.json")

    print("\n" + "=" * 118)
    print("RWTH48 V4.9 COMPLETE")
    print("Status        :", complete["Status"])
    print("Windows       :", f"{len(bundle.y):,}")
    print("Input shape   :", complete["InputShape"])
    print("Output length :", K)
    print("Outer folds   :", len(outer_test_batteries))
    print("Models        :", models)
    print("Elapsed       :", f"{elapsed / 3600.0:.2f} h")
    print("Output        :", out)
    print("=" * 118)

    if best_overall:
        print(
            f"Best overall pointwise: {best_overall['Model']} | "
            f"RMSE={best_overall['RMSE']:.6f} | "
            f"MAE={best_overall['MAE']:.6f} | "
            f"R2={best_overall['R2']:.6f}"
        )

    if lead_best:
        print("Best model at selected future leads:")
        for r in lead_best:
            print(
                f"  lead={r['Lead']:>3} | {r['BestModel']:<9} | "
                f"RMSE={r['RMSE']:.6f} | MAE={r['MAE']:.6f} | R2={r['R2']:.6f}"
            )

    if len(compare_df):
        print("\nTrajectory vs V4.6A single-point (common cohort):")
        print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()

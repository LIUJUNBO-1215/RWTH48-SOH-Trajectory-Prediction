# -*- coding: utf-8 -*-
r"""
RWTH48 V4.8B LITE
FixedTime Top2 × J/L 稀疏诊断公平消融 + 47-cell Outer LOBO
================================================================================

研究问题
--------
V4.7 已在 FixedTime cumulative-prefix 路线中冻结 Top2：
    V_TimeWeightedMean
    I_Slope_A_per_s

V4.8B 不再改变 HI，只回答：
    1) 多久做一次 120 s 短时诊断（J）？
    2) 预测时需要保留过去多少次诊断（L）？
    3) 在历史覆盖范围相近时，诊断次数能否从 10 次降到 5 次甚至 3 次？

冻结不变
--------
    FixedTime cumulative checkpoints : 0-30 / 0-60 / 0-90 / 0-120 s
    Top2 HI                          : V_TimeWeightedMean + I_Slope_A_per_s
    H                               : 10 / 50 / 100 cycles
    Models                          : RF / XGBoost / LSTM / BiLSTM
    Outer validation                : 47-cell LOBO
    X                               : HI only
    No cycle index / historical SOH / RPT SOH / capacity / future HI in X
    All history and t+H target must stay inside one StrictSegmentKey
    Exact history cycles only; no nearest-cycle substitution

默认 J/L 组合
------------
    J05_L10 : J=5,  L=10, span=45 cycles, 10 diagnoses  [V4.7 baseline]
    J10_L05 : J=10, L=5,  span=40 cycles,  5 diagnoses  [primary matched-span]
    J20_L03 : J=20, L=3,  span=40 cycles,  3 diagnoses  [primary matched-span]
    J05_L05 : J=5,  L=5,  span=20 cycles,  5 diagnoses  [history-length ablation]
    J10_L10 : J=10, L=10, span=90 cycles, 10 diagnoses  [long-history stress test]

主公平比较
----------
Primary matched-span common cohort：
    J05_L10 vs J10_L05 vs J20_L03

三组历史跨度约 40~45 cycles，但历史诊断次数：
    10 -> 5 -> 3

每个 H 都对 Battery + OriginCycle + TargetCycle 做 exact intersection，
只在完全相同的 held-out test windows 上比较模型误差。

次级公平比较
------------
    J05_L10 vs J05_L05 : 固定 J=5，比较历史长度/诊断次数
    J05_L10 vs J10_L10 : 固定 L=10，比较诊断间隔/历史跨度

注意：J10_L10 在 H=100 可能因 strict segment 长度而合法窗口极少或为空。
代码会先做 availability audit；无合法窗口的 config × H 会记录并跳过，绝不 nearest 替代。

V4.7 baseline 复用
------------------
若存在：
    RWTH48_V4_7_HI数量消融_SourceOnlyRanking_LOBO_LITE/
        13_ALL_OOF_HI_count_ablation.csv.gz
代码会验证 Top2 确实为冻结的两项 HI，并直接复用 J=5,L=10 的 Top2 OOF。
若文件不存在或使用 --retrain-baseline，则重新训练 baseline。

默认新电脑路径
--------------
workspace

运行
----
1) 先审计（强烈建议）：
    python -u "src/25_fixedtime_sparse_search_helper.py" --audit-only

2) 单折 smoke：
    python -u "src/25_fixedtime_sparse_search_helper.py" --only-battery RWTH-002 --horizons 10 --configs J05_L10 J10_L05 J20_L03

3) 正式全部：
    python -u "src/25_fixedtime_sparse_search_helper.py"

4) 只跑 CPU 树模型：
    python -u "src/25_fixedtime_sparse_search_helper.py" --models RF XGBoost

5) 只跑深度模型：
    python -u "src/25_fixedtime_sparse_search_helper.py" --models LSTM BiLSTM

6) 强制重训 cache：
    python -u "src/25_fixedtime_sparse_search_helper.py" --force-folds

重要输出
--------
00_config.json
01_input_compatibility_audit.csv
02_JL_availability_inventory.csv
03_primary_common_cohort_inventory.csv
04_fold_model_metrics.csv
05_own_cohort_pooled_metrics.csv
06_pairwise_common_cohort_metrics.csv
07_primary_matched_span_common_metrics.csv
08_primary_gain_vs_J05_L10.csv
09_measurement_efficiency_summary.csv
10_runtime_summary.csv
11_ALL_OOF_JL_ablation.csv.gz
12_acceptance_gate.csv
13_primary_common_predictions.csv.gz
Figures/
99_RWTH48_V4_8B_COMPLETE.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# =============================================================================
# 0. Frozen configuration
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")

DEFAULT_V46A_ROOT = BASE / "RWTH48_V4_6A_HIOnly_120sMultiCheckpoint_LOBO_LITE"
DEFAULT_STRICT_INPUT = (
    BASE
    / "RWTH48_V4_5C_FIX_最小冲击共同扰动处理"
    / "06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz"
)
DEFAULT_HI_INPUT = DEFAULT_V46A_ROOT / "03_prefix_HI_all_cycles.csv.gz"
DEFAULT_V46A_COMPLETE = DEFAULT_V46A_ROOT / "99_RWTH48_V4_6A_COMPLETE.json"

DEFAULT_V47_ROOT = BASE / "RWTH48_V4_7_HI数量消融_SourceOnlyRanking_LOBO_LITE"
DEFAULT_V47_OOF = DEFAULT_V47_ROOT / "13_ALL_OOF_HI_count_ablation.csv.gz"
DEFAULT_V47_COMPLETE = DEFAULT_V47_ROOT / "99_RWTH48_V4_7_COMPLETE.json"

DEFAULT_OUTPUT = BASE / "RWTH48_V4_8B_FixedTime_Top2_JL稀疏诊断消融_LOBO_LITE"

TARGET = "CyclingSOH_Charge_Canonical"
SEGMENT = "StrictSegmentKey"
EXCLUDED_BATTERIES = {"RWTH-028"}

CHECKPOINTS_S = (30, 60, 90, 120)
TOP2_HI = (
    "V_TimeWeightedMean",
    "I_Slope_A_per_s",
)
TOP2_COLUMNS = tuple(
    f"T{sec:03d}__{hi}"
    for sec in CHECKPOINTS_S
    for hi in TOP2_HI
)

DEFAULT_HORIZONS = (10, 50, 100)

# label -> (J, L, role)
JL_CONFIGS: Dict[str, Tuple[int, int, str]] = {
    "J05_L10": (5, 10, "baseline_matched_span"),
    "J10_L05": (10, 5, "primary_matched_span"),
    "J20_L03": (20, 3, "primary_matched_span"),
    "J05_L05": (5, 5, "history_length_ablation"),
    "J10_L10": (10, 10, "long_history_stress"),
}
PRIMARY_CONFIGS = ("J05_L10", "J10_L05", "J20_L03")
BASELINE_CONFIG = "J05_L10"

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
RANDOM_SEED = 20260821
FAST_GZIP = {"method": "gzip", "compresslevel": 1}

ALL_MODELS = ("RF", "XGBoost", "LSTM", "BiLSTM")
KEY = ["Horizon", "Battery", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly"]


# =============================================================================
# 1. CLI / utilities
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWTH48 V4.8B FixedTime Top2 J/L sparse-diagnostic fair ablation."
    )
    p.add_argument("--strict-input", type=Path, default=DEFAULT_STRICT_INPUT)
    p.add_argument("--hi-input", type=Path, default=DEFAULT_HI_INPUT)
    p.add_argument("--v46a-complete", type=Path, default=DEFAULT_V46A_COMPLETE)
    p.add_argument("--v47-oof", type=Path, default=DEFAULT_V47_OOF)
    p.add_argument("--v47-complete", type=Path, default=DEFAULT_V47_COMPLETE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)

    p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    p.add_argument(
        "--configs",
        nargs="+",
        choices=list(JL_CONFIGS),
        default=list(JL_CONFIGS),
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
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--force-folds", action="store_true")
    p.add_argument("--retrain-baseline", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: object, p: Path) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_battery(x: object) -> str:
    s = str(x).strip()
    m = re.search(r"(\d+)$", s)
    if m:
        return f"RWTH-{int(m.group(1)):03d}"
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
        R2=float(r2_score(yt, yp)) if len(yt) >= 2 and np.std(yt) > 1e-12 else np.nan,
        MAPE_pct=float(np.mean(np.abs(e) / np.maximum(np.abs(yt), 1e-9)) * 100.0),
        Bias=float(np.mean(e)),
    )


def configure_tensorflow_if_needed(models: Sequence[str]) -> Dict[str, object]:
    need = any(m in {"LSTM", "BiLSTM"} for m in models)
    info: Dict[str, object] = {
        "TensorFlowRequired": need,
        "TensorFlowAvailable": False,
        "TensorFlowVersion": "",
        "GPUDevices": [],
    }
    if not need:
        return info
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError("请求了 LSTM/BiLSTM，但当前环境无法 import TensorFlow。") from exc
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
# 2. Inputs
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
        pc = pd.to_numeric(d["GlobalCycle"].shift(), errors="coerce")
        ps = d["SourcePart"].shift().astype(str)
        br = d["Battery"].ne(pb) | (d["GlobalCycle"] - pc).ne(1)
        if d["SourcePart"].astype(str).str.len().gt(0).any():
            br = br | d["SourcePart"].astype(str).ne(ps)
        d["_seg"] = br.groupby(d["Battery"]).cumsum().astype(int)
        d[SEGMENT] = d["Battery"].astype(str) + "__AUTO" + d["_seg"].astype(str).str.zfill(4)
        d.drop(columns="_seg", inplace=True)

    if d.duplicated(["Battery", "GlobalCycle"]).any():
        raise RuntimeError("strict target存在重复 Battery+GlobalCycle")
    d = d[~d["Battery"].isin(EXCLUDED_BATTERIES)].copy()
    return d.sort_values(["Battery", "GlobalCycle"]).reset_index(drop=True)


def load_prefix_hi(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"V4.6A prefix HI不存在：{path}")
    d = pd.read_csv(path, compression="infer", low_memory=False)
    required = {"Battery", "GlobalCycle", *TOP2_COLUMNS}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError("prefix HI缺Top2冻结列：\n" + "\n".join(missing))
    d["Battery"] = d["Battery"].map(normalize_battery)
    d["GlobalCycle"] = pd.to_numeric(d["GlobalCycle"], errors="coerce")
    d = d[d["GlobalCycle"].notna()].copy()
    d["GlobalCycle"] = d["GlobalCycle"].astype(int)
    for c in TOP2_COLUMNS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[~d["Battery"].isin(EXCLUDED_BATTERIES)].copy()
    if d.duplicated(["Battery", "GlobalCycle"]).any():
        raise RuntimeError("prefix HI存在重复 Battery+GlobalCycle")
    return d.sort_values(["Battery", "GlobalCycle"]).reset_index(drop=True)


def validate_upstream(v46a_complete: Path, strict: pd.DataFrame, hi: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(check: str, observed: object, expected: object, ok: bool) -> None:
        rows.append({"Check": check, "Observed": observed, "Expected": expected, "PASS": int(bool(ok))})

    if not v46a_complete.exists():
        add("V4.6A COMPLETE exists", False, True, False)
    else:
        c = json.loads(v46a_complete.read_text(encoding="utf-8"))
        add("V4.6A status", c.get("Status"), "PASS", c.get("Status") == "PASS")
        add("Only HI as input", c.get("OnlyHIAsModelInput"), True, c.get("OnlyHIAsModelInput") is True)
        add("Historical SOH disabled", c.get("HistoricalSOHAsFeature"), False, c.get("HistoricalSOHAsFeature") is False)
        add("Periodic calibration disabled", c.get("PeriodicSOHCalibration"), False, c.get("PeriodicSOHCalibration") is False)
        cps = [int(x) for x in c.get("Checkpoints_s", [])]
        add("Cumulative checkpoints", cps, list(CHECKPOINTS_S), cps == list(CHECKPOINTS_S))

    bats = sorted(strict["Battery"].unique())
    hib = sorted(hi["Battery"].unique())
    add("47 strict batteries", len(bats), 47, len(bats) == 47)
    add("HI covers strict batteries", len(set(bats) & set(hib)), len(bats), set(bats).issubset(set(hib)))
    add("Top2 columns", len(TOP2_COLUMNS), 8, len(TOP2_COLUMNS) == 8)
    return pd.DataFrame(rows)


# =============================================================================
# 3. Exact J/L windows
# =============================================================================


@dataclass
class WindowBundle:
    X: np.ndarray
    y: np.ndarray
    battery: np.ndarray
    origin_cycle: np.ndarray
    target_cycle: np.ndarray
    segment: np.ndarray


def build_sparse_history_windows(merged: pd.DataFrame, J: int, L: int, H: int) -> WindowBundle:
    Xs: List[np.ndarray] = []
    ys: List[float] = []
    bs: List[str] = []
    origins: List[int] = []
    targets: List[int] = []
    segs: List[str] = []
    offsets = np.asarray([-J * k for k in range(L - 1, -1, -1)], dtype=int)

    for (battery, seg), g in merged.groupby(["Battery", SEGMENT], sort=False):
        g = g.sort_values("GlobalCycle").reset_index(drop=True)
        cycles = g["GlobalCycle"].to_numpy(int)
        cmap = {int(c): i for i, c in enumerate(cycles)}
        Xmat = g[list(TOP2_COLUMNS)].to_numpy(np.float32)
        yvec = g[TARGET].to_numpy(float)

        for origin in cycles:
            target = int(origin + H)
            if target not in cmap:
                continue
            hc = origin + offsets
            if not all(int(c) in cmap for c in hc):
                continue
            idx = [cmap[int(c)] for c in hc]
            y = yvec[cmap[target]]
            if not np.isfinite(y):
                continue
            Xs.append(Xmat[idx, :])
            ys.append(float(y))
            bs.append(str(battery))
            origins.append(int(origin))
            targets.append(target)
            segs.append(str(seg))

    if not Xs:
        return WindowBundle(
            X=np.empty((0, L, len(TOP2_COLUMNS)), np.float32),
            y=np.empty(0, float),
            battery=np.empty(0, object),
            origin_cycle=np.empty(0, int),
            target_cycle=np.empty(0, int),
            segment=np.empty(0, object),
        )
    return WindowBundle(
        X=np.stack(Xs).astype(np.float32, copy=False),
        y=np.asarray(ys, float),
        battery=np.asarray(bs, object),
        origin_cycle=np.asarray(origins, int),
        target_cycle=np.asarray(targets, int),
        segment=np.asarray(segs, object),
    )


def bundle_keys(bundle: WindowBundle, H: int) -> pd.DataFrame:
    return pd.DataFrame({
        "Horizon": int(H),
        "Battery": bundle.battery,
        "OriginCycle_KeyOnly": bundle.origin_cycle,
        "TargetCycle_KeyOnly": bundle.target_cycle,
        "SOH_True": bundle.y,
    })


# =============================================================================
# 4. Source balancing / scaler
# =============================================================================


def balanced_indices(battery: np.ndarray, source_batteries: Sequence[str], per_battery: int) -> np.ndarray:
    idx: List[int] = []
    barr = np.asarray(battery, object)
    for b in source_batteries:
        q = np.flatnonzero(barr == b)
        if len(q) == 0:
            continue
        if per_battery <= 0 or len(q) <= per_battery:
            pick = q
        else:
            pos = np.linspace(0, len(q) - 1, int(per_battery))
            pick = q[np.round(pos).astype(int)]
        idx.extend(pick.tolist())
    return np.asarray(sorted(set(idx)), dtype=int)


@dataclass
class SeqScaler:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = np.asarray(X, np.float32).copy()
        for j in range(z.shape[2]):
            a = z[:, :, j]
            bad = ~np.isfinite(a)
            if bad.any():
                a[bad] = self.median[j]
            z[:, :, j] = (a - self.mean[j]) / self.std[j]
        return z


def fit_scaler(X_source: np.ndarray) -> SeqScaler:
    flat = np.asarray(X_source, np.float64).reshape(-1, X_source.shape[2])
    med = np.nanmedian(flat, axis=0)
    med[~np.isfinite(med)] = 0.0
    filled = flat.copy()
    for j in range(filled.shape[1]):
        bad = ~np.isfinite(filled[:, j])
        filled[bad, j] = med[j]
    mean = np.mean(filled, axis=0)
    std = np.std(filled, axis=0)
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    return SeqScaler(med.astype(np.float32), mean.astype(np.float32), std.astype(np.float32))


# =============================================================================
# 5. Frozen models
# =============================================================================


def build_deep_model(kind: str, L: int, F: int):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    inp = tf.keras.Input(shape=(L, F), name="HI_history")
    if kind == "LSTM":
        x = tf.keras.layers.LSTM(32, activation="tanh", recurrent_activation="sigmoid")(inp)
    elif kind == "BiLSTM":
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(24, activation="tanh", recurrent_activation="sigmoid")
        )(inp)
    else:
        raise ValueError(kind)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    out = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inp, out, name=kind)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=DEEP_LR), loss="mse")
    return model


def train_predict_deep(kind: str, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                       epochs: int, batch: int, seed: int) -> Tuple[np.ndarray, float]:
    import tensorflow as tf
    set_seed(seed)
    ym = float(np.mean(ytr))
    ys = float(np.std(ytr))
    if not np.isfinite(ys) or ys < 1e-8:
        ys = 1.0
    yz = ((ytr - ym) / ys).astype(np.float32)
    model = build_deep_model(kind, Xtr.shape[1], Xtr.shape[2])
    t0 = time.perf_counter()
    model.fit(Xtr, yz, epochs=int(epochs), batch_size=int(batch), shuffle=True, verbose=0)
    predz = model.predict(Xte, batch_size=max(256, int(batch)), verbose=0).reshape(-1)
    elapsed = time.perf_counter() - t0
    pred = predz.astype(float) * ys + ym
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return pred, float(elapsed)


def train_predict_rf(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, threads: int) -> Tuple[np.ndarray, float]:
    params = dict(RF_PARAMS)
    params["n_jobs"] = max(1, int(threads))
    model = RandomForestRegressor(**params)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    elapsed = time.perf_counter() - t0
    del model
    gc.collect()
    return np.asarray(pred, float), float(elapsed)


def train_predict_xgb(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, threads: int) -> Tuple[np.ndarray, float]:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise ImportError("未安装 xgboost") from exc
    params = dict(XGB_PARAMS)
    params["n_jobs"] = max(1, int(threads))
    model = XGBRegressor(**params)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr, verbose=False)
    pred = model.predict(Xte)
    elapsed = time.perf_counter() - t0
    del model
    gc.collect()
    return np.asarray(pred, float), float(elapsed)


# =============================================================================
# 6. V4.7 baseline reuse
# =============================================================================


def load_v47_top2(path: Path, models: Sequence[str], horizons: Sequence[int], batteries: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path, compression="infer", low_memory=False)
    req = {*KEY, "SOH_True", "HI_Set", *[f"Pred_{m}" for m in models]}
    missing = sorted(req - set(d.columns))
    if missing:
        raise RuntimeError("V4.7 OOF缺列：" + str(missing))
    d["Battery"] = d["Battery"].map(normalize_battery)
    for c in ("Horizon", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[
        d["HI_Set"].astype(str).eq("Top2")
        & d["Horizon"].isin([int(x) for x in horizons])
        & d["Battery"].isin(list(batteries))
    ].copy()
    for c in ("Horizon", "OriginCycle_KeyOnly", "TargetCycle_KeyOnly"):
        d[c] = d[c].astype(int)
    d["SOH_True"] = pd.to_numeric(d["SOH_True"], errors="coerce")
    for m in models:
        d[f"Pred_{m}"] = pd.to_numeric(d[f"Pred_{m}"], errors="coerce")

    if "SelectedHI" in d.columns:
        bad = []
        expected = set(TOP2_HI)
        for s in d["SelectedHI"].dropna().astype(str).unique():
            if set(x for x in s.split("|") if x) != expected:
                bad.append(s)
        if bad:
            raise RuntimeError(
                "V4.7 Top2并非冻结的 FixedTime Top2，拒绝复用：" + str(bad[:5])
            )

    if d.duplicated(KEY).any():
        raise RuntimeError("V4.7 Top2 OOF存在重复key")

    J, L, role = JL_CONFIGS[BASELINE_CONFIG]
    d["JL_Config"] = BASELINE_CONFIG
    d["J"] = J
    d["L"] = L
    d["HistorySpan_cycles"] = (L - 1) * J
    d["DiagnosticCount"] = L
    d["HistoryScalarInputs"] = L * len(TOP2_COLUMNS)
    d["Source"] = "reused_V4.7_Top2"
    keep = [
        "JL_Config", "J", "L", "HistorySpan_cycles", "DiagnosticCount", "HistoryScalarInputs",
        *KEY, "SOH_True", *[f"Pred_{m}" for m in models], "Source"
    ]
    return d[keep].sort_values(KEY).reset_index(drop=True)


def audit_v47_against_bundle(v47: pd.DataFrame, bundle: WindowBundle, H: int,
                             batteries: Sequence[str]) -> Dict[str, object]:
    if v47.empty:
        return {"Horizon": int(H), "ReusedV47": 0, "PASS": 0, "Reason": "V4.7 OOF unavailable"}
    a = bundle_keys(bundle, H)
    a = a[a["Battery"].isin(list(batteries))].copy()
    b = v47[v47["Horizon"].eq(int(H))][KEY + ["SOH_True"]].copy()
    m = a.merge(b, on=KEY, how="outer", suffixes=("_rebuilt", "_v47"), indicator=True)
    both = m[m["_merge"].eq("both")].copy()
    maxdiff = float(np.nanmax(np.abs(both["SOH_True_rebuilt"] - both["SOH_True_v47"]))) if len(both) else np.nan
    rebuilt_only = int((m["_merge"] == "left_only").sum())
    v47_only = int((m["_merge"] == "right_only").sum())
    ok = rebuilt_only == 0 and v47_only == 0 and np.isfinite(maxdiff) and maxdiff <= 1e-12
    return {
        "Horizon": int(H),
        "ReusedV47": 1,
        "RebuiltRows": int(len(a)),
        "V47Rows": int(len(b)),
        "RebuiltOnly": rebuilt_only,
        "V47Only": v47_only,
        "MaxAbsSOHDiff": maxdiff,
        "PASS": int(ok),
        "Reason": "PASS" if ok else "KEY_OR_TARGET_MISMATCH",
    }


# =============================================================================
# 7. One fold / cache
# =============================================================================


def cache_ok(pred_path: Path, metric_path: Path, cfg: str, models: Sequence[str]) -> bool:
    if not pred_path.exists() or not metric_path.exists():
        return False
    try:
        p = pd.read_csv(pred_path, compression="infer", nrows=2)
        m = pd.read_csv(metric_path, nrows=20)
    except Exception:
        return False
    if "JL_Config" not in p.columns or not p["JL_Config"].astype(str).eq(cfg).all():
        return False
    if not all(f"Pred_{x}" in p.columns for x in models):
        return False
    return "Model" in m.columns and set(models).issubset(set(m["Model"].astype(str)))


def run_fold(bundle: WindowBundle, cfg: str, H: int, test_battery: str,
             models: Sequence[str], args: argparse.Namespace,
             pred_cache: Path, metric_cache: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not args.force_folds and cache_ok(pred_cache, metric_cache, cfg, models):
        return (
            pd.read_csv(pred_cache, compression="infer", low_memory=False),
            pd.read_csv(metric_cache, low_memory=False),
        )

    J, L, role = JL_CONFIGS[cfg]
    all_bats = sorted(set(bundle.battery.tolist()))
    source_bats = [b for b in all_bats if b != test_battery]
    test_idx = np.flatnonzero(bundle.battery == test_battery)
    if len(test_idx) == 0:
        return pd.DataFrame(), pd.DataFrame()

    deep_idx = balanced_indices(bundle.battery, source_bats, DEEP_PER_SOURCE_BATTERY)
    rf_idx = balanced_indices(bundle.battery, source_bats, RF_PER_SOURCE_BATTERY)
    xgb_idx = balanced_indices(bundle.battery, source_bats, XGB_PER_SOURCE_BATTERY)
    if len(deep_idx) == 0:
        raise RuntimeError(f"{cfg}|H={H}|{test_battery}: source训练窗口为空")

    scaler = fit_scaler(bundle.X[deep_idx])
    Xte = scaler.transform(bundle.X[test_idx])
    Xtflat = Xte.reshape(len(Xte), -1)

    pred_df = pd.DataFrame({
        "JL_Config": cfg,
        "J": J,
        "L": L,
        "HistorySpan_cycles": (L - 1) * J,
        "DiagnosticCount": L,
        "HistoryScalarInputs": L * len(TOP2_COLUMNS),
        "Horizon": int(H),
        "TestBattery": test_battery,
        "Battery": bundle.battery[test_idx],
        "OriginCycle_KeyOnly": bundle.origin_cycle[test_idx],
        "TargetCycle_KeyOnly": bundle.target_cycle[test_idx],
        "StrictSegmentKey_QCOnly": bundle.segment[test_idx],
        "SOH_True": bundle.y[test_idx],
        "Source": "trained_V4.8B",
    })

    metric_rows: List[Dict[str, object]] = []
    for mi, model in enumerate(models):
        if model == "RF":
            Xtr = scaler.transform(bundle.X[rf_idx]).reshape(len(rf_idx), -1)
            pred, runtime = train_predict_rf(Xtr, bundle.y[rf_idx], Xtflat, args.threads_model)
            ntrain = len(rf_idx)
            del Xtr
        elif model == "XGBoost":
            Xtr = scaler.transform(bundle.X[xgb_idx]).reshape(len(xgb_idx), -1)
            pred, runtime = train_predict_xgb(Xtr, bundle.y[xgb_idx], Xtflat, args.threads_model)
            ntrain = len(xgb_idx)
            del Xtr
        elif model in {"LSTM", "BiLSTM"}:
            Xtr = scaler.transform(bundle.X[deep_idx])
            seed = RANDOM_SEED + H * 1000 + J * 100 + L * 10 + mi
            pred, runtime = train_predict_deep(
                model, Xtr, bundle.y[deep_idx], Xte,
                args.deep_epochs, args.deep_batch, seed,
            )
            ntrain = len(deep_idx)
            del Xtr
        else:
            raise ValueError(model)

        pred_df[f"Pred_{model}"] = pred
        metric_rows.append({
            "JL_Config": cfg, "J": J, "L": L,
            "HistorySpan_cycles": (L - 1) * J,
            "DiagnosticCount": L,
            "HistoryScalarInputs": L * len(TOP2_COLUMNS),
            "Horizon": int(H), "TestBattery": test_battery,
            "Model": model,
            "TrainSourceBatteries": len(source_bats),
            "TrainWindows": int(ntrain), "TestWindows": int(len(test_idx)),
            "Runtime_s": float(runtime),
            **calc_metrics(bundle.y[test_idx], pred),
        })
        gc.collect()

    metric_df = pd.DataFrame(metric_rows)
    ensure_dir(pred_cache.parent)
    pred_df.to_csv(pred_cache, index=False, compression=FAST_GZIP)
    metric_df.to_csv(metric_cache, index=False, encoding="utf-8-sig")
    del Xte, Xtflat
    gc.collect()
    return pred_df, metric_df


# =============================================================================
# 8. Aggregation / common cohorts
# =============================================================================


def melt_fold_metrics_from_oof(oof: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    rows = []
    for (cfg, H, bat), g in oof.groupby(["JL_Config", "Horizon", "Battery"], sort=True):
        J = int(g["J"].iloc[0]); L = int(g["L"].iloc[0])
        for m in models:
            c = f"Pred_{m}"
            if c not in g.columns:
                continue
            rows.append({
                "JL_Config": cfg, "J": J, "L": L,
                "HistorySpan_cycles": (L - 1) * J,
                "DiagnosticCount": L,
                "Horizon": int(H), "Battery": bat, "Model": m,
                **calc_metrics(g["SOH_True"].to_numpy(float), g[c].to_numpy(float)),
            })
    return pd.DataFrame(rows)


def own_cohort_pooled(oof: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    rows = []
    for (cfg, H), g in oof.groupby(["JL_Config", "Horizon"], sort=True):
        J = int(g["J"].iloc[0]); L = int(g["L"].iloc[0])
        for m in models:
            c = f"Pred_{m}"
            if c not in g.columns:
                continue
            rows.append({
                "JL_Config": cfg, "J": J, "L": L,
                "HistorySpan_cycles": (L - 1) * J,
                "DiagnosticCount": L,
                "HistoryScalarInputs": L * len(TOP2_COLUMNS),
                "Horizon": int(H), "Model": m, "EvaluationCohort": "OWN",
                "Batteries": int(g["Battery"].nunique()),
                **calc_metrics(g["SOH_True"].to_numpy(float), g[c].to_numpy(float)),
            })
    return pd.DataFrame(rows)


def key_index(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df[KEY])


def subset_to_keys(df: pd.DataFrame, keys: pd.MultiIndex) -> pd.DataFrame:
    q = df.copy()
    q["_key"] = list(key_index(q))
    q = q[q["_key"].isin(set(keys.tolist()))].drop(columns="_key")
    return q.sort_values(KEY).reset_index(drop=True)


def common_keys_for_configs(oof: pd.DataFrame, H: int, configs: Sequence[str]) -> pd.MultiIndex:
    sets = []
    for cfg in configs:
        q = oof[oof["Horizon"].eq(int(H)) & oof["JL_Config"].eq(cfg)]
        if q.empty:
            return pd.MultiIndex.from_tuples([], names=KEY)
        sets.append(set(map(tuple, q[KEY].to_numpy(object))))
    common = set.intersection(*sets) if sets else set()
    if not common:
        return pd.MultiIndex.from_tuples([], names=KEY)
    return pd.MultiIndex.from_tuples(sorted(common), names=KEY)


def validate_truth_on_common(oof: pd.DataFrame, H: int, configs: Sequence[str], keys: pd.MultiIndex) -> float:
    if len(keys) == 0:
        return np.nan
    ref = None
    maxdiff = 0.0
    for cfg in configs:
        q = subset_to_keys(
            oof[oof["Horizon"].eq(int(H)) & oof["JL_Config"].eq(cfg)], keys
        )
        vals = q["SOH_True"].to_numpy(float)
        if ref is None:
            ref = vals
        else:
            if len(vals) != len(ref):
                return np.inf
            maxdiff = max(maxdiff, float(np.nanmax(np.abs(vals - ref))))
    return maxdiff


def pairwise_common_metrics(oof: pd.DataFrame, configs: Sequence[str], horizons: Sequence[int],
                            models: Sequence[str]) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        if cfg == BASELINE_CONFIG:
            continue
        for H in horizons:
            keys = common_keys_for_configs(oof, H, [BASELINE_CONFIG, cfg])
            if len(keys) == 0:
                rows.append({
                    "Horizon": int(H), "CompareConfig": cfg, "Model": "ALL",
                    "CommonWindows": 0, "CommonBatteries": 0,
                    "Baseline_RMSE": np.nan, "Config_RMSE": np.nan,
                    "RMSE_Improvement_vs_Baseline_pct": np.nan,
                    "TruthMaxAbsDiff": np.nan,
                })
                continue
            truthdiff = validate_truth_on_common(oof, H, [BASELINE_CONFIG, cfg], keys)
            b = subset_to_keys(oof[(oof["Horizon"].eq(H)) & (oof["JL_Config"].eq(BASELINE_CONFIG))], keys)
            q = subset_to_keys(oof[(oof["Horizon"].eq(H)) & (oof["JL_Config"].eq(cfg))], keys)
            for m in models:
                mb = calc_metrics(b["SOH_True"], b[f"Pred_{m}"])
                mq = calc_metrics(q["SOH_True"], q[f"Pred_{m}"])
                br = float(mb["RMSE"]); qr = float(mq["RMSE"])
                rows.append({
                    "Horizon": int(H), "CompareConfig": cfg,
                    "J": JL_CONFIGS[cfg][0], "L": JL_CONFIGS[cfg][1],
                    "HistorySpan_cycles": (JL_CONFIGS[cfg][1] - 1) * JL_CONFIGS[cfg][0],
                    "Model": m, "CommonWindows": int(len(keys)),
                    "CommonBatteries": int(b["Battery"].nunique()),
                    "Baseline_MAE": mb["MAE"], "Config_MAE": mq["MAE"],
                    "Baseline_RMSE": br, "Config_RMSE": qr,
                    "Baseline_R2": mb["R2"], "Config_R2": mq["R2"],
                    "RMSE_Improvement_vs_Baseline_pct": (br - qr) / br * 100.0 if br > 0 else np.nan,
                    "MAE_Improvement_vs_Baseline_pct": (float(mb["MAE"]) - float(mq["MAE"])) / float(mb["MAE"]) * 100.0 if float(mb["MAE"]) > 0 else np.nan,
                    "TruthMaxAbsDiff": truthdiff,
                })
    return pd.DataFrame(rows)


def primary_common_outputs(oof: pd.DataFrame, horizons: Sequence[int], models: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inv_rows = []
    metric_rows = []
    pred_parts = []
    available_primary = [c for c in PRIMARY_CONFIGS if c in set(oof["JL_Config"].unique())]

    for H in horizons:
        if set(PRIMARY_CONFIGS).issubset(set(available_primary)):
            cfgs = list(PRIMARY_CONFIGS)
        else:
            cfgs = [c for c in PRIMARY_CONFIGS if c in available_primary]
        if len(cfgs) < 2:
            continue
        keys = common_keys_for_configs(oof, H, cfgs)
        truthdiff = validate_truth_on_common(oof, H, cfgs, keys) if len(keys) else np.nan
        if len(keys):
            ref = subset_to_keys(oof[(oof["Horizon"].eq(H)) & (oof["JL_Config"].eq(cfgs[0]))], keys)
            bats = int(ref["Battery"].nunique())
        else:
            bats = 0
        inv_rows.append({
            "Horizon": int(H), "Configs": "|".join(cfgs),
            "CommonWindows": int(len(keys)), "CommonBatteries": bats,
            "TruthMaxAbsDiff": truthdiff,
            "PASS": int(len(keys) > 0 and np.isfinite(truthdiff) and truthdiff <= 1e-12),
        })
        if len(keys) == 0:
            continue
        for cfg in cfgs:
            q = subset_to_keys(oof[(oof["Horizon"].eq(H)) & (oof["JL_Config"].eq(cfg))], keys)
            pred_parts.append(q.assign(CommonGroup="PRIMARY_MATCHED_SPAN"))
            J, L, _ = JL_CONFIGS[cfg]
            for m in models:
                metric_rows.append({
                    "Horizon": int(H), "JL_Config": cfg,
                    "J": J, "L": L, "HistorySpan_cycles": (L - 1) * J,
                    "DiagnosticCount": L, "HistoryScalarInputs": L * len(TOP2_COLUMNS),
                    "Model": m, "CommonWindows": int(len(q)),
                    "CommonBatteries": int(q["Battery"].nunique()),
                    **calc_metrics(q["SOH_True"], q[f"Pred_{m}"]),
                })
    inv = pd.DataFrame(inv_rows)
    met = pd.DataFrame(metric_rows)
    pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return inv, met, pred


def primary_gain(primary_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if primary_metrics.empty:
        return pd.DataFrame()
    for (H, model), g in primary_metrics.groupby(["Horizon", "Model"], sort=True):
        b = g[g["JL_Config"].eq(BASELINE_CONFIG)]
        if b.empty:
            continue
        br = float(b.iloc[0]["RMSE"]); bm = float(b.iloc[0]["MAE"])
        for _, r in g.iterrows():
            rr = float(r["RMSE"]); rm = float(r["MAE"])
            rows.append({
                "Horizon": int(H), "Model": model, "JL_Config": r["JL_Config"],
                "J": int(r["J"]), "L": int(r["L"]),
                "HistorySpan_cycles": int(r["HistorySpan_cycles"]),
                "DiagnosticCount": int(r["DiagnosticCount"]),
                "HistoryScalarInputs": int(r["HistoryScalarInputs"]),
                "CommonWindows": int(r["CommonWindows"]),
                "RMSE": rr, "Baseline_RMSE": br,
                "RMSE_Improvement_vs_J05_L10_pct": (br - rr) / br * 100.0 if br > 0 else np.nan,
                "MAE": rm, "Baseline_MAE": bm,
                "MAE_Improvement_vs_J05_L10_pct": (bm - rm) / bm * 100.0 if bm > 0 else np.nan,
                "DiagnosticCountReduction_pct": (1.0 - float(r["DiagnosticCount"]) / 10.0) * 100.0,
                "HistoryScalarReduction_pct": (1.0 - float(r["HistoryScalarInputs"]) / 80.0) * 100.0,
                "NominalDiagnosticFrequencyReduction_pct": (1.0 - 5.0 / float(r["J"])) * 100.0,
            })
    return pd.DataFrame(rows)


def measurement_efficiency(gain: pd.DataFrame) -> pd.DataFrame:
    if gain.empty:
        return pd.DataFrame()
    q = gain.copy()
    q["Within5pctBaselineRMSE"] = (
        q["RMSE_Improvement_vs_J05_L10_pct"].fillna(-999) >= -5.0
    ).astype(int)
    q["Within10pctBaselineRMSE"] = (
        q["RMSE_Improvement_vs_J05_L10_pct"].fillna(-999) >= -10.0
    ).astype(int)
    # Largest diagnostic reduction first, then lowest RMSE.
    q = q.sort_values(
        ["Horizon", "Model", "Within5pctBaselineRMSE", "DiagnosticCountReduction_pct", "RMSE"],
        ascending=[True, True, False, False, True],
    )
    return q


# =============================================================================
# 9. Figures
# =============================================================================


def make_figures(primary: pd.DataFrame, gain: pd.DataFrame, out: Path) -> None:
    if primary.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    figdir = ensure_dir(out / "Figures")
    for model in sorted(primary["Model"].unique()):
        g = primary[primary["Model"].eq(model)].copy()
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for cfg, q in g.groupby("JL_Config", sort=False):
            ax.plot(q["Horizon"], q["RMSE"], marker="o", label=cfg)
        ax.set_xlabel("Forecast horizon H (cycles)")
        ax.set_ylabel("RMSE")
        ax.set_title(f"Primary matched-span J/L ablation - {model}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / f"01_primary_RMSE_{model}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 10. Main
# =============================================================================


def main() -> None:
    args = parse_args()
    t_all = time.perf_counter()

    horizons = sorted(set(int(x) for x in args.horizons))
    configs = list(dict.fromkeys(args.configs))
    models = list(dict.fromkeys(args.models))
    if any(h < 1 for h in horizons):
        raise ValueError("horizons必须>=1")
    if not configs or not models:
        raise ValueError("configs/models不能为空")

    out = ensure_dir(args.output_root)
    cache_root = ensure_dir(out / "Cache_Folds")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(args.threads_model))))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(max(1, int(args.threads_model))))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    set_seed(RANDOM_SEED)
    tf_info = configure_tensorflow_if_needed(models)

    config_json = {
        "Task": "RWTH48 V4.8B FixedTime Top2 J/L sparse-diagnostic fair ablation",
        "StrictInput": str(args.strict_input),
        "HIInput": str(args.hi_input),
        "V47OOF": str(args.v47_oof),
        "Output": str(out),
        "Checkpoints_s": list(CHECKPOINTS_S),
        "CheckpointDefinition": "cumulative 0-T prefixes",
        "FixedTop2HI": list(TOP2_HI),
        "Top2ColumnsPerObservation": len(TOP2_COLUMNS),
        "JLConfigs": {k: {"J": v[0], "L": v[1], "HistorySpan": (v[1]-1)*v[0], "Role": v[2]} for k, v in JL_CONFIGS.items()},
        "ConfigsRequested": configs,
        "PrimaryMatchedSpanConfigs": list(PRIMARY_CONFIGS),
        "Horizons": horizons,
        "Models": models,
        "OnlyHIAsModelInput": True,
        "CycleAsFeature": False,
        "HistoricalSOHAsFeature": False,
        "PeriodicSOHCalibration": False,
        "FutureHI": False,
        "NearestCycleSubstitution": False,
        "V47BaselineReusePreferred": not args.retrain_baseline,
        "TrainingCapsPerSourceBattery": {"RF": RF_PER_SOURCE_BATTERY, "XGBoost": XGB_PER_SOURCE_BATTERY, "LSTM": DEEP_PER_SOURCE_BATTERY, "BiLSTM": DEEP_PER_SOURCE_BATTERY},
        "DeepEpochs": int(args.deep_epochs),
        "DeepBatch": int(args.deep_batch),
        "TensorFlow": tf_info,
    }
    save_json(config_json, out / "00_config.json")

    print("=" * 118)
    print("RWTH48 V4.8B | FixedTime Top2 × J/L sparse-diagnostic fair ablation")
    print("STRICT    :", args.strict_input)
    print("HI        :", args.hi_input)
    print("V4.7 OOF  :", args.v47_oof)
    print("OUTPUT    :", out)
    print("TOP2      :", TOP2_HI)
    print("CONFIGS   :", configs)
    print("HORIZONS  :", horizons)
    print("MODELS    :", models)
    print("TF/GPU    :", tf_info)
    print("=" * 118, flush=True)

    print("[A] Loading frozen V4.5C-FIX + V4.6A Top2 inputs ...", flush=True)
    strict = load_strict_target(args.strict_input)
    hi = load_prefix_hi(args.hi_input)
    compat = validate_upstream(args.v46a_complete, strict, hi)
    compat.to_csv(out / "01_input_compatibility_audit.csv", index=False, encoding="utf-8-sig")
    if not bool((compat["PASS"].astype(int) == 1).all()):
        raise RuntimeError("上游兼容性审计失败：\n" + compat.to_string(index=False))

    merged = strict[["Battery", "GlobalCycle", SEGMENT, TARGET]].merge(
        hi[["Battery", "GlobalCycle", *TOP2_COLUMNS]],
        on=["Battery", "GlobalCycle"], how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise RuntimeError("strict target 与 Top2 HI 无交集")

    batteries = sorted(strict["Battery"].unique())
    if args.only_battery:
        b = normalize_battery(args.only_battery)
        if b not in batteries:
            raise ValueError(f"--only-battery不存在：{b}")
        outer_bats = [b]
    else:
        outer_bats = batteries

    print("[B] Building exact J/L windows + availability audit ...", flush=True)
    bundles: Dict[Tuple[str, int], WindowBundle] = {}
    inv_rows = []
    for cfg in configs:
        J, L, role = JL_CONFIGS[cfg]
        for H in horizons:
            bun = build_sparse_history_windows(merged, J, L, H)
            bundles[(cfg, H)] = bun
            vc = pd.Series(bun.battery).value_counts() if len(bun.y) else pd.Series(dtype=int)
            inv_rows.append({
                "JL_Config": cfg, "Role": role, "J": J, "L": L,
                "HistorySpan_cycles": (L - 1) * J,
                "DiagnosticCount": L,
                "HistoryScalarInputs": L * len(TOP2_COLUMNS),
                "Horizon": int(H), "TotalWindows": int(len(bun.y)),
                "BatteriesWithWindows": int(len(vc)),
                "MinWindowsPerBattery": int(vc.min()) if len(vc) else 0,
                "MedianWindowsPerBattery": float(vc.median()) if len(vc) else 0.0,
                "MaxWindowsPerBattery": int(vc.max()) if len(vc) else 0,
                "Status": "AVAILABLE" if len(bun.y) else "NO_WINDOWS",
            })
    availability = pd.DataFrame(inv_rows)
    availability.to_csv(out / "02_JL_availability_inventory.csv", index=False, encoding="utf-8-sig")

    # Optional frozen V4.7 baseline reuse and exact audit.
    v47 = pd.DataFrame()
    v47_audit_rows = []
    can_reuse_v47 = (BASELINE_CONFIG in configs) and (not args.retrain_baseline) and args.v47_oof.exists()
    if can_reuse_v47:
        v47 = load_v47_top2(args.v47_oof, models, horizons, outer_bats)
        for H in horizons:
            bun = bundles.get((BASELINE_CONFIG, H))
            if bun is None or len(bun.y) == 0:
                continue
            v47_audit_rows.append(audit_v47_against_bundle(v47, bun, H, outer_bats))
        v47_audit = pd.DataFrame(v47_audit_rows)
        v47_audit.to_csv(out / "02b_V47_Top2_exact_baseline_audit.csv", index=False, encoding="utf-8-sig")
        if len(v47_audit) and not bool((v47_audit["PASS"].astype(int) == 1).all()):
            raise RuntimeError("V4.7 Top2 baseline与重建 J=5,L=10 cohort不一致：\n" + v47_audit.to_string(index=False))
        print("    V4.7 Top2 baseline: exact PASS, will reuse.", flush=True)
    else:
        print("    V4.7 Top2 baseline: unavailable/retrain requested; baseline will be trained.", flush=True)

    # Pre-training primary cohort inventory from exact windows only.
    key_frames = {}
    primary_inv_pre = []
    for cfg in configs:
        for H in horizons:
            bun = bundles[(cfg, H)]
            key_frames[(cfg, H)] = bundle_keys(bun, H) if len(bun.y) else pd.DataFrame(columns=KEY+["SOH_True"])
    if set(PRIMARY_CONFIGS).issubset(set(configs)):
        for H in horizons:
            sets = []
            for cfg in PRIMARY_CONFIGS:
                q = key_frames[(cfg, H)]
                sets.append(set(map(tuple, q[KEY].to_numpy(object))))
            common = set.intersection(*sets) if sets else set()
            cbats = len(set(x[1] for x in common)) if common else 0
            primary_inv_pre.append({
                "Horizon": int(H), "Configs": "|".join(PRIMARY_CONFIGS),
                "CommonWindows_PreTraining": int(len(common)),
                "CommonBatteries_PreTraining": int(cbats),
            })
    pd.DataFrame(primary_inv_pre).to_csv(out / "03_primary_common_cohort_inventory.csv", index=False, encoding="utf-8-sig")

    if args.audit_only:
        print("\n--audit-only PASS: inputs and J/L availability audit completed.")
        print(availability.to_string(index=False))
        if primary_inv_pre:
            print("\nPrimary matched-span exact common cohort:")
            print(pd.DataFrame(primary_inv_pre).to_string(index=False))
        return

    print("[C] LOBO training / V4.7 baseline reuse ...", flush=True)
    pred_parts: List[pd.DataFrame] = []
    runtime_rows: List[Dict[str, object]] = []

    if can_reuse_v47 and not v47.empty:
        pred_parts.append(v47)

    for cfg in configs:
        J, L, role = JL_CONFIGS[cfg]
        for H in horizons:
            bun = bundles[(cfg, H)]
            if len(bun.y) == 0:
                print(f"  {cfg} | H={H}: NO WINDOWS -> skip", flush=True)
                continue
            if cfg == BASELINE_CONFIG and can_reuse_v47:
                print(f"  {cfg} | H={H}: reused V4.7 Top2 OOF", flush=True)
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
                    f"fold {fi:02d}/{len(outer_bats):02d} | held-out={test_bat}",
                    flush=True,
                )
                t0 = time.perf_counter()
                p, m = run_fold(bun, cfg, H, test_bat, models, args, pc, mc)
                if not p.empty:
                    pred_parts.append(p)
                    runtime_rows.append({
                        "JL_Config": cfg, "J": J, "L": L, "Horizon": H,
                        "TestBattery": test_bat, "FoldWall_s": float(time.perf_counter()-t0),
                        "PredictionCache": str(pc),
                    })
                    msg = " | ".join(f"{r.Model}:RMSE={float(r.RMSE):.6f}" for r in m.itertuples(index=False))
                    print("      " + msg, flush=True)

    if not pred_parts:
        raise RuntimeError("没有任何OOF预测")
    oof = pd.concat(pred_parts, ignore_index=True, sort=False)
    oof = oof[oof["JL_Config"].isin(configs) & oof["Horizon"].isin(horizons)].copy()
    oof["Battery"] = oof["Battery"].map(normalize_battery)
    oof = oof.sort_values(["JL_Config", "Horizon", "Battery", "OriginCycle_KeyOnly"]).reset_index(drop=True)

    # Duplicate integrity.
    if oof.duplicated(["JL_Config", *KEY]).any():
        dup = oof[oof.duplicated(["JL_Config", *KEY], keep=False)].head(20)
        raise RuntimeError("OOF存在重复 JL_Config + key：\n" + dup.to_string(index=False))

    print("[D] Common-cohort aggregation ...", flush=True)
    fold_metrics = melt_fold_metrics_from_oof(oof, models)
    own = own_cohort_pooled(oof, models)
    pairwise = pairwise_common_metrics(oof, configs, horizons, models) if BASELINE_CONFIG in configs else pd.DataFrame()
    primary_inv, primary_met, primary_pred = primary_common_outputs(oof, horizons, models)
    gain = primary_gain(primary_met)
    efficiency = measurement_efficiency(gain)
    runtime = pd.DataFrame(runtime_rows)

    fold_metrics.to_csv(out / "04_fold_model_metrics.csv", index=False, encoding="utf-8-sig")
    own.to_csv(out / "05_own_cohort_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(out / "06_pairwise_common_cohort_metrics.csv", index=False, encoding="utf-8-sig")
    primary_met.to_csv(out / "07_primary_matched_span_common_metrics.csv", index=False, encoding="utf-8-sig")
    gain.to_csv(out / "08_primary_gain_vs_J05_L10.csv", index=False, encoding="utf-8-sig")
    efficiency.to_csv(out / "09_measurement_efficiency_summary.csv", index=False, encoding="utf-8-sig")
    runtime.to_csv(out / "10_runtime_summary.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(out / "11_ALL_OOF_JL_ablation.csv.gz", index=False, compression=FAST_GZIP)
    if not primary_pred.empty:
        primary_pred.to_csv(out / "13_primary_common_predictions.csv.gz", index=False, compression=FAST_GZIP)

    # Acceptance gate.
    gate_rows: List[Dict[str, object]] = []
    def gate(check: str, ok: bool, detail: object) -> None:
        gate_rows.append({"Check": check, "PASS": int(bool(ok)), "Detail": str(detail)})

    gate("Input compatibility", bool((compat["PASS"].astype(int) == 1).all()), "see 01_input_compatibility_audit.csv")
    gate("47 strict batteries", len(batteries) == 47, len(batteries))
    gate("Top2 fixed", set(TOP2_HI) == {"V_TimeWeightedMean", "I_Slope_A_per_s"}, TOP2_HI)
    gate("No duplicate config OOF keys", not oof.duplicated(["JL_Config", *KEY]).any(), len(oof))
    finite_ok = True; detail = []
    for m in models:
        if f"Pred_{m}" not in oof.columns:
            finite_ok = False; detail.append(f"{m}:missing_column")
        else:
            bad = int((~np.isfinite(pd.to_numeric(oof[f"Pred_{m}"], errors="coerce").to_numpy(float))).sum())
            finite_ok = finite_ok and bad == 0
            detail.append(f"{m}:bad={bad}")
    gate("All requested predictions finite", finite_ok, "; ".join(detail))

    if set(PRIMARY_CONFIGS).issubset(set(configs)):
        primary_ok = len(primary_inv) == len(horizons) and bool((primary_inv["PASS"].astype(int) == 1).all())
        gate("Primary matched-span exact common cohort", primary_ok, primary_inv.to_dict("records"))
    else:
        gate("Primary matched-span exact common cohort", True, "not all primary configs requested")

    if can_reuse_v47 and len(v47_audit_rows):
        va = pd.DataFrame(v47_audit_rows)
        gate("Reused V4.7 Top2 exact compatibility", bool((va["PASS"].astype(int) == 1).all()), va.to_dict("records"))

    if not args.only_battery and BASELINE_CONFIG in set(oof["JL_Config"]):
        bcount = int(oof[oof["JL_Config"].eq(BASELINE_CONFIG)]["Battery"].nunique())
        gate("Formal baseline covers 47 held-out batteries", bcount == 47, bcount)
    else:
        gate("Smoke/formal fold scope", True, sorted(oof["Battery"].unique())[:10])

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(out / "12_acceptance_gate.csv", index=False, encoding="utf-8-sig")
    gate_pass = bool((gate_df["PASS"].astype(int) == 1).all())

    if not args.skip_figures:
        make_figures(primary_met, gain, out)

    elapsed = float(time.perf_counter() - t_all)
    best_primary = []
    if not primary_met.empty:
        for (H, m), g in primary_met.groupby(["Horizon", "Model"], sort=True):
            r = g.sort_values(["RMSE", "DiagnosticCount"]).iloc[0]
            best_primary.append({
                "Horizon": int(H), "Model": str(m), "BestJL": str(r["JL_Config"]),
                "J": int(r["J"]), "L": int(r["L"]),
                "RMSE": float(r["RMSE"]), "MAE": float(r["MAE"]),
                "CommonWindows": int(r["CommonWindows"]),
            })

    complete = {
        "Task": "RWTH48 V4.8B FixedTime Top2 J/L sparse-diagnostic fair ablation",
        "Status": "PASS" if gate_pass else "CHECK",
        "Batteries": 47,
        "OuterTestBatteriesRun": len(outer_bats),
        "FixedTop2HI": list(TOP2_HI),
        "Checkpoints_s": list(CHECKPOINTS_S),
        "ConfigsRequested": configs,
        "PrimaryMatchedSpanConfigs": list(PRIMARY_CONFIGS),
        "Horizons": horizons,
        "Models": models,
        "V47BaselineReused": bool(can_reuse_v47),
        "OnlyHIAsModelInput": True,
        "CycleAsFeature": False,
        "HistoricalSOHAsFeature": False,
        "PeriodicSOHCalibration": False,
        "NearestCycleSubstitution": False,
        "BestPrimaryMatchedSpanByHorizonModel": best_primary,
        "OOFRows": int(len(oof)),
        "Elapsed_s": elapsed,
        "TensorFlow": tf_info,
        "Important": [
            "Only J and L change; FixedTime cumulative checkpoints and Top2 HI remain frozen.",
            "J/L are scheduling variables only and never enter model X.",
            "Primary comparison uses exact common Battery/Origin/Target test cohorts across J05_L10, J10_L05 and J20_L03.",
            "No nearest-cycle substitution is allowed.",
            "J10_L10 is a long-history stress test and may lose H=100 coverage due strict segment length.",
            "V4.7 J05_L10 Top2 OOF is reused only after exact key and SOH truth equivalence checks.",
        ],
    }
    save_json(complete, out / "99_RWTH48_V4_8B_COMPLETE.json")

    print("\n" + "=" * 118)
    print("RWTH48 V4.8B COMPLETE")
    print("Status      :", complete["Status"])
    print("Outer folds :", len(outer_bats))
    print("Configs     :", configs)
    print("Horizons    :", horizons)
    print("V4.7 reused :", can_reuse_v47)
    print("OOF rows    :", f"{len(oof):,}")
    print("Elapsed     :", f"{elapsed / 3600.0:.2f} h")
    print("Output      :", out)
    print("=" * 118)

    if not primary_met.empty:
        print("\nPrimary matched-span common-cohort RMSE:")
        print(primary_met[["Horizon", "Model", "JL_Config", "J", "L", "CommonWindows", "RMSE", "MAE", "R2"]]
              .sort_values(["Horizon", "Model", "RMSE"]).to_string(index=False))
    if not gate_pass:
        print("\nWARNING: acceptance gate contains CHECK; inspect 12_acceptance_gate.csv")


if __name__ == "__main__":
    main()

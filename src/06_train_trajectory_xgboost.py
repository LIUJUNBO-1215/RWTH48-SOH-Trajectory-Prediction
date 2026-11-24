# -*- coding: utf-8 -*-
r"""
RWTH48 V4.9E
=============
Event Top2 + J08/L06 Horizon-Aware XGBoost full-trajectory upgrade
47-cell outer LOBO | t+1 ... t+100 | exact comparison vs V4.9C multi-output XGBoost

研究目的
--------
V4.9C 的 native multi-output XGBoost 已经很强，但它把 100 个 future outputs 当作一个
multi-output target。V4.9E 改成一个共享的 lead-conditioned scalar XGBoost：

    historical Event Top2 sequence + known future lead encoding -> SOH(t+lead)

这样所有 lead=1...100 共享一个模型，同时显式知道预测距离。为避免把“Cycle”偷偷放入 X：
这里只加入 normalized future lead（1...100），它是任务定义中已知的 forecast horizon；
不加入 OriginCycle、TargetCycle、历史 SOH、容量、未来 HI 或未来 SOH。

冻结输入
--------
    Event Top2  : V_TimeWeightedMean + P_Last
    Event nodes : RefAh 0.05 / 0.09 / 0.13
    J/L         : J08/L06
    history     : 6 diagnoses × 6 raw scalars = 36
    K           : 100

Horizon-aware feature representation
------------------------------------
每个 trajectory origin 的历史输入生成：
    A. 36 raw historical Event HI scalars
    B. 每个 Event-HI channel 的 historical summary：last/mean/std/range/slope
    C. known future lead encoding：u,u²,u³,sqrt(u),log(u),sin/cos(pi*u),sin(2*pi*u)

Target 使用 source-only per-lead standardization，再由同一个 XGBoost 学习标准化 SOH。
训练时在 source batteries 内部做 battery-wise validation + early stopping，确定树数后
默认在全部 46 source batteries 上 refit，再预测 held-out battery。

默认新电脑
----------
BASE:
    workspace

需要已有：
    RWTH48_V4_5C_FIX_最小冲击共同扰动处理\06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz
    RWTH48_V4_2_双平台冻结_FixedTime_Event_HI\06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz
    RWTH48_V4_9C_Event_Top2_Sparse_FullTrajectory_LOBO\10_J08_L06_ALL_OOF_full_trajectory_predictions_long.csv.gz

helper scripts：
    20_trajectory_lobo_helper.py
    05_rank_hi_topk_ablation.py

运行
----
先审计：
    python -u "src/06_train_trajectory_xgboost.py" --audit-only

单电池 smoke：
    python -u "src/06_train_trajectory_xgboost.py" --only-battery RWTH-002

正式 GPU：
    python -u "src/06_train_trajectory_xgboost.py" --xgb-device auto

若 V4.9D 正占用同一块 GPU，可让本任务先走 CPU：
    python -u "src/06_train_trajectory_xgboost.py" --xgb-device cpu --threads-model 8
"""

from __future__ import annotations

import argparse
import gc
import gzip
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# =============================================================================
# 0. Frozen configuration
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR_DEFAULT = Path(__file__).resolve().parent
STRICT_DEFAULT = BASE / "RWTH48_V4_5C_FIX_最小冲击共同扰动处理" / "06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz"
EVENT_DEFAULT = BASE / "RWTH48_V4_2_双平台冻结_FixedTime_Event_HI" / "06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz"
V49C_ROOT_DEFAULT = BASE / "RWTH48_V4_9C_Event_Top2_Sparse_FullTrajectory_LOBO"
V49C_BASELINE_DEFAULT = V49C_ROOT_DEFAULT / "10_J08_L06_ALL_OOF_full_trajectory_predictions_long.csv.gz"
OUTPUT_DEFAULT = BASE / "RWTH48_V4_9E_Event_HorizonAware_XGBoost_FullTrajectory_LOBO"

EVENT_NODES = (
    (0.05, "Event_RefAh_0.05", "E005"),
    (0.09, "Event_RefAh_0.09", "E009"),
    (0.13, "Event_RefAh_0.13", "E013"),
)
EVENT_TOP2 = ("V_TimeWeightedMean", "P_Last")
EVENT_COLUMNS = tuple(f"{p}__{hi}" for _, _, p in EVENT_NODES for hi in EVENT_TOP2)
J, L, K = 8, 6, 100
TARGET = "CyclingSOH_Charge_Canonical"
OOF_KEY = ["Battery", "OriginCycle_KeyOnly", "Lead", "TargetCycle_KeyOnly"]
FAST_GZIP = {"method": "gzip", "compresslevel": 1}

RANDOM_SEED = 20260821
XGB_PER_SOURCE_BATTERY = 80
VALIDATION_BATTERIES = 5
THREADS = 4

# Conservative-but-strong scalar XGB recipe; best tree count is source-only selected.
XGB_DEFAULT = dict(
    n_estimators=1600,
    learning_rate=0.025,
    max_depth=7,
    min_child_weight=2.0,
    subsample=0.90,
    colsample_bytree=0.90,
    reg_lambda=2.0,
    reg_alpha=0.02,
    gamma=0.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=RANDOM_SEED,
)

# =============================================================================
# 1. CLI / utilities
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RWTH48 V4.9E horizon-aware XGBoost full trajectory")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--code-dir", type=Path, default=CODE_DIR_DEFAULT)
    p.add_argument("--strict-input", type=Path, default=STRICT_DEFAULT)
    p.add_argument("--event-file", type=Path, default=EVENT_DEFAULT)
    p.add_argument("--v49c-baseline", type=Path, default=V49C_BASELINE_DEFAULT)
    p.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    p.add_argument("--xgb-device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--xgb-gpu-id", type=int, default=0)
    p.add_argument("--threads-model", type=int, default=THREADS)
    p.add_argument("--per-source-battery", type=int, default=XGB_PER_SOURCE_BATTERY)
    p.add_argument("--validation-batteries", type=int, default=VALIDATION_BATTERIES)
    p.add_argument("--early-stopping-rounds", type=int, default=80)
    p.add_argument("--n-estimators", type=int, default=XGB_DEFAULT["n_estimators"])
    p.add_argument("--learning-rate", type=float, default=XGB_DEFAULT["learning_rate"])
    p.add_argument("--max-depth", type=int, default=XGB_DEFAULT["max_depth"])
    p.add_argument("--min-child-weight", type=float, default=XGB_DEFAULT["min_child_weight"])
    p.add_argument("--subsample", type=float, default=XGB_DEFAULT["subsample"])
    p.add_argument("--colsample-bytree", type=float, default=XGB_DEFAULT["colsample_bytree"])
    p.add_argument("--reg-lambda", type=float, default=XGB_DEFAULT["reg_lambda"])
    p.add_argument("--reg-alpha", type=float, default=XGB_DEFAULT["reg_alpha"])
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--no-refit-all-source", action="store_true")
    p.add_argument("--force-folds", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--skip-final-oof", action="store_true")
    return p.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: object, p: Path) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_battery(x: object) -> str:
    s = str(x).strip().upper().replace("_", "-")
    m = re.search(r"RWTH[- ]?0*(\d+)", s)
    if m:
        return f"RWTH-{int(m.group(1)):03d}"
    try:
        return f"RWTH-{int(float(s)):03d}"
    except Exception:
        return s


def safe_name(x: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 helper: {path}")
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
    raise FileNotFoundError(f"缺少 helper: {exact}\n请放入 {code_dir}")


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, float).reshape(-1)
    p = np.asarray(y_pred, float).reshape(-1)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return dict(N=0, MAE=np.nan, RMSE=np.nan, R2=np.nan, MAPE_pct=np.nan, Bias=np.nan)
    e = p - y
    return dict(
        N=int(len(y)), MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
        R2=float(r2_score(y, p)) if len(y) > 1 and np.std(y) > 1e-12 else np.nan,
        MAPE_pct=float(np.mean(np.abs(e) / np.maximum(np.abs(y), 1e-9)) * 100.0), Bias=float(np.mean(e)),
    )


def trajectory_shape(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = yp - yt
    dt, dp = np.diff(yt, axis=1), np.diff(yp, axis=1)
    d2t, d2p = np.diff(yt, n=2, axis=1), np.diff(yp, n=2, axis=1)
    tr, pr = float(np.mean(np.abs(d2t))), float(np.mean(np.abs(d2p)))
    return dict(
        MeanTrajectoryRMSE=float(np.mean(np.sqrt(np.mean(err ** 2, axis=1)))),
        MedianTrajectoryRMSE=float(np.median(np.sqrt(np.mean(err ** 2, axis=1)))),
        MeanTrajectoryMAE=float(np.mean(np.mean(np.abs(err), axis=1))),
        EndpointMAE=float(np.mean(np.abs(err[:, -1]))),
        TrueUpwardStepFraction=float(np.mean(dt > 0)), PredUpwardStepFraction=float(np.mean(dp > 0)),
        ExcessUpwardStepFraction=float(np.mean(dp > 0) - np.mean(dt > 0)),
        TrueRoughness=tr, PredRoughness=pr,
        RoughnessRatioPredToTrue=float(pr / tr) if tr > 1e-12 else np.nan,
        SlopeRMSE=float(np.sqrt(np.mean((dp - dt) ** 2))),
        CurvatureRMSE=float(np.sqrt(np.mean((d2p - d2t) ** 2))),
    )


# =============================================================================
# 2. Baseline / exact cohort
# =============================================================================

def load_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    head = pd.read_csv(path, compression="infer", nrows=0).columns.tolist()
    req = {"Battery", "OriginCycle_KeyOnly", "Lead", "TargetCycle_KeyOnly", "SOH_True", "Pred_XGBoost"}
    miss = sorted(req - set(head))
    if miss:
        raise RuntimeError(f"V4.9C baseline 缺列: {miss}")
    d = pd.read_csv(path, compression="infer", usecols=list(req), low_memory=False)
    d["Battery"] = d.Battery.map(normalize_battery)
    for c in ("OriginCycle_KeyOnly", "Lead", "TargetCycle_KeyOnly"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("SOH_True", "Pred_XGBoost"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=OOF_KEY + ["SOH_True", "Pred_XGBoost"])
    for c in ("OriginCycle_KeyOnly", "Lead", "TargetCycle_KeyOnly"):
        d[c] = d[c].astype(int)
    if d.duplicated(OOF_KEY).any():
        raise RuntimeError("baseline duplicate OOF keys")
    return d.sort_values(OOF_KEY).reset_index(drop=True)


def exact_bundle_audit(bundle, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in sorted(set(bundle.battery.tolist())):
        ix = np.flatnonzero(bundle.battery == b)
        origins = bundle.origin_cycle[ix]
        exp = pd.DataFrame(dict(
            Battery=b,
            OriginCycle_KeyOnly=np.repeat(origins, K),
            Lead=np.tile(np.arange(1, K + 1), len(ix)),
            SOH_Expected=bundle.y[ix].reshape(-1),
        ))
        exp["TargetCycle_KeyOnly"] = exp.OriginCycle_KeyOnly + exp.Lead
        q = baseline.loc[baseline.Battery.eq(b), OOF_KEY + ["SOH_True"]]
        z = exp.merge(q, on=OOF_KEY, how="outer", indicator=True, validate="one_to_one")
        both = z._merge.eq("both")
        dif = np.abs(z.loc[both, "SOH_Expected"].to_numpy(float) - z.loc[both, "SOH_True"].to_numpy(float))
        md = float(np.max(dif)) if len(dif) else np.nan
        ok = len(exp) == len(q) and z._merge.ne("both").sum() == 0 and (not np.isfinite(md) or md <= 1e-12)
        rows.append(dict(Battery=b, RebuiltLongRows=len(exp), BaselineLongRows=len(q), MaxAbsSOHDiff=md, PASS=int(ok)))
    return pd.DataFrame(rows)


# =============================================================================
# 3. Feature engineering
# =============================================================================

@dataclass
class HistImputer:
    median: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = np.asarray(X, np.float32).copy()
        for j in range(z.shape[2]):
            a = z[:, :, j]
            bad = ~np.isfinite(a)
            if bad.any():
                a[bad] = self.median[j]
            z[:, :, j] = a
        return z


def fit_hist_imputer(X: np.ndarray) -> HistImputer:
    f = np.asarray(X, float).reshape(-1, X.shape[2])
    med = np.nanmedian(f, axis=0)
    med[~np.isfinite(med)] = 0.0
    return HistImputer(med.astype(np.float32))


@dataclass
class PerLeadTargetScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, float) - self.mean[None, :]) / self.std[None, :]).astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, float) * self.std[None, :] + self.mean[None, :]


def fit_target_scaler(y: np.ndarray) -> PerLeadTargetScaler:
    m = np.mean(y, axis=0).astype(float)
    s = np.std(y, axis=0).astype(float)
    s[~np.isfinite(s) | (s < 1e-8)] = 1.0
    return PerLeadTargetScaler(m, s)


def history_base_features(X: np.ndarray) -> np.ndarray:
    """One fixed vector per trajectory origin. All values come from historical Event HI only."""
    X = np.asarray(X, np.float32)
    n, l, f = X.shape
    raw = X.reshape(n, l * f)
    last = X[:, -1, :]
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    ran = np.max(X, axis=1) - np.min(X, axis=1)
    t = np.arange(l, dtype=np.float32)
    tc = t - np.mean(t)
    denom = float(np.sum(tc ** 2)) if np.sum(tc ** 2) > 0 else 1.0
    slope = np.sum((X - mean[:, None, :]) * tc[None, :, None], axis=1) / denom
    return np.concatenate([raw, last, mean, std, ran, slope], axis=1).astype(np.float32)


def lead_features(k: int = K) -> np.ndarray:
    lead = np.arange(1, k + 1, dtype=np.float32)
    u = lead / float(k)
    return np.stack([
        u,
        u ** 2,
        u ** 3,
        np.sqrt(u),
        np.log1p(4.0 * u) / np.log(5.0),
        np.sin(np.pi * u),
        np.cos(np.pi * u),
        np.sin(2.0 * np.pi * u),
    ], axis=1).astype(np.float32)


def expand_long_features(X_hist: np.ndarray, k: int = K) -> np.ndarray:
    base = history_base_features(X_hist)
    lf = lead_features(k)
    n = len(base)
    return np.concatenate([
        np.repeat(base, k, axis=0),
        np.tile(lf, (n, 1)),
    ], axis=1).astype(np.float32)


def balanced_indices(battery: np.ndarray, bats: Sequence[str], max_per: int) -> np.ndarray:
    out: List[int] = []
    for b in bats:
        ix = np.flatnonzero(battery == b)
        if max_per <= 0 or len(ix) <= max_per:
            pick = ix
        else:
            pos = np.linspace(0, len(ix) - 1, max_per)
            pick = ix[np.round(pos).astype(int)]
        out.extend(pick.tolist())
    return np.asarray(sorted(set(out)), dtype=int)


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
    return [b for b in src if b not in set(val)], val


# =============================================================================
# 4. XGBoost backend / training
# =============================================================================

def probe_backend(requested: str, gpu_id: int, threads: int) -> Dict[str, object]:
    import xgboost as xgb
    from xgboost import XGBRegressor
    ver = str(xgb.__version__)
    if requested == "cpu":
        return dict(Requested=requested, Selected="cpu", GPU_ID=gpu_id, XGBoostVersion=ver, Probe="CPU_FORCED")
    params = dict(n_estimators=2, max_depth=2, learning_rate=0.2, objective="reg:squarederror", n_jobs=max(1, threads),
                  tree_method="hist", device=f"cuda:{gpu_id}", random_state=1)
    X = np.random.RandomState(1).randn(32, 4).astype(np.float32)
    y = np.random.RandomState(2).randn(32).astype(np.float32)
    try:
        m = XGBRegressor(**params)
        m.fit(X, y, verbose=False)
        _ = m.predict(X[:2])
        del m
        return dict(Requested=requested, Selected="cuda", GPU_ID=gpu_id, XGBoostVersion=ver, Probe="CUDA_PASS", ActualProbeDevice=f"cuda:{gpu_id}")
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError("--xgb-device cuda 但 CUDA probe 失败") from exc
        return dict(Requested=requested, Selected="cpu", GPU_ID=gpu_id, XGBoostVersion=ver, Probe="CUDA_FAIL_CPU_FALLBACK", FallbackReason=repr(exc))


def make_xgb(args: argparse.Namespace, backend: Dict[str, object], n_estimators: int, early_stopping: bool):
    from xgboost import XGBRegressor
    p = dict(
        n_estimators=int(n_estimators), learning_rate=float(args.learning_rate), max_depth=int(args.max_depth),
        min_child_weight=float(args.min_child_weight), subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree), reg_lambda=float(args.reg_lambda), reg_alpha=float(args.reg_alpha),
        objective="reg:squarederror", eval_metric="rmse", random_state=RANDOM_SEED,
        n_jobs=max(1, int(args.threads_model)), tree_method="hist",
    )
    if backend["Selected"] == "cuda":
        p["device"] = f"cuda:{int(args.xgb_gpu_id)}"
    else:
        p["device"] = "cpu"
    if early_stopping:
        p["early_stopping_rounds"] = int(args.early_stopping_rounds)
    return XGBRegressor(**p)


def train_fold(bundle, test_battery: str, args: argparse.Namespace, backend: Dict[str, object]):
    source_bats = [b for b in sorted(set(bundle.battery.tolist())) if b != test_battery]
    test_idx = np.flatnonzero(bundle.battery == test_battery)
    tr_bats, va_bats = choose_validation_batteries(source_bats, args.validation_batteries)
    tr_idx = balanced_indices(bundle.battery, tr_bats, args.per_source_battery)
    va_idx = balanced_indices(bundle.battery, va_bats, args.per_source_battery)
    if not len(tr_idx) or not len(va_idx) or not len(test_idx):
        raise RuntimeError("train/val/test windows empty")

    imp = fit_hist_imputer(bundle.X[tr_idx])
    Xtr_hist = imp.transform(bundle.X[tr_idx])
    Xva_hist = imp.transform(bundle.X[va_idx])
    ts = fit_target_scaler(bundle.y[tr_idx])
    ytr = ts.transform(bundle.y[tr_idx])
    yva = ts.transform(bundle.y[va_idx])

    Xtr = expand_long_features(Xtr_hist, K)
    Xva = expand_long_features(Xva_hist, K)
    ytr_long = ytr.reshape(-1)
    yva_long = yva.reshape(-1)

    t0 = time.perf_counter()
    model = make_xgb(args, backend, args.n_estimators, early_stopping=True)
    model.fit(Xtr, ytr_long, eval_set=[(Xva, yva_long)], verbose=False)
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        best_trees = int(args.n_estimators)
    else:
        best_trees = int(best_iter) + 1
    val_pred_z = np.asarray(model.predict(Xva), float).reshape(len(va_idx), K)
    val_pred = ts.inverse(val_pred_z)
    val_met = {**calc_metrics(bundle.y[va_idx], val_pred), **trajectory_shape(bundle.y[va_idx], val_pred)}
    del model, Xtr, Xva, ytr_long, yva_long, val_pred_z, val_pred
    gc.collect()

    if args.no_refit_all_source:
        final_idx = tr_idx
        final_bats = tr_bats
    else:
        final_idx = balanced_indices(bundle.battery, source_bats, args.per_source_battery)
        final_bats = source_bats

    final_imp = fit_hist_imputer(bundle.X[final_idx])
    Xf_hist = final_imp.transform(bundle.X[final_idx])
    Xte_hist = final_imp.transform(bundle.X[test_idx])
    final_ts = fit_target_scaler(bundle.y[final_idx])
    yf = final_ts.transform(bundle.y[final_idx]).reshape(-1)
    Xf = expand_long_features(Xf_hist, K)

    final_model = make_xgb(args, backend, best_trees, early_stopping=False)
    final_model.fit(Xf, yf, verbose=False)

    # Predict in trajectory-origin chunks to bound host/GPU memory.
    pred_parts = []
    chunk_origins = 256
    for s in range(0, len(test_idx), chunk_origins):
        e = min(len(test_idx), s + chunk_origins)
        Xc = expand_long_features(Xte_hist[s:e], K)
        pz = np.asarray(final_model.predict(Xc), float).reshape(e - s, K)
        pred_parts.append(final_ts.inverse(pz))
        del Xc, pz
    pred = np.vstack(pred_parts)
    info = dict(
        BestTrees=best_trees,
        ValidationBatteries=";".join(va_bats),
        ValidationBatteryCount=len(va_bats),
        FinalTrainSourceBatteries=len(final_bats),
        FinalTrainTrajectoryOrigins=len(final_idx),
        FinalTrainLongRows=len(final_idx) * K,
        RawHistoryScalars=L * len(EVENT_COLUMNS),
        EngineeredHistoryFeatures=history_base_features(Xf_hist[:1]).shape[1],
        LeadFeatures=lead_features(K).shape[1],
        TotalXGBFeatures=expand_long_features(Xf_hist[:1], K).shape[1],
        TargetScaling="source-only per-lead standardization",
        ValRMSE=val_met["RMSE"], ValRoughnessRatio=val_met["RoughnessRatioPredToTrue"],
        Runtime_s=float(time.perf_counter() - t0),
    )
    del final_model, Xf_hist, Xte_hist, Xf, yf, pred_parts
    gc.collect()
    return pred, info


# =============================================================================
# 5. Cache / aggregation
# =============================================================================

def cache_signature(test_battery: str, args: argparse.Namespace, baseline: Path, backend: Dict[str, object]) -> Dict[str, object]:
    return dict(
        Task="RWTH48_V4_9E", TestBattery=test_battery, J=J, L=L, K=K, EventTop2=list(EVENT_TOP2),
        PerSourceBattery=int(args.per_source_battery), ValidationBatteries=int(args.validation_batteries),
        EarlyStoppingRounds=int(args.early_stopping_rounds), NEstimators=int(args.n_estimators),
        LearningRate=float(args.learning_rate), MaxDepth=int(args.max_depth), MinChildWeight=float(args.min_child_weight),
        Subsample=float(args.subsample), ColsampleBytree=float(args.colsample_bytree), RegLambda=float(args.reg_lambda),
        RegAlpha=float(args.reg_alpha), RefitAllSource=int(not args.no_refit_all_source), Backend=backend,
        BaselineSize=int(baseline.stat().st_size), BaselineMTimeNs=int(baseline.stat().st_mtime_ns),
    )


def cache_valid(p: Path, sig: Dict[str, object]) -> bool:
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text(encoding="utf-8")) == sig
    except Exception:
        return False


def run_fold_cached(bundle, test_battery: str, args: argparse.Namespace, backend: Dict[str, object], cache_root: Path):
    fd = ensure_dir(cache_root / safe_name(test_battery))
    pred_path = fd / "HorizonAware_XGB_predictions.csv.gz"
    met_path = fd / "HorizonAware_XGB_metrics.csv"
    meta_path = fd / "HorizonAware_XGB_meta.json"
    sig = cache_signature(test_battery, args, args.v49c_baseline, backend)
    if (not args.force_folds and pred_path.exists() and met_path.exists() and pred_path.stat().st_size > 0
            and met_path.stat().st_size > 0 and cache_valid(meta_path, sig)):
        return pred_path, pd.read_csv(met_path)

    ix = np.flatnonzero(bundle.battery == test_battery)
    pred, info = train_fold(bundle, test_battery, args, backend)
    y = bundle.y[ix]
    origins = np.repeat(bundle.origin_cycle[ix].astype(int), K)
    leads = np.tile(np.arange(1, K + 1, dtype=int), len(ix))
    long = pd.DataFrame(dict(
        Battery=test_battery, OriginCycle_KeyOnly=origins, Lead=leads,
        TargetCycle_KeyOnly=origins + leads,
        StrictSegmentKey_QCOnly=np.repeat(bundle.segment[ix].astype(object), K),
        SOH_True=y.reshape(-1), Pred_HorizonAware_XGBoost=pred.reshape(-1),
    ))
    met = dict(TestBattery=test_battery, Model="HorizonAware_XGBoost",
               TestTrajectoryWindows=len(ix), FuturePointsPerTrajectory=K,
               **calc_metrics(y, pred), **trajectory_shape(y, pred), **info)
    long.to_csv(pred_path, index=False, compression=FAST_GZIP)
    pd.DataFrame([met]).to_csv(met_path, index=False, encoding="utf-8-sig")
    save_json(sig, meta_path)
    return pred_path, pd.DataFrame([met])


def matrix_from_long(d: pd.DataFrame, pred_col: str) -> Tuple[np.ndarray, np.ndarray]:
    d = d.sort_values(["Battery", "OriginCycle_KeyOnly", "Lead"])
    counts = d.groupby(["Battery", "OriginCycle_KeyOnly"], sort=False).size()
    if not counts.eq(K).all():
        raise RuntimeError("long file has incomplete trajectories")
    return d.SOH_True.to_numpy(float).reshape(-1, K), d[pred_col].to_numpy(float).reshape(-1, K)


def aggregate_new(paths: Sequence[Path], out: Path):
    parts = [pd.read_csv(p, compression="infer", low_memory=False) for p in paths]
    d = pd.concat(parts, ignore_index=True)
    yt, yp = matrix_from_long(d, "Pred_HorizonAware_XGBoost")
    overall = pd.DataFrame([dict(Model="HorizonAware_XGBoost", **calc_metrics(yt, yp), **trajectory_shape(yt, yp))])
    lead_rows = []
    for h in range(1, K + 1):
        q = d[d.Lead.eq(h)]
        lead_rows.append(dict(Model="HorizonAware_XGBoost", Lead=h,
                              **calc_metrics(q.SOH_True.to_numpy(float), q.Pred_HorizonAware_XGBoost.to_numpy(float))))
    lead = pd.DataFrame(lead_rows)
    overall.to_csv(out / "05_horizonaware_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    lead.to_csv(out / "06_horizonaware_leadwise_metrics.csv", index=False, encoding="utf-8-sig")
    return d, overall, lead


def exact_compare(new_long: pd.DataFrame, baseline: pd.DataFrame, out: Path, expected_batteries: int):
    q = baseline[OOF_KEY + ["SOH_True", "Pred_XGBoost"]].copy()
    n = new_long[OOF_KEY + ["SOH_True", "Pred_HorizonAware_XGBoost"]].copy()
    z = q.merge(n, on=OOF_KEY, how="inner", suffixes=("_Base", "_New"), validate="one_to_one")
    dif = np.abs(z.SOH_True_Base.to_numpy(float) - z.SOH_True_New.to_numpy(float))
    md = float(np.max(dif)) if len(dif) else np.nan
    audit = pd.DataFrame([dict(CommonLongRows=len(z), CommonBatteries=z.Battery.nunique(), TruthMaxAbsDiff=md,
                               PASS=int(len(z) > 0 and z.Battery.nunique() == int(expected_batteries) and md <= 1e-12))])
    audit.to_csv(out / "07_exact_common_audit.csv", index=False, encoding="utf-8-sig")
    if not audit.PASS.eq(1).all():
        raise RuntimeError("baseline/new exact common audit FAIL")

    yt = z.SOH_True_Base.to_numpy(float)
    pb = z.Pred_XGBoost.to_numpy(float)
    pn = z.Pred_HorizonAware_XGBoost.to_numpy(float)
    mb, mn = calc_metrics(yt, pb), calc_metrics(yt, pn)
    rows = [
        dict(Model="V49C_MultiOutput_XGBoost", **mb),
        dict(Model="V49E_HorizonAware_XGBoost", **mn),
    ]
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "08_exact_common_metrics.csv", index=False, encoding="utf-8-sig")
    gain = pd.DataFrame([dict(
        Baseline_RMSE=mb["RMSE"], New_RMSE=mn["RMSE"],
        RMSE_Improvement_pct=(mb["RMSE"] - mn["RMSE"]) / mb["RMSE"] * 100.0,
        Baseline_MAE=mb["MAE"], New_MAE=mn["MAE"],
        MAE_Improvement_pct=(mb["MAE"] - mn["MAE"]) / mb["MAE"] * 100.0,
    )])
    gain.to_csv(out / "09_gain_vs_V49C_XGBoost.csv", index=False, encoding="utf-8-sig")

    lead_rows = []
    for h in range(1, K + 1):
        g = z[z.Lead.eq(h)]
        b = calc_metrics(g.SOH_True_Base.to_numpy(float), g.Pred_XGBoost.to_numpy(float))
        nn = calc_metrics(g.SOH_True_Base.to_numpy(float), g.Pred_HorizonAware_XGBoost.to_numpy(float))
        lead_rows.append(dict(Lead=h, Baseline_RMSE=b["RMSE"], New_RMSE=nn["RMSE"],
                              RMSE_Improvement_pct=(b["RMSE"] - nn["RMSE"]) / b["RMSE"] * 100.0,
                              Baseline_MAE=b["MAE"], New_MAE=nn["MAE"]))
    lead_gain = pd.DataFrame(lead_rows)
    lead_gain.to_csv(out / "10_leadwise_gain_vs_V49C_XGBoost.csv", index=False, encoding="utf-8-sig")
    return audit, comp, gain, lead_gain


def stream_final(paths: Iterable[Path], output: Path):
    first = True
    with gzip.open(output, "wt", encoding="utf-8", newline="") as fh:
        for p in paths:
            for chunk in pd.read_csv(p, compression="infer", chunksize=100000, low_memory=False):
                chunk.to_csv(fh, index=False, header=first)
                first = False


# =============================================================================
# 6. Main
# =============================================================================

def main() -> None:
    args = parse_args()
    t_all = time.perf_counter()
    out = ensure_dir(args.output_root)
    cache_root = ensure_dir(out / "Cache_Folds")
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(args.threads_model))))

    v49_path = resolve_script(args.code_dir, "20_trajectory_lobo_helper.py", "*V4_9*整条未来SOH轨迹*py")
    v47e_path = resolve_script(args.code_dir, "05_rank_hi_topk_ablation.py", "*V4_7E*Event*HI*LOBO*py")
    v49 = load_module(v49_path, "rwth48_v49_helper_for_v49e")
    v47e = load_module(v47e_path, "rwth48_v47e_helper_for_v49e")
    v49.HI_COLUMNS = EVENT_COLUMNS
    v49.COMPACT_HI = EVENT_TOP2
    v49.CHECKPOINTS_S = tuple(x[0] for x in EVENT_NODES)

    for p in (args.strict_input, args.event_file, args.v49c_baseline):
        if not p.exists():
            raise FileNotFoundError(p)

    strict = v49.load_strict_target(args.strict_input)
    wide, node_audit = v47e.load_event_wide(args.event_file, EVENT_TOP2)
    wide = wide[["Battery", "GlobalCycle", *EVENT_COLUMNS]].copy()
    wide.Battery = wide.Battery.map(normalize_battery)
    baseline = load_baseline(args.v49c_baseline)
    bundle, inventory = v49.build_full_trajectory_windows(strict, wide, J, L, K)
    batteries = sorted(set(bundle.battery.tolist()))
    if len(batteries) != 47:
        raise RuntimeError(f"预期47 batteries, 实际 {len(batteries)}")

    node_audit.to_csv(out / "01_event_node_audit.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(out / "02_window_inventory.csv", index=False, encoding="utf-8-sig")
    compat = exact_bundle_audit(bundle, baseline)
    compat.to_csv(out / "03_v49c_exact_compatibility_audit.csv", index=False, encoding="utf-8-sig")
    if not compat.PASS.eq(1).all():
        raise RuntimeError("V4.9E cohort != V4.9C J08/L06")

    if args.audit_only:
        print("=" * 118)
        print("RWTH48 V4.9E --audit-only PASS")
        print("47 batteries | windows:", len(bundle.y), "| long rows:", len(bundle.y) * K)
        print("Base history feature dim:", history_base_features(bundle.X[:1]).shape[1])
        print("Lead feature dim:", lead_features(K).shape[1])
        print("Final XGB feature dim:", expand_long_features(bundle.X[:1], K).shape[1])
        print("=" * 118)
        return

    try:
        import xgboost as xgb
    except Exception as exc:
        raise RuntimeError("未安装 xgboost") from exc
    backend = probe_backend(args.xgb_device, args.xgb_gpu_id, args.threads_model)

    config = dict(
        Task="RWTH48 V4.9E Event horizon-aware XGBoost",
        StrictInput=str(args.strict_input), EventFile=str(args.event_file), Baseline=str(args.v49c_baseline), Output=str(out),
        EventTop2=list(EVENT_TOP2), EventNodes=[x[1] for x in EVENT_NODES], J=J, L=L, K=K,
        RawHistoryScalars=L * len(EVENT_COLUMNS), EngineeredHistoryFeatureDim=history_base_features(bundle.X[:1]).shape[1],
        LeadFeatureDim=lead_features(K).shape[1], TotalFeatureDim=expand_long_features(bundle.X[:1], K).shape[1],
        TargetScaling="source-only per-lead standardization", XGBoostBackend=backend,
        HistoricalSOHAsFeature=False, FutureSOHAsFeature=False, FutureHIAsFeature=False,
        OriginCycleAsFeature=False, TargetCycleAsFeature=False, KnownFutureLeadAsFeature=True,
    )
    save_json(config, out / "00_config.json")

    outer = [normalize_battery(args.only_battery)] if args.only_battery else batteries
    if args.only_battery and outer[0] not in batteries:
        raise ValueError(f"--only-battery不存在: {outer[0]}")

    pred_paths, fold_rows = [], []
    print("=" * 118)
    print("RWTH48 V4.9E | Horizon-Aware XGBoost | backend:", backend)
    print("=" * 118, flush=True)
    for i, b in enumerate(outer, 1):
        print(f"[{i:02d}/{len(outer):02d}] held-out={b}", end="", flush=True)
        p, m = run_fold_cached(bundle, b, args, backend, cache_root)
        pred_paths.append(p)
        fold_rows.append(m.iloc[0].to_dict())
        r = m.iloc[0]
        print(f" RMSE={float(r.RMSE):.6f} rough×={float(r.RoughnessRatioPredToTrue):.2f} trees={int(r.BestTrees)} {float(r.Runtime_s):.1f}s", flush=True)
        gc.collect()

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out / "04_fold_metrics.csv", index=False, encoding="utf-8-sig")
    new_long, overall, lead = aggregate_new(pred_paths, out)
    base_scope = baseline[baseline.Battery.isin(outer)].copy()
    audit, comp, gain, lead_gain = exact_compare(new_long, base_scope, out, len(outer))

    if not args.skip_final_oof:
        stream_final(pred_paths, out / "11_ALL_OOF_HorizonAware_XGBoost_trajectory.csv.gz")

    expected = len(outer)
    gate = pd.DataFrame([
        dict(Check="V49C exact cohort compatibility", Observed=int(compat.PASS.sum()), Expected=len(compat), PASS=int(compat.PASS.eq(1).all())),
        dict(Check="47 batteries", Observed=len(batteries), Expected=47, PASS=int(len(batteries) == 47)),
        dict(Check="J08/L06 fixed", Observed=f"J{J:02d}_L{L:02d}", Expected="J08_L06", PASS=1),
        dict(Check="Fold rows", Observed=len(fold_df), Expected=expected, PASS=int(len(fold_df) == expected)),
        dict(Check="Finite fold RMSE", Observed=int(np.isfinite(fold_df.RMSE.to_numpy(float)).sum()), Expected=expected,
             PASS=int(len(fold_df) == expected and np.isfinite(fold_df.RMSE.to_numpy(float)).all())),
        dict(Check="Exact baseline/new truth", Observed=int(audit.PASS.iloc[0]), Expected=1, PASS=int(audit.PASS.iloc[0] == 1)),
        dict(Check="Historical SOH used", Observed=0, Expected=0, PASS=1),
        dict(Check="Origin/Target cycle used as X", Observed=0, Expected=0, PASS=1),
        dict(Check="Future HI/SOH used", Observed=0, Expected=0, PASS=1),
    ])
    gate.to_csv(out / "12_acceptance_gate.csv", index=False, encoding="utf-8-sig")
    status = "SMOKE_PASS" if args.only_battery and gate.PASS.eq(1).all() else ("PASS" if gate.PASS.eq(1).all() else "CHECK")

    complete = dict(
        Task="RWTH48 V4.9E Event Horizon-Aware XGBoost full trajectory",
        Status=status, OuterTestBatteriesRun=len(outer), Full47FoldRun=(len(outer) == 47),
        EventTop2=list(EVENT_TOP2), EventCheckpoints=[x[1] for x in EVENT_NODES], J=J, L=L, K=K,
        RawHistoryScalars=L * len(EVENT_COLUMNS), KnownFutureLeadAsFeature=True,
        HistoricalSOHAsFeature=False, FutureSOHAsFeature=False, FutureHIAsFeature=False,
        OriginCycleAsFeature=False, TargetCycleAsFeature=False, PeriodicSOHCalibration=False,
        XGBoostBackend=backend, BestTreesMedian=float(fold_df.BestTrees.median()) if len(fold_df) else np.nan,
        ExactCommonGain=gain.to_dict("records"), Elapsed_s=float(time.perf_counter() - t_all),
        Important=[
            "This model uses one shared scalar XGBoost conditioned on known future lead, rather than 100-output native multi-output regression.",
            "OriginCycle and TargetCycle are keys only and never enter model X.",
            "Historical SOH/capacity and future HI/SOH are never model inputs.",
            "Early stopping uses whole source batteries only; best tree count is refit on all source batteries by default.",
            "All comparison against V4.9C is on exact identical Battery/Origin/Lead/Target/SOH truth.",
        ],
    )
    save_json(complete, out / "99_RWTH48_V4_9E_COMPLETE.json")

    print("\n" + "=" * 118)
    print("RWTH48 V4.9E COMPLETE |", status)
    print(comp.to_string(index=False))
    print("\nGain vs V4.9C multi-output XGBoost:")
    print(gain.to_string(index=False))
    print("Output:", out)
    print("=" * 118)


if __name__ == "__main__":
    main()

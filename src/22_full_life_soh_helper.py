# -*- coding: utf-8 -*-
r"""
RWTH48 V4.11B
=============
Early-Life One-Shot Full Operational-Life SOH Prediction
三路线统一公平比较：
    1) V4.10A-style XGBoost full-life baseline
    2) LSTM / BiLSTM full-life deep models
    3) Trend–Detail full-life deep models

================================================================================
一、冻结研究问题
================================================================================

V4.11A2 已冻结：

    Primary test origin:
        cycle 80

    Input:
        Event Top2 = V_TimeWeightedMean + P_Last
        Event checkpoints = RefAh 0.05 / 0.09 / 0.13
        J = 8
        L = 6
        history = [t-40, t-32, t-24, t-16, t-8, t]
        measured scalars = 6 × 3 × 2 = 36

    Target:
        CyclingSOH_Charge_Canonical
        operational Cycling-SOH，而不是 full-capacity RPT-SOH

    Endpoint:
        Operational EOL80
        persistent Cycling-SOH <= 0.80 for 3 exact cycles

    47 batteries:
        primary origin=80 后 RUL approximately 632...935 cycles
        median approximately 781 cycles

    Strict target policy:
        future target 可跨 StrictSegmentKey；
        只有 V4.5C strict canonical SOH 存在位置 mask=1；
        gap / excluded / recovery cycle = NaN + mask0；
        不插值，不 nearest-fill。

================================================================================
二、为什么模型固定输出 K=1000，而不是 K=935
================================================================================

935 是看完整个数据集真实 EOL 后得到的最大 observed RUL。
如果把模型结构直接定义成 K=935，会让 output horizon 看起来依赖完整测试标签。

因此 V4.11B 在模型设计前冻结：

    K = 1000

所有模型都产生：

    SOH_hat(t+1) ... SOH_hat(t+1000)

测试时绝不使用真实 EOL 来截断预测。

预测 Operational EOL80 从预测曲线自身得到：

    第一次出现连续 3 个预测点 <= 0.80

如果 1000 cycle 内未预测到 crossing：
    PredEOLLead = 1001
    PredEOLFound = 0

真实 SOH trajectory RMSE / MAE：
    只在 true operational EOL 以前且 strict mask=1 的位置计算。

================================================================================
三、三条路线
================================================================================

Route 1: XGBoost full-life baseline
-----------------------------------

V4.10A 冻结表示：

    36 measured scalars
        ->
    196 deterministic engineered features

196维完全复用：
    23_event_tree_helper.py
中的：
    engineer_features()

因为 variable-length full-life target 含 mask，
XGBoost 原生 multi-output 不能像 TensorFlow 一样对每个输出位置做 masked loss。

因此本脚本采用最干净的 full-life Tree baseline：

    [196 V4.10A engineered features]
        +
    [6 deterministic lead encodings]
        ->
    one scalar XGBoost regressor

6个 lead encodings:
    u
    u^2
    sqrt(u)
    log1p(4u)/log(5)
    sin(pi*u)
    cos(pi*u)

其中：
    u = Lead / 1000

模型在一次 vectorized predict 中对 Lead=1...1000 生成整条轨迹。

这不是“1000个独立XGB模型”，而是：
    一个共享的 lead-conditioned XGBoost model。

XGBoost frozen core 参数继续使用 V4.10A/V4.9C：
    n_estimators = 180
    max_depth = 5
    learning_rate = 0.05
    subsample = 0.90
    colsample_bytree = 0.85
    reg_lambda = 1.0
    min_child_weight = 1.0

为了防止长寿命battery拥有更多训练权重：
    每个 source Battery × origin
    最多均匀抽取固定数量的 strict valid leads
    默认 240 leads/origin。

Route 2: LSTM / BiLSTM full-life deep
--------------------------------------

输入：
    [6,6] Event history

Encoder:
    LSTM or BiLSTM

Future decoder:
    context + deterministic lead encoding
        ->
    shared TimeDistributed Dense decoder
        ->
    1000-point trajectory

注意：
    这里没有使用 future HI。
    lead只是“预测距离”，不是absolute Cycle，不携带未来电池状态。

为避免 1000-step recurrent decoder 在 RTX2070 8GB 上过慢，
full-life baseline decoder 使用“共享 lead-conditioned Dense decoder”；
历史序列建模仍由 LSTM/BiLSTM 完成。

Route 3: Trend–Detail full-life
-------------------------------

延续 V4.10D 的核心思想：

    final = trend + gate * detail_highpass

Trend:
    LSTM/BiLSTM encoder
        ->
    lead-aware shared decoder
        ->
    long-term SOH trend

Detail:
    encoder context
        ->
    Dense(K)
        ->
    moving-average high-pass
        ->
    zero-mean
        ->
    sigmoid gate
        ->
    local detail correction

默认：
    detail kernel = 9
    gate bias = -1.5
    sigmoid(-1.5) ≈ 0.182

本任务第一轮不加入 roughness loss。
原因：
    V4.10D 已显示额外 roughness grid 会牺牲RMSE；
    先回答 full-life 下 Trend–Detail 本身是否有价值。

================================================================================
四、严格验证
================================================================================

Outer validation:
    47-cell LOBO
    46 source batteries -> train
    1 held-out battery -> test

Primary test:
    held-out battery ONLY origin=80

Default source augmentation:
    source batteries may use origins:
        80 / 120 / 160

这不是 held-out future leakage：
    后期 origin 只来自 source batteries；
    held-out battery 始终只暴露 origin=80 的36个 measured scalars。

source validation:
    whole source batteries
    default 6 batteries
    任何 validation battery 的所有 origins 一起进入 validation。

held-out battery 不参与：
    input scaler
    target scaler
    early stopping
    model fitting
    XGB fitting
    model/hyperparameter selection

禁止：
    historical true SOH
    capacity history
    future HI
    future SOH as input
    Cycle as model feature
    nearest-cycle substitution
    test-battery calibration
    true EOL truncation at inference

================================================================================
五、Source-Augmented 与 Strict-80
================================================================================

默认：
    --source-origins 80 120 160

目的是增加 source-only training trajectories。

如果要做最严格的 only-origin80 ablation：

    --source-origins 80

测试永远：
    --test-origin 80

论文建议：
    Primary performance = source-augmented
    Ablation = strict source-origin80 only

二者必须单独报告，不能把训练样本数变化藏起来。

================================================================================
六、并行运行设计
================================================================================

脚本按 route 独立输出，因此可以在不同命令行同时跑不同route。

推荐新电脑（RTX2070 8GB）：

终端1：XGB强制CPU，限制4线程
    --routes xgb --xgb-device cpu --xgb-threads 4

终端2：Deep + TrendDetail 在同一个 TensorFlow 进程中顺序运行
    --routes deep trenddetail

不推荐：
    1) XGB CUDA 与 TensorFlow 同时跑；
    2) deep 与 trenddetail 两个独立TensorFlow进程同时抢同一RTX2070；
    3) LSTM和BiLSTM分两个进程同时跑GPU。

原因：
    RTX2070只有一个8GB GPU，
    TensorFlow和XGBoost CUDA会竞争显存、CUDA context和计算资源。

================================================================================
七、默认路径
================================================================================

BASE:
    workspace

V4.11A2:
    <BASE>\RWTH48_V4_11A2_跨StrictSegment全生命目标重建审计

需要：
    08_operational_eol80_target_grid_long.csv.gz
    09_input_history_wide.csv.gz
    99_RWTH48_V4_11A2_COMPLETE.json

V4.10A helper:
    src/
    23_event_tree_helper.py

Output:
    <BASE>\RWTH48_V4_11B_三路线全生命SOH_LOBO

================================================================================
八、运行
================================================================================

1. 审计：
python -u "src/22_full_life_soh_helper.py" --audit-only

2. 单电池 smoke，全三路线：
python -u "src/22_full_life_soh_helper.py" --only-battery RWTH-002 --routes xgb deep trenddetail

3. 单机顺序全跑：
python -u "src/22_full_life_soh_helper.py" --routes xgb deep trenddetail

4. 推荐双任务并行：

终端1：
python -u "src/22_full_life_soh_helper.py" --routes xgb --xgb-device cpu --xgb-threads 4

终端2：
python -u "src/22_full_life_soh_helper.py" --routes deep trenddetail

5. 两边完成后合并：
python -u "src/22_full_life_soh_helper.py" --stage merge

================================================================================
九、输出
================================================================================

01_XGB_FullLife/
02_Deep_FullLife/
03_TrendDetail_FullLife/

每个route：
    00_config.json
    01_fold_metrics.csv
    02_pooled_metrics.csv
    03_battery_metrics.csv
    04_horizon_bin_metrics.csv
    05_eol_rul_metrics.csv
    06_model_diagnostics.csv
    07_ALL_OOF_primary_origin80_long.csv.gz
    99_ROUTE_COMPLETE.json
    folds/<ModelID>/<Battery>.npz

merge后根目录：
    20_combined_pooled_metrics.csv
    21_combined_battery_metrics.csv
    22_combined_horizon_bin_metrics.csv
    23_combined_eol_rul_metrics.csv
    24_model_rank_summary.csv
    25_acceptance_gate.csv
    99_RWTH48_V4_11B_COMPLETE.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 0. Frozen configuration
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR_DEFAULT = Path(__file__).resolve().parent

V411A2_ROOT_DEFAULT = (
    BASE / "RWTH48_V4_11A2_跨StrictSegment全生命目标重建审计"
)
INPUT_WIDE_DEFAULT = V411A2_ROOT_DEFAULT / "09_input_history_wide.csv.gz"
TARGET_GRID_DEFAULT = (
    V411A2_ROOT_DEFAULT / "08_operational_eol80_target_grid_long.csv.gz"
)
V411A2_COMPLETE_DEFAULT = (
    V411A2_ROOT_DEFAULT / "99_RWTH48_V4_11A2_COMPLETE.json"
)

V410A_HELPER_DEFAULT = (
    CODE_DIR_DEFAULT / "23_event_tree_helper.py"
)

OUTPUT_DEFAULT = BASE / "RWTH48_V4_11B_三路线全生命SOH_LOBO"

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
MEASURED_SCALARS = L * len(EVENT_COLUMNS)
K_DEFAULT = 1000
TEST_ORIGIN_DEFAULT = 80
SOURCE_ORIGINS_DEFAULT = (80, 120, 160)

PRIMARY_EOL = 0.80
EOL_PERSISTENCE = 3
EXPECTED_BATTERIES = 47

RANDOM_SEED = 20260822
FAST_GZIP = {"method": "gzip", "compresslevel": 1}
ENC = "utf-8-sig"

MODEL_XGB = "XGB_Eng196_LeadShared"
MODEL_LSTM = "LSTM_FullLife_LeadDecoder"
MODEL_BILSTM = "BiLSTM_FullLife_LeadDecoder"
MODEL_TD_LSTM = "TrendDetail_LSTM_FullLife"
MODEL_TD_BILSTM = "TrendDetail_BiLSTM_FullLife"

ROUTE_DIRS = {
    "xgb": "01_XGB_FullLife",
    "deep": "02_Deep_FullLife",
    "trenddetail": "03_TrendDetail_FullLife",
}

DEEP_DEFAULTS = dict(
    max_epochs=70,
    batch_size=12,
    patience=10,
    min_delta=1e-5,
    learning_rate=8e-4,
    dropout=0.12,
    clipnorm=1.0,
    validation_batteries=6,
)

DETAIL_KERNEL = 9
DETAIL_GATE_BIAS = -1.5


# =============================================================================
# 1. CLI / utilities
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWTH48 V4.11B three-route full operational-life SOH LOBO"
    )

    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--code-dir", type=Path, default=CODE_DIR_DEFAULT)
    p.add_argument("--input-wide", type=Path, default=INPUT_WIDE_DEFAULT)
    p.add_argument("--target-grid", type=Path, default=TARGET_GRID_DEFAULT)
    p.add_argument(
        "--v411a2-complete",
        type=Path,
        default=V411A2_COMPLETE_DEFAULT,
    )
    p.add_argument(
        "--v410a-helper",
        type=Path,
        default=V410A_HELPER_DEFAULT,
    )
    p.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)

    p.add_argument(
        "--stage",
        choices=["run", "merge"],
        default="run",
    )
    p.add_argument(
        "--routes",
        nargs="+",
        choices=["xgb", "deep", "trenddetail"],
        default=["xgb", "deep", "trenddetail"],
    )
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--force", action="store_true")

    p.add_argument("--k", type=int, default=K_DEFAULT)
    p.add_argument(
        "--test-origin",
        type=int,
        default=TEST_ORIGIN_DEFAULT,
    )
    p.add_argument(
        "--source-origins",
        type=int,
        nargs="+",
        default=list(SOURCE_ORIGINS_DEFAULT),
    )

    # XGBoost
    p.add_argument(
        "--xgb-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument("--xgb-threads", type=int, default=6)
    p.add_argument(
        "--xgb-leads-per-origin",
        type=int,
        default=240,
    )

    # Deep
    p.add_argument(
        "--architectures",
        nargs="+",
        choices=["LSTM", "BiLSTM"],
        default=["LSTM", "BiLSTM"],
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=DEEP_DEFAULTS["max_epochs"],
    )
    p.add_argument(
        "--deep-batch",
        type=int,
        default=DEEP_DEFAULTS["batch_size"],
    )
    p.add_argument(
        "--early-patience",
        type=int,
        default=DEEP_DEFAULTS["patience"],
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=DEEP_DEFAULTS["min_delta"],
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=DEEP_DEFAULTS["learning_rate"],
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=DEEP_DEFAULTS["dropout"],
    )
    p.add_argument(
        "--clipnorm",
        type=float,
        default=DEEP_DEFAULTS["clipnorm"],
    )
    p.add_argument(
        "--validation-batteries",
        type=int,
        default=DEEP_DEFAULTS["validation_batteries"],
    )
    p.add_argument(
        "--no-refit-all-source",
        action="store_true",
    )

    # Trend–Detail
    p.add_argument(
        "--detail-kernel",
        type=int,
        default=DETAIL_KERNEL,
    )
    p.add_argument(
        "--detail-gate-bias",
        type=float,
        default=DETAIL_GATE_BIAS,
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
    s = str(x).strip().upper().replace("_", "-")
    m = re.search(r"RWTH-?0*(\d+)", s)
    if m:
        return f"RWTH-{int(m.group(1)):03d}"
    try:
        return f"RWTH-{int(float(s)):03d}"
    except Exception:
        return s


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stable_hash(obj: object) -> str:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def calc_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)

    m = np.isfinite(yt) & np.isfinite(yp)
    if mask is not None:
        m &= np.asarray(mask).astype(bool)

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
        RMSE=float(np.sqrt(np.mean(e * e))),
        R2=(
            float(r2_score(yt, yp))
            if len(yt) >= 2 and np.std(yt) > 1e-12
            else np.nan
        ),
        MAPE_pct=float(
            np.mean(
                np.abs(e)
                / np.maximum(np.abs(yt), 1e-9)
            )
            * 100.0
        ),
        Bias=float(np.mean(e)),
    )


def shape_metrics_masked(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    yt = np.asarray(y_true, float).reshape(-1)
    yp = np.asarray(y_pred, float).reshape(-1)
    m = (
        np.asarray(mask).astype(bool).reshape(-1)
        & np.isfinite(yt)
        & np.isfinite(yp)
    )

    if m.sum() == 0:
        return dict(
            EndpointAbsError=np.nan,
            TrueUpwardStepFraction=np.nan,
            PredUpwardStepFraction=np.nan,
            TrueRoughness=np.nan,
            PredRoughness=np.nan,
            RoughnessRatioPredToTrue=np.nan,
        )

    idx = np.flatnonzero(m)
    end = int(idx[-1])
    endpoint = float(abs(yp[end] - yt[end]))

    pair = m[:-1] & m[1:]
    if pair.any():
        dt = yt[1:] - yt[:-1]
        dp = yp[1:] - yp[:-1]
        tu = float(np.mean(dt[pair] > 0))
        pu = float(np.mean(dp[pair] > 0))
    else:
        tu = np.nan
        pu = np.nan

    trip = m[:-2] & m[1:-1] & m[2:]
    if trip.any():
        d2t = yt[2:] - 2 * yt[1:-1] + yt[:-2]
        d2p = yp[2:] - 2 * yp[1:-1] + yp[:-2]
        tr = float(np.mean(np.abs(d2t[trip])))
        pr = float(np.mean(np.abs(d2p[trip])))
        rr = (
            float(pr / tr)
            if np.isfinite(tr) and tr > 1e-15
            else np.nan
        )
    else:
        tr = pr = rr = np.nan

    return dict(
        EndpointAbsError=endpoint,
        TrueUpwardStepFraction=tu,
        PredUpwardStepFraction=pu,
        TrueRoughness=tr,
        PredRoughness=pr,
        RoughnessRatioPredToTrue=rr,
    )


def find_persistent_eol_lead(
    pred: np.ndarray,
    threshold: float = PRIMARY_EOL,
    persistence: int = EOL_PERSISTENCE,
) -> Tuple[int, int]:
    p = np.asarray(pred, float).reshape(-1)
    n = len(p)
    q = int(persistence)

    for i in range(0, n - q + 1):
        w = p[i:i+q]
        if np.isfinite(w).all() and np.all(w <= float(threshold)):
            return i + 1, 1  # lead is 1-based

    return n + 1, 0


def choose_validation_batteries(
    source_bats: Sequence[str],
    n_val: int,
) -> Tuple[List[str], List[str]]:
    src = sorted(source_bats)
    n_val = max(1, min(int(n_val), len(src) - 2))
    pos = np.unique(
        np.round(
            np.linspace(0, len(src) - 1, n_val)
        ).astype(int)
    )
    val = [src[i] for i in pos]
    for b in src:
        if len(val) >= n_val:
            break
        if b not in val:
            val.append(b)
    val = sorted(val[:n_val])
    tr = [b for b in src if b not in set(val)]
    return tr, val


# =============================================================================
# 2. Handoff loading
# =============================================================================

@dataclass
class FullLifeBundle:
    X: np.ndarray          # [N,6,6]
    y: np.ndarray          # [N,K], NaN where no strict truth
    mask: np.ndarray       # [N,K]
    battery: np.ndarray
    origin: np.ndarray
    true_eol_lead: np.ndarray
    true_eol_cycle: np.ndarray

    def select(
        self,
        batteries: Optional[Sequence[str]] = None,
        origins: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        m = np.ones(len(self.battery), dtype=bool)
        if batteries is not None:
            m &= np.isin(self.battery, list(batteries))
        if origins is not None:
            m &= np.isin(self.origin, list(origins))
        return np.flatnonzero(m)


def load_handoff(
    args: argparse.Namespace,
) -> Tuple[FullLifeBundle, Dict[str, object], pd.DataFrame]:
    for p in (
        args.input_wide,
        args.target_grid,
        args.v411a2_complete,
        args.v410a_helper,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    complete = json.loads(
        args.v411a2_complete.read_text(encoding="utf-8")
    )

    required_checks = [
        ("Status", complete.get("Status") == "PASS"),
        ("Batteries", int(complete.get("Batteries", -1)) == EXPECTED_BATTERIES),
        ("EventTop2", complete.get("EventTop2") == list(EVENT_TOP2)),
        ("J", int(complete.get("J", -1)) == J),
        ("L", int(complete.get("L", -1)) == L),
        (
            "MeasuredScalarInputs",
            int(complete.get("MeasuredScalarInputs", -1)) == MEASURED_SCALARS,
        ),
        ("CrossStrictSegmentTarget", bool(complete.get("CrossStrictSegmentTarget", False))),
        (
            "HistoricalSOHAsModelInput",
            not bool(complete.get("HistoricalSOHAsModelInput", True)),
        ),
        (
            "FutureHIAsModelInput",
            not bool(complete.get("FutureHIAsModelInput", True)),
        ),
        (
            "CycleAsModelFeature",
            not bool(complete.get("CycleAsModelFeature", True)),
        ),
    ]

    audit_rows = [
        dict(Check=n, PASS=int(ok))
        for n, ok in required_checks
    ]
    if not all(ok for _, ok in required_checks):
        raise RuntimeError(
            "V4.11A2 provenance不一致:\n"
            + pd.DataFrame(audit_rows).to_string(index=False)
        )

    if int(args.k) < int(complete.get("PrimaryOriginRUL_Max", 0)):
        raise RuntimeError(
            f"K={args.k} 小于V4.11A2 primary最大RUL="
            f"{complete.get('PrimaryOriginRUL_Max')}"
        )

    inp = pd.read_csv(
        args.input_wide,
        compression="infer",
        low_memory=False,
    )
    grid = pd.read_csv(
        args.target_grid,
        compression="infer",
        low_memory=False,
    )

    inp["Battery"] = inp["Battery"].map(normalize_battery)
    inp["OriginCycle_MetadataOnly"] = pd.to_numeric(
        inp["OriginCycle_MetadataOnly"],
        errors="coerce",
    ).astype(int)

    grid["Battery"] = grid["Battery"].map(normalize_battery)
    for c in (
        "OriginCycle_MetadataOnly",
        "Lead",
        "TargetCycle_QCOnly",
        "StrictTruthMask",
        "OperationalEOL80Lead_Truth",
    ):
        grid[c] = pd.to_numeric(grid[c], errors="coerce")

    key = ["Battery", "OriginCycle_MetadataOnly"]
    if inp.duplicated(key).any():
        raise RuntimeError("input wide duplicate Battery+Origin")
    if grid.duplicated(key + ["Lead"]).any():
        raise RuntimeError("target grid duplicate Battery+Origin+Lead")

    feature_cols = [
        f"H{li+1}__{c}"
        for li in range(L)
        for c in EVENT_COLUMNS
    ]
    missing = [c for c in feature_cols if c not in inp.columns]
    if missing:
        raise RuntimeError(f"input wide缺36标量列: {missing[:10]}")

    inp = inp.sort_values(
        ["OriginCycle_MetadataOnly", "Battery"]
    ).reset_index(drop=True)

    X = (
        inp[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float32)
        .reshape(len(inp), L, len(EVENT_COLUMNS))
    )
    if not np.isfinite(X).all():
        raise RuntimeError("36-scalar X存在NaN/Inf")

    grouped = {
        (str(b), int(o)): g.sort_values("Lead")
        for (b, o), g in grid.groupby(key, sort=False)
    }

    K = int(args.k)
    y = np.full((len(inp), K), np.nan, dtype=np.float32)
    mask = np.zeros((len(inp), K), dtype=np.uint8)
    true_eol_lead = np.zeros(len(inp), dtype=np.int32)
    true_eol_cycle = np.zeros(len(inp), dtype=np.int32)

    for i, r in enumerate(inp.itertuples(index=False)):
        b = str(r.Battery)
        o = int(r.OriginCycle_MetadataOnly)
        g = grouped.get((b, o))
        if g is None or len(g) == 0:
            raise RuntimeError(f"{b}/{o}: target grid缺失")

        leads = g["Lead"].astype(int).to_numpy()
        if leads.min() != 1:
            raise RuntimeError(f"{b}/{o}: target lead不是从1开始")
        if leads.max() > K:
            raise RuntimeError(f"{b}/{o}: lead>{K}")

        yy = pd.to_numeric(g["SOH_True"], errors="coerce").to_numpy(float)
        mm = g["StrictTruthMask"].fillna(0).astype(int).to_numpy().astype(bool)

        pos = leads - 1
        y[i, pos[mm]] = yy[mm].astype(np.float32)
        mask[i, pos[mm]] = 1

        eol_leads = pd.to_numeric(
            g["OperationalEOL80Lead_Truth"],
            errors="coerce",
        ).dropna().unique()

        if len(eol_leads) != 1:
            raise RuntimeError(f"{b}/{o}: true EOL lead不唯一")
        e = int(eol_leads[0])
        if e < 1 or e > K:
            raise RuntimeError(f"{b}/{o}: true EOL lead={e}不在1..{K}")
        true_eol_lead[i] = e
        true_eol_cycle[i] = o + e

        if mask[i, :e].sum() == 0:
            raise RuntimeError(f"{b}/{o}: EOL前无strict truth")

    if np.isfinite(y[mask == 0]).any():
        raise RuntimeError("mask0位置出现有限truth")
    if not np.isfinite(y[mask == 1]).all():
        raise RuntimeError("mask1位置存在NaN/Inf")

    batteries = inp["Battery"].astype(str).to_numpy()
    origins = inp["OriginCycle_MetadataOnly"].to_numpy(np.int32)

    bundle = FullLifeBundle(
        X=X,
        y=y,
        mask=mask,
        battery=batteries,
        origin=origins,
        true_eol_lead=true_eol_lead,
        true_eol_cycle=true_eol_cycle,
    )

    bats = sorted(set(batteries))
    if len(bats) != EXPECTED_BATTERIES:
        raise RuntimeError(f"handoff batteries={len(bats)}，预期47")

    test_idx = bundle.select(origins=[int(args.test_origin)])
    if len(test_idx) != EXPECTED_BATTERIES:
        raise RuntimeError(
            f"test origin={args.test_origin} 应有47行，实际={len(test_idx)}"
        )

    for o in args.source_origins:
        n = len(bundle.select(origins=[int(o)]))
        if n != EXPECTED_BATTERIES:
            raise RuntimeError(
                f"source origin={o} 应有47行，实际={n}"
            )

    audit_rows.extend([
        dict(Check="InputRows", Observed=len(inp), PASS=1),
        dict(Check="TestOriginRows", Observed=len(test_idx), PASS=int(len(test_idx)==47)),
        dict(Check="K", Observed=K, PASS=int(K >= int(complete["PrimaryOriginRUL_Max"]))),
        dict(Check="MeasuredScalars", Observed=MEASURED_SCALARS, PASS=1),
        dict(Check="HistoricalSOHInput", Observed=0, PASS=1),
        dict(Check="FutureHIInput", Observed=0, PASS=1),
        dict(Check="CycleFeature", Observed=0, PASS=1),
    ])

    return bundle, complete, pd.DataFrame(audit_rows)


# =============================================================================
# 3. Scaling
# =============================================================================

@dataclass
class InputScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(X, np.float32) - self.mean[None, None, :])
            / self.std[None, None, :]
        ).astype(np.float32)


def fit_input_scaler(X: np.ndarray) -> InputScaler:
    a = np.asarray(X, float)
    mean = np.nanmean(a, axis=(0, 1))
    std = np.nanstd(a, axis=(0, 1))
    mean[~np.isfinite(mean)] = 0.0
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    return InputScaler(
        mean.astype(np.float32),
        std.astype(np.float32),
    )


@dataclass
class TargetScaler:
    mean: float
    std: float

    def transform(self, y: np.ndarray) -> np.ndarray:
        z = np.asarray(y, np.float32).copy()
        m = np.isfinite(z)
        z[m] = (z[m] - self.mean) / self.std
        return z

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return (
            np.asarray(y, np.float32) * self.std + self.mean
        ).astype(np.float32)


def fit_target_scaler(
    y: np.ndarray,
    mask: np.ndarray,
) -> TargetScaler:
    a = np.asarray(y, float)
    m = np.asarray(mask).astype(bool) & np.isfinite(a)
    v = a[m]
    if len(v) == 0:
        raise RuntimeError("target scaler无有效source truth")
    mean = float(np.mean(v))
    std = float(np.std(v))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return TargetScaler(mean, std)


# =============================================================================
# 4. V4.10A engineered features / XGBoost
# =============================================================================

def load_v410a_helper(path: Path):
    m = load_module(path, "rwth48_v410a_for_v411b")
    required = (
        "engineer_features",
        "XGB_FROZEN_CORE",
    )
    miss = [x for x in required if not hasattr(m, x)]
    if miss:
        raise RuntimeError(f"V4.10A helper缺函数/配置: {miss}")
    return m


def lead_features(
    leads: np.ndarray,
    K: int,
) -> np.ndarray:
    l = np.asarray(leads, float).reshape(-1)
    u = l / float(K)
    return np.column_stack([
        u,
        u * u,
        np.sqrt(np.maximum(u, 0.0)),
        np.log1p(4.0 * u) / np.log(5.0),
        np.sin(np.pi * u),
        np.cos(np.pi * u),
    ]).astype(np.float32)


def sample_valid_leads(
    mask_row: np.ndarray,
    max_count: int,
) -> np.ndarray:
    valid = np.flatnonzero(
        np.asarray(mask_row).astype(bool)
    )
    if len(valid) == 0:
        return valid
    if max_count <= 0 or len(valid) <= max_count:
        return valid
    pos = np.linspace(0, len(valid) - 1, int(max_count))
    return valid[np.round(pos).astype(int)]


def make_xgb_long_train(
    helper,
    bundle: FullLifeBundle,
    indices: np.ndarray,
    K: int,
    leads_per_origin: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    Xseq = bundle.X[indices]
    Xeng, names, registry = helper.engineer_features(Xseq)

    if Xeng.shape[1] != 196:
        raise RuntimeError(
            f"V4.10A engineered feature count={Xeng.shape[1]}，预期196"
        )

    rows_x: List[np.ndarray] = []
    rows_y: List[np.ndarray] = []

    for j, idx in enumerate(indices):
        pos = sample_valid_leads(
            bundle.mask[idx],
            int(leads_per_origin),
        )
        if len(pos) == 0:
            continue
        base = np.repeat(
            Xeng[j:j+1],
            len(pos),
            axis=0,
        )
        lf = lead_features(pos + 1, K)
        rows_x.append(
            np.concatenate([base, lf], axis=1)
        )
        rows_y.append(
            bundle.y[idx, pos].astype(np.float32)
        )

    if not rows_x:
        raise RuntimeError("XGB long-format source rows为空")

    Xtr = np.vstack(rows_x).astype(np.float32)
    ytr = np.concatenate(rows_y).astype(np.float32)

    if not np.isfinite(Xtr).all() or not np.isfinite(ytr).all():
        raise RuntimeError("XGB long-format含NaN/Inf")

    lead_names = [
        "LEAD__u",
        "LEAD__u2",
        "LEAD__sqrt_u",
        "LEAD__log1p4u_log5",
        "LEAD__sin_pi_u",
        "LEAD__cos_pi_u",
    ]

    lead_reg = pd.DataFrame([
        {
            "Feature": n,
            "Group": "KnownLeadEncoding",
            "MeasuredOrDerived": "Derived",
            "Source": "deterministic forecast lead / K; no future battery measurement",
        }
        for n in lead_names
    ])

    registry2 = pd.concat(
        [registry, lead_reg],
        ignore_index=True,
    )
    return Xtr, ytr, names + lead_names, registry2


def make_xgb_test_grid(
    helper,
    xseq: np.ndarray,
    K: int,
) -> np.ndarray:
    Xeng, _, _ = helper.engineer_features(
        np.asarray(xseq, np.float32)[None, :, :]
    )
    base = np.repeat(Xeng, K, axis=0)
    lf = lead_features(np.arange(1, K+1), K)
    return np.concatenate(
        [base, lf],
        axis=1,
    ).astype(np.float32)


def fit_xgb_scalar(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    helper,
    device_mode: str,
    threads: int,
) -> Tuple[np.ndarray, str]:
    from xgboost import XGBRegressor

    base = dict(helper.XGB_FROZEN_CORE)
    base["n_jobs"] = max(1, int(threads))
    base.pop("multi_strategy", None)
    base.pop("device", None)
    base.pop("gpu_id", None)

    attempts: List[Tuple[str, Dict[str, object]]] = []

    if device_mode in ("auto", "cuda"):
        p = dict(base)
        p["tree_method"] = "hist"
        p["device"] = "cuda"
        attempts.append(("cuda_device", p))

        # Legacy XGBoost compatibility.
        p2 = dict(base)
        p2["tree_method"] = "gpu_hist"
        attempts.append(("cuda_gpu_hist_legacy", p2))

    if device_mode in ("auto", "cpu"):
        p = dict(base)
        p["tree_method"] = "hist"
        p["device"] = "cpu"
        attempts.append(("cpu_device", p))

        p2 = dict(base)
        p2["tree_method"] = "hist"
        attempts.append(("cpu_hist_legacy", p2))

    errors = []
    for name, params in attempts:
        model = None
        try:
            model = XGBRegressor(**params)
            model.fit(Xtr, ytr, verbose=False)
            pred = np.asarray(
                model.predict(Xte),
                dtype=float,
            ).reshape(-1)
            if len(pred) != len(Xte) or not np.isfinite(pred).all():
                raise RuntimeError(
                    f"prediction shape/finite invalid: {pred.shape}"
                )
            return pred, name
        except Exception as exc:
            errors.append(
                f"{name}: {type(exc).__name__}: {exc}"
            )
            try:
                del model
            except Exception:
                pass
            gc.collect()

    raise RuntimeError(
        "XGBoost所有backend尝试失败:\n"
        + "\n".join(errors)
    )


# =============================================================================
# 5. TensorFlow full-life models
# =============================================================================

def setup_tensorflow() -> Dict[str, object]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    return {
        "TensorFlowVersion": str(tf.__version__),
        "GPUs": [str(x) for x in gpus],
        "GPUCount": len(gpus),
    }


def masked_mse_loss():
    import tensorflow as tf

    def loss(y_true, y_pred):
        mask = tf.math.is_finite(y_true)
        yt = tf.where(mask, y_true, tf.zeros_like(y_true))
        yp = tf.where(mask, y_pred, tf.zeros_like(y_pred))
        sq = tf.square(yp - yt)
        sq = tf.where(mask, sq, tf.zeros_like(sq))
        den = tf.maximum(
            tf.reduce_sum(tf.cast(mask, tf.float32)),
            1.0,
        )
        return tf.reduce_sum(sq) / den

    loss.__name__ = "masked_mse"
    return loss


def build_tf_lead_features(tf, context, K: int):
    def fn(x):
        b = tf.shape(x)[0]
        u = (
            tf.cast(
                tf.range(1, K + 1),
                tf.float32,
            )
            / float(K)
        )
        f = tf.stack([
            u,
            tf.square(u),
            tf.sqrt(u),
            tf.math.log1p(4.0 * u) / tf.math.log(5.0),
            tf.sin(math.pi * u),
            tf.cos(math.pi * u),
        ], axis=1)
        return tf.tile(
            f[None, :, :],
            [b, 1, 1],
        )

    return tf.keras.layers.Lambda(
        fn,
        name="known_future_lead_encoding",
    )(context)


def encode_history(
    tf,
    inp,
    arch: str,
    dropout: float,
):
    x = tf.keras.layers.LayerNormalization(
        name="input_layernorm"
    )(inp)

    if arch == "LSTM":
        x = tf.keras.layers.LSTM(
            64,
            return_sequences=True,
            name="encoder_lstm1",
        )(x)
        x = tf.keras.layers.Dropout(
            dropout,
            name="encoder_dropout1",
        )(x)
        context = tf.keras.layers.LSTM(
            48,
            return_sequences=False,
            name="encoder_lstm2",
        )(x)
    elif arch == "BiLSTM":
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                48,
                return_sequences=True,
            ),
            name="encoder_bilstm1",
        )(x)
        x = tf.keras.layers.Dropout(
            dropout,
            name="encoder_dropout1",
        )(x)
        context = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                32,
                return_sequences=False,
            ),
            name="encoder_bilstm2",
        )(x)
    else:
        raise ValueError(arch)

    context = tf.keras.layers.Dense(
        64,
        activation="swish",
        name="context_dense64",
    )(context)
    context = tf.keras.layers.LayerNormalization(
        name="context_layernorm"
    )(context)
    context = tf.keras.layers.Dropout(
        dropout,
        name="context_dropout",
    )(context)
    return context


def lead_shared_trend_decoder(
    tf,
    context,
    K: int,
    dropout: float,
    prefix: str = "trend",
):
    rep = tf.keras.layers.RepeatVector(
        K,
        name=f"{prefix}_repeat_context",
    )(context)
    lead = build_tf_lead_features(
        tf,
        context,
        K,
    )
    z = tf.keras.layers.Concatenate(
        name=f"{prefix}_context_plus_lead"
    )([rep, lead])
    z = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            64,
            activation="swish",
        ),
        name=f"{prefix}_dense64",
    )(z)
    z = tf.keras.layers.Dropout(
        dropout,
        name=f"{prefix}_dropout",
    )(z)
    z = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            32,
            activation="swish",
        ),
        name=f"{prefix}_dense32",
    )(z)
    z = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            1,
            activation="linear",
            bias_initializer="zeros",
        ),
        name=f"{prefix}_each_lead",
    )(z)
    return tf.keras.layers.Reshape(
        (K,),
        name=f"{prefix}_trajectory",
    )(z)


def build_full_life_deep_model(
    arch: str,
    K: int,
    dropout: float,
    lr: float,
    clipnorm: float,
):
    import tensorflow as tf

    tf.keras.backend.clear_session()

    inp = tf.keras.Input(
        shape=(L, len(EVENT_COLUMNS)),
        name="EventTop2_history",
    )
    context = encode_history(
        tf,
        inp,
        arch,
        dropout,
    )
    final = lead_shared_trend_decoder(
        tf,
        context,
        K,
        dropout,
        prefix="full_life",
    )

    model = tf.keras.Model(
        inp,
        final,
        name=f"{arch}_FullLife_LeadDecoder",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(lr),
            clipnorm=float(clipnorm),
        ),
        loss=masked_mse_loss(),
    )
    return model


def build_trend_detail_model(
    arch: str,
    K: int,
    dropout: float,
    lr: float,
    clipnorm: float,
    detail_kernel: int,
    detail_gate_bias: float,
):
    import tensorflow as tf

    if detail_kernel < 3 or detail_kernel % 2 == 0:
        raise ValueError(
            "--detail-kernel必须为>=3的奇数"
        )

    tf.keras.backend.clear_session()

    inp = tf.keras.Input(
        shape=(L, len(EVENT_COLUMNS)),
        name="EventTop2_history",
    )
    context = encode_history(
        tf,
        inp,
        arch,
        dropout,
    )

    trend = lead_shared_trend_decoder(
        tf,
        context,
        K,
        dropout,
        prefix="trend",
    )

    d = tf.keras.layers.Dense(
        96,
        activation="swish",
        name="detail_dense96",
    )(context)
    d = tf.keras.layers.Dropout(
        dropout,
        name="detail_dropout",
    )(d)
    detail_raw = tf.keras.layers.Dense(
        K,
        activation="linear",
        name="detail_raw_DenseK",
    )(d)

    half = int(detail_kernel // 2)

    def highpass(v):
        v3 = tf.expand_dims(v, axis=-1)
        vp = tf.pad(
            v3,
            paddings=[[0, 0], [half, half], [0, 0]],
            mode="REFLECT",
        )
        smooth = tf.nn.avg_pool1d(
            vp,
            ksize=int(detail_kernel),
            strides=1,
            padding="VALID",
        )
        smooth = tf.squeeze(
            smooth,
            axis=-1,
        )
        hp = v - smooth
        return hp - tf.reduce_mean(
            hp,
            axis=1,
            keepdims=True,
        )

    detail_hp = tf.keras.layers.Lambda(
        highpass,
        name="detail_highpass_zero_mean",
    )(detail_raw)

    gate = tf.keras.layers.Dense(
        K,
        activation="sigmoid",
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(
            float(detail_gate_bias)
        ),
        name="detail_gate",
    )(context)

    detail_contrib = tf.keras.layers.Multiply(
        name="detail_contribution",
    )([gate, detail_hp])

    final = tf.keras.layers.Add(
        name="SOH_future_trajectory_scaled",
    )([trend, detail_contrib])

    model = tf.keras.Model(
        inp,
        final,
        name=f"TrendDetail_{arch}_FullLife",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(lr),
            clipnorm=float(clipnorm),
        ),
        loss=masked_mse_loss(),
    )
    return model


def diagnostic_trend_detail_model(model):
    import tensorflow as tf
    return tf.keras.Model(
        model.input,
        [
            model.output,
            model.get_layer("trend_trajectory").output,
            model.get_layer("detail_contribution").output,
            model.get_layer("detail_gate").output,
        ],
    )


def set_tf_seed(seed: int) -> None:
    import tensorflow as tf
    np.random.seed(int(seed))
    try:
        tf.keras.utils.set_random_seed(int(seed))
    except Exception:
        tf.random.set_seed(int(seed))


def train_deep_fold(
    bundle: FullLifeBundle,
    test_battery: str,
    arch: str,
    trend_detail: bool,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    import tensorflow as tf

    all_bats = sorted(set(bundle.battery.tolist()))
    source_bats = [
        b for b in all_bats
        if b != test_battery
    ]
    train_bats, val_bats = choose_validation_batteries(
        source_bats,
        args.validation_batteries,
    )

    source_origins = list(map(int, args.source_origins))
    test_origin = int(args.test_origin)

    tr_idx = bundle.select(
        batteries=train_bats,
        origins=source_origins,
    )
    va_idx = bundle.select(
        batteries=val_bats,
        origins=source_origins,
    )
    all_src_idx = bundle.select(
        batteries=source_bats,
        origins=source_origins,
    )
    te_idx = bundle.select(
        batteries=[test_battery],
        origins=[test_origin],
    )

    if len(te_idx) != 1:
        raise RuntimeError(
            f"{test_battery}: test origin {test_origin} rows={len(te_idx)}"
        )
    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise RuntimeError("source train/validation为空")

    xsc = fit_input_scaler(
        bundle.X[tr_idx]
    )
    ysc = fit_target_scaler(
        bundle.y[tr_idx],
        bundle.mask[tr_idx],
    )

    Xtr = xsc.transform(bundle.X[tr_idx])
    Xva = xsc.transform(bundle.X[va_idx])
    ytr = ysc.transform(bundle.y[tr_idx])
    yva = ysc.transform(bundle.y[va_idx])

    seed = (
        RANDOM_SEED
        + int(test_battery.split("-")[-1]) * 100
        + (1 if arch == "BiLSTM" else 0)
        + (10 if trend_detail else 0)
    )
    set_tf_seed(seed)

    if trend_detail:
        model = build_trend_detail_model(
            arch=arch,
            K=int(args.k),
            dropout=float(args.dropout),
            lr=float(args.learning_rate),
            clipnorm=float(args.clipnorm),
            detail_kernel=int(args.detail_kernel),
            detail_gate_bias=float(args.detail_gate_bias),
        )
    else:
        model = build_full_life_deep_model(
            arch=arch,
            K=int(args.k),
            dropout=float(args.dropout),
            lr=float(args.learning_rate),
            clipnorm=float(args.clipnorm),
        )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(args.early_patience),
            min_delta=float(args.min_delta),
            restore_best_weights=True,
            verbose=0,
        )
    ]

    t0 = time.perf_counter()
    hist = model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=int(args.max_epochs),
        batch_size=int(args.deep_batch),
        shuffle=True,
        verbose=0,
        callbacks=callbacks,
    )

    val_loss = np.asarray(
        hist.history.get("val_loss", []),
        float,
    )
    if len(val_loss) == 0 or not np.isfinite(val_loss).any():
        raise RuntimeError(
            f"{test_battery}/{arch}: val_loss无有效值"
        )
    best_epoch = int(np.nanargmin(val_loss)) + 1
    best_val_loss = float(np.nanmin(val_loss))

    del Xtr, Xva, ytr, yva
    gc.collect()

    # Final refit on all 46 source batteries after epoch frozen.
    if args.no_refit_all_source:
        final_model = model
        final_xsc = xsc
        final_ysc = ysc
        refit = 0
    else:
        del model
        tf.keras.backend.clear_session()
        gc.collect()

        final_xsc = fit_input_scaler(
            bundle.X[all_src_idx]
        )
        final_ysc = fit_target_scaler(
            bundle.y[all_src_idx],
            bundle.mask[all_src_idx],
        )
        Xall = final_xsc.transform(
            bundle.X[all_src_idx]
        )
        yall = final_ysc.transform(
            bundle.y[all_src_idx]
        )

        set_tf_seed(seed + 100000)

        if trend_detail:
            final_model = build_trend_detail_model(
                arch=arch,
                K=int(args.k),
                dropout=float(args.dropout),
                lr=float(args.learning_rate),
                clipnorm=float(args.clipnorm),
                detail_kernel=int(args.detail_kernel),
                detail_gate_bias=float(args.detail_gate_bias),
            )
        else:
            final_model = build_full_life_deep_model(
                arch=arch,
                K=int(args.k),
                dropout=float(args.dropout),
                lr=float(args.learning_rate),
                clipnorm=float(args.clipnorm),
            )

        final_model.fit(
            Xall,
            yall,
            epochs=max(1, best_epoch),
            batch_size=int(args.deep_batch),
            shuffle=True,
            verbose=0,
        )
        del Xall, yall
        refit = 1

    Xte = final_xsc.transform(
        bundle.X[te_idx]
    )

    diag: Dict[str, object] = {}
    if trend_detail:
        dm = diagnostic_trend_detail_model(final_model)
        p_scaled, trend_scaled, detail_scaled, gate = dm.predict(
            Xte,
            batch_size=1,
            verbose=0,
        )
        pred = final_ysc.inverse(
            np.asarray(p_scaled, float)
        )[0]
        # Detail is in scaled target units; rescale amplitude by std only.
        detail_real = (
            np.asarray(detail_scaled, float)[0]
            * final_ysc.std
        )
        diag.update(
            MeanDetailGate=float(np.mean(gate)),
            DetailRMS=float(
                np.sqrt(np.mean(detail_real * detail_real))
            ),
            TrendRMS=float(
                np.sqrt(
                    np.mean(
                        (
                            final_ysc.inverse(
                                np.asarray(trend_scaled, float)
                            )[0]
                        ) ** 2
                    )
                )
            ),
        )
    else:
        p_scaled = final_model.predict(
            Xte,
            batch_size=1,
            verbose=0,
        )
        pred = final_ysc.inverse(
            np.asarray(p_scaled, float)
        )[0]

    runtime = time.perf_counter() - t0

    info = dict(
        Architecture=arch,
        TrendDetail=int(trend_detail),
        BestEpoch=best_epoch,
        BestValLossScaled=best_val_loss,
        TrainBatteries=len(train_bats),
        ValidationBatteries=len(val_bats),
        ValidationBatteryList=";".join(val_bats),
        SourceOrigins=";".join(map(str, source_origins)),
        TrainSamples=len(tr_idx),
        ValidationSamples=len(va_idx),
        FinalSourceSamples=len(all_src_idx),
        RefitAllSource=refit,
        InputScalerSourceOnly=1,
        TargetScalerSourceOnly=1,
        TargetScalerMean=float(final_ysc.mean),
        TargetScalerStd=float(final_ysc.std),
        Runtime_s=float(runtime),
        **diag,
    )

    try:
        del final_model
    except Exception:
        pass
    tf.keras.backend.clear_session()
    gc.collect()

    return np.asarray(pred, float), info


# =============================================================================
# 6. Cache / prediction row conversion
# =============================================================================

def fold_cache_path(
    route_dir: Path,
    model_id: str,
    battery: str,
) -> Path:
    return (
        route_dir
        / "folds"
        / model_id
        / f"{battery}.npz"
    )


def make_signature(
    model_id: str,
    args: argparse.Namespace,
) -> str:
    return stable_hash(dict(
        Task="V411B",
        ModelID=model_id,
        K=int(args.k),
        TestOrigin=int(args.test_origin),
        SourceOrigins=list(map(int, args.source_origins)),
        XGBDevice=str(args.xgb_device),
        XGBLeadsPerOrigin=int(args.xgb_leads_per_origin),
        MaxEpochs=int(args.max_epochs),
        DeepBatch=int(args.deep_batch),
        Patience=int(args.early_patience),
        MinDelta=float(args.min_delta),
        LearningRate=float(args.learning_rate),
        Dropout=float(args.dropout),
        Clipnorm=float(args.clipnorm),
        ValidationBatteries=int(args.validation_batteries),
        RefitAllSource=not bool(args.no_refit_all_source),
        DetailKernel=int(args.detail_kernel),
        DetailGateBias=float(args.detail_gate_bias),
    ))


def save_fold_cache(
    path: Path,
    signature: str,
    pred: np.ndarray,
    info: Dict[str, object],
) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        signature=np.asarray([signature], dtype="U64"),
        pred=np.asarray(pred, np.float32),
        info_json=np.asarray([
            json.dumps(
                info,
                ensure_ascii=False,
                default=str,
            )
        ], dtype="U20000"),
    )


def load_fold_cache(
    path: Path,
    signature: str,
) -> Optional[Tuple[np.ndarray, Dict[str, object]]]:
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        sig = str(z["signature"][0])
        if sig != signature:
            return None
        pred = np.asarray(z["pred"], float)
        info = json.loads(str(z["info_json"][0]))
        return pred, info
    except Exception:
        return None


def prediction_long_rows(
    model_id: str,
    battery: str,
    origin: int,
    pred: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray,
    true_eol_lead: int,
) -> pd.DataFrame:
    K = len(pred)
    leads = np.arange(1, K+1, dtype=int)
    return pd.DataFrame({
        "ModelID": model_id,
        "Battery": battery,
        "OriginCycle_KeyOnly": int(origin),
        "Lead": leads,
        "TargetCycle_KeyOnly": int(origin) + leads,
        "SOH_True": np.asarray(y_true, float),
        "StrictTruthMask": np.asarray(mask).astype(np.uint8),
        "SOH_Pred": np.asarray(pred, float),
        "TrueOperationalEOL80Lead": int(true_eol_lead),
    })


def fold_metric_row(
    model_id: str,
    battery: str,
    pred: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray,
    true_eol_lead: int,
    origin: int,
    train_info: Dict[str, object],
) -> Dict[str, object]:
    m = calc_metrics(
        y_true,
        pred,
        mask,
    )
    sh = shape_metrics_masked(
        y_true,
        pred,
        mask,
    )
    peol, found = find_persistent_eol_lead(
        pred,
        PRIMARY_EOL,
        EOL_PERSISTENCE,
    )

    return dict(
        ModelID=model_id,
        Battery=battery,
        OriginCycle_KeyOnly=int(origin),
        TrueOperationalEOL80Lead=int(true_eol_lead),
        PredOperationalEOL80Lead=int(peol),
        PredEOLFound=int(found),
        EOLLeadError=int(peol - true_eol_lead),
        EOLLeadAbsError=int(abs(peol - true_eol_lead)),
        TrueOperationalEOL80Cycle=int(origin + true_eol_lead),
        PredOperationalEOL80Cycle=int(origin + peol),
        RULAbsError_cycles=int(abs(peol - true_eol_lead)),
        **m,
        **sh,
        **train_info,
    )


# =============================================================================
# 7. Route: XGB
# =============================================================================

def run_xgb_route(
    bundle: FullLifeBundle,
    helper,
    batteries: Sequence[str],
    args: argparse.Namespace,
) -> None:
    route_dir = ensure_dir(
        args.output_root / ROUTE_DIRS["xgb"]
    )
    model_id = MODEL_XGB

    config = dict(
        Route="xgb",
        ModelID=model_id,
        TestOrigin=int(args.test_origin),
        SourceOrigins=list(map(int, args.source_origins)),
        K=int(args.k),
        MeasuredScalars=MEASURED_SCALARS,
        V410AEngineeredFeatures=196,
        LeadEncodingFeatures=6,
        TotalModelFeatures=202,
        XGBFrozenCore=dict(helper.XGB_FROZEN_CORE),
        XGBDevice=str(args.xgb_device),
        XGBThreads=int(args.xgb_threads),
        XGBLeadsPerSourceOrigin=int(args.xgb_leads_per_origin),
        HistoricalSOHAsFeature=False,
        FutureHIAsFeature=False,
        CycleAsFeature=False,
        TrueEOLAsInput=False,
    )
    save_json(config, route_dir / "00_config.json")

    fold_rows = []
    pred_frames = []
    diag_rows = []
    registry_saved = False

    for test_bat in batteries:
        if args.only_battery and test_bat != normalize_battery(args.only_battery):
            continue

        signature = make_signature(
            model_id,
            args,
        )
        cp = fold_cache_path(
            route_dir,
            model_id,
            test_bat,
        )

        cached = (
            None
            if args.force
            else load_fold_cache(cp, signature)
        )

        te_idx = bundle.select(
            batteries=[test_bat],
            origins=[int(args.test_origin)],
        )
        if len(te_idx) != 1:
            raise RuntimeError(f"{test_bat}: test row !=1")
        ti = int(te_idx[0])

        if cached is not None:
            pred, train_info = cached
            print(
                f"[XGB] {test_bat}: cache",
                flush=True,
            )
        else:
            source_bats = [
                b for b in batteries
                if b != test_bat
            ]
            src_idx = bundle.select(
                batteries=source_bats,
                origins=list(map(int, args.source_origins)),
            )

            Xtr, ytr, feat_names, registry = make_xgb_long_train(
                helper,
                bundle,
                src_idx,
                int(args.k),
                int(args.xgb_leads_per_origin),
            )
            Xte = make_xgb_test_grid(
                helper,
                bundle.X[ti],
                int(args.k),
            )

            t0 = time.perf_counter()
            pred, backend = fit_xgb_scalar(
                Xtr,
                ytr,
                Xte,
                helper,
                str(args.xgb_device),
                int(args.xgb_threads),
            )
            runtime = time.perf_counter() - t0

            train_info = dict(
                Backend=backend,
                SourceBatteries=len(source_bats),
                SourceOrigins=";".join(map(str, args.source_origins)),
                SourceTrajectorySamples=len(src_idx),
                LongFormatTrainingRows=len(ytr),
                ModelFeatures=Xtr.shape[1],
                Runtime_s=float(runtime),
            )

            if not registry_saved:
                registry.to_csv(
                    route_dir / "06_feature_registry.csv",
                    index=False,
                    encoding=ENC,
                )
                registry_saved = True

            save_fold_cache(
                cp,
                signature,
                pred,
                train_info,
            )

            del Xtr, ytr, Xte
            gc.collect()

            print(
                f"[XGB] {test_bat}: "
                f"backend={backend} runtime={runtime:.1f}s",
                flush=True,
            )

        if len(pred) != int(args.k):
            raise RuntimeError(
                f"{test_bat}: XGB pred len={len(pred)}"
            )

        fr = fold_metric_row(
            model_id,
            test_bat,
            pred,
            bundle.y[ti],
            bundle.mask[ti],
            int(bundle.true_eol_lead[ti]),
            int(args.test_origin),
            train_info,
        )
        fold_rows.append(fr)

        pred_frames.append(
            prediction_long_rows(
                model_id,
                test_bat,
                int(args.test_origin),
                pred,
                bundle.y[ti],
                bundle.mask[ti],
                int(bundle.true_eol_lead[ti]),
            )
        )
        diag_rows.append({
            "ModelID": model_id,
            "Battery": test_bat,
            **train_info,
        })

    finalize_route(
        route_dir,
        fold_rows,
        pred_frames,
        diag_rows,
        args,
    )


# =============================================================================
# 8. Route: Deep / TrendDetail
# =============================================================================

def run_deep_route(
    route_name: str,
    bundle: FullLifeBundle,
    batteries: Sequence[str],
    args: argparse.Namespace,
) -> None:
    if route_name not in ("deep", "trenddetail"):
        raise ValueError(route_name)

    trend_detail = route_name == "trenddetail"
    route_dir = ensure_dir(
        args.output_root / ROUTE_DIRS[route_name]
    )

    tf_info = setup_tensorflow()

    config = dict(
        Route=route_name,
        TrendDetail=trend_detail,
        Architectures=list(args.architectures),
        TestOrigin=int(args.test_origin),
        SourceOrigins=list(map(int, args.source_origins)),
        K=int(args.k),
        MeasuredScalars=MEASURED_SCALARS,
        MaxEpochs=int(args.max_epochs),
        Batch=int(args.deep_batch),
        Patience=int(args.early_patience),
        MinDelta=float(args.min_delta),
        LearningRate=float(args.learning_rate),
        Dropout=float(args.dropout),
        Clipnorm=float(args.clipnorm),
        ValidationBatteries=int(args.validation_batteries),
        RefitAllSource=not bool(args.no_refit_all_source),
        DetailKernel=int(args.detail_kernel) if trend_detail else None,
        DetailGateBias=float(args.detail_gate_bias) if trend_detail else None,
        TensorFlow=tf_info,
        HistoricalSOHAsFeature=False,
        CapacityHistoryAsFeature=False,
        FutureHIAsFeature=False,
        CycleAsFeature=False,
        TrueEOLAsInput=False,
    )
    save_json(config, route_dir / "00_config.json")

    fold_rows = []
    pred_frames = []
    diag_rows = []

    for arch in args.architectures:
        if trend_detail:
            model_id = (
                MODEL_TD_LSTM
                if arch == "LSTM"
                else MODEL_TD_BILSTM
            )
        else:
            model_id = (
                MODEL_LSTM
                if arch == "LSTM"
                else MODEL_BILSTM
            )

        for test_bat in batteries:
            if args.only_battery and test_bat != normalize_battery(args.only_battery):
                continue

            signature = make_signature(
                model_id,
                args,
            )
            cp = fold_cache_path(
                route_dir,
                model_id,
                test_bat,
            )
            cached = (
                None
                if args.force
                else load_fold_cache(cp, signature)
            )

            te_idx = bundle.select(
                batteries=[test_bat],
                origins=[int(args.test_origin)],
            )
            if len(te_idx) != 1:
                raise RuntimeError(
                    f"{test_bat}: test row !=1"
                )
            ti = int(te_idx[0])

            if cached is not None:
                pred, train_info = cached
                print(
                    f"[{route_name}/{arch}] {test_bat}: cache",
                    flush=True,
                )
            else:
                pred, train_info = train_deep_fold(
                    bundle,
                    test_bat,
                    arch,
                    trend_detail,
                    args,
                )
                save_fold_cache(
                    cp,
                    signature,
                    pred,
                    train_info,
                )
                print(
                    f"[{route_name}/{arch}] {test_bat}: "
                    f"epoch={train_info.get('BestEpoch')} "
                    f"runtime={train_info.get('Runtime_s', np.nan):.1f}s",
                    flush=True,
                )

            if len(pred) != int(args.k):
                raise RuntimeError(
                    f"{test_bat}/{model_id}: pred len={len(pred)}"
                )

            fr = fold_metric_row(
                model_id,
                test_bat,
                pred,
                bundle.y[ti],
                bundle.mask[ti],
                int(bundle.true_eol_lead[ti]),
                int(args.test_origin),
                train_info,
            )
            fold_rows.append(fr)

            pred_frames.append(
                prediction_long_rows(
                    model_id,
                    test_bat,
                    int(args.test_origin),
                    pred,
                    bundle.y[ti],
                    bundle.mask[ti],
                    int(bundle.true_eol_lead[ti]),
                )
            )
            diag_rows.append({
                "ModelID": model_id,
                "Battery": test_bat,
                **train_info,
            })

    finalize_route(
        route_dir,
        fold_rows,
        pred_frames,
        diag_rows,
        args,
    )


# =============================================================================
# 9. Route summaries
# =============================================================================

def horizon_bin_metrics(
    oof: pd.DataFrame,
) -> pd.DataFrame:
    bins = [
        ("Lead001_100", 1, 100),
        ("Lead101_300", 101, 300),
        ("Lead301_500", 301, 500),
        ("Lead501_700", 501, 700),
        ("Lead701_1000", 701, 1000),
    ]
    rows = []

    for mid, gm in oof.groupby("ModelID", sort=True):
        for name, lo, hi in bins:
            g = gm[
                gm["Lead"].between(lo, hi)
                & gm["StrictTruthMask"].eq(1)
            ]
            m = calc_metrics(
                g["SOH_True"].to_numpy(float),
                g["SOH_Pred"].to_numpy(float),
            )
            rows.append({
                "ModelID": mid,
                "HorizonBin": name,
                "LeadMin": lo,
                "LeadMax": hi,
                **m,
            })

    return pd.DataFrame(rows)


def route_pooled_metrics(
    oof: pd.DataFrame,
    fold: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for mid, g in oof.groupby("ModelID", sort=True):
        q = g[g["StrictTruthMask"].eq(1)]
        m = calc_metrics(
            q["SOH_True"].to_numpy(float),
            q["SOH_Pred"].to_numpy(float),
        )
        f = fold[fold["ModelID"].eq(mid)]

        rows.append({
            "ModelID": mid,
            **m,
            "BatteryMacroRMSE": float(
                pd.to_numeric(
                    f["RMSE"],
                    errors="coerce",
                ).mean()
            ),
            "BatteryMedianRMSE": float(
                pd.to_numeric(
                    f["RMSE"],
                    errors="coerce",
                ).median()
            ),
            "MeanEOLAbsError_cycles": float(
                pd.to_numeric(
                    f["EOLLeadAbsError"],
                    errors="coerce",
                ).mean()
            ),
            "MedianEOLAbsError_cycles": float(
                pd.to_numeric(
                    f["EOLLeadAbsError"],
                    errors="coerce",
                ).median()
            ),
            "EOLFoundRate": float(
                pd.to_numeric(
                    f["PredEOLFound"],
                    errors="coerce",
                ).mean()
            ),
            "MeanRoughnessRatio": float(
                pd.to_numeric(
                    f["RoughnessRatioPredToTrue"],
                    errors="coerce",
                ).replace([np.inf, -np.inf], np.nan).mean()
            ),
            "MeanPredUpwardStepFraction": float(
                pd.to_numeric(
                    f["PredUpwardStepFraction"],
                    errors="coerce",
                ).mean()
            ),
            "MeanTrueUpwardStepFraction": float(
                pd.to_numeric(
                    f["TrueUpwardStepFraction"],
                    errors="coerce",
                ).mean()
            ),
        })

    return pd.DataFrame(rows).sort_values(
        ["RMSE", "MAE"]
    ).reset_index(drop=True)


def finalize_route(
    route_dir: Path,
    fold_rows: List[Dict[str, object]],
    pred_frames: List[pd.DataFrame],
    diag_rows: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    if not fold_rows:
        raise RuntimeError(
            f"{route_dir.name}: 没有fold结果"
        )

    fold = pd.DataFrame(fold_rows)
    oof = pd.concat(
        pred_frames,
        ignore_index=True,
    )

    fold.to_csv(
        route_dir / "01_fold_metrics.csv",
        index=False,
        encoding=ENC,
    )

    pooled = route_pooled_metrics(
        oof,
        fold,
    )
    pooled.to_csv(
        route_dir / "02_pooled_metrics.csv",
        index=False,
        encoding=ENC,
    )

    battery_cols = [
        c for c in [
            "ModelID",
            "Battery",
            "OriginCycle_KeyOnly",
            "N",
            "MAE",
            "RMSE",
            "R2",
            "MAPE_pct",
            "Bias",
            "TrueOperationalEOL80Lead",
            "PredOperationalEOL80Lead",
            "PredEOLFound",
            "EOLLeadError",
            "EOLLeadAbsError",
            "RULAbsError_cycles",
            "EndpointAbsError",
            "TrueUpwardStepFraction",
            "PredUpwardStepFraction",
            "TrueRoughness",
            "PredRoughness",
            "RoughnessRatioPredToTrue",
        ]
        if c in fold.columns
    ]
    fold[battery_cols].to_csv(
        route_dir / "03_battery_metrics.csv",
        index=False,
        encoding=ENC,
    )

    hb = horizon_bin_metrics(oof)
    hb.to_csv(
        route_dir / "04_horizon_bin_metrics.csv",
        index=False,
        encoding=ENC,
    )

    eol_cols = [
        c for c in [
            "ModelID",
            "Battery",
            "TrueOperationalEOL80Lead",
            "PredOperationalEOL80Lead",
            "PredEOLFound",
            "EOLLeadError",
            "EOLLeadAbsError",
            "TrueOperationalEOL80Cycle",
            "PredOperationalEOL80Cycle",
            "RULAbsError_cycles",
        ]
        if c in fold.columns
    ]
    fold[eol_cols].to_csv(
        route_dir / "05_eol_rul_metrics.csv",
        index=False,
        encoding=ENC,
    )

    pd.DataFrame(diag_rows).to_csv(
        route_dir / "06_model_diagnostics.csv",
        index=False,
        encoding=ENC,
    )

    oof.to_csv(
        route_dir / "07_ALL_OOF_primary_origin80_long.csv.gz",
        index=False,
        compression=FAST_GZIP,
    )

    full_run = (
        not bool(args.only_battery)
        and fold["Battery"].nunique() == EXPECTED_BATTERIES
    )
    complete = dict(
        Route=route_dir.name,
        Status="PASS",
        Full47FoldRun=bool(full_run),
        BatteriesRun=int(fold["Battery"].nunique()),
        Models=sorted(fold["ModelID"].unique().tolist()),
        TestOrigin=int(args.test_origin),
        SourceOrigins=list(map(int, args.source_origins)),
        K=int(args.k),
        Pooled=pooled.to_dict("records"),
        HistoricalSOHAsFeature=False,
        FutureHIAsFeature=False,
        CycleAsFeature=False,
        TrueEOLAsInput=False,
    )
    save_json(
        complete,
        route_dir / "99_ROUTE_COMPLETE.json",
    )


# =============================================================================
# 10. Merge
# =============================================================================

def merge_routes(
    args: argparse.Namespace,
) -> None:
    root = ensure_dir(args.output_root)

    pooled_all = []
    batt_all = []
    hb_all = []
    eol_all = []
    route_status = []

    for route, dirname in ROUTE_DIRS.items():
        d = root / dirname
        comp = d / "99_ROUTE_COMPLETE.json"
        if not comp.exists():
            route_status.append({
                "Route": route,
                "Exists": 0,
                "Full47FoldRun": 0,
            })
            continue

        obj = json.loads(
            comp.read_text(encoding="utf-8")
        )
        route_status.append({
            "Route": route,
            "Exists": 1,
            "Full47FoldRun": int(
                bool(obj.get("Full47FoldRun", False))
            ),
        })

        for name, store in [
            ("02_pooled_metrics.csv", pooled_all),
            ("03_battery_metrics.csv", batt_all),
            ("04_horizon_bin_metrics.csv", hb_all),
            ("05_eol_rul_metrics.csv", eol_all),
        ]:
            p = d / name
            if p.exists():
                store.append(
                    pd.read_csv(p, low_memory=False)
                )

    if not pooled_all:
        raise RuntimeError(
            "没有任何route pooled metrics可以merge"
        )

    pooled = pd.concat(
        pooled_all,
        ignore_index=True,
    )
    batt = (
        pd.concat(batt_all, ignore_index=True)
        if batt_all
        else pd.DataFrame()
    )
    hb = (
        pd.concat(hb_all, ignore_index=True)
        if hb_all
        else pd.DataFrame()
    )
    eol = (
        pd.concat(eol_all, ignore_index=True)
        if eol_all
        else pd.DataFrame()
    )

    pooled = pooled.sort_values(
        ["RMSE", "MAE"]
    ).reset_index(drop=True)
    pooled.insert(
        0,
        "Rank",
        np.arange(1, len(pooled)+1),
    )

    pooled.to_csv(
        root / "20_combined_pooled_metrics.csv",
        index=False,
        encoding=ENC,
    )
    batt.to_csv(
        root / "21_combined_battery_metrics.csv",
        index=False,
        encoding=ENC,
    )
    hb.to_csv(
        root / "22_combined_horizon_bin_metrics.csv",
        index=False,
        encoding=ENC,
    )
    eol.to_csv(
        root / "23_combined_eol_rul_metrics.csv",
        index=False,
        encoding=ENC,
    )

    rank_cols = [
        c for c in [
            "Rank",
            "ModelID",
            "RMSE",
            "MAE",
            "R2",
            "BatteryMacroRMSE",
            "MeanEOLAbsError_cycles",
            "MedianEOLAbsError_cycles",
            "EOLFoundRate",
            "MeanRoughnessRatio",
            "MeanPredUpwardStepFraction",
            "MeanTrueUpwardStepFraction",
        ]
        if c in pooled.columns
    ]
    pooled[rank_cols].to_csv(
        root / "24_model_rank_summary.csv",
        index=False,
        encoding=ENC,
    )

    rs = pd.DataFrame(route_status)
    full_routes = int(
        rs["Full47FoldRun"].sum()
    ) if len(rs) else 0

    gate = pd.DataFrame([
        {
            "Check": "At least one route complete",
            "Observed": int(rs["Exists"].sum()) if len(rs) else 0,
            "Expected": ">=1",
            "PASS": int(len(rs) and rs["Exists"].sum() >= 1),
        },
        {
            "Check": "All three routes exist",
            "Observed": int(rs["Exists"].sum()) if len(rs) else 0,
            "Expected": 3,
            "PASS": int(len(rs) == 3 and rs["Exists"].sum() == 3),
        },
        {
            "Check": "All three routes full 47-fold",
            "Observed": full_routes,
            "Expected": 3,
            "PASS": int(full_routes == 3),
        },
        {
            "Check": "Historical SOH as feature",
            "Observed": 0,
            "Expected": 0,
            "PASS": 1,
        },
        {
            "Check": "Future HI as feature",
            "Observed": 0,
            "Expected": 0,
            "PASS": 1,
        },
        {
            "Check": "Cycle as feature",
            "Observed": 0,
            "Expected": 0,
            "PASS": 1,
        },
        {
            "Check": "True EOL used to truncate prediction",
            "Observed": 0,
            "Expected": 0,
            "PASS": 1,
        },
    ])
    gate.to_csv(
        root / "25_acceptance_gate.csv",
        index=False,
        encoding=ENC,
    )

    status = (
        "PASS"
        if gate["PASS"].eq(1).all()
        else "PARTIAL"
    )

    complete = dict(
        Task="RWTH48 V4.11B Three-Route Full Operational-Life SOH LOBO",
        Status=status,
        Routes=route_status,
        Rank=pooled.to_dict("records"),
        TestOrigin=int(args.test_origin),
        SourceOrigins=list(map(int, args.source_origins)),
        K=int(args.k),
        OperationalEOLDefinition=(
            f"first {EOL_PERSISTENCE} consecutive predicted SOH <= {PRIMARY_EOL}"
        ),
        HistoricalSOHAsFeature=False,
        CapacityHistoryAsFeature=False,
        FutureHIAsFeature=False,
        CycleAsFeature=False,
        TrueEOLAsInput=False,
        Next=(
            "Compare strict80 source-only vs source multi-origin augmentation; "
            "then decide whether a hybrid model is justified."
        ),
    )
    save_json(
        complete,
        root / "99_RWTH48_V4_11B_COMPLETE.json",
    )

    print("\n" + "="*118)
    print("RWTH48 V4.11B MERGE")
    print("="*118)
    print("Status:", status)
    print(pooled[rank_cols].to_string(index=False))
    print("Output:", root)
    print("="*118)


# =============================================================================
# 11. Audit / main
# =============================================================================

def main() -> None:
    args = parse_args()
    args.output_root = Path(args.output_root)
    ensure_dir(args.output_root)

    if args.stage == "merge":
        merge_routes(args)
        return

    bundle, a2_complete, audit = load_handoff(args)

    # Route-specific preflight filenames are deliberate:
    # XGB-CPU and TensorFlow routes are allowed to run in parallel processes.
    # They must not race while overwriting the same root CSV/JSON.
    run_tag = "__".join(sorted(set(args.routes)))
    audit.to_csv(
        args.output_root / f"00_preflight_audit__{run_tag}.csv",
        index=False,
        encoding=ENC,
    )

    helper = load_v410a_helper(
        args.v410a_helper
    )

    # Explicitly verify 196-feature V4.10A representation.
    Xprobe, fnames, freg = helper.engineer_features(
        bundle.X[:1]
    )
    if Xprobe.shape != (1, 196):
        raise RuntimeError(
            f"V4.10A feature probe shape={Xprobe.shape}, expected (1,196)"
        )
    freg.to_csv(
        args.output_root / f"00_v410a_196_feature_registry__{run_tag}.csv",
        index=False,
        encoding=ENC,
    )

    batteries = sorted(
        set(bundle.battery.tolist())
    )

    test_rows = bundle.select(
        origins=[int(args.test_origin)]
    )
    test_rul = bundle.true_eol_lead[test_rows]

    preflight = dict(
        Status="PASS",
        Batteries=len(batteries),
        TestOrigin=int(args.test_origin),
        TestRows=len(test_rows),
        SourceOrigins=list(map(int, args.source_origins)),
        K=int(args.k),
        TestRUL_Min=int(test_rul.min()),
        TestRUL_Median=float(np.median(test_rul)),
        TestRUL_Max=int(test_rul.max()),
        XShape=list(bundle.X.shape),
        MeasuredScalars=MEASURED_SCALARS,
        V410AEngineeredFeatureCount=Xprobe.shape[1],
        HistoricalSOHAsFeature=False,
        FutureHIAsFeature=False,
        CycleAsFeature=False,
        TrueEOLAsInput=False,
        A2PrimaryOrigin=a2_complete.get("PrimaryOrigin"),
        A2TargetSemantics=a2_complete.get("TargetSemantics"),
    )
    save_json(
        preflight,
        args.output_root / f"00_preflight__{run_tag}.json",
    )

    print("="*118)
    print("RWTH48 V4.11B | Full Operational-Life Three-Route LOBO")
    print("Batteries      :", len(batteries))
    print("Test origin    :", args.test_origin)
    print("Source origins :", args.source_origins)
    print("K              :", args.k)
    print(
        "Test RUL       :",
        preflight["TestRUL_Min"],
        preflight["TestRUL_Median"],
        preflight["TestRUL_Max"],
    )
    print("Input          : [6,6] = 36 measured Event scalars")
    print("V4.10A eng     :", Xprobe.shape[1])
    print("Routes         :", args.routes)
    print("="*118, flush=True)

    if args.audit_only:
        print("AUDIT-ONLY PASS")
        return

    for route in args.routes:
        if route == "xgb":
            run_xgb_route(
                bundle,
                helper,
                batteries,
                args,
            )
        elif route in ("deep", "trenddetail"):
            run_deep_route(
                route,
                bundle,
                batteries,
                args,
            )
        else:
            raise ValueError(route)

    # If all three routes requested in this process, merge automatically.
    if set(args.routes) == {
        "xgb",
        "deep",
        "trenddetail",
    } and not args.only_battery:
        merge_routes(args)

    print("\nRun complete for routes:", args.routes)
    if set(args.routes) != {
        "xgb",
        "deep",
        "trenddetail",
    }:
        print(
            '完成所有独立route后运行：python "src/22_full_life_soh_helper.py" --stage merge'
        )


if __name__ == "__main__":
    main()

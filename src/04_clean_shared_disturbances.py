# -*- coding: utf-8 -*-
r"""
RWTH48 V4.5C-FIX MINIMAL IMPACT
跨电池日历同步共同扰动识别 + 最小点级删除 + 严格时序分段
================================================================================

目标
----
解决两类问题，同时尽量减少对原始数据的冲击：

A. 已确认的 RPT / 新 ZYK session 恢复效应
   - 每个 session 前若干 cycle（默认 5）从主预测数据排除；
   - 不修改原始值；
   - 任何时序窗口不得跨 session。

B. session 内部跨电池同步的 common-mode experimental disturbance
   例如约 2013-03-07/08、2013-04-26 这类在很多电池上同时发生的异常。
   这里不再“整段删除”，而采用 MINIMAL-IMPACT 策略：

   1) 先按 SourcePart 文件时间 + session 内 cycle 周期估计每个 cycle 的实验日历时间；
   2) 以 2 h calendar bin 做跨电池同步检测；
   3) 只有当一个 bin 内很多电池同时出现异常 dSOH，才认定 common-mode event bin；
   4) 在 event bin 内，仍然只删除“该 cycle 自己也异常”的行；
   5) 不删除 event bin 内正常安静 cycle；
   6) 不做整段 interpolation / smoothing / monotonic correction；
   7) 删除后在缺口处断开 sequence segment，防止 LSTM 跨异常点构造历史窗口。

为什么这样更保守
----------------
- common-mode 识别使用“群体同步性”避免把单电池真实 knee 错删；
- row-level 删除又要求“本行自身异常”，避免整个时间窗一刀切；
- 持续性但非同步的真实老化变化不会因为幅度大就自动删除；
- 所有原始数值保留在 audit 中；
- 对 future physical features（V/I/T/HI）只通过 Battery+GlobalCycle removal key 同步排除，
  不对保留值作任何数值修正。

主输入
------
V4.5A:
workspace\RWTH48_V4_5A_CyclingSOH重建与RPT交叉验证
    \05_dense_cycling_vs_RPT_cycle_level.csv.gz

V4.5B:
workspace\RWTH48_V4_5B_协议扰动审计与鲁棒CyclingSOH
    \03_CLEAN_CyclingSOH_MODEL_READY.csv.gz

主输出
------
workspace\RWTH48_V4_5C_FIX_最小冲击共同扰动处理

00_config.json
01_calendar_bin_common_mode_audit.csv
02_detected_common_mode_event_bins.csv
03_cycle_level_common_mode_audit.csv.gz
04_MINIMAL_COMMON_MODE_REMOVAL_KEYS.csv
05_minimal_removed_common_mode_cycles.csv.gz
06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz
07_feature_handling_policy.csv
08_removal_reason_summary.csv
09_before_after_delta_summary.csv
10_window_availability.csv
11_battery_impact_summary.csv
12_structural_acceptance_gate.csv
13_scientific_evidence_gate.csv
Figures/
99_RWTH48_V4_5C_FIX_COMPLETE.json

下游物理参数处理
----------------
只要 Battery+GlobalCycle 出现在：
    04_MINIMAL_COMMON_MODE_REMOVAL_KEYS.csv
则该 cycle 对应的：
    SOH、capacity、V、I、T、power、resistance proxy、所有 HI
全部从主预测数据排除。

未出现在 removal key 中的 cycle：
    所有物理参数保持原值，不平滑、不替换、不校正。

SourcePart、CalendarBin、ZYKSessionIndex、CycleWithinSession、RPT flag：
    仅用于 QC / 分段 / 审计；
    不建议作为主 BiLSTM health features。

默认检测门槛
------------
- Calendar bin: 2 h
- 每 bin 至少 10 个 active batteries
- 每只电池 strong dSOH:
      |dSOH - battery_median| > max(0.015, 12 * robust_sigma)
- common-mode event bin:
      strong-cell fraction >= 0.50 且 dominant-sign/active >= 0.45
      OR strong-cell fraction >= 0.65 且 >=2 个独立 I/V 物理量有群体异常
      OR strong-cell fraction >= 0.80（极强同步证据）
- row-level removal:
      必须是 strong dSOH cycle；
      且位于 detected event bin；
      混合方向事件还要求该行至少 1 个独立 I/V 物理量明显异常，
      或 |dSOH| >= 0.03。

这些门槛都是保守的；V4.5A 正常逐 cycle P95 |dSOH| 约 0.0026，
因此 0.015 floor 已明显高于自然小幅波动。

运行
----
python "src/04_clean_shared_disturbances.py"

只生成表：
python "src/04_clean_shared_disturbances.py" --skip-figures

敏感性：
python "src/04_clean_shared_disturbances.py" --calendar-bin-hours 1
python "src/04_clean_shared_disturbances.py" --calendar-bin-hours 3
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Paths / frozen defaults
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")

DEFAULT_V45A_ROOT = BASE / "RWTH48_V4_5A_CyclingSOH重建与RPT交叉验证"
DEFAULT_V45B_ROOT = BASE / "RWTH48_V4_5B_协议扰动审计与鲁棒CyclingSOH"
DEFAULT_OUTPUT_ROOT = BASE / "RWTH48_V4_5C_FIX_最小冲击共同扰动处理"

V45A_FILE = "05_dense_cycling_vs_RPT_cycle_level.csv.gz"
V45B_FILE = "03_CLEAN_CyclingSOH_MODEL_READY.csv.gz"

TARGET = "CyclingSOH_Charge_Canonical"

RECOVERY_CYCLES = 5

CALENDAR_BIN_HOURS_DEFAULT = 2
MIN_ACTIVE_BATTERIES = 10

STRONG_DSOH_FLOOR = 0.015
STRONG_DSOH_MAD_MULT = 12.0

MODERATE_DSOH_FLOOR = 0.0075
MODERATE_DSOH_MAD_MULT = 8.0

EVENT_STRONG_CELL_FRACTION = 0.50
EVENT_DOMINANT_SIGN_ACTIVE_FRACTION = 0.45
EVENT_MIXED_STRONG_FRACTION = 0.65
EVENT_EXTREME_STRONG_FRACTION = 0.80

PHYS_ROW_Z_THRESHOLD = 5.0
PHYS_BIN_MEDIAN_ABS_Z_THRESHOLD = 5.0
MIN_BIN_PHYS_SUPPORT_FEATURES = 2

ROW_LARGE_DSOH_OVERRIDE = 0.030

# Only independent-ish I/V protocol variables are used as supporting evidence.
# Capacity is deliberately not counted as independent evidence because Cycling-SOH
# is itself normalized from charge capacity.
PHYS_SUPPORT_FEATURES = [
    "Charge_I_Mean_A",
    "Discharge_I_Mean_A",
    "Charge_V_Min",
    "Discharge_V_Max",
]

# For output/audit only
AUX_PHYS_FEATURES = [
    "Charge_Capacity_Ah_Primary",
    "Discharge_Capacity_Ah_Primary",
    "Charge_Capacity_CC4_Ah_Integral",
    "Discharge_Capacity_CC4_Ah_Integral",
    "Charge_Duration_s",
    "Discharge_Duration_s",
    "Charge_V_Min",
    "Charge_V_Max",
    "Discharge_V_Min",
    "Discharge_V_Max",
    "Charge_I_Mean_A",
    "Discharge_I_Mean_A",
    "WindowCapacityChargeDischargeRelGap",
]

WINDOW_LENGTHS = (20, 30, 50)
FORECAST_HORIZONS = (1, 10, 50, 100)

# Scientific gates
MAX_COMMON_MODE_REMOVAL_RATE_FROM_V45B = 0.02
MIN_V45B_KEY_MATCH_RATE = 0.999
MIN_NATURAL_KEEP_RATE = 0.995
MIN_CELLS_H100 = 45
MIN_EVENT_ACTIVE_BATTERIES = 10
MIN_EVENT_MEDIAN_STRONG_FRACTION = 0.50

FAST_GZIP = {"method": "gzip", "compresslevel": 1}
RANDOM_SEED = 20260820


# =============================================================================
# Utility
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWTH48 V4.5C-FIX minimal-impact cross-cell common-mode disturbance handling."
    )
    p.add_argument("--v45a-root", type=Path, default=DEFAULT_V45A_ROOT)
    p.add_argument("--v45b-root", type=Path, default=DEFAULT_V45B_ROOT)
    p.add_argument("--v45a-file", type=Path, default=None)
    p.add_argument("--v45b-file", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument(
        "--calendar-bin-hours",
        type=int,
        default=CALENDAR_BIN_HOURS_DEFAULT,
        choices=[1, 2, 3, 4],
    )
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: object, p: Path) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(x) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def robust_med_sigma(s: pd.Series) -> Tuple[float, float]:
    v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0, 0.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med, 1.4826 * mad


def pearson_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3 or x[m].std() < 1e-12 or y[m].std() < 1e-12:
        return np.nan
    return float(x[m].corr(y[m]))


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return np.nan
    xr = x[m].rank(method="average")
    yr = y[m].rank(method="average")
    if xr.std() < 1e-12 or yr.std() < 1e-12:
        return np.nan
    return float(xr.corr(yr))


def parse_source_start(s: object) -> pd.Timestamp:
    m = re.search(r"_(\d{8})_(\d{6})_", str(s))
    if not m:
        return pd.NaT
    return pd.to_datetime(
        m.group(1) + m.group(2),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )


# =============================================================================
# Input and session metadata
# =============================================================================

def load_inputs(v45a_path: Path, v45b_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not v45a_path.exists():
        raise FileNotFoundError(v45a_path)
    if not v45b_path.exists():
        raise FileNotFoundError(v45b_path)

    a = pd.read_csv(v45a_path, compression="infer", low_memory=False)
    b = pd.read_csv(v45b_path, compression="infer", low_memory=False)

    req_a = {
        "Battery", "GlobalCycle", "SourcePart",
        TARGET, "CyclingSOH_TargetUsable",
        "FullAgingCycleValid",
        "RPT_ExactMeasuredAnchor", "RPT_SOH_Measured",
        "Charge_Duration_s", "Discharge_Duration_s",
        *PHYS_SUPPORT_FEATURES,
    }
    missing = sorted(req_a - set(a.columns))
    if missing:
        raise RuntimeError(f"V4.5A missing required columns: {missing}")

    req_b = {"Battery", "GlobalCycle"}
    missing = sorted(req_b - set(b.columns))
    if missing:
        raise RuntimeError(f"V4.5B missing required columns: {missing}")

    for d in (a, b):
        d["Battery"] = d["Battery"].astype(str)
        d["GlobalCycle"] = pd.to_numeric(d["GlobalCycle"], errors="coerce")
        d.dropna(subset=["GlobalCycle"], inplace=True)
        d["GlobalCycle"] = d["GlobalCycle"].astype(int)

    if a.duplicated(["Battery", "GlobalCycle"]).any():
        raise RuntimeError("V4.5A contains duplicate Battery+GlobalCycle")
    if b.duplicated(["Battery", "GlobalCycle"]).any():
        raise RuntimeError("V4.5B contains duplicate Battery+GlobalCycle")

    return a, b


def add_session_calendar_metadata(a: pd.DataFrame, bin_hours: int) -> pd.DataFrame:
    x = a.sort_values(["Battery", "GlobalCycle"]).copy().reset_index(drop=True)

    x["CycleWithinZYKSession"] = (
        x.groupby(["Battery", "SourcePart"], sort=False).cumcount() + 1
    ).astype(int)

    x["SourcePartStartTime"] = x["SourcePart"].map(parse_source_start)

    # Use actual cycle duration information to estimate a robust nominal period per session.
    cycle_duration = (
        pd.to_numeric(x["Charge_Duration_s"], errors="coerce")
        + pd.to_numeric(x["Discharge_Duration_s"], errors="coerce")
    )
    x["_CycleTotalDuration_s"] = cycle_duration

    session_period = (
        x.groupby(["Battery", "SourcePart"], sort=False)["_CycleTotalDuration_s"]
        .transform("median")
    )
    # Fallback to 1 h only if metadata is missing/unreasonable.
    session_period = pd.to_numeric(session_period, errors="coerce")
    session_period = session_period.where(
        session_period.between(3000.0, 4200.0),
        3600.0,
    )
    x["NominalCyclePeriod_s"] = session_period

    x["EstimatedCycleStartTime"] = (
        x["SourcePartStartTime"]
        + pd.to_timedelta(
            (x["CycleWithinZYKSession"] - 1)
            * x["NominalCyclePeriod_s"],
            unit="s",
        )
    )

    # pandas floor supports e.g. 2h
    x["CalendarBin"] = x["EstimatedCycleStartTime"].dt.floor(f"{bin_hours}h")

    sess = (
        x.groupby(["Battery", "SourcePart"], as_index=False)
        .agg(SessionStartCycle=("GlobalCycle", "min"))
        .sort_values(["Battery", "SessionStartCycle"])
    )
    sess["ZYKSessionIndex"] = (
        sess.groupby("Battery").cumcount() + 1
    ).astype(int)

    x = x.merge(
        sess[["Battery", "SourcePart", "ZYKSessionIndex"]],
        on=["Battery", "SourcePart"],
        how="left",
        validate="many_to_one",
    )

    x["_PrevSourcePart"] = x.groupby("Battery", sort=False)["SourcePart"].shift()
    x["_PrevGlobalCycle"] = x.groupby("Battery", sort=False)["GlobalCycle"].shift()

    x["SameSessionConsecutiveFlag"] = (
        x["SourcePart"].astype(str).eq(x["_PrevSourcePart"].astype(str))
        & (
            x["GlobalCycle"]
            - pd.to_numeric(x["_PrevGlobalCycle"], errors="coerce")
        ).eq(1)
    ).astype(np.int8)

    x["NewZYKSessionFlag"] = (
        x["_PrevSourcePart"].notna()
        & x["SourcePart"].astype(str).ne(x["_PrevSourcePart"].astype(str))
    ).astype(np.int8)

    return x.drop(
        columns=["_PrevSourcePart", "_PrevGlobalCycle", "_CycleTotalDuration_s"],
        errors="ignore",
    )


def mark_v45b_retained(x: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    key = b[["Battery", "GlobalCycle"]].drop_duplicates().copy()
    key["V45B_Retained"] = np.int8(1)

    y = x.merge(
        key,
        on=["Battery", "GlobalCycle"],
        how="left",
        validate="one_to_one",
    )
    y["V45B_Retained"] = (
        pd.to_numeric(y["V45B_Retained"], errors="coerce")
        .fillna(0).astype(np.int8)
    )
    return y


# =============================================================================
# Row deltas and robust per-battery thresholds
# =============================================================================

def add_delta_features(x: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = x.sort_values(["Battery", "GlobalCycle"]).copy().reset_index(drop=True)

    delta_features = [TARGET] + PHYS_SUPPORT_FEATURES

    for f in delta_features:
        y[f"Delta__{f}"] = (
            pd.to_numeric(y[f], errors="coerce")
            - pd.to_numeric(
                y.groupby("Battery", sort=False)[f].shift(),
                errors="coerce",
            )
        )
        y.loc[
            y["SameSessionConsecutiveFlag"].ne(1),
            f"Delta__{f}",
        ] = np.nan

    threshold_rows = []

    # only normal candidate population for estimating robust scales
    scale_population = (
        y["V45B_Retained"].eq(1)
        & y["SameSessionConsecutiveFlag"].eq(1)
        & (y["CycleWithinZYKSession"] > RECOVERY_CYCLES)
    )

    for b, g in y.groupby("Battery", sort=True):
        gm = g.loc[scale_population.loc[g.index]]

        row = {"Battery": b}

        med, sig = robust_med_sigma(gm[f"Delta__{TARGET}"])
        row["DeltaSOH_Median"] = med
        row["DeltaSOH_RobustSigma"] = sig
        row["StrongDeltaSOHThreshold"] = max(
            STRONG_DSOH_FLOOR,
            abs(med) + STRONG_DSOH_MAD_MULT * sig,
        )
        row["ModerateDeltaSOHThreshold"] = max(
            MODERATE_DSOH_FLOOR,
            abs(med) + MODERATE_DSOH_MAD_MULT * sig,
        )

        for f in PHYS_SUPPORT_FEATURES:
            pm, ps = robust_med_sigma(gm[f"Delta__{f}"])
            row[f"{f}__DeltaMedian"] = pm
            row[f"{f}__DeltaRobustSigma"] = max(ps, 1e-12)

        threshold_rows.append(row)

    thresholds = pd.DataFrame(threshold_rows)

    y = y.merge(
        thresholds,
        on="Battery",
        how="left",
        validate="many_to_one",
    )

    dsoh = pd.to_numeric(y[f"Delta__{TARGET}"], errors="coerce")
    med = pd.to_numeric(y["DeltaSOH_Median"], errors="coerce")
    strong_thr = pd.to_numeric(y["StrongDeltaSOHThreshold"], errors="coerce")
    moderate_thr = pd.to_numeric(y["ModerateDeltaSOHThreshold"], errors="coerce")

    y["StrongDeltaSOHFlag"] = (
        y["V45B_Retained"].eq(1)
        & y["SameSessionConsecutiveFlag"].eq(1)
        & (y["CycleWithinZYKSession"] > RECOVERY_CYCLES)
        & ((dsoh - med).abs() > strong_thr)
    ).astype(np.int8)

    y["ModerateDeltaSOHFlag"] = (
        y["V45B_Retained"].eq(1)
        & y["SameSessionConsecutiveFlag"].eq(1)
        & (y["CycleWithinZYKSession"] > RECOVERY_CYCLES)
        & ((dsoh - med).abs() > moderate_thr)
    ).astype(np.int8)

    # Independent I/V row support.
    support_count = np.zeros(len(y), dtype=np.int16)

    for f in PHYS_SUPPORT_FEATURES:
        dm = pd.to_numeric(y[f"{f}__DeltaMedian"], errors="coerce")
        ds = pd.to_numeric(y[f"{f}__DeltaRobustSigma"], errors="coerce").clip(lower=1e-12)
        dv = pd.to_numeric(y[f"Delta__{f}"], errors="coerce")
        z = (dv - dm) / ds
        y[f"DeltaZ__{f}"] = z
        support_count += (np.abs(z.to_numpy(float)) >= PHYS_ROW_Z_THRESHOLD).astype(np.int16)

    y["RowIndependentPhysicalSupportCount"] = support_count

    return y, thresholds


# =============================================================================
# Calendar-bin cross-cell common-mode detection
# =============================================================================

def build_calendar_bin_audit(y: pd.DataFrame) -> pd.DataFrame:
    base = y[
        y["V45B_Retained"].eq(1)
        & (y["CycleWithinZYKSession"] > RECOVERY_CYCLES)
        & y["CalendarBin"].notna()
    ].copy()

    rows = []

    for t, g in base.groupby("CalendarBin", sort=True):
        active_batteries = int(g["Battery"].nunique())

        strong = g[g["StrongDeltaSOHFlag"].eq(1)]
        strong_batteries = int(strong["Battery"].nunique())

        neg_batteries = int(
            strong[
                pd.to_numeric(strong[f"Delta__{TARGET}"], errors="coerce") < 0
            ]["Battery"].nunique()
        )
        pos_batteries = int(
            strong[
                pd.to_numeric(strong[f"Delta__{TARGET}"], errors="coerce") > 0
            ]["Battery"].nunique()
        )

        strong_frac = (
            strong_batteries / active_batteries
            if active_batteries else 0.0
        )
        dominant_active_frac = (
            max(neg_batteries, pos_batteries) / active_batteries
            if active_batteries else 0.0
        )
        sign_coherence_among_strong = (
            max(neg_batteries, pos_batteries) / strong_batteries
            if strong_batteries else 0.0
        )

        row = {
            "CalendarBin": t,
            "ActiveBatteries": active_batteries,
            "StrongDeltaBatteries": strong_batteries,
            "StrongNegativeBatteries": neg_batteries,
            "StrongPositiveBatteries": pos_batteries,
            "StrongCellFraction": strong_frac,
            "DominantSignActiveFraction": dominant_active_frac,
            "SignCoherenceAmongStrong": sign_coherence_among_strong,
            "MedianDeltaSOH": safe_float(
                pd.to_numeric(g[f"Delta__{TARGET}"], errors="coerce").median()
            ),
            "MedianAbsDeltaSOH": safe_float(
                pd.to_numeric(g[f"Delta__{TARGET}"], errors="coerce").abs().median()
            ),
        }

        phys_support = 0
        for f in PHYS_SUPPORT_FEATURES:
            z = pd.to_numeric(g[f"DeltaZ__{f}"], errors="coerce").abs()
            medz = safe_float(z.median())
            row[f"MedianAbsDeltaZ__{f}"] = medz
            if np.isfinite(medz) and medz >= PHYS_BIN_MEDIAN_ABS_Z_THRESHOLD:
                phys_support += 1

        row["BinIndependentPhysicalSupportCount"] = int(phys_support)

        # Event logic:
        # 1) coherent cross-cell same-direction event
        coherent = (
            active_batteries >= MIN_ACTIVE_BATTERIES
            and strong_frac >= EVENT_STRONG_CELL_FRACTION
            and dominant_active_frac >= EVENT_DOMINANT_SIGN_ACTIVE_FRACTION
        )

        # 2) mixed sign but extensive synchronized physical disturbance
        mixed_with_physics = (
            active_batteries >= MIN_ACTIVE_BATTERIES
            and strong_frac >= EVENT_MIXED_STRONG_FRACTION
            and phys_support >= MIN_BIN_PHYS_SUPPORT_FEATURES
        )

        # 3) nearly universal SOH disturbance; group synchrony itself is strong evidence
        extreme_sync = (
            active_batteries >= MIN_ACTIVE_BATTERIES
            and strong_frac >= EVENT_EXTREME_STRONG_FRACTION
        )

        event = coherent or mixed_with_physics or extreme_sync

        if extreme_sync:
            event_type = "EXTREME_SYNC"
        elif coherent:
            event_type = "COHERENT_SIGN"
        elif mixed_with_physics:
            event_type = "MIXED_SIGN_PHYSICAL"
        else:
            event_type = "NONE"

        row["CommonModeEventBinFlag"] = int(event)
        row["CommonModeEventType"] = event_type
        rows.append(row)

    return pd.DataFrame(rows)


def assign_event_clusters(bin_audit: pd.DataFrame, bin_hours: int) -> pd.DataFrame:
    e = bin_audit[
        bin_audit["CommonModeEventBinFlag"].eq(1)
    ].copy().sort_values("CalendarBin").reset_index(drop=True)

    if e.empty:
        e["CommonModeEventID"] = []
        return e

    max_gap_h = bin_hours * 2 + 0.01

    event_ids = []
    current = 0
    prev = None

    for t in pd.to_datetime(e["CalendarBin"]):
        if prev is None or (t - prev).total_seconds() / 3600.0 > max_gap_h:
            current += 1
        event_ids.append(f"CME{current:03d}")
        prev = t

    e["CommonModeEventID"] = event_ids
    return e


# =============================================================================
# Minimal row-level removal
# =============================================================================

def apply_minimal_common_mode_removal(
    y: pd.DataFrame,
    event_bins: pd.DataFrame,
) -> pd.DataFrame:
    x = y.copy()

    map_cols = [
        "CalendarBin",
        "CommonModeEventBinFlag",
        "CommonModeEventType",
        "CommonModeEventID",
        "ActiveBatteries",
        "StrongDeltaBatteries",
        "StrongCellFraction",
        "DominantSignActiveFraction",
        "SignCoherenceAmongStrong",
        "BinIndependentPhysicalSupportCount",
    ]

    eb = event_bins[map_cols].copy() if not event_bins.empty else pd.DataFrame(
        columns=map_cols
    )

    x = x.merge(
        eb,
        on="CalendarBin",
        how="left",
        validate="many_to_one",
    )

    x["CommonModeEventBinFlag"] = (
        pd.to_numeric(x["CommonModeEventBinFlag"], errors="coerce")
        .fillna(0).astype(np.int8)
    )
    x["CommonModeEventType"] = x["CommonModeEventType"].fillna("NONE")
    x["CommonModeEventID"] = x["CommonModeEventID"].fillna("")

    dsoh_abs = pd.to_numeric(
        x[f"Delta__{TARGET}"], errors="coerce"
    ).abs()

    strong = x["StrongDeltaSOHFlag"].eq(1)
    in_event = x["CommonModeEventBinFlag"].eq(1)
    row_phys = (
        pd.to_numeric(
            x["RowIndependentPhysicalSupportCount"], errors="coerce"
        ).fillna(0) >= 1
    )
    huge_delta = dsoh_abs >= ROW_LARGE_DSOH_OVERRIDE

    event_type = x["CommonModeEventType"].astype(str)

    # For coherent/extreme event:
    # strong SOH response itself is enough because many independent batteries
    # are simultaneously affected.
    coherent_or_extreme = event_type.isin([
        "COHERENT_SIGN",
        "EXTREME_SYNC",
    ])

    # For mixed-direction event:
    # require independent I/V row evidence OR a very large dSOH.
    mixed = event_type.eq("MIXED_SIGN_PHYSICAL")

    remove = (
        in_event
        & strong
        & (
            coherent_or_extreme
            | (mixed & (row_phys | huge_delta))
        )
    )

    x["MinimalCommonModeRemoveFlag"] = remove.astype(np.int8)

    reason = np.full(len(x), "KEEP", dtype=object)
    reason[
        (
            x["V45B_Retained"].eq(0)
        ).to_numpy()
    ] = "EXCLUDE_V45B"

    reason[
        (
            x["V45B_Retained"].eq(1)
            & (
                x["CycleWithinZYKSession"]
                <= RECOVERY_CYCLES
            )
        ).to_numpy()
    ] = "EXCLUDE_RPT_SESSION_RECOVERY"

    reason[
        remove.to_numpy()
    ] = "EXCLUDE_MINIMAL_COMMON_MODE_POINT"

    # Full-cycle incompleteness applies only after previous exclusions
    full_valid = (
        pd.to_numeric(
            x["FullAgingCycleValid"], errors="coerce"
        ).fillna(0).astype(int).eq(1)
    )

    later_incomplete = (
        x["V45B_Retained"].eq(1)
        & (x["CycleWithinZYKSession"] > RECOVERY_CYCLES)
        & (~remove)
        & (~full_valid)
    )
    reason[later_incomplete.to_numpy()] = "EXCLUDE_INCOMPLETE_AGING_HALF_CYCLE"

    x["FinalDecision"] = reason

    x["StrictMinimalKeep"] = pd.Series(reason).eq("KEEP").astype(np.int8)

    return x


# =============================================================================
# Sequence segments
# =============================================================================

def build_strict_segments(x: pd.DataFrame) -> pd.DataFrame:
    d = x[x["StrictMinimalKeep"].eq(1)].copy()
    d = d.sort_values(["Battery", "GlobalCycle"]).reset_index(drop=True)

    prev_b = d["Battery"].shift()
    prev_sp = d["SourcePart"].shift()
    prev_c = d["GlobalCycle"].shift()

    br = (
        d["Battery"].ne(prev_b)
        | d["SourcePart"].astype(str).ne(prev_sp.astype(str))
        | (
            d["GlobalCycle"]
            - pd.to_numeric(prev_c, errors="coerce")
        ).ne(1)
    )

    d["StrictSequenceBreakFlag"] = br.astype(np.int8)
    d["StrictSegmentIndex"] = (
        br.groupby(d["Battery"]).cumsum().astype(int)
    )
    d["StrictSegmentKey"] = (
        d["Battery"].astype(str)
        + "__S"
        + d["StrictSegmentIndex"].astype(str).str.zfill(4)
    )

    return d


# =============================================================================
# Summaries
# =============================================================================

def delta_summary(x: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    def collect(df: pd.DataFrame, label: str) -> Dict[str, float]:
        vals = []

        for _, g in df.sort_values(
            ["Battery", "GlobalCycle"]
        ).groupby("Battery", sort=False):
            v = pd.to_numeric(g[TARGET], errors="coerce")
            c = pd.to_numeric(g["GlobalCycle"], errors="coerce")
            ds = v.diff().abs()
            dc = c.diff()
            q = ds[dc.eq(1)].dropna()
            vals.extend(q.to_numpy(float).tolist())

        a = np.asarray(vals, dtype=float)
        a = a[np.isfinite(a)]

        if len(a) == 0:
            return {
                "Series": label,
                "NConsecutivePairs": 0,
            }

        return {
            "Series": label,
            "NConsecutivePairs": int(len(a)),
            "MedianAbsDelta": float(np.quantile(a, 0.50)),
            "P90AbsDelta": float(np.quantile(a, 0.90)),
            "P95AbsDelta": float(np.quantile(a, 0.95)),
            "P99AbsDelta": float(np.quantile(a, 0.99)),
            "MaxAbsDelta": float(np.max(a)),
            "RateAbsDeltaGT0p02": float(np.mean(a > 0.02)),
            "RateAbsDeltaGT0p05": float(np.mean(a > 0.05)),
        }

    raw = x[
        x["V45B_Retained"].eq(1)
        & (x["CycleWithinZYKSession"] > RECOVERY_CYCLES)
    ].copy()

    return pd.DataFrame([
        collect(raw, "V45B_POST_RECOVERY_RAW"),
        collect(strict, "STRICT_MINIMAL_IMPACT"),
    ])


def window_availability(strict: pd.DataFrame) -> pd.DataFrame:
    seg = (
        strict.groupby(
            ["Battery", "StrictSegmentKey"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "Rows"})
    )

    rows = []

    for L in WINDOW_LENGTHS:
        for h in FORECAST_HORIZONS:
            per_b = []

            for b, g in seg.groupby("Battery", sort=True):
                nwin = int(
                    sum(
                        max(0, int(n) - L - h + 1)
                        for n in g["Rows"]
                    )
                )
                per_b.append((b, nwin))

            arr = np.asarray([n for _, n in per_b], dtype=int)

            rows.append({
                "HistoryLength": L,
                "Horizon": h,
                "TotalWindows": int(arr.sum()),
                "BatteriesWithAtLeast1Window": int((arr > 0).sum()),
                "MinWindowsPerBattery": int(arr.min()) if len(arr) else 0,
                "MedianWindowsPerBattery": float(np.median(arr)) if len(arr) else 0,
                "MaxWindowsPerBattery": int(arr.max()) if len(arr) else 0,
            })

    return pd.DataFrame(rows)


def battery_impact_summary(x: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for b, g in x.groupby("Battery", sort=True):
        v45b = int(g["V45B_Retained"].sum())
        cm = int(g["MinimalCommonModeRemoveFlag"].sum())
        recovery = int(g["FinalDecision"].eq("EXCLUDE_RPT_SESSION_RECOVERY").sum())
        keep = int(g["StrictMinimalKeep"].sum())

        rows.append({
            "Battery": b,
            "V45BRetainedRows": v45b,
            "MinimalCommonModeRemovedRows": cm,
            "CommonModeRemovalRateFromV45B": cm / max(1, v45b),
            "RPTSessionRecoveryExcludedRows": recovery,
            "StrictMinimalRows": keep,
            "StrictMinimalKeepRateFromV45B": keep / max(1, v45b),
            "CommonModeEventsTouched": int(
                g.loc[
                    g["MinimalCommonModeRemoveFlag"].eq(1),
                    "CommonModeEventID",
                ].replace("", np.nan).nunique()
            ),
        })

    return pd.DataFrame(rows)


def feature_policy() -> pd.DataFrame:
    rows = [
        {
            "FeatureGroup": "Cycling-SOH target",
            "Examples": TARGET,
            "MainPolicy": "USE_ONLY_IF_CYCLE_RETAINED",
            "NumericalModification": "NONE",
            "Reason": "Affected common-mode cycle removed as a whole; retained values unchanged.",
        },
        {
            "FeatureGroup": "Capacity",
            "Examples": "Charge/Discharge capacity, CC4 capacity",
            "MainPolicy": "USE_ONLY_IF_CYCLE_RETAINED",
            "NumericalModification": "NONE",
            "Reason": "Same Battery+GlobalCycle removal key applies.",
        },
        {
            "FeatureGroup": "Voltage / Current",
            "Examples": "V min/max, I mean, future V/I HIs",
            "MainPolicy": "USE_ONLY_IF_CYCLE_RETAINED",
            "NumericalModification": "NONE",
            "Reason": "Same Battery+GlobalCycle removal key applies.",
        },
        {
            "FeatureGroup": "Temperature / Power / Resistance / all future HI",
            "Examples": "T, P, R proxies, dVdt, entropy, dwell ratios",
            "MainPolicy": "FILTER_BY_REMOVAL_KEYS",
            "NumericalModification": "NONE",
            "Reason": "Current table may not contain every HI; downstream datasets must join removal keys.",
        },
        {
            "FeatureGroup": "Protocol metadata",
            "Examples": "SourcePart, CalendarBin, ZYKSessionIndex, CycleWithinSession, RPT flags",
            "MainPolicy": "QC_AND_SEGMENTATION_ONLY",
            "NumericalModification": "NONE",
            "Reason": "Do not use as primary health features; avoids learning laboratory schedule.",
        },
    ]
    return pd.DataFrame(rows)


# =============================================================================
# Gates
# =============================================================================

def structural_gate(
    a: pd.DataFrame,
    b: pd.DataFrame,
    audit: pd.DataFrame,
    strict: pd.DataFrame,
) -> Tuple[pd.DataFrame, bool]:
    rows = []

    def add(name, obs, exp, passed, detail=""):
        rows.append({
            "Check": name,
            "Observed": obs,
            "Expected": exp,
            "PASS": int(bool(passed)),
            "Detail": detail,
        })

    add(
        "V4.5A duplicate keys",
        int(a.duplicated(["Battery", "GlobalCycle"]).sum()),
        0,
        not a.duplicated(["Battery", "GlobalCycle"]).any(),
    )
    add(
        "V4.5B duplicate keys",
        int(b.duplicated(["Battery", "GlobalCycle"]).sum()),
        0,
        not b.duplicated(["Battery", "GlobalCycle"]).any(),
    )
    add(
        "Strict duplicate keys",
        int(strict.duplicated(["Battery", "GlobalCycle"]).sum()),
        0,
        not strict.duplicated(["Battery", "GlobalCycle"]).any(),
    )

    matched = int(audit["V45B_Retained"].sum())
    match_rate = matched / max(1, len(b))

    add(
        "V4.5B keys matched to V4.5A",
        match_rate,
        f">={MIN_V45B_KEY_MATCH_RATE}",
        match_rate >= MIN_V45B_KEY_MATCH_RATE,
    )

    bad_recovery = int(
        (
            pd.to_numeric(
                strict["CycleWithinZYKSession"], errors="coerce"
            ) <= RECOVERY_CYCLES
        ).sum()
    )
    add(
        "Recovery cycles leaked into strict model-ready",
        bad_recovery,
        0,
        bad_recovery == 0,
    )

    # Every common-mode removed point must be absent from strict.
    removed_keys = set(
        zip(
            audit.loc[
                audit["MinimalCommonModeRemoveFlag"].eq(1),
                "Battery",
            ],
            audit.loc[
                audit["MinimalCommonModeRemoveFlag"].eq(1),
                "GlobalCycle",
            ],
        )
    )
    strict_keys = set(zip(strict["Battery"], strict["GlobalCycle"]))
    leaked = len(removed_keys & strict_keys)

    add(
        "Common-mode removal keys leaked into strict table",
        leaked,
        0,
        leaked == 0,
    )

    gate = pd.DataFrame(rows)
    return gate, bool(gate["PASS"].eq(1).all())


def scientific_gate(
    audit: pd.DataFrame,
    event_bins: pd.DataFrame,
    strict: pd.DataFrame,
    windows: pd.DataFrame,
) -> Tuple[pd.DataFrame, bool]:
    rows = []

    def add(name, obs, exp, passed, detail=""):
        rows.append({
            "EvidenceCheck": name,
            "Observed": obs,
            "SuggestedThreshold": exp,
            "PASS": int(bool(passed)),
            "Detail": detail,
        })

    v45b = int(audit["V45B_Retained"].sum())
    cm = int(audit["MinimalCommonModeRemoveFlag"].sum())
    cm_rate = cm / max(1, v45b)

    add(
        "Minimal common-mode removal rate from V4.5B",
        cm_rate,
        f"<={MAX_COMMON_MODE_REMOVAL_RATE_FROM_V45B}",
        cm_rate <= MAX_COMMON_MODE_REMOVAL_RATE_FROM_V45B,
        "Confirms point-level filtering rather than broad interval deletion.",
    )

    # Natural rows outside event bins should overwhelmingly be retained.
    natural = (
        audit["V45B_Retained"].eq(1)
        & (audit["CycleWithinZYKSession"] > RECOVERY_CYCLES)
        & audit["StrongDeltaSOHFlag"].eq(0)
        & audit["ModerateDeltaSOHFlag"].eq(0)
        & (
            pd.to_numeric(audit["FullAgingCycleValid"], errors="coerce")
            .fillna(0).astype(int).eq(1)
        )
    )
    nat_keep = (
        float(audit.loc[natural, "StrictMinimalKeep"].mean())
        if natural.any() else np.nan
    )

    add(
        "Natural non-event cycle preservation",
        nat_keep,
        f">={MIN_NATURAL_KEEP_RATE}",
        np.isfinite(nat_keep) and nat_keep >= MIN_NATURAL_KEEP_RATE,
    )

    if len(event_bins):
        med_active = safe_float(event_bins["ActiveBatteries"].median())
        med_frac = safe_float(event_bins["StrongCellFraction"].median())
    else:
        med_active = np.nan
        med_frac = np.nan

    add(
        "Median active batteries in detected event bins",
        med_active,
        f">={MIN_EVENT_ACTIVE_BATTERIES}",
        np.isfinite(med_active) and med_active >= MIN_EVENT_ACTIVE_BATTERIES,
    )
    add(
        "Median strong-cell fraction in detected event bins",
        med_frac,
        f">={MIN_EVENT_MEDIAN_STRONG_FRACTION}",
        np.isfinite(med_frac) and med_frac >= MIN_EVENT_MEDIAN_STRONG_FRACTION,
    )

    # RPT relationship should remain strong.
    rpt = strict[
        pd.to_numeric(
            strict["RPT_ExactMeasuredAnchor"], errors="coerce"
        ).fillna(0).astype(int).eq(1)
        & pd.to_numeric(strict["RPT_SOH_Measured"], errors="coerce").notna()
    ]
    rho = (
        spearman_corr(rpt[TARGET], rpt["RPT_SOH_Measured"])
        if len(rpt) else np.nan
    )
    add(
        "Strict Cycling-SOH Spearman vs measured RPT-SOH",
        rho,
        ">=0.98",
        np.isfinite(rho) and rho >= 0.98,
    )

    h100 = windows[
        (windows["HistoryLength"] == 20)
        & (windows["Horizon"] == 100)
    ]
    cells = int(
        h100["BatteriesWithAtLeast1Window"].iloc[0]
    ) if len(h100) else 0

    add(
        "Cells with L=20 horizon=100 windows",
        cells,
        f">={MIN_CELLS_H100}",
        cells >= MIN_CELLS_H100,
    )

    gate = pd.DataFrame(rows)
    return gate, bool(gate["PASS"].eq(1).all())


# =============================================================================
# Figures
# =============================================================================

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_event_bins(
    bin_audit: pd.DataFrame,
    event_bins: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    if bin_audit.empty:
        return

    plt = setup_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        pd.to_datetime(bin_audit["CalendarBin"]),
        bin_audit["StrongCellFraction"],
        linewidth=0.8,
        label="Strong-cell fraction",
    )

    if len(event_bins):
        ax.scatter(
            pd.to_datetime(event_bins["CalendarBin"]),
            event_bins["StrongCellFraction"],
            s=28,
            marker="x",
            label="Detected common-mode bins",
        )

    ax.axhline(
        EVENT_STRONG_CELL_FRACTION,
        linestyle="--",
        linewidth=1,
    )
    ax.set_ylabel("Fraction of active batteries")
    ax.set_xlabel("Estimated experiment calendar time")
    ax.set_title("Cross-cell calendar-synchronous common-mode detection")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_minimal_trajectories(
    audit: pd.DataFrame,
    strict: pd.DataFrame,
    figdir: Path,
    dpi: int,
) -> int:
    plt = setup_matplotlib()
    batteries = sorted(audit["Battery"].unique())
    per_page = 16
    pages = 0

    strict_key_set = set(zip(strict["Battery"], strict["GlobalCycle"]))

    for start in range(0, len(batteries), per_page):
        chunk = batteries[start:start+per_page]
        ncols = 4
        nrows = int(math.ceil(len(chunk)/ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(16, 3.1*nrows),
            sharey=True,
        )
        axes = np.asarray(axes).reshape(-1)

        for i, b in enumerate(chunk):
            ax = axes[i]
            g = audit[
                audit["Battery"].eq(b)
            ].sort_values("GlobalCycle").copy()

            raw = pd.to_numeric(g[TARGET], errors="coerce")
            ax.plot(
                g["GlobalCycle"],
                raw,
                linewidth=0.45,
                alpha=0.28,
                label="Raw",
            )

            keep_mask = np.array([
                (bb, int(cc)) in strict_key_set
                for bb, cc in zip(
                    g["Battery"],
                    g["GlobalCycle"],
                )
            ])
            ky = raw.copy()
            ky.loc[~keep_mask] = np.nan

            ax.plot(
                g["GlobalCycle"],
                ky,
                linewidth=0.8,
                label="Strict minimal-impact",
            )

            cm = g[
                g["MinimalCommonModeRemoveFlag"].eq(1)
            ]
            if len(cm):
                ax.scatter(
                    cm["GlobalCycle"],
                    cm[TARGET],
                    s=10,
                    marker="x",
                    label="Common-mode point removed",
                )

            ax.set_title(b, fontsize=9)
            ax.grid(alpha=0.15)
            ax.set_ylim(0.10, 1.12)

            if i == 0:
                ax.legend(fontsize=5.5)

        for j in range(len(chunk), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            "Minimal-impact common-mode cleaning: only individually abnormal synchronized cycles removed",
            fontsize=11,
        )
        fig.tight_layout()

        pages += 1
        fig.savefig(
            figdir / f"02_minimal_impact_trajectories_page_{pages:02d}.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

    return pages


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    v45a_path = (
        args.v45a_file
        if args.v45a_file is not None
        else args.v45a_root / V45A_FILE
    )
    v45b_path = (
        args.v45b_file
        if args.v45b_file is not None
        else args.v45b_root / V45B_FILE
    )

    out = ensure_dir(args.output_root)
    figdir = ensure_dir(out / "Figures")

    print("=" * 118)
    print("RWTH48 V4.5C-FIX | 最小冲击跨电池共同扰动处理")
    print("V4.5A :", v45a_path)
    print("V4.5B :", v45b_path)
    print("OUTPUT:", out)
    print("Calendar bin:", args.calendar_bin_hours, "h")
    print("=" * 118, flush=True)

    config = {
        "Task": "RWTH48 V4.5C-FIX minimal-impact calendar-synchronous common-mode disturbance handling",
        "V45AInput": str(v45a_path),
        "V45BInput": str(v45b_path),
        "Output": str(out),
        "MainTarget": TARGET,
        "RecoveryCyclesPerSession": RECOVERY_CYCLES,
        "CalendarBinHours": args.calendar_bin_hours,
        "StrongDeltaSOHFloor": STRONG_DSOH_FLOOR,
        "StrongDeltaSOHMADMultiplier": STRONG_DSOH_MAD_MULT,
        "EventStrongCellFraction": EVENT_STRONG_CELL_FRACTION,
        "EventDominantSignActiveFraction": EVENT_DOMINANT_SIGN_ACTIVE_FRACTION,
        "EventMixedStrongFraction": EVENT_MIXED_STRONG_FRACTION,
        "EventExtremeStrongFraction": EVENT_EXTREME_STRONG_FRACTION,
        "PhysicalSupportFeatures": PHYS_SUPPORT_FEATURES,
        "MinimalImpactPolicy": [
            "Do not remove whole calendar intervals.",
            "First detect common-mode event bins using cross-cell synchrony.",
            "Within event bins, remove only cycles that are themselves strong dSOH anomalies.",
            "Mixed-sign events additionally require independent I/V evidence or |dSOH|>=0.03.",
            "Do not interpolate, smooth, clip, or force monotonicity.",
            "All downstream physical features must be filtered by the same Battery+GlobalCycle removal keys.",
            "All sequence windows must break at removed cycles and at SourcePart/session boundaries.",
        ],
    }
    save_json(config, out / "00_config.json")

    # A. load
    print("[A] Loading V4.5A + V4.5B ...", flush=True)
    a, b = load_inputs(v45a_path, v45b_path)

    # B. metadata / deltas
    print("[B] Building session/calendar metadata and robust row deltas ...", flush=True)
    x = add_session_calendar_metadata(
        a,
        args.calendar_bin_hours,
    )
    x = mark_v45b_retained(x, b)
    x, thresholds = add_delta_features(x)

    thresholds.to_csv(
        out / "_battery_delta_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # C. common-mode bins
    print("[C] Detecting calendar-synchronous common-mode event bins ...", flush=True)
    bin_audit = build_calendar_bin_audit(x)
    event_bins = assign_event_clusters(
        bin_audit,
        args.calendar_bin_hours,
    )

    bin_audit.to_csv(
        out / "01_calendar_bin_common_mode_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    event_bins.to_csv(
        out / "02_detected_common_mode_event_bins.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # D. minimal row removal
    print("[D] Applying minimal row-level removal ...", flush=True)
    audit = apply_minimal_common_mode_removal(
        x,
        event_bins,
    )

    audit.to_csv(
        out / "03_cycle_level_common_mode_audit.csv.gz",
        index=False,
        compression=FAST_GZIP,
    )

    removal_keys = (
        audit.loc[
            audit["MinimalCommonModeRemoveFlag"].eq(1),
            [
                "Battery",
                "GlobalCycle",
                "EstimatedCycleStartTime",
                "CalendarBin",
                "CommonModeEventID",
                "CommonModeEventType",
            ],
        ]
        .sort_values(["Battery", "GlobalCycle"])
        .reset_index(drop=True)
    )
    removal_keys.to_csv(
        out / "04_MINIMAL_COMMON_MODE_REMOVAL_KEYS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    removed = audit[
        audit["MinimalCommonModeRemoveFlag"].eq(1)
    ].copy()
    removed.to_csv(
        out / "05_minimal_removed_common_mode_cycles.csv.gz",
        index=False,
        compression=FAST_GZIP,
    )

    # E. strict sequence
    print("[E] Building strict minimal-impact model-ready sequences ...", flush=True)
    strict = build_strict_segments(audit)

    preferred = [
        "Battery", "GlobalCycle", "SourcePart",
        "EstimatedCycleStartTime",
        "ZYKSessionIndex", "CycleWithinZYKSession",
        "StrictSequenceBreakFlag",
        "StrictSegmentIndex", "StrictSegmentKey",
        TARGET,
        "CyclingSOH_Discharge_Canonical",
        "RPT_ExactMeasuredAnchor",
        "RPT_SOH_Measured",
        "RPT_SOH_InterpolatedOrMeasured",
    ]

    physical = [
        c for c in AUX_PHYS_FEATURES
        if c in strict.columns
    ]

    model_cols = [
        c for c in preferred + physical
        if c in strict.columns
    ]

    strict[model_cols].to_csv(
        out / "06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz",
        index=False,
        compression=FAST_GZIP,
    )

    feature_policy().to_csv(
        out / "07_feature_handling_policy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # F. summaries
    reason = (
        audit.groupby("FinalDecision", as_index=False)
        .agg(
            Rows=("GlobalCycle", "size"),
            Batteries=("Battery", "nunique"),
        )
    )
    reason["RateOfAllRows"] = reason["Rows"] / len(audit)
    reason.to_csv(
        out / "08_removal_reason_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dsum = delta_summary(audit, strict)
    dsum.to_csv(
        out / "09_before_after_delta_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    windows = window_availability(strict)
    windows.to_csv(
        out / "10_window_availability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bsum = battery_impact_summary(audit)
    bsum.to_csv(
        out / "11_battery_impact_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sg, sg_pass = structural_gate(
        a, b, audit, strict
    )
    sg.to_csv(
        out / "12_structural_acceptance_gate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    eg, eg_pass = scientific_gate(
        audit, event_bins, strict, windows
    )
    eg.to_csv(
        out / "13_scientific_evidence_gate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # G. figures
    pages = 0
    if not args.skip_figures:
        print("[F] Generating figures ...", flush=True)
        plot_event_bins(
            bin_audit,
            event_bins,
            figdir / "01_calendar_common_mode_events.png",
            args.dpi,
        )
        pages = plot_minimal_trajectories(
            audit,
            strict,
            figdir,
            args.dpi,
        )

    # H. completion
    elapsed = time.perf_counter() - t0

    v45b_n = int(audit["V45B_Retained"].sum())
    cm_n = int(audit["MinimalCommonModeRemoveFlag"].sum())
    strict_n = int(len(strict))

    complete = {
        "Task": "RWTH48 V4.5C-FIX minimal-impact common-mode disturbance handling",
        "Status": "PASS" if sg_pass else "CHECK",
        "ScientificEvidenceStatus": (
            "PASS_MINIMAL_IMPACT_COMMON_MODE_FILTER"
            if eg_pass
            else "CHECK_MINIMAL_IMPACT_EVIDENCE"
        ),
        "InputRowsV45A": int(len(a)),
        "RowsRetainedV45B": v45b_n,
        "DetectedCommonModeEventBins": int(len(event_bins)),
        "DetectedCommonModeEventClusters": int(
            event_bins["CommonModeEventID"].nunique()
        ) if len(event_bins) else 0,
        "MinimalCommonModeRowsRemoved": cm_n,
        "MinimalCommonModeRemovalRateFromV45B": (
            cm_n / max(1, v45b_n)
        ),
        "StrictMinimalModelReadyRows": strict_n,
        "StrictMinimalKeepRateFromV45B": (
            strict_n / max(1, v45b_n)
        ),
        "Batteries": int(audit["Battery"].nunique()),
        "RecoveryCyclesPerSession": RECOVERY_CYCLES,
        "CalendarBinHours": args.calendar_bin_hours,
        "TrajectoryPages": pages,
        "Elapsed_s": float(elapsed),
        "MainModelReady": str(
            out / "06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz"
        ),
        "RemovalKeyFile": str(
            out / "04_MINIMAL_COMMON_MODE_REMOVAL_KEYS.csv"
        ),
        "Important": [
            "No whole common-mode calendar interval is deleted.",
            "Only individually abnormal cycles inside cross-cell synchronized event bins are removed.",
            "Retained SOH and physical parameters are numerically unchanged.",
            "The same Battery+GlobalCycle removal keys must be applied to downstream V/I/T/HI datasets.",
            "Protocol metadata is QC-only and should not be a primary health feature.",
            "Sequences break at session boundaries and at any removed cycle gap.",
        ],
        "NextStep": (
            "If all gates PASS, use 06_STRICT_MINIMAL_IMPACT_MODEL_READY.csv.gz "
            "for V4.6 outer-LOBO LSTM/BiLSTM direct multi-horizon forecasting."
        ),
    }

    save_json(
        complete,
        out / "99_RWTH48_V4_5C_FIX_COMPLETE.json",
    )

    print("\n" + "=" * 118)
    print("RWTH48 V4.5C-FIX COMPLETE")
    print("Structural status :", complete["Status"])
    print("Scientific status :", complete["ScientificEvidenceStatus"])
    print("V4.5A rows        :", f"{len(a):,}")
    print("V4.5B retained    :", f"{v45b_n:,}")
    print("Event bins        :", complete["DetectedCommonModeEventBins"])
    print("Event clusters    :", complete["DetectedCommonModeEventClusters"])
    print("Common-mode removed:", f"{cm_n:,}")
    print("CM removal rate   :", f"{complete['MinimalCommonModeRemovalRateFromV45B']:.4%}")
    print("Strict rows       :", f"{strict_n:,}")
    print("Strict keep rate  :", f"{complete['StrictMinimalKeepRateFromV45B']:.4%}")
    print("Elapsed           :", f"{elapsed:.1f} s")
    print("Output            :", out)
    print("=" * 118)


if __name__ == "__main__":
    main()

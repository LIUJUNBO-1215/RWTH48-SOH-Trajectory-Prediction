# -*- coding: utf-8 -*-
"""
RWTH 48-cell Rawdata 全流程 V3：
1) 扫描 Rohdaten 下零散 leaf .zip（每个zip内通常一个csv）
2) 文件/电池级质量审计与同工况一致性审计
3) TBA_Zyk / CU / CUv2 / BOL / EIS 分类
4) 正式老化 TBA_Zyk 全局Cycle重建（保留局部缺号，不做 nearest-cycle 替代）
5) 物理无效值、完全重复记录、保守因果毛刺清洗
6) CU/CUv2/BOL 标准容量提取，构建稀疏SOH reference；仅离线插值得到逐cycle监督标签
7) Discharge FixedTime prefix HI：60/120/180/300/450/600/900/1200 s
8) Discharge Event prefix HI：
   A. PRIMARY RefAhFraction = 0.20/0.30/0.40 × 冻结BOL reference capacity（因果、跨cycle可比较）
   B. SECONDARY DeltaV = 0.20/0.30/0.40 V（敏感性/表示消融）
9) 所有 FixedTime/Event HI 同时包含 Voltage / Current / Power / Temperature / Throughput / Dynamic-R proxy
10) 输出 model-ready long tables、feature registry、建议battery-level split、论文级审计图

默认输入：
workspace/Rohdaten
默认输出：
workspace/RWTH48_全流程输出_V4

安装依赖：
    pip install numpy pandas matplotlib

直接运行：
    python "src/01_preprocess_soh_labels.py"

推荐正式运行（4核仅并行电池；单个电池内部按时间顺序处理）：
    python "src/01_preprocess_soh_labels.py" ^
      --input-root "workspace/Rohdaten" ^
      --output-root "workspace/RWTH48_全流程输出_V4" ^
      --n-jobs 4

只审计，不写清洗后的大体积时序：
    python src/01_preprocess_soh_labels.py --audit-only

完整跑但不保存clean timeseries（仍会在内存中清洗后提HI）：
    python src/01_preprocess_soh_labels.py --no-save-clean-timeseries

说明：
- 不在模型输入中使用当前cycle完整容量/完整放电时长等未来信息。
- Event_RefAhFraction 使用“冻结的早期/BOL reference capacity”，不是当前cycle最终容量，因此合法。
- 逐cycle SOH 插值只用于监督标签构建，不作为任何prefix特征。
- 毛刺清洗默认采用“同cycle+同State+同Step的过去窗口”基准，不使用未来样本。
- 不做低通/平滑整条SOH或V/I/T曲线，避免改变真实物理动态；只修明确传感器异常/孤立毛刺。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. CONFIG
# =============================================================================
DEFAULT_INPUT_ROOT = (Path(__file__).resolve().parents[1] / "workspace" / "Rohdaten")
DEFAULT_OUTPUT_ROOT = (Path(__file__).resolve().parents[1] / "workspace" / "RWTH48_全流程输出_V4")

FIXED_TIME_NODES_S: Tuple[int, ...] = (60, 120, 180, 300, 450, 600, 900, 1200)
EVENT_REF_AH_FRACTIONS: Tuple[float, ...] = (0.20, 0.30, 0.40)
EVENT_DELTAV_NODES_V: Tuple[float, ...] = (0.20, 0.30, 0.40)

# 传感器“硬无效”范围：只删除明显不可能/编码溢出的测量值。
# 注意：BOL_P2存在0/50/99°C等表征工况；ZYK中也出现过多个电池同日约60~68°C的共同热事件。
# 因此不能把 >60°C 一律当传感器毛刺删除。协议偏离与传感器失效分开审计。
VOLTAGE_HARD_RANGE = (2.0, 4.5)
CURRENT_HARD_RANGE = (-10.0, 10.0)
TEMPERATURE_HARD_RANGE = (-10.0, 120.0)

# 正式老化 TBA_Zyk 的“名义工况温度范围”仅用于协议审计，不自动清洗。
TEMPERATURE_NOMINAL_ZYK_RANGE = (5.0, 60.0)

# 整段温度通道失效：例如某个完整ZYK session全为0°C。
TEMP_SESSION_ZERO_ABS_C = 1.0
TEMP_SESSION_ZERO_RATIO = 0.95
TEMP_SESSION_MIN_ROWS = 1000

# 保守“过去窗口”毛刺检测；在同 Cycle + State + Step 内重置。
CAUSAL_SPIKE_WINDOW = 9
CAUSAL_SPIKE_MIN_HISTORY = 5
CAUSAL_SPIKE_THRESHOLDS = {
    "Voltage_V": 0.12,       # 同step相对过去中位数偏离 > 120 mV
    "Current_A": 0.90,       # 同step偏离 > 0.9 A
    "Temperature_C": 4.0,    # 同step偏离 > 4 C
}

# 放电识别
DISCHARGE_CURRENT_THRESHOLD_A = -0.03
MIN_PREFIX_ROWS = 8
MIN_DCH_DURATION_S = 30.0

# 标准容量提取
MIN_CAPACITY_AH = 0.5
MAX_CAPACITY_AH = 3.0
REFERENCE_EARLY_POINTS = 3
REFERENCE_CAPACITY_CURRENT_A = 0.50
REFERENCE_CAPACITY_CURRENT_TOL_A = 0.20

# 审计门限（可在输出后据48-cell总体再冻结）
MAX_DUPLICATE_RATE_FOR_PASS = 0.03
MAX_HARD_INVALID_RATE_FOR_PASS = 0.005
MAX_SPIKE_RATE_FOR_PASS = 0.01
MAX_INTERNAL_MISSING_CYCLE_RATE_FOR_PASS = 0.05
MIN_ZYK_FILES_FOR_PASS = 3
MIN_USABLE_CYCLES_FOR_PASS = 300

# TBA_Zyk 会话级健全性门控：用于排除“文件名叫ZYK、实际上不是有效老化循环”的异常会话。
# 阈值故意宽，正常会话约500+ rows/cycle、|I|max约4.1 A；已知异常RWTH-019会话仅~4 rows/cycle且|I|max<0.02 A。
MIN_ZYK_ROWS_PER_ACTIVE_CYCLE = 100.0
MIN_ZYK_MAX_ABS_CURRENT_A = 1.0

# 稳健协议一致性 z 分数阈值
PROTOCOL_ROBUST_Z_THRESHOLD = 5.0

RANDOM_SEED = 42

CANONICAL_MAP = {
    "Schritt": "Step",
    "Zustand": "State",
    "Zeit": "Timestamp",
    "Programmdauer": "ProgramDuration_ms",
    "Schrittdauer": "StepDuration_ms",
    "Zyklus": "LocalCycle",
    "Zyklusebene": "CycleLevel",
    "Prozedur": "Procedure",
    "Prozedurebene": "ProcedureLevel",
    "AhAkku": "AhBattery",
    "AhLad": "AhCharge",
    "AhEla": "AhDischarge",
    "AhStep": "AhStep",
    "Energie": "Energy_Wh",
    "WhStep": "WhStep",
    "Spannung": "Voltage_V",
    "Strom": "Current_A",
    "Temp1": "Temperature_C",
}

TS_REQUIRED = ("Timestamp", "State", "LocalCycle", "Voltage_V", "Current_A", "Temperature_C")
NUMERIC_COLS = (
    "Step", "ProgramDuration_ms", "StepDuration_ms", "LocalCycle", "CycleLevel",
    "ProcedureLevel", "AhBattery", "AhCharge", "AhDischarge", "AhStep",
    "Energy_Wh", "WhStep", "Voltage_V", "Current_A", "Temperature_C",
)


# =============================================================================
# 1. UTILITIES
# =============================================================================
def setup_font() -> None:
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    try:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in preferred:
            if name in installed:
                plt.rcParams["font.sans-serif"] = [name]
                break
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    compression = "gzip" if str(path).lower().endswith(".gz") else None
    df.to_csv(path, index=index, encoding="utf-8-sig", compression=compression)


def save_json(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def robust_median_mad(x: Iterable[float]) -> Tuple[float, float]:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return med, mad


def robust_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return 0.67448975 * (x - med) / mad


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RWTH48 Rawdata V4 全流程质量审计/预处理/Event+FixedTime HI")
    p.add_argument("--input-root", type=str, default=str(DEFAULT_INPUT_ROOT))
    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--n-jobs", type=int, default=1, help="按电池并行。Windows建议2~4，避免磁盘/内存拥堵。")
    p.add_argument("--audit-only", action="store_true", help="只做inventory与原始文件审计，不做特征/标签。")
    p.add_argument("--no-save-clean-timeseries", action="store_true", help="不保存大体积clean时序part，但仍提特征。")
    p.add_argument("--force", action="store_true", help="忽略battery完成标记并重跑。")
    p.add_argument("--only-battery", type=str, default="", help="例如002；为空处理全部。")
    p.add_argument("--skip-deltav-event", action="store_true", help="只提Primary RefAhFraction Event，不提DeltaV Event。")
    if argv is not None:
        return p.parse_args(list(argv))
    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        args, _ = p.parse_known_args()
        return args
    return p.parse_args()


# =============================================================================
# 2. FILE INVENTORY / METADATA PARSE
# =============================================================================
@dataclass
class FileMeta:
    zip_path: str
    zip_name: str
    battery: str
    test_start: str
    role: str
    group_hint: str
    channel_hint: str
    compressed_bytes: int
    parse_ok: int
    parse_note: str


def parse_leaf_zip_name(path: Path) -> FileMeta:
    name = path.name
    bat_m = re.search(r"ep\s+sanyo\s+(\d{3})", name, flags=re.I)
    battery = f"RWTH-{bat_m.group(1)}" if bat_m else "UNKNOWN"

    dt_m = re.search(r"=Massenalterung=(\d{4}-\d{2}-\d{2}\s+\d{6})=([^=]+)=", name, flags=re.I)
    if dt_m:
        raw_dt, role = dt_m.group(1), dt_m.group(2)
        try:
            test_start = pd.to_datetime(raw_dt, format="%Y-%m-%d %H%M%S").isoformat()
        except Exception:
            test_start = raw_dt
    else:
        test_start = ""
        role_m = re.search(r"=(TBA_[^=]+)=", name, flags=re.I)
        role = role_m.group(1) if role_m else "UNKNOWN"

    group_hint = ""
    parts = name.split("=")
    if len(parts) >= 3:
        group_hint = parts[2]

    ch_m = re.search(r"Format\d+=([^.]*)\.zip$", name, flags=re.I)
    channel_hint = ch_m.group(1).strip() if ch_m else ""

    return FileMeta(
        zip_path=str(path), zip_name=name, battery=battery, test_start=test_start,
        role=role, group_hint=group_hint, channel_hint=channel_hint,
        compressed_bytes=path.stat().st_size if path.exists() else -1,
        parse_ok=int(battery != "UNKNOWN" and role != "UNKNOWN"),
        parse_note="" if (battery != "UNKNOWN" and role != "UNKNOWN") else "filename_metadata_incomplete",
    )


def scan_inventory(input_root: Path) -> pd.DataFrame:
    zips = sorted([p for p in input_root.rglob("*.zip") if p.is_file()])
    rows = [asdict(parse_leaf_zip_name(p)) for p in zips]
    inv = pd.DataFrame(rows)
    if not inv.empty:
        inv["test_start_dt"] = pd.to_datetime(inv["test_start"], errors="coerce")
        inv = inv.sort_values(["battery", "test_start_dt", "zip_name"], na_position="last").reset_index(drop=True)
    return inv


def open_inner_csv_bytes(zip_path: Path) -> Tuple[str, bytes, List[str]]:
    """Leaf zip -> CSV bytes. Does not extract to disk."""
    with zipfile.ZipFile(zip_path, "r") as z:
        members = [n for n in z.namelist() if not n.endswith("/")]
        csvs = [n for n in members if n.lower().endswith(".csv")]
        if not csvs:
            return "", b"", members
        # Usually exactly one CSV. Choose largest if multiple.
        info = sorted((z.getinfo(n) for n in csvs), key=lambda x: x.file_size, reverse=True)[0]
        return info.filename, z.read(info.filename), members


def read_leaf_csv(zip_path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    inner_name, raw, members = open_inner_csv_bytes(zip_path)
    info = {
        "InnerMemberCount": len(members),
        "InnerCSVName": inner_name,
        "InnerCSVBytes": len(raw),
        "EmptyInnerCSV": int(len(raw) == 0),
    }
    if not inner_name or len(raw) == 0:
        return pd.DataFrame(), info

    # The second line is a unit row, not data.
    last_err = None
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, skiprows=[1], low_memory=False)
            info["Encoding"] = enc
            return df, info
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"无法解析CSV {zip_path}: {last_err}")


def canonicalize_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一列名，并自动识别每个试验通道自己的温度列（Temp1/Temp7/Temp12/...）。
    RWTH 48块电池的温度通道编号并不统一，因此不能只把 Temp1 写死为 Temperature_C。

    同时构建 AcqTime_s：
    - 首选 ProgramDuration_ms（采集程序单调运行时间），规避墙上时钟/DST校时造成的 Timestamp 回跳；
    - 若不可用，再退化为按原始行顺序修复的 Timestamp 累计时间。
    """
    original_cols = list(df.columns)
    d = df.rename(columns={c: CANONICAL_MAP.get(c, c) for c in df.columns}).copy()

    # ---- Temperature channel auto-detection ----
    temp_source = ""
    if "Temperature_C" in d.columns:
        # 通常来自 Temp1 -> Temperature_C 的固定映射。
        temp_source = "Temp1" if "Temp1" in original_cols else "Temperature_C"
    else:
        # 正式 ZYK/CU 文件通常只有一个 Temp<number> 列，例如 Temp7/Temp92/Temp116。
        cand = [c for c in d.columns if re.fullmatch(r"Temp\d+", str(c), flags=re.I)]
        if not cand and "Temp" in d.columns:
            cand = ["Temp"]
        if cand:
            scores = []
            for c in cand:
                v = pd.to_numeric(d[c], errors="coerce")
                finite = int(v.notna().sum())
                plausible = int(v.between(-20.0, 100.0).sum())
                # 先最大化合理有效点，再最大化有限点；最后偏好名称较短者。
                scores.append((plausible, finite, -len(str(c)), str(c)))
            temp_source = max(scores)[-1]
            d["Temperature_C"] = pd.to_numeric(d[temp_source], errors="coerce")

    for c in NUMERIC_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "Timestamp" in d.columns:
        d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
    if "State" in d.columns:
        d["State"] = d["State"].astype(str).str.strip().str.upper()
    if "Procedure" in d.columns:
        d["Procedure"] = d["Procedure"].astype(str).str.strip()

    # ---- Monotonic acquisition time ----
    time_basis = "NONE"
    if "ProgramDuration_ms" in d.columns:
        prog = pd.to_numeric(d["ProgramDuration_ms"], errors="coerce")
        finite = prog.notna()
        dif = prog.loc[finite].diff()
        # 容许极少量重复，但不容许明显回跳。
        backwards = int((dif < -1e-6).sum()) if finite.any() else 0
        if finite.mean() >= 0.95 and backwards == 0:
            d["AcqTime_s"] = prog / 1000.0
            time_basis = "ProgramDuration_ms"

    if "AcqTime_s" not in d.columns and "Timestamp" in d.columns:
        # 保留文件原始行顺序。墙上时钟负跳不作为真实负时间；
        # 用正时间增量的中位数替代负/零异常增量，仅作为兜底。
        tt = pd.to_datetime(d["Timestamp"], errors="coerce")
        dt = tt.diff().dt.total_seconds()
        pos = dt[(dt > 0) & np.isfinite(dt)]
        fallback = float(pos.median()) if len(pos) else 1.0
        inc = dt.copy()
        inc.iloc[0] = 0.0
        inc = inc.where((inc > 0) & np.isfinite(inc), fallback)
        if len(inc):
            inc.iloc[0] = 0.0
        d["AcqTime_s"] = inc.fillna(fallback).cumsum()
        time_basis = "Timestamp_repaired"

    d.attrs["TemperatureSourceColumn"] = temp_source
    d.attrs["AcqTimeBasis"] = time_basis
    return d


def zyk_session_sanity(d: pd.DataFrame) -> Dict[str, object]:
    """宽松会话门控，防止异常/测试文件错误推进 GlobalCycle offset。"""
    if d.empty or "LocalCycle" not in d.columns or "State" not in d.columns:
        return {
            "ZYKSessionUsable": 0, "ZYKSessionReason": "missing_core_columns",
            "ZYKRowsPerActiveCycle": np.nan, "ZYKMaxAbsCurrent_A": np.nan,
        }
    local = pd.to_numeric(d["LocalCycle"], errors="coerce")
    st = d["State"].astype(str).str.upper()
    active = local.notna() & (local > 0) & st.isin(["DCH", "CHA"])
    ids = sorted(set(local.loc[active].astype(int)))
    ncyc = len(ids)
    rows_per = len(d) / max(1, ncyc)
    if "Current_A" in d.columns:
        cur = pd.to_numeric(d.loc[active, "Current_A"], errors="coerce")
        max_abs_i = float(cur.abs().max()) if cur.notna().any() else np.nan
    else:
        max_abs_i = np.nan

    usable = (
        ncyc > 0
        and rows_per >= MIN_ZYK_ROWS_PER_ACTIVE_CYCLE
        and np.isfinite(max_abs_i)
        and max_abs_i >= MIN_ZYK_MAX_ABS_CURRENT_A
    )
    reasons = []
    if ncyc <= 0:
        reasons.append("no_active_cycles")
    if rows_per < MIN_ZYK_ROWS_PER_ACTIVE_CYCLE:
        reasons.append("too_few_rows_per_cycle")
    if not np.isfinite(max_abs_i) or max_abs_i < MIN_ZYK_MAX_ABS_CURRENT_A:
        reasons.append("no_real_cycling_current")
    return {
        "ZYKSessionUsable": int(usable),
        "ZYKSessionReason": "PASS" if usable else ";".join(reasons),
        "ZYKRowsPerActiveCycle": float(rows_per),
        "ZYKMaxAbsCurrent_A": max_abs_i,
    }


def _time_seconds(p: pd.DataFrame) -> np.ndarray:
    """返回按采集顺序的单调秒时间；优先 AcqTime_s。"""
    if p.empty:
        return np.asarray([], dtype=float)
    if "AcqTime_s" in p.columns:
        a = pd.to_numeric(p["AcqTime_s"], errors="coerce").to_numpy(float)
        finite = np.flatnonzero(np.isfinite(a))
        if len(finite):
            base = a[finite[0]]
            return a - base
    if "ProgramDuration_ms" in p.columns:
        a = pd.to_numeric(p["ProgramDuration_ms"], errors="coerce").to_numpy(float) / 1000.0
        finite = np.flatnonzero(np.isfinite(a))
        if len(finite):
            base = a[finite[0]]
            return a - base
    if "Timestamp" in p.columns:
        t = pd.to_datetime(p["Timestamp"], errors="coerce")
        if t.notna().any():
            dt = t.diff().dt.total_seconds()
            pos = dt[(dt > 0) & np.isfinite(dt)]
            fallback = float(pos.median()) if len(pos) else 1.0
            inc = dt.where((dt > 0) & np.isfinite(dt), fallback)
            if len(inc):
                inc.iloc[0] = 0.0
            return inc.fillna(fallback).cumsum().to_numpy(float)
    return np.arange(len(p), dtype=float)


# =============================================================================
# 3. RAW FILE AUDIT
# =============================================================================
def audit_leaf_file(meta: pd.Series) -> Dict[str, object]:
    row = {
        "Battery": meta["battery"], "Role": meta["role"], "TestStart": meta["test_start"],
        "ZipPath": meta["zip_path"], "ZipName": meta["zip_name"],
        "Status": "PASS", "Error": "",
    }
    try:
        df0, zinfo = read_leaf_csv(Path(meta["zip_path"]))
        row.update(zinfo)
        if df0.empty:
            row.update({"Rows": 0, "Status": "EMPTY"})
            return row

        d = canonicalize_timeseries(df0)
        row["Rows"] = len(d)
        row["Columns"] = len(d.columns)
        row["ColumnNames"] = "|".join(map(str, d.columns))
        row["TemperatureSourceColumn"] = d.attrs.get("TemperatureSourceColumn", "")
        row["AcqTimeBasis"] = d.attrs.get("AcqTimeBasis", "")

        ts_ok = all(c in d.columns for c in TS_REQUIRED)
        row["TimeseriesSchemaOK"] = int(ts_ok)
        row["TemperatureAvailable"] = int("Temperature_C" in d.columns)

        if "ProgramDuration_ms" in d.columns:
            prog = pd.to_numeric(d["ProgramDuration_ms"], errors="coerce")
            row["ProgramDurationBackwards"] = int((prog.diff() < -1e-6).sum())
        else:
            row["ProgramDurationBackwards"] = np.nan

        if str(meta.get("role", "")).upper() == "TBA_ZYK":
            row.update(zyk_session_sanity(d))

        if "Timestamp" in d.columns:
            row["TimestampNaN"] = int(d["Timestamp"].isna().sum())
            row["TimeBackwards"] = int((d["Timestamp"].diff().dt.total_seconds() < 0).sum())
            row["DuplicateTimestamp"] = int(d["Timestamp"].duplicated(keep=False).sum())
            row["StartTime"] = str(d["Timestamp"].min())
            row["EndTime"] = str(d["Timestamp"].max())
        else:
            row["TimestampNaN"] = np.nan
            row["TimeBackwards"] = np.nan
            row["DuplicateTimestamp"] = np.nan

        row["ExactDuplicateRows"] = int(d.duplicated().sum())
        row["ExactDuplicateRate"] = float(d.duplicated().mean())

        if "LocalCycle" in d.columns:
            active = d[(d["LocalCycle"] > 0) & d.get("State", pd.Series("", index=d.index)).isin(["DCH", "CHA"])]
            cyc = sorted(set(pd.to_numeric(active["LocalCycle"], errors="coerce").dropna().astype(int)))
            row["ActiveCycleCount"] = len(cyc)
            row["LocalCycleMin"] = min(cyc) if cyc else np.nan
            row["LocalCycleMax"] = max(cyc) if cyc else np.nan
            if cyc:
                expected = set(range(min(cyc), max(cyc) + 1))
                missing = sorted(expected - set(cyc))
                row["InternalMissingCycleCount"] = len(missing)
                row["InternalMissingCycleIDs"] = ";".join(map(str, missing[:200]))
                row["InternalMissingCycleRate"] = len(missing) / max(1, len(expected))
            else:
                row["InternalMissingCycleCount"] = 0
                row["InternalMissingCycleIDs"] = ""
                row["InternalMissingCycleRate"] = 0.0

        if "State" in d.columns:
            vc = d["State"].value_counts(dropna=False)
            for st in ("DCH", "CHA", "PAU", "STO", "EIS"):
                row[f"StateRows_{st}"] = int(vc.get(st, 0))

        hard_invalid = np.zeros(len(d), dtype=bool)
        for col, (lo, hi) in {
            "Voltage_V": VOLTAGE_HARD_RANGE,
            "Current_A": CURRENT_HARD_RANGE,
            "Temperature_C": TEMPERATURE_HARD_RANGE,
        }.items():
            if col in d.columns:
                x = pd.to_numeric(d[col], errors="coerce")
                bad = x.notna() & ~x.between(lo, hi)
                row[f"{col}_NaN"] = int(x.isna().sum())
                row[f"{col}_HardInvalid"] = int(bad.sum())
                row[f"{col}_Min"] = safe_float(x.min())
                row[f"{col}_Median"] = safe_float(x.median())
                row[f"{col}_Max"] = safe_float(x.max())
                hard_invalid |= bad.to_numpy()
        row["AnyHardInvalidRows"] = int(hard_invalid.sum())
        row["HardInvalidRate"] = float(hard_invalid.mean()) if len(d) else 0.0

        # Temperature: distinguish sensor impossibility from nominal-protocol excursion.
        # Protocol excursion is PRESERVED, not cleaned.
        if "Temperature_C" in d.columns:
            tx = pd.to_numeric(d["Temperature_C"], errors="coerce")
            role_u = str(meta.get("role", "")).upper()
            if role_u == "TBA_ZYK":
                plo, phi = TEMPERATURE_NOMINAL_ZYK_RANGE
                protocol_out = tx.notna() & ~tx.between(plo, phi)
                row["Temperature_C_ProtocolOutOfNominal"] = int(protocol_out.sum())
                row["Temperature_C_ProtocolOutOfNominalRate"] = float(protocol_out.mean()) if len(tx) else 0.0

                finite = tx[np.isfinite(tx)]
                zero_ratio = float((finite.abs() <= TEMP_SESSION_ZERO_ABS_C).mean()) if len(finite) else np.nan
                temp_session_usable = not (
                    len(d) >= TEMP_SESSION_MIN_ROWS
                    and np.isfinite(zero_ratio)
                    and zero_ratio >= TEMP_SESSION_ZERO_RATIO
                )
                row["TemperatureSessionUsable"] = int(temp_session_usable)
                row["TemperatureSessionZeroLikeRatio"] = zero_ratio
                row["TemperatureSessionReason"] = "PASS" if temp_session_usable else "near_all_zero_temperature_channel"
            else:
                row["Temperature_C_ProtocolOutOfNominal"] = np.nan
                row["Temperature_C_ProtocolOutOfNominalRate"] = np.nan
                row["TemperatureSessionUsable"] = np.nan
                row["TemperatureSessionZeroLikeRatio"] = np.nan
                row["TemperatureSessionReason"] = ""

        # EIS-specific health check
        if "Zreal1" in d.columns or "ActFreq" in d.columns:
            zr = pd.to_numeric(d.get("Zreal1"), errors="coerce") if "Zreal1" in d else pd.Series(dtype=float)
            zi = pd.to_numeric(d.get("Zimg1"), errors="coerce") if "Zimg1" in d else pd.Series(dtype=float)
            fr = pd.to_numeric(d.get("ActFreq"), errors="coerce") if "ActFreq" in d else pd.Series(dtype=float)
            row["EISFiniteRows"] = int((zr.notna() & zi.notna() & fr.notna()).sum()) if len(zr) else 0

    except Exception as exc:
        row["Status"] = "ERROR"
        row["Error"] = repr(exc)
    return row


# =============================================================================
# 4. CLEANING
# =============================================================================
def _past_rolling_median_by_segment(x: pd.Series, seg: pd.Series) -> pd.Series:
    # transform preserves original order/index; only past values via shift(1)
    def f(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(
            window=CAUSAL_SPIKE_WINDOW,
            min_periods=CAUSAL_SPIKE_MIN_HISTORY,
        ).median()
    return x.groupby(seg, sort=False).transform(f)


def clean_timeseries(d: pd.DataFrame, role: str = "") -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Conservative, deployment-safe cleaning. No future samples used for spike baseline."""
    x = d.copy()
    n0 = len(x)

    # Preserve acquisition order. Do NOT sort by wall-clock Timestamp:
    # RWTH files contain clock/DST corrections (e.g. 03:59 -> 03:00) while ProgramDuration_ms remains monotonic.
    x["_RawOrder"] = np.arange(len(x))

    # Drop exact duplicates after canonicalization.
    dup = x.duplicated(subset=[c for c in x.columns if c != "_RawOrder"], keep="first")
    exact_dup = int(dup.sum())
    x = x.loc[~dup].reset_index(drop=True)

    # Hard invalid -> NaN, preserving raw in *_raw.
    hard_counts = {}
    for col, (lo, hi) in {
        "Voltage_V": VOLTAGE_HARD_RANGE,
        "Current_A": CURRENT_HARD_RANGE,
        "Temperature_C": TEMPERATURE_HARD_RANGE,
    }.items():
        if col not in x.columns:
            continue
        x[col + "_raw"] = x[col]
        bad = x[col].notna() & ~x[col].between(lo, hi)
        x[col + "_HardInvalidFlag"] = bad.astype(np.int8)
        hard_counts[col] = int(bad.sum())
        x.loc[bad, col] = np.nan

    # Session-level temperature-channel failure: keep electrical data, but do NOT fabricate temperature.
    # Example in audit: a complete ZYK session can contain essentially all 0°C values.
    temp_session_unusable = False
    temp_session_zero_ratio = np.nan
    if str(role).upper() == "TBA_ZYK" and "Temperature_C" in x.columns:
        tx = pd.to_numeric(x["Temperature_C"], errors="coerce")
        finite = tx[np.isfinite(tx)]
        temp_session_zero_ratio = float((finite.abs() <= TEMP_SESSION_ZERO_ABS_C).mean()) if len(finite) else np.nan
        if (
            len(x) >= TEMP_SESSION_MIN_ROWS
            and np.isfinite(temp_session_zero_ratio)
            and temp_session_zero_ratio >= TEMP_SESSION_ZERO_RATIO
        ):
            temp_session_unusable = True
            x["Temperature_C_SessionUnavailableFlag"] = np.int8(1)
            x["Temperature_C"] = np.nan
        else:
            x["Temperature_C_SessionUnavailableFlag"] = np.int8(0)

    # Segment resets at cycle/state/step boundary. This protects legitimate current steps.
    seg_cols = [c for c in ("GlobalCycle", "LocalCycle", "State", "Step") if c in x.columns]
    if seg_cols:
        change = pd.Series(False, index=x.index)
        for c in seg_cols:
            change |= x[c].ne(x[c].shift(1))
        seg = change.cumsum()
    else:
        seg = pd.Series(np.zeros(len(x), dtype=int), index=x.index)

    spike_counts = {}
    for col, abs_thr in CAUSAL_SPIKE_THRESHOLDS.items():
        if col not in x.columns:
            continue
        base = _past_rolling_median_by_segment(x[col], seg)
        deviation = (x[col] - base).abs()
        spike = base.notna() & x[col].notna() & (deviation > abs_thr)

        # Extra safeguard: if signal is close to previous value change regime, don't flag.
        # Requiring deviation from previous sample too suppresses slow real drift false positives.
        prev = x[col].groupby(seg, sort=False).shift(1)
        spike &= (x[col] - prev).abs().gt(abs_thr * 0.60)

        x[col + "_CausalBaseline"] = base
        x[col + "_SpikeFlag"] = spike.astype(np.int8)
        x[col + "_Clean"] = x[col]
        x.loc[spike, col + "_Clean"] = base.loc[spike]

        # Hard invalid: causal forward fill only inside segment; no backfill/future value.
        x[col + "_Clean"] = x[col + "_Clean"].groupby(seg, sort=False).ffill()
        spike_counts[col] = int(spike.sum())

    # Standard aliases: downstream always uses *_Clean if available.
    for col in ("Voltage_V", "Current_A", "Temperature_C"):
        if col in x.columns and col + "_Clean" not in x.columns:
            x[col + "_Clean"] = x[col]

    x["AnyHardInvalidFlag"] = 0
    x["AnySpikeFlag"] = 0
    for col in ("Voltage_V", "Current_A", "Temperature_C"):
        if col + "_HardInvalidFlag" in x:
            x["AnyHardInvalidFlag"] |= x[col + "_HardInvalidFlag"].astype(np.int8)
        if col + "_SpikeFlag" in x:
            x["AnySpikeFlag"] |= x[col + "_SpikeFlag"].astype(np.int8)

    report = {
        "RowsRaw": n0,
        "RowsAfterExactDedup": len(x),
        "ExactDuplicatesRemoved": exact_dup,
        "ExactDuplicateRate": exact_dup / max(1, n0),
        "HardInvalidRows": int(x["AnyHardInvalidFlag"].sum()),
        "HardInvalidRate": float(x["AnyHardInvalidFlag"].mean()) if len(x) else 0.0,
        "SpikeRows": int(x["AnySpikeFlag"].sum()),
        "SpikeRate": float(x["AnySpikeFlag"].mean()) if len(x) else 0.0,
        **{f"HardInvalid_{k}": v for k, v in hard_counts.items()},
        **{f"Spike_{k}": v for k, v in spike_counts.items()},
        "TemperatureSessionUsable": int(not temp_session_unusable),
        "TemperatureSessionZeroLikeRatio": temp_session_zero_ratio,
        "TemperatureSessionReason": "PASS" if not temp_session_unusable else "near_all_zero_temperature_channel",
        "CleaningFutureInformationUsed": 0,
    }
    return x.drop(columns=["_RawOrder"], errors="ignore"), report


# =============================================================================
# 5. GLOBAL CYCLE RECONSTRUCTION / PREFIX FEATURE HELPERS
# =============================================================================
def assign_global_cycles(d: pd.DataFrame, cycle_offset: int) -> Tuple[pd.DataFrame, int, Dict[str, object]]:
    x = d.copy()
    local = pd.to_numeric(x.get("LocalCycle"), errors="coerce")
    active = local.notna() & (local > 0) & x.get("State", pd.Series("", index=x.index)).isin(["DCH", "CHA"])
    ids = sorted(set(local.loc[active].astype(int)))
    if not ids:
        x["GlobalCycle"] = np.nan
        return x, cycle_offset, {"ActiveCycles": 0, "MissingLocalCycles": 0, "MissingLocalIDs": ""}

    local_max = max(ids)
    # Each TBA_Zyk file normally resets active cycle to 1. Using offset+LocalCycle preserves
    # internal missing IDs such as 76 -> 107 as true global gaps.
    x["GlobalCycle"] = np.where(local > 0, cycle_offset + local, np.nan)
    missing = sorted(set(range(1, local_max + 1)) - set(ids))
    report = {
        "ActiveCycles": len(ids), "LocalCycleMin": min(ids), "LocalCycleMax": local_max,
        "MissingLocalCycles": len(missing), "MissingLocalIDs": ";".join(map(str, missing[:500])),
        "GlobalCycleMin": cycle_offset + min(ids), "GlobalCycleMax": cycle_offset + local_max,
    }
    return x, cycle_offset + local_max, report


def collapse_duplicate_timestamps(prefix: pd.DataFrame) -> pd.DataFrame:
    """
    对同一采集时刻的非完全重复更新做合并。
    优先使用 AcqTime_s/ProgramDuration_ms，而不是墙上 Timestamp，避免DST/校时回跳重排。
    """
    if prefix.empty:
        return prefix
    p = prefix.copy()
    if "AcqTime_s" in p.columns:
        key = "AcqTime_s"
    elif "ProgramDuration_ms" in p.columns:
        key = "ProgramDuration_ms"
    elif "Timestamp" in p.columns:
        key = "Timestamp"
    else:
        return p

    p = p.sort_values(key, kind="mergesort").copy()
    if not p[key].duplicated().any():
        return p.reset_index(drop=True)
    agg = {}
    for c in p.columns:
        if c == key:
            continue
        if pd.api.types.is_numeric_dtype(p[c]):
            agg[c] = "median"
        else:
            agg[c] = "last"
    return p.groupby(key, as_index=False, sort=True).agg(agg)


def integrate_trapz(t_s: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(t_s) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    tt, yy = t_s[mask], y[mask]
    order = np.argsort(tt)
    tt, yy = tt[order], yy[order]
    keep = np.r_[True, np.diff(tt) > 0]
    tt, yy = tt[keep], yy[keep]
    if len(tt) < 2:
        return np.nan
    return float(np.trapezoid(yy, tt))


def linear_slope(t_s: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(t_s) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    tt, yy = t_s[mask], y[mask]
    if np.ptp(tt) < 1e-9:
        return np.nan
    tc = tt - np.mean(tt)
    den = np.dot(tc, tc)
    return float(np.dot(tc, yy - np.mean(yy)) / den) if den > 0 else np.nan


def derivative_stats(t_s: np.ndarray, y: np.ndarray, prefix: str) -> Dict[str, float]:
    mask = np.isfinite(t_s) & np.isfinite(y)
    tt, yy = t_s[mask], y[mask]
    if len(tt) < 4:
        return {f"{prefix}_Mean": np.nan, f"{prefix}_Std": np.nan, f"{prefix}_MaxAbs": np.nan, f"{prefix}_Last": np.nan}
    order = np.argsort(tt)
    tt, yy = tt[order], yy[order]
    dt = np.diff(tt)
    dy = np.diff(yy)
    ok = dt > 1e-9
    if ok.sum() < 2:
        return {f"{prefix}_Mean": np.nan, f"{prefix}_Std": np.nan, f"{prefix}_MaxAbs": np.nan, f"{prefix}_Last": np.nan}
    v = dy[ok] / dt[ok]
    return {
        f"{prefix}_Mean": float(np.nanmean(v)),
        f"{prefix}_Std": float(np.nanstd(v)),
        f"{prefix}_MaxAbs": float(np.nanmax(np.abs(v))),
        f"{prefix}_Last": float(v[-1]),
    }


def entropy_hist(x: np.ndarray, bins: int = 16) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 10 or np.nanstd(a) < 1e-12:
        return 0.0
    hist, _ = np.histogram(a, bins=bins)
    p = hist.astype(float)
    p = p[p > 0]
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def prefix_features(prefix: pd.DataFrame, ref_capacity_ah: float) -> Dict[str, float]:
    p = collapse_duplicate_timestamps(prefix)
    if len(p) < MIN_PREFIX_ROWS:
        return {}

    ts = _time_seconds(p)
    if not np.isfinite(ts).any() or np.nanmax(ts) <= 0:
        return {}

    V = pd.to_numeric(p.get("Voltage_V_Clean", p.get("Voltage_V")), errors="coerce").to_numpy(float)
    I = pd.to_numeric(p.get("Current_A_Clean", p.get("Current_A")), errors="coerce").to_numpy(float)
    temp_obj = p.get("Temperature_C_Clean", p.get("Temperature_C"))
    if temp_obj is None:
        T = np.full(len(p), np.nan, dtype=float)
    else:
        T = pd.to_numeric(temp_obj, errors="coerce").to_numpy(float)
    P = V * I

    out: Dict[str, float] = {
        "PrefixDuration_s": float(np.nanmax(ts)),
        "SampleCount": int(len(p)),
        "ObservedHardInvalidRate": float(pd.to_numeric(p.get("AnyHardInvalidFlag", 0), errors="coerce").fillna(0).mean()) if "AnyHardInvalidFlag" in p else 0.0,
        "ObservedSpikeRate": float(pd.to_numeric(p.get("AnySpikeFlag", 0), errors="coerce").fillna(0).mean()) if "AnySpikeFlag" in p else 0.0,
    }

    def basic(a: np.ndarray, name: str):
        q = a[np.isfinite(a)]
        if len(q) == 0:
            for k in ("Mean", "Std", "Median", "Min", "Max", "First", "Last", "Range"):
                out[f"{name}_{k}"] = np.nan
            return
        out[f"{name}_Mean"] = float(np.mean(q))
        out[f"{name}_Std"] = float(np.std(q))
        out[f"{name}_Median"] = float(np.median(q))
        out[f"{name}_Min"] = float(np.min(q))
        out[f"{name}_Max"] = float(np.max(q))
        idx = np.flatnonzero(np.isfinite(a))
        out[f"{name}_First"] = float(a[idx[0]])
        out[f"{name}_Last"] = float(a[idx[-1]])
        out[f"{name}_Range"] = float(np.max(q) - np.min(q))

    basic(V, "V")
    basic(I, "I")
    basic(T, "Temp")
    basic(P, "P")

    dur = max(1e-9, float(np.nanmax(ts)))
    out["V_TimeWeightedMean"] = integrate_trapz(ts, V) / dur
    out["I_TimeWeightedMean"] = integrate_trapz(ts, I) / dur
    out["Temp_TimeWeightedMean"] = integrate_trapz(ts, T) / dur
    out["P_AreaSigned_Wh"] = integrate_trapz(ts, P) / 3600.0
    out["P_AreaAbs_Wh"] = integrate_trapz(ts, np.abs(P)) / 3600.0
    out["PartialThroughput_Ah"] = integrate_trapz(ts, np.abs(I)) / 3600.0
    out["PartialDischarge_Ah"] = integrate_trapz(ts, np.maximum(-I, 0.0)) / 3600.0
    out["PartialCharge_Ah"] = integrate_trapz(ts, np.maximum(I, 0.0)) / 3600.0
    out["PartialThroughput_Wh"] = integrate_trapz(ts, np.abs(P)) / 3600.0
    out["RefAhProgress"] = out["PartialDischarge_Ah"] / ref_capacity_ah if np.isfinite(ref_capacity_ah) and ref_capacity_ah > 0 else np.nan

    out["V_Slope_V_per_s"] = linear_slope(ts, V)
    out["I_Slope_A_per_s"] = linear_slope(ts, I)
    out["Temp_Slope_C_per_min"] = linear_slope(ts, T) * 60.0
    out["Temp_Rise_C"] = out.get("Temp_Last", np.nan) - out.get("Temp_First", np.nan)

    out.update(derivative_stats(ts, V, "dVdt"))
    out.update(derivative_stats(ts, I, "dIdt"))
    dT = derivative_stats(ts, T, "dTdt_C_per_s")
    out.update(dT)

    out["I_AbsMean"] = float(np.nanmean(np.abs(I)))
    out["I_RMS"] = float(np.sqrt(np.nanmean(I ** 2)))
    out["I_CV"] = float(np.nanstd(I) / max(1e-9, np.nanmean(np.abs(I))))
    out["I_Entropy"] = entropy_hist(I)
    out["V_CV"] = float(np.nanstd(V) / max(1e-9, abs(np.nanmean(V))))
    out["Temp_CV"] = float(np.nanstd(T) / max(1e-9, abs(np.nanmean(T))))
    out["Temp_Above30_Ratio"] = float(np.nanmean(T > 30.0)) if np.isfinite(T).any() else np.nan
    out["Temp_Above35_Ratio"] = float(np.nanmean(T > 35.0)) if np.isfinite(T).any() else np.nan

    # Dynamic dV/dI resistance proxy for sufficiently exciting current profile.
    m = np.isfinite(V) & np.isfinite(I)
    if m.sum() >= 10 and np.std(I[m]) > 0.05:
        ii = I[m] - np.mean(I[m])
        den = float(np.dot(ii, ii))
        out["R_DynamicSlope_Ohm"] = float(np.dot(ii, V[m] - np.mean(V[m])) / den) if den > 0 else np.nan
        out["VI_Correlation"] = float(np.corrcoef(I[m], V[m])[0, 1]) if m.sum() > 2 else np.nan
    else:
        out["R_DynamicSlope_Ohm"] = np.nan
        out["VI_Correlation"] = np.nan

    # Voltage dwell bands useful for SOH/plateau shape; all within observed prefix.
    bands = [(3.0, 3.2), (3.2, 3.4), (3.4, 3.5), (3.5, 3.6), (3.6, 3.7), (3.7, 3.8), (3.8, 4.0), (4.0, 4.2)]
    finite_v = np.isfinite(V)
    for lo, hi in bands:
        name = f"VBand_{str(lo).replace('.', 'p')}_{str(hi).replace('.', 'p')}_Ratio"
        out[name] = float(np.mean((V[finite_v] >= lo) & (V[finite_v] < hi))) if finite_v.any() else np.nan

    return out


def get_discharge_cycle(d: pd.DataFrame, global_cycle: int) -> pd.DataFrame:
    g = d[(pd.to_numeric(d["GlobalCycle"], errors="coerce") == global_cycle) & d["State"].eq("DCH")].copy()
    if g.empty:
        return g
    I = pd.to_numeric(g.get("Current_A_Clean", g.get("Current_A")), errors="coerce")
    active = I < DISCHARGE_CURRENT_THRESHOLD_A
    if active.any():
        first_idx = np.flatnonzero(active.to_numpy())[0]
        g = g.iloc[first_idx:].copy()
    if "AcqTime_s" in g.columns:
        g = g.sort_values("AcqTime_s", kind="mergesort")
    elif "ProgramDuration_ms" in g.columns:
        g = g.sort_values("ProgramDuration_ms", kind="mergesort")
    else:
        g = g.sort_values("Timestamp", kind="mergesort")
    return g


def event_cut_index_refah(g: pd.DataFrame, ref_capacity_ah: float, fraction: float) -> Optional[int]:
    if not np.isfinite(ref_capacity_ah) or ref_capacity_ah <= 0:
        return None
    p = collapse_duplicate_timestamps(g)
    if len(p) < MIN_PREFIX_ROWS:
        return None
    ts = _time_seconds(p)
    I = pd.to_numeric(p.get("Current_A_Clean", p.get("Current_A")), errors="coerce").to_numpy(float)
    target = ref_capacity_ah * fraction
    # cumulative trapezoidal discharge Ah
    dt = np.diff(ts)
    cur = np.maximum(-I, 0.0)
    inc = np.where((dt > 0) & np.isfinite(dt), 0.5 * (cur[:-1] + cur[1:]) * dt / 3600.0, 0.0)
    cum = np.r_[0.0, np.cumsum(np.nan_to_num(inc, nan=0.0))]
    hit = np.flatnonzero(cum >= target)
    return int(hit[0]) if len(hit) else None


def event_cut_index_deltav(g: pd.DataFrame, delta_v: float) -> Optional[int]:
    p = collapse_duplicate_timestamps(g)
    if len(p) < MIN_PREFIX_ROWS:
        return None
    V = pd.to_numeric(p.get("Voltage_V_Clean", p.get("Voltage_V")), errors="coerce")
    ts = pd.Series(_time_seconds(p), index=p.index)

    # Robust start voltage from first 30 s (or first 8 finite points), then causal rolling median.
    early = V[(ts <= 30) & V.notna()]
    if len(early) < 3:
        early = V.dropna().head(8)
    if len(early) < 3:
        return None
    v0 = float(early.median())
    smooth = V.rolling(window=5, min_periods=3).median()
    hit = smooth <= (v0 - delta_v)
    # Require 3 observed hits in a 5-sample window to avoid pulse-only crossing.
    persist = hit.astype(int).rolling(5, min_periods=3).sum() >= 3
    ids = np.flatnonzero(persist.fillna(False).to_numpy())
    return int(ids[0]) if len(ids) else None


def extract_prefix_feature_rows(
    d: pd.DataFrame,
    battery: str,
    ref_capacity_ah: float,
    include_deltav: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    cycle_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []

    cycles = sorted(set(pd.to_numeric(d.get("GlobalCycle"), errors="coerce").dropna().astype(int)))
    for cyc in cycles:
        g = get_discharge_cycle(d, cyc)
        if len(g) < MIN_PREFIX_ROWS:
            continue
        rel_s = _time_seconds(g)
        dur = float(np.nanmax(rel_s)) if np.isfinite(rel_s).any() else np.nan
        if not np.isfinite(dur) or dur < MIN_DCH_DURATION_S:
            continue

        V = pd.to_numeric(g.get("Voltage_V_Clean", g.get("Voltage_V")), errors="coerce")
        I = pd.to_numeric(g.get("Current_A_Clean", g.get("Current_A")), errors="coerce")
        temp_obj = g.get("Temperature_C_Clean", g.get("Temperature_C"))
        if temp_obj is None:
            Temp = pd.Series(np.nan, index=g.index, dtype=float)
        else:
            Temp = pd.to_numeric(temp_obj, errors="coerce")
        full_ah = integrate_trapz(rel_s, np.maximum(-I.to_numpy(float), 0)) / 3600.0
        wall_t = pd.to_datetime(g.get("Timestamp"), errors="coerce") if "Timestamp" in g else pd.Series(pd.NaT, index=g.index)
        cycle_rows.append({
            "Battery": battery, "GlobalCycle": cyc,
            "DCH_Start": str(wall_t.iloc[0]) if len(wall_t) else "",
            "DCH_End": str(wall_t.iloc[-1]) if len(wall_t) else "",
            "DCH_Duration_s": dur, "DCH_Rows": len(g), "DCH_ObservedAh": full_ah,
            "DCH_MeanCurrent_A": safe_float(I.mean()), "DCH_MinCurrent_A": safe_float(I.min()),
            "DCH_MaxCurrent_A": safe_float(I.max()), "DCH_V_Start": safe_float(V.iloc[0] if len(V) else np.nan),
            "DCH_V_End": safe_float(V.iloc[-1] if len(V) else np.nan), "DCH_V_Min": safe_float(V.min()),
            "DCH_V_Max": safe_float(V.max()), "DCH_Temp_Mean_C": safe_float(Temp.mean()),
            "DCH_Temp_Max_C": safe_float(Temp.max()), "DCH_Temp_Rise_C": safe_float(Temp.iloc[-1] - Temp.iloc[0]) if len(Temp) else np.nan,
            "HardInvalidRate": float(pd.to_numeric(g.get("AnyHardInvalidFlag", 0), errors="coerce").fillna(0).mean()) if "AnyHardInvalidFlag" in g else 0.0,
            "SpikeRate": float(pd.to_numeric(g.get("AnySpikeFlag", 0), errors="coerce").fillna(0).mean()) if "AnySpikeFlag" in g else 0.0,
        })

        for node in FIXED_TIME_NODES_S:
            mask = rel_s <= node + 1e-9
            reachable = int(np.any(mask) and np.nanmax(rel_s) >= node)
            coverage_rows.append({"Battery": battery, "GlobalCycle": cyc, "Scope": "FixedTime", "Node": node, "Reachable": reachable})
            if not reachable:
                continue
            pref = g.loc[mask].copy()
            f = prefix_features(pref, ref_capacity_ah)
            if not f:
                continue
            fixed_rows.append({
                "Battery": battery, "GlobalCycle": cyc, "Scope": "Discharge_FixedTime",
                "NodeType": "Time_s", "NodeValue": float(node), "ReferenceCapacity_Ah": ref_capacity_ah,
                **f,
            })

        p_collapsed = collapse_duplicate_timestamps(g)
        for frac in EVENT_REF_AH_FRACTIONS:
            idx = event_cut_index_refah(g, ref_capacity_ah, frac)
            reachable = int(idx is not None)
            coverage_rows.append({"Battery": battery, "GlobalCycle": cyc, "Scope": "Event_RefAhFraction", "Node": frac, "Reachable": reachable})
            if idx is None:
                continue
            pref = p_collapsed.iloc[:idx + 1].copy()
            f = prefix_features(pref, ref_capacity_ah)
            if not f:
                continue
            event_rows.append({
                "Battery": battery, "GlobalCycle": cyc, "Scope": "Discharge_Event_RefAhFraction",
                "NodeType": "RefAhFraction", "NodeValue": float(frac), "ReferenceCapacity_Ah": ref_capacity_ah,
                **f,
            })

        if include_deltav:
            for dv in EVENT_DELTAV_NODES_V:
                idx = event_cut_index_deltav(g, dv)
                reachable = int(idx is not None)
                coverage_rows.append({"Battery": battery, "GlobalCycle": cyc, "Scope": "Event_DeltaV", "Node": dv, "Reachable": reachable})
                if idx is None:
                    continue
                pref = p_collapsed.iloc[:idx + 1].copy()
                f = prefix_features(pref, ref_capacity_ah)
                if not f:
                    continue
                event_rows.append({
                    "Battery": battery, "GlobalCycle": cyc, "Scope": "Discharge_Event_DeltaV",
                    "NodeType": "DeltaV_V", "NodeValue": float(dv), "ReferenceCapacity_Ah": ref_capacity_ah,
                    **f,
                })

    return pd.DataFrame(fixed_rows), pd.DataFrame(event_rows), pd.DataFrame(cycle_rows), pd.DataFrame(coverage_rows)


# =============================================================================
# 6. CAPACITY / SOH LABELS FROM BOL + CU/CUv2
# =============================================================================
def extract_capacity_from_checkup(d: pd.DataFrame, role: str, test_start: str, global_cycle_before: int) -> Optional[Dict[str, object]]:
    if d.empty or "State" not in d or "AhStep" not in d:
        return None
    q = d[d["State"].eq("DCH")].copy()
    if q.empty:
        return None
    q["AhStep"] = pd.to_numeric(q["AhStep"], errors="coerce")
    q["CurrentAbs"] = pd.to_numeric(q.get("Current_A"), errors="coerce").abs()

    # Standard capacity must use the same low-rate capacity-test protocol across BOL/CU.
    # In the uploaded cell 002, the standard capacity step is ~0.512 A; BOL_P2 also contains
    # a ~2 A full discharge, which must NOT be mixed into the SOH reference.
    if "Step" in q:
        per_step = q.groupby("Step", dropna=False).agg(
            MaxAh=("AhStep", "max"), Rows=("AhStep", "size"), MedianAbsCurrent=("CurrentAbs", "median")
        ).reset_index()
        per_step = per_step[
            per_step["MaxAh"].between(MIN_CAPACITY_AH, MAX_CAPACITY_AH)
            & per_step["MedianAbsCurrent"].between(
                REFERENCE_CAPACITY_CURRENT_A - REFERENCE_CAPACITY_CURRENT_TOL_A,
                REFERENCE_CAPACITY_CURRENT_A + REFERENCE_CAPACITY_CURRENT_TOL_A,
            )
        ].copy()
        if per_step.empty:
            return None
        per_step["CurrentDistance"] = (per_step["MedianAbsCurrent"] - REFERENCE_CAPACITY_CURRENT_A).abs()
        best = per_step.sort_values(["MaxAh", "CurrentDistance", "Rows"], ascending=[False, True, False]).iloc[0]
        capacity = float(best["MaxAh"])
        step = safe_float(best["Step"])
        current = safe_float(best["MedianAbsCurrent"])
    else:
        return None
    if not (MIN_CAPACITY_AH <= capacity <= MAX_CAPACITY_AH):
        return None

    return {
        "Role": role, "TestStart": test_start, "GlobalCycle": int(global_cycle_before),
        "Capacity_Ah": capacity, "CapacityStep": step, "CapacityStepMedianAbsCurrent_A": current,
        "LabelSource": "BOL_or_CheckUp_MajorDischargeStep",
    }


def build_soh_labels(checkups: pd.DataFrame, all_global_cycles: Sequence[int], battery: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    if checkups.empty:
        return pd.DataFrame(), pd.DataFrame(), np.nan
    c = checkups.copy().sort_values(["GlobalCycle", "TestStart"]).reset_index(drop=True)

    # Multiple checkups at same global cycle -> median capacity.
    sparse = c.groupby("GlobalCycle", as_index=False).agg(
        Capacity_Ah=("Capacity_Ah", "median"),
        CheckupCount=("Capacity_Ah", "size"),
        TestStart=("TestStart", "first"),
    )
    # SOH reference Q0:
    # 1) Prefer a valid BOL standard-capacity point (P1/P1v2/P2) at the beginning of life.
    # 2) If BOL extraction is unavailable, fall back to the earliest plausible measured point.
    #    This fallback is explicitly recorded so it can be audited later.
    # The RWTH UR18650E cells are ~1.85 Ah nominal; the broad 1.5~2.2 Ah range is used
    # only to reject obvious partial-discharge candidates as Q0, not to reject later aged capacities.
    ref_source = "Unavailable"
    ref = np.nan
    if "Role" in c.columns:
        role_u = c["Role"].astype(str).str.upper()
        bol = c[
            role_u.isin({"TBA_BOL_P1", "TBA_BOL_P1V2", "TBA_BOL_P2"})
            & pd.to_numeric(c["Capacity_Ah"], errors="coerce").between(1.50, 2.20)
        ].copy()
        if not bol.empty:
            # Prefer the largest valid full standard-capacity discharge among BOL candidates;
            # partial discharges at the same nominal current must not define Q0.
            rr = bol.sort_values(["Capacity_Ah", "TestStart"], ascending=[False, True]).iloc[0]
            ref = float(rr["Capacity_Ah"])
            ref_source = f"BOL:{rr['Role']}"
    if not np.isfinite(ref):
        plausible = c[pd.to_numeric(c["Capacity_Ah"], errors="coerce").between(1.50, 2.20)].copy()
        if not plausible.empty:
            rr = plausible.sort_values(["GlobalCycle", "TestStart"]).iloc[0]
            ref = float(rr["Capacity_Ah"])
            ref_source = f"EarliestPlausible:{rr.get('Role', 'Unknown')}"
    sparse["Battery"] = battery
    sparse["ReferenceCapacity_Ah"] = ref
    sparse["ReferenceCapacitySource"] = ref_source
    sparse["SOH"] = sparse["Capacity_Ah"] / ref if np.isfinite(ref) and ref > 0 else np.nan
    sparse["SOH_LabelType"] = "Measured_CheckUp"

    cyc = np.asarray(sorted(set(int(x) for x in all_global_cycles if int(x) > 0)), dtype=int)
    if len(cyc) == 0 or len(sparse) < 2:
        return sparse, pd.DataFrame(), ref

    xp = sparse["GlobalCycle"].to_numpy(float)
    fp = sparse["SOH"].to_numpy(float)
    good = np.isfinite(xp) & np.isfinite(fp)
    xp, fp = xp[good], fp[good]
    if len(xp) < 2:
        return sparse, pd.DataFrame(), ref

    # Only interpolate inside measured support. Never extrapolate beyond first/last checkup.
    inside = (cyc >= np.min(xp)) & (cyc <= np.max(xp))
    labels = pd.DataFrame({"Battery": battery, "GlobalCycle": cyc})
    labels["SOH_Target_SameCycle"] = np.nan
    labels.loc[inside, "SOH_Target_SameCycle"] = np.interp(cyc[inside].astype(float), xp, fp)
    measured_map = dict(zip(sparse["GlobalCycle"].astype(int), sparse["SOH"]))
    labels["SOH_IsMeasuredCheckUp"] = labels["GlobalCycle"].map(lambda z: int(int(z) in measured_map))
    labels["SOH_LabelUsable"] = labels["SOH_Target_SameCycle"].notna().astype(int)
    labels["SOH_LabelSource"] = np.where(labels["SOH_IsMeasuredCheckUp"].eq(1), "Measured_CheckUp", "Offline_LinearInterpolation_BetweenCheckUps")
    labels["ReferenceCapacity_Ah"] = ref
    labels["ReferenceCapacitySource"] = ref_source
    labels["FutureInformationUsedAsModelInput"] = 0
    labels["ImportantNote"] = "Interpolation is supervised-label construction only; never a prefix/model input."
    return sparse, labels, ref


# =============================================================================
# 7. PER-BATTERY PIPELINE
# =============================================================================
def process_battery(
    battery: str,
    inv_b: pd.DataFrame,
    output_root: Path,
    save_clean_timeseries: bool,
    include_deltav: bool,
    force: bool,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    bdir = ensure_dir(output_root / "02_batteries" / f"Cell_{battery}")
    done = bdir / "99_COMPLETE.json"
    if done.exists() and not force:
        try:
            prev = json.loads(done.read_text(encoding="utf-8"))
            prev["SkippedExisting"] = 1
            return prev
        except Exception:
            pass

    files = inv_b.copy()
    files["test_start_dt"] = pd.to_datetime(files["test_start"], errors="coerce")
    files = files.sort_values(["test_start_dt", "zip_name"], na_position="last").reset_index(drop=True)

    file_reports = []
    clean_reports = []
    cycle_map_reports = []
    zyk_parts: List[pd.DataFrame] = []
    checkup_rows: List[Dict[str, object]] = []
    cycle_offset = 0

    for j, r in files.iterrows():
        role = str(r["role"])
        zp = Path(r["zip_path"])
        try:
            raw, zi = read_leaf_csv(zp)
            base_rep = {"Battery": battery, "Role": role, "TestStart": r["test_start"], "ZipName": r["zip_name"], **zi}
            if raw.empty:
                file_reports.append({**base_rep, "Status": "EMPTY", "Rows": 0})
                continue
            d = canonicalize_timeseries(raw)
            file_reports.append({**base_rep, "Status": "OK", "Rows": len(d)})

            role_u = role.upper()
            is_zyk = role_u == "TBA_ZYK"
            is_check = role_u in {"TBA_CU", "TBA_CUV2", "TBA_BOL_P2", "TBA_BOL_P1", "TBA_BOL_P1V2"}

            if is_zyk:
                sess = zyk_session_sanity(d)
                if not bool(sess.get("ZYKSessionUsable", 0)):
                    # 关键：异常伪ZYK不推进GlobalCycle offset，也不进入HI。
                    file_reports[-1].update({
                        "Status": "EXCLUDED_ZYK_SESSION",
                        **sess,
                    })
                    clean_reports.append({
                        "Battery": battery, "Role": role, "TestStart": r["test_start"],
                        "ZipName": r["zip_name"], **sess,
                        "RowsRaw": len(d), "RowsAfterExactDedup": 0,
                        "ExactDuplicatesRemoved": 0, "HardInvalidRows": 0,
                        "SpikeRows": 0, "CleaningFutureInformationUsed": 0,
                    })
                    continue

                d, new_offset, cmap = assign_global_cycles(d, cycle_offset)
                cmap.update({"Battery": battery, "Role": role, "TestStart": r["test_start"], "ZipName": r["zip_name"], "CycleOffsetBefore": cycle_offset, "CycleOffsetAfter": new_offset})
                cycle_map_reports.append(cmap)
                cycle_offset = new_offset

                dclean, crep = clean_timeseries(d, role=role)
                crep.update({"Battery": battery, "Role": role, "TestStart": r["test_start"], "ZipName": r["zip_name"]})
                clean_reports.append(crep)

                if save_clean_timeseries:
                    part_name = f"{j:04d}_{pd.to_datetime(r['test_start'], errors='coerce').strftime('%Y%m%d_%H%M%S') if pd.notna(pd.to_datetime(r['test_start'], errors='coerce')) else 'NA'}_TBA_Zyk_clean.csv.gz"
                    cols = [
                        "Timestamp", "AcqTime_s", "ProgramDuration_ms", "GlobalCycle", "LocalCycle", "Step", "State", "Procedure",
                        "AhBattery", "AhCharge", "AhDischarge", "AhStep", "Energy_Wh", "WhStep",
                        "Voltage_V_raw", "Voltage_V_Clean", "Current_A_raw", "Current_A_Clean",
                        "Temperature_C_raw", "Temperature_C_Clean", "Temperature_C_SessionUnavailableFlag",
                        "Voltage_V_HardInvalidFlag", "Current_A_HardInvalidFlag", "Temperature_C_HardInvalidFlag",
                        "Voltage_V_SpikeFlag", "Current_A_SpikeFlag", "Temperature_C_SpikeFlag",
                        "AnyHardInvalidFlag", "AnySpikeFlag",
                    ]
                    cols = [c for c in cols if c in dclean.columns]
                    save_csv(dclean[cols], bdir / "clean_timeseries_parts" / part_name)

                # Keep each ZYK part in RAM one by one; typical file size manageable. For feature extraction
                # we can accumulate only canonical cleaned columns. If memory is tight, concatenation occurs
                # after all files; 48-cell processing should use n_jobs<=4.
                keep = [c for c in [
                    "Timestamp", "AcqTime_s", "ProgramDuration_ms", "GlobalCycle", "LocalCycle", "Step", "State", "Procedure",
                    "AhStep", "Voltage_V", "Current_A", "Temperature_C",
                    "Voltage_V_Clean", "Current_A_Clean", "Temperature_C_Clean",
                    "AnyHardInvalidFlag", "AnySpikeFlag",
                ] if c in dclean.columns]
                zyk_parts.append(dclean[keep].copy())

            elif is_check:
                # Check-up/BOL capacity belongs to state after all ZYK cycles completed before this timestamp.
                dclean, crep = clean_timeseries(d, role=role)
                crep.update({"Battery": battery, "Role": role, "TestStart": r["test_start"], "ZipName": r["zip_name"]})
                clean_reports.append(crep)
                cap = extract_capacity_from_checkup(dclean, role, r["test_start"], cycle_offset)
                if cap is not None:
                    checkup_rows.append(cap)

        except Exception as exc:
            file_reports.append({
                "Battery": battery, "Role": role, "TestStart": r["test_start"], "ZipName": r["zip_name"],
                "Status": "ERROR", "Rows": np.nan, "Error": repr(exc),
            })

    file_df = pd.DataFrame(file_reports)
    clean_df = pd.DataFrame(clean_reports)
    cmap_df = pd.DataFrame(cycle_map_reports)
    save_csv(file_df, bdir / "00_file_processing_status.csv")
    save_csv(clean_df, bdir / "01_cleaning_summary.csv")
    save_csv(cmap_df, bdir / "02_global_cycle_reconstruction.csv")

    if not zyk_parts:
        result = {"Battery": battery, "Status": "FAIL_NO_ZYK", "Elapsed_s": time.perf_counter() - t0}
        save_json(result, done)
        return result

    zyk = pd.concat(zyk_parts, ignore_index=True)
    # CRITICAL: preserve acquisition chronology within each global cycle.
    # Do NOT sort by wall-clock Timestamp because DST/clock corrections can move backward by ~1 hour.
    sort_cols = ["GlobalCycle"]
    if "AcqTime_s" in zyk.columns:
        sort_cols.append("AcqTime_s")
    elif "ProgramDuration_ms" in zyk.columns:
        sort_cols.append("ProgramDuration_ms")
    elif "Timestamp" in zyk.columns:
        sort_cols.append("Timestamp")
    zyk = zyk.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    all_cycles = sorted(set(pd.to_numeric(zyk["GlobalCycle"], errors="coerce").dropna().astype(int)))

    checkups = pd.DataFrame(checkup_rows)
    sparse_soh, labels, ref_capacity = build_soh_labels(checkups, all_cycles, battery)
    save_csv(checkups, bdir / "03_checkup_capacity_raw.csv")
    save_csv(sparse_soh, bdir / "04_sparse_soh_reference.csv")
    save_csv(labels, bdir / "05_cycle_soh_labels.csv")

    fixed, event, cycle_summary, coverage = extract_prefix_feature_rows(
        zyk, battery=battery, ref_capacity_ah=ref_capacity, include_deltav=include_deltav
    )

    # Merge label only after features are built. Labels never influence prefix construction.
    label_cols = ["Battery", "GlobalCycle", "SOH_Target_SameCycle", "SOH_IsMeasuredCheckUp", "SOH_LabelUsable", "SOH_LabelSource"]
    if not labels.empty:
        fixed = fixed.merge(labels[label_cols], on=["Battery", "GlobalCycle"], how="left") if not fixed.empty else fixed
        event = event.merge(labels[label_cols], on=["Battery", "GlobalCycle"], how="left") if not event.empty else event
        cycle_summary = cycle_summary.merge(labels[label_cols], on=["Battery", "GlobalCycle"], how="left") if not cycle_summary.empty else cycle_summary

    save_csv(cycle_summary, bdir / "06_cycle_physical_summary.csv")
    save_csv(fixed, bdir / "07_fixedtime_features.csv.gz")
    save_csv(event, bdir / "08_event_features.csv.gz")
    save_csv(coverage, bdir / "09_prefix_coverage.csv")

    # Battery-level summary
    internal_missing = int(cmap_df.get("MissingLocalCycles", pd.Series(dtype=float)).fillna(0).sum()) if not cmap_df.empty else 0
    max_global = int(max(all_cycles)) if all_cycles else 0
    active_recorded = len(all_cycles)
    missing_rate = internal_missing / max(1, active_recorded + internal_missing)
    # Formal sensor-quality gate is based on TBA_Zyk aging data only. BOL may intentionally include
    # characterization conditions outside the aging temperature range and must not poison the gate.
    qclean = clean_df[clean_df.get("Role", pd.Series("", index=clean_df.index)).astype(str).str.upper().eq("TBA_ZYK")].copy() if not clean_df.empty else pd.DataFrame()
    total_clean_rows = int(qclean.get("RowsAfterExactDedup", pd.Series(dtype=float)).fillna(0).sum()) if not qclean.empty else 0
    total_spikes = int(qclean.get("SpikeRows", pd.Series(dtype=float)).fillna(0).sum()) if not qclean.empty else 0
    total_invalid = int(qclean.get("HardInvalidRows", pd.Series(dtype=float)).fillna(0).sum()) if not qclean.empty else 0
    total_dup = int(qclean.get("ExactDuplicatesRemoved", pd.Series(dtype=float)).fillna(0).sum()) if not qclean.empty else 0
    temp_unusable_sessions = int((pd.to_numeric(qclean.get("TemperatureSessionUsable", pd.Series(dtype=float)), errors="coerce") == 0).sum()) if not qclean.empty else 0

    summary = {
        "Battery": battery,
        "Status": "PASS",
        "LeafZipFiles": len(files),
        "ZYKFiles": int(files["role"].astype(str).str.upper().eq("TBA_ZYK").sum()),
        "CheckUpFiles": int(files["role"].astype(str).str.upper().isin(["TBA_CU", "TBA_CUV2"]).sum()),
        "EISFiles": int(files["role"].astype(str).str.contains("EIS", case=False, na=False).sum()),
        "RecordedGlobalCycles": active_recorded,
        "GlobalCycleMaxIncludingGaps": max_global,
        "InternalMissingCycles": internal_missing,
        "InternalMissingCycleRate": missing_rate,
        "ReferenceCapacity_Ah": ref_capacity,
        "MeasuredSOHPoints": len(sparse_soh),
        "UsableSOHLabelCycles": int(labels.get("SOH_LabelUsable", pd.Series(dtype=int)).sum()) if not labels.empty else 0,
        "CleanRows": total_clean_rows,
        "ExactDuplicatesRemoved": total_dup,
        "ExactDuplicateRate": total_dup / max(1, total_clean_rows + total_dup),
        "HardInvalidRows": total_invalid,
        "HardInvalidRate": total_invalid / max(1, total_clean_rows),
        "SpikeRows": total_spikes,
        "SpikeRate": total_spikes / max(1, total_clean_rows),
        "TemperatureUnavailableZYKSessions": temp_unusable_sessions,
        "FixedTimeFeatureRows": len(fixed),
        "EventFeatureRows": len(event),
        "Elapsed_s": time.perf_counter() - t0,
    }

    # Quality gates
    gates = {
        "EnoughZYKFiles": summary["ZYKFiles"] >= MIN_ZYK_FILES_FOR_PASS,
        "EnoughCycles": summary["RecordedGlobalCycles"] >= MIN_USABLE_CYCLES_FOR_PASS,
        "DuplicateRateOK": summary["ExactDuplicateRate"] <= MAX_DUPLICATE_RATE_FOR_PASS,
        "HardInvalidRateOK": summary["HardInvalidRate"] <= MAX_HARD_INVALID_RATE_FOR_PASS,
        "SpikeRateOK": summary["SpikeRate"] <= MAX_SPIKE_RATE_FOR_PASS,
        "MissingCycleRateOK": summary["InternalMissingCycleRate"] <= MAX_INTERNAL_MISSING_CYCLE_RATE_FOR_PASS,
        "ReferenceCapacityOK": np.isfinite(summary["ReferenceCapacity_Ah"]) and MIN_CAPACITY_AH <= summary["ReferenceCapacity_Ah"] <= MAX_CAPACITY_AH,
        "SOHLabelsAvailable": summary["MeasuredSOHPoints"] >= 2 and summary["UsableSOHLabelCycles"] > 0,
    }
    gate_df = pd.DataFrame([{"Check": k, "Pass": int(v)} for k, v in gates.items()])
    save_csv(gate_df, bdir / "10_quality_gate.csv")
    summary["QualityGatePASS"] = int(all(gates.values()))
    if not all(gates.values()):
        summary["Status"] = "WARN_REVIEW"

    save_json(summary, done)
    return summary


# =============================================================================
# 8. GLOBAL AGGREGATION / PROTOCOL AUDIT / SPLIT / FIGURES
# =============================================================================
def aggregate_battery_outputs(output_root: Path) -> Dict[str, pd.DataFrame]:
    bdirs = sorted((output_root / "02_batteries").glob("Cell_*")) if (output_root / "02_batteries").exists() else []
    fixeds, events, cycles, labels, sparse, summaries, coverages = [], [], [], [], [], [], []
    for b in bdirs:
        comp = b / "99_COMPLETE.json"
        if comp.exists():
            try:
                summaries.append(json.loads(comp.read_text(encoding="utf-8")))
            except Exception:
                pass
        for p, arr in [
            (b / "07_fixedtime_features.csv.gz", fixeds),
            (b / "08_event_features.csv.gz", events),
            (b / "06_cycle_physical_summary.csv", cycles),
            (b / "05_cycle_soh_labels.csv", labels),
            (b / "04_sparse_soh_reference.csv", sparse),
            (b / "09_prefix_coverage.csv", coverages),
        ]:
            if p.exists() and p.stat().st_size > 0:
                try:
                    arr.append(pd.read_csv(p, low_memory=False))
                except pd.errors.EmptyDataError:
                    pass
    return {
        "battery_summary": pd.DataFrame(summaries),
        "fixed": pd.concat(fixeds, ignore_index=True) if fixeds else pd.DataFrame(),
        "event": pd.concat(events, ignore_index=True) if events else pd.DataFrame(),
        "cycles": pd.concat(cycles, ignore_index=True) if cycles else pd.DataFrame(),
        "labels": pd.concat(labels, ignore_index=True) if labels else pd.DataFrame(),
        "sparse": pd.concat(sparse, ignore_index=True) if sparse else pd.DataFrame(),
        "coverage": pd.concat(coverages, ignore_index=True) if coverages else pd.DataFrame(),
    }


def protocol_consistency_audit(cycles: pd.DataFrame, battery_summary: pd.DataFrame) -> pd.DataFrame:
    if cycles.empty:
        return pd.DataFrame()
    metrics = cycles.groupby("Battery", as_index=False).agg(
        Median_DCH_Duration_s=("DCH_Duration_s", "median"),
        Median_DCH_MeanCurrent_A=("DCH_MeanCurrent_A", "median"),
        Median_DCH_ObservedAh=("DCH_ObservedAh", "median"),
        Median_DCH_V_Start=("DCH_V_Start", "median"),
        Median_DCH_V_End=("DCH_V_End", "median"),
        Median_DCH_Temp_C=("DCH_Temp_Mean_C", "median"),
        Median_DCH_TempRise_C=("DCH_Temp_Rise_C", "median"),
        CycleCount=("GlobalCycle", "nunique"),
    )
    if not battery_summary.empty and "ReferenceCapacity_Ah" in battery_summary:
        metrics = metrics.merge(battery_summary[["Battery", "ReferenceCapacity_Ah", "QualityGatePASS"]], on="Battery", how="left")

    # Robust-z alone is unstable when the 48 cells are *extremely* consistent.
    # Example: median discharge duration differs by only ~0.01 s, but MAD is so tiny that
    # an ordinary cell can receive |robust-z| > 5.  Therefore a protocol outlier now requires
    # BOTH a large robust-z and a physically meaningful absolute deviation from the cohort median.
    protocol_abs_tol = {
        "Median_DCH_Duration_s": 10.0,       # s
        "Median_DCH_MeanCurrent_A": 0.25,    # A
        "Median_DCH_ObservedAh": 0.15,       # Ah
        "Median_DCH_V_Start": 0.05,          # V
        "Median_DCH_V_End": 0.10,            # V
        "Median_DCH_Temp_C": 2.5,            # degC
        "ReferenceCapacity_Ah": 0.15,        # Ah
    }
    zcols = []
    flagcols = []
    for c in [
        "Median_DCH_Duration_s", "Median_DCH_MeanCurrent_A", "Median_DCH_ObservedAh",
        "Median_DCH_V_Start", "Median_DCH_V_End", "Median_DCH_Temp_C", "ReferenceCapacity_Ah",
    ]:
        if c in metrics:
            zc = c + "_RobustZ"
            metrics[zc] = robust_z(metrics[c])
            zcols.append(zc)
            med = pd.to_numeric(metrics[c], errors="coerce").median()
            dev = (pd.to_numeric(metrics[c], errors="coerce") - med).abs()
            fc = c + "_ProtocolOutlierFlag"
            metrics[fc] = (
                metrics[zc].abs().gt(PROTOCOL_ROBUST_Z_THRESHOLD)
                & dev.gt(float(protocol_abs_tol[c]))
            ).astype(int)
            flagcols.append(fc)
    metrics["ProtocolOutlier"] = metrics[flagcols].max(axis=1).astype(int) if flagcols else 0
    if flagcols:
        metrics["ProtocolOutlierReason"] = metrics.apply(
            lambda r: "|".join(fc.replace("_ProtocolOutlierFlag", "") for fc in flagcols if int(r.get(fc, 0)) == 1),
            axis=1,
        )
    else:
        metrics["ProtocolOutlierReason"] = ""
    metrics["SameConditionGate"] = ((metrics.get("QualityGatePASS", 1) == 1) & (metrics["ProtocolOutlier"] == 0)).astype(int)
    return metrics


def make_feature_registry(fixed: pd.DataFrame, event: pd.DataFrame) -> pd.DataFrame:
    meta = {
        "Battery", "GlobalCycle", "Scope", "NodeType", "NodeValue", "ReferenceCapacity_Ah",
        "SOH_Target_SameCycle", "SOH_IsMeasuredCheckUp", "SOH_LabelUsable", "SOH_LabelSource",
    }
    cols = sorted((set(fixed.columns) | set(event.columns)) - meta)
    rows = []
    for c in cols:
        if c.startswith("Temp") or c.startswith("dTdt"):
            fam = "Temperature"
        elif c.startswith("V") or c.startswith("dVdt"):
            fam = "Voltage"
        elif c.startswith("I") or c.startswith("dIdt"):
            fam = "Current"
        elif c.startswith("P"):
            fam = "Power"
        elif "Ah" in c or "Wh" in c or "Throughput" in c or "Progress" in c:
            fam = "Throughput"
        elif c.startswith("R_") or c.startswith("VI_"):
            fam = "DynamicResistanceProxy"
        elif "Invalid" in c or "Spike" in c:
            fam = "QualityAudit"
        else:
            fam = "TimingOrOther"
        rows.append({
            "Feature": c,
            "Family": fam,
            "CausalPrefixLegal": 1,
            "AvailableInFixedTime": int(c in fixed.columns),
            "AvailableInEvent": int(c in event.columns),
            "TemperatureFeature": int(fam == "Temperature"),
            "Note": "Derived only from observed prefix; no current-cycle full capacity/duration used.",
        })
    return pd.DataFrame(rows)


def suggest_split(protocol: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    if protocol.empty:
        return pd.DataFrame()
    d = protocol.copy()
    eligible = d[d["SameConditionGate"].eq(1)].copy()
    if eligible.empty:
        return pd.DataFrame()

    # Stratify approximately by lifecycle length quartile, then deterministic shuffle inside strata.
    try:
        eligible["LifeStratum"] = pd.qcut(eligible["CycleCount"].rank(method="first"), q=min(4, len(eligible)), labels=False, duplicates="drop")
    except Exception:
        eligible["LifeStratum"] = 0
    rng = np.random.default_rng(seed)
    eligible["_rand"] = rng.random(len(eligible))
    eligible = eligible.sort_values(["LifeStratum", "_rand"]).reset_index(drop=True)

    # ~2/3 train, 1/6 val, 1/6 test; for exactly48 => 32/8/8.
    n = len(eligible)
    n_train = int(round(n * 2 / 3))
    n_val = int(round(n / 6))
    n_test = n - n_train - n_val
    # Interleave strata via sorted table, then assign by index for balanced distribution.
    roles = np.array(["Train"] * n_train + ["Validation"] * n_val + ["Test"] * n_test, dtype=object)
    rng.shuffle(roles)
    eligible["SuggestedSplit"] = roles
    eligible["SplitSeed"] = seed
    return eligible.drop(columns=["_rand"], errors="ignore")


def make_figures(data: Dict[str, pd.DataFrame], protocol: pd.DataFrame, fig_dir: Path) -> None:
    ensure_dir(fig_dir)
    summary = data["battery_summary"]
    cycles = data["cycles"]
    sparse = data["sparse"]

    if not summary.empty and "RecordedGlobalCycles" in summary:
        s = summary.sort_values("Battery")
        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.bar(s["Battery"].astype(str), s["RecordedGlobalCycles"])
        ax.set_xlabel("Battery")
        ax.set_ylabel("Recorded global cycles")
        ax.set_title("RWTH48: usable aging cycles per cell")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout(); fig.savefig(fig_dir / "01_cycle_count_by_battery.png", dpi=220); plt.close(fig)

        if "SpikeRate" in s and "HardInvalidRate" in s:
            fig, ax = plt.subplots(figsize=(14, 5.5))
            ax.plot(s["Battery"].astype(str), s["SpikeRate"], marker="o", label="Causal spike rate")
            ax.plot(s["Battery"].astype(str), s["HardInvalidRate"], marker="o", label="Hard invalid rate")
            ax.set_xlabel("Battery"); ax.set_ylabel("Rate"); ax.set_title("RWTH48 sensor-cleaning audit")
            ax.tick_params(axis="x", rotation=90); ax.legend(); fig.tight_layout()
            fig.savefig(fig_dir / "02_sensor_issue_rate.png", dpi=220); plt.close(fig)

    if not protocol.empty:
        p = protocol.sort_values("Battery")
        for i, c in enumerate(["Median_DCH_Duration_s", "Median_DCH_MeanCurrent_A", "Median_DCH_Temp_C"]):
            if c not in p: continue
            fig, ax = plt.subplots(figsize=(14, 5.5))
            ax.plot(p["Battery"].astype(str), p[c], marker="o")
            ax.set_xlabel("Battery"); ax.set_ylabel(c); ax.set_title(f"Same-condition protocol audit: {c}")
            ax.tick_params(axis="x", rotation=90); fig.tight_layout()
            fig.savefig(fig_dir / f"{3+i:02d}_protocol_{c}.png", dpi=220); plt.close(fig)

    if not sparse.empty and {"GlobalCycle", "SOH", "Battery"}.issubset(sparse.columns):
        fig, ax = plt.subplots(figsize=(14, 7))
        for bat, g in sparse.groupby("Battery"):
            ax.plot(g["GlobalCycle"], g["SOH"], marker="o", markersize=2, linewidth=0.8, alpha=0.65)
        ax.set_xlabel("Global cycle"); ax.set_ylabel("SOH (check-up reference)")
        ax.set_title("RWTH48 measured check-up SOH trajectories")
        fig.tight_layout(); fig.savefig(fig_dir / "06_all_cells_sparse_soh.png", dpi=220); plt.close(fig)


def aggregate_and_save(output_root: Path) -> None:
    data = aggregate_battery_outputs(output_root)
    gdir = ensure_dir(output_root / "03_global")
    save_csv(data["battery_summary"], gdir / "00_battery_quality_summary.csv")
    save_csv(data["cycles"], gdir / "01_all_cycle_physical_summary.csv.gz")
    save_csv(data["labels"], gdir / "02_all_cycle_soh_labels.csv.gz")
    save_csv(data["sparse"], gdir / "03_all_sparse_soh_reference.csv")
    save_csv(data["fixed"], gdir / "04_ALL_FIXEDTIME_MODEL_READY.csv.gz")
    save_csv(data["event"], gdir / "05_ALL_EVENT_MODEL_READY.csv.gz")
    save_csv(data["coverage"], gdir / "06_ALL_prefix_coverage.csv.gz")

    protocol = protocol_consistency_audit(data["cycles"], data["battery_summary"])
    save_csv(protocol, gdir / "07_same_condition_protocol_audit.csv")
    registry = make_feature_registry(data["fixed"], data["event"])
    save_csv(registry, gdir / "08_feature_registry.csv")
    split = suggest_split(protocol)
    save_csv(split, gdir / "09_suggested_battery_level_split.csv")

    # Coverage macro
    if not data["coverage"].empty:
        cov = data["coverage"].groupby(["Scope", "Node"], as_index=False).agg(
            Rows=("Reachable", "size"), Reachable=("Reachable", "sum"), Reachability=("Reachable", "mean")
        )
        save_csv(cov, gdir / "10_prefix_coverage_summary.csv")

    make_figures(data, protocol, output_root / "04_figures")

    # Final issue summary
    issues = []
    if not data["battery_summary"].empty:
        for r in data["battery_summary"].itertuples(index=False):
            if int(getattr(r, "QualityGatePASS", 0)) != 1:
                issues.append({"Battery": getattr(r, "Battery", ""), "Issue": "BatteryQualityGateNotPASS"})
    if not protocol.empty:
        for r in protocol[protocol["ProtocolOutlier"].eq(1)].itertuples(index=False):
            issues.append({"Battery": getattr(r, "Battery", ""), "Issue": "ProtocolOutlier"})
    save_csv(pd.DataFrame(issues, columns=["Battery", "Issue"]), gdir / "11_issue_summary.csv")

    final = {
        "Task": "RWTH48 full raw-data audit/preprocess/Event+FixedTime HI",
        "BatteryCountProcessed": int(data["battery_summary"]["Battery"].nunique()) if not data["battery_summary"].empty else 0,
        "BatteryQualityPass": int(data["battery_summary"].get("QualityGatePASS", pd.Series(dtype=int)).sum()) if not data["battery_summary"].empty else 0,
        "SameConditionGatePass": int(protocol.get("SameConditionGate", pd.Series(dtype=int)).sum()) if not protocol.empty else 0,
        "FixedTimeRows": len(data["fixed"]),
        "EventRows": len(data["event"]),
        "SparseSOHRows": len(data["sparse"]),
        "Important": [
            "Primary Event is frozen-reference Ah fraction (20/30/40%), not current-cycle final-capacity fraction.",
            "Temperature features are extracted in both FixedTime and Event scopes.",
            "SOH interpolation is label-only and never enters prefix features.",
            "Global cycle gaps are preserved; no nearest-cycle substitution.",
            "V4 FIX: TBA_CUv2 is processed as a real periodic capacity check-up (V3 typo TBA_CUv3 removed).",
            "V4 FIX: TBA_BOL_P1v2 is accepted for BOL reference-capacity extraction.",
            "V4 FIX: within the standard-current band, the largest completed discharge step is selected for capacity.",
            "V4 FIX: protocol outlier requires robust-z AND a meaningful absolute physical deviation.",
        ],
    }
    save_json(final, output_root / "99_RWTH48_PIPELINE_COMPLETE.json")


# =============================================================================
# 9. MAIN
# =============================================================================
def main() -> None:
    setup_font()
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = ensure_dir(Path(args.output_root))
    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_root}")

    print("=" * 110)
    print("RWTH48 Rawdata | 质量审计 + 物理清洗 + SOH reference + Event/FixedTime HI")
    print("INPUT :", input_root)
    print("OUTPUT:", output_root)
    print("=" * 110)

    # Stage 1 inventory
    print("[1] Scan leaf ZIP inventory")
    inv = scan_inventory(input_root)
    if inv.empty:
        raise RuntimeError(f"未在 {input_root} 下发现zip文件")
    save_csv(inv.drop(columns=["test_start_dt"], errors="ignore"), output_root / "00_inventory" / "00_leaf_zip_inventory.csv")
    print("  ZIP files discovered:", len(inv), "raw parsed labels:", inv["battery"].nunique())

    # Stage 2 raw file audit. This reads each file once, so for a 5GB dataset it is intentionally comprehensive.
    # 只对文件名元数据完整的leaf ZIP执行审计/正式处理；例如外层“002.zip”容器不应当成leaf重复读取。
    valid_inv = inv[(pd.to_numeric(inv["parse_ok"], errors="coerce").fillna(0).astype(int) == 1) & inv["battery"].ne("UNKNOWN")].copy()
    ignored_nonleaf = len(inv) - len(valid_inv)
    print("  valid leaf ZIP files:", len(valid_inv), "actual batteries:", valid_inv["battery"].nunique(),
          "ignored non-leaf/unknown:", ignored_nonleaf)

    print("[2] Raw leaf-file quality audit")
    audit_rows = []
    for i, r in valid_inv.iterrows():
        if args.only_battery and str(r["battery"]) != (args.only_battery if args.only_battery.upper().startswith("RWTH-") else "RWTH-" + args.only_battery.zfill(3)):
            continue
        rep = audit_leaf_file(r)
        audit_rows.append(rep)
        if (len(audit_rows) % 25) == 0:
            print(f"  audited {len(audit_rows)} files")
    raw_audit = pd.DataFrame(audit_rows)
    save_csv(raw_audit, output_root / "01_quality_audit" / "01_raw_file_audit.csv")

    if not raw_audit.empty:
        ba = raw_audit.groupby("Battery", as_index=False).agg(
            Files=("ZipName", "size"),
            ErrorFiles=("Status", lambda s: int((s == "ERROR").sum())),
            EmptyFiles=("Status", lambda s: int((s == "EMPTY").sum())),
            Rows=("Rows", "sum"),
            ExactDuplicateRows=("ExactDuplicateRows", "sum"),
            HardInvalidRows=("AnyHardInvalidRows", "sum"),
            InternalMissingCycles=("InternalMissingCycleCount", "sum"),
        )
        save_csv(ba, output_root / "01_quality_audit" / "02_raw_battery_audit_summary.csv")

    if args.audit_only:
        print("Audit-only完成。")
        return

    # Stage 3 per battery. We intentionally do not parallelize file audit; for processing, battery-level parallel is optional.
    batteries = sorted(valid_inv["battery"].astype(str).unique())
    if args.only_battery:
        batteries = [b for b in batteries if b == (args.only_battery if args.only_battery.upper().startswith("RWTH-") else "RWTH-" + args.only_battery.zfill(3))]
    print("[3] Per-battery preprocessing + global cycles + labels + prefix HI")
    print("  batteries:", len(batteries), "n_jobs:", args.n_jobs)

    results = []
    if args.n_jobs <= 1:
        for k, bat in enumerate(batteries, 1):
            print(f"  [{k}/{len(batteries)}] Cell {bat}")
            res = process_battery(
                bat, valid_inv[valid_inv["battery"].astype(str).eq(bat)].copy(), output_root,
                save_clean_timeseries=not args.no_save_clean_timeseries,
                include_deltav=not args.skip_deltav_event,
                force=args.force,
            )
            results.append(res)
            print("     ", res.get("Status"), "cycles=", res.get("RecordedGlobalCycles"), "event_rows=", res.get("EventFeatureRows"), "time=", round(res.get("Elapsed_s", 0)/60, 1), "min")
    else:
        # Windows-safe process pool. Each worker handles one battery chronologically.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        max_workers = max(1, int(args.n_jobs))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {}
            for bat in batteries:
                fut = ex.submit(
                    process_battery,
                    bat, valid_inv[valid_inv["battery"].astype(str).eq(bat)].copy(), output_root,
                    not args.no_save_clean_timeseries,
                    not args.skip_deltav_event,
                    args.force,
                )
                futs[fut] = bat
            done_n = 0
            for fut in as_completed(futs):
                bat = futs[fut]
                done_n += 1
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {"Battery": bat, "Status": "ERROR", "Error": repr(exc)}
                results.append(res)
                print(f"  [{done_n}/{len(batteries)}] Cell {bat}: {res.get('Status')}")

    save_csv(pd.DataFrame(results), output_root / "03_global" / "_runtime_battery_results.csv")

    print("[4] Global aggregation + same-condition protocol audit + model-ready tables")
    aggregate_and_save(output_root)

    print("=" * 110)
    print("RWTH48 pipeline COMPLETE")
    print("核心输出：")
    print("  03_global/00_battery_quality_summary.csv")
    print("  03_global/07_same_condition_protocol_audit.csv")
    print("  03_global/04_ALL_FIXEDTIME_MODEL_READY.csv.gz")
    print("  03_global/05_ALL_EVENT_MODEL_READY.csv.gz")
    print("  03_global/08_feature_registry.csv")
    print("  03_global/09_suggested_battery_level_split.csv")
    print("  99_RWTH48_PIPELINE_COMPLETE.json")
    print("=" * 110)


if __name__ == "__main__":
    main()

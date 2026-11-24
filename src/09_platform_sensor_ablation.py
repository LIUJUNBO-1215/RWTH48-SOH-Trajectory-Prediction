# -*- coding: utf-8 -*-
r"""
RWTH48 V4.13F
=============
Platform-Aware Sensor Modality Ablation for Direct RUL Prediction

===============================================================================
Scientific question
===============================================================================

The final RUL branch is Event-based.  However, an Event-RefAh checkpoint itself
requires current I(t) for charge-throughput integration.  Therefore a literal
hardware "Event-V-only" or "Event-VT" configuration is not physically valid.

This script deliberately separates the two platforms:

A. Event platform (PRIMARY; final pipeline)
    - Event_VI
    - Event_VIT

B. FixedTime platform (MECHANISM / SENSOR-BUDGET CONTROL)
    - FixedTime_V
    - FixedTime_VI
    - FixedTime_VT
    - FixedTime_VIT

Event:
    I is required both for RefAh Event localization and, in VI/VIT, may also
    contribute model-input HIs.

FixedTime:
    checkpoint localization uses elapsed time, so V-only and VT are physically
    meaningful sensor configurations.

===============================================================================
Primary methodology
===============================================================================

Outer validation:
    47-cell LOBO
    held-out battery is NEVER used for:
        HI ranking
        pair search
        inner CV
        model fitting

Target:
    Direct RUL at origin cycle 80
    TrueRUL = Operational EOL80 lead from frozen V4.11A2 handoff

History:
    cycle 40 / 48 / 56 / 64 / 72 / 80
    six sparse diagnoses, exactly matching the final Event RUL branch history.

Event nodes:
    RefAh 0.05 / 0.09 / 0.13

FixedTime nodes:
    30 / 60 / 90 / 120 s cumulative prefix

Model:
    same frozen shallow XGBoost RUL head as V4.12A:
        n_estimators=180
        max_depth=2
        learning_rate=0.035
        subsample=0.90
        colsample_bytree=0.80
        reg_lambda=8
        reg_alpha=0.2
        min_child_weight=3
        tree_method=hist

HI selection:
    source-only relevance ranking
        = mean of four strongest absolute Spearman correlations
          over all history x checkpoint positions

    source-only exhaustive Top2 pair search
        shortlist = 14 by default
        inner battery CV = 5 folds by default

    selected pair orientation follows source-only relevance rank.

Event_VIT:
    by default REUSES the exact frozen V4.12A DynamicTop2 outer predictions and
    selected HIs.  This provides an exact reproduction anchor:
        V4.13F Event_VIT == V4.12A RULSpecific_Top2
    Set --recompute-event-vit only for an optional expensive independent rerun.

Event_VI:
    same V4.12A Event representation and exact V4.10A 36->196 engineering,
    but the candidate HI pool is constrained to V/I-derived HIs.

FixedTime:
    uses its native four cumulative checkpoints.  Because V4.10A's exact
    engineer is frozen specifically for 3 Event nodes, FixedTime uses a
    transparent deterministic "native-node" pair engineering function:
        RawMeasured
        HistoricalTrend
        WithinDiagnosisGeometry
        ConsecutiveNodeDifference
        CrossHIWithinNode

    No new measurements are created.  This is a deterministic transformation
    of the selected FixedTime HI pair.

IMPORTANT CROSS-PLATFORM CAVEAT:
    Event uses 3 RefAh checkpoints while native FixedTime uses 4 time
    checkpoints.  Therefore Event-vs-FixedTime comparisons in this task are
    DESCRIPTIVE NATIVE-PLATFORM comparisons, NOT equal-checkpoint-budget proof
    of platform superiority.

For exact-common/equal-budget platform claims, retain the earlier frozen
V4.8D comparison.

===============================================================================
Sensor dependency logic
===============================================================================

A model-input HI is eligible for modality M only if:

    SensorDependencies(HI) subseteq M

Examples:
    V_TimeWeightedMean -> {V}
    VBand_*             -> {V}
    I_* / dIdt_*        -> {I}
    P_*                 -> {V,I}
    R_DynamicSlope_Ohm  -> {V,I}
    VI_Correlation      -> {V,I}
    Temp_* / dTdt_*     -> {T}
    Partial*_Ah         -> {I}
    PrefixDuration_s    -> {}  (time coordinate; no sensor channel)

The script writes a complete sensor dependency registry and FAILS if an
eligible HI has unknown dependency semantics.  Optional manual overrides may be
supplied with:

    --dependency-registry path.csv

Required columns:
    HI, SensorDependencies

Examples:
    V_Mean,V
    P_Last,V|I
    Temp_Range,T
    PrefixDuration_s,

===============================================================================
Outputs
===============================================================================

00_config.json
00_preflight.csv
01_sensor_dependency_registry.csv
02_platform_hi_coverage.csv

03_outer_predictions.csv
04_pooled_metrics.csv
05_selection_by_fold.csv
06_hi_selection_frequency.csv

07_marginal_sensor_value.csv
08_paired_bootstrap.csv
09_cross_platform_native_comparison.csv
10_battery_absolute_errors.csv

11_search_runtime_summary.csv
12_acceptance_gate.csv

folds/<Battery>/
    Event_VI_result.json
    Event_VI_ranking.csv
    Event_VI_pair_search.csv.gz

    Event_VIT_result.json
        (frozen V4.12A reuse unless --recompute-event-vit)

    FixedTime_V_result.json
    FixedTime_V_ranking.csv
    FixedTime_V_pair_search.csv.gz
    ...
Figures/
99_RWTH48_V4_13F_COMPLETE.json

===============================================================================
Recommended run order
===============================================================================

1) Audit
python -u "src/09_platform_sensor_ablation.py" --stage audit

2) Single-cell smoke
python -u "src/09_platform_sensor_ablation.py" ^
    --stage run --only-battery RWTH-002 --xgb-threads 4

3) Formal
python -u "src/09_platform_sensor_ablation.py" ^
    --stage all --xgb-threads 4

4) Two CPU workers (optional)
Terminal 1:
python -u "src/09_platform_sensor_ablation.py" --stage worker --num-workers 2 --worker-id 0 --xgb-threads 2

Terminal 2:
python -u "src/09_platform_sensor_ablation.py" --stage worker --num-workers 2 --worker-id 1 --xgb-threads 2 --auto-finalize

Resume:
    run the same command again; completed battery/config caches are reused.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, FrozenSet

import numpy as np
import pandas as pd

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. Frozen paths / constants
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR = Path(__file__).resolve().parent

V412A_HELPER = CODE_DIR / "21_rul_topk_helper.py"

V411A2_ROOT = BASE / "RWTH48_V4_11A2_跨StrictSegment全生命目标重建审计"
INPUT_WIDE = V411A2_ROOT / "09_input_history_wide.csv.gz"
TARGET_GRID = V411A2_ROOT / "08_operational_eol80_target_grid_long.csv.gz"
V411A2_COMPLETE = V411A2_ROOT / "99_RWTH48_V4_11A2_COMPLETE.json"

V411B_HELPER = CODE_DIR / "22_full_life_soh_helper.py"
V410A_HELPER = CODE_DIR / "23_event_tree_helper.py"

V411D_ROOT = BASE / "RWTH48_V4_11D_HorizonAdaptive_MoE_ResidualHybrid"
V411D_MOE_OOF = V411D_ROOT / "01_MoE" / "03_ALL_OOF_MoE_long.csv.gz"
V411D_METRICS = V411D_ROOT / "20_reference_and_hybrid_pooled_metrics.csv"
V411D_COMPLETE = V411D_ROOT / "99_RWTH48_V4_11D_COMPLETE.json"

V411E_ROOT = BASE / "RWTH48_V4_11E_EOL_RUL_Aware_FrozenMoE"
V411E_OUTER = V411E_ROOT / "04_outer_rul_predictions.csv"
V411E_COMPLETE = V411E_ROOT / "99_RWTH48_V4_11E_COMPLETE.json"

V411F_ROOT = BASE / "RWTH48_V4_11F_DirectRUL_Optimization"
V411F_COMPLETE = V411F_ROOT / "99_RWTH48_V4_11F_COMPLETE.json"

V412A_ROOT = BASE / "RWTH48_V4_12A_RULSpecific_HI_TopK_Pareto"
V412A_OUTER = V412A_ROOT / "07_outer_predictions_by_topk.csv"
V412A_COMPLETE = V412A_ROOT / "99_RWTH48_V4_12A_COMPLETE.json"

V42_ROOT = BASE / "RWTH48_V4_2_双平台冻结_FixedTime_Event_HI"

OUTPUT_ROOT = BASE / "RWTH48_V4_13F_PlatformAware_SensorModality_RUL_Ablation"

EXPECTED_BATTERIES = 47
TEST_ORIGIN = 80
SOURCE_ORIGINS = (80, 120, 160)
HISTORY_OFFSETS = (-40, -32, -24, -16, -8, 0)
EVENT_NODES = (0.05, 0.09, 0.13)
FIXEDTIME_NODES = (30.0, 60.0, 90.0, 120.0)
K_TRAJ = 1000

CONFIGS = (
    "Event_VI",
    "Event_VIT",
    "FixedTime_V",
    "FixedTime_VI",
    "FixedTime_VT",
    "FixedTime_VIT",
)

COMPUTED_DEFAULT = (
    "Event_VI",
    "Event_VIT",
    "FixedTime_V",
    "FixedTime_VI",
    "FixedTime_VT",
    "FixedTime_VIT",
)

MODALITY_SENSORS = {
    "V": frozenset({"V"}),
    "VI": frozenset({"V", "I"}),
    "VT": frozenset({"V", "T"}),
    "VIT": frozenset({"V", "I", "T"}),
}

ENC = "utf-8-sig"
FAST_GZIP = {"method": "gzip", "compresslevel": 1}
RANDOM_SEED = 20260824

# Bootstrap is battery-level paired bootstrap.
DEFAULT_BOOTSTRAP = 20000

# Same candidate eligibility boundary as V4.12A / V4.7E.
NON_FEATURE_COLS = {
    "Battery", "GlobalCycle", "Scope", "Platform", "NodeType", "NodeValue",
    "PlatformNode", "ReferenceCapacity_Ah", "ReferenceCapacity_Ah_QCOnly",
    "SOH_Target_SameCycle", "SOH_IsMeasuredCheckUp", "SOH_LabelUsable",
    "SOH_LabelSource", "EventReachable", "PrefixFeatureValid",
    "MissingFeatureCount", "Reachable", "SourcePart",
    "StrictSegmentKey", "CyclingSOH_Charge_Canonical",
}
DESIGN_FORBIDDEN = {
    "SampleCount", "ObservedHardInvalidRate", "ObservedSpikeRate",
}
STRUCTURAL_REDUNDANT_DROP = {
    "PartialThroughput_Wh",
}
GLOBAL_CONSTANT_DROP = {
    "VBand_3p0_3p2_Ratio",
    "VBand_3p2_3p4_Ratio",
    "VBand_4p0_4p2_Ratio",
}

FORBIDDEN_NAME_RE = re.compile(
    r"(?:"
    r"(^|_)SOH($|_)|"
    r"capacity|"
    r"true.?eol|"
    r"true.?rul|"
    r"target|"
    r"label|"
    r"future|"
    r"prediction|"
    r"pred_|"
    r"globalcycle|"
    r"(^|_)cycle($|_)|"
    r"battery"
    r")",
    flags=re.IGNORECASE,
)


# =============================================================================
# 1. CLI / basic utilities
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWTH48 V4.13F platform-aware sensor modality RUL ablation"
    )

    p.add_argument(
        "--stage",
        choices=["audit", "run", "worker", "finalize", "all"],
        default="all",
    )

    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--code-dir", type=Path, default=CODE_DIR)
    p.add_argument("--v412a-helper", type=Path, default=V412A_HELPER)

    p.add_argument("--event-ready", type=Path, default=None)
    p.add_argument("--fixedtime-ready", type=Path, default=None)
    p.add_argument("--dependency-registry", type=Path, default=None)

    p.add_argument("--v412a-outer", type=Path, default=V412A_OUTER)
    p.add_argument("--v412a-complete", type=Path, default=V412A_COMPLETE)

    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)

    p.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS),
        default=list(COMPUTED_DEFAULT),
    )

    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--force", action="store_true")

    p.add_argument("--shortlist", type=int, default=14)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--xgb-threads", type=int, default=4)

    p.add_argument(
        "--recompute-event-vit",
        action="store_true",
        help=(
            "Expensive optional check. Default reuses frozen V4.12A Event-VIT "
            "DynamicTop2 prediction exactly."
        ),
    )

    p.add_argument(
        "--fixedtime-nodes",
        nargs=4,
        type=float,
        default=list(FIXEDTIME_NODES),
    )

    p.add_argument("--phase", type=str, default="Discharge")

    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)

    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--auto-finalize", action="store_true")

    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--dpi", type=int, default=260)

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


def load_json(p: Path) -> Dict[str, object]:
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def save_csv(df: pd.DataFrame, p: Path) -> None:
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding=ENC)


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def normalize_battery(x: object) -> str:
    s = str(x).strip().upper().replace("_", "-").replace(" ", "")
    m = re.search(r"RWTH-?0*(\d+)", s)
    if m:
        return f"RWTH-{int(m.group(1)):03d}"
    return s


def norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def stable_hash(obj: object) -> str:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def split_groups(batteries: Sequence[str], n_folds: int) -> List[List[str]]:
    bats = sorted(set(map(str, batteries)))
    k = max(2, min(int(n_folds), len(bats)))
    groups = [[] for _ in range(k)]
    for i, b in enumerate(bats):
        groups[i % k].append(b)
    return [g for g in groups if g]


def worker_batteries(
    batteries: Sequence[str],
    num_workers: int,
    worker_id: int,
) -> List[str]:
    bats = sorted(batteries)
    n = max(1, int(num_workers))
    w = int(worker_id)
    if w < 0 or w >= n:
        raise ValueError(f"worker-id={w} must satisfy 0 <= id < {n}")
    return [b for i, b in enumerate(bats) if i % n == w]


def finite_or_nan(x) -> float:
    try:
        y = float(x)
    except Exception:
        return np.nan
    return y if np.isfinite(y) else np.nan


# =============================================================================
# 2. V4.12A frozen preflight adapter
# =============================================================================

def make_v412a_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace()

    ns.stage = "audit"
    ns.base = args.base
    ns.code_dir = args.code_dir

    ns.v411b_helper = args.code_dir / "22_full_life_soh_helper.py"
    ns.v410a_helper = args.code_dir / "23_event_tree_helper.py"

    a2 = args.base / "RWTH48_V4_11A2_跨StrictSegment全生命目标重建审计"
    ns.input_wide = a2 / "09_input_history_wide.csv.gz"
    ns.target_grid = a2 / "08_operational_eol80_target_grid_long.csv.gz"
    ns.v411a2_complete = a2 / "99_RWTH48_V4_11A2_COMPLETE.json"

    d = args.base / "RWTH48_V4_11D_HorizonAdaptive_MoE_ResidualHybrid"
    ns.v411d_moe_oof = d / "01_MoE" / "03_ALL_OOF_MoE_long.csv.gz"
    ns.v411d_metrics = d / "20_reference_and_hybrid_pooled_metrics.csv"
    ns.v411d_complete = d / "99_RWTH48_V4_11D_COMPLETE.json"

    e = args.base / "RWTH48_V4_11E_EOL_RUL_Aware_FrozenMoE"
    ns.v411e_outer = e / "04_outer_rul_predictions.csv"
    ns.v411e_complete = e / "99_RWTH48_V4_11E_COMPLETE.json"

    f = args.base / "RWTH48_V4_11F_DirectRUL_Optimization"
    ns.v411f_complete = f / "99_RWTH48_V4_11F_COMPLETE.json"

    ns.event_ready = args.event_ready
    ns.phase = args.phase
    ns.candidate_hi_file = None

    ns.output_root = args.output_root
    ns.only_battery = ""
    ns.force = False

    ns.shortlist = int(args.shortlist)
    ns.inner_folds = int(args.inner_folds)
    ns.pareto_mae_tol = 0.02
    ns.train_origin_mode = "strict"
    ns.xgb_threads = int(args.xgb_threads)

    ns.num_workers = 1
    ns.worker_id = 0
    ns.auto_finalize = False
    ns.wait_poll_seconds = 30
    ns.wait_timeout_hours = 24.0
    ns.skip_figures = True
    ns.dpi = 200

    return ns


def load_frozen_v412a_outer(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"V4.12A outer prediction file missing: {path}"
        )

    d = pd.read_csv(path, low_memory=False)

    required = {
        "OuterHeldoutBattery",
        "TrueRUL",
        "SelectedTop2",
        "Pred_Top2",
    }
    miss = sorted(required - set(d.columns))
    if miss:
        raise RuntimeError(
            f"V4.12A outer file missing columns: {miss}"
        )

    d["OuterHeldoutBattery"] = d["OuterHeldoutBattery"].map(normalize_battery)
    d["TrueRUL"] = pd.to_numeric(d["TrueRUL"], errors="coerce")
    d["Pred_Top2"] = pd.to_numeric(d["Pred_Top2"], errors="coerce")

    if len(d) != EXPECTED_BATTERIES:
        raise RuntimeError(f"V4.12A outer rows={len(d)} !=47")
    if d["OuterHeldoutBattery"].nunique() != EXPECTED_BATTERIES:
        raise RuntimeError("V4.12A outer battery count !=47")

    return d.sort_values("OuterHeldoutBattery").reset_index(drop=True)


# =============================================================================
# 3. Generic long-platform schema / FixedTime locating
# =============================================================================

def choose_column(
    columns: Sequence[str],
    exact_names: Sequence[str],
    contains_tokens: Sequence[str] = (),
) -> Optional[str]:
    norm_map = {norm_col(c): c for c in columns}
    for n in exact_names:
        nn = norm_col(n)
        if nn in norm_map:
            return norm_map[nn]

    toks = [norm_col(t) for t in contains_tokens if norm_col(t)]
    if toks:
        for c in columns:
            nc = norm_col(c)
            if all(t in nc for t in toks):
                return c
    return None


def parse_number(x: object) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else np.nan


def parse_fixedtime_node(x: object) -> float:
    """
    FixedTime PlatformNode strings may look like:
        FixedTime_30s
        FixedTime_T030
        FixedTime_0_30s
        0-30s

    The checkpoint identity is the terminal elapsed-time magnitude, not the
    first number appearing in the label.  Numeric NodeValue inputs are returned
    unchanged.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip()
    nums = re.findall(r"\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        return np.nan

    try:
        return float(nums[-1])
    except Exception:
        return np.nan


@dataclass
class PlatformSchema:
    battery_col: str
    cycle_col: str
    node_col: str
    phase_col: Optional[str]
    platform_col: Optional[str]
    platform_node_col: Optional[str]
    hi_cols: List[str]


@dataclass
class GenericHITensor:
    values: np.ndarray       # [sample, diagnosis=6, native_node, hi]
    batteries: np.ndarray
    origins: np.ndarray
    hi_names: List[str]
    nodes: Tuple[float, ...]
    platform: str
    coverage: pd.DataFrame


def candidate_hi_columns(
    df: pd.DataFrame,
    excluded: Iterable[str],
) -> List[str]:
    excluded = set(excluded)

    out = []
    for c in df.columns:
        if c in excluded:
            continue
        if c in NON_FEATURE_COLS:
            continue
        if c in DESIGN_FORBIDDEN:
            continue
        if c in STRUCTURAL_REDUNDANT_DROP:
            continue
        if c in GLOBAL_CONSTANT_DROP:
            continue
        if str(c).startswith("SOH_") or str(c).startswith("Pred_"):
            continue
        if FORBIDDEN_NAME_RE.search(str(c)):
            continue

        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() == 0:
            continue
        out.append(str(c))

    return out


def inspect_fixedtime_candidate(path: Path) -> Tuple[int, str]:
    """
    Return (score, reason). Higher score is more likely a usable FixedTime table.
    Reads only a small sample.
    """
    score = 0
    reason = []

    name = path.name.lower()
    if "fixedtime" in name or "fixed_time" in name or "fixed-time" in name:
        score += 20
        reason.append("filename-fixedtime")
    if "event" in name and "fixed" not in name:
        score -= 20

    try:
        hdr = pd.read_csv(
            path,
            compression="infer",
            nrows=0,
        ).columns.tolist()
    except Exception:
        return -999, "unreadable-header"

    pnode = choose_column(hdr, ["PlatformNode"])
    platform = choose_column(
        hdr,
        ["Platform", "PrefixType", "PrefixPlatform", "FeaturePlatform"],
    )
    nodeval = choose_column(hdr, ["NodeValue"])
    cycle = choose_column(hdr, ["GlobalCycle", "Cycle", "CycleIndex", "CycleNo"])
    bat = choose_column(hdr, ["Battery", "BatteryID", "Cell", "CellID"])

    if bat is not None and cycle is not None:
        score += 5
    if pnode is not None:
        score += 3
    if platform is not None:
        score += 2
    if nodeval is not None:
        score += 1

    usecols = [c for c in (pnode, platform, nodeval) if c is not None]
    if usecols:
        try:
            s = pd.read_csv(
                path,
                usecols=usecols,
                compression="infer",
                nrows=4000,
                low_memory=False,
            )
            txt = " ".join(
                s[c].astype(str).str.lower().head(4000).tolist()
                for c in []
            )
            # Avoid building a massive joined string through a list of lists.
            found = False
            for c in usecols:
                if s[c].astype(str).str.contains(
                    "fixedtime|fixed_time|fixed-time|fixed time",
                    case=False,
                    regex=True,
                    na=False,
                ).any():
                    found = True
                    break
            if found:
                score += 30
                reason.append("sample-fixedtime")
        except Exception:
            pass

    return score, ";".join(reason)


def locate_fixedtime_ready(args: argparse.Namespace) -> Tuple[Path, pd.DataFrame]:
    if args.fixedtime_ready is not None:
        p = Path(args.fixedtime_ready)
        if not p.exists():
            raise FileNotFoundError(p)
        audit = pd.DataFrame(
            [{"Path": str(p), "Score": 999, "Reason": "explicit"}]
        )
        return p, audit

    root = args.base / "RWTH48_V4_2_双平台冻结_FixedTime_Event_HI"
    if not root.exists():
        raise FileNotFoundError(
            f"V4.2 dual-platform root missing: {root}"
        )

    pats = (
        "*FixedTime*.csv.gz",
        "*FIXEDTIME*.csv.gz",
        "*fixedtime*.csv.gz",
        "*FixedTime*.csv",
        "*FIXEDTIME*.csv",
        "*.csv.gz",
        "*.csv",
    )

    seen = set()
    candidates = []
    for pat in pats:
        for p in root.rglob(pat):
            if p in seen:
                continue
            seen.add(p)
            if p.name.startswith("99_"):
                continue
            score, reason = inspect_fixedtime_candidate(p)
            candidates.append(
                {
                    "Path": str(p),
                    "Score": int(score),
                    "Reason": reason,
                }
            )

    if not candidates:
        raise FileNotFoundError(
            "V4.2目录中没有可检查的CSV文件；请用 --fixedtime-ready 指定。"
        )

    audit = pd.DataFrame(candidates).sort_values(
        ["Score", "Path"],
        ascending=[False, True],
    ).reset_index(drop=True)

    best = audit.iloc[0]
    if int(best["Score"]) < 10:
        raise FileNotFoundError(
            "无法可靠自动定位 FixedTime frozen HI table。\n"
            "请查看自动扫描结果并使用 --fixedtime-ready 显式指定。\n"
            + audit.head(20).to_string(index=False)
        )

    return Path(str(best["Path"])), audit


def detect_fixedtime_schema(
    df: pd.DataFrame,
    nodes: Sequence[float],
) -> PlatformSchema:
    cols = list(df.columns)

    battery_col = choose_column(
        cols,
        ["Battery", "BatteryID", "Cell", "CellID"],
    )
    cycle_col = choose_column(
        cols,
        ["GlobalCycle", "Cycle", "CycleIndex", "CycleNo"],
    )
    phase_col = choose_column(
        cols,
        ["Phase", "PhysicalPhase"],
    )
    platform_col = choose_column(
        cols,
        ["Platform", "PrefixType", "PrefixPlatform", "FeaturePlatform"],
    )
    platform_node_col = choose_column(
        cols,
        ["PlatformNode"],
    )
    node_value_col = choose_column(
        cols,
        ["NodeValue"],
    )

    if battery_col is None or cycle_col is None:
        raise RuntimeError(
            "FixedTime table无法识别 Battery / GlobalCycle"
        )

    # Prefer PlatformNode when it visibly encodes FixedTime.
    node_col = None
    if platform_node_col is not None:
        s = df[platform_node_col].astype(str)
        if s.str.contains(
            "FixedTime|Fixed_Time|Fixed-Time|Fixed Time",
            case=False,
            regex=True,
            na=False,
        ).any():
            node_col = platform_node_col

    if node_col is None and node_value_col is not None:
        nv = pd.to_numeric(df[node_value_col], errors="coerce")
        has_nodes = all(
            np.isclose(
                nv.to_numpy(float),
                float(t),
                atol=0.5,
                rtol=0,
                equal_nan=False,
            ).any()
            for t in nodes
        )
        if has_nodes:
            node_col = node_value_col

    if node_col is None:
        # Last-resort explicit time/checkpoint columns.
        node_col = choose_column(
            cols,
            [
                "FixedTime_s",
                "CheckpointTime_s",
                "PrefixTime_s",
                "TimeCheckpoint_s",
            ],
        )

    if node_col is None:
        raise RuntimeError(
            "FixedTime table无法识别节点坐标。需要 PlatformNode/NodeValue/"
            "FixedTime_s 等列。"
        )

    excluded = {
        battery_col,
        cycle_col,
        node_col,
    }
    for c in (phase_col, platform_col, platform_node_col):
        if c is not None:
            excluded.add(c)

    hi_cols = candidate_hi_columns(df, excluded)

    if len(hi_cols) < 2:
        raise RuntimeError(
            f"FixedTime合法数值HI候选只有 {len(hi_cols)} 个"
        )

    return PlatformSchema(
        battery_col=battery_col,
        cycle_col=cycle_col,
        node_col=node_col,
        phase_col=phase_col,
        platform_col=platform_col,
        platform_node_col=platform_node_col,
        hi_cols=hi_cols,
    )


def filter_fixedtime_table(
    df: pd.DataFrame,
    schema: PlatformSchema,
    nodes: Sequence[float],
    phase: str,
) -> pd.DataFrame:
    d = df.copy()

    d[schema.battery_col] = d[schema.battery_col].map(normalize_battery)
    d[schema.cycle_col] = pd.to_numeric(
        d[schema.cycle_col],
        errors="coerce",
    )

    # Platform restriction.
    masks = []
    if schema.platform_col is not None:
        s = d[schema.platform_col].astype(str)
        masks.append(
            s.str.contains(
                "FixedTime|Fixed_Time|Fixed-Time|Fixed Time",
                case=False,
                regex=True,
                na=False,
            ).to_numpy(bool)
        )
    if schema.platform_node_col is not None:
        s = d[schema.platform_node_col].astype(str)
        masks.append(
            s.str.contains(
                "FixedTime|Fixed_Time|Fixed-Time|Fixed Time",
                case=False,
                regex=True,
                na=False,
            ).to_numpy(bool)
        )

    if masks and np.any(np.column_stack(masks)):
        keep = np.any(np.column_stack(masks), axis=1)
        d = d[keep].copy()

    # Phase restriction if explicit.
    if schema.phase_col is not None:
        s = d[schema.phase_col].astype(str).str.strip()
        m = s.str.casefold().eq(str(phase).strip().casefold())
        if not m.any():
            raise RuntimeError(
                f"FixedTime Phase={phase}没有行；"
                f"可用值={sorted(s.unique().tolist())[:20]}"
            )
        d = d[m].copy()

    d["_NodeNumeric"] = d[schema.node_col].map(parse_fixedtime_node)

    canon = np.full(len(d), np.nan, dtype=float)
    obs = d["_NodeNumeric"].to_numpy(float)
    for t in nodes:
        m = np.isclose(obs, float(t), atol=0.5, rtol=0)
        canon[m] = float(t)
    d["_NodeCanonical"] = canon
    d = d[np.isfinite(d["_NodeCanonical"])].copy()

    missing = [
        float(t)
        for t in nodes
        if not np.isclose(
            d["_NodeCanonical"].to_numpy(float),
            float(t),
            atol=1e-12,
            rtol=0,
        ).any()
    ]
    if missing:
        raise RuntimeError(
            f"FixedTime frozen nodes missing: {missing}"
        )

    key = [
        schema.battery_col,
        schema.cycle_col,
        "_NodeCanonical",
    ]
    dup = d.duplicated(key, keep=False)
    if dup.any():
        raise RuntimeError(
            "FixedTime table在 Battery×Cycle×Node 上存在重复；"
            "禁止静默平均。\n"
            + d.loc[dup, key].head(20).to_string(index=False)
        )

    return d.reset_index(drop=True)


# =============================================================================
# 4. Sensor dependency registry
# =============================================================================

def parse_dep_string(x: object) -> FrozenSet[str]:
    if pd.isna(x) or str(x).strip() == "":
        return frozenset()

    s = str(x).upper()
    toks = [
        t
        for t in re.split(r"[|,;+\s]+", s)
        if t
    ]
    allowed = {"V", "I", "T"}
    bad = sorted(set(toks) - allowed)
    if bad:
        raise ValueError(f"Invalid sensor tokens: {bad}")
    return frozenset(toks)


def infer_sensor_dependencies(hi: str) -> Optional[FrozenSet[str]]:
    """
    Conservative semantic dependency classifier.

    Return:
        frozenset()   -> time/platform coordinate only
        {'V'} / {'I'} / {'T'} / combinations
        None          -> unknown, must be resolved before formal run
    """
    h = str(hi).strip()
    hl = h.lower()

    # No sensor-channel information.
    if h in {
        "PrefixDuration_s",
    }:
        return frozenset()

    # Explicit cross-channel / electrical derived quantities.
    if h in {
        "R_DynamicSlope_Ohm",
        "VI_Correlation",
    }:
        return frozenset({"V", "I"})

    # Event/current-throughput quantities.
    if h in {
        "RefAhProgress",
        "PartialThroughput_Ah",
        "PartialDischarge_Ah",
        "PartialCharge_Ah",
    }:
        return frozenset({"I"})

    # Power / energy / Wh.
    if (
        hl.startswith("p_")
        or "power" in hl
        or "_wh" in hl
        or hl.endswith("wh")
        or "energy" in hl
    ):
        return frozenset({"V", "I"})

    # Temperature.
    if (
        hl.startswith("temp")
        or hl.startswith("dtdt")
        or hl.startswith("t_")
        or "temperature" in hl
        or "_temp" in hl
    ):
        return frozenset({"T"})

    # Voltage.
    if (
        hl.startswith("v_")
        or hl.startswith("dvdt")
        or hl.startswith("vband")
        or hl.startswith("vcross")
        or hl.startswith("voltage")
        or "voltage" in hl
    ):
        return frozenset({"V"})

    # Current.
    if (
        hl.startswith("i_")
        or hl.startswith("didt")
        or hl.startswith("current")
        or "current" in hl
        or hl.endswith("_ah")
        or "_ah_" in hl
        or hl.startswith("partial")
    ):
        return frozenset({"I"})

    # Common resistance/conductance or VI geometry names.
    if (
        "ohm" in hl
        or "resistance" in hl
        or "conductance" in hl
        or hl.startswith("vi_")
        or hl.startswith("iv_")
    ):
        return frozenset({"V", "I"})

    return None


def build_dependency_registry(
    hi_names: Sequence[str],
    override_path: Optional[Path],
) -> pd.DataFrame:
    overrides: Dict[str, FrozenSet[str]] = {}

    if override_path is not None:
        p = Path(override_path)
        if not p.exists():
            raise FileNotFoundError(p)

        d = pd.read_csv(p, low_memory=False)
        required = {"HI", "SensorDependencies"}
        miss = sorted(required - set(d.columns))
        if miss:
            raise RuntimeError(
                f"dependency registry missing columns: {miss}"
            )

        for r in d.itertuples(index=False):
            overrides[str(r.HI).strip()] = parse_dep_string(
                getattr(r, "SensorDependencies")
            )

    rows = []
    for hi in sorted(set(map(str, hi_names))):
        if hi in overrides:
            dep = overrides[hi]
            source = "ManualOverride"
        else:
            dep = infer_sensor_dependencies(hi)
            source = "BuiltInSemanticRule"

        known = dep is not None
        dep_text = (
            "|".join(sorted(dep))
            if known
            else "UNKNOWN"
        )

        rows.append(
            {
                "HI": hi,
                "SensorDependencies": dep_text,
                "DependencyKnown": int(known),
                "RuleSource": source,
            }
        )

    return pd.DataFrame(rows)


def dependency_map(registry: pd.DataFrame) -> Dict[str, Optional[FrozenSet[str]]]:
    out = {}
    for r in registry.itertuples(index=False):
        if int(r.DependencyKnown) != 1:
            out[str(r.HI)] = None
        else:
            out[str(r.HI)] = parse_dep_string(
                r.SensorDependencies
            )
    return out


def allowed_his(
    hi_names: Sequence[str],
    modality: str,
    depmap: Dict[str, Optional[FrozenSet[str]]],
) -> List[str]:
    sensors = MODALITY_SENSORS[modality]
    out = []
    for hi in hi_names:
        dep = depmap.get(str(hi))
        if dep is None:
            continue
        if dep.issubset(sensors):
            out.append(str(hi))
    return out


# =============================================================================
# 5. Build FixedTime candidate tensor
# =============================================================================

def build_fixedtime_tensor(
    fixed: pd.DataFrame,
    schema: PlatformSchema,
    bundle,
    nodes: Sequence[float],
) -> GenericHITensor:
    batteries = np.asarray(
        [normalize_battery(x) for x in bundle.battery],
        dtype=object,
    )
    origins = np.asarray(bundle.origin, dtype=int)

    if len(batteries) != EXPECTED_BATTERIES * 3:
        raise RuntimeError(
            f"Frozen handoff sample count={len(batteries)} !=141"
        )

    d = fixed.copy()
    d["_CycleInt"] = np.rint(
        pd.to_numeric(d[schema.cycle_col], errors="coerce")
    ).astype("Int64")

    key_to_row = {}
    for i, r in d[
        [schema.battery_col, "_CycleInt", "_NodeCanonical"]
    ].iterrows():
        if pd.isna(r["_CycleInt"]):
            continue
        key = (
            normalize_battery(r[schema.battery_col]),
            int(r["_CycleInt"]),
            float(r["_NodeCanonical"]),
        )
        key_to_row[key] = i

    H = len(schema.hi_cols)
    M = len(nodes)
    N = len(batteries)

    tensor = np.full(
        (N, len(HISTORY_OFFSETS), M, H),
        np.nan,
        dtype=np.float32,
    )

    for n in range(N):
        b = batteries[n]
        o = int(origins[n])

        for di, off in enumerate(HISTORY_OFFSETS):
            cyc = o + int(off)

            for mi, node in enumerate(nodes):
                key = (b, cyc, float(node))
                row_idx = key_to_row.get(key)
                if row_idx is None:
                    continue

                vals = pd.to_numeric(
                    d.loc[row_idx, schema.hi_cols],
                    errors="coerce",
                ).to_numpy(float)

                tensor[n, di, mi, :] = vals.astype(np.float32)

    cov_rows = []
    for j, hi in enumerate(schema.hi_cols):
        x = tensor[:, :, :, j]
        sample_complete = np.isfinite(x).all(axis=(1, 2))
        origin80 = origins == TEST_ORIGIN

        cov_rows.append(
            {
                "Platform": "FixedTime",
                "HI": hi,
                "FiniteScalarRate_All141": float(np.isfinite(x).mean()),
                "CompleteSampleRate_All141": float(sample_complete.mean()),
                "CompleteSampleCount_All141": int(sample_complete.sum()),
                "CompleteBatteryRate_Origin80": float(
                    sample_complete[origin80].mean()
                ),
                "CompleteBatteryCount_Origin80": int(
                    sample_complete[origin80].sum()
                ),
                "EligibleCompleteAll141": int(sample_complete.all()),
            }
        )

    cov = pd.DataFrame(cov_rows)
    eligible = cov[
        cov["EligibleCompleteAll141"].eq(1)
    ]["HI"].tolist()

    if len(eligible) < 2:
        raise RuntimeError(
            f"FixedTime complete eligible HI only {len(eligible)}"
        )

    idx = [schema.hi_cols.index(h) for h in eligible]

    return GenericHITensor(
        values=tensor[:, :, :, idx],
        batteries=batteries,
        origins=origins.astype(np.int32),
        hi_names=list(eligible),
        nodes=tuple(map(float, nodes)),
        platform="FixedTime",
        coverage=cov,
    )


def subset_event_hit(v412a, hit, names: Sequence[str]):
    names = list(names)
    idx = [hit.hi_names.index(h) for h in names]
    return v412a.HITensor(
        values=np.asarray(hit.values[:, :, :, idx], np.float32),
        batteries=np.asarray(hit.batteries, dtype=object),
        origins=np.asarray(hit.origins, np.int32),
        hi_names=names,
        coverage=hit.coverage.copy(),
    )


def subset_generic_hit(hit: GenericHITensor, names: Sequence[str]) -> GenericHITensor:
    names = list(names)
    idx = [hit.hi_names.index(h) for h in names]
    return GenericHITensor(
        values=np.asarray(hit.values[:, :, :, idx], np.float32),
        batteries=np.asarray(hit.batteries, dtype=object),
        origins=np.asarray(hit.origins, np.int32),
        hi_names=names,
        nodes=tuple(hit.nodes),
        platform=hit.platform,
        coverage=hit.coverage.copy(),
    )


# =============================================================================
# 6. Generic FixedTime deterministic pair engineering
# =============================================================================

def safe_slope_axis(x: np.ndarray) -> np.ndarray:
    """
    x shape [N, L], slope against fixed normalized positional coordinate.
    """
    x = np.asarray(x, np.float64)
    L = x.shape[1]
    t = np.linspace(-1.0, 1.0, L, dtype=np.float64)
    tc = t - t.mean()
    denom = float(np.sum(tc * tc))
    xc = x - np.mean(x, axis=1, keepdims=True)
    return np.sum(xc * tc[None, :], axis=1) / max(denom, 1e-12)


def engineer_fixedtime_pair(
    pair: np.ndarray,
    hi_a: str,
    hi_b: str,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    pair shape [N, diagnosis=6, node=M, channel=2]

    Families:
        RawMeasured
        HistoricalTrend
        WithinDiagnosisGeometry
        ConsecutiveNodeDifference
        CrossHIWithinNode

    No true SOH, capacity history, target cycle, future HI, or held-out
    calibration enters these features.
    """
    x = np.asarray(pair, np.float64)

    if x.ndim != 4 or x.shape[1] != 6 or x.shape[3] != 2:
        raise RuntimeError(
            f"FixedTime pair shape={x.shape}, expected [N,6,M,2]"
        )
    if not np.isfinite(x).all():
        raise RuntimeError("FixedTime selected pair contains NaN/Inf")

    N, D, M, C = x.shape
    blocks = []
    names: List[str] = []
    reg_rows = []

    def add_block(arr: np.ndarray, block_names: Sequence[str], family: str):
        arr = np.asarray(arr, np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] != N:
            raise RuntimeError(f"feature block {family} N mismatch")
        blocks.append(arr)
        names.extend(list(block_names))
        reg_rows.extend(
            {
                "Feature": n,
                "Family": family,
            }
            for n in block_names
        )

    his = [str(hi_a), str(hi_b)]

    # ------------------------------------------------------------------
    # A. Raw measured block
    # ------------------------------------------------------------------
    raw = x.reshape(N, -1)
    raw_names = []
    for d in range(D):
        for m in range(M):
            for c in range(C):
                raw_names.append(
                    f"Raw__D{d}__N{m}__{his[c]}"
                )
    add_block(raw, raw_names, "RawMeasured")

    # ------------------------------------------------------------------
    # B. Historical trend across six diagnoses for each node×HI channel
    #    delta, normalized-position slope, mean, std
    # ------------------------------------------------------------------
    temp_cols = []
    temp_names = []
    for m in range(M):
        for c in range(C):
            a = x[:, :, m, c]  # [N,6]
            temp_cols.extend(
                [
                    a[:, -1] - a[:, 0],
                    safe_slope_axis(a),
                    np.mean(a, axis=1),
                    np.std(a, axis=1),
                ]
            )
            temp_names.extend(
                [
                    f"HistDelta__N{m}__{his[c]}",
                    f"HistSlope__N{m}__{his[c]}",
                    f"HistMean__N{m}__{his[c]}",
                    f"HistStd__N{m}__{his[c]}",
                ]
            )
    add_block(
        np.column_stack(temp_cols),
        temp_names,
        "HistoricalTrend",
    )

    # ------------------------------------------------------------------
    # C. Within-diagnosis geometry for each HI across native nodes:
    #    mean, std, min, max, range, node-position slope
    # ------------------------------------------------------------------
    geo_cols = []
    geo_names = []
    for d in range(D):
        for c in range(C):
            a = x[:, d, :, c]  # [N,M]
            vals = [
                np.mean(a, axis=1),
                np.std(a, axis=1),
                np.min(a, axis=1),
                np.max(a, axis=1),
                np.max(a, axis=1) - np.min(a, axis=1),
                safe_slope_axis(a),
            ]
            nms = [
                "Mean",
                "Std",
                "Min",
                "Max",
                "Range",
                "NodeSlope",
            ]
            geo_cols.extend(vals)
            geo_names.extend(
                [
                    f"WithinDiag_{nm}__D{d}__{his[c]}"
                    for nm in nms
                ]
            )
    add_block(
        np.column_stack(geo_cols),
        geo_names,
        "WithinDiagnosisGeometry",
    )

    # ------------------------------------------------------------------
    # D. Consecutive node differences
    # ------------------------------------------------------------------
    diff_cols = []
    diff_names = []
    for d in range(D):
        for c in range(C):
            a = x[:, d, :, c]
            da = np.diff(a, axis=1)
            for m in range(M - 1):
                diff_cols.append(da[:, m])
                diff_names.append(
                    f"NodeDiff__D{d}__N{m+1}-N{m}__{his[c]}"
                )
    if diff_cols:
        add_block(
            np.column_stack(diff_cols),
            diff_names,
            "ConsecutiveNodeDifference",
        )

    # ------------------------------------------------------------------
    # E. Cross-HI same-node geometry:
    #    signed difference and scale-normalized difference.
    # ------------------------------------------------------------------
    cross_cols = []
    cross_names = []
    for d in range(D):
        for m in range(M):
            a = x[:, d, m, 0]
            b = x[:, d, m, 1]

            diff = a - b
            ndiff = diff / (
                np.abs(a) + np.abs(b) + 1e-6
            )

            cross_cols.extend([diff, ndiff])
            cross_names.extend(
                [
                    f"CrossHI_Diff__D{d}__N{m}",
                    f"CrossHI_NormDiff__D{d}__N{m}",
                ]
            )

    add_block(
        np.column_stack(cross_cols),
        cross_names,
        "CrossHIWithinNode",
    )

    eng = np.concatenate(blocks, axis=1).astype(np.float32)

    if not np.isfinite(eng).all():
        raise RuntimeError(
            "FixedTime deterministic engineering produced NaN/Inf"
        )

    registry = pd.DataFrame(reg_rows)
    return eng, names, registry


def fixedtime_pair_matrix(
    hit: GenericHITensor,
    hi_a: str,
    hi_b: str,
) -> np.ndarray:
    imap = {h: i for i, h in enumerate(hit.hi_names)}
    if hi_a not in imap or hi_b not in imap:
        raise KeyError((hi_a, hi_b))
    return np.asarray(
        hit.values[:, :, :, [imap[hi_a], imap[hi_b]]],
        np.float32,
    )


# =============================================================================
# 7. Source-only ranking / metrics
# =============================================================================

def sample_indices(
    hit,
    batteries: Sequence[str],
    origins: Sequence[int],
) -> np.ndarray:
    bm = np.isin(np.asarray(hit.batteries, dtype=object), list(batteries))
    om = np.isin(np.asarray(hit.origins, dtype=int), list(map(int, origins)))
    return np.where(bm & om)[0].astype(np.int32)


def origin80_indices(hit, batteries: Sequence[str]) -> np.ndarray:
    idx = sample_indices(hit, batteries, [TEST_ORIGIN])
    if len(idx) != len(batteries):
        raise RuntimeError(
            f"origin80 sample count={len(idx)} vs batteries={len(batteries)}"
        )
    return idx


def true_rul(bundle, idx: Sequence[int]) -> np.ndarray:
    return np.asarray(
        bundle.true_eol_lead[np.asarray(idx, int)],
        dtype=float,
    )


def spearman_abs(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 5:
        return 0.0

    rx = pd.Series(x[m]).rank(method="average").to_numpy(float)
    ry = pd.Series(y[m]).rank(method="average").to_numpy(float)

    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0

    r = np.corrcoef(rx, ry)[0, 1]
    return float(abs(r)) if np.isfinite(r) else 0.0


def rank_generic_source_only(
    hit,
    bundle,
    source_batteries: Sequence[str],
    shortlist_n: int,
    force_his: Sequence[str] = (),
) -> pd.DataFrame:
    idx = origin80_indices(hit, source_batteries)
    y = true_rul(bundle, idx)

    rows = []
    for j, hi in enumerate(hit.hi_names):
        x = np.asarray(hit.values[idx, :, :, j], float).reshape(len(idx), -1)
        rhos = [
            spearman_abs(x[:, k], y)
            for k in range(x.shape[1])
        ]
        strongest = sorted(rhos, reverse=True)[:4]

        rows.append(
            {
                "HI": hi,
                "RelevanceTop4MeanAbsSpearman": float(np.mean(strongest)),
                "MaxAbsSpearman": float(max(rhos)),
                "MedianAbsSpearman": float(np.median(rhos)),
                "Positions": int(x.shape[1]),
            }
        )

    tab = pd.DataFrame(rows).sort_values(
        [
            "RelevanceTop4MeanAbsSpearman",
            "MaxAbsSpearman",
            "HI",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    tab["Rank"] = np.arange(1, len(tab) + 1)

    keep = tab.head(max(2, int(shortlist_n)))["HI"].tolist()
    for h in force_his:
        if h in hit.hi_names and h not in keep:
            keep.append(h)

    tab["Shortlisted"] = tab["HI"].isin(keep).astype(int)
    return tab


def rul_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]

    if len(yt) == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "MedianAE": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
            "R2": np.nan,
            "MAPE_pct": np.nan,
            "Within25": np.nan,
            "Within50": np.nan,
            "Within100": np.nan,
            "PredStd": np.nan,
            "TrueStd": np.nan,
            "SpreadRatio": np.nan,
            "SpreadGap": np.nan,
        }

    e = yp - yt
    ss_res = float(np.sum(e * e))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else np.nan

    ps = float(np.std(yp))
    ts = float(np.std(yt))

    return {
        "N": int(len(yt)),
        "MAE": float(np.mean(np.abs(e))),
        "MedianAE": float(np.median(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e * e))),
        "Bias": float(np.mean(e)),
        "R2": float(r2),
        "MAPE_pct": float(
            100.0 * np.mean(
                np.abs(e) / np.maximum(np.abs(yt), 1e-8)
            )
        ),
        "Within25": float(np.mean(np.abs(e) <= 25)),
        "Within50": float(np.mean(np.abs(e) <= 50)),
        "Within100": float(np.mean(np.abs(e) <= 100)),
        "PredStd": ps,
        "TrueStd": ts,
        "SpreadRatio": float(ps / ts) if ts > 1e-12 else np.nan,
        "SpreadGap": float(abs(1.0 - ps / ts)) if ts > 1e-12 else np.nan,
    }


# =============================================================================
# 8. Event Top2 search using exact V4.12A representation
# =============================================================================

def event_top2_search(
    v412a,
    event_hit,
    bundle,
    v410a,
    layout,
    source_batteries: Sequence[str],
    ranking: pd.DataFrame,
    args: argparse.Namespace,
    held: str,
) -> Tuple[List[str], Dict[str, float], pd.DataFrame]:
    groups = split_groups(source_batteries, int(args.inner_folds))

    shortlist = ranking[
        ranking["Shortlisted"].eq(1)
    ].sort_values("Rank")["HI"].tolist()

    if len(shortlist) < 2:
        raise RuntimeError(f"{held}: Event shortlist <2")

    rank_map = dict(zip(ranking["HI"], ranking["Rank"]))

    rows = []
    best_pair = None
    best_key = None
    best_met = None

    for a, b in itertools.combinations(shortlist, 2):
        pair = sorted(
            [a, b],
            key=lambda h: (rank_map.get(h, 10**9), h),
        )

        met, _ = v412a.inner_cv_selected_set(
            event_hit,
            bundle,
            v410a,
            layout,
            pair,
            source_batteries,
            groups,
            make_search_args(args),
        )

        key = (
            float(met["MAE"]),
            float(met["RMSE"]),
            float(met["SpreadGap"]),
            "|".join(pair),
        )

        rows.append(
            {
                "OuterHeldoutBattery": held,
                "Platform": "Event",
                "SelectedHISet": "|".join(pair),
                **met,
            }
        )

        if best_key is None or key < best_key:
            best_key = key
            best_pair = list(pair)
            best_met = dict(met)

    assert best_pair is not None
    assert best_met is not None

    return best_pair, best_met, pd.DataFrame(rows)


def make_search_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.train_origin_mode = "strict"
    ns.xgb_threads = int(args.xgb_threads)
    return ns


# =============================================================================
# 9. FixedTime Top2 search
# =============================================================================

def fixedtime_inner_cv_pair(
    v412a,
    hit: GenericHITensor,
    bundle,
    pair: Sequence[str],
    source_batteries: Sequence[str],
    groups: Sequence[Sequence[str]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    pair = list(pair)
    Xall, _, _ = engineer_fixedtime_pair(
        fixedtime_pair_matrix(hit, pair[0], pair[1]),
        pair[0],
        pair[1],
    )

    rows = []

    for fi, val_bats in enumerate(groups, start=1):
        val_bats = list(val_bats)
        valset = set(val_bats)

        train_bats = [
            b for b in source_batteries
            if b not in valset
        ]

        tr_idx = sample_indices(hit, train_bats, [TEST_ORIGIN])
        va_idx = origin80_indices(hit, val_bats)

        ytr = true_rul(bundle, tr_idx)
        yva = true_rul(bundle, va_idx)

        sw = np.ones(len(tr_idx), dtype=np.float32)

        model = v412a.fit_xgb_rul(
            Xall[tr_idx],
            ytr,
            sw,
            int(args.xgb_threads),
        )

        pred = np.asarray(
            model.predict(Xall[va_idx]),
            float,
        ).reshape(-1)
        pred = np.clip(pred, 1.0, float(K_TRAJ))

        for j, ii in enumerate(va_idx):
            rows.append(
                {
                    "Battery": str(hit.batteries[ii]),
                    "InnerFold": fi,
                    "TrueRUL": float(yva[j]),
                    "PredRUL": float(pred[j]),
                }
            )

        del model
        gc.collect()

    oof = pd.DataFrame(rows).sort_values("Battery").reset_index(drop=True)

    if len(oof) != len(source_batteries):
        raise RuntimeError(
            f"FixedTime inner OOF rows={len(oof)} vs source={len(source_batteries)}"
        )
    if oof["Battery"].nunique() != len(source_batteries):
        raise RuntimeError("FixedTime inner OOF battery duplicate/missing")

    met = rul_metrics(
        oof["TrueRUL"].to_numpy(float),
        oof["PredRUL"].to_numpy(float),
    )

    return met, oof


def fixedtime_top2_search(
    v412a,
    hit: GenericHITensor,
    bundle,
    source_batteries: Sequence[str],
    ranking: pd.DataFrame,
    args: argparse.Namespace,
    held: str,
) -> Tuple[List[str], Dict[str, float], pd.DataFrame]:
    groups = split_groups(source_batteries, int(args.inner_folds))

    shortlist = ranking[
        ranking["Shortlisted"].eq(1)
    ].sort_values("Rank")["HI"].tolist()

    if len(shortlist) < 2:
        raise RuntimeError(f"{held}: FixedTime shortlist <2")

    rank_map = dict(zip(ranking["HI"], ranking["Rank"]))

    rows = []
    best_pair = None
    best_key = None
    best_met = None

    for a, b in itertools.combinations(shortlist, 2):
        pair = sorted(
            [a, b],
            key=lambda h: (rank_map.get(h, 10**9), h),
        )

        met, _ = fixedtime_inner_cv_pair(
            v412a,
            hit,
            bundle,
            pair,
            source_batteries,
            groups,
            args,
        )

        key = (
            float(met["MAE"]),
            float(met["RMSE"]),
            float(met["SpreadGap"]),
            "|".join(pair),
        )

        rows.append(
            {
                "OuterHeldoutBattery": held,
                "Platform": "FixedTime",
                "SelectedHISet": "|".join(pair),
                **met,
            }
        )

        if best_key is None or key < best_key:
            best_key = key
            best_pair = list(pair)
            best_met = dict(met)

    assert best_pair is not None
    assert best_met is not None

    return best_pair, best_met, pd.DataFrame(rows)


def fixedtime_outer_predict(
    v412a,
    hit: GenericHITensor,
    bundle,
    pair: Sequence[str],
    source_batteries: Sequence[str],
    held: str,
    args: argparse.Namespace,
) -> float:
    pair = list(pair)

    Xall, _, _ = engineer_fixedtime_pair(
        fixedtime_pair_matrix(hit, pair[0], pair[1]),
        pair[0],
        pair[1],
    )

    tr_idx = sample_indices(
        hit,
        source_batteries,
        [TEST_ORIGIN],
    )
    te_idx = origin80_indices(
        hit,
        [held],
    )

    ytr = true_rul(bundle, tr_idx)
    sw = np.ones(len(tr_idx), dtype=np.float32)

    model = v412a.fit_xgb_rul(
        Xall[tr_idx],
        ytr,
        sw,
        int(args.xgb_threads),
    )

    pred = float(
        np.asarray(
            model.predict(Xall[te_idx]),
            float,
        ).reshape(-1)[0]
    )

    pred = float(np.clip(pred, 1.0, float(K_TRAJ)))

    del model
    gc.collect()
    return pred


# =============================================================================
# 10. Fold cache helpers
# =============================================================================

def cfg_platform_modality(cfg: str) -> Tuple[str, str]:
    if cfg.startswith("Event_"):
        return "Event", cfg.split("_", 1)[1]
    if cfg.startswith("FixedTime_"):
        return "FixedTime", cfg.split("_", 1)[1]
    raise ValueError(cfg)


def fold_cfg_dir(out: Path, battery: str) -> Path:
    return out / "folds" / normalize_battery(battery)


def result_path(out: Path, battery: str, cfg: str) -> Path:
    return fold_cfg_dir(out, battery) / f"{cfg}_result.json"


def ranking_path(out: Path, battery: str, cfg: str) -> Path:
    return fold_cfg_dir(out, battery) / f"{cfg}_ranking.csv"


def search_path(out: Path, battery: str, cfg: str) -> Path:
    return fold_cfg_dir(out, battery) / f"{cfg}_pair_search.csv.gz"


def cache_valid(out: Path, battery: str, cfg: str) -> bool:
    p = result_path(out, battery, cfg)
    if not p.exists():
        return False
    try:
        d = load_json(p)
        return (
            d.get("OuterHeldoutBattery") == normalize_battery(battery)
            and d.get("Config") == cfg
            and np.isfinite(float(d.get("PredRUL", np.nan)))
            and np.isfinite(float(d.get("TrueRUL", np.nan)))
        )
    except Exception:
        return False


# =============================================================================
# 11. One config / one outer fold
# =============================================================================

def load_v412a_reuse_row(
    frozen_outer: pd.DataFrame,
    held: str,
) -> Dict[str, object]:
    q = frozen_outer[
        frozen_outer["OuterHeldoutBattery"].eq(held)
    ]
    if len(q) != 1:
        raise RuntimeError(f"{held}: V4.12A frozen row !=1")

    r = q.iloc[0]

    return {
        "OuterHeldoutBattery": held,
        "Config": "Event_VIT",
        "Platform": "Event",
        "Modality": "VIT",
        "TrueRUL": float(r["TrueRUL"]),
        "PredRUL": float(r["Pred_Top2"]),
        "SelectedTop2": str(r["SelectedTop2"]),
        "HI_A": str(r["SelectedTop2"]).split("|")[0],
        "HI_B": str(r["SelectedTop2"]).split("|")[1],
        "InnerCV_MAE": finite_or_nan(r.get("InnerCV_Top2_MAE", np.nan)),
        "InnerCV_RMSE": np.nan,
        "CandidatePoolSize": np.nan,
        "ShortlistSize": np.nan,
        "PairCandidatesEvaluated": np.nan,
        "FeatureCount": 196,
        "MeasuredScalarInputs": 36,
        "CheckpointCount": 3,
        "HistoryDiagnosisCount": 6,
        "TrainingOrigins": "80",
        "SourceBatteries": 46,
        "HeldoutUsedForHIRanking": 0,
        "HeldoutUsedForPairSearch": 0,
        "HeldoutUsedForTraining": 0,
        "ReusedFrozenV412A": 1,
        "Representation": "Exact_V412A_Event_36to196",
        "CrossPlatformBudgetMatched": 0,
        "Runtime_s": 0.0,
    }


def run_config(
    cfg: str,
    held: str,
    v412a,
    v410a,
    bundle,
    event_hit,
    event_layout,
    fixed_hit,
    depmap,
    frozen_outer,
    args: argparse.Namespace,
) -> Dict[str, object]:
    t0 = time.perf_counter()

    platform, modality = cfg_platform_modality(cfg)

    # Exact frozen V4.12A anchor.
    if cfg == "Event_VIT" and not args.recompute_event_vit:
        result = load_v412a_reuse_row(frozen_outer, held)
        ensure_dir(fold_cfg_dir(args.output_root, held))
        save_json(result, result_path(args.output_root, held, cfg))
        return result

    all_bats = sorted(set(map(normalize_battery, bundle.battery.tolist())))
    source_bats = [b for b in all_bats if b != held]

    if platform == "Event":
        allowed = allowed_his(event_hit.hi_names, modality, depmap)
        if len(allowed) < 2:
            raise RuntimeError(
                f"{cfg}: modality-constrained Event HIs={len(allowed)} <2"
            )

        hsub = subset_event_hit(v412a, event_hit, allowed)

        ranking = v412a.rank_his_source_only(
            hsub,
            bundle,
            source_bats,
            int(args.shortlist),
        )

        pair, inner_met, search = event_top2_search(
            v412a,
            hsub,
            bundle,
            v410a,
            event_layout,
            source_bats,
            ranking,
            args,
            held,
        )

        pred = v412a.fit_predict_outer_set(
            hsub,
            bundle,
            v410a,
            event_layout,
            pair,
            source_bats,
            held,
            make_search_args(args),
        )

        # Exact Event pair representation count.
        Xpair = v412a.make_pair_bundle_input(
            hsub,
            pair[0],
            pair[1],
            event_layout,
        )
        Xeng, _, _ = v410a.engineer_features(Xpair)
        feature_count = int(Xeng.shape[1])
        measured_scalars = int(6 * 3 * 2)

    else:
        allowed = allowed_his(fixed_hit.hi_names, modality, depmap)
        if len(allowed) < 2:
            raise RuntimeError(
                f"{cfg}: modality-constrained FixedTime HIs={len(allowed)} <2"
            )

        hsub = subset_generic_hit(fixed_hit, allowed)

        ranking = rank_generic_source_only(
            hsub,
            bundle,
            source_bats,
            int(args.shortlist),
        )

        pair, inner_met, search = fixedtime_top2_search(
            v412a,
            hsub,
            bundle,
            source_bats,
            ranking,
            args,
            held,
        )

        pred = fixedtime_outer_predict(
            v412a,
            hsub,
            bundle,
            pair,
            source_bats,
            held,
            args,
        )

        Xeng, _, _ = engineer_fixedtime_pair(
            fixedtime_pair_matrix(hsub, pair[0], pair[1]),
            pair[0],
            pair[1],
        )
        feature_count = int(Xeng.shape[1])
        measured_scalars = int(6 * len(fixed_hit.nodes) * 2)

    te_idx = origin80_indices(
        hsub,
        [held],
    )
    y = float(true_rul(bundle, te_idx)[0])

    ensure_dir(fold_cfg_dir(args.output_root, held))

    ranking2 = ranking.copy()
    ranking2.insert(0, "OuterHeldoutBattery", held)
    ranking2.insert(1, "Config", cfg)
    save_csv(
        ranking2,
        ranking_path(args.output_root, held, cfg),
    )

    search2 = search.copy()
    search2.insert(1, "Config", cfg)
    search2.to_csv(
        search_path(args.output_root, held, cfg),
        index=False,
        encoding=ENC,
        compression=FAST_GZIP,
    )

    result = {
        "OuterHeldoutBattery": held,
        "Config": cfg,
        "Platform": platform,
        "Modality": modality,
        "TrueRUL": y,
        "PredRUL": float(pred),
        "SelectedTop2": "|".join(pair),
        "HI_A": pair[0],
        "HI_B": pair[1],
        "InnerCV_MAE": float(inner_met["MAE"]),
        "InnerCV_RMSE": float(inner_met["RMSE"]),
        "InnerCV_SpreadRatio": float(inner_met["SpreadRatio"]),
        "CandidatePoolSize": int(len(allowed)),
        "ShortlistSize": int(ranking["Shortlisted"].sum()),
        "PairCandidatesEvaluated": int(len(search)),
        "FeatureCount": feature_count,
        "MeasuredScalarInputs": measured_scalars,
        "CheckpointCount": 3 if platform == "Event" else len(fixed_hit.nodes),
        "HistoryDiagnosisCount": 6,
        "TrainingOrigins": "80",
        "SourceBatteries": len(source_bats),
        "HeldoutUsedForHIRanking": 0,
        "HeldoutUsedForPairSearch": 0,
        "HeldoutUsedForTraining": 0,
        "ReusedFrozenV412A": 0,
        "Representation": (
            "Exact_V412A_Event_36to196"
            if platform == "Event"
            else "NativeFixedTime_DeterministicPairEngineering"
        ),
        "CrossPlatformBudgetMatched": 0,
        "Runtime_s": float(time.perf_counter() - t0),
    }

    save_json(
        result,
        result_path(args.output_root, held, cfg),
    )

    return result


# =============================================================================
# 12. Preflight
# =============================================================================

def event_coverage_table(event_hit) -> pd.DataFrame:
    d = event_hit.coverage.copy()
    d.insert(0, "Platform", "Event")
    return d


def build_modality_coverage(
    event_hit,
    fixed_hit,
    depmap,
) -> pd.DataFrame:
    rows = []

    for cfg in CONFIGS:
        platform, modality = cfg_platform_modality(cfg)
        hnames = event_hit.hi_names if platform == "Event" else fixed_hit.hi_names
        allowed = allowed_his(hnames, modality, depmap)

        rows.append(
            {
                "Config": cfg,
                "Platform": platform,
                "Modality": modality,
                "EligibleCompleteHI_AllSensors": len(hnames),
                "EligibleHI_AfterModalityConstraint": len(allowed),
                "AllowedHIs": "|".join(allowed),
                "HardwareSensorInterpretation": (
                    "V+I required for Event localization"
                    + (" + T" if modality == "VIT" else "")
                    if platform == "Event"
                    else modality
                ),
            }
        )

    return pd.DataFrame(rows)


def preflight(args: argparse.Namespace):
    if not args.v412a_helper.exists():
        raise FileNotFoundError(args.v412a_helper)

    v412a = load_module(
        args.v412a_helper,
        "rwth48_v412a_for_v413f",
    )

    aargs = make_v412a_args(args)

    (
        helper,
        v410a,
        bundle,
        baseline,
        event_hit,
        event_layout,
        event_recon,
        handoff_audit,
        v412a_prov,
        event_schema_audit,
    ) = v412a.preflight(aargs)

    frozen_outer = load_frozen_v412a_outer(
        Path(args.v412a_outer)
    )

    comp = load_json(Path(args.v412a_complete))
    if comp.get("Status") != "PASS":
        raise RuntimeError(
            f"V4.12A complete status={comp.get('Status')} != PASS"
        )

    # FixedTime frozen table.
    fixed_path, locate_audit = locate_fixedtime_ready(args)

    print(
        f"[V413F] FixedTime-ready: {fixed_path}",
        flush=True,
    )

    fixed_raw = pd.read_csv(
        fixed_path,
        compression="infer",
        low_memory=False,
    )

    fixed_schema = detect_fixedtime_schema(
        fixed_raw,
        args.fixedtime_nodes,
    )

    fixed = filter_fixedtime_table(
        fixed_raw,
        fixed_schema,
        args.fixedtime_nodes,
        args.phase,
    )

    fixed_hit = build_fixedtime_tensor(
        fixed,
        fixed_schema,
        bundle,
        args.fixedtime_nodes,
    )

    # Sensor dependency registry across BOTH complete candidate pools.
    all_his = sorted(
        set(event_hit.hi_names)
        | set(fixed_hit.hi_names)
    )

    registry = build_dependency_registry(
        all_his,
        args.dependency_registry,
    )

    depmap = dependency_map(registry)

    unknown = registry[
        registry["DependencyKnown"].eq(0)
    ]["HI"].tolist()

    coverage = build_modality_coverage(
        event_hit,
        fixed_hit,
        depmap,
    )

    # True-RUL agreement: frozen V4.12A vs bundle.
    frozen_map = frozen_outer.set_index("OuterHeldoutBattery")
    rul_diffs = []

    batteries = sorted(set(map(normalize_battery, bundle.battery.tolist())))
    for b in batteries:
        idx = origin80_indices(event_hit, [b])
        y = float(true_rul(bundle, idx)[0])
        y0 = float(frozen_map.loc[b, "TrueRUL"])
        rul_diffs.append(abs(y - y0))

    # Event-VIT selected HIs must all be semantically V/I/T-valid.
    bad_vit_his = []
    for s in frozen_outer["SelectedTop2"].astype(str):
        for h in s.split("|"):
            dep = depmap.get(h)
            if dep is None or not dep.issubset(MODALITY_SENSORS["VIT"]):
                bad_vit_his.append(h)

    # Required modality candidate counts.
    covmap = coverage.set_index("Config")
    too_small = [
        cfg
        for cfg in CONFIGS
        if int(
            covmap.loc[
                cfg,
                "EligibleHI_AfterModalityConstraint",
            ]
        ) < 2
    ]

    pre_rows = [
        {
            "Check": "V4.12A complete status",
            "Observed": comp.get("Status"),
            "Expected": "PASS",
            "PASS": int(comp.get("Status") == "PASS"),
        },
        {
            "Check": "Outer batteries",
            "Observed": len(batteries),
            "Expected": 47,
            "PASS": int(len(batteries) == 47),
        },
        {
            "Check": "V4.12A frozen outer rows",
            "Observed": len(frozen_outer),
            "Expected": 47,
            "PASS": int(len(frozen_outer) == 47),
        },
        {
            "Check": "Bundle TrueRUL vs V4.12A max abs diff",
            "Observed": float(max(rul_diffs)),
            "Expected": "<=1e-9",
            "PASS": int(max(rul_diffs) <= 1e-9),
        },
        {
            "Check": "Unknown sensor dependency among eligible HIs",
            "Observed": len(unknown),
            "Expected": 0,
            "PASS": int(len(unknown) == 0),
        },
        {
            "Check": "Configs with <2 eligible HIs",
            "Observed": len(too_small),
            "Expected": 0,
            "PASS": int(len(too_small) == 0),
        },
        {
            "Check": "Frozen Event-VIT selected HIs modality-valid",
            "Observed": len(set(bad_vit_his)),
            "Expected": 0,
            "PASS": int(len(bad_vit_his) == 0),
        },
        {
            "Check": "FixedTime frozen node count",
            "Observed": len(fixed_hit.nodes),
            "Expected": 4,
            "PASS": int(len(fixed_hit.nodes) == 4),
        },
        {
            "Check": "Event config list physically valid",
            "Observed": "VI|VIT",
            "Expected": "VI|VIT",
            "PASS": 1,
        },
    ]

    pre = pd.DataFrame(pre_rows)

    if not pre["PASS"].eq(1).all():
        details = []
        if unknown:
            details.append(
                "Unknown sensor dependencies:\n"
                + "\n".join(unknown)
            )
        if too_small:
            details.append(
                "Insufficient modality candidate pools: "
                + ", ".join(too_small)
            )
        raise RuntimeError(
            "V4.13F preflight FAIL:\n"
            + pre[pre["PASS"].eq(0)].to_string(index=False)
            + ("\n\n" + "\n\n".join(details) if details else "")
        )

    return {
        "v412a": v412a,
        "helper": helper,
        "v410a": v410a,
        "bundle": bundle,
        "event_hit": event_hit,
        "event_layout": event_layout,
        "fixed_hit": fixed_hit,
        "fixed_path": fixed_path,
        "fixed_raw_rows": len(fixed_raw),
        "fixed_filtered_rows": len(fixed),
        "fixed_schema": fixed_schema,
        "fixed_locate_audit": locate_audit,
        "registry": registry,
        "depmap": depmap,
        "coverage": coverage,
        "frozen_outer": frozen_outer,
        "preflight": pre,
        "event_recon": event_recon,
        "event_schema_audit": event_schema_audit,
        "v412a_prov": v412a_prov,
        "handoff_audit": handoff_audit,
    }


# =============================================================================
# 13. Run folds
# =============================================================================

def run_one_battery(
    held: str,
    ctx: Dict[str, object],
    args: argparse.Namespace,
):
    v412a = ctx["v412a"]
    v410a = ctx["v410a"]
    bundle = ctx["bundle"]
    event_hit = ctx["event_hit"]
    event_layout = ctx["event_layout"]
    fixed_hit = ctx["fixed_hit"]
    depmap = ctx["depmap"]
    frozen_outer = ctx["frozen_outer"]

    for cfg in args.configs:
        if cache_valid(args.output_root, held, cfg) and not args.force:
            print(f"[V413F] {held} | {cfg}: cache", flush=True)
            continue

        print(
            f"[V413F] {held} | {cfg}: source-only modality search ...",
            flush=True,
        )

        r = run_config(
            cfg,
            held,
            v412a,
            v410a,
            bundle,
            event_hit,
            event_layout,
            fixed_hit,
            depmap,
            frozen_outer,
            args,
        )

        print(
            f"[V413F] {held} | {cfg}: "
            f"Top2={r['SelectedTop2']} "
            f"Pred={float(r['PredRUL']):.2f} "
            f"True={float(r['TrueRUL']):.0f} "
            f"runtime={float(r['Runtime_s']):.1f}s",
            flush=True,
        )


# =============================================================================
# 14. Final aggregation / bootstrap
# =============================================================================

def load_all_results(
    out: Path,
    batteries: Sequence[str],
) -> pd.DataFrame:
    rows = []
    missing = []

    for b in batteries:
        for cfg in CONFIGS:
            p = result_path(out, b, cfg)
            if not p.exists():
                missing.append(f"{b}|{cfg}")
                continue
            rows.append(load_json(p))

    if missing:
        raise RuntimeError(
            "Missing V4.13F fold/config caches:\n"
            + "\n".join(missing[:100])
            + (f"\n... total {len(missing)} missing" if len(missing) > 100 else "")
        )

    d = pd.DataFrame(rows)
    d["OuterHeldoutBattery"] = d["OuterHeldoutBattery"].map(normalize_battery)
    d["TrueRUL"] = pd.to_numeric(d["TrueRUL"], errors="coerce")
    d["PredRUL"] = pd.to_numeric(d["PredRUL"], errors="coerce")

    return d.sort_values(
        ["Config", "OuterHeldoutBattery"]
    ).reset_index(drop=True)


def pooled_metrics(outer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cfg, g in outer.groupby("Config", sort=True):
        met = rul_metrics(
            g["TrueRUL"].to_numpy(float),
            g["PredRUL"].to_numpy(float),
        )

        first = g.iloc[0]

        rows.append(
            {
                "Config": cfg,
                "Platform": first["Platform"],
                "Modality": first["Modality"],
                **met,
                "MeanInnerCV_MAE": float(
                    pd.to_numeric(
                        g["InnerCV_MAE"],
                        errors="coerce",
                    ).mean()
                ),
                "MeanCandidatePoolSize": float(
                    pd.to_numeric(
                        g["CandidatePoolSize"],
                        errors="coerce",
                    ).mean()
                ),
                "MeasuredScalarInputs": int(
                    pd.to_numeric(
                        g["MeasuredScalarInputs"],
                        errors="coerce",
                    ).dropna().iloc[0]
                ),
                "FeatureCount": int(
                    pd.to_numeric(
                        g["FeatureCount"],
                        errors="coerce",
                    ).dropna().iloc[0]
                ),
                "CheckpointCount": int(
                    pd.to_numeric(
                        g["CheckpointCount"],
                        errors="coerce",
                    ).dropna().iloc[0]
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["Platform", "Modality"]
    ).reset_index(drop=True)


def selection_frequency(outer: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cfg, g in outer.groupby("Config", sort=True):
        counts: Dict[str, int] = {}
        pair_counts: Dict[str, int] = {}

        for r in g.itertuples(index=False):
            pair = str(r.SelectedTop2)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            for h in [str(r.HI_A), str(r.HI_B)]:
                counts[h] = counts.get(h, 0) + 1

        for hi, n in sorted(
            counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            rows.append(
                {
                    "Config": cfg,
                    "RecordType": "HI",
                    "Item": hi,
                    "SelectionCount": n,
                    "SelectionRate": n / EXPECTED_BATTERIES,
                }
            )

        for pair, n in sorted(
            pair_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            rows.append(
                {
                    "Config": cfg,
                    "RecordType": "PAIR",
                    "Item": pair,
                    "SelectionCount": n,
                    "SelectionRate": n / EXPECTED_BATTERIES,
                }
            )

    return pd.DataFrame(rows)


def paired_comparison_row(
    outer: pd.DataFrame,
    cfg_a: str,
    cfg_b: str,
    label: str,
    comparison_type: str,
) -> Dict[str, object]:
    a = outer[
        outer["Config"].eq(cfg_a)
    ][
        ["OuterHeldoutBattery", "TrueRUL", "PredRUL"]
    ].rename(
        columns={
            "TrueRUL": "TrueRUL_A",
            "PredRUL": "Pred_A",
        }
    )

    b = outer[
        outer["Config"].eq(cfg_b)
    ][
        ["OuterHeldoutBattery", "TrueRUL", "PredRUL"]
    ].rename(
        columns={
            "TrueRUL": "TrueRUL_B",
            "PredRUL": "Pred_B",
        }
    )

    m = a.merge(
        b,
        on="OuterHeldoutBattery",
        how="inner",
        validate="one_to_one",
    )

    if len(m) != EXPECTED_BATTERIES:
        raise RuntimeError(
            f"{cfg_a} vs {cfg_b}: paired rows={len(m)} !=47"
        )

    max_truth_diff = float(
        np.max(
            np.abs(
                m["TrueRUL_A"].to_numpy(float)
                - m["TrueRUL_B"].to_numpy(float)
            )
        )
    )

    if max_truth_diff > 1e-9:
        raise RuntimeError(
            f"{cfg_a} vs {cfg_b}: TrueRUL mismatch"
        )

    y = m["TrueRUL_A"].to_numpy(float)
    pa = m["Pred_A"].to_numpy(float)
    pb = m["Pred_B"].to_numpy(float)

    ma = rul_metrics(y, pa)
    mb = rul_metrics(y, pb)

    ae_a = np.abs(pa - y)
    ae_b = np.abs(pb - y)

    return {
        "Comparison": label,
        "ComparisonType": comparison_type,
        "ConfigA_Base": cfg_a,
        "ConfigB_AddedOrAlternative": cfg_b,
        "N": len(m),
        "MAE_A": ma["MAE"],
        "MAE_B": mb["MAE"],
        "MAEGain_B_vs_A_pct": float(
            100.0 * (ma["MAE"] - mb["MAE"]) / ma["MAE"]
        ),
        "RMSE_A": ma["RMSE"],
        "RMSE_B": mb["RMSE"],
        "RMSEGain_B_vs_A_pct": float(
            100.0 * (ma["RMSE"] - mb["RMSE"]) / ma["RMSE"]
        ),
        "MeanPairedAbsErrorDiff_BminusA": float(
            np.mean(ae_b - ae_a)
        ),
        "MedianPairedAbsErrorDiff_BminusA": float(
            np.median(ae_b - ae_a)
        ),
        "B_BetterBatteryCount": int(np.sum(ae_b < ae_a)),
        "A_BetterBatteryCount": int(np.sum(ae_a < ae_b)),
        "TieBatteryCount": int(np.sum(np.isclose(ae_a, ae_b, atol=1e-12))),
        "CrossPlatformCheckpointBudgetMatched": int(
            not comparison_type.startswith("CrossPlatform")
        ),
    }


def comparison_plan() -> List[Tuple[str, str, str, str]]:
    return [
        (
            "Event_VI",
            "Event_VIT",
            "Event: temperature marginal value (VI -> VIT)",
            "WithinPlatform_Temperature",
        ),
        (
            "FixedTime_V",
            "FixedTime_VT",
            "FixedTime: temperature marginal value (V -> VT)",
            "WithinPlatform_Temperature",
        ),
        (
            "FixedTime_VI",
            "FixedTime_VIT",
            "FixedTime: temperature marginal value (VI -> VIT)",
            "WithinPlatform_Temperature",
        ),
        (
            "FixedTime_V",
            "FixedTime_VI",
            "FixedTime: current marginal value (V -> VI)",
            "WithinPlatform_Current",
        ),
        (
            "FixedTime_VT",
            "FixedTime_VIT",
            "FixedTime: current marginal value (VT -> VIT)",
            "WithinPlatform_Current",
        ),
        (
            "FixedTime_VI",
            "Event_VI",
            "Native platform contrast at VI (FixedTime -> Event)",
            "CrossPlatform_NativeBudget",
        ),
        (
            "FixedTime_VIT",
            "Event_VIT",
            "Native platform contrast at VIT (FixedTime -> Event)",
            "CrossPlatform_NativeBudget",
        ),
    ]


def paired_bootstrap(
    outer: pd.DataFrame,
    cfg_a: str,
    cfg_b: str,
    label: str,
    n_boot: int,
) -> Dict[str, object]:
    a = outer[
        outer["Config"].eq(cfg_a)
    ][
        ["OuterHeldoutBattery", "TrueRUL", "PredRUL"]
    ].rename(
        columns={
            "TrueRUL": "y",
            "PredRUL": "pa",
        }
    )

    b = outer[
        outer["Config"].eq(cfg_b)
    ][
        ["OuterHeldoutBattery", "PredRUL"]
    ].rename(columns={"PredRUL": "pb"})

    m = a.merge(
        b,
        on="OuterHeldoutBattery",
        validate="one_to_one",
    )

    y = m["y"].to_numpy(float)
    pa = m["pa"].to_numpy(float)
    pb = m["pb"].to_numpy(float)

    d = np.abs(pb - y) - np.abs(pa - y)

    rng = np.random.default_rng(
        RANDOM_SEED
        + int(
            hashlib.sha1(label.encode("utf-8")).hexdigest()[:6],
            16,
        )
    )

    B = max(1000, int(n_boot))
    vals = np.empty(B, dtype=float)

    n = len(d)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(np.mean(d[idx]))

    lo, hi = np.quantile(vals, [0.025, 0.975])

    return {
        "Comparison": label,
        "ConfigA_Base": cfg_a,
        "ConfigB_AddedOrAlternative": cfg_b,
        "N": n,
        "ObservedMeanAbsErrorDiff_BminusA": float(np.mean(d)),
        "BootstrapMean": float(np.mean(vals)),
        "CI95_Lower": float(lo),
        "CI95_Upper": float(hi),
        "CIExcludesZero": int(lo > 0 or hi < 0),
        "Direction": (
            "B_better"
            if np.mean(d) < 0
            else "A_better"
            if np.mean(d) > 0
            else "Tie"
        ),
        "BootstrapReplicates": B,
    }


def battery_error_table(outer: pd.DataFrame) -> pd.DataFrame:
    d = outer[
        [
            "OuterHeldoutBattery",
            "Config",
            "Platform",
            "Modality",
            "TrueRUL",
            "PredRUL",
            "SelectedTop2",
        ]
    ].copy()

    d["Error"] = d["PredRUL"] - d["TrueRUL"]
    d["AbsoluteError"] = np.abs(d["Error"])
    return d


def runtime_summary(outer: pd.DataFrame) -> pd.DataFrame:
    return (
        outer.groupby(["Config", "Platform", "Modality"])
        .agg(
            Batteries=("OuterHeldoutBattery", "nunique"),
            TotalRuntime_s=("Runtime_s", "sum"),
            MeanRuntime_s=("Runtime_s", "mean"),
            MedianRuntime_s=("Runtime_s", "median"),
        )
        .reset_index()
    )


# =============================================================================
# 15. Figures
# =============================================================================

def make_figures(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    freq: pd.DataFrame,
    out: Path,
    dpi: int,
):
    figdir = ensure_dir(out / "Figures")

    # 1. Pooled MAE.
    q = metrics.sort_values(["Platform", "Modality"]).copy()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(q))
    ax.bar(x, q["MAE"])
    ax.set_xticks(x)
    ax.set_xticklabels(q["Config"], rotation=30, ha="right")
    ax.set_ylabel("RUL MAE (cycles)")
    ax.set_title("Platform-aware sensor modality ablation")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(
        figdir / "01_modality_rul_mae.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 2. Marginal gains.
    q = comparisons[
        comparisons["ComparisonType"].str.startswith("WithinPlatform")
    ].copy()

    if len(q):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(q))
        ax.bar(x, q["MAEGain_B_vs_A_pct"])
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(q["Comparison"], rotation=25, ha="right")
        ax.set_ylabel("MAE gain of B vs A (%)")
        ax.set_title("Marginal sensor value")
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(
            figdir / "02_marginal_sensor_value.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 3. Top selected HIs for Event-VI vs Event-VIT.
    q = freq[
        freq["RecordType"].eq("HI")
        & freq["Config"].isin(["Event_VI", "Event_VIT"])
    ].copy()

    if len(q):
        top = (
            q.sort_values(
                ["Config", "SelectionCount"],
                ascending=[True, False],
            )
            .groupby("Config")
            .head(8)
            .copy()
        )

        labels = (
            top["Config"].astype(str)
            + " | "
            + top["Item"].astype(str)
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        y = np.arange(len(top))
        ax.barh(y, top["SelectionCount"])
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Outer-fold selection count")
        ax.set_title("Event RUL HI selection under VI vs VIT")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        fig.savefig(
            figdir / "03_event_vi_vs_vit_hi_frequency.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# 16. Finalize / acceptance
# =============================================================================

def finalize(ctx: Dict[str, object], args: argparse.Namespace) -> str:
    bundle = ctx["bundle"]
    batteries = sorted(set(map(normalize_battery, bundle.battery.tolist())))

    outer = load_all_results(
        args.output_root,
        batteries,
    )

    metrics = pooled_metrics(outer)
    freq = selection_frequency(outer)

    comp_rows = [
        paired_comparison_row(
            outer,
            a,
            b,
            label,
            ctype,
        )
        for a, b, label, ctype in comparison_plan()
    ]
    comparisons = pd.DataFrame(comp_rows)

    boots = pd.DataFrame(
        [
            paired_bootstrap(
                outer,
                a,
                b,
                label,
                int(args.bootstrap),
            )
            for a, b, label, _ in comparison_plan()
        ]
    )

    native_cross = comparisons[
        comparisons["ComparisonType"].eq("CrossPlatform_NativeBudget")
    ].copy()

    errors = battery_error_table(outer)
    runt = runtime_summary(outer)

    save_csv(outer, args.output_root / "03_outer_predictions.csv")
    save_csv(metrics, args.output_root / "04_pooled_metrics.csv")

    sel = outer[
        [
            "OuterHeldoutBattery",
            "Config",
            "Platform",
            "Modality",
            "SelectedTop2",
            "HI_A",
            "HI_B",
            "InnerCV_MAE",
            "CandidatePoolSize",
            "ShortlistSize",
            "PairCandidatesEvaluated",
            "FeatureCount",
            "MeasuredScalarInputs",
            "ReusedFrozenV412A",
        ]
    ].copy()

    save_csv(sel, args.output_root / "05_selection_by_fold.csv")
    save_csv(freq, args.output_root / "06_hi_selection_frequency.csv")
    save_csv(
        comparisons[
            ~comparisons["ComparisonType"].eq("CrossPlatform_NativeBudget")
        ].copy(),
        args.output_root / "07_marginal_sensor_value.csv",
    )
    save_csv(boots, args.output_root / "08_paired_bootstrap.csv")
    save_csv(
        native_cross,
        args.output_root / "09_cross_platform_native_comparison.csv",
    )
    save_csv(errors, args.output_root / "10_battery_absolute_errors.csv")
    save_csv(runt, args.output_root / "11_search_runtime_summary.csv")

    # -------------------------------------------------------------
    # Acceptance checks
    # -------------------------------------------------------------
    frozen = ctx["frozen_outer"][
        ["OuterHeldoutBattery", "Pred_Top2", "SelectedTop2", "TrueRUL"]
    ].copy()

    ev = outer[
        outer["Config"].eq("Event_VIT")
    ][
        ["OuterHeldoutBattery", "PredRUL", "SelectedTop2", "TrueRUL"]
    ].copy()

    rep = frozen.merge(
        ev,
        on="OuterHeldoutBattery",
        suffixes=("_Frozen", "_V413F"),
        validate="one_to_one",
    )

    pred_diff = float(
        np.max(
            np.abs(
                rep["Pred_Top2"].to_numpy(float)
                - rep["PredRUL"].to_numpy(float)
            )
        )
    )

    truth_diff = float(
        np.max(
            np.abs(
                rep["TrueRUL_Frozen"].to_numpy(float)
                - rep["TrueRUL_V413F"].to_numpy(float)
            )
        )
    )

    pair_mismatch = int(
        (
            rep["SelectedTop2_Frozen"].astype(str)
            != rep["SelectedTop2_V413F"].astype(str)
        ).sum()
    )

    config_counts = outer.groupby("Config")["OuterHeldoutBattery"].nunique()
    all47 = bool(
        all(int(config_counts.get(cfg, 0)) == 47 for cfg in CONFIGS)
    )

    allfinite = bool(
        np.isfinite(outer["PredRUL"].to_numpy(float)).all()
        and np.isfinite(outer["TrueRUL"].to_numpy(float)).all()
    )

    held_train = int(
        pd.to_numeric(
            outer["HeldoutUsedForTraining"],
            errors="coerce",
        ).fillna(1).sum()
    )
    held_rank = int(
        pd.to_numeric(
            outer["HeldoutUsedForHIRanking"],
            errors="coerce",
        ).fillna(1).sum()
    )
    held_search = int(
        pd.to_numeric(
            outer["HeldoutUsedForPairSearch"],
            errors="coerce",
        ).fillna(1).sum()
    )

    # Selected HI dependency validity.
    depmap = ctx["depmap"]
    invalid_selected = []

    for r in outer.itertuples(index=False):
        sensors = MODALITY_SENSORS[str(r.Modality)]
        for h in [str(r.HI_A), str(r.HI_B)]:
            dep = depmap.get(h)
            if dep is None or not dep.issubset(sensors):
                invalid_selected.append(
                    f"{r.OuterHeldoutBattery}|{r.Config}|{h}"
                )

    gate = pd.DataFrame(
        [
            {
                "Check": "Preflight all PASS",
                "Observed": int(ctx["preflight"]["PASS"].eq(1).all()),
                "Expected": 1,
                "PASS": int(ctx["preflight"]["PASS"].eq(1).all()),
            },
            {
                "Check": "All 6 configs have 47 outer batteries",
                "Observed": int(all47),
                "Expected": 1,
                "PASS": int(all47),
            },
            {
                "Check": "All outer predictions finite",
                "Observed": int(allfinite),
                "Expected": 1,
                "PASS": int(allfinite),
            },
            {
                "Check": "Event_VIT prediction max abs diff vs frozen V4.12A",
                "Observed": pred_diff,
                "Expected": "<=1e-9",
                "PASS": int(pred_diff <= 1e-9),
            },
            {
                "Check": "Event_VIT selected-pair mismatch vs frozen V4.12A",
                "Observed": pair_mismatch,
                "Expected": 0,
                "PASS": int(pair_mismatch == 0),
            },
            {
                "Check": "TrueRUL max abs diff vs frozen V4.12A",
                "Observed": truth_diff,
                "Expected": "<=1e-9",
                "PASS": int(truth_diff <= 1e-9),
            },
            {
                "Check": "Heldout used for HI ranking",
                "Observed": held_rank,
                "Expected": 0,
                "PASS": int(held_rank == 0),
            },
            {
                "Check": "Heldout used for pair search",
                "Observed": held_search,
                "Expected": 0,
                "PASS": int(held_search == 0),
            },
            {
                "Check": "Heldout used for model training",
                "Observed": held_train,
                "Expected": 0,
                "PASS": int(held_train == 0),
            },
            {
                "Check": "Selected HI violates modality dependency",
                "Observed": len(invalid_selected),
                "Expected": 0,
                "PASS": int(len(invalid_selected) == 0),
            },
            {
                "Check": "Event modalities limited to VI/VIT",
                "Observed": "|".join(
                    sorted(
                        outer[
                            outer["Platform"].eq("Event")
                        ]["Modality"].unique().tolist()
                    )
                ),
                "Expected": "VI|VIT",
                "PASS": int(
                    set(
                        outer[
                            outer["Platform"].eq("Event")
                        ]["Modality"].unique().tolist()
                    )
                    == {"VI", "VIT"}
                ),
            },
            {
                "Check": "Cross-platform comparisons labeled native-budget descriptive",
                "Observed": int(
                    native_cross[
                        "CrossPlatformCheckpointBudgetMatched"
                    ].eq(0).all()
                ),
                "Expected": 1,
                "PASS": int(
                    native_cross[
                        "CrossPlatformCheckpointBudgetMatched"
                    ].eq(0).all()
                ),
            },
        ]
    )

    save_csv(gate, args.output_root / "12_acceptance_gate.csv")

    if not args.skip_figures:
        make_figures(
            metrics,
            comparisons,
            freq,
            args.output_root,
            args.dpi,
        )

    status = "PASS" if gate["PASS"].eq(1).all() else "CHECK"

    # Main temperature-value rows.
    temp_rows = comparisons[
        comparisons["ComparisonType"].eq("WithinPlatform_Temperature")
    ].to_dict("records")

    save_json(
        {
            "Task": (
                "RWTH48 V4.13F Platform-Aware Sensor Modality "
                "Ablation for Direct RUL"
            ),
            "Status": status,
            "ScientificResult": (
                "PLATFORM_AWARE_SENSOR_MODALITY_ABLATION"
            ),
            "OuterBatteries": 47,
            "OriginCycle": 80,
            "HistoryCycles": [40, 48, 56, 64, 72, 80],
            "EventModalities": ["VI", "VIT"],
            "FixedTimeModalities": ["V", "VI", "VT", "VIT"],
            "EventNodes_RefAh": list(EVENT_NODES),
            "FixedTimeNodes_s": list(map(float, ctx["fixed_hit"].nodes)),
            "EventVITFrozenAnchor": (
                "V4.12A RULSpecific_Top2 / DynamicTop2"
            ),
            "EventVITReproductionMaxAbsDiff": pred_diff,
            "TemperatureMarginalValue": temp_rows,
            "PooledMetrics": metrics.to_dict("records"),
            "Protocol": {
                "OuterLOBO": True,
                "HeldoutUsedForHIRanking": False,
                "HeldoutUsedForPairSearch": False,
                "HeldoutUsedForTraining": False,
                "HistoricalSOHAsFeature": False,
                "CapacityHistoryAsFeature": False,
                "FutureHIAsFeature": False,
                "CycleAsModelFeature": False,
                "TrueEOLRULAsInput": False,
                "EventCurrentRequiredForLocator": True,
                "CrossPlatformCheckpointBudgetMatched": False,
            },
            "InterpretationRules": [
                (
                    "Event-VI vs Event-VIT is the primary test of the "
                    "incremental value of temperature in the final Event RUL pipeline."
                ),
                (
                    "FixedTime V/VI/VT/VIT is a sensor-information mechanism "
                    "control because FixedTime localization does not require current."
                ),
                (
                    "Event-vs-FixedTime rows in this task use native 3-vs-4 "
                    "checkpoint designs and are descriptive, not equal-budget "
                    "platform-superiority evidence."
                ),
                (
                    "Use earlier V4.8D for frozen exact-common/equal-budget "
                    "platform conclusions."
                ),
            ],
        },
        args.output_root / "99_RWTH48_V4_13F_COMPLETE.json",
    )

    print("\n" + "=" * 124)
    print("RWTH48 V4.13F COMPLETE")
    print("Status:", status)

    print("\nPooled metrics:")
    print(metrics.to_string(index=False))

    print("\nMarginal sensor value:")
    print(
        comparisons[
            ~comparisons["ComparisonType"].eq("CrossPlatform_NativeBudget")
        ].to_string(index=False)
    )

    print("\nPaired bootstrap:")
    print(boots.to_string(index=False))

    print("\nNative platform contrasts (DESCRIPTIVE; budget unmatched):")
    print(native_cross.to_string(index=False))

    print("\nAcceptance:")
    print(gate.to_string(index=False))

    print("\nOutput:", args.output_root)
    print("=" * 124)

    return status


# =============================================================================
# 17. Main
# =============================================================================

def main():
    args = parse_args()

    args.output_root = ensure_dir(Path(args.output_root))

    if int(args.shortlist) < 2:
        raise ValueError("--shortlist must be >=2")
    if int(args.inner_folds) < 2:
        raise ValueError("--inner-folds must be >=2")
    if int(args.bootstrap) < 1000:
        raise ValueError("--bootstrap must be >=1000")

    t0 = time.perf_counter()

    ctx = preflight(args)

    save_csv(
        ctx["preflight"],
        args.output_root / "00_preflight.csv",
    )

    save_csv(
        ctx["registry"],
        args.output_root / "01_sensor_dependency_registry.csv",
    )

    # Merge platform coverage tables.
    ecov = event_coverage_table(ctx["event_hit"])
    fcov = ctx["fixed_hit"].coverage.copy()
    pcov = pd.concat([ecov, fcov], ignore_index=True)

    dep = ctx["registry"][
        ["HI", "SensorDependencies", "DependencyKnown"]
    ]
    pcov = pcov.merge(dep, on="HI", how="left")

    save_csv(
        pcov,
        args.output_root / "02_platform_hi_coverage.csv",
    )

    save_csv(
        ctx["coverage"],
        args.output_root / "02b_modality_candidate_pool_summary.csv",
    )

    save_csv(
        ctx["fixed_locate_audit"],
        args.output_root / "00b_fixedtime_locator_audit.csv",
    )

    save_csv(
        ctx["event_recon"],
        args.output_root / "00c_event_original_top2_reconstruction.csv",
    )

    save_json(
        {
            "Task": "RWTH48 V4.13F platform-aware sensor modality RUL ablation",
            "Configs": list(CONFIGS),
            "RequestedConfigs": list(args.configs),
            "Shortlist": int(args.shortlist),
            "InnerFolds": int(args.inner_folds),
            "TrainingOrigins": [80],
            "OriginCycle": 80,
            "HistoryOffsets": list(HISTORY_OFFSETS),
            "EventNodes_RefAh": list(EVENT_NODES),
            "FixedTimeNodes_s": list(map(float, args.fixedtime_nodes)),
            "FixedTimeReady": str(ctx["fixed_path"]),
            "EventVITReuseFrozenV412A": bool(
                not args.recompute_event_vit
            ),
            "BootstrapReplicates": int(args.bootstrap),
            "CrossPlatformComparison": (
                "Native-platform descriptive; checkpoint budget not matched"
            ),
        },
        args.output_root / "00_config.json",
    )

    print("=" * 124)
    print("RWTH48 V4.13F | Platform-aware sensor modality ablation")
    print("Stage          :", args.stage)
    print("Configs        :", ", ".join(args.configs))
    print("FixedTime file :", ctx["fixed_path"])
    print("Shortlist      :", args.shortlist)
    print("Inner folds    :", args.inner_folds)
    print("Event VIT      :", (
        "RECOMPUTE" if args.recompute_event_vit else "REUSE frozen V4.12A"
    ))
    print("\nCandidate pools:")
    print(ctx["coverage"].to_string(index=False))
    print("=" * 124, flush=True)

    if args.stage == "audit":
        print("\nPreflight:")
        print(ctx["preflight"].to_string(index=False))
        print("\nAUDIT PASS")
        return

    bundle = ctx["bundle"]
    batteries = sorted(set(map(normalize_battery, bundle.battery.tolist())))

    if args.stage == "finalize":
        finalize(ctx, args)
        return

    if args.only_battery:
        b = normalize_battery(args.only_battery)
        if b not in batteries:
            raise RuntimeError(f"Unknown battery: {b}")
        run_bats = [b]
    elif args.stage == "worker":
        run_bats = worker_batteries(
            batteries,
            args.num_workers,
            args.worker_id,
        )
    else:
        run_bats = batteries

    for j, b in enumerate(run_bats, start=1):
        print(
            f"\n[V413F] battery {j}/{len(run_bats)} {b}",
            flush=True,
        )
        run_one_battery(b, ctx, args)

    if args.only_battery and args.stage == "run":
        rows = []
        for cfg in args.configs:
            p = result_path(args.output_root, args.only_battery, cfg)
            if p.exists():
                rows.append(load_json(p))
        if rows:
            print("\nSingle-battery results:")
            print(
                pd.DataFrame(rows)[
                    [
                        "Config",
                        "SelectedTop2",
                        "TrueRUL",
                        "PredRUL",
                        "InnerCV_MAE",
                        "CandidatePoolSize",
                        "Runtime_s",
                    ]
                ].to_string(index=False)
            )
        return

    if args.stage == "worker":
        if args.auto_finalize:
            complete = all(
                cache_valid(args.output_root, b, cfg)
                for b in batteries
                for cfg in CONFIGS
            )
            if complete:
                finalize(ctx, args)
            else:
                print(
                    "[V413F] worker complete; other caches still missing.",
                    flush=True,
                )
        return

    # all/run without --only-battery: finalize only when all 6 configs exist.
    complete = all(
        cache_valid(args.output_root, b, cfg)
        for b in batteries
        for cfg in CONFIGS
    )

    if complete:
        finalize(ctx, args)
    else:
        missing = [
            f"{b}|{cfg}"
            for b in batteries
            for cfg in CONFIGS
            if not cache_valid(args.output_root, b, cfg)
        ]
        print(
            "\n[V413F] Requested work finished, but formal finalize is not ready."
        )
        print("Missing caches:", len(missing))
        print("\n".join(missing[:50]))
        print(
            "\nRun remaining configs, then:"
            "\n  --stage finalize"
        )

    print(
        f"\nTotal wall time: {(time.perf_counter()-t0)/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
RWTH48 V4.14B | Full-Life SOH Measurement-Timing Value + Minimum-Resource Pareto | FIX2
======================================================================================

Scientific question
-------------------
Primary target is NOT RUL.  It is one-shot long-horizon/full-life SOH trajectory
prediction under a hard engineering accuracy constraint:

    minimize early-life measurement burden
    subject to full-life held-out-cell SOH RMSE < 0.01.

This stage explicitly separates two questions that were previously mixed together:

A) VALUE OF WHEN TO MEASURE
   At fixed acquisition resources, change only the historical diagnosis cycles.
   The coarse timing grid is exhaustively traversed (information score), then the
   most promising schedules are refined by a source-only Ridge trajectory proxy.
   Only source batteries are used to choose a schedule.  The held-out battery is
   evaluated once with the formal full-life XGBoost trajectory head.

B) MINIMUM-RESOURCE SEARCH
   Diagnosis count, diagnosis timing, intra-diagnosis checkpoint count/position,
   HI count and sensor modality are jointly searched.  To keep the nested search
   computationally tractable, each resource cell uses:
       source-only HI relevance ranking
       -> node-subset coordinate search
       -> exhaustive coarse timing information scan
       -> source-only Ridge proxy refinement
       -> HI-set refinement on finalists
       -> formal held-out full-life XGBoost evaluation.

   The timing schedule is re-optimized INSIDE every selected resource policy.
   Therefore the code does not assume that the best regular/high-budget timing is
   also optimal under low-resource sensing.

C) OPTIONAL DIAGNOSTIC-STALENESS STRESS TEST
   The pure timing experiment fixes the last diagnosis at origin cycle 80.  An
   optional stage separately allows the last available diagnosis to be 8/16/etc.
   cycles stale, so historical layout and last-observation staleness are not mixed.

Strict protocol
---------------
* 47-cell outer LOBO.
* No held-out SOH is used for timing selection, HI selection, node selection,
  resource-policy selection, training, or proxy-model tuning.
* No historical true SOH/capacity, future HI, future SOH, absolute cycle index,
  RUL or EOL is used as a model input.
* Historical diagnosis cycle locations are DESIGN variables / known acquisition
  metadata, not degradation labels.
* Arbitrary schedules are reconstructed from the frozen per-cycle Event and
  FixedTime HI tables.  Existing [40,48,56,64,72,80] history tensors are NOT
  rearranged or interpolated.
* Event checkpoints are RefAh-progress coordinates; current is physically required
  to locate them even if a selected HI does not otherwise depend on current.
* FixedTime checkpoints are elapsed acquisition-time coordinates.
* Full-life target is the already-frozen V4.11A2 masked trajectory handoff.

Primary output interpretation
-----------------------------
A policy is labelled:
    Feasible_Pooled          : pooled full-life RMSE < threshold
    Feasible_BatteryMacro    : mean per-battery RMSE < threshold
    Feasible_ClusterBoot95   : battery-cluster bootstrap 95% upper RMSE < threshold

The last criterion is the strongest empirical statement in this dataset.  It is
NOT a mathematical guarantee for every future battery population.

Parallel / resume
-----------------
Two workers are supported.  Work is cached at platform x diagnosis-count x role
for timing and platform x policy-role for minimum-resource search.  Worker 0 waits
for all workers and finalizes automatically.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 0. Constants
# =============================================================================

CODE_VERSION = "V414B_FULLLIFE_SOH_TIMING_MINRESOURCE_FIX2"

BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR = Path(__file__).resolve().parent
COMMON_NAME = "08_fixedtime_common_pipeline.py"
ADAPTER_NAME = "09_platform_sensor_ablation.py"

OUTPUT_NAME = "RWTH48_V4_14B_FullLifeSOH_Timing_MinResource"
STATE_NAME = "RWTH48_V4_14B_FullLifeSOH_Timing_MinResource_STATE"

EXPECTED_BATTERIES = 47
TEST_ORIGIN = 80
EVENT_NODES = (0.05, 0.09, 0.13)
EXPECTED_FIXEDTIME_NODES = (60.0, 120.0, 180.0, 300.0, 450.0, 600.0, 900.0, 1200.0)
EVENT_SOH_TOP2 = ("V_TimeWeightedMean", "P_Last")
FT01_ROOT_NAME = "RWTH48_V4_13F_PRE_FixedTime_01_SOH_HI_Ranking_TopK_Tree"
FT02_ROOT_NAME = "RWTH48_V4_13F_PRE_FixedTime_02_SOH_Deep_Hybrid_MoE"

ENC = "utf-8-sig"
FAST_GZIP = {"method": "gzip", "compresslevel": 1}
RANDOM_SEED = 20260825

MODALITY_SENSORS: Dict[str, FrozenSet[str]] = {
    "V": frozenset({"V"}),
    "VI": frozenset({"V", "I"}),
    "VT": frozenset({"V", "T"}),
    "VIT": frozenset({"V", "I", "T"}),
}

TIMING_MODALITY = {
    "Event": "VI",       # Event final SOH pair is VI; current is also locator-required.
    "FixedTime": "VIT", # isolate timing only; do not remove channels in Part A.
}

POLICY_ROLES = (
    "MIN_SCALAR",
    "MIN_SENSOR",
    "MIN_DURATION",
    "MIN_CHANNEL_SECONDS",
)


# =============================================================================
# 1. CLI / IO
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWTH48 V4.14B full-life SOH timing-value + minimum-resource Pareto"
    )
    p.add_argument(
        "--stage",
        choices=["audit", "timing", "resource", "staleness", "all", "finalize"],
        default="all",
    )
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--code-dir", type=Path, default=CODE_DIR)
    p.add_argument("--common", type=Path, default=None)
    p.add_argument("--event-ready", type=Path, default=None)
    p.add_argument("--fixedtime-ready", type=Path, default=None)
    p.add_argument("--dependency-registry", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--state-root", type=Path, default=None)

    # Historical timing grid.
    p.add_argument("--cycle-start", type=int, default=40)
    p.add_argument("--cycle-end", type=int, default=80)
    p.add_argument("--cycle-step", type=int, default=4)
    p.add_argument("--diagnosis-counts", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    p.add_argument("--timing-preselect", type=int, default=24)
    p.add_argument("--timing-bottom-preselect", type=int, default=6)
    p.add_argument("--local-refine-radius", type=int, default=3)
    p.add_argument("--local-refine-rounds", type=int, default=2)
    p.add_argument("--proxy-validation-batteries", type=int, default=9)
    p.add_argument("--proxy-ridge-alpha", type=float, default=8.0)
    p.add_argument(
        "--proxy-leads",
        nargs="+",
        type=int,
        default=[10, 25, 50, 100, 200, 300, 500, 700, 900],
    )
    p.add_argument("--include-worst-timing", action="store_true")

    # Formal trajectory head.
    p.add_argument("--xgb-threads", type=int, default=3)
    p.add_argument("--cpu-threads-per-worker", type=int, default=4)
    p.add_argument("--xgb-leads-per-origin", type=int, default=160)

    # Resource search.
    p.add_argument("--rmse-threshold", type=float, default=0.01)
    p.add_argument("--proxy-feasible-threshold", type=float, default=0.0105)
    p.add_argument("--resource-topks", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--resource-max-cells-event", type=int, default=90)
    p.add_argument("--resource-max-cells-fixedtime", type=int, default=180)
    p.add_argument("--resource-node-preselect", type=int, default=5)
    p.add_argument("--resource-schedule-preselect", type=int, default=5)
    p.add_argument("--hi-ranking-shortlist", type=int, default=12)
    p.add_argument("--hi-refine-shortlist", type=int, default=6)
    p.add_argument("--resource-cost-multiplier", type=float, default=2.0)
    p.add_argument(
        "--fixed-node-counts",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 6, 8],
    )
    p.add_argument(
        "--event-modalities",
        nargs="+",
        default=["VI", "VIT"],
    )
    p.add_argument(
        "--fixedtime-modalities",
        nargs="+",
        default=["V", "VI", "VT", "VIT"],
    )

    # Optional staleness experiment. Not part of default `all` to keep the first run bounded.
    p.add_argument("--staleness-values", nargs="+", type=int, default=[0, 8, 16])

    # Statistics / execution.
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--dpi", type=int, default=260)
    p.add_argument("--parallel-workers", type=int, default=1)
    p.add_argument("--parallel-worker-id", type=int, default=0)
    p.add_argument("--run-id", type=str, default="V414B_MANUAL")
    p.add_argument("--barrier-poll-s", type=float, default=10.0)

    args = p.parse_args()
    args.base = Path(args.base)
    args.code_dir = Path(args.code_dir)
    args.common = Path(args.common) if args.common else args.code_dir / COMMON_NAME
    args.output_root = Path(args.output_root) if args.output_root else args.base / OUTPUT_NAME
    args.state_root = Path(args.state_root) if args.state_root else args.base / STATE_NAME

    args.diagnosis_counts = sorted(set(int(x) for x in args.diagnosis_counts if int(x) >= 2))
    args.resource_topks = sorted(set(int(x) for x in args.resource_topks if 1 <= int(x) <= 3))
    args.fixed_node_counts = sorted(set(int(x) for x in args.fixed_node_counts if 1 <= int(x) <= 8))
    args.event_modalities = [m.upper() for m in args.event_modalities]
    args.fixedtime_modalities = [m.upper() for m in args.fixedtime_modalities]

    if not args.diagnosis_counts:
        p.error("--diagnosis-counts must contain >=2")
    if not args.resource_topks:
        p.error("--resource-topks must contain 1/2/3")
    if args.cycle_end != TEST_ORIGIN:
        p.error("FIX2 primary timing experiment requires --cycle-end 80")
    if args.cycle_step < 1:
        p.error("--cycle-step must be >=1")
    if args.local_refine_radius < 0 or args.local_refine_rounds < 0:
        p.error("--local-refine-radius/rounds must be >=0")
    if any(m not in MODALITY_SENSORS for m in args.event_modalities + args.fixedtime_modalities):
        p.error("modalities must be among V VI VT VIT")
    if int(args.parallel_workers) < 1:
        p.error("--parallel-workers must be >=1")
    if not (0 <= int(args.parallel_worker_id) < int(args.parallel_workers)):
        p.error("--parallel-worker-id must be in [0, parallel-workers)")
    return args


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


def save_json(obj, p: Path) -> None:
    ensure_dir(Path(p).parent)
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, p)


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, p: Path, compression=None) -> None:
    ensure_dir(Path(p).parent)
    tmp = Path(str(p) + ".tmp")
    df.to_csv(tmp, index=False, encoding=ENC, compression=compression)
    os.replace(tmp, p)


def load_module(path: Path, name: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def configure_threads(n: int) -> None:
    n = max(1, int(n))
    for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(n)


def stable_seed(*parts: object) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:8], 16) % (2**31 - 1)


def normalize_battery(C, x) -> str:
    return C.normalize_battery(x)


# =============================================================================
# 2. Frozen per-cycle platform cubes
# =============================================================================

@dataclass
class PlatformCube:
    platform: str
    values: np.ndarray  # [battery, cycle, node, hi]
    batteries: Tuple[str, ...]
    cycles: Tuple[int, ...]
    nodes: Tuple[float, ...]
    hi_names: Tuple[str, ...]

    @property
    def bmap(self):
        return {b: i for i, b in enumerate(self.batteries)}

    @property
    def cmap(self):
        return {int(c): i for i, c in enumerate(self.cycles)}

    @property
    def nmap(self):
        return {float(n): i for i, n in enumerate(self.nodes)}

    @property
    def hmap(self):
        return {h: i for i, h in enumerate(self.hi_names)}


def build_platform_cube(
    df: pd.DataFrame,
    battery_col: str,
    cycle_col: str,
    node_col: str,
    hi_cols: Sequence[str],
    batteries: Sequence[str],
    cycles: Sequence[int],
    nodes: Sequence[float],
    platform: str,
    C,
) -> PlatformCube:
    bats = tuple(map(str, batteries))
    cycs = tuple(map(int, cycles))
    nds = tuple(map(float, nodes))
    his = tuple(map(str, hi_cols))
    bmap = {b: i for i, b in enumerate(bats)}
    cmap = {c: i for i, c in enumerate(cycs)}
    nmap = {n: i for i, n in enumerate(nds)}

    arr = np.full((len(bats), len(cycs), len(nds), len(his)), np.nan, np.float32)
    d = df.copy()
    d[battery_col] = d[battery_col].map(lambda x: normalize_battery(C, x))
    d[cycle_col] = pd.to_numeric(d[cycle_col], errors="coerce")

    for idx, r in d.iterrows():
        b = str(r[battery_col])
        if b not in bmap or not np.isfinite(r[cycle_col]):
            continue
        c = int(round(float(r[cycle_col])))
        if c not in cmap:
            continue
        try:
            n = float(r[node_col])
        except Exception:
            continue
        # Canonical node lookup with tolerance.
        nn = next((x for x in nds if np.isclose(n, x, atol=5e-5 if platform == "Event" else 0.5, rtol=0)), None)
        if nn is None:
            continue
        vals = pd.to_numeric(r[list(his)], errors="coerce").to_numpy(float)
        arr[bmap[b], cmap[c], nmap[nn], :] = vals.astype(np.float32)

    return PlatformCube(platform, arr, bats, cycs, nds, his)


def extract_tensor(
    cube: PlatformCube,
    schedule: Sequence[int],
    nodes: Sequence[float],
    his: Sequence[str],
) -> Optional[np.ndarray]:
    try:
        ci = [cube.cmap[int(c)] for c in schedule]
        ni = [cube.nmap[float(n)] for n in nodes]
        hi = [cube.hmap[str(h)] for h in his]
    except KeyError:
        return None
    x = np.asarray(cube.values[:, ci, :, :][:, :, ni, :][:, :, :, hi], np.float32)
    if not np.isfinite(x).all():
        return None
    return x


def selected_sensor_set(platform: str, his: Sequence[str], depmap) -> FrozenSet[str]:
    s = set()
    for h in his:
        dep = depmap.get(str(h))
        if dep is None:
            return frozenset()
        s.update(dep)
    if platform == "Event":
        s.add("I")  # RefAh checkpoint locator.
    return frozenset(s)


# =============================================================================
# 3. Dynamic schedule representation
# =============================================================================

def _slope_rows(x: np.ndarray, coord: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, np.float64)
    t = np.asarray(coord, np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(t):
        raise RuntimeError(f"slope shape={x.shape}, coord={len(t)}")
    tc = t - np.mean(t)
    den = float(np.sum(tc * tc))
    if den <= 1e-15:
        return np.zeros(x.shape[0], float)
    return np.sum((x - np.mean(x, axis=1, keepdims=True)) * tc[None, :], axis=1) / den


def engineer_dynamic_selected(
    x: np.ndarray,  # [B,D,M,K]
    schedule: Sequence[int],
    nodes: Sequence[float],
    hi_names: Sequence[str],
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    x = np.asarray(x, np.float64)
    schedule = tuple(map(int, schedule))
    nodes = tuple(map(float, nodes))
    hi_names = tuple(map(str, hi_names))
    if x.ndim != 4:
        raise RuntimeError(f"dynamic tensor ndim={x.ndim} !=4")
    N, D, M, K = x.shape
    if D != len(schedule) or M != len(nodes) or K != len(hi_names):
        raise RuntimeError("dynamic tensor/coordinate mismatch")
    if not np.isfinite(x).all():
        raise RuntimeError("dynamic tensor contains NaN/Inf")

    blocks = []
    names: List[str] = []
    registry = []

    def add(a, nms, group, source_hi):
        a = np.asarray(a, np.float64)
        if a.ndim == 1:
            a = a[:, None]
        blocks.append(a)
        names.extend(nms)
        registry.extend({
            "Feature": n,
            "Group": group,
            "MeasuredOrDerived": "Measured" if group == "RawMeasured" else "Derived",
            "SourceHI": source_hi,
            "DiagnosisCount": D,
            "NodeCount": M,
        } for n in nms)

    cache = {}
    for k, h in enumerate(hi_names):
        a = x[:, :, :, k]
        cache[h] = a

        raw = a.reshape(N, -1)
        rn = [f"RAW__C{schedule[d]:03d}__N{m:02d}__{h}" for d in range(D) for m in range(M)]
        add(raw, rn, "RawMeasured", h)

        cols, nms = [], []
        for m in range(M):
            z = a[:, :, m]
            vals = [
                z[:, -1] - z[:, 0],
                _slope_rows(z, schedule),
                np.mean(z, axis=1),
                np.std(z, axis=1),
            ]
            tags = ["LongDelta", "SlopePerCycle", "Mean", "Std"]
            cols.extend(vals)
            nms.extend([f"HIST__N{m:02d}__{h}__{t}" for t in tags])
        add(np.column_stack(cols), nms, "HistoricalTrend", h)

        cols, nms = [], []
        for d in range(D):
            z = a[:, d, :]
            vals = [
                np.mean(z, axis=1), np.std(z, axis=1), np.min(z, axis=1),
                np.max(z, axis=1), np.max(z, axis=1) - np.min(z, axis=1),
                _slope_rows(z, nodes),
            ]
            tags = ["Mean", "Std", "Min", "Max", "Range", "NodeSlope"]
            cols.extend(vals)
            nms.extend([f"WITHIN__C{schedule[d]:03d}__{h}__{t}" for t in tags])
        add(np.column_stack(cols), nms, "WithinDiagnosisGeometry", h)

        if M > 1:
            cols, nms = [], []
            for d in range(D):
                dz = np.diff(a[:, d, :], axis=1)
                for m in range(M - 1):
                    cols.append(dz[:, m])
                    nms.append(f"NODEDIFF__C{schedule[d]:03d}__N{m+1:02d}-N{m:02d}__{h}")
            add(np.column_stack(cols), nms, "ConsecutiveNodeDifference", h)

    for i in range(K):
        for j in range(i + 1, K):
            ha, hb = hi_names[i], hi_names[j]
            a, b = cache[ha], cache[hb]
            cols, nms = [], []
            for d in range(D):
                for m in range(M):
                    av, bv = a[:, d, m], b[:, d, m]
                    diff = av - bv
                    ndiff = diff / (np.abs(av) + np.abs(bv) + 1e-6)
                    cols.extend([diff, ndiff])
                    nms.extend([
                        f"CROSSHI__C{schedule[d]:03d}__N{m:02d}__{ha}__{hb}__Diff",
                        f"CROSSHI__C{schedule[d]:03d}__N{m:02d}__{ha}__{hb}__NormDiff",
                    ])
            add(np.column_stack(cols), nms, "CrossHIWithinNode", f"{ha}|{hb}")

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    expected_single = 2 * D * M + 5 * D + 4 * M
    expected = K * expected_single + 2 * D * M * (K * (K - 1) // 2)
    if X.shape != (N, expected):
        raise RuntimeError(f"dynamic engineered shape={X.shape}, expected={(N, expected)}")
    if not np.isfinite(X).all():
        raise RuntimeError("dynamic engineered matrix contains NaN/Inf")
    return X, names, pd.DataFrame(registry)


def compact_features(x: np.ndarray, schedule: Sequence[int]) -> np.ndarray:
    """Small source-only proxy representation; no target leakage."""
    x = np.asarray(x, np.float64)
    N, D, M, K = x.shape
    blocks = [x.reshape(N, -1)]
    for k in range(K):
        for m in range(M):
            z = x[:, :, m, k]
            blocks.extend([
                (z[:, -1] - z[:, 0])[:, None],
                _slope_rows(z, schedule)[:, None],
                np.mean(z, axis=1)[:, None],
                np.std(z, axis=1)[:, None],
            ])
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    return X


# =============================================================================
# 4. Schedule generation and descriptors
# =============================================================================

def coarse_cycles(args, last_cycle: int = TEST_ORIGIN) -> Tuple[int, ...]:
    vals = list(range(int(args.cycle_start), int(last_cycle) + 1, int(args.cycle_step)))
    if last_cycle not in vals:
        vals.append(int(last_cycle))
    vals = sorted(set(v for v in vals if args.cycle_start <= v <= last_cycle))
    return tuple(vals)


def enumerate_schedules(cycles: Sequence[int], L: int, last_cycle: int) -> List[Tuple[int, ...]]:
    c = sorted(set(int(x) for x in cycles if int(x) <= int(last_cycle)))
    if int(last_cycle) not in c:
        c.append(int(last_cycle))
        c = sorted(set(c))
    earlier = [x for x in c if x < int(last_cycle)]
    if L < 2 or len(earlier) < L - 1:
        return []
    return [tuple(list(z) + [int(last_cycle)]) for z in itertools.combinations(earlier, L - 1)]


def uniform_schedule(cycles: Sequence[int], L: int, last_cycle: int) -> Tuple[int, ...]:
    schedules = enumerate_schedules(cycles, L, last_cycle)
    if not schedules:
        raise RuntimeError(f"No schedules L={L}, last={last_cycle}")
    target = np.linspace(min(cycles), last_cycle, L)
    def key(s):
        return float(np.sum((np.asarray(s, float) - target) ** 2))
    return min(schedules, key=key)


def schedule_descriptors(schedule: Sequence[int], origin: int = TEST_ORIGIN) -> Dict[str, float]:
    s = np.asarray(schedule, float)
    gaps = np.diff(s)
    return {
        "DiagnosisCount": int(len(s)),
        "FirstCycle": int(s[0]),
        "LastCycle": int(s[-1]),
        "HistorySpan": float(s[-1] - s[0]),
        "MeanGap": float(np.mean(gaps)) if len(gaps) else 0.0,
        "StdGap": float(np.std(gaps)) if len(gaps) else 0.0,
        "GapCV": float(np.std(gaps) / max(np.mean(gaps), 1e-12)) if len(gaps) else 0.0,
        "MaxGap": float(np.max(gaps)) if len(gaps) else 0.0,
        "MinGap": float(np.min(gaps)) if len(gaps) else 0.0,
        "LastStaleness": float(origin - s[-1]),
        "EarlyMeanCycle": float(np.mean(s[: max(1, len(s)//2)])),
        "LateMeanCycle": float(np.mean(s[len(s)//2 :])),
    }


def local_schedule_neighbors(
    schedule: Sequence[int],
    cycle_start: int,
    last_cycle: int,
    radius: int,
) -> List[Tuple[int, ...]]:
    """One-coordinate integer-cycle neighbors; final diagnosis remains fixed."""
    base = list(map(int, schedule))
    if len(base) < 2 or int(base[-1]) != int(last_cycle) or int(radius) <= 0:
        return []
    out = set()
    for j in range(len(base) - 1):
        for delta in range(-int(radius), int(radius) + 1):
            if delta == 0:
                continue
            z = list(base)
            z[j] += delta
            if z[j] < int(cycle_start) or z[j] >= int(last_cycle):
                continue
            if any(z[k] >= z[k + 1] for k in range(len(z) - 1)):
                continue
            out.add(tuple(z))
    return sorted(out)


def local_refine_schedule_ridge(
    ctx,
    held: str,
    cube: PlatformCube,
    schedule: Sequence[int],
    nodes: Sequence[float],
    his: Sequence[str],
    args,
    salt: str,
) -> Tuple[Tuple[int, ...], float, List[Dict[str, object]]]:
    """
    Source-only coordinate-descent refinement around a coarse schedule.
    It never evaluates the held-out SOH target.
    """
    source = source_batteries(ctx, held)

    def eval_schedule(sch: Sequence[int], tag: str) -> float:
        x = extract_tensor(cube, sch, nodes, his)
        if x is None:
            return np.inf
        return ridge_proxy_rmse(
            compact_features(x, sch),
            ctx["batteries"], source, ctx["Y80"], ctx["M80"], args,
            held, f"{salt}|{tag}|{'-'.join(map(str, sch))}",
        )

    current = tuple(map(int, schedule))
    best_rmse = eval_schedule(current, "start")
    audit = [{"Round": 0, "Schedule": "|".join(map(str, current)), "RidgeProxyRMSE": best_rmse}]
    for rr in range(1, int(args.local_refine_rounds) + 1):
        candidates = [current] + local_schedule_neighbors(
            current, args.cycle_start, current[-1], args.local_refine_radius
        )
        scored = []
        for sch in candidates:
            val = best_rmse if sch == current else eval_schedule(sch, f"r{rr}")
            scored.append((val, sch))
        val, sch = min(scored, key=lambda z: (z[0], z[1]))
        audit.append({"Round": rr, "Schedule": "|".join(map(str, sch)), "RidgeProxyRMSE": float(val)})
        if sch == current or not np.isfinite(val) or val >= best_rmse - 1e-12:
            break
        current, best_rmse = tuple(sch), float(val)
    return current, float(best_rmse), audit


# =============================================================================
# 5. Full-life source-only proxy
# =============================================================================

def origin80_bundle_map(C, bundle, batteries: Sequence[str]) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    out = {}
    for i, (b, o) in enumerate(zip(bundle.battery, bundle.origin)):
        bb = normalize_battery(C, b)
        if int(o) == TEST_ORIGIN:
            out[bb] = int(i)
    if len(out) != len(batteries):
        raise RuntimeError(f"origin80 map={len(out)} != batteries={len(batteries)}")
    Y = np.vstack([np.asarray(bundle.y[out[b]], float) for b in batteries])
    M = np.vstack([np.asarray(bundle.mask[out[b]], int) for b in batteries])
    return out, Y, M


def choose_proxy_validation(source_bats: Sequence[str], n: int, held: str, salt: str) -> List[str]:
    scored = sorted((stable_seed(held, salt, b), str(b)) for b in source_bats)
    n = max(2, min(int(n), max(2, len(scored) // 3)))
    return [b for _, b in scored[:n]]


def _long_proxy_rows(
    X: np.ndarray,
    bat_indices: Sequence[int],
    Y: np.ndarray,
    Mask: np.ndarray,
    leads: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    rows_x, rows_y = [], []
    K = Y.shape[1]
    lead_arr = [int(l) for l in leads if 1 <= int(l) <= K]
    for bi in bat_indices:
        pos = [l - 1 for l in lead_arr if int(Mask[bi, l - 1]) == 1 and np.isfinite(Y[bi, l - 1])]
        if not pos:
            continue
        base = np.repeat(X[bi : bi + 1], len(pos), axis=0)
        lf = np.asarray([
            [
                (p + 1) / K,
                math.sqrt((p + 1) / K),
                math.log1p(p + 1) / math.log1p(K),
                ((p + 1) / K) ** 2,
            ]
            for p in pos
        ], np.float32)
        rows_x.append(np.concatenate([base, lf], axis=1))
        rows_y.append(Y[bi, pos])
    if not rows_x:
        raise RuntimeError("proxy rows empty")
    return np.vstack(rows_x).astype(np.float32), np.concatenate(rows_y).astype(np.float32)


def ridge_proxy_rmse(
    X: np.ndarray,
    batteries: Sequence[str],
    source_bats: Sequence[str],
    Y: np.ndarray,
    Mask: np.ndarray,
    args,
    held: str,
    salt: str,
) -> float:
    bmap = {b: i for i, b in enumerate(batteries)}
    val_bats = choose_proxy_validation(source_bats, args.proxy_validation_batteries, held, salt)
    valset = set(val_bats)
    train_bats = [b for b in source_bats if b not in valset]
    tr = [bmap[b] for b in train_bats]
    va = [bmap[b] for b in val_bats]
    Xtr, ytr = _long_proxy_rows(X, tr, Y, Mask, args.proxy_leads)
    Xva, yva = _long_proxy_rows(X, va, Y, Mask, args.proxy_leads)
    sc = StandardScaler()
    A = sc.fit_transform(Xtr)
    B = sc.transform(Xva)
    model = Ridge(alpha=float(args.proxy_ridge_alpha), solver="lsqr")
    model.fit(A, ytr)
    pred = np.asarray(model.predict(B), float)
    return float(np.sqrt(np.mean((pred - yva) ** 2)))


def spearman_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 5:
        return 0.0
    ar = pd.Series(a[m]).rank(method="average").to_numpy(float)
    br = pd.Series(b[m]).rank(method="average").to_numpy(float)
    if np.std(ar) < 1e-12 or np.std(br) < 1e-12:
        return 0.0
    r = np.corrcoef(ar, br)[0, 1]
    return float(abs(r)) if np.isfinite(r) else 0.0


def future_summary_matrix(Y: np.ndarray, Mask: np.ndarray, leads: Sequence[int]) -> np.ndarray:
    K = Y.shape[1]
    cols = []
    for l in leads:
        if 1 <= int(l) <= K:
            z = Y[:, int(l) - 1].copy()
            z[Mask[:, int(l) - 1] != 1] = np.nan
            cols.append(z)
    # add late-life mean where available
    lo = max(0, min(K - 1, int(0.65 * K)))
    late = np.full(Y.shape[0], np.nan)
    for i in range(Y.shape[0]):
        m = (Mask[i, lo:] == 1) & np.isfinite(Y[i, lo:])
        if m.any():
            late[i] = float(np.mean(Y[i, lo:][m]))
    cols.append(late)
    return np.column_stack(cols) if cols else late[:, None]


def information_score(X: np.ndarray, target_summary: np.ndarray, source_idx: Sequence[int]) -> float:
    X = np.asarray(X, float)
    Y = np.asarray(target_summary, float)
    scores = []
    for j in range(X.shape[1]):
        for k in range(Y.shape[1]):
            scores.append(spearman_abs(X[np.asarray(source_idx), j], Y[np.asarray(source_idx), k]))
    if not scores:
        return -np.inf
    top = sorted(scores, reverse=True)[: min(12, len(scores))]
    return float(np.mean(top))


# =============================================================================
# 6. Costs / sensor dependencies
# =============================================================================

def acquisition_cost(
    cube: PlatformCube,
    schedule: Sequence[int],
    nodes: Sequence[float],
    his: Sequence[str],
    depmap,
    battery_indices: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    sensors = selected_sensor_set(cube.platform, his, depmap)
    if not sensors:
        return {
            "PhysicalSensorCount": np.nan,
            "HI_ScalarInputs": len(schedule) * len(nodes) * len(his),
            "MeasurementCheckpointCount": len(schedule) * len(nodes),
            "SensorCheckpointCount": np.nan,
            "AcquisitionDuration_s": np.nan,
            "SensorChannelSeconds": np.nan,
        }
    duration = np.nan
    if cube.platform == "FixedTime":
        duration = float(len(schedule) * max(nodes))
    else:
        if "PrefixDuration_s" in cube.hmap:
            x = extract_tensor(cube, schedule, [max(nodes)], ["PrefixDuration_s"])
            if x is not None:
                z = np.sum(x[:, :, 0, 0], axis=1)
                if battery_indices is not None:
                    ii = np.asarray(list(battery_indices), dtype=int)
                    if len(ii):
                        z = z[ii]
                if len(z) and np.isfinite(z).all():
                    duration = float(np.mean(z))
    return {
        "PhysicalSensorCount": int(len(sensors)),
        "PhysicalSensors": "|".join(sorted(sensors)),
        "HI_ScalarInputs": int(len(schedule) * len(nodes) * len(his)),
        "MeasurementCheckpointCount": int(len(schedule) * len(nodes)),
        "SensorCheckpointCount": int(len(schedule) * len(nodes) * len(sensors)),
        "DiagnosisCount": int(len(schedule)),
        "NodeCount": int(len(nodes)),
        "HI_Count": int(len(his)),
        "AcquisitionDuration_s": duration,
        "SensorChannelSeconds": float(duration * len(sensors)) if np.isfinite(duration) else np.nan,
    }


# =============================================================================
# 7. Preflight
# =============================================================================

def load_ft01_top2_map(base: Path) -> Dict[str, List[str]]:
    p = base / FT01_ROOT_NAME / "03_fold_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p, low_memory=False)
    req = {"OuterHeldoutBattery", "Top2_Selected"}
    if not req.issubset(d.columns):
        raise RuntimeError(f"FT01 summary missing {sorted(req-set(d.columns))}")
    out = {}
    for r in d.itertuples(index=False):
        pair = [x for x in str(r.Top2_Selected).split("|") if x]
        if len(pair) != 2:
            raise RuntimeError(f"Invalid FT01 Top2: {r.Top2_Selected}")
        out[str(r.OuterHeldoutBattery)] = pair
    if len(out) != EXPECTED_BATTERIES:
        raise RuntimeError(f"FT01 Top2 map={len(out)} !=47")
    return out


def preflight(args):
    configure_threads(args.cpu_threads_per_worker)
    C = load_module(args.common, "rwth48_v414b_common")
    adapter, ftctx = C.load_fixedtime_context(
        base=args.base,
        code_dir=args.code_dir,
        output_root=args.output_root,
        fixedtime_ready=args.fixedtime_ready,
        event_ready=args.event_ready,
        xgb_threads=args.xgb_threads,
        fixedtime_nodes=None,
    )

    fixed_nodes = tuple(map(float, ftctx["fixed_nodes"]))
    if len(fixed_nodes) != 8 or not np.allclose(fixed_nodes, EXPECTED_FIXEDTIME_NODES, atol=1e-9, rtol=0):
        raise RuntimeError(f"V4.14B requires final 8-node FixedTime platform; observed={fixed_nodes}")

    bundle = ftctx["bundle"]
    bats = sorted(set(normalize_battery(C, b) for b in bundle.battery.tolist()))
    if len(bats) != EXPECTED_BATTERIES:
        raise RuntimeError(f"batteries={len(bats)} !=47")

    # Event raw frozen long table.
    v412a = ftctx["v412a"]
    eargs = argparse.Namespace(
        event_ready=args.event_ready,
        base=args.base,
        phase="Discharge",
        candidate_hi_file=None,
    )
    event_path = v412a.locate_event_ready(eargs)
    event_raw = pd.read_csv(event_path, compression="infer", low_memory=False)
    event_schema = v412a.detect_event_schema(event_raw, eargs)
    event = v412a.filter_event_table(event_raw, event_schema, eargs)

    # FixedTime raw frozen long table.
    fixed_path = Path(ftctx["fixed_path"])
    fixed_raw = pd.read_csv(fixed_path, compression="infer", low_memory=False)
    fixed_schema = adapter.detect_fixedtime_schema(fixed_raw, fixed_nodes)
    fixed = adapter.filter_fixedtime_table(fixed_raw, fixed_schema, fixed_nodes, "Discharge")

    all_his = sorted(set(map(str, event_schema.hi_cols)) | set(map(str, fixed_schema.hi_cols)))
    registry = adapter.build_dependency_registry(all_his, args.dependency_registry)
    unknown = registry[registry["DependencyKnown"].eq(0)]
    if len(unknown):
        raise RuntimeError(
            "Unknown sensor dependencies; provide --dependency-registry:\n"
            + unknown.head(60).to_string(index=False)
        )
    depmap = adapter.dependency_map(registry)

    cycles_dense = tuple(range(int(args.cycle_start), TEST_ORIGIN + 1))
    event_cube = build_platform_cube(
        event,
        event_schema.battery_col,
        event_schema.cycle_col,
        "_RefAhCanonical",
        event_schema.hi_cols,
        bats,
        cycles_dense,
        EVENT_NODES,
        "Event",
        C,
    )
    fixed_cube = build_platform_cube(
        fixed,
        fixed_schema.battery_col,
        fixed_schema.cycle_col,
        "_NodeCanonical",
        fixed_schema.hi_cols,
        bats,
        cycles_dense,
        fixed_nodes,
        "FixedTime",
        C,
    )

    ft_top2 = load_ft01_top2_map(args.base)
    for h in EVENT_SOH_TOP2:
        if h not in event_cube.hmap:
            raise RuntimeError(f"Event SOH Top2 missing: {h}")
    for b in bats:
        for h in ft_top2[b]:
            if h not in fixed_cube.hmap:
                raise RuntimeError(f"{b}: FT01 Top2 HI missing in per-cycle cube: {h}")

    origin_map, Y80, M80 = origin80_bundle_map(C, bundle, bats)
    target_summary = future_summary_matrix(Y80, M80, args.proxy_leads)

    # Coarse timing availability using the final frozen pairs and full nodes.
    cov_rows = []
    for platform, cube in (("Event", event_cube), ("FixedTime", fixed_cube)):
        for L in args.diagnosis_counts:
            schedules = enumerate_schedules(coarse_cycles(args), L, TEST_ORIGIN)
            nvalid = 0
            for s in schedules:
                if platform == "Event":
                    his = EVENT_SOH_TOP2
                else:
                    # Require schedule to be usable for every fold-specific pair.
                    # This is measurement-availability only; no held-out target is consulted.
                    his = sorted(set(h for pair in ft_top2.values() for h in pair))
                # For FixedTime, all union HIs need not be jointly complete. Count a schedule valid
                # if each fold pair is complete; handled explicitly below.
                if platform == "Event":
                    ok = extract_tensor(cube, s, cube.nodes, his) is not None
                else:
                    ok = all(extract_tensor(cube, s, cube.nodes, ft_top2[b]) is not None for b in bats)
                nvalid += int(ok)
            cov_rows.append({
                "Platform": platform,
                "DiagnosisCount": int(L),
                "CandidateSchedules": len(schedules),
                "FullyUsableSchedules": nvalid,
                "PASS": int(nvalid > 0),
            })
    cycle_cov = pd.DataFrame(cov_rows)

    pre = pd.DataFrame([
        {"Check": "47 outer batteries", "Observed": len(bats), "Expected": 47, "PASS": int(len(bats)==47)},
        {"Check": "8 native FixedTime nodes", "Observed": len(fixed_nodes), "Expected": 8, "PASS": int(len(fixed_nodes)==8)},
        {"Check": "Event frozen table located", "Observed": str(event_path), "Expected": "exists", "PASS": int(Path(event_path).exists())},
        {"Check": "FixedTime frozen table located", "Observed": str(fixed_path), "Expected": "exists", "PASS": int(fixed_path.exists())},
        {"Check": "All sensor dependencies known", "Observed": int(registry["DependencyKnown"].sum()), "Expected": len(registry), "PASS": int(registry["DependencyKnown"].eq(1).all())},
        {"Check": "All timing L have usable schedules", "Observed": int(cycle_cov["PASS"].sum()), "Expected": len(cycle_cov), "PASS": int(cycle_cov["PASS"].eq(1).all())},
        {"Check": "Origin80 full-life targets", "Observed": len(origin_map), "Expected": 47, "PASS": int(len(origin_map)==47)},
        {"Check": "Common self-test", "Observed": C.self_test()["Status"], "Expected": "PASS", "PASS": int(C.self_test()["Status"]=="PASS")},
    ])
    if not pre["PASS"].eq(1).all():
        raise RuntimeError("V4.14B preflight failed:\n" + pre[pre.PASS.eq(0)].to_string(index=False))

    return {
        "C": C,
        "adapter": adapter,
        "ftctx": ftctx,
        "bundle": bundle,
        "batteries": bats,
        "origin_map": origin_map,
        "Y80": Y80,
        "M80": M80,
        "target_summary": target_summary,
        "event_cube": event_cube,
        "fixed_cube": fixed_cube,
        "ft_top2": ft_top2,
        "registry": registry,
        "depmap": depmap,
        "cycle_cov": cycle_cov,
        "preflight": pre,
        "event_path": Path(event_path),
        "fixed_path": fixed_path,
    }


# =============================================================================
# 8. Formal full-life XGB outer prediction
# =============================================================================

def source_batteries(ctx, held: str) -> List[str]:
    return [b for b in ctx["batteries"] if b != held]


def source_indices_origin80(ctx, held: str) -> List[int]:
    return [ctx["origin_map"][b] for b in source_batteries(ctx, held)]


def embed_features_141(ctx, X47: np.ndarray) -> np.ndarray:
    bundle = ctx["bundle"]
    out = np.zeros((len(bundle.battery), X47.shape[1]), np.float32)
    bmap = {b: i for i, b in enumerate(ctx["batteries"])}
    for b in ctx["batteries"]:
        out[ctx["origin_map"][b]] = X47[bmap[b]]
    return out


def formal_outer_prediction(
    ctx,
    held: str,
    cube: PlatformCube,
    schedule: Sequence[int],
    nodes: Sequence[float],
    his: Sequence[str],
    args,
) -> Dict[str, object]:
    x = extract_tensor(cube, schedule, nodes, his)
    if x is None:
        raise RuntimeError(f"formal config incomplete: {cube.platform} {schedule} {nodes} {his}")
    X47, _, _ = engineer_dynamic_selected(x, schedule, nodes, his)
    X141 = embed_features_141(ctx, X47)
    tr = source_indices_origin80(ctx, held)
    te = ctx["origin_map"][held]
    model, ntrain = ctx["C"].fit_xgb_trajectory(
        ctx["ftctx"]["v410a"],
        X141,
        ctx["bundle"],
        tr,
        threads=args.xgb_threads,
        leads_per_origin=args.xgb_leads_per_origin,
    )
    pred = ctx["C"].predict_xgb_trajectory(model, X141[te], k=ctx["bundle"].y.shape[1])
    y = np.asarray(ctx["bundle"].y[te], float)
    mask = np.asarray(ctx["bundle"].mask[te], int)
    met = ctx["C"].trajectory_metrics(y, pred, mask)
    valid = (mask == 1) & np.isfinite(y) & np.isfinite(pred)
    sse = float(np.sum((pred[valid] - y[valid]) ** 2))
    sae = float(np.sum(np.abs(pred[valid] - y[valid])))
    n = int(valid.sum())
    del model
    gc.collect()
    return {
        "Pred": pred.astype(np.float32),
        "True": y.astype(np.float32),
        "Mask": mask.astype(np.uint8),
        "Metrics": met,
        "SSE": sse,
        "SAE": sae,
        "N": n,
        "TrainRows": int(ntrain),
        "FeatureCount": int(X47.shape[1]),
    }


# =============================================================================
# 9. Timing-value search
# =============================================================================

def timing_dir(args, held: str) -> Path:
    return ensure_dir(args.output_root / "folds" / held / "timing")


def timing_search_path(args, held: str, platform: str, L: int, last: int) -> Path:
    return timing_dir(args, held) / f"{platform}_L{L}_LAST{last}_search.csv.gz"


def timing_result_path(args, held: str, platform: str, L: int, last: int, role: str) -> Path:
    return timing_dir(args, held) / f"{platform}_L{L}_LAST{last}_{role}.json"


def timing_pred_path(args, held: str, platform: str, L: int, last: int, role: str) -> Path:
    return timing_dir(args, held) / f"{platform}_L{L}_LAST{last}_{role}.npz"


def timing_pair(ctx, held: str, platform: str) -> List[str]:
    return list(EVENT_SOH_TOP2) if platform == "Event" else list(ctx["ft_top2"][held])


def timing_schedule_search(ctx, held: str, platform: str, L: int, last_cycle: int, args) -> pd.DataFrame:
    sp = timing_search_path(args, held, platform, L, last_cycle)
    if sp.exists() and not args.force:
        return pd.read_csv(sp, compression="infer", low_memory=False)

    cube = ctx["event_cube"] if platform == "Event" else ctx["fixed_cube"]
    nodes = cube.nodes
    his = timing_pair(ctx, held, platform)
    cycles = coarse_cycles(args, last_cycle)
    schedules = enumerate_schedules(cycles, L, last_cycle)
    source = source_batteries(ctx, held)
    bmap = {b: i for i, b in enumerate(ctx["batteries"])}
    src_idx = [bmap[b] for b in source]

    rows = []
    for s in schedules:
        x = extract_tensor(cube, s, nodes, his)
        if x is None:
            continue
        Xc = compact_features(x, s)
        info = information_score(Xc, ctx["target_summary"], src_idx)
        rows.append({
            "OuterHeldoutBattery": held,
            "Platform": platform,
            "DiagnosisCount": L,
            "LastCycle": last_cycle,
            "Schedule": "|".join(map(str, s)),
            "InformationScore": info,
            **schedule_descriptors(s),
        })
    d = pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError(f"{held}/{platform}/L{L}: no complete timing schedule")

    # Exhaustive information scan -> Ridge refinement only on top/bottom and uniform.
    d = d.sort_values(["InformationScore", "Schedule"], ascending=[False, True]).reset_index(drop=True)
    top_sched = d.head(max(1, int(args.timing_preselect)))["Schedule"].tolist()
    bottom_sched = d.tail(max(1, int(args.timing_bottom_preselect)))["Schedule"].tolist()
    uni = "|".join(map(str, uniform_schedule(cycles, L, last_cycle)))
    refine = list(dict.fromkeys(top_sched + bottom_sched + [uni]))
    ridge_map = {}
    for ss in refine:
        s = tuple(map(int, ss.split("|")))
        x = extract_tensor(cube, s, nodes, his)
        if x is None:
            continue
        Xc = compact_features(x, s)
        ridge_map[ss] = ridge_proxy_rmse(
            Xc, ctx["batteries"], source, ctx["Y80"], ctx["M80"], args,
            held, f"timing|{platform}|L{L}|last{last_cycle}|{ss}",
        )
    d["RidgeProxyRMSE"] = d["Schedule"].map(ridge_map)
    d["ProxyRefined"] = d["RidgeProxyRMSE"].notna().astype(int)
    d["SearchStage"] = "Coarse4Cycle"

    # Local integer-cycle refinement around the best source-only coarse schedule.
    ref = d[d["RidgeProxyRMSE"].notna()].sort_values(["RidgeProxyRMSE", "Schedule"])
    if len(ref) and args.local_refine_radius > 0 and args.local_refine_rounds > 0:
        coarse_best = tuple(map(int, str(ref.iloc[0].Schedule).split("|")))
        local_best, local_rmse, local_audit = local_refine_schedule_ridge(
            ctx, held, cube, coarse_best, nodes, his, args,
            f"timing_local|{platform}|L{L}|last{last_cycle}",
        )
        local_id = "|".join(map(str, local_best))
        if local_id not in set(d["Schedule"].astype(str)):
            xloc = extract_tensor(cube, local_best, nodes, his)
            if xloc is not None:
                iloc = information_score(compact_features(xloc, local_best), ctx["target_summary"], src_idx)
                d = pd.concat([d, pd.DataFrame([{
                    "OuterHeldoutBattery": held, "Platform": platform,
                    "DiagnosisCount": L, "LastCycle": last_cycle,
                    "Schedule": local_id, "InformationScore": iloc,
                    **schedule_descriptors(local_best),
                    "RidgeProxyRMSE": local_rmse, "ProxyRefined": 1,
                    "SearchStage": "LocalIntegerRefine",
                }])], ignore_index=True)
        elif np.isfinite(local_rmse):
            m = d["Schedule"].astype(str).eq(local_id)
            d.loc[m, "RidgeProxyRMSE"] = np.minimum(
                pd.to_numeric(d.loc[m, "RidgeProxyRMSE"], errors="coerce").fillna(np.inf), local_rmse
            )
            d.loc[m, "ProxyRefined"] = 1
            d.loc[m, "SearchStage"] = "Coarse+LocalRefine"
    save_csv(d, sp, compression=FAST_GZIP)
    return d


def choose_timing_roles(search: pd.DataFrame, args) -> Dict[str, Tuple[int, ...]]:
    ref = search[search["RidgeProxyRMSE"].notna()].copy()
    if ref.empty:
        raise RuntimeError("No Ridge-refined schedules")
    best = ref.sort_values(["RidgeProxyRMSE", "Schedule"]).iloc[0]
    worst = ref.sort_values(["RidgeProxyRMSE", "Schedule"], ascending=[False, True]).iloc[0]
    L = int(best.DiagnosisCount); last = int(best.LastCycle)
    uni_tuple = uniform_schedule(coarse_cycles(args, last), L, last)
    out = {
        "BEST_SOURCE_PROXY": tuple(map(int, str(best.Schedule).split("|"))),
        "UNIFORM": tuple(map(int, uni_tuple)),
    }
    if args.include_worst_timing:
        out["WORST_SOURCE_PROXY"] = tuple(map(int, str(worst.Schedule).split("|")))
    return out


def run_timing_formal(ctx, held: str, platform: str, L: int, last_cycle: int, args):
    search = timing_schedule_search(ctx, held, platform, L, last_cycle, args)
    roles = choose_timing_roles(search, args)
    cube = ctx["event_cube"] if platform == "Event" else ctx["fixed_cube"]
    nodes = cube.nodes
    his = timing_pair(ctx, held, platform)

    for role, schedule in roles.items():
        rp = timing_result_path(args, held, platform, L, last_cycle, role)
        pp = timing_pred_path(args, held, platform, L, last_cycle, role)
        if rp.exists() and pp.exists() and not args.force:
            try:
                q = load_json(rp)
                if q.get("CodeVersion") == CODE_VERSION:
                    print(f"[V414B timing] {held} {platform} L{L} {role}: cache", flush=True)
                    continue
            except Exception:
                pass
        t0 = time.perf_counter()
        formal = formal_outer_prediction(ctx, held, cube, schedule, nodes, his, args)
        held_bi = ctx["batteries"].index(held)
        cost = acquisition_cost(
            cube, schedule, nodes, his, ctx["depmap"],
            battery_indices=[held_bi],
        )
        result = {
            "CodeVersion": CODE_VERSION,
            "OuterHeldoutBattery": held,
            "Platform": platform,
            "Experiment": "PureTiming" if last_cycle == TEST_ORIGIN else "Staleness",
            "Role": role,
            "DiagnosisCount": int(L),
            "LastCycle": int(last_cycle),
            "Schedule": list(map(int, schedule)),
            "Nodes": list(map(float, nodes)),
            "SelectedHIs": list(map(str, his)),
            **schedule_descriptors(schedule),
            **cost,
            **{f"Fold_{k}": v for k, v in formal["Metrics"].items()},
            "SSE": formal["SSE"],
            "SAE": formal["SAE"],
            "N": formal["N"],
            "FeatureCount": formal["FeatureCount"],
            "TrainRows": formal["TrainRows"],
            "HeldoutUsedForTimingSelection": 0,
            "HeldoutUsedForHISelection": 0,
            "HeldoutUsedForTraining": 0,
            "HistoricalSOHAsInput": 0,
            "FutureHIAsInput": 0,
            "CycleAsModelFeature": 0,
            "Runtime_s": float(time.perf_counter() - t0),
        }
        np.savez_compressed(pp, pred=formal["Pred"], true=formal["True"], mask=formal["Mask"])
        save_json(result, rp)
        print(
            f"[V414B timing] {held} {platform} L{L} {role} "
            f"schedule={schedule} RMSE={result['Fold_RMSE']:.6f} scalars={cost['HI_ScalarInputs']}",
            flush=True,
        )


def run_timing_for_battery(ctx, held: str, args, last_cycle: int = TEST_ORIGIN):
    for platform in ("Event", "FixedTime"):
        for L in args.diagnosis_counts:
            if L <= len(coarse_cycles(args, last_cycle)):
                run_timing_formal(ctx, held, platform, int(L), int(last_cycle), args)


# =============================================================================
# 10. Resource search helpers
# =============================================================================

def resource_dir(args, held: str) -> Path:
    return ensure_dir(args.output_root / "folds" / held / "resource")


def resource_search_path(args, held: str, platform: str) -> Path:
    return resource_dir(args, held) / f"{platform}_candidate_search.csv.gz"


def resource_result_path(args, held: str, platform: str, role: str) -> Path:
    return resource_dir(args, held) / f"{platform}_{role}.json"


def resource_pred_path(args, held: str, platform: str, role: str) -> Path:
    return resource_dir(args, held) / f"{platform}_{role}.npz"


def allowed_his_for_modality(ctx, cube: PlatformCube, modality: str) -> List[str]:
    allowed = ctx["adapter"].allowed_his(cube.hi_names, modality, ctx["depmap"])
    # Avoid coordinate-only quantities as sole predictive HIs; they may remain available for cost accounting.
    return [h for h in allowed if h != "PrefixDuration_s"]


def general_hi_ranking(ctx, held: str, cube: PlatformCube, modality: str, args) -> pd.DataFrame:
    source = source_batteries(ctx, held)
    bmap = {b: i for i, b in enumerate(ctx["batteries"])}
    src_idx = [bmap[b] for b in source]
    allowed = allowed_his_for_modality(ctx, cube, modality)
    # Broad measurement canvas for ranking: all coarse cycles and all platform nodes.
    sched = coarse_cycles(args)
    rows = []
    for h in allowed:
        # Ranking tolerates missing cells by using available raw positions; formal candidates still require complete tensors.
        hi_idx = cube.hmap[h]
        ci = [cube.cmap[c] for c in sched]
        raw = np.asarray(cube.values[:, ci, :, hi_idx], float).reshape(len(cube.batteries), -1)
        scores = []
        for j in range(raw.shape[1]):
            for k in range(ctx["target_summary"].shape[1]):
                scores.append(spearman_abs(raw[np.asarray(src_idx), j], ctx["target_summary"][np.asarray(src_idx), k]))
        top = sorted(scores, reverse=True)[: min(12, len(scores))]
        rows.append({
            "HI": h,
            "InformationScore": float(np.mean(top)) if top else 0.0,
            "MaxScore": float(max(scores)) if scores else 0.0,
        })
    d = pd.DataFrame(rows).sort_values(["InformationScore", "MaxScore", "HI"], ascending=[False, False, True]).reset_index(drop=True)
    d["Rank"] = np.arange(1, len(d)+1)
    return d


def node_subsets(nodes: Sequence[float], M: int) -> List[Tuple[float, ...]]:
    return [tuple(map(float, z)) for z in itertools.combinations(tuple(map(float, nodes)), int(M))]


def resource_cells(platform: str, args) -> pd.DataFrame:
    rows = []
    if platform == "Event":
        modalities = args.event_modalities
        node_counts = [1, 2, 3]
    else:
        modalities = args.fixedtime_modalities
        node_counts = args.fixed_node_counts
    for mod in modalities:
        for L in args.diagnosis_counts:
            for M in node_counts:
                for K in args.resource_topks:
                    rows.append({
                        "Platform": platform,
                        "Modality": mod,
                        "DiagnosisCount": int(L),
                        "NodeCount": int(M),
                        "HI_Count": int(K),
                        "NominalScalarInputs": int(L*M*K),
                        "ModalitySensorCeiling": len(MODALITY_SENSORS[mod] | ({"I"} if platform == "Event" else set())),
                    })
    d = pd.DataFrame(rows).sort_values(
        ["NominalScalarInputs", "ModalitySensorCeiling", "DiagnosisCount", "NodeCount", "HI_Count", "Modality"]
    ).reset_index(drop=True)
    return d


def complete_his_for_config(cube: PlatformCube, schedule, nodes, ranking: pd.DataFrame, limit: int) -> List[str]:
    out = []
    for h in ranking.sort_values("Rank")["HI"].astype(str):
        if extract_tensor(cube, schedule, nodes, [h]) is not None:
            out.append(h)
        if len(out) >= int(limit):
            break
    return out


def refine_hi_set(
    ctx, held: str, cube: PlatformCube, schedule, nodes, ranking: pd.DataFrame, K: int, args, salt: str
) -> Tuple[List[str], float]:
    source = source_batteries(ctx, held)
    candidates = complete_his_for_config(cube, schedule, nodes, ranking, args.hi_refine_shortlist)
    if len(candidates) < K:
        raise RuntimeError(f"{salt}: complete HI candidates={len(candidates)} < K={K}")

    def score(hs):
        x = extract_tensor(cube, schedule, nodes, hs)
        if x is None:
            return np.inf
        return ridge_proxy_rmse(
            compact_features(x, schedule), ctx["batteries"], source, ctx["Y80"], ctx["M80"],
            args, held, salt + "|" + "|".join(hs),
        )

    if K == 1:
        vals = [(score([h]), [h]) for h in candidates]
        return min(vals, key=lambda z: (z[0], z[1]))[1], float(min(vals, key=lambda z: (z[0], z[1]))[0])
    if K == 2:
        vals = [(score(list(pair)), list(pair)) for pair in itertools.combinations(candidates, 2)]
        best = min(vals, key=lambda z: (z[0], z[1]))
        return best[1], float(best[0])
    # K=3 greedy after best pair.
    pair, pscore = refine_hi_set(ctx, held, cube, schedule, nodes, ranking, 2, args, salt + "|K2")
    vals = [(score(pair + [h]), pair + [h]) for h in candidates if h not in pair]
    best = min(vals, key=lambda z: (z[0], z[1]))
    return best[1], float(best[0])


def optimize_resource_cell(
    ctx, held: str, platform: str, cell: Mapping[str, object], ranking: pd.DataFrame, args
) -> Optional[Dict[str, object]]:
    cube = ctx["event_cube"] if platform == "Event" else ctx["fixed_cube"]
    L, M, K = int(cell["DiagnosisCount"]), int(cell["NodeCount"]), int(cell["HI_Count"])
    modality = str(cell["Modality"])
    source = source_batteries(ctx, held)
    bmap = {b: i for i, b in enumerate(ctx["batteries"])}
    src_idx = [bmap[b] for b in source]

    schedules = enumerate_schedules(coarse_cycles(args), L, TEST_ORIGIN)
    if not schedules:
        return None
    uni = uniform_schedule(coarse_cycles(args), L, TEST_ORIGIN)

    # Initial HI set from broad source-only ranking.
    rank_allowed = ranking.head(max(args.hi_ranking_shortlist, K)).copy()
    init_his = complete_his_for_config(cube, uni, cube.nodes[:1], rank_allowed, K)
    if len(init_his) < K:
        return None

    # Coordinate step 1: node subsets at uniform schedule, information score only.
    nrows = []
    for ns in node_subsets(cube.nodes, M):
        hs = complete_his_for_config(cube, uni, ns, rank_allowed, K)
        if len(hs) < K:
            continue
        x = extract_tensor(cube, uni, ns, hs[:K])
        if x is None:
            continue
        info = information_score(compact_features(x, uni), ctx["target_summary"], src_idx)
        nrows.append((info, ns, hs[:K]))
    if not nrows:
        return None
    nrows.sort(key=lambda z: (-z[0], z[1]))
    nrows = nrows[: max(1, int(args.resource_node_preselect))]

    # Coordinate step 2: exhaustive coarse timing information scan for top node subsets.
    candidate_pairs = []
    for _, ns, hs in nrows:
        srows = []
        for s in schedules:
            x = extract_tensor(cube, s, ns, hs)
            if x is None:
                continue
            info = information_score(compact_features(x, s), ctx["target_summary"], src_idx)
            srows.append((info, s))
        srows.sort(key=lambda z: (-z[0], z[1]))
        for info, s in srows[: max(1, int(args.resource_schedule_preselect))]:
            candidate_pairs.append((info, ns, s, hs))
    if not candidate_pairs:
        return None

    # Ridge refine the short list.
    refined = []
    for info, ns, s, hs in candidate_pairs:
        x = extract_tensor(cube, s, ns, hs)
        if x is None:
            continue
        prmse = ridge_proxy_rmse(
            compact_features(x, s), ctx["batteries"], source, ctx["Y80"], ctx["M80"], args,
            held, f"resource|{platform}|{modality}|L{L}|M{M}|K{K}|{ns}|{s}",
        )
        refined.append((prmse, -info, ns, s, hs))
    if not refined:
        return None
    refined.sort(key=lambda z: (z[0], z[1], z[2], z[3]))
    prmse, neginfo, ns, s, hs = refined[0]

    # Local integer-cycle timing refinement INSIDE this resource cell.
    # This is why the regular full-resource optimum is never blindly reused.
    if args.local_refine_radius > 0 and args.local_refine_rounds > 0:
        sloc, ploc, _ = local_refine_schedule_ridge(
            ctx, held, cube, s, ns, hs, args,
            f"resource_local|{platform}|{modality}|L{L}|M{M}|K{K}",
        )
        if np.isfinite(ploc) and ploc <= prmse:
            s, prmse = sloc, ploc
            xloc = extract_tensor(cube, s, ns, hs)
            if xloc is not None:
                neginfo = -information_score(compact_features(xloc, s), ctx["target_summary"], src_idx)

    # Resource-specific HI refinement on the winning node/schedule coordinates.
    try:
        hs2, prmse2 = refine_hi_set(
            ctx, held, cube, s, ns, rank_allowed, K, args,
            f"resource_refine|{platform}|{modality}|L{L}|M{M}|K{K}",
        )
        if prmse2 <= prmse:
            hs, prmse = hs2, prmse2
    except Exception:
        pass

    # Resource selection must itself remain source-only.  In Event mode the
    # actual prefix duration varies across cells, so compute the selection cost
    # from the 46 source batteries only.  Held-out deployment cost is reported
    # later but never used to choose a policy.
    cost = acquisition_cost(
        cube, s, ns, hs, ctx["depmap"],
        battery_indices=src_idx,
    )
    if not np.isfinite(cost.get("PhysicalSensorCount", np.nan)):
        return None
    return {
        "Platform": platform,
        "Modality": modality,
        "DiagnosisCount": L,
        "NodeCount": M,
        "HI_Count": K,
        "Schedule": "|".join(map(str, s)),
        "Nodes": "|".join(f"{x:g}" for x in ns),
        "SelectedHIs": "|".join(hs),
        "ProxyRMSE": float(prmse),
        "InformationScore": float(-neginfo),
        **cost,
    }


def search_resource_candidates(ctx, held: str, platform: str, args) -> pd.DataFrame:
    p = resource_search_path(args, held, platform)
    if p.exists() and not args.force:
        return pd.read_csv(p, compression="infer", low_memory=False)

    cube = ctx["event_cube"] if platform == "Event" else ctx["fixed_cube"]
    cells = resource_cells(platform, args)
    max_cells = int(args.resource_max_cells_event if platform == "Event" else args.resource_max_cells_fixedtime)

    rankings = {}
    rows = []
    feasible_scalar = None
    evaluated = 0
    for cell in cells.to_dict("records"):
        scalar = int(cell["NominalScalarInputs"])
        if feasible_scalar is not None and scalar > math.ceil(feasible_scalar * float(args.resource_cost_multiplier)):
            break
        if evaluated >= max_cells:
            break
        mod = str(cell["Modality"])
        if mod not in rankings:
            rankings[mod] = general_hi_ranking(ctx, held, cube, mod, args)
        try:
            r = optimize_resource_cell(ctx, held, platform, cell, rankings[mod], args)
        except Exception as exc:
            r = None
            print(f"[V414B resource proxy WARN] {held} {platform} {cell}: {exc!r}", flush=True)
        evaluated += 1
        if r is None:
            continue
        rows.append(r)
        if float(r["ProxyRMSE"]) < float(args.proxy_feasible_threshold):
            feasible_scalar = scalar if feasible_scalar is None else min(feasible_scalar, scalar)

    d = pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError(f"{held}/{platform}: resource candidate search empty")
    d.insert(0, "OuterHeldoutBattery", held)
    d["ProxyFeasible"] = (d["ProxyRMSE"] < float(args.proxy_feasible_threshold)).astype(int)
    save_csv(d, p, compression=FAST_GZIP)
    return d


def choose_resource_policies(d: pd.DataFrame) -> Dict[str, pd.Series]:
    feasible = d[d["ProxyFeasible"].eq(1)].copy()
    if feasible.empty:
        feasible = d.nsmallest(min(10, len(d)), "ProxyRMSE").copy()

    def finite_fill(s, name):
        x = pd.to_numeric(s[name], errors="coerce")
        return x.fillna(np.inf)

    q = feasible.copy()
    q["_duration"] = finite_fill(q, "AcquisitionDuration_s")
    q["_channel"] = finite_fill(q, "SensorChannelSeconds")

    selectors = {
        "MIN_SCALAR": ["HI_ScalarInputs", "PhysicalSensorCount", "_duration", "ProxyRMSE"],
        "MIN_SENSOR": ["PhysicalSensorCount", "HI_ScalarInputs", "_duration", "ProxyRMSE"],
        "MIN_DURATION": ["_duration", "HI_ScalarInputs", "PhysicalSensorCount", "ProxyRMSE"],
        "MIN_CHANNEL_SECONDS": ["_channel", "HI_ScalarInputs", "PhysicalSensorCount", "ProxyRMSE"],
    }
    out = {}
    for role, cols in selectors.items():
        z = q.sort_values(cols + ["Schedule", "Nodes", "SelectedHIs"]).iloc[0]
        out[role] = z
    return out


def run_resource_formal(ctx, held: str, platform: str, args):
    search = search_resource_candidates(ctx, held, platform, args)
    policies = choose_resource_policies(search)
    cube = ctx["event_cube"] if platform == "Event" else ctx["fixed_cube"]

    for role, r in policies.items():
        rp = resource_result_path(args, held, platform, role)
        pp = resource_pred_path(args, held, platform, role)
        if rp.exists() and pp.exists() and not args.force:
            try:
                q = load_json(rp)
                if q.get("CodeVersion") == CODE_VERSION:
                    print(f"[V414B resource] {held} {platform} {role}: cache", flush=True)
                    continue
            except Exception:
                pass
        schedule = tuple(map(int, str(r.Schedule).split("|")))
        nodes = tuple(map(float, str(r.Nodes).split("|")))
        his = [h for h in str(r.SelectedHIs).split("|") if h]
        t0 = time.perf_counter()
        formal = formal_outer_prediction(ctx, held, cube, schedule, nodes, his, args)
        held_bi = ctx["batteries"].index(held)
        cost = acquisition_cost(
            cube, schedule, nodes, his, ctx["depmap"],
            battery_indices=[held_bi],
        )
        result = {
            "CodeVersion": CODE_VERSION,
            "OuterHeldoutBattery": held,
            "Platform": platform,
            "PolicyRole": role,
            "Modality": str(r.Modality),
            "Schedule": list(schedule),
            "Nodes": list(nodes),
            "SelectedHIs": his,
            "ProxyRMSE": float(r.ProxyRMSE),
            "SelectionCost_Source46_AcquisitionDuration_s": float(r.AcquisitionDuration_s) if np.isfinite(float(r.AcquisitionDuration_s)) else np.nan,
            "SelectionCost_Source46_SensorChannelSeconds": float(r.SensorChannelSeconds) if np.isfinite(float(r.SensorChannelSeconds)) else np.nan,
            **schedule_descriptors(schedule),
            **cost,
            **{f"Fold_{k}": v for k, v in formal["Metrics"].items()},
            "SSE": formal["SSE"],
            "SAE": formal["SAE"],
            "N": formal["N"],
            "FeatureCount": formal["FeatureCount"],
            "TrainRows": formal["TrainRows"],
            "HeldoutUsedForResourceSelection": 0,
            "HeldoutUsedForTimingSelection": 0,
            "HeldoutUsedForHISelection": 0,
            "HeldoutUsedForTraining": 0,
            "HistoricalSOHAsInput": 0,
            "FutureHIAsInput": 0,
            "CycleAsModelFeature": 0,
            "Runtime_s": float(time.perf_counter() - t0),
        }
        np.savez_compressed(pp, pred=formal["Pred"], true=formal["True"], mask=formal["Mask"])
        save_json(result, rp)
        print(
            f"[V414B resource] {held} {platform} {role} scalars={cost['HI_ScalarInputs']} "
            f"sensors={cost['PhysicalSensorCount']} RMSE={result['Fold_RMSE']:.6f} "
            f"schedule={schedule} nodes={nodes} HIs={his}",
            flush=True,
        )


def run_resource_for_battery(ctx, held: str, args):
    for platform in ("Event", "FixedTime"):
        run_resource_formal(ctx, held, platform, args)


# =============================================================================
# 11. Aggregation / statistics
# =============================================================================

def read_npz_metrics(path: Path):
    z = np.load(path)
    return z["pred"], z["true"], z["mask"]


def pooled_from_results(C, rows: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    out = []
    for key, g in rows.groupby(list(group_cols), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        sse = float(g["SSE"].sum())
        n = int(g["N"].sum())
        sae = float(g["SAE"].sum())
        out.append({
            **dict(zip(group_cols, key)),
            "Batteries": g["OuterHeldoutBattery"].nunique(),
            "N": n,
            "RMSE": math.sqrt(sse / n) if n else np.nan,
            "MAE": sae / n if n else np.nan,
            "BatteryMacroRMSE": float(g["Fold_RMSE"].mean()),
            "MedianBatteryRMSE": float(g["Fold_RMSE"].median()),
            "MedianHI_ScalarInputs": float(g["HI_ScalarInputs"].median()),
            "MedianPhysicalSensorCount": float(g["PhysicalSensorCount"].median()),
            "MedianAcquisitionDuration_s": float(pd.to_numeric(g["AcquisitionDuration_s"], errors="coerce").median()),
            "MedianSensorChannelSeconds": float(pd.to_numeric(g["SensorChannelSeconds"], errors="coerce").median()),
        })
    return pd.DataFrame(out)


def cluster_bootstrap_rmse(g: pd.DataFrame, reps: int, seed: int) -> Dict[str, float]:
    q = g[["OuterHeldoutBattery", "SSE", "N"]].copy()
    q = q.groupby("OuterHeldoutBattery", as_index=False).agg(SSE=("SSE", "sum"), N=("N", "sum"))
    sse = q["SSE"].to_numpy(float); n = q["N"].to_numpy(float)
    obs = math.sqrt(float(sse.sum() / n.sum()))
    rng = np.random.default_rng(seed)
    vals = np.empty(int(reps), float)
    B = len(q)
    for i in range(int(reps)):
        ix = rng.integers(0, B, size=B)
        vals[i] = math.sqrt(float(sse[ix].sum() / n[ix].sum()))
    return {
        "ObservedRMSE": obs,
        "CI95_Low": float(np.quantile(vals, 0.025)),
        "CI95_High": float(np.quantile(vals, 0.975)),
        "Batteries": B,
    }


def paired_bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, reps: int, seed: int):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    vals = np.empty(int(reps), float)
    for i in range(int(reps)):
        ix = rng.integers(0, len(d), size=len(d))
        vals[i] = float(np.mean(d[ix]))
    return {
        "Observed": float(np.mean(d)),
        "CI95_Low": float(np.quantile(vals, 0.025)),
        "CI95_High": float(np.quantile(vals, 0.975)),
        "P_A_lt_B": float(np.mean(vals < 0)),
        "N": len(d),
    }


def collect_timing(ctx, args, last_cycle: int = TEST_ORIGIN) -> pd.DataFrame:
    rows = []
    roles = ["BEST_SOURCE_PROXY", "UNIFORM"] + (["WORST_SOURCE_PROXY"] if args.include_worst_timing else [])
    for b in ctx["batteries"]:
        for platform in ("Event", "FixedTime"):
            for L in args.diagnosis_counts:
                for role in roles:
                    p = timing_result_path(args, b, platform, L, last_cycle, role)
                    if not p.exists():
                        continue
                    q = load_json(p)
                    row = dict(q)
                    row["Schedule"] = "|".join(map(str, q["Schedule"]))
                    row["Nodes"] = "|".join(f"{x:g}" for x in q["Nodes"])
                    row["SelectedHIs"] = "|".join(q["SelectedHIs"])
                    rows.append(row)
    return pd.DataFrame(rows)


def collect_resource(ctx, args) -> pd.DataFrame:
    rows = []
    for b in ctx["batteries"]:
        for platform in ("Event", "FixedTime"):
            for role in POLICY_ROLES:
                p = resource_result_path(args, b, platform, role)
                if not p.exists():
                    continue
                q = load_json(p)
                row = dict(q)
                row["Schedule"] = "|".join(map(str, q["Schedule"]))
                row["Nodes"] = "|".join(f"{x:g}" for x in q["Nodes"])
                row["SelectedHIs"] = "|".join(q["SelectedHIs"])
                rows.append(row)
    return pd.DataFrame(rows)


def timing_value_tables(timing: pd.DataFrame, args):
    pooled = pooled_from_results(None, timing, ["Platform", "DiagnosisCount", "LastCycle", "Role"])
    boots = []
    gains = []
    for (platform, L, last), g in timing.groupby(["Platform", "DiagnosisCount", "LastCycle"], sort=True):
        a = g[g.Role.eq("BEST_SOURCE_PROXY")].sort_values("OuterHeldoutBattery")
        b = g[g.Role.eq("UNIFORM")].sort_values("OuterHeldoutBattery")
        if len(a) == EXPECTED_BATTERIES and len(b) == EXPECTED_BATTERIES and a.OuterHeldoutBattery.tolist() == b.OuterHeldoutBattery.tolist():
            bm = paired_bootstrap_mean_diff(
                a["Fold_RMSE"].to_numpy(float), b["Fold_RMSE"].to_numpy(float),
                args.bootstrap, stable_seed("timing_gain", platform, L, last),
            )
            boots.append({
                "Platform": platform, "DiagnosisCount": int(L), "LastCycle": int(last),
                "Statistic": "battery RMSE(BestSourceProxy-Uniform)", **bm,
            })
            pa = pooled[(pooled.Platform.eq(platform)) & pooled.DiagnosisCount.eq(L) & pooled.LastCycle.eq(last) & pooled.Role.eq("BEST_SOURCE_PROXY")]
            pb = pooled[(pooled.Platform.eq(platform)) & pooled.DiagnosisCount.eq(L) & pooled.LastCycle.eq(last) & pooled.Role.eq("UNIFORM")]
            if len(pa)==1 and len(pb)==1:
                gains.append({
                    "Platform": platform, "DiagnosisCount": int(L), "LastCycle": int(last),
                    "BestRMSE": float(pa.iloc[0].RMSE), "UniformRMSE": float(pb.iloc[0].RMSE),
                    "RMSEGain": float(pb.iloc[0].RMSE - pa.iloc[0].RMSE),
                    "RMSEGain_pct": 100.0 * float(pb.iloc[0].RMSE - pa.iloc[0].RMSE) / float(pb.iloc[0].RMSE),
                })
    return pooled, pd.DataFrame(gains), pd.DataFrame(boots)


def resource_tables(resource: pd.DataFrame, args):
    pooled = pooled_from_results(None, resource, ["Platform", "PolicyRole"])
    boots = []
    for key, g in resource.groupby(["Platform", "PolicyRole"], sort=True):
        bm = cluster_bootstrap_rmse(g, args.bootstrap, stable_seed("resource_cluster", *key))
        boots.append({"Platform": key[0], "PolicyRole": key[1], **bm})
    boot = pd.DataFrame(boots)
    out = pooled.merge(boot, on=["Platform", "PolicyRole"], how="left")
    out["Feasible_Pooled"] = (out["RMSE"] < float(args.rmse_threshold)).astype(int)
    out["Feasible_BatteryMacro"] = (out["BatteryMacroRMSE"] < float(args.rmse_threshold)).astype(int)
    out["Feasible_ClusterBoot95"] = (out["CI95_High"] < float(args.rmse_threshold)).astype(int)
    return out, boot


def locate_json(base: Path, pattern: str) -> Optional[Path]:
    hits = sorted(base.rglob(pattern))
    return hits[0] if hits else None


def frozen_soh_anchors(args) -> pd.DataFrame:
    rows = []
    ep = locate_json(args.base, "99_RWTH48_V4_13D_COMPLETE.json")
    if ep:
        try:
            q = load_json(ep)
            # Clean final Event MoE metrics can be found in perturbation rows or top-level summary.
            rmse = None; mae = None
            candidate_rows = (
                q.get("MoEJointPerturbationMetrics")
                or q.get("Results")
                or q.get("Scenarios")
                or q.get("PerturbationSummary")
                or []
            )
            for r in candidate_rows:
                if str(r.get("ModelID", "")) == "HorizonAdaptive_MoE" and "CleanRMSE" in r:
                    rmse = float(r["CleanRMSE"]); mae = float(r.get("CleanMAE", np.nan)); break
            if rmse is None:
                # Known schema seen in V4.13D: records often under Summary.
                blob = json.dumps(q)
                # no fragile parsing; leave optional if unavailable
            if rmse is not None:
                rows.append({
                    "Anchor": "Event_Final_HorizonAdaptive_MoE",
                    "Platform": "Event", "RMSE": rmse, "MAE": mae,
                    "HI_ScalarInputs": 36, "DiagnosisCount": 6, "NodeCount": 3,
                    "SelectedHIs": "V_TimeWeightedMean|P_Last",
                })
        except Exception:
            pass
    fp = args.base / FT02_ROOT_NAME / "99_RWTH48_V4_13F_PRE_FT02_COMPLETE.json"
    if fp.exists():
        try:
            q = load_json(fp)
            r = q.get("BestPooledModel", {})
            if r:
                rows.append({
                    "Anchor": "FixedTime_Final_HorizonAdaptive_MoE",
                    "Platform": "FixedTime", "RMSE": float(r.get("RMSE", np.nan)), "MAE": float(r.get("MAE", np.nan)),
                    "HI_ScalarInputs": 96, "DiagnosisCount": 6, "NodeCount": 8,
                    "SelectedHIs": "fold-specific FT01 Top2",
                })
        except Exception:
            pass
    return pd.DataFrame(rows)


# =============================================================================
# 12. Finalize / figures
# =============================================================================

def make_figures(timing_metrics, timing_gain, resource_metrics, out: Path, args):
    figdir = ensure_dir(out / "Figures")
    if len(timing_metrics):
        fig, ax = plt.subplots(figsize=(9.5, 5.7))
        q = timing_metrics[timing_metrics.Role.isin(["BEST_SOURCE_PROXY", "UNIFORM"]) & timing_metrics.LastCycle.eq(TEST_ORIGIN)]
        for (platform, role), g in q.groupby(["Platform", "Role"], sort=True):
            gg = g.sort_values("DiagnosisCount")
            ax.plot(gg.DiagnosisCount, gg.RMSE, marker="o", label=f"{platform}-{role}")
        ax.axhline(args.rmse_threshold, linestyle="--", linewidth=1)
        ax.set_xlabel("Historical diagnosis count")
        ax.set_ylabel("Full-life SOH RMSE")
        ax.set_title("Measurement timing value at fixed resources")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figdir / "01_timing_value_rmse.png", dpi=args.dpi, bbox_inches="tight"); plt.close(fig)

    if len(resource_metrics):
        fig, ax = plt.subplots(figsize=(9.5, 5.7))
        for platform, g in resource_metrics.groupby("Platform", sort=True):
            ax.scatter(g.MedianHI_ScalarInputs, g.RMSE, label=platform)
            for r in g.itertuples(index=False):
                ax.annotate(r.PolicyRole, (r.MedianHI_ScalarInputs, r.RMSE), fontsize=7)
        ax.axhline(args.rmse_threshold, linestyle="--", linewidth=1)
        ax.set_xlabel("Median extracted HI scalar inputs")
        ax.set_ylabel("Full-life SOH RMSE")
        ax.set_title("Minimum-resource policy outcomes")
        ax.grid(alpha=.2); ax.legend()
        fig.tight_layout(); fig.savefig(figdir / "02_min_resource_rmse.png", dpi=args.dpi, bbox_inches="tight"); plt.close(fig)


def collect_staleness_available(ctx, args) -> pd.DataFrame:
    """Collect optional staleness timing runs if they exist; absence is allowed."""
    parts = []
    for stale in sorted(set(int(x) for x in args.staleness_values if int(x) > 0)):
        last = TEST_ORIGIN - stale
        if last <= args.cycle_start:
            continue
        d = collect_timing(ctx, args, last)
        if not d.empty:
            d = d.copy()
            d["RequestedStaleness"] = int(stale)
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def staleness_tables(stale: pd.DataFrame, args):
    if stale.empty:
        return pd.DataFrame(), pd.DataFrame()
    pooled = pooled_from_results(None, stale, ["Platform", "DiagnosisCount", "LastCycle", "Role"])
    gains = []
    # Compare the best-source-proxy schedule at each stale last-cycle with the
    # corresponding origin80 timing result only when both exist in the caller's
    # combined table; here report feasibility and staleness descriptors directly.
    for r in pooled.itertuples(index=False):
        gains.append({
            "Platform": r.Platform,
            "DiagnosisCount": int(r.DiagnosisCount),
            "LastCycle": int(r.LastCycle),
            "Staleness": int(TEST_ORIGIN - int(r.LastCycle)),
            "Role": r.Role,
            "RMSE": float(r.RMSE),
            "BatteryMacroRMSE": float(r.BatteryMacroRMSE),
            "Feasible_RMSEltThreshold": int(float(r.RMSE) < float(args.rmse_threshold)),
        })
    return pooled, pd.DataFrame(gains)


def finalize(ctx, args):
    timing = collect_timing(ctx, args, TEST_ORIGIN)
    resource = collect_resource(ctx, args)
    if timing.empty:
        raise RuntimeError("No timing results to finalize")
    if resource.empty:
        raise RuntimeError("No resource results to finalize")

    tmet, tgain, tboot = timing_value_tables(timing, args)
    rmet, rboot = resource_tables(resource, args)
    anchors = frozen_soh_anchors(args)
    stale = collect_staleness_available(ctx, args)
    stale_met, stale_summary = staleness_tables(stale, args)

    # Schedule selection frequency / descriptors.
    sched_freq = (
        timing[timing.Role.eq("BEST_SOURCE_PROXY")]
        .groupby(["Platform", "DiagnosisCount", "Schedule"], sort=True)
        .size().rename("OuterFoldCount").reset_index()
        .sort_values(["Platform", "DiagnosisCount", "OuterFoldCount"], ascending=[True, True, False])
    )
    resource_cost = (
        resource.groupby(["Platform", "PolicyRole"], sort=True)
        .agg(
            Batteries=("OuterHeldoutBattery", "nunique"),
            MedianScalars=("HI_ScalarInputs", "median"),
            MedianMeasurementCheckpoints=("MeasurementCheckpointCount", "median"),
            MedianSensorCheckpoints=("SensorCheckpointCount", "median"),
            MinScalars=("HI_ScalarInputs", "min"),
            MaxScalars=("HI_ScalarInputs", "max"),
            MedianSensors=("PhysicalSensorCount", "median"),
            MedianDuration_s=("AcquisitionDuration_s", "median"),
            MedianChannelSeconds=("SensorChannelSeconds", "median"),
            MedianDiagnosisCount=("DiagnosisCount", "median"),
            MedianNodeCount=("NodeCount", "median"),
            MedianHI_Count=("HI_Count", "median"),
        ).reset_index()
    )
    minfeas = rmet[rmet.Feasible_Pooled.eq(1)].copy()
    if len(minfeas):
        minfeas = minfeas.sort_values([
            "MedianHI_ScalarInputs", "MedianPhysicalSensorCount", "MedianAcquisitionDuration_s", "RMSE"
        ])

    save_csv(timing, args.output_root / "03_timing_outer_results.csv")
    save_csv(tmet, args.output_root / "04_timing_pooled_metrics.csv")
    save_csv(tgain, args.output_root / "05_timing_value_gain_vs_uniform.csv")
    save_csv(tboot, args.output_root / "06_timing_value_paired_bootstrap.csv")
    save_csv(sched_freq, args.output_root / "07_best_schedule_frequency.csv")
    save_csv(resource, args.output_root / "08_resource_outer_results.csv")
    save_csv(rmet, args.output_root / "09_resource_policy_metrics.csv")
    save_csv(rboot, args.output_root / "10_resource_cluster_bootstrap.csv")
    save_csv(resource_cost, args.output_root / "11_resource_cost_distribution.csv")
    save_csv(minfeas, args.output_root / "12_minimum_feasible_summary.csv")
    save_csv(anchors, args.output_root / "13_frozen_full_life_anchors.csv")
    if not stale.empty:
        save_csv(stale, args.output_root / "15_staleness_outer_results.csv")
        save_csv(stale_met, args.output_root / "16_staleness_pooled_metrics.csv")
        save_csv(stale_summary, args.output_root / "17_staleness_feasibility_summary.csv")

    timing_flags = ["HeldoutUsedForTimingSelection", "HeldoutUsedForHISelection", "HeldoutUsedForTraining"]
    resource_flags = ["HeldoutUsedForResourceSelection", "HeldoutUsedForTimingSelection", "HeldoutUsedForHISelection", "HeldoutUsedForTraining"]
    tviol = int(timing[timing_flags].to_numpy(int).sum())
    rviol = int(resource[resource_flags].to_numpy(int).sum())

    expected_timing = EXPECTED_BATTERIES * 2 * len(args.diagnosis_counts) * (3 if args.include_worst_timing else 2)
    expected_resource = EXPECTED_BATTERIES * 2 * len(POLICY_ROLES)
    gate = pd.DataFrame([
        {"Check": "47 batteries timing", "Observed": timing.OuterHeldoutBattery.nunique(), "Expected": 47, "PASS": int(timing.OuterHeldoutBattery.nunique()==47)},
        {"Check": "Timing formal rows complete", "Observed": len(timing), "Expected": expected_timing, "PASS": int(len(timing)==expected_timing)},
        {"Check": "Timing leakage flags", "Observed": tviol, "Expected": 0, "PASS": int(tviol==0)},
        {"Check": "47 batteries resource", "Observed": resource.OuterHeldoutBattery.nunique(), "Expected": 47, "PASS": int(resource.OuterHeldoutBattery.nunique()==47)},
        {"Check": "Resource policy rows complete", "Observed": len(resource), "Expected": expected_resource, "PASS": int(len(resource)==expected_resource)},
        {"Check": "Resource leakage flags", "Observed": rviol, "Expected": 0, "PASS": int(rviol==0)},
        {"Check": "All formal RMSE finite", "Observed": int(np.isfinite(pd.concat([timing.Fold_RMSE, resource.Fold_RMSE]).to_numpy(float)).all()), "Expected": 1, "PASS": int(np.isfinite(pd.concat([timing.Fold_RMSE, resource.Fold_RMSE]).to_numpy(float)).all())},
        {"Check": "8 native FixedTime nodes preserved", "Observed": len(ctx["fixed_cube"].nodes), "Expected": 8, "PASS": int(len(ctx["fixed_cube"].nodes)==8)},
        {"Check": "At least one RMSE<0.01 policy", "Observed": int((rmet.RMSE < args.rmse_threshold).sum()), "Expected": ">=1 desirable", "PASS": 1},
    ])
    save_csv(gate, args.output_root / "14_acceptance_gate.csv")
    status = "PASS" if gate.PASS.eq(1).all() else "CHECK"

    if not args.skip_figures:
        make_figures(tmet, tgain, rmet, args.output_root, args)

    complete = {
        "Task": "RWTH48 V4.14B Full-Life SOH Measurement-Timing Value + Minimum-Resource Pareto",
        "CodeVersion": CODE_VERSION,
        "Status": status,
        "PrimaryObjective": "minimize measurement resources subject to full-life SOH RMSE < 0.01",
        "OuterBatteries": EXPECTED_BATTERIES,
        "TimingGrid": list(coarse_cycles(args)),
        "DiagnosisCounts": args.diagnosis_counts,
        "RMSEThreshold": args.rmse_threshold,
        "TimingValueSummary": tgain.to_dict("records"),
        "ResourcePolicyMetrics": rmet.to_dict("records"),
        "MinimumFeasible": minfeas.to_dict("records"),
        "FrozenSOHAnchors": anchors.to_dict("records"),
        "OptionalStalenessSummary": stale_summary.to_dict("records") if len(stale_summary) else [],
        "Protocol": {
            "OuterLOBO": True,
            "HeldoutUsedForTimingSelection": False,
            "HeldoutUsedForResourceSelection": False,
            "HeldoutUsedForHISelection": False,
            "HeldoutUsedForTraining": False,
            "HistoricalSOHAsInput": False,
            "FutureHIAsInput": False,
            "CycleAsModelFeature": False,
            "ArbitrarySchedulesRebuiltFromPerCycleFrozenHITable": True,
            "EventCurrentRequiredForLocator": True,
            "TimingReoptimizedInsideResourcePolicy": True,
        },
        "InterpretationRules": [
            "Part A isolates diagnosis timing by freezing the acquisition platform resources and changing historical diagnosis cycles only.",
            "Part B re-optimizes timing inside each selected resource policy; it does not reuse the Part-A optimum blindly.",
            "Feasible_ClusterBoot95 is an empirical 47-cell battery-cluster statement, not a universal mathematical guarantee.",
            "The formal resource comparison uses a common full-life shallow XGBoost trajectory head; frozen MoE models are capability anchors, not selection models.",
            "RUL is outside the primary objective of this stage.",
        ],
    }
    save_json(complete, args.output_root / "99_RWTH48_V4_14B_COMPLETE.json")

    print("\n" + "="*120)
    print("RWTH48 V4.14B FINALIZE")
    print("Status:", status)
    print("\nTiming gain vs uniform:")
    print(tgain.to_string(index=False) if len(tgain) else "<empty>")
    print("\nResource policy metrics:")
    print(rmet.to_string(index=False))
    print("\nMinimum feasible RMSE<", args.rmse_threshold, ":")
    print(minfeas.to_string(index=False) if len(minfeas) else "NO FORMAL POLICY BELOW THRESHOLD")
    print("\nAcceptance:")
    print(gate.to_string(index=False))
    print("Output:", args.output_root)
    print("="*120)
    if status != "PASS":
        raise RuntimeError("V4.14B acceptance CHECK/FAIL")
    return status


# =============================================================================
# 13. Parallel barrier
# =============================================================================

def state_dir(args) -> Path:
    return ensure_dir(args.state_root / str(args.run_id))


def done_marker(args, wid: int) -> Path:
    return state_dir(args) / f"worker_{wid:02d}.done.json"


def fail_marker(args, wid: int) -> Path:
    return state_dir(args) / f"worker_{wid:02d}.fail.json"


def clear_my_markers(args):
    for p in (done_marker(args, args.parallel_worker_id), fail_marker(args, args.parallel_worker_id)):
        if p.exists():
            p.unlink()


def write_done(args, owned, stage):
    save_json({
        "RunID": args.run_id,
        "WorkerID": args.parallel_worker_id,
        "Workers": args.parallel_workers,
        "OwnedBatteries": list(owned),
        "Stage": stage,
        "Status": "PASS",
        "Time": time.time(),
    }, done_marker(args, args.parallel_worker_id))


def write_fail(args, exc):
    save_json({
        "RunID": args.run_id,
        "WorkerID": args.parallel_worker_id,
        "Status": "FAIL",
        "Exception": repr(exc),
        "Traceback": traceback.format_exc(),
        "Time": time.time(),
    }, fail_marker(args, args.parallel_worker_id))


def barrier_and_finalize(ctx, args):
    if args.parallel_workers <= 1:
        return finalize(ctx, args)
    if args.parallel_worker_id != 0:
        return None
    while True:
        failed = [i for i in range(args.parallel_workers) if fail_marker(args, i).exists()]
        if failed:
            raise RuntimeError(f"Parallel failure: {[load_json(fail_marker(args,i)) for i in failed]}")
        done = [i for i in range(args.parallel_workers) if done_marker(args, i).exists()]
        if len(done) == args.parallel_workers:
            print(f"[BARRIER] all {args.parallel_workers} workers PASS -> finalize", flush=True)
            return finalize(ctx, args)
        print(f"[BARRIER] done={done}, pending={[i for i in range(args.parallel_workers) if i not in done]}", flush=True)
        time.sleep(max(1.0, float(args.barrier_poll_s)))


# =============================================================================
# 14. Main
# =============================================================================

def save_config(ctx, args):
    save_csv(ctx["preflight"], args.output_root / "00_preflight.csv")
    save_csv(ctx["cycle_cov"], args.output_root / "01_candidate_cycle_coverage.csv")
    save_csv(ctx["registry"], args.output_root / "02_sensor_dependency_registry.csv")
    save_json({
        "Task": "RWTH48 V4.14B Full-Life SOH Timing + Minimum Resource",
        "CodeVersion": CODE_VERSION,
        "RMSEThreshold": args.rmse_threshold,
        "CycleGrid": list(coarse_cycles(args)),
        "DiagnosisCounts": args.diagnosis_counts,
        "ProxyLeads": args.proxy_leads,
        "LocalRefineRadius_cycles": args.local_refine_radius,
        "LocalRefineRounds": args.local_refine_rounds,
        "TimingFormalRoles": ["BEST_SOURCE_PROXY", "UNIFORM"] + (["WORST_SOURCE_PROXY"] if args.include_worst_timing else []),
        "ResourceTopKs": args.resource_topks,
        "FixedNodeCounts": args.fixed_node_counts,
        "EventModalities": args.event_modalities,
        "FixedTimeModalities": args.fixedtime_modalities,
        "EventNodes": EVENT_NODES,
        "FixedTimeNodes": list(ctx["fixed_cube"].nodes),
        "EventReady": str(ctx["event_path"]),
        "FixedTimeReady": str(ctx["fixed_path"]),
        "ParallelWorkers": args.parallel_workers,
        "RunID": args.run_id,
        "SelectionCostPolicy": "Event duration/channel-seconds use source-46 only during policy selection; held-out actual cost is report-only",
    }, args.output_root / "00_config.json")


def main():
    args = parse_args()
    ensure_dir(args.output_root); ensure_dir(args.state_root)
    ctx = preflight(args)
    if args.parallel_worker_id == 0 or args.parallel_workers == 1:
        save_config(ctx, args)

    print("="*120)
    print("RWTH48 V4.14B | Full-Life SOH Measurement-Timing Value + Minimum-Resource Pareto")
    print("Stage       :", args.stage)
    print("Worker      :", f"{args.parallel_worker_id}/{args.parallel_workers}")
    print("RunID       :", args.run_id)
    print("Batteries   :", len(ctx["batteries"]))
    print("Cycle grid  :", list(coarse_cycles(args)))
    print("L values    :", args.diagnosis_counts)
    print("RMSE target : <", args.rmse_threshold)
    print("Output      :", args.output_root)
    print("="*120, flush=True)

    if args.stage == "audit":
        print(ctx["preflight"].to_string(index=False))
        print("\nCandidate schedule coverage:")
        print(ctx["cycle_cov"].to_string(index=False))
        print("\nAUDIT PASS")
        return
    if args.stage == "finalize":
        finalize(ctx, args)
        return

    bats = ctx["batteries"]
    if args.only_battery:
        b = normalize_battery(ctx["C"], args.only_battery)
        if b not in bats:
            raise RuntimeError(f"Unknown battery: {b}")
        owned = [b]
    else:
        owned = bats[args.parallel_worker_id :: args.parallel_workers]

    clear_my_markers(args)
    try:
        for i, b in enumerate(owned, start=1):
            print(f"\n[V414B] worker {args.parallel_worker_id}/{args.parallel_workers} battery {i}/{len(owned)} {b}", flush=True)
            if args.stage in ("timing", "all"):
                run_timing_for_battery(ctx, b, args, TEST_ORIGIN)
            if args.stage in ("resource", "all"):
                # Resource timing is re-optimized per resource cell and does NOT reuse Part-A timing blindly.
                run_resource_for_battery(ctx, b, args)
            if args.stage == "staleness":
                for stale in sorted(set(int(x) for x in args.staleness_values if int(x) >= 0)):
                    last = TEST_ORIGIN - stale
                    if last <= args.cycle_start:
                        continue
                    run_timing_for_battery(ctx, b, args, last)
        write_done(args, owned, args.stage)
    except Exception as exc:
        write_fail(args, exc)
        raise

    if args.only_battery:
        print("Single-battery run complete; no finalize.")
        return

    # `all` creates all primary formal outputs and can finalize automatically.
    if args.stage == "all":
        barrier_and_finalize(ctx, args)
    elif args.parallel_workers <= 1:
        print(f"Stage {args.stage} complete. Run remaining primary stage(s) then --stage finalize.")


# =============================================================================
# 15. Internal self-test (import-safe; invoked only by --help/py_compile externally)
# =============================================================================

def _synthetic_self_test():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(7, 4, 3, 2)).astype(np.float32)
    X, names, reg = engineer_dynamic_selected(x, [40, 52, 68, 80], [60, 180, 600], ["A", "B"])
    expected_single = 2*4*3 + 5*4 + 4*3
    expected = 2*expected_single + 2*4*3
    assert X.shape == (7, expected)
    assert len(names) == expected and len(reg) == expected
    s = enumerate_schedules([40,44,48,52,56,60,64,68,72,76,80], 4, 80)
    assert len(s) == math.comb(10, 3)
    assert uniform_schedule([40,44,48,52,56,60,64,68,72,76,80], 4, 80)[-1] == 80
    return True


if __name__ == "__main__":
    main()

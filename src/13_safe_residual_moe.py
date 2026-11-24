# -*- coding: utf-8 -*-
r"""
RWTH48 V4.16 | Cross-Model Timing Replication | FIX1
=====================================================

Scientific question
-------------------
Does the Event timing advantage remain in the SAME DIRECTION across:
    1) XGBoost
    2) TrendDetail-LSTM
    3) TrendDetail-BiLSTM
    4) Horizon-adaptive MoE (XGB + TD-LSTM)

This is NOT a new timing search and NOT a model leaderboard.

Frozen sensing comparison
-------------------------
For every outer held-out battery, reuse V4.14B's already source-only selected:
    OPT      = Event L3 BEST_SOURCE_PROXY
    UNIFORM  = Event L3 UNIFORM

Both roles always use:
    diagnosis count : 3
    Event nodes      : 0.05 / 0.09 / 0.13 RefAh
    HIs              : V_TimeWeightedMean / P_Last
    measured scalars : 3 x 3 x 2 = 18
    physical sensors : V + I

Therefore, within a fold, the principal acquisition difference is timing.

Strict protocol
---------------
* 47-cell outer LOBO.
* The held-out battery is excluded from:
      model fitting,
      deep early stopping,
      source-origin training,
      MoE gate validation / gate search.
* V4.14B timing schedules are reused; no timing reselection occurs here.
* No historical true SOH/capacity is an input.
* No held-out future HI / future SOH is an input.
* Absolute GlobalCycle is NOT a model feature.
* Diagnosis timing is legal acquisition metadata.
* The deep sequence uses relative cycle positions/gaps, not absolute cycle IDs.

Source-origin augmentation
--------------------------
Requested defaults: 80 / 120 / 160.

For each retained source origin o, the origin-80 schedule is translated while
preserving the exact relative offsets. Example:
    origin80 schedule 44|68|80
    origin120         84|108|120
    origin160         124|148|160

An origin is retained GLOBALLY only if:
    * the frozen Event table has complete HIs for every unique Opt/Uniform
      schedule translated to that origin for all 47 batteries, AND
    * the frozen full-life target bundle contains all 47 battery-origin rows.

This retention check uses measurement/target availability only, never held-out
prediction error. If 120 or 160 is unavailable, it is globally dropped and the
audit records that fact. Origin80 is mandatory.

Representation
--------------
XGB:
    V4.14B dynamic engineered representation from the same measured tensor.

LSTM / BiLSTM:
    3 diagnosis time steps.
    Each step contains:
        6 measured HI scalars (3 nodes x 2 HIs)
        relative diagnosis position to prediction origin
        relative gap from previous diagnosis
    => 8 input channels per diagnosis step.

MoE:
    near expert = augmented XGB
    far expert  = augmented TrendDetail-LSTM
    gate        = source-only sigmoid horizon gate.

For each outer fold and timing role, gate-validation batteries are excluded
from temporary gate-expert fitting. After gate selection, final experts are
refit on all 46 source batteries.

Primary evidence
----------------
For each model:
    DeltaRMSE = RMSE_OPT - RMSE_UNIFORM
Negative is better.

Strong model-agnostic timing evidence:
    pooled DeltaRMSE < 0 for all four models,
preferably with paired battery bootstrap CI95 upper < 0 for multiple/all models.

Do NOT require LSTM/BiLSTM to beat XGB in absolute RMSE.
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
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. Constants
# =============================================================================

CODE_VERSION = "V417_SAFEGUARDED_RESIDUAL_MOE_FIX1"

BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR = Path(__file__).resolve().parent

V414B_NAME = "10_schedule_and_resource_search.py"
COMMON_NAME = "08_fixedtime_common_pipeline.py"

V414B_ROOT_NAME = "RWTH48_V4_14B_FullLifeSOH_Timing_MinResource"
OUTPUT_NAME = "RWTH48_V4_17_Safeguarded_Residual_MoE"
STATE_NAME = "RWTH48_V4_17_Safeguarded_Residual_MoE_STATE"

EXPECTED_BATTERIES = 47
TEST_ORIGIN = 80
EVENT_NODES = (0.05, 0.09, 0.13)
EVENT_HIS = ("V_TimeWeightedMean", "P_Last")
ROLES = ("BEST_SOURCE_PROXY", "UNIFORM")
ROLE_LABEL = {
    "BEST_SOURCE_PROXY": "OPT",
    "UNIFORM": "UNIFORM",
}
PRIMARY_MODELS = ("FrozenXGB", "SafeResidualMoE")

LEAD_BINS = (
    (1, 100, "Lead001_100"),
    (101, 300, "Lead101_300"),
    (301, 500, "Lead301_500"),
    (501, 700, "Lead501_700"),
    (701, 1000, "Lead701_1000"),
)

ENC = "utf-8-sig"


# =============================================================================
# 1. CLI / IO
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWTH48 V4.17 Safeguarded Residual MoE FIX1"
    )
    p.add_argument(
        "--stage",
        choices=["audit", "run", "all", "finalize"],
        default="all",
    )

    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--code-dir", type=Path, default=CODE_DIR)
    p.add_argument("--v414b", type=Path, default=None)
    p.add_argument("--common", type=Path, default=None)
    p.add_argument("--v414b-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--state-root", type=Path, default=None)

    p.add_argument("--event-ready", type=Path, default=None)
    p.add_argument("--fixedtime-ready", type=Path, default=None)
    p.add_argument("--dependency-registry", type=Path, default=None)

    # Shared source-origin augmentation.
    p.add_argument(
        "--source-origins",
        nargs="+",
        type=int,
        default=[80, 120, 160],
    )

    # XGB.
    p.add_argument("--xgb-threads", type=int, default=3)
    p.add_argument("--xgb-leads-per-origin", type=int, default=160)

    # Deep.
    p.add_argument("--deep-validation-batteries", type=int, default=7)
    p.add_argument("--deep-max-epochs", type=int, default=60)
    p.add_argument("--deep-patience", type=int, default=8)
    p.add_argument("--deep-batch", type=int, default=12)
    p.add_argument("--deep-dropout", type=float, default=0.12)
    p.add_argument("--deep-lr", type=float, default=8e-4)
    p.add_argument("--deep-clipnorm", type=float, default=1.0)

    # Safeguarded residual MoE: source-only residual/gate design.
    p.add_argument("--gate-validation-batteries", type=int, default=7)

    # Battery-level cross-fitting used to construct source residual labels.
    p.add_argument("--residual-cv-folds", type=int, default=3)
    p.add_argument("--residual-clip-quantile", type=float, default=0.995)
    p.add_argument("--residual-clip-min", type=float, default=0.005)
    p.add_argument("--residual-clip-max", type=float, default=0.08)

    # Hard guard and smooth far activation.
    p.add_argument(
        "--guard-grid",
        nargs="+",
        type=int,
        default=[300, 400, 500, 600, 700],
    )
    p.add_argument(
        "--gate-c-grid",
        nargs="+",
        type=float,
        default=[300, 400, 500, 600, 700, 800],
    )
    p.add_argument(
        "--gate-tau-grid",
        nargs="+",
        type=float,
        default=[50, 100, 150, 200],
    )
    p.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75, 1.00],
    )

    # Source-only no-harm constraints.
    p.add_argument(
        "--no-harm-horizons",
        nargs="+",
        type=int,
        default=[100, 300, 500],
    )
    p.add_argument(
        "--no-harm-epsilon",
        type=float,
        default=0.00010,
        help="absolute SOH RMSE tolerance on source validation",
    )
    p.add_argument(
        "--overall-no-harm-epsilon",
        type=float,
        default=0.00010,
    )
    p.add_argument(
        "--far-objective-start",
        type=int,
        default=701,
    )

    # Inference.
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--force", action="store_true")

    # Threads / parallel.
    p.add_argument("--cpu-threads-per-worker", type=int, default=4)
    p.add_argument("--parallel-workers", type=int, default=1)
    p.add_argument("--parallel-worker-id", type=int, default=0)
    p.add_argument("--run-id", type=str, default="V417_SAFE_RESIDUAL_MOE_20260826_FIX1")
    p.add_argument("--barrier-poll-s", type=float, default=15.0)

    # V4.14B compatibility fields.
    p.add_argument("--cycle-start", type=int, default=40)
    p.add_argument("--cycle-end", type=int, default=80)
    p.add_argument("--cycle-step", type=int, default=4)
    p.add_argument("--proxy-validation-batteries", type=int, default=9)
    p.add_argument("--proxy-ridge-alpha", type=float, default=8.0)
    p.add_argument(
        "--proxy-leads",
        nargs="+",
        type=int,
        default=[10, 25, 50, 100, 200, 300, 500, 700, 900],
    )
    p.add_argument("--local-refine-radius", type=int, default=3)
    p.add_argument("--local-refine-rounds", type=int, default=2)
    p.add_argument("--hi-ranking-shortlist", type=int, default=12)

    a = p.parse_args()

    a.base = Path(a.base)
    a.code_dir = Path(a.code_dir)
    a.v414b = Path(a.v414b) if a.v414b else a.code_dir / V414B_NAME
    a.common = Path(a.common) if a.common else a.code_dir / COMMON_NAME
    a.v414b_root = (
        Path(a.v414b_root)
        if a.v414b_root
        else a.base / V414B_ROOT_NAME
    )
    a.output_root = (
        Path(a.output_root)
        if a.output_root
        else a.base / OUTPUT_NAME
    )
    a.state_root = (
        Path(a.state_root)
        if a.state_root
        else a.base / STATE_NAME
    )

    a.source_origins = sorted(set(int(x) for x in a.source_origins))
    if TEST_ORIGIN not in a.source_origins:
        a.source_origins.insert(0, TEST_ORIGIN)

    if a.cycle_end != TEST_ORIGIN:
        p.error("--cycle-end must remain 80")
    if a.parallel_workers < 1:
        p.error("--parallel-workers must be >=1")
    if not (0 <= a.parallel_worker_id < a.parallel_workers):
        p.error("invalid --parallel-worker-id")
    if a.deep_validation_batteries < 2:
        p.error("--deep-validation-batteries must be >=2")
    if a.gate_validation_batteries < 2:
        p.error("--gate-validation-batteries must be >=2")
    if a.residual_cv_folds < 2:
        p.error("--residual-cv-folds must be >=2")
    if not (0.90 <= a.residual_clip_quantile < 1.0):
        p.error("--residual-clip-quantile must be in [0.90,1)")
    if a.residual_clip_min <= 0 or a.residual_clip_max <= 0:
        p.error("residual clip bounds must be >0")
    if a.residual_clip_min > a.residual_clip_max:
        p.error("residual clip min > max")
    if not a.guard_grid:
        p.error("--guard-grid cannot be empty")
    if min(a.guard_grid) < 1 or max(a.guard_grid) >= 1000:
        p.error("--guard-grid must be within 1..999")
    if any(x <= 0 for x in a.alpha_grid):
        p.error("--alpha-grid values must be >0")
    if a.far_objective_start < 1 or a.far_objective_start > 1000:
        p.error("--far-objective-start must be within 1..1000")
    if a.bootstrap < 1000:
        p.error("--bootstrap should be >=1000")
    return a


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, p: Path) -> None:
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding=ENC)


def save_json(obj, p: Path) -> None:
    ensure_dir(p.parent)

    def conv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x))

    p.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=conv),
        encoding="utf-8",
    )


def load_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def load_module(path: Path, name: str):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stable_seed(*parts) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % (2**31 - 1)


def configure_threads(n: int) -> None:
    n = max(1, int(n))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(n)


def import_tensorflow_with_retry(
    attempts: int = 8,
    delay_s: float = 4.0,
):
    """
    Windows-safe TensorFlow import.

    On some Windows installations two Python processes, antivirus/indexing, or
    a recently closed TensorFlow process can transiently hold _pywrap_tf2.dll.
    A single import then raises:
        DLL load failed ... another program is using this file.

    This is an environment/file-lock problem, not a scientific failure.
    We retry after removing partially imported tensorflow modules.
    """
    import gc
    import time as _time

    last = None
    for attempt in range(1, int(attempts) + 1):
        try:
            import tensorflow as tf
            # Touch a tiny TF operation so DLL initialization is actually tested.
            _ = tf.constant([1.0]) + tf.constant([2.0])
            return tf, attempt, None
        except Exception as exc:
            last = exc

            # Remove partially imported TF modules before retrying.
            for name in list(sys.modules):
                if (
                    name == "tensorflow"
                    or name.startswith("tensorflow.")
                    or name.startswith("_pywrap_tf")
                ):
                    sys.modules.pop(name, None)

            gc.collect()

            if attempt < int(attempts):
                print(
                    f"[V417] TensorFlow import attempt "
                    f"{attempt}/{attempts} failed: {exc}",
                    flush=True,
                )
                _time.sleep(float(delay_s))

    return None, int(attempts), last


def schedule_text(s: Sequence[int]) -> str:
    return "|".join(map(str, s))


# =============================================================================
# 2. V4.14B dependency / preflight
# =============================================================================

def make_b_args(args) -> argparse.Namespace:
    return argparse.Namespace(
        stage="audit",
        base=args.base,
        code_dir=args.code_dir,
        common=args.common,
        event_ready=args.event_ready,
        fixedtime_ready=args.fixedtime_ready,
        dependency_registry=args.dependency_registry,
        output_root=args.output_root,
        state_root=args.state_root,

        cycle_start=args.cycle_start,
        cycle_end=args.cycle_end,
        cycle_step=args.cycle_step,
        diagnosis_counts=[2, 3, 4, 5, 6],
        timing_preselect=24,
        timing_bottom_preselect=6,
        local_refine_radius=args.local_refine_radius,
        local_refine_rounds=args.local_refine_rounds,
        proxy_validation_batteries=args.proxy_validation_batteries,
        proxy_ridge_alpha=args.proxy_ridge_alpha,
        proxy_leads=args.proxy_leads,
        include_worst_timing=False,

        xgb_threads=args.xgb_threads,
        cpu_threads_per_worker=args.cpu_threads_per_worker,
        xgb_leads_per_origin=args.xgb_leads_per_origin,

        rmse_threshold=0.01,
        proxy_feasible_threshold=0.0105,

        resource_topks=[1, 2, 3],
        resource_max_cells_event=90,
        resource_max_cells_fixedtime=180,
        resource_node_preselect=5,
        resource_schedule_preselect=5,
        hi_ranking_shortlist=args.hi_ranking_shortlist,
        hi_refine_shortlist=6,
        resource_cost_multiplier=2.0,
        fixed_node_counts=[1, 2, 3, 4, 6, 8],
        event_modalities=["VI", "VIT"],
        fixedtime_modalities=["V", "VI", "VT", "VIT"],

        staleness_values=[0, 4, 8, 12, 16, 24],
        bootstrap=args.bootstrap,
        only_battery="",
        force=False,
        skip_figures=True,
        dpi=200,

        parallel_workers=1,
        parallel_worker_id=0,
        run_id=args.run_id,
        barrier_poll_s=args.barrier_poll_s,
    )


def old_json(args, battery: str, role: str) -> Path:
    return (
        args.v414b_root
        / "folds"
        / battery
        / "timing"
        / f"Event_L3_LAST80_{role}.json"
    )


def old_npz(args, battery: str, role: str) -> Path:
    return (
        args.v414b_root
        / "folds"
        / battery
        / "timing"
        / f"Event_L3_LAST80_{role}.npz"
    )


def load_schedule(args, battery: str, role: str) -> Tuple[int, int, int]:
    p = old_json(args, battery, role)
    if not p.exists():
        raise FileNotFoundError(p)
    q = load_json(p)
    s = tuple(map(int, q["Schedule"]))
    if len(s) != 3 or s[-1] != TEST_ORIGIN:
        raise RuntimeError(f"{battery}/{role}: invalid schedule={s}")
    if tuple(map(float, q.get("Nodes", EVENT_NODES))) != EVENT_NODES:
        raise RuntimeError(f"{battery}/{role}: Event nodes changed")
    his = tuple(map(str, q.get("SelectedHIs", EVENT_HIS)))
    if his != EVENT_HIS:
        raise RuntimeError(f"{battery}/{role}: Event HIs changed: {his}")
    if int(q.get("HI_ScalarInputs", 18)) != 18:
        raise RuntimeError(f"{battery}/{role}: scalar budget !=18")
    return s


def target_index_map(C, bundle) -> Dict[Tuple[str, int], int]:
    out = {}
    for i, (b, o) in enumerate(zip(bundle.battery, bundle.origin)):
        key = (C.normalize_battery(b), int(o))
        if key in out:
            raise RuntimeError(f"duplicate target bundle key {key}")
        out[key] = int(i)
    return out


def load_extended_event_cube(B, ctx, args, max_origin: int):
    v412a = ctx["ftctx"]["v412a"]
    eargs = argparse.Namespace(
        event_ready=args.event_ready,
        base=args.base,
        phase="Discharge",
        candidate_hi_file=None,
    )
    event_path = Path(ctx["event_path"])
    raw = pd.read_csv(event_path, compression="infer", low_memory=False)
    schema = v412a.detect_event_schema(raw, eargs)
    event = v412a.filter_event_table(raw, schema, eargs)

    cycles = tuple(range(args.cycle_start, int(max_origin) + 1))
    cube = B.build_platform_cube(
        event,
        schema.battery_col,
        schema.cycle_col,
        "_RefAhCanonical",
        schema.hi_cols,
        ctx["batteries"],
        cycles,
        EVENT_NODES,
        "Event",
        ctx["C"],
    )
    return cube


def translated_schedule(base_schedule: Sequence[int], origin: int) -> Tuple[int, ...]:
    shift = int(origin) - TEST_ORIGIN
    return tuple(int(c) + shift for c in base_schedule)


def preflight(args):
    configure_threads(args.cpu_threads_per_worker)

    if not args.v414b.exists():
        raise FileNotFoundError(args.v414b)
    if not args.common.exists():
        raise FileNotFoundError(args.common)
    if not args.v414b_root.exists():
        raise FileNotFoundError(args.v414b_root)

    complete_path = args.v414b_root / "99_RWTH48_V4_14B_COMPLETE.json"
    if not complete_path.exists():
        raise FileNotFoundError(complete_path)
    complete = load_json(complete_path)
    if complete.get("Status") != "PASS":
        raise RuntimeError("V4.14B status != PASS")

    B = load_module(args.v414b, "rwth48_v416_v414b")
    bargs = make_b_args(args)
    ctx = B.preflight(bargs)
    ctx["B"] = B
    ctx["bargs"] = bargs

    bats = list(ctx["batteries"])
    if len(bats) != EXPECTED_BATTERIES:
        raise RuntimeError(f"batteries={len(bats)} !=47")

    # Validate exact frozen timing inputs and collect unique schedules.
    unique_schedules = set()
    missing = []
    for b in bats:
        for role in ROLES:
            try:
                s = load_schedule(args, b, role)
                unique_schedules.add(s)
            except Exception as exc:
                missing.append(f"{b}/{role}: {exc}")
            if not old_npz(args, b, role).exists():
                missing.append(str(old_npz(args, b, role)))
    if missing:
        raise RuntimeError("V4.14B timing dependencies incomplete:\n" + "\n".join(missing[:40]))

    tmap = target_index_map(ctx["C"], ctx["bundle"])
    max_origin = max(args.source_origins)
    ext_cube = load_extended_event_cube(B, ctx, args, max_origin)

    # Global, outcome-free origin retention.
    origin_rows = []
    retained = []
    for origin in args.source_origins:
        target_ok = all((b, int(origin)) in tmap for b in bats)
        measurement_ok = True
        failed_schedule = ""
        for s in sorted(unique_schedules):
            shifted = translated_schedule(s, origin)
            x = B.extract_tensor(ext_cube, shifted, EVENT_NODES, EVENT_HIS)
            if x is None:
                measurement_ok = False
                failed_schedule = schedule_text(shifted)
                break
        keep = int(target_ok and measurement_ok)
        if keep:
            retained.append(int(origin))
        origin_rows.append({
            "RequestedOrigin": int(origin),
            "Target47Available": int(target_ok),
            "EventMeasurementAllSchedulesAvailable": int(measurement_ok),
            "FirstFailedTranslatedSchedule": failed_schedule,
            "Retained": keep,
        })

    if TEST_ORIGIN not in retained:
        raise RuntimeError("Origin80 failed global availability audit")

    ctx["target_index"] = tmap
    ctx["event_cube_ext"] = ext_cube
    ctx["retained_origins"] = tuple(retained)
    ctx["origin_audit"] = pd.DataFrame(origin_rows)
    ctx["unique_schedules"] = sorted(unique_schedules)

    # TensorFlow availability audit with Windows DLL-lock retry.
    tf, tf_attempts, tf_error = import_tensorflow_with_retry(
        attempts=8,
        delay_s=4.0,
    )
    if tf is not None:
        tf_status = "PASS"
        tf_version = str(tf.__version__)
    else:
        tf_status = f"FAIL: {tf_error}"
        tf_version = ""

    pre = pd.DataFrame([
        {
            "Check": "V4.14B PASS",
            "Observed": complete.get("Status"),
            "Expected": "PASS",
            "PASS": int(complete.get("Status") == "PASS"),
        },
        {
            "Check": "47 outer batteries",
            "Observed": len(bats),
            "Expected": 47,
            "PASS": int(len(bats) == 47),
        },
        {
            "Check": "All Event L3 Opt/Uniform caches",
            "Observed": len(missing),
            "Expected": 0,
            "PASS": int(not missing),
        },
        {
            "Check": "18-scalar frozen sensing",
            "Observed": "3 diagnoses x 3 nodes x 2 HIs",
            "Expected": "18",
            "PASS": 1,
        },
        {
            "Check": "Origin80 retained",
            "Observed": "|".join(map(str, retained)),
            "Expected": "contains 80",
            "PASS": int(TEST_ORIGIN in retained),
        },
        {
            "Check": "TensorFlow import",
            "Observed": (
                f"{tf_status} {tf_version} | attempts={tf_attempts}"
            ),
            "Expected": "PASS",
            "PASS": int(tf_status == "PASS"),
        },
        {
            "Check": "Common self-test",
            "Observed": ctx["C"].self_test()["Status"],
            "Expected": "PASS",
            "PASS": int(ctx["C"].self_test()["Status"] == "PASS"),
        },
    ])

    if not pre["PASS"].eq(1).all():
        raise RuntimeError(
            "V4.17 preflight failed:\n"
            + pre[pre["PASS"].eq(0)].to_string(index=False)
        )

    ctx["preflight_v416"] = pre
    return ctx


# =============================================================================
# 3. Shared representation
# =============================================================================

def sample_indices(
    ctx,
    batteries: Sequence[str],
    origins: Sequence[int],
) -> np.ndarray:
    idx = []
    for b in batteries:
        for o in origins:
            key = (str(b), int(o))
            if key in ctx["target_index"]:
                idx.append(ctx["target_index"][key])
    return np.asarray(idx, int)


def build_representation(
    ctx,
    base_schedule: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        Xeng_all [bundle_rows, engineered_dim]
        Xseq_all [bundle_rows, 3, 8]
        available [bundle_rows]
    """
    B = ctx["B"]
    bundle = ctx["bundle"]
    cube = ctx["event_cube_ext"]
    nrow = len(bundle.battery)

    Xeng_all = None
    Xseq_all = np.full((nrow, 3, 8), np.nan, np.float32)
    available = np.zeros(nrow, np.uint8)

    bmap = {b: i for i, b in enumerate(ctx["batteries"])}

    for origin in ctx["retained_origins"]:
        sch = translated_schedule(base_schedule, origin)
        x = B.extract_tensor(cube, sch, EVENT_NODES, EVENT_HIS)
        if x is None:
            raise RuntimeError(f"representation unavailable schedule={sch}")

        Xeng47, _, _ = B.engineer_dynamic_selected(
            x,
            sch,
            EVENT_NODES,
            EVENT_HIS,
        )

        if Xeng_all is None:
            Xeng_all = np.full(
                (nrow, Xeng47.shape[1]),
                np.nan,
                np.float32,
            )

        # 6 raw scalars per diagnosis.
        raw = x.reshape(len(ctx["batteries"]), 3, 6).astype(np.float32)

        # Legal acquisition metadata, relative to prediction origin.
        rel = (
            (np.asarray(sch, float) - float(origin)) / 40.0
        ).astype(np.float32)
        gap = np.zeros(3, np.float32)
        gap[1:] = (
            np.diff(np.asarray(sch, float)) / 40.0
        ).astype(np.float32)

        meta = np.column_stack([rel, gap]).astype(np.float32)
        seq47 = np.concatenate(
            [
                raw,
                np.repeat(meta[None, :, :], len(ctx["batteries"]), axis=0),
            ],
            axis=2,
        )

        for b in ctx["batteries"]:
            ti = ctx["target_index"][(b, int(origin))]
            bi = bmap[b]
            Xeng_all[ti] = Xeng47[bi]
            Xseq_all[ti] = seq47[bi]
            available[ti] = 1

    if Xeng_all is None:
        raise RuntimeError("empty engineered representation")

    selected = sample_indices(
        ctx,
        ctx["batteries"],
        ctx["retained_origins"],
    )
    if not np.isfinite(Xeng_all[selected]).all():
        raise RuntimeError("engineered representation contains NaN")
    if not np.isfinite(Xseq_all[selected]).all():
        raise RuntimeError("sequence representation contains NaN")

    return Xeng_all, Xseq_all, available


# =============================================================================
# 4. Deep model
# =============================================================================

class SequenceChannelScaler:
    """
    Standardize each input channel across sample x diagnosis axes.
    Shape is preserved: [N, 3, D].
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    @classmethod
    def fit(cls, x: np.ndarray):
        x = np.asarray(x, np.float32)
        mean = np.nanmean(x, axis=(0, 1))
        std = np.nanstd(x, axis=(0, 1))
        std = np.where(
            np.isfinite(std) & (std > 1e-8),
            std,
            1.0,
        )
        mean = np.where(np.isfinite(mean), mean, 0.0)
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        z = (x - self.mean[None, None, :]) / self.std[None, None, :]
        if not np.isfinite(z).all():
            raise RuntimeError("SequenceChannelScaler produced NaN/Inf")
        return z.astype(np.float32)


class MaskedTargetScaler:
    """
    One global SOH mean/std estimated from valid source truth only.
    NaN targets remain NaN so masked loss can ignore them.
    """
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    @classmethod
    def fit(cls, y: np.ndarray, mask: np.ndarray):
        y = np.asarray(y, np.float32)
        m = np.asarray(mask).astype(bool)
        ok = m & np.isfinite(y)
        vals = y[ok]
        if len(vals) == 0:
            raise RuntimeError("No valid SOH truth for target scaling")
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if not np.isfinite(std) or std <= 1e-8:
            std = 1.0
        return cls(mean, std)

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, np.float32)
        return ((y - self.mean) / self.std).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, np.float32)
        return (y * self.std + self.mean).astype(np.float32)


def set_tf_seed_local(seed: int) -> None:
    seed = int(seed) % (2**31 - 1)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

def choose_batteries(
    batteries: Sequence[str],
    n: int,
    salt: str,
) -> List[str]:
    scored = sorted(
        (stable_seed(salt, b), b)
        for b in sorted(set(map(str, batteries)))
    )
    n = max(2, min(int(n), max(2, len(scored) // 3)))
    return [b for _, b in scored[:n]]


def build_td_model(
    input_dim: int,
    arch: str,
    k: int,
    dropout: float,
    lr: float,
    clipnorm: float,
):
    import tensorflow as tf

    tf.keras.backend.clear_session()

    inp = tf.keras.Input(
        shape=(3, int(input_dim)),
        name="Event_L3_history",
    )

    a = str(arch).upper()
    if a != "LSTM":
        raise ValueError(
            f"FIX2 is MoE-only; standalone/BiLSTM architectures are disabled: {arch}"
        )

    z = tf.keras.layers.LSTM(
        48,
        return_sequences=False,
        name="encoder_lstm48",
    )(inp)

    z = tf.keras.layers.Dense(
        64,
        activation="swish",
        name="encoder_dense64",
    )(z)
    z = tf.keras.layers.Dropout(
        float(dropout),
        name="encoder_dropout",
    )(z)

    t = tf.keras.layers.Dense(
        96,
        activation="swish",
        name="trend_dense96",
    )(z)
    t = tf.keras.layers.Dropout(float(dropout))(t)
    trend = tf.keras.layers.Dense(
        int(k),
        activation="linear",
        name="trend_raw",
    )(t)

    d = tf.keras.layers.Dense(
        96,
        activation="swish",
        name="detail_dense96",
    )(z)
    d = tf.keras.layers.Dropout(float(dropout))(d)
    detail = tf.keras.layers.Dense(
        int(k),
        activation="linear",
        name="detail_raw",
    )(d)

    gate = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(-1.5),
        name="detail_gate",
    )(z)

    out = tf.keras.layers.Add(name="SOH_trajectory")([
        trend,
        tf.keras.layers.Multiply()([detail, gate]),
    ])

    model = tf.keras.Model(
        inp,
        out,
        name=f"Event_L3_{arch}_TrendDetailResidual",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(lr),
            clipnorm=float(clipnorm),
        ),
        loss=ctx_masked_loss(),
    )
    return model


def ctx_masked_loss():
    import tensorflow as tf

    def loss(y_true, y_pred):
        mask = tf.math.is_finite(y_true)
        yt = tf.where(mask, y_true, tf.zeros_like(y_true))
        yp = tf.where(mask, y_pred, tf.zeros_like(y_pred))
        sq = tf.where(mask, tf.square(yp - yt), tf.zeros_like(yp))
        den = tf.maximum(
            tf.reduce_sum(tf.cast(mask, tf.float32)),
            1.0,
        )
        return tf.reduce_sum(sq) / den

    loss.__name__ = "masked_mse"
    return loss


def make_battery_cv_folds(
    batteries: Sequence[str],
    n_folds: int,
    salt: str,
) -> List[List[str]]:
    bats = sorted(set(map(str, batteries)))
    if len(bats) < 3:
        raise RuntimeError("Too few batteries for residual cross-fitting")

    n_folds = max(2, min(int(n_folds), len(bats)))
    ordered = [
        b
        for _, b in sorted(
            (stable_seed(salt, b), b)
            for b in bats
        )
    ]
    folds = [[] for _ in range(n_folds)]
    for i, b in enumerate(ordered):
        folds[i % n_folds].append(b)
    return [f for f in folds if f]


def fit_xgb_for_residual_targets(
    ctx,
    Xeng: np.ndarray,
    train_bats: Sequence[str],
    origins: Sequence[int],
    args,
):
    idx = sample_indices(ctx, train_bats, origins)
    if len(idx) == 0:
        raise RuntimeError("Residual-target XGB train indices empty")

    return ctx["C"].fit_xgb_trajectory(
        ctx["ftctx"]["v410a"],
        Xeng,
        ctx["bundle"],
        idx,
        threads=args.xgb_threads,
        leads_per_origin=args.xgb_leads_per_origin,
    )


def crossfit_xgb_residual_targets(
    ctx,
    Xeng: np.ndarray,
    batteries: Sequence[str],
    origins: Sequence[int],
    args,
    salt: str,
):
    """
    Construct battery-level OOF residual targets:
        residual = true SOH - source-only XGB prediction.

    Each battery's residual target is predicted by an XGB model that did not
    train on that battery.
    """
    bats = sorted(set(map(str, batteries)))
    folds = make_battery_cv_folds(
        bats,
        args.residual_cv_folds,
        f"{salt}|RESIDUAL_CV",
    )

    bundle = ctx["bundle"]
    residual = np.full(
        np.asarray(bundle.y).shape,
        np.nan,
        np.float32,
    )
    base_oof = np.full_like(residual, np.nan)
    coverage_rows = []

    for fi, val_bats in enumerate(folds):
        val_set = set(val_bats)
        train_bats = [b for b in bats if b not in val_set]
        if not train_bats:
            raise RuntimeError("Residual CV training battery set empty")

        model, ntrain = fit_xgb_for_residual_targets(
            ctx,
            Xeng,
            train_bats,
            origins,
            args,
        )

        fold_rows = 0
        for b in val_bats:
            for origin in origins:
                key = (b, int(origin))
                if key not in ctx["target_index"]:
                    continue
                ti = ctx["target_index"][key]

                pred = ctx["C"].predict_xgb_trajectory(
                    model,
                    Xeng[ti],
                    k=bundle.y.shape[1],
                )
                y = np.asarray(bundle.y[ti], np.float32)
                m = np.asarray(bundle.mask[ti]).astype(bool)
                ok = m & np.isfinite(y) & np.isfinite(pred)

                base_oof[ti, ok] = np.asarray(pred, np.float32)[ok]
                residual[ti, ok] = y[ok] - np.asarray(pred, np.float32)[ok]
                fold_rows += int(ok.sum())

        coverage_rows.append({
            "Fold": fi,
            "TrainBatteries": len(train_bats),
            "ValidationBatteries": len(val_bats),
            "ValidationBatteryList": "|".join(val_bats),
            "XGBLongTrainingRows": int(ntrain),
            "ResidualTruthPoints": int(fold_rows),
        })

        del model
        gc.collect()

    selected = sample_indices(ctx, bats, origins)
    msel = np.asarray(bundle.mask[selected]).astype(bool)
    rsel = residual[selected]
    covered = msel & np.isfinite(rsel)

    required = int(msel.sum())
    got = int(covered.sum())
    if required != got:
        raise RuntimeError(
            f"Residual OOF coverage incomplete: got={got}, required={required}"
        )

    vals = np.abs(rsel[covered])
    q = float(np.quantile(vals, args.residual_clip_quantile))
    clip = float(
        np.clip(
            q,
            args.residual_clip_min,
            args.residual_clip_max,
        )
    )

    info = {
        "ResidualCVFolds": len(folds),
        "ResidualTruthPoints": got,
        "ResidualAbsQuantileRaw": q,
        "ResidualClip": clip,
        "ResidualCV": coverage_rows,
    }
    return residual, base_oof, info


def fit_residual_td_source_only(
    ctx,
    Xeng: np.ndarray,
    Xseq: np.ndarray,
    source_bats: Sequence[str],
    args,
    salt: str,
):
    """
    Train TrendDetail-LSTM to predict battery-level cross-fitted XGB residuals.
    """
    import tensorflow as tf

    bundle = ctx["bundle"]
    residual, base_oof, rinfo = crossfit_xgb_residual_targets(
        ctx,
        Xeng,
        source_bats,
        ctx["retained_origins"],
        args,
        salt,
    )

    val_bats = choose_batteries(
        source_bats,
        args.deep_validation_batteries,
        f"{salt}|RESIDUAL_DEEP_ES",
    )
    val_set = set(val_bats)
    train_bats = [b for b in source_bats if b not in val_set]

    tr_idx = sample_indices(ctx, train_bats, ctx["retained_origins"])
    va_idx = sample_indices(ctx, val_bats, ctx["retained_origins"])
    all_idx = sample_indices(ctx, source_bats, ctx["retained_origins"])

    if len(tr_idx) == 0 or len(va_idx) == 0 or len(all_idx) == 0:
        raise RuntimeError("Residual deep source split empty")

    xsc = SequenceChannelScaler.fit(Xseq[tr_idx])
    ysc = MaskedTargetScaler.fit(
        residual[tr_idx],
        bundle.mask[tr_idx],
    )

    Xtr = xsc.transform(Xseq[tr_idx])
    Xva = xsc.transform(Xseq[va_idx])
    ytr = ysc.transform(residual[tr_idx])
    yva = ysc.transform(residual[va_idx])

    set_tf_seed_local(stable_seed(salt, "RESIDUAL_TD", "ES"))

    model = build_td_model(
        Xseq.shape[2],
        "LSTM",
        bundle.y.shape[1],
        args.deep_dropout,
        args.deep_lr,
        args.deep_clipnorm,
    )

    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=max(1, int(args.deep_patience)),
        min_delta=1e-5,
        restore_best_weights=True,
        verbose=0,
    )

    hist = model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=max(1, int(args.deep_max_epochs)),
        batch_size=max(1, int(args.deep_batch)),
        shuffle=True,
        verbose=0,
        callbacks=[cb],
    )

    vals = np.asarray(hist.history.get("val_loss", []), float)
    if len(vals) and np.isfinite(vals).any():
        best_epoch = int(np.nanargmin(vals) + 1)
    else:
        best_epoch = max(1, len(hist.history.get("loss", [])))

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    # Final refit on all source batteries with the source-only OOF residual labels.
    xsc2 = SequenceChannelScaler.fit(Xseq[all_idx])
    ysc2 = MaskedTargetScaler.fit(
        residual[all_idx],
        bundle.mask[all_idx],
    )
    Xall = xsc2.transform(Xseq[all_idx])
    yall = ysc2.transform(residual[all_idx])

    set_tf_seed_local(stable_seed(salt, "RESIDUAL_TD", "REFIT"))

    model2 = build_td_model(
        Xseq.shape[2],
        "LSTM",
        bundle.y.shape[1],
        args.deep_dropout,
        args.deep_lr,
        args.deep_clipnorm,
    )
    model2.fit(
        Xall,
        yall,
        epochs=max(1, best_epoch),
        batch_size=max(1, int(args.deep_batch)),
        shuffle=True,
        verbose=0,
    )

    info = {
        "Architecture": "LSTM",
        "ModelKind": "TrendDetail_XGBResidual",
        "BestEpoch": best_epoch,
        "ValidationBatteries": "|".join(val_bats),
        "SourceBatteries": len(source_bats),
        "SourceOrigins": "|".join(map(str, ctx["retained_origins"])),
        "FinalTrainSamples": len(all_idx),
        **rinfo,
    }
    return model2, xsc2, ysc2, info


def predict_residual(
    ctx,
    model,
    xsc,
    ysc,
    Xseq_row: np.ndarray,
    clip_abs: float,
) -> np.ndarray:
    p = np.asarray(
        model.predict(
            xsc.transform(Xseq_row[None, :, :]),
            batch_size=1,
            verbose=0,
        ),
        np.float32,
    )[0]
    p = ysc.inverse(p)
    p = np.clip(p, -float(clip_abs), float(clip_abs))

    if len(p) != ctx["bundle"].y.shape[1] or not np.isfinite(p).all():
        raise RuntimeError("invalid residual TD prediction")
    return p.astype(np.float32)


# =============================================================================
# 5. Frozen XGB + safeguarded residual gate
# =============================================================================

def source_bats(ctx, held: str) -> List[str]:
    return [b for b in ctx["batteries"] if b != held]


def load_frozen_near_prediction(
    ctx,
    args,
    held: str,
    role: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    p = old_npz(args, held, role)
    if not p.exists():
        raise FileNotFoundError(p)

    z = np.load(p)
    pred = np.asarray(z["pred"], np.float32)
    true = np.asarray(z["true"], np.float32)
    mask = np.asarray(z["mask"], np.uint8)

    te = ctx["target_index"][(held, TEST_ORIGIN)]
    y_bundle = np.asarray(ctx["bundle"].y[te], np.float32)
    m_bundle = np.asarray(ctx["bundle"].mask[te], np.uint8)

    if len(pred) != len(y_bundle):
        raise RuntimeError(
            f"{held}/{role}: frozen XGB prediction length mismatch"
        )

    truth_equal = np.allclose(
        true,
        y_bundle,
        equal_nan=True,
        atol=1e-7,
        rtol=0,
    )
    mask_equal = np.array_equal(mask, m_bundle)

    if not truth_equal or not mask_equal:
        raise RuntimeError(
            f"{held}/{role}: V4.14B truth/mask mismatch "
            f"(truth={truth_equal}, mask={mask_equal})"
        )
    if not np.isfinite(pred).all():
        raise RuntimeError(f"{held}/{role}: frozen XGB prediction has NaN/Inf")

    return pred, y_bundle, m_bundle, {
        "FrozenTruthExact": int(truth_equal),
        "FrozenMaskExact": int(mask_equal),
        "FrozenPredictionRows": int(len(pred)),
    }


def fit_gate_xgb_strict80(
    ctx,
    Xeng,
    train_bats: Sequence[str],
    args,
):
    idx = sample_indices(ctx, train_bats, (TEST_ORIGIN,))
    return ctx["C"].fit_xgb_trajectory(
        ctx["ftctx"]["v410a"],
        Xeng,
        ctx["bundle"],
        idx,
        threads=args.xgb_threads,
        leads_per_origin=args.xgb_leads_per_origin,
    )


def predict_temp_xgb(ctx, model, Xeng_row):
    return ctx["C"].predict_xgb_trajectory(
        model,
        Xeng_row,
        k=ctx["bundle"].y.shape[1],
    )


def guarded_residual_weights(
    k: int,
    guard: int,
    c: float,
    tau: float,
) -> np.ndarray:
    lead = np.arange(1, int(k) + 1, dtype=float)
    w = np.zeros(int(k), np.float32)

    active = lead > int(guard)
    if not np.any(active):
        return w

    tau = max(float(tau), 1e-6)
    raw = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                (lead - float(c)) / tau,
                -40,
                40,
            )
        )
    )
    raw_guard = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                (float(guard) - float(c)) / tau,
                -40,
                40,
            )
        )
    )

    den = max(1.0 - raw_guard, 1e-8)
    ww = np.clip((raw - raw_guard) / den, 0.0, 1.0)
    w[active] = ww[active].astype(np.float32)
    w[~active] = 0.0
    return w


def rmse_vector(y, p, mask, lead_mask=None) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.asarray(mask).astype(bool)

    if lead_mask is not None:
        m = m & np.asarray(lead_mask).astype(bool)

    ok = m & np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return float("nan")

    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2)))


def search_safeguarded_gate(
    rows: pd.DataFrame,
    args,
) -> Dict[str, object]:
    d = rows.copy()
    d = d[
        pd.to_numeric(
            d["Mask"],
            errors="coerce",
        ).fillna(0).astype(int).eq(1)
    ].copy()

    if d.empty:
        raise RuntimeError("Safeguard gate validation rows empty")

    lead = pd.to_numeric(
        d["Lead"],
        errors="coerce",
    ).to_numpy(int)
    y = pd.to_numeric(
        d["SOH_True"],
        errors="coerce",
    ).to_numpy(float)
    base = pd.to_numeric(
        d["Pred_Base"],
        errors="coerce",
    ).to_numpy(float)
    resid = pd.to_numeric(
        d["Pred_Residual"],
        errors="coerce",
    ).to_numpy(float)

    mask = np.isfinite(y) & np.isfinite(base) & np.isfinite(resid)

    if not mask.any():
        raise RuntimeError("No finite safeguard validation points")

    base_overall = rmse_vector(y, base, mask)

    base_near = {}
    for h in args.no_harm_horizons:
        base_near[int(h)] = rmse_vector(
            y,
            base,
            mask,
            lead <= int(h),
        )

    far_start_used = int(args.far_objective_start)
    far_mask = lead >= far_start_used
    base_far = rmse_vector(y, base, mask, far_mask)

    if (
        (not np.isfinite(base_far))
        or int((mask & far_mask).sum()) < 100
    ):
        far_start_used = 501
        far_mask = lead >= far_start_used
        base_far = rmse_vector(y, base, mask, far_mask)

    candidate_rows = []

    candidate_rows.append({
        "Candidate": "BASE_XGB",
        "Guard": 1000,
        "c": 1000.0,
        "tau": 100.0,
        "alpha": 0.0,
        "OverallRMSE": base_overall,
        "FarRMSE": base_far,
        "FarStartUsed": far_start_used,
        "NearPass": 1,
        "OverallPass": 1,
        "Feasible": 1,
        "NearDetail": "|".join(
            f"{h}:{base_near[h]:.8f}/{base_near[h]:.8f}/1"
            for h in sorted(base_near)
        ),
    })

    k = int(np.nanmax(lead))

    for guard in sorted(set(map(int, args.guard_grid))):
        for c in sorted(set(map(float, args.gate_c_grid))):
            if c < float(guard):
                continue

            for tau in sorted(set(map(float, args.gate_tau_grid))):
                w_full = guarded_residual_weights(
                    k,
                    guard,
                    c,
                    tau,
                )
                w = w_full[lead - 1]

                for alpha in sorted(set(map(float, args.alpha_grid))):
                    pred = base + float(alpha) * w * resid

                    overall = rmse_vector(y, pred, mask)
                    far = rmse_vector(y, pred, mask, far_mask)

                    near_pass = True
                    near_detail = []

                    for h in sorted(base_near):
                        r = rmse_vector(
                            y,
                            pred,
                            mask,
                            lead <= int(h),
                        )
                        b = base_near[h]
                        passed = (
                            np.isfinite(r)
                            and np.isfinite(b)
                            and r
                            <= b + float(args.no_harm_epsilon)
                        )
                        near_pass = near_pass and bool(passed)
                        near_detail.append(
                            f"{h}:{r:.8f}/{b:.8f}/{int(bool(passed))}"
                        )

                    overall_pass = (
                        np.isfinite(overall)
                        and overall
                        <= base_overall
                        + float(args.overall_no_harm_epsilon)
                    )

                    feasible = bool(
                        near_pass
                        and overall_pass
                        and np.isfinite(far)
                    )

                    candidate_rows.append({
                        "Candidate": "SAFE_RESIDUAL",
                        "Guard": int(guard),
                        "c": float(c),
                        "tau": float(tau),
                        "alpha": float(alpha),
                        "OverallRMSE": float(overall),
                        "FarRMSE": float(far),
                        "FarStartUsed": far_start_used,
                        "NearPass": int(near_pass),
                        "OverallPass": int(overall_pass),
                        "Feasible": int(feasible),
                        "NearDetail": "|".join(near_detail),
                    })

    cand = pd.DataFrame(candidate_rows)
    feasible = cand[cand["Feasible"].eq(1)].copy()

    if feasible.empty:
        raise RuntimeError("No feasible safeguard candidate including fallback")

    feasible["_neg_guard"] = -pd.to_numeric(
        feasible["Guard"],
        errors="coerce",
    )

    feasible = feasible.sort_values(
        [
            "FarRMSE",
            "OverallRMSE",
            "_neg_guard",
            "alpha",
            "c",
            "tau",
        ],
        ascending=[True, True, True, True, True, True],
    )

    best = feasible.iloc[0].to_dict()
    best.pop("_neg_guard", None)

    best["BaseOverallRMSE"] = float(base_overall)
    best["BaseFarRMSE"] = float(base_far)
    best["FarStartUsed"] = int(far_start_used)
    best["CandidateCount"] = int(len(cand))
    best["FeasibleCandidateCount"] = int(cand["Feasible"].sum())
    best["FallbackSelected"] = int(float(best["alpha"]) == 0.0)
    best["SelectionCriterion"] = (
        "source-only no-harm feasible -> min far RMSE -> min overall RMSE "
        "-> larger guard -> smaller alpha"
    )
    best["CandidateAudit"] = cand.to_dict("records")

    return best


def gate_validation(
    ctx,
    Xeng,
    Xseq,
    held: str,
    args,
    salt: str,
):
    import tensorflow as tf

    source = source_bats(ctx, held)

    gate_bats = choose_batteries(
        source,
        args.gate_validation_batteries,
        f"{salt}|SAFE_GATE",
    )

    gate_set = set(gate_bats)
    fit_bats = [
        b
        for b in source
        if b not in gate_set
    ]

    xgb, ntrain = fit_gate_xgb_strict80(
        ctx,
        Xeng,
        fit_bats,
        args,
    )

    residual_td, xsc, ysc, rinfo = fit_residual_td_source_only(
        ctx,
        Xeng,
        Xseq,
        fit_bats,
        args,
        f"{salt}|TEMP_RESIDUAL_TD",
    )

    rows = []

    for b in gate_bats:
        te = ctx["target_index"][(b, TEST_ORIGIN)]

        p_base = predict_temp_xgb(
            ctx,
            xgb,
            Xeng[te],
        )

        p_resid = predict_residual(
            ctx,
            residual_td,
            xsc,
            ysc,
            Xseq[te],
            rinfo["ResidualClip"],
        )

        y = np.asarray(
            ctx["bundle"].y[te],
            float,
        )
        m = np.asarray(
            ctx["bundle"].mask[te],
            int,
        )

        for lead_i in range(1, len(y) + 1):
            rows.append({
                "Battery": b,
                "Lead": lead_i,
                "SOH_True": float(y[lead_i - 1]),
                "Pred_Base": float(p_base[lead_i - 1]),
                "Pred_Residual": float(p_resid[lead_i - 1]),
                "Mask": int(m[lead_i - 1]),
            })

    gate = search_safeguarded_gate(
        pd.DataFrame(rows),
        args,
    )

    gate["GateValidationBatteries"] = "|".join(gate_bats)
    gate["GateFitBatteries"] = "|".join(fit_bats)
    gate["TemporaryNearXGBTrainRows"] = int(ntrain)
    gate["TemporaryResidualTDInfo"] = rinfo

    del xgb, residual_td
    tf.keras.backend.clear_session()
    gc.collect()

    return gate


# =============================================================================
# 6. Fold run / cache
# =============================================================================

def fold_role_dir(args, held: str, role: str) -> Path:
    return ensure_dir(
        args.output_root
        / "folds"
        / held
        / ROLE_LABEL[role]
    )


def result_json(args, held: str, role: str) -> Path:
    return fold_role_dir(args, held, role) / "result.json"


def pred_npz(args, held: str, role: str) -> Path:
    return fold_role_dir(args, held, role) / "predictions.npz"


def trajectory_stats(y, p, mask) -> Dict[str, float]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mask = np.asarray(mask).astype(bool)

    ok = mask & np.isfinite(y) & np.isfinite(p)
    e = p[ok] - y[ok]

    return {
        "RMSE": float(np.sqrt(np.mean(e * e))),
        "MAE": float(np.mean(np.abs(e))),
        "Bias": float(np.mean(e)),
        "SSE": float(np.sum(e * e)),
        "SAE": float(np.sum(np.abs(e))),
        "N": int(len(e)),
    }


def run_fold_role(ctx, held: str, role: str, args):
    import tensorflow as tf

    jp = result_json(args, held, role)
    pp = pred_npz(args, held, role)

    if jp.exists() and pp.exists() and not args.force:
        q = load_json(jp)
        if q.get("CodeVersion") == CODE_VERSION:
            print(
                f"[V417] {held} {ROLE_LABEL[role]}: cache",
                flush=True,
            )
            return

    t0 = time.perf_counter()
    schedule = load_schedule(args, held, role)
    Xeng, Xseq, _ = build_representation(ctx, schedule)
    source = source_bats(ctx, held)

    # Exact frozen held-out base XGB.
    p_xgb, y, mask, anchor_audit = load_frozen_near_prediction(
        ctx,
        args,
        held,
        role,
    )

    # Source-only safeguard/gate selection.
    gate = gate_validation(
        ctx,
        Xeng,
        Xseq,
        held,
        args,
        f"{held}|{role}",
    )

    # Final residual TD refit on all 46 source batteries.
    residual_td, rxsc, rysc, rinfo = fit_residual_td_source_only(
        ctx,
        Xeng,
        Xseq,
        source,
        args,
        f"{held}|{role}|FINAL_RESIDUAL_TD",
    )

    te = ctx["target_index"][(held, TEST_ORIGIN)]
    p_resid = predict_residual(
        ctx,
        residual_td,
        rxsc,
        rysc,
        Xseq[te],
        rinfo["ResidualClip"],
    )

    k = len(p_xgb)
    if float(gate["alpha"]) == 0.0:
        w = np.zeros(k, np.float32)
    else:
        w = guarded_residual_weights(
            k,
            int(gate["Guard"]),
            float(gate["c"]),
            float(gate["tau"]),
        )

    correction = (
        float(gate["alpha"])
        * w
        * p_resid
    ).astype(np.float32)

    p_safe = (
        p_xgb
        + correction
    ).astype(np.float32)

    # Structural invariant: no correction inside the selected hard guard.
    guard = int(gate["Guard"])
    if guard >= k:
        inside = np.arange(k) < k
    else:
        inside = np.arange(1, k + 1) <= guard

    max_guard_diff = float(
        np.max(np.abs(p_safe[inside] - p_xgb[inside]))
        if np.any(inside)
        else 0.0
    )
    if max_guard_diff > 1e-8:
        raise RuntimeError(
            f"Near-guard invariance violated: {max_guard_diff}"
        )

    metrics = {
        "FrozenXGB": trajectory_stats(y, p_xgb, mask),
        "SafeResidualMoE": trajectory_stats(y, p_safe, mask),
    }

    residual_diag = trajectory_stats(
        np.zeros_like(p_resid),
        p_resid,
        mask,
    )

    old = load_json(old_json(args, held, role))

    q = {
        "CodeVersion": CODE_VERSION,
        "OuterHeldoutBattery": held,
        "Role": ROLE_LABEL[role],
        "V414BRole": role,
        "Schedule": list(schedule),
        "Nodes": list(EVENT_NODES),
        "SelectedHIs": list(EVENT_HIS),
        "DiagnosisCount": 3,
        "HI_ScalarInputs": 18,
        "PhysicalSensorCount": int(old.get("PhysicalSensorCount", 2)),
        "AcquisitionDuration_s": old.get("AcquisitionDuration_s"),
        "RetainedResidualSourceOrigins": list(ctx["retained_origins"]),
        "FrozenNearAudit": anchor_audit,
        "ResidualTDInfo": rinfo,
        "SafeguardGate": gate,
        "ResidualPredictionDiagnostic": residual_diag,
        "SelectedGuard": guard,
        "SelectedAlpha": float(gate["alpha"]),
        "SelectedC": float(gate["c"]),
        "SelectedTau": float(gate["tau"]),
        "FallbackSelected": int(gate["FallbackSelected"]),
        "MaxAbsDiffInsideGuard": max_guard_diff,
        "Metrics": metrics,
        "FormalXGBRetrained": False,
        "Runtime_s": float(time.perf_counter() - t0),
        "Protocol": {
            "OuterLOBO": True,
            "V414BTimingScheduleReused": True,
            "MoESpecificTimingSearch": False,
            "V414BFrozenNearPredictionReusedExactly": True,
            "FormalHeldoutXGBRetrained": False,
            "SourceResidualTargetsBatteryCrossFitted": True,
            "HeldoutUsedForResidualTargetConstruction": False,
            "HeldoutUsedForTraining": False,
            "HeldoutUsedForDeepEarlyStopping": False,
            "HeldoutUsedForSafeguardSelection": False,
            "HeldoutUsedForTemporaryGateXGB": False,
            "HistoricalSOHAsInput": False,
            "FutureHIAsInput": False,
            "AbsoluteCycleAsModelFeature": False,
            "TimingMetadataRelativeOnly": True,
            "HardGuardExactXGBInsideGuard": True,
            "FrozenXGBFallbackAvailable": True,
            "SameMeasurementBudgetOptVsUniform": True,
        },
    }

    np.savez_compressed(
        pp,
        true=np.asarray(y, np.float32),
        mask=np.asarray(mask, np.uint8),
        pred_FrozenXGB=p_xgb.astype(np.float32),
        pred_SafeResidualMoE=p_safe.astype(np.float32),
        pred_ResidualTD=p_resid.astype(np.float32),
        residual_weight=w.astype(np.float32),
        residual_correction=correction.astype(np.float32),
    )
    save_json(q, jp)

    del residual_td
    tf.keras.backend.clear_session()
    gc.collect()

    print(
        f"[V417] {held} {ROLE_LABEL[role]} schedule={schedule} "
        f"XGB={metrics['FrozenXGB']['RMSE']:.6f} "
        f"SafeMoE={metrics['SafeResidualMoE']['RMSE']:.6f} "
        f"guard={guard} alpha={gate['alpha']:.2f} "
        f"c={gate['c']} tau={gate['tau']} "
        f"fallback={gate['FallbackSelected']} "
        f"runtime={q['Runtime_s']/60:.1f}min",
        flush=True,
    )

def run_battery(ctx, held: str, args):
    for role in ROLES:
        run_fold_role(ctx, held, role, args)


# =============================================================================
# 7. Aggregation
# =============================================================================

def collect_outer(ctx, args) -> pd.DataFrame:
    rows = []
    for b in ctx["batteries"]:
        for role in ROLES:
            p = result_json(args, b, role)
            if not p.exists():
                continue
            q = load_json(p)
            for model in PRIMARY_MODELS:
                m = q["Metrics"][model]
                rows.append({
                    "OuterHeldoutBattery": b,
                    "Role": q["Role"],
                    "Model": model,
                    "Schedule": schedule_text(q["Schedule"]),
                    "RMSE": m["RMSE"],
                    "MAE": m["MAE"],
                    "Bias": m["Bias"],
                    "SSE": m["SSE"],
                    "SAE": m["SAE"],
                    "N": m["N"],
                    "DiagnosisCount": q["DiagnosisCount"],
                    "HI_ScalarInputs": q["HI_ScalarInputs"],
                    "PhysicalSensorCount": q["PhysicalSensorCount"],
                    "AcquisitionDuration_s": q["AcquisitionDuration_s"],
                    "Runtime_s": q["Runtime_s"],
                })
    return pd.DataFrame(rows)


def pooled_table(outer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, role), d in outer.groupby(["Model", "Role"]):
        n = int(d["N"].sum())
        rows.append({
            "Model": model,
            "Role": role,
            "Batteries": d["OuterHeldoutBattery"].nunique(),
            "N": n,
            "RMSE": float(np.sqrt(d["SSE"].sum() / n)),
            "MAE": float(d["SAE"].sum() / n),
            "BatteryMacroRMSE": float(d["RMSE"].mean()),
            "BatteryMacroMAE": float(d["MAE"].mean()),
            "MeanRuntimePerFoldRole_s": float(d["Runtime_s"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["Model", "Role"])


def timing_gain_table(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in PRIMARY_MODELS:
        d = pooled[pooled["Model"].eq(model)].set_index("Role")
        if not {"OPT", "UNIFORM"}.issubset(d.index):
            continue
        opt = d.loc["OPT"]
        uni = d.loc["UNIFORM"]
        rows.append({
            "Model": model,
            "OPT_RMSE": float(opt.RMSE),
            "UNIFORM_RMSE": float(uni.RMSE),
            "DeltaRMSE_OPT_minus_UNIFORM": float(opt.RMSE - uni.RMSE),
            "TimingGain_pct": float(
                (uni.RMSE - opt.RMSE) / uni.RMSE * 100.0
            ),
            "OPT_BatteryMacroRMSE": float(opt.BatteryMacroRMSE),
            "UNIFORM_BatteryMacroRMSE": float(uni.BatteryMacroRMSE),
            "Direction_OPT_Better": int(opt.RMSE < uni.RMSE),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(
    diff: np.ndarray,
    reps: int,
    seed: int,
) -> Dict[str, float]:
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    rng = np.random.default_rng(seed)
    boot = np.empty(int(reps), float)

    for i in range(int(reps)):
        idx = rng.integers(0, len(diff), len(diff))
        boot[i] = float(np.mean(diff[idx]))

    return {
        "ObservedMeanBatteryRMSEDifference": float(np.mean(diff)),
        "CI95_Low": float(np.quantile(boot, 0.025)),
        "CI95_High": float(np.quantile(boot, 0.975)),
        "P_OPT_Better": float(np.mean(boot < 0)),
        "Batteries": int(len(diff)),
        "BootstrapReps": int(reps),
    }


def bootstrap_table(outer: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for model in PRIMARY_MODELS:
        d = outer[outer["Model"].eq(model)]
        a = d[d["Role"].eq("OPT")].set_index("OuterHeldoutBattery")["RMSE"]
        b = d[d["Role"].eq("UNIFORM")].set_index("OuterHeldoutBattery")["RMSE"]
        common = sorted(set(a.index) & set(b.index))
        diff = np.asarray([a[x] - b[x] for x in common], float)
        rows.append({
            "Model": model,
            "Comparison": "OPT - UNIFORM",
            **paired_bootstrap(
                diff,
                args.bootstrap,
                stable_seed("BOOT", model),
            ),
            "FoldWinCount_OPT": int(np.sum(diff < 0)),
            "FoldTieCount": int(np.sum(np.isclose(diff, 0, atol=1e-12))),
            "FoldLossCount_OPT": int(np.sum(diff > 0)),
        })
    return pd.DataFrame(rows)


def horizon_table(ctx, args) -> pd.DataFrame:
    rows = []
    for b in ctx["batteries"]:
        for role in ROLES:
            p = pred_npz(args, b, role)
            if not p.exists():
                continue
            z = np.load(p)
            y = np.asarray(z["true"], float)
            mask = np.asarray(z["mask"]).astype(bool)
            lead = np.arange(1, len(y) + 1)

            for model in PRIMARY_MODELS:
                pred = np.asarray(z[f"pred_{model}"], float)
                for lo, hi, label in LEAD_BINS:
                    ok = (
                        mask
                        & (lead >= lo)
                        & (lead <= hi)
                        & np.isfinite(y)
                        & np.isfinite(pred)
                    )
                    if not ok.any():
                        continue
                    e = pred[ok] - y[ok]
                    rows.append({
                        "OuterHeldoutBattery": b,
                        "Role": ROLE_LABEL[role],
                        "Model": model,
                        "HorizonBin": label,
                        "N": int(ok.sum()),
                        "SSE": float(np.sum(e * e)),
                        "SAE": float(np.sum(np.abs(e))),
                        "RMSE": float(np.sqrt(np.mean(e * e))),
                    })

    d = pd.DataFrame(rows)
    out = []
    for (model, role, hb), g in d.groupby(["Model", "Role", "HorizonBin"]):
        n = int(g["N"].sum())
        out.append({
            "Model": model,
            "Role": role,
            "HorizonBin": hb,
            "Batteries": g["OuterHeldoutBattery"].nunique(),
            "N": n,
            "RMSE": float(np.sqrt(g["SSE"].sum() / n)),
            "MAE": float(g["SAE"].sum() / n),
            "BatteryMacroRMSE": float(g["RMSE"].mean()),
        })
    return pd.DataFrame(out)


def gate_table(ctx, args) -> pd.DataFrame:
    rows = []
    for b in ctx["batteries"]:
        for role in ROLES:
            p = result_json(args, b, role)
            if not p.exists():
                continue
            q = load_json(p)
            g = q["SafeguardGate"]
            rows.append({
                "OuterHeldoutBattery": b,
                "Role": q["Role"],
                "Candidate": g.get("Candidate"),
                "Guard": g.get("Guard"),
                "c": g.get("c"),
                "tau": g.get("tau"),
                "alpha": g.get("alpha"),
                "FallbackSelected": g.get("FallbackSelected"),
                "SourceOverallRMSE": g.get("OverallRMSE"),
                "SourceBaseOverallRMSE": g.get("BaseOverallRMSE"),
                "SourceFarRMSE": g.get("FarRMSE"),
                "SourceBaseFarRMSE": g.get("BaseFarRMSE"),
                "FarStartUsed": g.get("FarStartUsed"),
                "FeasibleCandidateCount": g.get("FeasibleCandidateCount"),
                "GateValidationBatteries": g.get("GateValidationBatteries"),
                "MaxAbsDiffInsideGuard": q.get("MaxAbsDiffInsideGuard"),
                "ResidualClip": q.get("ResidualTDInfo", {}).get("ResidualClip"),
            })
    return pd.DataFrame(rows)

def reference_anchor(ctx, args) -> pd.DataFrame:
    rows = []
    for b in ctx["batteries"]:
        for role in ROLES:
            p = old_npz(args, b, role)
            z = np.load(p)
            y = np.asarray(z["true"], float)
            pred = np.asarray(z["pred"], float)
            mask = np.asarray(z["mask"]).astype(bool)
            met = trajectory_stats(y, pred, mask)
            rows.append({
                "OuterHeldoutBattery": b,
                "Role": ROLE_LABEL[role],
                "ReferenceModel": "V4.14B_XGB_Strict80",
                **met,
            })
    return pd.DataFrame(rows)


def finalize(ctx, args):
    out = ensure_dir(args.output_root)

    outer = collect_outer(ctx, args)
    expected = EXPECTED_BATTERIES * 2 * len(PRIMARY_MODELS)
    if len(outer) != expected:
        raise RuntimeError(
            f"outer metric rows={len(outer)} expected={expected}"
        )

    save_csv(outer, out / "01_frozenXGB_safeResidualMoE_outer_metrics.csv")

    pooled = pooled_table(outer)
    save_csv(pooled, out / "02_frozenXGB_safeResidualMoE_pooled_metrics.csv")

    gain = timing_gain_table(pooled)
    save_csv(gain, out / "03_timing_gain_by_model.csv")

    boot = bootstrap_table(outer, args)
    save_csv(boot, out / "04_paired_battery_bootstrap.csv")

    horizon = horizon_table(ctx, args)
    save_csv(horizon, out / "05_horizon_metrics.csv")

    gates = gate_table(ctx, args)
    save_csv(gates, out / "06_safeguard_gate_summary.csv")

    guard_dist = (
        gates.groupby(
            ["Role", "Guard", "FallbackSelected"],
            dropna=False,
        )
        .size()
        .reset_index(name="Batteries")
        .sort_values(
            ["Role", "FallbackSelected", "Guard"]
        )
    )
    save_csv(
        guard_dist,
        out / "07_guard_fallback_distribution.csv",
    )

    near_audit = gates[
        [
            "OuterHeldoutBattery",
            "Role",
            "Guard",
            "alpha",
            "FallbackSelected",
            "MaxAbsDiffInsideGuard",
        ]
    ].copy()
    near_audit["NearGuardInvariantPASS"] = (
        pd.to_numeric(
            near_audit["MaxAbsDiffInsideGuard"],
            errors="coerce",
        ).fillna(np.inf) <= 1e-8
    ).astype(int)
    save_csv(
        near_audit,
        out / "08_near_guard_invariance.csv",
    )

    ref = reference_anchor(ctx, args)
    save_csv(ref, out / "09_frozen_xgb_cache_anchor.csv")

    # Scientific summary, not an integrity gate.
    scientific = gain.merge(
        boot[
            [
                "Model",
                "ObservedMeanBatteryRMSEDifference",
                "CI95_Low",
                "CI95_High",
                "P_OPT_Better",
                "FoldWinCount_OPT",
                "FoldLossCount_OPT",
            ]
        ],
        on="Model",
        how="left",
    )
    scientific["Bootstrap95Supports_OPT"] = (
        pd.to_numeric(scientific["CI95_High"], errors="coerce") < 0
    ).astype(int)

    direction_map = dict(
        zip(
            scientific["Model"].astype(str),
            pd.to_numeric(
                scientific["Direction_OPT_Better"],
                errors="coerce",
            ).fillna(0).astype(int),
        )
    )
    scientific["TimingSurvives_XGB_to_SafeResidualMoE"] = int(
        direction_map.get("FrozenXGB", 0) == 1
        and direction_map.get("SafeResidualMoE", 0) == 1
    )

    save_csv(
        scientific,
        out / "10_timing_survival_after_safeResidualMoE.csv",
    )

    gate = pd.DataFrame([
        {
            "Check": "Outer metric rows",
            "Observed": len(outer),
            "Expected": expected,
            "PASS": int(len(outer) == expected),
        },
        {
            "Check": "47 batteries per model-role",
            "Observed": int(
                outer.groupby(["Model", "Role"])["OuterHeldoutBattery"]
                .nunique().min()
            ),
            "Expected": 47,
            "PASS": int(
                outer.groupby(["Model", "Role"])["OuterHeldoutBattery"]
                .nunique().eq(47).all()
            ),
        },
        {
            "Check": "Both timing roles",
            "Observed": "|".join(sorted(outer["Role"].unique())),
            "Expected": "OPT|UNIFORM",
            "PASS": int(set(outer["Role"]) == {"OPT", "UNIFORM"}),
        },
        {
            "Check": "Two formal model families",
            "Observed": "|".join(sorted(outer["Model"].unique())),
            "Expected": "|".join(sorted(PRIMARY_MODELS)),
            "PASS": int(set(outer["Model"]) == set(PRIMARY_MODELS)),
        },
        {
            "Check": "Same 18 scalar budget",
            "Observed": "|".join(
                map(str, sorted(outer["HI_ScalarInputs"].unique()))
            ),
            "Expected": "18",
            "PASS": int(set(outer["HI_ScalarInputs"]) == {18}),
        },
        {
            "Check": "Finite RMSE",
            "Observed": int(np.isfinite(pd.to_numeric(outer["RMSE"])).sum()),
            "Expected": len(outer),
            "PASS": int(np.isfinite(pd.to_numeric(outer["RMSE"])).all()),
        },
        {
            "Check": "Exact Frozen-XGB inside selected hard guard",
            "Observed": float(
                pd.to_numeric(
                    gates["MaxAbsDiffInsideGuard"],
                    errors="coerce",
                ).max()
            ),
            "Expected": "<=1e-8",
            "PASS": int(
                pd.to_numeric(
                    gates["MaxAbsDiffInsideGuard"],
                    errors="coerce",
                ).fillna(np.inf).le(1e-8).all()
            ),
        },
        {
            "Check": "Source-only safeguard row count",
            "Observed": len(gates),
            "Expected": EXPECTED_BATTERIES * 2,
            "PASS": int(len(gates) == EXPECTED_BATTERIES * 2),
        },
        {
            "Check": "Origin80 retained",
            "Observed": "|".join(map(str, ctx["retained_origins"])),
            "Expected": "contains 80",
            "PASS": int(TEST_ORIGIN in ctx["retained_origins"]),
        },
    ])
    save_csv(gate, out / "11_acceptance_gate.csv")

    status = "PASS" if gate["PASS"].eq(1).all() else "CHECK"

    complete = {
        "Task": "RWTH48 V4.17 Safeguarded Residual MoE",
        "CodeVersion": CODE_VERSION,
        "Status": status,
        "PrimaryModels": list(PRIMARY_MODELS),
        "Roles": ["OPT", "UNIFORM"],
        "FrozenMeasurementBudget": {
            "DiagnosisCount": 3,
            "EventNodes": list(EVENT_NODES),
            "HIs": list(EVENT_HIS),
            "MeasuredScalars": 18,
            "PhysicalSensors": "V|I",
        },
        "RequestedSourceOrigins": args.source_origins,
        "RetainedSourceOrigins": list(ctx["retained_origins"]),
        "ScientificSummary": scientific.to_dict("records"),
        "Protocol": {
            "OuterLOBO": True,
            "V414BTimingSchedulesReused": True,
            "HeldoutUsedForTimingSelection": False,
            "HeldoutUsedForTraining": False,
            "HeldoutUsedForDeepEarlyStopping": False,
            "HeldoutUsedForMoEGateSelection": False,
            "HistoricalSOHAsInput": False,
            "FutureHIAsInput": False,
            "AbsoluteCycleAsModelFeature": False,
            "RelativeTimingMetadataAllowed": True,
            "SameBudgetOptVsUniform": True,
            "FormalHeldoutXGBRetrained": False,
            "FrozenV414BXGBPredictionReused": True,
            "StandaloneLSTMRepeated": False,
            "StandaloneBiLSTMRepeated": False,
            "TemporarySourceOnlyGateXGBAllowed": True,
            "SourceResidualTargetsBatteryCrossFitted": True,
            "HardGuardExactFrozenXGB": True,
            "FrozenXGBFallbackAvailable": True,
            "MoESpecificTimingSearch": False,
        },
        "InterpretationBoundary": (
            "V4.17 changes the hybrid logic after inspection of V4.16. "
            "Outer-LOBO is programmatically leakage-safe, but the same RWTH48 "
            "dataset is method-development evidence; external validation remains "
            "the strongest confirmation. Timing schedules remain V4.14B XGB-selected."
        ),
        "Acceptance": gate.to_dict("records"),
    }
    save_json(
        complete,
        out / "99_RWTH48_V4_17_COMPLETE.json",
    )

    print("\n" + "=" * 120)
    print("RWTH48 V4.17 FINALIZE |", status)
    print("=" * 120)
    print("\nTiming gain by model:")
    print(gain.to_string(index=False))
    print("\nPaired battery bootstrap:")
    print(boot.to_string(index=False))
    print("\nSafeguarded residual timing summary:")
    print(scientific.to_string(index=False))
    print("\nOutput:", out)
    print("=" * 120)


# =============================================================================
# 8. Parallel / resume
# =============================================================================

def state_dir(args) -> Path:
    return ensure_dir(args.state_root / args.run_id)


def done_marker(args, worker: int) -> Path:
    return state_dir(args) / f"worker_{worker:02d}.done.json"


def fail_marker(args, worker: int) -> Path:
    return state_dir(args) / f"worker_{worker:02d}.fail.json"


def write_done(args, owned):
    save_json(
        {
            "Worker": args.parallel_worker_id,
            "OwnedBatteries": list(owned),
            "Status": "PASS",
        },
        done_marker(args, args.parallel_worker_id),
    )


def write_fail(args, exc: Exception):
    save_json(
        {
            "Worker": args.parallel_worker_id,
            "Status": "FAIL",
            "Exception": repr(exc),
            "Traceback": traceback.format_exc(),
        },
        fail_marker(args, args.parallel_worker_id),
    )


def barrier_finalize(ctx, args):
    if args.parallel_worker_id != 0:
        return

    print("\nWorker 0 waiting for all workers...", flush=True)
    while True:
        fails = [
            fail_marker(args, w)
            for w in range(args.parallel_workers)
            if fail_marker(args, w).exists()
        ]
        if fails:
            raise RuntimeError(
                "Parallel worker failed:\n" + "\n".join(map(str, fails))
            )

        if all(
            done_marker(args, w).exists()
            for w in range(args.parallel_workers)
        ):
            break

        time.sleep(max(2.0, float(args.barrier_poll_s)))

    finalize(ctx, args)


# =============================================================================
# 9. Main
# =============================================================================

def main():
    args = parse_args()
    ensure_dir(args.output_root)
    ensure_dir(args.state_root)

    ctx = preflight(args)

    if args.parallel_worker_id == 0:
        save_csv(
            ctx["preflight_v416"],
            args.output_root / "00_preflight.csv",
        )
        save_csv(
            ctx["origin_audit"],
            args.output_root / "00_source_origin_availability.csv",
        )
        save_json(
            {
                "CodeVersion": CODE_VERSION,
                "FormalModels": list(PRIMARY_MODELS),
                "FormalXGBRetraining": False,
                "StandaloneLSTM": False,
                "StandaloneBiLSTM": False,
                "RequestedSourceOrigins": args.source_origins,
                "RetainedSourceOrigins": list(ctx["retained_origins"]),
                "DeepValidationBatteries": args.deep_validation_batteries,
                "GateValidationBatteries": args.gate_validation_batteries,
                "DeepMaxEpochs": args.deep_max_epochs,
                "DeepPatience": args.deep_patience,
                "DeepBatch": args.deep_batch,
                "XGBThreads": args.xgb_threads,
                "XGBLeadsPerOrigin": args.xgb_leads_per_origin,
                "ResidualCVFolds": args.residual_cv_folds,
                "ResidualClipQuantile": args.residual_clip_quantile,
                "ResidualClipMin": args.residual_clip_min,
                "ResidualClipMax": args.residual_clip_max,
                "GuardGrid": args.guard_grid,
                "GateCGrid": args.gate_c_grid,
                "GateTauGrid": args.gate_tau_grid,
                "AlphaGrid": args.alpha_grid,
                "NoHarmHorizons": args.no_harm_horizons,
                "NoHarmEpsilon": args.no_harm_epsilon,
                "OverallNoHarmEpsilon": args.overall_no_harm_epsilon,
                "FarObjectiveStart": args.far_objective_start,
                "ParallelWorkers": args.parallel_workers,
                "RunID": args.run_id,
            },
            args.output_root / "00_config.json",
        )

    if args.stage == "audit":
        print(ctx["preflight_v416"].to_string(index=False))
        print("\nSource-origin availability:")
        print(ctx["origin_audit"].to_string(index=False))
        print("\nAUDIT PASS")
        return

    if args.stage == "finalize":
        if args.parallel_worker_id != 0:
            raise RuntimeError("finalize must use worker 0")
        finalize(ctx, args)
        return

    bats = list(ctx["batteries"])
    if args.only_battery:
        want = str(args.only_battery).strip()
        bats = [b for b in bats if b == want]
        if not bats:
            raise RuntimeError(f"battery not found: {want}")

    owned = [
        b
        for i, b in enumerate(bats)
        if i % args.parallel_workers == args.parallel_worker_id
    ]

    print(
        f"Worker {args.parallel_worker_id}/{args.parallel_workers} "
        f"owns {len(owned)} batteries | source origins={ctx['retained_origins']}",
        flush=True,
    )

    try:
        for i, b in enumerate(owned, 1):
            print(f"\n[{i}/{len(owned)}] {b}", flush=True)
            run_battery(ctx, b, args)

        if args.only_battery:
            print("\nSmoke/only-battery run complete; no global finalize.")
            return

        write_done(args, owned)

        if args.stage == "all":
            barrier_finalize(ctx, args)

    except Exception as exc:
        write_fail(args, exc)
        raise


if __name__ == "__main__":
    main()

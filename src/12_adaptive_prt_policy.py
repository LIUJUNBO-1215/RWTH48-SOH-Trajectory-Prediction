# -*- coding: utf-8 -*-
r"""
RWTH48 V4.15 | Budget-Aware Adaptive Diagnostic Scheduling | FIX1
================================================================

Primary scientific question
---------------------------
Given the diagnostic information already acquired from a held-out battery:

    1) Is another diagnosis still worth its measurement cost?
    2) If yes, at which future cycle should it be acquired?
    3) Can easy batteries stop after fewer diagnoses while difficult batteries
       receive additional measurements?

The primary platform is Event because V4.14B/V4.14C established statistically
supported timing sensitivity for Event under fixed measurement budgets.

Frozen measurement unit
-----------------------
Each Event diagnosis uses the V4.14C champion sensing unit:
    RefAh nodes : 0.05 / 0.09 / 0.13
    HIs         : V_TimeWeightedMean / P_Last
    scalars     : 3 nodes x 2 HIs = 6 measured HI scalars / diagnosis
    sensors     : V + I (I is also required for RefAh localization)

Adaptive policy
---------------
* The first diagnosis cycle is chosen using SOURCE BATTERIES ONLY.
* After a diagnosis is physically acquired, the policy may use ONLY the held-out
  battery's ALREADY-ACQUIRED HI values to decide whether to STOP or where to
  acquire the NEXT diagnosis.
* When choosing a future diagnosis cycle, the held-out HI at that future cycle
  is NOT inspected.  Expected post-measurement risk is learned from source
  batteries as a conditional value-of-information model.
* Outer held-out SOH / future SOH / future HI are never used for policy
  selection or model training.

Source-only conditional-risk surrogate
--------------------------------------
For each candidate schedule state s:
    - build the measured Event tensor;
    - construct compact causal history features;
    - produce source-battery cross-fitted Ridge trajectory-proxy errors;
    - these per-source-battery errors are the state-risk labels.

At deployment state s:
    - fit a source-only risk model:
          current measured features -> current proxy risk
    - calibrate a conservative upper risk from source cross-fit residuals.
    - for every possible next cycle c:
          current measured features -> risk after future measurement at s+[c]
      The future held-out HI is NOT a predictor.
    - STOP if conservative current risk <= proxy-risk target.
    - Otherwise:
          * if an extension is predicted to meet the target, select the
            lowest expected acquisition-cost extension;
          * otherwise maximize expected risk reduction per incremental
            measurement cost;
          * if expected improvement is too small, stop economically.

Important interpretation
------------------------
The policy thresholds are PROXY-RISK targets, not mathematical guarantees that
the final XGBoost full-life RMSE equals the same number.  The final strict
outer-LOBO XGBoost evaluation determines the actual resource/accuracy Pareto.

Formal predictor
----------------
Once a held-out battery's adaptive schedule has been realized, the exact
V4.14B common full-life XGBoost trajectory head is trained using the other 46
source batteries measured at the same realized schedule.  Thus:
    adaptive policy selection = source-only + already-acquired held-out HIs
    final trajectory scoring  = strict held-out full-life SOH

Outputs include:
    adaptive schedule traces
    diagnosis-count distribution
    schedule frequency
    full-life / horizon-bin metrics
    resource usage
    paired battery bootstrap vs V4.14B Event Opt-L3 / Uniform-L3
    budget-accuracy Pareto
    acceptance gate
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
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 0. Constants
# =============================================================================

CODE_VERSION = "V415_BUDGET_AWARE_ADAPTIVE_DIAGNOSTIC_SCHEDULING_FIX1"

BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR = Path(__file__).resolve().parent

V414B_NAME = "10_schedule_and_resource_search.py"
COMMON_NAME = "08_fixedtime_common_pipeline.py"

V414B_ROOT_NAME = "RWTH48_V4_14B_FullLifeSOH_Timing_MinResource"
V414C_ROOT_NAME = "RWTH48_V4_14C_SOH_P0_Closeout"

OUTPUT_NAME = "RWTH48_V4_15_BudgetAware_Adaptive_Diagnostic_Scheduling"
STATE_NAME = "RWTH48_V4_15_BudgetAware_Adaptive_Diagnostic_Scheduling_STATE"

EXPECTED_BATTERIES = 47
ORIGIN = 80

EVENT_HIS = ("V_TimeWeightedMean", "P_Last")
EVENT_NODES = (0.05, 0.09, 0.13)

DEFAULT_CANDIDATE_CYCLES = (40, 44, 48, 52, 56, 60, 64, 68, 71, 72, 76, 80)
DEFAULT_PROXY_TARGETS = (0.008, 0.009, 0.010)

ENC = "utf-8-sig"
FAST_GZIP = {"method": "gzip", "compresslevel": 1}

LEAD_BINS = (
    (1, 100, "Lead001_100"),
    (101, 300, "Lead101_300"),
    (301, 500, "Lead301_500"),
    (501, 700, "Lead501_700"),
    (701, 1000, "Lead701_1000"),
)


# =============================================================================
# 1. CLI / IO
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWTH48 V4.15 Budget-Aware Adaptive Diagnostic Scheduling"
    )
    p.add_argument(
        "--stage",
        choices=["audit", "policy", "formal", "all", "finalize"],
        default="all",
    )

    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--code-dir", type=Path, default=CODE_DIR)
    p.add_argument("--v414b", type=Path, default=None)
    p.add_argument("--common", type=Path, default=None)
    p.add_argument("--old-v414b-root", type=Path, default=None)
    p.add_argument("--old-v414c-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--state-root", type=Path, default=None)

    p.add_argument("--event-ready", type=Path, default=None)
    p.add_argument("--fixedtime-ready", type=Path, default=None)
    p.add_argument("--dependency-registry", type=Path, default=None)

    # Adaptive scheduling domain.
    p.add_argument(
        "--candidate-cycles",
        nargs="+",
        type=int,
        default=list(DEFAULT_CANDIDATE_CYCLES),
    )
    p.add_argument("--max-diagnoses", type=int, default=3)
    p.add_argument("--min-diagnoses", type=int, default=1)
    p.add_argument(
        "--proxy-risk-targets",
        nargs="+",
        type=float,
        default=list(DEFAULT_PROXY_TARGETS),
    )

    # Source-only proxy trajectory and conditional risk model.
    p.add_argument(
        "--proxy-leads",
        nargs="+",
        type=int,
        default=[10, 25, 50, 100, 200, 300, 500, 700, 900],
    )
    p.add_argument("--proxy-ridge-alpha", type=float, default=8.0)
    p.add_argument("--proxy-crossfit-folds", type=int, default=5)

    p.add_argument("--risk-model-alpha", type=float, default=5.0)
    p.add_argument("--risk-model-folds", type=int, default=5)
    p.add_argument("--risk-model-coverage", type=float, default=0.90)
    p.add_argument("--initial-source-quantile", type=float, default=0.80)

    # Economic decision rule.
    p.add_argument("--min-expected-gain", type=float, default=0.00010)
    p.add_argument("--scalar-cost-weight", type=float, default=1.0)
    p.add_argument("--duration-cost-weight", type=float, default=0.25)
    p.add_argument("--duration-normalizer-s", type=float, default=220.0)

    # Formal frozen full-life XGB.
    p.add_argument("--xgb-threads", type=int, default=3)
    p.add_argument("--cpu-threads-per-worker", type=int, default=4)
    p.add_argument("--xgb-leads-per-origin", type=int, default=160)

    # V4.14B helper compatibility.
    p.add_argument("--cycle-start", type=int, default=40)
    p.add_argument("--cycle-end", type=int, default=80)
    p.add_argument("--cycle-step", type=int, default=4)
    p.add_argument("--proxy-validation-batteries", type=int, default=9)
    p.add_argument("--local-refine-radius", type=int, default=3)
    p.add_argument("--local-refine-rounds", type=int, default=2)
    p.add_argument("--hi-ranking-shortlist", type=int, default=12)

    # Execution / inference.
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--force", action="store_true")

    p.add_argument("--parallel-workers", type=int, default=1)
    p.add_argument("--parallel-worker-id", type=int, default=0)
    p.add_argument("--run-id", type=str, default="V415_ADAPTIVE_20260826")
    p.add_argument("--barrier-poll-s", type=float, default=10.0)
    p.add_argument("--checkpoint-every-schedules", type=int, default=20)

    a = p.parse_args()

    a.base = Path(a.base)
    a.code_dir = Path(a.code_dir)

    a.v414b = (
        Path(a.v414b)
        if a.v414b
        else a.code_dir / V414B_NAME
    )
    a.common = (
        Path(a.common)
        if a.common
        else a.code_dir / COMMON_NAME
    )
    a.old_v414b_root = (
        Path(a.old_v414b_root)
        if a.old_v414b_root
        else a.base / V414B_ROOT_NAME
    )
    a.old_v414c_root = (
        Path(a.old_v414c_root)
        if a.old_v414c_root
        else a.base / V414C_ROOT_NAME
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

    cycles = sorted(
        set(
            int(x)
            for x in a.candidate_cycles
            if a.cycle_start <= int(x) <= ORIGIN
        )
    )
    if not cycles:
        p.error("--candidate-cycles is empty")
    if cycles[-1] != ORIGIN:
        cycles.append(ORIGIN)
    a.candidate_cycles = cycles

    if a.cycle_end != ORIGIN:
        p.error("--cycle-end must remain 80")
    if not (1 <= a.min_diagnoses <= a.max_diagnoses):
        p.error("require 1 <= min-diagnoses <= max-diagnoses")
    if a.max_diagnoses > len(a.candidate_cycles):
        p.error("max-diagnoses exceeds number of candidate cycles")

    a.proxy_risk_targets = sorted(
        set(float(x) for x in a.proxy_risk_targets if 0 < float(x) < 0.1)
    )
    if not a.proxy_risk_targets:
        p.error("--proxy-risk-targets is empty")

    if not (0.50 <= a.risk_model_coverage < 1.0):
        p.error("--risk-model-coverage must be in [0.5,1)")
    if not (0.50 <= a.initial_source_quantile < 1.0):
        p.error("--initial-source-quantile must be in [0.5,1)")
    if a.proxy_crossfit_folds < 2 or a.risk_model_folds < 2:
        p.error("crossfit folds must be >=2")
    if a.parallel_workers < 1:
        p.error("--parallel-workers must be >=1")
    if not (0 <= a.parallel_worker_id < a.parallel_workers):
        p.error("invalid --parallel-worker-id")
    if a.bootstrap < 100:
        p.error("--bootstrap should be >=100")

    return a


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, p: Path, compression=None) -> None:
    ensure_dir(p.parent)
    kw = {"index": False, "encoding": ENC}
    if compression is not None:
        kw["compression"] = compression
    df.to_csv(p, **kw)


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
        if isinstance(x, set):
            return sorted(x)
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


def stable_seed(*items) -> int:
    h = hashlib.sha1("|".join(map(str, items)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def normalize_target_id(x: float) -> str:
    return f"PRT_{float(x):.4f}".replace(".", "p")


def schedule_id(schedule: Sequence[int]) -> str:
    return "_".join(map(str, schedule))


def schedule_text(schedule: Sequence[int]) -> str:
    return "|".join(map(str, schedule))


def parse_schedule_text(x: str) -> Tuple[int, ...]:
    return tuple(int(v) for v in str(x).split("|") if str(v).strip())


def finite_quantile(x: Sequence[float], q: float) -> float:
    z = np.asarray(list(x), float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return np.nan
    try:
        return float(np.quantile(z, q, method="higher"))
    except TypeError:
        return float(np.quantile(z, q, interpolation="higher"))


# =============================================================================
# 2. V4.14B compatibility / preflight
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
        force=args.force,
        skip_figures=True,
        dpi=200,

        parallel_workers=1,
        parallel_worker_id=0,
        run_id=args.run_id,
        barrier_poll_s=args.barrier_poll_s,
    )


def baseline_timing_json(
    args,
    held: str,
    L: int,
    role: str,
) -> Path:
    return (
        args.old_v414b_root
        / "folds"
        / held
        / "timing"
        / f"Event_L{L}_LAST80_{role}.json"
    )


def baseline_timing_npz(
    args,
    held: str,
    L: int,
    role: str,
) -> Path:
    return (
        args.old_v414b_root
        / "folds"
        / held
        / "timing"
        / f"Event_L{L}_LAST80_{role}.npz"
    )


def preflight(args):
    if not args.v414b.exists():
        raise FileNotFoundError(args.v414b)
    if not args.common.exists():
        raise FileNotFoundError(args.common)
    if not args.old_v414b_root.exists():
        raise FileNotFoundError(args.old_v414b_root)

    c14b_path = args.old_v414b_root / "99_RWTH48_V4_14B_COMPLETE.json"
    if not c14b_path.exists():
        raise FileNotFoundError(c14b_path)
    c14b = load_json(c14b_path)
    if c14b.get("Status") != "PASS":
        raise RuntimeError("V4.14B status != PASS")

    c14c_path = args.old_v414c_root / "99_RWTH48_V4_14C_COMPLETE.json"
    c14c = None
    if c14c_path.exists():
        c14c = load_json(c14c_path)
        if c14c.get("Status") != "PASS":
            raise RuntimeError("V4.14C status != PASS")

    B = load_module(
        args.v414b,
        "rwth48_v414b_for_v415",
    )

    bargs = make_b_args(args)
    ctx = B.preflight(bargs)

    bats = list(ctx["batteries"])
    if len(bats) != EXPECTED_BATTERIES:
        raise RuntimeError(f"batteries={len(bats)} != 47")

    for h in EVENT_HIS:
        if h not in ctx["event_cube"].hmap:
            raise RuntimeError(f"Event HI missing: {h}")

    if not np.allclose(
        np.asarray(ctx["event_cube"].nodes, float),
        np.asarray(EVENT_NODES, float),
        atol=1e-9,
        rtol=0,
    ):
        raise RuntimeError(
            f"Event nodes mismatch: {ctx['event_cube'].nodes}"
        )

    missing = []
    for b in bats:
        for p in (
            baseline_timing_json(args, b, 3, "BEST_SOURCE_PROXY"),
            baseline_timing_npz(args, b, 3, "BEST_SOURCE_PROXY"),
            baseline_timing_json(args, b, 3, "UNIFORM"),
            baseline_timing_npz(args, b, 3, "UNIFORM"),
        ):
            if not p.exists():
                missing.append(str(p))

    pre = pd.DataFrame([
        {
            "Check": "V4.14B PASS",
            "Observed": c14b.get("Status"),
            "Expected": "PASS",
            "PASS": int(c14b.get("Status") == "PASS"),
        },
        {
            "Check": "V4.14C PASS",
            "Observed": c14c.get("Status") if c14c else "NOT_FOUND",
            "Expected": "PASS preferred",
            "PASS": int(c14c is None or c14c.get("Status") == "PASS"),
        },
        {
            "Check": "47 outer batteries",
            "Observed": len(bats),
            "Expected": EXPECTED_BATTERIES,
            "PASS": int(len(bats) == EXPECTED_BATTERIES),
        },
        {
            "Check": "Event champion HIs",
            "Observed": "|".join(EVENT_HIS),
            "Expected": "present",
            "PASS": int(all(h in ctx["event_cube"].hmap for h in EVENT_HIS)),
        },
        {
            "Check": "Event champion nodes",
            "Observed": "|".join(map(str, ctx["event_cube"].nodes)),
            "Expected": "|".join(map(str, EVENT_NODES)),
            "PASS": 1,
        },
        {
            "Check": "Baseline Opt/Uniform caches",
            "Observed": len(missing),
            "Expected": 0,
            "PASS": int(len(missing) == 0),
        },
        {
            "Check": "Candidate cycle bounds",
            "Observed": f"{min(args.candidate_cycles)}..{max(args.candidate_cycles)}",
            "Expected": f"{args.cycle_start}..{ORIGIN}",
            "PASS": int(
                min(args.candidate_cycles) >= args.cycle_start
                and max(args.candidate_cycles) <= ORIGIN
            ),
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
            "V4.15 preflight failed:\n"
            + pre[pre["PASS"].eq(0)].to_string(index=False)
        )

    ctx.update(
        {
            "B": B,
            "bargs": bargs,
            "v414b_complete": c14b,
            "v414c_complete": c14c,
        }
    )
    return ctx, pre


# =============================================================================
# 3. Adaptive state space
# =============================================================================

def all_candidate_schedules(
    cycles: Sequence[int],
    max_diagnoses: int,
) -> List[Tuple[int, ...]]:
    out = []
    cc = tuple(sorted(set(map(int, cycles))))
    for L in range(1, int(max_diagnoses) + 1):
        out.extend(tuple(x) for x in itertools.combinations(cc, L))
    return out


def source_batteries(ctx, held: str) -> List[str]:
    return [b for b in ctx["batteries"] if b != held]


def split_groups(
    items: Sequence[str],
    k: int,
    salt: str,
) -> List[List[str]]:
    order = sorted(
        items,
        key=lambda b: hashlib.sha1(
            f"{salt}|{b}".encode("utf-8")
        ).hexdigest(),
    )
    k = max(2, min(int(k), len(order)))
    return [order[i::k] for i in range(k)]


def compact_state_features(
    ctx,
    schedule: Sequence[int],
) -> np.ndarray:
    B = ctx["B"]
    cube = ctx["event_cube"]
    x = B.extract_tensor(
        cube,
        schedule,
        cube.nodes,
        EVENT_HIS,
    )
    if x is None:
        raise RuntimeError(
            f"incomplete Event state schedule={schedule}"
        )
    return B.compact_features(x, schedule)


# =============================================================================
# 4. Source-only proxy risk per state
# =============================================================================

def state_cache_path(args, held: str) -> Path:
    return (
        ensure_dir(
            args.output_root
            / "folds"
            / held
            / "policy"
        )
        / "00_proxy_state_risks.csv.gz"
    )


def crossfit_proxy_battery_risk(
    ctx,
    held: str,
    schedule: Sequence[int],
    args,
) -> pd.DataFrame:
    """
    Produce one source-only cross-fitted trajectory-proxy RMSE per source battery.

    This is NOT the formal outer XGB score.  It is an inner/source-only
    difficulty label used to learn the adaptive policy.
    """
    B = ctx["B"]
    source = source_batteries(ctx, held)
    bmap = {
        b: i
        for i, b in enumerate(ctx["batteries"])
    }

    X = compact_state_features(ctx, schedule)
    groups = split_groups(
        source,
        args.proxy_crossfit_folds,
        f"PROXY_STATE|{held}|{schedule_text(schedule)}",
    )

    rows = []
    for fi, val_bats in enumerate(groups, 1):
        valset = set(val_bats)
        train_bats = [
            b for b in source
            if b not in valset
        ]

        tr = [bmap[b] for b in train_bats]
        Xtr, ytr = B._long_proxy_rows(
            X,
            tr,
            ctx["Y80"],
            ctx["M80"],
            args.proxy_leads,
        )

        sc = StandardScaler()
        Atr = sc.fit_transform(Xtr)

        model = Ridge(
            alpha=float(args.proxy_ridge_alpha),
            solver="lsqr",
        )
        model.fit(Atr, ytr)

        for b in val_bats:
            bi = bmap[b]
            Xva, yva = B._long_proxy_rows(
                X,
                [bi],
                ctx["Y80"],
                ctx["M80"],
                args.proxy_leads,
            )
            pred = np.asarray(
                model.predict(sc.transform(Xva)),
                float,
            )
            e = pred - np.asarray(yva, float)
            rows.append(
                {
                    "OuterHeldoutBattery": held,
                    "Schedule": schedule_text(schedule),
                    "DiagnosisCount": len(schedule),
                    "SourceBattery": b,
                    "ProxyFold": fi,
                    "ProxyRMSE": float(
                        np.sqrt(np.mean(e * e))
                    ),
                    "ProxyMAE": float(
                        np.mean(np.abs(e))
                    ),
                    "ProxyRows": int(len(e)),
                }
            )

    d = pd.DataFrame(rows)
    if d["SourceBattery"].nunique() != len(source):
        raise RuntimeError(
            f"{held}/{schedule}: source proxy coverage "
            f"{d['SourceBattery'].nunique()} != {len(source)}"
        )
    return d


def build_state_risk_cache(
    ctx,
    held: str,
    args,
) -> pd.DataFrame:
    p = state_cache_path(args, held)

    if p.exists() and not args.force:
        old = pd.read_csv(
            p,
            compression="infer",
            low_memory=False,
        )
    else:
        old = pd.DataFrame()

    completed = (
        set(old["Schedule"].astype(str))
        if len(old) and "Schedule" in old.columns
        else set()
    )

    schedules = all_candidate_schedules(
        args.candidate_cycles,
        args.max_diagnoses,
    )

    frames = [old] if len(old) else []
    newly = 0

    for si, s in enumerate(schedules, 1):
        sid = schedule_text(s)
        if sid in completed:
            continue

        d = crossfit_proxy_battery_risk(
            ctx,
            held,
            s,
            args,
        )
        frames.append(d)
        completed.add(sid)
        newly += 1

        if (
            newly % max(1, int(args.checkpoint_every_schedules))
            == 0
        ):
            cur = pd.concat(
                frames,
                ignore_index=True,
            )
            save_csv(
                cur,
                p,
                compression=FAST_GZIP,
            )
            print(
                f"[{held}] proxy state cache "
                f"{len(completed)}/{len(schedules)} schedules",
                flush=True,
            )

    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )

    if out.empty:
        raise RuntimeError(
            f"{held}: proxy state cache empty"
        )

    # Deduplicate safe-resume appends.
    out = out.drop_duplicates(
        ["Schedule", "SourceBattery"],
        keep="last",
    ).sort_values(
        ["DiagnosisCount", "Schedule", "SourceBattery"]
    ).reset_index(drop=True)

    need = len(schedules) * (len(ctx["batteries"]) - 1)
    if len(out) != need:
        missing_sched = sorted(
            set(schedule_text(s) for s in schedules)
            - set(out["Schedule"].astype(str))
        )
        raise RuntimeError(
            f"{held}: proxy cache rows={len(out)} expected={need}; "
            f"missing schedules={missing_sched[:10]}"
        )

    save_csv(
        out,
        p,
        compression=FAST_GZIP,
    )
    return out


def schedule_risk_rows(
    risk_cache: pd.DataFrame,
    schedule: Sequence[int],
) -> pd.DataFrame:
    sid = schedule_text(schedule)
    d = risk_cache[
        risk_cache["Schedule"].astype(str).eq(sid)
    ].copy()
    if d.empty:
        raise RuntimeError(
            f"risk cache missing schedule {sid}"
        )
    return d


def schedule_source_risk_summary(
    risk_cache: pd.DataFrame,
    schedule: Sequence[int],
    args,
) -> Dict[str, float]:
    d = schedule_risk_rows(
        risk_cache,
        schedule,
    )
    r = pd.to_numeric(
        d["ProxyRMSE"],
        errors="coerce",
    ).to_numpy(float)
    return {
        "ProxyRiskMean": float(np.nanmean(r)),
        "ProxyRiskMedian": float(np.nanmedian(r)),
        "ProxyRiskQ": finite_quantile(
            r,
            float(args.initial_source_quantile),
        ),
        "ProxyRiskMax": float(np.nanmax(r)),
    }


# =============================================================================
# 5. Conditional risk / value-of-information model
# =============================================================================

def fit_conditional_risk(
    ctx,
    held: str,
    current_schedule: Sequence[int],
    target_schedule: Sequence[int],
    risk_cache: pd.DataFrame,
    args,
    salt: str,
) -> Dict[str, float]:
    """
    Predict target-schedule proxy risk from CURRENT measured features.

    Crucially:
      X = held-out HIs only at current_schedule.
      y = source-battery risks after target_schedule.
    The held-out HIs at target_schedule minus current_schedule are never read.
    """
    source = source_batteries(ctx, held)
    bmap = {
        b: i
        for i, b in enumerate(ctx["batteries"])
    }

    Xall = np.asarray(
        compact_state_features(
            ctx,
            current_schedule,
        ),
        float,
    )

    dtarget = schedule_risk_rows(
        risk_cache,
        target_schedule,
    )
    ymap = dict(
        zip(
            dtarget["SourceBattery"].astype(str),
            pd.to_numeric(
                dtarget["ProxyRMSE"],
                errors="coerce",
            ),
        )
    )

    y = np.asarray(
        [float(ymap[b]) for b in source],
        float,
    )
    X = np.vstack(
        [Xall[bmap[b]] for b in source]
    )

    # Log-risk target improves positivity and relative calibration.
    eps = 1e-6
    ly = np.log(np.maximum(y, eps))

    groups = split_groups(
        source,
        args.risk_model_folds,
        f"RISK_MODEL|{held}|{salt}",
    )

    oof = np.full(len(source), np.nan, float)
    pos = {b: i for i, b in enumerate(source)}

    for val_bats in groups:
        valset = set(val_bats)
        tr_bats = [
            b for b in source
            if b not in valset
        ]
        tr_idx = [pos[b] for b in tr_bats]
        va_idx = [pos[b] for b in val_bats]

        sc = StandardScaler()
        A = sc.fit_transform(X[tr_idx])
        Bv = sc.transform(X[va_idx])

        m = Ridge(
            alpha=float(args.risk_model_alpha),
            solver="lsqr",
        )
        m.fit(A, ly[tr_idx])

        pp = np.exp(
            np.asarray(m.predict(Bv), float)
        )
        oof[np.asarray(va_idx, int)] = pp

    if not np.isfinite(oof).all():
        raise RuntimeError(
            f"{held}/{current_schedule}->{target_schedule}: "
            "risk-model OOF incomplete"
        )

    residual = np.abs(oof - y)
    q = finite_quantile(
        residual,
        float(args.risk_model_coverage),
    )

    sc = StandardScaler()
    A = sc.fit_transform(X)
    model = Ridge(
        alpha=float(args.risk_model_alpha),
        solver="lsqr",
    )
    model.fit(A, ly)

    xh = Xall[bmap[held]][None, :]
    pred = float(
        np.exp(
            model.predict(
                sc.transform(xh)
            )[0]
        )
    )

    upper = float(pred + q)

    return {
        "PredictedRisk": pred,
        "RiskCalibrationQ": q,
        "RiskUpper": upper,
        "RiskModelOOFRMSE": float(
            np.sqrt(
                np.mean(
                    (oof - y) ** 2
                )
            )
        ),
        "RiskTargetScheduleMean": float(
            np.mean(y)
        ),
        "RiskTargetScheduleQ": finite_quantile(
            y,
            float(args.initial_source_quantile),
        ),
    }


# =============================================================================
# 6. Cost model
# =============================================================================

def source_cost(
    ctx,
    held: str,
    schedule: Sequence[int],
) -> Dict[str, float]:
    B = ctx["B"]
    bmap = {
        b: i
        for i, b in enumerate(ctx["batteries"])
    }
    src = [
        bmap[b]
        for b in source_batteries(ctx, held)
    ]
    return B.acquisition_cost(
        ctx["event_cube"],
        schedule,
        ctx["event_cube"].nodes,
        EVENT_HIS,
        ctx["depmap"],
        battery_indices=src,
    )


def heldout_cost(
    ctx,
    held: str,
    schedule: Sequence[int],
) -> Dict[str, float]:
    B = ctx["B"]
    bi = ctx["batteries"].index(held)
    return B.acquisition_cost(
        ctx["event_cube"],
        schedule,
        ctx["event_cube"].nodes,
        EVENT_HIS,
        ctx["depmap"],
        battery_indices=[bi],
    )


def incremental_cost_score(
    current_cost: Mapping[str, float],
    next_cost: Mapping[str, float],
    args,
) -> float:
    ds = (
        float(next_cost["HI_ScalarInputs"])
        - float(current_cost["HI_ScalarInputs"])
    )
    dd = (
        float(next_cost["AcquisitionDuration_s"])
        - float(current_cost["AcquisitionDuration_s"])
    )

    scalar_unit = max(
        1.0,
        float(len(EVENT_NODES) * len(EVENT_HIS)),
    )
    duration_unit = max(
        1.0,
        float(args.duration_normalizer_s),
    )

    score = (
        float(args.scalar_cost_weight)
        * max(ds, 0.0)
        / scalar_unit
        + float(args.duration_cost_weight)
        * max(dd, 0.0)
        / duration_unit
    )
    return float(max(score, 1e-6))


# =============================================================================
# 7. Source-only first diagnosis
# =============================================================================

def initial_plan_table(
    ctx,
    held: str,
    risk_cache: pd.DataFrame,
    proxy_target: float,
    args,
) -> pd.DataFrame:
    schedules = all_candidate_schedules(
        args.candidate_cycles,
        args.max_diagnoses,
    )

    rows = []
    for s in schedules:
        if len(s) < args.min_diagnoses:
            continue
        r = schedule_source_risk_summary(
            risk_cache,
            s,
            args,
        )
        c = source_cost(
            ctx,
            held,
            s,
        )
        rows.append(
            {
                "OuterHeldoutBattery": held,
                "ProxyRiskTarget": float(proxy_target),
                "Schedule": schedule_text(s),
                "DiagnosisCount": len(s),
                "FirstCycle": int(s[0]),
                "LastCycle": int(s[-1]),
                **r,
                **c,
                "SourceFeasible": int(
                    np.isfinite(r["ProxyRiskQ"])
                    and r["ProxyRiskQ"] <= float(proxy_target)
                ),
            }
        )

    d = pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError(
            f"{held}: initial source plan table empty"
        )
    return d


def choose_initial_source_plan(
    d: pd.DataFrame,
) -> pd.Series:
    feasible = d[d["SourceFeasible"].eq(1)].copy()

    if len(feasible):
        # Minimum number of diagnoses first; then minimum measurement burden;
        # then conservative source risk.
        q = feasible.sort_values(
            [
                "DiagnosisCount",
                "HI_ScalarInputs",
                "AcquisitionDuration_s",
                "ProxyRiskQ",
                "Schedule",
            ],
            ascending=[True, True, True, True, True],
        )
        return q.iloc[0]

    # If source proxy target is unattainable, use the best conservative risk,
    # then the lower measurement cost.
    q = d.sort_values(
        [
            "ProxyRiskQ",
            "DiagnosisCount",
            "HI_ScalarInputs",
            "AcquisitionDuration_s",
            "Schedule",
        ],
        ascending=[True, True, True, True, True],
    )
    return q.iloc[0]


# =============================================================================
# 8. Adaptive sequential policy
# =============================================================================

def policy_dir(args, held: str) -> Path:
    return ensure_dir(
        args.output_root
        / "folds"
        / held
        / "policy"
    )


def policy_json_path(
    args,
    held: str,
    proxy_target: float,
) -> Path:
    return (
        policy_dir(args, held)
        / f"{normalize_target_id(proxy_target)}_policy.json"
    )


def policy_trace_path(
    args,
    held: str,
    proxy_target: float,
) -> Path:
    return (
        policy_dir(args, held)
        / f"{normalize_target_id(proxy_target)}_trace.csv"
    )


def run_adaptive_policy(
    ctx,
    held: str,
    proxy_target: float,
    risk_cache: pd.DataFrame,
    args,
) -> Dict[str, object]:
    jp = policy_json_path(
        args,
        held,
        proxy_target,
    )
    tp = policy_trace_path(
        args,
        held,
        proxy_target,
    )

    if jp.exists() and tp.exists() and not args.force:
        return load_json(jp)

    plan = initial_plan_table(
        ctx,
        held,
        risk_cache,
        proxy_target,
        args,
    )
    source_plan = choose_initial_source_plan(plan)
    source_plan_schedule = parse_schedule_text(
        source_plan["Schedule"]
    )

    # The first diagnosis cannot be battery-adaptive because no held-out
    # measurements exist before it.
    current = (int(source_plan_schedule[0]),)

    trace = [
        {
            "OuterHeldoutBattery": held,
            "ProxyRiskTarget": float(proxy_target),
            "Step": 0,
            "StateSchedule": "",
            "CurrentCycle": np.nan,
            "Action": "SOURCE_ONLY_FIRST_DIAGNOSIS",
            "ChosenNextCycle": int(current[0]),
            "ResultingSchedule": schedule_text(current),
            "PredictedCurrentRisk": np.nan,
            "CurrentRiskUpper": np.nan,
            "PredictedNextRisk": np.nan,
            "NextRiskUpper": np.nan,
            "ExpectedRiskGain": np.nan,
            "IncrementalCostScore": np.nan,
            "DecisionReason": (
                "First cycle selected from source-only schedule-risk table; "
                "no held-out HI was available or used."
            ),
            "HeldoutFutureHIUsedForDecision": 0,
        }
    ]

    stop_reason = ""
    step = 1

    while True:
        current_model = fit_conditional_risk(
            ctx,
            held,
            current,
            current,
            risk_cache,
            args,
            salt=(
                f"CURRENT|{held}|{proxy_target}|"
                f"{schedule_text(current)}"
            ),
        )

        cur_upper = float(
            current_model["RiskUpper"]
        )

        # Accuracy-aware stop.
        if (
            len(current) >= args.min_diagnoses
            and cur_upper <= float(proxy_target)
        ):
            stop_reason = "RISK_TARGET_MET"
            trace.append(
                {
                    "OuterHeldoutBattery": held,
                    "ProxyRiskTarget": float(proxy_target),
                    "Step": step,
                    "StateSchedule": schedule_text(current),
                    "CurrentCycle": int(current[-1]),
                    "Action": "STOP",
                    "ChosenNextCycle": np.nan,
                    "ResultingSchedule": schedule_text(current),
                    "PredictedCurrentRisk": current_model["PredictedRisk"],
                    "CurrentRiskUpper": cur_upper,
                    "PredictedNextRisk": np.nan,
                    "NextRiskUpper": np.nan,
                    "ExpectedRiskGain": 0.0,
                    "IncrementalCostScore": 0.0,
                    "DecisionReason": "Conservative current risk <= proxy target.",
                    "HeldoutFutureHIUsedForDecision": 0,
                }
            )
            break

        if len(current) >= args.max_diagnoses:
            stop_reason = "MAX_DIAGNOSES"
            trace.append(
                {
                    "OuterHeldoutBattery": held,
                    "ProxyRiskTarget": float(proxy_target),
                    "Step": step,
                    "StateSchedule": schedule_text(current),
                    "CurrentCycle": int(current[-1]),
                    "Action": "STOP",
                    "ChosenNextCycle": np.nan,
                    "ResultingSchedule": schedule_text(current),
                    "PredictedCurrentRisk": current_model["PredictedRisk"],
                    "CurrentRiskUpper": cur_upper,
                    "PredictedNextRisk": np.nan,
                    "NextRiskUpper": np.nan,
                    "ExpectedRiskGain": 0.0,
                    "IncrementalCostScore": 0.0,
                    "DecisionReason": "Maximum diagnosis count reached.",
                    "HeldoutFutureHIUsedForDecision": 0,
                }
            )
            break

        future = [
            int(c)
            for c in args.candidate_cycles
            if int(c) > int(current[-1])
            and int(c) not in current
        ]

        if not future:
            stop_reason = "NO_FUTURE_CANDIDATE"
            trace.append(
                {
                    "OuterHeldoutBattery": held,
                    "ProxyRiskTarget": float(proxy_target),
                    "Step": step,
                    "StateSchedule": schedule_text(current),
                    "CurrentCycle": int(current[-1]),
                    "Action": "STOP",
                    "ChosenNextCycle": np.nan,
                    "ResultingSchedule": schedule_text(current),
                    "PredictedCurrentRisk": current_model["PredictedRisk"],
                    "CurrentRiskUpper": cur_upper,
                    "PredictedNextRisk": np.nan,
                    "NextRiskUpper": np.nan,
                    "ExpectedRiskGain": 0.0,
                    "IncrementalCostScore": 0.0,
                    "DecisionReason": "No future candidate cycle remains.",
                    "HeldoutFutureHIUsedForDecision": 0,
                }
            )
            break

        current_cost = source_cost(
            ctx,
            held,
            current,
        )

        cand = []
        for c in future:
            ext = tuple(
                list(current) + [int(c)]
            )

            ext_model = fit_conditional_risk(
                ctx,
                held,
                current,
                ext,
                risk_cache,
                args,
                salt=(
                    f"EXT|{held}|{proxy_target}|"
                    f"{schedule_text(current)}->{schedule_text(ext)}"
                ),
            )
            next_cost = source_cost(
                ctx,
                held,
                ext,
            )
            cscore = incremental_cost_score(
                current_cost,
                next_cost,
                args,
            )

            gain = (
                cur_upper
                - float(ext_model["RiskUpper"])
            )
            utility = gain / cscore

            cand.append(
                {
                    "Cycle": int(c),
                    "Schedule": ext,
                    "PredictedRisk": float(
                        ext_model["PredictedRisk"]
                    ),
                    "RiskUpper": float(
                        ext_model["RiskUpper"]
                    ),
                    "ExpectedGain": float(gain),
                    "IncrementalCostScore": float(cscore),
                    "Utility": float(utility),
                    "TotalDuration_s": float(
                        next_cost["AcquisitionDuration_s"]
                    ),
                    "TotalScalars": int(
                        next_cost["HI_ScalarInputs"]
                    ),
                }
            )

        feasible = [
            z for z in cand
            if z["RiskUpper"] <= float(proxy_target)
        ]

        if feasible:
            # Once a future measurement is predicted to meet the risk target,
            # choose the least measurement-burden extension rather than the
            # absolutely smallest predicted error.
            chosen = min(
                feasible,
                key=lambda z: (
                    z["TotalScalars"],
                    z["TotalDuration_s"],
                    z["RiskUpper"],
                    z["Cycle"],
                ),
            )
            reason = "FUTURE_TARGET_FEASIBLE_MIN_COST"
        else:
            chosen = max(
                cand,
                key=lambda z: (
                    z["Utility"],
                    z["ExpectedGain"],
                    -z["TotalDuration_s"],
                    -z["Cycle"],
                ),
            )
            reason = "MAX_EXPECTED_RISK_REDUCTION_PER_COST"

            if chosen["ExpectedGain"] < float(args.min_expected_gain):
                stop_reason = "ECONOMIC_STOP_LOW_EXPECTED_GAIN"
                trace.append(
                    {
                        "OuterHeldoutBattery": held,
                        "ProxyRiskTarget": float(proxy_target),
                        "Step": step,
                        "StateSchedule": schedule_text(current),
                        "CurrentCycle": int(current[-1]),
                        "Action": "STOP",
                        "ChosenNextCycle": np.nan,
                        "ResultingSchedule": schedule_text(current),
                        "PredictedCurrentRisk": current_model["PredictedRisk"],
                        "CurrentRiskUpper": cur_upper,
                        "PredictedNextRisk": chosen["PredictedRisk"],
                        "NextRiskUpper": chosen["RiskUpper"],
                        "ExpectedRiskGain": chosen["ExpectedGain"],
                        "IncrementalCostScore": chosen["IncrementalCostScore"],
                        "DecisionReason": (
                            "Best source-predicted future measurement has "
                            "insufficient expected gain for its cost."
                        ),
                        "HeldoutFutureHIUsedForDecision": 0,
                    }
                )
                break

        # WAIT to the chosen future cycle, then physically acquire that diagnosis.
        # No held-out HI at chosen Cycle was used in selecting it.
        new_current = tuple(chosen["Schedule"])

        trace.append(
            {
                "OuterHeldoutBattery": held,
                "ProxyRiskTarget": float(proxy_target),
                "Step": step,
                "StateSchedule": schedule_text(current),
                "CurrentCycle": int(current[-1]),
                "Action": "WAIT_AND_MEASURE",
                "ChosenNextCycle": int(chosen["Cycle"]),
                "ResultingSchedule": schedule_text(new_current),
                "PredictedCurrentRisk": current_model["PredictedRisk"],
                "CurrentRiskUpper": cur_upper,
                "PredictedNextRisk": chosen["PredictedRisk"],
                "NextRiskUpper": chosen["RiskUpper"],
                "ExpectedRiskGain": chosen["ExpectedGain"],
                "IncrementalCostScore": chosen["IncrementalCostScore"],
                "DecisionReason": reason,
                "HeldoutFutureHIUsedForDecision": 0,
            }
        )

        current = new_current
        step += 1

    final_cost_source = source_cost(
        ctx,
        held,
        current,
    )

    out = {
        "CodeVersion": CODE_VERSION,
        "OuterHeldoutBattery": held,
        "PolicyID": normalize_target_id(proxy_target),
        "ProxyRiskTarget": float(proxy_target),
        "CandidateCycles": list(map(int, args.candidate_cycles)),
        "MaxDiagnoses": int(args.max_diagnoses),
        "SourceOnlyInitialPlan": schedule_text(source_plan_schedule),
        "SourceOnlyInitialPlanRiskQ": float(
            source_plan["ProxyRiskQ"]
        ),
        "SourceOnlyInitialPlanFeasible": int(
            source_plan["SourceFeasible"]
        ),
        "AdaptiveSchedule": list(map(int, current)),
        "DiagnosisCount": len(current),
        "StopReason": stop_reason,
        "SourceExpectedCost": final_cost_source,
        "Protocol": {
            "HeldoutSOHUsedForPolicy": False,
            "HeldoutFutureSOHUsedForPolicy": False,
            "HeldoutFutureHIUsedForNextCycleSelection": False,
            "HeldoutAlreadyMeasuredHIUsedForAdaptation": True,
            "FirstDiagnosisSourceOnly": True,
            "SourceProxyCrossFitted": True,
            "RiskModelSourceOnly": True,
            "AbsoluteCycleAsXGBModelFeature": False,
            "HistoricalTrueSOHAsInput": False,
        },
    }

    save_csv(
        pd.DataFrame(trace),
        tp,
    )
    save_json(
        out,
        jp,
    )
    return out


def run_policy_battery(
    ctx,
    held: str,
    args,
) -> None:
    risk_cache = build_state_risk_cache(
        ctx,
        held,
        args,
    )

    for target in args.proxy_risk_targets:
        q = run_adaptive_policy(
            ctx,
            held,
            float(target),
            risk_cache,
            args,
        )
        print(
            f"[{held}] {q['PolicyID']} -> "
            f"{q['AdaptiveSchedule']} "
            f"L={q['DiagnosisCount']} "
            f"stop={q['StopReason']}",
            flush=True,
        )


# =============================================================================
# 9. Formal strict outer-LOBO XGB evaluation
# =============================================================================

def formal_dir(
    args,
    held: str,
) -> Path:
    return ensure_dir(
        args.output_root
        / "folds"
        / held
        / "formal"
    )


def formal_schedule_npz(
    args,
    held: str,
    schedule: Sequence[int],
) -> Path:
    return (
        formal_dir(args, held)
        / f"schedule_{schedule_id(schedule)}.npz"
    )


def formal_schedule_json(
    args,
    held: str,
    schedule: Sequence[int],
) -> Path:
    return (
        formal_dir(args, held)
        / f"schedule_{schedule_id(schedule)}.json"
    )


def evaluate_unique_schedule(
    ctx,
    held: str,
    schedule: Sequence[int],
    args,
) -> Dict[str, object]:
    jp = formal_schedule_json(
        args,
        held,
        schedule,
    )
    npz = formal_schedule_npz(
        args,
        held,
        schedule,
    )

    if jp.exists() and npz.exists() and not args.force:
        return load_json(jp)

    B = ctx["B"]

    formal = B.formal_outer_prediction(
        ctx,
        held,
        ctx["event_cube"],
        schedule,
        ctx["event_cube"].nodes,
        EVENT_HIS,
        ctx["bargs"],
    )

    cost = heldout_cost(
        ctx,
        held,
        schedule,
    )

    q = {
        "CodeVersion": CODE_VERSION,
        "OuterHeldoutBattery": held,
        "Schedule": list(map(int, schedule)),
        **cost,
        **{
            f"Fold_{k}": v
            for k, v in formal["Metrics"].items()
        },
        "SSE": formal["SSE"],
        "SAE": formal["SAE"],
        "N": formal["N"],
        "TrainRows": formal["TrainRows"],
        "FeatureCount": formal["FeatureCount"],
        "Protocol": {
            "HeldoutUsedForXGBTraining": False,
            "HistoricalSOHAsInput": False,
            "FutureHIAsInput": False,
            "CycleAsModelFeature": False,
        },
    }

    np.savez_compressed(
        npz,
        pred=formal["Pred"],
        true=formal["True"],
        mask=formal["Mask"],
    )
    save_json(q, jp)

    return q


def policy_formal_json_path(
    args,
    held: str,
    proxy_target: float,
) -> Path:
    return (
        formal_dir(args, held)
        / f"{normalize_target_id(proxy_target)}_formal.json"
    )


def run_formal_battery(
    ctx,
    held: str,
    args,
) -> None:
    for target in args.proxy_risk_targets:
        pp = policy_json_path(
            args,
            held,
            target,
        )
        if not pp.exists():
            raise FileNotFoundError(
                f"policy cache missing: {pp}. "
                "Run --stage policy first or use --stage all."
            )

        pol = load_json(pp)
        schedule = tuple(
            map(int, pol["AdaptiveSchedule"])
        )

        base = evaluate_unique_schedule(
            ctx,
            held,
            schedule,
            args,
        )

        q = {
            "CodeVersion": CODE_VERSION,
            "OuterHeldoutBattery": held,
            "PolicyID": pol["PolicyID"],
            "ProxyRiskTarget": float(target),
            "AdaptiveSchedule": list(schedule),
            "DiagnosisCount": len(schedule),
            "StopReason": pol["StopReason"],
            **{
                k: v
                for k, v in base.items()
                if k not in (
                    "CodeVersion",
                    "OuterHeldoutBattery",
                    "Schedule",
                    "Protocol",
                )
            },
            "Protocol": {
                **pol["Protocol"],
                **base["Protocol"],
            },
        }

        save_json(
            q,
            policy_formal_json_path(
                args,
                held,
                target,
            ),
        )

        print(
            f"[{held}] FORMAL {pol['PolicyID']} "
            f"schedule={schedule} "
            f"RMSE={q.get('Fold_RMSE', np.nan):.6f}",
            flush=True,
        )


# =============================================================================
# 10. Baseline loaders / metrics
# =============================================================================

def trajectory_sums(
    y: np.ndarray,
    p: np.ndarray,
    m: np.ndarray,
) -> Tuple[float, float, int]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.asarray(m).astype(bool)

    ok = (
        m
        & np.isfinite(y)
        & np.isfinite(p)
    )
    e = p[ok] - y[ok]

    return (
        float(np.sum(e * e)),
        float(np.sum(np.abs(e))),
        int(len(e)),
    )


def battery_rmse_from_npz(p: Path) -> float:
    z = np.load(p)
    pred = np.asarray(z["pred"], float)
    true = np.asarray(z["true"], float)
    mask = np.asarray(z["mask"]).astype(bool)
    ok = (
        mask
        & np.isfinite(pred)
        & np.isfinite(true)
    )
    if not ok.any():
        return np.nan
    return float(
        np.sqrt(
            np.mean(
                (pred[ok] - true[ok]) ** 2
            )
        )
    )


def horizon_rows(
    battery: str,
    config: str,
    npz_path: Path,
) -> List[Dict[str, object]]:
    z = np.load(npz_path)
    p = np.asarray(z["pred"], float)
    y = np.asarray(z["true"], float)
    m = np.asarray(z["mask"]).astype(bool)
    lead = np.arange(1, len(y) + 1)

    rows = []
    for lo, hi, label in LEAD_BINS:
        ok = (
            m
            & (lead >= lo)
            & (lead <= hi)
            & np.isfinite(y)
            & np.isfinite(p)
        )
        if not ok.any():
            continue
        e = p[ok] - y[ok]
        rows.append(
            {
                "OuterHeldoutBattery": battery,
                "Config": config,
                "HorizonBin": label,
                "N": int(ok.sum()),
                "SSE": float(np.sum(e * e)),
                "SAE": float(np.sum(np.abs(e))),
                "RMSE": float(np.sqrt(np.mean(e * e))),
                "MAE": float(np.mean(np.abs(e))),
            }
        )
    return rows


# =============================================================================
# 11. Aggregation / bootstrap
# =============================================================================

def collect_policy_rows(
    ctx,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    trace = []

    for b in ctx["batteries"]:
        for target in args.proxy_risk_targets:
            p = policy_json_path(
                args,
                b,
                target,
            )
            if p.exists():
                q = load_json(p)
                rows.append(
                    {
                        "OuterHeldoutBattery": b,
                        "PolicyID": q["PolicyID"],
                        "ProxyRiskTarget": q["ProxyRiskTarget"],
                        "AdaptiveSchedule": schedule_text(q["AdaptiveSchedule"]),
                        "DiagnosisCount": q["DiagnosisCount"],
                        "StopReason": q["StopReason"],
                        "HeldoutFutureHIUsedForNextCycleSelection": int(
                            q["Protocol"][
                                "HeldoutFutureHIUsedForNextCycleSelection"
                            ]
                        ),
                    }
                )

            tp = policy_trace_path(
                args,
                b,
                target,
            )
            if tp.exists():
                t = pd.read_csv(
                    tp,
                    low_memory=False,
                )
                trace.append(t)

    return (
        pd.DataFrame(rows),
        pd.concat(trace, ignore_index=True)
        if trace
        else pd.DataFrame(),
    )


def collect_formal_rows(
    ctx,
    args,
) -> pd.DataFrame:
    rows = []
    for b in ctx["batteries"]:
        for target in args.proxy_risk_targets:
            p = policy_formal_json_path(
                args,
                b,
                target,
            )
            if not p.exists():
                continue
            q = load_json(p)
            rows.append(
                {
                    "OuterHeldoutBattery": b,
                    "PolicyID": q["PolicyID"],
                    "ProxyRiskTarget": q["ProxyRiskTarget"],
                    "AdaptiveSchedule": schedule_text(q["AdaptiveSchedule"]),
                    "DiagnosisCount": q["DiagnosisCount"],
                    "StopReason": q["StopReason"],
                    "Fold_RMSE": q.get("Fold_RMSE", np.nan),
                    "Fold_MAE": q.get("Fold_MAE", np.nan),
                    "SSE": q["SSE"],
                    "SAE": q["SAE"],
                    "N": q["N"],
                    "HI_ScalarInputs": q["HI_ScalarInputs"],
                    "PhysicalSensorCount": q["PhysicalSensorCount"],
                    "AcquisitionDuration_s": q["AcquisitionDuration_s"],
                    "SensorChannelSeconds": q["SensorChannelSeconds"],
                    "HeldoutUsedForXGBTraining": int(
                        q["Protocol"]["HeldoutUsedForXGBTraining"]
                    ),
                    "HeldoutFutureHIUsedForNextCycleSelection": int(
                        q["Protocol"][
                            "HeldoutFutureHIUsedForNextCycleSelection"
                        ]
                    ),
                }
            )
    return pd.DataFrame(rows)


def pooled_metrics(
    formal: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if formal.empty:
        return pd.DataFrame()

    for pid, d in formal.groupby("PolicyID"):
        n = int(pd.to_numeric(d["N"]).sum())
        sse = float(pd.to_numeric(d["SSE"]).sum())
        sae = float(pd.to_numeric(d["SAE"]).sum())

        counts = pd.to_numeric(
            d["DiagnosisCount"],
            errors="coerce",
        )

        rows.append(
            {
                "PolicyID": pid,
                "ProxyRiskTarget": float(
                    d["ProxyRiskTarget"].iloc[0]
                ),
                "Batteries": d["OuterHeldoutBattery"].nunique(),
                "N": n,
                "RMSE": float(np.sqrt(sse / n)),
                "MAE": float(sae / n),
                "BatteryMacroRMSE": float(
                    pd.to_numeric(
                        d["Fold_RMSE"],
                        errors="coerce",
                    ).mean()
                ),
                "MeanDiagnoses": float(counts.mean()),
                "MedianDiagnoses": float(counts.median()),
                "PctL1": float((counts == 1).mean() * 100),
                "PctL2": float((counts == 2).mean() * 100),
                "PctL3": float((counts == 3).mean() * 100),
                "MeanScalars": float(
                    pd.to_numeric(
                        d["HI_ScalarInputs"],
                        errors="coerce",
                    ).mean()
                ),
                "MedianScalars": float(
                    pd.to_numeric(
                        d["HI_ScalarInputs"],
                        errors="coerce",
                    ).median()
                ),
                "MeanSensors": float(
                    pd.to_numeric(
                        d["PhysicalSensorCount"],
                        errors="coerce",
                    ).mean()
                ),
                "MeanDuration_s": float(
                    pd.to_numeric(
                        d["AcquisitionDuration_s"],
                        errors="coerce",
                    ).mean()
                ),
                "MedianDuration_s": float(
                    pd.to_numeric(
                        d["AcquisitionDuration_s"],
                        errors="coerce",
                    ).median()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["MeanScalars", "RMSE"]
    ).reset_index(drop=True)


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    reps: int,
    seed: int,
) -> Dict[str, float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]

    diff = a - b
    obs = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    boots = np.empty(int(reps), float)

    for i in range(int(reps)):
        idx = rng.integers(
            0,
            len(diff),
            len(diff),
        )
        boots[i] = float(
            np.mean(diff[idx])
        )

    return {
        "ObservedMeanDifference": obs,
        "CI95_Low": float(np.quantile(boots, 0.025)),
        "CI95_High": float(np.quantile(boots, 0.975)),
        "P_AdaptiveBetter": float(np.mean(boots < 0)),
        "Batteries": int(len(diff)),
        "BootstrapReps": int(reps),
    }


def baseline_battery_table(
    ctx,
    args,
) -> pd.DataFrame:
    rows = []
    for b in ctx["batteries"]:
        for role in (
            "BEST_SOURCE_PROXY",
            "UNIFORM",
        ):
            npz = baseline_timing_npz(
                args,
                b,
                3,
                role,
            )
            if not npz.exists():
                continue
            rows.append(
                {
                    "OuterHeldoutBattery": b,
                    "Baseline": (
                        "Event_OptL3"
                        if role == "BEST_SOURCE_PROXY"
                        else "Event_UniformL3"
                    ),
                    "BatteryRMSE": battery_rmse_from_npz(npz),
                }
            )
    return pd.DataFrame(rows)


def adaptive_horizon_table(
    ctx,
    formal: pd.DataFrame,
    args,
) -> pd.DataFrame:
    rows = []
    for r in formal.itertuples(index=False):
        schedule = parse_schedule_text(
            r.AdaptiveSchedule
        )
        npz = formal_schedule_npz(
            args,
            r.OuterHeldoutBattery,
            schedule,
        )
        rows.extend(
            horizon_rows(
                r.OuterHeldoutBattery,
                r.PolicyID,
                npz,
            )
        )

    d = pd.DataFrame(rows)
    if d.empty:
        return d

    out = []
    for (pid, hb), q in d.groupby(
        ["Config", "HorizonBin"]
    ):
        n = int(q["N"].sum())
        out.append(
            {
                "PolicyID": pid,
                "HorizonBin": hb,
                "Batteries": q["OuterHeldoutBattery"].nunique(),
                "N": n,
                "RMSE": float(
                    np.sqrt(
                        q["SSE"].sum() / n
                    )
                ),
                "MAE": float(
                    q["SAE"].sum() / n
                ),
                "BatteryMacroRMSE": float(
                    q["RMSE"].mean()
                ),
            }
        )
    return pd.DataFrame(out)


# =============================================================================
# 12. Finalize
# =============================================================================

def read_optional_csv(p: Path) -> pd.DataFrame:
    return (
        pd.read_csv(p, low_memory=False)
        if p.exists()
        else pd.DataFrame()
    )


def finalize(
    ctx,
    args,
) -> None:
    out = ensure_dir(args.output_root)

    policy, trace = collect_policy_rows(
        ctx,
        args,
    )
    formal = collect_formal_rows(
        ctx,
        args,
    )

    if formal.empty:
        raise RuntimeError(
            "No formal adaptive results found"
        )

    save_csv(
        policy,
        out / "01_adaptive_policy_outer.csv",
    )
    save_csv(
        trace,
        out / "02_adaptive_policy_trace.csv",
    )
    save_csv(
        formal,
        out / "03_adaptive_formal_outer_results.csv",
    )

    pooled = pooled_metrics(formal)
    save_csv(
        pooled,
        out / "04_adaptive_pooled_metrics.csv",
    )

    count_dist = (
        formal.groupby(
            ["PolicyID", "DiagnosisCount"]
        )
        .size()
        .reset_index(name="Batteries")
    )
    count_dist["Fraction"] = count_dist.groupby(
        "PolicyID"
    )["Batteries"].transform(
        lambda z: z / z.sum()
    )
    save_csv(
        count_dist,
        out / "05_diagnosis_count_distribution.csv",
    )

    sched_freq = (
        formal.groupby(
            [
                "PolicyID",
                "DiagnosisCount",
                "AdaptiveSchedule",
            ]
        )
        .size()
        .reset_index(name="OuterFoldCount")
        .sort_values(
            [
                "PolicyID",
                "OuterFoldCount",
                "AdaptiveSchedule",
            ],
            ascending=[True, False, True],
        )
    )
    save_csv(
        sched_freq,
        out / "06_adaptive_schedule_frequency.csv",
    )

    baseline = baseline_battery_table(
        ctx,
        args,
    )

    compare_rows = []
    boot_rows = []

    for pid, d in formal.groupby("PolicyID"):
        amap = dict(
            zip(
                d["OuterHeldoutBattery"],
                pd.to_numeric(
                    d["Fold_RMSE"],
                    errors="coerce",
                ),
            )
        )

        for bname, bdf in baseline.groupby("Baseline"):
            bmap = dict(
                zip(
                    bdf["OuterHeldoutBattery"],
                    pd.to_numeric(
                        bdf["BatteryRMSE"],
                        errors="coerce",
                    ),
                )
            )
            common = sorted(
                set(amap) & set(bmap)
            )
            aa = np.asarray(
                [amap[b] for b in common],
                float,
            )
            bb = np.asarray(
                [bmap[b] for b in common],
                float,
            )

            br = paired_bootstrap(
                aa,
                bb,
                args.bootstrap,
                stable_seed(
                    "BOOT",
                    pid,
                    bname,
                ),
            )
            boot_rows.append(
                {
                    "PolicyID": pid,
                    "Comparison": (
                        f"{pid} - {bname}"
                    ),
                    **br,
                }
            )
            compare_rows.append(
                {
                    "PolicyID": pid,
                    "Baseline": bname,
                    "AdaptiveBatteryMacroRMSE": float(
                        np.mean(aa)
                    ),
                    "BaselineBatteryMacroRMSE": float(
                        np.mean(bb)
                    ),
                    "MeanBatteryRMSEDifference": float(
                        np.mean(aa - bb)
                    ),
                    "Batteries": len(common),
                }
            )

    compare = pd.DataFrame(compare_rows)
    boot = pd.DataFrame(boot_rows)

    save_csv(
        compare,
        out / "07_adaptive_vs_baselines.csv",
    )
    save_csv(
        boot,
        out / "08_paired_battery_bootstrap.csv",
    )

    horizon = adaptive_horizon_table(
        ctx,
        formal,
        args,
    )
    save_csv(
        horizon,
        out / "09_adaptive_horizon_metrics.csv",
    )

    # Merge validated reference points from V4.14B/V4.14C where available.
    pareto_rows = []
    for r in pooled.itertuples(index=False):
        pareto_rows.append(
            {
                "Source": "V4.15",
                "Config": r.PolicyID,
                "RMSE": r.RMSE,
                "MeanScalars": r.MeanScalars,
                "MeanSensors": r.MeanSensors,
                "MeanDuration_s": r.MeanDuration_s,
                "Adaptive": 1,
            }
        )

    old_pareto = read_optional_csv(
        args.old_v414c_root
        / "21_empirical_validated_pareto.csv"
    )
    if len(old_pareto):
        for r in old_pareto.itertuples(index=False):
            pareto_rows.append(
                {
                    "Source": "V4.14C",
                    "Config": str(r.Config),
                    "RMSE": float(r.RMSE),
                    "MeanScalars": float(r.Scalars),
                    "MeanSensors": float(r.Sensors),
                    "MeanDuration_s": float(r.Duration_s),
                    "Adaptive": 0,
                }
            )

    pareto = pd.DataFrame(pareto_rows)
    if len(pareto):
        pareto = pareto.sort_values(
            ["MeanScalars", "RMSE"]
        ).reset_index(drop=True)
        # Non-dominated with respect to scalar count and RMSE.
        best = np.inf
        nd = []
        for r in pareto.itertuples(index=False):
            is_nd = float(r.RMSE) < best - 1e-15
            nd.append(int(is_nd))
            if is_nd:
                best = min(best, float(r.RMSE))
        pareto["ScalarRMSE_Pareto"] = nd

    save_csv(
        pareto,
        out / "10_budget_accuracy_pareto.csv",
    )

    # Integrity acceptance gate, separate from scientific success.
    expected_rows = (
        len(ctx["batteries"])
        * len(args.proxy_risk_targets)
    )

    gate = pd.DataFrame([
        {
            "Check": "Adaptive policy rows",
            "Observed": len(policy),
            "Expected": expected_rows,
            "PASS": int(len(policy) == expected_rows),
        },
        {
            "Check": "Adaptive formal rows",
            "Observed": len(formal),
            "Expected": expected_rows,
            "PASS": int(len(formal) == expected_rows),
        },
        {
            "Check": "47 batteries per policy",
            "Observed": int(
                formal.groupby("PolicyID")[
                    "OuterHeldoutBattery"
                ].nunique().min()
            ),
            "Expected": 47,
            "PASS": int(
                formal.groupby("PolicyID")[
                    "OuterHeldoutBattery"
                ].nunique().eq(47).all()
            ),
        },
        {
            "Check": "No heldout future HI in policy decisions",
            "Observed": int(
                pd.to_numeric(
                    policy[
                        "HeldoutFutureHIUsedForNextCycleSelection"
                    ],
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "Expected": 0,
            "PASS": int(
                pd.to_numeric(
                    policy[
                        "HeldoutFutureHIUsedForNextCycleSelection"
                    ],
                    errors="coerce",
                ).fillna(0).sum() == 0
            ),
        },
        {
            "Check": "No heldout XGB training",
            "Observed": int(
                pd.to_numeric(
                    formal["HeldoutUsedForXGBTraining"],
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "Expected": 0,
            "PASS": int(
                pd.to_numeric(
                    formal["HeldoutUsedForXGBTraining"],
                    errors="coerce",
                ).fillna(0).sum() == 0
            ),
        },
        {
            "Check": "Diagnosis counts within policy bound",
            "Observed": (
                f"{formal['DiagnosisCount'].min()}.."
                f"{formal['DiagnosisCount'].max()}"
            ),
            "Expected": (
                f"{args.min_diagnoses}.."
                f"{args.max_diagnoses}"
            ),
            "PASS": int(
                formal["DiagnosisCount"].between(
                    args.min_diagnoses,
                    args.max_diagnoses,
                ).all()
            ),
        },
        {
            "Check": "Finite pooled RMSE",
            "Observed": int(
                np.isfinite(
                    pd.to_numeric(
                        pooled["RMSE"],
                        errors="coerce",
                    )
                ).sum()
            ),
            "Expected": len(pooled),
            "PASS": int(
                np.isfinite(
                    pd.to_numeric(
                        pooled["RMSE"],
                        errors="coerce",
                    )
                ).all()
            ),
        },
    ])

    save_csv(
        gate,
        out / "11_acceptance_gate.csv",
    )

    status = (
        "PASS"
        if gate["PASS"].eq(1).all()
        else "CHECK"
    )

    scientific = []
    for r in pooled.itertuples(index=False):
        scientific.append(
            {
                "PolicyID": r.PolicyID,
                "ProxyRiskTarget": r.ProxyRiskTarget,
                "ActualPooledRMSE": r.RMSE,
                "MeanDiagnoses": r.MeanDiagnoses,
                "MeanScalars": r.MeanScalars,
                "MeanDuration_s": r.MeanDuration_s,
                "ActualRMSE_lt_0p008": bool(
                    r.RMSE < 0.008
                ),
                "ActualRMSE_lt_0p010": bool(
                    r.RMSE < 0.010
                ),
            }
        )

    complete = {
        "Task": (
            "RWTH48 V4.15 Budget-Aware Adaptive "
            "Diagnostic Scheduling"
        ),
        "CodeVersion": CODE_VERSION,
        "Status": status,
        "Platform": "Event",
        "MeasurementUnit": {
            "Nodes": list(EVENT_NODES),
            "HIs": list(EVENT_HIS),
            "MeasuredScalarsPerDiagnosis": (
                len(EVENT_NODES) * len(EVENT_HIS)
            ),
            "PhysicalSensors": "V|I",
        },
        "CandidateCycles": list(args.candidate_cycles),
        "MaxDiagnoses": args.max_diagnoses,
        "ProxyRiskTargets": args.proxy_risk_targets,
        "RiskModelCoverage": args.risk_model_coverage,
        "ScientificOutcome": scientific,
        "Protocol": {
            "OuterLOBO": True,
            "FirstDiagnosisSourceOnly": True,
            "HeldoutAlreadyMeasuredHIUsedForAdaptation": True,
            "HeldoutFutureHIUsedForNextCycleSelection": False,
            "HeldoutSOHUsedForPolicy": False,
            "HeldoutUsedForXGBTraining": False,
            "HistoricalSOHAsInput": False,
            "FutureHIAsInput": False,
            "CycleAsXGBModelFeature": False,
            "SourceProxyCrossFitted": True,
            "ConditionalRiskModelSourceOnly": True,
        },
        "Important": [
            (
                "Proxy-risk targets control policy aggressiveness and are "
                "not asserted to equal the final outer XGB RMSE."
            ),
            (
                "A future diagnosis cycle is selected using only current "
                "measured held-out features and source-learned expected "
                "risk after that future measurement; the future held-out "
                "HI is not inspected before selection."
            ),
            (
                "Formal XGB scoring is performed only after the adaptive "
                "schedule has been sequentially realized."
            ),
        ],
        "Acceptance": gate.to_dict("records"),
    }

    save_json(
        complete,
        out / "99_RWTH48_V4_15_COMPLETE.json",
    )

    print("\n" + "=" * 118)
    print("RWTH48 V4.15 FINALIZE")
    print("=" * 118)
    print("Status:", status)
    print("\nAdaptive pooled metrics:")
    print(pooled.to_string(index=False))
    print("\nDiagnosis-count distribution:")
    print(count_dist.to_string(index=False))
    print("\nAcceptance:")
    print(gate.to_string(index=False))
    print("Output:", out)
    print("=" * 118)


# =============================================================================
# 13. Parallel / resume
# =============================================================================

def state_dir(args) -> Path:
    return ensure_dir(
        args.state_root
        / args.run_id
    )


def done_marker(
    args,
    worker: int,
) -> Path:
    return (
        state_dir(args)
        / f"worker_{worker:02d}.done.json"
    )


def fail_marker(
    args,
    worker: int,
) -> Path:
    return (
        state_dir(args)
        / f"worker_{worker:02d}.fail.json"
    )


def write_done(
    args,
    owned: Sequence[str],
    stage: str,
) -> None:
    save_json(
        {
            "Worker": args.parallel_worker_id,
            "Stage": stage,
            "OwnedBatteries": list(owned),
            "Status": "PASS",
        },
        done_marker(
            args,
            args.parallel_worker_id,
        ),
    )


def write_fail(
    args,
    exc: Exception,
) -> None:
    save_json(
        {
            "Worker": args.parallel_worker_id,
            "Status": "FAIL",
            "Exception": repr(exc),
            "Traceback": traceback.format_exc(),
        },
        fail_marker(
            args,
            args.parallel_worker_id,
        ),
    )


def barrier_and_finalize(
    ctx,
    args,
) -> None:
    if args.parallel_worker_id != 0:
        return

    print(
        "\nWorker 0 waiting for all formal workers...",
        flush=True,
    )

    while True:
        fails = [
            fail_marker(args, w)
            for w in range(args.parallel_workers)
            if fail_marker(args, w).exists()
        ]
        if fails:
            raise RuntimeError(
                "A parallel worker failed:\n"
                + "\n".join(map(str, fails))
            )

        done = [
            done_marker(args, w).exists()
            for w in range(args.parallel_workers)
        ]
        if all(done):
            break

        time.sleep(
            max(1.0, float(args.barrier_poll_s))
        )

    finalize(
        ctx,
        args,
    )


# =============================================================================
# 14. Main
# =============================================================================

def main():
    args = parse_args()

    ensure_dir(args.output_root)
    ensure_dir(args.state_root)

    ctx, pre = preflight(args)

    if args.parallel_worker_id == 0:
        save_csv(
            pre,
            args.output_root / "00_preflight.csv",
        )
        save_json(
            {
                "CodeVersion": CODE_VERSION,
                "CandidateCycles": args.candidate_cycles,
                "MaxDiagnoses": args.max_diagnoses,
                "ProxyRiskTargets": args.proxy_risk_targets,
                "ProxyLeads": args.proxy_leads,
                "ProxyCrossfitFolds": args.proxy_crossfit_folds,
                "RiskModelFolds": args.risk_model_folds,
                "RiskModelCoverage": args.risk_model_coverage,
                "MinExpectedGain": args.min_expected_gain,
                "XGBLeadsPerOrigin": args.xgb_leads_per_origin,
                "ParallelWorkers": args.parallel_workers,
                "RunID": args.run_id,
            },
            args.output_root / "00_config.json",
        )

    if args.stage == "audit":
        print(pre.to_string(index=False))
        print("\nAUDIT PASS")
        return

    bats = list(ctx["batteries"])
    if args.only_battery:
        want = str(args.only_battery).strip()
        bats = [b for b in bats if b == want]
        if not bats:
            raise RuntimeError(
                f"--only-battery not found: {want}"
            )

    owned = [
        b
        for i, b in enumerate(bats)
        if i % args.parallel_workers
        == args.parallel_worker_id
    ]

    print(
        f"Worker {args.parallel_worker_id}/"
        f"{args.parallel_workers} owns "
        f"{len(owned)} batteries",
        flush=True,
    )

    try:
        if args.stage in ("policy", "all"):
            for i, b in enumerate(owned, 1):
                print(
                    f"\n[{i}/{len(owned)}] POLICY {b}",
                    flush=True,
                )
                run_policy_battery(
                    ctx,
                    b,
                    args,
                )

        if args.stage in ("formal", "all"):
            for i, b in enumerate(owned, 1):
                print(
                    f"\n[{i}/{len(owned)}] FORMAL {b}",
                    flush=True,
                )
                run_formal_battery(
                    ctx,
                    b,
                    args,
                )

        if args.stage == "finalize":
            if args.parallel_worker_id != 0:
                raise RuntimeError(
                    "finalize must be run by worker 0"
                )
            finalize(ctx, args)
            return

        if args.stage == "all":
            write_done(
                args,
                owned,
                args.stage,
            )
            barrier_and_finalize(
                ctx,
                args,
            )

    except Exception as e:
        write_fail(
            args,
            e,
        )
        raise


if __name__ == "__main__":
    main()

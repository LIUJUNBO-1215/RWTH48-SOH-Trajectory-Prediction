# -*- coding: utf-8 -*-
r"""
RWTH48 V4.13F-PRE FixedTime Full Catch-Up — Common Library
==========================================================

Purpose
-------
This module supplies the shared, leakage-safe machinery used by the FixedTime
catch-up package.  It deliberately modernizes FixedTime to the methodological
level reached by the final Event branch WITHOUT changing the frozen physical
FixedTime acquisition platform.

Upstream physical platform
--------------------------
    cumulative FixedTime discharge prefixes:
        0-30 s / 0-60 s / 0-90 s / 0-120 s

Historical diagnosis schedule for the final full-life tasks
-----------------------------------------------------------
    diagnosis cycles:
        40 / 48 / 56 / 64 / 72 / 80
    i.e. J=8, L=6, span=40 cycles

Target semantics
----------------
    CyclingSOH_Charge_Canonical
    Operational EOL80 = frozen V4.11A2 definition
    No historical true SOH / capacity / future HI / cycle index in X.

The module imports the already-written V4.13F platform-aware script only as a
reliable FixedTime frozen-table adapter and as a source of the exact 47-cell
V4.11A2 target bundle.  Model selection and all FixedTime feature engineering
implemented here are source-only.

Core capabilities
-----------------
1. Full FixedTime candidate-HI tensor from the frozen V4.2/V4.6 ecosystem.
2. SOH-specific source-only HI ranking.
3. RUL-specific source-only HI ranking.
4. Top-K deterministic engineering for arbitrary FixedTime HI sets.
5. Lead-conditioned full-life XGBoost trajectory model.
6. Generic LSTM / BiLSTM one-shot full-life model.
7. TrendDetail deep trajectory model.
8. Residual hybrid and horizon-adaptive MoE.
9. Direct RUL XGBoost / LSTM / BiLSTM / hybrid utilities.
10. Battery-level bootstrap and nested source-only cross-conformal utilities.
11. SHAP-ready feature registries and feature-level perturbation helpers.

Scientific guardrail
--------------------
This is a FixedTime *modernization/catch-up* package.  It is intended to remove
algorithm-maturity confounding before any new platform comparison.  It does NOT
retroactively change the earlier exact-common Event-vs-FixedTime experiment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz


# =============================================================================
# Frozen project constants
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR = Path(__file__).resolve().parent

FT_BASE_HELPER = CODE_DIR / "09_platform_sensor_ablation.py"

FTCATCH_COMMON_VERSION = "FIX3_NATIVE_ALL_FROZEN_NODES"

EXPECTED_BATTERIES = 47
TEST_ORIGIN = 80
SOURCE_ORIGINS = (80, 120, 160)
HISTORY_OFFSETS = (-40, -32, -24, -16, -8, 0)
DIAG_CYCLES_ORIGIN80 = (40, 48, 56, 64, 72, 80)
FIXEDTIME_NODES = (30.0, 60.0, 90.0, 120.0)  # fallback only; runtime-overridden from frozen table

K = 1000
RANDOM_SEED = 20260824
ENC = "utf-8-sig"
FAST_GZIP = {"method": "gzip", "compresslevel": 1}

HORIZON_BINS = (
    ("Lead001_100", 1, 100),
    ("Lead101_300", 101, 300),
    ("Lead301_500", 301, 500),
    ("Lead501_700", 501, 700),
    ("Lead701_1000", 701, 1000),
)
SOH_RANK_HORIZONS = (100, 300, 500, 700, 1000)


# =============================================================================
# Basic utilities
# =============================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, p: Path) -> None:
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding=ENC)


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


def normalize_battery(x: object) -> str:
    s = str(x).strip().upper().replace("_", "-").replace(" ", "")
    m = re.search(r"RWTH-?0*(\d+)", s)
    return f"RWTH-{int(m.group(1)):03d}" if m else s


def stable_seed(*items: object) -> int:
    raw = "|".join(map(str, items)).encode("utf-8")
    return int(hashlib.sha1(raw).hexdigest()[:8], 16)


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


def split_groups(batteries: Sequence[str], n_folds: int) -> List[List[str]]:
    bats = sorted(set(map(normalize_battery, batteries)))
    k = max(2, min(int(n_folds), len(bats)))
    groups = [[] for _ in range(k)]
    for i, b in enumerate(bats):
        groups[i % k].append(b)
    return [g for g in groups if g]


def worker_assignment(
    batteries: Sequence[str],
    num_workers: int,
    worker_id: int,
) -> List[str]:
    bats = sorted(map(normalize_battery, batteries))
    n = max(1, int(num_workers))
    w = int(worker_id)
    if not (0 <= w < n):
        raise ValueError(f"worker-id={w} invalid for num-workers={n}")
    return [b for i, b in enumerate(bats) if i % n == w]


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


def rul_metrics(y_true, y_pred) -> Dict[str, float]:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]

    if len(yt) == 0:
        return {
            "N": 0, "MAE": np.nan, "MedianAE": np.nan,
            "RMSE": np.nan, "Bias": np.nan, "R2": np.nan,
            "MAPE_pct": np.nan, "Within25": np.nan,
            "Within50": np.nan, "Within100": np.nan,
            "PredStd": np.nan, "TrueStd": np.nan,
            "SpreadRatio": np.nan, "SpreadGap": np.nan,
        }

    e = yp - yt
    ss_res = float(np.sum(e * e))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    ps = float(np.std(yp))
    ts = float(np.std(yt))

    return {
        "N": int(len(yt)),
        "MAE": float(np.mean(np.abs(e))),
        "MedianAE": float(np.median(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e * e))),
        "Bias": float(np.mean(e)),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else np.nan,
        "MAPE_pct": float(
            100.0 * np.mean(np.abs(e) / np.maximum(np.abs(yt), 1e-8))
        ),
        "Within25": float(np.mean(np.abs(e) <= 25)),
        "Within50": float(np.mean(np.abs(e) <= 50)),
        "Within100": float(np.mean(np.abs(e) <= 100)),
        "PredStd": ps,
        "TrueStd": ts,
        "SpreadRatio": float(ps / ts) if ts > 1e-12 else np.nan,
        "SpreadGap": float(abs(1.0 - ps / ts)) if ts > 1e-12 else np.nan,
    }


def trajectory_metrics(y, p, mask) -> Dict[str, float]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = (
        np.asarray(mask).astype(bool)
        & np.isfinite(y)
        & np.isfinite(p)
    )

    if not m.any():
        return {
            "N": 0, "MAE": np.nan, "RMSE": np.nan,
            "Bias": np.nan, "R2": np.nan,
        }

    yt = y[m]
    yp = p[m]
    e = yp - yt
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))

    return {
        "N": int(len(yt)),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e * e))),
        "Bias": float(np.mean(e)),
        "R2": float(
            1.0 - np.sum(e * e) / ss_tot
            if ss_tot > 1e-15 else np.nan
        ),
    }


def shape_metrics(y, p, mask) -> Dict[str, float]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.asarray(mask).astype(bool) & np.isfinite(y) & np.isfinite(p)

    idx = np.flatnonzero(m)
    if len(idx) < 3:
        return {
            "TrueUpwardStepFraction": np.nan,
            "PredUpwardStepFraction": np.nan,
            "TrueRoughness": np.nan,
            "PredRoughness": np.nan,
            "RoughnessRatioPredToTrue": np.nan,
        }

    # Keep only adjacent lead pairs to avoid bridging missing target intervals.
    pair = idx[1:][np.diff(idx) == 1]
    prev = pair - 1
    if len(pair) == 0:
        return {
            "TrueUpwardStepFraction": np.nan,
            "PredUpwardStepFraction": np.nan,
            "TrueRoughness": np.nan,
            "PredRoughness": np.nan,
            "RoughnessRatioPredToTrue": np.nan,
        }

    dy = y[pair] - y[prev]
    dp = p[pair] - p[prev]

    # Roughness = mean absolute first difference; stable for gapped truth.
    tr = float(np.mean(np.abs(dy)))
    pr = float(np.mean(np.abs(dp)))

    return {
        "TrueUpwardStepFraction": float(np.mean(dy > 0)),
        "PredUpwardStepFraction": float(np.mean(dp > 0)),
        "TrueRoughness": tr,
        "PredRoughness": pr,
        "RoughnessRatioPredToTrue": float(pr / tr) if tr > 1e-15 else np.nan,
    }


def find_persistent_eol_lead(
    pred: np.ndarray,
    threshold: float = 0.80,
    consecutive: int = 3,
) -> Tuple[int, int]:
    p = np.asarray(pred, float).reshape(-1)
    below = np.isfinite(p) & (p <= float(threshold))

    run = 0
    for i, flag in enumerate(below):
        run = run + 1 if flag else 0
        if run >= int(consecutive):
            return int(i + 2 - consecutive), 1  # lead is one-based

    return int(len(p) + 1), 0


# =============================================================================
# Adapter to existing V4.13F / frozen V4.11A2 target bundle
# =============================================================================

def make_ftbase_args(
    base: Path,
    code_dir: Path,
    output_root: Path,
    fixedtime_ready: Optional[Path] = None,
    event_ready: Optional[Path] = None,
    xgb_threads: int = 4,
) -> argparse.Namespace:
    ns = argparse.Namespace()

    ns.stage = "audit"
    ns.base = Path(base)
    ns.code_dir = Path(code_dir)
    ns.v412a_helper = (
        Path(code_dir)
        / "21_rul_topk_helper.py"
    )

    ns.event_ready = event_ready
    ns.fixedtime_ready = fixedtime_ready
    ns.dependency_registry = None

    ns.v412a_outer = (
        Path(base)
        / "RWTH48_V4_12A_RULSpecific_HI_TopK_Pareto"
        / "07_outer_predictions_by_topk.csv"
    )
    ns.v412a_complete = (
        Path(base)
        / "RWTH48_V4_12A_RULSpecific_HI_TopK_Pareto"
        / "99_RWTH48_V4_12A_COMPLETE.json"
    )

    ns.output_root = Path(output_root)
    ns.configs = list(getattr(
        load_module(
            Path(code_dir)
            / "09_platform_sensor_ablation.py",
            "_rwth48_v413f_temp_for_args",
        ),
        "CONFIGS",
    ))
    ns.only_battery = ""
    ns.force = False
    ns.shortlist = 14
    ns.inner_folds = 5
    ns.xgb_threads = int(xgb_threads)
    ns.recompute_event_vit = False
    ns.fixedtime_nodes = list(FIXEDTIME_NODES)
    ns.phase = "Discharge"
    ns.bootstrap = 20000
    ns.num_workers = 1
    ns.worker_id = 0
    ns.auto_finalize = False
    ns.skip_figures = True
    ns.dpi = 200
    return ns



def _fixedtime_platform_phase_subset(ftbase, df, schema, phase: str) -> pd.DataFrame:
    """Filter only FixedTime platform/phase; keep all native node values."""
    d = df.copy()

    d[schema.battery_col] = d[schema.battery_col].map(normalize_battery)
    d[schema.cycle_col] = pd.to_numeric(
        d[schema.cycle_col],
        errors="coerce",
    )

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

    if masks:
        mm = np.column_stack(masks)
        if mm.size and np.any(mm):
            d = d[np.any(mm, axis=1)].copy()

    if schema.phase_col is not None:
        s = d[schema.phase_col].astype(str).str.strip()
        m = s.str.casefold().eq(str(phase).strip().casefold())
        if not m.any():
            raise RuntimeError(
                f"FixedTime Phase={phase}没有行；"
                f"可用值={sorted(s.unique().tolist())[:20]}"
            )
        d = d[m].copy()

    d["_NodeNumeric_Discovery"] = d[schema.node_col].map(
        ftbase.parse_fixedtime_node
    )
    d["_NodeNumeric_Discovery"] = pd.to_numeric(
        d["_NodeNumeric_Discovery"],
        errors="coerce",
    )
    d = d[np.isfinite(d["_NodeNumeric_Discovery"].to_numpy(float))].copy()

    return d


def discover_fixedtime_nodes(
    ftbase,
    fixed_raw: pd.DataFrame,
    fixed_schema,
    bundle,
    phase: str = "Discharge",
    requested_nodes: Optional[Sequence[float]] = None,
) -> Tuple[Tuple[float, ...], pd.DataFrame]:
    """
    Resolve the actual native FixedTime checkpoints from the frozen table.

    FIX3 rule:
      * The frozen table is authoritative.
      * If the user explicitly supplies --fixedtime-nodes, use exactly those
        nodes after validating their presence.
      * Otherwise use ALL native FixedTime nodes that have complete coverage
        over the battery×cycle keys required by the frozen 141-sample handoff.
      * Require at least two complete nodes.
      * Never silently truncate an 8-node frozen platform to an arbitrary 4-node
        subset.

    This matters because the actual RWTH48 frozen FixedTime platform contains
    eight complete checkpoints:
        60 / 120 / 180 / 300 / 450 / 600 / 900 / 1200 s
    on the current data.  The code does not hard-code those values; it discovers
    them from the frozen table.
    """
    d = _fixedtime_platform_phase_subset(
        ftbase,
        fixed_raw,
        fixed_schema,
        phase,
    )

    req = set()
    bats = np.asarray(
        [normalize_battery(x) for x in bundle.battery],
        dtype=object,
    )
    origins = np.asarray(bundle.origin, dtype=int)

    for b, o in zip(bats, origins):
        for off in HISTORY_OFFSETS:
            req.add((str(b), int(o) + int(off)))

    req_n = len(req)

    vals = d["_NodeNumeric_Discovery"].to_numpy(float)
    canon = np.where(
        np.isclose(vals, np.rint(vals), atol=1e-6, rtol=0),
        np.rint(vals),
        vals,
    )
    d["_NodeCandidate"] = canon.astype(float)

    rows = []

    for node, g in d.groupby("_NodeCandidate", sort=True):
        cyc = pd.to_numeric(
            g[fixed_schema.cycle_col],
            errors="coerce",
        )
        good = cyc.notna()

        keys = set(
            zip(
                g.loc[good, fixed_schema.battery_col].map(normalize_battery),
                np.rint(cyc[good].to_numpy(float)).astype(int),
            )
        )
        keys = {(str(b), int(c)) for b, c in keys}
        covered = len(req & keys)

        rows.append(
            {
                "Node_s": float(node),
                "Rows": int(len(g)),
                "RequiredBatteryCycleKeys": int(req_n),
                "CoveredRequiredKeys": int(covered),
                "RequiredCoverage": float(
                    covered / req_n if req_n else np.nan
                ),
                "CompleteRequiredCoverage": int(covered == req_n),
            }
        )

    audit = pd.DataFrame(rows)
    if audit.empty:
        raise RuntimeError(
            "FixedTime frozen table中无法解析任何时间节点。"
        )

    audit = audit.sort_values(
        ["Node_s"],
        ascending=[True],
    ).reset_index(drop=True)

    if requested_nodes is not None:
        nodes = tuple(sorted(map(float, requested_nodes)))

        if len(nodes) < 2 or len(set(nodes)) != len(nodes):
            raise RuntimeError(
                "--fixedtime-nodes至少包含2个互异节点，"
                f"当前={nodes}"
            )

        available = audit["Node_s"].to_numpy(float)
        missing = [
            float(n)
            for n in nodes
            if not np.isclose(
                available,
                float(n),
                atol=0.5,
                rtol=0,
            ).any()
        ]
        if missing:
            raise RuntimeError(
                "显式指定的FixedTime节点不存在: "
                f"{missing}\n实际Frozen节点覆盖：\n"
                + audit.to_string(index=False)
            )

        audit["Selected"] = audit["Node_s"].apply(
            lambda x: int(
                any(
                    np.isclose(float(x), n, atol=0.5, rtol=0)
                    for n in nodes
                )
            )
        )
        audit["SelectionReason"] = np.where(
            audit["Selected"].eq(1),
            "ExplicitRequested",
            "",
        )
        return nodes, audit

    complete = tuple(
        sorted(
            audit.loc[
                audit["CompleteRequiredCoverage"].eq(1),
                "Node_s",
            ].astype(float).tolist()
        )
    )

    if len(complete) < 2:
        raise RuntimeError(
            "完整覆盖冻结141样本所需历史键的FixedTime节点少于2个。\n"
            + audit.to_string(index=False)
        )

    nodes = complete
    audit["Selected"] = audit["Node_s"].isin(nodes).astype(int)
    audit["SelectionReason"] = np.where(
        audit["Selected"].eq(1),
        "AllFullCoverageFrozenNodes",
        "",
    )

    return nodes, audit


def load_fixedtime_context(
    base: Path = BASE,
    code_dir: Path = CODE_DIR,
    output_root: Optional[Path] = None,
    fixedtime_ready: Optional[Path] = None,
    event_ready: Optional[Path] = None,
    xgb_threads: int = 4,
    fixedtime_nodes: Optional[Sequence[float]] = None,
):
    """
    Load only the pieces required by the FixedTime catch-up.

    IMPORTANT:
    We intentionally do NOT call the full V4.13F preflight here because that
    preflight also enforces sensor-dependency classification for every HI.
    Sensor dependency is required for the later modality ablation, but it must
    not block the model-modernization catch-up if a FixedTime HI has a new name.

    We instead reuse:
        - V4.12A preflight for the frozen V4.11A2 target bundle;
        - V4.13F FixedTime table locator/schema/tensor adapter.
    """
    global FIXEDTIME_NODES

    base = Path(base)
    code_dir = Path(code_dir)

    helper_path = (
        code_dir
        / "09_platform_sensor_ablation.py"
    )
    if not helper_path.exists():
        raise FileNotFoundError(
            "需要13F平台适配器：\n"
            + str(helper_path)
        )

    ftbase = load_module(
        helper_path,
        "rwth48_v413f_adapter_for_ftcatchup",
    )

    out = (
        Path(output_root)
        if output_root is not None
        else base / "RWTH48_V4_13F_PRE_FixedTime_FullCatchUp"
    )

    args = argparse.Namespace(
        stage="audit",
        base=base,
        code_dir=code_dir,
        v412a_helper=(
            code_dir / "21_rul_topk_helper.py"
        ),
        event_ready=event_ready,
        fixedtime_ready=fixedtime_ready,
        dependency_registry=None,
        v412a_outer=(
            base
            / "RWTH48_V4_12A_RULSpecific_HI_TopK_Pareto"
            / "07_outer_predictions_by_topk.csv"
        ),
        v412a_complete=(
            base
            / "RWTH48_V4_12A_RULSpecific_HI_TopK_Pareto"
            / "99_RWTH48_V4_12A_COMPLETE.json"
        ),
        output_root=out,
        configs=list(ftbase.CONFIGS),
        only_battery="",
        force=False,
        shortlist=14,
        inner_folds=5,
        xgb_threads=int(xgb_threads),
        recompute_event_vit=False,
        fixedtime_nodes=list(FIXEDTIME_NODES),
        phase="Discharge",
        bootstrap=20000,
        num_workers=1,
        worker_id=0,
        auto_finalize=False,
        skip_figures=True,
        dpi=200,
    )

    # 1) Frozen target bundle / common full-life truth.
    v412a = load_module(
        args.v412a_helper,
        "rwth48_v412a_for_ftcatchup",
    )
    aargs = ftbase.make_v412a_args(args)

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

    # 2) Native FixedTime frozen HI table.
    fixed_path, locate_audit = ftbase.locate_fixedtime_ready(args)

    fixed_raw = pd.read_csv(
        fixed_path,
        compression="infer",
        low_memory=False,
    )

    fixed_schema = ftbase.detect_fixedtime_schema(
        fixed_raw,
        args.fixedtime_nodes,
    )

    discovered_nodes, node_audit = discover_fixedtime_nodes(
        ftbase,
        fixed_raw,
        fixed_schema,
        bundle,
        phase=args.phase,
        requested_nodes=fixedtime_nodes,
    )

    args.fixedtime_nodes = list(discovered_nodes)

    fixed = ftbase.filter_fixedtime_table(
        fixed_raw,
        fixed_schema,
        discovered_nodes,
        args.phase,
    )

    fixed_hit = ftbase.build_fixedtime_tensor(
        fixed,
        fixed_schema,
        bundle,
        discovered_nodes,
    )

    # Runtime frozen nodes drive feature names and node-position slopes.
    FIXEDTIME_NODES = tuple(map(float, discovered_nodes))

    batteries = sorted(set(map(normalize_battery, bundle.battery.tolist())))
    if len(batteries) != EXPECTED_BATTERIES:
        raise RuntimeError(
            f"FixedTime catch-up batteries={len(batteries)} !=47"
        )

    ctx = {
        "helper": helper,
        "v410a": v410a,
        "v412a": v412a,
        "bundle": bundle,
        "baseline": baseline,
        "event_hit": event_hit,
        "event_layout": event_layout,
        "event_recon": event_recon,
        "handoff_audit": handoff_audit,
        "v412a_prov": v412a_prov,
        "event_schema_audit": event_schema_audit,
        "fixed_hit": fixed_hit,
        "fixed_path": fixed_path,
        "fixed_raw_rows": len(fixed_raw),
        "fixed_filtered_rows": len(fixed),
        "fixed_schema": fixed_schema,
        "fixed_locate_audit": locate_audit,
        "fixed_node_audit": node_audit,
        "fixed_nodes": tuple(map(float, discovered_nodes)),
    }

    print(
        "[FTCATCH FIX3] FixedTime-ready: " + str(fixed_path),
        flush=True,
    )
    print(
        "[FTCATCH FIX3] Frozen FixedTime nodes: "
        + ", ".join(f"{x:g}s" for x in discovered_nodes),
        flush=True,
    )

    return ftbase, ctx


# =============================================================================
# FixedTime selected-HI deterministic representation
# =============================================================================

def single_hi_engineering(
    x: np.ndarray,
    hi_name: str,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Generic native-FixedTime single-HI deterministic engineering.

    x:
        [N, 6 diagnoses, M native FixedTime checkpoints]

    For M checkpoints:
        RawMeasured                = 6M
        HistoricalTrend           = 4M
        WithinDiagnosisGeometry   = 36
        ConsecutiveNodeDifference = 6(M-1)

        SingleHI feature count:
            16M + 30

    For the actual 8-node frozen platform:
        16*8 + 30 = 158 features / HI.
    """
    x = np.asarray(x, np.float64)

    if x.ndim != 3 or x.shape[1] != 6:
        raise RuntimeError(
            f"single HI shape={x.shape}, expected [N,6,M]"
        )
    if x.shape[2] != len(FIXEDTIME_NODES):
        raise RuntimeError(
            f"single HI node count={x.shape[2]} != frozen nodes "
            f"{len(FIXEDTIME_NODES)}"
        )
    if not np.isfinite(x).all():
        raise RuntimeError(f"{hi_name}: single HI contains NaN/Inf")

    N, D, M = x.shape
    blocks: List[np.ndarray] = []
    names: List[str] = []
    rows: List[Dict[str, object]] = []

    def add(arr, nms, family):
        arr = np.asarray(arr, np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        blocks.append(arr)
        names.extend(nms)
        rows.extend(
            {
                "Feature": n,
                "Group": family,
                "MeasuredOrDerived": (
                    "Measured" if family == "RawMeasured" else "Derived"
                ),
                "SourceHI": hi_name,
            }
            for n in nms
        )

    # A. raw: 6M
    raw = x.reshape(N, -1)
    raw_names = [
        f"RAW__D{d}__T{int(round(FIXEDTIME_NODES[m])):04d}__{hi_name}"
        for d in range(D)
        for m in range(M)
    ]
    add(raw, raw_names, "RawMeasured")

    # B. historical trend: 4M
    cycle_t = np.asarray(HISTORY_OFFSETS, float)
    tc = cycle_t - cycle_t.mean()
    den = float(np.sum(tc * tc))

    cols = []
    nms = []
    for m in range(M):
        a = x[:, :, m]
        slope = (
            np.sum(
                (a - np.mean(a, axis=1, keepdims=True))
                * tc[None, :],
                axis=1,
            )
            / max(den, 1e-12)
        )
        vals = [
            a[:, -1] - a[:, 0],
            slope,
            np.mean(a, axis=1),
            np.std(a, axis=1),
        ]
        tags = ["LongDelta", "SlopePerCycle", "Mean", "Std"]
        cols.extend(vals)
        nms.extend(
            [
                f"HIST__T{int(round(FIXEDTIME_NODES[m])):04d}"
                f"__{hi_name}__{t}"
                for t in tags
            ]
        )
    add(np.column_stack(cols), nms, "HistoricalTrend")

    # C. within-diagnosis geometry: 6 diagnoses x 6 stats = 36
    cols = []
    nms = []
    node_t = np.asarray(FIXEDTIME_NODES, float)
    ntc = node_t - node_t.mean()
    nden = float(np.sum(ntc * ntc))

    for d in range(D):
        a = x[:, d, :]
        slope = (
            np.sum(
                (a - np.mean(a, axis=1, keepdims=True))
                * ntc[None, :],
                axis=1,
            )
            / max(nden, 1e-12)
        )
        vals = [
            np.mean(a, axis=1),
            np.std(a, axis=1),
            np.min(a, axis=1),
            np.max(a, axis=1),
            np.max(a, axis=1) - np.min(a, axis=1),
            slope,
        ]
        tags = ["Mean", "Std", "Min", "Max", "Range", "NodeSlope"]
        cols.extend(vals)
        nms.extend(
            [
                f"WITHIN__D{d}__{hi_name}__{t}"
                for t in tags
            ]
        )
    add(np.column_stack(cols), nms, "WithinDiagnosisGeometry")

    # D. adjacent native-node differences: 6(M-1)
    cols = []
    nms = []
    for d in range(D):
        da = np.diff(x[:, d, :], axis=1)
        for m in range(M - 1):
            cols.append(da[:, m])
            nms.append(
                f"NODEDIFF__D{d}__T{int(round(FIXEDTIME_NODES[m])):04d}"
                f"_T{int(round(FIXEDTIME_NODES[m+1])):04d}__{hi_name}"
            )
    if cols:
        add(
            np.column_stack(cols),
            nms,
            "ConsecutiveNodeDifference",
        )

    X = np.concatenate(blocks, axis=1).astype(np.float32)

    expected = 16 * M + 30
    if X.shape[1] != expected:
        raise RuntimeError(
            f"single HI engineered count={X.shape[1]} !={expected} for M={M}"
        )

    return X, names, pd.DataFrame(rows)


def cross_hi_engineering(
    xa: np.ndarray,
    xb: np.ndarray,
    hi_a: str,
    hi_b: str,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Generic same-node cross-HI geometry.

    Per diagnosis × native checkpoint:
        signed difference
        scale-normalized difference

    Feature count:
        6 * M * 2 = 12M per HI pair.
    """
    a = np.asarray(xa, np.float64)
    b = np.asarray(xb, np.float64)

    if a.shape != b.shape or a.ndim != 3 or a.shape[1] != 6:
        raise RuntimeError(
            f"cross HI shape mismatch: {a.shape} vs {b.shape}"
        )

    M = a.shape[2]
    if M != len(FIXEDTIME_NODES):
        raise RuntimeError(
            f"cross HI node count={M} != frozen nodes {len(FIXEDTIME_NODES)}"
        )

    cols = []
    names = []
    rows = []

    for d in range(6):
        for m in range(M):
            av = a[:, d, m]
            bv = b[:, d, m]

            diff = av - bv
            ndiff = diff / (np.abs(av) + np.abs(bv) + 1e-6)

            for val, tag in [(diff, "Diff"), (ndiff, "NormDiff")]:
                nm = (
                    f"CROSSHI__D{d}__T{int(round(FIXEDTIME_NODES[m])):04d}"
                    f"__{hi_a}__{hi_b}__{tag}"
                )
                cols.append(val)
                names.append(nm)
                rows.append(
                    {
                        "Feature": nm,
                        "Group": "CrossHIWithinNode",
                        "MeasuredOrDerived": "Derived",
                        "SourceHI": f"{hi_a}|{hi_b}",
                    }
                )

    X = np.column_stack(cols).astype(np.float32)
    expected = 12 * M

    if X.shape[1] != expected:
        raise RuntimeError(
            f"cross pair engineered count={X.shape[1]} !={expected}"
        )

    return X, names, pd.DataFrame(rows)


def engineer_fixedtime_selected(
    fixed_hit,
    selected_his: Sequence[str],
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    General Top-K deterministic engineering.

    For M native checkpoints and K selected HIs:

        Single-HI block = 16M + 30
        Cross-HI block  = 12M per pair

        FeatureCount(M,K)
            = K(16M+30) + 12M*C(K,2)

    For the actual 8-node frozen FixedTime platform:
        Top1 = 158
        Top2 = 412
        Top3 = 762
        Top4 = 1208
        Top5 = 1750
    """
    selected = list(map(str, selected_his))
    if len(selected) < 1:
        raise ValueError("selected_his must be non-empty")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_his contains duplicates")

    imap = {h: i for i, h in enumerate(fixed_hit.hi_names)}
    miss = [h for h in selected if h not in imap]
    if miss:
        raise KeyError(f"FixedTime HI not found: {miss}")

    blocks = []
    names = []
    regs = []

    single_cache = {}

    for h in selected:
        x = np.asarray(
            fixed_hit.values[:, :, :, imap[h]],
            np.float32,
        )
        single_cache[h] = x
        Xs, ns, rs = single_hi_engineering(x, h)
        blocks.append(Xs)
        names.extend(ns)
        regs.append(rs)

    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            a = selected[i]
            b = selected[j]
            Xc, nc, rc = cross_hi_engineering(
                single_cache[a],
                single_cache[b],
                a,
                b,
            )
            blocks.append(Xc)
            names.extend(nc)
            regs.append(rc)

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    reg = pd.concat(regs, ignore_index=True)

    M = len(FIXEDTIME_NODES)
    expected = (
        (16 * M + 30) * len(selected)
        + (12 * M) * (len(selected) * (len(selected) - 1) // 2)
    )
    if X.shape[1] != expected:
        raise RuntimeError(
            f"Top{len(selected)} engineered count={X.shape[1]} !={expected}"
        )
    if len(names) != X.shape[1] or len(reg) != X.shape[1]:
        raise RuntimeError("FixedTime feature registry mismatch")
    if not np.isfinite(X).all():
        raise RuntimeError("FixedTime engineered features contain NaN/Inf")

    return X, names, reg


def raw_fixedtime_sequence(
    fixed_hit,
    selected_his: Sequence[str],
) -> np.ndarray:
    """
    Return [N,6,M*K_his] native measured sequence for deep models.
    Checkpoint-major then HI-major within each diagnosis.
    """
    selected = list(map(str, selected_his))
    imap = {h: i for i, h in enumerate(fixed_hit.hi_names)}

    idx = [imap[h] for h in selected]
    x = np.asarray(
        fixed_hit.values[:, :, :, idx],
        np.float32,
    )  # [N,6,4,K]

    if not np.isfinite(x).all():
        raise RuntimeError("Selected FixedTime raw sequence contains NaN/Inf")

    N = x.shape[0]
    return x.reshape(N, 6, 4 * len(selected)).astype(np.float32)


# =============================================================================
# Sample selection / target helpers
# =============================================================================

def sample_indices(
    fixed_hit,
    batteries: Sequence[str],
    origins: Sequence[int],
) -> np.ndarray:
    bm = np.isin(
        np.asarray(fixed_hit.batteries, dtype=object),
        list(map(normalize_battery, batteries)),
    )
    om = np.isin(
        np.asarray(fixed_hit.origins, dtype=int),
        list(map(int, origins)),
    )
    return np.where(bm & om)[0].astype(np.int32)


def origin80_indices(
    fixed_hit,
    batteries: Sequence[str],
) -> np.ndarray:
    idx = sample_indices(fixed_hit, batteries, [TEST_ORIGIN])
    if len(idx) != len(batteries):
        raise RuntimeError(
            f"origin80 rows={len(idx)} vs batteries={len(batteries)}"
        )
    return idx


def true_rul(bundle, idx: Sequence[int]) -> np.ndarray:
    return np.asarray(
        bundle.true_eol_lead[np.asarray(idx, int)],
        dtype=np.float32,
    )


# =============================================================================
# SOH-specific and RUL-specific source-only HI ranking
# =============================================================================

def rank_rul_his_source_only(
    fixed_hit,
    bundle,
    source_batteries: Sequence[str],
    shortlist: int = 14,
) -> pd.DataFrame:
    idx = origin80_indices(fixed_hit, source_batteries)
    y = true_rul(bundle, idx)

    rows = []
    for j, hi in enumerate(fixed_hit.hi_names):
        x = np.asarray(
            fixed_hit.values[idx, :, :, j],
            float,
        ).reshape(len(idx), -1)

        rhos = [spearman_abs(x[:, k], y) for k in range(x.shape[1])]
        top4 = sorted(rhos, reverse=True)[:4]

        rows.append(
            {
                "HI": hi,
                "Task": "RUL",
                "RelevanceScore": float(np.mean(top4)),
                "MaxAbsSpearman": float(max(rhos)),
                "MedianAbsSpearman": float(np.median(rhos)),
                "PositionCount": int(x.shape[1]),
            }
        )

    tab = pd.DataFrame(rows).sort_values(
        ["RelevanceScore", "MaxAbsSpearman", "HI"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    tab["Rank"] = np.arange(1, len(tab) + 1)
    tab["Shortlisted"] = (
        tab["Rank"] <= max(2, int(shortlist))
    ).astype(int)
    return tab


def rank_soh_his_source_only(
    fixed_hit,
    bundle,
    source_batteries: Sequence[str],
    shortlist: int = 14,
    horizons: Sequence[int] = SOH_RANK_HORIZONS,
) -> pd.DataFrame:
    """
    SOH-specific ranking differs from RUL ranking.

    For each HI:
      - 24 causal input positions = 6 diagnoses x 4 FixedTime nodes.
      - For each requested future lead endpoint, compute abs Spearman between
        every causal position and the corresponding future SOH target across
        source batteries with observed truth.
      - At each horizon retain the four strongest position correlations.
      - Overall score = mean across available horizon-specific top4 means.

    Only source batteries participate.
    """
    idx = origin80_indices(fixed_hit, source_batteries)

    rows = []

    for j, hi in enumerate(fixed_hit.hi_names):
        X = np.asarray(
            fixed_hit.values[idx, :, :, j],
            float,
        ).reshape(len(idx), -1)

        horizon_scores = []
        max_all = []

        row = {
            "HI": hi,
            "Task": "SOH",
            "PositionCount": int(X.shape[1]),
        }

        for h in horizons:
            pos = int(h) - 1
            if pos < 0 or pos >= bundle.y.shape[1]:
                row[f"H{int(h)}_Top4MeanAbsSpearman"] = np.nan
                continue

            y = np.asarray(bundle.y[idx, pos], float)
            m = (
                np.asarray(bundle.mask[idx, pos]).astype(bool)
                & np.isfinite(y)
            )

            if m.sum() < 5:
                row[f"H{int(h)}_Top4MeanAbsSpearman"] = np.nan
                continue

            rhos = [
                spearman_abs(X[m, k], y[m])
                for k in range(X.shape[1])
            ]

            top4 = sorted(rhos, reverse=True)[:4]
            score = float(np.mean(top4))

            horizon_scores.append(score)
            max_all.extend(rhos)
            row[f"H{int(h)}_Top4MeanAbsSpearman"] = score

        row["RelevanceScore"] = (
            float(np.mean(horizon_scores))
            if horizon_scores
            else 0.0
        )
        row["MaxAbsSpearman"] = (
            float(max(max_all))
            if max_all
            else 0.0
        )
        row["AvailableHorizons"] = int(len(horizon_scores))
        rows.append(row)

    tab = pd.DataFrame(rows).sort_values(
        ["RelevanceScore", "MaxAbsSpearman", "HI"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    tab["Rank"] = np.arange(1, len(tab) + 1)
    tab["Shortlisted"] = (
        tab["Rank"] <= max(2, int(shortlist))
    ).astype(int)
    return tab


# =============================================================================
# XGBoost trajectory model
# =============================================================================

def lead_features(leads: np.ndarray, k: int = K) -> np.ndarray:
    l = np.asarray(leads, float).reshape(-1)
    u = l / float(k)
    return np.column_stack(
        [
            u,
            u * u,
            np.sqrt(np.maximum(u, 0.0)),
            np.log1p(4.0 * u) / np.log(5.0),
            np.sin(np.pi * u),
            np.cos(np.pi * u),
        ]
    ).astype(np.float32)


def sample_valid_leads(
    mask_row: np.ndarray,
    max_count: int,
) -> np.ndarray:
    valid = np.flatnonzero(np.asarray(mask_row).astype(bool))
    if len(valid) == 0:
        return valid
    if max_count <= 0 or len(valid) <= max_count:
        return valid

    pos = np.linspace(0, len(valid) - 1, int(max_count))
    return valid[np.round(pos).astype(int)]


def xgb_trajectory_params(
    v410a,
    threads: int,
) -> Dict[str, object]:
    p = dict(v410a.XGB_FROZEN_CORE)
    p.pop("multi_strategy", None)
    p["n_jobs"] = max(1, int(threads))
    p["tree_method"] = "hist"
    return p


def make_xgb_long_train(
    Xeng: np.ndarray,
    bundle,
    indices: Sequence[int],
    leads_per_origin: int = 240,
) -> Tuple[np.ndarray, np.ndarray]:
    Xeng = np.asarray(Xeng, np.float32)
    rows_x = []
    rows_y = []

    for idx in np.asarray(indices, int):
        pos = sample_valid_leads(
            bundle.mask[idx],
            int(leads_per_origin),
        )
        if len(pos) == 0:
            continue

        base = np.repeat(
            Xeng[idx : idx + 1],
            len(pos),
            axis=0,
        )
        lf = lead_features(pos + 1, bundle.y.shape[1])

        rows_x.append(
            np.concatenate([base, lf], axis=1)
        )
        rows_y.append(
            np.asarray(bundle.y[idx, pos], np.float32)
        )

    if not rows_x:
        raise RuntimeError("XGB trajectory long-format training rows empty")

    Xtr = np.vstack(rows_x).astype(np.float32)
    ytr = np.concatenate(rows_y).astype(np.float32)

    if not np.isfinite(Xtr).all() or not np.isfinite(ytr).all():
        raise RuntimeError("XGB trajectory training contains NaN/Inf")
    return Xtr, ytr


def fit_xgb_trajectory(
    v410a,
    Xeng: np.ndarray,
    bundle,
    train_idx: Sequence[int],
    threads: int = 4,
    leads_per_origin: int = 240,
):
    from xgboost import XGBRegressor

    Xtr, ytr = make_xgb_long_train(
        Xeng,
        bundle,
        train_idx,
        leads_per_origin=leads_per_origin,
    )

    model = XGBRegressor(
        **xgb_trajectory_params(v410a, threads)
    )
    model.fit(Xtr, ytr, verbose=False)
    return model, int(len(ytr))


def predict_xgb_trajectory(
    model,
    Xeng_row: np.ndarray,
    k: int = K,
) -> np.ndarray:
    base = np.repeat(
        np.asarray(Xeng_row, np.float32).reshape(1, -1),
        int(k),
        axis=0,
    )
    lf = lead_features(np.arange(1, int(k) + 1), int(k))
    Xte = np.concatenate([base, lf], axis=1).astype(np.float32)
    p = np.asarray(model.predict(Xte), np.float32).reshape(-1)
    if len(p) != int(k) or not np.isfinite(p).all():
        raise RuntimeError("Invalid XGB trajectory prediction")
    return p


# =============================================================================
# Direct RUL XGBoost
# =============================================================================

RUL_XGB_PARAMS = {
    "n_estimators": 180,
    "max_depth": 2,
    "learning_rate": 0.035,
    "subsample": 0.90,
    "colsample_bytree": 0.80,
    "reg_lambda": 8.0,
    "reg_alpha": 0.20,
    "min_child_weight": 3.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
}


def fit_xgb_rul(
    X: np.ndarray,
    y: np.ndarray,
    threads: int = 4,
    sample_weight: Optional[np.ndarray] = None,
    params_override: Optional[Dict[str, object]] = None,
):
    from xgboost import XGBRegressor

    p = dict(RUL_XGB_PARAMS)
    if params_override:
        p.update(params_override)
    p["n_jobs"] = max(1, int(threads))

    model = XGBRegressor(**p)
    model.fit(
        np.asarray(X, np.float32),
        np.asarray(y, np.float32),
        sample_weight=sample_weight,
        verbose=False,
    )
    return model


# =============================================================================
# Generic deep trajectory models
# =============================================================================

@dataclass
class InputScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
        return (
            (X - self.mean[None, None, :])
            / self.std[None, None, :]
        ).astype(np.float32)


@dataclass
class TargetScaler:
    mean: float
    std: float

    def transform(self, y: np.ndarray) -> np.ndarray:
        z = np.asarray(y, np.float32).copy()
        m = np.isfinite(z)
        z[m] = (z[m] - self.mean) / self.std
        return z

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return (
            np.asarray(z, np.float32) * self.std + self.mean
        ).astype(np.float32)


def fit_input_scaler(X: np.ndarray) -> InputScaler:
    a = np.asarray(X, float)
    mean = np.nanmean(a, axis=(0, 1))
    std = np.nanstd(a, axis=(0, 1))
    mean[~np.isfinite(mean)] = 0.0
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    return InputScaler(mean.astype(np.float32), std.astype(np.float32))


def fit_target_scaler(y: np.ndarray, mask: np.ndarray) -> TargetScaler:
    a = np.asarray(y, float)
    m = np.asarray(mask).astype(bool) & np.isfinite(a)
    vals = a[m]
    if len(vals) == 0:
        raise RuntimeError("No valid source SOH target for scaler")

    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return TargetScaler(mean, std)


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


def set_tf_seed(seed: int):
    import tensorflow as tf

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    tf.keras.utils.set_random_seed(int(seed))

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_sequence_encoder(
    inp,
    arch: str,
    dropout: float,
):
    import tensorflow as tf

    a = str(arch).upper()

    if a == "LSTM":
        z = tf.keras.layers.LSTM(
            48,
            return_sequences=False,
            name="encoder_lstm48",
        )(inp)
    elif a == "BILSTM":
        z = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                32,
                return_sequences=False,
            ),
            name="encoder_bilstm32",
        )(inp)
    else:
        raise ValueError(f"Unsupported deep arch={arch}")

    z = tf.keras.layers.Dense(
        64,
        activation="swish",
        name="encoder_dense64",
    )(z)
    z = tf.keras.layers.Dropout(
        float(dropout),
        name="encoder_dropout",
    )(z)
    return z


def build_densek_model(
    input_dim: int,
    arch: str,
    k: int = K,
    dropout: float = 0.12,
    lr: float = 8e-4,
    clipnorm: float = 1.0,
):
    import tensorflow as tf

    tf.keras.backend.clear_session()

    inp = tf.keras.Input(
        shape=(6, int(input_dim)),
        name="FixedTime_history",
    )
    z = build_sequence_encoder(inp, arch, dropout)
    z = tf.keras.layers.Dense(
        128,
        activation="swish",
        name="densek_dense128",
    )(z)
    z = tf.keras.layers.Dropout(
        float(dropout),
        name="densek_dropout",
    )(z)
    out = tf.keras.layers.Dense(
        int(k),
        activation="linear",
        name="SOH_trajectory",
    )(z)

    model = tf.keras.Model(
        inp,
        out,
        name=f"FixedTime_{arch}_DenseK",
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
    input_dim: int,
    arch: str,
    k: int = K,
    dropout: float = 0.12,
    lr: float = 8e-4,
    clipnorm: float = 1.0,
    detail_gate_bias: float = -1.5,
):
    """
    Generic FixedTime counterpart of the Event TrendDetail idea:
        smooth global trend + gated detail residual.
    """
    import tensorflow as tf

    tf.keras.backend.clear_session()

    inp = tf.keras.Input(
        shape=(6, int(input_dim)),
        name="FixedTime_history",
    )
    z = build_sequence_encoder(inp, arch, dropout)

    # Trend branch
    t = tf.keras.layers.Dense(
        96,
        activation="swish",
        name="trend_dense96",
    )(z)
    t = tf.keras.layers.Dropout(
        float(dropout),
        name="trend_dropout",
    )(t)
    trend = tf.keras.layers.Dense(
        int(k),
        activation="linear",
        name="trend_raw",
    )(t)

    # Detail branch
    d = tf.keras.layers.Dense(
        96,
        activation="swish",
        name="detail_dense96",
    )(z)
    d = tf.keras.layers.Dropout(
        float(dropout),
        name="detail_dropout",
    )(d)
    detail = tf.keras.layers.Dense(
        int(k),
        activation="linear",
        name="detail_raw",
    )(d)

    gate = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(
            float(detail_gate_bias)
        ),
        name="detail_gate",
    )(z)

    detail_scaled = tf.keras.layers.Multiply(
        name="detail_scaled"
    )([detail, gate])

    out = tf.keras.layers.Add(
        name="SOH_trajectory"
    )([trend, detail_scaled])

    model = tf.keras.Model(
        inp,
        out,
        name=f"FixedTime_{arch}_TrendDetail",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(lr),
            clipnorm=float(clipnorm),
        ),
        loss=masked_mse_loss(),
    )
    return model


def fit_deep_with_source_es(
    Xseq: np.ndarray,
    bundle,
    train_idx: Sequence[int],
    validation_batteries: Sequence[str],
    fixed_hit,
    arch: str,
    model_kind: str = "TrendDetail",
    max_epochs: int = 70,
    patience: int = 10,
    batch_size: int = 12,
    dropout: float = 0.12,
    lr: float = 8e-4,
    clipnorm: float = 1.0,
    final_refit: bool = True,
    seed: int = RANDOM_SEED,
):
    """
    Source-only early stopping:
        - validation_batteries are source batteries, never outer heldout.
        - select BestEpoch.
        - optional final refit on all provided train_idx for BestEpoch.
    """
    import tensorflow as tf

    train_idx = np.asarray(train_idx, int)
    valset = set(map(normalize_battery, validation_batteries))

    src_bats = np.asarray(
        [normalize_battery(x) for x in fixed_hit.batteries],
        dtype=object,
    )

    is_val = np.asarray(
        [src_bats[i] in valset for i in train_idx],
        bool,
    )

    va_idx = train_idx[is_val]
    tr_idx = train_idx[~is_val]

    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise RuntimeError(
            "Source-only deep ES split produced empty train/validation"
        )

    xsc = fit_input_scaler(Xseq[tr_idx])
    ysc = fit_target_scaler(
        bundle.y[tr_idx],
        bundle.mask[tr_idx],
    )

    Xtr = xsc.transform(Xseq[tr_idx])
    Xva = xsc.transform(Xseq[va_idx])
    ytr = ysc.transform(bundle.y[tr_idx])
    yva = ysc.transform(bundle.y[va_idx])

    set_tf_seed(seed)

    builder = (
        build_trend_detail_model
        if str(model_kind).lower().startswith("trend")
        else build_densek_model
    )

    model = builder(
        input_dim=Xseq.shape[2],
        arch=arch,
        k=bundle.y.shape[1],
        dropout=dropout,
        lr=lr,
        clipnorm=clipnorm,
    )

    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=max(1, int(patience)),
        min_delta=1e-5,
        restore_best_weights=True,
        verbose=0,
    )

    hist = model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=max(1, int(max_epochs)),
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        verbose=0,
        callbacks=[cb],
    )

    vals = np.asarray(hist.history.get("val_loss", []), float)
    best_epoch = (
        int(np.nanargmin(vals) + 1)
        if len(vals) and np.isfinite(vals).any()
        else int(len(hist.history.get("loss", [])))
    )
    best_epoch = max(1, best_epoch)

    if not final_refit:
        return model, xsc, ysc, {
            "BestEpoch": best_epoch,
            "TrainSamples": len(tr_idx),
            "ValidationSamples": len(va_idx),
            "FinalRefit": False,
        }

    # Refit on all source samples with fresh all-source scalers.
    xsc2 = fit_input_scaler(Xseq[train_idx])
    ysc2 = fit_target_scaler(
        bundle.y[train_idx],
        bundle.mask[train_idx],
    )

    Xall = xsc2.transform(Xseq[train_idx])
    yall = ysc2.transform(bundle.y[train_idx])

    set_tf_seed(seed + 100000)

    model2 = builder(
        input_dim=Xseq.shape[2],
        arch=arch,
        k=bundle.y.shape[1],
        dropout=dropout,
        lr=lr,
        clipnorm=clipnorm,
    )
    model2.fit(
        Xall,
        yall,
        epochs=best_epoch,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        verbose=0,
    )

    del model
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

    return model2, xsc2, ysc2, {
        "BestEpoch": best_epoch,
        "TrainSamples": len(tr_idx),
        "ValidationSamples": len(va_idx),
        "FinalSourceSamples": len(train_idx),
        "FinalRefit": True,
    }


def predict_deep_trajectory(
    model,
    xsc: InputScaler,
    ysc: TargetScaler,
    Xseq_row: np.ndarray,
) -> np.ndarray:
    X = xsc.transform(
        np.asarray(Xseq_row, np.float32)[None, :, :]
    )
    ps = np.asarray(
        model.predict(X, batch_size=1, verbose=0),
        np.float32,
    )
    p = ysc.inverse(ps)[0]
    if len(p) != K or not np.isfinite(p).all():
        raise RuntimeError("Invalid deep trajectory prediction")
    return p.astype(np.float32)


# =============================================================================
# Direct RUL deep model
# =============================================================================

@dataclass
class ScalarScaler:
    mean: float
    std: float

    def transform(self, y):
        return (
            (np.asarray(y, np.float32) - self.mean)
            / self.std
        ).astype(np.float32)

    def inverse(self, z):
        return (
            np.asarray(z, np.float32) * self.std + self.mean
        ).astype(np.float32)


def fit_scalar_scaler(y: np.ndarray) -> ScalarScaler:
    y = np.asarray(y, float)
    mean = float(np.mean(y))
    std = float(np.std(y))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return ScalarScaler(mean, std)


def build_rul_deep_model(
    input_dim: int,
    arch: str,
    dropout: float = 0.12,
    lr: float = 8e-4,
    clipnorm: float = 1.0,
):
    import tensorflow as tf

    tf.keras.backend.clear_session()

    inp = tf.keras.Input(
        shape=(6, int(input_dim)),
        name="FixedTime_RUL_history",
    )
    z = build_sequence_encoder(inp, arch, dropout)
    z = tf.keras.layers.Dense(
        48,
        activation="swish",
        name="rul_dense48",
    )(z)
    z = tf.keras.layers.Dropout(
        float(dropout),
        name="rul_dropout",
    )(z)
    out = tf.keras.layers.Dense(
        1,
        activation="linear",
        name="DirectRUL",
    )(z)

    model = tf.keras.Model(
        inp,
        out,
        name=f"FixedTime_{arch}_DirectRUL",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(lr),
            clipnorm=float(clipnorm),
        ),
        loss="mse",
    )
    return model


def fit_rul_deep_source_es(
    Xseq: np.ndarray,
    y: np.ndarray,
    battery_labels: np.ndarray,
    source_batteries: Sequence[str],
    validation_batteries: Sequence[str],
    arch: str,
    max_epochs: int = 100,
    patience: int = 12,
    batch_size: int = 16,
    dropout: float = 0.12,
    lr: float = 8e-4,
    clipnorm: float = 1.0,
    seed: int = RANDOM_SEED,
):
    import tensorflow as tf

    source_set = set(map(normalize_battery, source_batteries))
    valset = set(map(normalize_battery, validation_batteries))

    bats = np.asarray(
        [normalize_battery(x) for x in battery_labels],
        dtype=object,
    )

    src = np.asarray([b in source_set for b in bats], bool)
    val = np.asarray([b in valset for b in bats], bool) & src
    tr = src & ~val

    tr_idx = np.flatnonzero(tr)
    va_idx = np.flatnonzero(val)

    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise RuntimeError("RUL deep source-only ES split empty")

    xsc = fit_input_scaler(Xseq[tr_idx])
    ysc = fit_scalar_scaler(y[tr_idx])

    Xtr = xsc.transform(Xseq[tr_idx])
    Xva = xsc.transform(Xseq[va_idx])
    ytr = ysc.transform(y[tr_idx])
    yva = ysc.transform(y[va_idx])

    set_tf_seed(seed)

    model = build_rul_deep_model(
        Xseq.shape[2],
        arch,
        dropout=dropout,
        lr=lr,
        clipnorm=clipnorm,
    )

    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=max(1, int(patience)),
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=0,
    )

    hist = model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=max(1, int(max_epochs)),
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        verbose=0,
        callbacks=[cb],
    )

    vals = np.asarray(hist.history.get("val_loss", []), float)
    best_epoch = (
        int(np.nanargmin(vals) + 1)
        if len(vals) and np.isfinite(vals).any()
        else max(1, len(hist.history.get("loss", [])))
    )

    # final all-source refit
    src_idx = np.flatnonzero(src)
    xsc2 = fit_input_scaler(Xseq[src_idx])
    ysc2 = fit_scalar_scaler(y[src_idx])

    set_tf_seed(seed + 100000)
    model2 = build_rul_deep_model(
        Xseq.shape[2],
        arch,
        dropout=dropout,
        lr=lr,
        clipnorm=clipnorm,
    )
    model2.fit(
        xsc2.transform(Xseq[src_idx]),
        ysc2.transform(y[src_idx]),
        epochs=best_epoch,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        verbose=0,
    )

    return model2, xsc2, ysc2, {
        "BestEpoch": int(best_epoch),
        "TrainBatteries": len(source_set - valset),
        "ValidationBatteries": len(valset),
        "FinalSourceBatteries": len(source_set),
    }


def predict_rul_deep(
    model,
    xsc: InputScaler,
    ysc: ScalarScaler,
    Xseq_row: np.ndarray,
) -> float:
    z = model.predict(
        xsc.transform(
            np.asarray(Xseq_row, np.float32)[None, :, :]
        ),
        batch_size=1,
        verbose=0,
    )
    p = float(ysc.inverse(np.asarray(z).reshape(-1))[0])
    return float(np.clip(p, 1.0, float(K)))


# =============================================================================
# Source-only inner CV helpers
# =============================================================================

def inner_cv_rul_xgb(
    fixed_hit,
    bundle,
    selected_his: Sequence[str],
    source_batteries: Sequence[str],
    n_folds: int,
    threads: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    Xeng, _, _ = engineer_fixedtime_selected(
        fixed_hit,
        selected_his,
    )

    groups = split_groups(source_batteries, n_folds)
    rows = []

    for fi, val_bats in enumerate(groups, start=1):
        valset = set(val_bats)
        train_bats = [b for b in source_batteries if b not in valset]

        tr = origin80_indices(fixed_hit, train_bats)
        va = origin80_indices(fixed_hit, val_bats)

        ytr = true_rul(bundle, tr)
        yva = true_rul(bundle, va)

        model = fit_xgb_rul(
            Xeng[tr],
            ytr,
            threads=threads,
        )
        p = np.asarray(model.predict(Xeng[va]), float).reshape(-1)
        p = np.clip(p, 1.0, float(K))

        for j, idx in enumerate(va):
            rows.append(
                {
                    "Battery": str(fixed_hit.batteries[idx]),
                    "InnerFold": fi,
                    "TrueRUL": float(yva[j]),
                    "PredRUL": float(p[j]),
                }
            )

        del model
        gc.collect()

    oof = pd.DataFrame(rows).sort_values("Battery").reset_index(drop=True)

    if len(oof) != len(source_batteries):
        raise RuntimeError(
            f"inner RUL OOF rows={len(oof)} vs source={len(source_batteries)}"
        )

    return rul_metrics(oof["TrueRUL"], oof["PredRUL"]), oof


def inner_cv_soh_xgb(
    ftctx,
    selected_his: Sequence[str],
    source_batteries: Sequence[str],
    n_folds: int,
    threads: int,
    leads_per_origin: int = 160,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    fixed_hit = ftctx["fixed_hit"]
    bundle = ftctx["bundle"]
    v410a = ftctx["v410a"]

    Xeng, _, _ = engineer_fixedtime_selected(
        fixed_hit,
        selected_his,
    )

    groups = split_groups(source_batteries, n_folds)
    pred_rows = []

    for fi, val_bats in enumerate(groups, start=1):
        valset = set(val_bats)
        train_bats = [b for b in source_batteries if b not in valset]

        tr = sample_indices(fixed_hit, train_bats, [TEST_ORIGIN])
        va = origin80_indices(fixed_hit, val_bats)

        model, ntrain = fit_xgb_trajectory(
            v410a,
            Xeng,
            bundle,
            tr,
            threads=threads,
            leads_per_origin=leads_per_origin,
        )

        for idx in va:
            p = predict_xgb_trajectory(
                model,
                Xeng[idx],
                bundle.y.shape[1],
            )

            for lead in range(1, bundle.y.shape[1] + 1):
                pos = lead - 1
                if not bool(bundle.mask[idx, pos]):
                    continue
                pred_rows.append(
                    {
                        "Battery": str(fixed_hit.batteries[idx]),
                        "InnerFold": fi,
                        "Lead": lead,
                        "SOH_True": float(bundle.y[idx, pos]),
                        "SOH_Pred": float(p[pos]),
                    }
                )

        del model
        gc.collect()

    oof = pd.DataFrame(pred_rows)
    met = trajectory_metrics(
        oof["SOH_True"].to_numpy(float),
        oof["SOH_Pred"].to_numpy(float),
        np.ones(len(oof), dtype=np.uint8),
    )
    return met, oof


# =============================================================================
# Source-only Top-K / pair search
# =============================================================================

def exhaustive_top2_rul(
    fixed_hit,
    bundle,
    source_batteries: Sequence[str],
    ranking: pd.DataFrame,
    shortlist: int,
    inner_folds: int,
    threads: int,
) -> Tuple[List[str], Dict[str, float], pd.DataFrame]:
    cand = (
        ranking.sort_values("Rank")
        .head(max(2, int(shortlist)))["HI"]
        .astype(str)
        .tolist()
    )
    rankmap = dict(zip(ranking["HI"], ranking["Rank"]))

    rows = []
    best = None
    best_key = None
    best_met = None

    for a, b in itertools_combinations(cand, 2):
        pair = sorted(
            [a, b],
            key=lambda h: (rankmap.get(h, 10**9), h),
        )
        met, _ = inner_cv_rul_xgb(
            fixed_hit,
            bundle,
            pair,
            source_batteries,
            inner_folds,
            threads,
        )
        key = (
            float(met["MAE"]),
            float(met["RMSE"]),
            float(met["SpreadGap"]),
            "|".join(pair),
        )
        rows.append(
            {
                "SelectedTop2": "|".join(pair),
                **met,
            }
        )
        if best_key is None or key < best_key:
            best_key = key
            best = pair
            best_met = met

    return list(best), dict(best_met), pd.DataFrame(rows)


def exhaustive_top2_soh(
    ftctx,
    source_batteries: Sequence[str],
    ranking: pd.DataFrame,
    shortlist: int,
    inner_folds: int,
    threads: int,
    leads_per_origin: int = 160,
) -> Tuple[List[str], Dict[str, float], pd.DataFrame]:
    cand = (
        ranking.sort_values("Rank")
        .head(max(2, int(shortlist)))["HI"]
        .astype(str)
        .tolist()
    )
    rankmap = dict(zip(ranking["HI"], ranking["Rank"]))

    rows = []
    best = None
    best_key = None
    best_met = None

    for a, b in itertools_combinations(cand, 2):
        pair = sorted(
            [a, b],
            key=lambda h: (rankmap.get(h, 10**9), h),
        )
        met, _ = inner_cv_soh_xgb(
            ftctx,
            pair,
            source_batteries,
            inner_folds,
            threads,
            leads_per_origin=leads_per_origin,
        )
        key = (
            float(met["RMSE"]),
            float(met["MAE"]),
            "|".join(pair),
        )
        rows.append(
            {
                "SelectedTop2": "|".join(pair),
                **met,
            }
        )
        if best_key is None or key < best_key:
            best_key = key
            best = pair
            best_met = met

    return list(best), dict(best_met), pd.DataFrame(rows)


def itertools_combinations(items: Sequence[str], r: int):
    # Local generator avoids importing itertools in user-facing stage scripts.
    n = len(items)
    if r != 2:
        import itertools
        yield from itertools.combinations(items, r)
        return
    for i in range(n):
        for j in range(i + 1, n):
            yield items[i], items[j]


# =============================================================================
# Horizon-adaptive MoE
# =============================================================================

def sigmoid_gate(
    k: int,
    c: float,
    tau: float,
) -> np.ndarray:
    lead = np.arange(1, int(k) + 1, dtype=float)
    z = np.clip((lead - float(c)) / max(float(tau), 1e-6), -40, 40)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def search_horizon_gate(
    rows: pd.DataFrame,
    c_grid: Sequence[float],
    tau_grid: Sequence[float],
) -> Dict[str, float]:
    """
    rows requires:
        Battery, Lead, SOH_True, Pred_Near, Pred_Far, Mask
    """
    required = {
        "Battery", "Lead", "SOH_True",
        "Pred_Near", "Pred_Far", "Mask",
    }
    miss = sorted(required - set(rows.columns))
    if miss:
        raise RuntimeError(f"gate search missing columns: {miss}")

    best = None
    best_key = None

    lead = rows["Lead"].to_numpy(float)
    y = rows["SOH_True"].to_numpy(float)
    near = rows["Pred_Near"].to_numpy(float)
    far = rows["Pred_Far"].to_numpy(float)
    mask = rows["Mask"].to_numpy(int).astype(bool)

    for c in c_grid:
        for tau in tau_grid:
            w = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        (lead - float(c)) / max(float(tau), 1e-6),
                        -40,
                        40,
                    )
                )
            )
            p = (1.0 - w) * near + w * far
            met = trajectory_metrics(y, p, mask)
            key = (float(met["RMSE"]), float(met["MAE"]), float(c), float(tau))
            if best_key is None or key < best_key:
                best_key = key
                best = {
                    "GateType": "Sigmoid",
                    "c": float(c),
                    "tau": float(tau),
                    **met,
                }

    return best


# =============================================================================
# RUL ensemble / bootstrap / conformal
# =============================================================================

def source_only_blend_alpha(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    grid: Sequence[float] = tuple(np.linspace(0, 1, 21)),
) -> Dict[str, float]:
    y = np.asarray(y, float)
    pa = np.asarray(p_a, float)
    pb = np.asarray(p_b, float)

    best = None
    best_key = None

    for a in grid:
        p = (1.0 - float(a)) * pa + float(a) * pb
        met = rul_metrics(y, p)
        key = (met["MAE"], met["RMSE"], abs(float(a) - 0.5))
        if best_key is None or key < best_key:
            best_key = key
            best = {"Alpha_B": float(a), **met}

    return best


def paired_bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 20000,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(int(seed))
    B = max(1000, int(n_boot))
    out = np.empty(B, dtype=float)

    for i in range(B):
        idx = rng.integers(0, len(x), size=len(x))
        out[i] = float(np.mean(x[idx]))

    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(np.mean(x)), float(lo), float(hi)


def finite_sample_abs_residual_quantile(
    abs_residuals: np.ndarray,
    coverage: float,
) -> float:
    r = np.sort(
        np.asarray(abs_residuals, float)[
            np.isfinite(abs_residuals)
        ]
    )
    n = len(r)
    if n == 0:
        return np.nan
    k = int(math.ceil((n + 1) * float(coverage)))
    k = min(max(k, 1), n)
    return float(r[k - 1])


def wilson_interval(success: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    phat = success / n
    den = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / den
    half = (
        z
        * math.sqrt(
            phat * (1 - phat) / n
            + z * z / (4 * n * n)
        )
        / den
    )
    return float(max(0.0, center - half)), float(min(1.0, center + half))


# =============================================================================
# Perturbation helpers
# =============================================================================

def perturb_selected_hi_tensor(
    fixed_hit,
    selected_his: Sequence[str],
    target_his: Sequence[str],
    delta_pct: float,
) -> np.ndarray:
    """
    Return a copy of selected raw tensor [N,6,4,Kselected] with the named HIs
    multiplied by (1 + delta_pct/100).
    """
    selected = list(map(str, selected_his))
    target = set(map(str, target_his))
    imap = {h: i for i, h in enumerate(fixed_hit.hi_names)}

    x = np.asarray(
        fixed_hit.values[
            :, :, :, [imap[h] for h in selected]
        ],
        np.float32,
    ).copy()

    factor = np.float32(1.0 + float(delta_pct) / 100.0)

    for j, h in enumerate(selected):
        if h in target:
            x[:, :, :, j] *= factor

    return x


def engineer_selected_tensor(
    tensor: np.ndarray,
    selected_his: Sequence[str],
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Same engineering as engineer_fixedtime_selected(), but operates on a
    supplied selected-HI tensor [N,6,4,K], used for perturbation tests.
    """
    selected = list(map(str, selected_his))
    x = np.asarray(tensor, np.float32)

    if (
        x.ndim != 4
        or x.shape[1] != 6
        or x.shape[2] != len(FIXEDTIME_NODES)
        or x.shape[3] != len(selected)
    ):
        raise RuntimeError(
            f"selected tensor shape={x.shape}, expected "
            f"[N,6,{len(FIXEDTIME_NODES)},{len(selected)}]"
        )

    blocks = []
    names = []
    regs = []

    for j, h in enumerate(selected):
        Xs, ns, rs = single_hi_engineering(x[:, :, :, j], h)
        blocks.append(Xs)
        names.extend(ns)
        regs.append(rs)

    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            Xc, nc, rc = cross_hi_engineering(
                x[:, :, :, i],
                x[:, :, :, j],
                selected[i],
                selected[j],
            )
            blocks.append(Xc)
            names.extend(nc)
            regs.append(rc)

    Xeng = np.concatenate(blocks, axis=1).astype(np.float32)
    return Xeng, names, pd.concat(regs, ignore_index=True)


# =============================================================================
# Pure-function regression test
# =============================================================================

def self_test() -> Dict[str, object]:
    rng = np.random.default_rng(20260824)

    class H:
        pass

    h = H()
    h.hi_names = ["V_Mean", "I_Mean", "Temp_Range"]

    # FIX3 native test uses the actual discovered-node count when available;
    # before data loading, emulate the current 8-node frozen platform.
    global FIXEDTIME_NODES
    old_nodes = FIXEDTIME_NODES
    test_nodes = (
        60.0, 120.0, 180.0, 300.0,
        450.0, 600.0, 900.0, 1200.0,
    )

    try:
        FIXEDTIME_NODES = test_nodes
        h.values = rng.normal(
            size=(10, 6, len(test_nodes), 3)
        ).astype(np.float32)

        X1, n1, r1 = engineer_fixedtime_selected(h, ["V_Mean"])
        X2, n2, r2 = engineer_fixedtime_selected(
            h,
            ["V_Mean", "I_Mean"],
        )
        X3, n3, r3 = engineer_fixedtime_selected(
            h,
            ["V_Mean", "I_Mean", "Temp_Range"],
        )

        assert X1.shape == (10, 158)
        assert X2.shape == (10, 412)
        assert X3.shape == (10, 762)
        assert len(n2) == len(r2) == 412
        assert np.isfinite(X3).all()

        q = finite_sample_abs_residual_quantile(
            np.arange(1, 47, dtype=float),
            0.90,
        )
        assert np.isfinite(q)

        return {
            "NativeNodeCountTest": len(test_nodes),
            "Top1FeatureCount": X1.shape[1],
            "Top2FeatureCount": X2.shape[1],
            "Top3FeatureCount": X3.shape[1],
            "Finite": bool(np.isfinite(X3).all()),
            "ConformalQuantileSanity": float(q),
            "Status": "PASS",
        }
    finally:
        FIXEDTIME_NODES = old_nodes


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))

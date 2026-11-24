# -*- coding: utf-8 -*-
r"""
RWTH48 V4.27
============
Final Adaptive-vs-Fixed Paired Statistical Closure | FIX1

Purpose
-------
Pure statistical post-processing. NO model training. NO timing search.

This script freezes the already-completed results from:
    V4.26 Adaptive PRT0.008 + XGB / SafeResidualMoE
    V4.17 Fixed Event Opt-L3 + FrozenXGB / SafeResidualMoE

and answers three final questions:

Q1. Adaptive SafeResidualMoE vs Fixed Opt-L3 SafeResidualMoE
    - same 47 held-out batteries
    - paired battery-RMSE bootstrap
    - resource saving is reported descriptively
    - IMPORTANT: CI crossing zero does NOT prove equivalence.

Q2. Adaptive SafeResidualMoE vs Fixed Opt-L3 XGB
    - same 47 held-out batteries
    - paired battery-RMSE bootstrap

Q3. Is the SafeResidualMoE correction more useful in Adaptive-L2 than Adaptive-L3?
    - within L2: paired SafeMoE-XGB bootstrap
    - within L3: paired SafeMoE-XGB bootstrap
    - L2-vs-L3 effect heterogeneity: independent stratified bootstrap
    - IMPORTANT: L2/L3 membership is produced by the adaptive policy, not randomized;
      therefore the between-group heterogeneity result is exploratory.

Scientific boundary
-------------------
This script must NOT be used to claim formal equivalence of Adaptive and Fixed-L3.
A non-significant paired difference means "no detectable difference" under this
sample, not mathematical/statistical equivalence. Formal equivalence/noninferiority
would require a predeclared margin and a confirmatory design.

Default input roots
-------------------
<BASE>\RWTH48_V4_26_Adaptive_SafeResidualMoE_Policy_Transfer
<BASE>\RWTH48_V4_17_Safeguarded_Residual_MoE

Default output
--------------
<BASE>\RWTH48_V4_27_Final_Adaptive_Fixed_Paired_Closure

Main outputs
------------
00_preflight.csv
01_battery_level_comparison_matrix.csv
02_primary_paired_bootstrap.csv
03_adaptive_L2_L3_model_effects.csv
04_L2_vs_L3_heterogeneity_bootstrap.csv
05_final_resource_accuracy_table.csv
06_final_method_hierarchy.csv
07_final_claim_registry.csv
08_acceptance_gate.csv
Figure_01_Final_Operating_Points.png
Figure_02_Paired_Effect_Forest.png
99_RWTH48_V4_27_COMPLETE.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ENC = "utf-8-sig"
CODE_VERSION = "V427_FINAL_ADAPTIVE_FIXED_PAIRED_CLOSURE_FIX1"

BASE_DEFAULT = (Path(__file__).resolve().parents[1] / "workspace")
V426_ROOT_NAME = "RWTH48_V4_26_Adaptive_SafeResidualMoE_Policy_Transfer"
V417_ROOT_NAME = "RWTH48_V4_17_Safeguarded_Residual_MoE"
OUTPUT_NAME = "RWTH48_V4_27_Final_Adaptive_Fixed_Paired_Closure"

EXPECTED_BATTERIES = 47
ADAPTIVE_POLICY_ID = "PRT_0p0080"
FIXED_SCALARS = 18.0


# =============================================================================
# 1. CLI / IO
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWTH48 V4.27 final paired statistical closure"
    )
    p.add_argument("--stage", choices=["audit", "all"], default="all")
    p.add_argument("--base", type=Path, default=BASE_DEFAULT)
    p.add_argument("--v426-root", type=Path, default=None)
    p.add_argument("--v417-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()

    a.base = Path(a.base)
    a.v426_root = Path(a.v426_root) if a.v426_root else a.base / V426_ROOT_NAME
    a.v417_root = Path(a.v417_root) if a.v417_root else a.base / V417_ROOT_NAME
    a.output_root = Path(a.output_root) if a.output_root else a.base / OUTPUT_NAME

    if a.bootstrap < 1000:
        p.error("--bootstrap must be >=1000")
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
    return json.loads(p.read_text(encoding="utf-8-sig"))


def normalize_battery(x: object) -> str:
    import re
    s = str(x).strip().upper().replace("_", "-")
    m = re.search(r"RWTH[- ]?0*(\d+)", s)
    if m:
        return f"RWTH-{int(m.group(1)):03d}"
    return s


def stable_seed(*items) -> int:
    raw = "|".join(map(str, items)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % (2**31 - 1)


# =============================================================================
# 2. Input loaders
# =============================================================================

def v426_outer_path(a) -> Path:
    return a.v426_root / "01_outer_battery_metrics.csv"


def v426_pooled_path(a) -> Path:
    return a.v426_root / "02_pooled_metrics.csv"


def v426_complete_path(a) -> Path:
    return a.v426_root / "99_RWTH48_V4_26_COMPLETE.json"


def v417_outer_path(a) -> Path:
    return a.v417_root / "01_frozenXGB_safeResidualMoE_outer_metrics.csv"


def v417_pooled_path(a) -> Path:
    return a.v417_root / "02_frozenXGB_safeResidualMoE_pooled_metrics.csv"


def v417_complete_path(a) -> Path:
    return a.v417_root / "99_RWTH48_V4_17_COMPLETE.json"


def load_v426(a) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    for p in (v426_outer_path(a), v426_pooled_path(a), v426_complete_path(a)):
        if not p.exists():
            raise FileNotFoundError(p)

    complete = load_json(v426_complete_path(a))
    if complete.get("Status") != "PASS":
        raise RuntimeError("V4.26 status != PASS")

    outer = pd.read_csv(v426_outer_path(a), low_memory=False)
    pooled = pd.read_csv(v426_pooled_path(a), low_memory=False)

    req = {
        "OuterHeldoutBattery", "Model", "Schedule", "DiagnosisCount",
        "HI_ScalarInputs", "RMSE", "MAE", "N"
    }
    if not req.issubset(outer.columns):
        raise RuntimeError(
            f"V4.26 outer missing columns: {sorted(req-set(outer.columns))}"
        )

    outer = outer.copy()
    outer["OuterHeldoutBattery"] = outer["OuterHeldoutBattery"].map(normalize_battery)
    outer["RMSE"] = pd.to_numeric(outer["RMSE"], errors="coerce")
    outer["DiagnosisCount"] = pd.to_numeric(outer["DiagnosisCount"], errors="coerce").astype(int)
    outer["HI_ScalarInputs"] = pd.to_numeric(outer["HI_ScalarInputs"], errors="coerce")

    return outer, pooled, complete


def load_v417(a) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    for p in (v417_outer_path(a), v417_pooled_path(a), v417_complete_path(a)):
        if not p.exists():
            raise FileNotFoundError(p)

    complete = load_json(v417_complete_path(a))
    if complete.get("Status") != "PASS":
        raise RuntimeError("V4.17 status != PASS")

    outer = pd.read_csv(v417_outer_path(a), low_memory=False)
    pooled = pd.read_csv(v417_pooled_path(a), low_memory=False)

    req = {
        "OuterHeldoutBattery", "Model", "Role", "RMSE"
    }
    if not req.issubset(outer.columns):
        raise RuntimeError(
            f"V4.17 outer missing columns: {sorted(req-set(outer.columns))}"
        )

    outer = outer.copy()
    outer["OuterHeldoutBattery"] = outer["OuterHeldoutBattery"].map(normalize_battery)
    outer["RMSE"] = pd.to_numeric(outer["RMSE"], errors="coerce")

    return outer, pooled, complete


# =============================================================================
# 3. Preflight / comparison matrix
# =============================================================================

def build_comparison_matrix(v426: pd.DataFrame, v417: pd.DataFrame) -> pd.DataFrame:
    ax = v426[v426["Model"].eq("Adaptive_XGB")].copy()
    am = v426[v426["Model"].eq("Adaptive_SafeResidualMoE")].copy()

    fx = v417[
        v417["Model"].eq("FrozenXGB")
        & v417["Role"].astype(str).str.upper().eq("OPT")
    ].copy()
    fm = v417[
        v417["Model"].eq("SafeResidualMoE")
        & v417["Role"].astype(str).str.upper().eq("OPT")
    ].copy()

    if any(len(x) != EXPECTED_BATTERIES for x in (ax, am, fx, fm)):
        raise RuntimeError(
            "Expected exactly 47 rows for each of: Adaptive XGB, Adaptive SafeMoE, "
            "Fixed Opt-L3 XGB, Fixed Opt-L3 SafeMoE. "
            f"Observed={len(ax)},{len(am)},{len(fx)},{len(fm)}"
        )

    a = ax[[
        "OuterHeldoutBattery", "Schedule", "DiagnosisCount", "HI_ScalarInputs",
        "AcquisitionDuration_s", "RMSE"
    ]].rename(columns={"RMSE": "Adaptive_XGB_RMSE"})

    b = am[[
        "OuterHeldoutBattery", "RMSE"
    ]].rename(columns={"RMSE": "Adaptive_SafeMoE_RMSE"})

    c = fx[[
        "OuterHeldoutBattery", "RMSE"
    ]].rename(columns={"RMSE": "Fixed_L3_XGB_RMSE"})

    d = fm[[
        "OuterHeldoutBattery", "RMSE"
    ]].rename(columns={"RMSE": "Fixed_L3_SafeMoE_RMSE"})

    out = a.merge(b, on="OuterHeldoutBattery", how="inner")
    out = out.merge(c, on="OuterHeldoutBattery", how="inner")
    out = out.merge(d, on="OuterHeldoutBattery", how="inner")

    if len(out) != EXPECTED_BATTERIES or out["OuterHeldoutBattery"].nunique() != EXPECTED_BATTERIES:
        raise RuntimeError("47-battery merge failed")

    out["AdaptiveSafe_minus_AdaptiveXGB"] = (
        out["Adaptive_SafeMoE_RMSE"] - out["Adaptive_XGB_RMSE"]
    )
    out["AdaptiveSafe_minus_FixedSafe"] = (
        out["Adaptive_SafeMoE_RMSE"] - out["Fixed_L3_SafeMoE_RMSE"]
    )
    out["AdaptiveSafe_minus_FixedXGB"] = (
        out["Adaptive_SafeMoE_RMSE"] - out["Fixed_L3_XGB_RMSE"]
    )

    out["ScalarSaving_vs_FixedL3"] = FIXED_SCALARS - out["HI_ScalarInputs"]
    out["ScalarSavingPct_vs_FixedL3"] = (
        out["ScalarSaving_vs_FixedL3"] / FIXED_SCALARS * 100.0
    )

    return out.sort_values("OuterHeldoutBattery").reset_index(drop=True)


def preflight(a, v426: pd.DataFrame, v417: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "Check": "V4.26 47 batteries per model",
            "Observed": int(v426.groupby("Model")["OuterHeldoutBattery"].nunique().min()),
            "Expected": 47,
            "PASS": int(v426.groupby("Model")["OuterHeldoutBattery"].nunique().eq(47).all()),
        },
        {
            "Check": "V4.17 fixed OPT 47 batteries per model",
            "Observed": int(
                v417[v417["Role"].astype(str).str.upper().eq("OPT")]
                .groupby("Model")["OuterHeldoutBattery"].nunique().min()
            ),
            "Expected": 47,
            "PASS": int(
                v417[v417["Role"].astype(str).str.upper().eq("OPT")]
                .groupby("Model")["OuterHeldoutBattery"].nunique().eq(47).all()
            ),
        },
        {
            "Check": "Merged battery comparison matrix",
            "Observed": len(matrix),
            "Expected": 47,
            "PASS": int(len(matrix) == 47),
        },
        {
            "Check": "Adaptive policy only L2/L3",
            "Observed": "|".join(map(str, sorted(matrix["DiagnosisCount"].unique()))),
            "Expected": "2|3",
            "PASS": int(set(matrix["DiagnosisCount"].unique()) == {2, 3}),
        },
        {
            "Check": "Adaptive L2 battery count",
            "Observed": int((matrix["DiagnosisCount"] == 2).sum()),
            "Expected": 27,
            "PASS": int((matrix["DiagnosisCount"] == 2).sum() == 27),
        },
        {
            "Check": "Adaptive L3 battery count",
            "Observed": int((matrix["DiagnosisCount"] == 3).sum()),
            "Expected": 20,
            "PASS": int((matrix["DiagnosisCount"] == 3).sum() == 20),
        },
        {
            "Check": "All battery RMSE finite",
            "Observed": int(
                np.isfinite(
                    matrix[[
                        "Adaptive_XGB_RMSE", "Adaptive_SafeMoE_RMSE",
                        "Fixed_L3_XGB_RMSE", "Fixed_L3_SafeMoE_RMSE"
                    ]].to_numpy(float)
                ).sum()
            ),
            "Expected": 47 * 4,
            "PASS": int(
                np.isfinite(
                    matrix[[
                        "Adaptive_XGB_RMSE", "Adaptive_SafeMoE_RMSE",
                        "Fixed_L3_XGB_RMSE", "Fixed_L3_SafeMoE_RMSE"
                    ]].to_numpy(float)
                ).all()
            ),
        },
    ]
    return pd.DataFrame(rows)


# =============================================================================
# 4. Bootstrap statistics
# =============================================================================

def paired_bootstrap(diff: Sequence[float], reps: int, seed: int) -> Dict[str, object]:
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        raise RuntimeError("paired bootstrap: no finite differences")

    rng = np.random.default_rng(seed)
    vals = np.empty(int(reps), float)

    batch = 1000
    for start in range(0, int(reps), batch):
        n = min(batch, int(reps) - start)
        idx = rng.integers(0, len(d), size=(n, len(d)))
        vals[start:start+n] = d[idx].mean(axis=1)

    return {
        "ObservedMeanDifference": float(d.mean()),
        "MedianDifference": float(np.median(d)),
        "CI95_Low": float(np.quantile(vals, 0.025)),
        "CI95_High": float(np.quantile(vals, 0.975)),
        "P_FirstBetter": float(np.mean(vals < 0)),
        "N": int(len(d)),
        "BootstrapReps": int(reps),
        "FirstWins": int((d < 0).sum()),
        "Ties": int((d == 0).sum()),
        "FirstLosses": int((d > 0).sum()),
    }


def primary_pairwise_bootstrap(matrix: pd.DataFrame, a) -> pd.DataFrame:
    comparisons = [
        (
            "Adaptive SafeResidualMoE - Adaptive XGB",
            "AdaptiveSafe_minus_AdaptiveXGB",
            "Negative means SafeResidualMoE improves the same adaptive policy",
        ),
        (
            "Adaptive SafeResidualMoE - Fixed Opt-L3 SafeResidualMoE",
            "AdaptiveSafe_minus_FixedSafe",
            "Negative means Adaptive SafeMoE has lower battery RMSE; CI including 0 is NOT equivalence",
        ),
        (
            "Adaptive SafeResidualMoE - Fixed Opt-L3 XGB",
            "AdaptiveSafe_minus_FixedXGB",
            "Negative means Adaptive SafeMoE has lower battery RMSE",
        ),
    ]

    rows = []
    for label, col, meaning in comparisons:
        r = paired_bootstrap(
            matrix[col].to_numpy(float),
            a.bootstrap,
            stable_seed(a.seed, label),
        )
        rows.append({
            "Comparison": label,
            "DifferenceDefinition": "First - Second",
            "Interpretation": meaning,
            "StatisticalLevel": "battery-paired",
            **r,
            "CI95EntirelyBelowZero": int(r["CI95_High"] < 0),
            "CI95IncludesZero": int(r["CI95_Low"] <= 0 <= r["CI95_High"]),
        })

    return pd.DataFrame(rows)


def subgroup_effects(matrix: pd.DataFrame, a) -> pd.DataFrame:
    rows = []

    # Overall + L2 + L3, always SafeMoE - XGB under identical adaptive schedules.
    groups = [
        ("All", matrix),
        ("L2", matrix[matrix["DiagnosisCount"] == 2]),
        ("L3", matrix[matrix["DiagnosisCount"] == 3]),
    ]

    for tag, d in groups:
        r = paired_bootstrap(
            d["AdaptiveSafe_minus_AdaptiveXGB"].to_numpy(float),
            a.bootstrap,
            stable_seed(a.seed, "SAFE_XGB", tag),
        )
        rows.append({
            "Subgroup": tag,
            "DiagnosisCount": np.nan if tag == "All" else int(tag[-1]),
            "Comparison": "Adaptive SafeResidualMoE - Adaptive XGB",
            "NegativeMeans": "SafeResidualMoE better",
            **r,
            "CI95EntirelyBelowZero": int(r["CI95_High"] < 0),
        })

    return pd.DataFrame(rows)


def independent_group_heterogeneity_bootstrap(matrix: pd.DataFrame, a) -> pd.DataFrame:
    """
    Compare mean paired model effect between the disjoint adaptive L2 and L3 groups.

    Effect per battery:
        delta_i = RMSE_AdaptiveSafe_i - RMSE_AdaptiveXGB_i

    Heterogeneity contrast:
        mean(delta_L2) - mean(delta_L3)

    Negative => SafeMoE correction is more beneficial in L2.

    Because L2/L3 assignment is adaptive-policy-selected rather than randomized,
    this is exploratory effect heterogeneity, not causal interaction proof.
    """
    d2 = matrix.loc[
        matrix["DiagnosisCount"] == 2,
        "AdaptiveSafe_minus_AdaptiveXGB"
    ].to_numpy(float)
    d3 = matrix.loc[
        matrix["DiagnosisCount"] == 3,
        "AdaptiveSafe_minus_AdaptiveXGB"
    ].to_numpy(float)

    d2 = d2[np.isfinite(d2)]
    d3 = d3[np.isfinite(d3)]
    if len(d2) == 0 or len(d3) == 0:
        raise RuntimeError("L2/L3 heterogeneity bootstrap: empty group")

    observed = float(d2.mean() - d3.mean())

    rng = np.random.default_rng(stable_seed(a.seed, "L2_VS_L3_HETEROGENEITY"))
    vals = np.empty(int(a.bootstrap), float)
    batch = 1000
    for start in range(0, int(a.bootstrap), batch):
        n = min(batch, int(a.bootstrap) - start)
        i2 = rng.integers(0, len(d2), size=(n, len(d2)))
        i3 = rng.integers(0, len(d3), size=(n, len(d3)))
        vals[start:start+n] = d2[i2].mean(axis=1) - d3[i3].mean(axis=1)

    row = {
        "Comparison": "Mean(Safe-XGB effect in L2) - Mean(Safe-XGB effect in L3)",
        "NegativeMeans": "SafeResidualMoE is more beneficial in adaptive L2 than adaptive L3",
        "ObservedHeterogeneity": observed,
        "L2_MeanEffect": float(d2.mean()),
        "L3_MeanEffect": float(d3.mean()),
        "CI95_Low": float(np.quantile(vals, 0.025)),
        "CI95_High": float(np.quantile(vals, 0.975)),
        "P_L2_MoreBeneficial": float(np.mean(vals < 0)),
        "L2_N": int(len(d2)),
        "L3_N": int(len(d3)),
        "BootstrapReps": int(a.bootstrap),
        "CI95EntirelyBelowZero": int(float(np.quantile(vals, 0.975)) < 0),
        "StatisticalBoundary": (
            "Exploratory heterogeneity only; L2/L3 policy membership is not randomized, "
            "so this is not a causal interaction test."
        ),
    }
    return pd.DataFrame([row])


# =============================================================================
# 5. Resource/accuracy + hierarchy + claims
# =============================================================================

def pooled_lookup(df: pd.DataFrame, model: str, role: Optional[str] = None) -> pd.Series:
    q = df[df["Model"].astype(str).eq(model)].copy()
    if role is not None and "Role" in q.columns:
        q = q[q["Role"].astype(str).str.upper().eq(role.upper())]
    if q.empty:
        raise RuntimeError(f"Cannot find pooled row model={model} role={role}")
    return q.iloc[0]


def final_resource_accuracy(v426_pooled: pd.DataFrame, v417_pooled: pd.DataFrame) -> pd.DataFrame:
    ax = pooled_lookup(v426_pooled, "Adaptive_XGB")
    am = pooled_lookup(v426_pooled, "Adaptive_SafeResidualMoE")
    fx = pooled_lookup(v417_pooled, "FrozenXGB", "OPT")
    fm = pooled_lookup(v417_pooled, "SafeResidualMoE", "OPT")

    rows = [
        {
            "OperatingPoint": "Adaptive 2/3 XGB",
            "Role": "Simple adaptive baseline",
            "Policy": ADAPTIVE_POLICY_ID,
            "MeanDiagnoses": float(ax["MeanDiagnoses"]),
            "MeanScalars": float(ax["MeanScalars"]),
            "PooledRMSE": float(ax["RMSE"]),
            "BatteryMacroRMSE": float(ax["BatteryMacroRMSE"]),
        },
        {
            "OperatingPoint": "Adaptive 2/3 SafeResidualMoE",
            "Role": "Preferred accuracy implementation of primary adaptive policy",
            "Policy": ADAPTIVE_POLICY_ID,
            "MeanDiagnoses": float(am["MeanDiagnoses"]),
            "MeanScalars": float(am["MeanScalars"]),
            "PooledRMSE": float(am["RMSE"]),
            "BatteryMacroRMSE": float(am["BatteryMacroRMSE"]),
        },
        {
            "OperatingPoint": "Fixed Event Opt-L3 XGB",
            "Role": "Fixed-budget XGB reference",
            "Policy": "Fixed Event Opt-L3",
            "MeanDiagnoses": 3.0,
            "MeanScalars": 18.0,
            "PooledRMSE": float(fx["RMSE"]),
            "BatteryMacroRMSE": float(fx["BatteryMacroRMSE"]),
        },
        {
            "OperatingPoint": "Fixed Event Opt-L3 SafeResidualMoE",
            "Role": "Accuracy-centric fixed-budget reference",
            "Policy": "Fixed Event Opt-L3",
            "MeanDiagnoses": 3.0,
            "MeanScalars": 18.0,
            "PooledRMSE": float(fm["RMSE"]),
            "BatteryMacroRMSE": float(fm["BatteryMacroRMSE"]),
        },
    ]
    d = pd.DataFrame(rows)

    adaptive_scalars = float(am["MeanScalars"])
    saving = (18.0 - adaptive_scalars) / 18.0 * 100.0
    d["ScalarSavingPct_vsFixedL3"] = np.where(
        d["Policy"].eq(ADAPTIVE_POLICY_ID),
        saving,
        0.0,
    )
    return d


def final_hierarchy() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Priority": 1,
            "Layer": "PRIMARY_DEPLOYMENT_POLICY",
            "Method": "Adaptive 2/3 PRT0.008",
            "Meaning": "Primary decision policy for whether/when another diagnosis is worth acquiring",
        },
        {
            "Priority": 2,
            "Layer": "PREFERRED_PREDICTOR_IMPLEMENTATION",
            "Method": "Adaptive 2/3 + SafeResidualMoE",
            "Meaning": "Same frozen adaptive measurement policy with the lower pooled-RMSE predictor implementation",
        },
        {
            "Priority": 3,
            "Layer": "SIMPLE_ADAPTIVE_BASELINE",
            "Method": "Adaptive 2/3 + XGB",
            "Meaning": "Simpler predictor under the identical adaptive acquisition policy",
        },
        {
            "Priority": 4,
            "Layer": "FIXED_BUDGET_ACCURACY_REFERENCE",
            "Method": "Fixed Event Opt-L3 + SafeResidualMoE",
            "Meaning": "Accuracy-centric fixed three-diagnosis reference",
        },
        {
            "Priority": 5,
            "Layer": "FIXED_BUDGET_XGB_REFERENCE",
            "Method": "Fixed Event Opt-L3 + XGB",
            "Meaning": "Matched fixed-budget XGB reference for timing and resource comparisons",
        },
    ])


def claim_registry(pairwise: pd.DataFrame, subgroup: pd.DataFrame, hetero: pd.DataFrame, resource: pd.DataFrame) -> pd.DataFrame:
    pmap = pairwise.set_index("Comparison")
    h = hetero.iloc[0]
    r = resource.set_index("OperatingPoint")

    adaptive = r.loc["Adaptive 2/3 SafeResidualMoE"]
    fixedsafe = r.loc["Fixed Event Opt-L3 SafeResidualMoE"]
    fixedxgb = r.loc["Fixed Event Opt-L3 XGB"]

    p_fixedsafe = pmap.loc[
        "Adaptive SafeResidualMoE - Fixed Opt-L3 SafeResidualMoE"
    ]
    p_fixedxgb = pmap.loc[
        "Adaptive SafeResidualMoE - Fixed Opt-L3 XGB"
    ]

    return pd.DataFrame([
        {
            "ClaimID": "C1",
            "Claim": "Adaptive PRT0.008 transfers to SafeResidualMoE without model-specific timing re-optimization.",
            "Status": "SUPPORTED",
            "Evidence": "V4.26 frozen-policy experiment; V4.27 statistical closure",
            "Boundary": "Does not establish a SafeResidualMoE-specific timing optimum.",
        },
        {
            "ClaimID": "C2",
            "Claim": (
                f"Adaptive SafeResidualMoE uses {adaptive['ScalarSavingPct_vsFixedL3']:.2f}% fewer measured scalars "
                f"than fixed L3 while its pooled RMSE ({adaptive['PooledRMSE']:.9f}) is close to the fixed-L3 "
                f"SafeResidualMoE reference ({fixedsafe['PooledRMSE']:.9f})."
            ),
            "Status": "SUPPORTED_DESCRIPTIVELY",
            "Evidence": "Frozen resource/pooled metrics",
            "Boundary": (
                f"Battery-paired CI={p_fixedsafe['CI95_Low']:.6g} to {p_fixedsafe['CI95_High']:.6g}; "
                "a CI including zero is not proof of equivalence."
            ),
        },
        {
            "ClaimID": "C3",
            "Claim": "Adaptive SafeResidualMoE has lower pooled RMSE than fixed Opt-L3 XGB at a lower average measurement budget.",
            "Status": "SUPPORTED_DESCRIPTIVELY",
            "Evidence": (
                f"Adaptive pooled={adaptive['PooledRMSE']:.9f}; fixed-XGB pooled={fixedxgb['PooledRMSE']:.9f}; "
                f"paired mean difference={p_fixedxgb['ObservedMeanDifference']:.6g}."
            ),
            "Boundary": "Use paired CI to determine whether battery-level superiority is statistically supported.",
        },
        {
            "ClaimID": "C4",
            "Claim": "SafeResidualMoE appears to provide greater correction benefit in adaptive L2 than adaptive L3.",
            "Status": "EXPLORATORY" if not bool(h["CI95EntirelyBelowZero"]) else "EXPLORATORY_WITH_BOOTSTRAP_SUPPORT",
            "Evidence": (
                f"L2-L3 heterogeneity={h['ObservedHeterogeneity']:.6g}, "
                f"95% CI [{h['CI95_Low']:.6g}, {h['CI95_High']:.6g}]."
            ),
            "Boundary": "L2/L3 membership is policy-selected, not randomized; do not make causal subgroup claims.",
        },
        {
            "ClaimID": "C5",
            "Claim": "Adaptive and fixed-L3 SafeResidualMoE are statistically equivalent.",
            "Status": "NOT_ESTABLISHED",
            "Evidence": "No predeclared equivalence/noninferiority margin for this post-hoc comparison.",
            "Boundary": "Do not replace 'no detectable difference' with 'equivalent'.",
        },
    ])


# =============================================================================
# 6. Figures
# =============================================================================

def make_operating_points_figure(resource: pd.DataFrame, out: Path, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    marker_map = {
        "Adaptive 2/3 XGB": "s",
        "Adaptive 2/3 SafeResidualMoE": "D",
        "Fixed Event Opt-L3 XGB": "o",
        "Fixed Event Opt-L3 SafeResidualMoE": "^",
    }
    offsets = {
        "Adaptive 2/3 XGB": (-125, 18),
        "Adaptive 2/3 SafeResidualMoE": (-125, -40),
        "Fixed Event Opt-L3 XGB": (12, 18),
        "Fixed Event Opt-L3 SafeResidualMoE": (12, -40),
    }

    for r in resource.itertuples(index=False):
        y = 100.0 * float(r.PooledRMSE)
        ax.scatter(
            float(r.MeanScalars), y,
            s=105,
            marker=marker_map.get(r.OperatingPoint, "o"),
            zorder=3,
        )
        ax.annotate(
            f"{r.OperatingPoint}\n{float(r.MeanScalars):.2f} scalars\nRMSE={y:.4f}%",
            (float(r.MeanScalars), y),
            xytext=offsets.get(r.OperatingPoint, (8, 8)),
            textcoords="offset points",
            fontsize=8.5,
            arrowprops=dict(arrowstyle="-", linewidth=0.7),
        )

    ax.set_xlabel("Mean measured scalars")
    ax.set_ylabel("Pooled RMSE (%)")
    ax.set_title("Final adaptive-vs-fixed operating points")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_effect_forest(pairwise: pd.DataFrame, subgroup: pd.DataFrame, hetero: pd.DataFrame, out: Path, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []

    for r in pairwise.itertuples(index=False):
        rows.append({
            "Label": r.Comparison,
            "Mean": r.ObservedMeanDifference,
            "Low": r.CI95_Low,
            "High": r.CI95_High,
            "Type": "paired",
        })

    for tag in ("L2", "L3"):
        r = subgroup[subgroup["Subgroup"].eq(tag)].iloc[0]
        rows.append({
            "Label": f"SafeMoE-XGB within Adaptive {tag}",
            "Mean": r["ObservedMeanDifference"],
            "Low": r["CI95_Low"],
            "High": r["CI95_High"],
            "Type": "subgroup",
        })

    h = hetero.iloc[0]
    rows.append({
        "Label": "L2-vs-L3 model-effect heterogeneity",
        "Mean": h["ObservedHeterogeneity"],
        "Low": h["CI95_Low"],
        "High": h["CI95_High"],
        "Type": "exploratory",
    })

    d = pd.DataFrame(rows)
    y = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    means = d["Mean"].to_numpy(float) * 100.0
    low = d["Low"].to_numpy(float) * 100.0
    high = d["High"].to_numpy(float) * 100.0
    xerr = np.vstack([means-low, high-means])

    ax.errorbar(
        means, y,
        xerr=xerr,
        fmt="o",
        capsize=4,
        linewidth=1.2,
    )
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Label"], fontsize=8.5)
    ax.set_xlabel("Mean battery-RMSE difference (percentage points)")
    ax.set_title("Paired model/resource effects with 95% bootstrap confidence intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 7. Main
# =============================================================================

def main() -> None:
    a = parse_args()
    out = ensure_dir(a.output_root)

    v426_outer, v426_pooled, c26 = load_v426(a)
    v417_outer, v417_pooled, c17 = load_v417(a)

    matrix = build_comparison_matrix(v426_outer, v417_outer)
    pre = preflight(a, v426_outer, v417_outer, matrix)
    save_csv(pre, out / "00_preflight.csv")

    if not pre["PASS"].eq(1).all():
        raise RuntimeError(
            "V4.27 preflight failed:\n" + pre[pre["PASS"].eq(0)].to_string(index=False)
        )

    if a.stage == "audit":
        print("=" * 120)
        print("RWTH48 V4.27 AUDIT PASS")
        print("=" * 120)
        print(pre.to_string(index=False))
        print("\nOutput:", out)
        print("=" * 120)
        return

    pairwise = primary_pairwise_bootstrap(matrix, a)
    subgroup = subgroup_effects(matrix, a)
    hetero = independent_group_heterogeneity_bootstrap(matrix, a)
    resource = final_resource_accuracy(v426_pooled, v417_pooled)
    hierarchy = final_hierarchy()
    claims = claim_registry(pairwise, subgroup, hetero, resource)

    save_csv(matrix, out / "01_battery_level_comparison_matrix.csv")
    save_csv(pairwise, out / "02_primary_paired_bootstrap.csv")
    save_csv(subgroup, out / "03_adaptive_L2_L3_model_effects.csv")
    save_csv(hetero, out / "04_L2_vs_L3_heterogeneity_bootstrap.csv")
    save_csv(resource, out / "05_final_resource_accuracy_table.csv")
    save_csv(hierarchy, out / "06_final_method_hierarchy.csv")
    save_csv(claims, out / "07_final_claim_registry.csv")

    # Acceptance checks are integrity/statistical-computability gates, not superiority gates.
    pidx = pairwise.set_index("Comparison")
    fixedsafe_row = pidx.loc[
        "Adaptive SafeResidualMoE - Fixed Opt-L3 SafeResidualMoE"
    ]

    gate = pd.DataFrame([
        {
            "Check": "47 battery comparison rows",
            "Observed": len(matrix),
            "Expected": 47,
            "PASS": int(len(matrix) == 47),
        },
        {
            "Check": "Primary three paired comparisons complete",
            "Observed": len(pairwise),
            "Expected": 3,
            "PASS": int(len(pairwise) == 3),
        },
        {
            "Check": "Adaptive subgroup counts L2/L3",
            "Observed": f"{int((matrix.DiagnosisCount==2).sum())}|{int((matrix.DiagnosisCount==3).sum())}",
            "Expected": "27|20",
            "PASS": int(
                (matrix.DiagnosisCount==2).sum() == 27
                and (matrix.DiagnosisCount==3).sum() == 20
            ),
        },
        {
            "Check": "All bootstrap intervals finite",
            "Observed": int(
                np.isfinite(
                    pairwise[["ObservedMeanDifference", "CI95_Low", "CI95_High"]].to_numpy(float)
                ).all()
                and np.isfinite(
                    subgroup[["ObservedMeanDifference", "CI95_Low", "CI95_High"]].to_numpy(float)
                ).all()
                and np.isfinite(
                    hetero[["ObservedHeterogeneity", "CI95_Low", "CI95_High"]].to_numpy(float)
                ).all()
            ),
            "Expected": 1,
            "PASS": int(
                np.isfinite(
                    pairwise[["ObservedMeanDifference", "CI95_Low", "CI95_High"]].to_numpy(float)
                ).all()
                and np.isfinite(
                    subgroup[["ObservedMeanDifference", "CI95_Low", "CI95_High"]].to_numpy(float)
                ).all()
                and np.isfinite(
                    hetero[["ObservedHeterogeneity", "CI95_Low", "CI95_High"]].to_numpy(float)
                ).all()
            ),
        },
        {
            "Check": "No formal equivalence claim generated",
            "Observed": int(
                not claims["Status"].astype(str).str.contains("EQUIVALENT", case=False, regex=False).any()
            ),
            "Expected": 1,
            "PASS": int(
                not claims["Status"].astype(str).str.contains("EQUIVALENT", case=False, regex=False).any()
            ),
        },
    ])
    save_csv(gate, out / "08_acceptance_gate.csv")

    make_operating_points_figure(
        resource,
        out / "Figure_01_Final_Operating_Points.png",
        a.dpi,
    )
    make_effect_forest(
        pairwise,
        subgroup,
        hetero,
        out / "Figure_02_Paired_Effect_Forest.png",
        a.dpi,
    )

    status = "PASS" if gate["PASS"].eq(1).all() else "CHECK"

    adaptive_safe = resource[
        resource["OperatingPoint"].eq("Adaptive 2/3 SafeResidualMoE")
    ].iloc[0]
    fixed_safe = resource[
        resource["OperatingPoint"].eq("Fixed Event Opt-L3 SafeResidualMoE")
    ].iloc[0]

    complete = {
        "Task": "RWTH48 V4.27 Final Adaptive-vs-Fixed Paired Statistical Closure",
        "CodeVersion": CODE_VERSION,
        "Status": status,
        "NoNewModelTraining": True,
        "NoNewTimingSearch": True,
        "PrimaryAdaptivePolicy": ADAPTIVE_POLICY_ID,
        "OperatingPointSummary": resource.to_dict("records"),
        "PrimaryPairedStatistics": pairwise.to_dict("records"),
        "AdaptiveL2L3ModelEffects": subgroup.to_dict("records"),
        "ExploratoryL2vsL3Heterogeneity": hetero.to_dict("records"),
        "FinalMethodHierarchy": hierarchy.to_dict("records"),
        "ClaimRegistry": claims.to_dict("records"),
        "KeyBoundary": (
            "A paired confidence interval including zero does not establish equivalence. "
            "Adaptive-vs-fixed near-equality should be described as similar/nearly the same pooled accuracy "
            "unless a predeclared equivalence/noninferiority analysis exists."
        ),
        "Acceptance": gate.to_dict("records"),
    }
    save_json(complete, out / "99_RWTH48_V4_27_COMPLETE.json")

    print("\n" + "=" * 120)
    print("RWTH48 V4.27 FINALIZE |", status)
    print("=" * 120)
    print("\nPrimary paired bootstrap:")
    print(pairwise.to_string(index=False))
    print("\nAdaptive L2/L3 model effects:")
    print(subgroup.to_string(index=False))
    print("\nL2-vs-L3 exploratory heterogeneity:")
    print(hetero.to_string(index=False))
    print("\nFinal resource/accuracy table:")
    print(resource.to_string(index=False))
    print("\nClaim registry:")
    print(claims.to_string(index=False))
    print("\nAcceptance:")
    print(gate.to_string(index=False))
    print("\nOutput:", out)
    print("=" * 120)


if __name__ == "__main__":
    main()

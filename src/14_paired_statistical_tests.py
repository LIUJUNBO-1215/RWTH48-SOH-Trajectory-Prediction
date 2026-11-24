# -*- coding: utf-8 -*-
r"""
RWTH48 V4.19A | Final Statistical Evidence + Paper Summary | FIX1
=================================================================

ZERO-TRAINING group.

This script consumes frozen V4.14B/V4.15/V4.16/V4.17/V4.18-Lite outputs and:
1) formally tests SafeResidualMoE vs FrozenXGB with battery-paired bootstrap;
2) repeats the model comparison by horizon;
3) identifies hard batteries / model wins / timing wins;
4) consolidates the resource-accuracy master table;
5) consolidates the XGB -> V4.16 MoE -> V4.17 SafeResidualMoE ablation;
6) archives the V4.18-Lite negative/model-timing result;
7) creates paper-ready CSVs and PNG figures.

No model fitting, no timing search, no heldout reuse decisions.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CODE_VERSION = "V419A_FINAL_STATISTICAL_EVIDENCE_FIX1"

BASE = (Path(__file__).resolve().parents[1] / "workspace")
V414B_NAME = "RWTH48_V4_14B_FullLifeSOH_Timing_MinResource"
V415_NAME = "RWTH48_V4_15_BudgetAware_Adaptive_Diagnostic_Scheduling"
V416_NAME = "RWTH48_V4_16_MoEOnly_Timing_Replication"
V417_NAME = "RWTH48_V4_17_Safeguarded_Residual_MoE"
V418_NAME = "RWTH48_V4_18_Lite_Local_MoE_Timing_Search"
OUT_NAME = "RWTH48_V4_19A_Final_Statistical_Evidence"

HORIZONS = (
    (1, 100, "Lead001_100"),
    (101, 300, "Lead101_300"),
    (301, 500, "Lead301_500"),
    (501, 700, "Lead501_700"),
    (701, 1000, "Lead701_1000"),
)

ENC = "utf-8-sig"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--v414b-root", type=Path, default=None)
    p.add_argument("--v415-root", type=Path, default=None)
    p.add_argument("--v416-root", type=Path, default=None)
    p.add_argument("--v417-root", type=Path, default=None)
    p.add_argument("--v418-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260827)
    a = p.parse_args()
    a.base = Path(a.base)
    a.v414b_root = Path(a.v414b_root) if a.v414b_root else a.base / V414B_NAME
    a.v415_root = Path(a.v415_root) if a.v415_root else a.base / V415_NAME
    a.v416_root = Path(a.v416_root) if a.v416_root else a.base / V416_NAME
    a.v417_root = Path(a.v417_root) if a.v417_root else a.base / V417_NAME
    a.v418_root = Path(a.v418_root) if a.v418_root else a.base / V418_NAME
    a.output_root = Path(a.output_root) if a.output_root else a.base / OUT_NAME
    if a.bootstrap < 1000:
        p.error("--bootstrap must be >=1000")
    return a


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, p: Path):
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding=ENC)


def save_json(obj, p: Path):
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def stable_seed(seed: int, text: str) -> int:
    h = hashlib.sha256(f"{seed}|{text}".encode()).hexdigest()[:8]
    return int(h, 16) % (2**31 - 1)


def metric(y, pred, mask) -> Dict[str, float]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    m = np.asarray(mask).astype(bool)
    ok = m & np.isfinite(y) & np.isfinite(pred)
    e = pred[ok] - y[ok]
    return {
        "RMSE": float(np.sqrt(np.mean(e * e))) if ok.any() else np.nan,
        "MAE": float(np.mean(np.abs(e))) if ok.any() else np.nan,
        "SSE": float(np.sum(e * e)),
        "SAE": float(np.sum(np.abs(e))),
        "N": int(ok.sum()),
    }


def paired_bootstrap(diff, reps: int, seed: int):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        raise RuntimeError("too few paired observations")
    rng = np.random.default_rng(seed)
    # vectorized in chunks to avoid excessive memory
    means = np.empty(reps, float)
    chunk = 1000
    for i in range(0, reps, chunk):
        n = min(chunk, reps - i)
        idx = rng.integers(0, len(d), size=(n, len(d)))
        means[i:i+n] = d[idx].mean(axis=1)
    return {
        "ObservedMeanDifference": float(d.mean()),
        "CI95_Low": float(np.quantile(means, .025)),
        "CI95_High": float(np.quantile(means, .975)),
        "P_FirstBetter": float(np.mean(means < 0)),
        "Batteries": int(len(d)),
        "BootstrapReps": int(reps),
        "FirstWins": int(np.sum(d < 0)),
        "Ties": int(np.sum(d == 0)),
        "FirstLosses": int(np.sum(d > 0)),
    }


def v417_pred(root: Path, b: str, role: str) -> Path:
    return root / "folds" / b / role / "predictions.npz"


def v417_result(root: Path, b: str, role: str) -> Path:
    return root / "folds" / b / role / "result.json"


def discover_batteries(root: Path) -> List[str]:
    folds = root / "folds"
    if not folds.exists():
        raise FileNotFoundError(folds)
    bats = sorted(
        p.name for p in folds.iterdir()
        if p.is_dir() and v417_pred(root, p.name, "OPT").exists()
    )
    return bats


def preflight(args, bats):
    rows = []
    roots = [
        ("V4.14B", args.v414b_root, "99_RWTH48_V4_14B_COMPLETE.json"),
        ("V4.15", args.v415_root, "99_RWTH48_V4_15_COMPLETE.json"),
        ("V4.16", args.v416_root, "99_RWTH48_V4_16_COMPLETE.json"),
        ("V4.17", args.v417_root, "99_RWTH48_V4_17_COMPLETE.json"),
        ("V4.18", args.v418_root, "99_RWTH48_V4_18_LITE_COMPLETE.json"),
    ]
    for name, root, fn in roots:
        p = root / fn
        ok = p.exists()
        status = ""
        if ok:
            status = str(load_json(p).get("Status", ""))
            ok = status == "PASS"
        rows.append({"Check": f"{name} COMPLETE", "Observed": status or "MISSING", "Expected": "PASS", "PASS": int(ok)})

    missing = 0
    for b in bats:
        for role in ("OPT", "UNIFORM"):
            missing += int(not v417_pred(args.v417_root, b, role).exists())
            missing += int(not v417_result(args.v417_root, b, role).exists())

    rows += [
        {"Check": "47 V4.17 batteries", "Observed": len(bats), "Expected": 47, "PASS": int(len(bats) == 47)},
        {"Check": "V4.17 fold files missing", "Observed": missing, "Expected": 0, "PASS": int(missing == 0)},
    ]
    df = pd.DataFrame(rows)
    if not df.PASS.eq(1).all():
        raise RuntimeError("V4.19A preflight failed:\n" + df[df.PASS.eq(0)].to_string(index=False))
    return df


def build_battery_metrics(args, bats):
    rows = []
    hrows = []
    for b in bats:
        for role in ("OPT", "UNIFORM"):
            z = np.load(v417_pred(args.v417_root, b, role))
            y = np.asarray(z["true"], float)
            mask = np.asarray(z["mask"]).astype(bool)
            preds = {
                "FrozenXGB": np.asarray(z["pred_FrozenXGB"], float),
                "SafeResidualMoE": np.asarray(z["pred_SafeResidualMoE"], float),
            }
            q = load_json(v417_result(args.v417_root, b, role))
            life = int(mask.sum())
            for model, pred in preds.items():
                m = metric(y, pred, mask)
                rows.append({
                    "Battery": b, "Role": role, "Model": model,
                    "RMSE": m["RMSE"], "MAE": m["MAE"], "N": m["N"],
                    "ValidFuturePoints": life,
                    "Schedule": "|".join(map(str, q["Schedule"])),
                    "Guard": int(q["SelectedGuard"]),
                    "Alpha": float(q["SelectedAlpha"]),
                    "Fallback": int(q["FallbackSelected"]),
                })
                lead = np.arange(1, len(y) + 1)
                for lo, hi, hb in HORIZONS:
                    hm = metric(y, pred, mask & (lead >= lo) & (lead <= hi))
                    if hm["N"]:
                        hrows.append({
                            "Battery": b, "Role": role, "Model": model,
                            "HorizonBin": hb, "RMSE": hm["RMSE"], "MAE": hm["MAE"], "N": hm["N"]
                        })
    return pd.DataFrame(rows), pd.DataFrame(hrows)


def model_bootstrap(battery, args):
    out = []
    for role in ("OPT", "UNIFORM"):
        p = battery[battery.Role.eq(role)].pivot(index="Battery", columns="Model", values="RMSE")
        d = p["SafeResidualMoE"] - p["FrozenXGB"]
        out.append({
            "Role": role,
            "Comparison": "SafeResidualMoE - FrozenXGB",
            **paired_bootstrap(d.to_numpy(), args.bootstrap, stable_seed(args.seed, f"overall-{role}"))
        })
    return pd.DataFrame(out)


def horizon_bootstrap(hbattery, args):
    out = []
    for role in ("OPT", "UNIFORM"):
        for hb in [x[2] for x in HORIZONS]:
            d = hbattery[(hbattery.Role.eq(role)) & (hbattery.HorizonBin.eq(hb))]
            p = d.pivot(index="Battery", columns="Model", values="RMSE")
            if not {"SafeResidualMoE", "FrozenXGB"}.issubset(p.columns):
                continue
            diff = p["SafeResidualMoE"] - p["FrozenXGB"]
            out.append({
                "Role": role, "HorizonBin": hb,
                "Comparison": "SafeResidualMoE - FrozenXGB",
                **paired_bootstrap(diff.to_numpy(), args.bootstrap, stable_seed(args.seed, f"h-{role}-{hb}"))
            })
    return pd.DataFrame(out)


def hard_battery_table(battery):
    p = battery.pivot_table(
        index=["Battery", "ValidFuturePoints"],
        columns=["Role", "Model"],
        values="RMSE"
    )
    p.columns = ["_".join(c) for c in p.columns]
    p = p.reset_index()
    p["ModelGain_OPT"] = p["OPT_SafeResidualMoE"] - p["OPT_FrozenXGB"]
    p["TimingGain_XGB"] = p["OPT_FrozenXGB"] - p["UNIFORM_FrozenXGB"]
    p["TimingGain_SafeMoE"] = p["OPT_SafeResidualMoE"] - p["UNIFORM_SafeResidualMoE"]
    p["WorstRank_SafeMoE_OPT"] = p["OPT_SafeResidualMoE"].rank(ascending=False, method="min").astype(int)
    p["ModelImprovementRank"] = p["ModelGain_OPT"].rank(ascending=True, method="min").astype(int)
    p["ModelDegradationRank"] = p["ModelGain_OPT"].rank(ascending=False, method="min").astype(int)
    return p.sort_values("OPT_SafeResidualMoE", ascending=False)


def find_existing(root: Path, names: List[str]):
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def build_master_table(args):
    rows = []

    # V4.14B timing L2-L6.
    p = args.v414b_root / "04_timing_pooled_metrics.csv"
    if p.exists():
        d = pd.read_csv(p, low_memory=False)
        for r in d.itertuples(index=False):
            rows.append({
                "Family": "V4.14B Timing",
                "Configuration": f"{r.Platform}_L{int(r.DiagnosisCount)}_{r.Role}",
                "Model": "FrozenXGB",
                "Platform": str(r.Platform),
                "Diagnoses": float(r.DiagnosisCount),
                "Scalars": float(r.MedianHI_ScalarInputs),
                "Sensors": float(r.MedianPhysicalSensorCount),
                "Duration_s": float(r.MedianAcquisitionDuration_s),
                "RMSE": float(r.RMSE),
                "MAE": float(r.MAE),
                "BatteryMacroRMSE": float(r.BatteryMacroRMSE),
            })

    # V4.15 adaptive.
    p = args.v415_root / "04_adaptive_pooled_metrics.csv"
    if p.exists():
        d = pd.read_csv(p, low_memory=False)
        for r in d.itertuples(index=False):
            rows.append({
                "Family": "V4.15 Adaptive",
                "Configuration": str(r.PolicyID),
                "Model": "FrozenXGB",
                "Platform": "Event",
                "Diagnoses": float(r.MeanDiagnoses),
                "Scalars": float(r.MeanScalars),
                "Sensors": float(r.MeanSensors),
                "Duration_s": float(r.MeanDuration_s),
                "RMSE": float(r.RMSE),
                "MAE": float(r.MAE),
                "BatteryMacroRMSE": float(r.BatteryMacroRMSE),
            })

    # V4.17 final.
    p = args.v417_root / "02_frozenXGB_safeResidualMoE_pooled_metrics.csv"
    if p.exists():
        d = pd.read_csv(p, low_memory=False)
        for r in d.itertuples(index=False):
            rows.append({
                "Family": "V4.17 Final",
                "Configuration": f"{r.Model}_{r.Role}",
                "Model": str(r.Model),
                "Platform": "Event",
                "Diagnoses": 3.0,
                "Scalars": 18.0,
                "Sensors": 2.0,
                "Duration_s": np.nan,
                "RMSE": float(r.RMSE),
                "MAE": float(r.MAE),
                "BatteryMacroRMSE": float(r.BatteryMacroRMSE),
            })

    return pd.DataFrame(rows)


def ablation_table(args):
    rows = []
    # V4.16
    p16 = args.v416_root / "02_frozenXGB_MoE_pooled_metrics.csv"
    if p16.exists():
        d = pd.read_csv(p16, low_memory=False)
        for r in d[d.Role.eq("OPT")].itertuples(index=False):
            rows.append({
                "Stage": "V4.16",
                "Model": str(r.Model),
                "NearProtection": "No" if str(r.Model) == "MoE" else "Base",
                "ResidualLearning": "No",
                "RMSE": float(r.RMSE),
                "MAE": float(r.MAE),
                "BatteryMacroRMSE": float(r.BatteryMacroRMSE),
            })
    p17 = args.v417_root / "02_frozenXGB_safeResidualMoE_pooled_metrics.csv"
    if p17.exists():
        d = pd.read_csv(p17, low_memory=False)
        for r in d[d.Role.eq("OPT")].itertuples(index=False):
            rows.append({
                "Stage": "V4.17",
                "Model": str(r.Model),
                "NearProtection": "Exact hard guard" if str(r.Model) == "SafeResidualMoE" else "Base",
                "ResidualLearning": "Yes" if str(r.Model) == "SafeResidualMoE" else "No",
                "RMSE": float(r.RMSE),
                "MAE": float(r.MAE),
                "BatteryMacroRMSE": float(r.BatteryMacroRMSE),
            })
    return pd.DataFrame(rows)


def negative_result(args):
    q = load_json(args.v418_root / "99_RWTH48_V4_18_LITE_COMPLETE.json")
    rows = []
    for r in q.get("PairedComparisons", []):
        rows.append({"Type": "PairedComparison", **r})
    for r in q.get("ModelTimingInteraction", []):
        rows.append({"Type": "ModelTimingInteraction", **r})
    return pd.DataFrame(rows)


def plots(out: Path, master, battery, hbattery, ablation):
    plt.rcParams["axes.unicode_minus"] = False

    # Figure 1: final horizon comparison.
    d = (
        hbattery[hbattery.Role.eq("OPT")]
        .groupby(["Model", "HorizonBin"], as_index=False)
        .agg(RMSE=("RMSE", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))))
    )
    order = [x[2] for x in HORIZONS]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for model, g in d.groupby("Model"):
        g = g.set_index("HorizonBin").reindex(order)
        ax.plot(order, g.RMSE.to_numpy() * 100, marker="o", label=model)
    ax.set_ylabel("Battery-level RMS aggregate RMSE (%)")
    ax.set_xlabel("Forecast horizon")
    ax.set_title("V4.17 OPT: horizon-wise model comparison")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "Figure_01_V417_Horizon_Model_Comparison.png", dpi=300)
    plt.close(fig)

    # Figure 2: resource-accuracy Pareto scatter.
    if not master.empty:
        d = master[np.isfinite(master.Scalars) & np.isfinite(master.RMSE)].copy()
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for family, g in d.groupby("Family"):
            ax.scatter(g.Scalars, g.RMSE * 100, label=family, s=45)
        ax.set_xlabel("Measured scalars")
        ax.set_ylabel("RMSE (%)")
        ax.set_title("Resource–accuracy landscape")
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(out / "Figure_02_Resource_Accuracy_Landscape.png", dpi=300)
        plt.close(fig)

    # Figure 3: per-battery model gain.
    p = battery[battery.Role.eq("OPT")].pivot(index="Battery", columns="Model", values="RMSE")
    gain = (p["SafeResidualMoE"] - p["FrozenXGB"]).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(gain)), gain.to_numpy() * 100)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(np.arange(len(gain)))
    ax.set_xticklabels(gain.index, rotation=90, fontsize=6)
    ax.set_ylabel("SafeMoE − XGB RMSE (percentage points)")
    ax.set_title("Per-battery model difference under OPT timing")
    fig.tight_layout()
    fig.savefig(out / "Figure_03_PerBattery_Model_Difference.png", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    out = ensure_dir(args.output_root)

    bats = discover_batteries(args.v417_root)
    pf = preflight(args, bats)
    save_csv(pf, out / "00_preflight.csv")

    battery, hbattery = build_battery_metrics(args, bats)
    save_csv(battery, out / "01_V417_battery_model_metrics.csv")
    save_csv(hbattery, out / "02_V417_battery_horizon_metrics.csv")

    mb = model_bootstrap(battery, args)
    hb = horizon_bootstrap(hbattery, args)
    hard = hard_battery_table(battery)
    master = build_master_table(args)
    abl = ablation_table(args)
    neg = negative_result(args)

    save_csv(mb, out / "03_model_paired_bootstrap.csv")
    save_csv(hb, out / "04_horizon_model_paired_bootstrap.csv")
    save_csv(hard, out / "05_hard_battery_analysis.csv")
    save_csv(master, out / "06_master_resource_accuracy_table.csv")
    save_csv(abl, out / "07_XGB_MoE_SafeMoE_ablation.csv")
    save_csv(neg, out / "08_V418_negative_model_timing_result.csv")

    # concise claim table
    claims = []
    opt = mb[mb.Role.eq("OPT")].iloc[0]
    claims.append({
        "Claim": "SafeResidualMoE vs FrozenXGB overall under OPT",
        "Observed": float(opt.ObservedMeanDifference),
        "CI95_Low": float(opt.CI95_Low),
        "CI95_High": float(opt.CI95_High),
        "Supported": int(opt.CI95_High < 0),
    })
    far = hb[(hb.Role.eq("OPT")) & (hb.HorizonBin.eq("Lead701_1000"))]
    if len(far):
        r = far.iloc[0]
        claims.append({
            "Claim": "SafeResidualMoE vs FrozenXGB at lead701-1000",
            "Observed": float(r.ObservedMeanDifference),
            "CI95_Low": float(r.CI95_Low),
            "CI95_High": float(r.CI95_High),
            "Supported": int(r.CI95_High < 0),
        })
    claims = pd.DataFrame(claims)
    save_csv(claims, out / "09_key_claim_support.csv")

    plots(out, master, battery, hbattery, abl)

    acceptance = pd.DataFrame([
        {"Check": "47 batteries", "Observed": len(bats), "Expected": 47, "PASS": int(len(bats) == 47)},
        {"Check": "V4.17 model battery rows", "Observed": len(battery), "Expected": 47*2*2, "PASS": int(len(battery) == 188)},
        {"Check": "Overall model bootstrap roles", "Observed": len(mb), "Expected": 2, "PASS": int(len(mb) == 2)},
        {"Check": "Horizon bootstrap rows", "Observed": len(hb), "Expected": 10, "PASS": int(len(hb) == 10)},
        {"Check": "Finite paired CI", "Observed": int(np.isfinite(mb[["CI95_Low","CI95_High"]].to_numpy()).sum()), "Expected": 4, "PASS": int(np.isfinite(mb[["CI95_Low","CI95_High"]].to_numpy()).all())},
    ])
    save_csv(acceptance, out / "10_acceptance_gate.csv")

    status = "PASS" if acceptance.PASS.eq(1).all() else "CHECK"
    complete = {
        "Task": "RWTH48 V4.19A Final Statistical Evidence",
        "CodeVersion": CODE_VERSION,
        "Status": status,
        "BootstrapReps": args.bootstrap,
        "KeyClaimSupport": claims.to_dict("records"),
        "Acceptance": acceptance.to_dict("records"),
        "Outputs": [
            "03_model_paired_bootstrap.csv",
            "04_horizon_model_paired_bootstrap.csv",
            "05_hard_battery_analysis.csv",
            "06_master_resource_accuracy_table.csv",
            "07_XGB_MoE_SafeMoE_ablation.csv",
            "08_V418_negative_model_timing_result.csv",
            "09_key_claim_support.csv",
            "Figure_01_V417_Horizon_Model_Comparison.png",
            "Figure_02_Resource_Accuracy_Landscape.png",
            "Figure_03_PerBattery_Model_Difference.png",
        ],
    }
    save_json(complete, out / "99_RWTH48_V4_19A_COMPLETE.json")

    print("\n" + "="*110)
    print("V4.19A COMPLETE |", status)
    print("="*110)
    print("\nOverall model paired bootstrap:")
    print(mb.to_string(index=False))
    print("\nHorizon model bootstrap:")
    print(hb.to_string(index=False))
    print("\nKey claim support:")
    print(claims.to_string(index=False))
    print("\nOutput:", out)
    print("="*110)


if __name__ == "__main__":
    main()

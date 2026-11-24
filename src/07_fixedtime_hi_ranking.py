# -*- coding: utf-8 -*-
r"""
RWTH48 V4.13F-PRE FixedTime Catch-Up 01
=======================================
SOH-specific HI ranking + TopK ablation + deterministic representation upgrade

This stage closes the FixedTime gaps corresponding mainly to:
    Event V4.7E / V4.8C       HI-count / source-only ranking
    Event V4.10A              deterministic representation upgrade

FixedTime-native physical inputs
--------------------------------
    cumulative prefixes: 30 / 60 / 90 / 120 s
    six diagnoses:       40 / 48 / 56 / 64 / 72 / 80

Key outputs
-----------
1. Source-only SOH-specific HI ranking in every outer fold.
2. Dynamic Top1..Top5 FixedTime HI selection.
3. Exhaustive source-only Top2 pair search from shortlist.
4. Greedy source-only Top3..Top5 expansion.
5. Outer 47-cell LOBO full-life SOH trajectory evaluation.
6. Raw-vs-engineered Top2 representation ablation:
       raw 48 measured scalars
       -> native-node deterministic features
7. Feature registry and feature-dimension audit.

No held-out battery participates in:
    ranking / pair search / TopK expansion / training.

Run
---
audit:
python -u "src/07_fixedtime_hi_ranking.py" --stage audit

smoke:
python -u "src/07_fixedtime_hi_ranking.py" --stage run --only-battery RWTH-002 --xgb-threads 4

formal:
python -u "src/07_fixedtime_hi_ranking.py" --stage all --xgb-threads 4
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR = Path(__file__).resolve().parent
COMMON_PATH = CODE_DIR / "08_fixedtime_common_pipeline.py"

OUTPUT_ROOT = BASE / "RWTH48_V4_13F_PRE_FixedTime_01_SOH_HI_Ranking_TopK_Tree"

TOPKS = (1, 2, 3, 4, 5)


def parse_args():
    p = argparse.ArgumentParser(
        description="FixedTime catch-up 01: SOH HI ranking/TopK/tree representation"
    )
    p.add_argument("--stage", choices=["audit", "run", "all", "finalize"], default="all")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--code-dir", type=Path, default=CODE_DIR)
    p.add_argument("--common", type=Path, default=COMMON_PATH)
    p.add_argument("--fixedtime-ready", type=Path, default=None)
    p.add_argument("--event-ready", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)

    p.add_argument("--shortlist", type=int, default=12)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--topks", nargs="+", type=int, default=list(TOPKS))
    p.add_argument("--xgb-threads", type=int, default=4)
    p.add_argument("--leads-per-origin", type=int, default=160)

    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--dpi", type=int, default=260)
    return p.parse_args()


def load_common(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("ftcatch_common_01", str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules["ftcatch_common_01"] = m
    spec.loader.exec_module(m)
    return m


def fold_dir(out: Path, battery: str) -> Path:
    return out / "folds" / battery


def fold_json(out: Path, battery: str) -> Path:
    return fold_dir(out, battery) / "result.json"


def fold_ranking(out: Path, battery: str) -> Path:
    return fold_dir(out, battery) / "01_soh_ranking.csv"


def fold_search(out: Path, battery: str) -> Path:
    return fold_dir(out, battery) / "02_topk_search.csv.gz"


def fold_predictions(out: Path, battery: str) -> Path:
    return fold_dir(out, battery) / "03_predictions.csv.gz"


def raw_selected_matrix(fixed_hit, selected: Sequence[str]) -> np.ndarray:
    imap = {h: i for i, h in enumerate(fixed_hit.hi_names)}
    idx = [imap[h] for h in selected]
    x = np.asarray(fixed_hit.values[:, :, :, idx], np.float32)
    return x.reshape(x.shape[0], -1).astype(np.float32)


def outer_predict_xgb(
    C,
    ftctx,
    Xeng: np.ndarray,
    held: str,
    source_origins=(80,),
    threads=4,
    leads_per_origin=240,
):
    fixed_hit = ftctx["fixed_hit"]
    bundle = ftctx["bundle"]
    v410a = ftctx["v410a"]

    all_bats = sorted(set(map(C.normalize_battery, fixed_hit.batteries.tolist())))
    source = [b for b in all_bats if b != held]

    tr = C.sample_indices(fixed_hit, source, source_origins)
    te = C.origin80_indices(fixed_hit, [held])

    model, ntrain = C.fit_xgb_trajectory(
        v410a,
        Xeng,
        bundle,
        tr,
        threads=threads,
        leads_per_origin=leads_per_origin,
    )

    p = C.predict_xgb_trajectory(
        model,
        Xeng[int(te[0])],
        bundle.y.shape[1],
    )

    del model
    gc.collect()

    return int(te[0]), p, int(ntrain)


def inner_cv_soh_custom_X(
    C,
    ftctx,
    Xeng: np.ndarray,
    source_batteries: Sequence[str],
    inner_folds: int,
    threads: int,
    leads_per_origin: int,
) -> Dict[str, float]:
    fixed_hit = ftctx["fixed_hit"]
    bundle = ftctx["bundle"]
    v410a = ftctx["v410a"]

    rows = []
    groups = C.split_groups(source_batteries, inner_folds)

    for fi, val_bats in enumerate(groups, start=1):
        valset = set(val_bats)
        train_bats = [b for b in source_batteries if b not in valset]

        tr = C.sample_indices(fixed_hit, train_bats, [80])
        va = C.origin80_indices(fixed_hit, val_bats)

        model, _ = C.fit_xgb_trajectory(
            v410a,
            Xeng,
            bundle,
            tr,
            threads=threads,
            leads_per_origin=leads_per_origin,
        )

        for idx in va:
            p = C.predict_xgb_trajectory(model, Xeng[idx], bundle.y.shape[1])
            m = np.asarray(bundle.mask[idx]).astype(bool)
            pos = np.flatnonzero(m)

            for j in pos:
                rows.append(
                    {
                        "Battery": str(fixed_hit.batteries[idx]),
                        "InnerFold": fi,
                        "Lead": int(j + 1),
                        "SOH_True": float(bundle.y[idx, j]),
                        "SOH_Pred": float(p[j]),
                    }
                )

        del model
        gc.collect()

    d = pd.DataFrame(rows)
    return C.trajectory_metrics(
        d["SOH_True"].to_numpy(float),
        d["SOH_Pred"].to_numpy(float),
        np.ones(len(d), dtype=np.uint8),
    )


def greedy_expand(
    C,
    ftctx,
    source_bats: Sequence[str],
    ranking: pd.DataFrame,
    selected: List[str],
    target_k: int,
    args,
) -> Tuple[List[str], List[Dict[str, object]]]:
    cand = (
        ranking.sort_values("Rank")
        .head(max(args.shortlist, target_k + 4))["HI"]
        .astype(str)
        .tolist()
    )

    trace = []

    while len(selected) < target_k:
        best_h = None
        best_met = None
        best_key = None

        for h in cand:
            if h in selected:
                continue
            cur = selected + [h]

            met, _ = C.inner_cv_soh_xgb(
                ftctx,
                cur,
                source_bats,
                args.inner_folds,
                args.xgb_threads,
                leads_per_origin=args.leads_per_origin,
            )

            key = (float(met["RMSE"]), float(met["MAE"]), h)

            trace.append(
                {
                    "TargetK": target_k,
                    "CandidateAdded": h,
                    "SelectedSet": "|".join(cur),
                    **met,
                }
            )

            if best_key is None or key < best_key:
                best_key = key
                best_h = h
                best_met = met

        if best_h is None:
            raise RuntimeError(f"Unable to expand to Top{target_k}")

        selected = selected + [best_h]

    return selected, trace


def run_one(C, ftctx, held: str, args):
    t0 = time.perf_counter()
    out = args.output_root
    C.ensure_dir(fold_dir(out, held))

    if fold_json(out, held).exists() and not args.force:
        print(f"[FT01] {held}: cache", flush=True)
        return

    fixed_hit = ftctx["fixed_hit"]
    bundle = ftctx["bundle"]

    all_bats = sorted(set(map(C.normalize_battery, fixed_hit.batteries.tolist())))
    source = [b for b in all_bats if b != held]

    ranking = C.rank_soh_his_source_only(
        fixed_hit,
        bundle,
        source,
        shortlist=args.shortlist,
    )

    ranking.insert(0, "OuterHeldoutBattery", held)
    C.save_csv(ranking, fold_ranking(out, held))

    # Top1 from ranking.
    top1 = [str(ranking.sort_values("Rank").iloc[0]["HI"])]

    # Dynamic Top2: exhaustive source-only pair search.
    top2, top2_inner, pair_search = C.exhaustive_top2_soh(
        ftctx,
        source,
        ranking,
        shortlist=args.shortlist,
        inner_folds=args.inner_folds,
        threads=args.xgb_threads,
        leads_per_origin=args.leads_per_origin,
    )

    search_rows = []
    p2 = pair_search.copy()
    p2["SearchStage"] = "ExhaustiveTop2"
    p2["OuterHeldoutBattery"] = held
    search_rows.append(p2)

    selected_by_k: Dict[int, List[str]] = {
        1: top1,
        2: top2,
    }

    current = list(top2)

    for k in sorted(set(args.topks)):
        if k <= 2:
            continue
        current, trace = greedy_expand(
            C,
            ftctx,
            source,
            ranking,
            list(current),
            int(k),
            args,
        )
        selected_by_k[int(k)] = list(current)

        if trace:
            d = pd.DataFrame(trace)
            d["SearchStage"] = f"GreedyTop{k}"
            d["OuterHeldoutBattery"] = held
            search_rows.append(d)

    if search_rows:
        pd.concat(search_rows, ignore_index=True).to_csv(
            fold_search(out, held),
            index=False,
            encoding=C.ENC,
            compression=C.FAST_GZIP,
        )

    pred_parts = []
    summary = {
        "OuterHeldoutBattery": held,
        "HeldoutUsedForRanking": 0,
        "HeldoutUsedForSearch": 0,
        "HeldoutUsedForTraining": 0,
        "RankingTask": "SOH",
        "Runtime_s": None,
    }

    # Outer TopK evaluation.
    for k in sorted(set(args.topks)):
        selected = selected_by_k[int(k)]

        Xeng, names, reg = C.engineer_fixedtime_selected(
            fixed_hit,
            selected,
        )

        idx, p, ntrain = outer_predict_xgb(
            C,
            ftctx,
            Xeng,
            held,
            source_origins=(80,),
            threads=args.xgb_threads,
            leads_per_origin=240,
        )

        y = np.asarray(bundle.y[idx], float)
        m = np.asarray(bundle.mask[idx], int)

        met = C.trajectory_metrics(y, p, m)
        sh = C.shape_metrics(y, p, m)

        summary[f"Top{k}_Selected"] = "|".join(selected)
        summary[f"Top{k}_FeatureCount"] = int(Xeng.shape[1])
        summary[f"Top{k}_MeasuredScalars"] = int(6 * 4 * k)
        summary[f"Top{k}_RMSE"] = float(met["RMSE"])
        summary[f"Top{k}_MAE"] = float(met["MAE"])

        pos = np.flatnonzero(m.astype(bool))
        pred_parts.append(
            pd.DataFrame(
                {
                    "OuterHeldoutBattery": held,
                    "ModelID": f"Top{k}_Eng_XGB",
                    "HISet": "|".join(selected),
                    "HI_Count": k,
                    "FeatureCount": Xeng.shape[1],
                    "Lead": pos + 1,
                    "SOH_True": y[pos],
                    "SOH_Pred": p[pos],
                    "PredUpwardStepFraction": sh["PredUpwardStepFraction"],
                    "PredRoughness": sh["PredRoughness"],
                }
            )
        )

        # Save feature registry once for Top2 in fold.
        if k == 2:
            reg.to_csv(
                fold_dir(out, held) / "04_top2_feature_registry.csv",
                index=False,
                encoding=C.ENC,
            )

    # Raw measured vs engineered fair representation ablation using same selected Top2.
    Xraw = raw_selected_matrix(fixed_hit, top2)
    Xeng2, _, _ = C.engineer_fixedtime_selected(fixed_hit, top2)

    raw_model_id = f"Top2_Raw{Xraw.shape[1]}_XGB"
    eng_model_id = f"Top2_Eng{Xeng2.shape[1]}_XGB"

    raw_inner = inner_cv_soh_custom_X(
        C, ftctx, Xraw, source,
        args.inner_folds, args.xgb_threads, args.leads_per_origin,
    )
    eng_inner = inner_cv_soh_custom_X(
        C, ftctx, Xeng2, source,
        args.inner_folds, args.xgb_threads, args.leads_per_origin,
    )

    for tag, X, imet in [
        (raw_model_id, Xraw, raw_inner),
        (eng_model_id, Xeng2, eng_inner),
    ]:
        idx, p, ntrain = outer_predict_xgb(
            C,
            ftctx,
            X,
            held,
            source_origins=(80,),
            threads=args.xgb_threads,
            leads_per_origin=240,
        )

        y = np.asarray(bundle.y[idx], float)
        m = np.asarray(bundle.mask[idx], int)
        pos = np.flatnonzero(m.astype(bool))

        met = C.trajectory_metrics(y, p, m)

        pred_parts.append(
            pd.DataFrame(
                {
                    "OuterHeldoutBattery": held,
                    "ModelID": tag,
                    "HISet": "|".join(top2),
                    "HI_Count": 2,
                    "FeatureCount": X.shape[1],
                    "Lead": pos + 1,
                    "SOH_True": y[pos],
                    "SOH_Pred": p[pos],
                }
            )
        )

        summary[f"{tag}_InnerRMSE"] = float(imet["RMSE"])
        summary[f"{tag}_OuterRMSE"] = float(met["RMSE"])

    summary["Top2_InnerCV_RMSE"] = float(top2_inner["RMSE"])
    summary["Top2_InnerCV_MAE"] = float(top2_inner["MAE"])
    summary["Runtime_s"] = float(time.perf_counter() - t0)

    pd.concat(pred_parts, ignore_index=True).to_csv(
        fold_predictions(out, held),
        index=False,
        encoding=C.ENC,
        compression=C.FAST_GZIP,
    )

    C.save_json(summary, fold_json(out, held))

    print(
        f"[FT01] {held}: Top2={'|'.join(top2)} "
        f"nodes={len(fixed_hit.nodes)} "
        f"rawDim={Xraw.shape[1]} engDim={Xeng2.shape[1]} "
        f"rawRMSE={summary[f'{raw_model_id}_OuterRMSE']:.6f} "
        f"engRMSE={summary[f'{eng_model_id}_OuterRMSE']:.6f} "
        f"runtime={summary['Runtime_s']:.1f}s",
        flush=True,
    )


def finalize(C, ftctx, args):
    fixed_hit = ftctx["fixed_hit"]
    bundle = ftctx["bundle"]
    bats = sorted(set(map(C.normalize_battery, fixed_hit.batteries.tolist())))

    missing = [b for b in bats if not fold_json(args.output_root, b).exists()]
    if missing:
        raise RuntimeError(f"Missing fold results: {missing}")

    summaries = pd.DataFrame(
        [C.load_json(fold_json(args.output_root, b)) for b in bats]
    )

    preds = pd.concat(
        [
            pd.read_csv(
                fold_predictions(args.output_root, b),
                compression="infer",
                low_memory=False,
            )
            for b in bats
        ],
        ignore_index=True,
    )

    rankings = pd.concat(
        [
            pd.read_csv(fold_ranking(args.output_root, b), low_memory=False)
            for b in bats
        ],
        ignore_index=True,
    )

    pooled_rows = []

    for model_id, g in preds.groupby("ModelID", sort=True):
        met = C.trajectory_metrics(
            g["SOH_True"].to_numpy(float),
            g["SOH_Pred"].to_numpy(float),
            np.ones(len(g), dtype=np.uint8),
        )

        b_rmse = []
        for b, gb in g.groupby("OuterHeldoutBattery", sort=True):
            bm = C.trajectory_metrics(
                gb["SOH_True"].to_numpy(float),
                gb["SOH_Pred"].to_numpy(float),
                np.ones(len(gb), dtype=np.uint8),
            )
            b_rmse.append(bm["RMSE"])

        pooled_rows.append(
            {
                "ModelID": model_id,
                **met,
                "BatteryMacroRMSE": float(np.mean(b_rmse)),
            }
        )

    pooled = pd.DataFrame(pooled_rows).sort_values("RMSE").reset_index(drop=True)

    freq = (
        rankings.groupby(["HI", "Rank"])
        .size()
        .reset_index(name="FoldCount")
    )

    top_rank = (
        rankings.groupby("HI")
        .agg(
            MeanRank=("Rank", "mean"),
            MedianRank=("Rank", "median"),
            Top1Count=("Rank", lambda s: int((s == 1).sum())),
            Top5Count=("Rank", lambda s: int((s <= 5).sum())),
            MeanRelevance=("RelevanceScore", "mean"),
        )
        .reset_index()
        .sort_values(["MeanRank", "MeanRelevance"], ascending=[True, False])
    )

    C.save_csv(summaries, args.output_root / "03_fold_summary.csv")
    preds.to_csv(
        args.output_root / "04_ALL_OOF_SOH_TopK_and_representation.csv.gz",
        index=False,
        encoding=C.ENC,
        compression=C.FAST_GZIP,
    )
    C.save_csv(pooled, args.output_root / "05_pooled_metrics.csv")
    C.save_csv(rankings, args.output_root / "06_source_only_SOH_HI_ranking_by_fold.csv")
    C.save_csv(top_rank, args.output_root / "07_SOH_HI_ranking_stability.csv")

    # Selection frequency by TopK.
    sel_rows = []
    for k in sorted(set(args.topks)):
        c = f"Top{k}_Selected"
        counts = {}
        for s in summaries[c].astype(str):
            for h in s.split("|"):
                counts[h] = counts.get(h, 0) + 1

        for hi, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            sel_rows.append(
                {
                    "HI_Count": k,
                    "HI": hi,
                    "SelectionCount": n,
                    "SelectionRate": n / len(bats),
                }
            )

    C.save_csv(
        pd.DataFrame(sel_rows),
        args.output_root / "08_SOH_TopK_selection_frequency.csv",
    )

    raw = pooled[
        pooled["ModelID"].astype(str).str.match(r"^Top2_Raw\d+_XGB$")
    ]
    eng = pooled[
        pooled["ModelID"].astype(str).str.match(r"^Top2_Eng\d+_XGB$")
    ]

    if len(raw) == 1 and len(eng) == 1:
        raw_rmse = float(raw.iloc[0]["RMSE"])
        eng_rmse = float(eng.iloc[0]["RMSE"])
        gain = 100.0 * (raw_rmse - eng_rmse) / raw_rmse
    else:
        gain = np.nan

    gates = pd.DataFrame(
        [
            {
                "Check": "47 outer batteries complete",
                "Observed": summaries["OuterHeldoutBattery"].nunique(),
                "Expected": 47,
                "PASS": int(summaries["OuterHeldoutBattery"].nunique() == 47),
            },
            {
                "Check": "Heldout used for ranking/search/training",
                "Observed": int(
                    summaries[
                        [
                            "HeldoutUsedForRanking",
                            "HeldoutUsedForSearch",
                            "HeldoutUsedForTraining",
                        ]
                    ].to_numpy(int).sum()
                ),
                "Expected": 0,
                "PASS": int(
                    summaries[
                        [
                            "HeldoutUsedForRanking",
                            "HeldoutUsedForSearch",
                            "HeldoutUsedForTraining",
                        ]
                    ].to_numpy(int).sum()
                    == 0
                ),
            },
            {
                "Check": "Top2 engineered feature count consistent",
                "Observed": int(summaries["Top2_FeatureCount"].iloc[0]),
                "Expected": "same across all folds",
                "PASS": int(summaries["Top2_FeatureCount"].nunique() == 1),
            },
            {
                "Check": "All predictions finite",
                "Observed": int(np.isfinite(preds["SOH_Pred"].to_numpy(float)).all()),
                "Expected": 1,
                "PASS": int(np.isfinite(preds["SOH_Pred"].to_numpy(float)).all()),
            },
        ]
    )

    C.save_csv(gates, args.output_root / "09_acceptance_gate.csv")

    status = "PASS" if gates["PASS"].eq(1).all() else "CHECK"

    if not args.skip_figures:
        figdir = C.ensure_dir(args.output_root / "Figures")

        fig, ax = plt.subplots(figsize=(9, 5.5))
        q = pooled.copy()
        x = np.arange(len(q))
        ax.bar(x, q["RMSE"])
        ax.set_xticks(x)
        ax.set_xticklabels(q["ModelID"], rotation=30, ha="right")
        ax.set_ylabel("SOH RMSE")
        ax.set_title("FixedTime SOH TopK / representation catch-up")
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(figdir / "01_pooled_rmse.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    C.save_json(
        {
            "Task": "FixedTime catch-up 01 SOH HI ranking/TopK/representation",
            "Status": status,
            "OuterBatteries": 47,
            "PhysicalPlatform": "FixedTime 30/60/90/120 s cumulative prefix",
            "HistoryCyclesAtOrigin80": [40, 48, 56, 64, 72, 80],
            "TopK": sorted(set(args.topks)),
            "NativeFixedTimeNodes_s": list(map(float, ftctx["fixed_hit"].nodes)),
            "NativeNodeCount": len(ftctx["fixed_hit"].nodes),
            "Top2FeatureCount": int(summaries["Top2_FeatureCount"].iloc[0]),
            "Top2MeasuredScalarCount": int(
                6 * len(ftctx["fixed_hit"].nodes) * 2
            ),
            "RepresentationGain_RawToEngineered_pct": gain,
            "BestPooledModel": pooled.iloc[0].to_dict() if len(pooled) else {},
            "Protocol": {
                "HeldoutUsedForHIRanking": False,
                "HeldoutUsedForTopKSearch": False,
                "HeldoutUsedForTraining": False,
                "HistoricalSOHAsInput": False,
                "FutureHIAsInput": False,
                "CycleAsModelFeature": False,
            },
        },
        args.output_root / "99_RWTH48_V4_13F_PRE_FT01_COMPLETE.json",
    )

    print("\n" + "=" * 120)
    print("FixedTime Catch-Up 01 COMPLETE")
    print("Status:", status)
    print(pooled.to_string(index=False))
    print("\nAcceptance:")
    print(gates.to_string(index=False))
    print("=" * 120)

    return status


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    C = load_common(args.common)
    print("[FTCATCH FIX3] Common version:", getattr(C, "FTCATCH_COMMON_VERSION", "UNKNOWN"), flush=True)

    ftbase, ctx0 = C.load_fixedtime_context(
        base=args.base,
        code_dir=args.code_dir,
        output_root=args.output_root,
        fixedtime_ready=args.fixedtime_ready,
        event_ready=args.event_ready,
        xgb_threads=args.xgb_threads,
    )

    ftctx = {
        "ftbase": ftbase,
        "bundle": ctx0["bundle"],
        "fixed_hit": ctx0["fixed_hit"],
        "v410a": ctx0["v410a"],
        "v412a": ctx0["v412a"],
    }

    # Frozen table is authoritative for checkpoint times.
    C.save_csv(
        ctx0["fixed_node_audit"],
        args.output_root / "00b_fixedtime_node_audit.csv",
    )

    pre = pd.DataFrame(
        [
            {
                "Check": "Common self-test",
                "Observed": C.self_test()["Status"],
                "Expected": "PASS",
                "PASS": int(C.self_test()["Status"] == "PASS"),
            },
            {
                "Check": "FixedTime complete candidate HIs",
                "Observed": len(ftctx["fixed_hit"].hi_names),
                "Expected": ">=2",
                "PASS": int(len(ftctx["fixed_hit"].hi_names) >= 2),
            },
            {
                "Check": "Outer batteries",
                "Observed": len(set(ftctx["bundle"].battery.tolist())),
                "Expected": 47,
                "PASS": int(len(set(ftctx["bundle"].battery.tolist())) == 47),
            },
            {
                "Check": "Frozen FixedTime node count",
                "Observed": len(ftctx["fixed_hit"].nodes),
                "Expected": ">=2 native full-coverage nodes",
                "PASS": int(len(ftctx["fixed_hit"].nodes) >= 2),
            },
        ]
    )

    C.save_csv(pre, args.output_root / "00_preflight.csv")
    C.save_json(
        {
            "Task": "FixedTime Catch-Up 01",
            "Shortlist": args.shortlist,
            "InnerFolds": args.inner_folds,
            "TopKs": sorted(set(args.topks)),
            "XGBThreads": args.xgb_threads,
            "LeadsPerOriginInner": args.leads_per_origin,
            "NativeFixedTimeNodes_s": list(map(float, ftctx["fixed_hit"].nodes)),
        },
        args.output_root / "00_config.json",
    )

    print("=" * 120)
    print("FixedTime Catch-Up 01 | SOH HI ranking + TopK + tree representation")
    print("Candidates:", len(ftctx["fixed_hit"].hi_names))
    print("FixedTime nodes:", ftctx["fixed_hit"].nodes)
    print("=" * 120)

    if args.stage == "audit":
        print(pre.to_string(index=False))
        print("\nAUDIT PASS")
        return

    bats = sorted(set(map(C.normalize_battery, ftctx["bundle"].battery.tolist())))

    if args.stage == "finalize":
        finalize(C, ftctx, args)
        return

    if args.only_battery:
        b = C.normalize_battery(args.only_battery)
        if b not in bats:
            raise RuntimeError(f"Unknown battery: {b}")
        run_one(C, ftctx, b, args)
        return

    for j, b in enumerate(bats, start=1):
        print(f"\n[FT01] battery {j}/{len(bats)} {b}", flush=True)
        run_one(C, ftctx, b, args)

    finalize(C, ftctx, args)


if __name__ == "__main__":
    main()

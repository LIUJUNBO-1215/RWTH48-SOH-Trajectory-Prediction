# -*- coding: utf-8 -*-
r"""
RWTH48 V4.24 | Adaptive-Primary Finalization | FIX1
===================================================

NO MODEL TRAINING.
NO NEW TIMING SEARCH.
NO NEW HI SEARCH.

Purpose
-------
This script makes the final manuscript hierarchy consistent with the completed
RWTH48 experiments:

PRIMARY DEPLOYMENT METHOD
    Event adaptive 2/3-diagnosis strategy (V4.15, PRT0.008 resource-saving point)

ACCURACY-CENTRIC FIXED-BUDGET REFERENCE
    Event Opt-L3 fixed 3-diagnosis schedule
    - FrozenXGB reference
    - V4.17 SafeResidualMoE model extension

Therefore:
    Adaptive 2/3 is the MAIN DEPLOYMENT strategy.
    Fixed L3 is retained as an accuracy-oriented control/reference.

Sparse SOH Anchor is intentionally NOT part of this RWTH48 manuscript:
    the RWTH48 branch does not contain a Sparse SOH Anchor experiment.
    No Sparse Anchor figure or claim is generated here.

This script completes:
1) aggressive auto-discovery / recovery of V4.15 battery-level adaptive schedules;
2) adaptive schedule Sankey-like flow;
3) adaptive-vs-fixed-L3 resource/accuracy comparison;
4) Holm correction for the 25 Event LxH paired timing tests from V4.23;
5) final manuscript hierarchy / claim registry with Sparse Anchor removed.

No new model fitting is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


CODE_VERSION = "V424_ADAPTIVE_PRIMARY_FINALIZATION_FIX1"

BASE = (Path(__file__).resolve().parents[1] / "workspace")
V423_NAME = "RWTH48_V4_23_Final_Figure_Statistics_Closure"
V419A_NAME = "RWTH48_V4_19A_Final_Statistical_Evidence"
V417_NAME = "RWTH48_V4_17_Safeguarded_Residual_MoE"
FINAL_CLOSURE_NAME = "RWTH48_V4_21_22_Final_Closure"

OUTPUT_NAME = "RWTH48_V4_24_Adaptive_Primary_Finalization"

EXPECTED_BATTERIES = 47
PRIMARY_POLICY = 0.008
FIXED_L3_SCALARS = 18
ADAPTIVE_MEAN_SCALARS_REFERENCE = 14.553
ADAPTIVE_MEAN_DIAGNOSES_REFERENCE = 2.4255
ADAPTIVE_RMSE_REFERENCE = 0.007329627
FIXED_L3_XGB_RMSE_REFERENCE = 0.007258238208392788
FIXED_L3_SAFE_RMSE_REFERENCE = 0.007194155708834688

ENC = "utf-8-sig"


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="RWTH48 V4.24 Adaptive-Primary Finalization"
    )

    p.add_argument(
        "--stage",
        choices=["audit", "adaptive", "holm", "finalize", "all"],
        default="all",
    )

    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--v415-root", type=Path, default=None)
    p.add_argument("--v423-root", type=Path, default=None)
    p.add_argument("--v419a-root", type=Path, default=None)
    p.add_argument("--v417-root", type=Path, default=None)
    p.add_argument("--final-closure-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)

    # Optional direct pointers if auto-discovery still cannot identify legacy data.
    p.add_argument("--adaptive-file", type=Path, default=None)
    p.add_argument("--resource-table", type=Path, default=None)
    p.add_argument("--lh-bootstrap-file", type=Path, default=None)

    p.add_argument("--policy-threshold", type=float, default=PRIMARY_POLICY)
    p.add_argument("--alpha", type=float, default=0.05)

    a = p.parse_args()
    a.base = Path(a.base)

    a.v415_root = (
        Path(a.v415_root)
        if a.v415_root
        else discover_v415_root(a.base)
    )

    a.v423_root = Path(a.v423_root) if a.v423_root else a.base / V423_NAME
    a.v419a_root = Path(a.v419a_root) if a.v419a_root else a.base / V419A_NAME
    a.v417_root = Path(a.v417_root) if a.v417_root else a.base / V417_NAME
    a.final_closure_root = (
        Path(a.final_closure_root)
        if a.final_closure_root
        else a.base / FINAL_CLOSURE_NAME
    )
    a.output_root = Path(a.output_root) if a.output_root else a.base / OUTPUT_NAME

    if a.adaptive_file is not None:
        a.adaptive_file = Path(a.adaptive_file)
    if a.resource_table is not None:
        a.resource_table = Path(a.resource_table)
    if a.lh_bootstrap_file is not None:
        a.lh_bootstrap_file = Path(a.lh_bootstrap_file)

    if not (0 < a.alpha < 1):
        p.error("--alpha must be in (0,1)")

    return a


# =============================================================================
# Basic utilities
# =============================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, p: Path):
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding=ENC)


def save_json(obj: Any, p: Path):
    ensure_dir(p.parent)

    def cv(x):
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
        json.dumps(obj, ensure_ascii=False, indent=2, default=cv),
        encoding="utf-8",
    )


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def normalize(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(cols: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    cols = list(map(str, cols))
    exact = {c: c for c in cols}
    for a in aliases:
        if a in exact:
            return a

    norm = {normalize(c): c for c in cols}
    for a in aliases:
        n = normalize(a)
        if n in norm:
            return norm[n]
    return None


def num(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


# =============================================================================
# Discover V4.15 root
# =============================================================================

def discover_v415_root(base: Path) -> Path:
    exact = base / "RWTH48_V4_15_BudgetAware_Adaptive_Diagnostic_Scheduling"
    if exact.exists():
        return exact

    candidates = []
    if base.exists():
        for p in base.iterdir():
            if not p.is_dir():
                continue
            n = p.name.lower()
            score = 0
            if "v4_15" in n or "v4.15" in n or "v415" in n:
                score += 10
            if "adaptive" in n:
                score += 5
            if "diagnos" in n:
                score += 2
            if score:
                candidates.append((score, p))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1].name))
        return candidates[0][1]

    return exact


# =============================================================================
# Adaptive schedule extraction
# =============================================================================

BATTERY_KEYS = (
    "Battery", "battery", "BatteryID", "battery_id", "TestBattery",
    "HeldoutBattery", "heldout_battery", "HeldOutBattery",
    "OuterBattery", "test_battery", "battery_name",
)

SCHEDULE_KEYS = (
    "Schedule", "schedule", "SelectedSchedule", "selected_schedule",
    "AdaptiveSchedule", "adaptive_schedule", "FinalSchedule", "final_schedule",
    "DiagnosisSchedule", "diagnosis_schedule", "Path", "path",
    "SelectedCycles", "selected_cycles", "DiagnosisCycles", "diagnosis_cycles",
)

POLICY_KEYS = (
    "Policy", "policy", "PolicyID", "policy_id",
    "Threshold", "threshold", "PRT", "prt",
    "RiskThreshold", "risk_threshold", "TargetRisk", "target_risk",
)

CYCLE1_KEYS = (
    "C1", "c1", "Cycle1", "cycle1", "Diagnosis1", "diagnosis1",
    "FirstCycle", "first_cycle", "FirstDiagnosisCycle",
)
CYCLE2_KEYS = (
    "C2", "c2", "Cycle2", "cycle2", "Diagnosis2", "diagnosis2",
    "SecondCycle", "second_cycle", "SecondDiagnosisCycle",
)
CYCLE3_KEYS = (
    "C3", "c3", "Cycle3", "cycle3", "Diagnosis3", "diagnosis3",
    "ThirdCycle", "third_cycle", "ThirdDiagnosisCycle",
)


def battery_like(v: Any) -> Optional[str]:
    s = str(v).strip()
    if not s:
        return None
    m = re.search(r"RWTH[-_ ]?0*(\d{1,3})", s, flags=re.I)
    if m:
        return f"RWTH-{int(m.group(1)):03d}"
    return None


def parse_policy(v: Any) -> Optional[float]:
    if v is None:
        return None

    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        if 0 < x < 0.1:
            return x
        return None

    s = str(v).lower().replace("p", ".")
    # handles PRT0.008 / 0p008 / threshold=0.008
    vals = re.findall(r"0\.\d+", s)
    for z in vals:
        x = float(z)
        if 0 < x < 0.1:
            return x

    # 0p008 may become 0.008 after replace
    return None


def plausible_cycle(x: Any) -> Optional[int]:
    try:
        z = int(round(float(x)))
    except Exception:
        return None
    if 35 <= z <= 100:
        return z
    return None


def parse_schedule(v: Any) -> Optional[Tuple[int, ...]]:
    if v is None:
        return None

    if isinstance(v, (list, tuple, np.ndarray)):
        vals = []
        for q in v:
            c = plausible_cycle(q)
            if c is not None:
                vals.append(c)
    else:
        s = str(v)
        nums = re.findall(r"(?<!\d)(\d{2,3})(?!\d)", s)
        vals = []
        for q in nums:
            c = plausible_cycle(q)
            if c is not None:
                vals.append(c)

    # preserve order, remove immediate/accidental duplicates
    out = []
    for c in vals:
        if not out or c != out[-1]:
            out.append(c)

    if len(out) not in (2, 3):
        return None
    if any(out[i] >= out[i+1] for i in range(len(out)-1)):
        return None
    return tuple(out)


def cycles_from_record(rec: Mapping[str, Any]) -> Optional[Tuple[int, ...]]:
    # direct schedule
    for k in SCHEDULE_KEYS:
        if k in rec:
            s = parse_schedule(rec[k])
            if s is not None:
                return s

    # normalized direct schedule key
    nk = {normalize(k): k for k in rec.keys()}
    for key in SCHEDULE_KEYS:
        if normalize(key) in nk:
            s = parse_schedule(rec[nk[normalize(key)]])
            if s is not None:
                return s

    def get_by_aliases(aliases):
        for k in aliases:
            if k in rec:
                return plausible_cycle(rec[k])
        for k in aliases:
            n = normalize(k)
            if n in nk:
                return plausible_cycle(rec[nk[n]])
        return None

    c1 = get_by_aliases(CYCLE1_KEYS)
    c2 = get_by_aliases(CYCLE2_KEYS)
    c3 = get_by_aliases(CYCLE3_KEYS)

    vals = [x for x in (c1, c2, c3) if x is not None]
    if len(vals) in (2,3) and all(vals[i] < vals[i+1] for i in range(len(vals)-1)):
        return tuple(vals)

    return None


def battery_from_record(rec: Mapping[str, Any]) -> Optional[str]:
    nk = {normalize(k): k for k in rec.keys()}

    for k in BATTERY_KEYS:
        if k in rec:
            b = battery_like(rec[k])
            if b:
                return b

    for k in BATTERY_KEYS:
        n = normalize(k)
        if n in nk:
            b = battery_like(rec[nk[n]])
            if b:
                return b

    # fallback: inspect values
    for v in rec.values():
        b = battery_like(v)
        if b:
            return b

    return None


def policy_from_record(rec: Mapping[str, Any]) -> Optional[float]:
    nk = {normalize(k): k for k in rec.keys()}

    for k in POLICY_KEYS:
        if k in rec:
            q = parse_policy(rec[k])
            if q is not None:
                return q

    for k in POLICY_KEYS:
        n = normalize(k)
        if n in nk:
            q = parse_policy(rec[nk[n]])
            if q is not None:
                return q

    return None


def schedule_score(
    battery: str,
    schedule: Tuple[int, ...],
    policy: Optional[float],
    target_policy: float,
) -> float:
    score = 0.0

    # correct policy is the strongest evidence
    if policy is not None:
        if abs(policy - target_policy) <= 5e-6:
            score += 20
        elif abs(policy - target_policy) <= 5e-4:
            score += 10
        else:
            score -= 5

    if len(schedule) in (2,3):
        score += 5

    # weak consistency checks from the completed V4.15 outcome;
    # these are not used to fabricate schedules, only to rank duplicates.
    if schedule[0] == 44:
        score += 2
    if schedule[-1] <= 80:
        score += 1

    return score


def extract_csv_candidates(p: Path, target_policy: float) -> List[dict]:
    try:
        d = pd.read_csv(p, low_memory=False)
    except Exception:
        return []

    rows = []

    for idx, rec in d.iterrows():
        r = rec.to_dict()
        b = battery_from_record(r)
        s = cycles_from_record(r)
        policy = policy_from_record(r)

        if b is None or s is None:
            continue

        rows.append({
            "Battery": b,
            "Schedule": "|".join(map(str, s)),
            "DiagnosisCount": len(s),
            "Policy": policy,
            "SourceFile": str(p),
            "SourceKind": "CSV",
            "SourceRow": int(idx),
            "CandidateScore": schedule_score(b, s, policy, target_policy),
        })

    return rows


def walk_json_records(obj: Any, inherited_policy=None):
    """
    Yields dictionaries recursively.  Parent policy is propagated to nested folds.
    """
    if isinstance(obj, dict):
        current_policy = inherited_policy
        q = policy_from_record(obj)
        if q is not None:
            current_policy = q

        yield obj, current_policy

        for v in obj.values():
            yield from walk_json_records(v, current_policy)

    elif isinstance(obj, list):
        for v in obj:
            yield from walk_json_records(v, inherited_policy)


def extract_json_candidates(p: Path, target_policy: float) -> List[dict]:
    try:
        obj = read_json(p)
    except Exception:
        return []

    rows = []
    n = 0

    for rec, inherited_policy in walk_json_records(obj):
        b = battery_from_record(rec)
        s = cycles_from_record(rec)
        policy = policy_from_record(rec)
        if policy is None:
            policy = inherited_policy

        if b is None or s is None:
            continue

        rows.append({
            "Battery": b,
            "Schedule": "|".join(map(str, s)),
            "DiagnosisCount": len(s),
            "Policy": policy,
            "SourceFile": str(p),
            "SourceKind": "JSON",
            "SourceRow": n,
            "CandidateScore": schedule_score(b, s, policy, target_policy),
        })
        n += 1

    return rows


def candidate_files(a) -> List[Path]:
    if a.adaptive_file is not None:
        return [a.adaptive_file]

    roots = []
    if a.v415_root.exists():
        roots.append(a.v415_root)

    # Also search adaptive/V4.15-like directories in BASE.
    if a.base.exists():
        for p in a.base.iterdir():
            if not p.is_dir() or p in roots:
                continue
            n = p.name.lower()
            if (
                "v4_15" in n
                or "v415" in n
                or ("adaptive" in n and "rwth48" in n)
            ):
                roots.append(p)

    files = []
    for root in roots:
        for ext in ("*.csv", "*.json"):
            files.extend(root.rglob(ext))

    # prioritize likely schedule/fold files
    def priority(p):
        n = p.name.lower()
        s = 0
        for k, w in (
            ("schedule", 10),
            ("adaptive", 8),
            ("policy", 7),
            ("outer", 5),
            ("fold", 4),
            ("result", 3),
            ("summary", 2),
        ):
            if k in n:
                s += w
        return (-s, len(str(p)), str(p))

    return sorted(set(files), key=priority)


def discover_adaptive_candidates(a) -> pd.DataFrame:
    rows = []

    for i, p in enumerate(candidate_files(a), 1):
        if p.suffix.lower() == ".csv":
            q = extract_csv_candidates(p, a.policy_threshold)
        elif p.suffix.lower() == ".json":
            q = extract_json_candidates(p, a.policy_threshold)
        else:
            q = []

        rows.extend(q)

    if not rows:
        return pd.DataFrame()

    d = pd.DataFrame(rows)

    # exact target policy preferred if available
    pol = pd.to_numeric(d["Policy"], errors="coerce")
    exact = np.isfinite(pol) & (np.abs(pol - a.policy_threshold) <= 5e-6)

    if exact.any():
        d.loc[exact, "CandidateScore"] += 20

    return d


def select_best_adaptive_set(candidates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select a coherent 47-battery set.

    First score each source file by:
    - unique battery coverage;
    - percentage of 2/3 diagnosis schedules;
    - first-cycle consistency;
    - candidate confidence.

    Then choose the best file.  If it has incomplete coverage, fill missing
    batteries from the globally highest-scoring candidates.
    """
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    file_rows = []

    for source, g in candidates.groupby("SourceFile"):
        gb = (
            g.sort_values(["CandidateScore"], ascending=False)
            .drop_duplicates("Battery")
        )

        first = gb["Schedule"].str.split("|").str[0]
        file_rows.append({
            "SourceFile": source,
            "SourceKind": gb["SourceKind"].iloc[0],
            "UniqueBatteries": gb["Battery"].nunique(),
            "Rows": len(gb),
            "MeanCandidateScore": float(gb["CandidateScore"].mean()),
            "PctTwoOrThree": float(gb["DiagnosisCount"].isin([2,3]).mean()),
            "PctFirst44": float(first.eq("44").mean()),
        })

    source_audit = pd.DataFrame(file_rows)

    source_audit["SetScore"] = (
        source_audit["UniqueBatteries"] * 100
        + source_audit["PctTwoOrThree"] * 10
        + source_audit["PctFirst44"] * 3
        + source_audit["MeanCandidateScore"]
    )
    source_audit = source_audit.sort_values("SetScore", ascending=False)

    best_source = source_audit.iloc[0]["SourceFile"]

    selected = (
        candidates[candidates["SourceFile"].eq(best_source)]
        .sort_values(["CandidateScore"], ascending=False)
        .drop_duplicates("Battery")
        .copy()
    )

    # fill missing from all candidates only if needed
    all_bats = sorted(candidates["Battery"].unique())
    missing = [b for b in all_bats if b not in set(selected["Battery"])]

    if missing:
        fill = (
            candidates[candidates["Battery"].isin(missing)]
            .sort_values(["CandidateScore"], ascending=False)
            .drop_duplicates("Battery")
        )
        selected = pd.concat([selected, fill], ignore_index=True)

    # if multiple candidate sources mention >47 battery names, keep RWTH-001..048
    # but exclude RWTH-028 only when 48 batteries were discovered.
    if selected["Battery"].nunique() > EXPECTED_BATTERIES:
        if "RWTH-028" in set(selected["Battery"]):
            selected = selected[~selected["Battery"].eq("RWTH-028")].copy()

    selected = (
        selected.sort_values(["Battery", "CandidateScore"], ascending=[True, False])
        .drop_duplicates("Battery")
        .sort_values("Battery")
        .reset_index(drop=True)
    )

    return selected, source_audit


# =============================================================================
# Adaptive figures and comparison
# =============================================================================

def schedule_edges(schedules: pd.DataFrame) -> pd.DataFrame:
    cnt = Counter()

    for s in schedules["Schedule"].astype(str):
        vals = [int(x) for x in s.split("|")]
        nodes = [f"C{x}" for x in vals] + ["STOP"]
        for u, v in zip(nodes[:-1], nodes[1:]):
            cnt[(u, v)] += 1

    return pd.DataFrame([
        {"Source": u, "Target": v, "Count": n}
        for (u, v), n in sorted(cnt.items())
    ])


def node_sort_key(n: str):
    if n == "STOP":
        return 999
    m = re.search(r"\d+", n)
    return int(m.group()) if m else 0


def draw_flow(edges: pd.DataFrame, outpng: Path):
    if edges.empty:
        return

    # Determine diagnosis stage rather than raw cycle as x-axis.
    # C44 can appear stage1, others as stage2/3.
    sources = set(edges["Source"])
    targets = set(edges["Target"])
    nodes = sorted(sources | targets, key=node_sort_key)

    # Build graph levels from C44.
    level = {"C44": 0}
    changed = True
    for _ in range(10):
        changed = False
        for r in edges.itertuples(index=False):
            if r.Source in level:
                new = level[r.Source] + 1
                if r.Target == "STOP":
                    new = max(new, 3)
                if r.Target not in level or new < level[r.Target]:
                    level[r.Target] = new
                    changed = True

    for n in nodes:
        if n not in level:
            level[n] = 1 if n != "STOP" else 3

    # Spread nodes vertically by cycle within each level.
    ypos = {}
    for lv in sorted(set(level.values())):
        ns = [n for n in nodes if level[n] == lv]
        ns = sorted(ns, key=node_sort_key)
        vals = np.linspace(0.12, 0.88, max(len(ns), 2))
        if len(ns) == 1:
            ypos[ns[0]] = 0.5
        else:
            for n, y in zip(ns, vals):
                ypos[n] = float(y)

    xpos = {n: 0.08 + 0.29*min(level[n], 3) for n in nodes}
    maxc = max(int(edges["Count"].max()), 1)

    fig, ax = plt.subplots(figsize=(11.8, 6.6))

    for r in edges.itertuples(index=False):
        x0, y0 = xpos[r.Source], ypos[r.Source]
        x1, y1 = xpos[r.Target], ypos[r.Target]

        verts = [
            (x0, y0),
            (x0 + 0.38*(x1-x0), y0),
            (x0 + 0.62*(x1-x0), y1),
            (x1, y1),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        patch = PathPatch(
            MplPath(verts, codes),
            fill=False,
            linewidth=0.8 + 9.0*r.Count/maxc,
            alpha=0.28,
        )
        ax.add_patch(patch)

        ax.text(
            (x0+x1)/2,
            (y0+y1)/2,
            str(int(r.Count)),
            fontsize=8,
            ha="center",
            va="center",
        )

    for n in nodes:
        x, y = xpos[n], ypos[n]
        ax.add_patch(Rectangle((x-0.018, y-0.028), 0.036, 0.056, alpha=0.82))
        ax.text(x, y+0.045, n, ha="center", va="bottom", fontsize=9)

    ax.text(0.08, 0.97, "1st diagnosis", ha="center", fontsize=10)
    ax.text(0.37, 0.97, "2nd diagnosis", ha="center", fontsize=10)
    ax.text(0.66, 0.97, "3rd diagnosis / branch", ha="center", fontsize=10)
    ax.text(0.95, 0.97, "STOP", ha="center", fontsize=10)

    ax.set_xlim(0, 1.03)
    ax.set_ylim(0.04, 1.0)
    ax.set_axis_off()
    ax.set_title("Primary deployment policy: adaptive 2/3-diagnosis paths")
    fig.tight_layout()
    fig.savefig(outpng, dpi=300)
    plt.close(fig)


def adaptive_summary(schedules: pd.DataFrame) -> pd.DataFrame:
    first = schedules["Schedule"].str.split("|").str[0]
    second = schedules["Schedule"].str.split("|").str[1]

    return pd.DataFrame([{
        "Batteries": int(schedules["Battery"].nunique()),
        "MeanDiagnoses": float(schedules["DiagnosisCount"].mean()),
        "TwoDiagnosisBatteries": int((schedules["DiagnosisCount"] == 2).sum()),
        "ThreeDiagnosisBatteries": int((schedules["DiagnosisCount"] == 3).sum()),
        "TwoDiagnosisFraction": float((schedules["DiagnosisCount"] == 2).mean()),
        "ThreeDiagnosisFraction": float((schedules["DiagnosisCount"] == 3).mean()),
        "DominantFirstCycle": int(first.value_counts().index[0]),
        "DominantFirstCycleCount": int(first.value_counts().iloc[0]),
        "UniqueSecondCycles": "|".join(sorted(second.dropna().unique(), key=lambda x: int(x))),
    }])


def find_resource_table(a) -> Optional[Path]:
    if a.resource_table is not None and a.resource_table.exists():
        return a.resource_table

    candidates = [
        a.v419a_root / "06_master_resource_accuracy_table.csv",
        a.final_closure_root / "03_Final_Paper_Summary" / "02_final_resource_accuracy_table.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # fallback search
    for root in (a.v419a_root, a.final_closure_root, a.v415_root):
        if root.exists():
            for p in root.rglob("*.csv"):
                n = p.name.lower()
                if "resource" in n and ("accuracy" in n or "master" in n):
                    return p
    return None


def row_text(df: pd.DataFrame) -> pd.Series:
    return df.astype(str).agg(" | ".join, axis=1).str.lower()


def extract_resource_operating_points(a) -> pd.DataFrame:
    p = find_resource_table(a)

    rows = []

    if p is not None:
        try:
            d = pd.read_csv(p, low_memory=False)
            txt = row_text(d)

            masks = {
                "Adaptive 2/3 PRT0.008": (
                    txt.str.contains("adaptive", regex=False)
                    & (
                        txt.str.contains("0.008", regex=False)
                        | txt.str.contains("0p008", regex=False)
                    )
                ),
                "Fixed Event Opt-L3 XGB": (
                    txt.str.contains("event", regex=False)
                    & txt.str.contains("l3", regex=False)
                    & (
                        txt.str.contains("best", regex=False)
                        | txt.str.contains("opt", regex=False)
                    )
                    & txt.str.contains("xgb", regex=False)
                ),
                "Fixed Event Opt-L3 SafeMoE": (
                    txt.str.contains("safe", regex=False)
                    & txt.str.contains("l3", regex=False)
                    & (
                        txt.str.contains("best", regex=False)
                        | txt.str.contains("opt", regex=False)
                    )
                ),
            }

            for label, mask in masks.items():
                g = d[mask]
                if len(g):
                    # preserve first best matching row in raw form
                    rec = g.iloc[0].to_dict()
                    rec2 = {
                        "OperatingPoint": label,
                        "Source": str(p),
                        "RawRow": json.dumps(
                            {str(k): str(v) for k,v in rec.items()},
                            ensure_ascii=False,
                        ),
                    }

                    # detect likely columns
                    rmsec = find_col(d.columns, ("RMSE", "PooledRMSE", "rmse"))
                    scalc = find_col(d.columns, ("Scalars", "MeasuredScalars", "measured_scalars"))
                    diagc = find_col(d.columns, ("DiagnosisCount", "Diagnoses", "AvgDiagnoses"))
                    durc = find_col(d.columns, ("Duration_s", "MeasurementDuration_s", "Duration", "Seconds"))

                    if rmsec:
                        rec2["RMSE"] = num(rec.get(rmsec))
                    if scalc:
                        rec2["MeasuredScalars"] = num(rec.get(scalc))
                    if diagc:
                        rec2["Diagnoses"] = num(rec.get(diagc))
                    if durc:
                        rec2["Duration_s"] = num(rec.get(durc))

                    rows.append(rec2)
        except Exception:
            pass

    # Guaranteed frozen reference values, explicitly labeled as such.
    # These are not fabricated; they are the frozen completed V4.15/V4.17 summary
    # constants used only when the master table could not be parsed.
    known = {
        "Adaptive 2/3 PRT0.008": {
            "RMSE": ADAPTIVE_RMSE_REFERENCE,
            "MeasuredScalars": ADAPTIVE_MEAN_SCALARS_REFERENCE,
            "Diagnoses": ADAPTIVE_MEAN_DIAGNOSES_REFERENCE,
        },
        "Fixed Event Opt-L3 XGB": {
            "RMSE": FIXED_L3_XGB_RMSE_REFERENCE,
            "MeasuredScalars": FIXED_L3_SCALARS,
            "Diagnoses": 3.0,
        },
        "Fixed Event Opt-L3 SafeMoE": {
            "RMSE": FIXED_L3_SAFE_RMSE_REFERENCE,
            "MeasuredScalars": FIXED_L3_SCALARS,
            "Diagnoses": 3.0,
        },
    }

    got = {r["OperatingPoint"] for r in rows}

    for label, vals in known.items():
        if label not in got:
            rows.append({
                "OperatingPoint": label,
                "Source": "Frozen completed-study reference",
                **vals,
            })

    out = pd.DataFrame(rows)

    # ensure one row each
    out = (
        out.sort_values(
            "Source",
            key=lambda s: s.ne("Frozen completed-study reference"),
            ascending=False,
        )
        .drop_duplicates("OperatingPoint")
        .reset_index(drop=True)
    )

    return out


def draw_adaptive_vs_fixed(points: pd.DataFrame, outpng: Path):
    d = points.copy()
    d["MeasuredScalars"] = pd.to_numeric(d["MeasuredScalars"], errors="coerce")
    d["RMSE"] = pd.to_numeric(d["RMSE"], errors="coerce")
    d = d.dropna(subset=["MeasuredScalars", "RMSE"])

    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    for r in d.itertuples(index=False):
        ax.scatter(r.MeasuredScalars, 100*r.RMSE, s=85)
        ax.text(
            r.MeasuredScalars + 0.25,
            100*r.RMSE,
            r.OperatingPoint,
            fontsize=8.5,
            va="center",
        )

    ax.set_xlabel("Measured scalars")
    ax.set_ylabel("Pooled RMSE (%)")
    ax.set_title("Adaptive primary deployment vs fixed-L3 accuracy references")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpng, dpi=300)
    plt.close(fig)


def run_adaptive(a):
    out = ensure_dir(a.output_root / "01_Adaptive_Primary")

    candidates = discover_adaptive_candidates(a)

    if candidates.empty:
        save_csv(
            pd.DataFrame([{
                "Status": "FAIL_TO_DISCOVER",
                "V415Root": str(a.v415_root),
                "Instruction": "Pass --adaptive-file with the V4.15 battery-level schedule CSV/JSON.",
            }]),
            out / "00_adaptive_discovery_audit.csv",
        )
        raise RuntimeError(
            "No V4.15 battery-level adaptive schedule was discovered. "
            "Pass --adaptive-file with the relevant CSV/JSON."
        )

    save_csv(candidates, out / "00_all_schedule_candidates.csv")

    schedules, source_audit = select_best_adaptive_set(candidates)
    save_csv(source_audit, out / "01_source_candidate_audit.csv")

    if schedules["Battery"].nunique() != EXPECTED_BATTERIES:
        save_csv(schedules, out / "02_adaptive_battery_schedules_PARTIAL.csv")
        raise RuntimeError(
            f"Adaptive schedule recovery found {schedules['Battery'].nunique()} "
            f"batteries, expected {EXPECTED_BATTERIES}. "
            "Inspect 01_source_candidate_audit.csv or pass --adaptive-file."
        )

    save_csv(schedules, out / "02_adaptive_battery_schedules.csv")

    edges = schedule_edges(schedules)
    save_csv(edges, out / "03_adaptive_sankey_edges.csv")

    dist = (
        schedules.groupby(["DiagnosisCount", "Schedule"])
        .size()
        .rename("Batteries")
        .reset_index()
        .sort_values(["DiagnosisCount", "Batteries"], ascending=[True, False])
    )
    save_csv(dist, out / "04_adaptive_schedule_distribution.csv")

    summary = adaptive_summary(schedules)
    save_csv(summary, out / "05_adaptive_summary.csv")

    draw_flow(edges, out / "Figure_Adaptive_Primary_Flow.png")

    points = extract_resource_operating_points(a)
    save_csv(points, out / "06_adaptive_vs_fixedL3_operating_points.csv")
    draw_adaptive_vs_fixed(
        points,
        out / "Figure_Adaptive_vs_FixedL3_ResourceAccuracy.png",
    )

    # Primary interpretation
    interp = pd.DataFrame([
        {
            "Method": "Adaptive 2/3 PRT0.008",
            "PaperRole": "PRIMARY DEPLOYMENT METHOD",
            "Reason": (
                "battery-specific online acquisition; about 19% fewer measured "
                "scalars than fixed L3 with only modest pooled accuracy loss"
            ),
        },
        {
            "Method": "Fixed Event Opt-L3 XGB",
            "PaperRole": "ACCURACY-CENTRIC FIXED-BUDGET CONTROL",
            "Reason": "same compact Event information design with all cells measured 3 times",
        },
        {
            "Method": "Fixed Event Opt-L3 SafeResidualMoE",
            "PaperRole": "MODEL-ENHANCED ACCURACY REFERENCE",
            "Reason": "lowest main Event pooled RMSE; model gain is not battery-level significant",
        },
    ])
    save_csv(interp, out / "07_primary_method_hierarchy.csv")

    save_json(
        {
            "Task": "V4.24 Adaptive Primary Recovery",
            "CodeVersion": CODE_VERSION,
            "Status": "PASS",
            "PrimaryPolicy": a.policy_threshold,
            "Batteries": int(schedules["Battery"].nunique()),
            "PrimaryMethod": "Adaptive 2/3 diagnosis PRT0.008",
            "FixedL3Role": "accuracy-centric fixed-budget control",
            "NewTraining": False,
            "NewTimingSearch": False,
        },
        out / "99_ADAPTIVE_PRIMARY_COMPLETE.json",
    )

    print("\nADAPTIVE PRIMARY COMPLETE")
    print(summary.to_string(index=False))
    print("\nTop schedules:")
    print(dist.head(20).to_string(index=False))
    print("\nOperating points:")
    print(points.to_string(index=False))


# =============================================================================
# Holm correction for V4.23 L-H timing tests
# =============================================================================

def find_lh_bootstrap_file(a) -> Path:
    if a.lh_bootstrap_file is not None:
        return a.lh_bootstrap_file

    p = a.v423_root / "02_L_H_TimingGain" / "04_TimingGain_L_H_paired_bootstrap.csv"
    if p.exists():
        return p

    raise FileNotFoundError(p)


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """
    Standard Holm step-down adjusted p-values.
    """
    p = np.asarray(pvals, float)
    m = len(p)

    order = np.argsort(p)
    sp = p[order]

    adj_sorted = np.empty(m, float)
    running = 0.0

    for i, pv in enumerate(sp):
        val = (m - i) * pv
        running = max(running, val)
        adj_sorted[i] = min(running, 1.0)

    adj = np.empty(m, float)
    adj[order] = adj_sorted
    return adj


def run_holm(a):
    out = ensure_dir(a.output_root / "02_LH_Holm_Correction")

    p = find_lh_bootstrap_file(a)
    d = pd.read_csv(p, low_memory=False)

    need = {
        "L", "HorizonBin", "ObservedMeanDifference",
        "CI95_Low", "CI95_High", "P_FirstBetter",
        "Batteries",
    }
    miss = need - set(d.columns)
    if miss:
        raise RuntimeError(f"L-H bootstrap table missing: {sorted(miss)}")

    pbetter = pd.to_numeric(d["P_FirstBetter"], errors="coerce").to_numpy(float)

    # The V4.23 bootstrap reports posterior/bootstrap probability that OPT has
    # smaller battery-level RMSE.  Use the opposite tail as a one-sided
    # bootstrap tail probability for OPT<Uniform.
    tail_one = 1.0 - pbetter
    tail_one = np.clip(tail_one, 0.0, 1.0)

    # finite-bootstrap floor: avoids exact zero when all bootstrap means agree
    reps = pd.to_numeric(d.get("BootstrapReps", 20000), errors="coerce").fillna(20000)
    floor = 1.0 / (reps.to_numpy(float) + 1.0)
    tail_one = np.maximum(tail_one, floor)

    # descriptive two-sided bootstrap tail
    tail_two = np.minimum(1.0, 2*np.minimum(pbetter, 1-pbetter))
    tail_two = np.maximum(tail_two, floor)

    d["BootstrapTailP_OneSided_OPTBetter"] = tail_one
    d["HolmAdjusted_OneSided"] = holm_adjust(tail_one)
    d["HolmSignificant_OneSided"] = (
        d["HolmAdjusted_OneSided"] <= a.alpha
    ).astype(int)

    d["BootstrapTailP_TwoSided"] = tail_two
    d["HolmAdjusted_TwoSided"] = holm_adjust(tail_two)
    d["HolmSignificant_TwoSided"] = (
        d["HolmAdjusted_TwoSided"] <= a.alpha
    ).astype(int)

    d["Direction"] = np.where(
        pd.to_numeric(d["ObservedMeanDifference"], errors="coerce") < 0,
        "OPT better",
        "Uniform better",
    )

    save_csv(d, out / "01_L_H_bootstrap_with_Holm.csv")

    summary = (
        d.groupby("L", as_index=False)
        .agg(
            Cells=("HorizonBin", "count"),
            RawCIExcludesZero=(
                "CI95_High",
                lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum()),
            ),
            HolmOneSidedSignificant=("HolmSignificant_OneSided", "sum"),
            HolmTwoSidedSignificant=("HolmSignificant_TwoSided", "sum"),
        )
    )
    save_csv(summary, out / "02_Holm_summary_by_L.csv")

    # Figure: one-sided Holm significance over L x H.
    horder = ["1-100", "101-300", "301-500", "501-700", "701-1000"]
    lorder = sorted(d["L"].unique())

    pv = (
        d.pivot(index="HorizonBin", columns="L", values="HolmAdjusted_OneSided")
        .reindex(horder)
        .reindex(columns=lorder)
    )
    sg = (
        d.pivot(index="HorizonBin", columns="L", values="HolmSignificant_OneSided")
        .reindex(horder)
        .reindex(columns=lorder)
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    im = ax.imshow(
        -np.log10(np.maximum(pv.to_numpy(float), 1e-6)),
        aspect="auto",
    )
    ax.set_xticks(range(len(lorder)))
    ax.set_xticklabels([f"L{x}" for x in lorder])
    ax.set_yticks(range(len(horder)))
    ax.set_yticklabels(horder)
    ax.set_xlabel("Diagnosis count")
    ax.set_ylabel("Forecast horizon")
    ax.set_title("Holm-adjusted bootstrap evidence for OPT < Uniform")
    fig.colorbar(im, ax=ax, label="-log10(adjusted bootstrap tail probability)")

    for i in range(len(horder)):
        for j in range(len(lorder)):
            val = pv.iloc[i, j]
            if np.isfinite(val):
                star = "*" if sg.iloc[i, j] == 1 else ""
                ax.text(j, i, f"{val:.3f}{star}", ha="center", va="center", fontsize=7.8)

    fig.tight_layout()
    fig.savefig(out / "Figure_L_H_Holm_Adjusted_Evidence.png", dpi=300)
    plt.close(fig)

    save_json(
        {
            "Task": "V4.24 L-H Holm correction",
            "CodeVersion": CODE_VERSION,
            "Status": "PASS",
            "Tests": int(len(d)),
            "Alpha": float(a.alpha),
            "OneSidedInterpretation": "OPT battery-level RMSE < Uniform",
            "ImportantBoundary": (
                "bootstrap tail probability is used for multiplicity-controlled "
                "evidence; the pre-specified overall Event L3 test remains the "
                "primary confirmatory inference"
            ),
        },
        out / "99_LH_HOLM_COMPLETE.json",
    )

    print("\nL-H HOLM COMPLETE")
    print(summary.to_string(index=False))


# =============================================================================
# Final manuscript hierarchy
# =============================================================================

def build_final_registry(a) -> pd.DataFrame:
    rows = [
        {
            "Order": 1,
            "Block": "Platform / information representation",
            "PrimaryMethod": "Event",
            "Control": "FixedTime",
            "PaperRole": "Methods + resource/accuracy foundation",
        },
        {
            "Order": 2,
            "Block": "Within-diagnosis information design",
            "PrimaryMethod": "Event nodes 0.05/0.09/0.13 + V_TimeWeightedMean + P_Last",
            "Control": "larger HI/node candidates",
            "PaperRole": "Methods + interpretability",
        },
        {
            "Order": 3,
            "Block": "Inter-diagnosis timing",
            "PrimaryMethod": "source-only optimized timing",
            "Control": "Uniform timing",
            "PaperRole": "CORE SCIENTIFIC CLAIM",
        },
        {
            "Order": 4,
            "Block": "Primary deployment",
            "PrimaryMethod": "Adaptive 2/3 diagnosis PRT0.008",
            "Control": "Fixed Event Opt-L3",
            "PaperRole": "PRIMARY DEPLOYMENT METHOD",
        },
        {
            "Order": 5,
            "Block": "Fixed-budget accuracy reference",
            "PrimaryMethod": "Fixed Event Opt-L3",
            "Control": "Uniform L3 / other L",
            "PaperRole": "ACCURACY-CENTRIC CONTROL / PARETO REFERENCE",
        },
        {
            "Order": 6,
            "Block": "Model extension",
            "PrimaryMethod": "SafeResidualMoE",
            "Control": "FrozenXGB",
            "PaperRole": "Supporting model enhancement; do not claim significant universal superiority",
        },
        {
            "Order": 7,
            "Block": "Forecast horizon",
            "PrimaryMethod": "L-H / TimingGain-H maps",
            "Control": "matched Uniform timing",
            "PaperRole": "Shows L3 global Pareto role and L5/L6 far-horizon benefit",
        },
        {
            "Order": 8,
            "Block": "Endpoint",
            "PrimaryMethod": "trajectory-derived RUL at SOH=0.85 sensitivity endpoint",
            "Control": "XGB/Uniform counterparts",
            "PaperRole": "Timing benefit transfers from SOH trajectory to RUL",
        },
        {
            "Order": 9,
            "Block": "Deployment robustness",
            "PrimaryMethod": "timing jitter + calibration consistency",
            "Control": "Clean / diagnosis-to-diagnosis drift",
            "PaperRole": "Engineering robustness",
        },
        {
            "Order": 10,
            "Block": "Interpretability",
            "PrimaryMethod": "SHAP + HI physics",
            "Control": "HI ranking / compact HI ablation",
            "PaperRole": "Supporting interpretation",
        },
    ]

    return pd.DataFrame(rows)


def final_claims() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ClaimID": "C1",
            "Claim": "At fixed diagnostic budget, diagnosis timing is an independent performance-design variable.",
            "Status": "PRIMARY",
        },
        {
            "ClaimID": "C2",
            "Claim": "Adaptive 2/3 diagnosis is the primary deployment strategy because it reduces average measurement burden while retaining near-fixed-L3 accuracy.",
            "Status": "PRIMARY DEPLOYMENT",
        },
        {
            "ClaimID": "C3",
            "Claim": "Fixed Event Opt-L3 is the accuracy-centric compact-budget reference and overall resource-accuracy Pareto knee, not the final deployment policy.",
            "Status": "REFERENCE",
        },
        {
            "ClaimID": "C4",
            "Claim": "The useful diagnostic budget depends on forecast horizon: L3 is efficient through most horizons, while L5/L6 can benefit the farthest horizon.",
            "Status": "MAIN SUPPORTING",
        },
        {
            "ClaimID": "C5",
            "Claim": "Timing optimization improves the 0.85-SOH derived-RUL endpoint relative to Uniform timing.",
            "Status": "MAIN SUPPORTING",
        },
        {
            "ClaimID": "C6",
            "Claim": "Stable persistent gain bias is substantially less damaging than diagnosis-to-diagnosis gain drift.",
            "Status": "ENGINEERING",
        },
        {
            "ClaimID": "C7",
            "Claim": "SafeResidualMoE provides modest pooled improvement but does not establish battery-level universal superiority over FrozenXGB.",
            "Status": "LIMITED MODEL CLAIM",
        },
        {
            "ClaimID": "C8",
            "Claim": "Sparse SOH Anchor is not part of the RWTH48 study and must not appear as an RWTH48 experimental contribution.",
            "Status": "EXCLUSION",
        },
    ])


def run_finalize(a):
    out = ensure_dir(a.output_root / "03_Final_Manuscript_Hierarchy")

    reg = build_final_registry(a)
    claims = final_claims()

    save_csv(reg, out / "01_final_experiment_hierarchy.csv")
    save_csv(claims, out / "02_final_claim_registry.csv")

    adaptive_ok = (
        a.output_root / "01_Adaptive_Primary" / "99_ADAPTIVE_PRIMARY_COMPLETE.json"
    ).exists()
    holm_ok = (
        a.output_root / "02_LH_Holm_Correction" / "99_LH_HOLM_COMPLETE.json"
    ).exists()

    acceptance = pd.DataFrame([
        {
            "Check": "Adaptive primary recovered",
            "Observed": "PASS" if adaptive_ok else "MISSING",
            "Expected": "PASS",
            "PASS": int(adaptive_ok),
        },
        {
            "Check": "L-H Holm correction",
            "Observed": "PASS" if holm_ok else "MISSING",
            "Expected": "PASS",
            "PASS": int(holm_ok),
        },
        {
            "Check": "Sparse Anchor excluded from RWTH hierarchy",
            "Observed": int(not reg.astype(str).apply(lambda s: s.str.contains("Sparse", case=False)).any().any()),
            "Expected": 1,
            "PASS": int(not reg.astype(str).apply(lambda s: s.str.contains("Sparse", case=False)).any().any()),
        },
    ])
    save_csv(acceptance, out / "03_acceptance_gate.csv")

    status = "PASS" if acceptance["PASS"].eq(1).all() else "CHECK"

    save_json(
        {
            "Task": "RWTH48 V4.24 Adaptive-Primary Finalization",
            "CodeVersion": CODE_VERSION,
            "Status": status,
            "PrimaryDeployment": "Adaptive 2/3 diagnosis PRT0.008",
            "FixedL3Role": "accuracy-centric fixed-budget reference",
            "SparseSOHAnchorInRWTH": False,
            "NoNewModelTraining": True,
            "NoNewTimingSearch": True,
            "Acceptance": acceptance.to_dict("records"),
        },
        a.output_root / "99_RWTH48_V4_24_COMPLETE.json",
    )

    print("\nFINAL MANUSCRIPT HIERARCHY:", status)
    print(reg.to_string(index=False))
    print("\nAcceptance:")
    print(acceptance.to_string(index=False))


# =============================================================================
# Audit
# =============================================================================

def run_audit(a):
    rows = [
        {
            "Check": "V4.15 adaptive root",
            "Observed": str(a.v415_root),
            "Expected": "exists",
            "PASS": int(a.v415_root.exists()),
        },
        {
            "Check": "V4.23 L-H bootstrap",
            "Observed": str(find_lh_bootstrap_file(a)) if (
                a.lh_bootstrap_file is not None
                or (a.v423_root / "02_L_H_TimingGain" / "04_TimingGain_L_H_paired_bootstrap.csv").exists()
            ) else "missing",
            "Expected": "exists",
            "PASS": int(
                (
                    a.lh_bootstrap_file is not None
                    and a.lh_bootstrap_file.exists()
                )
                or (
                    a.v423_root
                    / "02_L_H_TimingGain"
                    / "04_TimingGain_L_H_paired_bootstrap.csv"
                ).exists()
            ),
        },
        {
            "Check": "Sparse Anchor intentionally excluded",
            "Observed": "RWTH has no Sparse Anchor branch",
            "Expected": "excluded",
            "PASS": 1,
        },
    ]

    # dry discovery count
    if a.v415_root.exists():
        c = discover_adaptive_candidates(a)
        n = c["Battery"].nunique() if not c.empty else 0
    else:
        n = 0

    rows.append({
        "Check": "Adaptive candidate batteries discovered",
        "Observed": int(n),
        "Expected": ">=47 candidate battery IDs preferred",
        "PASS": int(n >= 47),
    })

    df = pd.DataFrame(rows)
    save_csv(df, a.output_root / "00_preflight.csv")
    print(df.to_string(index=False))

    return df


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()
    ensure_dir(a.output_root)

    if a.stage == "audit":
        run_audit(a)
        return

    if a.stage == "adaptive":
        run_adaptive(a)
        return

    if a.stage == "holm":
        run_holm(a)
        return

    if a.stage == "finalize":
        run_finalize(a)
        return

    if a.stage == "all":
        run_audit(a)
        run_adaptive(a)
        run_holm(a)
        run_finalize(a)
        return


if __name__ == "__main__":
    main()

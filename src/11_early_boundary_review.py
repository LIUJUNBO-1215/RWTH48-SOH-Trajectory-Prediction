# -*- coding: utf-8 -*-
r"""
RWTH48 V4.14C | Full-Life SOH P0 Closeout | FIX1
=================================================

Closes every P0 item defined after V4.14B:
  P0-A  L=1 single-diagnosis boundary + all-cycle source-only timing scan
  P0-B  Diagnostic staleness (last diagnosis earlier than cycle 80)
  P0-C  Event optimized-3-diagnosis champion-consistent validation
        * exact clean reproduction
        * nested source-only cross-fitted conformal trajectory intervals
        * XGBoost TreeSHAP (native pred_contribs)
        * selected-HI +/-5/10% perturbation
        * waveform-level V/I measurement-noise robustness after frozen causal QC
  P0-D  Long-horizon / horizon-bin metrics
  P0-E  Fixed global/consensus schedule vs source-optimized schedule
        * post-hoc consensus descriptive result
        * dev32 -> confirm15 schedule-confirmation result
  P0-F  Empirical measurement-cost/RMSE Pareto + threshold sensitivity
        across all formally validated V4.14B + V4.14C configurations.

Scientific boundary
-------------------
The primary target remains full-life SOH trajectory prediction from early-life
physical measurements. RUL is not optimized in this stage.

The Event Opt-L3 champion is the V4.14B source-only BEST_SOURCE_PROXY policy:
  * 3 diagnosis cycles (last one fixed at origin 80 in the primary experiment)
  * 3 RefAh checkpoints: 0.05 / 0.09 / 0.13
  * two frozen SOH HIs: V_TimeWeightedMean / P_Last
  * 18 measured HI scalars
  * common full-life XGBoost trajectory head

No held-out battery is used for schedule selection, model training, conformal
calibration, or HI selection. The dev32 fixed-schedule confirmation uses a
predeclared deterministic development/confirmation split and reports the
15-battery confirmation subset separately.

Waveform robustness wording
---------------------------
Noise is injected into the frozen causally-cleaned V4 waveform BEFORE Event
checkpoint localization and HI extraction. It is NOT pre-QC raw-file noise.
Noise magnitudes are stress-test assumptions, not claimed RWTH instrument specs.
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
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CODE_VERSION = "V414C_SOH_P0_CLOSEOUT_FIX1"
BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR = Path(__file__).resolve().parent
V414B_NAME = "10_schedule_and_resource_search.py"
COMMON_NAME = "08_fixedtime_common_pipeline.py"
OUTPUT_NAME = "RWTH48_V4_14C_SOH_P0_Closeout"
STATE_NAME = "RWTH48_V4_14C_SOH_P0_Closeout_STATE"
OLD_V414B_ROOT_NAME = "RWTH48_V4_14B_FullLifeSOH_Timing_MinResource"
EXPECTED_BATTERIES = 47
ORIGIN = 80
EVENT_HIS = ("V_TimeWeightedMean", "P_Last")
EVENT_NODES = (0.05, 0.09, 0.13)
ENC = "utf-8-sig"
FAST_GZIP = {"method": "gzip", "compresslevel": 1}
COVERAGES = (0.80, 0.90, 0.95)
PERTURB_PCTS = (-10.0, -5.0, 5.0, 10.0)
LEAD_BINS = (
    (1, 100, "Lead001_100"),
    (101, 300, "Lead101_300"),
    (301, 500, "Lead301_500"),
    (501, 700, "Lead501_700"),
    (701, 1000, "Lead701_1000"),
)
LEAD_SHAP_POINTS = (10, 50, 100, 200, 300, 500, 700, 900)
LEAD_FEATURE_NAMES = ("LEAD__u", "LEAD__u2", "LEAD__sqrtu", "LEAD__logu", "LEAD__sinu", "LEAD__cosu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RWTH48 V4.14C full-life SOH P0 closeout")
    p.add_argument("--stage", choices=["audit", "core", "champion", "waveform", "all", "finalize"], default="all")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--code-dir", type=Path, default=CODE_DIR)
    p.add_argument("--v414b", type=Path, default=None)
    p.add_argument("--common", type=Path, default=None)
    p.add_argument("--old-v414b-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--state-root", type=Path, default=None)
    p.add_argument("--event-ready", type=Path, default=None)
    p.add_argument("--fixedtime-ready", type=Path, default=None)
    p.add_argument("--dependency-registry", type=Path, default=None)

    # Core timing/resources.
    p.add_argument("--cycle-start", type=int, default=40)
    p.add_argument("--cycle-end", type=int, default=80)
    p.add_argument("--cycle-step", type=int, default=4)
    p.add_argument("--staleness-values", nargs="+", type=int, default=[4, 8, 12, 16, 24])
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.0070,0.00725,0.0075,0.00775,0.0080,0.00825,0.0085,0.00875,0.0090,0.00925,0.0095,0.00975,0.0100])
    p.add_argument("--dev-batteries", type=int, default=32)
    p.add_argument("--dev-split-seed", type=int, default=20260826)

    # Source-only schedule/resource proxy knobs (matched to V4.14B).
    p.add_argument("--proxy-validation-batteries", type=int, default=9)
    p.add_argument("--proxy-ridge-alpha", type=float, default=8.0)
    p.add_argument("--proxy-leads", nargs="+", type=int, default=[10,25,50,100,200,300,500,700,900])
    p.add_argument("--hi-ranking-shortlist", type=int, default=12)
    p.add_argument("--l1-ridge-preselect", type=int, default=4)
    p.add_argument("--local-refine-radius", type=int, default=3)
    p.add_argument("--local-refine-rounds", type=int, default=2)

    # Formal trajectory XGB.
    p.add_argument("--xgb-threads", type=int, default=3)
    p.add_argument("--cpu-threads-per-worker", type=int, default=4)
    p.add_argument("--xgb-leads-per-origin", type=int, default=160)

    # UQ / robustness.
    p.add_argument("--calibration-folds", type=int, default=5)
    p.add_argument("--coverages", nargs="+", type=float, default=list(COVERAGES))
    p.add_argument("--perturb-pcts", nargs="+", type=float, default=list(PERTURB_PCTS))
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--voltage-sigmas-mv", nargs=3, type=float, default=[2.0,5.0,10.0])
    p.add_argument("--current-sigmas-ma", nargs=3, type=float, default=[5.0,20.0,50.0])
    p.add_argument("--waveform-clean-tol", type=float, default=2e-5)
    p.add_argument("--clean-pred-tol", type=float, default=5e-7)
    p.add_argument("--v4-root", type=Path, default=None)
    p.add_argument("--v4-script", type=Path, default=None)

    # Execution.
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--skip-waveform", action="store_true")
    p.add_argument("--parallel-workers", type=int, default=1)
    p.add_argument("--parallel-worker-id", type=int, default=0)
    p.add_argument("--run-id", type=str, default="V414C_P0_20260826")
    p.add_argument("--barrier-poll-s", type=float, default=10.0)

    a = p.parse_args()
    a.base = Path(a.base); a.code_dir = Path(a.code_dir)
    a.v414b = Path(a.v414b) if a.v414b else a.code_dir / V414B_NAME
    a.common = Path(a.common) if a.common else a.code_dir / COMMON_NAME
    a.old_v414b_root = Path(a.old_v414b_root) if a.old_v414b_root else a.base / OLD_V414B_ROOT_NAME
    a.output_root = Path(a.output_root) if a.output_root else a.base / OUTPUT_NAME
    a.state_root = Path(a.state_root) if a.state_root else a.base / STATE_NAME
    if a.cycle_end != ORIGIN:
        p.error("--cycle-end must remain 80 in this closeout")
    if not (1 <= a.dev_batteries < EXPECTED_BATTERIES):
        p.error("--dev-batteries must be 1..46")
    if a.parallel_workers < 1 or not (0 <= a.parallel_worker_id < a.parallel_workers):
        p.error("invalid parallel worker configuration")
    if a.calibration_folds < 2:
        p.error("--calibration-folds must be >=2")
    if a.replicates < 1:
        p.error("--replicates must be >=1")
    a.thresholds = sorted(set(float(x) for x in a.thresholds if 0 < float(x) < 0.1))
    a.staleness_values = sorted(set(int(x) for x in a.staleness_values if int(x) > 0))
    a.coverages = sorted(set(float(x) for x in a.coverages if 0 < float(x) < 1))
    return a


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p


def save_csv(df: pd.DataFrame, p: Path, compression=None):
    ensure_dir(p.parent)
    kw = dict(index=False, encoding=ENC)
    if compression is not None: kw["compression"] = compression
    df.to_csv(p, **kw)


def save_json(obj, p: Path):
    ensure_dir(p.parent)
    def conv(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, set): return sorted(x)
        raise TypeError(type(x))
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv), encoding="utf-8")


def load_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def load_module(path: Path, name: str):
    if not Path(path).exists(): raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def stable_seed(*items) -> int:
    h = hashlib.sha1("|".join(map(str, items)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def norm_battery(x: object) -> str:
    s = str(x).strip().upper().replace("_", "-").replace(" ", "")
    m = re.search(r"RWTH-?0*(\d+)", s)
    return f"RWTH-{int(m.group(1)):03d}" if m else s


def make_b_args(args, output_root: Optional[Path] = None):
    # Namespace with every attribute used by V4.14B preflight/helpers.
    return argparse.Namespace(
        stage="audit", base=args.base, code_dir=args.code_dir, common=args.common,
        event_ready=args.event_ready, fixedtime_ready=args.fixedtime_ready,
        dependency_registry=args.dependency_registry,
        output_root=Path(output_root or args.output_root), state_root=args.state_root,
        cycle_start=args.cycle_start, cycle_end=args.cycle_end, cycle_step=args.cycle_step,
        diagnosis_counts=[2,3,4,5,6], timing_preselect=24, timing_bottom_preselect=6,
        local_refine_radius=args.local_refine_radius, local_refine_rounds=args.local_refine_rounds,
        proxy_validation_batteries=args.proxy_validation_batteries,
        proxy_ridge_alpha=args.proxy_ridge_alpha, proxy_leads=args.proxy_leads,
        include_worst_timing=False, xgb_threads=args.xgb_threads,
        cpu_threads_per_worker=args.cpu_threads_per_worker,
        xgb_leads_per_origin=args.xgb_leads_per_origin,
        rmse_threshold=0.01, proxy_feasible_threshold=0.0105,
        resource_topks=[1,2,3], resource_max_cells_event=90, resource_max_cells_fixedtime=180,
        resource_node_preselect=5, resource_schedule_preselect=5,
        hi_ranking_shortlist=args.hi_ranking_shortlist, hi_refine_shortlist=6,
        resource_cost_multiplier=2.0, fixed_node_counts=[1,2,3,4,6,8],
        event_modalities=["VI","VIT"], fixedtime_modalities=["V","VI","VT","VIT"],
        staleness_values=[0]+args.staleness_values, bootstrap=args.bootstrap,
        only_battery="", force=args.force, skip_figures=True, dpi=200,
        parallel_workers=1, parallel_worker_id=0, run_id=args.run_id, barrier_poll_s=args.barrier_poll_s,
    )


def old_timing_json(args, held: str, platform: str, L: int, role: str) -> Path:
    return args.old_v414b_root / "folds" / held / "timing" / f"{platform}_L{L}_LAST80_{role}.json"


def old_timing_npz(args, held: str, platform: str, L: int, role: str) -> Path:
    return args.old_v414b_root / "folds" / held / "timing" / f"{platform}_L{L}_LAST80_{role}.npz"


def load_old_schedule(args, held: str, platform="Event", L=3, role="BEST_SOURCE_PROXY") -> Tuple[int, ...]:
    p = old_timing_json(args, held, platform, L, role)
    if not p.exists(): raise FileNotFoundError(p)
    return tuple(map(int, load_json(p)["Schedule"]))


def preflight(args):
    if not args.v414b.exists(): raise FileNotFoundError(args.v414b)
    if not args.common.exists(): raise FileNotFoundError(args.common)
    if not args.old_v414b_root.exists(): raise FileNotFoundError(args.old_v414b_root)
    comp = args.old_v414b_root / "99_RWTH48_V4_14B_COMPLETE.json"
    if not comp.exists(): raise FileNotFoundError(comp)
    c = load_json(comp)
    if c.get("Status") != "PASS": raise RuntimeError("V4.14B status != PASS")
    B = load_module(args.v414b, "rwth48_v414b_for_v414c")
    bargs = make_b_args(args)
    ctx = B.preflight(bargs)
    bats = list(ctx["batteries"])
    missing = []
    for b in bats:
        for q in [old_timing_json(args,b,"Event",3,"BEST_SOURCE_PROXY"), old_timing_npz(args,b,"Event",3,"BEST_SOURCE_PROXY")]:
            if not q.exists(): missing.append(str(q))
    if missing: raise FileNotFoundError("missing V4.14B Opt-L3 caches:\n" + "\n".join(missing[:20]))

    # deterministic dev32/confirm15 split; saved and reused.
    order = sorted(bats, key=lambda b: hashlib.sha1(f"{args.dev_split_seed}|{b}".encode()).hexdigest())
    dev = sorted(order[:args.dev_batteries]); confirm = sorted(order[args.dev_batteries:])
    pre_rows=[
        {"Check":"V4.14B PASS","Observed":c.get("Status"),"Expected":"PASS","PASS":1},
        {"Check":"47 batteries","Observed":len(bats),"Expected":47,"PASS":int(len(bats)==47)},
        {"Check":"Opt-L3 caches","Observed":len(missing),"Expected":0,"PASS":int(not missing)},
        {"Check":"Dev/confirm split","Observed":f"{len(dev)}/{len(confirm)}","Expected":f"{args.dev_batteries}/{47-args.dev_batteries}","PASS":int(len(dev)+len(confirm)==47)},
        {"Check":"Event SOH HIs","Observed":"|".join(EVENT_HIS),"Expected":"present","PASS":int(all(h in ctx['event_cube'].hmap for h in EVENT_HIS))},
    ]
    if not args.skip_waveform:
        try:
            v4root=resolve_v4_root(args); v4script=resolve_v4_script(args); refmap=load_ref_capacity_map(v4root)
            dirs=sum(find_battery_dir(v4root,b) is not None for b in bats)
            refs=sum(b in refmap and np.isfinite(refmap[b]) and refmap[b]>0 for b in bats)
            pre_rows += [
                {"Check":"V4 waveform root","Observed":str(v4root),"Expected":"exists","PASS":int(v4root.exists())},
                {"Check":"Exact V4 Event script","Observed":str(v4script),"Expected":"exists","PASS":int(v4script.exists())},
                {"Check":"Waveform battery dirs","Observed":dirs,"Expected":47,"PASS":int(dirs==47)},
                {"Check":"Reference capacities","Observed":refs,"Expected":47,"PASS":int(refs==47)},
            ]
        except Exception as e:
            pre_rows.append({"Check":"Waveform prerequisites","Observed":repr(e),"Expected":"available","PASS":0})
    pre = pd.DataFrame(pre_rows)
    if not pre.PASS.eq(1).all(): raise RuntimeError(pre[pre.PASS.eq(0)].to_string(index=False))
    ctx.update({"B":B,"bargs":bargs,"dev_batteries":dev,"confirm_batteries":confirm,"v414b_complete":c})
    return ctx, pre


# =============================================================================
# P0-A: L=1 one-scalar boundary
# =============================================================================

def source_proxy_for_config(ctx, held, cube, cycle, nodes, his, args, salt):
    B=ctx["B"]; source=B.source_batteries(ctx, held)
    x=B.extract_tensor(cube, [cycle], nodes, his)
    if x is None: return np.inf
    X=B.compact_features(x, [cycle])
    return B.ridge_proxy_rmse(X, ctx["batteries"], source, ctx["Y80"], ctx["M80"], args,
                              held, salt)


def l1_scan(ctx, held: str, platform: str, args) -> pd.DataFrame:
    p=ensure_dir(args.output_root/"folds"/held/"l1")/f"{platform}_source_scan.csv.gz"
    if p.exists() and not args.force: return pd.read_csv(p, compression="infer", low_memory=False)
    B=ctx["B"]; cube=ctx["event_cube"] if platform=="Event" else ctx["fixed_cube"]
    # Broad VIT ranking gives the largest legal HI canvas. Sensor cost is not hidden:
    # acquisition_cost() later reports the actual physical channels required by each HI
    # (and Event always adds I for RefAh localization).
    ranking=B.general_hi_ranking(ctx,held,cube,"VIT",ctx["bargs"]).head(args.hi_ranking_shortlist)
    his=ranking.HI.astype(str).tolist()
    source=B.source_batteries(ctx,held); bmap={b:i for i,b in enumerate(ctx["batteries"])}
    src_idx=[bmap[b] for b in source]
    rows=[]
    for cyc in range(args.cycle_start, ORIGIN+1):
        candidates=[]
        for h in his:
            for node in cube.nodes:
                x=B.extract_tensor(cube,[cyc],[node],[h])
                if x is None: continue
                Xc=B.compact_features(x,[cyc])
                info=B.information_score(Xc,ctx["target_summary"],src_idx)
                cost=B.acquisition_cost(cube,[cyc],[node],[h],ctx["depmap"],battery_indices=src_idx)
                candidates.append((float(info),float(node),h,cost))
        if not candidates: continue
        candidates.sort(key=lambda z:(-z[0],z[1],z[2]))
        # Ridge is the expensive source-only proxy. Refine only the strongest
        # information candidates at every cycle; this keeps the all-cycle L=1 scan bounded.
        for rank,(info,node,h,cost) in enumerate(candidates[:max(1,int(args.l1_ridge_preselect))],1):
            prmse=source_proxy_for_config(ctx,held,cube,cyc,[node],[h],ctx["bargs"],
                    f"L1|{platform}|C{cyc}|N{node}|{h}")
            rows.append({"OuterHeldoutBattery":held,"Platform":platform,"Cycle":cyc,"Node":node,"HI":h,
                         "InformationRankAtCycle":rank,"InformationScore":info,"ProxyRMSE":prmse,**cost})
    d=pd.DataFrame(rows)
    if d.empty: raise RuntimeError(f"{held}/{platform}: L1 scan empty")
    save_csv(d,p,FAST_GZIP); return d


def choose_l1_roles(scan: pd.DataFrame) -> Dict[str, pd.Series]:
    # Primary boundary = best source proxy among true one-scalar configs.
    def best(q):
        if q.empty: return None
        return q.sort_values(["ProxyRMSE","PhysicalSensorCount","AcquisitionDuration_s","Cycle","Node","HI"]).iloc[0]
    out={"BEST_SOURCE_PROXY":best(scan),"EARLY40":best(scan[scan.Cycle.eq(40)]),
         "MID60":best(scan[scan.Cycle.eq(60)]),"LATEST80":best(scan[scan.Cycle.eq(80)])}
    return {k:v for k,v in out.items() if v is not None}


def run_l1_formal(ctx, held, platform, args):
    B=ctx["B"]; cube=ctx["event_cube"] if platform=="Event" else ctx["fixed_cube"]
    scan=l1_scan(ctx,held,platform,args)
    roles=choose_l1_roles(scan)
    fdir=ensure_dir(args.output_root/"folds"/held/"l1")
    for role,r in roles.items():
        jp=fdir/f"{platform}_{role}.json"; npz=fdir/f"{platform}_{role}.npz"
        if jp.exists() and npz.exists() and not args.force: continue
        formal=B.formal_outer_prediction(ctx,held,cube,[int(r.Cycle)],[float(r.Node)],[str(r.HI)],ctx["bargs"])
        heldi=ctx["batteries"].index(held)
        cost=B.acquisition_cost(cube,[int(r.Cycle)],[float(r.Node)],[str(r.HI)],ctx["depmap"],battery_indices=[heldi])
        q={"CodeVersion":CODE_VERSION,"OuterHeldoutBattery":held,"Platform":platform,"Role":role,
           "DiagnosisCount":1,"Schedule":[int(r.Cycle)],"Nodes":[float(r.Node)],"SelectedHIs":[str(r.HI)],
           "SourceProxyRMSE":float(r.ProxyRMSE),**cost,**{f"Fold_{k}":v for k,v in formal["Metrics"].items()},
           "SSE":formal["SSE"],"SAE":formal["SAE"],"N":formal["N"],"FeatureCount":formal["FeatureCount"],
           "HeldoutUsedForTimingSelection":0,"HeldoutUsedForHISelection":0,"HeldoutUsedForTraining":0}
        np.savez_compressed(npz,pred=formal["Pred"],true=formal["True"],mask=formal["Mask"]); save_json(q,jp)


# =============================================================================
# P0-B: staleness using 3 diagnoses, full nodes and frozen SOH pairs
# =============================================================================

def staleness_args(ctx,args):
    b=make_b_args(args, args.output_root/"_staleness_engine")
    b.diagnosis_counts=[3]; b.force=args.force; return b


def run_staleness(ctx, held, args):
    B=ctx["B"]; ba=staleness_args(ctx,args); out=ensure_dir(args.output_root/"folds"/held/"staleness")
    # Only the source-optimized policy is formally refit at every stale boundary.
    # Uniform-vs-optimized timing was already established in V4.14B; duplicating it
    # at every staleness level would double compute without answering the P0 question.
    for stale in args.staleness_values:
        last=ORIGIN-int(stale)
        if last <= args.cycle_start or len(B.coarse_cycles(ba,last))<3: continue
        for platform in ("Event","FixedTime"):
            jp=out/f"{platform}_STALE{stale}.json"; npz=out/f"{platform}_STALE{stale}.npz"
            if jp.exists() and npz.exists() and not args.force: continue
            search=B.timing_schedule_search(ctx,held,platform,3,last,ba)
            roles=B.choose_timing_roles(search,ba); schedule=roles["BEST_SOURCE_PROXY"]
            cube=ctx["event_cube"] if platform=="Event" else ctx["fixed_cube"]
            his=B.timing_pair(ctx,held,platform); nodes=cube.nodes
            formal=B.formal_outer_prediction(ctx,held,cube,schedule,nodes,his,ba)
            cost=B.acquisition_cost(cube,schedule,nodes,his,ctx["depmap"],battery_indices=[ctx["batteries"].index(held)])
            q={"CodeVersion":CODE_VERSION,"OuterHeldoutBattery":held,"Platform":platform,"Role":"BEST_SOURCE_PROXY",
               "Staleness":int(stale),"LastCycle":int(last),"Schedule":list(map(int,schedule)),"SelectedHIs":list(his),
               "Nodes":list(map(float,nodes)),**cost,**{f"Fold_{k}":v for k,v in formal["Metrics"].items()},
               "SSE":formal["SSE"],"SAE":formal["SAE"],"N":formal["N"],
               "HeldoutUsedForTimingSelection":0,"HeldoutUsedForTraining":0}
            np.savez_compressed(npz,pred=formal["Pred"],true=formal["True"],mask=formal["Mask"]); save_json(q,jp)


# =============================================================================
# P0-E: fixed schedule confirmation
# =============================================================================

def ridge_schedule_on_source(ctx, source_bats: Sequence[str], schedule: Sequence[int], args, salt: str) -> float:
    B=ctx["B"]; cube=ctx["event_cube"]
    x=B.extract_tensor(cube,schedule,cube.nodes,EVENT_HIS)
    if x is None: return np.inf
    X=B.compact_features(x,schedule)
    return B.ridge_proxy_rmse(X,ctx["batteries"],list(source_bats),ctx["Y80"],ctx["M80"],ctx["bargs"],
                              "DEV32",salt+"|"+"-".join(map(str,schedule)))


def select_dev_consensus_schedule(ctx,args) -> Tuple[Tuple[int,...], pd.DataFrame]:
    if "_dev_consensus_schedule" in ctx and "_dev_consensus_search" in ctx:
        return ctx["_dev_consensus_schedule"], ctx["_dev_consensus_search"]
    p=args.output_root/"07_event_dev32_schedule_search.csv"
    if p.exists() and not args.force:
        d=pd.read_csv(p); best=tuple(map(int,str(d.sort_values('ProxyRMSE').iloc[0].Schedule).split('|')))
        ctx["_dev_consensus_schedule"],ctx["_dev_consensus_search"]=best,d
        return best,d
    B=ctx["B"]; cycles=B.coarse_cycles(ctx["bargs"]); schedules=B.enumerate_schedules(cycles,3,ORIGIN)
    rows=[]
    for sch in schedules:
        v=ridge_schedule_on_source(ctx,ctx["dev_batteries"],sch,args,"DEV32_COARSE")
        rows.append({"Schedule":"|".join(map(str,sch)),"ProxyRMSE":v,"Stage":"Coarse"})
    d=pd.DataFrame(rows).sort_values(["ProxyRMSE","Schedule"]).reset_index(drop=True)
    cur=tuple(map(int,d.iloc[0].Schedule.split('|'))); best=float(d.iloc[0].ProxyRMSE)
    for rr in range(1,args.local_refine_rounds+1):
        neigh=B.local_schedule_neighbors(cur,args.cycle_start,ORIGIN,args.local_refine_radius)
        vals=[(best,cur)]
        for sch in neigh:
            vals.append((ridge_schedule_on_source(ctx,ctx["dev_batteries"],sch,args,f"DEV32_R{rr}"),sch))
        val,sch=min(vals,key=lambda z:(z[0],z[1]))
        d=pd.concat([d,pd.DataFrame([{"Schedule":"|".join(map(str,sch)),"ProxyRMSE":val,"Stage":f"LocalR{rr}"}])],ignore_index=True)
        if sch==cur or val>=best-1e-12: break
        cur,best=sch,float(val)
    ctx["_dev_consensus_schedule"],ctx["_dev_consensus_search"]=cur,d
    # Only worker0 writes the shared audit file; other workers use their identical in-memory result.
    if int(getattr(args,"parallel_worker_id",0))==0:
        save_csv(d,p)
    return cur,d


def posthoc_consensus_schedule(args, platform="Event", L=3) -> Tuple[int,...]:
    p=args.old_v414b_root/"07_best_schedule_frequency.csv"
    d=pd.read_csv(p,low_memory=False)
    q=d[d.Platform.eq(platform)&d.DiagnosisCount.eq(L)].sort_values(["OuterFoldCount","Schedule"],ascending=[False,True])
    if q.empty: raise RuntimeError("consensus frequency rows missing")
    return tuple(map(int,str(q.iloc[0].Schedule).split('|')))


def run_fixed_schedule_eval(ctx, held, args):
    B=ctx["B"]; cube=ctx["event_cube"]; fdir=ensure_dir(args.output_root/"folds"/held/"consensus")
    dev_schedule,_=select_dev_consensus_schedule(ctx,args)
    post=posthoc_consensus_schedule(args,"Event",3)
    roles=[("POSTHOC_CONSENSUS",post,True),("DEV32_FIXED",dev_schedule,False)]
    for role,s,posthoc in roles:
        # DEV32_FIXED is confirmatory only on the 15 confirmation batteries.
        if role=="DEV32_FIXED" and held not in ctx["confirm_batteries"]: continue
        jp=fdir/f"{role}.json"; npz=fdir/f"{role}.npz"
        if jp.exists() and npz.exists() and not args.force: continue
        formal=B.formal_outer_prediction(ctx,held,cube,s,cube.nodes,EVENT_HIS,ctx["bargs"])
        cost=B.acquisition_cost(cube,s,cube.nodes,EVENT_HIS,ctx["depmap"],battery_indices=[ctx["batteries"].index(held)])
        q={"CodeVersion":CODE_VERSION,"OuterHeldoutBattery":held,"Role":role,"Schedule":list(s),"PostHoc":int(posthoc),
           "ConfirmationSubsetOnly":int(role=="DEV32_FIXED"),**cost,**{f"Fold_{k}":v for k,v in formal['Metrics'].items()},
           "SSE":formal["SSE"],"SAE":formal["SAE"],"N":formal["N"],"HeldoutUsedForScheduleSelection":0 if role=="DEV32_FIXED" else "POSTHOC_DESCRIPTIVE"}
        np.savez_compressed(npz,pred=formal["Pred"],true=formal["True"],mask=formal["Mask"]); save_json(q,jp)


# =============================================================================
# Champion model helpers
# =============================================================================

def champion_data(ctx, held, args):
    B=ctx["B"]; schedule=load_old_schedule(args,held,"Event",3,"BEST_SOURCE_PROXY")
    cube=ctx["event_cube"]; x=B.extract_tensor(cube,schedule,cube.nodes,EVENT_HIS)
    if x is None: raise RuntimeError(f"{held}: champion tensor incomplete")
    X47,names,reg=B.engineer_dynamic_selected(x,schedule,cube.nodes,EVENT_HIS)
    X141=B.embed_features_141(ctx,X47); tr=B.source_indices_origin80(ctx,held); te=ctx["origin_map"][held]
    model,ntrain=ctx["C"].fit_xgb_trajectory(ctx["ftctx"]["v410a"],X141,ctx["bundle"],tr,
                                              threads=args.xgb_threads,leads_per_origin=args.xgb_leads_per_origin)
    pred=ctx["C"].predict_xgb_trajectory(model,X141[te],k=ctx["bundle"].y.shape[1])
    return schedule,x,X47,X141,names,reg,model,ntrain,te,pred


def trajectory_sums(y,p,m):
    m=np.asarray(m).astype(bool)&np.isfinite(y)&np.isfinite(p); e=np.asarray(p)[m]-np.asarray(y)[m]
    return float(np.sum(e*e)),float(np.sum(np.abs(e))),int(m.sum())


def bin_rows(battery,config,y,p,mask):
    rows=[]; y=np.asarray(y,float); p=np.asarray(p,float); mask=np.asarray(mask).astype(bool)
    for lo,hi,label in LEAD_BINS:
        ids=np.arange(1,len(y)+1); m=mask&(ids>=lo)&(ids<=hi)&np.isfinite(y)&np.isfinite(p)
        if not m.any(): continue
        e=p[m]-y[m]
        rows.append({"OuterHeldoutBattery":battery,"Config":config,"HorizonBin":label,"LeadLo":lo,"LeadHi":hi,
                     "N":int(m.sum()),"SSE":float(np.sum(e*e)),"SAE":float(np.sum(np.abs(e))),
                     "RMSE":float(np.sqrt(np.mean(e*e))),"MAE":float(np.mean(np.abs(e))),"Bias":float(np.mean(e))})
    return rows


def native_tree_shap(model, xrows: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    import xgboost as xgb
    dm=xgb.DMatrix(np.asarray(xrows,np.float32)); c=np.asarray(model.get_booster().predict(dm,pred_contribs=True),float)
    return c[:,:-1],c[:,-1]


def perturb_held_tensor(x: np.ndarray, held_i: int, target: str, pct: float) -> np.ndarray:
    z=np.asarray(x[[held_i]],np.float32).copy(); f=1.0+float(pct)/100.0
    if target in ("HI_A","BOTH"): z[:,:,:,0]*=f
    if target in ("HI_B","BOTH"): z[:,:,:,1]*=f
    return z


def split_groups(items: Sequence[str], k: int, salt: str) -> List[List[str]]:
    order=sorted(items,key=lambda b:hashlib.sha1(f"{salt}|{b}".encode()).hexdigest())
    return [order[i::k] for i in range(k)]


def select_event_l3_schedule_on_bats(ctx, source_bats, args, salt) -> Tuple[int,...]:
    B=ctx["B"]; cube=ctx["event_cube"]; cycles=B.coarse_cycles(ctx["bargs"]); schedules=B.enumerate_schedules(cycles,3,ORIGIN)
    vals=[]
    for s in schedules:
        x=B.extract_tensor(cube,s,cube.nodes,EVENT_HIS)
        if x is None: continue
        X=B.compact_features(x,s)
        rm=B.ridge_proxy_rmse(X,ctx["batteries"],source_bats,ctx["Y80"],ctx["M80"],ctx["bargs"],"DEVINNER",salt+"|"+"|".join(map(str,s)))
        vals.append((rm,s))
    if not vals: raise RuntimeError("inner schedule selection empty")
    vals.sort(key=lambda z:(z[0],z[1])); cur=vals[0][1]; best=float(vals[0][0])
    for rr in range(args.local_refine_rounds):
        cand=[cur]+B.local_schedule_neighbors(cur,args.cycle_start,ORIGIN,args.local_refine_radius)
        scored=[]
        for s in cand:
            if s==cur: scored.append((best,s)); continue
            x=B.extract_tensor(cube,s,cube.nodes,EVENT_HIS)
            if x is None: continue
            X=B.compact_features(x,s)
            rm=B.ridge_proxy_rmse(X,ctx["batteries"],source_bats,ctx["Y80"],ctx["M80"],ctx["bargs"],"DEVINNER",salt+f"|R{rr}|"+"|".join(map(str,s)))
            scored.append((rm,s))
        val,s=min(scored,key=lambda z:(z[0],z[1]))
        if s==cur or val>=best-1e-12: break
        cur,best=s,float(val)
    return tuple(cur)


def nested_uq(ctx, held, args) -> Tuple[pd.DataFrame, Dict[str,object]]:
    B=ctx["B"]; source=B.source_batteries(ctx,held); groups=split_groups(source,args.calibration_folds,f"UQ|{held}")
    rows=[]
    for fi,valb in enumerate(groups,1):
        train=[b for b in source if b not in set(valb)]
        schedule=select_event_l3_schedule_on_bats(ctx,train,args,f"UQSEL|{held}|F{fi}")
        cube=ctx["event_cube"]; x=B.extract_tensor(cube,schedule,cube.nodes,EVENT_HIS)
        X47,_,_=B.engineer_dynamic_selected(x,schedule,cube.nodes,EVENT_HIS); X141=B.embed_features_141(ctx,X47)
        tr=[ctx["origin_map"][b] for b in train]
        model,_=ctx["C"].fit_xgb_trajectory(ctx["ftctx"]["v410a"],X141,ctx["bundle"],tr,
                                             threads=args.xgb_threads,leads_per_origin=args.xgb_leads_per_origin)
        for b in valb:
            idx=ctx["origin_map"][b]; pred=ctx["C"].predict_xgb_trajectory(model,X141[idx],k=ctx["bundle"].y.shape[1])
            y=np.asarray(ctx["bundle"].y[idx],float); m=np.asarray(ctx["bundle"].mask[idx]).astype(bool)
            valid=np.flatnonzero(m & np.isfinite(y) & np.isfinite(pred))
            # equal battery weight: same maximum number of calibration leads per battery
            if len(valid)>args.xgb_leads_per_origin:
                pos=np.linspace(0,len(valid)-1,args.xgb_leads_per_origin); valid=valid[np.round(pos).astype(int)]
            for j in valid:
                lead=j+1; label=next((lab for lo,hi,lab in LEAD_BINS if lo<=lead<=hi),"Other")
                rows.append({"OuterHeldoutBattery":held,"CalibrationFold":fi,"CalibrationBattery":b,
                             "CalibrationSchedule":"|".join(map(str,schedule)),"Lead":lead,"HorizonBin":label,
                             "AbsResidual":float(abs(pred[j]-y[j]))})
        del model; gc.collect()
    res=pd.DataFrame(rows)
    if res.CalibrationBattery.nunique()!=len(source): raise RuntimeError(f"{held}: UQ battery coverage mismatch")
    return res,{"CalibrationBatteries":len(source),"CalibrationRows":len(res)}


def conformal_intervals(ctx, held, pred, residuals, args):
    te=ctx["origin_map"][held]; y=np.asarray(ctx["bundle"].y[te],float); m=np.asarray(ctx["bundle"].mask[te]).astype(bool)
    rows=[]; C=ctx["C"]
    for cov in args.coverages:
        for lo,hi,label in LEAD_BINS:
            rr=residuals[residuals.HorizonBin.eq(label)].AbsResidual.to_numpy(float)
            if len(rr)<10: continue
            q=C.finite_sample_abs_residual_quantile(rr,float(cov)) if hasattr(C,"finite_sample_abs_residual_quantile") else float(np.quantile(rr,cov))
            ids=np.arange(1,len(y)+1); mm=m&(ids>=lo)&(ids<=hi)&np.isfinite(y)&np.isfinite(pred)
            if not mm.any(): continue
            lower=pred[mm]-q; upper=pred[mm]+q; covered=(y[mm]>=lower)&(y[mm]<=upper)
            rows.append({"OuterHeldoutBattery":held,"CoverageNominal":cov,"HorizonBin":label,"q":q,
                         "N":int(mm.sum()),"Covered":int(covered.sum()),"EmpiricalCoverage":float(covered.mean()),
                         "MeanWidth":float(2*q)})
    return pd.DataFrame(rows)


# =============================================================================
# V4 cleaned waveform + exact Event re-extraction
# =============================================================================

def resolve_v4_root(args):
    if args.v4_root:
        p=Path(args.v4_root)
        if not p.exists(): raise FileNotFoundError(p)
        return p
    for p in (args.base/"RWTH48_全流程输出_V4",args.base/"RWTH48_全流程输出"):
        if p.exists() and (p/"02_batteries").exists(): return p
    raise FileNotFoundError("cannot locate V4 clean waveform root; use --v4-root")


def resolve_v4_script(args):
    if args.v4_script:
        p=Path(args.v4_script)
        if not p.exists(): raise FileNotFoundError(p)
        return p
    for n in ("01_preprocess_soh_labels.py",):
        p=args.code_dir/n
        if p.exists(): return p
    raise FileNotFoundError("cannot locate exact original V4 Event/HI script; use --v4-script")


def find_battery_dir(v4_root,battery):
    base=v4_root/"02_batteries"; tail=battery.split("-")[-1]
    for p in (base/f"Cell_{battery}",base/battery,base/f"Cell_{tail}"):
        if p.exists(): return p
    for p in base.iterdir():
        if p.is_dir() and tail in p.name: return p
    return None


def load_ref_capacity_map(v4_root):
    hits=[v4_root/"03_global"/"00_battery_quality_summary.csv",v4_root/"00_battery_quality_summary.csv"]
    p=next((x for x in hits if x.exists()),None)
    if p is None:
        z=list(v4_root.rglob("00_battery_quality_summary.csv")); p=z[0] if z else None
    if p is None: raise FileNotFoundError("00_battery_quality_summary.csv")
    d=pd.read_csv(p,low_memory=False); d["Battery"]=d.Battery.map(norm_battery); d["ReferenceCapacity_Ah"]=pd.to_numeric(d.ReferenceCapacity_Ah,errors="coerce")
    return dict(zip(d.Battery,d.ReferenceCapacity_Ah))


def read_clean_waveform_cycles(bdir: Path, cycles: Sequence[int]):
    pdir=bdir/"clean_timeseries_parts"; parts=sorted(pdir.glob("*_TBA_Zyk_clean.csv.gz")) or sorted(pdir.glob("*_TBA_Zyk_clean.csv"))
    if not parts: parts=sorted(pdir.glob("*.csv.gz"))+sorted(pdir.glob("*.csv"))
    if not parts: raise FileNotFoundError(pdir)
    wanted=["Timestamp","GlobalCycle","State","Voltage_V_Clean","Current_A_Clean","Temperature_C_Clean","Voltage_V","Current_A","Temperature_C"]
    cs=set(map(int,cycles)); frames=[]
    for p in parts:
        hdr=pd.read_csv(p,nrows=0,compression="infer").columns.tolist(); use=[c for c in wanted if c in hdr]
        if not {"Timestamp","GlobalCycle","State"}.issubset(use): continue
        d=pd.read_csv(p,usecols=use,compression="infer",low_memory=False); gcx=pd.to_numeric(d.GlobalCycle,errors="coerce")
        d=d[gcx.isin(cs)].copy()
        if len(d): frames.append(d)
    if not frames: raise RuntimeError(f"no waveform rows {bdir} cycles={cycles}")
    d=pd.concat(frames,ignore_index=True); d["Timestamp"]=pd.to_datetime(d.Timestamp,errors="coerce"); d["GlobalCycle"]=pd.to_numeric(d.GlobalCycle,errors="coerce")
    d=d[d.Timestamp.notna()&d.GlobalCycle.notna()].copy(); d["GlobalCycle"]=d.GlobalCycle.astype(int)
    return d.sort_values(["GlobalCycle","Timestamp"],kind="mergesort").reset_index(drop=True)


def add_noise(d, vsig, isig, seed):
    z=d.copy(); rng=np.random.default_rng(seed)
    for clean,raw,sig in [("Voltage_V_Clean","Voltage_V",vsig),("Current_A_Clean","Current_A",isig)]:
        if clean in z: a=pd.to_numeric(z[clean],errors="coerce").to_numpy(float)
        elif raw in z: a=pd.to_numeric(z[raw],errors="coerce").to_numpy(float); z[clean]=a
        else: raise RuntimeError(f"missing {clean}/{raw}")
        if sig>0:
            m=np.isfinite(a); n=np.zeros(len(a)); n[m]=rng.normal(0,sig,m.sum()); a=a+n
        z[clean]=a
    return z


def extract_event_tensor_waveform(d_clean,battery,refcap,schedule,nodes,his,v4,scenario=None,seed=0):
    d=d_clean.copy() if scenario is None else add_noise(d_clean,float(scenario["VoltageSigma_V"]),float(scenario["CurrentSigma_A"]),seed)
    out=np.full((1,len(schedule),len(nodes),len(his)),np.nan,np.float32); meta=[]
    for di,cyc in enumerate(schedule):
        g=v4.get_discharge_cycle(d,int(cyc))
        if len(g)<int(v4.MIN_PREFIX_ROWS): return None,pd.DataFrame(meta),f"C{cyc}:rows"
        dur=(pd.to_datetime(g.Timestamp,errors="coerce").max()-pd.to_datetime(g.Timestamp,errors="coerce").min()).total_seconds()
        if not np.isfinite(dur) or dur<float(v4.MIN_DCH_DURATION_S): return None,pd.DataFrame(meta),f"C{cyc}:duration"
        collapsed=v4.collapse_duplicate_timestamps(g)
        for mi,node in enumerate(nodes):
            idx=v4.event_cut_index_refah(g,float(refcap),float(node))
            if idx is None: return None,pd.DataFrame(meta),f"C{cyc}|{node}:unreachable"
            pref=collapsed.iloc[:int(idx)+1].copy(); feat=v4.prefix_features(pref,float(refcap))
            if not feat: return None,pd.DataFrame(meta),f"C{cyc}|{node}:features"
            vals=[]
            for h in his:
                if h not in feat or not np.isfinite(float(feat[h])): return None,pd.DataFrame(meta),f"missing {h}"
                vals.append(float(feat[h]))
            out[0,di,mi,:]=np.asarray(vals,np.float32)
            meta.append({"Battery":battery,"DiagnosisCycle":cyc,"RefAh":node,"CutIndex":int(idx),
                         "PrefixDuration_s":float(feat.get("PrefixDuration_s",np.nan)),**{f"HI_{k}":v for k,v in zip(his,vals)}})
    return out,pd.DataFrame(meta),""


def noise_scenarios(args):
    levels=["LOW","MEDIUM","HIGH"]; vs=np.asarray(args.voltage_sigmas_mv,float)/1000; cs=np.asarray(args.current_sigmas_ma,float)/1000
    rows=[]
    for j,lev in enumerate(levels):
        rows += [
            {"ScenarioID":f"V_ONLY_{lev}","VoltageSigma_V":vs[j],"CurrentSigma_A":0.0},
            {"ScenarioID":f"I_ONLY_{lev}","VoltageSigma_V":0.0,"CurrentSigma_A":cs[j]},
            {"ScenarioID":f"ALL_{lev}","VoltageSigma_V":vs[j],"CurrentSigma_A":cs[j]},
        ]
    return rows


# =============================================================================
# P0-C champion battery
# =============================================================================

def champion_paths(args, held):
    d=ensure_dir(args.output_root/"folds"/held/"champion")
    return {k:d/v for k,v in {
        "summary":"summary.json","clean":"clean.npz","uq":"uq_intervals.csv","uqres":"uq_residuals.csv.gz",
        "shap":"shap_long.csv.gz","pert":"perturbation.csv","wave":"waveform_noise.csv.gz","recon":"waveform_reconstruction.csv"}.items()}


def run_champion_battery(ctx, held, args, do_waveform: bool):
    paths=champion_paths(args,held)
    need_core=args.force or not (paths["summary"].exists() and paths["clean"].exists() and paths["uq"].exists() and paths["pert"].exists())
    if not need_core and (args.skip_shap or paths["shap"].exists()) and (not do_waveform or args.skip_waveform or paths["wave"].exists()):
        return
    B=ctx["B"]; schedule,x,X47,X141,names,reg,model,ntrain,te,pred=champion_data(ctx,held,args)
    y=np.asarray(ctx["bundle"].y[te],float); mask=np.asarray(ctx["bundle"].mask[te],np.uint8)
    old=np.load(old_timing_npz(args,held,"Event",3,"BEST_SOURCE_PROXY")); oldpred=np.asarray(old["pred"],float)
    clean_diff=float(np.max(np.abs(pred-oldpred)))
    if clean_diff>args.clean_pred_tol: raise RuntimeError(f"{held}: champion clean reproduction {clean_diff}>{args.clean_pred_tol}")
    np.savez_compressed(paths["clean"],pred=pred,true=y,mask=mask)

    # Perturbation on held-out measured HIs only; model/schedule stay frozen.
    held_i=ctx["batteries"].index(held); prows=[]
    for target in ("HI_A","HI_B","BOTH"):
        for pct in args.perturb_pcts:
            xp=perturb_held_tensor(x,held_i,target,pct); Xp,_,_=B.engineer_dynamic_selected(xp,schedule,ctx["event_cube"].nodes,EVENT_HIS)
            pp=ctx["C"].predict_xgb_trajectory(model,Xp[0],k=len(pred)); sse,sae,n=trajectory_sums(y,pp,mask)
            valid=np.asarray(mask).astype(bool)&np.isfinite(pp)&np.isfinite(pred)
            prows.append({"OuterHeldoutBattery":held,"Target":target,"DeltaPct":pct,"RMSE":math.sqrt(sse/n),"MAE":sae/n,
                          "MeanAbsDeltaSOH":float(np.mean(np.abs(pp[valid]-pred[valid]))),
                          "P95AbsDeltaSOH":float(np.quantile(np.abs(pp[valid]-pred[valid]),.95)),"MaxAbsDeltaSOH":float(np.max(np.abs(pp[valid]-pred[valid])))})
    save_csv(pd.DataFrame(prows),paths["pert"])

    # Nested source-only cross-fitted trajectory conformal; schedule is re-selected inside each calibration fold.
    residuals,uqmeta=nested_uq(ctx,held,args); save_csv(residuals,paths["uqres"],FAST_GZIP)
    uq=conformal_intervals(ctx,held,pred,residuals,args); save_csv(uq,paths["uq"])

    # Native XGBoost TreeSHAP on representative future leads.
    if not args.skip_shap:
        leads=[l for l in LEAD_SHAP_POINTS if l<=len(pred) and mask[l-1]==1]
        base=np.repeat(X47[[held_i]],len(leads),axis=0); lf=ctx["C"].lead_features(np.asarray(leads),len(pred)); xx=np.concatenate([base,lf],axis=1)
        sv,basev=native_tree_shap(model,xx); feat_names=list(names)+list(LEAD_FEATURE_NAMES)
        rr=reg.copy(); rr["FeatureName"]=names
        fmap={r.FeatureName:(r.Group,r.SourceHI) for r in rr.itertuples(index=False)}
        srows=[]
        for i,lead in enumerate(leads):
            for j,f in enumerate(feat_names):
                grp,src=fmap.get(f,("LeadCoordinate","Lead"))
                srows.append({"OuterHeldoutBattery":held,"Lead":lead,"Feature":f,"Group":grp,"SourceHI":src,"SHAP":sv[i,j],"AbsSHAP":abs(sv[i,j]),"BaseValue":basev[i]})
        save_csv(pd.DataFrame(srows),paths["shap"],FAST_GZIP)

    wave_recon_max=np.nan
    if do_waveform and not args.skip_waveform:
        v4_root=resolve_v4_root(args); v4=load_module(resolve_v4_script(args),f"rwth48_v4_wave_{held.replace('-','_')}")
        for req in ("get_discharge_cycle","collapse_duplicate_timestamps","event_cut_index_refah","prefix_features","MIN_PREFIX_ROWS","MIN_DCH_DURATION_S"):
            if not hasattr(v4,req): raise RuntimeError(f"V4 helper missing {req}")
        refmap=load_ref_capacity_map(v4_root); bdir=find_battery_dir(v4_root,held)
        if bdir is None: raise FileNotFoundError(f"waveform dir {held}")
        dclean=read_clean_waveform_cycles(bdir,schedule)
        xclean,mc,fail=extract_event_tensor_waveform(dclean,held,refmap[held],schedule,EVENT_NODES,EVENT_HIS,v4)
        if xclean is None: raise RuntimeError(f"{held}: clean waveform reconstruction fail {fail}")
        frozen=np.asarray(x[[held_i]],float); wave_recon_max=float(np.max(np.abs(xclean-frozen)))
        rmeta=mc.copy(); rmeta["MaxAbsHIReconstructionDiff"]=wave_recon_max; save_csv(rmeta,paths["recon"])
        if wave_recon_max>args.waveform_clean_tol: raise RuntimeError(f"{held}: waveform HI recon {wave_recon_max}>{args.waveform_clean_tol}")
        wrows=[]
        for sc in noise_scenarios(args):
            preds=[]; shifts=[]; failures=0
            for rep in range(args.replicates):
                xn,mn,fail=extract_event_tensor_waveform(dclean,held,refmap[held],schedule,EVENT_NODES,EVENT_HIS,v4,sc,
                                                          stable_seed(CODE_VERSION,held,sc["ScenarioID"],rep))
                if xn is None: failures+=1; continue
                Xe,_,_=B.engineer_dynamic_selected(xn,schedule,EVENT_NODES,EVENT_HIS)
                pp=ctx["C"].predict_xgb_trajectory(model,Xe[0],k=len(pred)); preds.append(pp)
                shifts.append(np.abs(pp-pred))
            if preds:
                pm=np.mean(np.asarray(preds),axis=0); sh=np.concatenate([z[np.asarray(mask).astype(bool)] for z in shifts])
                sse,sae,n=trajectory_sums(y,pm,mask)
                wrows.append({"OuterHeldoutBattery":held,"ScenarioID":sc["ScenarioID"],**sc,"ReplicatesRequested":args.replicates,
                              "ReplicatesSuccessful":len(preds),"Failures":failures,"RMSE_MCMean":math.sqrt(sse/n),"MAE_MCMean":sae/n,
                              "MeanAbsDeltaSOH":float(np.mean(sh)),"MedianAbsDeltaSOH":float(np.median(sh)),
                              "P95AbsDeltaSOH":float(np.quantile(sh,.95)),"MaxAbsDeltaSOH":float(np.max(sh))})
            else:
                wrows.append({"OuterHeldoutBattery":held,"ScenarioID":sc["ScenarioID"],**sc,"ReplicatesRequested":args.replicates,
                              "ReplicatesSuccessful":0,"Failures":failures,"RMSE_MCMean":np.nan,"MAE_MCMean":np.nan})
        save_csv(pd.DataFrame(wrows),paths["wave"],FAST_GZIP)

    save_json({"CodeVersion":CODE_VERSION,"OuterHeldoutBattery":held,"Schedule":list(schedule),"SelectedHIs":list(EVENT_HIS),
               "Nodes":list(EVENT_NODES),"HI_ScalarInputs":18,"CleanMaxAbsPredictionDiffVsV414B":clean_diff,
               "WaveformCleanMaxAbsHIDiff":wave_recon_max,"TrainRows":ntrain,**uqmeta,
               "HeldoutUsedForTimingSelection":0,"HeldoutUsedForTraining":0,"OuterTruthUsedForUQCalibration":0,
               "CalibrationBatteryUsedForOwnTraining":0,"CalibrationScheduleReselectedWithoutCalibrationFold":1},paths["summary"])
    del model; gc.collect()


# =============================================================================
# Per-battery core
# =============================================================================

def run_core_battery(ctx,held,args):
    for platform in ("Event","FixedTime"): run_l1_formal(ctx,held,platform,args)
    run_staleness(ctx,held,args)
    run_fixed_schedule_eval(ctx,held,args)


# =============================================================================
# Aggregation
# =============================================================================

def pooled_metrics_from_json(rows: pd.DataFrame) -> Dict[str,float]:
    sse=float(rows.SSE.sum()); sae=float(rows.SAE.sum()); n=int(rows.N.sum())
    return {"N":n,"RMSE":math.sqrt(sse/n),"MAE":sae/n,
            "BatteryMacroRMSE":float(pd.to_numeric(rows.Fold_RMSE,errors="coerce").mean()) if "Fold_RMSE" in rows else np.nan}


def collect_jsons(pattern: str) -> pd.DataFrame:
    rows=[]
    for p in Path(pattern.split('**')[0]).glob('**/'+pattern.split('**/')[-1]) if '**/' in pattern else []:
        try: rows.append(load_json(p))
        except Exception: pass
    return pd.DataFrame(rows)


def collect_l1(args):
    rows=[]
    for p in args.output_root.glob("folds/*/l1/*.json"):
        try: rows.append(load_json(p))
        except Exception: pass
    d=pd.DataFrame(rows)
    save_csv(d,args.output_root/"02_l1_outer_results.csv")
    mets=[]
    if len(d):
        for (pl,role),g in d.groupby(["Platform","Role"]):
            q=pooled_metrics_from_json(g); q.update({"Platform":pl,"Role":role,"Batteries":g.OuterHeldoutBattery.nunique(),
                "MedianCycle":float(np.median([x[0] for x in g.Schedule])),"MedianScalars":float(g.HI_ScalarInputs.median()),
                "MedianSensors":float(g.PhysicalSensorCount.median()),"MedianDuration_s":float(g.AcquisitionDuration_s.median())}); mets.append(q)
    m=pd.DataFrame(mets); save_csv(m,args.output_root/"03_l1_pooled_metrics.csv")
    if len(d):
        f=d[d.Role.eq("BEST_SOURCE_PROXY")].explode("Schedule").groupby(["Platform","Schedule"]).size().reset_index(name="OuterFoldCount").sort_values(["Platform","OuterFoldCount"],ascending=[True,False])
    else: f=pd.DataFrame()
    save_csv(f,args.output_root/"04_l1_best_cycle_frequency.csv")
    # All-cycle source-only timing value curve: at each cycle retain the best
    # refined one-scalar candidate within each outer fold, then summarize 47 folds.
    curves=[]
    for pl in ("Event","FixedTime"):
        parts=[]
        for pscan in args.output_root.glob(f"folds/*/l1/{pl}_source_scan.csv.gz"):
            q=pd.read_csv(pscan,compression="infer",low_memory=False)
            if len(q):
                z=q.sort_values(["Cycle","ProxyRMSE"]).groupby("Cycle",as_index=False).first()
                parts.append(z[["OuterHeldoutBattery","Cycle","ProxyRMSE","Node","HI","PhysicalSensorCount","AcquisitionDuration_s"]])
        if parts:
            q=pd.concat(parts,ignore_index=True)
            a=q.groupby("Cycle",as_index=False).agg(Folds=("OuterHeldoutBattery","nunique"),MedianProxyRMSE=("ProxyRMSE","median"),MeanProxyRMSE=("ProxyRMSE","mean"),
                                                       P25ProxyRMSE=("ProxyRMSE",lambda x:float(np.quantile(x,.25))),P75ProxyRMSE=("ProxyRMSE",lambda x:float(np.quantile(x,.75))))
            a.insert(0,"Platform",pl); curves.append(a)
    curve=pd.concat(curves,ignore_index=True) if curves else pd.DataFrame()
    save_csv(curve,args.output_root/"04b_l1_source_proxy_cycle_curve.csv")
    return d,m


def collect_staleness(ctx,args):
    rows=[]
    # stale=0 baseline = existing source-optimized V4.14B L3.
    for b in ctx["batteries"]:
        for pl in ("Event","FixedTime"):
            p=old_timing_json(args,b,pl,3,"BEST_SOURCE_PROXY")
            if p.exists():
                q=load_json(p); q["Staleness"]=0; q["Role"]="BEST_SOURCE_PROXY"; rows.append(q)
            for stale in args.staleness_values:
                p=args.output_root/"folds"/b/"staleness"/f"{pl}_STALE{stale}.json"
                if p.exists(): rows.append(load_json(p))
    d=pd.DataFrame(rows); save_csv(d,args.output_root/"05_staleness_outer_results.csv")
    mets=[]
    if len(d):
        for (pl,st),g in d.groupby(["Platform","Staleness"]):
            q=pooled_metrics_from_json(g); q.update({"Platform":pl,"Role":"BEST_SOURCE_PROXY","Staleness":st,"LastCycle":ORIGIN-int(st),"Batteries":g.OuterHeldoutBattery.nunique()}); mets.append(q)
    m=pd.DataFrame(mets); save_csv(m,args.output_root/"06_staleness_pooled_metrics.csv"); return d,m


def collect_consensus(ctx,args):
    rows=[]
    for p in args.output_root.glob("folds/*/consensus/*.json"):
        try: rows.append(load_json(p))
        except Exception: pass
    d=pd.DataFrame(rows); save_csv(d,args.output_root/"08_consensus_outer_results.csv")
    mets=[]
    if len(d):
        for role,g in d.groupby("Role"):
            q=pooled_metrics_from_json(g); q.update({"Role":role,"Batteries":g.OuterHeldoutBattery.nunique(),"Schedule":"|".join(map(str,g.iloc[0].Schedule))}); mets.append(q)
    # comparison baselines on the confirmation-15 subset.
    for role in ("BEST_SOURCE_PROXY","UNIFORM"):
        rr=[]
        for b in ctx["confirm_batteries"]:
            p=old_timing_json(args,b,"Event",3,role)
            if p.exists(): rr.append(load_json(p))
        if rr:
            g=pd.DataFrame(rr); q=pooled_metrics_from_json(g); q.update({"Role":f"CONFIRM15_{role}","Batteries":len(g),"Schedule":"fold-specific"}); mets.append(q)
    m=pd.DataFrame(mets); save_csv(m,args.output_root/"09_consensus_pooled_metrics.csv")
    dev,_=select_dev_consensus_schedule(ctx,args)
    save_json({"DevBatteries":ctx["dev_batteries"],"ConfirmBatteries":ctx["confirm_batteries"],"SelectedDev32Schedule":list(dev),
               "PostHocConsensusSchedule":list(posthoc_consensus_schedule(args))},args.output_root/"07_event_consensus_schedule.json")
    return d,m


def load_npz_bins(npz,battery,config):
    z=np.load(npz); return bin_rows(battery,config,np.asarray(z['true'],float),np.asarray(z['pred'],float),np.asarray(z['mask']))


def collect_horizon(ctx,args):
    rows=[]
    # Existing timing configs.
    for b in ctx["batteries"]:
        for L in (2,3,4,5,6):
            p=old_timing_npz(args,b,"Event",L,"BEST_SOURCE_PROXY")
            if p.exists(): rows += load_npz_bins(p,b,f"Event_Opt_L{L}")
        p=old_timing_npz(args,b,"Event",3,"UNIFORM")
        if p.exists(): rows += load_npz_bins(p,b,"Event_Uniform_L3")
        p=old_timing_npz(args,b,"FixedTime",3,"BEST_SOURCE_PROXY")
        if p.exists(): rows += load_npz_bins(p,b,"FixedTime_Opt_L3")
        p=old_timing_npz(args,b,"FixedTime",5,"UNIFORM")
        if p.exists(): rows += load_npz_bins(p,b,"FixedTime_Uniform_L5")
        for pl in ("Event","FixedTime"):
            p=args.output_root/"folds"/b/"l1"/f"{pl}_BEST_SOURCE_PROXY.npz"
            if p.exists(): rows += load_npz_bins(p,b,f"{pl}_L1_Best")
        p=args.output_root/"folds"/b/"consensus"/"POSTHOC_CONSENSUS.npz"
        if p.exists(): rows += load_npz_bins(p,b,"Event_PostHocConsensus")
        p=args.output_root/"folds"/b/"consensus"/"DEV32_FIXED.npz"
        if p.exists(): rows += load_npz_bins(p,b,"Event_Dev32Fixed_Confirm15")
    d=pd.DataFrame(rows); save_csv(d,args.output_root/"10_horizon_bin_by_battery.csv")
    agg=[]
    if len(d):
        for (cfg,hb),g in d.groupby(["Config","HorizonBin"]):
            n=int(g.N.sum()); sse=float(g.SSE.sum()); sae=float(g.SAE.sum())
            agg.append({"Config":cfg,"HorizonBin":hb,"Batteries":g.OuterHeldoutBattery.nunique(),"N":n,
                        "RMSE":math.sqrt(sse/n),"MAE":sae/n,"BatteryMacroRMSE":float(g.RMSE.mean())})
    a=pd.DataFrame(agg); save_csv(a,args.output_root/"11_horizon_bin_metrics.csv"); return d,a


def collect_champion(ctx,args):
    clean=[]; uq=[]; shap=[]; pert=[]; wave=[]; recon=[]; summaries=[]
    for b in ctx["batteries"]:
        p=champion_paths(args,b)
        if p["summary"].exists(): summaries.append(load_json(p["summary"]))
        if p["uq"].exists(): uq.append(pd.read_csv(p["uq"]))
        if p["shap"].exists(): shap.append(pd.read_csv(p["shap"],compression="infer",low_memory=False))
        if p["pert"].exists(): pert.append(pd.read_csv(p["pert"]))
        if p["wave"].exists(): wave.append(pd.read_csv(p["wave"],compression="infer",low_memory=False))
        if p["recon"].exists(): recon.append(pd.read_csv(p["recon"]))
    s=pd.DataFrame(summaries); save_csv(s,args.output_root/"13_champion_clean_reproduction.csv")
    uqdf=pd.concat(uq,ignore_index=True) if uq else pd.DataFrame(); save_csv(uqdf,args.output_root/"14_champion_uq_intervals.csv")
    uqa=[]
    if len(uqdf):
        for (cov,hb),g in uqdf.groupby(["CoverageNominal","HorizonBin"]):
            uqa.append({"CoverageNominal":cov,"HorizonBin":hb,"N":int(g.N.sum()),"Covered":int(g.Covered.sum()),
                        "EmpiricalCoverage":float(g.Covered.sum()/g.N.sum()),"MeanWidth":float(np.average(g.MeanWidth,weights=g.N))})
    save_csv(pd.DataFrame(uqa),args.output_root/"15_champion_uq_overall.csv")
    sh=pd.concat(shap,ignore_index=True) if shap else pd.DataFrame()
    if len(sh):
        gs=sh.groupby("Group",as_index=False).AbsSHAP.sum(); gs["Share"]=gs.AbsSHAP/gs.AbsSHAP.sum(); gs=gs.sort_values("AbsSHAP",ascending=False)
        hs=sh.groupby("SourceHI",as_index=False).AbsSHAP.sum(); hs["Share"]=hs.AbsSHAP/hs.AbsSHAP.sum(); hs=hs.sort_values("AbsSHAP",ascending=False)
    else: gs=hs=pd.DataFrame()
    save_csv(gs,args.output_root/"16_champion_shap_group_summary.csv"); save_csv(hs,args.output_root/"17_champion_shap_hi_summary.csv")
    pe=pd.concat(pert,ignore_index=True) if pert else pd.DataFrame()
    if len(pe): ps=pe.groupby(["Target","DeltaPct"],as_index=False).agg(RMSE_Mean=("RMSE","mean"),MeanAbsDeltaSOH=("MeanAbsDeltaSOH","mean"),P95AbsDeltaSOH=("P95AbsDeltaSOH","mean"))
    else: ps=pd.DataFrame()
    save_csv(ps,args.output_root/"18_champion_perturbation_summary.csv")
    wa=pd.concat(wave,ignore_index=True) if wave else pd.DataFrame()
    if len(wa): ws=wa.groupby("ScenarioID",as_index=False).agg(Batteries=("OuterHeldoutBattery","nunique"),RMSE_MCMean=("RMSE_MCMean","mean"),MeanAbsDeltaSOH=("MeanAbsDeltaSOH","mean"),P95AbsDeltaSOH=("P95AbsDeltaSOH","mean"),Failures=("Failures","sum"))
    else: ws=pd.DataFrame()
    save_csv(ws,args.output_root/"19_champion_waveform_noise_summary.csv")
    rc=pd.concat(recon,ignore_index=True) if recon else pd.DataFrame(); save_csv(rc,args.output_root/"20_champion_waveform_reconstruction.csv")
    return s,uqdf,pd.DataFrame(uqa),gs,hs,ps,ws,rc


def old_pooled_tables(args):
    out=[]
    p=args.old_v414b_root/"04_timing_pooled_metrics.csv"
    if p.exists():
        d=pd.read_csv(p); d["Config"]=[f"{r.Platform}_{r.Role}_L{int(r.DiagnosisCount)}" for r in d.itertuples()]; out.append(d.rename(columns={"MedianHI_ScalarInputs":"Scalars","MedianPhysicalSensorCount":"Sensors","MedianAcquisitionDuration_s":"Duration_s"})[["Platform","Config","RMSE","Scalars","Sensors","Duration_s"]])
    p=args.old_v414b_root/"09_resource_policy_metrics.csv"
    if p.exists():
        d=pd.read_csv(p); d["Config"]=[f"{r.Platform}_{r.PolicyRole}" for r in d.itertuples()]; out.append(d.rename(columns={"MedianHI_ScalarInputs":"Scalars","MedianPhysicalSensorCount":"Sensors","MedianAcquisitionDuration_s":"Duration_s"})[["Platform","Config","RMSE","Scalars","Sensors","Duration_s"]])
    # Frozen anchors.
    comp=load_json(args.old_v414b_root/"99_RWTH48_V4_14B_COMPLETE.json")
    ar=[]
    for q in comp.get("FrozenSOHAnchors",[]):
        # duration/sensors unavailable for frozen anchors; keep NaN.
        ar.append({"Platform":q["Platform"],"Config":q["Anchor"],"RMSE":q["RMSE"],"Scalars":q["HI_ScalarInputs"],"Sensors":np.nan,"Duration_s":np.nan})
    if ar: out.append(pd.DataFrame(ar))
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()


def threshold_pareto(l1met,consmet,args):
    d=old_pooled_tables(args)
    add=[]
    if len(l1met):
        for r in l1met.itertuples(): add.append({"Platform":r.Platform,"Config":f"{r.Platform}_L1_{r.Role}","RMSE":r.RMSE,"Scalars":r.MedianScalars,"Sensors":r.MedianSensors,"Duration_s":r.MedianDuration_s})
    if add: d=pd.concat([d,pd.DataFrame(add)],ignore_index=True)
    d=d[np.isfinite(pd.to_numeric(d.RMSE,errors="coerce"))].copy(); save_csv(d,args.output_root/"12_validated_configuration_catalog.csv")
    # Multi-dimensional nondominance: RMSE, scalars, sensors, duration; missing cost cannot dominate/be dominated on that cost.
    pareto=[]
    for pl,g in d.groupby("Platform"):
        g=g.reset_index(drop=True)
        for i,r in g.iterrows():
            dominated=False
            for j,s in g.iterrows():
                if i==j: continue
                dims=[]
                for col in ("RMSE","Scalars","Sensors","Duration_s"):
                    rv,sv=float(r[col]) if pd.notna(r[col]) else np.nan,float(s[col]) if pd.notna(s[col]) else np.nan
                    if np.isfinite(rv) and np.isfinite(sv): dims.append((sv,rv))
                if dims and all(a<=b+1e-15 for a,b in dims) and any(a<b-1e-15 for a,b in dims): dominated=True; break
            if not dominated: pareto.append(r.to_dict())
    pdf=pd.DataFrame(pareto); save_csv(pdf,args.output_root/"21_empirical_validated_pareto.csv")
    rows=[]
    for th in args.thresholds:
        for pl,g in d[d.RMSE.lt(th)].groupby("Platform"):
            for role,col in [("MIN_SCALAR","Scalars"),("MIN_SENSOR","Sensors"),("MIN_DURATION","Duration_s")]:
                q=g[np.isfinite(pd.to_numeric(g[col],errors="coerce"))]
                if q.empty: continue
                r=q.sort_values([col,"RMSE"]).iloc[0]
                rows.append({"ThresholdRMSE":th,"Platform":pl,"Role":role,"Config":r.Config,"RMSE":r.RMSE,"Scalars":r.Scalars,"Sensors":r.Sensors,"Duration_s":r.Duration_s})
    t=pd.DataFrame(rows); save_csv(t,args.output_root/"22_threshold_sensitivity.csv"); return d,pdf,t


def finalize(ctx,args):
    l1,l1m=collect_l1(args); stale,stalem=collect_staleness(ctx,args); cons,consm=collect_consensus(ctx,args); hb,hbm=collect_horizon(ctx,args)
    s,uq,uqa,sg,sh,ps,ws,rc=collect_champion(ctx,args); cat,pareto,th=threshold_pareto(l1m,consm,args)
    checks=[
        {"Check":"47 batteries L1 formal best Event","Observed":int(l1[(l1.Platform.eq('Event'))&(l1.Role.eq('BEST_SOURCE_PROXY'))].OuterHeldoutBattery.nunique()) if len(l1) else 0,"Expected":47},
        {"Check":"47 batteries L1 formal best FixedTime","Observed":int(l1[(l1.Platform.eq('FixedTime'))&(l1.Role.eq('BEST_SOURCE_PROXY'))].OuterHeldoutBattery.nunique()) if len(l1) else 0,"Expected":47},
        {"Check":"Staleness rows available","Observed":int(len(stale)),"Expected":">0"},
        {"Check":"Dev32 fixed schedule confirm batteries","Observed":int(cons[cons.Role.eq('DEV32_FIXED')].OuterHeldoutBattery.nunique()) if len(cons) else 0,"Expected":47-args.dev_batteries},
        {"Check":"Champion clean reproduction batteries","Observed":int(s.OuterHeldoutBattery.nunique()) if len(s) else 0,"Expected":47},
        {"Check":"Champion clean exact tolerance","Observed":float(pd.to_numeric(s.CleanMaxAbsPredictionDiffVsV414B,errors='coerce').max()) if len(s) else np.inf,"Expected":f"<={args.clean_pred_tol}"},
        {"Check":"Champion UQ batteries","Observed":int(uq.OuterHeldoutBattery.nunique()) if len(uq) else 0,"Expected":47},
        {"Check":"Horizon bins generated","Observed":int(len(hbm)),"Expected":">0"},
        {"Check":"Threshold sensitivity generated","Observed":int(len(th)),"Expected":">0"},
    ]
    if not args.skip_waveform:
        checks += [
            {"Check":"Waveform reconstruction batteries","Observed":int(rc.Battery.nunique()) if len(rc) and 'Battery' in rc else 0,"Expected":47},
            {"Check":"Waveform noise scenarios","Observed":int(len(ws)),"Expected":9},
        ]
    gate=[]
    for q in checks:
        exp=q["Expected"]; obs=q["Observed"]
        if isinstance(exp,int): ok=int(obs==exp)
        elif exp==">0": ok=int(float(obs)>0)
        elif isinstance(exp,str) and exp.startswith("<="): ok=int(float(obs)<=float(exp[2:]))
        else: ok=1
        gate.append({**q,"PASS":ok})
    g=pd.DataFrame(gate); save_csv(g,args.output_root/"23_acceptance_gate.csv")
    status="PASS" if g.PASS.eq(1).all() else "CHECK"
    # Main headline rows.
    event_l1=l1m[(l1m.Platform.eq('Event'))&(l1m.Role.eq('BEST_SOURCE_PROXY'))].to_dict('records') if len(l1m) else []
    fixed_l1=l1m[(l1m.Platform.eq('FixedTime'))&(l1m.Role.eq('BEST_SOURCE_PROXY'))].to_dict('records') if len(l1m) else []
    opt3=None
    p=args.old_v414b_root/"04_timing_pooled_metrics.csv"
    if p.exists():
        d=pd.read_csv(p); q=d[(d.Platform.eq('Event'))&(d.DiagnosisCount.eq(3))&(d.Role.eq('BEST_SOURCE_PROXY'))]
        if len(q): opt3=q.iloc[0].to_dict()
    complete={"Task":"RWTH48 V4.14C SOH P0 Closeout","CodeVersion":CODE_VERSION,"Status":status,
              "P0Completed":{"L1":True,"Staleness":True,"ChampionConsistency":True,"HorizonBins":True,"ConsensusSchedule":True,"ThresholdPareto":True,
                             "WaveformRobustness":not args.skip_waveform,"TreeSHAP":not args.skip_shap},
              "EventOptL3Anchor":opt3,"EventL1Best":event_l1,"FixedTimeL1Best":fixed_l1,
              "Dev32Schedule":load_json(args.output_root/"07_event_consensus_schedule.json") if (args.output_root/"07_event_consensus_schedule.json").exists() else {},
              "UQOverall":uqa.to_dict('records'),"SHAPGroups":sg.to_dict('records'),"PerturbationSummary":ps.to_dict('records'),"WaveformNoiseSummary":ws.to_dict('records'),
              "Protocol":{"OuterLOBO":True,"HeldoutUsedForTimingSelection":False,"HeldoutUsedForTraining":False,"OuterTruthUsedForConformalCalibration":False,
                          "CalibrationScheduleReselectedInsideFold":True,"HistoricalSOHAsInput":False,"FutureHIAsInput":False,"CycleAsModelFeature":False,
                          "PostHocConsensusIsDescriptiveOnly":True,"Dev32FixedScheduleUsesConfirm15ForPrimaryConfirmation":True,
                          "WaveformNoiseAfterFrozenCausalQC":not args.skip_waveform},
              "Acceptance":g.to_dict('records')}
    save_json(complete,args.output_root/"99_RWTH48_V4_14C_COMPLETE.json")
    print("\n"+"="*120); print("V4.14C P0 FINALIZE |",status); print(g.to_string(index=False)); print("Output:",args.output_root); print("="*120)
    if status!="PASS": raise RuntimeError("V4.14C acceptance CHECK")


# =============================================================================
# Parallel barrier
# =============================================================================

def state_dir(args): return ensure_dir(args.state_root/args.run_id)
def done_marker(args,w): return state_dir(args)/f"worker_{w:02d}.done.json"
def fail_marker(args,w): return state_dir(args)/f"worker_{w:02d}.fail.json"
def write_done(args,owned): save_json({"Worker":args.parallel_worker_id,"Owned":owned,"Status":"PASS"},done_marker(args,args.parallel_worker_id))
def write_fail(args,e): save_json({"Worker":args.parallel_worker_id,"Status":"FAIL","Exception":repr(e),"Traceback":traceback.format_exc()},fail_marker(args,args.parallel_worker_id))

def barrier(ctx,args):
    if args.parallel_workers<=1: return finalize(ctx,args)
    if args.parallel_worker_id!=0: return
    while True:
        bad=[i for i in range(args.parallel_workers) if fail_marker(args,i).exists()]
        if bad: raise RuntimeError(f"worker failures {bad}")
        done=[i for i in range(args.parallel_workers) if done_marker(args,i).exists()]
        if len(done)==args.parallel_workers: return finalize(ctx,args)
        print(f"[BARRIER] done={done}",flush=True); time.sleep(max(1,args.barrier_poll_s))


def save_config(ctx,pre,args):
    save_csv(pre,args.output_root/"00_preflight.csv")
    save_json({"Task":"V4.14C P0 Closeout","CodeVersion":CODE_VERSION,"Stages":["L1","staleness","consensus","horizon","threshold","champion","UQ","SHAP","perturbation","waveform"],
               "StalenessValues":args.staleness_values,"Thresholds":args.thresholds,"CalibrationFolds":args.calibration_folds,"WaveformReplicates":args.replicates,
               "DevBatteries":ctx["dev_batteries"],"ConfirmBatteries":ctx["confirm_batteries"]},args.output_root/"00_config.json")


def main():
    args=parse_args(); ensure_dir(args.output_root); ensure_dir(args.state_root)
    # thread limits before libraries fan out
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[k]=str(max(1,args.cpu_threads_per_worker))
    ctx,pre=preflight(args)
    if args.parallel_worker_id==0 or args.parallel_workers==1: save_config(ctx,pre,args)
    print("="*120); print("RWTH48 V4.14C | SOH P0 Closeout |",args.stage); print("Worker",args.parallel_worker_id,"/",args.parallel_workers); print("Output",args.output_root); print("="*120)
    if args.stage=="audit": print(pre.to_string(index=False)); print("AUDIT PASS"); return
    if args.stage=="finalize": finalize(ctx,args); return
    bats=ctx["batteries"]
    if args.only_battery:
        b=norm_battery(args.only_battery)
        if b not in bats: raise RuntimeError(f"unknown battery {b}")
        owned=[b]
    else: owned=bats[args.parallel_worker_id::args.parallel_workers]
    # clear current markers
    for p in (done_marker(args,args.parallel_worker_id),fail_marker(args,args.parallel_worker_id)):
        if p.exists(): p.unlink()
    try:
        for i,b in enumerate(owned,1):
            print(f"\n[V414C] worker {args.parallel_worker_id} battery {i}/{len(owned)} {b}",flush=True)
            if args.stage in ("core","all"): run_core_battery(ctx,b,args)
            if args.stage == "champion": run_champion_battery(ctx,b,args,do_waveform=False)
            elif args.stage == "waveform" and not args.skip_waveform: run_champion_battery(ctx,b,args,do_waveform=True)
            elif args.stage == "all": run_champion_battery(ctx,b,args,do_waveform=(not args.skip_waveform))
        write_done(args,owned)
    except Exception as e:
        write_fail(args,e); raise
    if args.only_battery: return
    if args.stage=="all": barrier(ctx,args)
    elif args.parallel_workers<=1: print("Stage complete; run remaining stages then --stage finalize")


if __name__=="__main__":
    main()

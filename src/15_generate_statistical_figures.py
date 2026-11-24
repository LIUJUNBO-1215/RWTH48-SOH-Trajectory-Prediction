# -*- coding: utf-8 -*-
"""
RWTH48 V4.23 | Final Figure & Statistics Closure | FIX1
=======================================================
NO MODEL TRAINING. NO TIMING SEARCH. NO NEW FEATURE SEARCH.

Mandatory:
  A) 0.85-SOH derived-RUL 47-cell paired bootstrap
  B) Event L2-L6 x forecast-horizon RMSE map
  C) Event L2-L6 x forecast-horizon timing-gain map + paired bootstrap
  D) Adaptive 2/3-diagnosis schedule extraction + flow/Sankey data

Optional auto-discovery:
  E) SHAP compact-HI interpretation figure
  F) Sparse SOH Anchor supplementary K-H map

Optional legacy artifacts that cannot be discovered are recorded as SKIPPED;
they do not block the mandatory modules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

CODE_VERSION = "V423_FINAL_FIGURE_STATISTICS_CLOSURE_FIX1"
BASE = (Path(__file__).resolve().parents[1] / "workspace")
V414B_NAME = "RWTH48_V4_14B_FullLifeSOH_Timing_MinResource"
V415_NAME = "RWTH48_V4_15_BudgetAware_Adaptive_Diagnostic_Scheduling"
FINAL_CLOSURE_NAME = "RWTH48_V4_21_22_Final_Closure"
OUTPUT_NAME = "RWTH48_V4_23_Final_Figure_Statistics_Closure"
ENC = "utf-8-sig"

EVENT_LS = (2, 3, 4, 5, 6)
ROLES = ("BEST_SOURCE_PROXY", "UNIFORM")
HORIZONS = (
    (1, 100, "1-100"),
    (101, 300, "101-300"),
    (301, 500, "301-500"),
    (501, 700, "501-700"),
    (701, 1000, "701-1000"),
)
SELECTED_HIS = ("V_TimeWeightedMean", "P_Last")
EVENT_NODES = (0.05, 0.09, 0.13)
FIXEDTIME_NODES = (60, 120, 180, 300, 450, 600, 900, 1200)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["audit", "rul85", "lh", "adaptive", "interpret", "all"], default="all")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--v414b-root", type=Path, default=None)
    p.add_argument("--v415-root", type=Path, default=None)
    p.add_argument("--final-closure-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--primary-rul-threshold", type=float, default=0.85)
    p.add_argument("--adaptive-schedule-file", type=Path, default=None)
    p.add_argument("--shap-summary", type=Path, default=None)
    p.add_argument("--sparse-anchor-file", type=Path, default=None)
    a = p.parse_args()
    a.base = Path(a.base)
    a.v414b_root = Path(a.v414b_root) if a.v414b_root else a.base / V414B_NAME
    a.v415_root = Path(a.v415_root) if a.v415_root else a.base / V415_NAME
    a.final_closure_root = Path(a.final_closure_root) if a.final_closure_root else a.base / FINAL_CLOSURE_NAME
    a.output_root = Path(a.output_root) if a.output_root else a.base / OUTPUT_NAME
    if a.adaptive_schedule_file: a.adaptive_schedule_file = Path(a.adaptive_schedule_file)
    if a.shap_summary: a.shap_summary = Path(a.shap_summary)
    if a.sparse_anchor_file: a.sparse_anchor_file = Path(a.sparse_anchor_file)
    if a.bootstrap < 1000: p.error("--bootstrap must be >=1000")
    if not (0.5 < a.primary_rul_threshold < 1.0): p.error("invalid RUL threshold")
    return a


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, p: Path):
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding=ENC)


def save_json(obj, p: Path):
    ensure_dir(p.parent)
    def cv(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x))
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=cv), encoding="utf-8")


def stable_seed(seed: int, text: str):
    return int(hashlib.sha256(f"{seed}|{text}".encode()).hexdigest()[:8], 16) % (2**31 - 1)


def paired_bootstrap(diff, reps: int, seed: int):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) < 2: raise RuntimeError("too few paired batteries")
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, float)
    for i in range(0, reps, 1000):
        n = min(1000, reps-i)
        idx = rng.integers(0, len(d), size=(n, len(d)))
        vals[i:i+n] = d[idx].mean(axis=1)
    return {
        "ObservedMeanDifference": float(d.mean()),
        "CI95_Low": float(np.quantile(vals, .025)),
        "CI95_High": float(np.quantile(vals, .975)),
        "P_FirstBetter": float(np.mean(vals < 0)),
        "Batteries": int(len(d)),
        "BootstrapReps": int(reps),
        "FirstWins": int(np.sum(d < 0)),
        "Ties": int(np.sum(d == 0)),
        "FirstLosses": int(np.sum(d > 0)),
    }


def normalize_col(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(cols: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    exact = {str(c): str(c) for c in cols}
    for x in aliases:
        if x in exact: return x
    norm = {normalize_col(c): c for c in cols}
    for x in aliases:
        if normalize_col(x) in norm: return norm[normalize_col(x)]
    return None


# -----------------------------------------------------------------------------
# Audit
# -----------------------------------------------------------------------------

def prediction_npz(a, battery: str, L: int, role: str):
    return a.v414b_root / "folds" / battery / "timing" / f"Event_L{L}_LAST80_{role}.npz"


def audit(a):
    rul = a.final_closure_root / "01_RUL_EOL" / "01_battery_RUL_EOL.csv"
    found_npz = 0
    for L in EVENT_LS:
        for role in ROLES:
            found_npz += len(list((a.v414b_root / "folds").glob(f"*/timing/Event_L{L}_LAST80_{role}.npz")))
    return pd.DataFrame([
        {"Check":"Final derived-RUL battery table","Observed":str(rul),"Expected":"exists","PASS":int(rul.exists())},
        {"Check":"V4.14B root","Observed":str(a.v414b_root),"Expected":"exists","PASS":int(a.v414b_root.exists())},
        {"Check":"V4.15 root","Observed":str(a.v415_root),"Expected":"exists","PASS":int(a.v415_root.exists())},
        {"Check":"V4.14B Event L2-L6 prediction NPZ files","Observed":found_npz,"Expected":470,"PASS":int(found_npz==470)},
    ])


# -----------------------------------------------------------------------------
# A. RUL85
# -----------------------------------------------------------------------------

def run_rul85(a):
    out = ensure_dir(a.output_root / "01_RUL85_Final")
    p = a.final_closure_root / "01_RUL_EOL" / "01_battery_RUL_EOL.csv"
    if not p.exists(): raise FileNotFoundError(p)
    d = pd.read_csv(p, low_memory=False)
    theta = float(a.primary_rul_threshold)
    need = {"Battery","Role","Model","Threshold","CrossingMode","AbsRUL_Error","TrueCensored","PredCensored"}
    miss = need - set(d.columns)
    if miss: raise RuntimeError(f"RUL table missing {sorted(miss)}")

    base = d[np.isclose(pd.to_numeric(d.Threshold, errors="coerce"), theta) & d.CrossingMode.astype(str).eq("monotone")].copy()
    coverage = base.groupby(["Role","Model"], as_index=False).agg(
        Batteries=("Battery","nunique"),
        Scored=("AbsRUL_Error", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        TrueCensored=("TrueCensored","sum"),
        PredCensored=("PredCensored","sum"),
    )
    save_csv(coverage, out/"01_RUL85_coverage.csv")

    x = base[base.TrueCensored.eq(0)&base.PredCensored.eq(0)&base.AbsRUL_Error.notna()].copy()
    metrics = x.groupby(["Role","Model"], as_index=False).agg(
        Batteries=("Battery","nunique"),
        MAE_cycles=("AbsRUL_Error","mean"),
        MedianAE_cycles=("AbsRUL_Error","median"),
    )
    if "RUL_Error" in x.columns:
        rr = x.groupby(["Role","Model"])["RUL_Error"].apply(lambda s: float(np.sqrt(np.mean(np.asarray(s,float)**2)))).rename("RMSE_cycles").reset_index()
        metrics = metrics.merge(rr, on=["Role","Model"], how="left")
    if "TrueRUL" in x.columns:
        mm = x.groupby(["Role","Model"]).apply(lambda g: float(100*np.mean(np.abs(g.RUL_Error)/np.maximum(np.abs(g.TrueRUL),1e-9))), include_groups=False).rename("MAPE_pct").reset_index()
        metrics = metrics.merge(mm, on=["Role","Model"], how="left")
    save_csv(metrics, out/"02_RUL85_metrics.csv")

    comps = [
        ("SafeMoE_OPT - XGB_OPT", ("OPT","SafeResidualMoE"), ("OPT","FrozenXGB")),
        ("XGB_OPT - XGB_UNIFORM", ("OPT","FrozenXGB"), ("UNIFORM","FrozenXGB")),
        ("SafeMoE_OPT - SafeMoE_UNIFORM", ("OPT","SafeResidualMoE"), ("UNIFORM","SafeResidualMoE")),
    ]
    rows, per = [], []
    for name, f, s in comps:
        a1 = x[x.Role.eq(f[0])&x.Model.eq(f[1])].set_index("Battery").AbsRUL_Error
        a2 = x[x.Role.eq(s[0])&x.Model.eq(s[1])].set_index("Battery").AbsRUL_Error
        both = a1.index.intersection(a2.index)
        diff = a1.loc[both].to_numpy(float)-a2.loc[both].to_numpy(float)
        rows.append({"Comparison":name,"Threshold":theta,"CrossingMode":"monotone","Metric":"Absolute RUL error (cycles)",**paired_bootstrap(diff,a.bootstrap,stable_seed(a.seed,"RUL85|"+name))})
        for b in both:
            per.append({"Comparison":name,"Battery":b,"FirstAbsError":float(a1.loc[b]),"SecondAbsError":float(a2.loc[b]),"Difference":float(a1.loc[b]-a2.loc[b])})
    boot = pd.DataFrame(rows)
    save_csv(boot, out/"03_RUL85_paired_bootstrap.csv")
    save_csv(pd.DataFrame(per), out/"04_RUL85_per_battery_differences.csv")

    fig, ax = plt.subplots(figsize=(8.5,5.0))
    y = np.arange(len(boot))
    lo = boot.ObservedMeanDifference-boot.CI95_Low
    hi = boot.CI95_High-boot.ObservedMeanDifference
    ax.errorbar(boot.ObservedMeanDifference, y, xerr=np.vstack([lo,hi]), fmt="o", capsize=4)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(boot.Comparison)
    ax.set_xlabel("Difference in absolute RUL error (cycles)")
    ax.set_title(f"Derived RUL paired bootstrap at SOH={theta:.2f}")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(out/"Figure_RUL85_Paired_Bootstrap.png", dpi=300); plt.close(fig)

    save_json({"Task":"0.85-SOH derived RUL final paired statistics","CodeVersion":CODE_VERSION,"Status":"PASS","Threshold":theta,"BootstrapReps":a.bootstrap,"NewTraining":False}, out/"99_RUL85_COMPLETE.json")
    print("\nRUL85 COMPLETE\n", metrics.to_string(index=False), "\n\n", boot.to_string(index=False))


# -----------------------------------------------------------------------------
# B/C. L-H / TimingGain-H
# -----------------------------------------------------------------------------

def discover_batteries_lh(a):
    root = a.v414b_root/"folds"
    if not root.exists(): return []
    return [p.name for p in sorted(root.iterdir()) if p.is_dir() and prediction_npz(a,p.name,3,"BEST_SOURCE_PROXY").exists()]


def _rmse(y,p,m):
    ok = np.asarray(m).astype(bool)&np.isfinite(y)&np.isfinite(p)
    if not ok.any(): return np.nan,0,0.0
    e = np.asarray(p)[ok]-np.asarray(y)[ok]
    return float(np.sqrt(np.mean(e*e))), int(ok.sum()), float(np.sum(e*e))


def run_lh(a):
    out = ensure_dir(a.output_root/"02_L_H_TimingGain")
    bats = discover_batteries_lh(a)
    if len(bats)!=47: raise RuntimeError(f"L-H batteries={len(bats)} !=47")
    brow, prow = [], []
    for L in EVENT_LS:
        for role in ROLES:
            sse = {hb:0.0 for _,_,hb in HORIZONS}; nn = {hb:0 for _,_,hb in HORIZONS}
            local = []
            for b in bats:
                p = prediction_npz(a,b,L,role)
                if not p.exists(): raise FileNotFoundError(p)
                z=np.load(p)
                if not {"true","mask","pred"}.issubset(z.files): raise RuntimeError(f"{p} keys={z.files}")
                y=np.asarray(z["true"],float); m=np.asarray(z["mask"]).astype(bool); pred=np.asarray(z["pred"],float); lead=np.arange(1,len(y)+1)
                for lo,hi,hb in HORIZONS:
                    r,n,ss=_rmse(y,pred,m&(lead>=lo)&(lead<=hi))
                    if n:
                        rec={"Battery":b,"L":L,"Role":role,"HorizonBin":hb,"RMSE":r,"N":n}; brow.append(rec); local.append(rec); sse[hb]+=ss; nn[hb]+=n
            loc=pd.DataFrame(local)
            for _,_,hb in HORIZONS:
                if nn[hb]:
                    mac=float(loc[loc.HorizonBin.eq(hb)].RMSE.mean())
                    prow.append({"L":L,"Role":role,"HorizonBin":hb,"PooledRMSE":float(np.sqrt(sse[hb]/nn[hb])),"BatteryMacroRMSE":mac,"N":nn[hb]})
    battery=pd.DataFrame(brow); pooled=pd.DataFrame(prow)
    save_csv(battery,out/"01_L_H_battery_metrics.csv"); save_csv(pooled,out/"02_L_H_pooled_metrics.csv")

    opt=pooled[pooled.Role.eq("BEST_SOURCE_PROXY")].rename(columns={"PooledRMSE":"OPT_PooledRMSE","BatteryMacroRMSE":"OPT_BatteryMacroRMSE"})
    uni=pooled[pooled.Role.eq("UNIFORM")].rename(columns={"PooledRMSE":"Uniform_PooledRMSE","BatteryMacroRMSE":"Uniform_BatteryMacroRMSE"})
    g=opt.merge(uni,on=["L","HorizonBin"],how="inner",suffixes=("_opt","_uni"))
    g["TimingGainPooledPct"]=100*(g.Uniform_PooledRMSE-g.OPT_PooledRMSE)/g.Uniform_PooledRMSE
    g["TimingGainMacroPct"]=100*(g.Uniform_BatteryMacroRMSE-g.OPT_BatteryMacroRMSE)/g.Uniform_BatteryMacroRMSE
    save_csv(g,out/"03_TimingGain_by_L_H.csv")

    br=[]
    for L in EVENT_LS:
        for _,_,hb in HORIZONS:
            q=battery[battery.L.eq(L)&battery.HorizonBin.eq(hb)].pivot(index="Battery",columns="Role",values="RMSE")
            if set(ROLES).issubset(q.columns):
                diff=q.BEST_SOURCE_PROXY-q.UNIFORM
                br.append({"L":L,"HorizonBin":hb,"Comparison":"OPT - Uniform",**paired_bootstrap(diff.to_numpy(),a.bootstrap,stable_seed(a.seed,f"LH|{L}|{hb}"))})
    boot=pd.DataFrame(br); save_csv(boot,out/"04_TimingGain_L_H_paired_bootstrap.csv")

    order=[h for _,_,h in HORIZONS]
    pv=opt.pivot(index="HorizonBin",columns="L",values="OPT_PooledRMSE").reindex(order).reindex(columns=list(EVENT_LS))
    fig,ax=plt.subplots(figsize=(8.2,5.6)); im=ax.imshow(pv.to_numpy()*100,aspect="auto")
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels([f"L{x}" for x in pv.columns]); ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
    ax.set_xlabel("Diagnosis count"); ax.set_ylabel("Forecast horizon"); ax.set_title("Event optimized timing: L-H pooled RMSE (%)"); fig.colorbar(im,ax=ax,label="RMSE (%)")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            if np.isfinite(pv.iloc[i,j]): ax.text(j,i,f"{pv.iloc[i,j]*100:.2f}",ha="center",va="center",fontsize=8)
    fig.tight_layout(); fig.savefig(out/"Figure_L_H_RMSE_Map.png",dpi=300); plt.close(fig)

    pvg=g.pivot(index="HorizonBin",columns="L",values="TimingGainPooledPct").reindex(order).reindex(columns=list(EVENT_LS))
    fig,ax=plt.subplots(figsize=(8.2,5.6)); im=ax.imshow(pvg.to_numpy(),aspect="auto")
    ax.set_xticks(range(len(pvg.columns))); ax.set_xticklabels([f"L{x}" for x in pvg.columns]); ax.set_yticks(range(len(pvg.index))); ax.set_yticklabels(pvg.index)
    ax.set_xlabel("Diagnosis count"); ax.set_ylabel("Forecast horizon"); ax.set_title("Timing gain over Uniform by L and horizon (%)"); fig.colorbar(im,ax=ax,label="RMSE reduction (%)")
    for i in range(pvg.shape[0]):
        for j in range(pvg.shape[1]):
            if np.isfinite(pvg.iloc[i,j]): ax.text(j,i,f"{pvg.iloc[i,j]:+.1f}",ha="center",va="center",fontsize=8)
    fig.tight_layout(); fig.savefig(out/"Figure_TimingGain_Horizon_Map.png",dpi=300); plt.close(fig)

    save_json({"Task":"L-H and TimingGain-H from frozen V4.14B predictions","CodeVersion":CODE_VERSION,"Status":"PASS","Batteries":47,"L":list(EVENT_LS),"NewTraining":False},out/"99_L_H_COMPLETE.json")
    print("\nL-H COMPLETE\n", g.to_string(index=False))


# -----------------------------------------------------------------------------
# D. Adaptive schedule discovery + Sankey-like flow
# -----------------------------------------------------------------------------
BATTERY_ALIASES=("Battery","battery","BatteryID","battery_id","TestBattery","HeldoutBattery")
SCHEDULE_ALIASES=("Schedule","schedule","SelectedSchedule","selected_schedule","AdaptiveSchedule","adaptive_schedule","FinalSchedule","DiagnosisSchedule","Path")
POLICY_ALIASES=("PolicyID","policy_id","Policy","policy","Threshold","threshold")


def parse_schedule(v)->Optional[Tuple[int,...]]:
    if v is None or (isinstance(v,float) and np.isnan(v)): return None
    nums=[int(x) for x in re.findall(r"\d+",str(v))]
    nums=[x for x in nums if 35<=x<=100]
    clean=[]
    for x in nums:
        if not clean or x!=clean[-1]: clean.append(x)
    if len(clean)<2 or any(clean[i]>=clean[i+1] for i in range(len(clean)-1)): return None
    return tuple(clean)


def discover_adaptive_schedules(a):
    files=[a.adaptive_schedule_file] if a.adaptive_schedule_file else []
    if not files and a.v415_root.exists():
        files=[p for p in a.v415_root.rglob("*.csv") if any(k in p.name.lower() for k in ("adaptive","schedule","policy","outer","fold"))]
        if not files: files=list(a.v415_root.rglob("*.csv"))
    best=None; source=""
    for p in files:
        try: d=pd.read_csv(p,low_memory=False)
        except Exception: continue
        bc=find_col(d.columns,BATTERY_ALIASES); sc=find_col(d.columns,SCHEDULE_ALIASES); pc=find_col(d.columns,POLICY_ALIASES)
        if bc is None or sc is None: continue
        q=d.copy()
        if pc is not None:
            mask=q[pc].astype(str).str.contains(r"0\.008|0p008",regex=True,case=False)
            if mask.any(): q=q[mask].copy()
        rows=[]
        for _,r in q.iterrows():
            s=parse_schedule(r[sc]); b=str(r[bc]).strip()
            if s and b: rows.append({"Battery":b,"Schedule":"|".join(map(str,s)),"DiagnosisCount":len(s)})
        x=pd.DataFrame(rows)
        if x.empty: continue
        x=x.drop_duplicates(["Battery","Schedule"])
        if best is None or x.Battery.nunique()>best.Battery.nunique(): best=x; source=str(p)
    if best is None: return pd.DataFrame(),""
    if best.groupby("Battery").size().max()>1:
        best=(best.groupby(["Battery","Schedule","DiagnosisCount"]).size().rename("Count").reset_index().sort_values(["Battery","Count"],ascending=[True,False]).drop_duplicates("Battery").drop(columns="Count"))
    return best.sort_values("Battery").reset_index(drop=True),source


def sankey_edges(schedules):
    e={}
    for r in schedules.itertuples(index=False):
        s=[f"C{x}" for x in map(int,str(r.Schedule).split("|"))]+["STOP"]
        for u,v in zip(s[:-1],s[1:]): e[(u,v)]=e.get((u,v),0)+1
    return pd.DataFrame([{"Source":u,"Target":v,"Count":n} for (u,v),n in sorted(e.items())])


def node_cycle(n):
    if n=="STOP": return 999
    m=re.search(r"\d+",n); return int(m.group()) if m else 0


def draw_sankey(edges,png):
    if edges.empty: return
    nodes=sorted(set(edges.Source).union(edges.Target),key=node_cycle); cycles=[node_cycle(n) for n in nodes if n!="STOP"]; lo,hi=min(cycles),max(cycles)
    xpos={n:(1.0 if n=="STOP" else 0.05+0.82*(node_cycle(n)-lo)/max(hi-lo,1)) for n in nodes}; ypos={n:0.5 for n in nodes}; mx=max(edges.Count.max(),1)
    fig,ax=plt.subplots(figsize=(11.5,6.0))
    for r in edges.itertuples(index=False):
        x0,y0=xpos[r.Source],ypos[r.Source]; x1,y1=xpos[r.Target],ypos[r.Target]
        path=MplPath([(x0,y0),(x0+(x1-x0)*.35,y0),(x0+(x1-x0)*.65,y1),(x1,y1)],[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4])
        ax.add_patch(PathPatch(path,fill=False,linewidth=1+8*r.Count/mx,alpha=.3)); ax.text((x0+x1)/2,(y0+y1)/2+0.035,str(int(r.Count)),ha="center",fontsize=8)
    for n in nodes:
        x,y=xpos[n],ypos[n]; ax.add_patch(Rectangle((x-.018,y-.03),.036,.06,alpha=.8)); ax.text(x,y+.055,n,ha="center",fontsize=9)
    ax.set_xlim(0,1.05); ax.set_ylim(.25,.75); ax.set_axis_off(); ax.set_title("Adaptive 2/3-diagnosis deployment paths"); fig.tight_layout(); fig.savefig(png,dpi=300); plt.close(fig)


def run_adaptive(a):
    out=ensure_dir(a.output_root/"03_Adaptive_Diagnosis_Flow")
    s,source=discover_adaptive_schedules(a)
    if s.empty:
        save_csv(pd.DataFrame([{"Status":"SKIPPED","Reason":"No battery-level adaptive schedule table discovered","RequestedPolicy":"PRT0.008"}]),out/"00_adaptive_discovery_audit.csv")
        print("\nAdaptive schedule discovery: SKIPPED"); return False
    save_csv(s,out/"01_adaptive_battery_schedules.csv"); e=sankey_edges(s); save_csv(e,out/"02_adaptive_sankey_edges.csv")
    dist=s.groupby(["DiagnosisCount","Schedule"]).size().rename("Batteries").reset_index().sort_values(["DiagnosisCount","Batteries"],ascending=[True,False]); save_csv(dist,out/"03_adaptive_schedule_distribution.csv")
    summary=pd.DataFrame([{"Source":source,"Batteries":s.Battery.nunique(),"MeanDiagnoses":s.DiagnosisCount.mean(),"TwoDiagnosisBatteries":int((s.DiagnosisCount==2).sum()),"ThreeDiagnosisBatteries":int((s.DiagnosisCount==3).sum()),"OtherDiagnosisBatteries":int((~s.DiagnosisCount.isin([2,3])).sum()),"FirstCycleMode":s.Schedule.str.split("|").str[0].mode().iloc[0]}]); save_csv(summary,out/"04_adaptive_summary.csv")
    draw_sankey(e,out/"Figure_Adaptive_Diagnosis_Flow.png")
    save_json({"Task":"Adaptive 2/3 diagnosis schedule flow","CodeVersion":CODE_VERSION,"Status":"PASS","Source":source,"Batteries":int(s.Battery.nunique()),"NewTraining":False},out/"99_ADAPTIVE_FLOW_COMPLETE.json")
    print("\nADAPTIVE FLOW COMPLETE\n",summary.to_string(index=False),"\n\n",dist.head(20).to_string(index=False)); return True


# -----------------------------------------------------------------------------
# E/F. SHAP + HI registry + Sparse Anchor optional
# -----------------------------------------------------------------------------

def result_roots(base):
    try: return [p for p in base.iterdir() if p.is_dir() and p.name.upper().startswith("RWTH48")]
    except Exception: return []


def detect_shap(p):
    try: d=pd.read_csv(p,low_memory=False)
    except Exception: return None
    fc=find_col(d.columns,("Feature","feature","HI","Variable","FeatureName")); vc=find_col(d.columns,("MeanAbsSHAP","mean_abs_shap","Importance","SHAPImportance","MeanAbsoluteSHAP"))
    if fc is None: return None
    if vc is None:
        for c in d.columns:
            if "shap" in str(c).lower() and pd.to_numeric(d[c],errors="coerce").notna().any(): vc=c; break
    if vc is None: return None
    x=pd.DataFrame({"Feature":d[fc].astype(str),"Importance":pd.to_numeric(d[vc],errors="coerce")}).dropna(); return x if not x.empty else None


def discover_shap(a):
    files=[a.shap_summary] if a.shap_summary else [p for r in result_roots(a.base) for p in r.rglob("*.csv") if "shap" in p.name.lower()]
    best=None; src=""; score=-1
    for p in files[:2000]:
        x=detect_shap(p)
        if x is None: continue
        sc=sum(h in set(x.Feature) for h in SELECTED_HIS)
        if sc>score: best=x; src=str(p); score=sc
    return (best if best is not None else pd.DataFrame()),src


def hi_registry():
    return pd.DataFrame([
        {"Layer":"Platform","Item":"Event","Definition":"cumulative relative-capacity prefix","FinalChoice":"Yes","PhysicalMeaning":"aligns diagnosis by electrochemical progress rather than absolute seconds"},
        {"Layer":"Event nodes","Item":"0.05|0.09|0.13","Definition":"cumulative 0-r prefix","FinalChoice":"Yes","PhysicalMeaning":"three early high-coverage information points"},
        {"Layer":"FixedTime nodes","Item":"60|120|180|300|450|600|900|1200 s","Definition":"cumulative 0-t prefix","FinalChoice":"Control","PhysicalMeaning":"absolute-time diagnostic information"},
        {"Layer":"HI","Item":"V_TimeWeightedMean","Definition":"time-weighted voltage mean","FinalChoice":"Yes","PhysicalMeaning":"integrated voltage/state information"},
        {"Layer":"HI","Item":"P_Last","Definition":"terminal instantaneous power","FinalChoice":"Yes","PhysicalMeaning":"local V-I operating response"},
        {"Layer":"Diagnosis count","Item":"L3","Definition":"3 diagnoses","FinalChoice":"Pareto knee","PhysicalMeaning":"resource-accuracy operating point"},
        {"Layer":"Resource","Item":"18 measured scalars","Definition":"3 diagnoses x 3 nodes x 2 HIs","FinalChoice":"Main method","PhysicalMeaning":"compact Event information budget"},
    ])


def detect_sparse(p):
    try: d=pd.read_csv(p,low_memory=False)
    except Exception: return None
    rc=find_col(d.columns,("RMSE","rmse","RMSE_SOH","SOH_RMSE")); kc=find_col(d.columns,("K","AnchorInterval","AnchorSpacing","SOHAnchorInterval","CalibrationInterval")); hc=find_col(d.columns,("H","Horizon","HorizonBin","ForecastHorizon","LeadBin"))
    if None in (rc,kc,hc): return None
    x=pd.DataFrame({"K":d[kc],"H":d[hc],"RMSE":pd.to_numeric(d[rc],errors="coerce")}).dropna(subset=["RMSE"]); return x if not x.empty else None


def sparse_plot(a,out):
    files=[a.sparse_anchor_file] if a.sparse_anchor_file else []
    if not files:
        for r in result_roots(a.base):
            if "sparse" in r.name.lower() or "anchor" in r.name.lower(): files.extend(r.rglob("*.csv"))
        if not files:
            for r in result_roots(a.base): files.extend([p for p in r.rglob("*.csv") if "anchor" in p.name.lower() or "sparse" in p.name.lower()])
    best=None; src=""
    for p in files[:1000]:
        x=detect_sparse(p)
        if x is not None and (best is None or len(x)>len(best)): best=x; src=str(p)
    if best is None: return {"Artifact":"Sparse SOH Anchor supplementary map","Status":"SKIPPED","Source":"","Reason":"No compatible K/H/RMSE table discovered"}
    save_csv(best,out/"04_SparseAnchor_K_H_discovered.csv"); pv=best.pivot_table(index="H",columns="K",values="RMSE",aggfunc="mean")
    fig,ax=plt.subplots(figsize=(8,5.4)); im=ax.imshow(pv.to_numpy()*100,aspect="auto"); ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns,rotation=45,ha="right"); ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index); ax.set_xlabel("Sparse SOH anchor spacing / K"); ax.set_ylabel("Forecast horizon / H"); ax.set_title("Sparse SOH Anchor supplementary K-H map"); fig.colorbar(im,ax=ax,label="RMSE (%)"); fig.tight_layout(); fig.savefig(out/"Figure_SparseAnchor_K_H_Supplementary.png",dpi=300); plt.close(fig)
    return {"Artifact":"Sparse SOH Anchor supplementary map","Status":"PASS","Source":src,"Reason":""}


def run_interpret(a):
    out=ensure_dir(a.output_root/"04_Interpretability_and_SparseAnchor"); save_csv(hi_registry(),out/"01_HI_Node_Platform_Design_Registry.csv")
    audit=[]; shap,src=discover_shap(a)
    if shap.empty:
        audit.append({"Artifact":"SHAP compact-HI figure","Status":"SKIPPED","Source":"","Reason":"No compatible SHAP summary CSV discovered"})
    else:
        s=shap.groupby("Feature",as_index=False).Importance.mean().sort_values("Importance",ascending=False); save_csv(s,out/"02_SHAP_summary_discovered.csv"); keep=s[s.Feature.isin(["Lead","lead","Horizon","ForecastLead","V_TimeWeightedMean","P_Last"])]
        q=(keep if len(keep)>=2 else s.head(10)).sort_values("Importance"); fig,ax=plt.subplots(figsize=(7.3,4.8)); ax.barh(q.Feature,q.Importance); ax.set_xlabel("Mean absolute SHAP importance"); ax.set_title("Final predictor interpretability"); fig.tight_layout(); fig.savefig(out/"Figure_SHAP_Final_CompactHI.png",dpi=300); plt.close(fig); audit.append({"Artifact":"SHAP compact-HI figure","Status":"PASS","Source":src,"Reason":""})
    audit.append(sparse_plot(a,out)); save_csv(pd.DataFrame(audit),out/"03_interpretability_sparse_audit.csv"); save_json({"Task":"Final interpretability and Sparse Anchor supplementary","CodeVersion":CODE_VERSION,"Status":"PASS","OptionalArtifacts":audit,"NewTraining":False},out/"99_INTERPRETABILITY_COMPLETE.json"); print("\nINTERPRETABILITY COMPLETE\n",pd.DataFrame(audit).to_string(index=False))


# -----------------------------------------------------------------------------
# Final index
# -----------------------------------------------------------------------------

def write_index(a,status):
    idx=pd.DataFrame([
        {"Block":"RUL85","Status":status.get("RUL85","NOT_RUN"),"PaperRole":"Endpoint extension","NewTraining":"No"},
        {"Block":"L-H / TimingGain-H","Status":status.get("LH","NOT_RUN"),"PaperRole":"Main resource-timing result","NewTraining":"No"},
        {"Block":"Adaptive 2/3 flow","Status":status.get("ADAPTIVE","NOT_RUN"),"PaperRole":"Main resource-saving extension","NewTraining":"No"},
        {"Block":"SHAP / HI design","Status":status.get("INTERPRET","NOT_RUN"),"PaperRole":"Interpretability","NewTraining":"No"},
        {"Block":"Sparse SOH Anchor","Status":status.get("INTERPRET","NOT_RUN"),"PaperRole":"Supplementary optional calibration","NewTraining":"No"},
    ]); save_csv(idx,a.output_root/"00_final_figure_statistics_index.csv"); save_json({"Task":"RWTH48 V4.23 Final Figure & Statistics Closure","CodeVersion":CODE_VERSION,"Status":"PASS","StageStatus":status,"NoNewModelTraining":True,"NoNewTimingSearch":True},a.output_root/"99_RWTH48_V4_23_COMPLETE.json")


def main():
    a=parse_args(); ensure_dir(a.output_root); pf=audit(a); save_csv(pf,a.output_root/"00_preflight.csv")
    if a.stage=="audit":
        print(pf.to_string(index=False)); mandatory={"Final derived-RUL battery table","V4.14B root","V4.14B Event L2-L6 prediction NPZ files"}; bad=pf[pf.Check.isin(mandatory)&pf.PASS.eq(0)]
        if len(bad): raise RuntimeError("Mandatory audit failure:\n"+bad.to_string(index=False))
        print("\nAUDIT PASS for mandatory modules."); return
    st={}
    if a.stage in ("rul85","all"): run_rul85(a); st["RUL85"]="PASS"
    if a.stage in ("lh","all"): run_lh(a); st["LH"]="PASS"
    if a.stage in ("adaptive","all"): st["ADAPTIVE"]="PASS" if run_adaptive(a) else "SKIPPED"
    if a.stage in ("interpret","all"): run_interpret(a); st["INTERPRET"]="PASS"
    write_index(a,st); print("\n"+"="*100+"\nV4.23 COMPLETE\nOutput:",a.output_root,"\n"+"="*100)


if __name__=="__main__":
    main()

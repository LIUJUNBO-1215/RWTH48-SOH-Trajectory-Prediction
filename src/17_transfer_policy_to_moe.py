# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, os, sys, time, traceback, re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

CODE_VERSION = "V426_ADAPTIVE_SAFE_MOE_POLICY_TRANSFER_FIX1"
BASE_DEFAULT = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR = Path(__file__).resolve().parent
V417_SCRIPT_NAME = "13_safe_residual_moe.py"
V414B_SCRIPT_NAME = "10_schedule_and_resource_search.py"
COMMON_SCRIPT_NAME = "08_fixedtime_common_pipeline.py"
V415_ROOT_NAME = "RWTH48_V4_15_BudgetAware_Adaptive_Diagnostic_Scheduling"
V417_ROOT_NAME = "RWTH48_V4_17_Safeguarded_Residual_MoE"
V424_ROOT_NAME = "RWTH48_V4_24_Adaptive_Primary_Finalization"
V414B_ROOT_NAME = "RWTH48_V4_14B_FullLifeSOH_Timing_MinResource"
OUTPUT_NAME = "RWTH48_V4_26_Adaptive_SafeResidualMoE_Policy_Transfer"
STATE_NAME = "RWTH48_V4_26_Adaptive_SafeResidualMoE_Policy_Transfer_STATE"
EXPECTED_BATTERIES = 47
TEST_ORIGIN = 80
PRIMARY_POLICY = 0.008
POLICY_ID = "PRT_0p0080"
EVENT_NODES = (0.05, 0.09, 0.13)
EVENT_HIS = ("V_TimeWeightedMean", "P_Last")
SCALARS_PER_DIAGNOSIS = len(EVENT_NODES) * len(EVENT_HIS)
HORIZON_BINS = ((1,100,"Lead001_100"),(101,300,"Lead101_300"),(301,500,"Lead301_500"),(501,700,"Lead501_700"),(701,1000,"Lead701_1000"))
ENC = "utf-8-sig"
_ACTIVE_DIAGNOSIS_COUNT = 3
_V417_MODULE = None


def parse_args():
    p=argparse.ArgumentParser(description="RWTH48 V4.26 Adaptive 2/3 -> SafeResidualMoE policy-transfer experiment")
    p.add_argument("--stage",choices=["audit","run","all","finalize"],default="all")
    p.add_argument("--base",type=Path,default=BASE_DEFAULT); p.add_argument("--code-dir",type=Path,default=None)
    p.add_argument("--v417-script",type=Path,default=None); p.add_argument("--v414b-script",type=Path,default=None); p.add_argument("--common-script",type=Path,default=None)
    p.add_argument("--v415-root",type=Path,default=None); p.add_argument("--v417-root",type=Path,default=None); p.add_argument("--v424-root",type=Path,default=None); p.add_argument("--v414b-root",type=Path,default=None)
    p.add_argument("--adaptive-schedule-file",type=Path,default=None); p.add_argument("--output-root",type=Path,default=None); p.add_argument("--state-root",type=Path,default=None)
    p.add_argument("--event-ready",type=Path,default=None); p.add_argument("--fixedtime-ready",type=Path,default=None); p.add_argument("--dependency-registry",type=Path,default=None)
    p.add_argument("--source-origins",nargs="+",type=int,default=[80,120,160]); p.add_argument("--xgb-threads",type=int,default=3); p.add_argument("--xgb-leads-per-origin",type=int,default=160)
    p.add_argument("--deep-validation-batteries",type=int,default=7); p.add_argument("--deep-max-epochs",type=int,default=60); p.add_argument("--deep-patience",type=int,default=8); p.add_argument("--deep-batch",type=int,default=12); p.add_argument("--deep-dropout",type=float,default=0.12); p.add_argument("--deep-lr",type=float,default=8e-4); p.add_argument("--deep-clipnorm",type=float,default=1.0)
    p.add_argument("--gate-validation-batteries",type=int,default=7); p.add_argument("--residual-cv-folds",type=int,default=3); p.add_argument("--residual-clip-quantile",type=float,default=0.995); p.add_argument("--residual-clip-min",type=float,default=0.005); p.add_argument("--residual-clip-max",type=float,default=0.08)
    p.add_argument("--guard-grid",nargs="+",type=int,default=[300,400,500,600,700]); p.add_argument("--gate-c-grid",nargs="+",type=float,default=[300,400,500,600,700,800]); p.add_argument("--gate-tau-grid",nargs="+",type=float,default=[50,100,150,200]); p.add_argument("--alpha-grid",nargs="+",type=float,default=[0.25,0.50,0.75,1.00])
    p.add_argument("--no-harm-horizons",nargs="+",type=int,default=[100,300,500]); p.add_argument("--no-harm-epsilon",type=float,default=0.00010); p.add_argument("--overall-no-harm-epsilon",type=float,default=0.00010); p.add_argument("--far-objective-start",type=int,default=701)
    p.add_argument("--cycle-start",type=int,default=40); p.add_argument("--cycle-end",type=int,default=80); p.add_argument("--cycle-step",type=int,default=4); p.add_argument("--proxy-validation-batteries",type=int,default=9); p.add_argument("--proxy-ridge-alpha",type=float,default=8.0); p.add_argument("--proxy-leads",nargs="+",type=int,default=[10,25,50,100,200,300,500,700,900]); p.add_argument("--local-refine-radius",type=int,default=3); p.add_argument("--local-refine-rounds",type=int,default=2); p.add_argument("--hi-ranking-shortlist",type=int,default=12)
    p.add_argument("--bootstrap",type=int,default=10000); p.add_argument("--cpu-threads-per-worker",type=int,default=4); p.add_argument("--parallel-workers",type=int,default=1); p.add_argument("--parallel-worker-id",type=int,default=0); p.add_argument("--only-battery",type=str,default=""); p.add_argument("--force",action="store_true")
    p.add_argument("--run-id",type=str,default="V426_ADAPTIVE_SAFE_MOE_20260828_FIX1"); p.add_argument("--barrier-poll-s",type=float,default=15.0); p.add_argument("--barrier-timeout-hours",type=float,default=24.0); p.add_argument("--auto-finalize",action="store_true")
    a=p.parse_args(); a.base=Path(a.base); a.code_dir=Path(a.code_dir) if a.code_dir else CODE_DIR
    a.v417_script=Path(a.v417_script) if a.v417_script else a.code_dir/V417_SCRIPT_NAME; a.v414b_script=Path(a.v414b_script) if a.v414b_script else a.code_dir/V414B_SCRIPT_NAME; a.common_script=Path(a.common_script) if a.common_script else a.code_dir/COMMON_SCRIPT_NAME
    a.v415_root=Path(a.v415_root) if a.v415_root else a.base/V415_ROOT_NAME; a.v417_root=Path(a.v417_root) if a.v417_root else a.base/V417_ROOT_NAME; a.v424_root=Path(a.v424_root) if a.v424_root else a.base/V424_ROOT_NAME; a.v414b_root=Path(a.v414b_root) if a.v414b_root else a.base/V414B_ROOT_NAME
    a.output_root=Path(a.output_root) if a.output_root else a.base/OUTPUT_NAME; a.state_root=Path(a.state_root) if a.state_root else a.base/STATE_NAME
    if a.adaptive_schedule_file: a.adaptive_schedule_file=Path(a.adaptive_schedule_file)
    if a.parallel_workers<1: p.error("--parallel-workers must be >=1")
    if not 0<=a.parallel_worker_id<a.parallel_workers: p.error("--parallel-worker-id must be in [0, parallel-workers)")
    if a.bootstrap<1000: p.error("--bootstrap must be >=1000")
    return a


def configure_threads(a):
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[k]=str(max(1,int(a.cpu_threads_per_worker)))
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH","true"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")

def ensure_dir(p): p=Path(p); p.mkdir(parents=True,exist_ok=True); return p
def save_csv(df,p): ensure_dir(Path(p).parent); df.to_csv(p,index=False,encoding=ENC)
def save_json(obj,p):
    ensure_dir(Path(p).parent)
    def cv(x):
        if isinstance(x,Path): return str(x)
        if isinstance(x,np.integer): return int(x)
        if isinstance(x,np.floating): return float(x)
        if isinstance(x,np.ndarray): return x.tolist()
        raise TypeError(type(x))
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=cv),encoding="utf-8")
def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
def stable_seed(*items): return int(hashlib.sha256("|".join(map(str,items)).encode()).hexdigest()[:8],16)%(2**31-1)
def schedule_text(s): return "|".join(map(str,s))
def schedule_id(s): return "_".join(map(str,s))
def normalize_battery(x):
    s=str(x).strip().upper().replace("_","-"); m=re.search(r"RWTH[- ]?0*(\d+)",s); return f"RWTH-{int(m.group(1)):03d}" if m else s


def discover_schedule_file(a):
    if a.adaptive_schedule_file is not None:
        if not a.adaptive_schedule_file.exists(): raise FileNotFoundError(a.adaptive_schedule_file)
        return a.adaptive_schedule_file
    p=a.v424_root/"01_Adaptive_Primary"/"02_adaptive_battery_schedules.csv"
    if p.exists(): return p
    cand=[]
    if a.v424_root.exists():
        for q in a.v424_root.rglob("*.csv"):
            n=q.name.lower()
            if "adaptive" in n and "schedule" in n and "distribution" not in n and "candidate" not in n: cand.append(q)
    if cand: return sorted(cand,key=lambda x:(len(x.parts),str(x)))[0]
    raise FileNotFoundError("Cannot locate V4.24 battery-level adaptive schedule table. Pass --adaptive-schedule-file.")

def parse_schedule_value(x):
    vals=[int(v) for v in re.findall(r"\d+",str(x))]
    if len(vals) not in (2,3): raise ValueError(f"Adaptive schedule must contain 2 or 3 diagnoses: {x}")
    if any(b<=a for a,b in zip(vals[:-1],vals[1:])): raise ValueError(vals)
    if max(vals)>TEST_ORIGIN: raise ValueError(vals)
    return tuple(vals)

def load_adaptive_schedules(a):
    p=discover_schedule_file(a); d=pd.read_csv(p,low_memory=False)
    bc=next((c for c in d.columns if str(c).lower() in ("battery","outerheldoutbattery","batteryid","battery_id")),None)
    sc=next((c for c in d.columns if str(c).lower() in ("schedule","adaptiveschedule","selectedschedule","diagnosiscycles")),None)
    if bc is None or sc is None: raise RuntimeError(f"Missing Battery/Schedule columns in {p}: {list(d.columns)}")
    out=pd.DataFrame({"Battery":d[bc].map(normalize_battery),"Schedule":d[sc].map(lambda x:schedule_text(parse_schedule_value(x)))})
    out["DiagnosisCount"]=out.Schedule.str.split("|").str.len().astype(int); out=out.drop_duplicates("Battery").sort_values("Battery").reset_index(drop=True)
    if "RWTH-028" in set(out.Battery): out=out[out.Battery!="RWTH-028"].copy()
    if out.Battery.nunique()!=47: raise RuntimeError(f"Adaptive schedule batteries={out.Battery.nunique()} !=47")
    if not out.DiagnosisCount.isin([2,3]).all(): raise RuntimeError("Primary policy has non-L2/L3 schedules")
    return out,p


def make_v417_args(a):
    return argparse.Namespace(stage="audit",base=a.base,code_dir=a.code_dir,v414b=a.v414b_script,common=a.common_script,v414b_root=a.v414b_root,output_root=a.v417_root,state_root=a.state_root/"_V417_HELPER_STATE",event_ready=a.event_ready,fixedtime_ready=a.fixedtime_ready,dependency_registry=a.dependency_registry,source_origins=list(a.source_origins),xgb_threads=a.xgb_threads,xgb_leads_per_origin=a.xgb_leads_per_origin,deep_validation_batteries=a.deep_validation_batteries,deep_max_epochs=a.deep_max_epochs,deep_patience=a.deep_patience,deep_batch=a.deep_batch,deep_dropout=a.deep_dropout,deep_lr=a.deep_lr,deep_clipnorm=a.deep_clipnorm,gate_validation_batteries=a.gate_validation_batteries,residual_cv_folds=a.residual_cv_folds,residual_clip_quantile=a.residual_clip_quantile,residual_clip_min=a.residual_clip_min,residual_clip_max=a.residual_clip_max,guard_grid=list(a.guard_grid),gate_c_grid=list(a.gate_c_grid),gate_tau_grid=list(a.gate_tau_grid),alpha_grid=list(a.alpha_grid),no_harm_horizons=list(a.no_harm_horizons),no_harm_epsilon=a.no_harm_epsilon,overall_no_harm_epsilon=a.overall_no_harm_epsilon,far_objective_start=a.far_objective_start,bootstrap=a.bootstrap,only_battery="",force=False,cpu_threads_per_worker=a.cpu_threads_per_worker,parallel_workers=1,parallel_worker_id=0,run_id=a.run_id+"|V417_HELPER",barrier_poll_s=a.barrier_poll_s,cycle_start=a.cycle_start,cycle_end=a.cycle_end,cycle_step=a.cycle_step,proxy_validation_batteries=a.proxy_validation_batteries,proxy_ridge_alpha=a.proxy_ridge_alpha,proxy_leads=list(a.proxy_leads),local_refine_radius=a.local_refine_radius,local_refine_rounds=a.local_refine_rounds,hi_ranking_shortlist=a.hi_ranking_shortlist)


def build_td_model_variable(input_dim,arch,k,dropout,lr,clipnorm):
    global _ACTIVE_DIAGNOSIS_COUNT,_V417_MODULE
    import tensorflow as tf
    D=int(_ACTIVE_DIAGNOSIS_COUNT)
    if D not in (2,3): raise RuntimeError(D)
    tf.keras.backend.clear_session(); inp=tf.keras.Input(shape=(D,int(input_dim)),name=f"Event_L{D}_history")
    if str(arch).upper()!="LSTM": raise ValueError(arch)
    z=tf.keras.layers.LSTM(48,return_sequences=False,name="encoder_lstm48")(inp); z=tf.keras.layers.Dense(64,activation="swish",name="encoder_dense64")(z); z=tf.keras.layers.Dropout(float(dropout),name="encoder_dropout")(z)
    t=tf.keras.layers.Dense(96,activation="swish",name="trend_dense96")(z); t=tf.keras.layers.Dropout(float(dropout))(t); trend=tf.keras.layers.Dense(int(k),activation="linear",name="trend_raw")(t)
    d=tf.keras.layers.Dense(96,activation="swish",name="detail_dense96")(z); d=tf.keras.layers.Dropout(float(dropout))(d); detail=tf.keras.layers.Dense(int(k),activation="linear",name="detail_raw")(d)
    gate=tf.keras.layers.Dense(1,activation="sigmoid",bias_initializer=tf.keras.initializers.Constant(-1.5),name="detail_gate")(z); out=tf.keras.layers.Add(name="SOH_trajectory")([trend,tf.keras.layers.Multiply()([detail,gate])])
    model=tf.keras.Model(inp,out,name=f"Event_L{D}_{arch}_TrendDetailResidual"); model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr),clipnorm=float(clipnorm)),loss=_V417_MODULE.ctx_masked_loss()); return model


def build_representation_variable(ctx,base_schedule):
    V=ctx["V417"]; B=ctx["B"]; bundle=ctx["bundle"]; cube=ctx["event_cube_ext"]; base_schedule=tuple(map(int,base_schedule)); D=len(base_schedule)
    if D not in (2,3): raise RuntimeError(D)
    nrow=len(bundle.battery); Xeng_all=None; Xseq_all=np.full((nrow,D,8),np.nan,np.float32); available=np.zeros(nrow,np.uint8); bmap={b:i for i,b in enumerate(ctx["batteries"])}
    for origin in ctx["retained_origins"]:
        sch=V.translated_schedule(base_schedule,origin); x=B.extract_tensor(cube,sch,EVENT_NODES,EVENT_HIS)
        if x is None: raise RuntimeError(f"representation unavailable schedule={sch}")
        Xeng47,_,_=B.engineer_dynamic_selected(x,sch,EVENT_NODES,EVENT_HIS)
        if Xeng_all is None: Xeng_all=np.full((nrow,Xeng47.shape[1]),np.nan,np.float32)
        raw=x.reshape(len(ctx["batteries"]),D,SCALARS_PER_DIAGNOSIS).astype(np.float32); rel=((np.asarray(sch,float)-float(origin))/40.0).astype(np.float32); gap=np.zeros(D,np.float32); gap[1:]=(np.diff(np.asarray(sch,float))/40.0).astype(np.float32); meta=np.column_stack([rel,gap]).astype(np.float32); seq47=np.concatenate([raw,np.repeat(meta[None,:,:],len(ctx["batteries"]),axis=0)],axis=2)
        for b in ctx["batteries"]:
            key=(b,int(origin))
            if key not in ctx["target_index"]: continue
            ti=ctx["target_index"][key]; bi=bmap[b]; Xeng_all[ti]=Xeng47[bi]; Xseq_all[ti]=seq47[bi]; available[ti]=1
    selected=V.sample_indices(ctx,ctx["batteries"],ctx["retained_origins"])
    if Xeng_all is None or not np.isfinite(Xeng_all[selected]).all() or not np.isfinite(Xseq_all[selected]).all(): raise RuntimeError("non-finite variable representation")
    return Xeng_all,Xseq_all,available


def v415_complete_path(a): return a.v415_root/"99_RWTH48_V4_15_COMPLETE.json"
def v417_complete_path(a): return a.v417_root/"99_RWTH48_V4_17_COMPLETE.json"
def v415_schedule_npz(a,b,s): return a.v415_root/"folds"/b/"formal"/f"schedule_{schedule_id(s)}.npz"
def v415_policy_formal_json(a,b): return a.v415_root/"folds"/b/"formal"/f"{POLICY_ID}_formal.json"

def load_exact_v415_xgb(ctx,a,held,schedule):
    p=v415_schedule_npz(a,held,schedule)
    if not p.exists(): raise FileNotFoundError(p)
    z=np.load(p); pred=np.asarray(z["pred"],np.float32); true=np.asarray(z["true"],np.float32); mask=np.asarray(z["mask"],np.uint8); te=ctx["target_index"][(held,TEST_ORIGIN)]; y=np.asarray(ctx["bundle"].y[te],np.float32); m=np.asarray(ctx["bundle"].mask[te],np.uint8)
    teq=np.allclose(true,y,equal_nan=True,atol=1e-7,rtol=0); meq=np.array_equal(mask,m)
    if not teq or not meq: raise RuntimeError(f"{held}: V4.15 truth/mask mismatch")
    fj=v415_policy_formal_json(a,held); pse=None
    if fj.exists():
        q=load_json(fj); qs=q.get("AdaptiveSchedule")
        if qs is not None:
            pse=int(tuple(map(int,qs))==tuple(map(int,schedule)))
            if not pse: raise RuntimeError(f"{held}: V4.15 policy schedule mismatch")
    return pred,y,m,{"V415FrozenPredictionFile":str(p),"FrozenTruthExact":int(teq),"FrozenMaskExact":int(meq),"PolicyScheduleExact":pse}


def preflight(a):
    global _V417_MODULE
    req=[a.v417_script,a.v414b_script,a.common_script,a.v415_root,a.v417_root,a.v414b_root,v415_complete_path(a),v417_complete_path(a)]
    for p in req:
        if not Path(p).exists(): raise FileNotFoundError(p)
    c15=load_json(v415_complete_path(a)); c17=load_json(v417_complete_path(a))
    if c15.get("Status")!="PASS" or c17.get("Status")!="PASS": raise RuntimeError("V4.15/V4.17 upstream status not PASS")
    schedules,source=load_adaptive_schedules(a); V=load_module(a.v417_script,"rwth48_v426_v417_helper"); _V417_MODULE=V; V.configure_threads(a.cpu_threads_per_worker); tf,attempts,err=V.import_tensorflow_with_retry(attempts=8,delay_s=4.0)
    if tf is None: raise RuntimeError(f"TensorFlow import failed: {err}")
    V.build_td_model=build_td_model_variable; vargs=make_v417_args(a); ctx=V.preflight(vargs); ctx["V417"]=V; ctx["vargs"]=vargs
    bats=list(map(normalize_battery,ctx["batteries"])); smap=dict(zip(schedules.Battery,schedules.Schedule)); rows=[]
    if len(bats)!=47: raise RuntimeError(f"batteries={len(bats)}")
    for stext in sorted(set(schedules.Schedule)):
        sch=tuple(map(int,stext.split("|"))); ok=1; detail=""
        try: build_representation_variable(ctx,sch)
        except Exception as e: ok=0; detail=repr(e)
        rows.append({"CheckType":"UniqueScheduleRepresentation","Battery":"","Schedule":stext,"Observed":ok,"Expected":1,"PASS":ok,"Detail":detail})
    for b in bats:
        if b not in smap: raise RuntimeError(f"missing schedule {b}")
        sch=tuple(map(int,smap[b].split("|"))); p=v415_schedule_npz(a,b,sch); rows.append({"CheckType":"ExactV415FrozenXGB","Battery":b,"Schedule":schedule_text(sch),"Observed":int(p.exists()),"Expected":1,"PASS":int(p.exists()),"Detail":str(p)})
    audit=pd.DataFrame(rows); ctx["adaptive_schedules"]=schedules; ctx["adaptive_schedule_map"]=smap; ctx["schedule_source"]=source; ctx["preflight_v426"]=audit
    if not audit.PASS.eq(1).all(): raise RuntimeError("preflight failed\n"+audit[audit.PASS.eq(0)].to_string(index=False))
    return ctx


def fold_dir(a,b): return ensure_dir(a.output_root/"folds"/b)
def fold_result_json(a,b): return fold_dir(a,b)/"result.json"
def fold_pred_npz(a,b): return fold_dir(a,b)/"predictions.npz"
def trajectory_stats(y,p,m):
    y=np.asarray(y,float); p=np.asarray(p,float); m=np.asarray(m).astype(bool); ok=m & np.isfinite(y) & np.isfinite(p)
    if not ok.any(): return {"RMSE":np.nan,"MAE":np.nan,"Bias":np.nan,"SSE":np.nan,"SAE":np.nan,"N":0}
    e=p[ok]-y[ok]; return {"RMSE":float(np.sqrt(np.mean(e*e))),"MAE":float(np.mean(np.abs(e))),"Bias":float(np.mean(e)),"SSE":float(np.sum(e*e)),"SAE":float(np.sum(np.abs(e))),"N":int(ok.sum())}
def source_batteries(ctx,held): return [b for b in ctx["batteries"] if normalize_battery(b)!=normalize_battery(held)]


def run_one(ctx,a,held):
    global _ACTIVE_DIAGNOSIS_COUNT
    import tensorflow as tf
    jp=fold_result_json(a,held); pp=fold_pred_npz(a,held)
    if jp.exists() and pp.exists() and not a.force:
        q=load_json(jp)
        if q.get("CodeVersion")==CODE_VERSION: print(f"[V426] {held}: cache",flush=True); return
    schedule=tuple(map(int,ctx["adaptive_schedule_map"][normalize_battery(held)].split("|"))); D=len(schedule); _ACTIVE_DIAGNOSIS_COUNT=D; t0=time.perf_counter(); Xeng,Xseq,_=build_representation_variable(ctx,schedule); source=source_batteries(ctx,held); p_xgb,y,mask,xgb_audit=load_exact_v415_xgb(ctx,a,held,schedule); V=ctx["V417"]; vargs=ctx["vargs"]
    gate=V.gate_validation(ctx,Xeng,Xseq,held,vargs,f"{held}|ADAPTIVE_PRT0.008|L{D}")
    if float(gate["alpha"])==0.0:
        p_resid=np.zeros_like(p_xgb,np.float32); p_safe=p_xgb.copy(); correction=np.zeros_like(p_xgb,np.float32); w=np.zeros_like(p_xgb,np.float32); rinfo={"SkippedFinalResidualTD":True,"Reason":"alpha=0 fallback"}
    else:
        residual_td,rxsc,rysc,rinfo=V.fit_residual_td_source_only(ctx,Xeng,Xseq,source,vargs,f"{held}|ADAPTIVE_PRT0.008|L{D}|FINAL_RESIDUAL_TD"); te=ctx["target_index"][(held,TEST_ORIGIN)]; p_resid=V.predict_residual(ctx,residual_td,rxsc,rysc,Xseq[te],rinfo["ResidualClip"]); w=V.guarded_residual_weights(len(p_xgb),int(gate["Guard"]),float(gate["c"]),float(gate["tau"])).astype(np.float32); correction=(float(gate["alpha"])*w*p_resid).astype(np.float32); p_safe=(p_xgb+correction).astype(np.float32); del residual_td; tf.keras.backend.clear_session(); gc.collect()
    guard=int(gate["Guard"]); inside=np.arange(1,len(p_xgb)+1)<=guard; maxdiff=float(np.max(np.abs(p_safe[inside]-p_xgb[inside])) if inside.any() else 0.0)
    if maxdiff>1e-8: raise RuntimeError(f"{held}: hard guard violated {maxdiff}")
    xmet=trajectory_stats(y,p_xgb,mask); smet=trajectory_stats(y,p_safe,mask); duration=np.nan; fj=v415_policy_formal_json(a,held)
    if fj.exists(): duration=load_json(fj).get("AcquisitionDuration_s",np.nan)
    q={"CodeVersion":CODE_VERSION,"OuterHeldoutBattery":held,"PolicyID":POLICY_ID,"AdaptiveSchedule":list(schedule),"DiagnosisCount":D,"EventNodes":list(EVENT_NODES),"SelectedHIs":list(EVENT_HIS),"HI_ScalarInputs":D*SCALARS_PER_DIAGNOSIS,"PhysicalSensorCount":2,"AcquisitionDuration_s":duration,"AdaptiveScheduleSource":str(ctx["schedule_source"]),"FrozenAdaptiveXGBAudit":xgb_audit,"ResidualTDInfo":rinfo,"SafeguardGate":gate,"SelectedGuard":guard,"SelectedAlpha":float(gate["alpha"]),"SelectedC":float(gate["c"]),"SelectedTau":float(gate["tau"]),"FallbackSelected":int(gate["FallbackSelected"]),"MaxAbsDiffInsideGuard":maxdiff,"Metrics":{"Adaptive_XGB":xmet,"Adaptive_SafeResidualMoE":smet},"Runtime_s":float(time.perf_counter()-t0),"Protocol":{"OuterLOBO":True,"AdaptivePolicyFrozen":True,"AdaptiveTimingReoptimizedForSafeMoE":False,"SameAdaptiveScheduleXGBvsSafeMoE":True,"V415FrozenAdaptiveXGBPredictionReusedExactly":True,"SourceResidualTargetsBatteryCrossFitted":True,"SafeguardSelectedSourceOnly":True,"HeldoutUsedForTraining":False,"HeldoutUsedForSafeguardSelection":False,"HistoricalSOHAsInput":False,"FutureHIAsInput":False,"AbsoluteCycleAsModelFeature":False,"HardGuardExactXGBInsideGuard":True}}
    np.savez_compressed(pp,true=np.asarray(y,np.float32),mask=np.asarray(mask,np.uint8),pred_Adaptive_XGB=np.asarray(p_xgb,np.float32),pred_Adaptive_SafeResidualMoE=np.asarray(p_safe,np.float32),pred_ResidualTD=np.asarray(p_resid,np.float32),residual_weight=np.asarray(w,np.float32),residual_correction=np.asarray(correction,np.float32)); save_json(q,jp)
    print(f"[V426] {held} L{D} {schedule} XGB={xmet['RMSE']:.6f} Safe={smet['RMSE']:.6f} guard={guard} alpha={gate['alpha']:.2f} runtime={q['Runtime_s']/60:.1f}m",flush=True)


def state_dir(a): return ensure_dir(a.state_root/a.run_id)
def done_marker(a,w): return state_dir(a)/f"worker_{w:02d}.done.json"
def fail_marker(a,w): return state_dir(a)/f"worker_{w:02d}.fail.json"
def worker_batteries(ctx,a):
    bats=sorted(map(normalize_battery,ctx["batteries"]))
    if a.only_battery:
        b=normalize_battery(a.only_battery)
        if b not in bats: raise RuntimeError(b)
        return [b]
    return [b for i,b in enumerate(bats) if i%a.parallel_workers==a.parallel_worker_id]
def run_worker(ctx,a):
    bats=worker_batteries(ctx,a); done_marker(a,a.parallel_worker_id).unlink(missing_ok=True); fail_marker(a,a.parallel_worker_id).unlink(missing_ok=True)
    try:
        for i,b in enumerate(bats,1): print(f"\n[V426] worker {a.parallel_worker_id} {i}/{len(bats)} {b}",flush=True); run_one(ctx,a,b)
        save_json({"CodeVersion":CODE_VERSION,"Worker":a.parallel_worker_id,"Status":"PASS","Batteries":bats},done_marker(a,a.parallel_worker_id))
    except Exception as e:
        save_json({"CodeVersion":CODE_VERSION,"Worker":a.parallel_worker_id,"Status":"FAIL","Error":repr(e),"Traceback":traceback.format_exc()},fail_marker(a,a.parallel_worker_id)); raise
def wait_for_workers(a):
    deadline=time.time()+a.barrier_timeout_hours*3600
    while True:
        fails=[fail_marker(a,w) for w in range(a.parallel_workers) if fail_marker(a,w).exists()]
        if fails: raise RuntimeError("worker failed\n"+"\n".join(map(str,fails)))
        done=[done_marker(a,w).exists() for w in range(a.parallel_workers)]
        if all(done): return
        if time.time()>deadline: raise TimeoutError("worker barrier timeout")
        print(f"[V426] waiting {sum(done)}/{a.parallel_workers}",flush=True); time.sleep(max(2,a.barrier_poll_s))


def collect_outer(ctx,a):
    rows=[]
    for b in sorted(map(normalize_battery,ctx["batteries"])):
        p=fold_result_json(a,b)
        if not p.exists(): continue
        q=load_json(p)
        for model in ("Adaptive_XGB","Adaptive_SafeResidualMoE"):
            m=q["Metrics"][model]; rows.append({"OuterHeldoutBattery":b,"Model":model,"Schedule":schedule_text(q["AdaptiveSchedule"]),"DiagnosisCount":q["DiagnosisCount"],"HI_ScalarInputs":q["HI_ScalarInputs"],"AcquisitionDuration_s":q.get("AcquisitionDuration_s"),"RMSE":m["RMSE"],"MAE":m["MAE"],"Bias":m["Bias"],"SSE":m["SSE"],"SAE":m["SAE"],"N":m["N"],"SelectedGuard":q["SelectedGuard"],"SelectedAlpha":q["SelectedAlpha"],"SelectedC":q["SelectedC"],"SelectedTau":q["SelectedTau"],"FallbackSelected":q["FallbackSelected"],"MaxAbsDiffInsideGuard":q["MaxAbsDiffInsideGuard"],"Runtime_s":q["Runtime_s"]})
    return pd.DataFrame(rows)
def pooled_metrics(outer):
    rows=[]
    for model,d in outer.groupby("Model"):
        n=int(d.N.sum()); rows.append({"Model":model,"Batteries":d.OuterHeldoutBattery.nunique(),"N":n,"RMSE":float(np.sqrt(d.SSE.sum()/n)),"MAE":float(d.SAE.sum()/n),"BatteryMacroRMSE":float(d.RMSE.mean()),"MeanDiagnoses":float(d.DiagnosisCount.mean()),"MeanScalars":float(d.HI_ScalarInputs.mean()),"MeanDuration_s":float(pd.to_numeric(d.AcquisitionDuration_s,errors="coerce").mean())})
    return pd.DataFrame(rows)
def by_L_metrics(outer):
    rows=[]
    for (model,L),d in outer.groupby(["Model","DiagnosisCount"]):
        n=int(d.N.sum()); rows.append({"Model":model,"DiagnosisCount":int(L),"Batteries":d.OuterHeldoutBattery.nunique(),"N":n,"RMSE":float(np.sqrt(d.SSE.sum()/n)),"MAE":float(d.SAE.sum()/n),"BatteryMacroRMSE":float(d.RMSE.mean())})
    return pd.DataFrame(rows)
def paired_bootstrap(diff,reps,seed):
    d=np.asarray(diff,float); d=d[np.isfinite(d)]; rng=np.random.default_rng(seed); vals=np.empty(reps,float)
    for i in range(0,reps,1000):
        n=min(1000,reps-i); idx=rng.integers(0,len(d),(n,len(d))); vals[i:i+n]=d[idx].mean(1)
    return {"ObservedMeanDifference":float(d.mean()),"CI95_Low":float(np.quantile(vals,.025)),"CI95_High":float(np.quantile(vals,.975)),"P_FirstBetter":float(np.mean(vals<0)),"Batteries":len(d),"BootstrapReps":reps,"FirstWinCount":int((d<0).sum()),"TieCount":int((d==0).sum()),"FirstLossCount":int((d>0).sum())}
def paired_model_bootstrap(outer,a):
    x=outer[outer.Model.eq("Adaptive_XGB")][["OuterHeldoutBattery","RMSE"]].rename(columns={"RMSE":"XGB"}); s=outer[outer.Model.eq("Adaptive_SafeResidualMoE")][["OuterHeldoutBattery","RMSE"]].rename(columns={"RMSE":"Safe"}); d=x.merge(s,on="OuterHeldoutBattery"); d["Diff"]=d.Safe-d.XGB; return pd.DataFrame([{"Comparison":"Adaptive_SafeResidualMoE - Adaptive_XGB","NegativeMeans":"SafeResidualMoE better",**paired_bootstrap(d.Diff,a.bootstrap,stable_seed(CODE_VERSION,"PAIR"))}])

def fixed_l3_reference(a):
    for p in [a.v417_root/"02_frozenXGB_safeResidualMoE_pooled_metrics.csv",a.v417_root/"02_frozenXGB_MoE_pooled_metrics.csv"]:
        if p.exists():
            d=pd.read_csv(p); q=d[d.Role.astype(str).str.upper().eq("OPT")].copy()
            if not q.empty:
                q["OperatingPoint"]=q.Model.map({"FrozenXGB":"Fixed Event Opt-L3 XGB","SafeResidualMoE":"Fixed Event Opt-L3 SafeResidualMoE"}).fillna(q.Model); q["MeanDiagnoses"]=3.; q["MeanScalars"]=18.; return q
    return pd.DataFrame()
def make_points(pooled,fixed):
    rows=[]
    for _,r in pooled.iterrows(): rows.append({"OperatingPoint":"Adaptive 2/3 XGB" if r.Model=="Adaptive_XGB" else "Adaptive 2/3 SafeResidualMoE","MeanDiagnoses":r.MeanDiagnoses,"MeanScalars":r.MeanScalars,"RMSE":r.RMSE,"BatteryMacroRMSE":r.BatteryMacroRMSE})
    for _,r in fixed.iterrows(): rows.append({"OperatingPoint":r.OperatingPoint,"MeanDiagnoses":3.,"MeanScalars":18.,"RMSE":r.RMSE,"BatteryMacroRMSE":r.get("BatteryMacroRMSE",np.nan)})
    return pd.DataFrame(rows)
def draw_resource_accuracy(points,outpng):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,5.7)); mk={"Adaptive 2/3 XGB":"s","Adaptive 2/3 SafeResidualMoE":"D","Fixed Event Opt-L3 XGB":"o","Fixed Event Opt-L3 SafeResidualMoE":"^"}; offs={"Adaptive 2/3 XGB":(-72,20),"Adaptive 2/3 SafeResidualMoE":(-72,-35),"Fixed Event Opt-L3 XGB":(12,22),"Fixed Event Opt-L3 SafeResidualMoE":(12,-35)}
    for r in points.itertuples(index=False): ax.scatter(r.MeanScalars,100*r.RMSE,s=95,marker=mk.get(r.OperatingPoint,"o")); ax.annotate(f"{r.OperatingPoint}\n{r.MeanScalars:.2f} scalars\nRMSE={100*r.RMSE:.4f}%",(r.MeanScalars,100*r.RMSE),xytext=offs.get(r.OperatingPoint,(8,8)),textcoords="offset points",fontsize=8.5,arrowprops=dict(arrowstyle="-",linewidth=.7))
    ax.set_xlabel("Mean measured scalars"); ax.set_ylabel("Pooled RMSE (%)"); ax.set_title("Adaptive-policy transfer across prediction models"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(outpng,dpi=300); plt.close(fig)

def finalize(ctx,a):
    missing=[b for b in sorted(map(normalize_battery,ctx["batteries"])) if not (fold_result_json(a,b).exists() and fold_pred_npz(a,b).exists())]
    if missing: raise RuntimeError(f"missing folds {missing}")
    outer=collect_outer(ctx,a); pooled=pooled_metrics(outer); byL=by_L_metrics(outer); boot=paired_model_bootstrap(outer,a); fixed=fixed_l3_reference(a); points=make_points(pooled,fixed)
    save_csv(outer,a.output_root/"01_outer_battery_metrics.csv"); save_csv(pooled,a.output_root/"02_pooled_metrics.csv"); save_csv(boot,a.output_root/"03_paired_battery_bootstrap.csv"); save_csv(byL,a.output_root/"04_metrics_by_diagnosis_count.csv"); save_csv(points,a.output_root/"05_resource_accuracy_operating_points.csv")
    draw_resource_accuracy(points,a.output_root/"Figure_01_Adaptive_Model_Transfer_ResourceAccuracy.png")
    x=float(pooled.loc[pooled.Model.eq("Adaptive_XGB"),"RMSE"].iloc[0]); s=float(pooled.loc[pooled.Model.eq("Adaptive_SafeResidualMoE"),"RMSE"].iloc[0]); b=boot.iloc[0]; scientific=pd.DataFrame([{"Question":"Frozen Adaptive PRT0.008 transfer to SafeResidualMoE without timing re-optimization","Adaptive_XGB_RMSE":x,"Adaptive_SafeResidualMoE_RMSE":s,"RelativePooledGainPct":100*(x-s)/x,"MeanBatteryDifference_SafeMinusXGB":b.ObservedMeanDifference,"CI95_Low":b.CI95_Low,"CI95_High":b.CI95_High,"P_SafeBetter":b.P_FirstBetter,"TimingReoptimizedForSafeMoE":False}]); save_csv(scientific,a.output_root/"06_scientific_summary.csv")
    hard=float(pd.to_numeric(outer.MaxAbsDiffInsideGuard,errors="coerce").max()); xrepro=abs(x-0.007329626731119783); gate=pd.DataFrame([{"Check":"47 batteries","Observed":outer.OuterHeldoutBattery.nunique(),"Expected":47,"PASS":int(outer.OuterHeldoutBattery.nunique()==47)},{"Check":"94 model rows","Observed":len(outer),"Expected":94,"PASS":int(len(outer)==94)},{"Check":"Adaptive XGB exact pooled reproduction","Observed":xrepro,"Expected":"<=1e-9","PASS":int(xrepro<=1e-9)},{"Check":"Hard guard invariant","Observed":hard,"Expected":"<=1e-8","PASS":int(hard<=1e-8)}]); save_csv(gate,a.output_root/"07_acceptance_gate.csv"); status="PASS" if gate.PASS.eq(1).all() else "CHECK"; save_json({"Task":"RWTH48 V4.26 Adaptive SafeResidualMoE Policy Transfer","CodeVersion":CODE_VERSION,"Status":status,"PrimaryPolicy":POLICY_ID,"TimingReoptimizedForSafeMoE":False,"ScientificSummary":scientific.to_dict("records"),"Acceptance":gate.to_dict("records")},a.output_root/"99_RWTH48_V4_26_COMPLETE.json")
    print("\n"+"="*120); print("RWTH48 V4.26 FINALIZE",status); print(pooled.to_string(index=False)); print("\n",boot.to_string(index=False)); print("\n",gate.to_string(index=False)); print("Output:",a.output_root); print("="*120)


def main():
    a=parse_args(); configure_threads(a); ensure_dir(a.output_root); ensure_dir(a.state_root); ctx=preflight(a); save_csv(ctx["preflight_v426"],a.output_root/"00_preflight.csv"); save_csv(ctx["adaptive_schedules"],a.output_root/"00_frozen_adaptive_schedules.csv")
    if a.stage=="audit": print("RWTH48 V4.26 AUDIT PASS\n",ctx["preflight_v426"].to_string(index=False)); return
    if a.stage in ("run","all"):
        run_worker(ctx,a)
        if a.auto_finalize and a.parallel_worker_id==0: wait_for_workers(a); finalize(ctx,a)
        elif a.parallel_workers==1 and a.stage=="all": finalize(ctx,a)
        return
    if a.stage=="finalize": finalize(ctx,a)

if __name__=="__main__": main()

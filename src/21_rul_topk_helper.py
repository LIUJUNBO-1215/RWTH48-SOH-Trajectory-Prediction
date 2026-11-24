# -*- coding: utf-8 -*-
r"""
RWTH48 V4.12A
=============
RUL-specific Event HI Re-screening + TopK Pareto Search
RUL专用Event HI重筛选 + Top2/Top3/Top4/Top5严格source-only Pareto搜索

================================================================================
一、研究动机
================================================================================

当前冻结结果：

SOH trajectory:
    V4.11D Horizon-Adaptive MoE
    full-life SOH RMSE ~= 0.007903779

RUL:
    V4.11E Direct RUL XGB
    MAE ~= 61.98 cycles
    RMSE ~= 74.43 cycles
    MAPE ~= 7.99%
    47/47 batteries available

V4.11F 已系统测试：
    source-only affine calibration
    tail-balanced weighting
    weighted 80/120/160 source augmentation
    RUL target vs absolute EOL-cycle target
    XGB/Ridge/PLS/ExtraTrees
    nonnegative ensemble

但没有超过 V4.11E。
V4.11F 的主要结构性问题仍是 prediction spread compression：

    true RUL std  ~= 79.85 cycles
    V4.11E pred std ~= 44.57 cycles
    spread ratio ~= 0.558

而 V4.11F 新路线 spread ratio 进一步下降。

因此下一步不再继续调整同一组“SOH trajectory Top2 HI”的回归器，
而是直接回答：

    最适合重建未来SOH轨迹的HI，
    是否也是最适合区分最终寿命/RUL的HI？

V4.12A 对合法 Event HI 候选进行 RUL-specific source-only 重筛选。

================================================================================
二、核心约束
================================================================================

仍然使用相同早期测量窗口：

    primary test origin = cycle 80

    history offsets:
        [-40, -32, -24, -16, -8, 0]

    即 cycle:
        40,48,56,64,72,80

Event checkpoints:
    RefAh = [0.05, 0.09, 0.13]

每个HI:
    6 diagnoses x 3 checkpoints = 18 measured scalars

TopK:
    Top2 -> 36 measured scalars
    Top3 -> 54 measured scalars
    Top4 -> 72 measured scalars
    Top5 -> 90 measured scalars

注意：
    这里增加的是从“相同早期Event测量数据”中选择的合法HI，
    不是新增未来测量，不新增SOH/容量历史。

禁止：
    historical true SOH
    capacity history
    future HI
    future SOH
    Cycle as model feature
    true RUL/EOL as input
    held-out battery calibration
    held-out battery参与HI筛选
    nearest-cycle substitution

================================================================================
三、输入
================================================================================

1) Event model-ready / frozen Event HI table
--------------------------------------------

优先自动寻找：

    05_ALL_EVENT_MODEL_READY.csv.gz

若没有，再寻找：

    06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz

也可显式指定：
    --event-ready "完整路径"

脚本会自动识别常见列：
    Battery
    GlobalCycle / Cycle
    RefAh
    Phase
    PrefixType / Platform

若存在 Phase：
    默认只保留 Discharge
    可 --phase Charge 修改

若存在平台列：
    自动保留 Event。

2) V4.11B helper + V4.10A helper
--------------------------------

用于加载 V4.11A2 已冻结的 full-life handoff：

    X = [141,6,6]
    battery
    origin = 80/120/160
    true_eol_lead
    true_eol_cycle

并用于调用 V4.10A 的冻结 deterministic feature engineering。

3) V4.11E baseline
------------------

直接复用：
    04_outer_rul_predictions.csv

其中：
    TrueRUL
    DirectRUL

作为 exact-common frozen baseline。

================================================================================
四、非常重要：原Top2重建一致性检查
================================================================================

原 Event Top2：

    V_TimeWeightedMean
    P_Last

V4.12A 会从 Event-ready 表重新构造：

    6 diagnoses x 3 checkpoints x 2 HIs

然后自动尝试：

    HI顺序
    checkpoint-major / HI-major
    history正序 / 反序

与 V4.11B frozen bundle.X 比较。

只有找到一个 layout，使 reconstructed original Top2 与 frozen bundle.X
在数值容差内一致，才允许继续。

否则：
    直接 FAIL。

这一步保证：
    V4.12A 没有悄悄改变 Event Top2 的测量语义/排列。

================================================================================
五、V4.10A exact deterministic transform 如何推广到TopK
================================================================================

V4.10A 对原Top2：

    36 measured scalars
        ->
    196 deterministic model features

V4.12A 不重新发明另一套特征工程。

对于任意候选HI pair：
    把它们映射到原Top2的两个channel位置
        ->
    调用同一个 V4.10A engineer_features
        ->
    196 deterministic pair features

对于 TopK>2：

    selected = [HI_anchor, HI_2, ..., HI_K]

构造：
    pair(anchor, HI_2)
    pair(anchor, HI_3)
    ...
    pair(anchor, HI_K)

每个pair 196 features，最后拼接：

    Top2: 196
    Top3: 392
    Top4: 588
    Top5: 784

这不增加物理测量。
只是对所选HI进行相同 deterministic transform。

这样避免：
    Top2用一套特征工程
    Top3/4/5又用另一套完全不同的工程。

================================================================================
六、严格47-cell outer LOBO
================================================================================

每个 outer held-out battery h：

    source batteries = other 46

所有：
    HI relevance ranking
    pair search
    greedy forward selection
    TopK source CV
    Pareto K selection

全部只使用 source batteries。

held-out h 的 RUL 只能在全部选择冻结以后用于最终一次 outer prediction。

================================================================================
七、候选HI预筛
================================================================================

Event-ready 可能包含数百个HI。
全组合搜索成本不可接受。

第一步，在每个 outer fold 内仅用 source batteries 计算
每个HI group的 RUL relevance：

    每个HI有18个位置
    对18个位置分别计算 Spearman(|rho|)
    取 strongest positions 的稳健汇总

默认：
    relevance = mean(top4 abs Spearman)

然后取：
    shortlist top 14

同时强制把原Top2加入 shortlist：
    V_TimeWeightedMean
    P_Last

这一步只是 source-only candidate reduction。

================================================================================
八、Top2 exhaustive pair + Top3~Top5 greedy
================================================================================

在 shortlist 内：

Top2:
    遍历所有 unordered pairs
    每个pair做 source battery 5-fold CV
    固定 shallow XGB
    score = source OOF RUL MAE

Top3:
    冻结最佳Top2
    逐个尝试添加一个剩余HI
    选择 source OOF MAE 最低者

Top4 / Top5:
    继续forward greedy。

每个候选同时记录：
    MAE
    RMSE
    Bias
    R2
    MAPE
    spread ratio

================================================================================
九、RUL spread Pareto
================================================================================

V4.11F 证明：
    只看MAE可能会继续压缩RUL预测范围。

所以 V4.12A 不使用人为加权loss直接改变模型，
而是明确输出 Top2~Top5 的二维 source-only Pareto：

    objective 1:
        source CV MAE -> minimize

    objective 2:
        abs(1 - SpreadRatio) -> minimize

每个 outer fold 的 ParetoSelected K 规则：

    1) 找 Top2~Top5 中最小 inner MAE
    2) 保留 MAE <= min_MAE * 1.02 的候选
    3) 在这些候选里选 abs(1-spread) 最小者
    4) 再按 MAE、小K tie-break

即：
    只有当额外HI的MAE基本不损失时，
    才允许用更好的RUL spread换取更大的K。

这不是用 outer held-out result 选择K。

================================================================================
十、训练origin
================================================================================

V4.12A 默认采用：

    --train-origin-mode strict

即HI搜索期间只使用 source origin80。
原因：
    当前任务是隔离“HI选择”本身的贡献，
    不把 V4.11F 已经证明不稳定的 multi-origin tuning 再混入搜索。

可做敏感性分析：
    --train-origin-mode augmented

augmented模式：
    source origin80/120/160都训练
    每节battery总权重保持1：
        每个origin权重=1/3
    inner validation仍只看origin80。

正式主结论建议先使用 strict。

================================================================================
十一、最终比较
================================================================================

Outer predictions:
    Frozen V4.11E DirectRUL
    RULSpecific_Top2
    RULSpecific_Top3
    RULSpecific_Top4
    RULSpecific_Top5
    RULSpecific_ParetoSelected

重点成功标准：

    MAE < 61.98 cycles
    且 SpreadRatio > 0.558

理想：
    MAE < 55
    最好 < 50

如果 TopK 改善MAE但 spread ratio继续下降，
不能简单称为解决了寿命辨识问题。

================================================================================
十二、输出
================================================================================

00_config.json
00_preflight.csv
01_event_input_schema_audit.csv
02_original_top2_reconstruction_audit.csv
03_candidate_hi_coverage.csv

folds/<Battery>.npz

04_hi_relevance_by_outer_fold.csv
05_pair_and_greedy_search_audit.csv
06_selected_hi_by_outer_fold.csv
07_outer_predictions_by_topk.csv

08_topk_outer_metrics.csv
09_topk_life_bin_metrics.csv
10_hi_selection_frequency.csv
11_gain_vs_v411e.csv
12_pareto_selection_frequency.csv
13_source_only_protocol_audit.csv
14_acceptance_gate.csv

Figures/
    01_topk_mae.png
    02_mae_vs_spread.png
    03_hi_frequency.png
    04_true_vs_pred_champion.png
    05_life_bin_mae.png

99_RWTH48_V4_12A_COMPLETE.json

================================================================================
十三、运行
================================================================================

审计：
python -u "src/21_rul_topk_helper.py" --stage audit

单电池 smoke：
python -u "src/21_rul_topk_helper.py" --stage run --only-battery RWTH-002 --xgb-threads 4

正式：
python -u "src/21_rul_topk_helper.py" --stage all --xgb-threads 4

两个CPU worker：
终端1：
python -u "src/21_rul_topk_helper.py" --stage worker --num-workers 2 --worker-id 0 \
    --xgb-threads 2 --auto-finalize

终端2：
python -u "src/21_rul_topk_helper.py" --stage worker --num-workers 2 --worker-id 1 \
    --xgb-threads 2
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. Frozen configuration
# =============================================================================

BASE = (Path(__file__).resolve().parents[1] / "workspace")
CODE_DIR_DEFAULT = Path(__file__).resolve().parent

V411B_HELPER_DEFAULT = (
    CODE_DIR_DEFAULT / "22_full_life_soh_helper.py"
)
V410A_HELPER_DEFAULT = (
    CODE_DIR_DEFAULT / "23_event_tree_helper.py"
)

V411A2_ROOT_DEFAULT = (
    BASE / "RWTH48_V4_11A2_跨StrictSegment全生命目标重建审计"
)
INPUT_WIDE_DEFAULT = V411A2_ROOT_DEFAULT / "09_input_history_wide.csv.gz"
TARGET_GRID_DEFAULT = (
    V411A2_ROOT_DEFAULT / "08_operational_eol80_target_grid_long.csv.gz"
)
V411A2_COMPLETE_DEFAULT = (
    V411A2_ROOT_DEFAULT / "99_RWTH48_V4_11A2_COMPLETE.json"
)

V411D_ROOT_DEFAULT = (
    BASE / "RWTH48_V4_11D_HorizonAdaptive_MoE_ResidualHybrid"
)
V411D_MOE_OOF_DEFAULT = (
    V411D_ROOT_DEFAULT / "01_MoE" / "03_ALL_OOF_MoE_long.csv.gz"
)
V411D_METRICS_DEFAULT = (
    V411D_ROOT_DEFAULT / "20_reference_and_hybrid_pooled_metrics.csv"
)
V411D_COMPLETE_DEFAULT = (
    V411D_ROOT_DEFAULT / "99_RWTH48_V4_11D_COMPLETE.json"
)

V411E_ROOT_DEFAULT = (
    BASE / "RWTH48_V4_11E_EOL_RUL_Aware_FrozenMoE"
)
V411E_OUTER_DEFAULT = V411E_ROOT_DEFAULT / "04_outer_rul_predictions.csv"
V411E_COMPLETE_DEFAULT = V411E_ROOT_DEFAULT / "99_RWTH48_V4_11E_COMPLETE.json"

V411F_ROOT_DEFAULT = (
    BASE / "RWTH48_V4_11F_DirectRUL_Optimization"
)
V411F_COMPLETE_DEFAULT = V411F_ROOT_DEFAULT / "99_RWTH48_V4_11F_COMPLETE.json"

OUTPUT_DEFAULT = BASE / "RWTH48_V4_12A_RULSpecific_HI_TopK_Pareto"

EXPECTED_BATTERIES = 47
TEST_ORIGIN = 80
SOURCE_ORIGINS = (80, 120, 160)
K_TRAJ = 1000

HISTORY_OFFSETS = (-40, -32, -24, -16, -8, 0)
REFAH_TARGETS = (0.05, 0.09, 0.13)

ORIGINAL_TOP2 = ("V_TimeWeightedMean", "P_Last")

TOPK_VALUES = (2, 3, 4, 5)
DEFAULT_SHORTLIST = 14
DEFAULT_INNER_FOLDS = 5
DEFAULT_PARETO_MAE_TOL = 0.02

RANDOM_SEED = 20260823
ENC = "utf-8-sig"
FAST_GZIP = {"method": "gzip", "compresslevel": 1}
RECON_ATOL = 5e-7

METHOD_BASELINE = "F0_V411E_DirectRUL"

FORBIDDEN_NAME_RE = re.compile(
    r"(?:"
    r"(^|_)SOH($|_)|"
    r"capacity|"
    r"true.?eol|"
    r"true.?rul|"
    r"target|"
    r"label|"
    r"future|"
    r"prediction|"
    r"pred_|"
    r"globalcycle|"
    r"(^|_)cycle($|_)|"
    r"battery"
    r")",
    flags=re.IGNORECASE,
)

META_HINTS = {
    "battery", "batteryid", "cell", "cellid",
    "cycle", "globalcycle", "cycleindex", "cycleno",
    "refah", "ref_ah", "referenceah", "checkpoint",
    "phase", "prefixtype", "platform", "strictsegmentkey",
    "sourcepart", "split", "fold",
    "scope", "nodetype", "nodevalue", "platformnode",
    "eventreachable", "prefixfeaturevalid", "missingfeaturecount",
    "reachable", "samplecount", "observedhardinvalidrate",
    "observedspikerate",
}

# Keep the RUL-specific candidate pool on exactly the same design-level
# eligibility boundary used by the frozen V4.7E Event-HI ablation helper.
V47E_NON_FEATURE_COLS = {
    "Battery", "GlobalCycle", "Scope", "Platform", "NodeType", "NodeValue",
    "PlatformNode", "ReferenceCapacity_Ah", "ReferenceCapacity_Ah_QCOnly",
    "SOH_Target_SameCycle", "SOH_IsMeasuredCheckUp", "SOH_LabelUsable",
    "SOH_LabelSource", "EventReachable", "PrefixFeatureValid",
    "MissingFeatureCount", "Reachable", "SourcePart",
    "StrictSegmentKey", "CyclingSOH_Charge_Canonical",
}
V47E_DESIGN_FORBIDDEN = {
    "SampleCount", "ObservedHardInvalidRate", "ObservedSpikeRate",
}
V47E_STRUCTURAL_REDUNDANT_DROP = {
    "PartialThroughput_Wh",
}
V47E_GLOBAL_CONSTANT_DROP = {
    "VBand_3p0_3p2_Ratio",
    "VBand_3p2_3p4_Ratio",
    "VBand_4p0_4p2_Ratio",
}


# =============================================================================
# 1. CLI / utilities
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWTH48 V4.12A RUL-specific Event HI TopK Pareto"
    )

    p.add_argument(
        "--stage",
        choices=["audit", "run", "finalize", "all", "worker"],
        default="all",
    )

    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--code-dir", type=Path, default=CODE_DIR_DEFAULT)
    p.add_argument("--v411b-helper", type=Path, default=V411B_HELPER_DEFAULT)
    p.add_argument("--v410a-helper", type=Path, default=V410A_HELPER_DEFAULT)

    p.add_argument("--input-wide", type=Path, default=INPUT_WIDE_DEFAULT)
    p.add_argument("--target-grid", type=Path, default=TARGET_GRID_DEFAULT)
    p.add_argument("--v411a2-complete", type=Path, default=V411A2_COMPLETE_DEFAULT)

    p.add_argument("--v411d-moe-oof", type=Path, default=V411D_MOE_OOF_DEFAULT)
    p.add_argument("--v411d-metrics", type=Path, default=V411D_METRICS_DEFAULT)
    p.add_argument("--v411d-complete", type=Path, default=V411D_COMPLETE_DEFAULT)

    p.add_argument("--v411e-outer", type=Path, default=V411E_OUTER_DEFAULT)
    p.add_argument("--v411e-complete", type=Path, default=V411E_COMPLETE_DEFAULT)
    p.add_argument("--v411f-complete", type=Path, default=V411F_COMPLETE_DEFAULT)

    p.add_argument("--event-ready", type=Path, default=None)
    p.add_argument("--phase", type=str, default="Discharge")
    p.add_argument("--candidate-hi-file", type=Path, default=None)

    p.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    p.add_argument("--only-battery", type=str, default="")
    p.add_argument("--force", action="store_true")

    p.add_argument("--shortlist", type=int, default=DEFAULT_SHORTLIST)
    p.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    p.add_argument("--pareto-mae-tol", type=float, default=DEFAULT_PARETO_MAE_TOL)
    p.add_argument(
        "--train-origin-mode",
        choices=["strict", "augmented"],
        default="strict",
    )
    p.add_argument("--xgb-threads", type=int, default=4)

    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--auto-finalize", action="store_true")
    p.add_argument("--wait-poll-seconds", type=int, default=30)
    p.add_argument("--wait-timeout-hours", type=float, default=24.0)

    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--dpi", type=int, default=260)

    return p.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def normalize_battery(x: object) -> str:
    s = str(x).strip().upper().replace("_", "-").replace(" ", "")
    m = re.search(r"RWTH-?0*(\d+)", s)
    if m:
        return f"RWTH-{int(m.group(1)):03d}"
    return s


def norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def stable_hash(obj: object) -> str:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def split_battery_groups(
    batteries: Sequence[str],
    n_folds: int,
) -> List[List[str]]:
    bats = sorted(set(map(str, batteries)))
    k = max(2, min(int(n_folds), len(bats)))
    groups = [[] for _ in range(k)]
    for i, b in enumerate(bats):
        groups[i % k].append(b)
    return [g for g in groups if g]


def fold_cache_path(out: Path, battery: str) -> Path:
    return out / "folds" / f"{battery}.npz"


def selected_worker_batteries(
    batteries: Sequence[str],
    args: argparse.Namespace,
) -> List[str]:
    bats = list(sorted(batteries))

    if args.only_battery:
        b = normalize_battery(args.only_battery)
        if b not in bats:
            raise RuntimeError(f"--only-battery={b} 不存在")
        return [b]

    nw = max(1, int(args.num_workers))
    wid = int(args.worker_id)
    if wid < 0 or wid >= nw:
        raise ValueError("--worker-id必须满足0 <= id < num-workers")

    return [
        b for i, b in enumerate(bats)
        if i % nw == wid
    ]


# =============================================================================
# 2. Load frozen V4.11 handoff / baseline
# =============================================================================

def make_v411b_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.base = args.base
    ns.code_dir = args.code_dir
    ns.input_wide = args.input_wide
    ns.target_grid = args.target_grid
    ns.v411a2_complete = args.v411a2_complete
    ns.v410a_helper = args.v410a_helper
    ns.output_root = args.output_root

    ns.stage = "run"
    ns.routes = ["xgb"]
    ns.audit_only = False
    ns.only_battery = args.only_battery
    ns.force = args.force

    ns.k = K_TRAJ
    ns.test_origin = TEST_ORIGIN
    ns.source_origins = list(SOURCE_ORIGINS)

    ns.xgb_device = "cpu"
    ns.xgb_threads = args.xgb_threads
    ns.xgb_leads_per_origin = 240

    ns.architectures = ["LSTM"]
    ns.max_epochs = 70
    ns.deep_batch = 12
    ns.early_patience = 10
    ns.min_delta = 1e-5
    ns.learning_rate = 8e-4
    ns.dropout = 0.12
    ns.clipnorm = 1.0
    ns.validation_batteries = 6
    ns.no_refit_all_source = False
    ns.detail_kernel = 9
    ns.detail_gate_bias = -1.5
    return ns


def load_v411e_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    d = pd.read_csv(path, low_memory=False)

    required = {
        "OuterHeldoutBattery",
        "TrueRUL",
        "DirectRUL",
    }
    miss = sorted(required - set(d.columns))
    if miss:
        raise RuntimeError(f"V4.11E baseline缺列: {miss}")

    d["OuterHeldoutBattery"] = d["OuterHeldoutBattery"].map(normalize_battery)

    if len(d) != EXPECTED_BATTERIES:
        raise RuntimeError(f"V4.11E rows={len(d)} !=47")
    if d["OuterHeldoutBattery"].nunique() != EXPECTED_BATTERIES:
        raise RuntimeError("V4.11E battery unique !=47")
    return d


# =============================================================================
# 3. Locate / normalize Event table
# =============================================================================

def locate_event_ready(args: argparse.Namespace) -> Path:
    if args.event_ready is not None:
        p = Path(args.event_ready)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    direct_candidates = [
        args.base / "RWTH48_V4_2_双平台冻结_FixedTime_Event_HI"
        / "05_ALL_EVENT_MODEL_READY.csv.gz",
        args.base / "RWTH48_V4_2_双平台冻结_FixedTime_Event_HI"
        / "06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz",
        args.base / "05_ALL_EVENT_MODEL_READY.csv.gz",
        args.base / "06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz",
    ]
    for p in direct_candidates:
        if p.exists():
            return p

    patterns = (
        "05_ALL_EVENT_MODEL_READY*.csv.gz",
        "06_EVENT_REFAH_FROZEN_LABEL_READY*.csv.gz",
        "*ALL_EVENT_MODEL_READY*.csv.gz",
    )
    hits = []
    for pat in patterns:
        try:
            hits.extend(args.base.rglob(pat))
        except Exception:
            pass

    hits = sorted(
        set(hits),
        key=lambda p: (
            0 if "V4_2" in str(p) else 1,
            len(str(p)),
            str(p),
        ),
    )
    if not hits:
        raise FileNotFoundError(
            "未找到05_ALL_EVENT_MODEL_READY.csv.gz或"
            "06_EVENT_REFAH_FROZEN_LABEL_READY.csv.gz；"
            "请用 --event-ready 显式指定。"
        )
    return hits[0]


def choose_column(
    columns: Sequence[str],
    exact_norm_names: Sequence[str],
    contains_tokens: Sequence[str] = (),
) -> Optional[str]:
    """
    Robust column resolver.

    IMPORTANT:
    The old implementation evaluated ``all([]) == True`` when
    ``contains_tokens`` was empty.  Therefore an optional column such as
    Phase/Platform that had no exact match incorrectly fell back to the FIRST
    dataframe column (in the current Event file that happened to be Battery).
    This caused Phase filtering to compare "Discharge" against RWTH-xxx IDs.

    Rule:
      1) exact normalized-name match first;
      2) token fallback ONLY when at least one token is explicitly supplied;
      3) otherwise return None.
    """
    norm_map = {norm_col(c): c for c in columns}

    for n in exact_norm_names:
        nn = norm_col(n)
        if nn in norm_map:
            return norm_map[nn]

    tokens = [norm_col(t) for t in contains_tokens if norm_col(t)]
    if tokens:
        for c in columns:
            nc = norm_col(c)
            if all(t in nc for t in tokens):
                return c

    return None


def parse_refah_value(x: object) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().lower()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else np.nan


@dataclass
class EventSchema:
    battery_col: str
    cycle_col: str
    refah_col: str
    phase_col: Optional[str]
    platform_col: Optional[str]
    hi_cols: List[str]


def detect_event_schema(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> EventSchema:
    cols = list(df.columns)

    battery_col = choose_column(
        cols,
        ["Battery", "BatteryID", "Cell", "CellID"],
    )
    cycle_col = choose_column(
        cols,
        ["GlobalCycle", "Cycle", "CycleIndex", "CycleNo"],
    )
    # V4.2 frozen Event table is a LONG node table.
    # Authoritative Event checkpoint identity is:
    #     PlatformNode = Event_RefAh_0.05 / 0.09 / 0.13
    # with NodeValue = 0.05 / 0.09 / 0.13.
    #
    # ReferenceCapacity_Ah is the battery's reference-capacity metadata
    # used to construct those fractions; it is NOT a checkpoint coordinate.
    #
    # Prefer PlatformNode because this is exactly what the frozen V4.7E/V4.9C
    # loaders use. Fall back to NodeValue, then only to explicit RefAh columns.
    platform_node_col = choose_column(
        cols,
        ["PlatformNode"],
    )
    node_value_col = choose_column(
        cols,
        ["NodeValue"],
    )

    refah_col = None
    if platform_node_col is not None:
        vals = df[platform_node_col].astype(str)
        if vals.str.contains(
            "Event_RefAh_",
            case=False,
            regex=False,
            na=False,
        ).any():
            refah_col = platform_node_col

    if refah_col is None and node_value_col is not None:
        nv = pd.to_numeric(df[node_value_col], errors="coerce")
        has_frozen_nodes = any(
            np.isclose(
                nv.to_numpy(float),
                r,
                atol=5e-5,
                rtol=0,
                equal_nan=False,
            ).any()
            for r in REFAH_TARGETS
        )
        if has_frozen_nodes:
            refah_col = node_value_col

    if refah_col is None:
        refah_col = choose_column(
            cols,
            ["RefAh", "Ref_Ah", "CheckpointAh", "RefAhFraction"],
        )

    phase_col = choose_column(
        cols,
        ["Phase", "PhysicalPhase"],
    )
    platform_col = choose_column(
        cols,
        ["PrefixType", "Platform", "PrefixPlatform", "FeaturePlatform"],
    )

    # Optional metadata columns must never alias the required key columns.
    required_key_cols = {battery_col, cycle_col, refah_col}
    if phase_col in required_key_cols:
        raise RuntimeError(
            f"Phase列误识别为关键列: phase_col={phase_col}, "
            f"battery={battery_col}, cycle={cycle_col}, refah={refah_col}"
        )
    if platform_col in required_key_cols:
        raise RuntimeError(
            f"Platform列误识别为关键列: platform_col={platform_col}, "
            f"battery={battery_col}, cycle={cycle_col}, refah={refah_col}"
        )

    if battery_col is None:
        raise RuntimeError("Event table无法识别Battery列")
    if cycle_col is None:
        raise RuntimeError("Event table无法识别Cycle/GlobalCycle列")
    if refah_col is None:
        raise RuntimeError(
            "Event table无法识别RefAh列。"
            "当前V4.12A要求long Event checkpoint table。"
        )

    excluded = {
        battery_col,
        cycle_col,
        refah_col,
    }
    if phase_col:
        excluded.add(phase_col)
    if platform_col:
        excluded.add(platform_col)

    explicit_candidates = None
    if args.candidate_hi_file is not None:
        cpath = Path(args.candidate_hi_file)
        if not cpath.exists():
            raise FileNotFoundError(cpath)
        ct = pd.read_csv(cpath, low_memory=False)
        if len(ct.columns) == 0:
            raise RuntimeError("candidate-hi-file为空")
        # First column by default.
        explicit_candidates = set(
            ct.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        )

    hi_cols = []
    for c in cols:
        if c in excluded:
            continue

        # Exact frozen V4.7E design-level exclusions first.
        if c in V47E_NON_FEATURE_COLS:
            continue
        if c in V47E_DESIGN_FORBIDDEN:
            continue
        if c in V47E_STRUCTURAL_REDUNDANT_DROP:
            continue
        if c in V47E_GLOBAL_CONSTANT_DROP:
            continue
        if str(c).startswith("SOH_") or str(c).startswith("Pred_"):
            continue
        if "Capacity" in str(c) and c not in {"PartialCapacity_Ah"}:
            continue

        # Additional defensive guards.
        if FORBIDDEN_NAME_RE.search(str(c)):
            continue
        if norm_col(c) in META_HINTS:
            continue
        if explicit_candidates is not None and c not in explicit_candidates:
            continue

        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() == 0:
            continue
        hi_cols.append(c)

    if len(hi_cols) < 2:
        raise RuntimeError(
            f"合法数值HI候选只有{len(hi_cols)}个，无法TopK搜索"
        )

    return EventSchema(
        battery_col=battery_col,
        cycle_col=cycle_col,
        refah_col=refah_col,
        phase_col=phase_col,
        platform_col=platform_col,
        hi_cols=hi_cols,
    )


def filter_event_table(
    df: pd.DataFrame,
    schema: EventSchema,
    args: argparse.Namespace,
) -> pd.DataFrame:
    d = df.copy()

    d[schema.battery_col] = d[schema.battery_col].map(normalize_battery)
    d[schema.cycle_col] = pd.to_numeric(
        d[schema.cycle_col],
        errors="coerce",
    )
    d["_RefAhNumeric"] = d[schema.refah_col].map(parse_refah_value)

    if schema.platform_col is not None:
        s = d[schema.platform_col].astype(str)
        m = s.str.contains("Event", case=False, regex=False, na=False)
        if m.any():
            d = d[m].copy()

    if schema.phase_col is not None:
        want = str(args.phase).strip()
        s = d[schema.phase_col].astype(str).str.strip()
        m = s.str.casefold().eq(want.casefold())
        if not m.any():
            raise RuntimeError(
                f"Phase={want}没有任何行；"
                f"可用值示例={sorted(s.dropna().unique().tolist())[:20]}"
            )
        d = d[m].copy()

    # Only the three frozen checkpoints.
    keep_ref = np.zeros(len(d), dtype=bool)
    ref = d["_RefAhNumeric"].to_numpy(float)
    for r in REFAH_TARGETS:
        keep_ref |= np.isclose(ref, r, atol=5e-5, rtol=0)
    d = d[keep_ref].copy()

    if len(d) == 0:
        raise RuntimeError(
            "Event checkpoint过滤后为0行。"
            f" refah_col={schema.refah_col!r}。"
            "V4.2冻结表应使用PlatformNode="
            "Event_RefAh_0.05/0.09/0.13（或NodeValue=0.05/0.09/0.13），"
            "不要把ReferenceCapacity_Ah当成RefAh节点。"
        )

    # Canonicalize exact checkpoint labels.
    rr = d["_RefAhNumeric"].to_numpy(float)
    canon = np.full(len(d), np.nan, dtype=float)
    for r in REFAH_TARGETS:
        m = np.isclose(rr, r, atol=5e-5, rtol=0)
        canon[m] = r
    d["_RefAhCanonical"] = canon

    node_counts = (
        d.groupby("_RefAhCanonical", dropna=False)
        .size()
        .to_dict()
    )
    missing_nodes = [
        r for r in REFAH_TARGETS
        if int(node_counts.get(r, 0)) == 0
    ]
    if missing_nodes:
        raise RuntimeError(
            f"Event冻结节点缺失: {missing_nodes}; "
            f"detected counts={node_counts}; "
            f"refah_col={schema.refah_col!r}"
        )

    key = [
        schema.battery_col,
        schema.cycle_col,
        "_RefAhCanonical",
    ]
    dup = d.duplicated(key, keep=False)
    if dup.any():
        sample = d.loc[dup, key].head(20)
        raise RuntimeError(
            "Event table在Battery×Cycle×RefAh上存在重复行。"
            "V4.12A禁止静默平均，请先检查Phase/Platform过滤。\n"
            + sample.to_string(index=False)
        )

    return d


# =============================================================================
# 4. Build candidate HI tensor for frozen origins
# =============================================================================

@dataclass
class HITensor:
    values: np.ndarray     # [sample, diagnosis=6, checkpoint=3, hi]
    batteries: np.ndarray  # [sample]
    origins: np.ndarray    # [sample]
    hi_names: List[str]
    coverage: pd.DataFrame


def build_hi_tensor(
    event: pd.DataFrame,
    schema: EventSchema,
    bundle,
    args: argparse.Namespace,
) -> HITensor:
    # Use frozen handoff sample order exactly.
    batteries = np.asarray(bundle.battery, dtype=object)
    origins = np.asarray(bundle.origin, dtype=int)

    if len(batteries) != EXPECTED_BATTERIES * 3:
        raise RuntimeError(
            f"handoff samples={len(batteries)}，预期141"
        )

    # Create key-indexed event table.
    d = event.copy()
    d["_CycleInt"] = np.rint(
        pd.to_numeric(d[schema.cycle_col], errors="coerce")
    ).astype("Int64")

    key_to_row = {}
    for i, r in d[
        [schema.battery_col, "_CycleInt", "_RefAhCanonical"]
    ].iterrows():
        if pd.isna(r["_CycleInt"]):
            continue
        key = (
            normalize_battery(r[schema.battery_col]),
            int(r["_CycleInt"]),
            float(r["_RefAhCanonical"]),
        )
        key_to_row[key] = i

    H = len(schema.hi_cols)
    N = len(batteries)
    tensor = np.full(
        (N, len(HISTORY_OFFSETS), len(REFAH_TARGETS), H),
        np.nan,
        dtype=np.float32,
    )

    for n in range(N):
        b = normalize_battery(batteries[n])
        o = int(origins[n])

        for di, off in enumerate(HISTORY_OFFSETS):
            cyc = o + int(off)
            for ri, refah in enumerate(REFAH_TARGETS):
                key = (b, cyc, float(refah))
                if key not in key_to_row:
                    continue
                row_idx = key_to_row[key]
                vals = pd.to_numeric(
                    d.loc[row_idx, schema.hi_cols],
                    errors="coerce",
                ).to_numpy(dtype=float)
                tensor[n, di, ri, :] = vals.astype(np.float32)

    coverage_rows = []
    for j, hi in enumerate(schema.hi_cols):
        x = tensor[:, :, :, j]
        sample_complete = np.isfinite(x).all(axis=(1, 2))

        origin80_mask = origins == TEST_ORIGIN
        coverage_rows.append({
            "HI": hi,
            "FiniteScalarRate_All141": float(np.isfinite(x).mean()),
            "CompleteSampleRate_All141": float(sample_complete.mean()),
            "CompleteSampleCount_All141": int(sample_complete.sum()),
            "CompleteBatteryRate_Origin80": float(
                sample_complete[origin80_mask].mean()
            ),
            "CompleteBatteryCount_Origin80": int(
                sample_complete[origin80_mask].sum()
            ),
            "EligibleCompleteAll141": int(sample_complete.all()),
        })

    cov = pd.DataFrame(coverage_rows)

    # Main search requires complete values across all frozen source origin samples.
    eligible = cov[
        cov["EligibleCompleteAll141"].eq(1)
    ]["HI"].tolist()

    # Original Top2 must be available.
    missing_top2 = [h for h in ORIGINAL_TOP2 if h not in schema.hi_cols]
    if missing_top2:
        raise RuntimeError(
            f"Event table缺少原Top2 HI: {missing_top2}"
        )

    if any(h not in eligible for h in ORIGINAL_TOP2):
        bad = cov[cov["HI"].isin(ORIGINAL_TOP2)]
        raise RuntimeError(
            "原Top2在141个frozen samples中不是100% complete:\n"
            + bad.to_string(index=False)
        )

    if len(eligible) < 5:
        raise RuntimeError(
            f"100% complete合法HI仅{len(eligible)}个，不足Top5"
        )

    idx = [schema.hi_cols.index(h) for h in eligible]
    tensor = tensor[:, :, :, idx]

    return HITensor(
        values=tensor,
        batteries=np.asarray(
            [normalize_battery(x) for x in batteries],
            dtype=object,
        ),
        origins=origins.astype(np.int32),
        hi_names=list(eligible),
        coverage=cov,
    )


# =============================================================================
# 5. Infer original frozen [6,6] layout
# =============================================================================

@dataclass
class FrozenLayout:
    hi_order: Tuple[str, str]
    checkpoint_major: bool
    reverse_history: bool
    max_abs_diff: float
    mean_abs_diff: float


def pair_tensor_to_bundle_layout(
    pair_values: np.ndarray,  # [N,6,3,2], columns in pair_names order
    pair_names: Sequence[str],
    layout: FrozenLayout,
) -> np.ndarray:
    x = np.asarray(pair_values, np.float32)

    # Reorder the two HIs into the original frozen slot semantics.
    name_to_local = {str(h): i for i, h in enumerate(pair_names)}
    # For arbitrary new pair, layout.hi_order names are semantic slots only.
    # The caller passes pair_names ordered [slot0_value, slot1_value] when new.
    if set(pair_names) == set(layout.hi_order):
        order = [name_to_local[h] for h in layout.hi_order]
        x = x[:, :, :, order]
    else:
        # New candidate pair is already supplied in slot order.
        pass

    if layout.reverse_history:
        x = x[:, ::-1, :, :]

    if layout.checkpoint_major:
        # [N,6,3,2] -> [cp0hi0, cp0hi1, cp1hi0, ...]
        y = x.reshape(x.shape[0], x.shape[1], 6)
    else:
        # [N,6,2,3] -> [hi0cp0, hi0cp1, ..., hi1cp2]
        y = np.transpose(x, (0, 1, 3, 2)).reshape(
            x.shape[0], x.shape[1], 6
        )

    return y.astype(np.float32)


def infer_frozen_layout(
    hit: HITensor,
    bundle,
) -> Tuple[FrozenLayout, pd.DataFrame]:
    hi_to_idx = {h: i for i, h in enumerate(hit.hi_names)}
    idx = [hi_to_idx[h] for h in ORIGINAL_TOP2]
    pair = hit.values[:, :, :, idx]

    rows = []
    candidates = []

    for hi_order in (
        ORIGINAL_TOP2,
        tuple(reversed(ORIGINAL_TOP2)),
    ):
        for checkpoint_major in (True, False):
            for reverse_history in (False, True):
                layout = FrozenLayout(
                    hi_order=tuple(hi_order),
                    checkpoint_major=checkpoint_major,
                    reverse_history=reverse_history,
                    max_abs_diff=np.inf,
                    mean_abs_diff=np.inf,
                )

                y = pair_tensor_to_bundle_layout(
                    pair,
                    ORIGINAL_TOP2,
                    layout,
                )
                target = np.asarray(bundle.X, np.float32)

                if y.shape != target.shape:
                    raise RuntimeError(
                        f"reconstruction shape={y.shape} vs bundle={target.shape}"
                    )

                diff = np.abs(y.astype(float) - target.astype(float))
                maxd = float(np.nanmax(diff))
                meand = float(np.nanmean(diff))

                row = {
                    "HIOrder": "|".join(hi_order),
                    "CheckpointMajor": int(checkpoint_major),
                    "ReverseHistory": int(reverse_history),
                    "MaxAbsDiff": maxd,
                    "MeanAbsDiff": meand,
                    "PASS": int(maxd <= RECON_ATOL),
                }
                rows.append(row)

                candidates.append(
                    FrozenLayout(
                        hi_order=tuple(hi_order),
                        checkpoint_major=checkpoint_major,
                        reverse_history=reverse_history,
                        max_abs_diff=maxd,
                        mean_abs_diff=meand,
                    )
                )

    tab = pd.DataFrame(rows).sort_values(
        ["MaxAbsDiff", "MeanAbsDiff"]
    ).reset_index(drop=True)

    best = min(
        candidates,
        key=lambda z: (z.max_abs_diff, z.mean_abs_diff),
    )

    if best.max_abs_diff > RECON_ATOL:
        raise RuntimeError(
            "无法从Event-ready重建V4.11B frozen original Top2输入。"
            f" best max abs diff={best.max_abs_diff:.6g} > {RECON_ATOL}。\n"
            "请检查 --event-ready 是否指向V4.2冻结Event表、Phase是否正确。"
        )

    return best, tab


# =============================================================================
# 6. Pair-block exact V4.10A deterministic engineering
# =============================================================================

def make_pair_bundle_input(
    hit: HITensor,
    hi_a: str,
    hi_b: str,
    layout: FrozenLayout,
) -> np.ndarray:
    hi_to_idx = {h: i for i, h in enumerate(hit.hi_names)}
    if hi_a not in hi_to_idx or hi_b not in hi_to_idx:
        raise KeyError((hi_a, hi_b))

    pair = hit.values[
        :, :, :, [hi_to_idx[hi_a], hi_to_idx[hi_b]]
    ]

    # New pair is intentionally mapped:
    #   first selected HI -> frozen channel slot 0
    #   second selected HI -> frozen channel slot 1
    #
    # pair_tensor_to_bundle_layout only auto-reorders when names are exactly
    # original Top2; use synthetic names here to preserve supplied order.
    synthetic_layout = FrozenLayout(
        hi_order=("__slot0__", "__slot1__"),
        checkpoint_major=layout.checkpoint_major,
        reverse_history=layout.reverse_history,
        max_abs_diff=0.0,
        mean_abs_diff=0.0,
    )
    return pair_tensor_to_bundle_layout(
        pair,
        [hi_a, hi_b],
        synthetic_layout,
    )


def engineer_selected_hi_set(
    hit: HITensor,
    selected: Sequence[str],
    layout: FrozenLayout,
    v410a,
) -> np.ndarray:
    selected = list(selected)
    if len(selected) < 2:
        raise ValueError("selected HI set至少2个")

    anchor = selected[0]
    blocks = []

    for other in selected[1:]:
        Xpair = make_pair_bundle_input(
            hit,
            anchor,
            other,
            layout,
        )
        eng, names, registry = v410a.engineer_features(Xpair)

        if eng.shape[1] != 196:
            raise RuntimeError(
                f"V4.10A pair engineered count={eng.shape[1]} !=196"
            )
        if not np.isfinite(eng).all():
            raise RuntimeError(
                f"pair({anchor},{other}) engineered features存在NaN/Inf"
            )
        blocks.append(np.asarray(eng, np.float32))

    return np.concatenate(blocks, axis=1).astype(np.float32)


# =============================================================================
# 7. Source-only RUL relevance ranking
# =============================================================================

def sample_indices_for_batteries_origins(
    hit: HITensor,
    batteries: Sequence[str],
    origins: Sequence[int],
) -> np.ndarray:
    bm = np.isin(hit.batteries, list(batteries))
    om = np.isin(hit.origins, list(map(int, origins)))
    return np.where(bm & om)[0].astype(np.int32)


def source_origin80_indices(
    hit: HITensor,
    batteries: Sequence[str],
) -> np.ndarray:
    idx = sample_indices_for_batteries_origins(
        hit,
        batteries,
        [TEST_ORIGIN],
    )
    if len(idx) != len(batteries):
        raise RuntimeError(
            f"origin80 sample count={len(idx)} vs batteries={len(batteries)}"
        )
    return idx


def true_rul_for_indices(bundle, idx: Sequence[int]) -> np.ndarray:
    return np.asarray(
        bundle.true_eol_lead[np.asarray(idx, int)],
        dtype=float,
    )


def spearman_abs(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return 0.0

    sx = pd.Series(x[m]).rank(method="average").to_numpy(float)
    sy = pd.Series(y[m]).rank(method="average").to_numpy(float)

    if np.std(sx) < 1e-12 or np.std(sy) < 1e-12:
        return 0.0
    r = np.corrcoef(sx, sy)[0, 1]
    return float(abs(r)) if np.isfinite(r) else 0.0


def rank_his_source_only(
    hit: HITensor,
    bundle,
    source_batteries: Sequence[str],
    shortlist_n: int,
) -> pd.DataFrame:
    idx = source_origin80_indices(
        hit,
        source_batteries,
    )
    y = true_rul_for_indices(bundle, idx)

    rows = []
    for j, hi in enumerate(hit.hi_names):
        x = hit.values[idx, :, :, j].reshape(len(idx), -1)
        rhos = [
            spearman_abs(x[:, k], y)
            for k in range(x.shape[1])
        ]
        strongest = sorted(rhos, reverse=True)[:4]
        score = float(np.mean(strongest))
        rows.append({
            "HI": hi,
            "RelevanceTop4MeanAbsSpearman": score,
            "MaxAbsSpearman": float(max(rhos)),
            "MedianAbsSpearman18": float(np.median(rhos)),
        })

    tab = pd.DataFrame(rows).sort_values(
        [
            "RelevanceTop4MeanAbsSpearman",
            "MaxAbsSpearman",
            "HI",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    tab["Rank"] = np.arange(1, len(tab) + 1)

    keep = tab.head(max(2, int(shortlist_n)))["HI"].tolist()

    for h in ORIGINAL_TOP2:
        if h not in keep:
            keep.append(h)

    tab["Shortlisted"] = tab["HI"].isin(keep).astype(int)
    return tab


# =============================================================================
# 8. RUL XGB + inner CV
# =============================================================================

XGB_RUL_PARAMS = {
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
    sample_weight: np.ndarray,
    threads: int,
):
    from xgboost import XGBRegressor

    p = dict(XGB_RUL_PARAMS)
    p["n_jobs"] = max(1, int(threads))

    model = XGBRegressor(**p)
    model.fit(
        X,
        y,
        sample_weight=sample_weight,
        verbose=False,
    )
    return model


def rul_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    e = yp - yt

    ss_res = float(np.sum(e * e))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    ps = float(np.std(yp))
    ts = float(np.std(yt))

    return {
        "N": len(yt),
        "MAE": float(np.mean(np.abs(e))),
        "MedianAE": float(np.median(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e * e))),
        "Bias": float(np.mean(e)),
        "R2": float(r2),
        "MAPE_pct": float(
            100 * np.mean(np.abs(e) / np.maximum(np.abs(yt), 1e-8))
        ),
        "Within25": float(np.mean(np.abs(e) <= 25)),
        "Within50": float(np.mean(np.abs(e) <= 50)),
        "Within100": float(np.mean(np.abs(e) <= 100)),
        "PredStd": ps,
        "TrueStd": ts,
        "SpreadRatio": float(ps / ts) if ts > 1e-12 else np.nan,
        "SpreadGap": float(abs(1.0 - ps / ts)) if ts > 1e-12 else np.nan,
    }


def inner_cv_selected_set(
    hit: HITensor,
    bundle,
    v410a,
    layout: FrozenLayout,
    selected: Sequence[str],
    source_batteries: Sequence[str],
    groups: Sequence[Sequence[str]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    selected = list(selected)
    Xall = engineer_selected_hi_set(
        hit,
        selected,
        layout,
        v410a,
    )

    rows = []

    for fi, val_bats in enumerate(groups, start=1):
        val_bats = list(val_bats)
        train_bats = [
            b for b in source_batteries
            if b not in set(val_bats)
        ]

        if args.train_origin_mode == "strict":
            train_origins = [80]
        else:
            train_origins = list(SOURCE_ORIGINS)

        tr_idx = sample_indices_for_batteries_origins(
            hit,
            train_bats,
            train_origins,
        )
        va_idx = source_origin80_indices(
            hit,
            val_bats,
        )

        ytr = true_rul_for_indices(bundle, tr_idx)
        yva = true_rul_for_indices(bundle, va_idx)

        if args.train_origin_mode == "strict":
            sw = np.ones(len(tr_idx), dtype=np.float32)
        else:
            # Equal total contribution per source battery:
            # 3 origins -> each weight 1/3.
            sw = np.full(len(tr_idx), 1.0 / 3.0, dtype=np.float32)

        model = fit_xgb_rul(
            Xall[tr_idx],
            ytr,
            sw,
            int(args.xgb_threads),
        )
        pred = np.asarray(
            model.predict(Xall[va_idx]),
            float,
        ).reshape(-1)
        pred = np.clip(pred, 1.0, float(K_TRAJ))

        for j, idx in enumerate(va_idx):
            rows.append({
                "Battery": str(hit.batteries[idx]),
                "InnerFold": fi,
                "TrueRUL": float(yva[j]),
                "PredRUL": float(pred[j]),
            })

        del model
        gc.collect()

    oof = pd.DataFrame(rows).sort_values("Battery").reset_index(drop=True)

    if len(oof) != len(source_batteries):
        raise RuntimeError(
            f"inner OOF rows={len(oof)} vs source={len(source_batteries)}"
        )
    if oof["Battery"].nunique() != len(source_batteries):
        raise RuntimeError("inner OOF battery重复/缺失")

    met = rul_metrics(
        oof["TrueRUL"].to_numpy(float),
        oof["PredRUL"].to_numpy(float),
    )
    return met, oof


# =============================================================================
# 9. Top2 exhaustive + Top3..5 greedy
# =============================================================================

def candidate_key_for_search(met: Dict[str, float], his: Sequence[str]):
    return (
        float(met["MAE"]),
        float(met["RMSE"]),
        float(met["SpreadGap"]),
        "|".join(his),
    )


def search_topk_source_only(
    hit: HITensor,
    bundle,
    v410a,
    layout: FrozenLayout,
    source_batteries: Sequence[str],
    ranking: pd.DataFrame,
    args: argparse.Namespace,
    outer_held: str,
) -> Tuple[
    Dict[int, List[str]],
    pd.DataFrame,
    pd.DataFrame,
]:
    groups = split_battery_groups(
        source_batteries,
        int(args.inner_folds),
    )

    shortlist = ranking[
        ranking["Shortlisted"].eq(1)
    ].sort_values("Rank")["HI"].tolist()

    if len(shortlist) < 5:
        raise RuntimeError("shortlist不足5")

    search_rows = []
    selected_by_k: Dict[int, List[str]] = {}
    summary_rows = []

    # ------------------------------------------------------------------
    # Top2 exhaustive pair search
    # Pair orientation uses source relevance rank: higher-ranked HI first.
    # ------------------------------------------------------------------
    rank_map = dict(zip(ranking["HI"], ranking["Rank"]))

    best_pair = None
    best_pair_key = None
    best_pair_met = None

    for a, b in itertools.combinations(shortlist, 2):
        pair = sorted(
            [a, b],
            key=lambda h: (rank_map.get(h, 10**9), h),
        )

        met, oof = inner_cv_selected_set(
            hit,
            bundle,
            v410a,
            layout,
            pair,
            source_batteries,
            groups,
            args,
        )
        key = candidate_key_for_search(met, pair)

        search_rows.append({
            "OuterHeldoutBattery": outer_held,
            "StepK": 2,
            "CandidateAdded": pair[1],
            "SelectedHISet": "|".join(pair),
            **met,
        })

        if best_pair_key is None or key < best_pair_key:
            best_pair_key = key
            best_pair = list(pair)
            best_pair_met = met

    assert best_pair is not None
    assert best_pair_met is not None

    selected_by_k[2] = list(best_pair)
    summary_rows.append({
        "OuterHeldoutBattery": outer_held,
        "TopK": 2,
        "SelectedHISet": "|".join(best_pair),
        **best_pair_met,
    })

    # ------------------------------------------------------------------
    # Top3..5 greedy forward
    # ------------------------------------------------------------------
    current = list(best_pair)

    for k in (3, 4, 5):
        remaining = [h for h in shortlist if h not in current]
        best = None
        best_key = None
        best_met = None

        for h in remaining:
            cand = current + [h]
            met, oof = inner_cv_selected_set(
                hit,
                bundle,
                v410a,
                layout,
                cand,
                source_batteries,
                groups,
                args,
            )
            key = candidate_key_for_search(met, cand)

            search_rows.append({
                "OuterHeldoutBattery": outer_held,
                "StepK": k,
                "CandidateAdded": h,
                "SelectedHISet": "|".join(cand),
                **met,
            })

            if best_key is None or key < best_key:
                best_key = key
                best = list(cand)
                best_met = met

        assert best is not None
        assert best_met is not None

        current = list(best)
        selected_by_k[k] = list(current)
        summary_rows.append({
            "OuterHeldoutBattery": outer_held,
            "TopK": k,
            "SelectedHISet": "|".join(current),
            **best_met,
        })

    search_tab = pd.DataFrame(search_rows)
    summary = pd.DataFrame(summary_rows).sort_values("TopK")

    return selected_by_k, search_tab, summary


def source_only_choose_pareto_k(
    summary: pd.DataFrame,
    mae_tol: float,
) -> Tuple[int, pd.DataFrame]:
    d = summary.copy().sort_values("TopK")

    # Mark nondominated in (MAE, SpreadGap).
    pareto = []
    for i, r in d.iterrows():
        dominated = False
        for j, s in d.iterrows():
            if i == j:
                continue
            no_worse = (
                float(s["MAE"]) <= float(r["MAE"])
                and float(s["SpreadGap"]) <= float(r["SpreadGap"])
            )
            strictly = (
                float(s["MAE"]) < float(r["MAE"])
                or float(s["SpreadGap"]) < float(r["SpreadGap"])
            )
            if no_worse and strictly:
                dominated = True
                break
        pareto.append(int(not dominated))

    d["ParetoNonDominated"] = pareto

    min_mae = float(d["MAE"].min())
    tol_limit = min_mae * (1.0 + float(mae_tol))
    eligible = d[d["MAE"] <= tol_limit].copy()

    if len(eligible) == 0:
        eligible = d[d["MAE"].eq(min_mae)].copy()

    chosen = eligible.sort_values(
        ["SpreadGap", "MAE", "TopK"]
    ).iloc[0]

    d["WithinMAETolerance"] = (
        d["MAE"] <= tol_limit
    ).astype(int)
    d["ParetoSelected"] = d["TopK"].eq(
        int(chosen["TopK"])
    ).astype(int)

    return int(chosen["TopK"]), d


# =============================================================================
# 10. Final outer fit/predict
# =============================================================================

def fit_predict_outer_set(
    hit: HITensor,
    bundle,
    v410a,
    layout: FrozenLayout,
    selected: Sequence[str],
    source_batteries: Sequence[str],
    held: str,
    args: argparse.Namespace,
) -> float:
    Xall = engineer_selected_hi_set(
        hit,
        selected,
        layout,
        v410a,
    )

    if args.train_origin_mode == "strict":
        train_origins = [80]
    else:
        train_origins = list(SOURCE_ORIGINS)

    tr_idx = sample_indices_for_batteries_origins(
        hit,
        source_batteries,
        train_origins,
    )
    te_idx = source_origin80_indices(
        hit,
        [held],
    )

    ytr = true_rul_for_indices(bundle, tr_idx)

    if args.train_origin_mode == "strict":
        sw = np.ones(len(tr_idx), dtype=np.float32)
    else:
        sw = np.full(len(tr_idx), 1.0 / 3.0, dtype=np.float32)

    model = fit_xgb_rul(
        Xall[tr_idx],
        ytr,
        sw,
        int(args.xgb_threads),
    )
    pred = float(
        np.asarray(
            model.predict(Xall[te_idx]),
            float,
        ).reshape(-1)[0]
    )
    pred = float(np.clip(pred, 1.0, float(K_TRAJ)))

    del model
    gc.collect()
    return pred


# =============================================================================
# 11. One outer fold
# =============================================================================

def fold_signature(args: argparse.Namespace, event_path: Path) -> str:
    return stable_hash({
        "Task": "V412A",
        "EventReady": str(event_path),
        "Phase": args.phase,
        "HistoryOffsets": HISTORY_OFFSETS,
        "RefAhTargets": REFAH_TARGETS,
        "OriginalTop2": ORIGINAL_TOP2,
        "Shortlist": int(args.shortlist),
        "InnerFolds": int(args.inner_folds),
        "ParetoMAETol": float(args.pareto_mae_tol),
        "TrainOriginMode": args.train_origin_mode,
        "XGB": XGB_RUL_PARAMS,
    })


def run_one_outer_fold(
    hit: HITensor,
    bundle,
    v410a,
    layout: FrozenLayout,
    baseline: pd.DataFrame,
    batteries: Sequence[str],
    held: str,
    event_path: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    cp = fold_cache_path(args.output_root, held)
    sig = fold_signature(args, event_path)

    if cp.exists() and not args.force:
        try:
            z = np.load(cp, allow_pickle=False)
            if str(z["signature"][0]) == sig:
                result = json.loads(str(z["result_json"][0]))
                print(f"[V412A] {held}: cache", flush=True)
                return result
        except Exception:
            pass

    t0 = time.perf_counter()
    source = [b for b in batteries if b != held]

    ranking = rank_his_source_only(
        hit,
        bundle,
        source,
        int(args.shortlist),
    )
    ranking.insert(0, "OuterHeldoutBattery", held)

    selected_by_k, search_tab, summary = search_topk_source_only(
        hit,
        bundle,
        v410a,
        layout,
        source,
        ranking,
        args,
        held,
    )

    pareto_k, pareto_summary = source_only_choose_pareto_k(
        summary,
        float(args.pareto_mae_tol),
    )

    pred_by_k = {}
    for k in TOPK_VALUES:
        pred_by_k[k] = fit_predict_outer_set(
            hit,
            bundle,
            v410a,
            layout,
            selected_by_k[k],
            source,
            held,
            args,
        )

    test_idx = source_origin80_indices(hit, [held])
    true_rul = float(bundle.true_eol_lead[test_idx[0]])

    brow = baseline[
        baseline["OuterHeldoutBattery"].eq(held)
    ]
    if len(brow) != 1:
        raise RuntimeError(f"{held}: V4.11E baseline row !=1")
    base_pred = float(brow.iloc[0]["DirectRUL"])

    result = {
        "OuterHeldoutBattery": held,
        "TrueRUL": true_rul,
        "V411E_DirectRUL": base_pred,

        "SelectedTop2": "|".join(selected_by_k[2]),
        "SelectedTop3": "|".join(selected_by_k[3]),
        "SelectedTop4": "|".join(selected_by_k[4]),
        "SelectedTop5": "|".join(selected_by_k[5]),

        "Pred_Top2": pred_by_k[2],
        "Pred_Top3": pred_by_k[3],
        "Pred_Top4": pred_by_k[4],
        "Pred_Top5": pred_by_k[5],

        "ParetoSelectedK": pareto_k,
        "ParetoSelectedHISet": "|".join(selected_by_k[pareto_k]),
        "Pred_ParetoSelected": pred_by_k[pareto_k],

        "InnerCV_Top2_MAE": float(
            summary.loc[summary["TopK"].eq(2), "MAE"].iloc[0]
        ),
        "InnerCV_Top3_MAE": float(
            summary.loc[summary["TopK"].eq(3), "MAE"].iloc[0]
        ),
        "InnerCV_Top4_MAE": float(
            summary.loc[summary["TopK"].eq(4), "MAE"].iloc[0]
        ),
        "InnerCV_Top5_MAE": float(
            summary.loc[summary["TopK"].eq(5), "MAE"].iloc[0]
        ),

        "InnerCV_Top2_SpreadRatio": float(
            summary.loc[summary["TopK"].eq(2), "SpreadRatio"].iloc[0]
        ),
        "InnerCV_Top3_SpreadRatio": float(
            summary.loc[summary["TopK"].eq(3), "SpreadRatio"].iloc[0]
        ),
        "InnerCV_Top4_SpreadRatio": float(
            summary.loc[summary["TopK"].eq(4), "SpreadRatio"].iloc[0]
        ),
        "InnerCV_Top5_SpreadRatio": float(
            summary.loc[summary["TopK"].eq(5), "SpreadRatio"].iloc[0]
        ),

        "Runtime_s": float(time.perf_counter() - t0),

        "HeldoutUsedForHIRanking": 0,
        "HeldoutUsedForHISelection": 0,
        "HeldoutUsedForKSelection": 0,
        "HeldoutUsedForTraining": 0,
        "HistoricalSOHAsFeature": 0,
        "CapacityHistoryAsFeature": 0,
        "FutureHIAsFeature": 0,
        "CycleAsFeature": 0,
        "TrueEOLRULAsInput": 0,
        "TestBatteryCalibration": 0,
    }

    ensure_dir(cp.parent)
    np.savez_compressed(
        cp,
        signature=np.asarray([sig], dtype="U64"),
        result_json=np.asarray(
            [json.dumps(result, ensure_ascii=False, default=str)],
            dtype="U200000",
        ),
        ranking_json=np.asarray(
            [ranking.to_json(orient="records")],
            dtype="U1000000",
        ),
        search_json=np.asarray(
            [search_tab.to_json(orient="records")],
            dtype="U3000000",
        ),
        pareto_summary_json=np.asarray(
            [pareto_summary.to_json(orient="records")],
            dtype="U500000",
        ),
    )

    print(
        f"[V412A] {held}: "
        f"Top2={selected_by_k[2]} "
        f"Top5={selected_by_k[5]} "
        f"ParetoK={pareto_k} | "
        f"true={true_rul:.0f} "
        f"base={base_pred:.1f} "
        f"T2={pred_by_k[2]:.1f} "
        f"T3={pred_by_k[3]:.1f} "
        f"T4={pred_by_k[4]:.1f} "
        f"T5={pred_by_k[5]:.1f} "
        f"P={pred_by_k[pareto_k]:.1f} "
        f"runtime={result['Runtime_s']:.1f}s",
        flush=True,
    )

    return result


# =============================================================================
# 12. Run orchestration / workers
# =============================================================================

def run_selected_folds(
    hit,
    bundle,
    v410a,
    layout,
    baseline,
    batteries,
    event_path,
    args,
) -> List[Dict[str, object]]:
    todo = selected_worker_batteries(
        batteries,
        args,
    )

    results = []
    for held in todo:
        results.append(
            run_one_outer_fold(
                hit,
                bundle,
                v410a,
                layout,
                baseline,
                batteries,
                held,
                event_path,
                args,
            )
        )

    tag = (
        f"worker{args.worker_id}of{args.num_workers}"
        if args.stage == "worker"
        else "run"
    )
    pd.DataFrame(results).to_csv(
        args.output_root / f"00_runtime_{tag}.csv",
        index=False,
        encoding=ENC,
    )
    return results


def all_fold_caches_ready(
    out: Path,
    batteries: Sequence[str],
) -> Tuple[int, List[str]]:
    missing = [
        b for b in batteries
        if not fold_cache_path(out, b).exists()
    ]
    return len(batteries) - len(missing), missing


def wait_for_all_caches(
    out: Path,
    batteries: Sequence[str],
    args: argparse.Namespace,
) -> None:
    poll = max(5, int(args.wait_poll_seconds))
    timeout_h = float(args.wait_timeout_hours)
    t0 = time.time()
    last = -1

    while True:
        ready, missing = all_fold_caches_ready(out, batteries)
        if ready != last:
            print(
                f"[worker wait] fold caches ready {ready}/{len(batteries)}",
                flush=True,
            )
            last = ready

        pd.DataFrame({
            "Battery": batteries,
            "Ready": [
                int(fold_cache_path(out, b).exists())
                for b in batteries
            ],
        }).to_csv(
            out / "00_worker_wait_status.csv",
            index=False,
            encoding=ENC,
        )

        if not missing:
            return

        if timeout_h > 0 and (time.time() - t0) / 3600 >= timeout_h:
            raise TimeoutError(
                f"等待fold caches超过{timeout_h:g}h，missing={missing}"
            )
        time.sleep(poll)


# =============================================================================
# 13. Finalization
# =============================================================================

def load_fold_artifacts(
    out: Path,
    batteries: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = []
    rankings = []
    searches = []
    summaries = []

    for b in batteries:
        cp = fold_cache_path(out, b)
        if not cp.exists():
            raise FileNotFoundError(cp)
        z = np.load(cp, allow_pickle=False)

        results.append(
            json.loads(str(z["result_json"][0]))
        )
        rankings.append(
            pd.DataFrame(
                json.loads(str(z["ranking_json"][0]))
            )
        )
        searches.append(
            pd.DataFrame(
                json.loads(str(z["search_json"][0]))
            )
        )
        summaries.append(
            pd.DataFrame(
                json.loads(str(z["pareto_summary_json"][0]))
            )
        )

    return (
        pd.DataFrame(results),
        pd.concat(rankings, ignore_index=True),
        pd.concat(searches, ignore_index=True),
        pd.concat(summaries, ignore_index=True),
    )


def method_metrics_from_outer(
    outer: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mapping = {
        METHOD_BASELINE: "V411E_DirectRUL",
        "RULSpecific_Top2": "Pred_Top2",
        "RULSpecific_Top3": "Pred_Top3",
        "RULSpecific_Top4": "Pred_Top4",
        "RULSpecific_Top5": "Pred_Top5",
        "RULSpecific_ParetoSelected": "Pred_ParetoSelected",
    }

    yt = outer["TrueRUL"].to_numpy(float)
    rows = []
    errors = []

    for method, col in mapping.items():
        yp = outer[col].to_numpy(float)
        met = rul_metrics(yt, yp)
        rows.append({
            "Method": method,
            **met,
        })

        for i, b in enumerate(outer["OuterHeldoutBattery"]):
            errors.append({
                "Battery": b,
                "Method": method,
                "TrueRUL": float(yt[i]),
                "PredRUL": float(yp[i]),
                "Error": float(yp[i] - yt[i]),
                "AbsError": float(abs(yp[i] - yt[i])),
            })

    return (
        pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True),
        pd.DataFrame(errors),
    )


def life_bin_metrics(
    outer: pd.DataFrame,
) -> pd.DataFrame:
    mapping = {
        METHOD_BASELINE: "V411E_DirectRUL",
        "RULSpecific_Top2": "Pred_Top2",
        "RULSpecific_Top3": "Pred_Top3",
        "RULSpecific_Top4": "Pred_Top4",
        "RULSpecific_Top5": "Pred_Top5",
        "RULSpecific_ParetoSelected": "Pred_ParetoSelected",
    }

    yt = outer["TrueRUL"].to_numpy(float)
    q33 = float(np.quantile(yt, 1 / 3))
    q67 = float(np.quantile(yt, 2 / 3))

    rows = []
    for bin_name, mask in (
        ("ShortRUL", yt <= q33),
        ("MiddleRUL", (yt > q33) & (yt <= q67)),
        ("LongRUL", yt > q67),
    ):
        for method, col in mapping.items():
            met = rul_metrics(
                yt[mask],
                outer[col].to_numpy(float)[mask],
            )
            rows.append({
                "LifeBin": bin_name,
                "Q33": q33,
                "Q67": q67,
                "Method": method,
                **met,
            })
    return pd.DataFrame(rows)


def hi_selection_frequency(
    outer: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for k in TOPK_VALUES:
        col = f"SelectedTop{k}"
        for r in outer.itertuples(index=False):
            his = str(getattr(r, col)).split("|")
            for rank, hi in enumerate(his, start=1):
                rows.append({
                    "TopK": k,
                    "Battery": r.OuterHeldoutBattery,
                    "HI": hi,
                    "SelectionRank": rank,
                })

    d = pd.DataFrame(rows)
    out = (
        d.groupby(["TopK", "HI"])
        .agg(
            OuterFoldCount=("Battery", "nunique"),
            MeanSelectionRank=("SelectionRank", "mean"),
        )
        .reset_index()
    )
    out["SelectionRate"] = out["OuterFoldCount"] / EXPECTED_BATTERIES
    return out.sort_values(
        ["TopK", "OuterFoldCount", "MeanSelectionRank"],
        ascending=[True, False, True],
    )


def gain_vs_baseline(metrics: pd.DataFrame) -> pd.DataFrame:
    b = metrics[
        metrics["Method"].eq(METHOD_BASELINE)
    ].iloc[0]

    rows = []
    for _, r in metrics.iterrows():
        if r["Method"] == METHOD_BASELINE:
            continue
        rows.append({
            "Model": r["Method"],
            "MAE": r["MAE"],
            "BaselineMAE": b["MAE"],
            "Gain_MAE_pct": (
                100 * (b["MAE"] - r["MAE"]) / b["MAE"]
            ),
            "RMSE": r["RMSE"],
            "BaselineRMSE": b["RMSE"],
            "Gain_RMSE_pct": (
                100 * (b["RMSE"] - r["RMSE"]) / b["RMSE"]
            ),
            "SpreadRatio": r["SpreadRatio"],
            "BaselineSpreadRatio": b["SpreadRatio"],
            "SpreadRatioDelta": r["SpreadRatio"] - b["SpreadRatio"],
            "R2": r["R2"],
            "MAPE_pct": r["MAPE_pct"],
        })
    return pd.DataFrame(rows).sort_values(
        "Gain_MAE_pct",
        ascending=False,
    )


def protocol_audit(
    outer: pd.DataFrame,
    recon: pd.DataFrame,
    layout: FrozenLayout,
    hit: HITensor,
) -> pd.DataFrame:
    rows = []

    def add(check, observed, expected, ok):
        rows.append({
            "Check": check,
            "Observed": observed,
            "Expected": expected,
            "PASS": int(bool(ok)),
        })

    add("Outer folds", len(outer), 47, len(outer) == 47)
    add(
        "Unique batteries",
        outer["OuterHeldoutBattery"].nunique(),
        47,
        outer["OuterHeldoutBattery"].nunique() == 47,
    )
    add(
        "Original Top2 reconstruction max abs diff",
        layout.max_abs_diff,
        f"<= {RECON_ATOL}",
        layout.max_abs_diff <= RECON_ATOL,
    )
    add(
        "Eligible complete HI count",
        len(hit.hi_names),
        ">=5",
        len(hit.hi_names) >= 5,
    )

    for c in (
        "HeldoutUsedForHIRanking",
        "HeldoutUsedForHISelection",
        "HeldoutUsedForKSelection",
        "HeldoutUsedForTraining",
        "HistoricalSOHAsFeature",
        "CapacityHistoryAsFeature",
        "FutureHIAsFeature",
        "CycleAsFeature",
        "TrueEOLRULAsInput",
        "TestBatteryCalibration",
    ):
        s = int(
            pd.to_numeric(
                outer[c],
                errors="coerce",
            ).fillna(1).sum()
        )
        add(c, s, 0, s == 0)

    for c in (
        "Pred_Top2",
        "Pred_Top3",
        "Pred_Top4",
        "Pred_Top5",
        "Pred_ParetoSelected",
    ):
        ok = np.isfinite(
            pd.to_numeric(
                outer[c],
                errors="coerce",
            ).to_numpy(float)
        ).all()
        add(f"{c} finite", int(ok), 1, ok)

    return pd.DataFrame(rows)


def make_figures(
    outer,
    metrics,
    life,
    hi_freq,
    out,
    dpi,
) -> None:
    figdir = ensure_dir(out / "Figures")

    # 1 MAE
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    q = metrics.sort_values("MAE")
    ax.bar(np.arange(len(q)), q["MAE"].to_numpy(float))
    ax.set_xticks(np.arange(len(q)))
    ax.set_xticklabels(q["Method"], rotation=28, ha="right")
    ax.set_ylabel("RUL MAE (cycles)")
    ax.set_title("V4.12A outer-LOBO TopK comparison")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(figdir / "01_topk_mae.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 2 MAE vs spread
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for _, r in metrics.iterrows():
        ax.scatter(r["SpreadRatio"], r["MAE"], s=55)
        ax.annotate(
            r["Method"],
            (r["SpreadRatio"], r["MAE"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction spread / true spread")
    ax.set_ylabel("RUL MAE (cycles)")
    ax.set_title("Accuracy vs RUL spread")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figdir / "02_mae_vs_spread.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 3 HI frequency for Top5
    q = hi_freq[hi_freq["TopK"].eq(5)].head(20)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.barh(
        np.arange(len(q)),
        q["OuterFoldCount"].to_numpy(),
    )
    ax.set_yticks(np.arange(len(q)))
    ax.set_yticklabels(q["HI"])
    ax.invert_yaxis()
    ax.set_xlabel("Selected outer-fold count")
    ax.set_title("Top5 RUL-specific HI selection frequency")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(figdir / "03_hi_frequency.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 4 champion true vs pred
    champ = metrics.sort_values(["MAE", "RMSE"]).iloc[0]["Method"]
    colmap = {
        METHOD_BASELINE: "V411E_DirectRUL",
        "RULSpecific_Top2": "Pred_Top2",
        "RULSpecific_Top3": "Pred_Top3",
        "RULSpecific_Top4": "Pred_Top4",
        "RULSpecific_Top5": "Pred_Top5",
        "RULSpecific_ParetoSelected": "Pred_ParetoSelected",
    }
    yt = outer["TrueRUL"].to_numpy(float)
    yp = outer[colmap[champ]].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(yt, yp, alpha=.8)
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"Champion: {champ}")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(
        figdir / "04_true_vs_pred_champion.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 5 life bins
    methods = list(metrics["Method"])
    bins = ["ShortRUL", "MiddleRUL", "LongRUL"]
    x = np.arange(len(bins))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for j, method in enumerate(methods):
        q = life[life["Method"].eq(method)].set_index("LifeBin")
        vals = [float(q.loc[b, "MAE"]) for b in bins]
        ax.bar(
            x + (j - (len(methods)-1)/2) * width,
            vals,
            width=width,
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("MAE (cycles)")
    ax.set_title("RUL error by lifetime range")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(figdir / "05_life_bin_mae.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def finalize(
    hit,
    bundle,
    layout,
    recon,
    batteries,
    args,
) -> None:
    outer, rankings, searches, summaries = load_fold_artifacts(
        args.output_root,
        batteries,
    )
    outer = outer.sort_values(
        "OuterHeldoutBattery"
    ).reset_index(drop=True)

    outer.to_csv(
        args.output_root / "07_outer_predictions_by_topk.csv",
        index=False,
        encoding=ENC,
    )
    rankings.to_csv(
        args.output_root / "04_hi_relevance_by_outer_fold.csv",
        index=False,
        encoding=ENC,
    )
    searches.to_csv(
        args.output_root / "05_pair_and_greedy_search_audit.csv",
        index=False,
        encoding=ENC,
    )
    summaries.to_csv(
        args.output_root / "06_selected_hi_by_outer_fold.csv",
        index=False,
        encoding=ENC,
    )

    metrics, errors = method_metrics_from_outer(outer)
    metrics.to_csv(
        args.output_root / "08_topk_outer_metrics.csv",
        index=False,
        encoding=ENC,
    )
    errors.to_csv(
        args.output_root / "08b_outer_error_by_battery.csv",
        index=False,
        encoding=ENC,
    )

    life = life_bin_metrics(outer)
    life.to_csv(
        args.output_root / "09_topk_life_bin_metrics.csv",
        index=False,
        encoding=ENC,
    )

    hi_freq = hi_selection_frequency(outer)
    hi_freq.to_csv(
        args.output_root / "10_hi_selection_frequency.csv",
        index=False,
        encoding=ENC,
    )

    gains = gain_vs_baseline(metrics)
    gains.to_csv(
        args.output_root / "11_gain_vs_v411e.csv",
        index=False,
        encoding=ENC,
    )

    pareto_freq = (
        outer.groupby("ParetoSelectedK")
        .size()
        .reset_index(name="OuterFoldCount")
        .sort_values("ParetoSelectedK")
    )
    pareto_freq["SelectionRate"] = (
        pareto_freq["OuterFoldCount"] / EXPECTED_BATTERIES
    )
    pareto_freq.to_csv(
        args.output_root / "12_pareto_selection_frequency.csv",
        index=False,
        encoding=ENC,
    )

    protocol = protocol_audit(
        outer,
        recon,
        layout,
        hit,
    )
    protocol.to_csv(
        args.output_root / "13_source_only_protocol_audit.csv",
        index=False,
        encoding=ENC,
    )

    protocol.to_csv(
        args.output_root / "14_acceptance_gate.csv",
        index=False,
        encoding=ENC,
    )

    if not args.skip_figures:
        make_figures(
            outer,
            metrics,
            life,
            hi_freq,
            args.output_root,
            int(args.dpi),
        )

    status = (
        "PASS"
        if len(protocol) > 0 and protocol["PASS"].eq(1).all()
        else "CHECK"
    )

    champion = metrics.sort_values(
        ["MAE", "RMSE"]
    ).iloc[0].to_dict()

    baseline = metrics[
        metrics["Method"].eq(METHOD_BASELINE)
    ].iloc[0].to_dict()

    hi_top = (
        hi_freq[hi_freq["TopK"].eq(5)]
        .head(15)
        .to_dict("records")
    )

    complete = {
        "Task": "RWTH48 V4.12A RUL-specific Event HI TopK Pareto Search",
        "Status": status,
        "EventReady": str(args.event_ready_resolved),
        "Phase": args.phase,
        "HistoryOffsets": list(HISTORY_OFFSETS),
        "RefAhTargets": list(REFAH_TARGETS),
        "OriginalTop2": list(ORIGINAL_TOP2),
        "EligibleCompleteHI": len(hit.hi_names),
        "FrozenLayout": {
            "HIOrder": list(layout.hi_order),
            "CheckpointMajor": layout.checkpoint_major,
            "ReverseHistory": layout.reverse_history,
            "MaxAbsDiff": layout.max_abs_diff,
        },
        "TrainOriginMode": args.train_origin_mode,
        "Shortlist": int(args.shortlist),
        "InnerFolds": int(args.inner_folds),
        "ParetoMAETolerance": float(args.pareto_mae_tol),
        "Baseline": baseline,
        "Champion": champion,
        "Methods": metrics.to_dict("records"),
        "GainVsV411E": gains.to_dict("records"),
        "ParetoKFrequency": pareto_freq.to_dict("records"),
        "Top5HIFrequencyTop15": hi_top,
        "HistoricalSOHAsFeature": False,
        "CapacityHistoryAsFeature": False,
        "FutureHIAsFeature": False,
        "CycleAsFeature": False,
        "TrueEOLRULAsInput": False,
        "HeldoutUsedForHIRanking": False,
        "HeldoutUsedForHISelection": False,
        "HeldoutUsedForKSelection": False,
        "TestBatteryCalibration": False,
        "Next": (
            "If any RUL-specific TopK beats the exact V4.11E DirectRUL baseline "
            "with equal or improved spread ratio, freeze the recurrently selected HI set "
            "and run V4.12B final Direct-RUL protocol confirmation. "
            "If no TopK beats baseline, treat the ~62-cycle RUL error as an "
            "information-limit signal under the current sparse Event measurement budget."
        ),
    }

    save_json(
        complete,
        args.output_root / "99_RWTH48_V4_12A_COMPLETE.json",
    )

    print("\n" + "=" * 120)
    print("RWTH48 V4.12A FINALIZE")
    print("=" * 120)
    print("Status:", status)
    print("\nOuter metrics:")
    print(metrics.to_string(index=False))
    print("\nGain vs V4.11E:")
    print(gains.to_string(index=False))
    print("\nPareto K frequency:")
    print(pareto_freq.to_string(index=False))
    print("\nTop5 HI frequency:")
    print(
        hi_freq[hi_freq["TopK"].eq(5)]
        .head(20)
        .to_string(index=False)
    )
    print("\nAcceptance:")
    print(protocol.to_string(index=False))
    print("\nOutput:", args.output_root)
    print("=" * 120)


# =============================================================================
# 14. Preflight
# =============================================================================

def preflight(args: argparse.Namespace):
    helper = load_module(
        args.v411b_helper,
        "rwth48_v411b_for_v412a",
    )
    v411b_args = make_v411b_args(args)
    bundle, a2_complete, handoff_audit = helper.load_handoff(v411b_args)
    v410a = helper.load_v410a_helper(args.v410a_helper)

    required_bundle_attrs = (
        "X",
        "y",
        "mask",
        "battery",
        "origin",
        "true_eol_lead",
        "true_eol_cycle",
        "select",
    )
    missing = [
        a for a in required_bundle_attrs
        if not hasattr(bundle, a)
    ]
    if missing:
        raise RuntimeError(
            "V4.11B FullLifeBundle schema不匹配，缺少: "
            + ", ".join(missing)
        )

    if np.asarray(bundle.X).shape != (
        EXPECTED_BATTERIES * 3,
        6,
        6,
    ):
        raise RuntimeError(
            f"bundle.X shape={np.asarray(bundle.X).shape}, "
            "预期(141,6,6)"
        )

    dcomp = load_json(args.v411d_complete)
    if dcomp.get("Status") != "PASS":
        raise RuntimeError("V4.11D COMPLETE不是PASS")

    ecomp = load_json(args.v411e_complete)
    # CHECK from old 1e-12 tolerance is acceptable, but provenance must be clean.
    provenance_keys_e = (
        "HistoricalSOHAsFeature",
        "CapacityHistoryAsFeature",
        "FutureHIAsFeature",
        "CycleAsFeature",
        "TrueEOLRULAsInput",
        "HeldoutOrigin120_160Used",
        "OuterHeldoutUsedForRULHeadSelection",
        "OuterHeldoutUsedForFusionAlphaSelection",
        "GlobalSourceOOFUsedForFusionTuning",
    )
    prov_rows = []
    for key in provenance_keys_e:
        obs = bool(ecomp.get(key, True))
        prov_rows.append({
            "Check": f"V411E_{key}",
            "Observed": obs,
            "Expected": False,
            "PASS": int(obs is False),
        })

    # V4.11F negative-ablation completion is informative but not mandatory.
    if args.v411f_complete.exists():
        fcomp = load_json(args.v411f_complete)
        prov_rows.append({
            "Check": "V411F_completion_status",
            "Observed": fcomp.get("Status"),
            "Expected": "PASS",
            "PASS": int(fcomp.get("Status") == "PASS"),
        })

    baseline = load_v411e_baseline(args.v411e_outer)

    event_path = locate_event_ready(args)
    args.event_ready_resolved = event_path

    print(f"[V412A] Event-ready: {event_path}", flush=True)
    event_raw = pd.read_csv(
        event_path,
        compression="infer",
        low_memory=False,
    )
    schema = detect_event_schema(event_raw, args)

    print(
        "[V412A] Detected Event schema | "
        f"Battery={schema.battery_col!r} | "
        f"Cycle={schema.cycle_col!r} | "
        f"RefAhNodeCoordinate={schema.refah_col!r} | "
        f"Phase={schema.phase_col!r} | "
        f"Platform={schema.platform_col!r} | "
        f"NumericHI={len(schema.hi_cols)}",
        flush=True,
    )

    # The V4.2 06_EVENT_REFAH_FROZEN_LABEL_READY file can already be
    # Event-specific and may legitimately contain no explicit Phase column.
    # In that case phase_col=None and no phase filtering is applied.
    event = filter_event_table(
        event_raw,
        schema,
        args,
    )

    hit = build_hi_tensor(
        event,
        schema,
        bundle,
        args,
    )

    layout, recon = infer_frozen_layout(
        hit,
        bundle,
    )

    # Exact true RUL agreement with V4.11E.
    bmap = baseline.set_index("OuterHeldoutBattery")
    bats = sorted(set(map(normalize_battery, bundle.battery.tolist())))
    for b in bats:
        idx = np.where(
            (hit.batteries == b)
            & (hit.origins == TEST_ORIGIN)
        )[0]
        if len(idx) != 1:
            raise RuntimeError(f"{b}: origin80 sample !=1")
        y = float(bundle.true_eol_lead[idx[0]])
        y0 = float(bmap.loc[b, "TrueRUL"])
        prov_rows.append({
            "Check": f"{b}_TrueRUL_match_V411E",
            "Observed": y,
            "Expected": y0,
            "PASS": int(np.isclose(y, y0, atol=1e-9, rtol=0)),
        })

    prov = pd.DataFrame(prov_rows)
    if not prov["PASS"].eq(1).all():
        raise RuntimeError(
            "V4.12A provenance FAIL:\n"
            + prov[prov["PASS"].eq(0)].to_string(index=False)
        )

    schema_audit = pd.DataFrame([
        {
            "EventReady": str(event_path),
            "RawRows": len(event_raw),
            "FilteredRows": len(event),
            "BatteryColumn": schema.battery_col,
            "CycleColumn": schema.cycle_col,
            "RefAhColumn": schema.refah_col,
            "RefAhSemanticRule": (
                "PlatformNode/NodeValue frozen Event node; "
                "ReferenceCapacity_Ah is metadata, not checkpoint"
            ),
            "PhaseColumn": schema.phase_col,
            "PlatformColumn": schema.platform_col,
            "RequestedPhase": args.phase,
            "DetectedNumericHICandidates": len(schema.hi_cols),
            "CompleteEligibleHI": len(hit.hi_names),
            "BatteriesFiltered": event[schema.battery_col].nunique(),
        }
    ])

    return (
        helper,
        v410a,
        bundle,
        baseline,
        hit,
        layout,
        recon,
        handoff_audit,
        prov,
        schema_audit,
    )


# =============================================================================
# 15. Main
# =============================================================================

def main() -> None:
    args = parse_args()
    args.output_root = ensure_dir(Path(args.output_root))
    t0 = time.perf_counter()

    (
        helper,
        v410a,
        bundle,
        baseline,
        hit,
        layout,
        recon,
        handoff_audit,
        prov,
        schema_audit,
    ) = preflight(args)

    # Audits
    pd.concat(
        [
            handoff_audit.assign(Block="V411A2_Handoff"),
            prov.assign(Block="UpstreamProvenance"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(
        args.output_root / "00_preflight.csv",
        index=False,
        encoding=ENC,
    )

    schema_audit.to_csv(
        args.output_root / "01_event_input_schema_audit.csv",
        index=False,
        encoding=ENC,
    )
    recon.to_csv(
        args.output_root / "02_original_top2_reconstruction_audit.csv",
        index=False,
        encoding=ENC,
    )
    hit.coverage.to_csv(
        args.output_root / "03_candidate_hi_coverage.csv",
        index=False,
        encoding=ENC,
    )

    save_json(
        {
            "Task": "RWTH48 V4.12A",
            "Stage": args.stage,
            "EventReady": str(args.event_ready_resolved),
            "Phase": args.phase,
            "HistoryOffsets": list(HISTORY_OFFSETS),
            "RefAhTargets": list(REFAH_TARGETS),
            "OriginalTop2": list(ORIGINAL_TOP2),
            "EligibleCompleteHI": len(hit.hi_names),
            "Shortlist": int(args.shortlist),
            "InnerFolds": int(args.inner_folds),
            "ParetoMAETolerance": float(args.pareto_mae_tol),
            "TrainOriginMode": args.train_origin_mode,
            "FrozenLayout": {
                "HIOrder": list(layout.hi_order),
                "CheckpointMajor": layout.checkpoint_major,
                "ReverseHistory": layout.reverse_history,
                "MaxAbsDiff": layout.max_abs_diff,
            },
            "XGB": XGB_RUL_PARAMS,
            "HistoricalSOHAsFeature": False,
            "CapacityHistoryAsFeature": False,
            "FutureHIAsFeature": False,
            "CycleAsFeature": False,
            "TrueEOLRULAsInput": False,
            "TestBatteryCalibration": False,
        },
        args.output_root / "00_config.json",
    )

    batteries = sorted(
        set(
            hit.batteries[
                hit.origins == TEST_ORIGIN
            ].tolist()
        )
    )
    if len(batteries) != EXPECTED_BATTERIES:
        raise RuntimeError(f"origin80 batteries={len(batteries)} !=47")

    print("=" * 120)
    print("RWTH48 V4.12A | RUL-specific Event HI TopK Pareto Search")
    print("Stage                  :", args.stage)
    print("Event-ready            :", args.event_ready_resolved)
    print("Phase                  :", args.phase)
    print("Batteries              :", len(batteries))
    print("Complete eligible HIs  :", len(hit.hi_names))
    print("Shortlist              :", args.shortlist)
    print("TopK                   :", TOPK_VALUES)
    print("Inner battery folds    :", args.inner_folds)
    print("Train origin mode      :", args.train_origin_mode)
    print("Pareto MAE tolerance   :", args.pareto_mae_tol)
    print("Frozen layout          :", layout)
    print("=" * 120, flush=True)

    if args.stage == "audit":
        print("AUDIT PASS")
        return

    if args.stage in ("run", "all", "worker"):
        run_selected_folds(
            hit,
            bundle,
            v410a,
            layout,
            baseline,
            batteries,
            args.event_ready_resolved,
            args,
        )

    if args.only_battery:
        print(
            "\nSingle-battery smoke complete; full finalize skipped.",
            flush=True,
        )
        return

    if args.stage == "worker":
        save_json(
            {
                "WorkerID": int(args.worker_id),
                "NumWorkers": int(args.num_workers),
                "Status": "DONE",
                "Batteries": selected_worker_batteries(batteries, args),
            },
            args.output_root
            / f"98_WORKER_{args.worker_id}_OF_{args.num_workers}_COMPLETE.json",
        )

        if args.auto_finalize:
            wait_for_all_caches(
                args.output_root,
                batteries,
                args,
            )
            finalize(
                hit,
                bundle,
                layout,
                recon,
                batteries,
                args,
            )
        return

    if args.stage in ("finalize", "all"):
        finalize(
            hit,
            bundle,
            layout,
            recon,
            batteries,
            args,
        )

    print(
        f"\nV4.12A stage {args.stage} elapsed="
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

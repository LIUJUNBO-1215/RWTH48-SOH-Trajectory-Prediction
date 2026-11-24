# Two-scale diagnostic resource allocation for lithium-ion battery SOH trajectory prediction
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24509121.svg)](https://doi.org/10.5281/zenodo.24509121)

**Author:** Junbo Liu  
**Affiliation:** School of Electrical Engineering, Chongqing University, Chongqing, China  
**Research data source:** [RWTH Aachen University battery dataset](https://doi.org/10.18154/RWTH-2021-04545)

## About this repository

This repository accompanies a manuscript intended for the *Journal of Energy Storage*. It preserves the existing analysis source scripts, selected derived data, fold-wise experiments, summary and statistical outputs, and **all ten final manuscript figures** (Figs. 1–8 and A.1–A.2). The study uses 47 held-out cells and examines RefAhFraction and FixedTime measurement representations, equal-budget observation schedules, joint resource selection, and adaptive PRT measurements.

**Scope of reproducibility:** The archived CSV results can be inspected and cross-checked; 25 source scripts compile successfully and are provided for inspection and research reuse. **The repository has not been independently rerun end-to-end from raw RWTH waveforms.** Original V4 preprocessing caches, exact dependency versions, and complete per-prediction-point arrays are not all available. In particular, keeping finished figure PNGs does not establish that each figure can be regenerated directly from the retained CSVs. See [`docs/reproduction_status.md`](docs/reproduction_status.md).

## Directory layout

| Directory | Contents and purpose |
|---|---|
| `src/` | 18 primary research Python programs (01–18), plus seven recovered helper modules (19–25): SOH preprocessing, partial-cycle HI extraction, ranking/ablation, XGBoost and SafeResidualMoE models, observation scheduling, resource optimization, adaptive PRT, and statistical postprocessing. Each is a retained historical script rather than a guaranteed stage of a portable one-command pipeline. |
| `configs/` | Recovered frozen experimental configurations for preprocessing and timing/resource studies. Local machine paths were redacted, while scientific settings were retained. These are configuration snapshots, not automatically loaded by every script. |
| `data/tables/` | Eight machine-readable tables named `Table_S1.csv`–`Table_S8.csv`. Names are retained to match manuscript/archive citations; these are **not** a new standalone Supplementary Information document. |
| `data/rankings/` | Complete 141 (47 cells × 3 horizons) source-only HI ranking groups with 73 indicators/group; selected-HI manifest and rank-frequency aggregation. |
| `data/features/` | Candidate-HI and dual-platform feature registries describing available diagnostic inputs. |
| `data/ORIGINAL_DATA_SOURCE.md` | Location, DOI, and access information for RWTH raw data, which are not duplicated here. |
| `results/preprocessing/` | Node coverage, battery-level reachability, and model eligibility records. |
| `results/ablation/` | Detailed Top-K out-of-fold evaluation metrics covering 60 configurations. |
| `results/scheduling/` | Outer-fold schedule evaluations and pooled timing/accuracy metrics. |
| `results/resources/` | Outer-fold joint resource-search results and target-specific operating-point metrics. |
| `results/adaptive/` | PRT decision traces, per-cell stopping outcomes, and pooled threshold comparisons. |
| `results/predictions/` | Battery-level fixed/adaptive and XGBoost/SafeResidualMoE comparisons; **not** full pointwise SOH prediction arrays. |
| `results/statistics/` | Horizon-stratified and battery-paired bootstrap outputs. |
| `figures/main/` | **The eight final main-text figures** in PNG: `Figure_01.png` through `Figure_08.png`. |
| `figures/appendix/` | **The two final Appendix figures** in PNG: `Figure_A1.png`, `Figure_A2.png`. |
| `tests/` | `check_archive_integrity.py`: checks file structure, 141×73 rankings, key expected row counts, Python syntax and compilation, numbered script references, known local-path patterns, figure presence, and exact-byte duplicate detection. It is **not** a training test. |
| `docs/` | Concise explanation of tested reproducibility and known missing inputs. Internal historical filename mappings and hash lists are intentionally not included. |

## Manuscript tables and data files

The displayed Appendix tables A.1–A.5 correspond to historical summary data S3–S6 and S8. The file nomenclature is preserved so manuscript links remain unambiguous.

| Paper item | CSV in this repository | Content |
|---|---|---|
| HI dictionary (manuscript archive citation) | `data/tables/Table_S1.csv` | 73 enabled HI definitions and units |
| Top-K results (manuscript archive citation) | `data/tables/Table_S2.csv` | 60 predictor × horizon × K configurations |
| Appendix Table A.1 | `data/tables/Table_S3.csv` | RefAhFraction uniform vs. optimized L2–L6 results |
| Appendix Table A.2 | `data/tables/Table_S4.csv` | Two-platform L2–L6 timing/accuracy comparison |
| Appendix Table A.3 | `data/tables/Table_S5.csv` | Objective-specific joint resource policies |
| Appendix Table A.4 | `data/tables/Table_S6.csv` | 25 Holm-adjusted comparisons |
| Held-out battery results (manuscript archive citation) | `data/tables/Table_S7.csv` | 47 battery-level summary records |
| Appendix Table A.5 | `data/tables/Table_S8.csv` | Three PRT stopping settings |
| Fold-wise HI rankings | `data/rankings/01_source_only_hi_ranking.csv` | 10,293 records = 141 groups × 73 HIs |

## Figure-to-evidence guide

These are the **relevant retained data files and scripts**. The column is a provenance guide, **not a claim that the named script exactly recreates the published image without additional inputs**. Each final PNG is provided under `figures/`.

| Figure | Topic | Retained related evidence | Code stage | Regeneration status |
|---|---|---|---|---|
| Fig. 1 | Cycling-SOH trajectories and interquartile range | RWTH raw data (external); preprocessing coverage in `results/preprocessing/` | `src/01_preprocess_soh_labels.py` | Final PNG available; full raw SOH time series not archived here |
| Fig. 2 | Early diagnostic HI contrasts at comparable SOH | HI definitions and registries in `data/`; external RWTH data | `src/02_build_refah_features.py` | Final PNG available; exact plotting inputs not fully archived |
| Fig. 3 | Two-scale methods framework | Methods summarized in `src/` and `README.md` | Multiple stages | Final schematic available; no editable drawing source |
| Fig. 4 | Within-cycle measurement duration and fraction | `results/preprocessing/` and external RWTH data | `src/02_build_refah_features.py` | Final PNG available; original per-trigger series not independently verified |
| Fig. 5 | RefAhFraction schedule optimization and predictions | `results/scheduling/`, `results/predictions/` | `src/10_schedule_and_resource_search.py` | Final PNG available; representative pointwise curves are not fully retained |
| Fig. 6 | FixedTime timing and resource comparison | `data/tables/Table_S4.csv`; `results/scheduling/` | `src/08_fixedtime_common_pipeline.py`, `src/10_schedule_and_resource_search.py` | Numerical aggregates present; exact plotting rerun not verified |
| Fig. 7 | Adaptive PRT decisions and resource/error differences | `results/adaptive/`, `data/tables/Table_S8.csv` | `src/12_adaptive_prt_policy.py`, `src/16_finalize_adaptive_results.py` | Decision-level records retained; exact plotting rerun not verified |
| Fig. 8 | Horizon-specific changes and predictor robustness | `results/statistics/`, `results/predictions/` | `src/14_paired_statistical_tests.py`, `src/18_compare_fixed_and_adaptive.py` | Numerical outputs retained; exact plotting rerun not verified |
| Fig. A.1 | Top-K ablation | `data/tables/Table_S2.csv`, `results/ablation/` | `src/05_rank_hi_topk_ablation.py` | Final PNG and 60-configuration data retained |
| Fig. A.2 | Holm-adjusted horizon-by-L evidence | `data/tables/Table_S6.csv`, `results/statistics/` | `src/14_paired_statistical_tests.py`, `src/15_generate_statistical_figures.py` | Final PNG and adjusted p values retained |

## Code and data use

Run the standard-library archive integrity check from the repository root:

```bash
python tests/check_archive_integrity.py
```

This verifies retained-file consistency; it **does not retrain** any model or prove that all figure outputs can be regenerated. See the known limitations in `docs/reproduction_status.md`.

The original RWTH dataset is maintained by its provider and should be obtained through [the official dataset record](https://doi.org/10.18154/RWTH-2021-04545); its large raw waveforms are not re-uploaded here. Default paths for the recovered scripts now point to a repository-relative workspace; historical intermediate directory names are retained. Additional input files and local configuration may still be required. Because a portable full-run protocol has not been verified, no single command is advertised as reproducing the entire article.

## Citation and reuse

`CITATION.cff` identifies the researcher and code collection. Update it with the actual public repository URL and, if created, Zenodo DOI **after publication of the repository/release**. No DOI or repository URL is invented here. The author has not yet granted a standalone software license; no license is implied by making the repository public. The RWTH source-data terms are separate.

## Local workspace and environment

The default processing workspace is `workspace/` at the repository root, separate from archived `data/` and `results/`. Download the original dataset using the DOI above and place its extracted `Rohdaten/` directory in `workspace/Rohdaten/` for the default preprocessing path. The default V4 output is `workspace/RWTH48_全流程输出_V4/`. The source scripts retain historical intermediate directory names and script-specific command-line settings. `src/19`–`src/25` supply recovered historical helper modules; the numbering is not an execution order. `requirements.txt` lists statically detected third-party imports (including TensorFlow for deep-learning routes and LightGBM for optional tree-model routes); versions and compatibility have not been validated. The workspace is excluded from Git. These changes do not establish a complete executable pipeline.


### Setup and limited checks

Use a separate Python environment (Python 3.11 is a reasonable starting point, not a recovered original environment). From the repository root:

```bash
python -m pip install -r requirements.txt
python tests/check_archive_integrity.py
python src/01_preprocess_soh_labels.py --help
python src/08_fixedtime_common_pipeline.py
```

The last command runs an included synthetic feature-engineering self-test, not a training experiment. The other processing scripts expose script-specific `--help` options. For raw preprocessing, after supplying the original data, `python src/01_preprocess_soh_labels.py` uses the default workspace paths. Subsequent scripts require their corresponding upstream inputs; the compact exported CSVs are not substitutes for full intermediate model inputs. Do not run the numbered scripts blindly in numerical order.

### Source program index

| File | Purpose |
|---|---|
| `src/01_preprocess_soh_labels.py` | Raw waveform audit, cleaning and SOH label preprocessing |
| `src/02_build_refah_features.py` | RefAhFraction feature extraction |
| `src/03_build_fixedtime_features.py` | FixedTime and Event feature extraction |
| `src/04_clean_shared_disturbances.py` | Shared disturbance cleaning |
| `src/05_rank_hi_topk_ablation.py` | Source-only HI ranking and Top-K ablation |
| `src/06_train_trajectory_xgboost.py` | Horizon-aware XGBoost trajectory modeling |
| `src/07_fixedtime_hi_ranking.py` | FixedTime ranking and Top-K evaluation |
| `src/08_fixedtime_common_pipeline.py` | Common FixedTime adapters and feature engineering; synthetic self-test |
| `src/09_platform_sensor_ablation.py` | Platform-aware sensor modality and RUL ablation |
| `src/10_schedule_and_resource_search.py` | Timing schedules and joint diagnostic-resource search |
| `src/11_early_boundary_review.py` | Early-boundary and SOH closeout checks |
| `src/12_adaptive_prt_policy.py` | Adaptive PRT scheduling |
| `src/13_safe_residual_moe.py` | Safeguarded residual mixture-of-experts |
| `src/14_paired_statistical_tests.py` | Paired and horizon-stratified statistical evidence |
| `src/15_generate_statistical_figures.py` | Statistical figure generation and closure |
| `src/16_finalize_adaptive_results.py` | Adaptive primary results finalization |
| `src/17_transfer_policy_to_moe.py` | Adaptive policy transfer to SafeResidualMoE |
| `src/18_compare_fixed_and_adaptive.py` | Final fixed/adaptive paired comparison |
| `src/19_fixedtime_ranking_helper.py` | Historical FixedTime HI-ranking and model helper |
| `src/20_trajectory_lobo_helper.py` | Historical LOBO trajectory-modeling helper |
| `src/21_rul_topk_helper.py` | Historical RUL-specific HI Top-K helper |
| `src/22_full_life_soh_helper.py` | Historical full-life SOH modeling helper |
| `src/23_event_tree_helper.py` | Historical Event tree-model helper |
| `src/24_event_sparse_search_helper.py` | Historical Event sparse-diagnosis helper |
| `src/25_fixedtime_sparse_search_helper.py` | Historical FixedTime sparse-diagnosis helper |

### Intermediate inputs intentionally excluded

The workspace is ignored by Git. Large V4/V4.5A/V4.5B/V4.6A intermediate tables, strict-segment and V4.11A2 operational-EOL target/history bundles, fold caches, and complete pointwise prediction arrays are not included. Some stages need historical outputs whose generating scripts are not included. These are genuine requirements of the historical programs, not missing links to the compact result CSVs. See `docs/reproduction_status.md` for the validation boundary.

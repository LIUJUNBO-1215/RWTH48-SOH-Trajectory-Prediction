# Reproducibility and data-scope notes

## Verified for this archived release
- Eight appendix-linked tables have 73, 60, 5, 10, 8, 25, 47, and 3 data rows, respectively.
- The source-only ranking export contains 141 groups with 73 HIs each (10,293 rows); the two highest-ranked feature codes are stable across those groups.
- All 25 source files (18 primary programs and seven historical helper modules) compile as Python source.
- The package contains 8 main and 2 Appendix final manuscript images.
- Existing published-table snapshots were compared with retained output exports in an earlier audit. This does **not** imply model reruns.

## Not fully included or independently rerun
- Complete source-to-results recreation from RWTH raw waveform measurements.
- Original V4 preprocessing caches and all per-battery reference-capacity provenance fields.
- Complete true/predicted SOH arrays with validity masks for each prediction point.
- The exact dependency environment with pinned versions and a tested portable invocation.
- All original figure-generation inputs and a verified one-command figure rebuild.

The scripts in `src/` are historical research programs, not a complete portable executable workflow. Known default machine-specific paths were redirected to a repository-relative workspace; internal numbered script references were updated and seven historical helper scripts were recovered. Historical intermediate inputs and script-specific configurations remain required; configuration JSON machine paths have been redacted. Do not describe the repository as independently end-to-end reproduced or claim that the final figure images are recomputed by the included integrity check.

## Archive path and helper update

Seven historical helper modules were recovered for inspection. All 24 command-line entry points successfully displayed help from outside the repository; the common FixedTime module (08) passed its synthetic feature-engineering self-test. The dynamic preprocessing imports in scripts 02 and 03 also passed. No training or raw-data pipeline rerun was performed. Required historical inputs such as V4.5A/V4.5B tables, strict-segment tables and fold caches are not established as recoverable by this archive. Requirements are unpinned and have not been tested.

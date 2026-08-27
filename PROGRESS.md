# Project Progress

Concise chronological project log. Detailed technical information lives in `docs/`.

## 2026-08-27

### Completed
- Implemented and hardened `patchtst_v1_5`, a supervised PhysioJEPA-style PatchTST variant with grouped-Conv tokenization, rotary self-attention, attentive pooling, a strict `--physiojepa-fidelity` preset, eager-only execution, tokenizer-derived patch counts, and dedicated v1.5 tests. See `docs/training.md`, `docs/model_description.md`, and `docs/data_description.md`.
- Added `docs/model_description.md` to compare `patchtst_v1`, `patchtst_v1_5`, and `patchtst_v2`, including their key architecture differences and typical configs.
- Added `docs/data_description.md` to summarize waveform inputs, split-specific normalization, regression targets, and event-label bundles.
- Reorganized the docs to remove long histories, duplicate interpretation, and stale queue notes. See `docs/README.md`.
- Added early stopping to `scripts/train_patchtst.py` and documented the trainer and submitter behavior. See `docs/training.md`.
- Replaced the mistaken standard-cohort reruns with fresh early-stopping-enabled vasopressor-free `patchtst_v1` jobs. See `docs/training_vaso_free.md`.
- Deleted the unused `anchor_event_centric_physiojepa_style` event-target mode from the code path and CLI, and added `anchor_horizon_filtered` for stricter hypotension negatives. See `docs/target_generation.md`.
- Completed regression and classification evaluation for the fresh vasopressor-free `patchtst_v1` runs, including the filtered-label hypotension GPU eval jobs `26851239` and `26851240`. See `docs/regression_results_v1_vaso_free_sorted.md`, `docs/classification_results_v1_vaso_free.md`, and `docs/training_vaso_free.md`.
- Added exact positive/valid counts behind the vasopressor-free classification prevalences for filtered and non-filtered `5m` and `10m` tables. See `docs/classification_results_v1_vaso_free.md`.
- Added per-condition negative-removal counts for filtered vasopressor-free `5m` and `10m` hypotension labels, including outcome-window validity/event overlap. See `docs/data_description.md`.
- Added the corresponding test-split filtering breakdown and reconciled it with the filtered classification denominators. See `docs/data_description.md` and `docs/classification_results_v1_vaso_free.md`.
- Deleted 44 center-mode `t+0` feature-regression run directories that used overlapping input/target windows; retained the shared `all_targets.npz` bundle and all gap-mode runs.
- Deleted obsolete event-centric target artifacts, the unreferenced WFDB training loader, and four old/leakage-prone submission/build scripts. Retained `slurm/train_patchtst.sh` because current gap-mode and event submitters use it as a shared runner.

### In Progress
- No training or evaluation SLURM jobs are currently pending.
- Last successful `squeue` check observed one non-repo-specific interactive CPU shell running: SLURM `26851827`; a later status query could not connect to Slurm from this environment, so verify before treating it as still active.
- The optional standard event target artifact `outputs/targets/event_targets_standard_5m_10m.npz` is absent and is not used by current models; rebuild it only if standard-cohort event experiments resume.
- Python `compileall` passed; `pytest` was unavailable in the active environment, and the `git` executable was unavailable, so Git working-tree status was not independently verifiable.
- Root-level `patch_probe_tmp.txt` and empty `stats.ipynb` were left untouched because their ownership/purpose was not established.

### Next Steps
1. Compare the fresh vasopressor-free `patchtst_v1` regression and classification results against the older `pre-v1` baselines; treat `docs/evaluation.md` regression results as legacy because their center-mode `t+0` targets overlap the input.
2. Decide whether to launch strict `patchtst_v1_5 --physiojepa-fidelity` supervised runs on the vasopressor-free cohort; no such runs have been launched.
3. If `v1_5` is promoted, create a submitter that explicitly pins the current leakage-safe target bundles and document the jobs in `docs/training_vaso_free.md`.
4. Rebuild and verify the missing standard `5m`/`10m` event bundle only if standard-cohort event experiments are needed.

## 2026-08-26

### Completed
- Rebuilt and verified vasopressor-free event targets with minute-level aggregation and current `5`-minute event semantics. See `docs/target_generation.md`.
- Added bootstrap event-metric confidence intervals and refreshed evaluation summaries. See `docs/evaluation.md`.
- Added target-stat summaries and refreshed Bland-Altman notes. See `docs/target_statistics.md` and `docs/bland_altman_analysis.md`.
- Added `patchtst_v2`, shared normalization, and standard `5m`/`10m` event-target support. See `docs/training.md` and `docs/target_generation.md`.

### Next Steps
- Compare vasopressor-free results against PhysioJEPA.
- Decide whether to expand `patchtst_v2` beyond pilot status.

## 2026-08-25

### Completed
- Finished the clean vasopressor-free PatchTST batch and evaluated the saved models. See `docs/training_vaso_free.md`.
- Added vasopressor-free regression, event, and threshold-sweep summaries. See `docs/training_vaso_free.md`.

### Next Steps
- Compare against PhysioJEPA.
- Decide whether to submit the vasopressor-free `t+20m` gap-mode regression batch.

## 2026-08-24

### Completed
- Prepared the vasopressor-free overlap cohort assets and fixed the regression `NaN` issues. See `docs/training_vaso_free.md`.
- Resubmitted the clean vasopressor-free regression and event batches. See `docs/training_vaso_free.md`.

## 2026-08-23

### Completed
- Evaluated the original full-cohort PatchTST baseline and computed simple baselines. See `docs/evaluation.md`.

### Next Steps
- Compare PatchTST against PhysioJEPA.
- Extend to later horizons for the strongest targets.

## 2026-08-19 to 2026-08-21

### Completed
- Built the initial documentation, target bundles, splits, dataset pipeline, and single-target PatchTST trainer. See `docs/training.md` and `docs/target_generation.md`.
- Replaced the original WFDB-at-train-time path with the NumPy/mmap pipeline and stabilized training after several failed early batches. See `docs/training.md`.

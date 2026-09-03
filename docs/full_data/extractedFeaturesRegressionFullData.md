# Extracted Feature Regression - Full Data

This page tracks downstream regression models trained on the full-data v7 extracted-feature cache. It mirrors the model families and evaluation discipline used in `docs/v7_extracted_features/extractedFeaturesRegression.md`, but uses the segment-aware full-data cohort from `data_m3_120s_prediction`.

## Current Status

- Full-data v7 extracted features are available at `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v7/full_data_vasopressor_free_waveform_features_v7` with shape `(1969515, 20, 93)`.
- Full-data combined v7+v8 extracted features are available at `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/combined_v7_v8/full_data_vasopressor_free_waveform_features_v7_v8_segment_plan` with shape `(1969515, 20, 287)`.
- A segment-aware regression target builder has been prepared: `scripts/build_full_data_feature_regression_targets.py`.
- The full-data regression target bundle is available at `outputs/targets/feature_targets_gap_full_data.npz` with shape `(1969515, 78)` and `128227907` valid target values.
- A second-pass hardened, non-overwriting regression target bundle is available at `outputs/targets/feature_targets_gap_full_data_hardened_v2.npz`; it is numerically identical to the active production bundle and adds `split_labels`, relationship-based timing metadata, cache/source identity audits, and post-write serializer validation.
- Completed full-data `t+0m_gap` regression jobs for `history_xgb`, `full_sequence_xgb`, and `transformer`, writing results under `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/regression`. The first XGBoost submission OOMed and was resubmitted after memory fixes; the completed batch has `78/78` metrics and prediction exports.
- Submitted full-data combined v7+v8 `t+0m_gap` regression jobs for `history_xgb`, `full_sequence_xgb`, and `transformer`, writing results under `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data_v7_v8/regression`; active jobs were canceled on `2026-09-03` for deadlines and should be resumed as soon as possible.

## Planned Setup

- Cohort: high-confidence segment-level vasopressor-free full-data windows.
- Split discipline: full-data `patient_splits.json`, carried row-wise in the feature cache as `split_labels.npy`.
- Input: v7 extracted waveform features, `(N, 20, 93)`, preprocessed train-only by `scripts/train_feature_models.py`.
- Target names: same `26` base physiological/correlation targets as `docs/v7_extracted_features/extractedFeaturesRegression.md`.
- Target horizons: `0`, `20`, and `60` minutes with `gap` semantics in the target bundle; initial model submitter trains the documented `t+0m_gap` targets.
- Alignment key: `anchor_id`, not `(patient_id, anchor_time)`, because full-data `anchor_time` is segment-relative and can repeat within a patient.

## Regression Target Comparability

Unlike the classification targets, the v7 and full-data regression targets use the same target semantics. Both predict future physiological/source feature values with `gap` timing, where `target_center = anchor_center + input_window_minutes + horizon_minutes`. With the 20-minute input window, `t+0m_gap` therefore points to the target window immediately after the input window, not to an overlapping center-time target.

The full-data differences are cohort and alignment machinery rather than target definition. The earlier v7 bundle uses the older vasopressor-free overlap cohort and aligns future source rows by `(patient_id, anchor_time)`; this full-data bundle uses the segment-aware full-data cohort and aligns by `anchor_id`/segment-aware source identity because full-data anchor times are segment-relative and can repeat within a patient. The hardened full-data rebuild added validation and provenance metadata but did not change the numeric targets or masks used by the completed regression runs.

## Commands

Rebuild regression targets:

```bash
sbatch slurm/build_full_data_feature_regression_targets.sh
```

The Slurm wrapper writes the default production path `outputs/targets/feature_targets_gap_full_data.npz`. To rebuild without overwriting the active bundle, run the Python builder directly with an explicit alternate `--output` path, as shown in the recommendation below.

Submit the same extracted-feature model families used in the prior v7 docs:

```bash
INCLUDE_CLASSIFICATION=0 bash slurm/submit_feature_models_full_data_v7.sh
```

The submitter supports `MODELS=...`, `DRY_RUN=1`, `INCLUDE_REGRESSION=0`, and `INCLUDE_CLASSIFICATION=0` environment overrides.

Submit the combined v7+v8 comparison batch:

```bash
INCLUDE_CLASSIFICATION=0 \
MODELS='history_xgb full_sequence_xgb transformer' \
CACHE_DIR=/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/combined_v7_v8/full_data_vasopressor_free_waveform_features_v7_v8_segment_plan \
OUTPUT_ROOT=/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data_v7_v8 \
REGRESSION_TARGETS=outputs/targets/feature_targets_gap_full_data_hardened_v2.npz \
FEATURE_CACHE_LABEL=v7_v8 \
bash slurm/submit_feature_models_full_data_v7.sh
```

To force only tabular submissions to a different CPU partition, set `TABULAR_PARTITION`, for example `TABULAR_PARTITION=cpu_short`. This leaves Transformer submissions on their SLURM script's GPU partition unless `SEQUENCE_PARTITION` is also set.

## Combined v7+v8 Regression Submission

Submitted on `2026-09-02` after building and validating the combined cache.

- Cache: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/combined_v7_v8/full_data_vasopressor_free_waveform_features_v7_v8_segment_plan`
- Cache shape: `(1969515, 20, 287)`.
- Targets: `outputs/targets/feature_targets_gap_full_data_hardened_v2.npz`.
- Models: `history_xgb`, `full_sequence_xgb`, `transformer`.
- Output root: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data_v7_v8/regression`.
- Submitted jobs: `78`.
- Initial regression job IDs: `26980716`-`26980811`, excluding the interleaved classification IDs listed in `docs/full_data/extractedFeaturesClassificationFullData.md`.
- Partition move: the pending `cpu_medium` tabular regression jobs were canceled and resubmitted to `cpu_short`; Transformer regression jobs were left on `gl40s_short`.
- Active tabular regression job IDs: `26980891`-`26980937`, plus `26980939`, `26980941`, `26980943`, `26980945`, and `26980947`.
- Active Transformer regression job IDs: `26980786`-`26980811`.
- Handoff status at `2026-09-02 18:20 EDT`: tabular jobs `26980891`-`26980915` were running on `cpu_short`; remaining tabular jobs `26980916`-`26980937`, `26980939`, `26980941`, `26980943`, `26980945`, and `26980947` were pending on `QOSMaxMemoryPerUser`; Transformer regression jobs `26980786`-`26980811` were pending on `gl40s_short` priority. The output root had `25` regression run directories but no `metrics.json`, `test_predictions.npz`, `model.pkl`, or `config.json` artifacts yet.

Implementation notes:

- `scripts/build_combined_v7_v8_feature_cache.py` writes the combined cache by streaming the aligned v7 and v8 arrays into output memmaps.
- `FeaturePreprocessor` now recognizes v8 feature definitions and allows all-missing training columns by imputing `0`, using mean `0` and std `1`, and preserving the mask indicator. This is needed because default v8 intentionally stores disabled audit-only feature families as all missing.

Before interpreting results, audit each run directory for `metrics.json`, `test_predictions.npz`, and `model.pkl` for XGBoost models.


## Combined v7+v8 Regression Cancellation, 2026-09-03

The active combined v7+v8 full-data regression jobs were canceled for now to free SLURM capacity for deadlines. This was a scheduling decision, not a result-based decision; the combined v7+v8 regression batch should be resumed as soon as possible.

- Canceled tabular regression jobs: `26980916`-`26980937`, `26980939`, `26980941`, `26980943`, `26980945`, and `26980947`.
- Canceled Transformer regression jobs: `26980786`-`26980811`.
- No combined v7+v8 regression metrics should be interpreted until the batch is resubmitted, completed, and audited for `metrics.json`, `test_predictions.npz`, and XGBoost `model.pkl` artifacts.
- Handoff artifact audit after cancellation found `48` regression run directories under `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data_v7_v8/regression`, with `25` `config.json`, `25` `metrics.json`, `25` `test_predictions.npz`, and `25` XGBoost `model.pkl` files. These are partial artifacts from the interrupted batch, not a complete v7+v8 result set.

## Raw-Waveform PatchTST Models

The full-data 4-channel raw-waveform `patchtst_v1` regression setup, normalization stats, validation checks, and submitted SLURM jobs are documented in `docs/full_data/full_data_raw_waveform_models.md`.

## Regression Job Completion

Submitted on `2026-08-31` with:

```bash
INCLUDE_CLASSIFICATION=0 MODELS='history_xgb full_sequence_xgb transformer' bash slurm/submit_feature_models_full_data_v7.sh
```

Original job ranges:

- `history_xgb`: SLURM `26934307`-`26934332`; these OOMed during training and were resubmitted as `26934627`-`26934652` after the tabular memory fix.
- `full_sequence_xgb`: SLURM `26934333`-`26934358`; these OOMed during training and were resubmitted as `26934653`-`26934678` after the tabular memory fix.
- `transformer`: SLURM `26934359`-`26934384`.

Completion check on `2026-09-01`:

- `squeue -u dk5565` showed no full-data regression jobs still queued or running; only unrelated interactive `bash` job `26949817` was active.
- Artifact audit under `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/regression` found `78/78` `metrics.json` files, `78/78` `test_predictions.npz` files, and `52/52` expected XGBoost `model.pkl` files.
- Each prediction archive was reopened and checked for `sample_ids`, `patient_time_sample_ids`, `anchor_ids`, `targets`, `predictions`, `row_indices`, and `split_labels`; prediction counts matched `metrics.json`.

Expected artifacts per output directory:

- `metrics.json`
- `test_predictions.npz`, including `sample_ids`, `patient_time_sample_ids`, `anchor_ids`, `targets`, `predictions`, `row_indices`, and `split_labels`
- `model.pkl` for the two XGBoost model families

Target-bundle rebuild:

- Pending hardened target rebuild SLURM `26940116` was cancelled and the same build was run locally because it is CPU-only.
- First output path: `outputs/targets/feature_targets_gap_full_data_hardened_20260831.npz`; this artifact was superseded by the second-pass build below because its metadata predates the final timing/source-identity metadata corrections.
- Second-pass output path: `outputs/targets/feature_targets_gap_full_data_hardened_v2.npz`.
- The existing production target bundle `outputs/targets/feature_targets_gap_full_data.npz` was intentionally left untouched while regression jobs were running.
- The second-pass hardened output shape is `(1969515, 78)` with `128227907` valid target values.
- Diff against the active production bundle found identical `feature_targets`, `feature_mask`, `anchor_ids`, `anchor_patient_ids`, `anchor_times`, `segment_ids`, `segment_names`, `input_start_times`, `input_end_times`, `feature_target_names`, and `feature_spec`.
- The second-pass hardened bundle adds `split_labels` plus richer metadata: relationship-based `timing_semantics`, `cache_identity_validation`, `current_source_identity_audit`, explicit feature/correlation/target names, aggregation, and `post_write_bundle_audit`.
- The second-pass current-source identity audit matched `1969515/1969515` cache anchors exactly.
- No regression training resubmission is needed for target correctness.
- Decision: use the hardened bundle for future submissions. The hardened target pipeline changes validation/provenance metadata, not the numeric targets or masks used by the completed regression batch.

## Result Tables



Full-data results refreshed on `2026-09-01` with `10000` patient-cluster bootstrap replicates and seed `42`. Tables include runs with completed `metrics.json`, `test_predictions.npz`, and XGBoost `model.pkl` artifacts where applicable. Point estimates are the original global test-set metrics; confidence intervals are percentile 95% CIs from resampling test patients with replacement and including all windows for sampled patients.



Unique test-patient counts across model-target runs: `265, 266, 267, 268, 269`. Invalid R2 bootstrap replicates across all runs: `0`.



Machine-readable CI outputs are saved at `outputs/feature_models/full_data_regression_patient_bootstrap_ci_2026-09-01.json` and `outputs/feature_models/full_data_regression_patient_bootstrap_ci_2026-09-01.csv`; each run directory also has a `metrics_with_ci.json` sidecar with separate numeric CI fields.



### `history_xgb`

Completed targets: `26/26`.

| Target | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Test Predictions | Test Patients |
|---|---:|---:|---:|---:|---:|
| `PLETH_amp_t_plus_0m_gap` | 0.0751 [0.0676, 0.0823] | 0.1273 [0.1169, 0.1371] | 0.9579 [0.9455, 0.9674] | 257659 | 269 |
| `dPdt_max_t_plus_0m_gap` | 93.0741 [86.0484, 100.4084] | 149.7872 [136.7376, 163.1547] | 0.8871 [0.8687, 0.9028] | 258583 | 268 |
| `PP_t_plus_0m_gap` | 4.5845 [4.2484, 4.9606] | 7.4164 [6.7930, 8.0828] | 0.8704 [0.8498, 0.8877] | 258229 | 265 |
| `SBP_t_plus_0m_gap` | 5.7149 [5.4047, 6.0349] | 8.4450 [7.9535, 8.9550] | 0.8690 [0.8420, 0.8921] | 258473 | 265 |
| `ABP_area_t_plus_0m_gap` | 1.3463 [1.2311, 1.4747] | 2.1851 [1.9864, 2.3908] | 0.8687 [0.8395, 0.8925] | 258982 | 268 |
| `RESP_amp_t_plus_0m_gap` | 0.1077 [0.0994, 0.1164] | 0.1605 [0.1476, 0.1734] | 0.8577 [0.8007, 0.8954] | 259242 | 269 |
| `DBP_t_plus_0m_gap` | 3.2047 [2.9864, 3.4436] | 5.5841 [5.1163, 6.0851] | 0.8546 [0.8197, 0.8782] | 258473 | 265 |
| `PLETH_ACDC_t_plus_0m_gap` | 0.0694 [0.0641, 0.0758] | 0.1066 [0.0952, 0.1220] | 0.8486 [0.7955, 0.8797] | 257659 | 269 |
| `MAP_t_plus_0m_gap` | 4.3035 [4.0253, 4.6101] | 6.9803 [6.4172, 7.5860] | 0.8384 [0.8038, 0.8640] | 258861 | 266 |
| `ShockIdx_t_plus_0m_gap` | 0.0675 [0.0619, 0.0739] | 0.1056 [0.0939, 0.1190] | 0.7994 [0.7382, 0.8454] | 258146 | 265 |
| `PVI_t_plus_0m_gap` | 6.4888 [6.0092, 7.0607] | 9.4274 [8.3395, 10.8257] | 0.6975 [0.6077, 0.7676] | 257288 | 269 |
| `HR_t_plus_0m_gap` | 5.0936 [4.4731, 5.7231] | 8.0224 [7.0461, 8.9755] | 0.6844 [0.6203, 0.7385] | 259254 | 269 |
| `PPV_t_plus_0m_gap` | 6.4955 [5.8085, 7.2545] | 10.7110 [9.3970, 12.0361] | 0.6397 [0.5257, 0.7210] | 257423 | 267 |
| `HR_range_t_plus_0m_gap` | 9.3546 [8.4673, 10.2086] | 12.8810 [11.8065, 13.8922] | 0.6338 [0.5578, 0.6861] | 259255 | 269 |
| `ABP_tau_t_plus_0m_gap` | 0.2332 [0.2108, 0.2570] | 0.4396 [0.3986, 0.4828] | 0.6085 [0.5649, 0.6471] | 258337 | 265 |
| `HRV_RMSSD_t_plus_0m_gap` | 30.5174 [27.8497, 33.2037] | 41.3961 [38.0027, 44.7161] | 0.5832 [0.5122, 0.6456] | 259255 | 269 |
| `RR_t_plus_0m_gap` | 1.2879 [1.2250, 1.3539] | 1.7174 [1.6379, 1.7989] | 0.5375 [0.4767, 0.5945] | 259218 | 269 |
| `PLETH_ACDC_PLETH_amp_t_plus_0m_gap` | 0.0678 [0.0591, 0.0773] | 0.1350 [0.1201, 0.1500] | 0.3654 [0.2888, 0.4516] | 259581 | 269 |
| `ECG_Ramp_t_plus_0m_gap` | 0.2138 [0.1917, 0.2367] | 0.2734 [0.2494, 0.2971] | 0.3490 [0.2723, 0.4211] | 259261 | 269 |
| `ABP_area_ShockIdx_t_plus_0m_gap` | 0.3097 [0.2906, 0.3286] | 0.4057 [0.3852, 0.4255] | 0.2630 [0.2244, 0.2971] | 259581 | 269 |
| `ABP_area_ABP_tau_t_plus_0m_gap` | 0.4203 [0.4076, 0.4331] | 0.5073 [0.4917, 0.5225] | 0.2177 [0.1755, 0.2594] | 259581 | 269 |
| `PTT_t_plus_0m_gap` | 25.7840 [24.0949, 27.5509] | 32.6925 [30.6278, 34.8215] | 0.1473 [0.0792, 0.2001] | 258114 | 268 |
| `ShockIdx_ABP_tau_t_plus_0m_gap` | 0.4366 [0.4275, 0.4454] | 0.5159 [0.5055, 0.5256] | 0.0911 [0.0684, 0.1106] | 259581 | 269 |
| `PLETH_ACDC_ABP_tau_t_plus_0m_gap` | 0.3791 [0.3707, 0.3869] | 0.4550 [0.4464, 0.4630] | 0.0667 [0.0484, 0.0852] | 259581 | 269 |
| `PLETH_ACDC_ShockIdx_t_plus_0m_gap` | 0.3885 [0.3815, 0.3955] | 0.4639 [0.4567, 0.4711] | 0.0624 [0.0482, 0.0756] | 259581 | 269 |
| `PLETH_amp_ShockIdx_t_plus_0m_gap` | 0.3787 [0.3717, 0.3856] | 0.4532 [0.4459, 0.4603] | 0.0381 [0.0258, 0.0489] | 259581 | 269 |

### `full_sequence_xgb`

Completed targets: `26/26`.

| Target | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Test Predictions | Test Patients |
|---|---:|---:|---:|---:|---:|
| `PLETH_amp_t_plus_0m_gap` | 0.0756 [0.0682, 0.0828] | 0.1279 [0.1176, 0.1378] | 0.9575 [0.9450, 0.9670] | 257659 | 269 |
| `dPdt_max_t_plus_0m_gap` | 94.6011 [87.7163, 101.7571] | 150.0291 [137.2045, 163.3311] | 0.8867 [0.8676, 0.9031] | 258583 | 268 |
| `PP_t_plus_0m_gap` | 4.5843 [4.2525, 4.9573] | 7.3498 [6.7594, 8.0029] | 0.8727 [0.8497, 0.8924] | 258229 | 265 |
| `ABP_area_t_plus_0m_gap` | 1.3423 [1.2244, 1.4742] | 2.1689 [1.9659, 2.3796] | 0.8707 [0.8401, 0.8954] | 258982 | 268 |
| `SBP_t_plus_0m_gap` | 5.7480 [5.4425, 6.0704] | 8.4252 [7.9537, 8.9212] | 0.8697 [0.8403, 0.8960] | 258473 | 265 |
| `RESP_amp_t_plus_0m_gap` | 0.1091 [0.1008, 0.1178] | 0.1625 [0.1495, 0.1756] | 0.8541 [0.7957, 0.8930] | 259242 | 269 |
| `DBP_t_plus_0m_gap` | 3.2373 [3.0168, 3.4736] | 5.6185 [5.1544, 6.1074] | 0.8528 [0.8186, 0.8757] | 258473 | 265 |
| `MAP_t_plus_0m_gap` | 4.3137 [4.0394, 4.6169] | 6.9596 [6.4039, 7.5535] | 0.8394 [0.8030, 0.8665] | 258861 | 266 |
| `PLETH_ACDC_t_plus_0m_gap` | 0.0740 [0.0676, 0.0813] | 0.1131 [0.1001, 0.1302] | 0.8297 [0.7690, 0.8643] | 257659 | 269 |
| `ShockIdx_t_plus_0m_gap` | 0.0690 [0.0623, 0.0768] | 0.1104 [0.0943, 0.1306] | 0.7807 [0.7360, 0.8072] | 258146 | 265 |
| `PVI_t_plus_0m_gap` | 6.5458 [6.0588, 7.1182] | 9.5251 [8.4425, 10.9104] | 0.6912 [0.6008, 0.7613] | 257288 | 269 |
| `HR_t_plus_0m_gap` | 5.0788 [4.4562, 5.7185] | 8.0127 [7.0396, 8.9747] | 0.6851 [0.6201, 0.7394] | 259254 | 269 |
| `HR_range_t_plus_0m_gap` | 9.3801 [8.4681, 10.2480] | 12.9521 [11.8507, 13.9819] | 0.6298 [0.5531, 0.6826] | 259255 | 269 |
| `PPV_t_plus_0m_gap` | 6.6289 [5.9232, 7.4057] | 10.8932 [9.5251, 12.2692] | 0.6273 [0.5149, 0.7077] | 257423 | 267 |
| `ABP_tau_t_plus_0m_gap` | 0.2360 [0.2135, 0.2599] | 0.4386 [0.3994, 0.4802] | 0.6102 [0.5643, 0.6492] | 258337 | 265 |
| `HRV_RMSSD_t_plus_0m_gap` | 30.5394 [27.8532, 33.2348] | 41.4283 [37.9750, 44.8191] | 0.5826 [0.5099, 0.6453] | 259255 | 269 |
| `RR_t_plus_0m_gap` | 1.2984 [1.2317, 1.3692] | 1.7366 [1.6512, 1.8259] | 0.5270 [0.4699, 0.5817] | 259218 | 269 |
| `PLETH_ACDC_PLETH_amp_t_plus_0m_gap` | 0.0683 [0.0596, 0.0777] | 0.1351 [0.1203, 0.1501] | 0.3645 [0.2872, 0.4514] | 259581 | 269 |
| `ECG_Ramp_t_plus_0m_gap` | 0.2143 [0.1919, 0.2373] | 0.2729 [0.2491, 0.2964] | 0.3514 [0.2779, 0.4218] | 259261 | 269 |
| `ABP_area_ShockIdx_t_plus_0m_gap` | 0.3160 [0.2964, 0.3354] | 0.4115 [0.3905, 0.4318] | 0.2415 [0.2007, 0.2773] | 259581 | 269 |
| `ABP_area_ABP_tau_t_plus_0m_gap` | 0.4320 [0.4207, 0.4438] | 0.5171 [0.5023, 0.5319] | 0.1874 [0.1477, 0.2256] | 259581 | 269 |
| `PTT_t_plus_0m_gap` | 26.0059 [24.2864, 27.8103] | 32.8708 [30.8158, 34.9532] | 0.1380 [0.0696, 0.1910] | 258114 | 268 |
| `ShockIdx_ABP_tau_t_plus_0m_gap` | 0.4443 [0.4342, 0.4537] | 0.5221 [0.5117, 0.5320] | 0.0687 [0.0520, 0.0835] | 259581 | 269 |
| `PLETH_ACDC_ABP_tau_t_plus_0m_gap` | 0.3808 [0.3724, 0.3888] | 0.4567 [0.4482, 0.4649] | 0.0594 [0.0421, 0.0767] | 259581 | 269 |
| `PLETH_ACDC_ShockIdx_t_plus_0m_gap` | 0.3912 [0.3839, 0.3987] | 0.4666 [0.4590, 0.4742] | 0.0516 [0.0384, 0.0638] | 259581 | 269 |
| `PLETH_amp_ShockIdx_t_plus_0m_gap` | 0.3806 [0.3734, 0.3878] | 0.4551 [0.4476, 0.4625] | 0.0298 [0.0186, 0.0389] | 259581 | 269 |

### `transformer`

Completed targets: `26/26`.

| Target | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Test Predictions | Test Patients |
|---|---:|---:|---:|---:|---:|
| `PLETH_amp_t_plus_0m_gap` | 0.0762 [0.0689, 0.0832] | 0.1285 [0.1183, 0.1383] | 0.9571 [0.9445, 0.9666] | 257659 | 269 |
| `dPdt_max_t_plus_0m_gap` | 89.8857 [82.6471, 97.3374] | 146.1141 [132.8685, 159.3102] | 0.8925 [0.8733, 0.9092] | 258583 | 268 |
| `PP_t_plus_0m_gap` | 4.5769 [4.2512, 4.9474] | 7.3434 [6.7315, 8.0112] | 0.8730 [0.8498, 0.8927] | 258229 | 265 |
| `SBP_t_plus_0m_gap` | 5.6561 [5.3451, 5.9874] | 8.4289 [7.9228, 8.9524] | 0.8695 [0.8423, 0.8929] | 258473 | 265 |
| `ABP_area_t_plus_0m_gap` | 1.3985 [1.2806, 1.5264] | 2.2227 [2.0148, 2.4331] | 0.8642 [0.8316, 0.8905] | 258982 | 268 |
| `PLETH_ACDC_t_plus_0m_gap` | 0.0662 [0.0620, 0.0709] | 0.1013 [0.0943, 0.1089] | 0.8633 [0.8358, 0.8840] | 257659 | 269 |
| `RESP_amp_t_plus_0m_gap` | 0.1068 [0.0992, 0.1147] | 0.1598 [0.1477, 0.1719] | 0.8588 [0.7995, 0.8986] | 259242 | 269 |
| `DBP_t_plus_0m_gap` | 3.2873 [3.0732, 3.5192] | 5.6831 [5.2157, 6.1805] | 0.8494 [0.8125, 0.8746] | 258473 | 265 |
| `MAP_t_plus_0m_gap` | 4.2801 [4.0088, 4.5745] | 7.0228 [6.4688, 7.6161] | 0.8365 [0.7994, 0.8634] | 258861 | 266 |
| `ShockIdx_t_plus_0m_gap` | 0.0702 [0.0619, 0.0807] | 0.1243 [0.0962, 0.1614] | 0.7222 [0.6862, 0.7667] | 258146 | 265 |
| `PVI_t_plus_0m_gap` | 6.6362 [6.0832, 7.2650] | 9.9628 [8.7155, 11.4237] | 0.6622 [0.5615, 0.7429] | 257288 | 269 |
| `HR_t_plus_0m_gap` | 5.9737 [5.4302, 6.5251] | 8.6136 [7.7195, 9.4763] | 0.6361 [0.5758, 0.6854] | 259254 | 269 |
| `ABP_tau_t_plus_0m_gap` | 0.2160 [0.1949, 0.2385] | 0.4339 [0.3923, 0.4774] | 0.6185 [0.5729, 0.6593] | 258337 | 265 |
| `PPV_t_plus_0m_gap` | 6.5026 [5.8227, 7.2490] | 11.0780 [9.6383, 12.5404] | 0.6146 [0.4784, 0.7149] | 257423 | 267 |
| `HR_range_t_plus_0m_gap` | 9.8642 [8.9522, 10.7425] | 13.8179 [12.6931, 14.9261] | 0.5786 [0.4776, 0.6475] | 259255 | 269 |
| `RR_t_plus_0m_gap` | 1.2604 [1.1964, 1.3296] | 1.6969 [1.6121, 1.7848] | 0.5484 [0.4827, 0.6126] | 259218 | 269 |
| `HRV_RMSSD_t_plus_0m_gap` | 32.1746 [29.4928, 34.8299] | 43.4780 [40.2937, 46.5711] | 0.5402 [0.4678, 0.6019] | 259255 | 269 |
| `PLETH_ACDC_PLETH_amp_t_plus_0m_gap` | 0.0723 [0.0635, 0.0818] | 0.1372 [0.1220, 0.1525] | 0.3447 [0.2624, 0.4364] | 259581 | 269 |
| `ECG_Ramp_t_plus_0m_gap` | 0.2215 [0.1992, 0.2448] | 0.2936 [0.2679, 0.3192] | 0.2492 [0.1539, 0.3364] | 259261 | 269 |
| `ABP_area_ShockIdx_t_plus_0m_gap` | 0.3074 [0.2874, 0.3273] | 0.4134 [0.3916, 0.4344] | 0.2345 [0.1928, 0.2718] | 259581 | 269 |
| `ABP_area_ABP_tau_t_plus_0m_gap` | 0.4218 [0.4066, 0.4363] | 0.5115 [0.4939, 0.5282] | 0.2048 [0.1541, 0.2557] | 259581 | 269 |
| `ShockIdx_ABP_tau_t_plus_0m_gap` | 0.4388 [0.4292, 0.4477] | 0.5197 [0.5088, 0.5300] | 0.0775 [0.0488, 0.1038] | 259581 | 269 |
| `PLETH_ACDC_ABP_tau_t_plus_0m_gap` | 0.3798 [0.3721, 0.3871] | 0.4574 [0.4494, 0.4648] | 0.0566 [0.0381, 0.0752] | 259581 | 269 |
| `PLETH_ACDC_ShockIdx_t_plus_0m_gap` | 0.3871 [0.3801, 0.3938] | 0.4654 [0.4580, 0.4724] | 0.0565 [0.0353, 0.0760] | 259581 | 269 |
| `PLETH_amp_ShockIdx_t_plus_0m_gap` | 0.3770 [0.3701, 0.3839] | 0.4527 [0.4454, 0.4598] | 0.0402 [0.0234, 0.0553] | 259581 | 269 |
| `PTT_t_plus_0m_gap` | 27.6674 [25.9546, 29.5488] | 35.5098 [33.3634, 37.7830] | -0.0060 [-0.1014, 0.0720] | 258114 | 268 |

### Best Completed Model Per Target

All three submitted model families completed. Best-model counts by held-out R2: `history_xgb` 14/26, `full_sequence_xgb` 5/26, `transformer` 7/26.

| Target | Best Completed Model | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Test Predictions | Test Patients |
|---|---|---:|---:|---:|---:|---:|
| `HR_t_plus_0m_gap` | `full_sequence_xgb` | 5.0788 [4.4562, 5.7185] | 8.0127 [7.0396, 8.9747] | 0.6851 [0.6201, 0.7394] | 259254 | 269 |
| `RR_t_plus_0m_gap` | `transformer` | 1.2604 [1.1964, 1.3296] | 1.6969 [1.6121, 1.7848] | 0.5484 [0.4827, 0.6126] | 259218 | 269 |
| `SBP_t_plus_0m_gap` | `full_sequence_xgb` | 5.7480 [5.4425, 6.0704] | 8.4252 [7.9537, 8.9212] | 0.8697 [0.8403, 0.8960] | 258473 | 265 |
| `DBP_t_plus_0m_gap` | `history_xgb` | 3.2047 [2.9864, 3.4436] | 5.5841 [5.1163, 6.0851] | 0.8546 [0.8197, 0.8782] | 258473 | 265 |
| `PP_t_plus_0m_gap` | `transformer` | 4.5769 [4.2512, 4.9474] | 7.3434 [6.7315, 8.0112] | 0.8730 [0.8498, 0.8927] | 258229 | 265 |
| `MAP_t_plus_0m_gap` | `full_sequence_xgb` | 4.3137 [4.0394, 4.6169] | 6.9596 [6.4039, 7.5535] | 0.8394 [0.8030, 0.8665] | 258861 | 266 |
| `ABP_area_t_plus_0m_gap` | `full_sequence_xgb` | 1.3423 [1.2244, 1.4742] | 2.1689 [1.9659, 2.3796] | 0.8707 [0.8401, 0.8954] | 258982 | 268 |
| `PLETH_ACDC_t_plus_0m_gap` | `transformer` | 0.0662 [0.0620, 0.0709] | 0.1013 [0.0943, 0.1089] | 0.8633 [0.8358, 0.8840] | 257659 | 269 |
| `PLETH_amp_t_plus_0m_gap` | `history_xgb` | 0.0751 [0.0676, 0.0823] | 0.1273 [0.1169, 0.1371] | 0.9579 [0.9455, 0.9674] | 257659 | 269 |
| `ECG_Ramp_t_plus_0m_gap` | `full_sequence_xgb` | 0.2143 [0.1919, 0.2373] | 0.2729 [0.2491, 0.2964] | 0.3514 [0.2779, 0.4218] | 259261 | 269 |
| `HRV_RMSSD_t_plus_0m_gap` | `history_xgb` | 30.5174 [27.8497, 33.2037] | 41.3961 [38.0027, 44.7161] | 0.5832 [0.5122, 0.6456] | 259255 | 269 |
| `HR_range_t_plus_0m_gap` | `history_xgb` | 9.3546 [8.4673, 10.2086] | 12.8810 [11.8065, 13.8922] | 0.6338 [0.5578, 0.6861] | 259255 | 269 |
| `ShockIdx_t_plus_0m_gap` | `history_xgb` | 0.0675 [0.0619, 0.0739] | 0.1056 [0.0939, 0.1190] | 0.7994 [0.7382, 0.8454] | 258146 | 265 |
| `PPV_t_plus_0m_gap` | `history_xgb` | 6.4955 [5.8085, 7.2545] | 10.7110 [9.3970, 12.0361] | 0.6397 [0.5257, 0.7210] | 257423 | 267 |
| `PVI_t_plus_0m_gap` | `history_xgb` | 6.4888 [6.0092, 7.0607] | 9.4274 [8.3395, 10.8257] | 0.6975 [0.6077, 0.7676] | 257288 | 269 |
| `PTT_t_plus_0m_gap` | `history_xgb` | 25.7840 [24.0949, 27.5509] | 32.6925 [30.6278, 34.8215] | 0.1473 [0.0792, 0.2001] | 258114 | 268 |
| `dPdt_max_t_plus_0m_gap` | `transformer` | 89.8857 [82.6471, 97.3374] | 146.1141 [132.8685, 159.3102] | 0.8925 [0.8733, 0.9092] | 258583 | 268 |
| `ABP_tau_t_plus_0m_gap` | `transformer` | 0.2160 [0.1949, 0.2385] | 0.4339 [0.3923, 0.4774] | 0.6185 [0.5729, 0.6593] | 258337 | 265 |
| `RESP_amp_t_plus_0m_gap` | `transformer` | 0.1068 [0.0992, 0.1147] | 0.1598 [0.1477, 0.1719] | 0.8588 [0.7995, 0.8986] | 259242 | 269 |
| `PLETH_ACDC_PLETH_amp_t_plus_0m_gap` | `history_xgb` | 0.0678 [0.0591, 0.0773] | 0.1350 [0.1201, 0.1500] | 0.3654 [0.2888, 0.4516] | 259581 | 269 |
| `ABP_area_ABP_tau_t_plus_0m_gap` | `history_xgb` | 0.4203 [0.4076, 0.4331] | 0.5073 [0.4917, 0.5225] | 0.2177 [0.1755, 0.2594] | 259581 | 269 |
| `ABP_area_ShockIdx_t_plus_0m_gap` | `history_xgb` | 0.3097 [0.2906, 0.3286] | 0.4057 [0.3852, 0.4255] | 0.2630 [0.2244, 0.2971] | 259581 | 269 |
| `PLETH_amp_ShockIdx_t_plus_0m_gap` | `transformer` | 0.3770 [0.3701, 0.3839] | 0.4527 [0.4454, 0.4598] | 0.0402 [0.0234, 0.0553] | 259581 | 269 |
| `PLETH_ACDC_ShockIdx_t_plus_0m_gap` | `history_xgb` | 0.3885 [0.3815, 0.3955] | 0.4639 [0.4567, 0.4711] | 0.0624 [0.0482, 0.0756] | 259581 | 269 |
| `ShockIdx_ABP_tau_t_plus_0m_gap` | `history_xgb` | 0.4366 [0.4275, 0.4454] | 0.5159 [0.5055, 0.5256] | 0.0911 [0.0684, 0.1106] | 259581 | 269 |
| `PLETH_ACDC_ABP_tau_t_plus_0m_gap` | `history_xgb` | 0.3791 [0.3707, 0.3869] | 0.4550 [0.4464, 0.4630] | 0.0667 [0.0484, 0.0852] | 259581 | 269 |

## Bland-Altman Agreement Summary

This table summarizes the existing full-data best-model Bland-Altman analyses in `blandaltman_full_features/`. Statistics use `difference = prediction - reference` and `average = (prediction + reference) / 2`, matching the plot-generation convention. Model rows use patient-cluster bootstrap percentile 95% CIs with patients resampled as clusters and all windows retained for sampled patients. The `TRAIN-MEAN NULL MODEL` predicts the training-set mean of the corresponding target for every test observation, in the same original physical units as the Bland-Altman plots; brackets on model and null rows are patient-cluster bootstrap percentile 95% CIs.

Bootstrap replicates: `2000` by default. Random seed: `42` by default. No repository-defined clinically meaningful tolerances were found for these regression targets, so `Within Tolerance` is `NA` until `BLAND_ALTMAN_TOLERANCES` in `scripts/summarize_bland_altman_full_features.py` is populated with explicit defensible thresholds.

**Reference rows.** `PERFECT PREDICTION` denotes exact agreement (`prediction = reference`), giving zero bias, zero LoA width, zero proportional-bias slope, and 100% coverage. `TRAIN-MEAN NULL MODEL` predicts the training-set mean of the corresponding target for every test observation and represents a model with no patient- or window-specific predictive information. The null prediction is derived exclusively from the training data. Lower absolute bias is better, lower LoA half-width is better, proportional-bias slope closer to zero is better, and higher coverage is better. Do not compare raw Bias or LoA half-width across targets with different physical units; compare those primarily within a target.

Machine-readable outputs: `blandaltman_full_features/bland_altman_agreement_summary.csv` and `blandaltman_full_features/bland_altman_agreement_summary.json`.

| Target | Model | Bias [95% CI] | LoA Half-Width [95% CI] | Proportional-Bias Slope [95% CI] | Within Tolerance [95% CI] | N Predictions | N Patients |
|---|---|---:|---:|---:|---:|---:|---:|
| `ALL TARGETS` | `PERFECT PREDICTION` | 0 | 0 | 0 | 100% | - | - |
| `ABP_area_ABP_tau_t_plus_0m_gap` | `history_xgb` | -0.0126 [-0.0428, 0.0213] | 0.9941 [0.9629, 1.0222] | -1.0432 [-1.1098, -0.9718] | NA | 259581 | 269 |
| `ABP_area_ABP_tau_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 0.0035 [-0.0522, 0.0605] | 1.1243 [1.0960, 1.1516] | -2.0000 [-2.0000, -2.0000] | NA | 259581 | 269 |
| `ABP_area_ShockIdx_t_plus_0m_gap` | `history_xgb` | 0.0154 [-0.0036, 0.0338] | 0.7946 [0.7538, 0.8350] | -0.8492 [-0.9161, -0.7923] | NA | 259581 | 269 |
| `ABP_area_ShockIdx_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 0.0369 [-0.0076, 0.0788] | 0.9262 [0.8869, 0.9639] | -2.0000 [-2.0000, -2.0000] | NA | 259581 | 269 |
| `ABP_area_t_plus_0m_gap` | `full_sequence_xgb` | 0.0127 [-0.1117, 0.1391] | 4.2510 [3.8442, 4.6511] | -0.0827 [-0.1010, -0.0669] | NA | 258982 | 268 |
| `ABP_area_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 0.9079 [0.0341, 1.6225] | 11.8216 [10.8533, 12.7481] | -2.0000 [-2.0000, -2.0000] | NA | 258982 | 268 |
| `ABP_tau_t_plus_0m_gap` | `transformer` | -0.0052 [-0.0241, 0.0130] | 0.8505 [0.7721, 0.9356] | -0.2333 [-0.2875, -0.1835] | NA | 258337 | 265 |
| `ABP_tau_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | -0.0366 [-0.1076, 0.0315] | 1.3770 [1.2370, 1.5217] | -2.0000 [-2.0000, -2.0000] | NA | 258337 | 265 |
| `DBP_t_plus_0m_gap` | `history_xgb` | 0.0117 [-0.1660, 0.1752] | 10.9448 [10.0129, 11.9442] | -0.0920 [-0.1092, -0.0793] | NA | 258473 | 265 |
| `DBP_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 0.1172 [-2.4123, 2.3447] | 28.7029 [25.4682, 31.7273] | -2.0000 [-2.0000, -2.0000] | NA | 258473 | 265 |
| `ECG_Ramp_t_plus_0m_gap` | `full_sequence_xgb` | 0.0135 [-0.0035, 0.0304] | 0.5343 [0.4873, 0.5792] | -0.5982 [-0.6961, -0.5024] | NA | 259261 | 269 |
| `ECG_Ramp_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 0.0065 [-0.0296, 0.0426] | 0.6642 [0.6115, 0.7172] | -2.0000 [-2.0000, -2.0000] | NA | 259261 | 269 |
| `HRV_RMSSD_t_plus_0m_gap` | `history_xgb` | -1.5341 [-4.8025, 1.5077] | 81.0808 [74.4020, 87.3173] | -0.2828 [-0.3459, -0.2235] | NA | 259255 | 269 |
| `HRV_RMSSD_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 2.3784 [-6.2252, 11.2383] | 125.6763 [119.8402, 130.7147] | -2.0000 [-2.0000, -2.0000] | NA | 259255 | 269 |
| `HR_range_t_plus_0m_gap` | `history_xgb` | -0.6573 [-1.8313, 0.4211] | 25.2139 [23.0401, 27.2325] | -0.2390 [-0.2903, -0.1873] | NA | 259255 | 269 |
| `HR_range_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 0.4672 [-2.7045, 3.8296] | 41.7212 [37.5915, 45.3751] | -2.0000 [-2.0000, -2.0000] | NA | 259255 | 269 |
| `HR_t_plus_0m_gap` | `full_sequence_xgb` | -0.0633 [-0.8325, 0.7094] | 15.7045 [13.7788, 17.6102] | -0.2094 [-0.2474, -0.1654] | NA | 259254 | 269 |
| `HR_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | -1.1089 [-2.5928, 0.4552] | 27.9882 [25.8803, 29.9725] | -2.0000 [-2.0000, -2.0000] | NA | 259254 | 269 |
| `MAP_t_plus_0m_gap` | `full_sequence_xgb` | -0.0173 [-0.2885, 0.2544] | 13.6407 [12.5636, 14.8276] | -0.1222 [-0.1434, -0.1028] | NA | 258861 | 266 |
| `MAP_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 1.0017 [-1.5618, 3.4517] | 34.0390 [30.3614, 37.6771] | -2.0000 [-2.0000, -2.0000] | NA | 258861 | 266 |
| `PLETH_ACDC_ABP_tau_t_plus_0m_gap` | `history_xgb` | 0.0010 [-0.0141, 0.0169] | 0.8917 [0.8744, 0.9070] | -1.5925 [-1.6397, -1.5443] | NA | 259581 | 269 |
| `PLETH_ACDC_ABP_tau_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 0.0099 [-0.0120, 0.0307] | 0.9230 [0.9056, 0.9390] | -2.0000 [-2.0000, -2.0000] | NA | 259581 | 269 |
| `PLETH_ACDC_PLETH_amp_t_plus_0m_gap` | `history_xgb` | -0.0011 [-0.0049, 0.0031] | 0.2646 [0.2350, 0.2939] | -0.6494 [-0.7602, -0.5550] | NA | 259581 | 269 |
| `PLETH_ACDC_PLETH_amp_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | -0.0035 [-0.0160, 0.0099] | 0.3322 [0.2936, 0.3707] | -2.0000 [-2.0000, -2.0000] | NA | 259581 | 269 |
| `PLETH_ACDC_ShockIdx_t_plus_0m_gap` | `history_xgb` | 0.0141 [-0.0069, 0.0363] | 0.9088 [0.8947, 0.9221] | -1.6637 [-1.6973, -1.6274] | NA | 259581 | 269 |
| `PLETH_ACDC_ShockIdx_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 0.0151 [-0.0105, 0.0425] | 0.9390 [0.9230, 0.9546] | -2.0000 [-2.0000, -2.0000] | NA | 259581 | 269 |
| `PLETH_ACDC_t_plus_0m_gap` | `transformer` | -0.0041 [-0.0092, 0.0006] | 0.1983 [0.1846, 0.2127] | -0.0436 [-0.0689, -0.0116] | NA | 257659 | 269 |
| `PLETH_ACDC_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 0.0045 [-0.0309, 0.0421] | 0.5370 [0.4973, 0.5737] | -2.0000 [-2.0000, -2.0000] | NA | 257659 | 269 |
| `PLETH_amp_ShockIdx_t_plus_0m_gap` | `transformer` | -0.0263 [-0.0464, -0.0060] | 0.8858 [0.8716, 0.8989] | -1.5684 [-1.6066, -1.5283] | NA | 259581 | 269 |
| `PLETH_amp_ShockIdx_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 0.0100 [-0.0144, 0.0354] | 0.9056 [0.8899, 0.9210] | -2.0000 [-2.0000, -2.0000] | NA | 259581 | 269 |
| `PLETH_amp_t_plus_0m_gap` | `history_xgb` | -0.0004 [-0.0047, 0.0040] | 0.2494 [0.2291, 0.2677] | -0.0263 [-0.0328, -0.0210] | NA | 257659 | 269 |
| `PLETH_amp_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 0.0268 [-0.0712, 0.1331] | 1.2160 [1.1282, 1.2836] | -2.0000 [-2.0000, -2.0000] | NA | 257659 | 269 |
| `PPV_t_plus_0m_gap` | `history_xgb` | 0.1284 [-0.6360, 0.8854] | 20.9921 [18.2031, 23.5071] | -0.2972 [-0.3807, -0.2265] | NA | 257423 | 267 |
| `PPV_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | -0.5784 [-3.2803, 1.8209] | 34.9750 [29.3439, 39.7020] | -2.0000 [-2.0000, -2.0000] | NA | 257423 | 267 |
| `PP_t_plus_0m_gap` | `transformer` | 0.9760 [0.6744, 1.3069] | 14.2653 [13.0826, 15.4980] | -0.0637 [-0.0762, -0.0530] | NA | 258229 | 265 |
| `PP_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 2.0511 [-0.7533, 4.6272] | 40.3827 [37.0691, 43.9377] | -2.0000 [-2.0000, -2.0000] | NA | 258229 | 265 |
| `PTT_t_plus_0m_gap` | `history_xgb` | -2.8375 [-6.2986, 0.8088] | 63.8355 [59.4986, 67.9180] | -1.0794 [-1.1661, -0.9970] | NA | 258114 | 268 |
| `PTT_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | -7.4844 [-11.3683, -3.1577] | 69.3925 [64.7858, 73.7798] | -2.0000 [-2.0000, -2.0000] | NA | 258114 | 268 |
| `PVI_t_plus_0m_gap` | `history_xgb` | 0.0583 [-0.5937, 0.6858] | 18.4773 [16.3219, 21.0653] | -0.2102 [-0.2709, -0.1660] | NA | 257288 | 269 |
| `PVI_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | -0.1480 [-2.6002, 2.0121] | 33.5966 [30.6889, 36.2388] | -2.0000 [-2.0000, -2.0000] | NA | 257288 | 269 |
| `RESP_amp_t_plus_0m_gap` | `transformer` | 0.0115 [0.0048, 0.0174] | 0.3125 [0.2882, 0.3373] | -0.0671 [-0.0796, -0.0505] | NA | 259242 | 269 |
| `RESP_amp_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | -0.0288 [-0.1045, 0.0408] | 0.8337 [0.6909, 0.9997] | -2.0000 [-2.0000, -2.0000] | NA | 259242 | 269 |
| `RR_t_plus_0m_gap` | `transformer` | -0.0040 [-0.0977, 0.0898] | 3.3259 [3.1539, 3.4955] | -0.2998 [-0.3689, -0.2337] | NA | 259218 | 269 |
| `RR_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | -0.1004 [-0.3783, 0.1744] | 4.9493 [4.6049, 5.3002] | -2.0000 [-2.0000, -2.0000] | NA | 259218 | 269 |
| `SBP_t_plus_0m_gap` | `full_sequence_xgb` | 0.0450 [-0.3625, 0.4537] | 16.5132 [15.5959, 17.4933] | -0.0869 [-0.0949, -0.0788] | NA | 258473 | 265 |
| `SBP_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 2.0803 [-0.9473, 5.3048] | 45.7404 [41.5866, 50.5330] | -2.0000 [-2.0000, -2.0000] | NA | 258473 | 265 |
| `ShockIdx_ABP_tau_t_plus_0m_gap` | `history_xgb` | -0.0142 [-0.0491, 0.0171] | 1.0107 [0.9878, 1.0293] | -1.5159 [-1.5620, -1.4691] | NA | 259581 | 269 |
| `ShockIdx_ABP_tau_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | -0.0184 [-0.0622, 0.0203] | 1.0605 [1.0365, 1.0816] | -2.0000 [-2.0000, -2.0000] | NA | 259581 | 269 |
| `ShockIdx_t_plus_0m_gap` | `history_xgb` | -0.0031 [-0.0113, 0.0055] | 0.2069 [0.1839, 0.2328] | -0.1702 [-0.2091, -0.1430] | NA | 258146 | 265 |
| `ShockIdx_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | -0.0275 [-0.0625, 0.0054] | 0.4622 [0.3757, 0.5618] | -2.0000 [-2.0000, -2.0000] | NA | 258146 | 265 |
| `dPdt_max_t_plus_0m_gap` | `transformer` | -2.5512 [-8.5934, 3.2950] | 286.3405 [260.4128, 312.3149] | -0.0494 [-0.0594, -0.0394] | NA | 258583 | 268 |
| `dPdt_max_t_plus_0m_gap` | `TRAIN-MEAN NULL MODEL` | 30.3112 [-37.1582, 93.2573] | 873.6652 [789.5939, 953.0501] | -2.0000 [-2.0000, -2.0000] | NA | 258583 | 268 |

## Verification

- `scripts/build_full_data_feature_regression_targets.py` syntax checked.
- `scripts/train_feature_models.py` now supports `anchor_id` target alignment for segment-aware full-data bundles while retaining legacy patient/time alignment for old bundles.
- Source schema check passed after `X_stats.npy` became readable: `X_stats.npy` shape `(2847597, 19, 109)`, `corr_features_focused.npy` shape `(2847597, 7)`.
- Built `outputs/targets/feature_targets_gap_full_data.npz` with horizons `0`, `20`, and `60` using `gap` semantics and the same `26` base targets as `docs/v7_extracted_features/extractedFeaturesRegression.md`.
- Target diagnostics: rows with future source are `1808846` at `t+0m_gap`, `1677548` at `t+20m_gap`, and `1463716` at `t+60m_gap`.
- Downstream alignment smoke check passed for `MAP_t_plus_0m_gap`: `1969515` feature-cache anchors matched by `anchor_id`, `1801312` valid labels, `0` missing target rows.

## 2026-08-31 Target Builder Hardening

The full-data regression target builder was audited and hardened without changing feature extraction or target semantics.

Implemented safeguards:

- `patient_ids.npy`, `seg_names.npy`, and other string metadata are loaded without memory mapping so object-dtype arrays are accepted safely.
- Source targets are keyed by `(patient_id, seg_name, quantized_window_time)` with six-decimal timestamp quantization, duplicate-source rejection, and segment-grouped `searchsorted` matching instead of one dictionary entry per source row.
- Cache integrity is validated before target construction: `values.npy` must be 3D, `mask.npy` must match it, `metadata.json` must agree with tensor shape, merged caches must have `_SUCCESS`, and all identity arrays must have length `N`.
- Cache identity is validated against `anchors.csv`: row counts, row-wise patient IDs, row-wise strict integral anchor IDs, row-wise split labels, unique `anchor_id`, and unique segment-aware anchor identity.
- Patient, segment, split, and source identifiers reject missing, blank, `nan`, `none`, and `null` values before key construction.
- `anchor_times.npy` is preserved as the canonical bundle `anchor_time`; `window_time` remains the source lookup time. Timing metadata now records validated relationships rather than asserting a semantic time basis that was not independently proven.
- A current-source identity audit verifies that every cache anchor maps exactly back to a current source row before any future offset is applied.
- `X_stats` shape handling now validates 2D and both 3D layouts, rejects ambiguous feature axes, and implements only configured mean aggregation.
- Correlation-feature shape and ordering are validated against the actual source sidecar `corr_features_focused_names.json:features`.
- The production source root does not contain source-side `X_stats` feature-name metadata; the builder records `x_stats_feature_order_verified: false` and an explicit historical-assumption status instead of claiming verification. Synthetic tests cover exact `X_stats` metadata verification and wrong-order rejection.
- Final source target matrix shape and target-name ordering are validated before matching.
- Per-segment time-spacing diagnostics are reported for source lookup times so cross-segment jumps do not obscure the primary spacing.
- Per-horizon diagnostics now separate source-row matches from finite target availability and include match rates, rows with any valid target, entirely non-finite matched rows, valid-value counts/fractions, and per-feature valid counts/fractions.
- Optional `--min-source-match-rate` can make low nonzero match rates fatal when a run has an established expected boundary-loss threshold.
- Target bundles now preserve `split_labels` in addition to stable anchor/segment identity arrays already written by `save_target_bundle()`.
- After writing, the builder reopens the target bundle and validates required identity arrays, target/mask shapes, mask equals target finiteness, unique saved anchor IDs, and metadata target-name order.
- Downstream feature-model alignment now prefers `anchor_id`, then segment-aware composite identity, then legacy `(patient_id, quantized_anchor_time)` only when unique.

Validation run:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m py_compile scripts/build_full_data_feature_regression_targets.py scripts/train_feature_models.py scripts/train_patchtst.py waveform_baselines/target_builders.py waveform_baselines/wf_features/cache.py
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_full_data_feature_regression_targets
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_target_generation tests.test_waveform_feature_pipeline
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest discover tests
```

All discovered tests passed: `164` tests in `36.109` seconds.

Second hardening validation:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m py_compile scripts/build_full_data_feature_regression_targets.py scripts/train_feature_models.py scripts/train_patchtst.py waveform_baselines/target_builders.py waveform_baselines/wf_features/cache.py tests/test_full_data_feature_regression_targets.py
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_full_data_feature_regression_targets
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_target_generation tests.test_waveform_feature_pipeline
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest discover tests
```

All discovered tests passed after the second hardening pass: `169` tests in `28.785` seconds.

Real-data subset smoke:

- Built a temporary `/tmp` target bundle from two real full-data segments: `p000160/3531764_0003` and `p000188/3317157_0002`.
- Smoke output shape was `(8, 78)` with `624` valid target values.
- Gap offsets were verified as `1200`, `2400`, and `4800` seconds for horizons `0`, `20`, and `60`.
- Manually inspected anchors matched source rows from the same patient and segment only.
- For `gap, horizon=0`, inspected target centers began immediately after the input interval; for example anchor center `600.0` seconds had input `[0.0, 1200.0]` and target center `1800.0`, corresponding to target window `[1200.0, 2400.0]`.
- Subset source match rates were `1.0` for all three horizons.

Second smoke run:

- Command: inline Python smoke harness using `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python` to build a temporary bundle from early and near-end anchors in two real segments.
- Audit CSV: `outputs/targets/full_data_regression_target_smoke_audit_20260831.csv`.
- Smoke output shape was `(12, 78)` with `468` valid target values.
- Current-source identity audit matched `12/12` anchors: match rate `1.0`.
- Horizon match rates were `0.5` for `0`, `20`, and `60` minute gap horizons because early anchors matched and near-end anchors intentionally had no future source within the same segment.
- Rows whose matched source was entirely non-finite: `0` at all smoke horizons.
- Valid value fraction was `0.5` at all smoke horizons; `HR` valid fraction was `0.5` at all smoke horizons.
- Manual audit rows confirmed `gap h=0` uses `+1200` seconds, `gap h=20` uses `+2400` seconds, `gap h=60` uses `+4800` seconds, matched future rows stayed in the same `(patient_id, seg_name)`, and near-end anchors remained all-NaN/all-false rather than crossing segment boundaries.
- Post-write bundle audit passed for required arrays and mask/target consistency.

Recommended production rebuild command:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/build_full_data_feature_regression_targets.py \
  --output outputs/targets/feature_targets_gap_full_data_hardened_v2.npz \
  --feature-horizon-mode gap \
  --horizons 0 20 60
```

Use a different `--output` path if preserving the existing `hardened_v2` artifact.

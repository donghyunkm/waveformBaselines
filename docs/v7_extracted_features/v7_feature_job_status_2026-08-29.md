# v7 Feature Job Status - 2026-08-29

## Summary

The corrected v7 waveform-feature extraction and merge finished successfully.
The subsequent v7 feature-model training jobs did not finish successfully.

## Feature Extraction

Checked jobs:

- extraction array: SLURM `26873594`
- shard merge: SLURM `26873626`
- dependent jobs: SLURM `26873627`-`26873635`

`squeue` no longer listed any of these jobs. The production v7 feature cache
has a merged `_SUCCESS` marker at:

```text
/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/v7/vasopressor_free_waveform_features_v7
```

All 32 shard directories also had `_SUCCESS` markers.

The merge log reported:

- `n_samples`: `334833`
- shape: `(334833, 20, 93)`

The merged cache files were present:

- `_SUCCESS`
- `values.npy`
- `mask.npy`
- `patient_ids.npy`
- `anchor_times.npy`
- `anchor_ids.npy`
- `split_labels.npy`
- `metadata.json`
- `feature_quality_report.json`

Some shard stderr files contained WFDB runtime warnings, but the merge completed
and produced the validated cache.

## Downstream Training

The v7 downstream feature-model jobs failed during target alignment:

- tabular/persistence: SLURM `26873627`-`26873631`
- GRU/Transformer: SLURM `26873632`-`26873635`

Representative error:

```text
ValueError: Missing target rows for 3610 feature cache anchors: ('p003866', 4253333041.1600003), ('p003866', 4253333191.1600003), ('p003866', 4253333341.1600003), ('p003866', 4253333491.1600003), ('p003866', 4253333641.1600003)
```

The v7 feature-model output directories under `outputs/feature_models/*_v7`
contained only `preprocessing.json` files at inspection time. No completed
model metrics or checkpoints should be interpreted for these v7 dependent runs.

## Next Steps

- Diagnose why `3610` anchors in the merged v7 feature cache do not have rows
  in the selected target bundle.
- Decide whether to filter the feature cache to target-covered anchors or adjust
  target generation/alignment so the intended anchor universe matches.
- Resubmit the v7 tabular/persistence and sequence model jobs after alignment
  is fixed.
- Complete the full-cohort v7 feature-quality audit before interpreting
  retrained downstream model results.

## Follow-up

The cache/target mismatch was diagnosed as sub-microsecond floating-point
representation drift in `anchor_time`, not a real difference in the intended
anchor universe. Rounding `(patient_id, anchor_time)` joins to 6 decimal places
aligned both current target bundles to the production cache with zero missing
rows:

- regression bundle: `334833` rows, `323813` valid `MAP_t_plus_0m_gap` targets
- filtered classification bundle: `334833` rows, `34338` valid
  `hypotension_within_5m` targets

A local persistence smoke completed after the fix:

```json
{
  "mae": 4.719614505767822,
  "rmse": 7.760484064479423,
  "r2": 0.7158088088035583
}
```

The v7 full-cohort feature-quality audit summary was written to
`outputs/feature_models/v7_full_cohort_feature_quality_audit_2026-08-29.json`.
It found no zero-valid features and low maximum missingness (`6.47%` for
`delta_pleth_amplitude_median`), but flagged an ECG/ABP rate-disagreement tail:
`23.4%` of paired tokens had absolute ECG/ABP rate difference above `20` bpm.
Representative overlays were saved under `docs/figures/`.

Resubmitted v7 jobs:

- regression tabular/persistence: `26898023`, `26898026`, `26898027`
- regression sequence: `26898042`, `26898048`
- classification tabular: `26898024`, `26898025`
- classification sequence: `26898044`, `26898046`

Completed at follow-up inspection:

- persistence regression `26898023`: MAE `4.7196`, RMSE `7.7605`, R2 `0.7158`
- current-state XGBoost regression `26898027`: MAE `4.3627`, RMSE `6.7218`, R2 `0.7874`
- GRU regression `26898042`: MAE `4.2501`, RMSE `6.5019`, R2 `0.8010`
- Transformer regression `26898048`: MAE `4.1050`, RMSE `6.2483`, R2 `0.8163`
- current-state XGBoost classification `26898025`: AUROC `0.9939`, AUPRC
  `0.8950`, specificity at 85% sensitivity `0.9902`
- GRU classification `26898046`: AUROC `0.9905`, AUPRC `0.8592`,
  specificity at 85% sensitivity `0.9818`
- Transformer classification `26898044`: AUROC `0.9850`, AUPRC `0.6647`,
  specificity at 85% sensitivity `0.9754`

Later queue/result check:

- history XGBoost regression jobs `26898378`-`26898403` finished and produced all `26` metrics artifacts
- duplicate history MAP jobs `26898026` and `26898383` both wrote `outputs/feature_models/history_xgb_feature_MAP_t_plus_0m_gap_v7`; their stdout logs report identical zero-missing alignment counts and identical metrics, with empty stderr
- history XGBoost classification `26898024` finished with zero missing aligned targets and empty stderr; metrics were AUROC `0.9955`, AUPRC `0.9215`, specificity at 85% sensitivity `0.9922`

Additional all-target v7 regression jobs were submitted with
`slurm/submit_feature_regression_v7_all_targets.sh` for all 26 `t+0m_gap`
targets and four model families: current-state XGBoost, history XGBoost, GRU,
and Transformer. The submitter skipped existing MAP metrics and submitted
`101` new jobs:

- current-state XGBoost: `26898353`-`26898377`
- history XGBoost: `26898378`-`26898403`
- GRU: `26898404`-`26898428`
- Transformer: `26898429`-`26898453`

Latest generated status artifact:
`outputs/feature_models/v7_all_target_regression_status_2026-08-29.json`.
The refreshed artifact records current-state XGBoost, history XGBoost, GRU, and
Transformer complete for `26/26`. See `docs/v7_extracted_features/extractedFeaturesRegression.md`
for the refreshed history-XGBoost and best-completed-v7 tables.

# Bland-Altman Analysis

Compact summary of the vasopressor-free regression agreement plots under `outputs/patchtst/vasopressor_free/`.

## Artifacts

- summary: `outputs/patchtst/vasopressor_free/bland_altman_summary.json`
- per-task stats: `outputs/patchtst/vasopressor_free/<task>/bland_altman_stats.json`
- per-task figures: `outputs/patchtst/vasopressor_free/<task>/bland_altman.png`
- flat export: `blandaltman/`

## Main Patterns

- Mean bias is usually small relative to the limits of agreement.
- The main failure mode is range compression, not global offset.
- Pressure-family targets, `HR`, `RR`, `PTT`, and interaction targets show the clearest mean-dependent bias.
- `PPV`, `ABP_area`, `ABP_tau`, `PVI`, `ECG_Ramp`, and `RESP_amp` show the clearest heteroscedasticity.

## Best Agreement

- `PLETH_ACDC`
- `dPdt_max`
- `PVI`
- `HR`
- `HR_range`

These targets have the tightest residual bands relative to scale, though several still show range compression at the extremes.

## Weakest Agreement

- interaction targets
- `PPV`
- `ABP_tau`
- `RESP_amp`
- `RR`

These failures look systematic rather than outlier-driven.

## Full-Data Extracted-Feature Best Models

Generated on `2026-09-01` for the best completed full-data extracted-feature regression model per target, using the best-model selections from `outputs/feature_models/full_data_regression_patient_bootstrap_ci_2026-09-01.csv`.

Command:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/make_full_data_best_bland_altman_plots.py --output-dir blandaltman_full_features
```

Artifacts:

- flat plot export: `blandaltman_full_features/`
- plots: `26` PNG files, one per target/model pair
- per-plot stats: `blandaltman_full_features/<target>__<model>.json`
- combined summary: `blandaltman_full_features/bland_altman_full_features_summary.json`

The plots use held-out test predictions from the selected best model for each target, with differences defined as `prediction - target` and limits of agreement as mean difference plus/minus `1.96` sample standard deviations.

## v7 Extracted-Feature Best Models

Generated on `2026-09-01` for the documented best completed v7 extracted-feature regression model per target in `docs/v7_extracted_features/extractedFeaturesRegression.md`.

Commands:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/make_v7_extractedfeatures_best_bland_altman_plots.py
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/summarize_bland_altman_v7_extractedfeatures.py --bootstrap-replicates 2000 --seed 42
```

Artifacts:

- flat plot export: `blandaltman_v7_extractedfeatures/`
- plots: `26` PNG files, one per documented target/model winner
- per-plot stats: `blandaltman_v7_extractedfeatures/<target>__<model>.json`
- combined plot summary: `blandaltman_v7_extractedfeatures/bland_altman_v7_extractedfeatures_summary.json`
- current GRU-vs-PatchTST agreement summary outputs: `blandaltman_v7_extractedfeatures/bland_altman_agreement_summary.{csv,json,md}`
- current paired comparison outputs: `blandaltman_v7_extractedfeatures/bland_altman_paired_significance.{csv,json,md}` and `blandaltman_v7_extractedfeatures/bland_altman_paired_validation.csv`
- Markdown tables inserted in `docs/v7_extracted_features/extractedFeaturesRegression.md`

The PNG plots use the saved v7 prediction exports under `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/regression` and were generated for the documented best completed v7 model per target. Those plots use all finite held-out test predictions from each selected model's `test_predictions.npz`; for example, `ABP_area_t_plus_0m_gap__gru.png` used `35701` test predictions from `97` patients.

The current agreement and paired-comparison tables were regenerated on `2026-09-01` by `scripts/paired_bland_altman_v7_vs_patchtst.py` for `gru` labeled as `V7` versus `PatchTST v1 raw waveform`, not for the best v7 model on every target. They use the same Bland-Altman metric definitions, align raw row-level predictions by stable patient/anchor sample identity, and report `52` descriptive rows, `78` paired comparison rows, and `26` validation rows. The paired comparison uses `10000` paired patient-cluster bootstrap replicates with seed `20260901`, no p-values, and no multiple-comparison adjustment. Undefined Within Tolerance rows/columns are omitted from the displayed/current CSV tables because no target-specific tolerance thresholds are defined.

## Full-Data Agreement Summary Table

Generated on `2026-09-01` after the full-data Bland-Altman plots. The reusable script is:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/summarize_bland_altman_full_features.py --bootstrap-replicates 2000 --seed 42
```

Artifacts:

- Markdown table inserted in `docs/full_data/extractedFeaturesRegressionFullData.md` under `Bland-Altman Agreement Summary`
- machine-readable CSV: `blandaltman_full_features/bland_altman_agreement_summary.csv`
- machine-readable JSON: `blandaltman_full_features/bland_altman_agreement_summary.json`
- standalone Markdown: `blandaltman_full_features/bland_altman_agreement_summary.md`

The table has one target-independent `PERFECT PREDICTION` row, one model row per plotted target/model pair, and one `TRAIN-MEAN NULL MODEL` row per target. The null model predicts the target's training-set mean from `outputs/targets/feature_targets_gap_full_data.npz`, using the full-data v7 feature cache `split_labels.npy == train`; it does not use the test-set mean. Model and null rows use patient-cluster bootstrap percentile 95% CIs with `2000` replicates and seed `42`. Validation checks in the JSON confirm that HR, RR, and SBP recalculated model rows match the existing plotted bias/lower-LoA/upper-LoA lines exactly and that null-model bias and slope sanity checks passed. No repository-defined clinically meaningful tolerances were found, so `Within Tolerance` remains `NA` until explicit thresholds are added to `BLAND_ALTMAN_TOLERANCES` in the script.

# Extracted Feature Classification

This page tracks downstream classification work for the waveform-feature representation introduced on `2026-08-28`.

Current status:

- feature representation implemented
- classification baselines implemented
- v7 shard merge completed
- original v7 classification dependent jobs failed during target alignment before metrics/checkpoints
- target alignment fixed by 6-decimal anchor-time rounding; v7 classification dependent jobs were resubmitted
- completed v7 classification metrics and prediction exports exist for current-state XGBoost, history XGBoost, full-sequence XGBoost, full-sequence MLP, GRU, Transformer, and TCN

## Fixed Task Setup

The implementation is designed to reuse the repository's existing classification setup documented in `docs/v1_vasopressor_free/classification_results_v1_vaso_free.md`.

Preserved inputs to the classification workflow:

- cohort: vasopressor-free overlap cohort
- patient-disjoint splits: `outputs/splits/vasopressor_free_splits.json`
- event target bundles, including filtered and non-filtered variants
- evaluation convention: AUROC, AUPRC, and specificity at 85% sensitivity from the held-out test-score threshold sweep, matching the current table convention

## Implemented Models

Via `scripts/train_feature_models.py`; see `docs/v7_extracted_features/extractedFeaturesModelDescription.md` for architecture and preprocessing details:

- current-state baseline: `current_state_xgb` or `current_state_linear`
- ordered history summary baseline: `history_xgb`
- full-sequence tabular baseline: `full_sequence_xgb`
- full-sequence neural baseline: `full_sequence_mlp`
- temporal convolution baseline: `tcn`
- temporal sequence baseline: `gru`
- primary temporal model: `transformer`

All consume the same cached `(20, 93)` physiological feature sequence, after train-only preprocessing.

## Current Filtered Bundle

This is the filtered comparison dataset used by the displayed v7 extracted-feature classification results. It is the earlier vasopressor-free legacy counterpart to the full-data `Current Filtered Bundle` section in `docs/full_data/extractedFeaturesClassificationFullData.md`.

Important semantic difference from full data: this bundle uses the legacy `anchor_horizon_filtered` target path, not the corrected full-data `anchor_onset_within_horizon` target path. Therefore its target names do not include `_onset_`, and its negative filtering uses older global/recording rules rather than the full-data complete-source-recording last-confirmed-onset cutoff.

The legacy and corrected labels are similar in spirit because both avoid training on anchors where an event is already active in the input window, but they are not equivalent. The legacy `anchor_horizon` positive is based on finding a fully sustained event inside the future horizon window itself. With a `5m` horizon and `5m` sustain rule, an event that starts `3` minutes after prediction time and continues for `5` minutes can be missed by the legacy label because only the first `2` event minutes fall inside the `5m` horizon. The corrected full-data onset label instead asks whether the event episode starts inside the horizon and allows the `5m` confirmation period to extend beyond the horizon. Thus the corrected full-data label is closer to "does a new confirmed episode begin soon?", while this v7 bundle is closer to "is a sustained event fully visible within the horizon window?"

Artifacts:

- Bundle: `outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz`
- Metadata JSON: `outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.json`
- Shape: `(334833, 4)`
- Target columns: `hypotension_within_5m`, `tachycardia_within_5m`, `hypotension_within_10m`, `tachycardia_within_10m`
- Alignment smoke check: `334833` feature-cache rows, `34338` valid `hypotension_within_5m` targets, zero missing aligned rows

Filtering rules:

- Positive labels are preserved from the base anchor-horizon label.
- Hypotension negatives are retained only when the corresponding outcome window is valid and clean.
- Hypotension negatives from recordings with any sustained hypotension event are removed.
- Remaining hypotension negatives later than the global mean last-event time across positive recordings are removed.
- These legacy filters were only used for hypotension. Tachycardia columns are present for compatibility with the broader event-target pipeline but were not used for the displayed v7 extracted-feature classification comparison.

Filtered target counts after legacy negative selection:

| Target | Filtered valid | Filtered positive | Filtered negative | Filtered prevalence | Train filtered valid/pos | Val filtered valid/pos | Test filtered valid/pos |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hypotension_within_5m` | 34,338 | 1,783 | 32,555 | 5.192% | 26,713 / 1,406 | 3,975 / 183 | 3,650 / 194 |
| `hypotension_within_10m` | 36,314 | 3,957 | 32,357 | 10.897% | 28,259 / 3,108 | 4,182 / 412 | 3,873 / 437 |
| `tachycardia_within_5m` | 280,714 | 824 | 279,890 | 0.294% | 218,679 / 627 | 30,405 / 123 | 31,630 / 74 |
| `tachycardia_within_10m` | 280,729 | 1,885 | 278,844 | 0.671% | 218,685 / 1,436 | 30,409 / 283 | 31,635 / 166 |

Filtered negative-removal reasons for hypotension:

| Target | Kept negatives | Base invalid | Positive | Recording contains sustained hypotension | After global mean last-event cutoff | Outcome window invalid | Clean-window failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hypotension_within_5m` | 32,555 | 54,668 | 1,783 | 109,946 | 87,617 | 48,148 | 116 |
| `hypotension_within_10m` | 32,357 | 54,511 | 3,957 | 107,838 | 87,671 | 48,383 | 116 |

Meaning:

- `Kept negatives`: valid base negatives that survived all legacy filtering.
- `Base invalid`: rows already invalid before negative filtering; computed as all anchor rows minus positives and base negative candidates.
- `Positive`: future-event positives retained by the filters.
- `Recording contains sustained hypotension`: base negatives removed because their recording had any sustained hypotension event.
- `After global mean last-event cutoff`: base negatives removed because they occurred later than the global mean last-event time across positive recordings.
- `Outcome window invalid`: base negatives removed because the required outcome window did not have sufficient valid coverage.
- `Clean-window failed`: rows with valid outcome windows that nevertheless contained hypotensive minutes.

Held-out test split filtering breakdown:

| Target | Kept negatives | Base invalid | Positive | Recording contains sustained hypotension | After global mean last-event cutoff | Outcome window invalid | Clean-window failed | Final valid test examples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hypotension_within_5m` | 3,456 | 4,518 | 194 | 14,661 | 9,153 | 4,078 | 13 | 3,650 |
| `hypotension_within_10m` | 3,436 | 4,519 | 437 | 14,433 | 9,155 | 4,104 | 13 | 3,873 |

These counts explain why the displayed `5m` hypotension benchmark is much smaller and more enriched than the unfiltered event-label setting: it keeps all `194` test positives but reduces the valid test denominator to `3650`.

## v7 Filtered 5m Hypotension Results

Comparison baseline is the filtered vasopressor-free `patchtst_v1` `5m` hypotension result in `docs/v1_vasopressor_free/classification_results_v1_vaso_free.md`: AUROC `0.975`, AUPRC `0.594`, specificity at 85% sensitivity `0.959`, with `194 / 3650` positives. The v7 feature-model rows below use `outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz` and should be interpreted with the v7 ECG/ABP rate-disagreement caveat in `docs/v7_extracted_features/extractedFeatures.md`.

| Model | N | Pos | AUROC | Delta AUROC | Delta AUROC 95% CI | AUROC p | AUPRC | Delta AUPRC | Delta AUPRC 95% CI | AUPRC p | Spec @85% | Delta Spec | Delta Spec 95% CI | Spec p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PatchTST v1 raw waveform | 3650 | 194 | 0.9746 | - | - | - | 0.5939 | - | - | - | 0.9592 | - | - | - |
| Current-state XGBoost | 3650 | 194 | 0.9939 | +0.0193 | [+0.0044, +0.0439] | 0.8401 | 0.8950 | +0.3011 | [+0.0506, +0.5020] | 0.05059 | 0.9902 | +0.0310 | [+0.0072, +0.0690] | 0.9301 |
| History XGBoost | 3650 | 194 | 0.9955 | +0.0209 | [+0.0052, +0.0452] | 0.8284 | 0.9215 | +0.3276 | [+0.0591, +0.5274] | 0.0282 | 0.9922 | +0.0330 | [+0.0083, +0.0729] | 0.9213 |
| Full-sequence XGBoost | 3650 | 194 | 0.9941 | +0.0195 | [+0.0050, +0.0416] | 0.8342 | 0.8986 | +0.3047 | [+0.0627, +0.4618] | 0.0362 | 0.9931 | +0.0339 | [+0.0084, +0.0715] | 0.9271 |
| Full-sequence MLP | 3650 | 194 | 0.9655 | -0.0091 | [-0.0318, +0.0096] | 0.9242 | 0.5339 | -0.0600 | [-0.2870, +0.1831] | 0.6616 | 0.9375 | -0.0217 | [-0.0570, +0.0129] | 0.9484 |
| GRU | 3650 | 194 | 0.9890 | +0.0144 | [+0.0039, +0.0297] | 0.8794 | 0.8266 | +0.2327 | [+0.0485, +0.3549] | 0.06139 | 0.9771 | +0.0179 | [+0.0018, +0.0403] | 0.9558 |
| Transformer | 3650 | 194 | 0.9896 | +0.0150 | [+0.0011, +0.0364] | 0.8732 | 0.8280 | +0.2342 | [+0.0011, +0.4426] | 0.1636 | 0.9789 | +0.0197 | [+0.0007, +0.0462] | 0.9556 |
| TCN | 3650 | 194 | 0.9897 | +0.0151 | [+0.0004, +0.0378] | 0.8741 | 0.7741 | +0.1802 | [-0.0623, +0.4294] | 0.3378 | 0.9832 | +0.0240 | [+0.0033, +0.0573] | 0.9456 |


Higher is better for all listed classification metrics. The table uses aligned held-out predictions for all significance columns. Every delta, CI, and p-value is row-specific and compares that row’s model against `PatchTST v1 raw waveform` on the same held-out test observations; these are not pairwise comparisons among v7 models. Delta convention is `comparison model - PatchTST`, so positive deltas favor the row’s model. The 95% CI is a confidence interval for the true metric delta: if the same study were repeated many times and a 95% CI were built the same way each time, about 95% of those intervals would contain the true population-level delta. P-values are patient-clustered paired permutation p-values. All completed v7 feature models except full-sequence MLP improved over the prior filtered `patchtst_v1` `5m` result on AUROC, AUPRC, and specificity at 85% sensitivity. History XGBoost is the best completed model on AUROC and AUPRC. Full-sequence XGBoost has the highest specificity at 85% sensitivity (`0.9931`), just ahead of history XGBoost (`0.9922`), while current-state XGBoost remains close behind; the sequence models trail the tabular XGBoost baselines for this filtered hypotension task.

## Results Analysis

For filtered `5m` hypotension classification, the XGBoost, GRU, TCN, and Transformer v7 engineered-feature models outperform the prior raw-waveform `patchtst_v1` baseline, while full-sequence MLP underperforms it. The strongest AUROC/AUPRC result is history XGBoost (`AUROC=0.9955`, `AUPRC=0.9215`), while full-sequence XGBoost gives the highest specificity at 85% sensitivity (`0.9931`). The main gain over PatchTST is in precision-recall performance: AUPRC rises from `0.594` to `0.9215` for history XGBoost, which is the most relevant improvement given the low test prevalence (`194/3650`, `5.3%`).

The tabular XGBoost models, including full-sequence XGBoost, beat the GRU, TCN, and Transformer sequence models on all reported metrics. This suggests that, for this filtered short-horizon hypotension task, explicit physiological features and nonlinear tabular interactions carry more useful signal than learned temporal sequence modeling over the same feature cache. The small advantage of history XGBoost and full-sequence XGBoost over current-state XGBoost indicates that recent temporal context helps, but the context is captured well by tabular representations rather than requiring a neural sequence model.

Full-sequence MLP is the weakest v7 classifier here (`AUROC=0.9655`, `AUPRC=0.5339`, specificity at 85% sensitivity `0.9375`) and is the only completed v7 classifier below PatchTST. Its PatchTST-relative deltas are not statistically significant by the paired permutation tests. Transformer, GRU, and TCN improve over PatchTST but remain below the XGBoost variants. Before treating the classification result as final, the next useful checks are probability calibration, threshold stability, and inspection under the v7 ECG/ABP rate-disagreement caveat documented in `docs/v7_extracted_features/extractedFeatures.md`.

## Prediction Export Reruns

The reported filtered `5m` v7 classification rows were rerun with aligned test prediction export enabled so significance testing can be performed for every table row. Outputs are written under `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/classification`. The original four classification prediction-export jobs were included in the `134`-job rerun batch submitted on 2026-08-29; full-sequence XGBoost and full-sequence MLP were added by separate completed jobs.

## Expected Classification Inputs

- physiological cache: `(N, 20, 93)`
- validity mask: `(N, 20, 93)`
- model-ready tensor after preprocessing: `(N, 20, 186)`
- current-state baseline input: last token only
- history-XGBoost baseline input: `930` summary features
- full-sequence XGBoost input: flattened full sequence with `3720` features
- full-sequence MLP input: flattened full sequence with `3720` features

## Evaluation Discipline

- preprocessing fit: training split only
- threshold selection: validation split only
- final metrics: test split only
- patient overlap across splits: not allowed

## Jobs

- Completed TCN-only classification significance job `26919472`; outputs written to `outputs/feature_models/classification_patchtst_v7_significance_tcn_only_2026-08-29.{json,csv,md}` with 1 result and zero failures.

- Completed TCN classification job `26919423`; metrics and `test_predictions.npz` are available at `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/classification/tcn_hypotension_within_5m_filtered_v7`.

- Completed MLP-inclusive classification significance job `26919239` with models `current_state_xgb`, `history_xgb`, `full_sequence_xgb`, `full_sequence_mlp`, `gru`, and `transformer`; outputs written to `outputs/feature_models/classification_patchtst_v7_significance_with_mlp_2026-08-29.{json,csv,md}`.

- Optimized classification significance resampling, cancelled superseded job `26919215`, and completed optimized analysis as SLURM `26919217`; final MLP-inclusive non-TCN table source is `26919239`.

- Superseded optimized completed-model classification significance job `26919215` was cancelled before the final MLP-inclusive rerun.

- Completed `full_sequence_mlp` GPU jobs `26919162`-`26919188`; all regression/classification metrics and prediction exports are available for downstream significance testing.

- Completed `full_sequence_xgb` jobs `26919125`-`26919151` to generate regression/classification prediction exports for downstream significance testing.

- Superseded slow classification significance job `26919099` was cancelled; optimized reruns produced the displayed significance outputs.

SLURM wrappers added:

- `slurm/extract_waveform_features.sh`
- `slurm/train_feature_model.sh`

Job status:

- invalidated `v1` jobs `26870150`, `26870151`, `26870154`, `26870155` were cancelled because the audited feature representation changed numerically
- v2 extraction/model jobs `26870362`–`26870371` were cancelled because the feature representation had numerical correctness bugs
- v3 extraction/model jobs `26870874`, `26870876`-`26870884` were cancelled because the final audit found additional numerical correctness issues
- interim v4 jobs `26871441`, `26871446`-`26871454` were cancelled after the final refined-foot endpoint correction
- v4 extraction/model jobs `26871618`, `26871691`-`26871699` were cancelled because the final v4 audit found additional numerical correctness issues
- v6 extraction/dependent jobs `26873175`, `26873177`-`26873185` were cancelled before completion; pre-hardening v7 jobs `26873436`, `26873437`, `26873469`-`26873477` were cancelled
- corrected v7 extraction array `26873594` and merge `26873626` completed
- v7 dependent classification jobs failed during target alignment before metrics/checkpoints: tabular jobs `26873630`-`26873631` and sequence jobs `26873634`-`26873635`
- alignment fix smoke passed against the production v7 cache and `outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz`: `334833` rows, `34338` valid `hypotension_within_5m` targets, zero missing rows
- completed resubmitted v7 classification job `26898025` current-state XGBoost: AUROC `0.9939`, AUPRC `0.8950`, specificity at 85% sensitivity `0.9902`
- completed resubmitted v7 classification job `26898046` GRU: AUROC `0.9905`, AUPRC `0.8592`, specificity at 85% sensitivity `0.9818`
- completed resubmitted v7 classification job `26898044` Transformer: AUROC `0.9850`, AUPRC `0.6647`, specificity at 85% sensitivity `0.9754`
- completed resubmitted v7 classification job `26898024` history XGBoost: AUROC `0.9955`, AUPRC `0.9215`, specificity at 85% sensitivity `0.9922`; log reported zero missing aligned targets and empty stderr
- completed full-sequence XGBoost classification job `26919151`: AUROC `0.9941`, AUPRC `0.8986`, specificity at 85% sensitivity `0.9931`; prediction export includes `3650` test rows, `194` positives, and `66` patients
- completed full-sequence MLP classification job `26919188`: AUROC `0.9655`, AUPRC `0.5339`, specificity at 85% sensitivity `0.9375`; prediction export includes `3650` test rows, `194` positives, and `66` patients

## Next Steps

1. Add predicted-probability calibration and threshold-stability diagnostics for the completed v7 classification models.
2. Review whether TCN changes the classification narrative; current TCN metrics remain below XGBoost models and have high PatchTST-relative paired permutation p-values.
3. Interpret classification results with the v7 ECG/ABP rate-disagreement audit caveat in `docs/v7_extracted_features/extractedFeatures.md`.
4. Decide whether to extend feature-model classification beyond filtered `5m` hypotension.

# Extracted Feature Classification - Full Data

This page tracks full-data extracted-feature classification targets and submitted model jobs for the segment-aware `data_m3_120s_prediction` cohort. It documents the corrected anchor-based hypotension, tachycardia, and hypoxia target bundles. As of the 2026-09-01 handoff check, all 18 v7 filtered classification runs for history XGBoost, full-sequence XGBoost, and Transformer have completed with `metrics.json` and `test_predictions.npz`; XGBoost runs also have `model.pkl`. On 2026-09-02, the same 18-job grid was submitted for the combined v7+v8 feature cache; active jobs were canceled on 2026-09-03 for deadlines and should be resumed as soon as possible.

## Raw-Waveform PatchTST Models

The full-data 4-channel raw-waveform `patchtst_v1` classification setup and submitted SLURM jobs are documented in `docs/full_data/full_data_raw_waveform_models.md`.

## Comparability With Earlier v7 Results

The full-data classification metrics below should not be read as the same benchmark as `docs/v7_extracted_features/extractedFeaturesClassification.md`. They are production-scale follow-up experiments on a different cohort, split, and target-construction path.

Key differences from the earlier v7 extracted-feature classification table:

- Cohort and split: the earlier v7 table uses the older vasopressor-free overlap cohort with `334833` feature-cache rows from `887` patients and reports filtered `5m` hypotension on `3650` test examples. This full-data page uses the segment-aware `data_m3_120s_prediction` cohort with `1969515` feature-cache rows from `1758` high-confidence vaso-free patients and reports filtered `5m` hypotension on `100490` test examples.
- Target semantics: the earlier v7 table uses `outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz` with legacy `anchor_horizon_filtered` labels. The full-data runs use corrected anchor-onset labels such as `hypotension_onset_within_5m`, absolute timestamp reconstruction, `anchor_id`/segment-aware alignment, `strict-clean-horizon` negatives, and complete-source-recording last-onset filtering.
- Class balance: the earlier filtered `5m` hypotension test set has `194 / 3650` positives (`5.315%`). The full-data filtered `5m` hypotension test set has `1895 / 100490` positives (`1.886%`). Because AUPRC is prevalence-sensitive, AUPRC values are not directly comparable without accounting for the different base rates.
- Task difficulty: the earlier filtered hypotension benchmark is small and enriched after aggressive negative removal. The full-data benchmark is broader, more heterogeneous, and lower-prevalence, so lower AUROC/AUPRC does not by itself indicate a model regression.
- Event-feature match: hypotension and tachycardia have direct or near-direct signal in the v7 waveform features through ABP/MAP and ECG/ABP rate features. Hypoxia labels come from `SpO2 < 90`, while the extracted waveform feature cache is built from waveform channels rather than direct SpO2 numeric features, so weaker hypoxia performance is expected.

For this reason, the correct interpretation is that the earlier v7 page shows strong performance on the older filtered `5m` hypotension benchmark, while this page is a harder segment-aware full-data stress test with corrected onset labels and lower event prevalence.

One subtle but important point: filtering out anchors with active input-window hypotension does not make the old and new target definitions identical. With a `5m` horizon and `5m` sustain rule, suppose hypotension starts `3` minutes after prediction time and continues for `5` minutes. The event is a new onset inside the horizon, so the corrected full-data label is positive. The legacy `anchor_horizon` label can miss it because the full `5` sustained minutes are not contained inside the `5m` horizon window. The corrected full-data target therefore labels confirmed event starts in the horizon, while allowing confirmation to extend past the horizon boundary.

## Current Artifacts

- Feature cache: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v7/full_data_vasopressor_free_waveform_features_v7`, shape `(1969515, 20, 93)`.
- Combined v7+v8 feature cache: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/combined_v7_v8/full_data_vasopressor_free_waveform_features_v7_v8_segment_plan`, shape `(1969515, 20, 287)`.
- Aligned numerics cache: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/numerics/full_data_v1`, shape `(1969515, 5, 1200)`.
- Primary unfiltered target bundle: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m.npz`.
- Current hardened filtered target bundle: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.npz`.
- Tachycardia+hypoxia filtered target bundle: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz`.

## Combined v7+v8 Classification Submission

Submitted on `2026-09-02` after building and validating the combined feature cache.

- Cache: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/combined_v7_v8/full_data_vasopressor_free_waveform_features_v7_v8_segment_plan`.
- Cache shape: `(1969515, 20, 287)`.
- Models: `history_xgb`, `full_sequence_xgb`, `transformer`.
- Events/horizons: hypotension, tachycardia, and hypoxia at `5m` and `10m`.
- Output root: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data_v7_v8/classification`.
- Submitted jobs: `18`.
- Hypotension target bundle: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.npz`.
- Tachycardia/hypoxia target bundle: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz`.
- Initial job IDs: `26980728`, `26980730`, `26980732`, `26980734`, `26980736`, `26980738`, `26980740`, `26980742`, `26980744`, `26980746`, `26980748`, `26980750`, `26980752`, `26980754`, `26980756`, `26980758`, `26980760`, `26980762`.
- Partition move: the pending `cpu_medium` tabular classification jobs were canceled and resubmitted to `cpu_short`; Transformer classification jobs were left on `gl40s_short`.
- Active tabular classification job IDs: `26980938`, `26980940`, `26980942`, `26980944`, `26980946`, `26980948`, `26980949`, `26980950`, `26980951`, `26980952`, `26980953`, `26980954`.
- Active Transformer classification job IDs: `26980744`, `26980746`, `26980748`, `26980750`, `26980760`, `26980762`.
- Handoff status at `2026-09-02 18:20 EDT`: tabular classification jobs `26980938`, `26980940`, `26980942`, `26980944`, `26980946`, and `26980948`-`26980954` were pending on `QOSMaxMemoryPerUser`; Transformer classification jobs `26980744`, `26980746`, `26980748`, `26980750`, `26980760`, and `26980762` were pending on `gl40s_short` priority. The classification output root had no run directories or result artifacts yet.

Commands used:

```bash
INCLUDE_REGRESSION=0 \
INCLUDE_CLASSIFICATION=1 \
MODELS='history_xgb full_sequence_xgb transformer' \
CLASSIFICATION_EVENTS='hypotension' \
CACHE_DIR=/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/combined_v7_v8/full_data_vasopressor_free_waveform_features_v7_v8_segment_plan \
OUTPUT_ROOT=/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data_v7_v8 \
CLASSIFICATION_TARGETS=outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.npz \
FEATURE_CACHE_LABEL=v7_v8 \
bash slurm/submit_feature_models_full_data_v7.sh
```

For the active tabular resubmission, the same commands were rerun with `MODELS='history_xgb full_sequence_xgb'` and `TABULAR_PARTITION=cpu_short`. Transformer jobs from the initial submission were not canceled.

```bash
INCLUDE_REGRESSION=0 \
INCLUDE_CLASSIFICATION=1 \
MODELS='history_xgb full_sequence_xgb transformer' \
CLASSIFICATION_EVENTS='tachycardia hypoxia' \
CACHE_DIR=/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/combined_v7_v8/full_data_vasopressor_free_waveform_features_v7_v8_segment_plan \
OUTPUT_ROOT=/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data_v7_v8 \
CLASSIFICATION_TARGETS=outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz \
FEATURE_CACHE_LABEL=v7_v8 \
bash slurm/submit_feature_models_full_data_v7.sh
```

Implementation notes:

- `scripts/build_combined_v7_v8_feature_cache.py` writes the combined cache by streaming the aligned v7 and v8 arrays into output memmaps.
- `FeaturePreprocessor` now recognizes v8 feature definitions and allows all-missing training columns by imputing `0`, using mean `0` and std `1`, and preserving the mask indicator. This is needed because default v8 intentionally stores disabled audit-only feature families as all missing.

Before interpreting results, audit each run directory for `metrics.json`, `test_predictions.npz`, and `model.pkl` for XGBoost models.


## Combined v7+v8 Classification Cancellation, 2026-09-03

The active combined v7+v8 full-data classification jobs were canceled for now to free SLURM capacity for deadlines. This was a scheduling decision, not a result-based decision; the combined v7+v8 classification batch should be resumed as soon as possible.

- Canceled tabular classification jobs: `26980938`, `26980940`, `26980942`, `26980944`, `26980946`, and `26980948`-`26980954`.
- Canceled Transformer classification jobs: `26980744`, `26980746`, `26980748`, `26980750`, `26980760`, and `26980762`.
- No combined v7+v8 classification metrics should be interpreted until the batch is resubmitted, completed, and audited for `metrics.json`, `test_predictions.npz`, and XGBoost `model.pkl` artifacts.
- Handoff artifact audit after cancellation found `2` classification run directories under `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data_v7_v8/classification`, with no `config.json`, `metrics.json`, `test_predictions.npz`, or `model.pkl` artifacts. These directories are incomplete placeholders from the interrupted batch.

## Target Definition

Each extracted-feature row remains an anchor candidate. For horizon `H`, input ending at `t0`, and sustained duration `S=5` minutes:

```text
positive iff a confirmed event episode onset e satisfies t0 <= e < t0 + H
```

The five-minute confirmation period may extend beyond the forecast horizon. Intervals are half-open `[start, end)`. An onset exactly at `t0 + H` is excluded from that horizon. A target cell is invalid if the same event is already active in the 20-minute input window.

Event definitions currently supported by the corrected full-data builder:

- Hypotension: MAP-only, `ABP Mean <= 65`.
- Tachycardia: `HR > 110`.
- Hypoxia: `SpO2 < 90`.

Primary negatives use `observable-no-onset`: a valid negative has enough input, forecast, and confirmation coverage to rule out any confirmed onset in the horizon. Isolated abnormal minutes for the target event that do not form a sustained episode may remain negative.

Filtered negatives use `strict-clean-horizon`: a valid negative must have no event-positive minutes for the target event in the forecast horizon.

## Timestamp Basis

Root cause of the earlier very small valid-label count: the old path mixed time bases. The feature-cache anchor timestamps were treated as absolute seconds, but in this cache `anchor_times.npy` and `anchors.csv::window_time` are segment-relative centers. The aligned numerics cache stores absolute centers.

The corrected loader joins `anchor_ids.npy` and `anchor_times.npy` to `anchors.csv` by `anchor_id`, identifies the timestamp basis, and writes canonical bundle times in absolute seconds.

- Feature-cache `anchor_times.npy`: segment-relative.
- `anchors.csv::window_time`: segment-relative; absolute match fraction `0.0`, segment-relative match fraction `1.0`.
- `numerics_window_times.npy`: absolute numeric-window centers.
- Label working basis: absolute seconds.
- Canonical output basis: absolute seconds.

The builder also preserves all 150-second-stride anchors by using phase-specific minute timelines per segment instead of snapping or rejecting 30-second-offset anchors.

Timestamp preflight passed with anchor-center p95/max distance `0.0` seconds, matching-center fraction `1.0`, complete input coverage fraction `0.989056`, complete forecast coverage fraction `0.952645`, and complete confirmation coverage fraction `0.937069`.

## Primary Bundle

Command configuration:

```bash
--target-mode anchor_onset_within_horizon
--events hypotension
--numerics-source aligned-array
--numerics-window-time-basis absolute
--aligned-time-basis absolute
--event-horizons 5 10
--hypotension-definition map-only
--sustain-minutes 5
--negative-policy observable-no-onset
--negative-exclusion-scope none
--no-late-negative-cutoff
--min-valid-fraction-per-minute 0.0166666666666667
--timestamp-alignment-tolerance-seconds 1.0
```

Artifacts:

- Bundle: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m.npz`
- Audit CSV: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m.audit.csv`
- Metadata JSON: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m.json`
- Target columns: `hypotension_onset_within_5m`, `hypotension_onset_within_10m`
- Shape: `(1969515, 2)`
- Valid target cells: `3091979`
- Unique anchors with at least one valid target: `1566030`
- Time-alignment failures: `0`
- Audit-context failures: `0`
- Base, filtered, and final arrays are identical under the primary no-filter configuration.

| Target | Valid | Positive | Negative | Prevalence | Train valid/pos | Val valid/pos | Test valid/pos |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hypotension_onset_within_5m` | 1,566,030 | 12,110 | 1,553,920 | 0.773% | 1,123,287 / 8,703 | 218,884 / 1,515 | 223,859 / 1,892 |
| `hypotension_onset_within_10m` | 1,525,949 | 22,645 | 1,503,304 | 1.484% | 1,094,586 / 16,271 | 213,351 / 2,833 | 218,012 / 3,541 |

Per-target invalid reasons are anchor-target cells that could not receive a valid positive/negative label.

| Target | Insufficient input | Insufficient forecast | Insufficient confirmation | Active event in input |
|---|---:|---:|---:|---:|
| `hypotension_onset_within_5m` | 101,660 | 45,372 | 33,988 | 222,465 |
| `hypotension_onset_within_10m` | 101,660 | 87,820 | 31,621 | 222,465 |

Meaning:

- `Insufficient input`: not enough valid MAP data in the 20-minute input window.
- `Insufficient forecast`: future MAP coverage does not extend far enough through the requested horizon.
- `Insufficient confirmation`: the horizon is observable, but there is not enough additional future coverage to confirm a sustained 5-minute onset near the horizon end.
- `Active event in input`: MAP hypotension was already present during the input window.

## Current Filtered Bundle

This is the current stricter comparison dataset. It keeps the corrected onset-positive semantics and filters only base negatives. The earlier `recording_lastpos_positive_recordings_filtered` bundle used segment chunks as the recording group; `recording_lastonset_hardened_filtered` fixed the group ID but still scanned only anchor-bearing chunks. Both are superseded for future use by the complete-scan bundle below.

Command configuration. These are now the script defaults, so the flags are shown for reproducibility rather than because they must all be typed manually:

```bash
--target-mode anchor_onset_within_horizon
--events hypotension
--numerics-source aligned-array
--numerics-window-time-basis absolute
--aligned-time-basis absolute
--event-horizons 5 10
--hypotension-definition map-only
--sustain-minutes 5
--negative-policy strict-clean-horizon
--negative-exclusion-scope none
--apply-late-negative-cutoff
--late-cutoff-group-scope recording
--late-cutoff-strategy group-last-positive
--exclude-late-cutoff-groups-without-positives
--late-cutoff-candidate forecast_endpoint
--min-valid-fraction-per-minute 0.0166666666666667
--timestamp-alignment-tolerance-seconds 1.0
```

Filtering rules:

- Base negatives must satisfy `strict-clean-horizon`.
- `negative_exclusion_scope = none`, so the builder does not exclude every negative merely because its recording contains a confirmed onset.
- The recording cutoff group is the complete source recording, identified as `patient_id/source_record_name`, where `source_record_name` strips the final numeric chunk suffix from `seg_name` such as `p000188/3317157_0002 -> p000188/3317157`. The full feature cache has `21,833` segment chunks and `2,619` source recordings.
- Recording-level confirmed MAP-only sustained hypotension onsets are detected on one canonical recording-level minute timeline built from all numerics-bearing chunks for the selected source recording, not by pooling anchor-phase timelines or scanning only chunks that contain feature-cache anchors.
- Recordings with no confirmed MAP-only sustained hypotension onset contribute no negative controls.
- In recordings with confirmed onsets, base negatives are removed only when `forecast_endpoint > last_confirmed_onset_in_that_recording`. Equality is retained.
- Positives are retained; no positive labels were removed by filtering.

Artifacts:

- Bundle: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.npz`
- Audit CSV: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.audit.csv`
- Metadata JSON: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.json`
- Valid target cells: `1355868`
- Unique anchors with at least one valid target: `695134`
- Time-alignment failures: `0`
- Audit-context failures: `0`
- Positive labels removed by filters: `0`
- Five-minute-to-ten-minute horizon consistency violations: `0`
- Source recordings audited for cutoff: `2619`
- Anchor-bearing segment chunks in audited recordings: `21833`
- Numerics-bearing segment chunks scanned: `21833`
- Audited recordings with additional no-anchor numerics chunks: `0` in the current cache
- Confirmed-onset recordings whose last onset came from a no-anchor chunk: `0` in the current cache
- Source recordings with confirmed onsets used for cutoff: `1346`
- Mean last-confirmed-onset elapsed time across onset-containing recordings, for context only: `140058.366` seconds.

| Target | Valid | Positive | Negative | Prevalence | Train valid/pos | Val valid/pos | Test valid/pos |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hypotension_onset_within_5m` | 691,240 | 12,137 | 679,103 | 1.756% | 502,552 / 8,725 | 88,198 / 1,517 | 100,490 / 1,895 |
| `hypotension_onset_within_10m` | 664,628 | 22,700 | 641,928 | 3.415% | 482,915 / 16,318 | 85,111 / 2,837 | 96,602 / 3,545 |

Filtered negative-removal reasons describe what happened after base onset labels were built.

| Target | Kept negatives | Base invalid | Positive | After recording last-confirmed-onset cutoff | No confirmed-onset recording |
|---|---:|---:|---:|---:|---:|
| `hypotension_onset_within_5m` | 679,103 | 432,716 | 12,137 | 297,958 | 547,601 |
| `hypotension_onset_within_10m` | 641,928 | 485,615 | 22,700 | 287,148 | 532,124 |

Meaning:

- `Kept negatives`: valid base strict-clean negatives that survived all filtering.
- `Base invalid`: rows already invalid before negative filtering.
- `Positive`: confirmed future-onset positives retained by the filters.
- `After recording last-confirmed-onset cutoff`: base negatives whose forecast endpoint occurred after the last confirmed onset in the same source recording.
- `No confirmed-onset recording`: base negatives from recordings with no confirmed onset, so the recording was excluded from the negative-control set.

## Tachycardia and Hypoxia Filtered Bundle

This bundle was built on 2026-08-31 for the requested non-hypotension event labels only. It uses the same corrected full-data anchor-onset logic and filtered negative-selection semantics as the current hypotension filtered bundle. The existing hypotension jobs and hypotension bundle were not changed.

Command:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/build_full_data_event_targets.py \
  --events tachycardia hypoxia \
  --output outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz \
  --audit-csv outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.audit.csv \
  --target-mode anchor_onset_within_horizon \
  --event-horizons 5 10 \
  --sustain-minutes 5 \
  --negative-policy strict-clean-horizon \
  --negative-exclusion-scope none \
  --apply-late-negative-cutoff \
  --late-cutoff-group-scope recording \
  --late-cutoff-strategy group-last-positive \
  --exclude-late-cutoff-groups-without-positives \
  --late-cutoff-candidate forecast_endpoint \
  --numerics-source aligned-array \
  --numerics-window-time-basis absolute \
  --aligned-time-basis absolute \
  --progress-every 1000
```

Artifacts:

- Bundle: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz`
- Audit CSV: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.audit.csv`
- Metadata JSON: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.json`
- Target columns: `tachycardia_onset_within_5m`, `hypoxia_onset_within_5m`, `tachycardia_onset_within_10m`, `hypoxia_onset_within_10m`
- Shape: `(1969515, 4)`
- Valid target cells: `1843822`
- Unique anchors with at least one valid target: `753264`
- Time-alignment failures: `0`
- Audit-context failures: `0`; audit CSV rows: `1400`
- Positive labels removed by filters: `0`
- Five-minute-to-ten-minute horizon consistency violations: `0` for tachycardia and `0` for hypoxia
- Source recordings audited for cutoff: `2619`
- Anchor-bearing segment chunks in audited recordings: `21833`
- Numerics-bearing segment chunks scanned: `21833`
- Audited recordings with additional no-anchor numerics chunks: `0` in the current cache
- Confirmed-onset recordings whose last onset came from a no-anchor chunk: `0` in the current cache
- Source recordings with confirmed tachycardia onsets used for cutoff: `850`
- Source recordings with confirmed hypoxia onsets used for cutoff: `568`
- Mean last-confirmed-onset elapsed time, for context only: tachycardia `144765.035` seconds; hypoxia `168028.944` seconds.

Channel diagnostics:

| Event | Channel | Index | Finite sample fraction | Valid minute fraction | Valid minutes / total |
|---|---|---:|---:|---:|---:|
| Tachycardia | `HR` | 4 | 0.995811 | 0.996723 | 5,251,362 / 5,268,629 |
| Hypoxia | `SpO2` | 2 | 0.963291 | 0.973712 | 5,130,125 / 5,268,629 |

Filtered target counts after strict-clean-horizon negative selection, recording-level last-confirmed-onset cutoff, and exclusion of recordings with no confirmed onset:

| Target | Filtered valid | Filtered positive | Filtered negative | Filtered prevalence | Train filtered valid/pos | Val filtered valid/pos | Test filtered valid/pos |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tachycardia_onset_within_5m` | 496,729 | 6,320 | 490,409 | 1.272% | 358,244 / 4,416 | 64,180 / 757 | 74,305 / 1,147 |
| `hypoxia_onset_within_5m` | 440,933 | 1,993 | 438,940 | 0.452% | 324,055 / 1,484 | 53,943 / 263 | 62,935 / 246 |
| `tachycardia_onset_within_10m` | 479,911 | 11,882 | 468,029 | 2.476% | 346,363 / 8,310 | 62,051 / 1,420 | 71,497 / 2,152 |
| `hypoxia_onset_within_10m` | 426,249 | 3,751 | 422,498 | 0.880% | 312,921 / 2,792 | 52,420 / 499 | 60,908 / 460 |

Filtered negative-removal reasons:

| Target | Kept negatives | Base invalid | Positive | After recording last-confirmed-onset cutoff | No confirmed-onset recording |
|---|---:|---:|---:|---:|---:|
| `tachycardia_onset_within_5m` | 490,409 | 273,214 | 6,320 | 294,141 | 905,431 |
| `hypoxia_onset_within_5m` | 438,940 | 220,619 | 1,993 | 202,828 | 1,105,135 |
| `tachycardia_onset_within_10m` | 468,029 | 319,694 | 11,882 | 285,735 | 884,175 |
| `hypoxia_onset_within_10m` | 422,498 | 272,122 | 3,751 | 195,847 | 1,075,297 |

## Minute-Validity Sensitivity

Here, sensitivity means a data-coverage sensitivity check: the same corrected label builder was rerun with stricter `--min-valid-fraction-per-minute` thresholds to measure how many labels remain valid. It does not refer to model sensitivity/recall, and it was not used to choose a threshold from downstream model performance.

Threshold meanings:

- `1/60`: at least one finite 1 Hz numeric sample in a minute.
- `0.50`: at least 30 finite samples in a minute.
- `0.80`: at least 48 finite samples in a minute.

Higher thresholds require cleaner minute-level numeric coverage. Among labels that remained jointly valid, the `0.50` and `0.80` builds agreed 100% with the `1/60` reference labels. The MAP finite-samples-per-minute distribution from the aligned numerics cache is saved at `outputs/targets/full_data_map_finite_samples_per_minute_distribution.json`: `98.146%` of anchor-minutes have at least one finite sample, `97.236%` have at least `30/60`, `96.140%` have at least `48/60`, and `93.800%` have all `60/60`. A defensible stricter default for future clinically cleaner builds is `0.80`; keep `1/60` only for backward-compatible exploratory comparisons.

| Min valid fraction/min | Bundle | Valid cells | Unique valid anchors | 5m pos | 10m pos | Episode count | 5m/10m violations | Agreement with 1/60 on jointly valid labels |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1/60 | `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m.npz` | 3,091,979 | 1,566,030 | 12,110 | 22,645 | 37,594 | 0 | reference |
| 0.50 | `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_minvalid_0p50.npz` | 2,838,855 | 1,445,626 | 11,035 | 20,425 | 38,104 | 0 | 100.0% for both targets |
| 0.80 | `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_minvalid_0p80.npz` | 2,559,020 | 1,310,463 | 9,849 | 17,972 | 38,952 | 0 | 100.0% for both targets |


## Filtered Model Jobs

Submitted on 2026-08-31 using `slurm/submit_feature_models_full_data_v7.sh` with:

```bash
INCLUDE_REGRESSION=0 \
INCLUDE_CLASSIFICATION=1 \
MODELS="history_xgb full_sequence_xgb transformer" \
bash slurm/submit_feature_models_full_data_v7.sh
```

All jobs use the complete-scan filtered target bundle `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.npz`. Follow-up on `2026-09-01` showed the first submission batch failed quickly with exit code `1:0` and produced no result artifacts because the training code requested legacy names such as `hypotension_within_5m` while the corrected target bundle stores onset names such as `hypotension_onset_within_5m`. `scripts/train_patchtst.py::TargetExtractor._event_col_index` now accepts both naming conventions, and the failed jobs were resubmitted as `26949866`-`26949871`.

Latest handoff check on `2026-09-01` found all six hypotension result directories complete. `squeue -u dk5565` no longer listed the resubmitted extracted-feature classification jobs.

| Model | Horizon | Failed SLURM job | Resubmitted SLURM job | Current state at handoff | Output directory |
|---|---:|---:|---:|---|---|
| `history_xgb` | 5m | `26944638` | `26949866` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/history_xgb_hypotension_within_5m_filtered_v7` |
| `history_xgb` | 10m | `26944639` | `26949867` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/history_xgb_hypotension_within_10m_filtered_v7` |
| `full_sequence_xgb` | 5m | `26944640` | `26949868` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/full_sequence_xgb_hypotension_within_5m_filtered_v7` |
| `full_sequence_xgb` | 10m | `26944641` | `26949869` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/full_sequence_xgb_hypotension_within_10m_filtered_v7` |
| `transformer` | 5m | `26944642` | `26949870` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/transformer_hypotension_within_5m_filtered_v7` |
| `transformer` | 10m | `26944643` | `26949871` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/transformer_hypotension_within_10m_filtered_v7` |

Completed filtered hypotension results inspected from `metrics.json`:

| Model | Horizon | SLURM job | AUROC | AUPRC | Specificity at 85% sensitivity | Test rows | Positives | Test prevalence | Test patients |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `history_xgb` | 5m | `26949866` | 0.9148 | 0.2421 | 0.8294 | 100,490 | 1,895 | 1.886% | 156 |
| `history_xgb` | 10m | `26949867` | 0.8933 | 0.3350 | 0.7814 | 96,602 | 3,545 | 3.670% | 156 |
| `full_sequence_xgb` | 5m | `26949868` | 0.9109 | 0.2416 | 0.8243 | 100,490 | 1,895 | 1.886% | 156 |
| `full_sequence_xgb` | 10m | `26949869` | 0.8933 | 0.3441 | 0.7671 | 96,602 | 3,545 | 3.670% | 156 |
| `transformer` | 5m | `26949870` | 0.9108 | 0.2156 | 0.8302 | 100,490 | 1,895 | 1.886% | 156 |
| `transformer` | 10m | `26949871` | 0.8929 | 0.3243 | 0.7781 | 96,602 | 3,545 | 3.670% | 156 |

## Tachycardia and Hypoxia Filtered Model Jobs

Submitted on 2026-08-31 using the same full-data extracted-feature training path as the filtered hypotension jobs:

```bash
INCLUDE_REGRESSION=0 \
INCLUDE_CLASSIFICATION=1 \
MODELS="history_xgb full_sequence_xgb transformer" \
CLASSIFICATION_EVENTS="tachycardia hypoxia" \
CLASSIFICATION_TARGETS="outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz" \
bash slurm/submit_feature_models_full_data_v7.sh
```

The first tachycardia/hypoxia submission batch also failed with the same target-name lookup issue and was resubmitted on `2026-09-01` as `26949872`-`26949883`. Latest handoff check found all tachycardia and hypoxia result directories complete. `squeue -u dk5565` no longer listed the resubmitted extracted-feature classification jobs.

| Model | Event | Horizon | Failed SLURM job | Resubmitted SLURM job | Current state at handoff | Output directory |
|---|---|---:|---:|---:|---|---|
| `history_xgb` | tachycardia | 5m | `26945429` | `26949872` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/history_xgb_tachycardia_within_5m_filtered_v7` |
| `history_xgb` | tachycardia | 10m | `26945430` | `26949873` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/history_xgb_tachycardia_within_10m_filtered_v7` |
| `history_xgb` | hypoxia | 5m | `26945431` | `26949874` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/history_xgb_hypoxia_within_5m_filtered_v7` |
| `history_xgb` | hypoxia | 10m | `26945432` | `26949875` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/history_xgb_hypoxia_within_10m_filtered_v7` |
| `full_sequence_xgb` | tachycardia | 5m | `26945433` | `26949876` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/full_sequence_xgb_tachycardia_within_5m_filtered_v7` |
| `full_sequence_xgb` | tachycardia | 10m | `26945434` | `26949877` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/full_sequence_xgb_tachycardia_within_10m_filtered_v7` |
| `full_sequence_xgb` | hypoxia | 5m | `26945435` | `26949878` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/full_sequence_xgb_hypoxia_within_5m_filtered_v7` |
| `full_sequence_xgb` | hypoxia | 10m | `26945436` | `26949879` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/full_sequence_xgb_hypoxia_within_10m_filtered_v7` |
| `transformer` | tachycardia | 5m | `26945437` | `26949880` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/transformer_tachycardia_within_5m_filtered_v7` |
| `transformer` | tachycardia | 10m | `26945438` | `26949881` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/transformer_tachycardia_within_10m_filtered_v7` |
| `transformer` | hypoxia | 5m | `26945439` | `26949882` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/transformer_hypoxia_within_5m_filtered_v7` |
| `transformer` | hypoxia | 10m | `26945440` | `26949883` | completed | `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/classification/transformer_hypoxia_within_10m_filtered_v7` |

Completed filtered tachycardia/hypoxia results inspected from `metrics.json`:

| Model | Event | Horizon | SLURM job | AUROC | AUPRC | Specificity at 85% sensitivity | Test rows | Positives | Test prevalence | Test patients |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `history_xgb` | tachycardia | 5m | `26949872` | 0.9067 | 0.2113 | 0.8168 | 74,305 | 1,147 | 1.544% | 92 |
| `history_xgb` | tachycardia | 10m | `26949873` | 0.8926 | 0.3278 | 0.7960 | 71,497 | 2,152 | 3.010% | 90 |
| `history_xgb` | hypoxia | 5m | `26949874` | 0.6633 | 0.0090 | 0.3248 | 62,935 | 246 | 0.391% | 67 |
| `history_xgb` | hypoxia | 10m | `26949875` | 0.6534 | 0.0145 | 0.3337 | 60,908 | 460 | 0.755% | 66 |
| `full_sequence_xgb` | tachycardia | 5m | `26949876` | 0.9007 | 0.2137 | 0.8012 | 74,305 | 1,147 | 1.544% | 92 |
| `full_sequence_xgb` | tachycardia | 10m | `26949877` | 0.8918 | 0.3546 | 0.7798 | 71,497 | 2,152 | 3.010% | 90 |
| `full_sequence_xgb` | hypoxia | 5m | `26949878` | 0.6643 | 0.0074 | 0.3545 | 62,935 | 246 | 0.391% | 67 |
| `full_sequence_xgb` | hypoxia | 10m | `26949879` | 0.6485 | 0.0204 | 0.3171 | 60,908 | 460 | 0.755% | 66 |
| `transformer` | tachycardia | 5m | `26949880` | 0.9017 | 0.1815 | 0.7955 | 74,305 | 1,147 | 1.544% | 92 |
| `transformer` | tachycardia | 10m | `26949881` | 0.8972 | 0.3146 | 0.8049 | 71,497 | 2,152 | 3.010% | 90 |
| `transformer` | hypoxia | 5m | `26949882` | 0.6156 | 0.0060 | 0.2591 | 62,935 | 246 | 0.391% | 67 |
| `transformer` | hypoxia | 10m | `26949883` | 0.6372 | 0.0134 | 0.3513 | 60,908 | 460 | 0.755% | 66 |

Expected result files per completed job are `metrics.json`, `test_predictions.npz`, and for XGBoost jobs `model.pkl`. The 2026-09-01 artifact audit found all expected filtered classification artifacts: `18/18` `metrics.json`, `18/18` `test_predictions.npz`, and `12/12` XGBoost `model.pkl` files.

## Verification

- Syntax check passed for `scripts/build_full_data_event_targets.py`, `scripts/extract_full_data_numerics.py`, `scripts/merge_full_data_numerics.py`, `waveform_baselines/anchor_labeling.py`, `waveform_baselines/event_episodes.py`, `waveform_baselines/event_timeline.py`, and `tests/test_corrected_event_labeling.py`.
- Syntax check passed for the target-name compatibility patch: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m py_compile scripts/train_patchtst.py tests/test_waveform_feature_pipeline.py`.
- Focused onset-name compatibility test passed: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_waveform_feature_pipeline.FeaturePipelineTests.test_target_extractor_accepts_onset_event_target_names`.
- Full-data target extractor smoke check passed for `hypotension` 5m/10m, `tachycardia` 5m, and `hypoxia` 10m onset targets, with valid counts matching the documented bundles.
- `slurm/submit_feature_models_full_data_v7.sh` syntax check passed after adding `CLASSIFICATION_EVENTS`; the default remains `hypotension`, so existing hypotension submissions are preserved unless `CLASSIFICATION_EVENTS` is explicitly set.
- Focused target tests passed after adding tachycardia+hypoxia support: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_corrected_event_labeling tests.test_target_generation` (`38` tests discovered in this environment).
- Timestamp preflight passed with `0` time-alignment failures.
- Debug build passed on `50` segments and wrote `outputs/targets/debug_event_targets_full_data_anchor_onset_v2_5m_10m.npz` plus `outputs/targets/debug_event_targets_full_data_anchor_onset_v2_5m_10m.audit.csv`.
- Primary full build and hardened filtered full build both passed post-build checks: target names are exact, canonical bundle times are absolute, positives are preserved by filters, final negatives are valid base strict-clean negatives, and valid 5-minute positives are 10-minute positives whenever both labels are valid.
- Complete-recording cutoff smoke build passed on `5` segments and wrote `outputs/targets/debug_event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_smoke.npz`; requested-event diagnostics report only hypotension for `--events hypotension`.
- Tachycardia+hypoxia full-data filtered build passed post-build checks: target names are exact, canonical bundle times are absolute, positives are preserved by filters, final negatives are valid base strict-clean negatives, and valid 5-minute positives are 10-minute positives whenever both labels are valid for each event.

## Remaining Limitations

- Current label construction requires 1 Hz aligned numerics.
- Existing merged numerics metadata did not contain `sampling_rate_hz`; this run records that `1.0 Hz` was inferred from the legacy 1200-sample window length. New merged numerics metadata now writes this explicitly.
- The primary and minute-validity sensitivity bundles predate the complete-source-recording cutoff. They remain valid for unfiltered target semantics; filtered model runs should use the complete-scan filtered bundle above.
- Episode counts in the sensitivity table are counted on the phase-specific minute grids used to preserve all 150-second-stride anchors.
- Full-data classification training submissions from `2026-08-31` failed before artifact creation because requested event target names were not found in the target bundles; target-name lookup has been patched and the failed jobs were resubmitted as `26949866`-`26949883`.

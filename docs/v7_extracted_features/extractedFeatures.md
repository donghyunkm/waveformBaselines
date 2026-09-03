# Extracted Waveform Features

This document is the authoritative description of the current waveform-feature pipeline for the vasopressor-free cohort. Historical versions are summarized only in Version History.

## Current Version

- current feature version: `v7`
- per-sample physiological cache shape: `(20, 93)`
- model-ready sequence shape after preprocessing: `(20, 186)` because validity masks are concatenated to imputed/normalized feature values
- cache root: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/v7`
- production run name: `vasopressor_free_waveform_features_v7`
- v7 extraction job: SLURM array `26873594` (`0-31`), completed
- v7 merge job: SLURM `26873626`, completed
- merged cache shape: `(334833, 20, 93)`
- dependent v7 jobs: tabular/persistence `26873627`-`26873631`; GRU/Transformer `26873632`-`26873635`, failed during target alignment before usable metrics/checkpoints

The previous v5 extraction and dependent jobs were cancelled after this audit reproduced confirmed ECG over-detection and found detector/config consistency issues: `26872137`-`26872146`. The serial v6 extraction/dependent jobs `26873175`, `26873177`-`26873185` were cancelled before completion after the final ECG XQRS-input/provenance correction required v7. Earlier v3/v4 jobs are listed under Version History.

## Representation

Each sample is a calibrated 20-minute ICU waveform segment with channel order:

- `II`: ECG lead II
- `ABP`: arterial blood pressure
- `PLETH`: photoplethysmography
- `RESP`: respiratory waveform

The extractor returns 20 non-overlapping one-minute feature tokens. Ten-second micro-windows are used for SQI and validity bookkeeping. Beat/cycle morphology and HRV are computed over the full valid one-minute token where that is methodologically appropriate.

## Input/Target Timing

The waveform feature input is the fixed 20-minute interval centered on `anchor_time`:

```text
input_start = anchor_time - 600 s
input_end   = anchor_time + 600 s
loaded waveform interval = [input_start, input_end)
```

Gap-mode regression and event-classification targets start at `input_end`. The feature extractor receives only the loaded input slice. Tests assert the expected load boundaries, classification and regression target starts, and that perturbing samples after the input endpoint does not change extracted features.

## Missing-Data Handling

Filtering is gap-aware:

- only NaN gaps up to `max_interpolated_gap_seconds = 0.2` are linearly interpolated for filtering
- longer missing gaps remain unavailable
- contiguous finite/usable runs are filtered separately
- runs with length `<= scipy.signal.filtfilt` pad requirement are left as NaN
- unsupported short runs are not passed through unfiltered

For the installed SciPy behavior, the pad requirement is computed from the actual filter coefficients:

```python
padlen = 3 * max(len(a), len(b))
```

`filtfilt` is called with that explicit `padlen`, and only finite runs with `segment.size > padlen` are filtered.

## Filtering

All filtering is performed inside the fixed input interval.

- ECG XQRS input: raw calibrated ECG with only short-gap interpolation up to `0.2 s`; long gaps remain unavailable
- ECG energy-fallback/diagnostic detector signal: `5-20 Hz` bandpass
- ECG morphology signal: `0.5-40 Hz` bandpass
- ABP detector signal: `0.5-12 Hz` bandpass
- ABP morphology signal: low-pass with `20 Hz` ceiling
- PLETH detector signal: `0.5-8 Hz` bandpass
- PLETH morphology signal: low-pass with `8 Hz` ceiling
- RESP detector signal: configurable `0.05-1.5 Hz` bandpass

Detector-specific timing signals are used only for timing. Calibrated/raw measurement signals are used for absolute amplitude and pressure values where applicable.

## ECG Features

ECG R peaks are detected separately within each contiguous finite XQRS-input run. Each detected peak stores its finite-run identity.

Detection and RR validity:

- configured primary detector: `ecg_detector = "xqrs"` by default; accepted values are `xqrs` and `energy`
- XQRS input: raw calibrated ECG after gap-safe interpolation of short gaps only; the `5-20 Hz` ECG signal is not fed to XQRS
- if `ecg_detector = "xqrs"` and WFDB is unavailable, extraction fails clearly rather than silently changing algorithms
- fallback detector: polarity-invariant QRS slope-energy detector with a `0.25 s` refractory period
- per-run fallback is controlled by `ecg_allow_energy_fallback = True`; failed/empty XQRS runs may use the energy detector, while `False` leaves those runs undetected
- RR plausibility: `[60/220, 60/30] s`, equivalent to `30-220 bpm`
- a valid RR interval must be physiologically plausible and connect two peaks from the same finite detector run
- RR intervals never cross long unavailable gaps

Morphology mapping:

- each detector R peak is mapped to the strongest absolute local extremum in the morphology signal within `+/-0.08 s`
- the morphology search requires sufficient finite morphology samples
- raw R-amplitude lookup requires the mapped raw ECG sample to be finite
- QRS widths are computed only within continuous finite morphology runs
- beat-template consistency uses finite morphology segments from `[-0.12, +0.18] s` around mapped peaks

HRV rules:

- SDNN uses valid RR intervals and requires `ecg_hrv_min_beats = 8`
- RMSSD, SDSD, pNN20, and pNN50 use only adjacent valid RR-interval pairs in the original RR sequence
- successive-difference metrics require `ecg_hrv_min_successive_pairs = 3`
- invalid RR intervals are not removed in a way that creates artificial adjacency

ECG polarity note: v7 uses configured finite-run ECG detection and maps each detector event to the strongest absolute local morphology extremum, so upright and inverted QRS complexes are handled consistently for timing, amplitude, QRS width, and morphology segments. QRS width uses polarity-invariant morphology amplitude.

## ABP Features

ABP timing and measurement are separated:

- detector ABP locates pulse peaks and inter-peak troughs
- one detector trough is selected between each pair of adjacent detector peaks
- detector troughs are refined once onto nearby calibrated/raw local minima
- refined troughs are shared boundaries, so adjacent beats have exact common endpoints
- candidate beats are constructed refined-trough to refined-trough
- detector peaks are refined once onto nearby calibrated/raw local maxima
- calibrated/raw ABP is used for SBP, DBP, MAP, pulse pressure, and pulse area
- full-beat global raw maxima/minima are not used for SBP/DBP

ABP validity:

- plausible SBP: `[50, 260] mmHg`
- plausible DBP: `[20, 180] mmHg`
- pulse pressure must be positive
- beat duration must imply pulse rate in `[30, 220] bpm`
- `abp_valid_pulse_fraction` denominator is all measurable candidate ABP beats

MAP formula:

```text
MAP_beat = mean(raw calibrated ABP samples over the refined trough-to-trough beat)
abp_map_median_mmhg = median(MAP_beat over valid ABP beats)
```

The `(SBP + 2*DBP)/3` approximation is not used for the cached MAP feature.

ABP morphology:

- pulse area is trapezoidal area of raw ABP above the refined foot over the refined beat
- upstroke duration is refined foot to refined peak
- decay duration is refined peak to next refined foot
- upstroke slope is raw amplitude divided by refined upstroke duration
- dP/dt uses the morphology-preserving ABP signal over the refined beat
- morphology consistency uses only physiologically valid ABP beats

## PLETH Features

PLETH uses the same detector-guided trough-to-trough logic as ABP:

- detector PLETH locates pulse peaks and inter-peak troughs
- detector troughs are refined once onto nearby raw/native local minima
- refined troughs are shared boundaries across adjacent pulses
- detector peaks are refined once onto nearby raw/native local maxima
- full-beat global raw extrema are not used
- valid pulse requires positive amplitude, finite rise/decay times, and width in `[0.3, 2.0] s`
- morphology consistency uses only valid PLETH pulses

PLETH amplitude, slope, area, and delta features are native/device-scaled values. They are not calibrated physiological units like mmHg and may vary with monitor gain, sensor placement, hardware, and recording configuration. They are retained in v7 but must be audited across the full cohort for extreme between-recording scale effects.

## RESP Features

RESP peaks and troughs are detected separately within each contiguous finite filtered run. Cycles are never constructed across long missing gaps.

Detection and cycle rules:

- same-type peaks and troughs use minimum spacing derived from `resp_min_cycle_s`; the current configuration uses `0.75 s`
- prominence: `max(0.2 * robust_scale(filtered), 1e-4)`
- candidate cycle: finite trough-to-peak-to-trough candidate within the same finite detector run
- accepted cycle: candidate with duration in `[0.75, 15.0] s`
- `resp_valid_cycle_fraction = accepted_cycles / candidate_cycles`
- no candidates gives `NaN`; candidates but no accepted cycles gives `0.0`

RESP polarity is not assumed to map reliably to inspiration/expiration. Directional names are therefore mechanical:

- `resp_rise_time_median_s`
- `resp_fall_time_median_s`
- `resp_rise_fall_ratio_median`
- `resp_rise_slope_median`
- `resp_fall_slope_median`

RESP does not expose `extreme_value_fraction` in v7. The min/max dwell statistic has weaker interpretation for slow respiratory cycles and was not added solely for symmetry.

## Cross-Signal Features

v7 includes only coarse rate-agreement features that do not depend on millisecond channel synchronization:

- `cross_ecg_abp_rate_diff_bpm = ecg_hr_bpm - abp_pulse_rate_bpm`
- `cross_ecg_pleth_rate_diff_bpm = ecg_hr_bpm - pleth_pulse_rate_bpm`
- `cross_ecg_abp_rate_agreement = 1 / (1 + abs(diff) / 10)`
- `cross_ecg_pleth_rate_agreement = 1 / (1 + abs(diff) / 10)`

PAT/PTT-style timing features remain out of scope until channel synchronization is explicitly validated.

## Delta Features

First differences are causal:

```text
delta_f[t] = f[t] - f[t - 1]
```

The first token has missing delta values. A delta is present only when both current and previous base features are valid.

## SQI Definitions

Micro-window denominator is always the six 10-second regions in the one-minute token.

- missing micro-window: finite-sample fraction below `quality_min_finite_fraction = 1.0`
- flatline micro-window: sufficiently observed and finite-sample SD below channel threshold
- valid micro-window: sufficiently observed and not flatline
- missing and flatline are mutually exclusive
- all-missing micro-windows count as missing, not flatline

Flatline thresholds:

- ECG: `1e-3`
- ABP: `0.5 mmHg`
- PLETH: `1e-4 native units`
- RESP: `1e-4 native units`

Extreme-value fraction:

- available for ECG, ABP, and PLETH
- computed over finite samples only
- fraction of finite samples within `extreme_value_atol_fraction = 0.01` of the token's finite observed minimum or maximum
- returns `NaN` when no finite samples exist
- this is not hardware saturation detection

Beat/cycle validity denominators:

- ECG plausible beat fraction: all detected same-run RR intervals
- ABP valid pulse fraction: all measurable candidate ABP beats
- PLETH valid pulse fraction: all measurable candidate PLETH beats
- RESP valid cycle fraction: finite trough-to-peak-to-trough candidate cycles
- no candidates gives `NaN`

## Morphology Consistency

Morphology consistency uses the same template algorithm for ECG, ABP, PLETH, and RESP after channel-specific segmentation:

- collect finite beat/cycle segments from the channel's valid morphology population
- require at least 3 segments
- resample each segment linearly to 64 points
- create a leave-one-out pointwise median template for each segment
- compute Pearson correlation between the segment and its leave-one-out template
- aggregate with the median correlation
- return `NaN` when insufficient segments or degenerate variance prevents correlation

No explicit amplitude normalization is applied before correlation beyond Pearson correlation's centering/scaling.

## Normalization Policy

`FeaturePreprocessor` concatenates the raw validity mask as model inputs and does not z-normalize binary/quality-style channels: validity fractions, missingness fractions, flatline fractions, extreme-value fractions, plausible/valid pulse fractions, bounded RR-derived pNN20/pNN50 fractions, morphology correlations, and coarse agreement scores. Ordinary continuous physiological and morphology variables are train-set z-normalized, including timing ratios such as `resp_rise_fall_ratio_median` and `abp_upstroke_decay_ratio_median`.

## Feature Cache

The v7 physiological cache contains:

- `values.npy`: physiological features, shape `(N, 20, 93)`
- `mask.npy`: validity mask, same shape
- `patient_ids.npy`
- `anchor_times.npy`
- `anchor_ids.npy`
- `split_labels.npy`
- `metadata.json` with feature names, units, descriptions, channel order, timing config, interpolation config, and split/target-bundle provenance
- `feature_quality_report.json`

The cache is task-independent and does not store `targets.npy` or `target_mask.npy`. The optional `target_bundle_path` is metadata-only provenance. Downstream classification and regression code loads the appropriate existing target bundle and performs an explicit lookup by `(patient_id, anchor_time)`; duplicate target keys are rejected.

Physiological feature values remain in interpretable units where possible. Model normalization is not written back into `values.npy`.

## v7 Sharded Extraction

The production v7 cache is built by a 32-task SLURM array followed by a validated merge. This is an operational/data-integrity change only; `FEATURE_VERSION` remains `v7`.

Sharding invariants:

- the full aligned anchor table is filtered to split-file patients, then canonically sorted by globally unique `anchor_id` before sharding
- global `(patient_id, anchor_time)` and `anchor_id` uniqueness are checked before sharding so cross-shard duplicates cannot evade validation
- shard assignment is deterministic positional stride over the canonical table: shard `i` receives rows `i::shard_count`
- the current wrapper uses `#SBATCH --array=0-31`, passes `SLURM_ARRAY_TASK_ID` as `shard_index`, and writes zero-padded unique shard outputs such as `vasopressor_free_waveform_features_v7_shard_000`
- each shard logs `shard_index`, `shard_count`, `output_name`, and `cache_dir` before extraction and writes shard metadata plus `_SUCCESS` after its files are complete

Merge invariants:

- merge requires exactly shard indices `0..31`, loaded from zero-padded shard directories, with no missing, duplicate, or unexpected shard index
- all shards must have identical feature version, feature order, units/descriptions, sampling rate, channel order, window count, complete `extraction_config`, split provenance, and target-bundle provenance
- every shard array must have matching `values`/`mask` shapes and metadata arrays of length `N`
- after concatenation, merged `(patient_id, anchor_time)` and `anchor_id` keys are checked globally for uniqueness
- merged anchor IDs and `(patient_id, anchor_time)` keys must exactly match the expected canonical full-cohort anchor set
- final arrays are sorted back to canonical `anchor_id` order regardless of shard completion or filesystem order
- merged split sample counts and deduplicated split patient counts are recomputed from merged arrays
- ECG detector diagnostics are summed by key across shards with nonnegative/count consistency checks
- `feature_quality_report.json` is recomputed from the full merged arrays, not copied or averaged from shard reports
- merged metadata sets `is_merged_cache = true`, `source_shard_count = 32`, `shard_index = null`, and `shard_count = null`
- merge writes to a temporary directory, validates it, writes `_SUCCESS`, then renames it into the production cache path so downstream jobs cannot consume a partial merged cache
- downstream feature-model training calls `load_feature_cache(..., require_success=True)` and will not train without the completion marker

## Model Preprocessing

`FeaturePreprocessor` fits training-split-only parameters:

- imputation value: train median of observed valid values per feature
- normalization mean: train mean of observed valid values per feature
- normalization standard deviation: train standard deviation of observed valid values per feature
- transform step: missing values are imputed with the training median, then z-normalized using the observed-data training mean/std
- validation/test/inference reuse the training-fitted parameters
- masks are concatenated as model inputs

`HistorySummaryBuilder` creates tabular summaries over the 20-token history:

- mean, median, SD, min, max, first, last, last-minus-first, slope, valid fraction
- slope is in feature units per minute
- slope regression uses actual valid minute indices and does not compress missing time gaps

## Baselines / Sequence Models

Implemented downstream consumers use the same v7 feature cache:

- persistence baseline for BP regression
- current-state baseline using final token only
- XGBoost history-summary baseline
- GRU sequence model
- Transformer sequence model

The Transformer positional capacity is configurable through `SequenceModelConfig.max_seq_len`. It validates that positional capacity is at least the input sequence length plus the CLS token when used; tests cover 20, 40, and 120 token inputs.

## Ordered Feature List

1. `ecg_hr_bpm` (`bpm`) - Median heart rate from valid same-run ECG RR intervals.
2. `ecg_rr_median_s` (`s`) - Median valid same-run RR interval.
3. `ecg_rr_iqr_s` (`s`) - Valid same-run RR interquartile range.
4. `ecg_rr_min_s` (`s`) - Minimum valid same-run RR interval.
5. `ecg_rr_max_s` (`s`) - Maximum valid same-run RR interval.
6. `ecg_hrv_sdnn_s` (`s`) - SDNN over valid same-run RR intervals.
7. `ecg_hrv_rmssd_s` (`s`) - RMSSD over valid adjacent same-run RR pairs.
8. `ecg_hrv_sdsd_s` (`s`) - SDSD over valid adjacent same-run RR pairs.
9. `ecg_hrv_pnn20` (`fraction`) - Fraction of valid adjacent RR differences greater than 20 ms.
10. `ecg_hrv_pnn50` (`fraction`) - Fraction of valid adjacent RR differences greater than 50 ms.
11. `ecg_r_amp_median` (`mV_or_native`) - Median calibrated R-peak amplitude.
12. `ecg_r_amp_iqr` (`mV_or_native`) - IQR of R-peak amplitude.
13. `ecg_qrs_width_median_s` (`s`) - Approximate QRS width from morphology signal.
14. `ecg_qrs_width_iqr_s` (`s`) - QRS width IQR.
15. `ecg_max_abs_slope` (`unit_per_s`) - Maximum absolute ECG slope within finite morphology runs.
16. `ecg_morphology_consistency` (`corr`) - Median beat-template correlation.
17. `ecg_valid_micro_fraction` (`fraction`) - Fraction of valid ECG micro-windows.
18. `ecg_missing_micro_fraction` (`fraction`) - Fraction of ECG micro-windows with insufficient finite coverage.
19. `ecg_plausible_beat_fraction` (`fraction`) - Fraction of same-run RR intervals in physiologic range.
20. `ecg_flatline_fraction` (`fraction`) - Fraction of flat ECG micro-windows.
21. `ecg_extreme_value_fraction` (`fraction`) - Fraction of finite ECG samples near the token min/max.
22. `abp_sbp_median_mmhg` (`mmHg`) - Median refined raw systolic blood pressure.
23. `abp_dbp_median_mmhg` (`mmHg`) - Median refined raw diastolic blood pressure.
24. `abp_map_median_mmhg` (`mmHg`) - Median direct beat-mean arterial pressure.
25. `abp_pulse_pressure_median_mmhg` (`mmHg`) - Median pulse pressure.
26. `abp_pulse_rate_bpm` (`bpm`) - Pulse rate from valid ABP beats.
27. `abp_sbp_sd_mmhg` (`mmHg`) - SBP standard deviation.
28. `abp_sbp_iqr_mmhg` (`mmHg`) - SBP IQR.
29. `abp_dbp_sd_mmhg` (`mmHg`) - DBP standard deviation.
30. `abp_dbp_iqr_mmhg` (`mmHg`) - DBP IQR.
31. `abp_map_sd_mmhg` (`mmHg`) - MAP standard deviation.
32. `abp_map_iqr_mmhg` (`mmHg`) - MAP IQR.
33. `abp_pp_sd_mmhg` (`mmHg`) - Pulse pressure standard deviation.
34. `abp_pp_iqr_mmhg` (`mmHg`) - Pulse pressure IQR.
35. `abp_upstroke_slope_median` (`mmHg_per_s`) - Median refined raw ABP upstroke slope.
36. `abp_dpdt_max_median` (`mmHg_per_s`) - Median maximum positive dP/dt from morphology-preserving ABP.
37. `abp_dpdt_min_median` (`mmHg_per_s`) - Median maximum negative dP/dt from morphology-preserving ABP.
38. `abp_pulse_area_median` (`mmHg_s`) - Median raw ABP beat area above refined foot.
39. `abp_pulse_width_median_s` (`s`) - Median refined trough-to-trough pulse width.
40. `abp_upstroke_duration_median_s` (`s`) - Median refined foot-to-peak duration.
41. `abp_decay_duration_median_s` (`s`) - Median refined peak-to-next-foot duration.
42. `abp_upstroke_decay_ratio_median` (`ratio`) - Median refined upstroke/decay timing ratio.
43. `abp_morphology_consistency` (`corr`) - Median valid-beat template correlation.
44. `abp_valid_pulse_fraction` (`fraction`) - Fraction of valid ABP pulses.
45. `abp_plausible_sbp_fraction` (`fraction`) - Fraction of candidate pulses with plausible SBP.
46. `abp_plausible_dbp_fraction` (`fraction`) - Fraction of candidate pulses with plausible DBP.
47. `abp_sbp_gt_dbp_fraction` (`fraction`) - Fraction of candidate pulses with SBP greater than DBP.
48. `abp_valid_micro_fraction` (`fraction`) - Fraction of valid ABP micro-windows.
49. `abp_missing_micro_fraction` (`fraction`) - Fraction of ABP micro-windows with insufficient finite coverage.
50. `abp_flatline_fraction` (`fraction`) - Fraction of flat ABP micro-windows.
51. `abp_extreme_value_fraction` (`fraction`) - Fraction of finite ABP samples near the token min/max.
52. `pleth_pulse_rate_bpm` (`bpm`) - Pulse rate from valid PLETH pulses.
53. `pleth_amplitude_median` (`native`) - Median pulse amplitude.
54. `pleth_amplitude_iqr` (`native`) - Pulse amplitude IQR.
55. `pleth_rise_time_median_s` (`s`) - Median refined foot-to-peak rise time.
56. `pleth_decay_time_median_s` (`s`) - Median refined peak-to-next-foot decay time.
57. `pleth_rise_slope_median` (`unit_per_s`) - Median rise slope.
58. `pleth_decay_slope_median` (`unit_per_s`) - Median decay slope.
59. `pleth_width_median_s` (`s`) - Median refined trough-to-trough pulse width.
60. `pleth_area_median` (`unit_s`) - Median pulse area above refined foot.
61. `pleth_morphology_consistency` (`corr`) - Median valid-pulse template correlation.
62. `pleth_valid_pulse_fraction` (`fraction`) - Fraction of valid PLETH pulses.
63. `pleth_valid_micro_fraction` (`fraction`) - Fraction of valid PLETH micro-windows.
64. `pleth_missing_micro_fraction` (`fraction`) - Fraction of PLETH micro-windows with insufficient finite coverage.
65. `pleth_flatline_fraction` (`fraction`) - Fraction of flat PLETH micro-windows.
66. `pleth_extreme_value_fraction` (`fraction`) - Fraction of finite PLETH samples near the token min/max.
67. `resp_rate_bpm` (`breaths_per_min`) - Respiratory rate from accepted trough-to-peak-to-trough cycles.
68. `resp_cycle_length_median_s` (`s`) - Median accepted respiratory cycle length.
69. `resp_cycle_length_iqr_s` (`s`) - Accepted respiratory cycle-length IQR.
70. `resp_amplitude_median` (`native`) - Median respiratory amplitude.
71. `resp_amplitude_iqr` (`native`) - Respiratory amplitude IQR.
72. `resp_rise_time_median_s` (`s`) - Median trough-to-peak rise time for accepted RESP cycles.
73. `resp_fall_time_median_s` (`s`) - Median peak-to-trough fall time for accepted RESP cycles.
74. `resp_rise_fall_ratio_median` (`ratio`) - Median RESP rise/fall timing ratio.
75. `resp_rise_slope_median` (`unit_per_s`) - Median RESP rise slope.
76. `resp_fall_slope_median` (`unit_per_s`) - Median RESP fall slope.
77. `resp_cycle_area_median` (`unit_s`) - Median cycle area.
78. `resp_morphology_consistency` (`corr`) - Median accepted-cycle template correlation.
79. `resp_valid_cycle_fraction` (`fraction`) - Accepted trough-start cycles divided by candidate trough-start cycles.
80. `resp_valid_micro_fraction` (`fraction`) - Fraction of valid RESP micro-windows.
81. `resp_missing_micro_fraction` (`fraction`) - Fraction of RESP micro-windows with insufficient finite coverage.
82. `resp_flatline_fraction` (`fraction`) - Fraction of flat RESP micro-windows.
83. `cross_ecg_abp_rate_diff_bpm` (`bpm`) - ECG heart rate minus ABP pulse rate.
84. `cross_ecg_pleth_rate_diff_bpm` (`bpm`) - ECG heart rate minus PLETH pulse rate.
85. `cross_ecg_abp_rate_agreement` (`fraction`) - Coarse ECG/ABP rate agreement.
86. `cross_ecg_pleth_rate_agreement` (`fraction`) - Coarse ECG/PLETH rate agreement.
87. `delta_ecg_hr_bpm` (`bpm`) - Current ECG heart rate minus previous minute.
88. `delta_abp_map_median_mmhg` (`mmHg`) - Current MAP minus previous minute.
89. `delta_abp_sbp_median_mmhg` (`mmHg`) - Current SBP minus previous minute.
90. `delta_abp_dbp_median_mmhg` (`mmHg`) - Current DBP minus previous minute.
91. `delta_abp_pulse_pressure_median_mmhg` (`mmHg`) - Current pulse pressure minus previous minute.
92. `delta_pleth_amplitude_median` (`native`) - Current PLETH amplitude minus previous minute.
93. `delta_resp_rate_bpm` (`breaths_per_min`) - Current respiratory rate minus previous minute.

## Validation and Tests

Completed before v7 full extraction submission:

- focused unit/synthetic suite: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_waveform_feature_pipeline`
- result: `56` tests passed
- full unit-test discovery: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest discover tests`
- result: `82` tests passed
- syntax check: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m compileall waveform_baselines/wf_features scripts tests`
- result: passed
- real-data smoke cache: `/tmp/waveform_feature_cache_smoke_v7/v7/smoke_local_v7`
- smoke shape: `(4, 20, 93)`
- synthetic detector benchmark: `outputs/feature_models/v7_synthetic_ecg_detector_benchmarks.json`
- real-smoke rate agreement summary: `outputs/feature_models/v7_smoke_rate_agreement.json`
- smoke plots: `docs/figures/waveform_features_v7_smoke_sample0.png`, `docs/figures/waveform_features_v7_smoke_sample1.png`, `docs/figures/waveform_features_v7_smoke_sample2.png`

Smoke value check:

- ECG extraction diagnostics: `80` finite ECG detector runs, `80` XQRS attempts, `80` XQRS-used runs, `0` energy fallbacks
- 60-bpm synthetic ECG: `60` detected peaks and `59.52` bpm for upright and inverted QRS; canonical ECG-ABP/PLETH rates agree within `0.48` bpm
- ECG synthetic sweep: upright/inverted `40`, `60`, `80`, `120`, `160`, and `200` bpm recovered without 2x/3x over-detection
- ECG-ABP paired smoke tokens `56`: median absolute difference `0.00` bpm, p90 `0.69`, p95 `1.39`, within `5` bpm `1.00`, >`20` bpm `0.00`, no observed 2x/3x systematic pattern
- ECG-PLETH paired smoke tokens `80`: median absolute difference `0.00` bpm, p90 `0.76`, p95 `1.39`, within `5` bpm `1.00`, >`20` bpm `0.00`, no observed 2x/3x systematic pattern
- `resp_invalid_micro_fraction` absent from v7 metadata
- target arrays are absent from the smoke cache; `target_bundle_path` is metadata-only

## Full-Cohort Quality Audit

The v7 production array extraction `26873594` and merge `26873626` completed, all 32 shard directories plus the merged cache have `_SUCCESS` markers, and the merged cache shape is `(334833, 20, 93)`.

Full-cohort audit summary from `outputs/feature_models/v7_full_cohort_feature_quality_audit_2026-08-29.json`:

- samples: `334833`; patients: `885`; split sample counts: train `262408`, validation `35892`, test `36533`
- no zero-valid features were found
- maximum feature missing/nonfinite fraction: `delta_pleth_amplitude_median`, `0.0647`
- ECG HR distribution remained bounded by configured plausibility limits; `0.215%` of valid ECG-HR tokens were above `180` bpm and `0.001%` were below `40` bpm
- ABP summary pressures were mostly plausible: valid-token fractions above SBP `220` mmHg, DBP `140` mmHg, and below MAP `40` mmHg were `0.0166%`, `0.263%`, and `0.0356%`
- PLETH per-sample median amplitude showed a wide but bounded native-scale range: median `1.68`, p99 `1.96`, max `3.88`
- RESP rate remained a caveat at high rates: `3.70%` of valid RESP-rate tokens were above `60` breaths/min
- ECG/PLETH rate agreement was mostly strong by median but had a tail: median absolute difference `0.36` bpm, p95 `56.6` bpm, `8.63%` above `20` bpm
- ECG/ABP rate agreement has the largest remaining quality caveat: median absolute difference `0.33` bpm, p95 `89.3` bpm, `23.4%` above `20` bpm and `20.7%` above `40` bpm

Low ECG/ABP agreement overlays were generated for representative worst-tail examples:

- `docs/figures/waveform_features_v7_full_audit_low_ecg_abp_313876_m13.png`
- `docs/figures/waveform_features_v7_full_audit_low_ecg_abp_118193_m19.png`
- `docs/figures/waveform_features_v7_full_audit_low_ecg_abp_302246_m02.png`

The inspected `118193`/minute `19` overlay shows the ABP detector selecting secondary pulsatile peaks in addition to systolic peaks, inflating ABP pulse rate relative to ECG. This is an empirical quality caveat for interpreting cross-signal features and downstream model results, not a target-alignment issue.

Downstream v7 jobs `26873627`-`26873635` failed during target alignment before usable metrics/checkpoints. Representative error:

```text
ValueError: Missing target rows for 3610 feature cache anchors, e.g. ('p003866', 4253333041.1600003).
```

The mismatch was caused by sub-microsecond floating-point representation differences between otherwise identical cache and target anchor times. `scripts/train_feature_models.py` now rounds target-join anchor times to 6 decimal places; direct production-cache alignment checks reported zero missing rows for `outputs/targets/feature_targets_gap_vasopressor_free.npz` and `outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz`.

## v7 Final ECG Correction Summary

| Issue | Assessment | Action | Numeric representation changed? |
| ----- | ---------- | ------ | ------------------------------- |
| XQRS input path | numerical correctness problem | XQRS now receives raw calibrated ECG with short-gap interpolation only, preserving finite-run boundaries; the `5-20 Hz` detector signal is retained only for energy fallback and diagnostics. | Yes |
| ECG detector reproducibility | API/provenance cleanup | `ExtractionConfig` now explicitly records `ecg_detector` and `ecg_allow_energy_fallback`; invalid detector names fail early. Requesting XQRS without WFDB raises a clear error instead of silently using energy detection. | No physiological change by itself |
| Per-run fallback policy | API/provenance cleanup | XQRS empty/failed finite runs can use the energy detector only when `ecg_allow_energy_fallback=True`; detector peak sets are never combined. | No additional change |
| ECG diagnostics | provenance cleanup | Cache metadata aggregates ECG detector run counts: total runs, XQRS attempts/used/failed/zero/exception counts, energy fallback runs, direct energy runs, and no-detection runs when present. | No |
| ECG benchmark/smoke validation | validation | Focused ECG synthetic benchmark and real smoke agreement stayed accurate; the 4-sample smoke used XQRS for all `80` ECG finite runs. | No additional change |
| RESP documentation | documentation cleanup | RESP same-type extrema spacing documentation now matches code: derived from `resp_min_cycle_s`, currently `0.75 s`. | No |
| RR terminology | documentation cleanup | Remaining stale interval wording was replaced with valid RR interval/pair/difference wording. | No |

## v6 Final Correction Summary

| Issue | Assessment | Action | Numeric representation changed? |
| ----- | ---------- | ------ | ------------------------------- |
| ECG over-detection/polarity | numerical correctness problem | Reproduced v5 60-bpm synthetic ECG as `180` detections and `178.57` bpm; replaced simple positive `find_peaks` with finite-run WFDB XQRS plus a QRS-energy fallback and absolute-extremum morphology mapping. | Yes |
| ECG HR benchmark | validation | Added upright/inverted 40, 60, 80, 120, 160, and 200 bpm synthetic benchmarks with noise and baseline wander; output recorded in `outputs/feature_models/v6_synthetic_detector_benchmarks.json`. | No additional change |
| ECG missing-data behavior | already correct after detector replacement | Preserved finite-run IDs, same-run RR validity, no cross-gap RR, no all-NaN physiologic zeros, and HRV adjacency. | No additional change |
| ECG HRV terminology | documentation/API cleanup | Descriptions now say valid RR intervals and RR-based variability rather than strict normal-to-normal intervals because no ectopic/PVC cleaning is implemented. | No |
| RESP detector/config consistency | numerical correctness problem | Derived same-type extrema spacing from `resp_min_cycle_s`, preserving finite-run trough-to-peak-to-trough cycles and resolving 65-80 brpm half-rate aliasing. | Yes |
| ABP detector/config consistency | numerical correctness problem | Derived ABP detector spacing from `60 / abp_max_pulse_bpm` and allowed half-sample tolerance in duration validity at the configured max rate. | Yes |
| Pulsatile bad-trough robustness | numerical correctness problem | Failed trough refinements are local invalid markers; only beats touching the bad boundary are skipped. Shared refined boundaries remain reused for valid adjacent beats. | Yes |
| Split integrity | API/data-integrity cleanup | Split files now reject duplicate patient assignment, extraction filters to split-file patients, and extracted rows cannot have `unknown` split labels. Patient/sample split counts are stored in metadata. | No physiological change |
| Feature key uniqueness | API/data-integrity cleanup | Cache build rejects duplicate `(patient_id, anchor_time)` and duplicate `anchor_id`; metadata array lengths must equal `N`. | No physiological change |
| Target join | API/data-integrity cleanup | Feature-model target alignment remains key-based by `(patient_id, anchor_time)`, rejects duplicate/missing keys, ignores extra target rows without shifting alignment, and reports sample/valid/extra counts. | No physiological change |
| Cache metadata/provenance | API cleanup | Cache metadata now includes complete `extraction_config = self.config.to_dict()`. | No |
| Cache overwrite protection | API cleanup | Nonempty cache directories fail by default; `--overwrite` is required to replace them. | No |
| Sampling-rate/window source | API cleanup | Waveform loading derives window length from `self.config.input_samples` and raises if waveform metadata `fs` differs from extraction config. | No |
| Preprocessing zero-observation features | API/data-integrity cleanup | `FeaturePreprocessor.fit()` raises for any feature with zero valid training observations and records train valid counts/fractions. | No physiological change |
| RESP ratio normalization | preprocessing cleanup | `resp_rise_fall_ratio_median` is now normalized as an ordinary continuous timing ratio. | Model preprocessing only |
| Final validation | validation | Focused waveform tests passed `50`; full discovery passed `76`; compileall passed; v6 smoke cache and overlay plots were generated. | No additional change |

After the train/validation feature-quality audit passes and the target-alignment issue is resolved, the v7 extractor can be treated as the frozen waveform-feature baseline.

## v5 Audit Summary

| Issue | Assessment | Action | Numeric representation changed? |
| ----- | ---------- | ------ | ------------------------------- |
| `1` ECG RR across long gaps | numerical correctness problem | Preserve finite-run IDs for R peaks and require same-run RR intervals. | Yes |
| `2` RESP finite-run extrema | numerical correctness problem | Detect RESP peaks/troughs within finite filtered runs. | Yes |
| `3` RESP cycle phase | numerical correctness problem | Standardize RESP cycles to trough-to-peak-to-trough only. | Yes |
| `4` target-cache docs/API | documentation/API cleanup | Keep cache task-independent; remove target arrays from docs; target path is metadata-only; downstream joins by `(patient_id, anchor_time)` and rejects duplicates. | No physiological change |
| `5` ABP/PLETH refined boundaries | numerical correctness problem | Refine troughs once and share boundaries across adjacent beats. | Yes |
| `6` morphology consistency population | numerical correctness problem | Compute ABP/PLETH morphology consistency from physiologically valid pulses only. | Yes |
| `7` preprocessing documentation | documentation-only | Document observed-valid train medians/means/stds and transform order exactly. | No |
| `8` target-bundle plumbing | documentation/API cleanup | Remove `target_bundle_path` from waveform sample loading; retain builder field only as metadata provenance. | No |
| `9` stale clipping names | documentation/API cleanup | Rename internal `clipping_atol_fraction` to `extreme_value_atol_fraction`. | No numerical change |
| `10` stale invalid micro field | documentation/API cleanup | Remove unused internal `invalid_micro_fraction`. | No feature change |
| `11` ABP descriptions | documentation/API cleanup | Update source feature descriptions to refined-foot/refined-peak wording. | No numerical change |
| `12` v4 regression guard | already correct | Retained v4 filtering, ECG NaN safety, raw ABP/PLETH measurements, HRV adjacency, SQI, Transformer capacity, and timing tests. | No additional change |
| `13` new tests | test gap | Expanded focused suite to 39 tests and full discovery to 65 tests. | No |
| `14` version handling | operational | Bumped `FEATURE_VERSION` to `v5`; v4 retained for provenance and superseded. | Namespace/numerical version change |
| `15` SLURM handling | operational | Cancelled affected v4 jobs and submitted v5 extraction/dependent jobs. | No additional feature change |
| `16` post-pass direction | process | Stop implementation-level redesign unless full-cohort quality audit reveals empirical failures. | No |

## Version History

### v7

Current production candidate; production array `26873594` and merge `26873626` completed. Changes from v6:

- ECG detector selection is explicit in `ExtractionConfig` through `ecg_detector` and `ecg_allow_energy_fallback`
- requesting `ecg_detector = "xqrs"` without WFDB raises a clear error instead of silently switching algorithms
- WFDB XQRS receives raw calibrated ECG with only short-gap interpolation; long gaps remain finite-run boundaries
- the `5-20 Hz` ECG detector signal is retained for the QRS-energy fallback and diagnostics, not as XQRS input
- per-run XQRS failures or zero detections use energy fallback only when explicitly allowed
- cache metadata aggregates ECG detector diagnostics so XQRS/fallback usage is auditable
- sharded extraction and merge integrity were hardened without changing physiological feature algorithms
- RESP extrema-spacing documentation and stale RR terminology were corrected

Superseded v6 jobs cancelled before completion:

- serial extraction: `26873175`
- tabular/persistence: `26873177`-`26873181`
- GRU/Transformer: `26873182`-`26873185`

Pre-hardening v7 sharded jobs cancelled before completion:

- extraction array: `26873436`
- shard merge: `26873437`
- tabular/persistence: `26873469`-`26873473`
- GRU/Transformer: `26873474`-`26873477`

Final original v7 job status:

- extraction array: `26873594` (`0-31`), completed
- shard merge: `26873626`, completed
- tabular/persistence: `26873627`-`26873631`, failed during target alignment
- GRU/Transformer: `26873632`-`26873635`, failed during target alignment

Resubmitted after the target-alignment fix:

- tabular/persistence: `26898023`-`26898027`
- GRU/Transformer: `26898042`, `26898044`, `26898046`, `26898048`

Additional all-target v7 regression jobs were submitted after the initial MAP-only comparison to cover all 26 `t+0m_gap` targets from `docs/v1_vasopressor_free/regression_results_v1_vaso_free_sorted.md`. Current-state XGBoost jobs are `26898353`-`26898377`, history XGBoost jobs are `26898378`-`26898403`, GRU jobs are `26898404`-`26898428`, and Transformer jobs are `26898429`-`26898453`. See `docs/v7_extracted_features/extractedFeaturesRegression.md` for current completion status and comparisons.

### v6

Superseded on `2026-08-28` before completion after the final ECG XQRS-input/provenance correction required v7. Changes from v5:

- ECG detection uses finite-run WFDB XQRS with QRS-energy fallback instead of simple positive `find_peaks`
- ECG morphology mapping and QRS-width estimation are polarity-consistent via strongest absolute local extrema
- RESP extrema spacing is derived from `resp_min_cycle_s`, preserving 10-80 brpm support without half-rate aliasing
- ABP detector spacing is derived from `abp_max_pulse_bpm`, preserving detection through 220 bpm
- one failed ABP/PLETH trough refinement invalidates only local beats
- cache build filters to split-file patients, rejects split/key integrity problems, serializes full extraction config, and blocks accidental overwrite
- downstream feature-model target alignment explicitly reports key-based join counts and rejects missing/duplicate keys
- `resp_rise_fall_ratio_median` is normalized as a continuous morphology feature

Cancelled serial v6 jobs:

- extraction: `26873175`
- tabular/persistence: `26873177`-`26873181`
- GRU/Transformer: `26873182`-`26873185`

Cancelled superseded v5 jobs:

- extraction: `26872137`
- tabular/persistence: `26872138`-`26872142`
- GRU/Transformer: `26872143`-`26872146`

### v5

Superseded on `2026-08-28`. Changes from v4:

- ECG valid RR intervals cannot cross long finite-run gaps
- RESP extrema detection is finite-run aware
- RESP cycles use one mechanical phase: trough-to-peak-to-trough
- ABP/PLETH refined trough boundaries are shared exactly by adjacent beats
- ABP/PLETH morphology consistency uses physiologically valid pulses only
- feature cache remains task-independent; target bundles are joined downstream by `(patient_id, anchor_time)`
- internal cleanup of extreme-value and invalid-micro naming

Cancelled v5 jobs:

- extraction: `26872137`
- tabular/persistence: `26872138`-`26872142`
- GRU/Transformer: `26872143`-`26872146`

### v4

Superseded on `2026-08-28`. v4 fixed short-run `filtfilt`, ECG finite-run peak detection, NaN-safe ECG morphology, detector-local ABP/PLETH raw extrema, ABP beat-duration validity, all-missing extreme-value NaN behavior, HRV minimum-pair rules, RESP invalid-fraction schema cleanup, and Transformer positional capacity. It was cancelled after the final audit found remaining numerical issues in ECG cross-gap RR intervals, RESP finite-run/cycle phase handling, pulsatile refined boundary sharing, and morphology-consistency populations.

Cancelled v4 jobs:

- extraction: `26871618`
- tabular/persistence: `26871691`-`26871695`
- GRU/Transformer: `26871696`-`26871699`

Cancelled interim v4 jobs:

- extraction: `26871441`
- tabular/persistence: `26871446`-`26871450`
- GRU/Transformer: `26871451`-`26871454`

### v3

Superseded on `2026-08-28`. v3 fixed major v2 issues such as trough-to-trough pulsatile segmentation, long-gap interpolation, HRV adjacency, RESP valid-cycle denominator, missing-vs-flatline SQI, and history slopes. It was cancelled before completion because v4 found remaining numerical issues.

Cancelled v3 jobs:

- extraction: `26870874`
- tabular/persistence: `26870880`-`26870884`
- GRU/Transformer: `26870876`-`26870879`

### v2

Superseded by v3 after confirmed numerical issues. v2 introduced raw ABP pressure measurements, direct beat-wise MAP, ECG morphology separation, RESP high-rate recovery, duplicate SDNN removal, and polarity-neutral RESP names, but still had later-discovered segmentation and missingness problems.

Cancelled v2 jobs:

- `26870362`-`26870371`

### v1

Initial implementation. Superseded after the first methodology audit found numerical and naming issues.

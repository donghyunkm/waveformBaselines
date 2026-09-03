# Extracted Features Verification

This page consolidates the verification work for the v7 extracted waveform-feature cache. The implementation details and ordered feature definitions remain in `extractedFeatures.md`; this page focuses on evidence that was checked, generated artifacts, plots, and known caveats.

## Scope

- feature version: `v7`
- production cache: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/v7/vasopressor_free_waveform_features_v7`
- merged cache shape: `(334833, 20, 93)`
- token count: `6696660`
- patients: `885`
- split sample counts: train `262408`, validation `35892`, test `36533`

## Verification Status

The v7 extractor was syntax checked, unit/synthetic tested, smoke tested on real waveform samples, run as a 32-shard production extraction, merged with shard/cache integrity checks, and audited at full-cohort scale. This supports using v7 as the frozen extracted-feature baseline, with the ECG/ABP rate-disagreement caveat below.

Verification does not mean every detector output is physiologically correct in every noisy ICU segment. The full-cohort audit found no zero-valid features and generally plausible distributions, but it also found a substantial ECG/ABP rate-disagreement tail that should be considered when interpreting cross-signal features and downstream models.

## Tests and Smoke Runs

- Focused waveform-feature suite: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_waveform_feature_pipeline`; `56` tests passed.
- Full repository test discovery: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest discover tests`; `82` tests passed.
- Syntax check: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m compileall waveform_baselines/wf_features scripts tests`; passed.
- Real-data smoke cache: `/tmp/waveform_feature_cache_smoke_v7/v7/smoke_local_v7`.
- Smoke cache shape: `(4, 20, 93)`.
- Synthetic detector benchmark: `outputs/feature_models/v7_synthetic_ecg_detector_benchmarks.json`.
- Real-smoke rate agreement summary: `outputs/feature_models/v7_smoke_rate_agreement.json`.
- Full-cohort audit summary: `outputs/feature_models/v7_full_cohort_feature_quality_audit_2026-08-29.json`.

## Plot Reading Notes

In the waveform-overlay plots, the blue curve is the raw input waveform for the channel being inspected. The yellow/orange curve is the detector signal: a filtered helper signal used to make peaks, troughs, or extrema easier to find. It is not a separately measured physiological channel and is not directly stored as the feature value. ECG uses a `5-20 Hz` detector curve for R-peak detection; ABP and PLETH use bandpass detector curves for pulse peaks/troughs; RESP uses a `0.05-1.5 Hz` filtered curve for respiratory extrema.

The bottom minute-level trajectory panel uses colors differently: each colored line is a feature trajectory across the 20 input minute tokens, with labels in the legend.

## Synthetic and Real-Smoke Checks

The synthetic checks were used to verify detector behavior on controlled waveforms before relying on real ICU segments. The v7 synthetic ECG sweep recovered upright and inverted ECG examples at `40`, `60`, `80`, `120`, `160`, and `200` bpm without the prior 2x/3x over-detection pattern. A canonical 60-bpm ECG/ABP/PLETH case gave 60 detected ECG peaks and `59.52` bpm for upright and inverted ECG, with ECG-ABP and ECG-PLETH rates agreeing within `0.48` bpm.

<img src="../figures/extracted_features_smoke/synthetic_feature_smoke.png" alt="Synthetic extracted-feature smoke verification plot" width="720" />

The real-data v7 smoke run produced `80` finite ECG detector runs, `80` XQRS attempts, `80` XQRS-used decisions, and `0` energy-fallback cases. Paired real-smoke rate agreement was tight: ECG-ABP had `56` paired tokens with median absolute difference `0.00` bpm, p90 `0.69`, p95 `1.39`, and all paired tokens within `5` bpm; ECG-PLETH had `80` paired tokens with median absolute difference `0.00` bpm, p90 `0.76`, p95 `1.39`, and all paired tokens within `5` bpm.

<img src="../figures/waveform_features_v7_smoke_sample0.png" alt="v7 real-data smoke plot sample 0" width="720" />

<img src="../figures/waveform_features_v7_smoke_sample1.png" alt="v7 real-data smoke plot sample 1" width="720" />

<img src="../figures/waveform_features_v7_smoke_sample2.png" alt="v7 real-data smoke plot sample 2" width="720" />

## Full-Cohort Audit

The full production extraction completed as SLURM array `26873594`, with merge job `26873626`. All 32 shard directories and the merged cache had `_SUCCESS` markers.

Key audit results:

- zero-valid features: none
- maximum feature missing/nonfinite fraction: `delta_pleth_amplitude_median`, `0.0647`
- ECG heart-rate outlier fractions among valid tokens: `0.215%` above 180 bpm and `0.001%` below 40 bpm
- ABP pressure plausibility fractions among valid tokens: `0.0166%` SBP above 220, `0.263%` DBP above 140, `0.0356%` MAP below 40
- PLETH median amplitude: median `1.68`, p99 `1.96`, max about `3.99`
- RESP high-rate caveat: `3.70%` of valid RESP-rate tokens were above 60 breaths/min
- ECG-PLETH rate agreement: median absolute difference `0.36` bpm, p95 `56.6`, `8.63%` above 20 bpm
- ECG-ABP rate agreement: median absolute difference `0.33` bpm, p95 `89.3`, `23.4%` above 20 bpm, `20.7%` above 40 bpm

The low ECG/ABP agreement overlays below were generated to inspect the full-cohort tail. The inspected `118193` minute `19` example showed the ABP detector selecting secondary pulsatile peaks in addition to systolic peaks, inflating ABP pulse rate relative to ECG. This is a feature-quality caveat for ABP pulse-rate and cross-signal rate features, not evidence of target-alignment failure.

<img src="../figures/waveform_features_v7_full_audit_low_ecg_abp_118193_m19.png" alt="v7 full-cohort low ECG/ABP agreement audit plot for patient 118193 minute 19" width="720" />

<img src="../figures/waveform_features_v7_full_audit_low_ecg_abp_302246_m02.png" alt="v7 full-cohort low ECG/ABP agreement audit plot for patient 302246 minute 2" width="720" />

<img src="../figures/waveform_features_v7_full_audit_low_ecg_abp_313876_m13.png" alt="v7 full-cohort low ECG/ABP agreement audit plot for patient 313876 minute 13" width="720" />

## Source-Feature Agreement Audit

A same-window agreement audit now compares selected v7 extracted features against the historical source feature arrays under `/gpfs/data/eh3828lab/derived_datasets/baselines/output_v2`. The audit script is `scripts/audit_v7_feature_source_agreement.py`; outputs are `outputs/feature_models/v7_source_agreement_2026-09-02.{json,csv,md}`. The source arrays are comparators, not adjudicated clinical ground truth, because detector implementations, window aggregation, and feature definitions may differ from v7.

Command:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/audit_v7_feature_source_agreement.py
```

Row alignment passed exactly: `334833 / 334833` v7 cache rows matched source rows by `(patient_id, anchor_time)` after 6-decimal timestamp quantization. The table below reports the best of v7 `mean`, `median`, and `last` 20-token summaries for each comparator.

| Source comparator | v7 feature | v7 summary | N common | Pearson r | Spearman r | Bias | MAE | RMSE | p95 abs error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HR | `ecg_hr_bpm` | last | 334726 | -0.1745 | -0.2489 | -18.18 | 21.55 | 32.05 | 63.83 |
| RR | `resp_rate_bpm` | mean | 334828 | 0.3072 | 0.2807 | 1.48 | 6.897 | 9.994 | 21.70 |
| SBP | `abp_sbp_median_mmhg` | mean | 334747 | 0.9451 | 0.9371 | -2.724 | 3.191 | 8.082 | 21.57 |
| DBP | `abp_dbp_median_mmhg` | mean | 334747 | 0.9889 | 0.9862 | 0.5452 | 1.023 | 2.115 | 4.753 |
| MAP | `abp_map_median_mmhg` | mean | 334747 | 0.9793 | 0.9855 | -0.1422 | 1.272 | 3.093 | 4.920 |
| PP | `abp_pulse_pressure_median_mmhg` | mean | 334564 | 0.9146 | 0.8987 | -2.997 | 3.714 | 9.302 | 24.36 |
| ABP_area | `abp_pulse_area_median` | mean | 334747 | 0.8689 | 0.8512 | -0.9646 | 1.736 | 3.578 | 8.640 |
| PLETH_amp | `pleth_amplitude_median` | mean | 330615 | 0.9867 | 0.9715 | -0.00937 | 0.0331 | 0.0951 | 0.1123 |
| HRV_RMSSD | `ecg_hrv_rmssd_s` | mean | 334600 | 0.4690 | 0.5603 | -0.0254 | 0.0558 | 0.0793 | 0.1663 |
| dPdt_max | `abp_dpdt_max_median` | mean | 334683 | 0.9443 | 0.9289 | -44.68 | 62.23 | 159.4 | 362.8 |
| RESP_amp | `resp_amplitude_median` | mean | 334833 | 0.8442 | 0.8215 | 0.0141 | 0.1233 | 0.2067 | 0.4311 |

Interpretation:

- ABP pressure features (`SBP`, `DBP`, `MAP`, `PP`) show strong source agreement, with best Pearson correlations from `0.915` to `0.989`.
- Native-scale morphology/amplitude comparators are also informative: `PLETH_amp` is very high agreement, while `ABP_area`, `dPdt_max`, and `RESP_amp` are strong but definition-sensitive.
- `HR` and `RR` source agreement is weak. Given v7 synthetic and real-smoke detector checks, these should be treated as source-definition/detector disagreement flags requiring outlier review rather than proof that the v7 detectors are wrong.
- Most v7 columns still do not have source comparators: validity/missingness fractions, flatline/extreme-value fractions, morphology consistency, many HRV descriptors, PLETH/ABP timing morphology, cross-signal agreement scores, and deltas remain waveform-derived QC/physiology features.

## Known Caveats and Follow-Ups

- ECG/ABP pulse-rate disagreement has a real full-cohort tail, driven at least in part by ABP secondary-peak selection in noisy or complex segments.
- RESP rates above 60 breaths/min occur in `3.70%` of valid RESP-rate tokens and should be treated as a quality caveat for RESP-derived features.
- PLETH amplitude is reported on the native processed waveform scale; it was verified as bounded rather than calibrated to physical units.
- Target-bundle alignment initially failed for several downstream jobs because of sub-microsecond `anchor_time` differences. The training join was fixed by rounding join keys to 6 decimals in `scripts/train_feature_models.py`; production checks found zero missing rows after the fix.

## Related Artifacts

- feature pipeline doc: `docs/v7_extracted_features/extractedFeatures.md`
- job status and audit notes: `docs/v7_extracted_features/v7_feature_job_status_2026-08-29.md`
- full-cohort audit JSON: `outputs/feature_models/v7_full_cohort_feature_quality_audit_2026-08-29.json`
- source-feature agreement audit: `outputs/feature_models/v7_source_agreement_2026-09-02.{json,csv,md}`
- synthetic detector benchmark JSON: `outputs/feature_models/v7_synthetic_ecg_detector_benchmarks.json`
- smoke rate-agreement JSON: `outputs/feature_models/v7_smoke_rate_agreement.json`

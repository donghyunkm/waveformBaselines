# Supplemental Waveform Features v8

`v8` is a supplemental raw-waveform feature cache intended to be joined with the frozen `v7` extracted-feature cache by identical `anchor_id`, `patient_id`, `anchor_time`, and `split_label`. It does not replace v7 and does not re-store the existing 93 v7 features.

Code:

- `waveform_baselines/wf_features_v8/definitions.py`
- `waveform_baselines/wf_features_v8/config.py`
- `waveform_baselines/wf_features_v8/pipeline.py`
- `waveform_baselines/wf_features_v8/morphology.py`
- `waveform_baselines/wf_features_v8/cache.py`
- `scripts/build_waveform_features_v8.py`
- `scripts/audit_waveform_features_v8.py`
- `tests/test_waveform_feature_v8.py`

The extractor reuses v7 channel order, sampling rate, 20-minute input length, one-minute tokenization, and v7 detector/segmentation primitives. The production writer verifies v7/v8 sample identity and feature-name disjointness before writing `_SUCCESS`.

Cross-signal timing, pulse-deficit, basic PLETH fiducial, PLETH derivative-fiducial, advanced ABP morphology, and systolic-time features are present for audit use but disabled by default:

```text
enable_cross_signal_timing = False
enable_pulse_deficit_features = False
enable_pleth_fiducials = False
enable_pleth_derivative_fiducials = False
enable_abp_advanced_morphology = False
enable_systolic_time_features = False
```

Disabled features are written as `NaN` with `mask=false` unless the corresponding audit flag is explicitly enabled.

## Cache Format

```text
featureExtraction/v8/<run_name>/
    values.npy
    mask.npy
    patient_ids.npy
    anchor_times.npy
    anchor_ids.npy
    split_labels.npy
    segment_ids.npy
    segment_names.npy
    anchors.csv
    metadata.json
    feature_quality_report.json
    alignment_report.json
    _SUCCESS
```

The current definition set has `194` v8 columns, so `values.shape = (N, 20, 194)`. The external version remains `v8`; stale-cache protection is handled by metadata fields:

```text
feature_version = v8
feature_schema_revision = 7
feature_schema_hash = a72aeb3b2a2942a899851cffab6d43228e939a82653514907b4fe8748cae229b
```

`load_v8_feature_cache()` rejects caches whose feature version, ordered feature names, feature dimension, schema revision, or schema hash do not match the current code.

## Literature-Driven Expansion

The September 2026 expansion keeps v8 in place and adds a compact subset motivated by:

- Pal et al., `npj Cardiovascular Health` 2025, which emphasizes the onset -> systolic peak -> dicrotic notch -> diastolic peak landmark chain for ABP/PPG and large landmark-derived feature sets: https://www.nature.com/articles/s44325-025-00096-0
- Goda, Charlton, and Behar pyPPG, `Physiological Measurement` 2024, which standardizes PPG fiducials, VPG/APG landmarks, and derivative biomarkers: https://pmc.ncbi.nlm.nih.gov/articles/PMC11003363/ and https://pyppg.readthedocs.io/en/latest/
- Tsai et al., `BMC Medical Informatics and Decision Making` 2025, which derives ABP/ECG/RESP/SpO2 shock-prediction features and highlights ABP `TimeSBP2DBP` SampEn and respiratory-waveform variability: https://link.springer.com/article/10.1186/s12911-025-03108-2

SpO2 is intentionally not added to the waveform-only v8 input contract; it remains a separate numerics/multimodal feature source.

## Ordered Feature List

The exact ordered feature list, units, roles, source channels, default-enabled flags, and minimum-observation rules are encoded in `waveform_baselines/wf_features_v8/definitions.py` and serialized into every v8 `metadata.json`. The current ordered ranges are:

- 1-9: ABP and PLETH respiratory variation: `abp_ppv_pct`, `abp_spv_mmhg`, `abp_sbp_resp_variation_pct`, `abp_dbp_resp_variation_pct`, `abp_map_resp_variation_pct`, `abp_pulse_area_resp_variation_pct`, `pleth_resp_amplitude_variation_pct`, `pleth_area_resp_variation_pct`, `pleth_width_resp_variation_pct`.
- 10-18: disabled synchronization-gated PAT/PTT-like timing: `ecg_abp_pat_median_ms`, `ecg_abp_pat_iqr_ms`, `ecg_abp_pat_sd_ms`, `ecg_pleth_pat_median_ms`, `ecg_pleth_pat_iqr_ms`, `ecg_pleth_pat_sd_ms`, `abp_pleth_delay_median_ms`, `abp_pleth_delay_iqr_ms`, `abp_pleth_delay_sd_ms`.
- 19-24: fixed-window SBP-RR coupling and sequence-method baroreflex features.
- 25-33: rolling 5-minute ABP nonlinear dynamics and signed SBP/DBP successive-difference features.
- 34-42: nonlinear and frequency-domain rolling 5-minute HRV features.
- 43-58: disabled advanced ABP notch, diastolic-peak, normalized area-fraction, and tau features.
- 59-90: one-minute short-timescale slopes, burden features, RESP instability, and polarity-invariant RESP coupling.
- 91-96: rolling 5-minute ECG rhythm-instability and QRS morphology features.
- 97-100: disabled synchronization-gated ECG-to-pulse mechanical deficit features.
- 101-110: default-on scale-invariant PLETH morphology features.
- 111-132: disabled PLETH notch/secondary-peak and VPG/APG derivative fiducials. The former `pleth_b_time_median_s` remains removed because it was not a validated APG b-wave fiducial.
- 133-149: temporally ordered ABP lability, DBP lability, and ABP morphology-outlier features.
- 150-157: ECG-/PLETH-derived respiration and cross-modal respiratory-rate agreement.
- 158-179: rolling 5-minute RESP pattern, RESP rate-variability, and selected morphology-dynamics features.
- 180-189: waveform-quality and hardware-artifact component features. The dead, unimplemented `abp_flush_like_fraction` schema entry was removed before freeze.
- 190-194: disabled ABP systolic-time features.

## Definitions and Validity

Detailed units, source channels, minimum required beats/cycles, valid ranges, synchronization requirements, and default enabled status are recorded in each cache `metadata.json` and in `waveform_baselines/wf_features_v8/definitions.py`.

Key conventions:

- PPV: median across valid RESP cycles of `100 * (PPmax - PPmin) / ((PPmax + PPmin) / 2)`.
- SPV: median across valid RESP cycles of `SBPmax - SBPmin` in mmHg.
- PLETH respiratory amplitude variation: median across valid RESP cycles of `100 * (ampmax - ampmin) / ampmax`; this is not labeled PVI because calibrated perfusion index is unavailable.
- Baroreflex: all baroreflex and generic SBP-RR features are fixed full-causal-5-minute features with at least 180 seconds of ABP coverage. Sequence-method gain uses monotonic SBP runs with concordant delayed RR changes over lags 0, 1, and 2 beats, but each SBP sequence is accepted at most once using the best valid lag. ABP beat indices must be consecutive in the same ABP finite run, RR observations must be valid and continuous in the same ECG finite run, and invalid beats/gaps break sequences. Generic SBP-RR lag correlations are named separately and are not interpreted as BRS.
- Sample entropy: full rolling 5-minute causal RR window, longest continuous valid RR run only, `m=2`, `r=0.2*SD`, Chebyshev distance, minimum 30 consecutive RR intervals, and at least 240 seconds RR coverage.
- DFA alpha1: full rolling 5-minute causal RR window, longest continuous valid RR run only, scales 4, 6, 8, 10, 12, and 16 beats, linear fit on log fluctuation versus log scale, minimum 40 consecutive RR intervals.
- Frequency-domain HRV: variance-normalized Lomb-Scargle RR PSD with total 0.0033-0.40 Hz, LF 0.04-0.15 Hz, HF 0.15-0.40 Hz, minimum 30 consecutive RR intervals and 240 seconds coverage.
- ECG rhythm instability: full rolling 5-minute causal RR/QRS history only; RR adjacency requires adjacent valid RR intervals in the same finite run, and invalid RR values are not compressed away. Detector-to-morphology QRS mapping stores one entry per detector R peak; failed mappings are `-1`, and morphology evidence is never borrowed from a neighboring beat.
- QRS/ABP/PLETH morphology outliers: denominators are segments with valid normalized template comparisons. Degenerate or unusable segments remain `NaN` in their original positions.
- PLETH morphology: valid native-scale pulses are normalized by refined foot-to-refined-peak amplitude for shape features; widths use the detector-refined physiological peak, not the full-segment global maximum. STT uses maximum positive derivative only over the refined-foot-to-refined-peak upstroke. Kurtosis uses Fisher convention.
- CV features use the conventional coefficient of variation, `SD / abs(mean)`, with finite/minimum-count checks. They are not described as robust CVs.
- ABP lability: ARV, RMSSD, p95 successive changes, MAP drops, and scale changes only use adjacent valid same-run beats. Decline run length is counted in beats; `N` consecutive declining eligible transitions reports `N+1`, no declines reports `0`, and no eligible transitions returns missing.
- Derived respiration: ECG R-peak amplitude and PLETH amplitude/area surrogates are detrended and analyzed with Lomb-Scargle over the configured respiratory band using the selected continuous event finite run. Continuous-run selection ranks candidates by temporal duration, then event count, then latest event time. PLETH amplitude/area components are centered and scaled only after selecting the continuous run, so disconnected gain-shifted PLETH runs do not affect the analyzed run. Strength is integrated power in `f_peak +/- derived_resp_peak_half_width_hz` divided by integrated total respiratory-band power, making it less dependent on frequency-grid resolution. PLETH amplitude and area components are optional independent surrogates.
- RESP pauses: detected separately from normal-cycle extraction using finite observed RESP intervals with at least 10 seconds without accepted extrema and surrounding respiratory activity; long NaN gaps and complete flatlines return missing rather than apnea.
- RESP coupling: `resp_*_max_abs_correlation` features search RESP time shifts over `+/- resp_coupling_max_lag_seconds` and take the maximum absolute correlation, preserving polarity invariance.
- RESP nonlinear interval and amplitude-dynamics features: interval SampEn, amplitude-envelope CV, sigh count, suppressed-amplitude burden, and periodic-envelope spectra use the selected longest continuous accepted-cycle finite run and require the configured 240 seconds of cycle-run coverage. Pause burden remains separate because it is measured directly from observed RESP signal intervals.
- Burden fractions are beat or RR-observation fractions as indicated in the feature name. Burden thresholds intentionally avoid MAP <=65 and SBP <=90.
- Quality features are interpretable artifact components, not validated diagnostic labels. Spectral energy is computed separately within finite runs and duration-weighted before forming ratios; baseline-jump and ABP step-change comparisons use median absolute change plus MAD scale thresholds, preserve 5-second window identity, and missing windows break continuity. PLETH plateau and quantization metrics are finite-run-local and use normalized adjacent differences rather than fixed native-unit rounding.
- Advanced ABP morphology: dicrotic notch detection uses the morphology segment, while notch pressure ratio, systolic area, diastolic area, and tau fitting use the aligned raw calibrated ABP segment. The systolic/diastolic area ratio uses the same linear foot-to-next-foot baseline as the systolic area fraction. Tau fitting excludes the immediate post-notch interval and starts after a detected rebound when present; rebound prominence uses the refined systolic peak when available.
- ABP nonlinear dynamics: full rolling 5-minute features use the longest continuous valid ABP beat run. SampEn uses `m=2`, `r=0.2*SD`, and at least `min_abp_nonlinear_beats`. `T_peak_to_DBP` is measured from systolic peak to the next refined foot, not to the notch.
- PLETH derivative fiducials: audit-only VPG/APG features use a dedicated derivative-preserving full-window PLETH morphology signal, Savitzky-Golay smoothing with a window derived from seconds and sampling rate, and pyPPG-style ordered fiducials. VPG/APG/JPG derivative samples inside half a Savitzky-Golay window of finite-run boundaries are masked unreliable. VPG `u` is the largest first derivative before the systolic peak, `v` is bounded by the diastolic peak, and `w` is tied to post-notch late-derivative morphology. APG detection follows `a -> b -> e -> c -> d`; missing APG `c`/`d` extrema can fall back only to a single sign-correct JPG zero crossing in the appropriate interval, otherwise they remain missing. Individual features aggregate their own valid beat populations, while `pleth_derivative_fiducial_valid_fraction` is the complete `u/v/w/a/b/c/d/e` set fraction. Derivative ratios are amplitude ratios in APG space; the aging index is `(b-c-d-e)/a`. pyPPG remains an optional validation dependency, not a production dependency.
- RESP RRV: full rolling 5-minute breath-interval variability uses the longest continuous RESP cycle run. Time-domain, RMSSD/SDSD/Poincare, and spectral features all require approximately 240 seconds of cycle-run coverage.
- Morphology dynamics: selected 5-minute descriptors summarize ABP pulse-area CV, ABP dP/dt CV/slope, PLETH width CV, and audit-only PLETH reflection/APG-ratio variability. Each dynamic feature selects its own longest continuous valid beat run and slopes use event times, not beat index.

## Smoke Run

Latest frozen default command:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/build_waveform_features_v8.py \
  --cache-root /tmp/waveform_feature_cache_smoke_v8_frozen \
  --output-name smoke_local_v8_frozen \
  --max-samples 2 \
  --overwrite \
  --allow-prefix-v7-alignment
```

Result:

- Cache: `/tmp/waveform_feature_cache_smoke_v8_frozen/v8/smoke_local_v8_frozen`
- Shape: `(2, 20, 194)`
- Alignment against frozen full-data v7 prefix: passed
- v7/v8 feature-name overlap: `0`
- Audit JSON: `/tmp/waveform_feature_cache_smoke_v8_frozen/smoke_local_v8_frozen_audit.json`

Enabled pre-existing and new one-minute features generally had nonzero validity on the smoke set. Rolling 5-minute features are missing for early tokens by design. Timing, pulse-deficit, PLETH fiducial, advanced ABP morphology, and systolic-time columns were all missing in the default smoke because production flags were intentionally disabled.

Frozen smoke audit review flags:

- Default smoke finite features: `130`; all-missing features: `64`, primarily disabled experimental/synchronization-gated columns.
- Near-zero variance on this tiny smoke set: `resp_pause_count`, `resp_longest_pause_s`, `ecg_derived_resp_rate_5m`, `resp_pause_burden_5m`.
- High v8-v8 correlations with absolute correlation at least 0.95: `52` pairs in the default smoke. These are unstable two-sample smoke flags, not removal decisions.
- High v7-v8 correlations with absolute correlation at least 0.95: `40` pairs on the two-sample prefix comparison. These are smoke-size flags only and were not used for feature removal.

Selected default smoke coverage:

- `abp_step_change_count`: valid fraction `1.0`, median `0.0`, IQR `1.0`, min `0.0`, max `6.0`.
- `ecg_baseline_jump_count`: valid fraction `1.0`, median `0.0`, IQR `0.0`, min `0.0`, max `3.0`.
- `resp_baseline_jump_count`: valid fraction `1.0`, median `0.0`, IQR `0.0`, min `0.0`, max `2.0`.
- `pleth_plateau_fraction`: valid fraction `1.0`, median `0.3518`, IQR `0.1105`, min `0.0320`, max `0.5423`.
- `pleth_quantization_index`: valid fraction `1.0`, median `0.3866`, IQR `0.2461`, min `0.0992`, max `0.5965`.
- `hrv_sampen_5m`: valid fraction `0.45`, median `1.2197`, IQR `0.9569`.
- `resp_interval_rmssd_5m`: valid fraction `0.8`, median `1.8172`, IQR `0.1889`.

Frozen timed smoke command:

```bash
/usr/bin/time -p /gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/build_waveform_features_v8.py \
  --cache-root /tmp/waveform_feature_cache_smoke_v8_frozen_timed \
  --output-name smoke_local_v8_frozen_timed \
  --max-samples 1 \
  --overwrite \
  --allow-prefix-v7-alignment
```

Result: `real 27.16`, `user 26.26`, `sys 0.94` seconds for one 20-minute sample including cache writes and alignment validation.

Small extreme-pressure smoke scan over the frozen 2-anchor smoke found `2591` detected ABP beats, `2579` passing current validity, `8` with `SBP < 50`, `10` with `DBP < 20`, `8` with `MAP < 40`, and `0` with `SBP > 220`. Earlier printed examples were very low or negative-pressure detections and were invalid under current thresholds; full-cohort QC should still include representative waveform overlays before using severe-hypotension censoring conclusions analytically.

## Tests

Command:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_waveform_feature_v8
```

Focused result: `63` tests run, `62` passed and `1` optional pyPPG validation hook skipped because pyPPG is not installed in this environment. Full discovery result: `254` tests run, `253` passed and `1` optional pyPPG hook skipped. Coverage includes guarded real-data ECG detector parity when the full-data waveform assets are available, RR/baroreflex adjacency, QRS mapping alignment, pulse-deficit run resets and target-run matching, respiratory pause safety, conservative PLETH notch/diastolic-peak handling, PLETH derivative `u/a` validity, APG `c/d` JPG fallback and ambiguous-fallback rejection, PLETH morphology, ABP lability and DBP lability, ABP nonlinear dynamics, RESP temporal-coverage gating, PLETH derivative fiducials, RESP RRV, run-local PLETH-derived respiration, robust outlier thresholds, PLETH quality gain invariance, downstream-derived feature handling, artifact directionality, schema validation/stale-cache rejection, shape-validation failure behavior, and token-level causality.

Syntax check:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m compileall waveform_baselines/wf_features_v8 scripts/build_waveform_features_v8.py scripts/audit_waveform_features_v8.py tests/test_waveform_feature_v8.py
```

Result: passed.

## Required Real-Data Audit Before Production Timing/Morphology

Before enabling cross-signal timing in production, run representative extraction with `--enable-cross-signal-timing`, inspect PAT/PTT distributions, verify within-recording stability, check for channel/integer-sample offsets, and generate waveform overlays around ECG R peaks, ABP feet, and PLETH feet.

Before enabling advanced ABP morphology, inspect notch detections, systolic/diastolic area splits, tau fit residuals, and low/high/missing tail examples on real ICU waveforms.

This revision-6 freeze state was superseded by the final revision-7 qualifying-run fix before full feature extraction/QC validation. Full extraction QC should still include larger-cohort feature distribution tables, continuous-run coverage summaries, PLETH/ABP landmark validity fractions, manual extreme-pressure waveform inspection, redundancy/stability flags, and profiler-level timing for detector, entropy, derivative-fiducial, template-distance, and baroreflex components.


Frozen smoke continuous-history coverage over the 2-anchor smoke final 5-minute histories:

- Median longest RR run duration: `255.096 s`; HRV coverage-satisfied fraction: `0.5`.
- Median longest ABP run duration: `297.288 s`; ABP nonlinear coverage-satisfied fraction: `1.0`.
- Median longest PLETH run duration: `68.736 s`; PLETH morphology-dynamics coverage-satisfied fraction: `0.0`.
- Median longest RESP run duration: `293.936 s`; RESP coverage-satisfied fraction: `1.0`.

Audit-enabled frozen smoke path: `/tmp/waveform_feature_cache_smoke_v8_frozen/v8/smoke_local_v8_frozen_audit_enabled`, shape `(2, 20, 194)`, v7-prefix alignment passed. Selected audit-enabled landmark validity-summary columns were finite on all 40 tokens. Median fractions: PLETH notch `0.2301`, PLETH diastolic peak `0.0592`, complete derivative fiducial set `0.2426`, ABP notch `0.00685`, ABP diastolic peak `0.0`, ABP tau `0.0`. Median derivative timing fractions: `u 0.1776`, `v 0.7302`, `w 0.6741`, `a 0.1646`, `b 0.2562`, `c 0.2938`, `d 0.3694`, `e 0.4597`.

## Final V8 Freeze Cleanup, 2026-09-01

This final cleanup did not add or remove features and did not change any broad feature family. Lomb-HRV now receives `config.min_hrv_coverage_seconds` from `nonlinear_hrv_features()` instead of using an independent hard-coded coverage threshold. The three generic SBP-RR correlation definitions now explicitly describe the selected sufficiently covered continuous valid ABP population used by the implementation. The external feature version remains `v8`; this revision-6 cleanup state was superseded by the final revision-7 qualifying-run fix below, and feature count remains `194`.

Final cleanup validation:

- Compileall passed for `waveform_baselines/wf_features_v8`, `scripts/build_waveform_features_v8.py`, `scripts/audit_waveform_features_v8.py`, and `tests/test_waveform_feature_v8.py`.
- Focused V8 tests: `64` run, `63` passed, `1` optional pyPPG hook skipped.
- Full unittest discovery: `260` run, `259` passed, `1` optional pyPPG hook skipped.
- Schema checks: `194` feature names, `194` unique names, all produced feature keys registered, `_put_row()` rejects unknown keys, metadata includes ordered names, revision/hash, channel order, sampling rate, feature-window duration, rolling-history duration, and extraction config.
- Config-validation matrix rejected invalid channel order, fixed-duration/schema mismatches, invalid width levels, rhythm-history overflow, non-positive counts, invalid fractions/frequency ranges/lag step/tau bounds, negative score separations, and insufficient Savitzky-Golay polynomial order/window for JPG-backed derivative fiducials.

Final cleanup smoke caches:

- Default: `/tmp/waveform_feature_cache_smoke_v8_final_cleanup/v8/smoke_local_v8_final_cleanup`, shape `(2, 20, 194)`, v7-prefix alignment passed.
- Audit-enabled: `/tmp/waveform_feature_cache_smoke_v8_final_cleanup/v8/smoke_local_v8_final_cleanup_audit_enabled`, shape `(2, 20, 194)`, v7-prefix alignment passed.
- Default audit JSON: `/tmp/waveform_feature_cache_smoke_v8_final_cleanup/smoke_local_v8_final_cleanup_audit.json`.
- Audit-enabled JSON: `/tmp/waveform_feature_cache_smoke_v8_final_cleanup/smoke_local_v8_final_cleanup_audit_enabled.json`.

Final cleanup smoke QC summary:

- Default smoke finite features: `130`; features with `>95%` missingness: `64`, primarily disabled experimental/synchronization-gated columns.
- Audit-enabled finite features: `183`; features with `>95%` missingness: `13`.
- Near-zero variance flags: default `4` (`resp_pause_count`, `resp_longest_pause_s`, `ecg_derived_resp_rate_5m`, `resp_pause_burden_5m`); audit-enabled `6` with ABP tau columns added.
- `<1%` unique finite values: `0`.
- High absolute v8-v8 correlation flags on the tiny smoke set: default `52`, audit-enabled `62`; no features were removed from these smoke-size flags.
- Median longest-run durations over the final 5-minute histories of the 2 smoke anchors: RR `255.096 s`, ABP `0.0 s` under the ABP nonlinear coverage-qualified selector for this exact final-window audit, PLETH `0.0 s`, RESP `293.344 s`. Coverage-satisfied fractions: HRV `0.5`, ABP nonlinear `0.0`, PLETH morphology `0.0`, RESP `1.0`.
- Audit-enabled landmark validity-summary medians: PLETH notch `0.2301`, PLETH diastolic peak `0.0592`, derivative complete set `0.2426`, VPG/APG timing fractions `u 0.1776`, `v 0.7302`, `w 0.6741`, `a 0.1646`, `b 0.2562`, `c 0.2938`, `d 0.3694`, `e 0.4597`; ABP notch `0.00685`, ABP diastolic peak `0.0`, ABP tau `0.0`.
- Severe-pressure smoke audit over the same final 2 anchors found `723` detected ABP beats, `723` valid beats, and no beats with `SBP < 50`, `DBP < 20`, `MAP < 40`, or `SBP > 220`. Bounds were intentionally unchanged. Larger extraction/QC must still collect pre-filter otherwise-morphologically-valid extreme-pressure cases, affected records/patients, distributions, and waveform examples before drawing conclusions about severe-hypotension censoring.

Final cleanup runtime benchmark:

```bash
/usr/bin/time -p /gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/build_waveform_features_v8.py \
  --cache-root /tmp/waveform_feature_cache_smoke_v8_final_cleanup_timed \
  --output-name smoke_local_v8_final_cleanup_timed \
  --max-samples 1 \
  --overwrite \
  --allow-prefix-v7-alignment
```

Result: `real 31.72`, `user 26.60`, `sys 1.09` seconds for one 20-minute sample including cache writes and alignment validation.

This revision-6 cleanup state was superseded by the final revision-7 qualifying-run fix below. The next step is full V8 extraction and QC validation, including the larger severe-hypotension audit described above.

## Final V8 Qualifying-Run Freeze Fix, 2026-09-01

This final targeted pass fixed count-plus-coverage run selection without adding or removing features. Features that require both an event count and continuous temporal coverage now select the longest run among candidates that already satisfy both requirements, using the existing deterministic ranking of duration, then count, then latest end time. This avoids returning missing from a longer sparse run while ignoring a slightly shorter qualifying run.

Updated call sites:

- `_longest_valid_rr_run()` accepts `min_count` and `min_duration_s`; nonlinear HRV and `ecg_rr_irregularity_index_5m` pass the configured count and RR coverage requirements directly.
- `abp_nonlinear_dynamics_features()` selects qualifying ABP runs for SampEn features and uses `minimum_abp_successive_pairs + 1` beats for successive-difference summaries.
- `morphology_dynamics_features()` selects qualifying ABP area/dP-dt and PLETH width runs up front.
- `pleth_experimental_morphology_dynamics_features()` selects qualifying reflection-index and APG b/a runs up front.
- `derived_respiration_features()`, `respiratory_pattern_features()`, and `respiratory_rate_variability_features()` pass RESP cycle count and coverage requirements directly into RESP run selection where applicable.

Definition cleanup:

- `hrv_poincare_sd1_sd2_ratio_5m` now documents configured continuous RR temporal coverage rather than a hard-coded `240 s` requirement.
- `baroreflex_sequence_fraction_5m` now documents the pairable unique monotonic SBP-sequence denominator.

Final schema identity after this fix:

```text
feature_version = v8
feature_count = 194
feature_schema_revision = 7
feature_schema_hash = a72aeb3b2a2942a899851cffab6d43228e939a82653514907b4fe8748cae229b
```

Final validation after this pass:

- Compileall passed for `waveform_baselines/wf_features_v8` and `tests/test_waveform_feature_v8.py`.
- Focused V8 tests: `67` run, `66` passed, `1` optional pyPPG hook skipped.
- Full unittest discovery: `264` run, `263` passed, `1` optional pyPPG hook skipped.
- Schema checks: `194` feature names, all unique; all emitted keys registered in `FEATURE_DEFINITIONS_V8`; enabled/default, experimental, quality, and synchronization metadata counts unchanged (`135`, `41`, `10`, `13`).
- Revision-7 smoke cache: `/tmp/waveform_feature_cache_smoke_v8_revision7/v8/smoke_local_v8_revision7`, shape `(2, 20, 194)`, v7-prefix alignment passed.
- Revision-7 smoke audit JSON: `/tmp/waveform_feature_cache_smoke_v8_revision7/smoke_local_v8_revision7_audit.json`. Default smoke finite features: `130`; `>95%` missing: `64`; near-zero variance flags: `4`; `<1%` unique finite values: `0`; high absolute v8-v8 correlation flags on this tiny smoke: `52`.

V8 is frozen at schema revision 7. Stop modifying the feature extractor; next work is representative real-data V8 extraction/QC audit, then full extraction.

## V8 Runtime Optimization Pass, 2026-09-01

A first correctness-preserving runtime pass optimized expensive helper internals without changing the v8 schema or removing feature families. The feature count remains `194`, external version remains `v8`, and schema revision/hash remain revision `7` / `a72aeb3b2a2942a899851cffab6d43228e939a82653514907b4fe8748cae229b` because ordered feature definitions did not change.

Implemented optimizations:

- `_template_distances()` now computes exact leave-one-out medians from per-coordinate sorted order statistics and vectorizes correlation/distance calculations instead of repeatedly calling `np.median(np.delete(...))` inside the beat loop.
- `sample_entropy()` now uses exact Chebyshev-radius pair counting with `scipy.spatial.cKDTree.query_pairs()` over sliding-window embeddings, preserving the previous unordered-pair SampEn convention.
- `detect_pulses(..., kind="pleth")` no longer computes derivative morphology, VPG, APG, JPG, or derivative segment object arrays unless `enable_pleth_derivative_fiducials=True`. These derivative arrays are only consumed by disabled-by-default derivative fiducial paths.
- `baroreflex_features()` now computes `_paired_sbp_rr_for_lag()` once for lags `0`, `1`, and `2`, then reuses the lag cache for lag correlations and sequence detection.

Validation:

- Added exact reference tests for `_template_distances()` covering 3-beat, odd/even beat-count, tied-value, random/noisy, and larger-beat cases.
- Added exact reference tests for `sample_entropy()` across deterministic, noisy, NaN-containing, and BP-like sequences.
- Added a regression test confirming disabled default PLETH derivative fiducial mode does not call `_savgol_derivative_by_finite_run()` and does not materialize derivative segment arrays.
- Focused V8 tests passed: `70` run, `69` passed, `1` optional pyPPG hook skipped.
- Full unittest discovery passed: `267` run, `266` passed, `1` optional pyPPG hook skipped.
- Compileall passed for `waveform_baselines/wf_features_v8`, `scripts/build_waveform_features_v8.py`, `scripts/audit_waveform_features_v8.py`, and `tests/test_waveform_feature_v8.py`.

Representative one-window timing after this pass, using the same first full-data anchor profiling method as the pre-optimization profile:

| Component | Before | After | Notes |
|---|---:|---:|---|
| Total `extract_v8_feature_sequence()` | `24.125 s` | `19.501 s` | `(20, 194)` output, finite count `2234` after pass |
| `detect_pulses()` | `11.089 s` | `10.879 s` | Still `72` calls; structural detector reuse remains the main target |
| `detect_ecg_events()` | `8.175 s` | `6.567 s` | Still `36` calls; speedup mostly from faster template work inside ECG morphology |
| `_template_distances()` | `4.231 s` | `0.967 s` | Exact leave-one-out definition preserved by tests |
| `rhythm_features()` | `1.897 s` | `0.424 s` | Benefits from faster QRS template distance path |
| `sample_entropy()` | `0.923 s` | `0.038 s` | Exact SampEn convention preserved by tests |
| `abp_nonlinear_dynamics_features()` | `0.780 s` | `0.036 s` | Mostly SampEn speedup |
| `baroreflex_features()` | `0.405 s` | `0.199 s` | Lag pairing reused |

Remaining bottleneck: redundant window-local detection is still present in `extract_v8_feature_sequence()`. The optimized profile still makes `36` ECG detector calls and `72` ABP/PLETH pulse detector calls per 20-minute input. The next major optimization should preserve the current extractor as a reference and introduce shared preprocessing/event caches or guarded event slicing so overlapping one-minute and rolling five-minute windows stop repeating expensive filtering and detection work.

## V8 Cached-Detection Optimization Pass, 2026-09-02

A second runtime pass addressed the detector-dominated profile while preserving the causal reference extractor as the default production path. This pass did not change the ordered 194-feature schema.

Implemented changes:

- Added `extract_v8_feature_sequence_reference()` for the previous detector-per-window implementation.
- Added `extract_v8_feature_sequence_cached_global()` as an explicit optimized path that detects ECG, ABP pulses, PLETH pulses, and RESP cycles once over the full 20-minute input, then constructs one-minute and rolling 5-minute event views with `searchsorted()`-based slicing.
- Added `slice_ecg_events()`, `slice_pulse_events()`, and `slice_resp_events()` helpers. Cached events stay in full-input coordinates and are rebased only for the existing local-window feature functions.
- Kept `extract_v8_feature_sequence()` mapped to the causal reference path for now because the global detector can allow later within-input waveform changes to alter earlier token detector/filter context. The cached-global path is available intentionally through `scripts/build_waveform_features_v8.py --use-global-event-cache`.
- Replaced repeated `_segment_run_id()` finite-run scans inside V8 detector paths with cached sample-level run-ID maps. This preserves output semantics for the reference path while removing thousands of repeated `finite_runs()` scans.
- Added focused coverage that the cached-global extractor makes exactly one ECG detector call, two pulse detector calls, and one RESP detector call on a 20-minute input.

Representative one-window timing, first full-data anchor:

| Metric | Old V8 | Helper-optimized V8 | Run-map causal/reference V8 | Cached-global V8 |
|---|---:|---:|---:|---:|
| Total time/sample | `24.125 s` | `19.501 s` | `4.807 s` | `2.178 s` |
| Speedup vs old V8 | `1.00x` | `1.24x` | `5.02x` | `11.08x` |
| Speedup vs helper-optimized V8 | `0.81x` | `1.00x` | `4.06x` | `8.95x` |
| ECG detector calls | not measured | `36` | `36` | `1` |
| Pulse detector calls | not measured | `72` | `72` | `2` |
| RESP detector calls | not measured | `36` | `36` | `1` |
| ECG detector total time | not measured | `6.567 s` | `1.992 s` | `0.386 s` |
| Pulse detector total time | not measured | `10.879 s` | `0.768 s` | `0.149 s` |
| RESP detector total time | not measured | `0.078 s` | `0.076 s` | `0.014 s` |
| `_template_distances()` total time | `4.231 s` | `0.967 s` | `0.958 s` | `0.268 s` |
| `sample_entropy()` total time | `0.923 s` | `0.038 s` | `0.038 s` | `0.038 s` |

One-pass detector microprofile after cached run-ID maps:

| Component | Runtime | Events |
|---|---:|---:|
| full-input ABP `detect_pulses()` | `0.074 s` | `1308` beats |
| full-input PLETH `detect_pulses()` | `0.084 s` | `1612` pulses |
| full-input `detect_ecg_events()` | `0.404 s` | `1347` peaks |

Cached-global feature agreement against the causal reference on the same anchor:

| Metric | Value |
|---|---:|
| Reference finite count | `2234` |
| Cached-global finite count | `2190` |
| Finite-mask agreement | `0.9866` |
| Both-finite compared values | `2186` |
| MAE | `0.3343` |
| Median absolute error | `0.0` |
| p95 absolute error | `0.5835` |
| Max absolute error | `49.7645` |

Largest mask differences were in rolling QRS morphology/template features (`ecg_qrs_morphology_outlier_fraction_5m`, `ecg_qrs_template_distance_p95_5m`, and `ecg_ectopic_like_beat_fraction_5m`, each differing on `0.8` of tokens). Largest value differences were respiratory-variation features, especially PLETH respiratory area/amplitude/width variation. These are plausible consequences of continuous detector/filter context and boundary semantics, but they require multi-anchor QC before making cached-global the default production extractor.

Validation after this pass:

- Focused V8 tests: `71` run, `70` passed, `1` optional pyPPG hook skipped.
- Full unittest discovery: `268` run, `267` passed, `1` optional pyPPG hook skipped.
- Compileall passed for `waveform_baselines/wf_features_v8`, `scripts/build_waveform_features_v8.py`, `scripts/audit_waveform_features_v8.py`, and `tests/test_waveform_feature_v8.py`.

Full-data overlap analysis:

| Metric | Value |
|---|---:|
| Extraction samples | `1969515` |
| Unique source segments | `21833` |
| Unique patients | `1758` |
| Median anchor stride | `150 s` |
| Requested sample-minutes | `39390300` |
| Approximate unique recording-minutes covered | `5305865` |
| Approximate duplication factor | `7.42x` |

The dataset is not one-minute stride; adjacent anchors are spaced by `150 s`. Even so, 20-minute windows overlap heavily. A recording-level feature-timeline architecture could still remove about `7.4x` duplicated waveform coverage before accounting for detector/cache reuse, and remains the preferred long-term architecture for full-dataset throughput.

## V8 Post-Detection Optimization Pass, 2026-09-02

A third runtime pass targeted the now-dominant post-detection costs after global detector caching reduced detector calls to one full-input ECG pass, two pulse passes, and one RESP pass. The ordered 194-feature schema remains unchanged and `extract_v8_feature_sequence()` still points to the causal/reference path by default.

Implemented changes:

- Added `include_qrs_template` to `detect_ecg_events()` and set `include_qrs_template=False` for `extract_v8_feature_sequence_cached_global()`. The full-input QRS template correlations were not consumed by cached-global feature aggregation; rolling QRS morphology still calculates its window-local template statistics.
- Refactored RESP pause detection so `detect_resp_pause_durations()` can consume cached `filtered` RESP and `ordered_extrema` from `detect_resp_cycles()`. Minute burden and rolling respiratory-pattern features now reuse the already-created RESP cache instead of filtering and detecting extrema again.
- Replaced per-beat `scipy.stats.skew()` / `scipy.stats.kurtosis()` calls in PLETH shape features with local formulas matching SciPy's bias-corrected Fisher definitions to numerical precision. This avoids thousands of expensive SciPy decorator/wrapper calls without changing the feature definition.
- Vectorized the main SBP/RR search in `_paired_sbp_rr_for_lag()` while preserving the transition-validity checks used by baroreflex sequence features.
- Benchmarked a one-time full-input PLETH shape primitive cache. It was exactly value-equivalent on the representative anchor but not a net latency win for per-20-minute cached-global extraction, so it remains available behind `include_pleth_shape_primitives` but is not enabled by the cached-global extractor.

Representative post-change timing on the same first full-data anchor:

| Version | Runtime/sample | Speedup vs original | Speedup vs cached-global baseline |
|---|---:|---:|---:|
| Original | `24.125 s` | `1.00x` | `0.09x` |
| Helper-optimized | `19.501 s` | `1.24x` | `0.11x` |
| Causal/reference before this pass | `4.807 s` | `5.02x` | `0.45x` |
| Cached-global baseline | `2.178 s` | `11.08x` | `1.00x` |
| New optimized causal/reference | `4.175 s` | `5.78x` | `0.52x` |
| New optimized cached-global | `1.476 s` | `16.35x` | `1.48x` |

The cached-global target of `<1.5 s/sample` was met on this representative anchor. Cached-global agreement against the causal/reference path was unchanged from the previous pass because the changes either preserve formulas exactly or remove unused full-input QRS template work:

| Metric | Value |
|---|---:|
| Reference finite count | `2234` |
| Cached-global finite count | `2190` |
| Finite-mask agreement | `0.9866` |
| Both-finite compared values | `2186` |
| MAE | `0.3343` |
| Median absolute error | `0.0` |
| p95 absolute error | `0.5835` |
| Max absolute error | `49.7645` |

Post-change cached-global component breakdown on the same anchor:

| Component | Calls | Time | % total |
|---|---:|---:|---:|
| `detect_ecg_events()` | `1` | `0.323 s` | `21.6%` |
| `pleth_shape_features()` | `20` | `0.287 s` | `19.1%` |
| `_template_distances()` | `56` | `0.180 s` | `12.0%` |
| `detect_pulses()` | `2` | `0.149 s` | `9.9%` |
| `nonlinear_hrv_features()` | `16` | `0.109 s` | `7.3%` |
| `baroreflex_features()` | `16` | `0.103 s` | `6.9%` |
| `abp_lability_features()` | `20` | `0.100 s` | `6.7%` |
| `coupling_features()` | `20` | `0.076 s` | `5.1%` |
| `rhythm_features()` | `16` | `0.071 s` | `4.7%` |
| `derived_respiration_features()` | `16` | `0.069 s` | `4.6%` |
| `waveform_quality_features()` | `20` | `0.047 s` | `3.1%` |
| `respiratory_pattern_features()` | `16` | `0.040 s` | `2.7%` |
| `sample_entropy()` | `95` | `0.038 s` | `2.5%` |
| `abp_nonlinear_dynamics_features()` | `16` | `0.035 s` | `2.4%` |
| `respiratory_rate_variability_features()` | `16` | `0.028 s` | `1.8%` |
| RESP pause detection | `36` | `0.015 s` | `1.0%` |

Validation after this pass:

- Verified the local skew/kurtosis formulas match `scipy.stats.skew(..., bias=False)` and `scipy.stats.kurtosis(..., fisher=True, bias=False)` to tight numerical tolerance on random finite vectors.
- Compileall passed for `waveform_baselines/wf_features_v8`, `scripts/build_waveform_features_v8.py`, `scripts/audit_waveform_features_v8.py`, and `tests/test_waveform_feature_v8.py`.
- Focused V8 tests: `71` run, `70` passed, `1` optional pyPPG hook skipped.
- Full unittest discovery: `268` run, `267` passed, `1` optional pyPPG hook skipped.

Feature-row phase and deduplication audit for the full-data anchors:

| Metric | Value |
|---|---:|
| Extraction samples | `1969515` |
| Unique source segments | `21833` |
| Distinct feature-grid phases | `2` |
| Phase `0.0 s` count | `989882` |
| Phase `30.0 s` count | `979633` |
| Requested feature rows | `39390300` |
| Unique `(segment_id, feature_window_start_sample)` rows | `10502565` |
| Feature-row duplication factor | `3.75x` |

The earlier waveform-coverage duplication estimate remains about `7.42x`, but exact feature-row reuse is lower because the `150 s` anchor stride alternates between `0 s` and `30 s` minute-grid phases. A recording-level V8 architecture should therefore derive the required phases from anchors and compute unique feature rows per `(recording/segment, phase, feature_window_start)` rather than assuming a single universal minute grid.

## V8 Exact Micro-Optimization and Semantic Deduplication Audit, 2026-09-02

A fourth V8 runtime pass focused on exact optimizations after cached-global extraction reached the `1.476 s/sample` range. The feature schema remains unchanged at `194` columns, and `extract_v8_feature_sequence()` still uses the causal/reference implementation by default.

Implemented exact changes:

- Replaced `_center_scale_segment()`'s repeated `np.median()` / `np.percentile()` calls with a single sorted 1D pass using NumPy's default linear percentile definition. A direct randomized check confirmed equivalence to the previous NumPy calculation to tight tolerance.
- Optimized the `rhythm_features()` local RR baseline calculation by masking only the small local neighborhood instead of constructing a full-length boolean array for every RR interval.
- Added pulse raw observability metadata: `_raw_run_id_by_sample`, `_raw_bad_prefix`, `_raw_sum_prefix`, and `_raw_sum_sq_prefix`. `_observable_forward_pairs()` and `pulse_deficit_features()` now use O(1) interval finite/run/std checks instead of repeated `_segment_run_id(start, end - 1, np.isfinite(raw))`, `np.isfinite(raw[start:end]).all()`, and `np.std(raw[start:end])`.
- Preserved cached-global run-ID coordinate consistency by slicing the full-input raw run-ID map into pulse event windows, so interval run IDs and pulse event `run_id` values use the same numbering. Added a regression test covering a selected window after an earlier finite-data gap.
- Added `build_cross_signal_pair_cache()` and wired enabled timing/pulse-deficit features to share ECG-to-ABP and ECG-to-PLETH pair tables. Pulse-deficit source matching now uses a boolean mask instead of Python `set` membership.
- Vectorized `coupling_features()` RESP lag correlation across the configured lag grid while preserving NaN handling, minimum-count behavior, zero-variance checks, and lag resolution.

Representative timing on the same first full-data anchor:

| Version | Runtime/sample | Speedup vs original | Accuracy notes |
|---|---:|---:|---|
| Original | `24.125 s` | `1.00x` | reference old |
| Previous cached-global | `2.178 s` | `11.08x` | finite-mask agreement `0.9866` vs causal/reference |
| Current cached-global before this pass | `1.476 s` | `16.35x` | finite-mask agreement `0.9866` vs causal/reference |
| New optimized cached-global | `1.192 s` | `20.23x` | finite-mask agreement unchanged at `0.9866` |
| Current causal/reference before this pass | `4.175 s` | `5.78x` | gold standard |
| New optimized causal/reference | `3.305 s` | `7.30x` | gold standard |
| Deduplicated causal/reference effective throughput | not implemented | not measured | semantic component duplication audited below |

Cached-global agreement against the causal/reference path on the representative anchor was unchanged after the exact optimizations:

| Metric | Value |
|---|---:|
| Reference finite count | `2234` |
| Cached-global finite count | `2190` |
| Finite-mask agreement | `0.9866` |
| Both-finite compared values | `2186` |
| MAE | `0.3343` |
| Median absolute error | `0.0` |
| p95 absolute error | `0.5835` |
| Max absolute error | `49.7645` |

Post-change cached-global component timers on the same anchor:

| Component | Calls | Time | % total |
|---|---:|---:|---:|
| `detect_ecg_events()` | `1` | `0.316 s` | `26.2%` |
| `detect_pulses()` | `2` | `0.151 s` | `12.5%` |
| `pleth_shape_features()` | `20` | `0.132 s` | `11.0%` |
| `nonlinear_hrv_features()` | `16` | `0.109 s` | `9.0%` |
| `baroreflex_features()` | `16` | `0.103 s` | `8.5%` |
| `derived_respiration_features()` | `16` | `0.069 s` | `5.7%` |
| `rhythm_features()` | `16` | `0.061 s` | `5.0%` |
| `waveform_quality_features()` | `20` | `0.046 s` | `3.9%` |
| `_template_distances()` | `56` | `0.045 s` | `3.7%` |
| `respiratory_pattern_features()` | `16` | `0.040 s` | `3.3%` |
| `sample_entropy()` | `95` | `0.037 s` | `3.1%` |
| `abp_lability_features()` | `20` | `0.036 s` | `2.9%` |
| `respiratory_rate_variability_features()` | `16` | `0.028 s` | `2.3%` |
| `_resp_variation_features()` | `40` | `0.024 s` | `2.0%` |
| `slice_pulse_events()` | `72` | `0.018 s` | `1.5%` |
| RESP pause detection | `36` | `0.015 s` | `1.2%` |
| `coupling_features()` | `20` | `0.007 s` | `0.6%` |

Final cached-global cProfile after this pass still shows profiling overhead (`1.963 s` profiled vs `1.192 s` unprofiled median), but the top self-time functions are now diffuse: `lombscargle` (`0.170 s`), NumPy reductions (`0.133 s`), `finite_runs` (`0.118 s`), `find_local_peaks` (`0.099 s`), pulse extraction (`0.036 s`), `_paired_sbp_rr_for_lag()` (`0.036 s`), local width/gradient operations, and rhythm scanning. There is no remaining single non-detector pure-Python hotspot comparable to the original redundant detector/template costs.

Validation after this pass:

- Focused V8 tests: `72` run, `71` passed, `1` optional pyPPG hook skipped.
- Full unittest discovery: `269` run, `268` passed, `1` skipped.
- Compileall passed for `waveform_baselines/wf_features_v8` and `tests/test_waveform_feature_v8.py` during focused validation.

Semantic row/component deduplication audit for exact causal/reference extraction:

| Metric | Value |
|---|---:|
| Extraction samples | `1969515` |
| Unique source segments | `21833` |
| Requested minute components | `39390300` |
| Unique minute components | `10502565` |
| Minute duplication factor | `3.75x` |
| Requested 5-minute history components | `31512240` |
| Unique 5-minute history components | `10327901` |
| History duplication factor | `3.05x` |
| Requested component units | `70902540` |
| Unique component units | `20830466` |
| Component duplication factor | `3.40x` |
| Requested semantic rows | `39390300` |
| Unique semantic row variants keyed by history availability | `18205961` |
| Semantic row duplication factor | `2.16x` |

The component-level duplication factor is the more relevant target for an exact causal implementation because one-minute and 5-minute-history feature components can be cached separately. Using the current representative causal/reference runtime as a rough upper-bound model gives an estimated `~1.75 s/requested sample` equivalent after component deduplication, but this has not been implemented or benchmarked. The next architecture should refactor the reference extractor into shared minute-component and history-component functions, verify exact parity, and then process anchors grouped by source segment/recording so each unique component is computed once.


## V8 Exact Component Cache Prototype, 2026-09-02

The causal/reference extractor has been refactored internally into explicit one-minute and 5-minute history components without changing feature definitions:

- `extract_v8_minute_component()` performs the exact one-minute detector and feature-family work previously embedded in the reference sequence loop.
- `extract_v8_history_component()` performs the exact rolling 5-minute detector and history-feature work previously embedded in the reference sequence loop.
- `extract_v8_feature_sequence_reference()` now assembles rows from those components and remains the causal gold-standard path.
- `extract_v8_feature_sequence_reference_cached_components()` adds an exact component cache keyed by segment/config and absolute sample interval.

The component cache preserves the existing warm-up behavior: rows 0-3 in each 20-row sample still receive one-minute features only, even when earlier waveform data exists in the source recording. Rows 4-19 receive the cached history component for the exact `[row_end - 300 s, row_end]` interval.

An opt-in build-script mode was added:

```bash
scripts/build_waveform_features_v8.py --use-reference-component-cache
```

This mode is mutually exclusive with `--use-global-event-cache`. It keeps causal/reference detector semantics and records component-cache stats in `metadata.json` plus progress/final JSON logs. The current implementation reuses components across anchors processed by one script process/shard; it does not yet group/load a whole source recording once, so it is a safe intermediate step rather than the final recording-level architecture.

Focused real-data component-cache benchmark over 8 overlapping anchors from one source segment (`p043738/3251946_0031`, anchors at 600-1650 s):

| Metric | Plain reference | Component-cached reference |
|---|---:|---:|
| Wall time | `30.428 s` | `13.806 s` |
| Effective speedup | `1.00x` | `2.20x` |
| Effective seconds/requested sample | `3.804 s` | `1.726 s` |
| Mask equality vs plain reference | baseline | `true` |
| Strict value equality vs plain reference | baseline | `true` |
| Max abs diff on both-finite values | baseline | `0.0` |
| Minute components requested/computed/hits | `160/160/0` | `160/70/90` |
| History components requested/computed/hits | `128/128/0` | `128/62/66` |

CLI smoke for the new flag:

```bash
/usr/bin/time -p /gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/build_waveform_features_v8.py \
  --cache-root /tmp/waveform_feature_cache_component_smoke \
  --output-name smoke_v8_component_cache_cli \
  --max-samples 3 \
  --overwrite \
  --allow-prefix-v7-alignment \
  --use-reference-component-cache
```

Result: cache `/tmp/waveform_feature_cache_component_smoke/v8/smoke_v8_component_cache_cli`, shape `(3, 20, 194)`, v7-prefix alignment passed, wall time `10.94 s`. Component-cache summary: minute requested/computed/hits `60/45/15`, history requested/computed/hits `48/37/11`.

Validation after the component-cache implementation:

- Compileall passed for `waveform_baselines/wf_features_v8`, `scripts/build_waveform_features_v8.py`, `scripts/audit_waveform_features_v8.py`, and `tests/test_waveform_feature_v8.py`.
- Focused V8 tests: `74` run, `73` passed, `1` optional pyPPG hook skipped.
- Full unittest discovery: `271` run, `270` passed, `1` optional pyPPG hook skipped.

New tests cover reference-vs-component-cache parity, shared 0-second/30-second phase reuse behavior, first-four-row history warm-up preservation, and pulse-deficit run-ID consistency after earlier finite gaps.

Remaining dataset-level work:

- Move from process-local component caching to segment/recording-grouped extraction so each source waveform is loaded once per worker.
- Sort anchors by source segment and time, compute/reuse unique one-minute and 5-minute components per segment, then assemble the original 20-row samples.
- Parallelize by segment/recording rather than by feature row.
- Re-benchmark on representative multi-record shards and estimate full-dataset CPU-hours from observed component hit rates.


## V8 Lean Components and Segment-Planned Reuse, 2026-09-02

A follow-up exact optimization pass made the component implementation leaner and added a segment-planned execution path.

Exact detector-output pruning:

- `detect_ecg_events()` now accepts `include_morphology`; core mode returns only R peaks, run IDs, RR intervals, RR times, and RR validity/run continuity arrays.
- `extract_v8_minute_component()` uses ECG core mode plus `include_qrs_template=False`, because one-minute features do not consume QRS morphology, R-amplitude derived respiration, or QRS template correlations.
- `extract_v8_history_component()` still keeps ECG morphology for rolling rhythm/QRS features, but disables unused detector-level QRS template correlations; `rhythm_features()` continues to calculate its exact window-specific template statistics.
- `detect_pulses()` now accepts `include_segments`, `include_observability`, and `include_raw_signal` output flags. History ABP omits raw/object segment arrays and timing observability prefixes; history PLETH does the same unless PLETH fiducial/derivative history features are enabled.
- One-minute pulse detection still keeps beat segments for PLETH shape, ABP lability/template morphology, waveform quality, and optional morphology/fiducial families. It only omits raw-signal observability metadata when timing and pulse-deficit features are disabled.

Segment-planned exact extraction:

- `extract_v8_reference_component_plan_for_segment()` takes a loaded source segment waveform plus all input-start samples for that segment, enumerates unique one-minute and 5-minute history intervals, computes each unique component once, finalizes components to dense `float32` values plus bool masks, and assembles the requested `(N, 20, 194)` sequences.
- `scripts/build_waveform_features_v8.py --use-reference-segment-component-plan` exposes this mode. It is mutually exclusive with `--use-global-event-cache` and `--use-reference-component-cache`.
- Segment-planned mode uses segment-aware sharding so one source segment is not split across workers within that mode. Current balancing is based on contiguous anchor-count targets; future production tuning can replace this with the measured unique-component work estimate.

Representative lean component timings over real intervals from `p043738/3251946_0031`:

| Component | Mean | Median | p95 | N |
|---|---:|---:|---:|---:|
| 1-minute component | `0.0240 s` | `0.0240 s` | `0.0245 s` | `30` |
| 5-minute history component | `0.1190 s` | `0.1190 s` | `0.1214 s` | `20` |

The weighted full-data compute-speedup estimate should use those separate costs rather than treating minute and history components as equal. With the existing full-data component counts, this gives an estimated exact component-compute speedup of `3.17x`:

```text
uncached = 39.39M * T_minute + 31.51M * T_history ~= 1305 CPU-hours
cached   = 10.50M * T_minute + 10.33M * T_history ~= 412 CPU-hours
speedup  ~= 3.17x
```

The CPU-hour estimate excludes output serialization, scheduler overhead, and possible I/O bottlenecks, but it is a better compute model than the unweighted `3.40x` component-count ratio because history components cost about `5x` a minute component.

Same 8-anchor overlap benchmark after lean modes:

| Path | Runtime | Speedup vs plain reference | Agreement |
|---|---:|---:|---|
| Plain reference, 8 anchors | `18.774 s` | `1.00x` | baseline |
| Dynamic component cache | `8.917 s` | `2.11x` | exact, max diff `0.0` |
| Segment-planned component extraction | `8.908 s` | `2.11x` | exact, max diff `0.0` |

The earlier 8-anchor benchmark before lean detector-output pruning was `30.428 s` plain reference and `13.806 s` component-cached reference. The new lean implementation is faster while retaining strict output equality.

Longer 50-anchor same-segment benchmark:

| Metric | Value |
|---|---:|
| Segment | `p043738/3251946_0031` |
| Anchors | `50` |
| Dynamic component-cache wall time | `37.864 s` |
| Segment-planned wall time | `37.785 s` |
| Segment-planned effective sec/sample | `0.756 s` |
| Minute requested/computed/hits | `1000 / 280 / 720` |
| Minute reuse factor | `3.57x` |
| History requested/computed/hits | `800 / 272 / 528` |
| History reuse factor | `2.94x` |
| Dense minute/history component cache bytes | `271600 / 263840` |

At this 50-anchor scale, reuse factors are close to the full-data audit (`3.75x` minute and `3.05x` history). Segment planning is not materially faster than dynamic caching for a single segment because both compute the same unique components, but it avoids per-anchor 20-minute extraction, stores compact dense component rows, reports deterministic unique work, and enables segment-aware SLURM sharding.

CLI smoke for segment-planned extraction:

```bash
/usr/bin/time -p /gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python scripts/build_waveform_features_v8.py \
  --cache-root /tmp/waveform_feature_cache_segment_plan_smoke \
  --output-name smoke_v8_segment_plan_cli \
  --max-samples 3 \
  --overwrite \
  --allow-prefix-v7-alignment \
  --use-reference-segment-component-plan
```

Result: cache `/tmp/waveform_feature_cache_segment_plan_smoke/v8/smoke_v8_segment_plan_cli`, shape `(3, 20, 194)`, v7-prefix alignment passed, wall time `10.14 s`. Component stats were minute `60/45/15` requested/computed/hits and history `48/37/11` requested/computed/hits.

Cached-global remains opt-in. A three-run spot timing on the representative first full-data anchor after this pass was `1.086`, `1.089`, and `1.090 s` (`1.089 s` median), but this path still has the known nonlocal detector-context semantics and is not the exact causal reference path.

Validation after this pass:

- Compileall passed for `waveform_baselines/wf_features_v8`, `scripts/build_waveform_features_v8.py`, `scripts/audit_waveform_features_v8.py`, and `tests/test_waveform_feature_v8.py`.
- Focused V8 tests: `79` run, `78` passed, `1` optional pyPPG hook skipped.
- Full unittest discovery: `276` run, `275` passed, `1` optional pyPPG hook skipped.

New validation covers ECG core-mode equality for R/RR fields, lean pulse-mode equality for history-required pulse fields, lean minute/history component parity against full-materialization feature dictionaries, segment-planned parity against cached reference assembly, phase handling, warm-up behavior, and finite-gap run-ID consistency.

Remaining work for production-scale exact extraction:

- Replace the current row-based V8 SLURM array with `--use-reference-segment-component-plan` and segment-aware shard sizing.
- Balance shard work by estimated unique-component cost rather than only anchor count.
- Benchmark several multi-segment shards end to end, including write time and peak RSS.
- Consider chunked writes or memmap output only if serialization becomes a measured bottleneck.




## V8 Segment-Planned Submission Preparation, 2026-09-02

The old full V8 revision-7 row-sharded SLURM array was cancelled before submitting the optimized exact segment-planned path:

```text
cancelled: SLURM 26961873, job name full_wf_v8, array 0-2047%128
submitted: SLURM 26968656, job name full_wf_v8_segplan, array 0-127%128
```

The V8 extraction submitter now targets the optimized exact causal segment-planned path:

```bash
sbatch slurm/extract_full_data_vasofree_waveform_features_v8_array.sh
```

Current submitter settings:

```text
job name: full_wf_v8_segplan
array: 0-127%128
shard count: 128
output prefix: full_data_vasopressor_free_waveform_features_v8_segment_plan_shard
mode: --use-reference-segment-component-plan
threads: OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 by default
```

The merge script now expects the matching 128 segment-planned shards and writes a distinct final cache name:

```bash
sbatch slurm/merge_full_data_vasofree_waveform_features_v8.sh
```

```text
shard prefix: full_data_vasopressor_free_waveform_features_v8_segment_plan_shard
output name: full_data_vasopressor_free_waveform_features_v8_segment_plan
shard count: 128
```

Rationale for 128 shards instead of the earlier 2048:

- The segment-planned exact path uses weighted segment-aware sharding, so each shard owns complete source segments and preserves component reuse.
- With the current `~416.7` CPU-hour exact unique-component estimate, `128` concurrent shards imply an ideal average of about `3.3` CPU-hours per shard.
- The `18:00:00` time limit should be sufficient unless shard work is much more imbalanced than predicted.
- Using `128` total shards avoids thousands of small output shard caches while keeping the same maximum concurrency as the old `2048%128` array.

Submitter smoke test:

```bash
SLURM_ARRAY_TASK_ID=0 bash slurm/extract_full_data_vasofree_waveform_features_v8_array.sh \
  --cache-root /tmp/waveform_feature_cache_v8_segplan_submit_smoke \
  --output-name smoke_v8_segplan_submit \
  --max-samples 10 \
  --overwrite \
  --allow-prefix-v7-alignment
```

Result: passed V7 prefix alignment with shape `(10, 20, 194)`. The smoke selected one segment, computed `80` unique minute components and `72` unique history components, and used the segment-planned exact path.

Current status check before handoff:

```text
SLURM 26968656: 128 RUNNING tasks
stdout logs: 128
stderr logs: 128, nonempty stderr logs: 0
completed segment-plan shard directories: 0
completed _SUCCESS files: 0
```

No merge has been submitted yet. Merge only after all `128` shard caches complete successfully.

## V8 Final Profile-Gated Efficiency Pass, 2026-09-02

This pass followed a strict profile gate: no feature definitions, detector semantics, filters, windows, SampEn, Lomb grids, or feature families were changed. The profiler was run on 120 unique one-minute components and 120 unique 5-minute history components from eight real waveform segments:

```text
p000160/3531764_0003
p000188/3317157_0002
p000188/3317157_0004
p000188/3317157_0007
p000188/3641721_0003
p000188/3285727_0007
p000188/3285727_0009
p000188/3285727_0013
```

Initial cProfile component timings:

| Component | N | Mean | Median | p95 |
|---|---:|---:|---:|---:|
| 1-minute | `120` | `0.0591 s` | `0.0601 s` | `0.0663 s` |
| 5-minute history | `120` | `0.2605 s` | `0.2625 s` | `0.2866 s` |

The 1-minute profile was dominated by normal detector and morphology work: pulse detection (`2.025 s`, `28.6%` cumulative), ECG detection (`2.014 s`, `28.4%`), PLETH shape (`1.523 s`, `21.5%`), waveform quality (`0.541 s`, `7.6%`), and ABP lability (`0.394 s`, `5.6%`) over the 120 profiled components. No obvious exact redundancy was retained for the minute component.

The 5-minute history profile showed one actionable redundancy: `_map_detector_peaks_to_morphology_aligned()` used `4.952 s` cumulative time (`15.8%`) and repeated the same local morphology slicing/finite/median/extrema work in two passes. The mapper was refactored to cache the first-pass local finite offsets, dominant-polarity evidence, and positive/negative candidate extrema, then reuse those cached local results for final dominant-polarity alignment. The morphology alignment decision rule is unchanged.

Equivalence and post-change profile:

- A standalone randomized check compared the new mapper against the old two-pass logic over `2000` finite/gappy/tied morphology cases: exact mapped-index equality.
- Post-change 5-minute history cProfile: mean `0.2445 s`, median `0.2454 s`, p95 `0.2704 s` over the same 120 component jobs.
- Mapper cumulative time dropped from `4.952 s` to `3.019 s` over 120 history components.
- Total profiled history-component time dropped from `31.264 s` to `29.336 s` (`~6.2%` under cProfile).

Remaining post-change history hotspots are primarily intrinsic computations: ECG detection (`13.134 s`, `44.8%`), pulse detection (`9.290 s`, `31.7%`), XQRS (`8.706 s`, `29.7%`), `_extract_pulsatile_beats()` (`7.457 s`, `25.4%`), the optimized mapper (`3.019 s`, `10.3%`), rhythm features (`2.468 s`, `8.4%`), Lomb-Scargle (`1.511 s`, `5.2%`), template distances (`1.359 s`, `4.6%`), and DFA (`0.834 s`, `2.8%`). The Lomb, template, and DFA paths were not changed because their remaining costs reflect scientific calculations and did not reveal a simple exact redundancy above the acceptance threshold.

Repeated 50-anchor segment-planned benchmark after the mapper change, on `p043738/3251946_0031`:

| Rep | Wall time | Effective s/sample |
|---|---:|---:|
| 1 | `52.411 s` | `1.048` |
| 2 | `52.635 s` | `1.053` |
| 3 | `53.020 s` | `1.060` |
| 4 | `53.020 s` | `1.060` |
| 5 | `53.078 s` | `1.062` |

Summary: median `53.020 s`, min `52.411 s`, max `53.078 s`, IQR `0.385 s`, median effective `1.060 s/requested sample`. These absolute times are slower than the earlier warm-cache 50-anchor benchmark, so they should not be compared directly as a throughput regression; they are useful for variability and exactness checks on the current node.

The 8-anchor parity check after the retained mapper change remained exact:

| Path | Runtime | Agreement |
|---|---:|---|
| Plain reference | `26.485 s` | baseline |
| Segment-planned causal | `12.535 s` | `mask_equal=True`, strict value equality, max diff `0.0` |

Validation:

- `compileall` passed for the V8 package, V8 build/audit scripts, and V8 tests.
- Focused V8 tests passed: `81` run, `80` passed, `1` optional pyPPG test skipped.
- Full test discovery passed: `278` run, `277` passed, `1` skipped.

Stop decision: no additional feature-computation edits were made. Timing/pulse-deficit fast paths are not active in the default profile because synchronization-gated timing and pulse-deficit features are disabled. Fixed-grid caching was not retained because the visible `linspace` time is mostly per-segment morphology resampling, not reusable frequency-grid creation. Further throughput work should be production operational work: weighted segment-plan shard benchmarks, segment-level process parallelism, input/output timing, peak RSS, and shard-load validation.

## V8 Planner Overhead Cleanup, 2026-09-02

The next exact-efficiency pass focused on implementation overhead around the segment-planned causal path, not on changing detector or feature definitions. The authoritative extraction paths remain:

- causal/reference: exact isolated 1-minute and 5-minute detector context;
- cached-global: faster approximate-context diagnostic path;
- segment-planned causal: exact reference components computed once per unique segment interval and assembled into requested samples.

Code changes:

- Cached the ordered V8 feature schema and `name -> column` mapping once per process.
- Replaced `_put_row()` full-schema scans with sparse finalization over only the produced feature dictionary items, while preserving unknown-feature validation.
- Replaced segment-plan assembly dictionary lookups and nested sample/minute loops with `np.unique(..., return_inverse=True)` component IDs, dense component gathers, chunked history overlays, and `np.copyto(..., where=history_mask)`.
- Changed lean pulse history mode so `include_observability=False` does not allocate a full sample-level run-ID map. Pulse event run IDs are assigned from `finite_runs()` boundaries with `searchsorted()`, preserving the same run numbering.
- Reworked `_resp_variation_features()` to use sorted pulse-time `searchsorted()` slices and one RESP-cycle pass, including ABP SPV and PLETH respiratory amplitude variation.
- Added weighted segment-aware sharding for `--use-reference-segment-component-plan`. Shards are assigned complete segments using a greedy unique-component work estimate, while returned anchors preserve canonical `anchor_id` output order for V7/V8 alignment.

A segment-planner timing breakdown on 50 anchors from `p043738/3251946_0031` showed that planner overhead is now negligible:

| Step | Time |
|---|---:|
| Planning | `0.00026 s` |
| Unique minute compute/finalize | `6.573 s` |
| Unique history compute/finalize | `30.570 s` |
| Vectorized assembly | `0.00029 s` |
| Total measured | `37.143 s` |

The optimized 50-anchor same-segment benchmark was:

| Metric | Value |
|---|---:|
| Wall time | `37.175 s` |
| Effective time/requested sample | `0.743 s` |
| Minute requested/computed/hits | `1000 / 280 / 720` |
| History requested/computed/hits | `800 / 272 / 528` |
| Finite output values | `118951` |

The 8-anchor parity benchmark after the changes remained exact:

| Path | Runtime | Agreement |
|---|---:|---|
| Plain reference | `18.706 s` | baseline |
| Segment-planned causal | `8.970 s` | `mask_equal=True`, strict value equality, max diff `0.0` |

Measured component timing on real intervals:

| Component | Mean | Median | p95 |
|---|---:|---:|---:|
| 1-minute component | `0.02417 s` | `0.02411 s` | `0.02461 s` |
| 5-minute history component | `0.12066 s` | `0.12055 s` | `0.12232 s` |

Using those measured means and the full-data duplication audit:

| Estimate | CPU-hours |
|---|---:|
| Uncached causal component compute | `1320.7` |
| Unique-component cached causal compute | `416.7` |
| Weighted exact compute speedup | `3.17x` |

A one-time active-range float64 conversion was tested because source segments are float32, but it was not retained: converting the 50-anchor active range to float64 first took `37.331 s`, slightly slower than the view-based path (`37.175 s`). Component functions still cast their isolated windows exactly as before.

Validation:

- Focused V8 tests passed: `81` run, `80` passed, `1` optional pyPPG test skipped.
- Full test discovery passed: `278` run, `277` passed, `1` skipped.
- `compileall` passed for the V8 package, build scripts, audit script, and V8 tests.
- Segment-plan CLI smoke with weighted segment sharding and V7 prefix alignment passed for 10 samples: shape `(10, 20, 194)`, minute requested/computed/hits `200 / 80 / 120`, history requested/computed/hits `160 / 72 / 88`.

Interpretation: remaining segment-planned runtime is dominated by unique component computation, especially 5-minute history components. Planning, schema finalization, and assembly are no longer meaningful bottlenecks on the measured workload. Further exact speed work should target only measured >5% component hotspots or production throughput issues such as multi-segment shard balance, process-level segment parallelism, peak RSS, and output I/O.

## V8 Extraction/QC Start, 2026-09-01

After freezing revision 7, a 10-sample QC subset was extracted before full submission.

Default QC cache:

```text
/tmp/waveform_feature_cache_v8_qc/v8/qc10_v8_revision7_default
```

Result: shape `(10, 20, 194)`, v7-prefix alignment passed. Audit JSON: `/tmp/waveform_feature_cache_v8_qc/qc10_v8_revision7_default_audit.json`. Default QC summary: `130` finite features, `64` features with `>95%` missingness, near-zero variance flags `resp_pause_count`, `resp_longest_pause_s`, and `resp_pause_burden_5m`, `<1%` unique finite flags `resp_pause_count` and `resp_longest_pause_s`, and `114` high-correlation flags on this small subset. These are QC flags, not feature-removal decisions.

Audit-enabled QC cache:

```text
/tmp/waveform_feature_cache_v8_qc/v8/qc10_v8_revision7_audit_enabled
```

Result: shape `(10, 20, 194)`, v7-prefix alignment passed. Audit JSON: `/tmp/waveform_feature_cache_v8_qc/qc10_v8_revision7_audit_enabled_audit.json`. Audit-enabled summary: `186` finite features, `8` features with `>95%` missingness, same RESP pause near-zero flags, and `134` high-correlation flags on this small subset. Median validity-summary fractions: PLETH notch `0.1875`, PLETH diastolic peak `0.0328`, derivative complete set `0.3223`, ABP notch `0.0274`, ABP diastolic peak `0.0`, ABP tau `0.0`. Median VPG/APG timing fractions: `u 0.1668`, `v 0.7002`, `w 0.6369`, `a 0.1193`, `b 0.2238`, `c 0.2663`, `d 0.3458`, `e 0.4545`.

Continuous-coverage check over the 10 QC final-history windows found median qualifying longest-run durations RR `149.016 s`, ABP `100.844 s`, PLETH `0.0 s`, RESP `294.684 s`; coverage-satisfied fractions were HRV `0.5`, ABP nonlinear `0.5`, PLETH morphology `0.0`, RESP `1.0`.

Severe-pressure QC over the same 10 anchors found `3582` detected ABP beats and `3550` valid beats. There were `8` extreme low-pressure detections, all from one patient/segment (`p000160/3531764_0003`), all rejected by current pressure bounds: `SBP < 50`: `0`, `DBP < 20`: `8`, `MAP < 40`: `0`, `SBP > 220`: `0`, rejected fraction among extreme cases `1.0`. The extreme cases had normal-high SBP/MAP but near-zero or negative DBP (`DBP` min/median/max `-14.86/-14.86/3.52`, `SBP` median `175.62`, `MAP` median `94.37`), consistent with foot-detection/artifact rather than true severe hypotension. Representative overlay PNGs and `examples.json` are under `/tmp/waveform_feature_cache_v8_qc/extreme_pressure_examples/`. Larger full-extraction QC should repeat this audit before any ABP bound change.

Full default V8 extraction was submitted as a 2048-shard CPU array with concurrency cap 128:

```text
SLURM job: 26961873
Shard prefix: full_data_vasopressor_free_waveform_features_v8_rev7_shard
Cache root: /gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v8
Wrapper: slurm/extract_full_data_vasofree_waveform_features_v8_array.sh
Merge wrapper: slurm/merge_full_data_vasofree_waveform_features_v8.sh
Merge script: scripts/merge_waveform_features_v8_shards.py
```

Early shard logs showed clean startup with `194` features, revision-7 metadata, and about `961-962` samples per shard. Handoff checks through `2026-09-01 17:55 EDT` found `128` running array tasks and the remaining tasks pending on `JobArrayTaskLimit`; representative shard logs `logs/full_wf_v8_26961873_0.out` and `logs/full_wf_v8_26961873_127.out` still show only startup records, no nonempty stderr was observed in checked early logs, and `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v8` did not exist yet because shard caches are written at task completion. The merge should be run only after all array tasks complete successfully.

## Handoff Status - 2026-09-02

Fresh handoff check from repo files, logs, and `squeue -u dk5565`:

- Active full-data extraction: SLURM array `26968656`, job name `full_wf_v8_segplan`, `128/128` tasks running for about `7h39m` at the check.
- Current command path: `slurm/extract_full_data_vasofree_waveform_features_v8_array.sh` calling `scripts/build_waveform_features_v8.py --use-reference-segment-component-plan` with revision-7, `194` v8 features, and default-disabled audit-only feature families.
- Log-derived progress across all 128 started shard logs: `1200108 / 1969515` samples processed (`60.9%`), with shard fractions ranging from `52.1%` to `65.0%`.
- Error scan: `128` stdout logs and `128` stderr logs were present under `logs/full_wf_v8_segplan_26968656_*.{out,err}`; no `Traceback`, `Error`, `Exception`, `FAILED`, `Killed`, `TIMEOUT`, or `RuntimeError` markers were found in stderr.
- Output state: no final `_SUCCESS` shard markers or merged v8 cache directory were present yet under `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v8`; shard caches appear to be written at task completion.
- Merge state: no merge job has been submitted. Submit `sbatch slurm/merge_full_data_vasofree_waveform_features_v8.sh` only after all `128` segment-plan shard caches complete successfully.

Planned validation after extraction completes:

1. Confirm all `128` shard directories have `_SUCCESS`, `metadata.json`, `feature_quality_report.json`, and `alignment_report.json`.
2. Merge with `slurm/merge_full_data_vasofree_waveform_features_v8.sh`.
3. Validate merged v8 shape, revision/hash, v7-prefix alignment, v7/v8 feature-name disjointness, split counts, missingness, and quality summaries.
4. Run `scripts/audit_waveform_features_v8.py` on the merged cache and inspect severe-pressure, RESP, PLETH/ABP morphology, and rate/coupling outlier examples before treating v8 as empirically validated.


## Handoff Status - 2026-09-02 11:55 EDT

Fresh handoff check from repo files, logs, and `squeue -u dk5565`:

- Active full-data extraction: SLURM array `26968656`, job name `full_wf_v8_segplan`, `128/128` tasks running for about `11h35m` at the check.
- Current command path: `slurm/extract_full_data_vasofree_waveform_features_v8_array.sh` calling `scripts/build_waveform_features_v8.py --use-reference-segment-component-plan` with revision-7, `194` v8 features, and default-disabled audit-only feature families.
- Log-derived progress across all 128 shard logs: `1836790 / 1969515` samples processed (`93.26%`), with shard fractions ranging from `80.95%` to `97.60%`.
- Error scan: `128` stdout logs and `128` stderr logs were present under `logs/full_wf_v8_segplan_26968656_*.{out,err}`; `38` stderr logs were nonempty, but no `Traceback`, `Error`, `Exception`, `FAILED`, `Killed`, `TIMEOUT`, or `RuntimeError` markers were found. Representative nonempty stderr contents were WFDB runtime warnings from normalization.
- Output state: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v8` was still absent, so no final shard `_SUCCESS` markers or merged v8 cache were present yet. This remains consistent with shard caches being written at task completion.
- Merge state: no merge job has been submitted. Submit `sbatch slurm/merge_full_data_vasofree_waveform_features_v8.sh` only after all `128` segment-plan shard caches complete successfully.

Immediate next checks after the array leaves `squeue`:

1. Confirm all `128` shard directories exist and each has `_SUCCESS`, `metadata.json`, `feature_quality_report.json`, and `alignment_report.json`.
2. Scan all `logs/full_wf_v8_segplan_26968656_*.err` again for failure markers and inspect any non-warning stderr.
3. Submit the merge with `sbatch slurm/merge_full_data_vasofree_waveform_features_v8.sh` only after every shard passes.
4. After merge, validate merged v8 shape, revision/hash, v7-prefix alignment, feature-name disjointness, split counts, missingness, and quality summaries before using v8 downstream.

## Segment-Plan Shard Recovery - 2026-09-02

After SLURM array `26968656` left `squeue`, all `128` segment-plan shard directories existed, but none had `_SUCCESS`. Every shard stderr ended with the same final validation failure:

```text
ValueError: v7/v8 tensor sample/time shapes differ: (1969515, 20) vs (~15385, 20)
```

The shard arrays had already been written with expected shard-local shapes such as `(15387, 20, 194)`. The failure was in the writer's final alignment check: per-shard v8 caches were compared against the full v7 tensor unless `--allow-prefix-v7-alignment` was used. That prefix mode is appropriate for smoke caches, but not for segment-aware shards because their rows are selected by `anchor_id`, not by first-N position.

Fix:

- Added `subset_feature_cache_by_anchor_ids()` in `waveform_baselines/wf_features_v8/cache.py`.
- Updated `scripts/build_waveform_features_v8.py` and `scripts/audit_waveform_features_v8.py` so caches with shard metadata validate against matching v7 `anchor_id`s in v8 order.
- Added a regression test covering out-of-order shard-anchor subsetting.

Validation:

- Focused tests passed:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest \
  tests.test_waveform_feature_v8.WaveformFeatureV8Tests.test_v7_cache_can_be_subset_to_shard_anchor_order \
  tests.test_waveform_feature_v8.WaveformFeatureV8Tests.test_v7_v8_alignment_and_combined_loader
```

- `compileall` passed for `waveform_baselines/wf_features_v8/cache.py`, `scripts/build_waveform_features_v8.py`, `scripts/audit_waveform_features_v8.py`, and `tests/test_waveform_feature_v8.py`.
- Recovery validation loaded the full v7 cache once, validated all existing v8 shard caches by anchor ID, wrote `alignment_report.json` and `_SUCCESS` for each shard, and covered `1,969,515` rows across `128/128` shards.

Merge state:

- Submitted merge job `26980077`, then canceled it while still pending on `Priority` and ran the merge locally.
- The first local merge attempt was killed with exit code `137` because the merge script concatenated the full `values` and `mask` tensors in memory.
- The merge script now streams `values.npy` and `mask.npy` into output memmaps and writes an aggregate shard-summary quality report by default. The aggregate report has exact counts, missingness, mean/std, min, and max from shard summaries; quantile fields are weighted shard-summary estimates. Use `--quality-mode exact` only when there is enough memory/time for an exact full-cache quality scan.
- Local merge completed successfully.

Merged cache:

```text
/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v8/full_data_vasopressor_free_waveform_features_v8_segment_plan
```

Validation:

- Merged shape: `(1969515, 20, 194)`.
- `_SUCCESS` is present.
- Schema revision: `7`.
- Schema hash matches current code: `a72aeb3b2a2942a899851cffab6d43228e939a82653514907b4fe8748cae229b`.
- Full v7/v8 alignment passed for `1,969,515` samples, `20` feature windows, v7 feature count `93`, v8 feature count `194`, and feature-name overlap count `0`.
- No leftover local merge temp directories remained after successful rename.

Next check:

1. Run full-cache `scripts/audit_waveform_features_v8.py` QC before using v8 downstream.

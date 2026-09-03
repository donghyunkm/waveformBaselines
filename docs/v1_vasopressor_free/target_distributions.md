# Target Distribution Analysis

Computed on `2026-08-28` for the same vasopressor-free regression setup used in `docs/v1_vasopressor_free/regression_results_v1_vaso_free_sorted.md`.

## Scope

- cohort: vasopressor-free overlap cohort
- split file: `outputs/splits/vasopressor_free_splits.json`
- target bundle: `outputs/targets/feature_targets_gap_vasopressor_free.npz`
- analyzed targets: the `26` leakage-safe `*_t_plus_0m_gap` regression targets evaluated in the regression results doc
- detailed descriptive statistics below use the `train` split, because future normalization or transformation decisions should be fit on training data only
- `val` and `test` are included only to check for distribution shift

## Methods / Conventions

- Missing counts are split totals minus finite valid target values from the saved `feature_mask` and target array.
- Obvious outliers are defined as Tukey 1.5*IQR outliers using fences fit on the train split for each target.
- `kurtosis` is reported as excess kurtosis.
- The central-range histogram uses the train 1st-99th percentile x-range.
- Multimodality is a heuristic flag based on smoothed train histograms; treat it as suggestive, not definitive.
- Units for waveform morphology and interaction targets are inferred from feature names because the saved bundle does not store unit metadata explicitly.
- Transformation parameters (`z` mean/std, Box-Cox lambda, Yeo-Johnson lambda) are fit on the train split only.
- Transformed descriptive statistics are reported on the transformed train targets only; val/test are not used to fit or tune transformations.

## Artifacts

- analysis generator: `scripts/analyze_target_distributions.py`
- machine-readable summary: `outputs/targets/target_distributions_vasopressor_free_t0.json`
- raw-distribution figures: `docs/figures/target_distributions/`
- raw-vs-transformed comparison figures: `docs/figures/target_distributions/transformations/`

## Regeneration

The current document and JSON summary were generated from the saved leakage-safe vasopressor-free `t+0` gap target bundle with:

```bash
mkdir -p /tmp/mpl
MPLCONFIGDIR=/tmp/mpl \
  /gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python \
  scripts/analyze_target_distributions.py \
  --target-path outputs/targets/feature_targets_gap_vasopressor_free.npz \
  --splits-path outputs/splits/vasopressor_free_splits.json \
  --metadata-path outputs/targets/feature_targets_gap_vasopressor_free.json \
  --output-json outputs/targets/target_distributions_vasopressor_free_t0.json \
  --output-doc docs/v1_vasopressor_free/target_distributions.md \
  --figure-dir docs/figures/target_distributions \
  --date 2026-08-28
```

Completed in this document:

- train-split raw target characterization for the `26` leakage-safe `*_t_plus_0m_gap` targets
- train-only transformation fitting and comparison for `z`-score, selective `log1p`, selective Box-Cox, and Yeo-Johnson
- per-target future-experiment recommendations

Not done here:

- no target arrays were modified in place
- no model training or re-evaluation was launched from this analysis
- recommendations are distribution-based only and remain unverified against downstream regression performance

## Summary Table

| Target | N | Mean | Median | Std | Skewness | Min-Max | % Zero | Distribution summary | Transform follow-up |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| `HR` | 253053 | 103.72 | 101.75 | 14.391 | 0.4382 | 55.784 - 145.67 | 0.00% | approximately symmetric | simple z-score normalization is likely enough if any normalization is used |
| `RR` | 253050 | 26.660 | 26.771 | 2.490 | -0.4607 | 3.000 - 35.945 | 0.00% | approximately symmetric; affected by extreme values | simple z-score normalization is likely enough if any normalization is used |
| `SBP` | 253053 | 127.13 | 125.93 | 22.655 | 0.3067 | 41.265 - 227.94 | 0.00% | approximately symmetric; train outliers concentrated in a few patients | simple z-score normalization is likely enough if any normalization is used |
| `DBP` | 253053 | 62.526 | 61.060 | 13.892 | 0.9985 | 22.202 - 181.45 | 0.00% | right-skewed; affected by extreme values | investigate Box-Cox or log-type transform |
| `PP` | 252945 | 64.747 | 64.143 | 20.617 | 0.0824 | 5.002 - 157.72 | 0.00% | approximately symmetric | simple z-score normalization is likely enough if any normalization is used |
| `MAP` | 253053 | 84.680 | 83.301 | 15.443 | 0.7113 | 32.157 - 184.64 | 0.00% | right-skewed; affected by extreme values | simple z-score normalization is likely enough if any normalization is used |
| `ABP_area` | 253053 | 16.142 | 15.426 | 5.940 | 0.7028 | 0.0261 - 49.596 | 0.00% | right-skewed; affected by extreme values | simple z-score normalization is likely enough if any normalization is used |
| `PLETH_ACDC` | 250423 | 0.8804 | 0.9606 | 0.2465 | -1.136 | 0.0075 - 1.867 | 0.00% | left-skewed; affected by extreme values | simple z-score normalization is likely enough if any normalization is used |
| `PLETH_amp` | 250423 | 1.378 | 1.663 | 0.5783 | -0.8796 | 0.0081 - 3.752 | 0.00% | left-skewed; possible multimodality | simple z-score normalization is likely enough if any normalization is used |
| `ECG_Ramp` | 253053 | 0.4684 | 0.3354 | 0.3627 | 1.107 | 0.0000 - 4.468 | 0.14% | right-skewed; possible multimodality; affected by extreme values | investigate log1p |
| `HRV_RMSSD` | 253052 | 97.129 | 94.578 | 64.672 | 0.3713 | 3.838 - 494.78 | 0.00% | approximately symmetric | simple z-score normalization is likely enough if any normalization is used |
| `HR_range` | 253053 | 62.221 | 71.300 | 20.663 | -0.9100 | 2.707 - 98.024 | 0.00% | left-skewed | simple z-score normalization is likely enough if any normalization is used |
| `ShockIdx` | 253053 | 0.8462 | 0.8181 | 0.2033 | 1.253 | 0.3677 - 3.062 | 0.00% | right-skewed; heavy-tailed; affected by extreme values | investigate Box-Cox or log-type transform |
| `PPV` | 252892 | 14.598 | 8.707 | 15.296 | 2.101 | 0.3978 - 99.829 | 0.00% | right-skewed; heavy-tailed; affected by extreme values | investigate Box-Cox or log-type transform |
| `PVI` | 250422 | 24.861 | 20.570 | 16.220 | 1.067 | 1.226 - 96.066 | 0.00% | right-skewed; affected by extreme values | investigate Box-Cox or log-type transform |
| `PTT` | 251938 | 192.60 | 200.00 | 36.826 | -0.9750 | 52.000 - 248.00 | 0.00% | left-skewed; possible multimodality; affected by extreme values | simple z-score normalization is likely enough if any normalization is used |
| `dPdt_max` | 253026 | 1,013.8 | 954.51 | 444.97 | 0.8894 | 10.176 - 4,152.0 | 0.00% | right-skewed; affected by extreme values | investigate Box-Cox or log-type transform |
| `ABP_tau` | 252881 | 1.166 | 1.024 | 0.6726 | 3.581 | 0.1127 - 9.901 | 0.00% | right-skewed; heavy-tailed; affected by extreme values | investigate Box-Cox or log-type transform |
| `RESP_amp` | 253053 | 0.6386 | 0.6013 | 0.3743 | 2.067 | 0.0281 - 8.163 | 0.00% | right-skewed; heavy-tailed; affected by extreme values | investigate Box-Cox or log-type transform |
| `PLETH_ACDC_PLETH_amp` | 253053 | 0.9099 | 0.9827 | 0.1870 | -3.653 | -1.000 - 1.000 | 1.04% | left-skewed; heavy-tailed; bounded with spike at +/-1 | bounded correlation feature; simple z-score may be insufficient, but log/Box-Cox are not appropriate |
| `ABP_area_ABP_tau` | 253053 | -0.2518 | -0.3854 | 0.5867 | 0.4663 | -1.000 - 1.000 | 0.07% | approximately symmetric | simple z-score normalization is likely enough if any normalization is used |
| `ABP_area_ShockIdx` | 253053 | -0.5242 | -0.7252 | 0.4987 | 1.176 | -1.000 - 1.000 | 0.00% | right-skewed; bounded with spike at +/-1; affected by extreme values | bounded correlation feature; simple z-score may be insufficient, but log/Box-Cox are not appropriate |
| `PLETH_amp_ShockIdx` | 253053 | 0.0334 | 0.0337 | 0.4614 | -0.0496 | -1.000 - 1.000 | 1.04% | approximately symmetric | simple z-score normalization is likely enough if any normalization is used |
| `PLETH_ACDC_ShockIdx` | 253053 | 0.0800 | 0.0949 | 0.4808 | -0.1347 | -1.0000 - 1.000 | 1.04% | approximately symmetric | simple z-score normalization is likely enough if any normalization is used |
| `ShockIdx_ABP_tau` | 253053 | 0.2301 | 0.3267 | 0.5419 | -0.4510 | -1.000 - 1.000 | 0.07% | approximately symmetric | simple z-score normalization is likely enough if any normalization is used |
| `PLETH_ACDC_ABP_tau` | 253053 | 0.1176 | 0.1359 | 0.4674 | -0.1942 | -0.9969 - 1.000 | 1.11% | approximately symmetric | simple z-score normalization is likely enough if any normalization is used |

## Transformation Analysis

| Target | Transformation | Valid? | Lambda | Skewness Before | Skewness After | Kurtosis After | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| `HR` | `Raw` | yes | — | 0.4382 | 0.4382 | -0.6320 | reference raw distribution |
| `HR` | `Z-score` | yes | — | 0.4382 | 0.4382 | -0.6320 | scale changed only; shape unchanged as expected |
| `HR` | `log1p` | no | — | 0.4382 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `HR` | `Box-Cox` | no | — | 0.4382 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `HR` | `Yeo-Johnson` | yes | -0.7324 | 0.4382 | 0.0272 | -0.8041 | moderately reduces skewness |
| `RR` | `Raw` | yes | — | -0.4607 | -0.4607 | 1.704 | reference raw distribution |
| `RR` | `Z-score` | yes | — | -0.4607 | -0.4607 | 1.704 | scale changed only; shape unchanged as expected |
| `RR` | `log1p` | no | — | -0.4607 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `RR` | `Box-Cox` | no | — | -0.4607 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `RR` | `Yeo-Johnson` | yes | 2.189 | -0.4607 | 0.0439 | 0.5830 | moderately reduces skewness; reduces heavy-tail behavior |
| `SBP` | `Raw` | yes | — | 0.3067 | 0.3067 | 0.0530 | reference raw distribution |
| `SBP` | `Z-score` | yes | — | 0.3067 | 0.3067 | 0.0530 | scale changed only; shape unchanged as expected |
| `SBP` | `log1p` | no | — | 0.3067 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `SBP` | `Box-Cox` | no | — | 0.3067 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `SBP` | `Yeo-Johnson` | yes | 0.4279 | 0.3067 | 0.0024 | 0.0524 | moderately reduces skewness |
| `DBP` | `Raw` | yes | — | 0.9985 | 0.9985 | 2.882 | reference raw distribution |
| `DBP` | `Z-score` | yes | — | 0.9985 | 0.9985 | 2.882 | scale changed only; shape unchanged as expected |
| `DBP` | `log1p` | yes | — | 0.9985 | 0.0950 | 0.5304 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `DBP` | `Box-Cox` | yes | -0.1054 | 0.9985 | -0.0038 | 0.4941 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `DBP` | `Yeo-Johnson` | yes | -0.1245 | 0.9985 | -0.0043 | 0.4876 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `PP` | `Raw` | yes | — | 0.0824 | 0.0824 | 0.1878 | reference raw distribution |
| `PP` | `Z-score` | yes | — | 0.0824 | 0.0824 | 0.1878 | scale changed only; shape unchanged as expected |
| `PP` | `log1p` | no | — | 0.0824 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `PP` | `Box-Cox` | no | — | 0.0824 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `PP` | `Yeo-Johnson` | yes | 0.9371 | 0.0824 | 0.0113 | 0.1975 | little shape change |
| `MAP` | `Raw` | yes | — | 0.7113 | 0.7113 | 1.342 | reference raw distribution |
| `MAP` | `Z-score` | yes | — | 0.7113 | 0.7113 | 1.342 | scale changed only; shape unchanged as expected |
| `MAP` | `log1p` | yes | — | 0.7113 | 0.0450 | 0.4961 | substantially reduces skewness |
| `MAP` | `Box-Cox` | yes | -0.0568 | 0.7113 | -0.0018 | 0.5215 | substantially reduces skewness |
| `MAP` | `Yeo-Johnson` | yes | -0.0707 | 0.7113 | -0.0021 | 0.5173 | substantially reduces skewness |
| `ABP_area` | `Raw` | yes | — | 0.7028 | 0.7028 | 1.106 | reference raw distribution |
| `ABP_area` | `Z-score` | yes | — | 0.7028 | 0.7028 | 1.106 | scale changed only; shape unchanged as expected |
| `ABP_area` | `log1p` | yes | — | 0.7028 | -0.7782 | 2.499 | increases heavy-tail behavior; compresses the upper tail |
| `ABP_area` | `Box-Cox` | yes | 0.5369 | 0.7028 | 0.0511 | 0.5857 | substantially reduces skewness |
| `ABP_area` | `Yeo-Johnson` | yes | 0.4803 | 0.7028 | 0.0338 | 0.5148 | substantially reduces skewness |
| `PLETH_ACDC` | `Raw` | yes | — | -1.136 | -1.136 | 0.5788 | reference raw distribution |
| `PLETH_ACDC` | `Z-score` | yes | — | -1.136 | -1.136 | 0.5788 | scale changed only; shape unchanged as expected |
| `PLETH_ACDC` | `log1p` | no | — | -1.136 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `PLETH_ACDC` | `Box-Cox` | no | — | -1.136 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `PLETH_ACDC` | `Yeo-Johnson` | yes | 4.743 | -1.136 | -0.2583 | -0.7033 | substantially reduces skewness; reduces heavy-tail behavior |
| `PLETH_amp` | `Raw` | yes | — | -0.8796 | -0.8796 | -0.7785 | reference raw distribution |
| `PLETH_amp` | `Z-score` | yes | — | -0.8796 | -0.8796 | -0.7785 | scale changed only; shape unchanged as expected |
| `PLETH_amp` | `log1p` | no | — | -0.8796 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `PLETH_amp` | `Box-Cox` | no | — | -0.8796 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `PLETH_amp` | `Yeo-Johnson` | yes | 3.120 | -0.8796 | -0.5107 | -1.204 | moderately reduces skewness |
| `ECG_Ramp` | `Raw` | yes | — | 1.107 | 1.107 | 0.8044 | reference raw distribution |
| `ECG_Ramp` | `Z-score` | yes | — | 1.107 | 1.107 | 0.8044 | scale changed only; shape unchanged as expected |
| `ECG_Ramp` | `log1p` | yes | — | 1.107 | 0.7172 | -0.5054 | moderately reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `ECG_Ramp` | `Box-Cox` | no | — | 1.107 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `ECG_Ramp` | `Yeo-Johnson` | yes | -1.869 | 1.107 | 0.1811 | -1.179 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `HRV_RMSSD` | `Raw` | yes | — | 0.3713 | 0.3713 | -0.4565 | reference raw distribution |
| `HRV_RMSSD` | `Z-score` | yes | — | 0.3713 | 0.3713 | -0.4565 | scale changed only; shape unchanged as expected |
| `HRV_RMSSD` | `log1p` | no | — | 0.3713 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `HRV_RMSSD` | `Box-Cox` | no | — | 0.3713 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `HRV_RMSSD` | `Yeo-Johnson` | yes | 0.5088 | 0.3713 | -0.1956 | -1.058 | little shape change |
| `HR_range` | `Raw` | yes | — | -0.9100 | -0.9100 | -0.3814 | reference raw distribution |
| `HR_range` | `Z-score` | yes | — | -0.9100 | -0.9100 | -0.3814 | scale changed only; shape unchanged as expected |
| `HR_range` | `log1p` | no | — | -0.9100 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `HR_range` | `Box-Cox` | no | — | -0.9100 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `HR_range` | `Yeo-Johnson` | yes | 2.022 | -0.9100 | -0.4823 | -1.085 | moderately reduces skewness |
| `ShockIdx` | `Raw` | yes | — | 1.253 | 1.253 | 5.373 | reference raw distribution |
| `ShockIdx` | `Z-score` | yes | — | 1.253 | 1.253 | 5.373 | scale changed only; shape unchanged as expected |
| `ShockIdx` | `log1p` | yes | — | 1.253 | 0.6516 | 1.364 | substantially reduces skewness; reduces heavy-tail behavior |
| `ShockIdx` | `Box-Cox` | yes | -0.2952 | 1.253 | -0.0024 | 0.0858 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `ShockIdx` | `Yeo-Johnson` | yes | -1.791 | 1.253 | 0.0046 | -0.1077 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `PPV` | `Raw` | yes | — | 2.101 | 2.101 | 4.687 | reference raw distribution |
| `PPV` | `Z-score` | yes | — | 2.101 | 2.101 | 4.687 | scale changed only; shape unchanged as expected |
| `PPV` | `log1p` | yes | — | 2.101 | 0.3881 | -0.5429 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `PPV` | `Box-Cox` | yes | -0.0840 | 2.101 | 0.0138 | -0.5090 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `PPV` | `Yeo-Johnson` | yes | -0.2142 | 2.101 | 0.0424 | -0.6796 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `PVI` | `Raw` | yes | — | 1.067 | 1.067 | 0.7389 | reference raw distribution |
| `PVI` | `Z-score` | yes | — | 1.067 | 1.067 | 0.7389 | scale changed only; shape unchanged as expected |
| `PVI` | `log1p` | yes | — | 1.067 | -0.1772 | -0.5604 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `PVI` | `Box-Cox` | yes | 0.1679 | 1.067 | -0.0229 | -0.6024 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `PVI` | `Yeo-Johnson` | yes | 0.1197 | 1.067 | -0.0165 | -0.6434 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `PTT` | `Raw` | yes | — | -0.9750 | -0.9750 | 0.8978 | reference raw distribution |
| `PTT` | `Z-score` | yes | — | -0.9750 | -0.9750 | 0.8978 | scale changed only; shape unchanged as expected |
| `PTT` | `log1p` | no | — | -0.9750 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `PTT` | `Box-Cox` | no | — | -0.9750 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `PTT` | `Yeo-Johnson` | yes | 2.612 | -0.9750 | -0.1800 | -0.7105 | substantially reduces skewness; reduces heavy-tail behavior |
| `dPdt_max` | `Raw` | yes | — | 0.8894 | 0.8894 | 1.941 | reference raw distribution |
| `dPdt_max` | `Z-score` | yes | — | 0.8894 | 0.8894 | 1.941 | scale changed only; shape unchanged as expected |
| `dPdt_max` | `log1p` | yes | — | 0.8894 | -1.445 | 5.361 | worsens skewness; increases heavy-tail behavior; compresses the upper tail |
| `dPdt_max` | `Box-Cox` | yes | 0.5532 | 0.8894 | 0.0638 | 0.7587 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `dPdt_max` | `Yeo-Johnson` | yes | 0.5523 | 0.8894 | 0.0634 | 0.7570 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `ABP_tau` | `Raw` | yes | — | 3.581 | 3.581 | 24.940 | reference raw distribution |
| `ABP_tau` | `Z-score` | yes | — | 3.581 | 3.581 | 24.940 | scale changed only; shape unchanged as expected |
| `ABP_tau` | `log1p` | yes | — | 3.581 | 1.132 | 3.438 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `ABP_tau` | `Box-Cox` | yes | 0.0414 | 3.581 | 0.0132 | 1.958 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `ABP_tau` | `Yeo-Johnson` | yes | -0.9843 | 3.581 | -0.0565 | 0.7796 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `RESP_amp` | `Raw` | yes | — | 2.067 | 2.067 | 18.347 | reference raw distribution |
| `RESP_amp` | `Z-score` | yes | — | 2.067 | 2.067 | 18.347 | scale changed only; shape unchanged as expected |
| `RESP_amp` | `log1p` | yes | — | 2.067 | 0.4136 | 0.8421 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `RESP_amp` | `Box-Cox` | yes | 0.3832 | 2.067 | -0.0007 | 0.3608 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `RESP_amp` | `Yeo-Johnson` | yes | -0.5961 | 2.067 | 0.0083 | -0.3294 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `PLETH_ACDC_PLETH_amp` | `Raw` | yes | — | -3.653 | -3.653 | 15.712 | reference raw distribution |
| `PLETH_ACDC_PLETH_amp` | `Z-score` | yes | — | -3.653 | -3.653 | 15.712 | scale changed only; shape unchanged as expected |
| `PLETH_ACDC_PLETH_amp` | `log1p` | no | — | -3.653 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `PLETH_ACDC_PLETH_amp` | `Box-Cox` | no | — | -3.653 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `PLETH_ACDC_PLETH_amp` | `Yeo-Johnson` | yes | 14.222 | -3.653 | -1.021 | -0.3272 | substantially reduces skewness; reduces heavy-tail behavior; introduces an extreme learned lambda; expands the bounded scale aggressively |
| `ABP_area_ABP_tau` | `Raw` | yes | — | 0.4663 | 0.4663 | -1.089 | reference raw distribution |
| `ABP_area_ABP_tau` | `Z-score` | yes | — | 0.4663 | 0.4663 | -1.089 | scale changed only; shape unchanged as expected |
| `ABP_area_ABP_tau` | `log1p` | no | — | 0.4663 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `ABP_area_ABP_tau` | `Box-Cox` | no | — | 0.4663 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `ABP_area_ABP_tau` | `Yeo-Johnson` | yes | 0.3675 | 0.4663 | 0.1848 | -1.344 | moderately reduces skewness |
| `ABP_area_ShockIdx` | `Raw` | yes | — | 1.176 | 1.176 | 0.3748 | reference raw distribution |
| `ABP_area_ShockIdx` | `Z-score` | yes | — | 1.176 | 1.176 | 0.3748 | scale changed only; shape unchanged as expected |
| `ABP_area_ShockIdx` | `log1p` | no | — | 1.176 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `ABP_area_ShockIdx` | `Box-Cox` | no | — | 1.176 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `ABP_area_ShockIdx` | `Yeo-Johnson` | yes | -0.7751 | 1.176 | 0.4444 | -1.203 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `PLETH_amp_ShockIdx` | `Raw` | yes | — | -0.0496 | -0.0496 | -0.8936 | reference raw distribution |
| `PLETH_amp_ShockIdx` | `Z-score` | yes | — | -0.0496 | -0.0496 | -0.8936 | scale changed only; shape unchanged as expected |
| `PLETH_amp_ShockIdx` | `log1p` | no | — | -0.0496 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `PLETH_amp_ShockIdx` | `Box-Cox` | no | — | -0.0496 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `PLETH_amp_ShockIdx` | `Yeo-Johnson` | yes | 1.062 | -0.0496 | -0.0117 | -0.8977 | little shape change |
| `PLETH_ACDC_ShockIdx` | `Raw` | yes | — | -0.1347 | -0.1347 | -0.9422 | reference raw distribution |
| `PLETH_ACDC_ShockIdx` | `Z-score` | yes | — | -0.1347 | -0.1347 | -0.9422 | scale changed only; shape unchanged as expected |
| `PLETH_ACDC_ShockIdx` | `log1p` | no | — | -0.1347 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `PLETH_ACDC_ShockIdx` | `Box-Cox` | no | — | -0.1347 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `PLETH_ACDC_ShockIdx` | `Yeo-Johnson` | yes | 1.171 | -0.1347 | -0.0332 | -0.9709 | little shape change |
| `ShockIdx_ABP_tau` | `Raw` | yes | — | -0.4510 | -0.4510 | -0.9527 | reference raw distribution |
| `ShockIdx_ABP_tau` | `Z-score` | yes | — | -0.4510 | -0.4510 | -0.9527 | scale changed only; shape unchanged as expected |
| `ShockIdx_ABP_tau` | `log1p` | no | — | -0.4510 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `ShockIdx_ABP_tau` | `Box-Cox` | no | — | -0.4510 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `ShockIdx_ABP_tau` | `Yeo-Johnson` | yes | 1.602 | -0.4510 | -0.1456 | -1.212 | moderately reduces skewness |
| `PLETH_ACDC_ABP_tau` | `Raw` | yes | — | -0.1942 | -0.1942 | -0.8643 | reference raw distribution |
| `PLETH_ACDC_ABP_tau` | `Z-score` | yes | — | -0.1942 | -0.1942 | -0.8643 | scale changed only; shape unchanged as expected |
| `PLETH_ACDC_ABP_tau` | `log1p` | no | — | -0.1942 | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `PLETH_ACDC_ABP_tau` | `Box-Cox` | no | — | -0.1942 | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `PLETH_ACDC_ABP_tau` | `Yeo-Johnson` | yes | 1.245 | -0.1942 | -0.0462 | -0.9292 | little shape change |

## Recommended Future Transformation Tests

| Target | Recommendation |
|---|---|
| `HR` | keep raw target as baseline; transformation probably unnecessary |
| `RR` | keep raw target as baseline; transformation probably unnecessary |
| `SBP` | keep raw target as baseline; transformation probably unnecessary |
| `DBP` | keep raw target as baseline; test log1p + z-score |
| `PP` | keep raw target as baseline; transformation probably unnecessary |
| `MAP` | keep raw target as baseline; test log1p + z-score |
| `ABP_area` | keep raw target as baseline; test Box-Cox + z-score |
| `PLETH_ACDC` | keep raw target as baseline; test Yeo-Johnson + z-score |
| `PLETH_amp` | keep raw target as baseline; test z-score only |
| `ECG_Ramp` | keep raw target as baseline; test Yeo-Johnson + z-score |
| `HRV_RMSSD` | keep raw target as baseline; transformation probably unnecessary |
| `HR_range` | keep raw target as baseline; test z-score only |
| `ShockIdx` | keep raw target as baseline; test Box-Cox + z-score |
| `PPV` | keep raw target as baseline; test Box-Cox + z-score |
| `PVI` | keep raw target as baseline; test log1p + z-score |
| `PTT` | keep raw target as baseline; test Yeo-Johnson + z-score |
| `dPdt_max` | keep raw target as baseline; test Box-Cox + z-score |
| `ABP_tau` | keep raw target as baseline; test Yeo-Johnson + z-score |
| `RESP_amp` | keep raw target as baseline; test Yeo-Johnson + z-score |
| `PLETH_ACDC_PLETH_amp` | keep raw target as baseline; test z-score only |
| `ABP_area_ABP_tau` | keep raw target as baseline; test z-score only |
| `ABP_area_ShockIdx` | keep raw target as baseline; test Yeo-Johnson + z-score |
| `PLETH_amp_ShockIdx` | keep raw target as baseline; test z-score only |
| `PLETH_ACDC_ShockIdx` | keep raw target as baseline; test z-score only |
| `ShockIdx_ABP_tau` | keep raw target as baseline; test z-score only |
| `PLETH_ACDC_ABP_tau` | keep raw target as baseline; test z-score only |

## Per-Target Details

## `HR`

- target key: `HR_t_plus_0m_gap`
- units: `bpm`
- train distribution summary: approximately symmetric
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 55.784 |
| max | 145.67 |
| mean | 103.72 |
| median | 101.75 |
| std | 14.391 |
| IQR | 21.761 |
| skewness | 0.4382 |
| kurtosis (excess) | -0.6320 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 79.931 | 83.541 | 92.119 | 101.75 | 113.88 | 130.22 | 137.13 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 13 | 0.01% |
| low outliers | 13 | 0.01% |
| high outliers | 0 | 0.00% |

Train outlier fences: lower `59.477`, upper `146.52`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 103.72 | 101.75 | 14.391 | 21.761 | 0.00% | 0.00% | 0.01% | 0.0000 | 0.0000 | reference |
| val | 35059 | 104.88 | 104.28 | 14.562 | 22.826 | 0.00% | 0.00% | 0.00% | 0.0652 | 0.1164 | no clear shift |
| test | 35701 | 104.43 | 103.51 | 14.910 | 23.587 | 0.00% | 0.00% | 0.00% | 0.0438 | 0.0809 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `13` across `1` patients
- largest single-patient share of train outliers: `100.00%`
- top-5-patient share of train outliers: `100.00%`

### Plots

![HR full histogram](figures/target_distributions/HR_hist_full.png)

![HR central histogram](figures/target_distributions/HR_hist_central.png)

![HR split boxplot](figures/target_distributions/HR_boxplot.png)

![HR ECDF](figures/target_distributions/HR_ecdf.png)

### Short Interpretation

- `HR` (bpm) has median `101.75` and IQR `21.761`; approximately symmetric; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; transformation probably unnecessary`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 103.72 | 101.75 | 14.391 | 0.4382 | 0.4382 | -0.6320 | 1.626 | 55.784 | 145.67 | 79.931 | 101.75 | 137.13 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | -0.1371 | 1.000 | 0.4382 | 0.4382 | -0.6320 | 1.626 | -3.331 | 2.915 | -1.653 | -0.1371 | 2.321 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | 0.4382 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | 0.4382 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | -0.7324 | 1.320 | 1.320 | 0.0045 | 0.4382 | 0.0272 | -0.8041 | 1.272 | 1.295 | 1.330 | 1.311 | 1.320 | 1.328 | moderately reduces skewness |

### Transformation Comparison Plot

![HR transformation comparison](figures/target_distributions/transformations/HR_transform_compare.png)

## `RR`

- target key: `RR_t_plus_0m_gap`
- units: `breaths/min`
- train distribution summary: approximately symmetric; affected by extreme values
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253050 | 9358 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 3.000 |
| max | 35.945 |
| mean | 26.660 |
| median | 26.771 |
| std | 2.490 |
| IQR | 3.041 |
| skewness | -0.4607 |
| kurtosis (excess) | 1.704 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 20.069 | 22.459 | 25.202 | 26.771 | 28.243 | 30.541 | 32.060 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 4942 | 1.95% |
| low outliers | 3855 | 1.52% |
| high outliers | 1087 | 0.43% |

Train outlier fences: lower `20.640`, upper `32.805`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253050 | 26.660 | 26.771 | 2.490 | 3.041 | 0.00% | 0.00% | 1.95% | 0.0000 | 0.0000 | reference |
| val | 35059 | 26.861 | 26.963 | 2.466 | 2.885 | 0.00% | 0.00% | 2.47% | 0.0395 | 0.0633 | no clear shift |
| test | 35701 | 26.650 | 26.775 | 2.365 | 3.014 | 0.00% | 0.00% | 1.18% | 0.0156 | 0.0015 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `4942` across `162` patients
- largest single-patient share of train outliers: `5.61%`
- top-5-patient share of train outliers: `22.20%`

### Plots

![RR full histogram](figures/target_distributions/RR_hist_full.png)

![RR central histogram](figures/target_distributions/RR_hist_central.png)

![RR split boxplot](figures/target_distributions/RR_boxplot.png)

![RR ECDF](figures/target_distributions/RR_ecdf.png)

### Short Interpretation

- `RR` (breaths/min) has median `26.771` and IQR `3.041`; approximately symmetric; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; transformation probably unnecessary`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 26.660 | 26.771 | 2.490 | -0.4607 | -0.4607 | 1.704 | 1.739 | 3.000 | 35.945 | 20.069 | 26.771 | 32.060 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | 0.0446 | 1.000 | -0.4607 | -0.4607 | 1.704 | 1.739 | -9.502 | 3.729 | -2.647 | 0.0446 | 2.169 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | -0.4607 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | -0.4607 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 2.189 | 660.27 | 659.15 | 126.47 | -0.4607 | 0.0439 | 0.5830 | 1.942 | 9.038 | 1,231.5 | 359.93 | 659.15 | 965.58 | moderately reduces skewness; reduces heavy-tail behavior |

### Transformation Comparison Plot

![RR transformation comparison](figures/target_distributions/transformations/RR_transform_compare.png)

## `SBP`

- target key: `SBP_t_plus_0m_gap`
- units: `mmHg`
- train distribution summary: approximately symmetric; train outliers concentrated in a few patients
- split-shift summary: `mild shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 41.265 |
| max | 227.94 |
| mean | 127.13 |
| median | 125.93 |
| std | 22.655 |
| IQR | 31.702 |
| skewness | 0.3067 |
| kurtosis (excess) | 0.0530 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 82.151 | 92.998 | 110.46 | 125.93 | 142.16 | 165.93 | 184.36 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 1866 | 0.74% |
| low outliers | 311 | 0.12% |
| high outliers | 1555 | 0.61% |

Train outlier fences: lower `62.908`, upper `189.72`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 127.13 | 125.93 | 22.655 | 31.702 | 0.00% | 0.00% | 0.74% | 0.0000 | 0.0000 | reference |
| val | 35059 | 126.81 | 125.76 | 23.557 | 33.773 | 0.00% | 0.00% | 0.87% | 0.0248 | 0.0056 | no clear shift |
| test | 35701 | 130.68 | 130.98 | 21.560 | 30.631 | 0.00% | 0.00% | 0.09% | 0.0857 | 0.1592 | mild shift |

### Extreme Values By Patient

- train Tukey outliers: `1866` across `29` patients
- largest single-patient share of train outliers: `31.08%`
- top-5-patient share of train outliers: `79.42%`

### Plots

![SBP full histogram](figures/target_distributions/SBP_hist_full.png)

![SBP central histogram](figures/target_distributions/SBP_hist_central.png)

![SBP split boxplot](figures/target_distributions/SBP_boxplot.png)

![SBP ECDF](figures/target_distributions/SBP_ecdf.png)

### Short Interpretation

- `SBP` (mmHg) has median `125.93` and IQR `31.702`; approximately symmetric; train outliers concentrated in a few patients; zeros `0.00%`; split comparison `mild shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; transformation probably unnecessary`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 127.13 | 125.93 | 22.655 | 0.3067 | 0.3067 | 0.0530 | 1.843 | 41.265 | 227.94 | 82.151 | 125.93 | 184.36 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | -0.0530 | 1.000 | 0.3067 | 0.3067 | 0.0530 | 1.843 | -3.790 | 4.450 | -1.986 | -0.0530 | 2.526 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | 0.3067 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | 0.3067 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 0.4279 | 16.237 | 16.234 | 1.412 | 0.3067 | 0.0024 | 0.0524 | 1.645 | 9.263 | 21.566 | 13.159 | 16.234 | 19.500 | moderately reduces skewness |

### Transformation Comparison Plot

![SBP transformation comparison](figures/target_distributions/transformations/SBP_transform_compare.png)

## `DBP`

- target key: `DBP_t_plus_0m_gap`
- units: `mmHg`
- train distribution summary: right-skewed; affected by extreme values
- split-shift summary: `no clear shift`
- transformation follow-up: investigate Box-Cox or log-type transform

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 22.202 |
| max | 181.45 |
| mean | 62.526 |
| median | 61.060 |
| std | 13.892 |
| IQR | 17.163 |
| skewness | 0.9985 |
| kurtosis (excess) | 2.882 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 37.095 | 42.982 | 53.082 | 61.060 | 70.245 | 86.537 | 104.41 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 5156 | 2.04% |
| low outliers | 92 | 0.04% |
| high outliers | 5064 | 2.00% |

Train outlier fences: lower `27.337`, upper `95.991`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 62.526 | 61.060 | 13.892 | 17.163 | 0.00% | 0.00% | 2.04% | 0.0000 | 0.0000 | reference |
| val | 35059 | 62.086 | 61.030 | 12.841 | 16.907 | 0.00% | 0.00% | 1.13% | 0.0175 | 0.0017 | no clear shift |
| test | 35701 | 61.600 | 59.833 | 13.530 | 15.563 | 0.00% | 0.00% | 1.83% | 0.0464 | 0.0715 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `5156` across `166` patients
- largest single-patient share of train outliers: `5.78%`
- top-5-patient share of train outliers: `20.11%`

### Plots

![DBP full histogram](figures/target_distributions/DBP_hist_full.png)

![DBP central histogram](figures/target_distributions/DBP_hist_central.png)

![DBP split boxplot](figures/target_distributions/DBP_boxplot.png)

![DBP ECDF](figures/target_distributions/DBP_ecdf.png)

### Short Interpretation

- `DBP` (mmHg) has median `61.060` and IQR `17.163`; right-skewed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; investigate Box-Cox or log-type transform.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test log1p + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 62.526 | 61.060 | 13.892 | 0.9985 | 0.9985 | 2.882 | 2.526 | 22.202 | 181.45 | 37.095 | 61.060 | 104.41 | reference raw distribution |
| `Z-score` | yes | — | 0.0000 | -0.1056 | 1.000 | 0.9985 | 0.9985 | 2.882 | 2.526 | -2.903 | 8.560 | -1.831 | -0.1056 | 3.015 | scale changed only; shape unchanged as expected |
| `log1p` | yes | — | 4.129 | 4.128 | 0.2123 | 0.9985 | 0.0950 | 0.5304 | 1.922 | 3.144 | 5.206 | 3.640 | 4.128 | 4.658 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Box-Cox` | yes | -0.1054 | 3.335 | 3.336 | 0.1399 | 0.9985 | -0.0038 | 0.4941 | 1.862 | 2.644 | 4.003 | 3.005 | 3.336 | 3.674 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Yeo-Johnson` | yes | -0.1245 | 3.227 | 3.228 | 0.1269 | 0.9985 | -0.0043 | 0.4876 | 1.860 | 2.602 | 3.831 | 2.927 | 3.228 | 3.534 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |

### Transformation Comparison Plot

![DBP transformation comparison](figures/target_distributions/transformations/DBP_transform_compare.png)

## `PP`

- target key: `PP_t_plus_0m_gap`
- units: `mmHg`
- train distribution summary: approximately symmetric
- split-shift summary: `mild shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 252945 | 9463 | 3.61% |
| val | 35892 | 35052 | 840 | 2.34% |
| test | 36533 | 35638 | 895 | 2.45% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 5.002 |
| max | 157.72 |
| mean | 64.747 |
| median | 64.143 |
| std | 20.617 |
| IQR | 27.105 |
| skewness | 0.0824 |
| kurtosis (excess) | 0.1878 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 15.232 | 31.007 | 51.226 | 64.143 | 78.331 | 98.925 | 113.39 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 2448 | 0.97% |
| low outliers | 989 | 0.39% |
| high outliers | 1459 | 0.58% |

Train outlier fences: lower `10.568`, upper `118.99`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 252945 | 64.747 | 64.143 | 20.617 | 27.105 | 0.00% | 0.00% | 0.97% | 0.0000 | 0.0000 | reference |
| val | 35052 | 64.822 | 62.494 | 22.409 | 31.980 | 0.00% | 0.00% | 1.00% | 0.0585 | 0.0608 | no clear shift |
| test | 35638 | 69.340 | 69.697 | 19.882 | 26.879 | 0.00% | 0.00% | 0.43% | 0.1083 | 0.2049 | mild shift |

### Extreme Values By Patient

- train Tukey outliers: `2448` across `93` patients
- largest single-patient share of train outliers: `18.01%`
- top-5-patient share of train outliers: `46.41%`

### Plots

![PP full histogram](figures/target_distributions/PP_hist_full.png)

![PP central histogram](figures/target_distributions/PP_hist_central.png)

![PP split boxplot](figures/target_distributions/PP_boxplot.png)

![PP ECDF](figures/target_distributions/PP_ecdf.png)

### Short Interpretation

- `PP` (mmHg) has median `64.143` and IQR `27.105`; approximately symmetric; zeros `0.00%`; split comparison `mild shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; transformation probably unnecessary`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 64.747 | 64.143 | 20.617 | 0.0824 | 0.0824 | 0.1878 | 1.817 | 5.002 | 157.72 | 15.232 | 64.143 | 113.39 | reference raw distribution |
| `Z-score` | yes | — | 0.0000 | -0.0293 | 1.000 | 0.0824 | 0.0824 | 0.1878 | 1.817 | -2.898 | 4.509 | -2.402 | -0.0293 | 2.359 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | 0.0824 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | 0.0824 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 0.9371 | 52.688 | 52.389 | 15.898 | 0.0824 | 0.0113 | 0.1975 | 1.782 | 4.655 | 122.08 | 13.469 | 52.389 | 89.534 | little shape change |

### Transformation Comparison Plot

![PP transformation comparison](figures/target_distributions/transformations/PP_transform_compare.png)

## `MAP`

- target key: `MAP_t_plus_0m_gap`
- units: `mmHg`
- train distribution summary: right-skewed; affected by extreme values
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 32.157 |
| max | 184.64 |
| mean | 84.680 |
| median | 83.301 |
| std | 15.443 |
| IQR | 19.883 |
| skewness | 0.7113 |
| kurtosis (excess) | 1.342 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 55.600 | 62.475 | 73.798 | 83.301 | 93.681 | 112.26 | 128.49 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 4362 | 1.72% |
| low outliers | 287 | 0.11% |
| high outliers | 4075 | 1.61% |

Train outlier fences: lower `43.973`, upper `123.51`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 84.680 | 83.301 | 15.443 | 19.883 | 0.00% | 0.00% | 1.72% | 0.0000 | 0.0000 | reference |
| val | 35059 | 84.329 | 83.018 | 14.556 | 18.678 | 0.00% | 0.00% | 1.24% | 0.0180 | 0.0143 | no clear shift |
| test | 35701 | 85.286 | 83.927 | 14.577 | 18.256 | 0.00% | 0.00% | 1.27% | 0.0409 | 0.0315 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `4362` across `140` patients
- largest single-patient share of train outliers: `5.50%`
- top-5-patient share of train outliers: `23.41%`

### Plots

![MAP full histogram](figures/target_distributions/MAP_hist_full.png)

![MAP central histogram](figures/target_distributions/MAP_hist_central.png)

![MAP split boxplot](figures/target_distributions/MAP_boxplot.png)

![MAP ECDF](figures/target_distributions/MAP_ecdf.png)

### Short Interpretation

- `MAP` (mmHg) has median `83.301` and IQR `19.883`; right-skewed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test log1p + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 84.680 | 83.301 | 15.443 | 0.7113 | 0.7113 | 1.342 | 2.273 | 32.157 | 184.64 | 55.600 | 83.301 | 128.49 | reference raw distribution |
| `Z-score` | yes | — | 0.0000 | -0.0893 | 1.000 | 0.7113 | 0.7113 | 1.342 | 2.273 | -3.401 | 6.473 | -1.883 | -0.0893 | 2.837 | scale changed only; shape unchanged as expected |
| `log1p` | yes | — | 4.435 | 4.434 | 0.1774 | 0.7113 | 0.0450 | 0.4961 | 1.821 | 3.501 | 5.224 | 4.036 | 4.434 | 4.864 | substantially reduces skewness |
| `Box-Cox` | yes | -0.0568 | 3.910 | 3.910 | 0.1396 | 0.7113 | -0.0018 | 0.5215 | 1.794 | 3.150 | 4.516 | 3.592 | 3.910 | 4.243 | substantially reduces skewness |
| `Yeo-Johnson` | yes | -0.0707 | 3.806 | 3.806 | 0.1296 | 0.7113 | -0.0021 | 0.5173 | 1.793 | 3.101 | 4.367 | 3.511 | 3.806 | 4.115 | substantially reduces skewness |

### Transformation Comparison Plot

![MAP transformation comparison](figures/target_distributions/transformations/MAP_transform_compare.png)

## `ABP_area`

- target key: `ABP_area_t_plus_0m_gap`
- units: `mmHg*s (inferred)`
- train distribution summary: right-skewed; affected by extreme values
- split-shift summary: `possible meaningful shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 0.0261 |
| max | 49.596 |
| mean | 16.142 |
| median | 15.426 |
| std | 5.940 |
| IQR | 7.726 |
| skewness | 0.7028 |
| kurtosis (excess) | 1.106 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 4.591 | 7.620 | 12.021 | 15.426 | 19.748 | 26.519 | 33.726 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 4486 | 1.77% |
| low outliers | 93 | 0.04% |
| high outliers | 4393 | 1.74% |

Train outlier fences: lower `0.4316`, upper `31.337`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 16.142 | 15.426 | 5.940 | 7.726 | 0.00% | 0.00% | 1.77% | 0.0000 | 0.0000 | reference |
| val | 35059 | 16.548 | 16.225 | 5.901 | 8.365 | 0.00% | 0.00% | 1.24% | 0.0570 | 0.1035 | no clear shift |
| test | 35701 | 17.846 | 17.386 | 6.339 | 8.327 | 0.00% | 0.00% | 3.73% | 0.1248 | 0.2536 | possible meaningful shift |

### Extreme Values By Patient

- train Tukey outliers: `4486` across `64` patients
- largest single-patient share of train outliers: `11.01%`
- top-5-patient share of train outliers: `38.99%`

### Plots

![ABP_area full histogram](figures/target_distributions/ABP_area_hist_full.png)

![ABP_area central histogram](figures/target_distributions/ABP_area_hist_central.png)

![ABP_area split boxplot](figures/target_distributions/ABP_area_boxplot.png)

![ABP_area ECDF](figures/target_distributions/ABP_area_ecdf.png)

### Short Interpretation

- `ABP_area` (mmHg*s (inferred)) has median `15.426` and IQR `7.726`; right-skewed; affected by extreme values; zeros `0.00%`; split comparison `possible meaningful shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test Box-Cox + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 16.142 | 15.426 | 5.940 | 0.7028 | 0.7028 | 1.106 | 2.369 | 0.0261 | 49.596 | 4.591 | 15.426 | 33.726 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | -0.1205 | 1.000 | 0.7028 | 0.7028 | 1.106 | 2.369 | -2.713 | 5.632 | -1.945 | -0.1205 | 2.960 | scale changed only; shape unchanged as expected |
| `log1p` | yes | — | 2.778 | 2.799 | 0.3696 | 0.7028 | -0.7782 | 2.499 | 1.607 | 0.0258 | 3.924 | 1.721 | 2.799 | 3.547 | increases heavy-tail behavior; compresses the upper tail |
| `Box-Cox` | yes | 0.5369 | 6.288 | 6.230 | 1.645 | 0.7028 | 0.0511 | 0.5857 | 1.954 | -1.599 | 13.288 | 2.359 | 6.230 | 10.455 | substantially reduces skewness |
| `Yeo-Johnson` | yes | 0.4803 | 5.946 | 5.904 | 1.359 | 0.7028 | 0.0338 | 0.5148 | 1.929 | 0.0259 | 11.627 | 2.677 | 5.904 | 9.360 | substantially reduces skewness |

### Transformation Comparison Plot

![ABP_area transformation comparison](figures/target_distributions/transformations/ABP_area_transform_compare.png)

## `PLETH_ACDC`

- target key: `PLETH_ACDC_t_plus_0m_gap`
- units: `ratio`
- train distribution summary: left-skewed; affected by extreme values
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 250423 | 11985 | 4.57% |
| val | 35892 | 34438 | 1454 | 4.05% |
| test | 36533 | 34846 | 1687 | 4.62% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 0.0075 |
| max | 1.867 |
| mean | 0.8804 |
| median | 0.9606 |
| std | 0.2465 |
| IQR | 0.2925 |
| skewness | -1.136 |
| kurtosis (excess) | 0.5788 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1705 | 0.3568 | 0.7638 | 0.9606 | 1.056 | 1.151 | 1.212 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 10221 | 4.08% |
| low outliers | 10220 | 4.08% |
| high outliers | 1 | 0.00% |

Train outlier fences: lower `0.3250`, upper `1.495`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 250423 | 0.8804 | 0.9606 | 0.2465 | 0.2925 | 0.00% | 0.00% | 4.08% | 0.0000 | 0.0000 | reference |
| val | 34438 | 0.8501 | 0.9422 | 0.2593 | 0.3297 | 0.00% | 0.00% | 5.70% | 0.0468 | 0.0628 | no clear shift |
| test | 34846 | 0.8844 | 0.9694 | 0.2286 | 0.2736 | 0.00% | 0.00% | 2.54% | 0.0586 | 0.0300 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `10221` across `146` patients
- largest single-patient share of train outliers: `6.39%`
- top-5-patient share of train outliers: `20.75%`

### Plots

![PLETH_ACDC full histogram](figures/target_distributions/PLETH_ACDC_hist_full.png)

![PLETH_ACDC central histogram](figures/target_distributions/PLETH_ACDC_hist_central.png)

![PLETH_ACDC split boxplot](figures/target_distributions/PLETH_ACDC_boxplot.png)

![PLETH_ACDC ECDF](figures/target_distributions/PLETH_ACDC_ecdf.png)

### Short Interpretation

- `PLETH_ACDC` (ratio) has median `0.9606` and IQR `0.2925`; left-skewed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test Yeo-Johnson + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 0.8804 | 0.9606 | 0.2465 | -1.136 | -1.136 | 0.5788 | 0.8598 | 0.0075 | 1.867 | 0.1705 | 0.9606 | 1.212 | reference raw distribution |
| `Z-score` | yes | — | 0.0000 | 0.3256 | 1.000 | -1.136 | -1.136 | 0.5788 | 0.8598 | -3.541 | 4.003 | -2.880 | 0.3256 | 1.346 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | -1.136 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | -1.136 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 4.743 | 4.574 | 4.927 | 2.213 | -1.136 | -0.2583 | -0.7033 | 1.192 | 0.0076 | 30.943 | 0.2340 | 4.927 | 8.897 | substantially reduces skewness; reduces heavy-tail behavior |

### Transformation Comparison Plot

![PLETH_ACDC transformation comparison](figures/target_distributions/transformations/PLETH_ACDC_transform_compare.png)

## `PLETH_amp`

- target key: `PLETH_amp_t_plus_0m_gap`
- units: `pleth AU`
- train distribution summary: left-skewed; possible multimodality
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 250423 | 11985 | 4.57% |
| val | 35892 | 34438 | 1454 | 4.05% |
| test | 36533 | 34846 | 1687 | 4.62% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 0.0081 |
| max | 3.752 |
| mean | 1.378 |
| median | 1.663 |
| std | 0.5783 |
| IQR | 0.9028 |
| skewness | -0.8796 |
| kurtosis (excess) | -0.7785 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1353 | 0.3139 | 0.9302 | 1.663 | 1.833 | 1.923 | 1.953 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 1 | 0.00% |
| low outliers | 0 | 0.00% |
| high outliers | 1 | 0.00% |

Train outlier fences: lower `-0.4241`, upper `3.187`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 250423 | 1.378 | 1.663 | 0.5783 | 0.9028 | 0.00% | 0.00% | 0.00% | 0.0000 | 0.0000 | reference |
| val | 34438 | 1.341 | 1.603 | 0.5746 | 0.9895 | 0.00% | 0.00% | 0.00% | 0.0482 | 0.0660 | no clear shift |
| test | 34846 | 1.398 | 1.684 | 0.5774 | 0.8921 | 0.00% | 0.00% | 0.00% | 0.0261 | 0.0237 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `1` across `1` patients
- largest single-patient share of train outliers: `100.00%`
- top-5-patient share of train outliers: `100.00%`

### Plots

![PLETH_amp full histogram](figures/target_distributions/PLETH_amp_hist_full.png)

![PLETH_amp central histogram](figures/target_distributions/PLETH_amp_hist_central.png)

![PLETH_amp split boxplot](figures/target_distributions/PLETH_amp_boxplot.png)

![PLETH_amp ECDF](figures/target_distributions/PLETH_amp_ecdf.png)

### Short Interpretation

- `PLETH_amp` (pleth AU) has median `1.663` and IQR `0.9028`; left-skewed; possible multimodality; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test z-score only`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 1.378 | 1.663 | 0.5783 | -0.8796 | -0.8796 | -0.7785 | 0.3214 | 0.0081 | 3.752 | 0.1353 | 1.663 | 1.953 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | 0.4929 | 1.000 | -0.8796 | -0.8796 | -0.7785 | 0.3214 | -2.369 | 4.105 | -2.149 | 0.4929 | 0.9946 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | -0.8796 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | -0.8796 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 3.120 | 5.323 | 6.487 | 3.043 | -0.8796 | -0.5107 | -1.204 | 0.4497 | 0.0082 | 41.131 | 0.1557 | 6.487 | 9.078 | moderately reduces skewness |

### Transformation Comparison Plot

![PLETH_amp transformation comparison](figures/target_distributions/transformations/PLETH_amp_transform_compare.png)

## `ECG_Ramp`

- target key: `ECG_Ramp_t_plus_0m_gap`
- units: `ECG AU`
- train distribution summary: right-skewed; possible multimodality; affected by extreme values
- split-shift summary: `no clear shift`
- transformation follow-up: investigate log1p

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 0.0000 |
| max | 4.468 |
| mean | 0.4684 |
| median | 0.3354 |
| std | 0.3627 |
| IQR | 0.5163 |
| skewness | 1.107 |
| kurtosis (excess) | 0.8044 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0611 | 0.0971 | 0.1801 | 0.3354 | 0.6964 | 1.202 | 1.524 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 365 | 0.14% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 3513 | 1.39% |
| low outliers | 0 | 0.00% |
| high outliers | 3513 | 1.39% |

Train outlier fences: lower `-0.5944`, upper `1.471`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 0.4684 | 0.3354 | 0.3627 | 0.5163 | 0.14% | 0.00% | 1.39% | 0.0000 | 0.0000 | reference |
| val | 35059 | 0.4675 | 0.3122 | 0.3760 | 0.5481 | 0.10% | 0.00% | 0.97% | 0.0665 | 0.0450 | no clear shift |
| test | 35701 | 0.4650 | 0.3477 | 0.3653 | 0.4853 | 0.00% | 0.00% | 2.14% | 0.0431 | 0.0238 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `3513` across `40` patients
- largest single-patient share of train outliers: `8.60%`
- top-5-patient share of train outliers: `34.30%`

### Plots

![ECG_Ramp full histogram](figures/target_distributions/ECG_Ramp_hist_full.png)

![ECG_Ramp central histogram](figures/target_distributions/ECG_Ramp_hist_central.png)

![ECG_Ramp split boxplot](figures/target_distributions/ECG_Ramp_boxplot.png)

![ECG_Ramp ECDF](figures/target_distributions/ECG_Ramp_ecdf.png)

### Short Interpretation

- `ECG_Ramp` (ECG AU) has median `0.3354` and IQR `0.5163`; right-skewed; possible multimodality; affected by extreme values; zeros `0.14%`; split comparison `no clear shift`; investigate log1p.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test Yeo-Johnson + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 0.4684 | 0.3354 | 0.3627 | 1.107 | 1.107 | 0.8044 | 2.303 | 0.0000 | 4.468 | 0.0611 | 0.3354 | 1.524 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | -0.3668 | 1.000 | 1.107 | 1.107 | 0.8044 | 2.303 | -1.292 | 11.029 | -1.123 | -0.3668 | 2.912 | scale changed only; shape unchanged as expected |
| `log1p` | yes | — | 0.3569 | 0.2892 | 0.2279 | 1.107 | 0.7172 | -0.5054 | 1.754 | 0.0000 | 1.699 | 0.0593 | 0.2892 | 0.9260 | moderately reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Box-Cox` | no | — | — | — | — | 1.107 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | -1.869 | 0.2372 | 0.2234 | 0.1087 | 1.107 | 0.1811 | -1.179 | 1.121 | 0.0000 | 0.5127 | 0.0562 | 0.2234 | 0.4403 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |

### Transformation Comparison Plot

![ECG_Ramp transformation comparison](figures/target_distributions/transformations/ECG_Ramp_transform_compare.png)

## `HRV_RMSSD`

- target key: `HRV_RMSSD_t_plus_0m_gap`
- units: `ms (inferred)`
- train distribution summary: approximately symmetric
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253052 | 9356 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 3.838 |
| max | 494.78 |
| mean | 97.129 |
| median | 94.578 |
| std | 64.672 |
| IQR | 112.53 |
| skewness | 0.3713 |
| kurtosis (excess) | -0.4565 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 7.122 | 10.129 | 36.297 | 94.578 | 148.83 | 199.99 | 247.83 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 269 | 0.11% |
| low outliers | 0 | 0.00% |
| high outliers | 269 | 0.11% |

Train outlier fences: lower `-132.50`, upper `317.62`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253052 | 97.129 | 94.578 | 64.672 | 112.53 | 0.00% | 0.00% | 0.11% | 0.0000 | 0.0000 | reference |
| val | 35059 | 104.52 | 106.79 | 62.513 | 98.098 | 0.00% | 0.00% | 0.33% | 0.0771 | 0.1086 | no clear shift |
| test | 35701 | 107.15 | 107.31 | 69.035 | 111.36 | 0.00% | 0.00% | 0.50% | 0.0573 | 0.1131 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `269` across `12` patients
- largest single-patient share of train outliers: `73.61%`
- top-5-patient share of train outliers: `92.57%`

### Plots

![HRV_RMSSD full histogram](figures/target_distributions/HRV_RMSSD_hist_full.png)

![HRV_RMSSD central histogram](figures/target_distributions/HRV_RMSSD_hist_central.png)

![HRV_RMSSD split boxplot](figures/target_distributions/HRV_RMSSD_boxplot.png)

![HRV_RMSSD ECDF](figures/target_distributions/HRV_RMSSD_ecdf.png)

### Short Interpretation

- `HRV_RMSSD` (ms (inferred)) has median `94.578` and IQR `112.53`; approximately symmetric; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; transformation probably unnecessary`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 97.129 | 94.578 | 64.672 | 0.3713 | 0.3713 | -0.4565 | 1.362 | 3.838 | 494.78 | 7.122 | 94.578 | 247.83 | reference raw distribution |
| `Z-score` | yes | — | 0.0000 | -0.0394 | 1.0000 | 0.3713 | 0.3713 | -0.4565 | 1.362 | -1.443 | 6.149 | -1.392 | -0.0394 | 2.330 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | 0.3713 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | 0.3713 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 0.5088 | 16.941 | 18.038 | 7.437 | 0.3713 | -0.1956 | -1.058 | 0.9839 | 2.418 | 44.261 | 3.740 | 18.038 | 30.585 | little shape change |

### Transformation Comparison Plot

![HRV_RMSSD transformation comparison](figures/target_distributions/transformations/HRV_RMSSD_transform_compare.png)

## `HR_range`

- target key: `HR_range_t_plus_0m_gap`
- units: `bpm`
- train distribution summary: left-skewed
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 2.707 |
| max | 98.024 |
| mean | 62.221 |
| median | 71.300 |
| std | 20.663 |
| IQR | 29.919 |
| skewness | -0.9100 |
| kurtosis (excess) | -0.3814 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 11.955 | 20.750 | 47.952 | 71.300 | 77.871 | 83.616 | 88.609 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 2 | 0.00% |
| low outliers | 2 | 0.00% |
| high outliers | 0 | 0.00% |

Train outlier fences: lower `3.074`, upper `122.75`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 62.221 | 71.300 | 20.663 | 29.919 | 0.00% | 0.00% | 0.00% | 0.0000 | 0.0000 | reference |
| val | 35059 | 64.244 | 72.106 | 19.038 | 22.785 | 0.00% | 0.00% | 0.00% | 0.0551 | 0.0269 | no clear shift |
| test | 35701 | 65.066 | 73.065 | 19.893 | 24.109 | 0.00% | 0.00% | 0.28% | 0.0629 | 0.0590 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `2` across `2` patients
- largest single-patient share of train outliers: `50.00%`
- top-5-patient share of train outliers: `100.00%`

### Plots

![HR_range full histogram](figures/target_distributions/HR_range_hist_full.png)

![HR_range central histogram](figures/target_distributions/HR_range_hist_central.png)

![HR_range split boxplot](figures/target_distributions/HR_range_boxplot.png)

![HR_range ECDF](figures/target_distributions/HR_range_ecdf.png)

### Short Interpretation

- `HR_range` (bpm) has median `71.300` and IQR `29.919`; left-skewed; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test z-score only`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 62.221 | 71.300 | 20.663 | -0.9100 | -0.9100 | -0.3814 | 0.5785 | 2.707 | 98.024 | 11.955 | 71.300 | 88.609 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | 0.4394 | 1.000 | -0.9100 | -0.9100 | -0.3814 | 0.5785 | -2.880 | 1.733 | -2.433 | 0.4394 | 1.277 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | -0.9100 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | -0.9100 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 2.022 | 2,405.8 | 2,843.0 | 1,235.0 | -0.9100 | -0.4823 | -1.085 | 0.7366 | 6.503 | 5,371.0 | 87.367 | 2,843.0 | 4,388.4 | moderately reduces skewness |

### Transformation Comparison Plot

![HR_range transformation comparison](figures/target_distributions/transformations/HR_range_transform_compare.png)

## `ShockIdx`

- target key: `ShockIdx_t_plus_0m_gap`
- units: `ratio`
- train distribution summary: right-skewed; heavy-tailed; affected by extreme values
- split-shift summary: `no clear shift`
- transformation follow-up: investigate Box-Cox or log-type transform

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 0.3677 |
| max | 3.062 |
| mean | 0.8462 |
| median | 0.8181 |
| std | 0.2033 |
| IQR | 0.2583 |
| skewness | 1.253 |
| kurtosis (excess) | 5.373 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5060 | 0.5727 | 0.7022 | 0.8181 | 0.9605 | 1.210 | 1.424 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 4729 | 1.87% |
| low outliers | 0 | 0.00% |
| high outliers | 4729 | 1.87% |

Train outlier fences: lower `0.3148`, upper `1.348`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 0.8462 | 0.8181 | 0.2033 | 0.2583 | 0.00% | 0.00% | 1.87% | 0.0000 | 0.0000 | reference |
| val | 35059 | 0.8619 | 0.8477 | 0.2153 | 0.3041 | 0.00% | 0.00% | 2.25% | 0.0643 | 0.1143 | no clear shift |
| test | 35701 | 0.8251 | 0.8088 | 0.1875 | 0.2529 | 0.00% | 0.00% | 1.09% | 0.0428 | 0.0363 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `4729` across `126` patients
- largest single-patient share of train outliers: `13.91%`
- top-5-patient share of train outliers: `33.26%`

### Plots

![ShockIdx full histogram](figures/target_distributions/ShockIdx_hist_full.png)

![ShockIdx central histogram](figures/target_distributions/ShockIdx_hist_central.png)

![ShockIdx split boxplot](figures/target_distributions/ShockIdx_boxplot.png)

![ShockIdx ECDF](figures/target_distributions/ShockIdx_ecdf.png)

### Short Interpretation

- `ShockIdx` (ratio) has median `0.8181` and IQR `0.2583`; right-skewed; heavy-tailed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; investigate Box-Cox or log-type transform.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test Box-Cox + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 0.8462 | 0.8181 | 0.2033 | 1.253 | 1.253 | 5.373 | 2.344 | 0.3677 | 3.062 | 0.5060 | 0.8181 | 1.424 | reference raw distribution |
| `Z-score` | yes | — | 0.0000 | -0.1382 | 1.000 | 1.253 | 1.253 | 5.373 | 2.344 | -2.354 | 10.897 | -1.673 | -0.1382 | 2.840 | scale changed only; shape unchanged as expected |
| `log1p` | yes | — | 0.6074 | 0.5978 | 0.1056 | 1.253 | 0.6516 | 1.364 | 2.035 | 0.3131 | 1.402 | 0.4095 | 0.5978 | 0.8853 | substantially reduces skewness; reduces heavy-tail behavior |
| `Box-Cox` | yes | -0.2952 | -0.2076 | -0.2068 | 0.2421 | 1.253 | -0.0024 | 0.0858 | 1.633 | -1.164 | 0.9530 | -0.7544 | -0.2068 | 0.3354 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Yeo-Johnson` | yes | -1.791 | 0.3669 | 0.3669 | 0.0346 | 1.253 | 0.0046 | -0.1077 | 1.600 | 0.2397 | 0.5129 | 0.2902 | 0.3669 | 0.4439 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |

### Transformation Comparison Plot

![ShockIdx transformation comparison](figures/target_distributions/transformations/ShockIdx_transform_compare.png)

## `PPV`

- target key: `PPV_t_plus_0m_gap`
- units: `%`
- train distribution summary: right-skewed; heavy-tailed; affected by extreme values
- split-shift summary: `no clear shift`
- transformation follow-up: investigate Box-Cox or log-type transform

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 252892 | 9516 | 3.63% |
| val | 35892 | 35036 | 856 | 2.38% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 0.3978 |
| max | 99.829 |
| mean | 14.598 |
| median | 8.707 |
| std | 15.296 |
| IQR | 13.366 |
| skewness | 2.101 |
| kurtosis (excess) | 4.687 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 1.412 | 2.208 | 4.782 | 8.707 | 18.148 | 48.641 | 73.260 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 21842 | 8.64% |
| low outliers | 0 | 0.00% |
| high outliers | 21842 | 8.64% |

Train outlier fences: lower `-15.267`, upper `38.197`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 252892 | 14.598 | 8.707 | 15.296 | 13.366 | 0.00% | 0.00% | 8.64% | 0.0000 | 0.0000 | reference |
| val | 35036 | 16.546 | 8.950 | 18.206 | 17.669 | 0.00% | 0.00% | 11.68% | 0.0571 | 0.0182 | no clear shift |
| test | 35701 | 14.702 | 8.628 | 15.241 | 12.552 | 0.00% | 0.00% | 9.24% | 0.0365 | 0.0059 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `21842` across `308` patients
- largest single-patient share of train outliers: `2.61%`
- top-5-patient share of train outliers: `12.19%`

### Plots

![PPV full histogram](figures/target_distributions/PPV_hist_full.png)

![PPV central histogram](figures/target_distributions/PPV_hist_central.png)

![PPV split boxplot](figures/target_distributions/PPV_boxplot.png)

![PPV ECDF](figures/target_distributions/PPV_ecdf.png)

### Short Interpretation

- `PPV` (%) has median `8.707` and IQR `13.366`; right-skewed; heavy-tailed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; investigate Box-Cox or log-type transform.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test Box-Cox + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 14.598 | 8.707 | 15.296 | 2.101 | 2.101 | 4.687 | 4.830 | 0.3978 | 99.829 | 1.412 | 8.707 | 73.260 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | -0.3851 | 1.000 | 2.101 | 2.101 | 4.687 | 4.830 | -0.9283 | 5.572 | -0.8620 | -0.3851 | 3.835 | scale changed only; shape unchanged as expected |
| `log1p` | yes | — | 2.382 | 2.273 | 0.8291 | 2.101 | 0.3881 | -0.5429 | 1.699 | 0.3349 | 4.613 | 0.8804 | 2.273 | 4.308 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Box-Cox` | yes | -0.0840 | 2.014 | 1.979 | 0.7691 | 2.101 | 0.0138 | -0.5090 | 1.470 | -0.9586 | 3.818 | 0.3399 | 1.979 | 3.605 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Yeo-Johnson` | yes | -0.2142 | 1.822 | 1.799 | 0.4892 | 2.101 | 0.0424 | -0.6796 | 1.397 | 0.3231 | 2.931 | 0.8023 | 1.799 | 2.813 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |

### Transformation Comparison Plot

![PPV transformation comparison](figures/target_distributions/transformations/PPV_transform_compare.png)

## `PVI`

- target key: `PVI_t_plus_0m_gap`
- units: `%`
- train distribution summary: right-skewed; affected by extreme values
- split-shift summary: `no clear shift`
- transformation follow-up: investigate Box-Cox or log-type transform

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 250422 | 11986 | 4.57% |
| val | 35892 | 34437 | 1455 | 4.05% |
| test | 36533 | 34839 | 1694 | 4.64% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 1.226 |
| max | 96.066 |
| mean | 24.861 |
| median | 20.570 |
| std | 16.220 |
| IQR | 21.532 |
| skewness | 1.067 |
| kurtosis (excess) | 0.7389 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 3.750 | 6.154 | 12.276 | 20.570 | 33.808 | 57.397 | 72.945 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 5829 | 2.33% |
| low outliers | 0 | 0.00% |
| high outliers | 5829 | 2.33% |

Train outlier fences: lower `-20.022`, upper `66.106`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 250422 | 24.861 | 20.570 | 16.220 | 21.532 | 0.00% | 0.00% | 2.33% | 0.0000 | 0.0000 | reference |
| val | 34437 | 24.709 | 19.995 | 16.894 | 22.422 | 0.00% | 0.00% | 2.90% | 0.0298 | 0.0267 | no clear shift |
| test | 34839 | 24.744 | 20.738 | 15.581 | 21.236 | 0.00% | 0.00% | 1.81% | 0.0156 | 0.0078 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `5829` across `143` patients
- largest single-patient share of train outliers: `6.30%`
- top-5-patient share of train outliers: `21.39%`

### Plots

![PVI full histogram](figures/target_distributions/PVI_hist_full.png)

![PVI central histogram](figures/target_distributions/PVI_hist_central.png)

![PVI split boxplot](figures/target_distributions/PVI_boxplot.png)

![PVI ECDF](figures/target_distributions/PVI_ecdf.png)

### Short Interpretation

- `PVI` (%) has median `20.570` and IQR `21.532`; right-skewed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; investigate Box-Cox or log-type transform.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test log1p + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 24.861 | 20.570 | 16.220 | 1.067 | 1.067 | 0.7389 | 2.432 | 1.226 | 96.066 | 3.750 | 20.570 | 72.945 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | -0.2645 | 1.000 | 1.067 | 1.067 | 0.7389 | 2.432 | -1.457 | 4.390 | -1.301 | -0.2645 | 2.965 | scale changed only; shape unchanged as expected |
| `log1p` | yes | — | 3.055 | 3.071 | 0.6473 | 1.067 | -0.1772 | -0.5604 | 1.278 | 0.8002 | 4.575 | 1.558 | 3.071 | 4.303 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Box-Cox` | yes | 0.1679 | 3.955 | 3.939 | 1.131 | 1.067 | -0.0229 | -0.6024 | 1.393 | 0.2073 | 6.861 | 1.480 | 3.939 | 6.282 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Yeo-Johnson` | yes | 0.1197 | 3.724 | 3.712 | 0.9298 | 1.067 | -0.0165 | -0.6434 | 1.377 | 0.8398 | 6.092 | 1.713 | 3.712 | 5.629 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |

### Transformation Comparison Plot

![PVI transformation comparison](figures/target_distributions/transformations/PVI_transform_compare.png)

## `PTT`

- target key: `PTT_t_plus_0m_gap`
- units: `ms (inferred)`
- train distribution summary: left-skewed; possible multimodality; affected by extreme values
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 251938 | 10470 | 3.99% |
| val | 35892 | 34871 | 1021 | 2.84% |
| test | 36533 | 35669 | 864 | 2.36% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 52.000 |
| max | 248.00 |
| mean | 192.60 |
| median | 200.00 |
| std | 36.826 |
| IQR | 47.413 |
| skewness | -0.9750 |
| kurtosis (excess) | 0.8978 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 75.478 | 122.94 | 172.33 | 200.00 | 219.74 | 240.00 | 248.00 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 5573 | 2.21% |
| low outliers | 5573 | 2.21% |
| high outliers | 0 | 0.00% |

Train outlier fences: lower `101.21`, upper `290.86`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 251938 | 192.60 | 200.00 | 36.826 | 47.413 | 0.00% | 0.00% | 2.21% | 0.0000 | 0.0000 | reference |
| val | 34871 | 187.71 | 198.31 | 40.542 | 56.428 | 0.00% | 0.00% | 4.08% | 0.0654 | 0.0356 | no clear shift |
| test | 35669 | 193.48 | 199.27 | 33.870 | 46.899 | 0.00% | 0.00% | 1.14% | 0.0351 | 0.0155 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `5573` across `100` patients
- largest single-patient share of train outliers: `7.34%`
- top-5-patient share of train outliers: `28.24%`

### Plots

![PTT full histogram](figures/target_distributions/PTT_hist_full.png)

![PTT central histogram](figures/target_distributions/PTT_hist_central.png)

![PTT split boxplot](figures/target_distributions/PTT_boxplot.png)

![PTT ECDF](figures/target_distributions/PTT_ecdf.png)

### Short Interpretation

- `PTT` (ms (inferred)) has median `200.00` and IQR `47.413`; left-skewed; possible multimodality; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test Yeo-Johnson + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 192.60 | 200.00 | 36.826 | -0.9750 | -0.9750 | 0.8978 | 1.012 | 52.000 | 248.00 | 75.478 | 200.00 | 248.00 | reference raw distribution |
| `Z-score` | yes | — | 0.0000 | 0.2009 | 1.0000 | -0.9750 | -0.9750 | 0.8978 | 1.012 | -3.818 | 1.504 | -3.180 | 0.2009 | 1.504 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | -0.9750 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | -0.9750 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 2.612 | 386,045.5 | 396,789.0 | 160,167.6 | -0.9750 | -0.1800 | -0.7105 | 1.253 | 12,204.5 | 694,170.8 | 31,804.6 | 396,789.0 | 694,170.8 | substantially reduces skewness; reduces heavy-tail behavior |

### Transformation Comparison Plot

![PTT transformation comparison](figures/target_distributions/transformations/PTT_transform_compare.png)

## `dPdt_max`

- target key: `dPdt_max_t_plus_0m_gap`
- units: `mmHg/s (inferred)`
- train distribution summary: right-skewed; affected by extreme values
- split-shift summary: `mild shift`
- transformation follow-up: investigate Box-Cox or log-type transform

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253026 | 9382 | 3.58% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35668 | 865 | 2.37% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 10.176 |
| max | 4,152.0 |
| mean | 1,013.8 |
| median | 954.51 |
| std | 444.97 |
| IQR | 545.27 |
| skewness | 0.8894 |
| kurtosis (excess) | 1.941 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 160.07 | 379.71 | 717.28 | 954.51 | 1,262.6 | 1,798.0 | 2,371.2 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 5296 | 2.09% |
| low outliers | 0 | 0.00% |
| high outliers | 5296 | 2.09% |

Train outlier fences: lower `-100.62`, upper `2,080.5`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253026 | 1,013.8 | 954.51 | 444.97 | 545.27 | 0.00% | 0.00% | 2.09% | 0.0000 | 0.0000 | reference |
| val | 35059 | 1,009.4 | 878.15 | 518.49 | 609.93 | 0.00% | 0.00% | 5.09% | 0.0824 | 0.1401 | mild shift |
| test | 35668 | 1,122.5 | 1,061.6 | 486.81 | 621.78 | 0.00% | 0.00% | 4.82% | 0.1023 | 0.1963 | mild shift |

### Extreme Values By Patient

- train Tukey outliers: `5296` across `67` patients
- largest single-patient share of train outliers: `23.28%`
- top-5-patient share of train outliers: `46.54%`

### Plots

![dPdt_max full histogram](figures/target_distributions/dPdt_max_hist_full.png)

![dPdt_max central histogram](figures/target_distributions/dPdt_max_hist_central.png)

![dPdt_max split boxplot](figures/target_distributions/dPdt_max_boxplot.png)

![dPdt_max ECDF](figures/target_distributions/dPdt_max_ecdf.png)

### Short Interpretation

- `dPdt_max` (mmHg/s (inferred)) has median `954.51` and IQR `545.27`; right-skewed; affected by extreme values; zeros `0.00%`; split comparison `mild shift`; investigate Box-Cox or log-type transform.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test Box-Cox + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 1,013.8 | 954.51 | 444.97 | 0.8894 | 0.8894 | 1.941 | 2.598 | 10.176 | 4,152.0 | 160.07 | 954.51 | 2,371.2 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | -0.1332 | 1.0000 | 0.8894 | 0.8894 | 1.941 | 2.598 | -2.255 | 7.053 | -1.919 | -0.1332 | 3.051 | scale changed only; shape unchanged as expected |
| `log1p` | yes | — | 6.813 | 6.862 | 0.5133 | 0.8894 | -1.445 | 5.361 | 1.610 | 2.414 | 8.332 | 5.082 | 6.862 | 7.772 | worsens skewness; increases heavy-tail behavior; compresses the upper tail |
| `Box-Cox` | yes | 0.5532 | 79.335 | 78.649 | 20.318 | 0.8894 | 0.0638 | 0.7587 | 2.087 | 4.716 | 179.65 | 28.154 | 78.649 | 131.29 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Yeo-Johnson` | yes | 0.5523 | 78.992 | 78.313 | 20.177 | 0.8894 | 0.0634 | 0.7570 | 2.086 | 5.057 | 178.57 | 28.162 | 78.313 | 130.58 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |

### Transformation Comparison Plot

![dPdt_max transformation comparison](figures/target_distributions/transformations/dPdt_max_transform_compare.png)

## `ABP_tau`

- target key: `ABP_tau_t_plus_0m_gap`
- units: `s (inferred)`
- train distribution summary: right-skewed; heavy-tailed; affected by extreme values
- split-shift summary: `mild shift`
- transformation follow-up: investigate Box-Cox or log-type transform

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 252881 | 9527 | 3.63% |
| val | 35892 | 35037 | 855 | 2.38% |
| test | 36533 | 35634 | 899 | 2.46% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 0.1127 |
| max | 9.901 |
| mean | 1.166 |
| median | 1.024 |
| std | 0.6726 |
| IQR | 0.6137 |
| skewness | 3.581 |
| kurtosis (excess) | 24.940 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2249 | 0.5140 | 0.7713 | 1.024 | 1.385 | 2.199 | 3.689 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 10829 | 4.28% |
| low outliers | 0 | 0.00% |
| high outliers | 10829 | 4.28% |

Train outlier fences: lower `-0.1493`, upper `2.306`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 252881 | 1.166 | 1.024 | 0.6726 | 0.6137 | 0.00% | 0.00% | 4.28% | 0.0000 | 0.0000 | reference |
| val | 35037 | 1.136 | 1.042 | 0.5929 | 0.5460 | 0.00% | 0.00% | 4.68% | 0.0432 | 0.0295 | no clear shift |
| test | 35634 | 1.089 | 0.9213 | 0.5810 | 0.5947 | 0.00% | 0.00% | 3.36% | 0.1061 | 0.1667 | mild shift |

### Extreme Values By Patient

- train Tukey outliers: `10829` across `225` patients
- largest single-patient share of train outliers: `5.14%`
- top-5-patient share of train outliers: `15.66%`

### Plots

![ABP_tau full histogram](figures/target_distributions/ABP_tau_hist_full.png)

![ABP_tau central histogram](figures/target_distributions/ABP_tau_hist_central.png)

![ABP_tau split boxplot](figures/target_distributions/ABP_tau_boxplot.png)

![ABP_tau ECDF](figures/target_distributions/ABP_tau_ecdf.png)

### Short Interpretation

- `ABP_tau` (s (inferred)) has median `1.024` and IQR `0.6137`; right-skewed; heavy-tailed; affected by extreme values; zeros `0.00%`; split comparison `mild shift`; investigate Box-Cox or log-type transform.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test Yeo-Johnson + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 1.166 | 1.024 | 0.6726 | 3.581 | 3.581 | 24.940 | 4.343 | 0.1127 | 9.901 | 0.2249 | 1.024 | 3.689 | reference raw distribution |
| `Z-score` | yes | — | 0.0000 | -0.2111 | 1.000 | 3.581 | 3.581 | 24.940 | 4.343 | -1.565 | 12.987 | -1.398 | -0.2111 | 3.751 | scale changed only; shape unchanged as expected |
| `log1p` | yes | — | 0.7377 | 0.7049 | 0.2506 | 3.581 | 1.132 | 3.438 | 2.825 | 0.1068 | 2.389 | 0.2029 | 0.7049 | 1.545 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Box-Cox` | yes | 0.0414 | 0.0368 | 0.0233 | 0.4879 | 3.581 | 0.0132 | 1.958 | 2.248 | -2.087 | 2.405 | -1.447 | 0.0233 | 1.341 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Yeo-Johnson` | yes | -0.9843 | 0.5105 | 0.5083 | 0.1136 | 3.581 | -0.0565 | 0.7796 | 1.944 | 0.1014 | 0.9192 | 0.1839 | 0.5083 | 0.7940 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |

### Transformation Comparison Plot

![ABP_tau transformation comparison](figures/target_distributions/transformations/ABP_tau_transform_compare.png)

## `RESP_amp`

- target key: `RESP_amp_t_plus_0m_gap`
- units: `resp AU`
- train distribution summary: right-skewed; heavy-tailed; affected by extreme values
- split-shift summary: `mild shift`
- transformation follow-up: investigate Box-Cox or log-type transform

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | 0.0281 |
| max | 8.163 |
| mean | 0.6386 |
| median | 0.6013 |
| std | 0.3743 |
| IQR | 0.5307 |
| skewness | 2.067 |
| kurtosis (excess) | 18.347 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0892 | 0.1517 | 0.3563 | 0.6013 | 0.8870 | 1.204 | 1.782 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 0 | 0.00% |
| Tukey outliers using train 1.5*IQR fences | 3171 | 1.25% |
| low outliers | 0 | 0.00% |
| high outliers | 3171 | 1.25% |

Train outlier fences: lower `-0.4398`, upper `1.683`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 0.6386 | 0.6013 | 0.3743 | 0.5307 | 0.00% | 0.00% | 1.25% | 0.0000 | 0.0000 | reference |
| val | 35059 | 0.5508 | 0.5108 | 0.3107 | 0.4809 | 0.00% | 0.00% | 0.49% | 0.0986 | 0.1705 | mild shift |
| test | 35701 | 0.6346 | 0.5970 | 0.3644 | 0.4779 | 0.00% | 0.00% | 1.59% | 0.0382 | 0.0081 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `3171` across `48` patients
- largest single-patient share of train outliers: `8.42%`
- top-5-patient share of train outliers: `38.76%`

### Plots

![RESP_amp full histogram](figures/target_distributions/RESP_amp_hist_full.png)

![RESP_amp central histogram](figures/target_distributions/RESP_amp_hist_central.png)

![RESP_amp split boxplot](figures/target_distributions/RESP_amp_boxplot.png)

![RESP_amp ECDF](figures/target_distributions/RESP_amp_ecdf.png)

### Short Interpretation

- `RESP_amp` (resp AU) has median `0.6013` and IQR `0.5307`; right-skewed; heavy-tailed; affected by extreme values; zeros `0.00%`; split comparison `mild shift`; investigate Box-Cox or log-type transform.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test Yeo-Johnson + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 0.6386 | 0.6013 | 0.3743 | 2.067 | 2.067 | 18.347 | 2.224 | 0.0281 | 8.163 | 0.0892 | 0.6013 | 1.782 | reference raw distribution |
| `Z-score` | yes | — | 0.0000 | -0.0995 | 1.000 | 2.067 | 2.067 | 18.347 | 2.224 | -1.631 | 20.100 | -1.468 | -0.0995 | 3.054 | scale changed only; shape unchanged as expected |
| `log1p` | yes | — | 0.4706 | 0.4708 | 0.2119 | 2.067 | 0.4136 | 0.8421 | 1.672 | 0.0277 | 2.215 | 0.0854 | 0.4708 | 1.023 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Box-Cox` | yes | 0.3832 | -0.5006 | -0.4621 | 0.4905 | 2.067 | -0.0007 | 0.3608 | 1.508 | -1.945 | 3.225 | -1.576 | -0.4621 | 0.6465 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |
| `Yeo-Johnson` | yes | -0.5961 | 0.4004 | 0.4105 | 0.1581 | 2.067 | 0.0083 | -0.3294 | 1.422 | 0.0275 | 1.230 | 0.0833 | 0.4105 | 0.7659 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |

### Transformation Comparison Plot

![RESP_amp transformation comparison](figures/target_distributions/transformations/RESP_amp_transform_compare.png)

## `PLETH_ACDC_PLETH_amp`

- target key: `PLETH_ACDC_PLETH_amp_t_plus_0m_gap`
- units: `correlation coefficient`
- train distribution summary: left-skewed; heavy-tailed; bounded with spike at +/-1; affected by extreme values
- split-shift summary: `no clear shift`
- transformation follow-up: bounded correlation feature; simple z-score may be insufficient, but log/Box-Cox are not appropriate

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | -1.000 |
| max | 1.000 |
| mean | 0.9099 |
| median | 0.9827 |
| std | 0.1870 |
| IQR | 0.0871 |
| skewness | -3.653 |
| kurtosis (excess) | 15.712 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.5604 | 0.9129 | 0.9827 | 1.000 | 1.000 | 1.000 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 2632 | 1.04% |
| negative values | 1384 | 0.55% |
| Tukey outliers using train 1.5*IQR fences | 29549 | 11.68% |
| low outliers | 29549 | 11.68% |
| high outliers | 0 | 0.00% |

Train outlier fences: lower `0.7823`, upper `1.131`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 0.9099 | 0.9827 | 0.1870 | 0.0871 | 1.04% | 0.55% | 11.68% | 0.0000 | 0.0000 | reference |
| val | 35059 | 0.9126 | 0.9864 | 0.1889 | 0.0789 | 1.77% | 0.27% | 10.96% | 0.0355 | 0.0426 | no clear shift |
| test | 35701 | 0.9027 | 0.9838 | 0.2144 | 0.0814 | 2.41% | 0.67% | 11.25% | 0.0155 | 0.0119 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `29549` across `553` patients
- largest single-patient share of train outliers: `1.92%`
- top-5-patient share of train outliers: `8.67%`

### Plots

![PLETH_ACDC_PLETH_amp full histogram](figures/target_distributions/PLETH_ACDC_PLETH_amp_hist_full.png)

![PLETH_ACDC_PLETH_amp central histogram](figures/target_distributions/PLETH_ACDC_PLETH_amp_hist_central.png)

![PLETH_ACDC_PLETH_amp split boxplot](figures/target_distributions/PLETH_ACDC_PLETH_amp_boxplot.png)

![PLETH_ACDC_PLETH_amp ECDF](figures/target_distributions/PLETH_ACDC_PLETH_amp_ecdf.png)

### Short Interpretation

- `PLETH_ACDC_PLETH_amp` (correlation coefficient) has median `0.9827` and IQR `0.0871`; left-skewed; heavy-tailed; bounded with spike at +/-1; affected by extreme values; zeros `1.04%`; split comparison `no clear shift`; bounded correlation feature; simple z-score may be insufficient, but log/Box-Cox are not appropriate. includes noticeable mass at +/-1.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test z-score only`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 0.9099 | 0.9827 | 0.1870 | -3.653 | -3.653 | 15.712 | 0.1983 | -1.000 | 1.000 | 0.0000 | 0.9827 | 1.000 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | 0.3895 | 1.0000 | -3.653 | -3.653 | 15.712 | 0.1983 | -10.214 | 0.4819 | -4.866 | 0.3895 | 0.4819 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | -3.653 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | -3.653 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 14.222 | 982.08 | 1,187.9 | 436.72 | -3.653 | -1.021 | -0.3272 | 0.2473 | -0.0818 | 1,343.8 | 0.0000 | 1,187.9 | 1,343.8 | substantially reduces skewness; reduces heavy-tail behavior; introduces an extreme learned lambda; expands the bounded scale aggressively |

### Transformation Comparison Plot

![PLETH_ACDC_PLETH_amp transformation comparison](figures/target_distributions/transformations/PLETH_ACDC_PLETH_amp_transform_compare.png)

## `ABP_area_ABP_tau`

- target key: `ABP_area_ABP_tau_t_plus_0m_gap`
- units: `correlation coefficient`
- train distribution summary: approximately symmetric
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | -1.000 |
| max | 1.000 |
| mean | -0.2518 |
| median | -0.3854 |
| std | 0.5867 |
| IQR | 1.029 |
| skewness | 0.4663 |
| kurtosis (excess) | -1.089 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| -0.9941 | -0.9650 | -0.7877 | -0.3854 | 0.2414 | 0.7962 | 0.9312 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 177 | 0.07% |
| negative values | 166816 | 65.92% |
| Tukey outliers using train 1.5*IQR fences | 0 | 0.00% |
| low outliers | 0 | 0.00% |
| high outliers | 0 | 0.00% |

Train outlier fences: lower `-2.332`, upper `1.785`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | -0.2518 | -0.3854 | 0.5867 | 1.029 | 0.07% | 65.92% | 0.00% | 0.0000 | 0.0000 | reference |
| val | 35059 | -0.2063 | -0.2990 | 0.5800 | 1.035 | 0.07% | 63.00% | 0.00% | 0.0440 | 0.0840 | no clear shift |
| test | 35701 | -0.2368 | -0.3681 | 0.5993 | 1.071 | 0.19% | 64.51% | 0.00% | 0.0152 | 0.0169 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `0` across `0` patients
- largest single-patient share of train outliers: `0.00%`
- top-5-patient share of train outliers: `0.00%`

### Plots

![ABP_area_ABP_tau full histogram](figures/target_distributions/ABP_area_ABP_tau_hist_full.png)

![ABP_area_ABP_tau central histogram](figures/target_distributions/ABP_area_ABP_tau_hist_central.png)

![ABP_area_ABP_tau split boxplot](figures/target_distributions/ABP_area_ABP_tau_boxplot.png)

![ABP_area_ABP_tau ECDF](figures/target_distributions/ABP_area_ABP_tau_ecdf.png)

### Short Interpretation

- `ABP_area_ABP_tau` (correlation coefficient) has median `-0.3854` and IQR `1.029`; approximately symmetric; zeros `0.07%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test z-score only`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | -0.2518 | -0.3854 | 0.5867 | 0.4663 | 0.4663 | -1.089 | 1.279 | -1.000 | 1.000 | -0.9941 | -0.3854 | 0.9312 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | -0.2278 | 1.000 | 0.4663 | 0.4663 | -1.089 | 1.279 | -1.275 | 2.133 | -1.265 | -0.2278 | 2.016 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | 0.4663 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | 0.4663 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 0.3675 | -0.3649 | -0.4305 | 0.6398 | 0.4663 | 0.1848 | -1.344 | 0.9841 | -1.287 | 0.7894 | -1.278 | -0.4305 | 0.7445 | moderately reduces skewness |

### Transformation Comparison Plot

![ABP_area_ABP_tau transformation comparison](figures/target_distributions/transformations/ABP_area_ABP_tau_transform_compare.png)

## `ABP_area_ShockIdx`

- target key: `ABP_area_ShockIdx_t_plus_0m_gap`
- units: `correlation coefficient`
- train distribution summary: right-skewed; bounded with spike at +/-1; affected by extreme values
- split-shift summary: `no clear shift`
- transformation follow-up: bounded correlation feature; simple z-score may be insufficient, but log/Box-Cox are not appropriate

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | -1.000 |
| max | 1.000 |
| mean | -0.5242 |
| median | -0.7252 |
| std | 0.4987 |
| IQR | 0.6569 |
| skewness | 1.176 |
| kurtosis (excess) | 0.3748 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| -1.000 | -0.9890 | -0.9179 | -0.7252 | -0.2610 | 0.5729 | 0.8543 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 0 | 0.00% |
| negative values | 210327 | 83.12% |
| Tukey outliers using train 1.5*IQR fences | 6855 | 2.71% |
| low outliers | 0 | 0.00% |
| high outliers | 6855 | 2.71% |

Train outlier fences: lower `-1.903`, upper `0.7244`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | -0.5242 | -0.7252 | 0.4987 | 0.6569 | 0.00% | 83.12% | 2.71% | 0.0000 | 0.0000 | reference |
| val | 35059 | -0.5156 | -0.6911 | 0.4814 | 0.6478 | 0.00% | 83.67% | 2.31% | 0.0400 | 0.0519 | no clear shift |
| test | 35701 | -0.4779 | -0.6649 | 0.5202 | 0.7300 | 0.00% | 80.75% | 3.45% | 0.0482 | 0.0918 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `6855` across `431` patients
- largest single-patient share of train outliers: `2.29%`
- top-5-patient share of train outliers: `9.79%`

### Plots

![ABP_area_ShockIdx full histogram](figures/target_distributions/ABP_area_ShockIdx_hist_full.png)

![ABP_area_ShockIdx central histogram](figures/target_distributions/ABP_area_ShockIdx_hist_central.png)

![ABP_area_ShockIdx split boxplot](figures/target_distributions/ABP_area_ShockIdx_boxplot.png)

![ABP_area_ShockIdx ECDF](figures/target_distributions/ABP_area_ShockIdx_ecdf.png)

### Short Interpretation

- `ABP_area_ShockIdx` (correlation coefficient) has median `-0.7252` and IQR `0.6569`; right-skewed; bounded with spike at +/-1; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; bounded correlation feature; simple z-score may be insufficient, but log/Box-Cox are not appropriate. includes noticeable mass at +/-1.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test Yeo-Johnson + z-score`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | -0.5242 | -0.7252 | 0.4987 | 1.176 | 1.176 | 0.3748 | 2.404 | -1.000 | 1.000 | -1.000 | -0.7252 | 0.8543 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | -0.4031 | 1.000 | 1.176 | 1.176 | 0.3748 | 2.404 | -0.9541 | 3.056 | -0.9541 | -0.4031 | 2.764 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | 1.176 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | 1.176 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | -0.7751 | -1.067 | -1.276 | 0.8347 | 1.176 | 0.4444 | -1.203 | 1.170 | -2.106 | 0.5363 | -2.106 | -1.276 | 0.4907 | substantially reduces skewness; reduces heavy-tail behavior; compresses the upper tail |

### Transformation Comparison Plot

![ABP_area_ShockIdx transformation comparison](figures/target_distributions/transformations/ABP_area_ShockIdx_transform_compare.png)

## `PLETH_amp_ShockIdx`

- target key: `PLETH_amp_ShockIdx_t_plus_0m_gap`
- units: `correlation coefficient`
- train distribution summary: approximately symmetric
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | -1.000 |
| max | 1.000 |
| mean | 0.0334 |
| median | 0.0337 |
| std | 0.4614 |
| IQR | 0.7236 |
| skewness | -0.0496 |
| kurtosis (excess) | -0.8936 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| -0.8817 | -0.7279 | -0.3245 | 0.0337 | 0.3990 | 0.7727 | 0.9073 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 2630 | 1.04% |
| negative values | 117826 | 46.56% |
| Tukey outliers using train 1.5*IQR fences | 0 | 0.00% |
| low outliers | 0 | 0.00% |
| high outliers | 0 | 0.00% |

Train outlier fences: lower `-1.410`, upper `1.484`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 0.0334 | 0.0337 | 0.4614 | 0.7236 | 1.04% | 46.56% | 0.00% | 0.0000 | 0.0000 | reference |
| val | 35059 | 0.0257 | 0.0111 | 0.4624 | 0.7240 | 1.77% | 47.38% | 0.00% | 0.0163 | 0.0313 | no clear shift |
| test | 35701 | 0.0045 | 0.0000 | 0.4671 | 0.7212 | 2.41% | 48.41% | 0.00% | 0.0325 | 0.0466 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `0` across `0` patients
- largest single-patient share of train outliers: `0.00%`
- top-5-patient share of train outliers: `0.00%`

### Plots

![PLETH_amp_ShockIdx full histogram](figures/target_distributions/PLETH_amp_ShockIdx_hist_full.png)

![PLETH_amp_ShockIdx central histogram](figures/target_distributions/PLETH_amp_ShockIdx_hist_central.png)

![PLETH_amp_ShockIdx split boxplot](figures/target_distributions/PLETH_amp_ShockIdx_boxplot.png)

![PLETH_amp_ShockIdx ECDF](figures/target_distributions/PLETH_amp_ShockIdx_ecdf.png)

### Short Interpretation

- `PLETH_amp_ShockIdx` (correlation coefficient) has median `0.0337` and IQR `0.7236`; approximately symmetric; zeros `1.04%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test z-score only`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 0.0334 | 0.0337 | 0.4614 | -0.0496 | -0.0496 | -0.8936 | 1.207 | -1.000 | 1.000 | -0.8817 | 0.0337 | 0.9073 | reference raw distribution |
| `Z-score` | yes | — | 0.0000 | 0.0005 | 1.0000 | -0.0496 | -0.0496 | -0.8936 | 1.207 | -2.240 | 2.095 | -1.983 | 0.0005 | 1.894 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | -0.0496 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | -0.0496 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 1.062 | 0.0390 | 0.0337 | 0.4620 | -0.0496 | -0.0117 | -0.8977 | 1.233 | -0.9763 | 1.024 | -0.8628 | 0.0337 | 0.9277 | little shape change |

### Transformation Comparison Plot

![PLETH_amp_ShockIdx transformation comparison](figures/target_distributions/transformations/PLETH_amp_ShockIdx_transform_compare.png)

## `PLETH_ACDC_ShockIdx`

- target key: `PLETH_ACDC_ShockIdx_t_plus_0m_gap`
- units: `correlation coefficient`
- train distribution summary: approximately symmetric
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | -1.0000 |
| max | 1.000 |
| mean | 0.0800 |
| median | 0.0949 |
| std | 0.4808 |
| IQR | 0.7668 |
| skewness | -0.1347 |
| kurtosis (excess) | -0.9422 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| -0.8861 | -0.7302 | -0.2947 | 0.0949 | 0.4721 | 0.8232 | 0.9311 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 2632 | 1.04% |
| negative values | 108164 | 42.74% |
| Tukey outliers using train 1.5*IQR fences | 0 | 0.00% |
| low outliers | 0 | 0.00% |
| high outliers | 0 | 0.00% |

Train outlier fences: lower `-1.445`, upper `1.622`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 0.0800 | 0.0949 | 0.4808 | 0.7668 | 1.04% | 42.74% | 0.00% | 0.0000 | 0.0000 | reference |
| val | 35059 | 0.0714 | 0.0679 | 0.4796 | 0.7619 | 1.77% | 43.85% | 0.00% | 0.0189 | 0.0352 | no clear shift |
| test | 35701 | 0.0512 | 0.0457 | 0.4798 | 0.7519 | 2.41% | 44.69% | 0.00% | 0.0339 | 0.0642 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `0` across `0` patients
- largest single-patient share of train outliers: `0.00%`
- top-5-patient share of train outliers: `0.00%`

### Plots

![PLETH_ACDC_ShockIdx full histogram](figures/target_distributions/PLETH_ACDC_ShockIdx_hist_full.png)

![PLETH_ACDC_ShockIdx central histogram](figures/target_distributions/PLETH_ACDC_ShockIdx_hist_central.png)

![PLETH_ACDC_ShockIdx split boxplot](figures/target_distributions/PLETH_ACDC_ShockIdx_boxplot.png)

![PLETH_ACDC_ShockIdx ECDF](figures/target_distributions/PLETH_ACDC_ShockIdx_ecdf.png)

### Short Interpretation

- `PLETH_ACDC_ShockIdx` (correlation coefficient) has median `0.0949` and IQR `0.7668`; approximately symmetric; zeros `1.04%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test z-score only`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 0.0800 | 0.0949 | 0.4808 | -0.1347 | -0.1347 | -0.9422 | 1.091 | -1.0000 | 1.000 | -0.8861 | 0.0949 | 0.9311 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | 0.0310 | 1.000 | -0.1347 | -0.1347 | -0.9422 | 1.091 | -2.246 | 1.913 | -2.009 | 0.0310 | 1.770 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | -0.1347 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | -0.1347 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 1.171 | 0.0971 | 0.0956 | 0.4849 | -0.1347 | -0.0332 | -0.9709 | 1.153 | -0.9366 | 1.069 | -0.8349 | 0.0956 | 0.9916 | little shape change |

### Transformation Comparison Plot

![PLETH_ACDC_ShockIdx transformation comparison](figures/target_distributions/transformations/PLETH_ACDC_ShockIdx_transform_compare.png)

## `ShockIdx_ABP_tau`

- target key: `ShockIdx_ABP_tau_t_plus_0m_gap`
- units: `correlation coefficient`
- train distribution summary: approximately symmetric
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | -1.000 |
| max | 1.000 |
| mean | 0.2301 |
| median | 0.3267 |
| std | 0.5419 |
| IQR | 0.9001 |
| skewness | -0.4510 |
| kurtosis (excess) | -0.9527 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| -0.9158 | -0.7549 | -0.1931 | 0.3267 | 0.7070 | 0.9335 | 0.9813 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 177 | 0.07% |
| negative values | 84523 | 33.40% |
| Tukey outliers using train 1.5*IQR fences | 0 | 0.00% |
| low outliers | 0 | 0.00% |
| high outliers | 0 | 0.00% |

Train outlier fences: lower `-1.543`, upper `2.057`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 0.2301 | 0.3267 | 0.5419 | 0.9001 | 0.07% | 33.40% | 0.00% | 0.0000 | 0.0000 | reference |
| val | 35059 | 0.2134 | 0.3043 | 0.5425 | 0.9148 | 0.07% | 34.79% | 0.00% | 0.0144 | 0.0249 | no clear shift |
| test | 35701 | 0.2441 | 0.3432 | 0.5457 | 0.8918 | 0.19% | 32.34% | 0.00% | 0.0164 | 0.0184 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `0` across `0` patients
- largest single-patient share of train outliers: `0.00%`
- top-5-patient share of train outliers: `0.00%`

### Plots

![ShockIdx_ABP_tau full histogram](figures/target_distributions/ShockIdx_ABP_tau_hist_full.png)

![ShockIdx_ABP_tau central histogram](figures/target_distributions/ShockIdx_ABP_tau_hist_central.png)

![ShockIdx_ABP_tau split boxplot](figures/target_distributions/ShockIdx_ABP_tau_boxplot.png)

![ShockIdx_ABP_tau ECDF](figures/target_distributions/ShockIdx_ABP_tau_ecdf.png)

### Short Interpretation

- `ShockIdx_ABP_tau` (correlation coefficient) has median `0.3267` and IQR `0.9001`; approximately symmetric; zeros `0.07%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test z-score only`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 0.2301 | 0.3267 | 0.5419 | -0.4510 | -0.4510 | -0.9527 | 0.7273 | -1.000 | 1.000 | -0.9158 | 0.3267 | 0.9813 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | 0.1781 | 1.000 | -0.4510 | -0.4510 | -0.9527 | 0.7273 | -2.270 | 1.421 | -2.115 | 0.1781 | 1.386 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | -0.4510 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | -0.4510 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 1.602 | 0.3216 | 0.3575 | 0.5855 | -0.4510 | -0.1456 | -1.212 | 0.8600 | -0.7982 | 1.271 | -0.7420 | 0.3575 | 1.242 | moderately reduces skewness |

### Transformation Comparison Plot

![ShockIdx_ABP_tau transformation comparison](figures/target_distributions/transformations/ShockIdx_ABP_tau_transform_compare.png)

## `PLETH_ACDC_ABP_tau`

- target key: `PLETH_ACDC_ABP_tau_t_plus_0m_gap`
- units: `correlation coefficient`
- train distribution summary: approximately symmetric
- split-shift summary: `no clear shift`
- transformation follow-up: simple z-score normalization is likely enough if any normalization is used

### Valid / Missing Counts

| Split | Total anchors | Valid | Missing | Missing % |
|---|---:|---:|---:|---:|
| train | 262408 | 253053 | 9355 | 3.57% |
| val | 35892 | 35059 | 833 | 2.32% |
| test | 36533 | 35701 | 832 | 2.28% |

### Train Statistics

| Statistic | Value |
|---|---:|
| min | -0.9969 |
| max | 1.000 |
| mean | 0.1176 |
| median | 0.1359 |
| std | 0.4674 |
| IQR | 0.7314 |
| skewness | -0.1942 |
| kurtosis (excess) | -0.8643 |

### Train Percentiles

| 1st | 5th | 25th | 50th | 75th | 95th | 99th |
|---:|---:|---:|---:|---:|---:|---:|
| -0.8678 | -0.6869 | -0.2329 | 0.1359 | 0.4985 | 0.8302 | 0.9325 |

### Train Zero / Negative / Outlier Counts

| Metric | Count | Percent of valid train values |
|---|---:|---:|
| exact zeros | 2809 | 1.11% |
| negative values | 100030 | 39.53% |
| Tukey outliers using train 1.5*IQR fences | 0 | 0.00% |
| low outliers | 0 | 0.00% |
| high outliers | 0 | 0.00% |

Train outlier fences: lower `-1.330`, upper `1.596`.

### Split Comparison

| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 253053 | 0.1176 | 0.1359 | 0.4674 | 0.7314 | 1.11% | 39.53% | 0.00% | 0.0000 | 0.0000 | reference |
| val | 35059 | 0.1116 | 0.1191 | 0.4650 | 0.7259 | 1.84% | 39.87% | 0.00% | 0.0135 | 0.0230 | no clear shift |
| test | 35701 | 0.0672 | 0.0643 | 0.4708 | 0.7250 | 2.60% | 43.02% | 0.00% | 0.0503 | 0.0979 | no clear shift |

### Extreme Values By Patient

- train Tukey outliers: `0` across `0` patients
- largest single-patient share of train outliers: `0.00%`
- top-5-patient share of train outliers: `0.00%`

### Plots

![PLETH_ACDC_ABP_tau full histogram](figures/target_distributions/PLETH_ACDC_ABP_tau_hist_full.png)

![PLETH_ACDC_ABP_tau central histogram](figures/target_distributions/PLETH_ACDC_ABP_tau_hist_central.png)

![PLETH_ACDC_ABP_tau split boxplot](figures/target_distributions/PLETH_ACDC_ABP_tau_boxplot.png)

![PLETH_ACDC_ABP_tau ECDF](figures/target_distributions/PLETH_ACDC_ABP_tau_ecdf.png)

### Short Interpretation

- `PLETH_ACDC_ABP_tau` (correlation coefficient) has median `0.1359` and IQR `0.7314`; approximately symmetric; zeros `1.11%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Transformation Analysis

- recommendation for future experiments: `keep raw target as baseline; test z-score only`

| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Raw` | yes | — | 0.1176 | 0.1359 | 0.4674 | -0.1942 | -0.1942 | -0.8643 | 1.089 | -0.9969 | 1.000 | -0.8678 | 0.1359 | 0.9325 | reference raw distribution |
| `Z-score` | yes | — | -0.0000 | 0.0393 | 1.0000 | -0.1942 | -0.1942 | -0.8643 | 1.089 | -2.384 | 1.888 | -2.108 | 0.0393 | 1.744 | scale changed only; shape unchanged as expected |
| `log1p` | no | — | — | — | — | -0.1942 | — | — | — | — | — | — | — | — | skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like |
| `Box-Cox` | no | — | — | — | — | -0.1942 | — | — | — | — | — | — | — | — | skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox |
| `Yeo-Johnson` | yes | 1.245 | 0.1417 | 0.1381 | 0.4759 | -0.1942 | -0.0462 | -0.9292 | 1.173 | -0.9082 | 1.100 | -0.7983 | 0.1381 | 1.021 | little shape change |

### Transformation Comparison Plot

![PLETH_ACDC_ABP_tau transformation comparison](figures/target_distributions/transformations/PLETH_ACDC_ABP_tau_transform_compare.png)

## Overall Interpretation

### Per-Target Interpretation

- `HR` (bpm) has median `101.75` and IQR `21.761`; approximately symmetric; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `RR` (breaths/min) has median `26.771` and IQR `3.041`; approximately symmetric; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `SBP` (mmHg) has median `125.93` and IQR `31.702`; approximately symmetric; train outliers concentrated in a few patients; zeros `0.00%`; split comparison `mild shift`; simple z-score normalization is likely enough if any normalization is used.
- `DBP` (mmHg) has median `61.060` and IQR `17.163`; right-skewed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; investigate Box-Cox or log-type transform.
- `PP` (mmHg) has median `64.143` and IQR `27.105`; approximately symmetric; zeros `0.00%`; split comparison `mild shift`; simple z-score normalization is likely enough if any normalization is used.
- `MAP` (mmHg) has median `83.301` and IQR `19.883`; right-skewed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `ABP_area` (mmHg*s (inferred)) has median `15.426` and IQR `7.726`; right-skewed; affected by extreme values; zeros `0.00%`; split comparison `possible meaningful shift`; simple z-score normalization is likely enough if any normalization is used.
- `PLETH_ACDC` (ratio) has median `0.9606` and IQR `0.2925`; left-skewed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `PLETH_amp` (pleth AU) has median `1.663` and IQR `0.9028`; left-skewed; possible multimodality; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `ECG_Ramp` (ECG AU) has median `0.3354` and IQR `0.5163`; right-skewed; possible multimodality; affected by extreme values; zeros `0.14%`; split comparison `no clear shift`; investigate log1p.
- `HRV_RMSSD` (ms (inferred)) has median `94.578` and IQR `112.53`; approximately symmetric; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `HR_range` (bpm) has median `71.300` and IQR `29.919`; left-skewed; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `ShockIdx` (ratio) has median `0.8181` and IQR `0.2583`; right-skewed; heavy-tailed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; investigate Box-Cox or log-type transform.
- `PPV` (%) has median `8.707` and IQR `13.366`; right-skewed; heavy-tailed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; investigate Box-Cox or log-type transform.
- `PVI` (%) has median `20.570` and IQR `21.532`; right-skewed; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; investigate Box-Cox or log-type transform.
- `PTT` (ms (inferred)) has median `200.00` and IQR `47.413`; left-skewed; possible multimodality; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `dPdt_max` (mmHg/s (inferred)) has median `954.51` and IQR `545.27`; right-skewed; affected by extreme values; zeros `0.00%`; split comparison `mild shift`; investigate Box-Cox or log-type transform.
- `ABP_tau` (s (inferred)) has median `1.024` and IQR `0.6137`; right-skewed; heavy-tailed; affected by extreme values; zeros `0.00%`; split comparison `mild shift`; investigate Box-Cox or log-type transform.
- `RESP_amp` (resp AU) has median `0.6013` and IQR `0.5307`; right-skewed; heavy-tailed; affected by extreme values; zeros `0.00%`; split comparison `mild shift`; investigate Box-Cox or log-type transform.
- `PLETH_ACDC_PLETH_amp` (correlation coefficient) has median `0.9827` and IQR `0.0871`; left-skewed; heavy-tailed; bounded with spike at +/-1; affected by extreme values; zeros `1.04%`; split comparison `no clear shift`; bounded correlation feature; simple z-score may be insufficient, but log/Box-Cox are not appropriate. includes noticeable mass at +/-1.
- `ABP_area_ABP_tau` (correlation coefficient) has median `-0.3854` and IQR `1.029`; approximately symmetric; zeros `0.07%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `ABP_area_ShockIdx` (correlation coefficient) has median `-0.7252` and IQR `0.6569`; right-skewed; bounded with spike at +/-1; affected by extreme values; zeros `0.00%`; split comparison `no clear shift`; bounded correlation feature; simple z-score may be insufficient, but log/Box-Cox are not appropriate. includes noticeable mass at +/-1.
- `PLETH_amp_ShockIdx` (correlation coefficient) has median `0.0337` and IQR `0.7236`; approximately symmetric; zeros `1.04%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `PLETH_ACDC_ShockIdx` (correlation coefficient) has median `0.0949` and IQR `0.7668`; approximately symmetric; zeros `1.04%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `ShockIdx_ABP_tau` (correlation coefficient) has median `0.3267` and IQR `0.9001`; approximately symmetric; zeros `0.07%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.
- `PLETH_ACDC_ABP_tau` (correlation coefficient) has median `0.1359` and IQR `0.7314`; approximately symmetric; zeros `1.11%`; split comparison `no clear shift`; simple z-score normalization is likely enough if any normalization is used.

### Recommended Future Tests

- `HR`: keep raw target as baseline; transformation probably unnecessary
- `RR`: keep raw target as baseline; transformation probably unnecessary
- `SBP`: keep raw target as baseline; transformation probably unnecessary
- `DBP`: keep raw target as baseline; test log1p + z-score
- `PP`: keep raw target as baseline; transformation probably unnecessary
- `MAP`: keep raw target as baseline; test log1p + z-score
- `ABP_area`: keep raw target as baseline; test Box-Cox + z-score
- `PLETH_ACDC`: keep raw target as baseline; test Yeo-Johnson + z-score
- `PLETH_amp`: keep raw target as baseline; test z-score only
- `ECG_Ramp`: keep raw target as baseline; test Yeo-Johnson + z-score
- `HRV_RMSSD`: keep raw target as baseline; transformation probably unnecessary
- `HR_range`: keep raw target as baseline; test z-score only
- `ShockIdx`: keep raw target as baseline; test Box-Cox + z-score
- `PPV`: keep raw target as baseline; test Box-Cox + z-score
- `PVI`: keep raw target as baseline; test log1p + z-score
- `PTT`: keep raw target as baseline; test Yeo-Johnson + z-score
- `dPdt_max`: keep raw target as baseline; test Box-Cox + z-score
- `ABP_tau`: keep raw target as baseline; test Yeo-Johnson + z-score
- `RESP_amp`: keep raw target as baseline; test Yeo-Johnson + z-score
- `PLETH_ACDC_PLETH_amp`: keep raw target as baseline; test z-score only
- `ABP_area_ABP_tau`: keep raw target as baseline; test z-score only
- `ABP_area_ShockIdx`: keep raw target as baseline; test Yeo-Johnson + z-score
- `PLETH_amp_ShockIdx`: keep raw target as baseline; test z-score only
- `PLETH_ACDC_ShockIdx`: keep raw target as baseline; test z-score only
- `ShockIdx_ABP_tau`: keep raw target as baseline; test z-score only
- `PLETH_ACDC_ABP_tau`: keep raw target as baseline; test z-score only

- Targets with strong positive skew or heavy upper tails are the clearest candidates for future nonlinear transform checks.
- Targets with negative support are poor Box-Cox candidates and, if transformation is revisited later, are better matched to Yeo-Johnson or simple z-scoring.
- The bounded correlation targets deserve separate review because they already live on a constrained `[-1, 1]` scale with some boundary mass.
- Split comparisons should be read as a guardrail against learning a train-specific transform that does not transfer cleanly to validation or test.

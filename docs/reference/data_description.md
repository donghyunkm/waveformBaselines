# Data Description

This document describes the waveform inputs, cohort structure, target bundles, and normalization used by the current supervised PatchTST training pipeline.

## Waveform Inputs

Main waveform cache:

- `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms`

Upstream sources:

- raw matched waveforms: `/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched/`
- derived numerics/features: `/gpfs/data/eh3828lab/derived_datasets/baselines/output_v2/`

Current canonical training window:

- length: `20` minutes
- sampling rate: `125 Hz`
- samples per channel: `150000`

Raw channel order on disk:

- `II`, `ABP`, `PLETH`, `RESP`

Common default model channel order:

- `ABP`, `II`, `PLETH`

Current channel-count convention:

- `patchtst_v1`: `3` channels, `ABP,II,PLETH`
- `patchtst_v1_5`: `3` channels, `ABP,II,PLETH`
- `patchtst_v2`: the repo's `4`-channel experimental variant, `II,PLETH,ABP,RESP`

The raw waveform cache stores `4` channels on disk. Current `v1` and `v1.5` training and evaluation results use only the `3`-channel subset above. Treat `v2` as the repo's `4`-channel path unless a run config explicitly overrides `--channels`.

Training uses pre-extracted per-patient `.npy` waveforms with memory-mapped reads through `waveform_baselines/numpy_dataset.py`.

## Splits and Cohorts

### Standard cohort

- splits file: `outputs/splits/splits.json`
- standard target bundle: `outputs/targets/all_targets.npz`

### Vasopressor-free overlap cohort

- splits file: `outputs/splits/vasopressor_free_splits.json`
- anchors: `outputs/targets/vasopressor_free_overlap_anchors.csv`

Current vasopressor-free overlap cohort size:

- patients: `887`
- anchor rows/windows: `334833`
- train: `692` patients, `263894` windows
- val: `98` patients, `36026` windows
- test: `97` patients, `36645` windows

This vasopressor-free cohort is the main setting used for the recent `v1` reruns and rebuilt event labels.

## Normalization

The current normalization scheme is split-specific shared per-channel z-scoring applied at dataset load time.

Important points:

- normalization is external to the model
- statistics are computed from the train split
- the same train-split stats are reused for val and test
- normalization is shared across windows, not per-patient and not per-window
- no RevIN or model-internal waveform normalization is applied

Relevant files:

- loader: `waveform_baselines/normalization.py`
- stats script: `scripts/compute_waveform_normalization_stats.py`
- standard stats example: `normalization_stats_splits.json`
- vasopressor-free stats file: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms/normalization_stats_vasopressor_free_splits.json`

This is the "new normalization scheme" currently referenced in the recent training runs: split-specific shared waveform normalization, replacing older looser normalization assumptions.

## Training Sample Filtering

The supervised training dataset is not simply "all windows in the split."

Current filtering behavior:

- waveform windows come from the split-specific anchor list
- targets are loaded from a target bundle
- invalid target entries are removed before training
- `SingleTargetDataset` filters out windows whose selected target is masked/invalid
- batching is patient-grouped through `waveform_baselines/patient_sampler.py`

As a result, the effective number of training windows depends on:

- cohort
- task type
- selected target
- horizon
- target-generation mode

## Regression Targets

Regression targets are scalar physiological or cross-signal feature values predicted at future horizons.

Source arrays:

- `X_stats.npy`
- `corr_features_focused.npy`

Base target count:

- `26`

Target families:

- waveform/vital features such as `HR`, `RR`, `SBP`, `DBP`, `PP`, `MAP`
- morphology/derived features such as `ABP_area`, `PLETH_ACDC`, `PLETH_amp`, `ECG_Ramp`
- variability/timing/derived features such as `HRV_RMSSD`, `HR_range`, `ShockIdx`, `PPV`, `PVI`, `PTT`, `dPdt_max`, `ABP_tau`, `RESP_amp`
- correlation / interaction features such as `PLETH_ACDC_PLETH_amp`, `ABP_area_ABP_tau`, `ABP_area_ShockIdx`, `PLETH_amp_ShockIdx`, `PLETH_ACDC_ShockIdx`, `ShockIdx_ABP_tau`, `PLETH_ACDC_ABP_tau`

Configured horizons:

- `0`
- `20`
- `60` minutes

Total regression targets:

- `78`

### Regression target semantics

Two bundle semantics exist, but only gap mode is suitable for new
leakage-safe regression runs:

- standard bundle behavior: `center` mode; the `t+0` target is aligned to the
  same anchor as the input and overlaps the 20-minute input window, so those
  `t+0` regression results are legacy/invalid for leakage-free reporting
- vasopressor-free gap-mode behavior: the target window begins after the input
  window and is the current regression setup

For the vasopressor-free gap-mode bundle:

- input window: `[-10, +10]` minutes around the anchor
- target window: `[+10, +30]` minutes

Current regression bundles:

- standard legacy bundle: `outputs/targets/all_targets.npz` (retain for
  compatibility with target-stat artifacts and legacy event outputs; do not
  use for new `t+0` regression runs)
- vasopressor-free gap mode: `outputs/targets/feature_targets_gap_vasopressor_free.npz`

## Classification / Event Targets

Classification targets are binary future-event labels.

Current event types:

- hypotension
- tachycardia

Label source:

- `X_numerics.npy`

Current sustained-event rule:

- `5` consecutive valid minutes

Thresholds:

- hypotension: `MAP <= 65`
- tachycardia: `HR > 110`

Stable-input rule:

- anchors are excluded for a given event if that event is already active during the input window

### Event target-generation modes

#### `anchor_horizon`

Directly labels each anchor/horizon pair from the post-window future outcome.

#### `anchor_horizon_filtered`

Keeps the same positive definition, but applies stricter hypotension-negative filtering:

- the relevant `5`-minute outcome window must contain zero hypotensive minutes
- the outcome-time label must be valid
- a recording contributes no hypotension negatives if it contains any sustained hypotension event anywhere
- negatives later than the mean last-event time across positive recordings are dropped

This mode only changes the hypotension negative class; it is intended to make near-event and partially invalid negatives less noisy.

### Event bundles

- standard `5m`/`10m`: `outputs/targets/event_targets_standard_5m_10m.npz`
  (optional standard-cohort bundle; not used by current models and currently
  absent from the workspace; rebuild only if standard event experiments
  resume)
- vasopressor-free rebuilt multi-horizon: `outputs/targets/event_targets_vasopressor_free.npz`
- vasopressor-free dedicated `5m`: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/targets/event_targets_vasopressor_free_5m.npz`
- vasopressor-free filtered-negative `5m`/`10m`: `outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz`

The current model runs are vasopressor-free and use only the three
vasopressor-free event bundles listed above. The standard event bundle is
retained as a reproducible future-experiment path, not as a current model
dependency.

### Current filtered hypotension prevalence

For `anchor_horizon_filtered` on the vasopressor-free cohort:

- hypotension `5m`: `1783 / 34338 = 0.0519`
- hypotension `10m`: `3957 / 36314 = 0.1090`

These prevalences are much higher than the earlier unfiltered rare-event evaluation prevalence because the stricter negative-filtering logic removes a large number of easy or noisy negatives.

### Filtered-negative removal counts

The following counts are for the baseline negative candidates before
`anchor_horizon_filtered` is applied. Counts are attributed sequentially in
the same order as the implementation: recording-level exclusion, late-time
cutoff, invalid outcome window, then hypotensive minutes in an otherwise valid
outcome window.

Here, `outcome window invalid` means that the five expected minute bins could
not be evaluated reliably because at least one expected minute timestamp was
missing or its MAP values were all non-finite. It does not mean that the
window contained hypotension. The label check always covers five minutes:
from `input_end_time + horizon` through the following four minutes. Therefore,
the `5m` task checks `[t+5m, t+9m]`, and the `10m` task checks `[t+10m,
t+14m]`, where `t` is the end of the input window. A valid window containing
any hypotensive minute is counted separately in the next column.

| Horizon | Baseline negative candidates | Negative recording has any sustained hypotension | Negative is later than mean last-event time | Outcome window invalid | Valid outcome window contains hypotensive minutes | Retained negatives |
|---|---:|---:|---:|---:|---:|---:|
| `5m` | `278,382` | `109,946` | `87,617` | `48,148` | `116` | `32,555` |
| `10m` | `276,365` | `107,838` | `87,671` | `48,383` | `116` | `32,357` |

The outcome-window removal reported by the target-builder diagnostics is the
combined total of the last two columns: `48,264` for `5m` and `48,499` for
`10m`. Before sequential attribution, the invalid-window and hypotensive-minute
categories overlap for `13` candidates at each horizon; therefore their raw
category counts are `48,148` and `129` for `5m`, and `48,383` and `129` for
`10m`. The table assigns those overlaps to the invalid-window condition first.

For the held-out test split used in the classification results, the same
breakdown is:

| Horizon | Baseline negative candidates | Negative recording has any sustained hypotension | Negative is later than mean last-event time | Outcome window invalid | Valid outcome window contains hypotensive minutes | Retained negatives | Positives | Final valid test examples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `5m` | `31,361` | `14,661` | `9,153` | `4,078` | `13` | `3,456` | `194` | `3,650` |
| `10m` | `31,141` | `14,433` | `9,155` | `4,104` | `13` | `3,436` | `437` | `3,873` |

These test-split counts use the global mean last-event cutoff learned from the
positive recordings in the full vasopressor-free cohort. The invalid-window
and hypotensive-minute categories overlap for `1` test negative at each
horizon; the table assigns that overlap to invalid-window removal first.

## Practical Interpretation

The effective supervised problem depends on both the target family and the bundle:

- regression uses dense scalar targets with target-specific masking
- event classification uses sparse binary labels and is much more sensitive to label design
- hypotension performance is especially affected by event semantics, anchor rules, and negative filtering
- recent vasopressor-free runs should be interpreted together with the split-specific normalization file and the exact target bundle path, because those choices materially change the training distribution

## Main Files

- dataset loader: `waveform_baselines/numpy_dataset.py`
- target builder: `scripts/build_targets.py`
- task specs: `waveform_baselines/task_specs.py`
- event target logic: `waveform_baselines/target_builders.py`
- trainer: `scripts/train_patchtst.py`

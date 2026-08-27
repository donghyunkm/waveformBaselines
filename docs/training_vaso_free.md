# Vasopressor-Free Training

This file tracks the vasopressor-free overlap cohort, its key artifacts, the current reruns, and the main completed results.

## Cohort

- splits: `outputs/splits/vasopressor_free_splits.json`
- anchors: `outputs/targets/vasopressor_free_overlap_anchors.csv`
- waveforms: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms`

Overlap cohort size:

- patients: `887`
- restricted anchor rows: `334833`
- split counts:
  - train: `692` patients, `263894` windows
  - val: `98` patients, `36026` windows
  - test: `97` patients, `36645` windows

## Artifacts

### Regression

- bundle: `outputs/targets/feature_targets_gap_vasopressor_free.npz`
- semantics: adjacent-window gap mode
- input: `[-10, +10]` minutes
- target: `[+10, +30]` minutes

### Events

- rebuilt multi-horizon bundle: `outputs/targets/event_targets_vasopressor_free.npz`
- dedicated `5m` bundle: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/targets/event_targets_vasopressor_free_5m.npz`
- event rule: `5` consecutive valid minutes after minute-level aggregation
- hypotension: `MAP <= 65`
- tachycardia: `HR > 110`

### Normalization

- shared stats file: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms/normalization_stats_vasopressor_free_splits.json`

## Completed Current Reruns (2026-08-27)

These are the current `v1` jobs.

### Regression batch

- run tag: `vasopressor_free_v1_es`
- model: `patchtst_v1`
- batch size: `512`
- epochs: `50`
- early stopping: patience `5`, minimum epochs `10`, minimum delta `0.0`
- jobs: `26844048`–`26844073`

### Hypotension event reruns

- run tag: `vasopressor_free_v1_events_5m_10m_es`
- model: `patchtst_v1`
- batch size: `512`
- epochs: `50`
- early stopping: patience `5`, minimum epochs `10`, minimum delta `0.0`
- jobs:
  - `26844046`: hypotension `10m`
  - `26844047`: hypotension `5m`

### Notes

- Earlier same-day reruns under `vasopressor_free_v1` and `vasopressor_free_v1_events_5m_10m` were cancelled and replaced so the current jobs start from scratch with early stopping enabled.
- Earlier completed results in this file are `pre-v1`.
- The current regression and event training jobs completed; no current training
  jobs remain pending.

## Completed Filtered-Hypotension `v1` Training (2026-08-27)

These runs use `anchor_horizon_filtered`, which heavily trims the hypotension negative class.

### Filtered event batch

- run tag: `vasopressor_free_v1_events_5m_10m_anchor_horizon_filtered_es`
- model: `patchtst_v1`
- target bundle: `outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz`
- jobs:
  - `26850864`: hypotension `5m`
  - `26850865`: hypotension `10m`

### Training outcomes

| Horizon | Train Windows | Val Windows | Train Patients | Val Patients | Best Val Loss | Best Epoch | Stop Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `5m` | `26,301` | `3,975` | `408` | `60` | `0.094466` | `12` | `17` |
| `10m` | `27,847` | `4,182` | `418` | `62` | `0.222887` | `7` | `12` |

### Test-set evaluation

- The filtered test split is much smaller than the unfiltered one:
  - `5m`: `3,650` valid test windows across `66` patients
  - `10m`: `3,873` valid test windows across `66` patients
- Local evaluation from the interactive sandbox was not practical because:
  - `num_workers > 0` failed when Python multiprocessing could not open the required local listener socket
  - `num_workers = 0` ran on CPU only and did not finish within the interactive timeout
- GPU evaluation jobs were therefore submitted on 2026-08-27 and completed successfully:
  - `26851239`: filtered hypotension `5m` evaluation
  - `26851240`: filtered hypotension `10m` evaluation

### Filtered test outcomes

| Horizon | N Valid | Prevalence | AUROC | AUPRC | Train Epochs |
|---|---:|---:|---:|---:|---:|
| `5m` | `3,650` | `0.05315` | `0.975` | `0.594` | `12` |
| `10m` | `3,873` | `0.11283` | `0.821` | `0.477` | `7` |

Artifacts:

- `5m` metrics: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_anchor_horizon_filtered_es/event_hypotension_within_5m/test_metrics.json`
- `10m` metrics: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_anchor_horizon_filtered_es/event_hypotension_within_10m/test_metrics.json`
- `5m` predictions: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_anchor_horizon_filtered_es/event_hypotension_within_5m/test_predictions.npz`
- `10m` predictions: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_anchor_horizon_filtered_es/event_hypotension_within_10m/test_predictions.npz`

The comparison against the unfiltered `v1` results is summarized in `docs/classification_results_v1_vaso_free.md`.

## Completed `pre-v1` Vasopressor-Free Results

### Regression

Best `R²` targets from the completed `pre-v1` batch under `outputs/patchtst/vasopressor_free/`:

| Target | R² |
|---|---:|
| `PLETH_ACDC` | `0.759` |
| `dPdt_max` | `0.755` |
| `PVI` | `0.736` |
| `HR` | `0.700` |
| `HR_range` | `0.685` |

Weakest targets:

| Target | R² |
|---|---:|
| `RESP_amp` | `-0.079` |
| `RR` | `-0.032` |
| `PLETH_amp×ShockIdx` | `-0.005` |

PatchTST clearly beat the train-mean baseline on most regression tasks.

### Event classification

#### `pre-v1` rebuilt-label results

These are the most relevant completed vasopressor-free event results so far.

| Event | AUROC | AUPRC |
|---|---:|---:|
| Tachycardia `5m` | `0.948` | `0.087` |
| Tachycardia `10m` | `0.936` | `0.113` |
| Tachycardia `60m` | `0.887` | `0.263` |
| Tachycardia `90m` | `0.863` | `0.310` |
| Hypotension `5m` | `0.700` | `0.012` |
| Hypotension `10m` | `0.652` | `0.022` |
| Hypotension `60m` | `0.534` | `0.081` |
| Hypotension `90m` | `0.472` | `0.105` |

Main takeaways:

- Tachycardia ranking is strong across horizons.
- Hypotension remains weak, especially long-horizon.
- Hypotension `90m` is slightly anti-predictive in the rebuilt-label run.
- For rare-event tasks, threshold-free ranking metrics are more informative than the default `0.5` threshold.

## Output Locations

- completed `pre-v1` regression/event artifacts: `outputs/patchtst/vasopressor_free/`
- rebuilt-label event artifacts: `outputs/patchtst/vasopressor_free_rebuilt_events/`
- rebuilt-label `5m` artifacts: `outputs/patchtst/vasopressor_free_rebuilt_events_5m_only/`

## Next Questions

- How much does the fresh `v1` rerun improve over the `pre-v1` vasopressor-free results?
- Does early stopping reduce wasted training without hurting the stronger regression targets?
- Is hypotension fundamentally weak on this cohort, or mostly limited by label design and event prevalence?

The filtered-negative evaluation is complete and documented in
`docs/classification_results_v1_vaso_free.md`. The remaining questions are
planned analysis, not completed experiments.

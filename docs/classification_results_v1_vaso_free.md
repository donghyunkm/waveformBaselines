# Vasopressor-Free `patchtst_v1` Classification Results

Compact summary of the fresh early-stopping-enabled vasopressor-free `patchtst_v1` hypotension evaluation completed on 2026-08-27.

This page separates the older non-filtered-label `v1` evaluation from the later `anchor_horizon_filtered` evaluation.

## Filtered Data Scope

- run tag: `vasopressor_free_v1_events_5m_10m_anchor_horizon_filtered_es`
- model: `patchtst_v1`
- task: single-target event classification
- cohort: vasopressor-free overlap cohort
- split file: `outputs/splits/vasopressor_free_splits.json`
- target bundle: `outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz`

## Filtered Data Hypotension Results

| Horizon | N Valid | Positives / Valid | Prevalence | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | Spec @ 85% Sens | Threshold @ 85% Sens | Train Epochs |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| `5m` | `3,650` | `194 / 3,650` | `0.05315` | `0.975` | `[0.968, 0.981]` | `0.594` | `[0.525, 0.665]` | `0.959` | `0.12179` | `12` |
| `10m` | `3,873` | `437 / 3,873` | `0.11283` | `0.821` | `[0.798, 0.843]` | `0.477` | `[0.431, 0.522]` | `0.586` | `0.02387` | `7` |

## Non-Filtered Data Scope

- run tag: `vasopressor_free_v1_events_5m_10m_es`
- model: `patchtst_v1`
- task: single-target event classification
- cohort: vasopressor-free overlap cohort
- split file: `outputs/splits/vasopressor_free_splits.json`
- `10m` target bundle: `outputs/targets/event_targets_vasopressor_free.npz`
- `5m` target bundle: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/targets/event_targets_vasopressor_free_5m.npz`

## Non-Filtered Data Hypotension Results

| Horizon | N Valid | Positives / Valid | Prevalence | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | Spec @ 85% Sens | Threshold @ 85% Sens | Train Epochs |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| `5m` | `31,555` | `194 / 31,555` | `0.00615` | `0.854` | `[0.834, 0.875]` | `0.037` | `[0.026, 0.055]` | `0.683` | `0.00339` | `5` |
| `10m` | `31,578` | `437 / 31,578` | `0.01384` | `0.889` | `[0.878, 0.900]` | `0.081` | `[0.069, 0.099]` | `0.803` | `0.01133` | `5` |

## Notes

- All four evaluations produced finite predictions for every evaluated test sample.
- The prevalence numerators are identical across the filtered and non-filtered tables (`194` positives for `5m`, `437` positives for `10m`); the prevalence shift comes from the much smaller valid denominator after `anchor_horizon_filtered` label filtering.
- For the filtered test split, the negative counts reconcile as `31,361` baseline negatives minus `27,905` sequential removals plus `194` positives = `3,650` valid examples for `5m`, and `31,141` minus `27,705` plus `437` = `3,873` for `10m`. The full-cohort filtering counts are documented separately in `docs/data_description.md`.
- The filtered dataset is much smaller and much less imbalanced than the non-filtered one, so its metrics are not directly prevalence-matched to the older table.
- On filtered data, the `5m` model is stronger than the `10m` model on both AUROC and AUPRC.
- On non-filtered data, both models predicted no positives at the default `0.5` threshold, so default-threshold sensitivity, precision, and F1 were all `0.0`.
- The useful signal is still best understood through ranking metrics and chosen operating points rather than the default threshold alone.

## Source Files

- Filtered `5m` metrics: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_anchor_horizon_filtered_es/event_hypotension_within_5m/test_metrics.json`
- Filtered `10m` metrics: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_anchor_horizon_filtered_es/event_hypotension_within_10m/test_metrics.json`
- Filtered `5m` predictions: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_anchor_horizon_filtered_es/event_hypotension_within_5m/test_predictions.npz`
- Filtered `10m` predictions: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_anchor_horizon_filtered_es/event_hypotension_within_10m/test_predictions.npz`
- Non-filtered `5m` metrics: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_es/event_hypotension_within_5m/test_metrics.json`
- Non-filtered `10m` metrics: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_es/event_hypotension_within_10m/test_metrics.json`
- Non-filtered `5m` predictions: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_es/event_hypotension_within_5m/test_predictions.npz`
- Non-filtered `10m` predictions: `outputs/patchtst/vasopressor_free_v1_events_5m_10m_es/event_hypotension_within_10m/test_predictions.npz`

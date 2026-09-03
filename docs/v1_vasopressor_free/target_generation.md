# Target Generation

This document defines the current target-building setup.

## Inputs

- waveform features: `/gpfs/data/eh3828lab/derived_datasets/baselines/output_v2`
- waveform anchors: aligned 20-minute windows
- main scripts:
  - `scripts/build_aligned_20m_index.py`
  - `scripts/build_targets.py`

## Canonical Input Window

- window length: `20` minutes
- sampling rate: `125 Hz`
- samples per channel: `150000`
- raw signal order on disk: `II`, `ABP`, `PLETH`, `RESP`
- default model channels: `ABP`, `II`, `PLETH`

## Regression Targets

- source arrays: `X_stats.npy`, `corr_features_focused.npy`
- base targets: `26`
- horizons: `0`, `20`, `60` minutes
- total regression targets: `78`

Current bundles:

- standard legacy mixed bundle: `outputs/targets/all_targets.npz`; its
  center-mode `t+0` regression targets overlap the input and must not be used
  for new leakage-free regression experiments
- vasopressor-free gap-mode: `outputs/targets/feature_targets_gap_vasopressor_free.npz`

## Event Targets

- events: hypotension, tachycardia
- labels come from `X_numerics.npy`
- current rule: `5` consecutive valid minutes
- hypotension threshold: `MAP <= 65`
- tachycardia threshold: `HR > 110`
- stable-input filter excludes anchors where the event is already active during the input window

Current event modes:

- `anchor_horizon`: label each anchor+horizon pair directly
- `anchor_horizon_filtered`: keep the same positive definition, but keep hypotension negatives only when the outcome window is clean and valid, the source recording never contains any sustained hypotension event, and the outcome timestamp is not later than the mean last-event time across positive recordings

Current bundles:

- vasopressor-free rebuilt bundle: `outputs/targets/event_targets_vasopressor_free.npz`
- vasopressor-free `5m` bundle: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/targets/event_targets_vasopressor_free_5m.npz`
- vasopressor-free filtered-negative `5m`/`10m` bundle: `outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz`
- standard `5m`/`10m` bundle: `outputs/targets/event_targets_standard_5m_10m.npz`
  (optional future-experiment path; not used by current models and currently
  absent from the workspace)

## Important Current State

- job `26838720` reportedly rebuilt standard `5m`/`10m` event targets on 2026-08-26, but the optional output is absent in the current workspace; rerun the documented command only if standard-cohort event experiments are needed
- the intended standard path uses `outputs/index/aligned_20m_anchors.csv` so event horizons begin at the end of the 20-minute input window
- vasopressor-free event labels were rebuilt on 2026-08-25 with minute-level aggregation before the `5`-minute rule
- current vasopressor-free regression runs use gap mode, with a 20-minute
  input followed immediately by the 20-minute target window
- current vasopressor-free classification runs use `input_end_time` as the
  future-label boundary, so their outcome windows do not overlap the input

## Key Commands

Standard aligned index:

```bash
PYTHONPATH=. /gpfs/home/dk5565/.conda/envs/physiojepa/bin/python \
  scripts/build_aligned_20m_index.py \
  --output-csv outputs/index/aligned_20m_anchors.csv
```

Standard event targets (`5m`, `10m`):

```bash
PYTHONPATH=. /gpfs/home/dk5565/.conda/envs/physiojepa/bin/python \
  scripts/build_targets.py \
  --anchors-csv outputs/index/aligned_20m_anchors.csv \
  --skip-feature-targets \
  --event-horizons 5 10 \
  --output outputs/targets/event_targets_standard_5m_10m.npz
```

Vasopressor-free rebuilt events:

```bash
PYTHONPATH=. /gpfs/home/dk5565/.conda/envs/physiojepa/bin/python \
  scripts/build_targets.py \
  --anchors-csv outputs/targets/vasopressor_free_overlap_anchors.csv \
  --skip-feature-targets \
  --output outputs/targets/event_targets_vasopressor_free.npz
```

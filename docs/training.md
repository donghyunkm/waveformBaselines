# Training

This file describes the current training pipeline and the parts that still matter.

## Current Baseline

- model family: single-target PatchTST
- current default baseline variant: `patchtst_v1`
- supervised PhysioJEPA-style variant: `patchtst_v1_5`
  - use `--physiojepa-fidelity` for the strict `125/125`, `512`, `8`, `3`, `2048`, batch-size `32` preset
- newer experimental variant: `patchtst_v2`
- current active waveform cache for most runs: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms`

## Data and Splits

- waveform cache: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms`
- raw source: `/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched/`
- features/numerics source: `/gpfs/data/eh3828lab/derived_datasets/baselines/output_v2/`
- standard split file: `outputs/splits/splits.json`
- legacy standard target bundle: `outputs/targets/all_targets.npz`

## Dataset Pipeline

Main components:

- `waveform_baselines/numpy_dataset.py`
- `waveform_baselines/patient_sampler.py`
- `scripts/extract_waveforms.py`

Current behavior:

- waveforms are pre-extracted to per-patient `.npy` files
- training uses memory-mapped reads
- batches are patient-grouped to reduce file-switch overhead
- invalid targets are filtered before training

Training and evaluation use `NumpyWaveformDataset`; WFDB is used only during
the waveform pre-extraction step in `scripts/extract_waveforms.py`.

For leakage-safe regression, use the vasopressor-free gap-mode bundle and an
explicit `--feature-horizon-mode gap`. Do not use the direct defaults in
`slurm/train_patchtst.sh` for new regression runs: they point to the legacy
center-mode `all_targets.npz` bundle and `t+0`.

## Normalization

Current normalization is shared train-split per-channel z-scoring, not per-patient normalization.

- stats script: `scripts/compute_waveform_normalization_stats.py`
- loader: `waveform_baselines/normalization.py`
- standard stats file example: `normalization_stats_splits.json`
- vasopressor-free stats file example: `normalization_stats_vasopressor_free_splits.json`

The dataset now requires the split-specific stats file when `--normalize` is enabled.

## Trainer

Main script: `scripts/train_patchtst.py`

Important current behavior:

- one model per target
- mixed precision forward pass, loss in fp32
- `best_model.pt` tracks best validation loss
- `latest_model.pt` is saved every epoch for resume
- `--resume` restores optimizer state, epoch, best validation loss, and early-stopping state

### Early stopping

Optional early stopping is now supported:

- `--early-stopping-patience`
- `--early-stopping-min-epochs`
- `--early-stopping-min-delta`

Checkpoint state also stores `epochs_without_improvement`.

## Current Variants

### `patchtst_v1`

- channel-independent temporal encoder
- no cross-channel attention
- mean pooling

### `patchtst_v1_5`

- supervised end-to-end adaptation of the PatchTST encoder and attentive classifier from `benmfox/PhysioJEPA`
- channel-specific grouped-Conv1d tokenizer with zero end-padding
- shared channel-independent Transformer encoder with rotary Q/K attention and post-norm TST blocks
- shared per-channel attentive pooler and final `Linear(C*d_model, 1)` head
- retains this repo's existing supervised targets, losses, optimizer, scheduler, and dataset-level normalization
- intentionally omits PhysioJEPA masked pretraining, reconstruction heads, pretrained checkpoint loading, and encoder freezing
- `patchtst_v1_5` currently runs in eager mode; `torch.compile` remains enabled only for `patchtst_v1`

### `patchtst_v2`

- experimental variant
- local cross-channel fusion
- optional attention pooling
- intended for 4-channel experiments

## Useful Submitters

- generic train wrapper: `slurm/train_patchtst.sh`
- vasopressor-free regression rerun: `slurm/submit_patchtst_regression_t0_gap_vasopressor_free.sh`
- vasopressor-free event rerun: `slurm/submit_patchtst_event_vasopressor_free.sh`

## Current Notes

- `patchtst_v1` is the main baseline.
- `patchtst_v1_5` is the PhysioJEPA-architecture supervised adaptation.
- `patchtst_v2` is still experimental.
- the old long job histories and failed early batches are omitted here on purpose; `PROGRESS.md` is the place to find the chronology.

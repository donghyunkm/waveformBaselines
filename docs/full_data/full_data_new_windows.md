# New `data_m3_120s_prediction` Windows

This note records what is in `/gpfs/data/eh3828lab/derived_datasets/physionet_restricted/mimic_derived_data/data_m3_120s_prediction`, how it relates to the repo's current v1 vasopressor-free pipeline, and what is required to build regression and classification datasets that stay identical to the current v1 setup except for the source window set.

## Handoff State (2026-08-30)

The segment-level vasopressor-free manifest for this dataset has been built and hardened. A full-data-specific v7 waveform-feature extractor, shard merger, and SLURM submitters have been added, smoke-tested, and run to completion. Regression target bundles, numerics arrays, classification target bundles, and model-training jobs have not yet been built for the full-data window grid.

Latest status check on `2026-08-30`: extraction array `26920063` and dependent merge job `26920070` were no longer listed by `squeue`. All `128` shard directories plus the merged cache had `_SUCCESS` markers. The merge log reported the expected shape `(1969515, 20, 93)` and `n_samples=1969515` for `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v7/full_data_vasopressor_free_waveform_features_v7`.

The merged cache row arrays were checked after completion:

- `values.npy`: `(1969515, 20, 93)`, `float32`
- `mask.npy`: `(1969515, 20, 93)`, `bool`
- `patient_ids.npy`, `anchor_times.npy`, `anchor_ids.npy`, `split_labels.npy`, `segment_ids.npy`, `segment_names.npy`: all row-aligned to `1969515`
- split counts: `1416785` train, `269872` val, `282858` test
- coverage: `1758` unique patients and `21833` unique vaso-free segments

The merged `feature_quality_report.json` contains all `93` v7 features with no zero-valid features. The highest missing/non-finite fractions are the first-difference ABP features (`6.41%`), `delta_pleth_amplitude_median` (`5.82%`), `delta_resp_rate_bpm` (`5.35%`), `delta_ecg_hr_bpm` (`5.22%`), and `cross_ecg_abp_rate_diff_bpm`/agreement (`1.54%`). Shard stderr files were nonempty for `25/128` shards, with sampled messages limited to WFDB normalization runtime warnings; the merge stderr was empty.

The previously active unrelated target-normalized vasopressor-free regression batch has completed evaluation. Results are summarized in `docs/v1_vasopressor_free/regression_results_v1_vaso_free_target_normalized.md`, with artifacts under `outputs/patchtst/vasopressor_free_v1_target_norm_es/`. The first submission (`26864294`-`26864306`) failed because the shared wrapper referenced a removed Python environment, and the replacement wrapper now uses `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python`.

The repository's Git status was not independently verified because `git` is unavailable on the current node. This is an environment limitation, not evidence of a clean worktree. Use a node with Git before committing or rebasing.

## Scope

Target behavior to preserve from the current extracted-feature v7 path where applicable:

- waveform feature input channels: `II,ABP,PLETH,RESP`
- input window: centered `20` minutes at `125 Hz` (`150000` samples)
- feature representation: v7 extracted waveform features, `20` feature windows by `93` features
- regression semantics: same `26` base targets, same gap-mode target semantics used by `feature_targets_gap_vasopressor_free.npz`
- classification semantics: same event definitions and same `anchor_horizon` / `anchor_horizon_filtered` label logic used by the current v1 hypotension runs
- split discipline: full-data `patient_splits.json` patient-level train/val/test splits only
- optional regression target normalization: same train-only z-score path now available in `scripts/train_patchtst.py`

Relevant existing docs:

- `docs/v1_vasopressor_free/classification_results_v1_vaso_free.md`
- `docs/v1_vasopressor_free/regression_results_v1_vaso_free_sorted.md`
- `docs/reference/data_description.md`
- `docs/v1_vasopressor_free/target_generation.md`
- `docs/v1_vasopressor_free/training_vaso_free.md`

## Files In `data_m3_120s_prediction`

Top-level files observed:

- `X_stats.npy`
- `block_start_times.npy`
- `cluster_labels.npy`
- `corr_features_focused.npy`
- `corr_features_focused_names.json`
- `n_windows.txt`
- `patient_ids.npy`
- `patient_splits.json`
- `seg_names.npy`
- `segment_metadata.json`
- `window_times.npy`

What each file appears to be:

- `patient_ids.npy`
  - row-aligned patient ID array
  - shape confirmed: `(2847597,)`
- `window_times.npy`
  - row-aligned window center times
  - shape confirmed: `(2847597,)`
  - values are relative to segment start, not absolute timestamps
  - first segment sample: `600.0, 750.0, 900.0, ...`
  - stride within a segment is `150` seconds
- `seg_names.npy`
  - row-aligned waveform segment basename such as `3531764_0003`
  - shape confirmed: `(2847597,)`
- `segment_metadata.json`
  - segment-level metadata with fields:
    - `patient_id`
    - `seg_name`
    - `seg_start_secs`
    - `start_anchor`
    - `origin`
    - `n_windows`
  - confirmed segment rows: `32044`
  - confirmed unique patients: `2056`
  - confirmed `sum(n_windows) = 2847597`
- `patient_splits.json`
  - authoritative patient-level split assignment for this dataset
  - contains all `2056` patients exactly once
- `corr_features_focused.npy`
  - row-aligned focused correlation features
  - shape confirmed: `(2847597, 7)`
- `corr_features_focused_names.json`
  - confirmed names exactly match the repo's current `7` correlation targets:
    - `PLETH_ACDC_PLETH_amp`
    - `ABP_area_ABP_tau`
    - `ABP_area_ShockIdx`
    - `PLETH_amp_ShockIdx`
    - `PLETH_ACDC_ShockIdx`
    - `ShockIdx_ABP_tau`
    - `PLETH_ACDC_ABP_tau`
- `cluster_labels.npy`
  - row-aligned cluster assignment array
  - shape confirmed: `(2847597,)`
  - not used by the current supervised v1 pipeline
- `block_start_times.npy`
  - row-aligned array with shape `(2847597,)`
  - observed value is constant `600.0` throughout sampled rows and global min/max
  - does not appear to be needed for the current supervised dataset construction
- `n_windows.txt`
  - plain-text total window count
  - confirmed content: `2847597`
- `X_stats.npy`
  - likely the physiological feature source analogous to the current regression feature source
  - direct read access was blocked by filesystem permissions during this inspection, so its exact shape could not be confirmed here

## Confirmed Counts

These counts are directly confirmed from the readable arrays and metadata:

- patients: `2056`
- windows: `2847597`
- segment rows: `32044`

Split counts from `patient_splits.json`:

| Split | Patients | Windows |
|---|---:|---:|
| `train` | `1440` | `2049696` |
| `val` | `308` | `398887` |
| `test` | `308` | `399014` |

Per-patient window count summary:

- min: `2`
- median: `722`
- max: `21116`

`segment_metadata.json` also reconciles exactly with the total window count:

- `sum(n_windows) = 2847597`

## Match To `mimic3_waveforms_matched`

The row-level segment naming matches the raw matched waveform directory layout.

Examples:

- dataset segment `p000160 / 3531764_0003`
  - matched files exist at `mimic3_waveforms_matched/p00/p000160/3531764_0003.{hea,dat}`
- dataset segment `p000188 / 3285727_0015`
  - matched files exist at `mimic3_waveforms_matched/p00/p000188/3285727_0015.{hea,dat}`

Important join facts:

- patient directory rule: `mimic3_waveforms_matched/{patient_id[:3]}/{patient_id}/`
- waveform segment key: `seg_name`
- absolute anchor time can be reconstructed as:
  - `anchor_time = seg_start_secs + window_times[row]`
- because `window_times.npy` is segment-relative, `patient_id + window_time` is not unique across the full dataset
- the true row identity is segment-aware:
  - at minimum `(patient_id, seg_name, window_time)`
  - for absolute-time use, `(patient_id, anchor_time)` after adding `seg_start_secs`

## Relationship To The Existing Vasopressor-Free Cohort

Current repo v1 vasopressor-free cohort:

- `887` patients
- `334833` windows
- splits from `outputs/splits/vasopressor_free_splits.json`

Comparison against the new dataset:

- all current `887` vasopressor-free patients are present in the new `2056`-patient dataset
- no current vasopressor-free patient is missing from the new dataset
- the new dataset adds `1169` patients beyond the current cohort
- the current `887` patients account for `1192845` windows inside the new dataset
- the added `1169` patients account for `1654752` windows

The old repo split file should not be reused for this new dataset:

- the new dataset already provides its own patient-level split assignment in `patient_splits.json`
- the old `887` patients land in the new split map as:
  - `636` train
  - `120` val
  - `131` test

## Segment-Level Vasopressor-Free Manifest

The active full-data manifest is segment-level, not patient-level:

- manifest: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_waveform_manifest.csv`
- QC: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_waveform_manifest.qc.json`
- free-segment list: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_free_segments.txt`
- confirmed continuous vasopressor intervals: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_vasopressor_intervals.csv`
- uncertain vasopressor evidence: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_uncertain_vasopressor_evidence.csv`

Important QC from the completed build:

- total waveform segments: `32044`
- valid timestamp segments: `32044`
- metadata/WFDB timestamp mismatches: `0`
- ICU linkage: `30492` `matched_full`, `120` `matched_partial`, `1432` `unmatched`
- confirmed vasopressor-overlap segments: `8658`
- high-confidence vasopressor-free segments: `21833`
- unknown segments: `1553`
- vasopressor-free among classified segments: `71.6047%`

The manifest's `vasopressor_free=True` is conservative. It requires valid segment timing, metadata/WFDB timestamp agreement where available, full containment in one unambiguous ICU stay, no confirmed vasopressor overlap, and no unresolved CareVue or MetaVision evidence that could apply. Partial ICU matches and unresolved uncertainty remain `NA`, not `True`.

See `docs/full_data/record_level_vasopressor_free_manifest.md` for reconstruction details and full QC.

## Full-Data Vaso-Free Feature Extraction

Added scripts:

- `scripts/extract_full_data_vasofree_waveform_features.py`
- `scripts/merge_full_data_waveform_feature_shards.py`
- `slurm/extract_full_data_vasofree_waveform_features_array.sh`
- `slurm/merge_full_data_vasofree_waveform_features.sh`

The extractor builds a segment-aware anchor table from:

- `/gpfs/data/eh3828lab/derived_datasets/physionet_restricted/mimic_derived_data/data_m3_120s_prediction/patient_ids.npy`
- `/gpfs/data/eh3828lab/derived_datasets/physionet_restricted/mimic_derived_data/data_m3_120s_prediction/seg_names.npy`
- `/gpfs/data/eh3828lab/derived_datasets/physionet_restricted/mimic_derived_data/data_m3_120s_prediction/window_times.npy`
- `/gpfs/data/eh3828lab/derived_datasets/physionet_restricted/mimic_derived_data/data_m3_120s_prediction/patient_splits.json`

It joins those rows to the hardened manifest by `segment_id = patient_id + "/" + seg_name` and keeps only `vasopressor_free == True` segments. The full expected full-data vaso-free extraction set is `1969515` windows from `21833` high-confidence vaso-free segments.

Output location:

- final merged cache: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v7/full_data_vasopressor_free_waveform_features_v7`
- shard caches: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v7/full_data_vasopressor_free_waveform_features_v7_shard_###`

Cache contents include `values.npy`, `mask.npy`, `patient_ids.npy`, `anchor_times.npy`, `anchor_ids.npy`, `split_labels.npy`, `segment_ids.npy`, `segment_names.npy`, `anchors.csv`, `metadata.json`, `feature_quality_report.json`, and `_SUCCESS`.

Validation completed before full submission:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python \
  scripts/extract_full_data_vasofree_waveform_features.py \
  --cache-root /tmp/full_data_feat_smoke \
  --output-name smoke \
  --max-samples 2 \
  --overwrite
```

Smoke result: shape `(2, 20, 93)`, `expected_full_n_samples=1969515`.

Two-shard smoke extraction plus partial merge also passed:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python \
  scripts/merge_full_data_waveform_feature_shards.py \
  --cache-root /tmp/full_data_feat_merge_smoke2 \
  --shard-name-prefix shard \
  --shard-count 2 \
  --output-name merged \
  --overwrite \
  --allow-partial
```

Smoke merge result: shape `(4, 20, 93)`.

The full SLURM extraction completed:

```bash
sbatch slurm/extract_full_data_vasofree_waveform_features_array.sh
```

Submitted jobs:

- extraction array: `26920063`
- dependent merge: `26920070`

Validated merged output:

- cache: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v7/full_data_vasopressor_free_waveform_features_v7`
- shape: `(1969515, 20, 93)`
- success markers: `128/128` shards plus merged cache
- quality report: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v7/full_data_vasopressor_free_waveform_features_v7/feature_quality_report.json`

## Why The Older Raw-Waveform Pipeline Could Not Be Reused As-Is

Two core assumptions in the older raw-waveform training pipeline did not hold for this dataset.

### 1. The waveform cache is currently one continuous segment per patient

Current repo behavior:

- `scripts/extract_waveforms.py` picks one best segment per patient
- `waveform_baselines/numpy_dataset.py` loads one file per patient
- metadata stores one `seg_start_secs` and one anchor list per patient

New dataset behavior:

- `2056` patients spread across `32044` segments
- many patients have many segments
- the row identity is segment-aware, not patient-only

Consequence for the older pipeline:

- the current per-patient waveform cache format cannot represent all `2847597` windows without dropping most segments
- concatenating segments into one fake continuous patient file would be unsafe unless explicit gap handling is added, because the current loader assumes every anchor index refers to real contiguous signal

Current resolution for extracted features:

- `scripts/extract_full_data_vasofree_waveform_features.py` reads the matched WFDB segment for each full-data window directly, using `segment_id` and `segment_path`
- the resulting v7 extracted-feature cache is row-aligned to the full-data window grid after filtering to high-confidence segment-level `vasopressor_free=True`
- this does not replace the older raw-waveform NumPy loader for PatchTST training; it creates an extracted-feature cache suitable for the existing extracted-feature model family after aligned targets are built

### 2. The current target builders expect a unique `(patient_id, anchor_time)` grid from the source arrays

Current repo target builders use:

- regression: `patient_ids.npy` + `window_times.npy` + `X_stats.npy` + `corr_features_focused.npy`
- classification: `numerics_patient_ids.npy` + `numerics_window_times.npy` + `X_numerics.npy`

The old source under `/gpfs/data/eh3828lab/derived_datasets/baselines/output_v2` is an older absolute-time grid with only `666492` regression rows and `673283` numerics rows.

Alignment check against the new dataset:

- using `anchor_time = seg_start_secs + window_time`, only a minority of sampled new anchors match the old `output_v2` timestamp grid
- this is expected because `output_v2` corresponds to the older overlap extraction, not the new `2056`-patient window set

Consequence:

- the current `output_v2` artifacts cannot simply be pointed at the new anchors
- new regression and classification targets must be generated on the new window grid

## What Is Needed To Build The New Regression Dataset

### Confirmed available pieces

- row-aligned `patient_ids.npy`
- row-aligned `window_times.npy`
- row-aligned `seg_names.npy`
- segment-to-absolute-time mapping via `segment_metadata.json`
- row-aligned `corr_features_focused.npy`
- an `X_stats.npy` file is present, but its readable schema was not confirmable because of permissions
- patient-level splits are already supplied in `patient_splits.json`

### Required remaining steps

1. Reuse the segment-aware anchor table logic from `scripts/extract_full_data_vasofree_waveform_features.py`.
   - Required columns to preserve current semantics:
     - `patient_id`
     - `segment_id`
     - `seg_name`
     - `window_time`
     - `anchor_id`
     - `split_label`
   - Compute:
     - `anchor_time = seg_start_secs + window_time`
     - `input_start_time = anchor_time - 600`
     - `input_end_time = anchor_time + 600`
   - Keep segment provenance in all target bundles so joins never fall back to patient-only time keys

2. Build a new split artifact from `patient_splits.json` if the downstream trainer requires a repo-style split JSON instead of row-level `split_labels.npy`.
   - Convert the patient map into the repo split JSON shape:
     - `train`
     - `val`
     - `test`
   - Include split stats using the new window counts above

3. Make regression targets segment-aware and absolute-time-safe.
   - If `X_stats.npy` is readable and row-aligned to the other arrays, the simplest path is:
     - materialize a new regression source bundle keyed by absolute anchor time
     - or add a new builder that uses `(patient_id, seg_name, row_idx)` / absolute anchor time instead of `(patient_id, window_time)`
   - The current `build_feature_regression_targets()` function cannot safely use the new directory as-is because:
     - `window_times.npy` is relative, not absolute
     - `patient_id + window_time` is not unique across segments

4. Preserve the existing regression task definition.
   - same `26` base targets
   - same focused correlation names
   - same horizons `0, 20, 60`
   - same gap-mode semantics for leakage-safe runs
   - for the current v1 comparison path, use the same `t+0` gap-mode setup as `docs/v1_vasopressor_free/regression_results_v1_vaso_free_sorted.md`

5. Preserve current target-normalization behavior for comparable regression training.
   - optional regression target normalization:
     - same train-only z-score path used by the current `vasopressor_free_v1_target_norm_es` batch

### Regression target build status

- `X_stats.npy` is now readable from this environment.
- Built `outputs/targets/feature_targets_gap_full_data.npz` on `2026-08-31` using the same `26` base targets, focused correlations, horizons `0, 20, 60`, and gap-mode semantics as the current extracted-feature regression setup.
- The saved target bundle has shape `(1969515, 78)` and `128227907` valid target values. See `docs/full_data/extractedFeaturesRegressionFullData.md`.

## What Is Needed To Build The New Classification Dataset

### Confirmed current state

- the new directory does not contain:
  - `numerics_patient_ids.npy`
  - `numerics_window_times.npy`
  - `X_numerics.npy`
- the current event builder depends on those numerics arrays

### Required steps

1. Build the same new anchor table described above.

2. Produce numerics windows on the new anchor grid.
   - Needed outputs, matching current builder expectations in spirit:
     - row-aligned patient IDs
     - row-aligned window times
     - row-aligned numerics windows
   - They must be aligned to the same `2056` patients / `32044` segments / `2847597` windows
   - They must be keyed by absolute anchor time or another segment-aware key that avoids collisions

3. Reuse the current label semantics unchanged.
   - stable input window requirement
   - minute-level aggregation
   - `5` consecutive valid minutes
   - hypotension threshold `MAP <= 65`
   - tachycardia threshold `HR > 110`
   - event modes:
     - `anchor_horizon`
     - `anchor_horizon_filtered`

4. Build the same classification bundles needed for the current v1 comparison.
   - at minimum, to mirror the documented v1 hypotension evaluations:
     - unfiltered `5m` / `10m`
     - filtered-negative `5m` / `10m`

### Current classification target path

- `data_m3_120s_prediction` still does not contain row-aligned numerics arrays.
- To preserve the prior target definition, full-data numerics should be materialized from the same bedside-monitor waveform numerics source used by `icuDataExtraction`: `/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched/RECORDS-numerics`.
- Prepared scripts: `scripts/extract_full_data_numerics.py`, `scripts/merge_full_data_numerics.py`, and `scripts/build_full_data_event_targets.py`.
- Production target construction should use the aligned-array route, which calls the existing `waveform_baselines.target_builders.build_event_targets()` implementation with the same `anchor_horizon_filtered` event spec.
- A small compatibility check on `5` prior vasopressor-free patients (`3294` rows) found exact old-vs-new equality for extracted numerics and unfiltered target generation: `0` numerics mismatches, max absolute numerics difference `0.0`, `0` target-mask mismatches, and `0` target-value mismatches on jointly valid labels.
- Submitted dependent classification-preparation chain on `2026-08-30`: extraction array `26931914` (`0-127`), merge `26931915`, and target build `26931916`. Superseding handoff checks confirmed extraction/merge completion and later corrected full-data event target bundles for hypotension, tachycardia, and hypoxia. Current target artifacts, filtering decisions, and completed extracted-feature model results are documented in `docs/full_data/extractedFeaturesClassificationFullData.md`.
- Filtered-label construction and validation are no longer pending in this construction note; use `docs/full_data/extractedFeaturesClassificationFullData.md` as the source of truth for current filtered target counts and model results.

## Remaining Implementation Plan

The cleanest path was to keep the dataset representation segment-aware end-to-end. The v7 waveform-feature cache, regression targets, aligned numerics, corrected event targets, and extracted-feature downstream runs have since been implemented; current status lives in `docs/full_data/extractedFeaturesRegressionFullData.md` and `docs/full_data/extractedFeaturesClassificationFullData.md`.

### Step 1. Verify the v7 feature extraction

Completed on `2026-08-30`:

- `128` shard `_SUCCESS` files
- merged cache `_SUCCESS`
- merged shape `(1969515, 20, 93)`
- split counts: `1416785` train, `269872` val, `282858` test
- no zero-valid features in the merged feature-quality report

### Step 2. Create target/split artifacts for full-data rows

Prepared outputs and scripts:

- regression target bundle: `outputs/targets/feature_targets_gap_full_data.npz`
- classification target bundles: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.npz` and `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz`
- regression builder: `scripts/build_full_data_feature_regression_targets.py` / `slurm/build_full_data_feature_regression_targets.sh`
- classification builder: `scripts/build_full_data_event_targets.py` / `slurm/build_full_data_event_targets.sh`
- target bundles include `anchor_ids`, `segment_ids`, and `segment_names` so downstream jobs can align by `anchor_id` instead of duplicate-prone `(patient_id, anchor_time)` keys

### Step 3. Build new regression targets on the same anchor rows

Completed on `2026-08-31`:

- regression target bundle: `outputs/targets/feature_targets_gap_full_data.npz`
- same `26` base target names across `0`, `20`, and `60` minute gap horizons as `feature_targets_gap_vasopressor_free.npz`
- gap-mode target lookup is segment-aware and stays within the same `segment_id`
- MAP `t+0m_gap` downstream alignment smoke check matched all `1969515` feature-cache anchors by `anchor_id`

### Step 4. Build new numerics windows and event targets

Completed and superseded by corrected anchor-onset v2 target bundles:

- extraction array `26931914`: completed
- merge `26931915`: completed
- merged numerics default output: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/numerics/full_data_v1`
- current hypotension target bundle: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.npz`
- current tachycardia/hypoxia target bundle: `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz`
- current target definitions, counts, and validation are documented in `docs/full_data/extractedFeaturesClassificationFullData.md`

### Step 5. Submit extracted-feature regression/classification jobs

Completed for the documented full-data extracted-feature comparison paths:

- submitter: `slurm/submit_feature_models_full_data_v7.sh`
- output root: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data`
- regression results: `docs/full_data/extractedFeaturesRegressionFullData.md`
- classification results: `docs/full_data/extractedFeaturesClassificationFullData.md`

## Bottom Line

Confirmed:

- this new dataset contains exactly `2056` patients and `2847597` windows
- it maps naturally to `mimic3_waveforms_matched` via `(patient_id, seg_name)`
- all current `887` vasopressor-free overlap patients are included inside it

Also confirmed:

- the older/general repo pipeline cannot be reused unchanged for the full-data cohort, because parts of it assume one continuous waveform segment per patient and the existing target-builder arrays are on a different timestamp grid

Therefore the remaining work is:

1. fix file-level read permissions or provide a readable replacement for full-data `X_stats.npy`
2. build `outputs/targets/feature_targets_gap_full_data.npz` with `slurm/build_full_data_feature_regression_targets.sh`
3. monitor target build `26931916` until the event target bundle finishes successfully
4. validate `outputs/targets/event_targets_full_data_anchor_horizon_filtered_5m_10m.npz` metadata/counts after `26931916` completes
5. submit full-data extracted-feature models with `slurm/submit_feature_models_full_data_v7.sh` and update the full-data result docs

The main unresolved blocker is file-level read access to `X_stats.npy` for regression. The `data_m3_120s_prediction` directory is accessible, but `X_stats.npy` itself is owned by `ms17929:ms17929` with mode `770`, so `dk5565` cannot open it until permissions/group ownership are changed or a readable copy is provided. Classification no longer depends on that file, but the submitted aligned numerics extraction/merge and target-build jobs must finish successfully before classification models can be launched.

# Full-Data Raw-Waveform Models - Confirmed Vasopressor Overlap

This page tracks raw-waveform PatchTST experiments on the full-data segment-aware cohort restricted to waveform segments with confirmed vasopressor overlap. It is the exposed-cohort counterpart to `docs/full_data/full_data_raw_waveform_models.md`, which uses high-confidence vasopressor-free segments.

## Cohort Definition

The cohort is built from the segment-level manifest:

```text
/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_waveform_manifest.csv
```

Included rows satisfy:

```text
has_vasopressor_overlap == True
```

Rows with uncertain or unknown vasopressor status are excluded. This is a conservative confirmed-exposure cohort, not the logical complement of vasopressor-free.

Manifest-level counts from the existing QC:

- confirmed vasopressor-overlap segments: `8658`
- high-confidence vasopressor-free segments: `21833`
- unknown/uncertain segments: `1553`

Built anchor-cache counts on `2026-09-03`:

- windows: `726876`
- segments: `8658`
- patients: `752`
- split windows: train `520040`, val `108835`, test `98001`

## Artifacts

Anchor cache:

```text
/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/raw_anchors/full_data_vasopressor_present_waveform_anchors
```

Regression target bundle:

```text
outputs/targets/feature_targets_gap_full_data_vasopressor_present.npz
```

Event target bundle:

```text
outputs/targets/event_targets_full_data_vasopressor_present_anchor_onset_v2_5m_10m.npz
```

Aligned numerics cache for event targets:

```text
/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/numerics/full_data_vasopressor_present_v1
```

Raw-waveform output run tags:

```text
full_data_vasopressor_present_v1_4ch_es
full_data_vasopressor_present_v1_4ch_events_es
full_data_vasopressor_present_v15_4ch_30s_es
full_data_vasopressor_present_v15_4ch_30s_events_es
```

## Saved Scripts

Anchor and target preparation:

- `scripts/build_full_data_vasopressor_present_anchor_cache.py`
- `slurm/build_full_data_vasopressor_present_anchor_cache.sh`
- `slurm/compute_full_data_patchtst_waveform_normalization_vasopressor_present.sh`
- `slurm/build_full_data_feature_regression_targets_vasopressor_present.sh`
- `slurm/extract_full_data_numerics_vasopressor_present_array.sh`
- `slurm/merge_full_data_numerics_vasopressor_present.sh`
- `slurm/build_full_data_event_targets_vasopressor_present.sh`

Model submission:

- `slurm/submit_patchtst_regression_t0_gap_full_data_4ch_vasopressor_present.sh`
- `slurm/submit_patchtst_classification_full_data_4ch_vasopressor_present.sh`
- `slurm/submit_patchtst_regression_t0_gap_full_data_4ch_30s_v15_vasopressor_present.sh`
- `slurm/submit_patchtst_classification_full_data_4ch_30s_v15_vasopressor_present.sh`

The raw-waveform anchor cache intentionally writes empty placeholder `values.npy` and `mask.npy` arrays with shape `(N, 0, 0)`. Existing target builders validate this cache contract, while raw-waveform training reads WFDB segments through `anchors.csv`.

## Model Setup

Regression uses the same 20-minute full-data raw-waveform setup as the vasopressor-free experiment:

- model variant: `patchtst_v1`
- channels: `II,ABP,PLETH,RESP`
- dataset format: `full_data_segments`
- input window: centered 20 minutes at 125 Hz, shape `(4, 150000)`
- target semantics: `t+0m_gap` for the initial regression training jobs
- target normalization: disabled
- waveform normalization: train-split stats computed from the confirmed-overlap anchor cache, not reused from vasopressor-free runs

Classification uses the same raw-waveform model family and the corrected full-data anchor-onset event-label semantics:

- events: hypotension, tachycardia, hypoxia
- horizons: `5m`, `10m`
- target mode: `anchor_onset_within_horizon`
- negative policy: `strict-clean-horizon`

Interpret classification under active vasopressor exposure carefully: predicting hypotension/tachycardia/hypoxia during treatment is a different clinical task than predicting events in vasopressor-free records.

## Completed Preparation, 2026-09-03

- Added and syntax-checked the vasopressor-present anchor, normalization, target, numerics, and model-submit scripts.
- Built a 100-row smoke anchor cache under `/tmp/full_data_vasopressor_present_anchor_smoke`.
- Built the full confirmed-overlap anchor cache with `726876` windows from `8658` segments and `752` patients.
- Built regression targets at `outputs/targets/feature_targets_gap_full_data_vasopressor_present.npz`, shape `(726876, 78)`, with `46882053` valid values.
- Smoke-tested one unnormalized val raw-waveform example from the new anchor cache: anchor `3386`, patient `p000188`, segment `p000188/3285727_0045`, waveform shape `(4, 150000)`, all finite after dataset loading.

Validation commands completed:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m py_compile   scripts/build_full_data_vasopressor_present_anchor_cache.py   scripts/build_full_data_feature_regression_targets.py   scripts/build_full_data_event_targets.py   scripts/compute_waveform_normalization_stats.py

bash -n   slurm/build_full_data_vasopressor_present_anchor_cache.sh   slurm/compute_full_data_patchtst_waveform_normalization_vasopressor_present.sh   slurm/build_full_data_feature_regression_targets_vasopressor_present.sh   slurm/build_full_data_event_targets_vasopressor_present.sh   slurm/extract_full_data_numerics_vasopressor_present_array.sh   slurm/merge_full_data_numerics_vasopressor_present.sh   slurm/submit_patchtst_regression_t0_gap_full_data_4ch_vasopressor_present.sh   slurm/submit_patchtst_classification_full_data_4ch_vasopressor_present.sh
```

## Completed Preparation Jobs, 2026-09-03

Completed prerequisite jobs:

- waveform normalization: SLURM `26985172`
- aligned numerics extraction array: SLURM `26985173` (`128` array tasks, all completed successfully; longest observed task `01:01:03`)
- numerics merge after extraction: SLURM `26985174`
- event target build after numerics merge: SLURM `26985175`

Regression and classification prerequisites are complete for both the 20-minute `patchtst_v1` and trailing-30-second `patchtst_v1_5` vasopressor-present submitters. Training jobs `26985976`-`26985995` have been submitted on `gl40s_short` and are pending by priority.

## Handoff Audit, 2026-09-03

Current state from repo artifacts and `squeue`:

- Anchor cache exists and has `_SUCCESS`, `anchors.csv`, `metadata.json`, and placeholder `values.npy`/`mask.npy` arrays with shape `(726876, 0, 0)`.
- Regression target bundle exists at `outputs/targets/feature_targets_gap_full_data_vasopressor_present.npz`, shape `(726876, 78)`, with `46882053` valid values.
- Waveform normalization stats are complete: `26985172` finished successfully in `00:30:19` and wrote `normalization_stats_patient_splits.json` under the anchor cache. Train-split stats: `II` mean `0.366344` std `0.283187`; `ABP` mean `75.452424` std `26.936483`; `PLETH` mean `1.434991` std `0.784668`; `RESP` mean `0.251278` std `0.571643`.
- Aligned numerics cache is complete: array `26985173` completed all `128` shards and merge job `26985174` completed in `00:01:09`. The merged cache under `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/numerics/full_data_vasopressor_present_v1` has `_SUCCESS`, `X_numerics.npy` shape `(726876, 5, 1200)`, `anchor_ids.npy`, `numerics_patient_ids.npy`, `numerics_seg_names.npy`, `numerics_window_times.npy`, `n_windows.txt`, and `numerics_metadata.json`. Metadata reports `726876` rows, `128` parts, vital names `ABP Mean`, `PULSE`, `SpO2`, `RESP`, and `HR`, `1200` samples/window, `1.0 Hz`, absolute center times.
- Event target bundle is complete: `26985175` completed in `00:07:52` and wrote `outputs/targets/event_targets_full_data_vasopressor_present_anchor_onset_v2_5m_10m.npz`, `.json`, and `.audit.csv`. The bundle has `726876` anchors and six target columns: `hypotension_onset_within_5m`, `tachycardia_onset_within_5m`, `hypoxia_onset_within_5m`, `hypotension_onset_within_10m`, `tachycardia_onset_within_10m`, and `hypoxia_onset_within_10m`. Timestamp audit context failures were `0`; stderr for merge and event-target jobs was empty.
- An attempted event-target build against the old vasopressor-free aligned numerics failed with `Feature anchors and numeric windows do not appear to use the same time coordinate`; do not reuse `numerics/full_data_v1` for this cohort because it is row-aligned to the vasopressor-free anchor cache.
- The original pending prep chain `26984462`-`26984465` was canceled and superseded after checking that `cpu_short` has `MaxTime=12:00:00`.
- Final SLURM accounting: `26985172` normalization `COMPLETED` (`00:30:19`), `26985174` merge `COMPLETED` (`00:01:09`), and `26985175` event targets `COMPLETED` (`00:07:52`); the `26985173` numerics array tasks completed successfully before merge. No `2698517*` jobs remained in `squeue` after the final check.
- Saved submitters cover both 20-minute `patchtst_v1` regression/classification and trailing-30-second `patchtst_v1_5` regression/classification for the vaso-present cohort. All twenty requested jobs have been submitted; no training results are available yet.


Filtered event target counts from the completed bundle:

| Target | Valid | Positives | Negatives | Positive Rate |
|---|---:|---:|---:|---:|
| `hypotension_onset_within_5m` | 296062 | 10720 | 285342 | 0.036209 |
| `tachycardia_onset_within_5m` | 200352 | 2520 | 197832 | 0.012578 |
| `hypoxia_onset_within_5m` | 160661 | 881 | 159780 | 0.005484 |
| `hypotension_onset_within_10m` | 280023 | 19887 | 260136 | 0.071019 |
| `tachycardia_onset_within_10m` | 193759 | 4758 | 189001 | 0.024556 |
| `hypoxia_onset_within_10m` | 154711 | 1644 | 153067 | 0.010626 |

Filtered negative-removal reasons describe what happened after base onset labels were built:

| Target | Kept negatives | Base invalid | Positive | After recording last-confirmed-onset cutoff | No confirmed-onset recording |
|---|---:|---:|---:|---:|---:|
| `hypotension_onset_within_5m` | 285342 | 298957 | 10720 | 67046 | 64811 |
| `tachycardia_onset_within_5m` | 197832 | 131057 | 2520 | 105431 | 290036 |
| `hypoxia_onset_within_5m` | 159780 | 121993 | 881 | 95353 | 348869 |
| `hypotension_onset_within_10m` | 260136 | 320191 | 19887 | 64025 | 62637 |
| `tachycardia_onset_within_10m` | 189001 | 147355 | 4758 | 102132 | 283630 |
| `hypoxia_onset_within_10m` | 153067 | 141383 | 1644 | 92044 | 338738 |

Meaning:

- `Kept negatives`: valid base strict-clean negatives that survived all filtering.
- `Base invalid`: rows already invalid before negative filtering.
- `Positive`: confirmed future-onset positives retained by the filters.
- `After recording last-confirmed-onset cutoff`: base negatives whose forecast endpoint occurred after the last confirmed onset in the same source recording.
- `No confirmed-onset recording`: base negatives from recordings with no confirmed onset for that event, so the recording was excluded from the negative-control set.

The event-target wrapper now explicitly pins the current classification-label definition: `--target-mode anchor_onset_within_horizon`, `--negative-policy strict-clean-horizon`, `--late-cutoff-group-scope recording`, `--late-cutoff-strategy group-last-positive`, `--exclude-late-cutoff-groups-without-positives`, and `--late-cutoff-candidate forecast_endpoint`. This matches the most recent full-data onset-label configuration rather than the older legacy `anchor_horizon_filtered` labels.


## Training Readiness, 2026-09-03

The vaso-present cohort is ready for the requested raw-waveform training launch, subject to GPU queue capacity. All four paths dry-ran successfully against the completed vaso-present artifacts, full-data patient splits, and 4-channel waveform cache (`II,ABP,PLETH,RESP`).

Ready dry-run commands:

```bash
DRY_RUN=1 FEATURE_LIST='MAP PLETH_ACDC ABP_tau dPdt_max' bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch_vasopressor_present.sh
DRY_RUN=1 EVENTS='hypotension tachycardia hypoxia' HORIZONS='5 10' bash slurm/submit_patchtst_classification_full_data_4ch_vasopressor_present.sh
DRY_RUN=1 FEATURE_LIST='MAP PLETH_ACDC ABP_tau dPdt_max' bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch_30s_v15_vasopressor_present.sh
DRY_RUN=1 EVENTS='hypotension tachycardia hypoxia' HORIZONS='5 10' bash slurm/submit_patchtst_classification_full_data_4ch_30s_v15_vasopressor_present.sh
```

Launch commands:

```bash
FEATURE_LIST='MAP PLETH_ACDC ABP_tau dPdt_max' bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch_vasopressor_present.sh
EVENTS='hypotension tachycardia hypoxia' HORIZONS='5 10' bash slurm/submit_patchtst_classification_full_data_4ch_vasopressor_present.sh
FEATURE_LIST='MAP PLETH_ACDC ABP_tau dPdt_max' bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch_30s_v15_vasopressor_present.sh
EVENTS='hypotension tachycardia hypoxia' HORIZONS='5 10' bash slurm/submit_patchtst_classification_full_data_4ch_30s_v15_vasopressor_present.sh
```

These commands would submit `20` jobs total: `4` 20-minute regression, `6` 20-minute classification, `4` 30-second regression, and `6` 30-second classification.


Additional preflight on 2026-09-03 found and fixed a launch-level issue for `patchtst_v1_5 --physiojepa-fidelity` jobs. The shared wrapper `slurm/train_patchtst.sh` previously injected architecture defaults (`--batch-size 512`, `--d-model 128`, `--n-layers 4`) before all user args; because the fidelity preset validates explicit CLI flags, those wrapper defaults conflicted with the required v1.5 preset (`batch_size=32`, `d_model=512`, `n_layers=3`). The wrapper now omits architecture-sensitive defaults whenever `--physiojepa-fidelity` is present, while preserving common defaults such as learning rate, workers, normalization, logging, and resume.

Validation after the fix:

- `bash -n` passed for `slurm/train_patchtst.sh` and all four vaso-present training submitters.
- Dry-runs passed for both 30-second vaso-present v1.5 submitters with the requested regression feature list and classification event/horizon set.
- Parser-level preflight with the real wrapper-style argument order passed for representative 30-second v1.5 regression (`MAP`) and classification (`hypotension`, 5 min) commands without starting training. The parsed fidelity architecture was `patch_len=125`, `stride=125`, `d_model=512`, `n_heads=8`, `n_layers=3`, `d_ff=2048`, and `batch_size=32`. A non-fidelity preflight also passed and preserved the existing `patchtst_v1` wrapper defaults (`d_model=128`, `n_heads=8`, `n_layers=4`, `batch_size=512`).
- This does not prove long GPU training will finish; queue/runtime, filesystem I/O, CUDA/node failures, and optimization behavior remain unverified until submitted jobs run.


Submitted training jobs on 2026-09-03:

- 20-minute `patchtst_v1` regression, run tag `full_data_vasopressor_present_v1_4ch_es`: `26985976` MAP, `26985977` PLETH_ACDC, `26985978` ABP_tau, `26985979` dPdt_max.
- 20-minute `patchtst_v1` classification, run tag `full_data_vasopressor_present_v1_4ch_events_es`: `26985980` hypotension 5m, `26985981` hypotension 10m, `26985982` tachycardia 5m, `26985983` tachycardia 10m, `26985984` hypoxia 5m, `26985985` hypoxia 10m.
- Trailing-30-second `patchtst_v1_5 --physiojepa-fidelity` regression, run tag `full_data_vasopressor_present_v15_4ch_30s_es`: `26985986` MAP, `26985987` PLETH_ACDC, `26985988` ABP_tau, `26985989` dPdt_max.
- Trailing-30-second `patchtst_v1_5 --physiojepa-fidelity` classification, run tag `full_data_vasopressor_present_v15_4ch_30s_events_es`: `26985990` hypotension 5m, `26985991` hypotension 10m, `26985992` tachycardia 5m, `26985993` tachycardia 10m, `26985994` hypoxia 5m, `26985995` hypoxia 10m.

Latest post-submit `squeue` check showed all twenty jobs pending on `gl40s_short` with reason `Priority`; approximate pending ranks among all `gl40s_short` jobs were `327`-`346`, and `squeue --start` reported `N/A` starts for them. No `logs/ptst_vaso*269859*.{out,err}` files or output roots existed yet, so no training results are available.

## A100 Duplicate Submission, 2026-09-03 10:02 EDT

The original `gl40s_short` jobs `26985976`-`26985995` were left untouched. To race the queue without log or output collisions, the four vaso-present submitters now support:

- `PARTITION`, default `gl40s_short`
- `EXCLUDE_NODES`, default empty
- `JOB_NAME_SUFFIX`, default empty
- `LOG_NAME_SUFFIX`, defaulting to `JOB_NAME_SUFFIX`
- `RUN_TAG`, already supported and used here with an `_a100` suffix

Syntax checks passed for all four submitters before launch. Dry-runs confirmed distinct `a100_short` partition, `_a100` job/log names, and `_a100` output run tags.

Submitted `a100_short` copies:

- 20-minute `patchtst_v1` regression, run tag `full_data_vasopressor_present_v1_4ch_es_a100`: `26988967` MAP, `26988968` PLETH_ACDC, `26988969` ABP_tau, and `26988970` dPdt_max.
- 20-minute `patchtst_v1` classification, run tag `full_data_vasopressor_present_v1_4ch_events_es_a100`: `26988971` hypotension 5m, `26988972` hypotension 10m, `26988973` tachycardia 5m, `26988974` tachycardia 10m, `26988975` hypoxia 5m, and `26988976` hypoxia 10m.
- Trailing-30-second `patchtst_v1_5 --physiojepa-fidelity` regression, run tag `full_data_vasopressor_present_v15_4ch_30s_es_a100`: `26988977` MAP, `26988978` PLETH_ACDC, `26988979` ABP_tau, and `26988980` dPdt_max.
- Trailing-30-second `patchtst_v1_5 --physiojepa-fidelity` classification, run tag `full_data_vasopressor_present_v15_4ch_30s_events_es_a100`: `26988985` hypotension 5m, `26988986` hypotension 10m, `26988987` tachycardia 5m, `26988988` tachycardia 10m, `26988989` hypoxia 5m, and `26988990` hypoxia 10m.

Initial post-submit `squeue` showed `26988967`-`26988970` running on `a100_short`; the remaining `a100_short` copies were pending with reason `Priority`. The first running logs used the expected `_a100` paths, but jobs that landed on `a100-4012` or `a100-4021` failed immediately during CUDA initialization with:

```text
RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

The bad-node placements were:

- `a100-4012`: `26988967` MAP regression, `26988971` hypotension 5m, `26988973` tachycardia 5m, `26988975` hypoxia 5m, `26988976` hypoxia 10m, `26988977` 30-second MAP, and `26988978` 30-second PLETH_ACDC.
- `a100-4021`: `26988970` dPdt_max regression, `26988972` hypotension 10m, and `26988974` tachycardia 10m.

`26988968` PLETH_ACDC and `26988969` ABP_tau continued running on `a100-4022` and `a100-4023`. The still-pending a100 copies were updated with `ExcNodeList=a100-4012,a100-4021`, and failed copies were resubmitted with `EXCLUDE_NODES=a100-4012,a100-4021`:

- replacement 20-minute regression: `26989008` MAP and `26989009` dPdt_max
- replacement 20-minute classification: `26989012` hypotension 5m, `26989013` hypotension 10m, `26989014` tachycardia 5m, `26989015` tachycardia 10m, `26989016` hypoxia 5m, and `26989017` hypoxia 10m
- replacement 30-second regression: `26989018` MAP and `26989019` PLETH_ACDC

Latest `squeue` after replacements showed an active a100 set of twenty jobs: `26988968`, `26988969`, `26988979`, `26988980`, `26988985`-`26988990`, `26989008`, `26989009`, and `26989012`-`26989019`.

## GL40S Cancellation, 2026-09-03 10:11 EDT

After confirming the distinct `a100_short` copies were active or queued, the original pending `gl40s_short` vaso-present jobs were canceled:

- 20-minute regression: `26985976`-`26985979`
- 20-minute classification: `26985980`-`26985985`
- 30-second v1.5 regression: `26985986`-`26985989`
- 30-second v1.5 classification: `26985990`-`26985995`

Post-cancel `squeue` showed no remaining entries for `26985976`-`26985995`. The `a100_short` copies remained present.

## A100 Progress Check, 2026-09-03 10:30 EDT

Current live training jobs:

- `26988968` PLETH_ACDC 20-minute regression: running on `a100-4022`; log reached epoch 1 step 900/917 with no fatal error.
- `26988969` ABP_tau 20-minute regression: running on `a100-4023`; log reached epoch 1 step 900/923 with no fatal error.
- `26988980` dPdt_max trailing-30-second regression: running on `a100-4023`; log reached epoch 1 step 6400/14791 at about 18.6 steps/s with no fatal error.

Additional A100 jobs landed on `a100-4028` and failed during CUDA initialization with the same CUDA-unavailable error as the earlier `a100-4012`/`a100-4021` failures. Newly failed IDs:

- 30-second regression: `26988979` ABP_tau.
- 30-second classification: `26988985` hypotension 5m, `26988986` hypotension 10m, `26988987` tachycardia 5m, `26988988` tachycardia 10m, `26988989` hypoxia 5m, and `26988990` hypoxia 10m.
- 20-minute regression/classification replacements: `26989008` MAP, `26989009` dPdt_max, `26989012` hypotension 5m, `26989013` hypotension 10m, `26989014` tachycardia 5m, `26989015` tachycardia 10m, and `26989016` hypoxia 5m.

The remaining pending A100 jobs `26989017` hypoxia 10m and `26989018`-`26989019` 30-second MAP/PLETH_ACDC were updated to `ExcNodeList=a100-4012,a100-4021,a100-4028`. Failed copies were resubmitted with `EXCLUDE_NODES=a100-4012,a100-4021,a100-4028`:

- 20-minute regression: `26989262` MAP and `26989263` dPdt_max.
- 20-minute classification: `26989264` hypotension 5m, `26989265` hypotension 10m, `26989266` tachycardia 5m, `26989267` tachycardia 10m, `26989268` hypoxia 5m, and `26989269` hypoxia 10m.
- 30-second regression: `26989270` ABP_tau.
- 30-second classification: `26989271` hypotension 5m, `26989272` hypotension 10m, `26989273` tachycardia 5m, `26989274` tachycardia 10m, `26989275` hypoxia 5m, and `26989276` hypoxia 10m.

Post-resubmission `squeue` showed running A100 jobs `26988968`, `26988969`, and `26988980`; pending A100 jobs `26989017`-`26989019` and `26989262`-`26989276`; and all pending jobs waiting by `Priority`. Spot checks with `scontrol show job` confirmed the expanded exclusion on the older pending jobs and new replacements.

## Commands

Rebuild the anchor cache:

```bash
sbatch slurm/build_full_data_vasopressor_present_anchor_cache.sh
```

Compute waveform normalization:

```bash
sbatch slurm/compute_full_data_patchtst_waveform_normalization_vasopressor_present.sh
```

Build regression targets:

```bash
sbatch slurm/build_full_data_feature_regression_targets_vasopressor_present.sh
```

Build classification targets:

```bash
num=$(sbatch slurm/extract_full_data_numerics_vasopressor_present_array.sh | awk '{print $NF}')
merge=$(sbatch --dependency=afterok:${num} slurm/merge_full_data_numerics_vasopressor_present.sh | awk '{print $NF}')
sbatch --dependency=afterok:${merge} slurm/build_full_data_event_targets_vasopressor_present.sh
```

Dry-run selected submissions:

```bash
DRY_RUN=1 FEATURE_LIST=MAP bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch_vasopressor_present.sh
DRY_RUN=1 EVENTS=hypotension HORIZONS=5 bash slurm/submit_patchtst_classification_full_data_4ch_vasopressor_present.sh
DRY_RUN=1 FEATURE_LIST=MAP bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch_30s_v15_vasopressor_present.sh
DRY_RUN=1 EVENTS=hypotension HORIZONS=5 bash slurm/submit_patchtst_classification_full_data_4ch_30s_v15_vasopressor_present.sh
```

Resubmit training only after canceling the current queued jobs or choosing distinct `RUN_TAG` values to avoid duplicate jobs writing to the same output roots:

```bash
FEATURE_LIST='MAP PLETH_ACDC ABP_tau dPdt_max' bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch_vasopressor_present.sh
EVENTS='hypotension tachycardia hypoxia' HORIZONS='5 10' bash slurm/submit_patchtst_classification_full_data_4ch_vasopressor_present.sh
FEATURE_LIST='MAP PLETH_ACDC ABP_tau dPdt_max' bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch_30s_v15_vasopressor_present.sh
EVENTS='hypotension tachycardia hypoxia' HORIZONS='5 10' bash slurm/submit_patchtst_classification_full_data_4ch_30s_v15_vasopressor_present.sh
```

## Next Steps

1. Monitor SLURM jobs `26985976`-`26985995`; inspect `logs/ptst_vaso*269859*.{out,err}` as soon as they start.
2. Verify the four output roots after jobs start: `outputs/patchtst/full_data_vasopressor_present_v1_4ch_es`, `outputs/patchtst/full_data_vasopressor_present_v1_4ch_events_es`, `outputs/patchtst/full_data_vasopressor_present_v15_4ch_30s_es`, and `outputs/patchtst/full_data_vasopressor_present_v15_4ch_30s_events_es`.
3. Evaluate only after final checkpoints are written; require aligned `test_metrics.json` and `test_predictions.npz` before comparing vaso-present, vasopressor-free, 20-minute, and 30-second models.
4. If queue latency becomes unacceptable, consider canceling and resubmitting a small subset to `a100_short` or `gl40s_long` with distinct run tags; avoid duplicate jobs writing to the same run tag.

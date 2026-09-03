# Full-Data Raw-Waveform Models

This page tracks raw-waveform models trained on the full-data segment-aware vasopressor-free cohort from `data_m3_120s_prediction`. The current workstream uses the same raw-waveform `patchtst_v1` architecture for full-data regression and classification, with 4 input channels (`II,ABP,PLETH,RESP`) and segment-aware `anchor_id` target alignment.

Detailed extracted-feature target construction and extracted-feature model results remain in `docs/full_data/extractedFeaturesRegressionFullData.md` and `docs/full_data/extractedFeaturesClassificationFullData.md`.

## Shared Code Paths

- `waveform_baselines/full_data_dataset.py`: segment-aware WFDB-backed raw waveform loader for full-data windows.
- `scripts/train_patchtst.py --dataset-format full_data_segments`: training path that uses `anchor_id` target lookup when available.
- `scripts/eval_patchtst.py --dataset-format full_data_segments`: evaluation path that exports `sample_ids`, `patient_time_sample_ids`, `patient_ids`, `anchor_times`, `anchor_ids`, `segment_ids`, and `segment_names`.
- `scripts/compute_waveform_normalization_stats.py --channels II,ABP,PLETH,RESP`: full-data waveform-normalization support.
- `slurm/compute_full_data_patchtst_waveform_normalization.sh`: computes the required full-data train-split waveform stats.
- `slurm/submit_patchtst_regression_t0_gap_full_data_4ch.sh`: submits all 26 raw-waveform regression jobs.
- `slurm/submit_patchtst_classification_full_data_4ch.sh`: submits the raw-waveform classification jobs.

## Shared Data And Normalization

- Waveform source: full-data feature cache `anchors.csv` under `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v7/full_data_vasopressor_free_waveform_features_v7`.
- Split discipline: full-data `patient_splits.json`, with split labels carried by the anchor cache.
- Dataset format: `full_data_segments`.
- Channels: `II,ABP,PLETH,RESP`.
- Waveform normalization: train-split 4-channel stats computed from unique training-split full-data segments referenced by `anchors.csv`; target normalization is disabled for regression.


## Handoff Status, 2026-09-02 18:20 EDT

Fresh handoff check from repo outputs and `squeue -u dk5565`:

- Running regression jobs: `26957107` PPV, `26957108` PVI, `26957109` PTT, `26957110` dPdt_max, `26957111` ABP_tau, and `26957112` RESP_amp.
- Pending regression jobs on `gl40s_short` priority: `26957113` PLETH_ACDC_PLETH_amp, `26957114` ABP_area_ABP_tau, `26957115` ABP_area_ShockIdx, `26957116` PLETH_amp_ShockIdx, `26957117` PLETH_ACDC_ShockIdx, `26957118` ShockIdx_ABP_tau, and `26957119` PLETH_ACDC_ABP_tau.
- Pending classification jobs on `gl40s_short` priority: `26957120` hypotension 5m, `26957121` hypotension 10m, `26957122` tachycardia 5m, `26957123` tachycardia 10m, `26957124` hypoxia 5m, and `26957125` hypoxia 10m.
- Regression output artifact counts under `outputs/patchtst/full_data_v1_4ch_es`: `26` run directories, `26` `config.json`, `19` `best_model.pt`, `19` `latest_model.pt`, `8` `test_metrics.json`, and `8` `test_predictions.npz`.
- Regression evaluation exports observed for: `HR`, `DBP`, `MAP`, `ABP_area`, `ECG_Ramp`, `HRV_RMSSD`, `HR_range`, and `ABP_tau`. Treat these as partial exports until the remaining checkpoints finish and all targets are evaluated consistently.
- Classification output artifact counts under `outputs/patchtst/full_data_v1_4ch_events_es`: `4` run directories with `config.json` only; `0` checkpoints, `0` metrics, and `0` prediction exports. Hypoxia has not created output directories yet.
- Do not use the full-data raw-waveform runs in downstream significance or Bland-Altman comparisons until all intended checkpoints have completed and aligned evaluation exports have been audited.

After all checkpoints finish, evaluate completed regression runs with:

```bash
sbatch slurm/eval_patchtst.sh --output-base outputs/patchtst/full_data_v1_4ch_es
```

Run or adapt the same evaluation path for `outputs/patchtst/full_data_v1_4ch_events_es` after the classification jobs have checkpoints.


## Deadline Cancellation, 2026-09-03

To reduce active SLURM load for near-term deadlines, the remaining full-data 20-minute raw-waveform `patchtst_v1` regression work was narrowed temporarily to `ABP_tau` and `dPdt_max`. The canceled targets should be resumed soon after the deadline pressure clears; this was a scheduling decision, not an experimental conclusion.

- Kept running regression jobs: `26957110` dPdt_max and `26957111` ABP_tau.
- Canceled running/pending regression jobs: `26957107` PPV, `26957108` PVI, `26957109` PTT, `26957112` RESP_amp, `26957113` PLETH_ACDC_PLETH_amp, `26957114` ABP_area_ABP_tau, `26957115` ABP_area_ShockIdx, `26957116` PLETH_amp_ShockIdx, `26957117` PLETH_ACDC_ShockIdx, `26957118` ShockIdx_ABP_tau, and `26957119` PLETH_ACDC_ABP_tau.
- Kept regression evaluation jobs: `26981967` PLETH_ACDC, `26981973` dPdt_max, and `26981974` ABP_tau. MAP already had a local evaluation export and did not have an active eval job in this cancellation pass.
- Canceled regression evaluation jobs: `26981964` RR, `26981965` SBP, `26981966` PP, `26981968` PLETH_amp, `26981969` ShockIdx, `26981970` PPV, `26981971` PVI, `26981972` PTT, `26981975` RESP_amp, `26981976` PLETH_ACDC_PLETH_amp, `26981977` ABP_area_ABP_tau, `26981978` ABP_area_ShockIdx, `26981979` PLETH_amp_ShockIdx, `26981980` PLETH_ACDC_ShockIdx, `26981981` ShockIdx_ABP_tau, and `26981982` PLETH_ACDC_ABP_tau.
- Classification jobs were intentionally left untouched.

## Handoff Audit, 2026-09-03

Current state from repo artifacts, logs, and `squeue` after the deadline cancellation pass:

- Running regression jobs: `26957110` dPdt_max and `26957111` ABP_tau. Recent logs show dPdt_max in epoch `21` and ABP_tau in epoch `25`; neither should be treated as final until the jobs exit successfully and final evals complete.
- Running classification jobs: `26957120` hypotension 5m, `26957121` hypotension 10m, and `26957123` tachycardia 10m. Pending classification jobs: `26957124` hypoxia 5m and `26957125` hypoxia 10m. `26957122` tachycardia 5m is no longer in `squeue`; its eval job `26981985` is pending on priority.
- Pending kept eval jobs: `26981967` PLETH_ACDC by priority, plus dependency-held `26981973` dPdt_max and `26981974` ABP_tau. Classification eval `26981985` tachycardia 5m is pending on priority; `26981983`, `26981984`, and `26981986`-`26981988` remain dependency-held behind their training jobs.
- Artifact audit under `outputs/patchtst/full_data_v1_4ch_es`: `26` run directories, `26` configs, `21` `best_model.pt`, `21` `latest_model.pt`, and `8` test metric/prediction exports. Existing exports remain partial.
- Artifact audit under `outputs/patchtst/full_data_v1_4ch_events_es`: `4` run directories, `4` configs, `4` checkpoints, and no test metric/prediction exports.
- Canceled regression targets and evals in the deadline section should be resumed after deadline pressure clears; the cancellation does not imply those targets failed or are scientifically unimportant.

## Status Check, 2026-09-03 09:53 EDT

Current state from live `squeue`, logs, and artifacts:

- `squeue -u dk5565` shows only kept regression training job `26957110` dPdt_max still running on `gl40s_short`; latest log reached epoch `32`.
- Dependent dPdt_max eval `26981973` remains pending on dependency; no `ptst_eval_dPdt_max_26981973` log files exist yet.
- Final eval `26981967` for `PLETH_ACDC` completed at `07:28 EDT`: R2 `0.8879`, corr `0.9424`, RMSE `0.0917`, MAE `0.0604`, with `257659` valid test windows.
- Final eval `26981974` for `ABP_tau` completed at `09:39 EDT`: R2 `0.6510`, corr `0.8121`, RMSE `0.4150`, MAE `0.1907`, with `258337` valid test windows. This supersedes the earlier interim ABP_tau export.
- The 20-minute classification training jobs `26957120`-`26957125` completed via early stopping and all six dependent eval exports are present under `outputs/patchtst/full_data_v1_4ch_events_es`.
- Classification test metrics now available: hypotension 5m AUROC `0.9106`, AUPRC `0.2141`; hypotension 10m AUROC `0.8769`, AUPRC `0.2644`; tachycardia 5m AUROC `0.8944`, AUPRC `0.2107`; tachycardia 10m AUROC `0.8915`, AUPRC `0.3287`; hypoxia 5m AUROC `0.5508`, AUPRC `0.004242`; hypoxia 10m AUROC `0.5563`, AUPRC `0.008415`.
- Regression artifact counts under `outputs/patchtst/full_data_v1_4ch_es`: `26` run directories, `26` configs, `21` checkpoints, and `9` aligned test metric/prediction exports. Remaining dPdt_max export is still pending completion of `26957110` and eval `26981973`.
- Classification artifact counts under `outputs/patchtst/full_data_v1_4ch_events_es`: `6` run directories, `6` configs, `6` checkpoints, and `6` aligned test metric/prediction exports.
- `sacct` was unavailable during this check due a Slurm accounting connection failure, so completed states are inferred from logs, artifacts, and absence from `squeue`.

## Related 30-Second Raw-Waveform Models

The separate trailing-30-second `patchtst_v1_5` full-data setup is tracked in `docs/full_data/full_data_30s_waveform_models.md`.

## Test Evaluation Status - 2026-09-02

A local evaluation run was started for checkpointed full-data raw-waveform PatchTST regression outputs, then stopped after the user requested the remaining work be submitted to Slurm. It produced test metrics and prediction exports for `8` target directories. One of those targets, `ABP_tau`, was evaluated from an interim checkpoint while training job `26957111` was still running; dependent final evaluation job `26981974` was submitted and should overwrite it after training completes.

Available local evaluation outputs under `outputs/patchtst/full_data_v1_4ch_es`:

| Target | R2 | Corr | RMSE | MAE | Valid Test Windows | Checkpoint Epoch | Caveat |
|---|---:|---:|---:|---:|---:|---:|---|
| `ABP_area_t_plus_0m_gap` | 0.8879 | 0.9435 | 2.0192 | 1.2299 | 258,982 | 30 | final checkpoint |
| `MAP_t_plus_0m_gap` | 0.8350 | 0.9151 | 7.0552 | 4.3995 | 258,861 | 40 | final checkpoint |
| `DBP_t_plus_0m_gap` | 0.8185 | 0.9072 | 6.2391 | 3.7942 | 258,473 | 13 | final checkpoint |
| `HR_t_plus_0m_gap` | 0.6793 | 0.8266 | 8.0860 | 5.1155 | 259,254 | 11 | final checkpoint |
| `HR_range_t_plus_0m_gap` | 0.6336 | 0.8013 | 12.8853 | 9.3967 | 259,255 | 9 | final checkpoint |
| `HRV_RMSSD_t_plus_0m_gap` | 0.6176 | 0.7947 | 39.6531 | 28.0944 | 259,255 | 21 | final checkpoint |
| `ABP_tau_t_plus_0m_gap` | 0.6105 | 0.7820 | 0.4384 | 0.2055 | 258,337 | 12 | interim; final eval pending `26981974` |
| `ECG_Ramp_t_plus_0m_gap` | 0.3415 | 0.5931 | 0.2750 | 0.2211 | 259,261 | 6 | final checkpoint |

Slurm evaluation submissions for remaining/final metrics:

- Immediate completed-regression evals pending on GPU priority: `26981964` RR, `26981965` SBP, `26981966` PP, `26981967` PLETH_ACDC, `26981968` PLETH_amp, and `26981969` ShockIdx.
- Dependency-held regression evals: `26981970` PPV after `26957107`, `26981971` PVI after `26957108`, `26981972` PTT after `26957109`, `26981973` dPdt_max after `26957110`, `26981974` ABP_tau after `26957111`, `26981975` RESP_amp after `26957112`, `26981976` PLETH_ACDC_PLETH_amp after `26957113`, `26981977` ABP_area_ABP_tau after `26957114`, `26981978` ABP_area_ShockIdx after `26957115`, `26981979` PLETH_amp_ShockIdx after `26957116`, `26981980` PLETH_ACDC_ShockIdx after `26957117`, `26981981` ShockIdx_ABP_tau after `26957118`, and `26981982` PLETH_ACDC_ABP_tau after `26957119`.
- Dependency-held classification evals: `26981983` hypotension 5m after `26957120`, `26981984` hypotension 10m after `26957121`, `26981985` tachycardia 5m after `26957122`, `26981986` tachycardia 10m after `26957123`, `26981987` hypoxia 5m after `26957124`, and `26981988` hypoxia 10m after `26957125`.

Evaluation infrastructure changes made for this handoff:

- `scripts/eval_patchtst.py --all --skip-existing` skips directories that already have both `test_metrics.json` and `test_predictions.npz`.
- `slurm/eval_patchtst.sh` now defaults to `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python`, because `/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python` is not present on this node.
- `slurm/eval_patchtst.sh` supports `EVAL_ALL=0` for single-checkpoint evaluation jobs.

## Regression Setup

A 4-channel raw-waveform `patchtst_v1` path has been prepared for the same full-data segment-aware vasopressor-free cohort used by the extracted-feature regression runs.

Key choices:

- model variant: `patchtst_v1`
- channels: `II,ABP,PLETH,RESP`
- dataset format: `full_data_segments`
- waveform source: full-data feature cache `anchors.csv`, which supplies `anchor_id`, split labels, segment IDs, segment names, and WFDB segment paths
- target bundle: `outputs/targets/feature_targets_gap_full_data_hardened_v2.npz`
- target semantics: `t+0m_gap`
- target normalization: disabled
- waveform normalization: train-split 4-channel stats computed from unique training-split full-data segments referenced by `anchors.csv`
- output run tag: `full_data_v1_4ch_es`

Prepared code paths:

- `waveform_baselines/full_data_dataset.py`: segment-aware WFDB-backed raw waveform loader for full-data windows
- `scripts/train_patchtst.py --dataset-format full_data_segments`: training path that uses `anchor_id` target lookup when available
- `scripts/eval_patchtst.py --dataset-format full_data_segments`: evaluation path that exports `sample_ids`, `patient_time_sample_ids`, `patient_ids`, `anchor_times`, `anchor_ids`, `segment_ids`, and `segment_names`
- `scripts/compute_waveform_normalization_stats.py --channels II,ABP,PLETH,RESP`: full-data normalization support
- `slurm/compute_full_data_patchtst_waveform_normalization.sh`: computes the required full-data train-split waveform stats
- `slurm/submit_patchtst_regression_t0_gap_full_data_4ch.sh`: submits all 26 raw-waveform regression jobs

Current status:

- First full-data 4-channel waveform-normalization submission `26950998` failed immediately because the wrapper did not expose the repo package on `PYTHONPATH`.
- The wrapper was fixed and replacement normalization job `26951001` was submitted on `2026-09-01`; initial `squeue` showed it running on `cpu_medium`.
- Replacement normalization job `26951001` completed and wrote `normalization_stats_patient_splits.json` under the full-data anchor cache.
- Training jobs were submitted as SLURM `26954507`-`26954532` on `2026-09-01`. Follow-up status check found seven jobs still running (`26954507`-`26954513`) with `best_model.pt`, `latest_model.pt`, and `config.json`; the other 19 jobs (`26954514`-`26954532`) left the queue before epoch output and stderr ends in a TorchInductor remote/autotune cache `json.decoder.JSONDecodeError`. No evaluation outputs exist yet.
- The failure was addressed by making `slurm/train_patchtst.sh` use job-local `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and `CUDA_CACHE_PATH` values under `/tmp`. `scripts/train_patchtst.py` also supports `PATCHTST_DISABLE_COMPILE=1` as a fallback if compile-cache failures recur.
- The 19 failed regression targets were resubmitted on `2026-09-01` as SLURM `26957101`-`26957119` using the same run tag and output directories. These jobs were pending on `gl40s_short` priority at the post-submit check.

Saved full-data 4-channel waveform-normalization stats:

| Channel | Mean | Std | Valid Count |
|---|---:|---:|---:|
| `II` | `0.302546` | `0.317446` | `28776115393` |
| `ABP` | `86.060442` | `31.186258` | `28775938092` |
| `PLETH` | `1.538612` | `0.765433` | `28775573554` |
| `RESP` | `0.332023` | `0.402085` | `28776313043` |

The 26-target raw-waveform batch was launched with:

```bash
bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch.sh
```

Dry-run the commands with:

```bash
DRY_RUN=1 bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch.sh
```

Evaluate completed checkpoints with:

```bash
sbatch slurm/eval_patchtst.sh --output-base outputs/patchtst/full_data_v1_4ch_es
```

Validation completed before submission:

- Syntax check passed for `waveform_baselines/full_data_dataset.py`, `waveform_baselines/normalization.py`, `scripts/train_patchtst.py`, `scripts/eval_patchtst.py`, `scripts/compute_waveform_normalization_stats.py`, and `tests/test_full_data_patchtst_dataset.py`.
- Focused tests passed: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_full_data_patchtst_dataset tests.test_numpy_dataset_config tests.test_eval_patchtst_config_loading`.
- Submitter dry run produced all 26 expected `patchtst_v1` commands with `--dataset-format full_data_segments`, `--channels II,ABP,PLETH,RESP`, `--n-channels 4`, `--feature-horizon-mode gap`, and no `--normalize-targets`.
- Follow-up code review fixed a full-data grouping bug in `SingleTargetDataset`: the wrapper no longer assumes `len(patient_ids) == len(patient_boundaries)`, because the full-data loader groups boundaries by segment while patient IDs are unique patients. The focused test now covers multiple segments for one patient.
- Real full-data index construction passed with counts: train `1,416,785` windows / `1,229` patients / `15,739` segment groups; val `269,872` / `255` / `2,919`; test `282,858` / `274` / `3,175`.
- Real MAP target-filtering smoke passed without waveform loading: train `1,295,977` valid windows, val `246,474` valid windows.
- Real WFDB sample smoke passed for val anchor `605` (`p000188/3285727_0007`): waveform shape `(4, 150000)`, dtype `float32`, finite values, valid target. A `patchtst_v1` forward pass returned finite shape `(1, 1)`.

## Classification Setup

A 4-channel raw-waveform `patchtst_v1` classification batch was prepared using the same segment-aware full-data loader and waveform normalization as the full-data raw-waveform regression batch.

Key choices:

- model variant: `patchtst_v1`
- channels: `II,ABP,PLETH,RESP`
- dataset format: `full_data_segments`
- waveform source: full-data feature cache `anchors.csv`, which supplies `anchor_id`, split labels, segment IDs, segment names, and WFDB segment paths
- waveform normalization: `normalization_stats_patient_splits.json` under the full-data anchor cache
- run tag: `full_data_v1_4ch_events_es`
- epochs: `50`
- early stopping: patience `5`, minimum epochs `10`, minimum delta `0.0`

Submitted jobs on `2026-09-01`:

| Event | Horizon | Target Bundle | SLURM |
|---|---:|---|---:|
| `hypotension` | `5m` | `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.npz` | `26954651` |
| `hypotension` | `10m` | `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.npz` | `26954652` |
| `tachycardia` | `5m` | `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz` | `26954653` |
| `tachycardia` | `10m` | `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz` | `26954654` |
| `hypoxia` | `5m` | `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz` | `26954655` |
| `hypoxia` | `10m` | `outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_tachycardia_hypoxia_recording_complete_scan_filtered.npz` | `26954656` |

Initial `squeue` check showed all six jobs pending on `gl40s_short` priority. Follow-up status check found that all six jobs had left the queue without checkpoints or evaluation outputs. Hypotension and tachycardia jobs wrote `config.json` then failed before epoch output with the same TorchInductor remote/autotune cache `json.decoder.JSONDecodeError` seen in failed regression jobs. Hypoxia jobs failed during argument parsing because `train_patchtst.py --event-name` accepted only `hypotension` and `tachycardia`.

The classification path was fixed on `2026-09-01` so PatchTST accepts `hypoxia` when named event-target columns are available, while preserving legacy two-event indexing for unnamed target bundles. `scripts/eval_patchtst.py` was updated to accept hypoxia for later evaluation. All six failed classification jobs were resubmitted as SLURM `26957120`-`26957125` and were pending on `gl40s_short` priority at the post-submit check.

## Handoff Status - 2026-09-01

Current `squeue -u dk5565` state for raw-waveform PatchTST jobs:

- Running regression jobs: `26954507` HR, `26954508` RR, `26954509` SBP, `26954510` DBP, `26954511` PP, `26954512` MAP, `26954513` ABP_area.
- Failed regression jobs: `26954514` PLETH_ACDC, `26954515` PLETH_amp, `26954516` ECG_Ramp, `26954517` HRV_RMSSD, `26954518` HR_range, `26954519` ShockIdx, `26954520` PPV, `26954521` PVI, `26954522` PTT, `26954523` dPdt_max, `26954524` ABP_tau, `26954525` RESP_amp, `26954526` PLETH_ACDC_PLETH_amp, `26954527` ABP_area_ABP_tau, `26954528` ABP_area_ShockIdx, `26954529` PLETH_amp_ShockIdx, `26954530` PLETH_ACDC_ShockIdx, `26954531` ShockIdx_ABP_tau, and `26954532` PLETH_ACDC_ABP_tau.
- Failed classification jobs: `26954651` hypotension 5m, `26954652` hypotension 10m, `26954653` tachycardia 5m, `26954654` tachycardia 10m, `26954655` hypoxia 5m, `26954656` hypoxia 10m.
- Resubmitted regression jobs: `26957101`-`26957119`, corresponding in order to the failed regression target list above.
- Resubmitted classification jobs: `26957120` hypotension 5m, `26957121` hypotension 10m, `26957122` tachycardia 5m, `26957123` tachycardia 10m, `26957124` hypoxia 5m, and `26957125` hypoxia 10m.

Running regression logs show normal early-epoch training progress: HR reached epoch `5`, RR `7`, SBP `4`, DBP `4`, PP `5`, MAP `7`, and ABP_area `6` at the latest check. Stderr for these running jobs contains PyTorch warnings about nested tensors and learning-rate scheduler call order, but no fatal error in the inspected tails.

The failed regression jobs created only `config.json` under their output directories and no `best_model.pt`, `latest_model.pt`, `test_metrics.json`, or `test_predictions.npz`. The failed hypotension/tachycardia classification jobs similarly created only `config.json`. The failed hypoxia jobs created no output directory.

Accounting caveat: `sacct` was temporarily unavailable during the follow-up status check, so completed/failed state is inferred from absence in `squeue`, logs, and output artifacts.


### Status Check - 2026-09-01 17:55 EDT

`squeue -u dk5565` at `2026-09-01 17:55 EDT` shows the original seven full-data raw-waveform PatchTST regression jobs still running: `26954507` HR, `26954508` RR, `26954509` SBP, `26954510` DBP, `26954511` PP, `26954512` MAP, and `26954513` ABP_area. Resubmitted regression job `26957101` for PLETH_ACDC is also running. Resubmitted regression jobs `26957102`-`26957119` and classification jobs `26957120`-`26957125` remain pending with reason `QOSMaxMemoryPerUser`.

Output artifact counts: regression output directories exist for all `26` targets, with `config.json` in all `26` and `best_model.pt`/`latest_model.pt` in `8/26`; no regression `test_metrics.json` or `test_predictions.npz` files exist yet. Classification output directories exist for the four jobs that previously reached config writing, with `config.json` in `4/4`; there are still no classification checkpoints or evaluation outputs.

Recent running regression logs show ongoing training rather than fatal errors: HR around epoch `13`, RR `15`, SBP `11`, DBP `11`, PP `13`, MAP `16`, ABP_area `15`, and resubmitted PLETH_ACDC `2`. Evaluation still needs to be run after training checkpoints finish.

Training jobs save checkpoints/config/history. Downstream statistical-significance testing requires running `scripts/eval_patchtst.py` through `slurm/eval_patchtst.sh` after checkpoints finish; eval writes `test_predictions.npz` with predictions, targets, masks, `sample_ids`, `patient_time_sample_ids`, `patient_ids`, `anchor_times`, `anchor_ids`, `segment_ids`, and `segment_names`.

# Full-Data 30-Second Raw-Waveform Models

This page tracks raw-waveform models trained on the full-data segment-aware vasopressor-free cohort from `data_m3_120s_prediction` using only the trailing `30` seconds of the original 20-minute input horizon. The targets remain the same as the full-data raw-waveform and extracted-feature workstreams: segment-aware `anchor_id` alignment, full-data patient splits, and the existing regression/event target bundles.

The 20-minute full-data raw-waveform `patchtst_v1` jobs remain documented in `docs/full_data/full_data_raw_waveform_models.md`.

## PatchTST v1.5 Setup

A separate 4-channel raw-waveform `patchtst_v1_5` path has been prepared to use only the trailing `30` seconds of the existing full-data input horizon while preserving the same segment-aware targets. The target bundles, `anchor_id` alignment, split discipline, and regression/event label semantics are unchanged from the 20-minute full-data raw-waveform runs.

Key choices:

- model variant: `patchtst_v1_5` with `--physiojepa-fidelity`
- channels: `II,ABP,PLETH,RESP`
- dataset format: `full_data_segments`
- input crop: `--seq-len 3750 --input-window-position input_end`, which reads the final `30` seconds ending at the original 20-minute input boundary
- tokenizer under the fidelity preset: `patch_len=125`, `stride=125`, so each token is `1` second at `125 Hz` and each channel contributes `30` tokens
- regression targets: same `outputs/targets/feature_targets_gap_full_data_hardened_v2.npz` bundle and `t+0m_gap` semantics
- event targets: same full-data onset target bundles used by the 20-minute full-data raw-waveform classification setup
- waveform normalization: same train-split 4-channel normalization stats under the full-data anchor cache
- regression run tag: `full_data_v15_4ch_30s_es`
- classification run tag: `full_data_v15_4ch_30s_events_es`

Prepared code paths:

- `waveform_baselines/full_data_dataset.py`: added `input_window_position`; default `center` preserves existing behavior, while `input_end` returns the trailing crop from the original input interval.
- `scripts/train_patchtst.py`: saves and uses `input_window_position` in `TrainConfig`, limited to `full_data_segments` for non-centered crops.
- `scripts/eval_patchtst.py`: accepts/saves explicit `--input-window-position` overrides and otherwise evaluates from the saved training config.
- `slurm/submit_patchtst_regression_t0_gap_full_data_4ch_30s_v15.sh`: prepared all-target regression submitter.
- `slurm/submit_patchtst_classification_full_data_4ch_30s_v15.sh`: prepared event-classification submitter.
- `slurm/train_patchtst.sh`: on 2026-09-03, fixed the shared wrapper so `--physiojepa-fidelity` jobs no longer inherit conflicting architecture defaults (`batch_size=512`, `d_model=128`, `n_layers=4`). Fidelity jobs now let `scripts/train_patchtst.py` apply the strict preset (`batch_size=32`, `d_model=512`, `n_layers=3`, `d_ff=2048`, `patch_len=stride=125`).

Validation completed before submission:

- Syntax check passed for `waveform_baselines/full_data_dataset.py`, `scripts/train_patchtst.py`, `scripts/eval_patchtst.py`, and `tests/test_full_data_patchtst_dataset.py`.
- Focused tests passed: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_full_data_patchtst_dataset tests.test_eval_patchtst_config_loading tests.test_patchtst_smoke` (`20` tests).
- New full-data dataset test verifies `input_window_position=input_end` returns `[anchor_time + 570s, anchor_time + 600s)` for a 30-second-equivalent trailing crop from the original centered 20-minute input interval.
- Regression submitter dry-run passed for `MAP`; classification submitter dry-run passed for `hypotension` at `5m`.

Launch regression with:

```bash
bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch_30s_v15.sh
```

Launch classification with:

```bash
bash slurm/submit_patchtst_classification_full_data_4ch_30s_v15.sh
```

Dry-run with:

```bash
DRY_RUN=1 bash slurm/submit_patchtst_regression_t0_gap_full_data_4ch_30s_v15.sh
DRY_RUN=1 bash slurm/submit_patchtst_classification_full_data_4ch_30s_v15.sh
```

## Submitted Jobs, 2026-09-02

Submitted the full 30-second `patchtst_v1_5` workload: `26` regression targets and `6` event-classification targets. Initial `squeue` check showed all `32` jobs pending on `gl40s_short` with reason `Priority`; they were later moved to `a100_short`. No output artifacts had been created yet.

| Task | Target | SLURM | Initial state |
|---|---|---:|---|
| Regression | `HR_t_plus_0m_gap` | `26981233` | Pending, `Priority` |
| Regression | `RR_t_plus_0m_gap` | `26981234` | Pending, `Priority` |
| Regression | `SBP_t_plus_0m_gap` | `26981235` | Pending, `Priority` |
| Regression | `DBP_t_plus_0m_gap` | `26981236` | Pending, `Priority` |
| Regression | `PP_t_plus_0m_gap` | `26981237` | Pending, `Priority` |
| Regression | `MAP_t_plus_0m_gap` | `26981238` | Pending, `Priority` |
| Regression | `ABP_area_t_plus_0m_gap` | `26981239` | Pending, `Priority` |
| Regression | `PLETH_ACDC_t_plus_0m_gap` | `26981240` | Pending, `Priority` |
| Regression | `PLETH_amp_t_plus_0m_gap` | `26981241` | Pending, `Priority` |
| Regression | `ECG_Ramp_t_plus_0m_gap` | `26981242` | Pending, `Priority` |
| Regression | `HRV_RMSSD_t_plus_0m_gap` | `26981243` | Pending, `Priority` |
| Regression | `HR_range_t_plus_0m_gap` | `26981244` | Pending, `Priority` |
| Classification | `hypotension_within_5m` | `26981245` | Pending, `Priority` |
| Regression | `ShockIdx_t_plus_0m_gap` | `26981246` | Pending, `Priority` |
| Classification | `hypotension_within_10m` | `26981247` | Pending, `Priority` |
| Regression | `PPV_t_plus_0m_gap` | `26981248` | Pending, `Priority` |
| Classification | `tachycardia_within_5m` | `26981249` | Pending, `Priority` |
| Regression | `PVI_t_plus_0m_gap` | `26981250` | Pending, `Priority` |
| Classification | `tachycardia_within_10m` | `26981251` | Pending, `Priority` |
| Regression | `PTT_t_plus_0m_gap` | `26981252` | Pending, `Priority` |
| Classification | `hypoxia_within_5m` | `26981253` | Pending, `Priority` |
| Regression | `dPdt_max_t_plus_0m_gap` | `26981254` | Pending, `Priority` |
| Classification | `hypoxia_within_10m` | `26981255` | Pending, `Priority` |
| Regression | `ABP_tau_t_plus_0m_gap` | `26981256` | Pending, `Priority` |
| Regression | `RESP_amp_t_plus_0m_gap` | `26981257` | Pending, `Priority` |
| Regression | `PLETH_ACDC_PLETH_amp_t_plus_0m_gap` | `26981258` | Pending, `Priority` |
| Regression | `ABP_area_ABP_tau_t_plus_0m_gap` | `26981259` | Pending, `Priority` |
| Regression | `ABP_area_ShockIdx_t_plus_0m_gap` | `26981260` | Pending, `Priority` |
| Regression | `PLETH_amp_ShockIdx_t_plus_0m_gap` | `26981261` | Pending, `Priority` |
| Regression | `PLETH_ACDC_ShockIdx_t_plus_0m_gap` | `26981262` | Pending, `Priority` |
| Regression | `ShockIdx_ABP_tau_t_plus_0m_gap` | `26981263` | Pending, `Priority` |
| Regression | `PLETH_ACDC_ABP_tau_t_plus_0m_gap` | `26981264` | Pending, `Priority` |

### Partition Move - 2026-09-02

All submitted 30-second `patchtst_v1_5` jobs were moved in place from `gl40s_short` to `a100_short` with `scontrol update`. Handoff verification at `2026-09-02 18:20 EDT` showed all `32` jobs on `a100_short`, still `PENDING` with reason `Priority`.

## Current Status, 2026-09-03

- Implementation, submitters, and focused validation are complete.
- After deadline cancellation, kept 30-second `patchtst_v1_5` regression jobs `26981238`, `26981240`, `26981254`, and `26981256` were moved back to `gl40s_short` on `2026-09-03`; post-move `squeue` showed them pending with reason `Priority`.
- 30-second `patchtst_v1_5` classification jobs `26981245`, `26981247`, `26981249`, `26981251`, `26981253`, and `26981255` were moved back to `gl40s_short` on `2026-09-03`; post-move `squeue` showed them pending with reason `Priority`.
- No 30-second output directories or `ptst_v15_30s_*_269812*.{out,err}` logs existed at the handoff check, so none of these jobs had started writing artifacts yet.
- Existing 20-minute `patchtst_v1` full-data jobs are also still running or pending on GPU priority, so keep outputs separated by run tag.


## Deadline Cancellation, 2026-09-03

To reduce active SLURM load for near-term deadlines, the 30-second `patchtst_v1_5` regression queue was narrowed temporarily to `MAP`, `PLETH_ACDC`, `ABP_tau`, and `dPdt_max`. The canceled regression targets should be resumed soon after the deadline pressure clears; this was a scheduling decision, not an experimental conclusion.

- Kept pending regression jobs: `26981238` MAP, `26981240` PLETH_ACDC, `26981254` dPdt_max, and `26981256` ABP_tau. On `2026-09-03`, these jobs and the kept 30-second classification jobs were moved from `a100_short` back to `gl40s_short`; post-move `squeue` showed all ten `ptst_v15_30s_*` jobs pending with reason `Priority`.
- Canceled pending regression jobs: `26981233` HR, `26981234` RR, `26981235` SBP, `26981236` DBP, `26981237` PP, `26981239` ABP_area, `26981241` PLETH_amp, `26981242` ECG_Ramp, `26981243` HRV_RMSSD, `26981244` HR_range, `26981246` ShockIdx, `26981248` PPV, `26981250` PVI, `26981252` PTT, `26981257` RESP_amp, `26981258` PLETH_ACDC_PLETH_amp, `26981259` ABP_area_ABP_tau, `26981260` ABP_area_ShockIdx, `26981261` PLETH_amp_ShockIdx, `26981262` PLETH_ACDC_ShockIdx, `26981263` ShockIdx_ABP_tau, and `26981264` PLETH_ACDC_ABP_tau.
- Classification jobs were intentionally left untouched.

## Handoff Audit, 2026-09-03

Current state from repo artifacts and `squeue` after the partition move back to `gl40s_short`:

- Kept regression jobs `26981238` MAP, `26981240` PLETH_ACDC, `26981254` dPdt_max, and `26981256` ABP_tau are pending on `gl40s_short` with reason `Priority`.
- Kept classification jobs `26981245`, `26981247`, `26981249`, `26981251`, `26981253`, and `26981255` are pending on `gl40s_short` with reason `Priority`.
- Output roots `outputs/patchtst/full_data_v15_4ch_30s_es` and `outputs/patchtst/full_data_v15_4ch_30s_events_es` do not exist yet, so these jobs had not started writing artifacts at handoff.
- The canceled regression jobs should be resubmitted after deadline pressure clears. Keep this run tag separate from the 20-minute `patchtst_v1` outputs.

## Status Check, 2026-09-03 09:53 EDT

The kept vasopressor-free 30-second `patchtst_v1_5` jobs left `squeue` but did not create output roots. Their logs show immediate argument-parse failure from the pre-fix shared wrapper defaults:

```text
train_patchtst.py: error: --physiojepa-fidelity conflicts with explicit --d-model=128; expected 512.
```

Affected kept jobs:

- regression: `26981238` MAP, `26981240` PLETH_ACDC, `26981254` dPdt_max, and `26981256` ABP_tau
- classification: `26981245` hypotension 5m, `26981247` hypotension 10m, `26981249` tachycardia 5m, `26981251` tachycardia 10m, `26981253` hypoxia 5m, and `26981255` hypoxia 10m

No directories exist under `outputs/patchtst/full_data_v15_4ch_30s_es` or `outputs/patchtst/full_data_v15_4ch_30s_events_es`. These jobs should be resubmitted with the corrected `slurm/train_patchtst.sh` wrapper if the vasopressor-free 30-second results are still needed. This failure is separate from the newer vaso-present jobs `26985986`-`26985995`, which were submitted after the wrapper fix and remain queued.

### Replacement Submission, 2026-09-03 09:57 EDT

The failed kept jobs were resubmitted with the corrected `slurm/train_patchtst.sh` wrapper after dry-run verification. Live `squeue` shows all replacements pending on `gl40s_short` with reason `Priority`.

- Regression replacements, run tag `full_data_v15_4ch_30s_es`: `26988945` MAP, `26988946` PLETH_ACDC, `26988947` dPdt_max, and `26988948` ABP_tau.
- Classification replacements, run tag `full_data_v15_4ch_30s_events_es`: `26988951` hypotension 5m, `26988952` hypotension 10m, `26988953` tachycardia 5m, `26988954` tachycardia 10m, `26988955` hypoxia 5m, and `26988956` hypoxia 10m.
- The initial sandboxed submit attempt failed with `sbatch: error: Error creating slurm stream socket: Operation not permitted`; the successful submissions were run with escalated Slurm access.

## Next Steps

1. Monitor replacement 30-second `patchtst_v1_5` jobs `26988945`-`26988948` and `26988951`-`26988956`; inspect logs once they start and verify the fidelity preset no longer conflicts with wrapper defaults.
2. Resume the canceled 30-second regression targets soon after deadline pressure clears.
3. After checkpoints complete, evaluate the new run tags with `slurm/eval_patchtst.sh` so `test_metrics.json` and `test_predictions.npz` are available before any model comparison.
4. Compare against the 20-minute `patchtst_v1` full-data runs only after both paths have aligned evaluation exports.

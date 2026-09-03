# PatchTST Evaluation

This file summarizes the completed full-cohort 3-channel PatchTST evaluation from 2026-08-23. Treat all results here as `pre-v1`.

These are historical results only. The standard `all_targets.npz` bundle used
by this evaluation contains center-mode `t+0` regression targets that overlap
the 20-minute input, so these regression results must not be used as
leakage-free results. Current leakage-safe regression results are documented
in `docs/v1_vasopressor_free/regression_results_v1_vaso_free_sorted.md` and use gap mode.

## Setup

- evaluation job: `26752665`
- cohort: full cohort
- split: standard test split (`249` patients, about `95k` windows per task)
- model: 3-channel PatchTST (`ABP`, `II`, `PLETH`)
- artifacts: `outputs/patchtst/test_results_summary.json`, `outputs/patchtst/<task>/test_metrics.json`

## Main Results

### Best regression targets by `R²`

| Target | R² |
|---|---:|
| `PLETH_ACDC` | `0.939` |
| `PVI` | `0.921` |
| `HR` | `0.811` |
| `dPdt_max` | `0.777` |
| `ABP_tau` | `0.758` |

### Weakest regression targets by `R²`

| Target | R² |
|---|---:|
| `RESP_amp` | `0.057` |
| `RR` | `0.055` |
| `PLETH_amp×ShockIdx` | `0.039` |

### Event classification

| Event | AUROC | AUPRC | F1 | Prevalence |
|---|---:|---:|---:|---:|
| Tachycardia `10m` | `0.990` | `0.971` | `0.922` | `13.8%` |
| Hypotension `10m` | `0.785` | `0.489` | `0.452` | `23.9%` |

## Takeaways

- Strongest targets are waveform-local morphology targets such as `PLETH_ACDC`, `PVI`, and `dPdt_max`.
- Cross-feature interaction targets and respiration-related targets are much weaker.
- Tachycardia is easy for this baseline.
- Hypotension is useful but clearly harder.

## Notes

- Regression baseline was the train-mean predictor.
- Classification baseline was random ranking.
- Evaluation later gained bootstrap CIs for event metrics, but this `pre-v1` result set predates the current rerun batches.
- On 2026-08-27, `scripts/eval_patchtst.py` was hardened so evaluation now reconstructs dataset paths, splits, target bundle, task selection, and model architecture from each checkpoint's saved training config by default, and checkpoint loading now strips only `_orig_mod.` before requiring `strict=True`. This prevents mixed `--all` batches from accidentally reusing the current CLI/default `waveform_dir`, `splits_path`, `target_path`, or model variant across unrelated experiments.

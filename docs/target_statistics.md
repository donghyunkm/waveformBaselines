# Regression Target Statistics

Computed on 2026-08-26 from saved target bundles. This is a compact summary of the saved CSV/JSON artifacts.

## Artifacts

- standard: `outputs/targets/feature_target_stats.{json,csv}`
- vasopressor-free: `outputs/targets/feature_target_stats_vasopressor_free.{json,csv}`
- script: `scripts/compute_target_stats.py`

## Main Findings

### Full cohort

- Coverage is nearly complete for most base vitals on `t+0m`.
- PPG-derived targets such as `PVI`, `PLETH_ACDC`, and `PLETH_amp` have slightly lower coverage, about `97.9%` on the test split.
- The broadest targets are `dPdt_max`, `HRV_RMSSD`, and `PTT`.
- The heaviest-tailed targets include `PPV`, `ABP_tau`, and multiplicative interaction features.

### Vasopressor-free overlap cohort

- Coverage is slightly lower than the full cohort because of the smaller cohort and gap-mode target definition.
- `PVI`, `PLETH_ACDC`, and `PLETH_amp` are again the least complete targets.
- Hemodynamic targets shift upward relative to the full cohort, especially `MAP`, `SBP`, and `dPdt_max`.
- The same target families remain hardest: `PPV`, `ABP_tau`, and interaction features.

## Practical Interpretation

- Easy targets are usually either compact (`HR`) or tightly tied to waveform morphology (`PLETH_ACDC`, `PVI`, `dPdt_max`).
- Hard targets are usually interaction, ratio, or cross-channel timing targets.
- Raw variance alone is not a good difficulty metric.
- Outlier load and tail behavior are more informative than scale by itself.

## Regeneration

```bash
PYTHONPATH=. /gpfs/home/dk5565/.conda/envs/physiojepa/bin/python \
  scripts/compute_target_stats.py \
  --target-path outputs/targets/all_targets.npz \
  --splits-path outputs/splits/splits.json \
  --output-json outputs/targets/feature_target_stats.json \
  --output-csv outputs/targets/feature_target_stats.csv

PYTHONPATH=. /gpfs/home/dk5565/.conda/envs/physiojepa/bin/python \
  scripts/compute_target_stats.py \
  --target-path outputs/targets/feature_targets_gap_vasopressor_free.npz \
  --splits-path outputs/splits/vasopressor_free_splits.json \
  --output-json outputs/targets/feature_target_stats_vasopressor_free.json \
  --output-csv outputs/targets/feature_target_stats_vasopressor_free.csv
```

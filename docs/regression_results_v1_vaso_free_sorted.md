# Vasopressor-Free `patchtst_v1` Regression Results

Compact summary of the fresh early-stopping-enabled vasopressor-free `patchtst_v1` regression evaluation completed on 2026-08-27.

## Scope

- run tag: `vasopressor_free_v1_es`
- model: `patchtst_v1`
- task: single-target regression
- cohort: vasopressor-free overlap cohort
- split file: `outputs/splits/vasopressor_free_splits.json`
- target bundle: `outputs/targets/feature_targets_gap_vasopressor_free.npz`
- evaluation summary: `outputs/patchtst/vasopressor_free_v1_es/test_results_summary.json`

## Evaluation Summary

- evaluated targets: `26`
- all evaluated targets had finite predictions: `n_nan_preds = 0` for every task
- targets with `R² >= 0.7`: `8`
- targets with `R² >= 0.5`: `13`
- targets with negative `R²`: `5`
- mean `R²` across targets: `0.412`
- median `R²` across targets: `0.485`
- early-stopping training epochs ranged from `2` to `50`, with median `13`

## Full Results Sorted By `R²`

| Target | R² | Corr | RMSE | MAE | N Valid | Train Epochs |
|---|---:|---:|---:|---:|---:|---:|
| `PLETH_amp` | `0.893` | `0.945` | `0.188` | `0.123` | `34846` | `13` |
| `PP` | `0.815` | `0.908` | `8.546` | `5.633` | `35638` | `32` |
| `DBP` | `0.811` | `0.903` | `5.877` | `3.575` | `35701` | `32` |
| `SBP` | `0.809` | `0.904` | `9.427` | `6.646` | `35701` | `21` |
| `ABP_area` | `0.773` | `0.881` | `3.017` | `1.994` | `35701` | `37` |
| `dPdt_max` | `0.773` | `0.884` | `232.043` | `160.863` | `35668` | `50` |
| `MAP` | `0.746` | `0.866` | `7.342` | `4.906` | `35701` | `22` |
| `PVI` | `0.713` | `0.846` | `8.344` | `6.164` | `34839` | `24` |
| `HR_range` | `0.634` | `0.800` | `12.031` | `9.129` | `35701` | `11` |
| `HR` | `0.624` | `0.792` | `9.141` | `6.331` | `35701` | `6` |
| `PLETH_ACDC` | `0.623` | `0.808` | `0.140` | `0.109` | `34846` | `13` |
| `ShockIdx` | `0.602` | `0.778` | `0.118` | `0.081` | `35701` | `10` |
| `HRV_RMSSD` | `0.565` | `0.759` | `45.509` | `31.921` | `35701` | `12` |
| `PPV` | `0.485` | `0.713` | `10.937` | `6.721` | `35701` | `22` |
| `ABP_tau` | `0.382` | `0.621` | `0.457` | `0.248` | `35634` | `14` |
| `ECG_Ramp` | `0.356` | `0.605` | `0.293` | `0.215` | `35701` | `23` |
| `ABP_area_ShockIdx` | `0.156` | `0.424` | `0.478` | `0.386` | `35701` | `6` |
| `ABP_area_ABP_tau` | `0.113` | `0.337` | `0.565` | `0.485` | `35701` | `23` |
| `ShockIdx_ABP_tau` | `0.012` | `0.120` | `0.542` | `0.463` | `35701` | `9` |
| `PLETH_ACDC_PLETH_amp` | `0.007` | `0.115` | `0.214` | `0.112` | `35701` | `3` |
| `RR` | `0.005` | `0.095` | `2.359` | `1.845` | `35701` | `13` |
| `PLETH_amp_ShockIdx` | `-0.010` | `0.091` | `0.469` | `0.389` | `35701` | `5` |
| `PLETH_ACDC_ABP_tau` | `-0.015` | `0.100` | `0.474` | `0.392` | `35701` | `19` |
| `PLETH_ACDC_ShockIdx` | `-0.026` | `0.061` | `0.486` | `0.405` | `35701` | `5` |
| `RESP_amp` | `-0.059` | `-0.036` | `0.375` | `0.270` | `35701` | `10` |
| `PTT` | `-0.074` | `0.017` | `35.099` | `28.932` | `35669` | `2` |

## Main Takeaways

- The fresh `v1` rerun stayed numerically stable through training and evaluation; no regression task produced `NaN` predictions.
- Core blood-pressure and plethysmography-derived targets remained the strongest signals on the vasopressor-free cohort.
- Several interaction features and `PTT` remained weak or anti-predictive, so these are the clearest candidates for redesign or de-prioritization.
- Early stopping reduced many runs well below the `50`-epoch cap while still yielding strong top-end performance.

# Vasopressor-Free `patchtst_v1` Regression Results With Target Normalization

Compact summary of the target-normalized vasopressor-free `patchtst_v1` regression evaluation completed on 2026-08-28.

## Scope

- run tag: `vasopressor_free_v1_target_norm_es`
- model: `patchtst_v1`
- waveform inputs: `3` channels, `ABP,II,PLETH`
- task: single-target regression
- cohort: vasopressor-free overlap cohort
- split file: `outputs/splits/vasopressor_free_splits.json`
- target bundle: `outputs/targets/feature_targets_gap_vasopressor_free.npz`
- target normalization: train-split z-score per target
- evaluation summary: `outputs/patchtst/vasopressor_free_v1_target_norm_es/test_results_summary.json`

## Evaluation Summary

- evaluated targets: `26`
- all evaluated targets had finite predictions: `n_nan_preds = 0` for every task
- targets with `R² >= 0.7`: `8`
- targets with `R² >= 0.5`: `13`
- targets with negative `R²`: `3`
- mean `R²` across targets: `0.419`
- median `R²` across targets: `0.483`
- early-stopping training epochs ranged from `2` to `42`, with median `13.5`

## Significance Testing

- For each target, significance was tested between the normalized and non-normalized runs on the same test windows using a paired two-sided t-test on per-window squared errors.
- Because the target values are identical across the paired runs, this is equivalent to testing whether the observed `ΔR²` differs from `0`.
- `q` values use Benjamini-Hochberg FDR correction across the `26` targets; `Sig` is called at `q < 0.05`.

## Full Results Sorted By `R²`

| Target | R² | ΔR² vs No Norm | p | q | Sig | Corr | RMSE | MAE | N Valid | Train Epochs | Best Val Loss |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `PLETH_amp` | `0.934` | `+0.040` | `2.11e-184` | `1.10e-183` | `improved` | `0.966` | `0.148` | `0.093` | `34846` | `32` | `0.054699` |
| `dPdt_max` | `0.854` | `+0.082` | `<1e-300` | `<1e-300` | `improved` | `0.925` | `185.691` | `123.445` | `35668` | `38` | `0.127780` |
| `PP` | `0.831` | `+0.016` | `3.21e-23` | `4.63e-23` | `improved` | `0.914` | `8.174` | `5.283` | `35638` | `38` | `0.126229` |
| `DBP` | `0.811` | `-0.000` | `9.97e-01` | `9.97e-01` | `ns` | `0.901` | `5.877` | `3.514` | `35701` | `32` | `0.140108` |
| `ABP_area` | `0.799` | `+0.025` | `2.75e-27` | `4.21e-27` | `improved` | `0.897` | `2.844` | `1.824` | `35701` | `42` | `0.190753` |
| `SBP` | `0.796` | `-0.013` | `3.57e-07` | `4.64e-07` | `worsened` | `0.894` | `9.741` | `6.654` | `35701` | `17` | `0.125521` |
| `MAP` | `0.744` | `-0.002` | `1.57e-01` | `1.77e-01` | `ns` | `0.863` | `7.372` | `5.005` | `35701` | `16` | `0.164087` |
| `PLETH_ACDC` | `0.703` | `+0.080` | `1.40e-144` | `5.19e-144` | `improved` | `0.840` | `0.125` | `0.085` | `34846` | `23` | `0.259912` |
| `HR` | `0.688` | `+0.064` | `4.73e-202` | `6.15e-201` | `improved` | `0.831` | `8.332` | `5.548` | `35701` | `8` | `0.293828` |
| `ShockIdx` | `0.667` | `+0.064` | `1.43e-71` | `3.37e-71` | `improved` | `0.823` | `0.108` | `0.073` | `35701` | `18` | `0.224181` |
| `PVI` | `0.648` | `-0.065` | `5.96e-190` | `5.16e-189` | `worsened` | `0.811` | `9.241` | `6.819` | `34839` | `11` | `0.355802` |
| `HR_range` | `0.602` | `-0.032` | `2.46e-55` | `4.93e-55` | `worsened` | `0.776` | `12.548` | `9.184` | `35701` | `12` | `0.361932` |
| `HRV_RMSSD` | `0.565` | `-0.000` | `8.48e-01` | `8.82e-01` | `ns` | `0.765` | `45.523` | `32.125` | `35701` | `12` | `0.439800` |
| `PPV` | `0.400` | `-0.085` | `1.60e-85` | `4.62e-85` | `worsened` | `0.636` | `11.805` | `7.846` | `35701` | `4` | `0.753939` |
| `ABP_tau` | `0.353` | `-0.029` | `6.30e-12` | `8.62e-12` | `worsened` | `0.599` | `0.467` | `0.283` | `35634` | `4` | `0.559552` |
| `ECG_Ramp` | `0.300` | `-0.056` | `1.33e-61` | `2.89e-61` | `worsened` | `0.548` | `0.306` | `0.232` | `35701` | `28` | `0.796574` |
| `ABP_area_ShockIdx` | `0.196` | `+0.040` | `8.71e-82` | `2.26e-81` | `improved` | `0.444` | `0.467` | `0.369` | `35701` | `15` | `0.797808` |
| `ABP_area_ABP_tau` | `0.110` | `-0.003` | `1.63e-01` | `1.77e-01` | `ns` | `0.333` | `0.565` | `0.485` | `35701` | `10` | `0.888093` |
| `PLETH_ACDC_PLETH_amp` | `0.040` | `+0.033` | `3.74e-146` | `1.62e-145` | `improved` | `0.214` | `0.210` | `0.110` | `35701` | `8` | `0.970703` |
| `RR` | `0.017` | `+0.012` | `2.29e-06` | `2.84e-06` | `improved` | `0.167` | `2.345` | `1.818` | `35701` | `22` | `0.930423` |
| `ShockIdx_ABP_tau` | `0.009` | `-0.004` | `1.55e-02` | `1.84e-02` | `worsened` | `0.133` | `0.543` | `0.461` | `35701` | `20` | `0.964769` |
| `PLETH_amp_ShockIdx` | `0.007` | `+0.017` | `1.16e-52` | `2.15e-52` | `improved` | `0.104` | `0.465` | `0.387` | `35701` | `6` | `0.986329` |
| `PLETH_ACDC_ShockIdx` | `0.001` | `+0.027` | `4.80e-99` | `1.56e-98` | `improved` | `0.093` | `0.480` | `0.400` | `35701` | `6` | `0.953632` |
| `RESP_amp` | `-0.023` | `+0.036` | `3.00e-185` | `1.95e-184` | `improved` | `-0.044` | `0.369` | `0.268` | `35701` | `8` | `0.692249` |
| `PLETH_ACDC_ABP_tau` | `-0.034` | `-0.019` | `2.41e-41` | `4.17e-41` | `worsened` | `0.054` | `0.479` | `0.395` | `35701` | `11` | `0.945506` |
| `PTT` | `-0.127` | `-0.053` | `8.02e-36` | `1.30e-35` | `worsened` | `0.041` | `35.949` | `28.943` | `35669` | `2` | `1.218771` |

## Comparison Against Non-Normalized `vasopressor_free_v1_es`

- `13` targets improved in `R²` and `13` worsened numerically; after FDR correction, `13` were significantly improved, `9` significantly worsened, and `4` were not significantly different
- largest significant `R²` gains: `dPdt_max` `+0.082`, `PLETH_ACDC` `+0.080`, `ShockIdx` `+0.064`, `HR` `+0.064`, `PLETH_amp` `+0.040`
- largest significant `R²` drops: `PPV` `-0.085`, `PVI` `-0.065`, `ECG_Ramp` `-0.056`, `PTT` `-0.053`, `HR_range` `-0.032`
- non-significant targets at `q < 0.05`: `DBP`, `MAP`, `HRV_RMSSD`, `ABP_area_ABP_tau`

## Main Takeaways

- Target normalization preserved the strong blood-pressure and plethysmography targets and improved the very top end, especially `PLETH_amp` and `dPdt_max`.
- The batch reduced the count of negative-`R²` targets from `5` to `3`, but several interaction features remained near-zero and `PTT` remained clearly weak.
- The gains were selective rather than uniform, so future reruns should treat z-scoring as a target-specific baseline rather than an across-the-board improvement; the significance results reinforce that some of the small deltas are effectively ties.

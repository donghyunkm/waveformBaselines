# Bland-Altman Analysis

Compact summary of the vasopressor-free regression agreement plots under `outputs/patchtst/vasopressor_free/`.

## Artifacts

- summary: `outputs/patchtst/vasopressor_free/bland_altman_summary.json`
- per-task stats: `outputs/patchtst/vasopressor_free/<task>/bland_altman_stats.json`
- per-task figures: `outputs/patchtst/vasopressor_free/<task>/bland_altman.png`
- flat export: `blandaltman/`

## Main Patterns

- Mean bias is usually small relative to the limits of agreement.
- The main failure mode is range compression, not global offset.
- Pressure-family targets, `HR`, `RR`, `PTT`, and interaction targets show the clearest mean-dependent bias.
- `PPV`, `ABP_area`, `ABP_tau`, `PVI`, `ECG_Ramp`, and `RESP_amp` show the clearest heteroscedasticity.

## Best Agreement

- `PLETH_ACDC`
- `dPdt_max`
- `PVI`
- `HR`
- `HR_range`

These targets have the tightest residual bands relative to scale, though several still show range compression at the extremes.

## Weakest Agreement

- interaction targets
- `PPV`
- `ABP_tau`
- `RESP_amp`
- `RR`

These failures look systematic rather than outlier-driven.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import extreme_value_fraction, micro_window_slices


@dataclass(frozen=True)
class SignalQualitySummary:
    valid_micro_fraction: float
    missing_micro_fraction: float
    flatline_fraction: float
    extreme_value_fraction: float
    valid_micro_mask: tuple[bool, ...]


def assess_signal_quality(
    values: np.ndarray,
    micro_window_samples: int,
    min_std: float,
    min_finite_fraction: float = 0.95,
    extreme_value_atol_fraction: float = 0.01,
) -> SignalQualitySummary:
    arr = np.asarray(values, dtype=np.float64)
    micro_valid: list[bool] = []
    missing_flags: list[bool] = []
    flatline_flags: list[bool] = []
    for slc in micro_window_slices(arr.size, micro_window_samples):
        window = arr[slc]
        finite_mask = np.isfinite(window)
        finite_fraction = float(np.mean(finite_mask)) if window.size else 0.0
        finite = window[finite_mask]
        missing = finite_fraction < min_finite_fraction
        if finite.size == 0:
            micro_valid.append(False)
            missing_flags.append(True)
            flatline_flags.append(False)
            continue
        std = float(np.std(finite))
        flat = (not missing) and (std < min_std)
        valid = (not missing) and (not flat)
        micro_valid.append(valid)
        missing_flags.append(missing)
        flatline_flags.append(flat)
    valid_micro_fraction = float(np.mean(micro_valid)) if micro_valid else float("nan")
    missing_micro_fraction = float(np.mean(missing_flags)) if missing_flags else float("nan")
    flatline_fraction = float(np.mean(flatline_flags)) if flatline_flags else float("nan")
    extreme_fraction = extreme_value_fraction(arr, atol_fraction=extreme_value_atol_fraction)
    return SignalQualitySummary(
        valid_micro_fraction=valid_micro_fraction,
        missing_micro_fraction=missing_micro_fraction,
        flatline_fraction=flatline_fraction,
        extreme_value_fraction=extreme_fraction,
        valid_micro_mask=tuple(micro_valid),
    )

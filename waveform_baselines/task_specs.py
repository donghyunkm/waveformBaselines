from __future__ import annotations

from dataclasses import dataclass

WAVEFORM_FEATURE_NAMES = [
    "HR",
    "RR",
    "SBP",
    "DBP",
    "PP",
    "MAP",
    "ABP_area",
    "PLETH_ACDC",
    "PLETH_amp",
    "ECG_Ramp",
    "HRV_RMSSD",
    "HR_range",
    "ShockIdx",
    "PPV",
    "PVI",
    "PTT",
    "dPdt_max",
    "ABP_tau",
    "RESP_amp",
]

FOCUSED_CORRELATION_NAMES = [
    "PLETH_ACDC_PLETH_amp",
    "ABP_area_ABP_tau",
    "ABP_area_ShockIdx",
    "PLETH_amp_ShockIdx",
    "PLETH_ACDC_ShockIdx",
    "ShockIdx_ABP_tau",
    "PLETH_ACDC_ABP_tau",
]


@dataclass(frozen=True)
class FeatureRegressionTaskSpec:
    """Schema for future physiological feature prediction."""

    horizons_min: tuple[int, ...] = (0, 20, 60)
    horizon_mode: str = "center"
    input_window_minutes: int = 20
    feature_names: tuple[str, ...] = tuple(WAVEFORM_FEATURE_NAMES)
    correlation_names: tuple[str, ...] = tuple(FOCUSED_CORRELATION_NAMES)
    aggregation: str = "mean"

    @property
    def base_target_names(self) -> tuple[str, ...]:
        return self.feature_names + self.correlation_names

    @property
    def target_names(self) -> tuple[str, ...]:
        names = []
        for horizon in self.horizons_min:
            suffix = f"t_plus_{horizon}m"
            if self.horizon_mode == "gap":
                suffix = f"{suffix}_gap"
            for base_name in self.base_target_names:
                names.append(f"{base_name}_{suffix}")
        return tuple(names)

    @property
    def target_dim(self) -> int:
        return len(self.target_names)


@dataclass(frozen=True)
class EventTaskSpec:
    """Schema for future binary event prediction."""

    horizons_min: tuple[int, ...] = (5, 10, 60, 90)
    target_generation_mode: str = "anchor_horizon"
    hypotension_threshold: float = 65.0
    tachycardia_threshold: float = 110.0
    hypoxia_threshold: float = 90.0
    sustain_minutes: int = 5
    hypotension_channel: int = 0
    tachycardia_channel: int = 4
    hypoxia_channel: int = 2
    event_names: tuple[str, ...] = ("hypotension", "tachycardia")

    @property
    def target_names(self) -> tuple[str, ...]:
        names = []
        for horizon in self.horizons_min:
            for event_name in self.event_names:
                names.append(f"{event_name}_within_{horizon}m")
        return tuple(names)

    @property
    def target_dim(self) -> int:
        return len(self.target_names)


DEFAULT_FEATURE_TASK = FeatureRegressionTaskSpec()
DEFAULT_EVENT_TASK = EventTaskSpec()

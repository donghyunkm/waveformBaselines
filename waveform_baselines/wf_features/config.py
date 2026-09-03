from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

FS = 125
INPUT_WINDOW_SECONDS = 20 * 60
FEATURE_WINDOW_SECONDS = 60
MICRO_WINDOW_SECONDS = 10
CHANNEL_ORDER = ("II", "ABP", "PLETH", "RESP")
FEATURE_VERSION = "v7"
CACHE_ROOT = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction"
)


@dataclass(frozen=True)
class ExtractionConfig:
    sampling_rate_hz: int = FS
    input_window_seconds: int = INPUT_WINDOW_SECONDS
    feature_window_seconds: int = FEATURE_WINDOW_SECONDS
    micro_window_seconds: int = MICRO_WINDOW_SECONDS
    channel_order: tuple[str, ...] = CHANNEL_ORDER
    feature_version: str = FEATURE_VERSION
    ecg_hrv_min_beats: int = 8
    ecg_hrv_min_successive_pairs: int = 3
    morphology_template_points: int = 64
    quality_min_finite_fraction: float = 1.0
    extreme_value_atol_fraction: float = 0.01
    max_interpolated_gap_seconds: float = 0.2
    ecg_detector: str = "xqrs"
    ecg_allow_energy_fallback: bool = True
    ecg_detector_low_hz: float = 5.0
    ecg_detector_high_hz: float = 20.0
    ecg_morphology_low_hz: float = 0.5
    ecg_morphology_high_hz: float = 40.0
    ecg_peak_search_radius_s: float = 0.08
    abp_detector_low_hz: float = 0.5
    abp_detector_high_hz: float = 12.0
    abp_morphology_high_hz: float = 20.0
    abp_peak_search_radius_s: float = 0.08
    abp_trough_search_radius_s: float = 0.12
    abp_min_pulse_bpm: float = 30.0
    abp_max_pulse_bpm: float = 220.0
    pleth_detector_low_hz: float = 0.5
    pleth_detector_high_hz: float = 8.0
    pleth_morphology_high_hz: float = 8.0
    pleth_peak_search_radius_s: float = 0.08
    pleth_trough_search_radius_s: float = 0.12
    resp_detector_low_hz: float = 0.05
    resp_detector_high_hz: float = 1.5
    resp_min_cycle_s: float = 0.75
    resp_max_cycle_s: float = 15.0

    def __post_init__(self) -> None:
        if self.ecg_detector not in {"xqrs", "energy"}:
            raise ValueError(f"Unsupported ECG detector {self.ecg_detector!r}; expected 'xqrs' or 'energy'")

    @property
    def input_samples(self) -> int:
        return self.sampling_rate_hz * self.input_window_seconds

    @property
    def feature_window_samples(self) -> int:
        return self.sampling_rate_hz * self.feature_window_seconds

    @property
    def micro_window_samples(self) -> int:
        return self.sampling_rate_hz * self.micro_window_seconds

    @property
    def n_feature_windows(self) -> int:
        return self.input_window_seconds // self.feature_window_seconds

    @property
    def micro_windows_per_feature_window(self) -> int:
        return self.feature_window_seconds // self.micro_window_seconds

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_EXTRACTION_CONFIG = ExtractionConfig()

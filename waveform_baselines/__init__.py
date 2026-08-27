"""Core utilities for waveform baseline experiments."""

from .data_index import (
    RAW_SIGNAL_ORDER,
    WAVEFORM_MODEL_CHANNELS,
    AlignedWaveformDataset,
    build_aligned_20m_anchor_table,
    load_raw_waveform_manifest,
)
from .task_specs import (
    DEFAULT_EVENT_TASK,
    DEFAULT_FEATURE_TASK,
    EventTaskSpec,
    FeatureRegressionTaskSpec,
)

__all__ = [
    "DEFAULT_EVENT_TASK",
    "DEFAULT_FEATURE_TASK",
    "EventTaskSpec",
    "FeatureRegressionTaskSpec",
    "RAW_SIGNAL_ORDER",
    "WAVEFORM_MODEL_CHANNELS",
    "AlignedWaveformDataset",
    "build_aligned_20m_anchor_table",
    "load_raw_waveform_manifest",
]

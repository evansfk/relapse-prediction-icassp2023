"""Data loading, augmentations, preprocessing, and synthetic wearable data generation."""

from .augmentations import (
    JitterTransform,
    PermutationTransform,
    ScalingTransform,
    TFCFrequencyAugmentation,
    TFCTimeAugmentation,
    TSTCCAugmentation,
    add_frequency,
    remove_frequency,
)
from .base import BaseTransform
from .datasets import TFCDataset, TSTCCDataset, WearableTensorDataset, compute_fft_magnitude
from .preprocessing import (
    SurvivalLabelResult,
    append_time_of_day_features,
    build_survival_labels,
    clean_wearable_signals,
    encode_cyclical_time,
    extract_consecutive_windows,
    standardize_per_sample,
)
from .synthetic import SyntheticWearableDataset, generate_synthetic_wearable_dataset

__all__ = [
    "BaseTransform",
    "JitterTransform",
    "PermutationTransform",
    "ScalingTransform",
    "SurvivalLabelResult",
    "SyntheticWearableDataset",
    "TFCDataset",
    "TFCFrequencyAugmentation",
    "TFCTimeAugmentation",
    "TSTCCAugmentation",
    "TSTCCDataset",
    "WearableTensorDataset",
    "add_frequency",
    "append_time_of_day_features",
    "build_survival_labels",
    "clean_wearable_signals",
    "compute_fft_magnitude",
    "encode_cyclical_time",
    "extract_consecutive_windows",
    "generate_synthetic_wearable_dataset",
    "remove_frequency",
    "standardize_per_sample",
]

"""Typed configuration dataclasses matching the IEEE ICASSP 2023 experimental setups."""

from __future__ import annotations

import math

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

type WindowHours = Literal[4, 8, 12]

WINDOW_HOURS_TO_LENGTH: dict[WindowHours, int] = {4: 240, 8: 480, 12: 720}

CHANNEL_NAMES: tuple[str, ...] = (
    "linacc_x",
    "linacc_y",
    "linacc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "heart_rate",
    "rr_interval",
    "hour_sin",
    "hour_cos",
)

CHANNEL_GROUPS: dict[str, tuple[int, ...]] = {
    "accelerometer": (0, 1, 2),
    "gyroscope": (3, 4, 5),
    "heart_rate": (6, 7),
    "hour_of_day": (8, 9),
}

_TUPLE_SIZES: dict[str, int] = {
    "betas": 2,
    "risk_quantiles": 2,
    "hidden_channels": 3,
    "kernel_sizes": 3,
    "dilations": 3,
}
_MIN_TC_CHANNELS = 4


def select_channel_indices(exclude_groups: Sequence[str] = (), /) -> list[int]:
    """Return retained channel indices after excluding modality groups (Table 3 ablations).

    Args:
        exclude_groups: Modality group names from ``CHANNEL_GROUPS`` to omit.

    Returns:
        Sorted list of retained channel indices in ``0..9``.

    Raises:
        ValueError: If an unknown modality group name is provided.
    """
    unknown = set(exclude_groups) - set(CHANNEL_GROUPS)
    if unknown:
        msg = f"Unknown channel groups {sorted(unknown)}; expected subset of {sorted(CHANNEL_GROUPS)}"
        raise ValueError(msg)

    excluded: set[int] = set()
    for group in exclude_groups:
        excluded.update(CHANNEL_GROUPS[group])

    return [idx for idx in range(len(CHANNEL_NAMES)) if idx not in excluded]


def _coerce_yaml_value(field_name: str, field_type: str, val: object, /) -> object:
    """Coerce YAML 1.1 scientific-notation strings and validate tuple-field lengths."""
    if field_name in _TUPLE_SIZES and isinstance(val, (list, tuple)):
        expected_len = _TUPLE_SIZES[field_name]
        if len(val) != expected_len:
            msg = f"Expected {expected_len} elements for {field_name!r}, got {len(val)}"
            raise ValueError(msg)
        elem_is_int = "int" in field_type
        return tuple(int(v) if elem_is_int else float(v) for v in val)
    if isinstance(val, str) and field_type.startswith("float"):
        return float(val)
    if isinstance(val, int) and not isinstance(val, bool) and field_type.startswith("float"):
        return float(val)
    return val


@dataclass(slots=True, kw_only=True)
class BaseConfig:
    """Base configuration supporting YAML deserialization with schema validation."""

    @classmethod
    def from_yaml(cls, path: str | Path, /) -> Self:
        """Load a configuration dataclass from a YAML mapping file.

        Args:
            path: Path to a YAML configuration file.

        Returns:
            Populated configuration instance.

        Raises:
            ValueError: If the YAML root is not a mapping or contains unknown/invalid keys.
        """
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded is None:
            raw: dict[str, Any] = {}
        elif isinstance(loaded, dict):
            raw = loaded
        else:
            msg = f"Expected YAML mapping in {path}, got {type(loaded).__name__}"
            raise ValueError(msg)

        valid_keys = {f.name for f in fields(cls)}
        unknown_keys = set(raw) - valid_keys
        if unknown_keys:
            msg = f"Unknown keys for {cls.__name__}: {sorted(unknown_keys)}"
            raise ValueError(msg)

        coerced: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in raw:
                coerced[f.name] = _coerce_yaml_value(f.name, str(f.type), raw[f.name])
        return cls(**coerced)


@dataclass(slots=True, kw_only=True)
class TFCConfig(BaseConfig):
    """Hyperparameters for Time-Frequency Consistency (TFC) SSL pretraining.

    Defaults match the ICASSP 2023 experimental configuration across 4h, 8h, and 12h windows.
    """

    input_channels: int = 10
    seq_len: int = 480
    kernel_size: int = 12
    stride: int = 2
    final_out_channels: int = 128
    dropout: float = 0.4
    projection_hidden_dim: int = 256
    projection_dim: int = 128
    temperature: float = 0.2
    use_cosine_similarity: bool = True
    consistency_weight: float = 0.5
    consistency_mode: Literal["tf", "clamped_margin"] = "tf"
    jitter_ratio: float = 0.3
    jitter_scale_ratio: float = 1.1
    scale_loc: float = 1.0
    max_seg: int = 16
    freq_mask_ratio: float = 0.1
    freq_add_ratio: float = 0.1
    lr: float = 3e-3
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 3e-4
    use_lr_scheduler: bool = True
    batch_size: int = 128
    epochs: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate TFC hyperparameters."""
        if self.consistency_mode not in {"tf", "clamped_margin"}:
            msg = f"consistency_mode must be 'tf' or 'clamped_margin', got {self.consistency_mode!r}"
            raise ValueError(msg)
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not math.isfinite(self.lr) or self.lr <= 0:
            msg = f"temperature and lr must be finite > 0, got temperature={self.temperature}, lr={self.lr}"
            raise ValueError(msg)
        if self.input_channels < 1 or self.seq_len < 1 or self.epochs < 1 or self.batch_size < 1:
            msg = "input_channels, seq_len, epochs, and batch_size must be >= 1"
            raise ValueError(msg)


@dataclass(slots=True, kw_only=True)
class TSTCCConfig(BaseConfig):
    """Hyperparameters for Temporal and Contextual Contrasting (TS-TCC) SSL pretraining.

    Defaults match the ICASSP 2023 experimental configuration. Note that
    ``final_out_channels * L_enc`` equals ``768`` (4h), ``1408`` (8h), and ``2048`` (12h).
    """

    input_channels: int = 10
    seq_len: int = 480
    kernel_size: int = 16
    stride: int = 3
    final_out_channels: int = 64
    dropout: float = 0.35
    tc_hidden_dim: int = 256
    tc_timesteps: int = 9
    transformer_depth: int = 4
    transformer_heads: int = 4
    transformer_mlp_dim: int = 64
    transformer_dropout: float = 0.1
    temperature: float = 0.2
    use_cosine_similarity: bool = True
    lambda1: float = 1.0
    lambda2: float = 0.7
    jitter_ratio: float = 0.1
    jitter_scale_ratio: float = 2.0
    scale_loc: float = 1.0
    max_seg: int = 12
    lr: float = 3e-3
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 3e-4
    batch_size: int = 128
    epochs: int = 200
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate TS-TCC hyperparameters."""
        if self.tc_timesteps < 1 or self.final_out_channels < _MIN_TC_CHANNELS:
            msg = f"tc_timesteps must be >= 1 and final_out_channels >= {_MIN_TC_CHANNELS}"
            raise ValueError(msg)
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not math.isfinite(self.lr) or self.lr <= 0:
            msg = f"temperature and lr must be finite > 0, got temperature={self.temperature}, lr={self.lr}"
            raise ValueError(msg)
        if self.input_channels < 1 or self.seq_len < 1 or self.epochs < 1 or self.batch_size < 1:
            msg = "input_channels, seq_len, epochs, and batch_size must be >= 1"
            raise ValueError(msg)


@dataclass(slots=True, kw_only=True)
class MixingUpConfig(BaseConfig):
    """Hyperparameters for Mixing-Up Contrastive Learning SSL pretraining.

    Defaults match the ICASSP 2023 experimental configuration (Section 3).
    """

    input_channels: int = 10
    seq_len: int = 480
    hidden_channels: tuple[int, int, int] = (128, 256, 128)
    kernel_sizes: tuple[int, int, int] = (7, 5, 3)
    dilations: tuple[int, int, int] = (2, 4, 8)
    projection_dim: int = 128
    alpha: float = 1.0
    temperature: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate Mixing-Up hyperparameters."""
        if (
            not math.isfinite(self.alpha)
            or self.alpha <= 0
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            msg = f"alpha and temperature must be finite > 0, got alpha={self.alpha}, temperature={self.temperature}"
            raise ValueError(msg)
        if (
            not math.isfinite(self.lr)
            or self.lr <= 0
            or self.input_channels < 1
            or self.epochs < 1
            or self.batch_size < 1
        ):
            msg = "lr must be > 0 and input_channels, epochs, batch_size >= 1"
            raise ValueError(msg)


@dataclass(slots=True, kw_only=True)
class SurvivalConfig(BaseConfig):
    """Hyperparameters for downstream survival analysis and evaluation (Sections 3-4).

    Defaults match the ICASSP 2023 ``pysurvival==0.1.2`` evaluation setup:
    full-batch (``batch_size=None``) ``lr=1e-4, dropout=0.2, epochs=1000``, uniform MTLR
    time grid with ``l2_smooth=1e-2``, ``max_depth=5 / min_samples_leaf=10 / max_samples=0.63``
    survival forests, ``drop_first_onehot=True``, and ``ibs_mode="paper"`` (``[0, t_max] / t_max``).
    """

    test_size: float = 0.4
    include_user_id: bool = True
    include_static: bool = False
    include_past_episodes: bool = False
    include_event_free_days: bool = False
    drop_first_onehot: bool = True
    ibs_mode: Literal["paper", "quantile"] = "paper"
    deepsurv_hidden_units: int = 150
    deepsurv_activation: Literal["tanh", "relu"] = "tanh"
    deepsurv_dropout: float = 0.2
    deepsurv_lr: float = 1e-4
    deepsurv_weight_decay: float = 1e-4
    deepsurv_epochs: int = 1000
    deepsurv_batch_size: int | None = None
    mtlr_hidden_units: int = 150
    mtlr_bins: int = 150
    mtlr_binning: Literal["uniform", "quantile"] = "uniform"
    mtlr_l2_smooth: float = 1e-2
    mtlr_activation: Literal["tanh", "relu"] = "tanh"
    mtlr_dropout: float = 0.2
    mtlr_lr: float = 1e-4
    mtlr_weight_decay: float = 1e-4
    mtlr_epochs: int = 1000
    mtlr_batch_size: int | None = None
    n_trees: int = 200
    max_depth: int | None = 5
    min_samples_leaf: int = 10
    max_samples: float | None = 0.63
    risk_quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate survival configuration literals and split bounds."""
        if not (0.0 < self.test_size < 1.0):
            msg = f"test_size must lie strictly in (0, 1), got {self.test_size}"
            raise ValueError(msg)
        if self.ibs_mode not in {"paper", "quantile"}:
            msg = f"ibs_mode must be 'paper' or 'quantile', got {self.ibs_mode!r}"
            raise ValueError(msg)
        if self.deepsurv_activation not in {"tanh", "relu"} or self.mtlr_activation not in {"tanh", "relu"}:
            msg = "deepsurv_activation and mtlr_activation must be 'tanh' or 'relu'"
            raise ValueError(msg)
        if self.mtlr_binning not in {"uniform", "quantile"}:
            msg = f"mtlr_binning must be 'uniform' or 'quantile', got {self.mtlr_binning!r}"
            raise ValueError(msg)

"""PyTorch Dataset classes for SSL pretraining and downstream evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch

from torch import Tensor
from torch.utils.data import Dataset

from .augmentations import TFCFrequencyAugmentation, TFCTimeAugmentation, TSTCCAugmentation

if TYPE_CHECKING:
    import numpy as np

    from numpy.typing import NDArray

_EXPECTED_TENSOR_NDIM = 3


def _to_float_tensor(samples: NDArray[np.floating] | Tensor, /) -> Tensor:
    tensor = (
        samples.detach().clone().float()
        if isinstance(samples, Tensor)
        else torch.as_tensor(samples, dtype=torch.float32)
    )
    if tensor.ndim != _EXPECTED_TENSOR_NDIM:
        msg = f"Expected 3D wearable tensor (S, C, L), got shape {tuple(tensor.shape)}"
        raise ValueError(msg)
    return tensor


def compute_fft_magnitude(x: Tensor, /) -> Tensor:
    """Compute normalized FFT magnitude spectrum matching ``dataloader_TFC.py``.

    Args:
        x: Time-domain tensor of shape ``(..., L)``.

    Returns:
        Real-valued normalized magnitude spectrum of shape ``(..., L)``.
    """
    window_length = x.shape[-1]
    mag: Tensor = torch.fft.fft(x, dim=-1).abs() / float(window_length)
    return mag


class WearableTensorDataset(Dataset[tuple[Tensor, Tensor]]):
    """Standard labeled or unlabeled wearable time-series dataset returning ``(x, label)``.

    Args:
        samples: Wearable time series of shape ``(S, C, L)``.
        labels: Optional integer labels of shape ``(S,)`` (defaults to zeros).
    """

    def __init__(
        self, samples: NDArray[np.floating] | Tensor, /, *, labels: NDArray[np.integer] | Tensor | None = None
    ) -> None:
        self.x = _to_float_tensor(samples)
        if labels is None:
            self.y = torch.zeros(len(self.x), dtype=torch.long)
        else:
            self.y = torch.as_tensor(labels, dtype=torch.long)
            if len(self.y) != len(self.x):
                msg = f"Mismatched sample ({len(self.x)}) and label ({len(self.y)}) counts"
                raise ValueError(msg)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.x)

    @override
    def __getitem__(self, index: int, /) -> tuple[Tensor, Tensor]:
        return self.x[index], self.y[index]


class TSTCCDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Dataset returning ``(x, weak_aug, strong_aug)`` views for TS-TCC pretraining.

    Args:
        samples: Unlabeled wearable time series of shape ``(S, C, L)``.
        jitter_scale_ratio: Standard deviation for weak scaling augmentation.
        jitter_ratio: Standard deviation for strong jitter augmentation.
        max_seg: Maximum temporal segments for strong permutation augmentation.
        scale_loc: Mean of the Gaussian scaling factor distribution.
        seed: Optional random seed for deterministic view generation.
    """

    def __init__(
        self,
        samples: NDArray[np.floating] | Tensor,
        /,
        *,
        jitter_scale_ratio: float,
        jitter_ratio: float,
        max_seg: int,
        scale_loc: float,
        seed: int | None = None,
    ) -> None:
        self.x = _to_float_tensor(samples)
        self.augment = TSTCCAugmentation(
            jitter_scale_ratio=jitter_scale_ratio,
            jitter_ratio=jitter_ratio,
            max_seg=max_seg,
            scale_loc=scale_loc,
            seed=seed,
        )

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.x)

    @override
    def __getitem__(self, index: int, /) -> tuple[Tensor, Tensor, Tensor]:
        x_i = self.x[index]
        weak, strong = self.augment(x_i)
        return x_i, weak, strong


class TFCDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Dataset returning ``(x_t, x_t_aug, x_f, x_f_aug)`` views for TFC pretraining.

    Args:
        samples: Unlabeled wearable time series of shape ``(S, C, L)``.
        jitter_ratio: Time-domain jitter noise standard deviation.
        jitter_scale_ratio: Time-domain scaling standard deviation.
        max_seg: Time-domain maximum permutation segments.
        scale_loc: Mean of the time-domain Gaussian scaling factor distribution.
        freq_mask_ratio: Frequency-domain masking probability.
        freq_add_ratio: Frequency-domain perturbation probability.
        seed: Optional random seed for deterministic view generation.
    """

    def __init__(
        self,
        samples: NDArray[np.floating] | Tensor,
        /,
        *,
        jitter_ratio: float,
        jitter_scale_ratio: float,
        max_seg: int,
        scale_loc: float,
        freq_mask_ratio: float,
        freq_add_ratio: float,
        seed: int | None = None,
    ) -> None:
        self.x_t = _to_float_tensor(samples)
        self.x_f = compute_fft_magnitude(self.x_t)
        self.time_aug = TFCTimeAugmentation(
            jitter_ratio=jitter_ratio,
            jitter_scale_ratio=jitter_scale_ratio,
            max_seg=max_seg,
            scale_loc=scale_loc,
            seed=seed,
        )
        self.freq_aug = TFCFrequencyAugmentation(
            mask_ratio=freq_mask_ratio, perturb_ratio=freq_add_ratio, seed=None if seed is None else seed + 100
        )

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.x_t)

    @override
    def __getitem__(self, index: int, /) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x_t = self.x_t[index]
        x_f = self.x_f[index]
        return x_t, self.time_aug(x_t), x_f, self.freq_aug(x_f)

"""Time-domain and frequency-domain augmentations for SSL pretraining.

Combines the clean transform abstractions from ``relapse-prediction`` with the
TS-TCC (Eldele et al., IJCAI 2021) and TFC (Zhang et al., NeurIPS 2022)
augmentation banks used in Fekas et al. (IEEE ICASSP 2023).
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING, override

import numpy as np
import torch

from torch import Tensor
from torch.utils.data import get_worker_info

from .base import NDIM_2D, NDIM_3D, BaseTransform

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MIN_PERM_SEGMENTS = 2
_MIN_PERM_TIMESTEPS = 4
_NUM_TFC_TIME_AUGS = 3
_FREQ_AUG_PROB = 0.5


class _WorkerSafeNumpyRNGMixin:
    """Re-seed NumPy ``Generator`` per ``DataLoader`` worker and per epoch."""

    _base_seed: int | None
    _offset: int
    _rng: np.random.Generator
    _worker_seed: int | None

    def _init_rng(self, seed: int | None, /, *, offset: int = 0) -> None:
        self._base_seed = seed
        self._offset = offset
        self._rng = np.random.default_rng(None if seed is None else [seed, offset])
        self._worker_seed = None

    def _active_rng(self) -> np.random.Generator:
        worker = get_worker_info()
        if worker is not None and self._worker_seed != worker.seed:
            seeds = (
                [worker.seed, worker.id, self._offset]
                if self._base_seed is None
                else [self._base_seed, worker.seed, worker.id, self._offset]
            )
            self._rng = np.random.default_rng(seeds)
            self._worker_seed = worker.seed
        return self._rng


class JitterTransform(_WorkerSafeNumpyRNGMixin, BaseTransform):
    """Additive zero-mean Gaussian noise augmentation (Um et al., 2017).

    Args:
        sigma: Standard deviation of the Gaussian noise ``N(0, sigma)``.
        seed: Optional random seed for deterministic augmentation.
    """

    def __init__(self, *, sigma: float = 0.1, seed: int | None = None) -> None:
        if not math.isfinite(sigma) or sigma < 0:
            msg = f"sigma must be finite and non-negative, got {sigma}"
            raise ValueError(msg)
        self.sigma = sigma
        self._init_rng(seed)

    @override
    def transform[FloatT: np.floating](self, data: NDArray[FloatT], /) -> NDArray[FloatT]:
        rng = self._active_rng()
        noise = rng.normal(loc=0.0, scale=self.sigma, size=data.shape).astype(data.dtype, copy=False)
        return (data + noise).astype(data.dtype, copy=False)


class ScalingTransform(_WorkerSafeNumpyRNGMixin, BaseTransform):
    """Per-channel multiplicative scaling augmentation (Um et al., 2017).

    Multiplies each channel in each sample by a scalar drawn from ``N(loc, sigma)``.

    Args:
        sigma: Standard deviation of the scaling factor distribution.
        loc: Mean of the scaling factor distribution (default ``1.0``).
        seed: Optional random seed for deterministic augmentation.
    """

    def __init__(self, *, sigma: float = 0.1, loc: float = 1.0, seed: int | None = None) -> None:
        if not math.isfinite(sigma) or sigma < 0 or not math.isfinite(loc):
            msg = f"sigma must be finite >= 0 and loc must be finite, got sigma={sigma}, loc={loc}"
            raise ValueError(msg)
        self.sigma = sigma
        self.loc = loc
        self._init_rng(seed)

    @override
    def transform[FloatT: np.floating](self, data: NDArray[FloatT], /) -> NDArray[FloatT]:
        rng = self._active_rng()
        batch_size, n_channels, _ = data.shape
        factors = rng.normal(loc=self.loc, scale=self.sigma, size=(batch_size, n_channels)).astype(
            data.dtype, copy=False
        )
        return (data * factors[..., np.newaxis]).astype(data.dtype, copy=False)


class PermutationTransform(_WorkerSafeNumpyRNGMixin, BaseTransform):
    """Random temporal segment permutation augmentation.

    Splits each sample along the time axis into ``2..max_segments`` chunks and randomly
    permutes those chunks identically across all channels to preserve cross-channel alignment.

    Args:
        max_segments: Maximum number of temporal segments (must be ``>= 2``).
        random_seg_mode: If ``True``, sample random interior split indices; otherwise split evenly.
        seed: Optional random seed for deterministic augmentation.
    """

    def __init__(self, *, max_segments: int = 5, random_seg_mode: bool = True, seed: int | None = None) -> None:
        if max_segments < _MIN_PERM_SEGMENTS:
            msg = f"max_segments must be >= {_MIN_PERM_SEGMENTS}, got {max_segments}"
            raise ValueError(msg)
        self.max_segments = max_segments
        self.random_seg_mode = random_seg_mode
        self._init_rng(seed)

    @override
    def transform[FloatT: np.floating](self, data: NDArray[FloatT], /) -> NDArray[FloatT]:
        rng = self._active_rng()
        batch_size, _, n_timesteps = data.shape
        if n_timesteps < _MIN_PERM_TIMESTEPS:
            return data.copy()

        effective_max_segs = min(self.max_segments, n_timesteps - 1)
        orig_steps = np.arange(n_timesteps)
        num_segs = rng.integers(_MIN_PERM_SEGMENTS, effective_max_segs + 1, size=batch_size)
        warped_indices = np.tile(orig_steps, (batch_size, 1))

        for b in range(batch_size):
            n_seg = int(num_segs[b])
            if self.random_seg_mode and n_timesteps >= n_seg:
                split_points = np.sort(rng.choice(np.arange(1, n_timesteps), size=n_seg - 1, replace=False))
                splits = list(np.split(orig_steps, split_points))
            else:
                splits = list(np.array_split(orig_steps, n_seg))
            rng.shuffle(splits)
            warped_indices[b] = np.concatenate(splits)

        return np.take_along_axis(data, warped_indices[:, np.newaxis, :], axis=2)


class TSTCCAugmentation:
    """Weak and strong view generator for TS-TCC (Eldele et al., IJCAI 2021).

    - Weak augmentation: scaling (``scale_loc=1.0`` matches Um et al. ``arXiv:1706.00527``;
      set ``scale_loc=2.0`` for upstream ``TS-TCC``'s mean-2.0 scaling).
    - Strong augmentation: temporal permutation followed by Gaussian jitter.

    Args:
        jitter_scale_ratio: Standard deviation for weak scaling.
        jitter_ratio: Standard deviation for strong jitter.
        max_seg: Maximum segment count for strong permutation.
        scale_loc: Mean of the Gaussian scaling factor distribution.
        seed: Optional random seed.
    """

    def __init__(
        self,
        *,
        jitter_scale_ratio: float = 2.0,
        jitter_ratio: float = 0.1,
        max_seg: int = 12,
        scale_loc: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.weak_scale = ScalingTransform(sigma=jitter_scale_ratio, loc=scale_loc, seed=seed)
        self.strong_perm = PermutationTransform(
            max_segments=max(_MIN_PERM_SEGMENTS, max_seg), seed=None if seed is None else seed + 1
        )
        self.strong_jitter = JitterTransform(sigma=jitter_ratio, seed=None if seed is None else seed + 2)

    def __call__(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        """Generate ``(weak_aug, strong_aug)`` tensors from a batch or single sample.

        Args:
            x: Tensor of shape ``(B, C, L)`` or ``(C, L)``.

        Returns:
            Tuple of ``(weak_aug, strong_aug)`` matching ``x`` shape, dtype, and device.
        """
        if x.ndim not in {NDIM_2D, NDIM_3D}:
            msg = f"Expected 2D (C, L) or 3D (B, C, L) tensor, got {x.ndim}D"
            raise ValueError(msg)
        was_2d = x.ndim == NDIM_2D
        arr = x.detach().cpu().numpy()
        weak = self.weak_scale(arr)
        strong = self.strong_jitter(self.strong_perm(arr))
        weak_t = torch.from_numpy(weak).to(device=x.device, dtype=x.dtype)
        strong_t = torch.from_numpy(strong).to(device=x.device, dtype=x.dtype)
        if was_2d:
            return weak_t.squeeze(0), strong_t.squeeze(0)
        return weak_t, strong_t


class TFCTimeAugmentation(_WorkerSafeNumpyRNGMixin):
    """Time-domain augmentation bank for TFC (randomly picks jitter, scaling, or permutation).

    Args:
        jitter_ratio: Jitter noise standard deviation.
        jitter_scale_ratio: Scaling factor standard deviation.
        max_seg: Maximum segments for permutation.
        scale_loc: Mean of the Gaussian scaling factor distribution (``1.0`` per Um et al.,
            or ``2.0`` per upstream ``TFC-pretraining``).
        seed: Optional random seed.
    """

    def __init__(
        self,
        *,
        jitter_ratio: float = 0.3,
        jitter_scale_ratio: float = 1.1,
        max_seg: int = 16,
        scale_loc: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.jitter = JitterTransform(sigma=jitter_ratio, seed=seed)
        self.scaling = ScalingTransform(
            sigma=jitter_scale_ratio, loc=scale_loc, seed=None if seed is None else seed + 1
        )
        self.permutation = PermutationTransform(
            max_segments=max(_MIN_PERM_SEGMENTS, max_seg), seed=None if seed is None else seed + 2
        )
        self._init_rng(seed, offset=3)

    def __call__(self, x: Tensor, /) -> Tensor:
        """Apply one randomly selected time-domain augmentation per sample in the batch.

        Args:
            x: Input time-series tensor of shape ``(B, C, L)`` or ``(C, L)``.

        Returns:
            Augmented tensor with the same shape, dtype, and device as ``x``.
        """
        if x.ndim not in {NDIM_2D, NDIM_3D}:
            msg = f"Expected 2D (C, L) or 3D (B, C, L) tensor, got {x.ndim}D"
            raise ValueError(msg)
        was_2d = x.ndim == NDIM_2D
        arr = x.detach().cpu().numpy()
        if was_2d:
            arr = arr[np.newaxis, ...]

        rng = self._active_rng()
        choices = rng.integers(0, _NUM_TFC_TIME_AUGS, size=arr.shape[0])
        out = np.empty_like(arr)

        for idx, transform in enumerate((self.jitter, self.scaling, self.permutation)):
            mask = choices == idx
            if mask.any():
                out[mask] = transform(arr[mask])

        out_t = torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
        return out_t.squeeze(0) if was_2d else out_t


def _rand_like_tensor(tensor: Tensor, /, *, generator: torch.Generator | None) -> Tensor:
    """Sample uniform ``[0, 1)`` noise matching ``tensor`` shape, dtype, and device."""
    if generator is None:
        return torch.rand(tensor.shape, device=tensor.device, dtype=tensor.dtype)
    return torch.rand(tensor.shape, generator=generator, device=generator.device, dtype=tensor.dtype).to(
        device=tensor.device
    )


def remove_frequency(
    x_freq: Tensor, /, *, mask_ratio: float = 0.1, generator: torch.Generator | None = None
) -> Tensor:
    """Randomly mask out spectral components with probability ``mask_ratio``.

    Args:
        x_freq: Frequency-domain magnitude tensor of shape ``(B, C, L)`` or ``(C, L)``.
        mask_ratio: Fraction of frequency bins set to zero.
        generator: Optional CPU ``torch.Generator`` for deterministic sampling.

    Returns:
        Masked frequency-domain tensor on the same device as ``x_freq``.
    """
    mask = _rand_like_tensor(x_freq, generator=generator) > mask_ratio
    return x_freq * mask.to(dtype=x_freq.dtype)


def add_frequency(
    x_freq: Tensor,
    /,
    *,
    perturb_ratio: float = 0.1,
    amplitude_scale: float = 0.1,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Randomly inject small per-sample, per-channel frequency perturbations into ``x_freq``.

    Args:
        x_freq: Frequency-domain magnitude tensor of shape ``(B, C, L)`` or ``(C, L)``.
        perturb_ratio: Fraction of frequency bins perturbed.
        amplitude_scale: Perturbation magnitude relative to each sample/channel max spectral amplitude.
        generator: Optional CPU ``torch.Generator`` for deterministic sampling.

    Returns:
        Perturbed frequency-domain tensor on the same device as ``x_freq``.
    """
    mask = (_rand_like_tensor(x_freq, generator=generator) > (1.0 - perturb_ratio)).to(dtype=x_freq.dtype)
    max_amp = x_freq.amax(dim=-1, keepdim=True).clamp(min=1e-6)
    random_amp = _rand_like_tensor(x_freq, generator=generator) * (max_amp * amplitude_scale)
    return x_freq + mask * random_amp


class TFCFrequencyAugmentation:
    """Frequency-domain augmentation bank for TFC (randomly picks frequency removal or addition).

    Args:
        mask_ratio: Fraction of bins masked by ``remove_frequency``.
        perturb_ratio: Fraction of bins perturbed by ``add_frequency``.
        seed: Optional seed for deterministic spectral perturbations.
    """

    def __init__(self, *, mask_ratio: float = 0.1, perturb_ratio: float = 0.1, seed: int | None = None) -> None:
        if not (0.0 <= mask_ratio <= 1.0) or not (0.0 <= perturb_ratio <= 1.0):
            msg = f"mask_ratio and perturb_ratio must lie in [0, 1], got {mask_ratio}, {perturb_ratio}"
            raise ValueError(msg)
        self.mask_ratio = mask_ratio
        self.perturb_ratio = perturb_ratio
        self._base_seed = seed
        self._worker_seed: int | None = None
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def _active_generator(self) -> torch.Generator | None:
        worker = get_worker_info()
        if worker is not None and self._worker_seed != worker.seed:
            gen = torch.Generator()
            mixed = worker.seed + worker.id * 100_003 + (0 if self._base_seed is None else self._base_seed * 1_000_003)
            gen.manual_seed(mixed % (2**63 - 1))
            self._generator = gen
            self._worker_seed = worker.seed
        return self._generator

    def __call__(self, x_freq: Tensor, /) -> Tensor:
        """Apply one randomly selected frequency-domain augmentation per sample.

        Args:
            x_freq: Frequency-domain tensor of shape ``(B, C, L)`` or ``(C, L)``.

        Returns:
            Augmented frequency-domain tensor with the same shape, dtype, and device.
        """
        if x_freq.ndim not in {NDIM_2D, NDIM_3D}:
            msg = f"Expected 2D (C, L) or 3D (B, C, L) frequency tensor, got {x_freq.ndim}D"
            raise ValueError(msg)
        gen = self._active_generator()
        was_2d = x_freq.ndim == NDIM_2D
        batch = x_freq.unsqueeze(0) if was_2d else x_freq
        aug_1 = remove_frequency(batch, mask_ratio=self.mask_ratio, generator=gen)
        aug_2 = add_frequency(batch, perturb_ratio=self.perturb_ratio, generator=gen)

        if gen is None:
            coin = torch.rand(batch.shape[0], 1, 1, device=batch.device)
        else:
            coin = torch.rand(batch.shape[0], 1, 1, generator=gen).to(device=batch.device)
        select_first = (coin < _FREQ_AUG_PROB).to(dtype=batch.dtype)
        out = select_first * aug_1 + (1.0 - select_first) * aug_2
        return out.squeeze(0) if was_2d else out

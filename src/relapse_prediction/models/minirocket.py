"""Self-contained PyTorch Multivariate MiniRocket feature extractor.

Implements the unsupervised Multivariate MiniRocket baseline (Dempster et al., KDD 2021,
``arXiv:2012.08791``; matching ``tsai``'s ``MiniRocketFeatures`` used in Fekas et al.,
IEEE ICASSP 2023 Table 2 and Figures 2-3) without heavy external dependencies such as
``tsai``/``fastai`` or Numba compilation.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor

if TYPE_CHECKING:
    from numpy.typing import NDArray

_NUM_BASE_KERNELS = 84
_KERNEL_LENGTH = 9
_MAX_CHANNELS_PER_KERNEL = 9
_EXPECTED_INPUT_NDIM = 3
_GOLDEN_RATIO = (1.0 + float(np.sqrt(5.0))) / 2.0


def _build_84_kernels() -> Tensor:
    """Build the 84 canonical MiniRocket length-9 kernels with weights in ``{-1, 2}``."""
    kernels = torch.full((_NUM_BASE_KERNELS, _KERNEL_LENGTH), -1.0, dtype=torch.float32)
    for idx, pos in enumerate(combinations(range(_KERNEL_LENGTH), 3)):
        kernels[idx, list(pos)] = 2.0
    return kernels


def _compute_dilations(
    *, seq_len: int, num_features: int, max_dilations_per_kernel: int
) -> tuple[list[int], list[int]]:
    """Compute exponentially spaced dilations and log-density-weighted feature counts.

    Matches ``_fit_dilations`` in Dempster et al. (2021) and ``tsai``: preserves
    ``return_counts=True`` multiplicities when ``int(2**x)`` maps multiple log-spaced
    exponents onto fine dilations ``d = 1, 2, 3, ...``.
    """
    num_features_per_kernel = max(1, num_features // _NUM_BASE_KERNELS)
    true_max_dilations = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / float(true_max_dilations)

    max_exp = np.log2(max(1.0, (seq_len - 1) / (_KERNEL_LENGTH - 1)))
    raw_dilations = np.logspace(0.0, max_exp, num=true_max_dilations, base=2.0).astype(np.int64)
    unique_dilations, raw_counts = np.unique(np.maximum(raw_dilations, 1), return_counts=True)

    scaled_counts = np.floor(raw_counts * multiplier).astype(np.int64)
    remainder = int(num_features_per_kernel - int(scaled_counts.sum()))
    if remainder > 0:
        scaled_counts[:remainder] += 1

    nonzero = scaled_counts > 0
    return [int(d) for d in unique_dilations[nonzero]], [int(c) for c in scaled_counts[nonzero]]


class MiniRocketExtractor:
    """Unsupervised Multivariate MiniRocket feature extractor.

    Applies the 84 fixed length-9 ``{-1, 2}`` convolutional kernels across exponentially
    spaced dilations, log-uniformly sized random channel subsets, and alternating
    padded/valid convolutions, calibrating bias thresholds from unclamped golden-ratio
    low-discrepancy quantiles on the fitting cohort.

    Args:
        num_features: Output feature count (rounded down to a multiple of 84; default ``9996``).
        max_dilations_per_kernel: Upper bound on log-spaced dilation candidates per kernel.
        device: PyTorch device used for convolution and quantile computation.
        seed: Random seed controlling channel combinations and bias-calibration samples.
    """

    def __init__(
        self,
        *,
        num_features: int = 9996,
        max_dilations_per_kernel: int = 32,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        if num_features < _NUM_BASE_KERNELS:
            msg = f"num_features must be >= {_NUM_BASE_KERNELS}, got {num_features}"
            raise ValueError(msg)
        self.num_features = (num_features // _NUM_BASE_KERNELS) * _NUM_BASE_KERNELS
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.device = torch.device(device)
        self.seed = seed

        self._kernels = _build_84_kernels()
        self._dilations: list[int] = []
        self._counts_per_dilation: list[int] = []
        self._channel_weights: list[Tensor] = []
        self._biases: list[Tensor] = []
        self._n_channels: int = 0
        self._seq_len: int = 0
        self._fitted: bool = False

    @property
    def output_dim(self) -> int:
        """Exact dimensionality of extracted MiniRocket features."""
        return self.num_features

    def _init_channel_projections(self, n_channels: int, /) -> None:
        """Sample log-uniform channel-combination matrices per ``(dilation, kernel)``."""
        rng = np.random.default_rng(self.seed)
        self._channel_weights = []
        max_channels = min(n_channels, _MAX_CHANNELS_PER_KERNEL)
        max_exp_channels = float(np.log2(max_channels + 1))

        for _ in self._dilations:
            mask = np.zeros((_NUM_BASE_KERNELS, n_channels), dtype=np.float32)
            for k in range(_NUM_BASE_KERNELS):
                raw_n = int(2.0 ** float(rng.uniform(0.0, max_exp_channels)))
                n_sel = max(1, min(raw_n, max_channels))
                chosen = rng.choice(n_channels, size=n_sel, replace=False)
                mask[k, chosen] = 1.0
            self._channel_weights.append(torch.from_numpy(mask))

    def fit(self, x: NDArray[np.floating[Any]] | Tensor, /) -> Self:
        """Fit dilation schedule, channel projections, and quantile bias thresholds.

        Args:
            x: Training time series of shape ``(S, C, L)`` with ``S >= 1`` and ``L >= 9``.

        Returns:
            ``self`` for method chaining.
        """
        x_t = torch.as_tensor(x, dtype=torch.float32)
        if x_t.ndim != _EXPECTED_INPUT_NDIM or x_t.shape[0] == 0:
            msg = f"Expected non-empty 3D input (S, C, L), got shape {tuple(x_t.shape)}"
            raise ValueError(msg)

        n_samples, n_channels, seq_len = x_t.shape
        if seq_len < _KERNEL_LENGTH:
            msg = f"MiniRocket requires seq_len >= {_KERNEL_LENGTH}, got {seq_len}"
            raise ValueError(msg)

        self._n_channels = n_channels
        self._seq_len = seq_len
        self._dilations, self._counts_per_dilation = _compute_dilations(
            seq_len=seq_len, num_features=self.num_features, max_dilations_per_kernel=self.max_dilations_per_kernel
        )
        self._init_channel_projections(n_channels)

        rng = np.random.default_rng(self.seed + 1)
        self._biases = []

        max_cal = max(_NUM_BASE_KERNELS, len(self._dilations) * _NUM_BASE_KERNELS)
        n_cal = min(n_samples, max_cal)
        cal_idx = rng.choice(n_samples, size=n_cal, replace=False)
        x_cal = x_t[cal_idx].to(self.device)

        q_step = 0
        with torch.no_grad():
            for dil_idx, (dilation, count) in enumerate(zip(self._dilations, self._counts_per_dilation, strict=True)):
                conv_out = self._convolve_dilation(x_cal, dil_idx=dil_idx, dilation=dilation)
                biases_d = torch.empty((_NUM_BASE_KERNELS, count), dtype=torch.float32, device=self.device)
                for k in range(_NUM_BASE_KERNELS):
                    sample_i = int(rng.integers(0, n_cal))
                    vals = conv_out[sample_i, k]
                    quantiles = torch.tensor(
                        [((q_step + j + 1) * _GOLDEN_RATIO) % 1.0 for j in range(count)],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    q_step += count
                    biases_d[k] = torch.quantile(vals, quantiles)
                self._biases.append(biases_d.cpu())

        self._fitted = True
        return self

    def _convolve_dilation(self, x_t: Tensor, /, *, dil_idx: int, dilation: int) -> Tensor:
        """Convolve ``x_t`` with the 84 MiniRocket kernels using symmetric ``4 * dilation`` padding."""
        ch_w = self._channel_weights[dil_idx].to(device=x_t.device, dtype=x_t.dtype)
        kernels = self._kernels.to(device=x_t.device, dtype=x_t.dtype)
        weight = ch_w.unsqueeze(-1) * kernels.unsqueeze(1)
        padding = ((_KERNEL_LENGTH - 1) * dilation) // 2
        return F.conv1d(x_t, weight, padding=padding, dilation=dilation)

    def _compute_alternating_ppv(self, conv_out: Tensor, bias: Tensor, /, *, dil_idx: int, dilation: int) -> Tensor:
        """Compute PPV alternating full padded vs. valid interior slices per ``(dil_idx + k) % 2``."""
        batch_len = conv_out.shape[0]
        count = bias.shape[1]
        padding = ((_KERNEL_LENGTH - 1) * dilation) // 2
        pad_parity = dil_idx % 2
        unpad_parity = 1 - pad_parity

        ppv = torch.empty((batch_len, _NUM_BASE_KERNELS, count), dtype=conv_out.dtype, device=conv_out.device)
        padded_cmp = conv_out[:, pad_parity::2].unsqueeze(2) > bias[pad_parity::2].unsqueeze(0).unsqueeze(-1)
        ppv[:, pad_parity::2] = padded_cmp.float().mean(dim=-1)

        valid_conv = conv_out[:, unpad_parity::2, padding:-padding] if padding > 0 else conv_out[:, unpad_parity::2]
        unpadded_cmp = valid_conv.unsqueeze(2) > bias[unpad_parity::2].unsqueeze(0).unsqueeze(-1)
        ppv[:, unpad_parity::2] = unpadded_cmp.float().mean(dim=-1)
        return ppv.reshape(batch_len, _NUM_BASE_KERNELS * count)

    def transform(self, x: NDArray[np.floating[Any]] | Tensor, /, *, batch_size: int = 256) -> NDArray[np.float32]:
        """Transform time-series windows ``(S, C, L)`` into MiniRocket PPV features ``(S, F)``.

        Args:
            x: Input array or tensor of shape ``(S, C, L)`` matching fitted ``(C, L)``.
            batch_size: Mini-batch size for memory-bounded feature extraction.

        Returns:
            ``float32`` NumPy feature array of shape ``(S, output_dim)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If input shape does not match fitted ``(C, L)``.
        """
        if not self._fitted:
            msg = "MiniRocketExtractor must be fitted before calling transform()"
            raise RuntimeError(msg)

        x_t = torch.as_tensor(x, dtype=torch.float32)
        if x_t.ndim != _EXPECTED_INPUT_NDIM or x_t.shape[1] != self._n_channels or x_t.shape[2] != self._seq_len:
            msg = f"Expected shape (S, {self._n_channels}, {self._seq_len}), got {tuple(x_t.shape)}"
            raise ValueError(msg)
        if x_t.shape[0] == 0:
            return np.empty((0, self.num_features), dtype=np.float32)

        chunks: list[NDArray[np.float32]] = []
        with torch.no_grad():
            for start in range(0, x_t.shape[0], batch_size):
                batch = x_t[start : start + batch_size].to(self.device)
                batch_feats: list[Tensor] = []
                for dil_idx, dilation in enumerate(self._dilations):
                    conv_out = self._convolve_dilation(batch, dil_idx=dil_idx, dilation=dilation)
                    bias = self._biases[dil_idx].to(device=batch.device, dtype=batch.dtype)
                    batch_feats.append(
                        self._compute_alternating_ppv(conv_out, bias, dil_idx=dil_idx, dilation=dilation)
                    )
                chunks.append(np.asarray(torch.cat(batch_feats, dim=1).cpu().numpy(), dtype=np.float32))

        return np.concatenate(chunks, axis=0)

    def fit_transform(self, x: NDArray[np.floating[Any]] | Tensor, /, *, batch_size: int = 256) -> NDArray[np.float32]:
        """Fit on ``x`` and return extracted PPV features."""
        return self.fit(x).transform(x, batch_size=batch_size)

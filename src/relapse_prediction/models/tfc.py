"""Time-Frequency Consistency (TFC) encoder (Zhang et al., NeurIPS 2022)."""

from __future__ import annotations

from typing import override

import torch

from torch import Tensor, nn

from relapse_prediction.config import TFCConfig
from relapse_prediction.data.datasets import compute_fft_magnitude

from .base import BaseSSLEncoder, build_conv_encoder_blocks

_EXPECTED_NDIM = 3


class TFCModel(BaseSSLEncoder):
    """Dual time-domain and frequency-domain contrastive encoder for TFC.

    Note on paper fidelity:
        Section 3 of Fekas et al. (IEEE ICASSP 2023) refers to 3-layer 1-D ResNets,
        whereas the trained TFC encoder (adapted from upstream ``mims-harvard/TFC-pretraining``)
        uses two 3-block 1-D CNN branches (``32 -> 64 -> 128``) with 2-layer cross-space
        projectors (``256 -> 128``). Concatenating ``[z_time, z_freq]`` yields the
        256-dimensional embeddings reported in Section 4.

    Args:
        config: :class:`TFCConfig` specifying input channels, sequence length, and architecture.
    """

    def __init__(self, config: TFCConfig | None = None, /) -> None:
        super().__init__()
        cfg = config or TFCConfig()
        self.config = cfg

        self.time_encoder = build_conv_encoder_blocks(
            input_channels=cfg.input_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            final_out_channels=cfg.final_out_channels,
            dropout=cfg.dropout,
        )
        self.freq_encoder = build_conv_encoder_blocks(
            input_channels=cfg.input_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            final_out_channels=cfg.final_out_channels,
            dropout=cfg.dropout,
        )

        # Infer flattened CNN output dim in eval mode so BatchNorm running stats stay untouched.
        self.time_encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.input_channels, cfg.seq_len)
            flat_dim = int(self.time_encoder(dummy).reshape(1, -1).shape[1])
        self.time_encoder.train()

        self.projector_t = nn.Sequential(
            nn.Linear(flat_dim, cfg.projection_hidden_dim),
            nn.BatchNorm1d(cfg.projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.projection_hidden_dim, cfg.projection_dim),
        )
        self.projector_f = nn.Sequential(
            nn.Linear(flat_dim, cfg.projection_hidden_dim),
            nn.BatchNorm1d(cfg.projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.projection_hidden_dim, cfg.projection_dim),
        )
        self._embedding_dim = 2 * cfg.projection_dim

    @property
    @override
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @override
    def forward(self, x_in_t: Tensor, x_in_f: Tensor, /) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute time and frequency representations and cross-space projections.

        Args:
            x_in_t: Time-domain batch of shape ``(B, C, L)``.
            x_in_f: Frequency-domain FFT magnitude batch of shape ``(B, C, L)``.

        Returns:
            Tuple ``(h_time, z_time, h_freq, z_freq)``.
        """
        expected_tail = (self.config.input_channels, self.config.seq_len)
        if (
            x_in_t.ndim != _EXPECTED_NDIM
            or x_in_f.ndim != _EXPECTED_NDIM
            or x_in_t.shape[1:] != expected_tail
            or x_in_f.shape != x_in_t.shape
        ):
            msg = (
                f"TFCModel expected matching 3D inputs (B, {self.config.input_channels}, {self.config.seq_len}), "
                f"got {tuple(x_in_t.shape)} and {tuple(x_in_f.shape)}"
            )
            raise ValueError(msg)
        h_time: Tensor = self.time_encoder(x_in_t).reshape(x_in_t.shape[0], -1)
        z_time: Tensor = self.projector_t(h_time)

        h_freq: Tensor = self.freq_encoder(x_in_f).reshape(x_in_f.shape[0], -1)
        z_freq: Tensor = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq

    @override
    def encode(self, x: Tensor, /) -> Tensor:
        """Extract concatenated ``[z_time, z_freq]`` embeddings of shape ``(B, 256)``."""
        x_f = compute_fft_magnitude(x)
        _, z_time, _, z_freq = self.forward(x, x_f)
        return torch.cat([z_time, z_freq], dim=1)

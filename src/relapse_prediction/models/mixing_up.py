"""Mixing-Up Contrastive Learning FCN encoder (Wickstrøm et al., PRL 2022)."""

from __future__ import annotations

from typing import override

from torch import Tensor, nn

from relapse_prediction.config import MixingUpConfig

from .base import BaseSSLEncoder


class MixingUpFCN(BaseSSLEncoder):
    """Three-layer dilated 1-D Fully Convolutional Network with projection head.

    Matches the Mixing-Up architecture trained in Fekas et al. (IEEE ICASSP 2023):
    three dilated Conv1d + BatchNorm1d + ReLU blocks followed by global average pooling
    producing 128-dimensional embeddings (``h``) and a 2-layer MLP projection head (``z``).

    Args:
        config: :class:`MixingUpConfig` specifying channels, kernels, dilations, and projection dim.
    """

    def __init__(self, config: MixingUpConfig | None = None, /) -> None:
        super().__init__()
        cfg = config or MixingUpConfig()
        c1, c2, c3 = cfg.hidden_channels
        k1, k2, k3 = cfg.kernel_sizes
        d1, d2, d3 = cfg.dilations

        p1 = ((k1 - 1) * d1) // 2
        p2 = ((k2 - 1) * d2) // 2
        p3 = ((k3 - 1) * d3) // 2

        self.encoder = nn.Sequential(
            nn.Conv1d(cfg.input_channels, c1, kernel_size=k1, padding=p1, dilation=d1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=p2, dilation=d2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=k3, padding=p3, dilation=d3),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(c3, cfg.projection_dim),
            nn.BatchNorm1d(cfg.projection_dim),
            nn.ReLU(),
            nn.Linear(cfg.projection_dim, cfg.projection_dim),
        )
        self._embedding_dim = c3

    @property
    @override
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @override
    def forward(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        """Compute projection ``z`` and backbone representation ``h``.

        Args:
            x: Input tensor of shape ``(B, C, L)``.

        Returns:
            Tuple ``(z, h)`` where ``z`` has shape ``(B, projection_dim)`` and ``h`` has
            shape ``(B, embedding_dim)``.
        """
        h: Tensor = self.encoder(x)
        z: Tensor = self.proj_head(h)
        return z, h

    @override
    def encode(self, x: Tensor, /) -> Tensor:
        """Extract 128-dimensional backbone embeddings for downstream tasks."""
        h: Tensor = self.encoder(x)
        return h

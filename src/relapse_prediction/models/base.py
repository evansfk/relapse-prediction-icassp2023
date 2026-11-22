"""Base protocol and shared 1-D CNN builder for self-supervised time-series encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


def build_conv_encoder_blocks(
    *, input_channels: int, kernel_size: int, stride: int, final_out_channels: int, dropout: float
) -> nn.Sequential:
    """Build the 3-block 1-D CNN backbone shared by TFC and TS-TCC."""
    return nn.Sequential(
        nn.Conv1d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            padding=kernel_size // 2,
        ),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        nn.Dropout(dropout),
        nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
        nn.BatchNorm1d(final_out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
    )


class BaseSSLEncoder(nn.Module, ABC):
    """Common interface for SSL encoders exposing fixed-width sample embeddings."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality ``D`` of the vector returned by :meth:`encode`."""

    @abstractmethod
    def encode(self, x: Tensor, /) -> Tensor:
        """Encode a batch of multivariate time series ``(B, C, L)`` into ``(B, D)`` embeddings.

        Args:
            x: Wearable time-series tensor of shape ``(B, C, L)``.

        Returns:
            Feature tensor of shape ``(B, D)``.
        """

"""Base interface for multivariate time-series augmentations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

NDIM_2D = 2
NDIM_3D = 3


class BaseTransform(ABC):
    """Base class for multivariate time-series transformations.

    All transformations operate on 3D arrays of shape ``(batch_size, n_channels, n_timesteps)``.
    If a 2D array ``(n_channels, n_timesteps)`` is passed to ``__call__``, a batch dimension
    is prepended automatically.
    """

    @abstractmethod
    def transform[FloatT: np.floating](self, data: NDArray[FloatT], /) -> NDArray[FloatT]:
        """Transform a 3D batch of multivariate time series.

        Args:
            data: Array of shape ``(batch_size, n_channels, n_timesteps)``.

        Returns:
            Transformed array with the same shape and dtype as ``data``.
        """

    def __call__[FloatT: np.floating](self, data: NDArray[FloatT], /) -> NDArray[FloatT]:
        """Validate dimensions and apply the transformation.

        Args:
            data: Array of shape ``(batch_size, n_channels, n_timesteps)`` or
                ``(n_channels, n_timesteps)``.

        Returns:
            Transformed 3D array of shape ``(batch_size, n_channels, n_timesteps)``.

        Raises:
            TypeError: If ``data`` is not a NumPy ``ndarray``.
            ValueError: If ``data`` is neither 2D nor 3D.
        """
        raw: object = data
        if not isinstance(raw, np.ndarray):
            msg = f"Input data must be a numpy ndarray, got {type(raw).__name__}"
            raise TypeError(msg)

        if data.ndim == NDIM_2D:
            batch: NDArray[FloatT] = data[np.newaxis, ...]
        elif data.ndim == NDIM_3D:
            batch = data
        else:
            msg = f"Expected 2D or 3D array, got {data.ndim}D array with shape {data.shape}"
            raise ValueError(msg)

        return self.transform(batch)

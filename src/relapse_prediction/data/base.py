"""Base interface for multivariate time-series augmentations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from relapse_prediction.utils.plotting import plot_mts

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

_NDIM_2D = 2
_NDIM_3D = 3


class BaseTransform(ABC):
    """Base class for multivariate time-series transformations.

    All transformations operate on 3D arrays of shape ``(batch_size, n_channels, n_timesteps)``.
    If a 2D array ``(n_channels, n_timesteps)`` is passed to ``__call__``, a batch dimension
    is prepended automatically.
    """

    @abstractmethod
    def transform[FloatT: np.floating[Any]](self, data: NDArray[FloatT], /) -> NDArray[FloatT]:
        """Transform a 3D batch of multivariate time series.

        Args:
            data: Array of shape ``(batch_size, n_channels, n_timesteps)``.

        Returns:
            Transformed array with the same shape and dtype as ``data``.
        """

    def __call__[FloatT: np.floating[Any]](self, data: NDArray[FloatT], /) -> NDArray[FloatT]:
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

        if data.ndim == _NDIM_2D:
            batch: NDArray[FloatT] = data[np.newaxis, ...]
        elif data.ndim == _NDIM_3D:
            batch = data
        else:
            msg = f"Expected 2D or 3D array, got {data.ndim}D array with shape {data.shape}"
            raise ValueError(msg)

        return self.transform(batch)

    def plot_transform(self, data: NDArray[np.floating[Any]], /, *, show: bool = True) -> Figure:
        """Visualize an original sample alongside its transformed version.

        Args:
            data: 2D ``(C, L)`` or 3D ``(B, C, L)`` time-series array.
            show: Whether to call ``plt.show()``.

        Returns:
            Matplotlib ``Figure`` comparing original and transformed series.
        """
        transformed = self(data)
        original = data if data.ndim == _NDIM_2D else data[0]
        return plot_mts([original, transformed[0]], titles=["Original", self.__class__.__name__], show=show)

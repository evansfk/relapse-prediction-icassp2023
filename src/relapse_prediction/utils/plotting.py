"""Visualization utilities for multivariate time series and survival results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

MAX_SERIES = 5
_MTS_NDIM = 2
_CHANCE_C_INDEX = 0.5


def _validate_mts_input(mts_seq: Sequence[NDArray[np.floating[Any]]], /, *, titles: Sequence[str] | None) -> None:
    """Validate inputs for :func:`plot_mts`.

    Args:
        mts_seq: Sequence of 2D arrays with shape ``(num_channels, time_steps)``.
        titles: Optional column titles matching ``len(mts_seq)``.

    Raises:
        TypeError: If ``mts_seq`` is not a sequence of NumPy arrays.
        ValueError: If series count exceeds ``MAX_SERIES``, channel counts disagree,
            or title count does not match series count.
    """
    if not isinstance(mts_seq, Sequence) or not all(isinstance(mts, np.ndarray) for mts in mts_seq):
        msg = "Expected a sequence of NDArray, e.g., plot_mts([array1, array2], titles=['A', 'B'])."
        raise TypeError(msg)

    num_series = len(mts_seq)
    if num_series == 0 or num_series > MAX_SERIES:
        msg = f"Expected between 1 and {MAX_SERIES} timeseries, got {num_series}."
        raise ValueError(msg)

    n_channels = mts_seq[0].shape[0]
    for idx, mts in enumerate(mts_seq):
        if mts.ndim != _MTS_NDIM:
            msg = f"Timeseries at index {idx} must be 2D (C, L), got shape {mts.shape}."
            raise ValueError(msg)
        if mts.shape[0] != n_channels:
            msg = f"Timeseries at index {idx} has {mts.shape[0]} channels; expected {n_channels}."
            raise ValueError(msg)

    if titles is not None and len(titles) != num_series:
        msg = f"Number of titles ({len(titles)}) does not match number of timeseries ({num_series})."
        raise ValueError(msg)


def _plot_single_mts(
    mts: NDArray[np.floating[Any]],
    axes: list[Axes],
    /,
    *,
    title: str | None,
    channel_names: Sequence[str] | None,
    is_leftmost: bool,
) -> None:
    """Plot a single multivariate time series down a column of axes."""
    x = np.arange(mts.shape[1])
    for i, ax in enumerate(axes):
        ax.plot(x, mts[i])
        ax.grid(visible=True)
        if is_leftmost:
            label = channel_names[i] if channel_names is not None else f"Channel {i}"
            ax.set_ylabel(label)
    if title is not None:
        axes[0].set_title(title)


def plot_mts(
    mts_seq: Sequence[NDArray[np.floating[Any]]],
    /,
    *,
    titles: Sequence[str] | None = None,
    channel_names: Sequence[str] | None = None,
    show: bool = True,
) -> Figure:
    """Plot a sequence of multivariate time series side by side.

    Args:
        mts_seq: Sequence of 2D arrays of shape ``(num_channels, time_steps)``.
        titles: Optional titles for each column.
        channel_names: Optional row labels for each channel.
        show: Whether to call ``plt.show()``.

    Returns:
        The created Matplotlib ``Figure``.
    """
    _validate_mts_input(mts_seq, titles=titles)

    num_series = len(mts_seq)
    n_channels = mts_seq[0].shape[0]

    fig, axs = plt.subplots(n_channels, num_series, figsize=(6 * num_series, 1.8 * n_channels), sharex=True)
    axs_arr = np.atleast_2d(axs)
    if n_channels == 1 and num_series > 1:
        axs_arr = axs_arr.reshape(1, num_series)
    elif num_series == 1 and n_channels > 1:
        axs_arr = axs_arr.reshape(n_channels, 1)

    titles_list = list(titles) if titles is not None else [f"Series {i}" for i in range(num_series)]

    for col, mts in enumerate(mts_seq):
        column_axes: list[Axes] = [axs_arr[row, col] for row in range(n_channels)]
        _plot_single_mts(mts, column_axes, title=titles_list[col], channel_names=channel_names, is_leftmost=(col == 0))

    for col in range(num_series):
        axs_arr[-1, col].set_xlabel("Time Steps (minutes)")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_survival_comparison(results: pd.DataFrame, /, *, show: bool = False) -> Figure:
    """Plot C-index and ``-log(IBS)`` comparison bars across survival models (cf. Fig. 3).

    Args:
        results: DataFrame containing ``survival_model``, ``c_index``, and ``neg_log_ibs``
            (and optionally ``embedding``).
        show: Whether to call ``plt.show()``.

    Returns:
        The created Matplotlib ``Figure``.
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex=True)
    models = list(dict.fromkeys(results["survival_model"].tolist()))
    embeddings = (
        list(dict.fromkeys(results["embedding"].tolist())) if "embedding" in results.columns else ["embeddings"]
    )

    x = np.arange(len(models))
    width = 0.8 / max(len(embeddings), 1)

    for idx, emb in enumerate(embeddings):
        subset = results[results["embedding"] == emb] if "embedding" in results.columns else results
        offset = (idx - (len(embeddings) - 1) / 2.0) * width
        c_vals = [float(subset.loc[subset["survival_model"] == m, "c_index"].mean()) for m in models]
        ibs_vals = [float(subset.loc[subset["survival_model"] == m, "neg_log_ibs"].mean()) for m in models]
        axes[0].bar(x + offset, c_vals, width=width, label=emb)
        axes[1].bar(x + offset, ibs_vals, width=width, label=emb)

    axes[0].set_ylabel("C-index")
    axes[0].axhline(_CHANCE_C_INDEX, linestyle="--", linewidth=0.9, color="gray")
    axes[0].grid(visible=True, axis="y", alpha=0.4)
    axes[0].legend(ncol=len(embeddings), loc="upper center", bbox_to_anchor=(0.5, 1.18))

    axes[1].set_ylabel(r"$-\log(\mathrm{IBS})$")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].grid(visible=True, axis="y", alpha=0.4)

    fig.tight_layout()
    if show:
        plt.show()
    return fig

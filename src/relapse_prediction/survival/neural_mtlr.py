"""PyTorch Neural Multi-Task Logistic Regression (Neural MTLR) survival model.

Modern replacement for ``pysurvival.models.multi_task.NeuralMultiTaskModel`` matching
Fotso (2018) and the Fekas et al. (IEEE ICASSP 2023) benchmark notebooks (``2.0-4.0``):
1-hidden-layer ``tanh`` MLP (default 150 units, 150 discrete time bins on ``[0, 1.1 * max(T)]``,
and ``l2_smooth`` temporal smoothness regularization on adjacent output time weights).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self, override

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from relapse_prediction.losses.survival import MTLRLoss, mtlr_cumulative_scores
from relapse_prediction.utils.seed import seed_everything

from .base import BaseSurvivalModel, build_survival_mlp

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MIN_BINS = 2
_UNIFORM_GRID_EXPANSION = 1.1


class _MTLRNet(nn.Module):
    """Feed-forward network outputting discrete-time MTLR logits."""

    def __init__(
        self, *, in_features: int, hidden_units: int, n_bins: int, activation: Literal["tanh", "relu"], dropout: float
    ) -> None:
        super().__init__()
        self.net = build_survival_mlp(
            in_features=in_features,
            hidden_units=hidden_units,
            out_features=n_bins,
            activation=activation,
            dropout=dropout,
        )

    @property
    def output_layer(self) -> nn.Linear:
        """Final linear projection mapping hidden activations to ``n_bins`` time logits."""
        layer = self.net[-1]
        if not isinstance(layer, nn.Linear):
            msg = f"Expected final layer to be nn.Linear, got {type(layer).__name__}"
            raise TypeError(msg)
        return layer

    def temporal_smoothness_penalty(self) -> Tensor:
        """Compute Yu et al. / ``pysurvival`` ``norm_diff`` temporal smoothness penalty."""
        out = self.output_layer
        w_diff = out.weight[1:] - out.weight[:-1]
        b_diff = out.bias[1:] - out.bias[:-1]
        return w_diff.pow(2).sum() + b_diff.pow(2).sum()

    @override
    def forward(self, x: Tensor, /) -> Tensor:
        out: Tensor = self.net(x)
        return out


class NeuralMTLRModel(BaseSurvivalModel):
    """Neural Multi-Task Logistic Regression (Neural MTLR) survival model.

    Args:
        hidden_units: Hidden layer width (default ``150``, matching notebooks ``2.0-4.0``).
        bins: Number of discrete time intervals (default ``150``).
        binning: ``"uniform"`` builds an evenly spaced grid over ``(0, 1.1 * max(T)]`` matching
            ``pysurvival``; ``"quantile"`` places knots at interior empirical event quantiles.
        l2_smooth: Temporal smoothness penalty weight on adjacent output time bins (``1e-2`` in ``pysurvival``).
        activation: Hidden activation (``"tanh"`` or ``"relu"``).
        dropout: Dropout probability (default ``0.2`` per ``pysurvival``).
        lr: Adam learning rate (default ``1e-4`` per ``pysurvival``).
        weight_decay: L2 regularization strength.
        epochs: Training epochs (default ``1000`` per ``pysurvival``).
        batch_size: Mini-batch size, or ``None`` for full-batch optimization (default ``None``).
        device: Target PyTorch device.
        seed: Random seed.
    """

    def __init__(
        self,
        *,
        hidden_units: int = 150,
        bins: int = 150,
        binning: Literal["uniform", "quantile"] = "uniform",
        l2_smooth: float = 1e-2,
        activation: Literal["tanh", "relu"] = "tanh",
        dropout: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 1000,
        batch_size: int | None = None,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        if bins < _MIN_BINS:
            msg = f"bins must be >= {_MIN_BINS}, got {bins}"
            raise ValueError(msg)
        if batch_size is not None and batch_size < 1:
            msg = f"batch_size must be >= 1 or None, got {batch_size}"
            raise ValueError(msg)
        if binning not in {"uniform", "quantile"}:
            msg = f"binning must be 'uniform' or 'quantile', got {binning!r}"
            raise ValueError(msg)
        self.hidden_units = hidden_units
        self.bins = bins
        self.binning: Literal["uniform", "quantile"] = binning
        self.l2_smooth = l2_smooth
        self.activation: Literal["tanh", "relu"] = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.seed = seed

        self._scaler = StandardScaler()
        self._net: _MTLRNet | None = None
        self._time_bins: NDArray[np.float64] = np.empty(0, dtype=np.float64)

    @property
    @override
    def name(self) -> str:
        return "NeuralMTLR"

    @property
    @override
    def times(self) -> NDArray[np.float64]:
        if self._net is None:
            msg = "NeuralMTLRModel must be fitted before accessing times"
            raise RuntimeError(msg)
        return self._time_bins

    def _build_time_bins(self, *, time: NDArray[np.float64], event: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Construct strictly increasing discrete time-interval upper bounds."""
        if self.binning == "uniform":
            max_t = max(float(time.max()), 1e-3) * _UNIFORM_GRID_EXPANSION
            return np.linspace(0.0, max_t, num=self.bins + 1, dtype=np.float64)[1:]

        ref_times = time[event] if np.any(event) else time
        unique_t = np.unique(ref_times)
        n_bins = min(self.bins, max(_MIN_BINS, len(unique_t)))
        quantiles = np.linspace(0.0, 1.0, num=n_bins + 2)[1:-1]
        edges = np.unique(np.quantile(ref_times, quantiles))
        if len(edges) < _MIN_BINS:
            t_min, t_max = float(time.min()), float(time.max())
            if np.isclose(t_min, t_max):
                t_max = t_min + 1.0
            edges = np.linspace(t_min, t_max, num=max(_MIN_BINS, n_bins))
        return np.asarray(edges, dtype=np.float64)

    @override
    def fit(self, x: NDArray[np.floating], /, *, time: NDArray[np.floating], event: NDArray[np.bool_]) -> Self:
        seed_everything(self.seed)
        x_scaled = self._scaler.fit_transform(np.asarray(x, dtype=np.float64)).astype(np.float32)
        t_arr = np.asarray(time, dtype=np.float64).reshape(-1)
        e_arr = np.asarray(event, dtype=bool).reshape(-1)

        self._time_bins = self._build_time_bins(time=t_arr, event=e_arr)
        n_bins = len(self._time_bins)

        bin_indices = np.clip(np.searchsorted(self._time_bins, t_arr, side="right").astype(np.int64), 0, n_bins)

        self._net = _MTLRNet(
            in_features=x_scaled.shape[1],
            hidden_units=self.hidden_units,
            n_bins=n_bins,
            activation=self.activation,
            dropout=self.dropout,
        ).to(self.device)

        dataset = TensorDataset(torch.from_numpy(x_scaled), torch.from_numpy(bin_indices), torch.from_numpy(e_arr))
        eff_batch = len(dataset) if self.batch_size is None else min(self.batch_size, len(dataset))
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        loader = DataLoader(dataset, batch_size=eff_batch, shuffle=self.batch_size is not None, generator=gen)

        criterion = MTLRLoss()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self._net.train()
        n_total = float(max(1, len(dataset)))
        for _ in range(self.epochs):
            for bx, bb, be in loader:
                optimizer.zero_grad()
                logits: Tensor = self._net(bx.to(self.device))
                loss: Tensor = criterion(logits, bin_indices=bb.to(self.device), event=be.to(self.device))
                if self.l2_smooth > 0.0:
                    loss = loss + (self.l2_smooth / n_total) * self._net.temporal_smoothness_penalty()
                torch.autograd.backward(loss)
                optimizer.step()

        return self

    @override
    def predict_survival(
        self, x: NDArray[np.floating], /, *, times: NDArray[np.floating] | None = None
    ) -> NDArray[np.float64]:
        if self._net is None:
            msg = "NeuralMTLRModel must be fitted before calling predict methods"
            raise RuntimeError(msg)

        x_scaled = self._scaler.transform(np.asarray(x, dtype=np.float64)).astype(np.float32)
        self._net.eval()
        with torch.no_grad():
            logits: Tensor = self._net(torch.from_numpy(x_scaled).to(self.device))
            scores = mtlr_cumulative_scores(logits)
            pmf = np.asarray(F.softmax(scores, dim=1).cpu().numpy(), dtype=np.float64)

        surv_at_bins = np.clip(1.0 - np.cumsum(pmf[:, :-1], axis=1), 0.0, 1.0)
        if times is None:
            return surv_at_bins

        query_t = np.asarray(times, dtype=np.float64).reshape(-1)
        t_knots = np.concatenate([[0.0], self._time_bins])
        out = np.empty((surv_at_bins.shape[0], len(query_t)), dtype=np.float64)
        for i in range(surv_at_bins.shape[0]):
            s_knots = np.concatenate([[1.0], surv_at_bins[i]])
            out[i] = np.interp(query_t, t_knots, s_knots, left=1.0, right=float(s_knots[-1]))
        return np.clip(out, 0.0, 1.0)

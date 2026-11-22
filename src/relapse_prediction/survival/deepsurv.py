"""PyTorch DeepSurv (NonLinearCoxPH) survival model with Breslow baseline hazard.

Modern replacement for ``pysurvival.models.semi_parametric.NonLinearCoxPHModel`` matching
the architecture and hyperparameters used in Fekas et al. (IEEE ICASSP 2023):
1-hidden-layer ``tanh`` MLP (default 150 units) trained with the negative Cox partial
log-likelihood.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self, override

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from relapse_prediction.losses.survival import CoxPHLoss
from relapse_prediction.utils.seed import seed_everything

from .base import BaseSurvivalModel, build_survival_mlp

if TYPE_CHECKING:
    from numpy.typing import NDArray

_LOG_RISK_CLIP = 20.0
_MIN_DENOM_FLOOR = 1e-12
_MIN_COX_BATCH_SIZE = 2


class _DeepSurvNet(nn.Module):
    """Feed-forward log-risk network for DeepSurv."""

    def __init__(
        self, *, in_features: int, hidden_units: int, activation: Literal["tanh", "relu"], dropout: float
    ) -> None:
        super().__init__()
        self.net = build_survival_mlp(
            in_features=in_features, hidden_units=hidden_units, out_features=1, activation=activation, dropout=dropout
        )

    @override
    def forward(self, x: Tensor, /) -> Tensor:
        out: Tensor = self.net(x).squeeze(-1)
        return out


class DeepSurvModel(BaseSurvivalModel):
    """DeepSurv nonlinear Cox proportional hazards survival model.

    Args:
        hidden_units: Hidden layer width (default ``150``, matching paper notebooks).
        activation: Hidden activation (``"tanh"`` in paper notebooks, or ``"relu"``).
        dropout: Dropout probability in the hidden layer (default ``0.2`` per ``pysurvival``).
        lr: Adam learning rate (default ``1e-4`` per ``pysurvival``).
        weight_decay: L2 regularization strength.
        epochs: Training epochs (default ``1000`` per ``pysurvival``).
        batch_size: Mini-batch size, or ``None`` for full-batch optimization (default ``None``,
            matching ``pysurvival``'s exact full-risk-set Cox optimization).
        device: Target PyTorch device (``"cpu"``, ``"cuda"``, or ``torch.device``).
        seed: Random seed for reproducible initialization.
    """

    def __init__(
        self,
        *,
        hidden_units: int = 150,
        activation: Literal["tanh", "relu"] = "tanh",
        dropout: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 1000,
        batch_size: int | None = None,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        if batch_size is not None and batch_size < _MIN_COX_BATCH_SIZE:
            msg = (
                f"batch_size must be >= {_MIN_COX_BATCH_SIZE} or None for CoxPH risk-set comparisons, got {batch_size}"
            )
            raise ValueError(msg)
        self.hidden_units = hidden_units
        self.activation: Literal["tanh", "relu"] = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.seed = seed

        self._scaler = StandardScaler()
        self._net: _DeepSurvNet | None = None
        self._times: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._baseline_cum_hazard: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._risk_offset: float = 0.0

    @property
    @override
    def name(self) -> str:
        return "DeepSurv"

    @property
    @override
    def times(self) -> NDArray[np.float64]:
        if self._net is None:
            msg = "DeepSurvModel must be fitted before accessing times"
            raise RuntimeError(msg)
        return self._times

    def _fit_breslow_baseline(
        self, log_risk: NDArray[np.float64], /, *, time: NDArray[np.float64], event: NDArray[np.bool_]
    ) -> None:
        """Estimate the Breslow cumulative baseline hazard ``H_0(t)`` on training data."""
        self._risk_offset = float(np.mean(log_risk))
        exp_risk = np.exp(np.clip(log_risk - self._risk_offset, -_LOG_RISK_CLIP, _LOG_RISK_CLIP))

        event_times = np.unique(time[event])
        if len(event_times) == 0:
            self._times = np.unique(time)
            self._baseline_cum_hazard = np.zeros_like(self._times, dtype=np.float64)
            return

        increments = np.empty(len(event_times), dtype=np.float64)
        for idx, t_k in enumerate(event_times):
            d_k = float(np.sum(event & (time == t_k)))
            denom = float(np.sum(exp_risk[time >= t_k]))
            increments[idx] = d_k / max(denom, _MIN_DENOM_FLOOR)

        self._times = event_times
        self._baseline_cum_hazard = np.cumsum(increments)

    @override
    def fit(self, x: NDArray[np.floating], /, *, time: NDArray[np.floating], event: NDArray[np.bool_]) -> Self:
        seed_everything(self.seed)
        x_scaled = self._scaler.fit_transform(np.asarray(x, dtype=np.float64)).astype(np.float32)
        t_arr = np.asarray(time, dtype=np.float64).reshape(-1)
        e_arr = np.asarray(event, dtype=bool).reshape(-1)

        self._net = _DeepSurvNet(
            in_features=x_scaled.shape[1],
            hidden_units=self.hidden_units,
            activation=self.activation,
            dropout=self.dropout,
        ).to(self.device)

        dataset = TensorDataset(
            torch.from_numpy(x_scaled),
            torch.from_numpy(t_arr.astype(np.float32)),
            torch.from_numpy(e_arr.astype(np.float32)),
        )
        eff_batch = len(dataset) if self.batch_size is None else min(self.batch_size, len(dataset))
        # Avoid singleton remainder batches that yield zero CoxPH gradient.
        drop_last = len(dataset) > eff_batch and (len(dataset) % eff_batch == 1)
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        loader = DataLoader(
            dataset, batch_size=eff_batch, shuffle=self.batch_size is not None, drop_last=drop_last, generator=gen
        )

        criterion = CoxPHLoss()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self._net.train()
        for _ in range(self.epochs):
            for bx, bt, be in loader:
                optimizer.zero_grad()
                pred: Tensor = self._net(bx.to(self.device))
                loss: Tensor = criterion(pred, time=bt.to(self.device), event=be.to(self.device))
                if loss.requires_grad:
                    torch.autograd.backward(loss)
                    optimizer.step()

        self._net.eval()
        with torch.no_grad():
            raw_risk: Tensor = self._net(torch.from_numpy(x_scaled).to(self.device))
            train_log_risk = np.asarray(raw_risk.cpu().numpy(), dtype=np.float64)
        self._fit_breslow_baseline(train_log_risk, time=t_arr, event=e_arr)
        return self

    def _predict_log_risk(self, x: NDArray[np.floating], /) -> NDArray[np.float64]:
        if self._net is None:
            msg = "DeepSurvModel must be fitted before calling predict methods"
            raise RuntimeError(msg)
        x_scaled = self._scaler.transform(np.asarray(x, dtype=np.float64)).astype(np.float32)
        self._net.eval()
        with torch.no_grad():
            pred: Tensor = self._net(torch.from_numpy(x_scaled).to(self.device))
            return np.asarray(pred.cpu().numpy(), dtype=np.float64)

    @override
    def predict_cumulative_hazard(
        self, x: NDArray[np.floating], /, *, times: NDArray[np.floating] | None = None
    ) -> NDArray[np.float64]:
        log_risk = self._predict_log_risk(x)
        exp_risk = np.exp(np.clip(log_risk - self._risk_offset, -_LOG_RISK_CLIP, _LOG_RISK_CLIP))

        if times is None:
            h0 = self._baseline_cum_hazard
        else:
            query_t = np.asarray(times, dtype=np.float64).reshape(-1)
            idx = np.searchsorted(self._times, query_t, side="right") - 1
            h0 = np.where(idx >= 0, self._baseline_cum_hazard[np.clip(idx, 0, len(self._times) - 1)], 0.0)

        return np.outer(exp_risk, h0)

    @override
    def predict_survival(
        self, x: NDArray[np.floating], /, *, times: NDArray[np.floating] | None = None
    ) -> NDArray[np.float64]:
        cum_hazard = self.predict_cumulative_hazard(x, times=times)
        return np.asarray(np.exp(-cum_hazard), dtype=np.float64)

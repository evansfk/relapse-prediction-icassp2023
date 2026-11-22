"""Unified base interface and shared neural MLP builder for survival models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Self, assert_never

import numpy as np

from torch import nn

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SURVIVAL_CLIP_EPS = 1e-8


def build_survival_mlp(
    *, in_features: int, hidden_units: int, out_features: int, activation: Literal["tanh", "relu"], dropout: float
) -> nn.Sequential:
    """Build the Xavier-initialized 1-hidden-layer MLP shared by DeepSurv and Neural MTLR."""
    if activation == "tanh":
        act: nn.Module = nn.Tanh()
    elif activation == "relu":
        act = nn.ReLU()
    else:
        assert_never(activation)
    fc1 = nn.Linear(in_features, hidden_units)
    fc2 = nn.Linear(hidden_units, out_features)
    for layer in (fc1, fc2):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    return nn.Sequential(fc1, act, nn.Dropout(dropout), fc2)


class BaseSurvivalModel(ABC):
    """Abstract base class for neural and tree-based right-censored survival models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier (e.g., ``"DeepSurv"``, ``"NeuralMTLR"``)."""

    @property
    @abstractmethod
    def times(self) -> NDArray[np.float64]:
        """Fitted model time grid ``(J,)`` over which survival and hazard functions are defined."""

    @abstractmethod
    def fit(self, x: NDArray[np.floating], /, *, time: NDArray[np.floating], event: NDArray[np.bool_]) -> Self:
        """Fit the survival model on training covariates and right-censored targets.

        Args:
            x: Covariate matrix of shape ``(N, F)``.
            time: Observed time-to-relapse or censoring time of shape ``(N,)``.
            event: Boolean event indicators (``True`` = relapse, ``False`` = right-censored).

        Returns:
            Fitted ``self``.
        """

    @abstractmethod
    def predict_survival(
        self, x: NDArray[np.floating], /, *, times: NDArray[np.floating] | None = None
    ) -> NDArray[np.float64]:
        """Predict survival probabilities ``S(t | x) = P(T >= t | x)``.

        Args:
            x: Covariate matrix of shape ``(N, F)``.
            times: Optional 1D evaluation times ``(J,)``; defaults to :attr:`times`.

        Returns:
            Array of shape ``(N, J)`` with values in ``[0, 1]``.
        """

    def predict_cumulative_hazard(
        self, x: NDArray[np.floating], /, *, times: NDArray[np.floating] | None = None
    ) -> NDArray[np.float64]:
        """Predict cumulative hazard ``H(t | x) = -log S(t | x)``.

        Args:
            x: Covariate matrix of shape ``(N, F)``.
            times: Optional 1D evaluation times ``(J,)``; defaults to :attr:`times`.

        Returns:
            Array of shape ``(N, J)`` of non-negative cumulative hazard values.
        """
        surv = np.clip(self.predict_survival(x, times=times), _SURVIVAL_CLIP_EPS, 1.0)
        return np.asarray(-np.log(surv), dtype=np.float64)

    def predict_risk(self, x: NDArray[np.floating], /) -> NDArray[np.float64]:
        """Compute scalar risk score ``r(x) = sum_j H(t_j | x)`` (paper Section 3).

        Args:
            x: Covariate matrix of shape ``(N, F)``.

        Returns:
            1D risk-score array of shape ``(N,)`` where higher values indicate higher relapse risk.
        """
        cum_hazard = self.predict_cumulative_hazard(x)
        return np.asarray(cum_hazard.sum(axis=1), dtype=np.float64)

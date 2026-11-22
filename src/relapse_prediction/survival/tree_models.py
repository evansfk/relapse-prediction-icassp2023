"""Tree-based right-censored survival models powered by ``scikit-survival``.

Provides modern, maintained Python 3.12+ wrappers matching the two ensemble survival
models evaluated in Fekas et al. (IEEE ICASSP 2023):
- :class:`ExtraSurvivalTreesModel` (``sksurv.ensemble.ExtraSurvivalTrees``)
- :class:`RandomSurvivalForestModel` (``sksurv.ensemble.RandomSurvivalForest``),
  serving as the maintained modern counterpart to ``pysurvival``'s abandoned
  ``ConditionalSurvivalForestModel`` (documented explicitly in README and docstrings).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self, override

import numpy as np

from sksurv.ensemble import ExtraSurvivalTrees, RandomSurvivalForest

from relapse_prediction.evaluation.survival_metrics import to_structured_survival

from .base import BaseSurvivalModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class _SkSurvForestProtocol(Protocol):
    unique_times_: NDArray[np.float64]

    def fit(self, x: NDArray[np.float64], y: NDArray[np.void]) -> _SkSurvForestProtocol: ...
    def predict_survival_function(
        self, x: NDArray[np.float64]
    ) -> list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]: ...
    def predict_cumulative_hazard_function(
        self, x: NDArray[np.float64]
    ) -> list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]: ...


class _SkSurvForestWrapper(BaseSurvivalModel):
    """Shared wrapper around ``scikit-survival`` forest estimators."""

    def __init__(self, estimator: _SkSurvForestProtocol, /, *, model_name: str) -> None:
        self._estimator = estimator
        self._name = model_name
        self._times: NDArray[np.float64] = np.empty(0, dtype=np.float64)

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def times(self) -> NDArray[np.float64]:
        if len(self._times) == 0:
            msg = f"{self._name} must be fitted before accessing times"
            raise RuntimeError(msg)
        return self._times

    def _prepare_eval_inputs(
        self, x: NDArray[np.floating[Any]], /, *, times: NDArray[np.floating[Any]] | None, caller: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Validate fitted state and return ``(x_arr, query_t, eval_t, before_first_mask)``."""
        if len(self._times) == 0:
            msg = f"{self._name} must be fitted before calling {caller}"
            raise RuntimeError(msg)
        x_arr = np.asarray(x, dtype=np.float64)
        query_t = self._times if times is None else np.asarray(times, dtype=np.float64).reshape(-1)
        before_first = query_t < self._times[0]
        eval_t = np.clip(query_t, float(self._times[0]), float(self._times[-1]))
        return x_arr, query_t, eval_t, before_first

    @override
    def fit(
        self, x: NDArray[np.floating[Any]], /, *, time: NDArray[np.floating[Any]], event: NDArray[np.bool_]
    ) -> Self:
        x_arr = np.asarray(x, dtype=np.float64)
        y_struct = to_structured_survival(time=time, event=event)
        self._estimator.fit(x_arr, y_struct)
        self._times = np.asarray(self._estimator.unique_times_, dtype=np.float64)
        return self

    @override
    def predict_survival(
        self, x: NDArray[np.floating[Any]], /, *, times: NDArray[np.floating[Any]] | None = None
    ) -> NDArray[np.float64]:
        x_arr, _, eval_t, before_first = self._prepare_eval_inputs(x, times=times, caller="predict_survival")
        surv_fns = self._estimator.predict_survival_function(x_arr)
        curves = np.clip(np.vstack([fn(eval_t) for fn in surv_fns]).astype(np.float64), 0.0, 1.0)
        # Preserve S(t | x) = 1.0 strictly prior to the earliest observed event time t_0.
        if np.any(before_first):
            curves[:, before_first] = 1.0
        return curves

    @override
    def predict_cumulative_hazard(
        self, x: NDArray[np.floating[Any]], /, *, times: NDArray[np.floating[Any]] | None = None
    ) -> NDArray[np.float64]:
        x_arr, _, eval_t, before_first = self._prepare_eval_inputs(x, times=times, caller="predict_cumulative_hazard")
        chf_fns = self._estimator.predict_cumulative_hazard_function(x_arr)
        curves = np.vstack([fn(eval_t) for fn in chf_fns]).astype(np.float64)
        # Preserve H(t | x) = 0.0 strictly prior to the earliest observed event time t_0.
        if np.any(before_first):
            curves[:, before_first] = 0.0
        return curves


class ExtraSurvivalTreesModel(_SkSurvForestWrapper):
    """Extremely Randomized Survival Trees (``num_trees=200`` in the paper).

    Defaults match ``pysurvival.models.survival_forest.ExtraSurvivalTreesModel``:
    ``n_estimators=200, max_depth=5, min_samples_leaf=10, max_samples=0.63``.

    Args:
        n_estimators: Number of trees in the ensemble (default ``200``).
        max_depth: Maximum tree depth (default ``5`` per ``pysurvival``; ``None`` for unlimited).
        min_samples_leaf: Minimum samples per leaf node (default ``10`` per ``pysurvival``).
        max_samples: Bootstrap sample fraction or count (default ``0.63`` per ``pysurvival``).
        n_jobs: Parallel jobs passed to ``scikit-survival``.
        seed: Random seed.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = 5,
        min_samples_leaf: int = 10,
        max_samples: float | None = 0.63,
        n_jobs: int = -1,
        seed: int = 42,
    ) -> None:
        estimator = ExtraSurvivalTrees(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=max(6, 2 * min_samples_leaf),
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=seed,
        )
        super().__init__(estimator, model_name="ExtraSurvivalTrees")


class RandomSurvivalForestModel(_SkSurvForestWrapper):
    """Random Survival Forest (maintained modern counterpart to ``pysurvival``'s C-SurvivalForest).

    Note:
        The original ICASSP 2023 notebooks used ``pysurvival.models.survival_forest.ConditionalSurvivalForestModel``
        (abandoned since 2019, incompatible with Python >= 3.8). ``scikit-survival``'s
        ``RandomSurvivalForest`` provides the closest actively maintained survival forest
        implementation in Python 3.12+, configured by default with ``pysurvival``'s
        ``max_depth=5, min_samples_leaf=10, max_samples=0.63``.

    Args:
        n_estimators: Number of trees in the ensemble (default ``200``).
        max_depth: Maximum tree depth (default ``5`` per ``pysurvival``; ``None`` for unlimited).
        min_samples_leaf: Minimum samples per leaf node (default ``10`` per ``pysurvival``).
        max_samples: Bootstrap sample fraction or count (default ``0.63`` per ``pysurvival``).
        n_jobs: Parallel jobs passed to ``scikit-survival``.
        seed: Random seed.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = 5,
        min_samples_leaf: int = 10,
        max_samples: float | None = 0.63,
        n_jobs: int = -1,
        seed: int = 42,
    ) -> None:
        estimator = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=max(6, 2 * min_samples_leaf),
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=seed,
        )
        super().__init__(estimator, model_name="RandomSurvivalForest")

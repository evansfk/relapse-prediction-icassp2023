"""Survival analysis evaluation metrics (C-index, Brier Score, IBS, -log(IBS)) and risk grouping.

Implements the Section 3 equations from Fekas et al. (IEEE ICASSP 2023) using
``scikit-survival``'s IPCW estimators and provides 3-tier risk stratification (Fig. 4).
All survival APIs enforce consistent ``(time, event)`` keyword arguments to eliminate
positional-argument swap hazards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from sksurv.metrics import brier_score, concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from relapse_prediction.survival.base import BaseSurvivalModel

_EPS_TIME = 1e-6
_MIN_IBS_FLOOR = 1e-12


@dataclass(frozen=True)
class SurvivalMetrics:
    """Discrimination and calibration summary for a survival model."""

    c_index: float
    ibs: float
    neg_log_ibs: float
    concordant_pairs: int
    discordant_pairs: int
    tied_risk_pairs: int


def to_structured_survival(*, time: NDArray[np.floating[Any]], event: NDArray[np.bool_]) -> NDArray[np.void]:
    """Convert ``(time, event)`` arrays into a ``scikit-survival`` structured array."""
    structured: NDArray[np.void] = Surv.from_arrays(
        event=np.asarray(event, dtype=bool), time=np.asarray(time, dtype=np.float64)
    )
    return structured


def compute_c_index(
    *, time: NDArray[np.floating[Any]], event: NDArray[np.bool_], risk_scores: NDArray[np.floating[Any]]
) -> tuple[float, int, int, int]:
    """Compute Harrell's concordance index (C-index) for right-censored data.

    Matches the formula printed in Section 3: higher ``risk_scores`` correspond to shorter
    predicted time-to-relapse.

    Args:
        time: Observed or censored times ``(N,)``.
        event: Boolean event indicators ``(N,)`` (``True`` = relapse observed).
        risk_scores: Predicted scalar risk scores ``r(x)`` of shape ``(N,)``.

    Returns:
        Tuple ``(c_index, concordant, discordant, tied_risk)``.
    """
    c_idx, concordant, discordant, tied_risk, _ = concordance_index_censored(
        np.asarray(event, dtype=bool), np.asarray(time, dtype=np.float64), np.asarray(risk_scores, dtype=np.float64)
    )
    return float(c_idx), int(concordant), int(discordant), int(tied_risk)


def _valid_eval_times(
    time_train: NDArray[np.floating[Any]],
    time_test: NDArray[np.floating[Any]],
    /,
    *,
    num_points: int,
    ibs_mode: Literal["paper", "quantile"] = "paper",
) -> NDArray[np.float64]:
    """Construct an evaluation time grid strictly inside the IPCW support of both train and test."""
    t_tr = np.asarray(time_train, dtype=np.float64)
    t_te = np.asarray(time_test, dtype=np.float64)
    strict_low = max(float(t_te.min()), float(t_tr.min())) + _EPS_TIME
    strict_high = min(float(t_te.max()), float(t_tr.max())) - _EPS_TIME

    if strict_high <= strict_low:
        msg = (
            f"No overlapping interior time support between train [{t_tr.min():.4g}, {t_tr.max():.4g}] "
            f"and test [{t_te.min():.4g}, {t_te.max():.4g}]"
        )
        raise ValueError(msg)

    if ibs_mode == "paper":
        return np.linspace(strict_low, strict_high, num=num_points, dtype=np.float64)

    t_min = max(float(np.quantile(t_te, 0.05)), strict_low)
    t_max = min(float(np.quantile(t_te, 0.95)), strict_high)
    if t_max <= t_min:
        t_min = strict_low
        t_max = strict_high
    return np.linspace(t_min, t_max, num=num_points, dtype=np.float64)


def compute_ibs_and_brier(
    *,
    time_train: NDArray[np.floating[Any]],
    event_train: NDArray[np.bool_],
    time_test: NDArray[np.floating[Any]],
    event_test: NDArray[np.bool_],
    survival_probs: NDArray[np.floating[Any]],
    eval_times: NDArray[np.floating[Any]],
    ibs_mode: Literal["paper", "quantile"] = "paper",
) -> tuple[float, float, NDArray[np.float64]]:
    """Compute IPCW Brier Scores, Integrated Brier Score (IBS), and ``-log(IBS)``.

    Args:
        time_train: Training event/censoring times.
        event_train: Training event indicators (used to fit censoring estimator ``G(t)``).
        time_test: Test event/censoring times.
        event_test: Test event indicators.
        survival_probs: Predicted survival probabilities ``S(t | x)`` of shape ``(N_test, len(eval_times))``.
        eval_times: Strictly increasing evaluation time points.
        ibs_mode: ``"paper"`` integrates ``(1 / t_max) * int_0^{t_max} BS(t) dt`` matching
            Paper Section 3 and ``pysurvival`` (anchoring ``BS(0) = 0`` at ``t = 0``);
            ``"quantile"`` normalizes over ``eval_times[-1] - eval_times[0]``.

    Returns:
        Tuple ``(ibs, neg_log_ibs, brier_curve)``.
    """
    t_tr = np.asarray(time_train, dtype=np.float64)
    t_te = np.asarray(time_test, dtype=np.float64)
    e_te = np.asarray(event_test, dtype=bool)
    times_arr = np.asarray(eval_times, dtype=np.float64)

    # Administratively censor test subjects observed at/beyond max(time_train):
    # sksurv's CensoringDistributionEstimator requires every test_time < max(time_train),
    # while for any eval time t < max(time_train) a subject with T_i >= max(time_train) is
    # identically still event-free (1_{T_i > t} == 1, 1_{T_i <= t} == 0).
    max_tr = float(np.max(t_tr))
    if float(times_arr.max()) < max_tr:
        exceeds = t_te >= max_tr
        if np.any(exceeds):
            admin_censor_time = 0.5 * (float(times_arr.max()) + max_tr)
            t_te = np.where(exceeds, admin_censor_time, t_te)
            e_te = e_te & ~exceeds

    y_tr = to_structured_survival(time=t_tr, event=event_train)
    y_te = to_structured_survival(time=t_te, event=e_te)
    surv_clipped = np.clip(np.asarray(survival_probs, dtype=np.float64), 0.0, 1.0)

    _, scores_raw = brier_score(y_tr, y_te, surv_clipped, times_arr)
    scores = np.asarray(scores_raw, dtype=np.float64)
    if ibs_mode == "paper":
        t_with_zero = np.concatenate([[0.0], times_arr])
        bs_with_zero = np.concatenate([[0.0], scores])
        ibs_val = float(np.trapezoid(bs_with_zero, t_with_zero) / max(float(times_arr[-1]), _EPS_TIME))
    else:
        ibs_val = float(integrated_brier_score(y_tr, y_te, surv_clipped, times_arr))
    neg_log_ibs = float(-np.log(max(ibs_val, _MIN_IBS_FLOOR)))
    return ibs_val, neg_log_ibs, scores


def evaluate_survival_model(
    model: BaseSurvivalModel,
    x_test: NDArray[np.floating[Any]],
    /,
    *,
    time_train: NDArray[np.floating[Any]],
    event_train: NDArray[np.bool_],
    time_test: NDArray[np.floating[Any]],
    event_test: NDArray[np.bool_],
    num_time_points: int = 50,
    ibs_mode: Literal["paper", "quantile"] = "paper",
) -> SurvivalMetrics:
    """Evaluate a fitted survival model on C-index, IBS, and ``-log(IBS)``.

    Args:
        model: Fitted :class:`BaseSurvivalModel`.
        x_test: Test feature matrix ``(N_test, F)``.
        time_train: Training survival times (for IPCW censoring weights).
        event_train: Training event flags.
        time_test: Test survival times.
        event_test: Test event flags.
        num_time_points: Number of time points used to integrate the Brier score.
        ibs_mode: Integration mode (``"paper"`` ``[0, t_max] / t_max`` or ``"quantile"``).

    Returns:
        :class:`SurvivalMetrics` summary.
    """
    risk_scores = model.predict_risk(x_test)
    c_idx, concordant, discordant, tied_risk = compute_c_index(
        time=time_test, event=event_test, risk_scores=risk_scores
    )

    eval_times = _valid_eval_times(time_train, time_test, num_points=num_time_points, ibs_mode=ibs_mode)
    surv_probs = model.predict_survival(x_test, times=eval_times)
    ibs_val, neg_log_ibs, _ = compute_ibs_and_brier(
        time_train=time_train,
        event_train=event_train,
        time_test=time_test,
        event_test=event_test,
        survival_probs=surv_probs,
        eval_times=eval_times,
        ibs_mode=ibs_mode,
    )
    return SurvivalMetrics(
        c_index=c_idx,
        ibs=ibs_val,
        neg_log_ibs=neg_log_ibs,
        concordant_pairs=concordant,
        discordant_pairs=discordant,
        tied_risk_pairs=tied_risk,
    )


def stratify_risk_groups(
    risk_scores: NDArray[np.floating[Any]],
    /,
    *,
    quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    cutoffs: tuple[float, float] | None = None,
) -> tuple[NDArray[np.str_], tuple[float, float]]:
    """Partition scalar risk scores into ``low``, ``medium``, and ``high`` risk tiers (Fig. 4).

    Pass ``cutoffs`` fitted on training-set risk scores to avoid test-set threshold leakage.

    Args:
        risk_scores: 1D array of predicted risk scores ``(N,)``.
        quantiles: ``(q_low, q_high)`` thresholds in ``(0, 1)`` (defaults to tertiles),
            used when ``cutoffs`` is ``None``.
        cutoffs: Optional pre-fitted ``(low_cutoff, high_cutoff)`` thresholds.

    Returns:
        Tuple ``(labels, (low_cutoff, high_cutoff))`` where ``labels`` has values in
        ``{"low", "medium", "high"}``.
    """
    scores = np.asarray(risk_scores, dtype=np.float64).reshape(-1)
    if cutoffs is not None:
        low_cut, high_cut = float(cutoffs[0]), float(cutoffs[1])
        if high_cut < low_cut:
            msg = f"Expected low_cut <= high_cut, got {cutoffs}"
            raise ValueError(msg)
    else:
        q_low, q_high = quantiles
        if not (0.0 < q_low < q_high < 1.0):
            msg = f"Expected 0 < q_low < q_high < 1, got {quantiles}"
            raise ValueError(msg)
        low_cut = float(np.quantile(scores, q_low))
        high_cut = float(np.quantile(scores, q_high))

    if np.isclose(low_cut, high_cut):
        labels = np.where(scores < low_cut, "low", np.where(scores > high_cut, "high", "medium"))
    else:
        labels = np.where(scores <= low_cut, "low", np.where(scores <= high_cut, "medium", "high"))
    return labels, (low_cut, high_cut)

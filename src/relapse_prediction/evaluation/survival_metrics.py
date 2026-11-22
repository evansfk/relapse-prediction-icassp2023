"""Survival analysis evaluation metrics (C-index, Brier Score, IBS, -log(IBS)) and risk grouping.

Implements the Section 3 equations from Fekas et al. (IEEE ICASSP 2023) using
``scikit-survival``'s IPCW estimators and provides 3-tier risk stratification (Fig. 4).
All survival APIs enforce consistent ``(time, event)`` keyword arguments to eliminate
positional-argument swap hazards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np

from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from relapse_prediction.survival.base import BaseSurvivalModel

_EPS_TIME = 1e-6
_MIN_IBS_FLOOR = 1e-12


@dataclass(frozen=True, slots=True, kw_only=True)
class SurvivalMetrics:
    """Discrimination and calibration summary for a survival model."""

    c_index: float
    ibs: float
    neg_log_ibs: float
    concordant_pairs: int
    discordant_pairs: int
    tied_risk_pairs: int


def to_structured_survival(*, time: NDArray[np.floating], event: NDArray[np.bool_]) -> NDArray[np.void]:
    """Convert ``(time, event)`` arrays into a ``scikit-survival`` structured array."""
    structured: NDArray[np.void] = Surv.from_arrays(
        event=np.asarray(event, dtype=bool), time=np.asarray(time, dtype=np.float64)
    )
    return structured


def compute_c_index(
    *, time: NDArray[np.floating], event: NDArray[np.bool_], risk_scores: NDArray[np.floating]
) -> tuple[float, int, int, int]:
    """Compute Harrell's concordance index (C-index) for right-censored data.

    Matches the unweighted Harrell concordance formula printed in Section 3: higher
    ``risk_scores`` correspond to shorter predicted time-to-relapse (whereas ``pysurvival``'s
    C++ extension cited via Uno et al. [30] weighted pairs by test-set IPCW). Returns ``0.5``
    if ``event`` contains zero comparable pairs.

    Args:
        time: Observed or censored times ``(N,)``.
        event: Boolean event indicators ``(N,)`` (``True`` = relapse observed).
        risk_scores: Predicted scalar risk scores ``r(x)`` of shape ``(N,)``.

    Returns:
        Tuple ``(c_index, concordant, discordant, tied_risk)``.
    """
    e_arr = np.asarray(event, dtype=bool)
    t_arr = np.asarray(time, dtype=np.float64)
    r_arr = np.asarray(risk_scores, dtype=np.float64)
    if not bool(np.isfinite(r_arr).all()):
        msg = "risk_scores must be finite"
        raise ValueError(msg)
    if not np.any(e_arr):
        return 0.5, 0, 0, 0
    try:
        c_idx, concordant, discordant, tied_risk, _ = concordance_index_censored(e_arr, t_arr, r_arr)
    except NoComparablePairException:
        return 0.5, 0, 0, 0
    return float(c_idx), int(concordant), int(discordant), int(tied_risk)


def _valid_eval_times(
    time_train: NDArray[np.floating],
    time_test: NDArray[np.floating],
    /,
    *,
    num_points: int,
    ibs_mode: Literal["paper", "quantile"],
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
    time_train: NDArray[np.floating],
    event_train: NDArray[np.bool_],
    time_test: NDArray[np.floating],
    event_test: NDArray[np.bool_],
    survival_probs: NDArray[np.floating],
    eval_times: NDArray[np.floating],
    ibs_mode: Literal["paper", "quantile"] = "paper",
) -> tuple[float, float, NDArray[np.float64]]:
    """Compute Graf et al. (1999) IPCW Brier Scores, Integrated Brier Score (IBS), and ``-log(IBS)``.

    Args:
        time_train: Training event/censoring times.
        event_train: Training event indicators (used to fit censoring estimator ``G(t)``).
        time_test: Test event/censoring times.
        event_test: Test event indicators.
        survival_probs: Predicted survival probabilities ``S(t | x)`` of shape ``(N_test, len(eval_times))``.
        eval_times: Strictly increasing evaluation time points ``< max(time_train)``.
        ibs_mode: ``"paper"`` integrates ``(1 / t_max) * int_0^{t_max} BS(t) dt`` matching
            Paper Section 3 and ``pysurvival`` (anchoring ``BS(0) = 0`` at ``t = 0``);
            ``"quantile"`` normalizes over ``eval_times[-1] - eval_times[0]``.

    Returns:
        Tuple ``(ibs, neg_log_ibs, brier_curve)``.
    """
    t_tr = np.asarray(time_train, dtype=np.float64).reshape(-1)
    t_te = np.asarray(time_test, dtype=np.float64).reshape(-1)
    e_te = np.asarray(event_test, dtype=bool).reshape(-1)
    times_arr = np.asarray(eval_times, dtype=np.float64).reshape(-1)
    surv_clipped = np.clip(np.asarray(survival_probs, dtype=np.float64), 0.0, 1.0)

    if len(times_arr) == 0 or float(times_arr.min()) < 0.0:
        msg = "eval_times must be non-empty and non-negative"
        raise ValueError(msg)
    max_tr = float(np.max(t_tr))
    if float(times_arr.max()) >= max_tr:
        msg = f"eval_times must lie strictly below max(time_train)={max_tr}, got max={times_arr.max()}"
        raise ValueError(msg)
    if surv_clipped.shape != (len(t_te), len(times_arr)):
        msg = f"Expected survival_probs shape {(len(t_te), len(times_arr))}, got {surv_clipped.shape}"
        raise ValueError(msg)

    y_tr = to_structured_survival(time=t_tr, event=event_train)
    cens_est = CensoringDistributionEstimator().fit(y_tr)

    # Evaluate G_hat(t) at every eval_time and G_hat(T_i) at uncensored test event times < max(t_tr).
    g_eval = np.maximum(np.asarray(cens_est.predict_proba(times_arr), dtype=np.float64), _MIN_IBS_FLOOR)
    obs_event_mask = e_te & (t_te < max_tr)
    g_subj = np.ones(len(t_te), dtype=np.float64)
    if np.any(obs_event_mask):
        g_subj[obs_event_mask] = np.maximum(
            np.asarray(cens_est.predict_proba(t_te[obs_event_mask]), dtype=np.float64), _MIN_IBS_FLOOR
        )

    # Weight uncensored events by 1 / G_hat(T_i) and still-at-risk subjects by 1 / G_hat(t_k) (Graf et al., 1999).
    event_by_t = (t_te[:, np.newaxis] <= times_arr[np.newaxis, :]) & e_te[:, np.newaxis]
    at_risk_after_t = t_te[:, np.newaxis] > times_arr[np.newaxis, :]
    term_event = np.where(event_by_t, (surv_clipped**2) / g_subj[:, np.newaxis], 0.0)
    term_surv = np.where(at_risk_after_t, ((1.0 - surv_clipped) ** 2) / g_eval[np.newaxis, :], 0.0)
    scores = np.asarray(np.mean(term_event + term_surv, axis=0), dtype=np.float64)

    if ibs_mode == "paper":
        t_with_zero = np.concatenate([[0.0], times_arr])
        bs_with_zero = np.concatenate([[0.0], scores])
        ibs_val = float(np.trapezoid(bs_with_zero, t_with_zero) / max(float(times_arr[-1]), _EPS_TIME))
    elif ibs_mode == "quantile":
        span = float(times_arr[-1] - times_arr[0])
        ibs_val = float(np.trapezoid(scores, times_arr) / span) if span > _EPS_TIME else float(np.mean(scores))
    else:
        assert_never(ibs_mode)
    neg_log_ibs = float(-np.log(max(ibs_val, _MIN_IBS_FLOOR)))
    return ibs_val, neg_log_ibs, scores


def evaluate_survival_model(
    model: BaseSurvivalModel,
    x_test: NDArray[np.floating],
    /,
    *,
    time_train: NDArray[np.floating],
    event_train: NDArray[np.bool_],
    time_test: NDArray[np.floating],
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
    risk_scores: NDArray[np.floating],
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
    if not bool(np.isfinite(scores).all()):
        msg = "risk_scores must contain only finite values"
        raise ValueError(msg)
    if cutoffs is not None:
        low_cut, high_cut = float(cutoffs[0]), float(cutoffs[1])
        if not np.isfinite(low_cut) or not np.isfinite(high_cut) or high_cut < low_cut:
            msg = f"Expected finite low_cut <= high_cut, got {cutoffs}"
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

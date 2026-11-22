"""Survival analysis loss functions for DeepSurv and Neural MTLR."""

from __future__ import annotations

from typing import override

import torch

from torch import Tensor, nn

_EXPECTED_LOGITS_NDIM = 2


class CoxPHLoss(nn.Module):
    """Negative Cox proportional hazards partial log-likelihood with exact Breslow ties.

    Used to train DeepSurv (Katzman et al., 2018; ``NonLinearCoxPHModel`` in the paper).
    All subjects tied at time ``t_k`` share the exact full risk-set denominator
    ``sum_{j : t_j >= t_k} exp(r_j)``, making the loss order-invariant under batch shuffling.
    """

    @override
    def forward(self, log_hazard: Tensor, /, *, time: Tensor, event: Tensor) -> Tensor:
        """Compute average negative Breslow partial log-likelihood over uncensored events.

        Args:
            log_hazard: Predicted log-risk scores of shape ``(B,)`` or ``(B, 1)``.
            time: Observed survival or censoring times of shape ``(B,)``.
            event: Binary event indicators (``1`` = relapse observed, ``0`` = right-censored),
                shape ``(B,)``.

        Returns:
            Scalar loss tensor (``0.0`` if no uncensored events appear in the batch).
        """
        risk = log_hazard.reshape(-1)
        t = time.reshape(-1)
        e = event.reshape(-1).to(dtype=risk.dtype)

        n_events = e.sum()
        if n_events <= 0:
            return risk.sum() * 0.0

        order = torch.argsort(t, descending=True)
        t_sorted = t[order]
        risk_sorted = risk[order]
        e_sorted = e[order]

        # Group tied times so every sample at t_k reads the prefix log-sum-exp at the
        # last index of its unique time block (exact Breslow risk set R(t_k) = {j: t_j >= t_k}).
        log_cum = torch.logcumsumexp(risk_sorted, dim=0)
        _, inv, counts = torch.unique_consecutive(t_sorted, return_inverse=True, return_counts=True)
        last_idx = torch.cumsum(counts, dim=0) - 1
        log_risk_set = log_cum[last_idx[inv]]

        partial_ll = (risk_sorted - log_risk_set) * e_sorted
        return -partial_ll.sum() / n_events


def mtlr_cumulative_scores(logits: Tensor, /) -> Tensor:
    """Map ``(B, K)`` MTLR interval logits to ``(B, K + 1)`` cumulative survival scores.

    Column ``k`` (``0 <= k < K``) equals ``sum_{j=k}^{K-1} logits[:, j]`` (event in interval
    ``k``), and column ``K`` equals ``0`` (surviving beyond the last time boundary).

    Args:
        logits: Raw network outputs of shape ``(B, K)``.

    Returns:
        Cumulative score tensor of shape ``(B, K + 1)``.

    Raises:
        ValueError: If ``logits`` is not a 2D tensor.
    """
    if logits.ndim != _EXPECTED_LOGITS_NDIM:
        msg = f"Expected 2D MTLR logits (B, K), got shape {tuple(logits.shape)}"
        raise ValueError(msg)
    suffix = torch.flip(torch.cumsum(torch.flip(logits, dims=[1]), dim=1), dims=[1])
    zeros = torch.zeros((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
    return torch.cat([suffix, zeros], dim=1)


class MTLRLoss(nn.Module):
    """Discrete-time Multi-Task Logistic Regression (Neural MTLR) negative log-likelihood.

    Implements the right-censored survival likelihood of Yu et al. (2011) and Fotso (2018):
    - Uncensored sample with event in bin ``k``: ``-log P(T in bin k | x)``
    - Right-censored sample at bin ``k``: ``-log sum_{m >= k} P(T in bin m | x)``
    """

    @override
    def forward(self, logits: Tensor, /, *, bin_indices: Tensor, event: Tensor) -> Tensor:
        """Compute mean negative log-likelihood for a batch of MTLR predictions.

        Args:
            logits: Raw network outputs of shape ``(B, K)`` for ``K`` time intervals.
            bin_indices: Integer time-interval indices in ``0..K`` of shape ``(B,)``.
            event: Boolean or 0/1 event indicators of shape ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        scores = mtlr_cumulative_scores(logits)
        log_z = torch.logsumexp(scores, dim=1)

        bins = bin_indices.reshape(-1).long().clamp(0, logits.shape[1])
        is_event = event.reshape(-1).bool()
        batch_idx = torch.arange(logits.shape[0], device=logits.device)

        losses = torch.empty(logits.shape[0], device=logits.device, dtype=logits.dtype)

        if is_event.any():
            ev_rows = batch_idx[is_event]
            ev_bins = bins[is_event]
            losses[is_event] = log_z[is_event] - scores[ev_rows, ev_bins]

        is_censored = ~is_event
        if is_censored.any():
            cens_scores = scores[is_censored]
            cens_bins = bins[is_censored].unsqueeze(1)
            col_idx = torch.arange(scores.shape[1], device=logits.device).unsqueeze(0)
            valid_mask = col_idx >= cens_bins
            masked_scores = cens_scores.masked_fill(~valid_mask, float("-inf"))
            log_surv = torch.logsumexp(masked_scores, dim=1)
            losses[is_censored] = log_z[is_censored] - log_surv

        return losses.mean()

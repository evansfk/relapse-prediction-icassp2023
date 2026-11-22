"""Soft-label contrastive loss for Mixing-Up SSL (Wickstrøm et al., PRL 2022)."""

from __future__ import annotations

import math

from typing import override

import torch
import torch.nn.functional as F

from torch import Tensor, nn

_EXPECTED_NDIM = 2


def _resolve_lam_vector(
    lam: float | Tensor, batch_size: int, /, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Validate and broadcast ``lam`` into a finite 1D tensor of shape ``(B,)`` in ``[0, 1]``."""
    lam_t = torch.as_tensor(lam, device=device, dtype=dtype)
    if lam_t.ndim == 0:
        lam_vec = lam_t.expand(batch_size)
    elif lam_t.shape == (batch_size,):
        lam_vec = lam_t
    else:
        msg = f"Expected scalar or shape ({batch_size},) for lam, got {tuple(lam_t.shape)}"
        raise ValueError(msg)
    if not bool(torch.isfinite(lam_vec).all()) or bool(((lam_vec < 0.0) | (lam_vec > 1.0)).any()):
        msg = "lam must be finite and lie in [0, 1]"
        raise ValueError(msg)
    return lam_vec


class MixUpLoss(nn.Module):
    """Mixing-Up contrastive loss predicting the convex mixing proportion ``lam``.

    Supports two clean mixing modes without anchor duplication:
    - **Same-batch permutation mixing** (pass ``z_2=None, perm=perm`` where ``x_2 = x_1[perm]``):
      computes ``(B, B)`` logits against unique anchors ``z_1`` with soft targets
      ``targets[i, i] += lam`` and ``targets[i, perm[i]] += 1 - lam``, avoiding duplicate
      anchor columns and the ``log(2)`` loss floor.
    - **Disjoint-batch mixing** (pass two disjoint batches ``z_1`` and ``z_2`` with ``perm=None``):
      computes ``(B, 2B)`` logits matching Wickstrøm et al. (PRL 2022).

    Args:
        temperature: Contrastive temperature ``tau`` (default ``0.5``).
    """

    def __init__(self, *, temperature: float = 0.5) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            msg = f"temperature must be finite and positive, got {temperature}"
            raise ValueError(msg)
        self.temperature = temperature

    @override
    def forward(
        self,
        z_aug: Tensor,
        z_1: Tensor,
        z_2: Tensor | None = None,
        /,
        *,
        lam: float | Tensor,
        perm: Tensor | None = None,
    ) -> Tensor:
        """Compute Mixing-Up soft cross-entropy loss.

        Args:
            z_aug: Projected embeddings of mixed inputs, shape ``(B, D)``.
            z_1: Projected embeddings of first anchor inputs, shape ``(B, D)``.
            z_2: Optional projected embeddings of disjoint second anchors, shape ``(B, D)``.
            lam: Mixing coefficient in ``[0, 1]`` (scalar float or tensor of shape ``(B,)``).
            perm: Optional index permutation tensor ``(B,)`` when mixing within ``z_1``.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If neither or both of ``z_2`` and ``perm`` are supplied, or shapes mismatch.
        """
        if (z_2 is None) == (perm is None):
            msg = "Provide exactly one of disjoint `z_2` or same-batch `perm`"
            raise ValueError(msg)
        if z_aug.shape != z_1.shape or z_aug.dim() != _EXPECTED_NDIM:
            msg = f"Expected matching 2D tensors (B, D), got {tuple(z_aug.shape)} and {tuple(z_1.shape)}"
            raise ValueError(msg)

        batch_size = z_aug.shape[0]
        device = z_aug.device
        z_1_norm = F.normalize(z_1, dim=1)
        z_aug_norm = F.normalize(z_aug, dim=1)

        lam_vec = _resolve_lam_vector(lam, batch_size, device=device, dtype=z_aug.dtype)

        if perm is not None:
            if perm.shape != (batch_size,):
                msg = f"Expected perm shape ({batch_size},), got {tuple(perm.shape)}"
                raise ValueError(msg)
            perm_long = perm.to(device=device, dtype=torch.long)
            if bool(((perm_long < 0) | (perm_long >= batch_size)).any()):
                msg = f"perm indices must lie in [0, {batch_size})"
                raise ValueError(msg)
            row_idx = torch.arange(batch_size, device=device)
            soft_targets = torch.zeros((batch_size, batch_size), device=device, dtype=z_aug.dtype)
            soft_targets.index_put_((row_idx, row_idx), lam_vec, accumulate=True)
            soft_targets.index_put_((row_idx, perm_long), 1.0 - lam_vec, accumulate=True)
            logits = (z_aug_norm @ z_1_norm.T) / self.temperature
        else:
            if z_2 is None:
                msg = "Expected z_2 tensor when perm is None"
                raise ValueError(msg)
            if z_2.shape != z_1.shape:
                msg = f"Expected z_2 shape {tuple(z_1.shape)}, got {tuple(z_2.shape)}"
                raise ValueError(msg)
            z_2_norm = F.normalize(z_2, dim=1)
            eye = torch.eye(batch_size, device=device, dtype=z_aug.dtype)
            lam_col = lam_vec.unsqueeze(1)
            soft_targets = torch.cat([lam_col * eye, (1.0 - lam_col) * eye], dim=1)
            logits = torch.cat([z_aug_norm @ z_1_norm.T, z_aug_norm @ z_2_norm.T], dim=1) / self.temperature

        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()

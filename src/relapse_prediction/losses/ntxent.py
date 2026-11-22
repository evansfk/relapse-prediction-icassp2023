"""NT-Xent and Poly-NT-Xent contrastive losses.

``NTXentLoss`` is verified against ``pytorch-metric-learning`` for both values and
gradients. ``NTXentLossPoly`` implements the polynomial-augmented contrastive objective
used in Time-Frequency Consistency (Zhang et al., NeurIPS 2022).
"""

from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F

from torch import Tensor, nn

_EXPECTED_NDIM = 2
_MIN_BATCH_SIZE = 2


def _negative_pair_mask(batch_size: int, device: torch.device, /) -> Tensor:
    """Build boolean mask excluding self-pairs and positive pairs in a ``(2B, 2B)`` similarity matrix."""
    total_n = 2 * batch_size
    row_idx = torch.arange(total_n, device=device)
    pos_idx = (row_idx + batch_size) % total_n
    neg_mask = torch.ones((total_n, total_n), device=device, dtype=torch.bool)
    neg_mask.fill_diagonal_(fill_value=False)
    neg_mask[row_idx, pos_idx] = False
    return neg_mask


def _validate_pair_inputs(z1: Tensor, z2: Tensor, /, *, caller: str) -> None:
    """Validate that ``z1`` and ``z2`` are matching 2D tensors with batch size ``>= 2``."""
    if z1.shape != z2.shape:
        msg = f"{caller}: shape mismatch {tuple(z1.shape)} != {tuple(z2.shape)}"
        raise ValueError(msg)
    if z1.dim() != _EXPECTED_NDIM:
        msg = f"{caller}: expected 2D tensors (N, D), got {z1.dim()}D"
        raise ValueError(msg)
    if z1.shape[0] < _MIN_BATCH_SIZE:
        msg = f"{caller}: requires batch size >= {_MIN_BATCH_SIZE}, got {z1.shape[0]}"
        raise ValueError(msg)


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy (NT-Xent) contrastive loss.

    Args:
        temperature: Softmax temperature scaling similarity scores.
        use_cosine_similarity: If ``True``, L2-normalize representations inside the
            autograd graph before dot-product similarity (exact cosine similarity).
    """

    def __init__(self, *, temperature: float = 0.5, use_cosine_similarity: bool = True) -> None:
        super().__init__()
        if temperature <= 0:
            msg = f"temperature must be positive, got {temperature}"
            raise ValueError(msg)
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity

    def forward_with_logits(self, z1: Tensor, z2: Tensor, /) -> tuple[Tensor, Tensor]:
        """Compute NT-Xent loss and the ``(2N, 2N)`` pairwise similarity matrix.

        Args:
            z1: First view projections of shape ``(N, D)``.
            z2: Second view projections of shape ``(N, D)``.

        Returns:
            Tuple ``(loss, similarity_matrix)``.
        """
        _validate_pair_inputs(z1, z2, caller="NTXentLoss")

        representations = torch.cat([z1, z2], dim=0)
        if self.use_cosine_similarity:
            # Always normalize inside autograd so cosine-similarity gradients remain exact.
            representations = F.normalize(representations, dim=1)
        similarity_matrix = torch.matmul(representations, representations.T)

        batch_size = z1.shape[0]
        total_n = 2 * batch_size
        row_idx = torch.arange(total_n, device=z1.device)
        pos_idx = (row_idx + batch_size) % total_n

        pos_logits = similarity_matrix[row_idx, pos_idx].unsqueeze(1) / self.temperature
        neg_mask = _negative_pair_mask(batch_size, z1.device)
        neg_logits = similarity_matrix[neg_mask].view(total_n, total_n - 2) / self.temperature

        # Numerically stable log-domain cross-entropy (avoids float32 exp underflow).
        all_logits = torch.cat([pos_logits, neg_logits], dim=1)
        log_denom = torch.logsumexp(all_logits, dim=1)
        loss = (log_denom - pos_logits.squeeze(1)).mean()
        return loss, similarity_matrix

    @override
    def forward(self, z1: Tensor, z2: Tensor, /) -> Tensor:
        """Compute scalar NT-Xent loss between view projections ``z1`` and ``z2``.

        Args:
            z1: First view projections of shape ``(N, D)``.
            z2: Second view projections of shape ``(N, D)``.

        Returns:
            Scalar loss tensor.
        """
        loss, _ = self.forward_with_logits(z1, z2)
        return loss


class NTXentLossPoly(nn.Module):
    """Poly-loss-augmented NT-Xent objective from TFC (Zhang et al., NeurIPS 2022).

    Adds the first-order polynomial correction term ``epsilon * (1 / B - p_t)`` to
    standard NT-Xent cross-entropy, supporting arbitrary dynamic batch sizes on CPU or GPU.

    Args:
        temperature: Contrastive softmax temperature (default ``0.2`` in TFC).
        use_cosine_similarity: Whether to use cosine similarity instead of dot product.
    """

    def __init__(self, *, temperature: float = 0.2, use_cosine_similarity: bool = True) -> None:
        super().__init__()
        if temperature <= 0:
            msg = f"temperature must be positive, got {temperature}"
            raise ValueError(msg)
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity

    @override
    def forward(self, z1: Tensor, z2: Tensor, /) -> Tensor:
        """Compute Poly-NT-Xent loss between view projections ``z1`` and ``z2``.

        Args:
            z1: Tensor of shape ``(B, D)``.
            z2: Tensor of shape ``(B, D)``.

        Returns:
            Scalar loss tensor.
        """
        _validate_pair_inputs(z1, z2, caller="NTXentLossPoly")
        batch_size = z1.shape[0]
        total_n = 2 * batch_size

        representations = torch.cat([z2, z1], dim=0)
        if self.use_cosine_similarity:
            representations = F.normalize(representations, dim=1)
        similarity_matrix = torch.matmul(representations, representations.T)

        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos], dim=0).view(total_n, 1)

        neg_mask = _negative_pair_mask(batch_size, z1.device)
        negatives = similarity_matrix[neg_mask].view(total_n, -1)

        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(total_n, device=z1.device, dtype=torch.long)

        ce = F.cross_entropy(logits, labels, reduction="mean")
        probs = F.softmax(logits, dim=-1)
        # Equivalent to TFC's original mean over all entries of the positive-class one-hot probability.
        pt = probs[:, 0].mean() / float(logits.shape[-1])
        epsilon = float(batch_size)
        return ce + epsilon * (1.0 / float(batch_size) - pt)

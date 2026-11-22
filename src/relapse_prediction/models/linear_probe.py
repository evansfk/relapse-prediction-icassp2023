"""Linear classifier probe for the Person Identification proxy task (Section 4, Table 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import torch
import torch.nn.functional as F

from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from relapse_prediction.evaluation.classification import ClassificationMetrics, compute_classification_metrics
from relapse_prediction.utils.seed import seed_everything

if TYPE_CHECKING:
    import numpy as np

    from numpy.typing import NDArray


class LinearProbe(nn.Module):
    """Single linear classification layer evaluated on frozen SSL embeddings.

    Args:
        feature_dim: Input embedding dimension ``D``.
        num_classes: Number of unique user classes (14 in Table 2).
    """

    def __init__(self, *, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    @override
    def forward(self, x: Tensor, /) -> Tensor:
        logits: Tensor = self.classifier(x.reshape(x.shape[0], -1))
        return logits


@dataclass(frozen=True)
class LinearProbeResult:
    """Trained probe and validation/test metrics."""

    model: LinearProbe
    val_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    best_epoch: int


def train_linear_probe(
    *,
    x_train: NDArray[np.floating[Any]] | Tensor,
    y_train: NDArray[np.integer[Any]] | Tensor,
    x_val: NDArray[np.floating[Any]] | Tensor,
    y_val: NDArray[np.integer[Any]] | Tensor,
    x_test: NDArray[np.floating[Any]] | Tensor,
    y_test: NDArray[np.integer[Any]] | Tensor,
    num_classes: int,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> LinearProbeResult:
    """Train a linear classifier on frozen embeddings with validation macro-F1 early stopping.

    Implements the frozen-embedding 14-class linear probe (Section 4, Table 2) and computes
    true macro-AUPRC rather than reusing macro-Precision.

    Args:
        x_train: Training embeddings ``(N_tr, D)``.
        y_train: Integer user-ID labels ``(N_tr,)``.
        x_val: Validation embeddings ``(N_val, D)``.
        y_val: Validation labels ``(N_val,)``.
        x_test: Test embeddings ``(N_te, D)``.
        y_test: Test labels ``(N_te,)``.
        num_classes: Number of target user classes.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        patience: Early-stopping patience on validation macro-F1.
        device: Target PyTorch device.
        seed: Random seed for reproducible initialization and batch shuffling.

    Returns:
        :class:`LinearProbeResult` containing the best model and validation/test metrics.
    """
    seed_everything(seed)
    dev = torch.device(device)
    x_tr = torch.as_tensor(x_train, dtype=torch.float32)
    y_tr = torch.as_tensor(y_train, dtype=torch.long)
    x_v = torch.as_tensor(x_val, dtype=torch.float32)
    y_v = torch.as_tensor(y_val, dtype=torch.long)
    x_te = torch.as_tensor(x_test, dtype=torch.float32)
    y_te = torch.as_tensor(y_test, dtype=torch.long)

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr), batch_size=min(batch_size, len(x_tr)), shuffle=True, generator=gen
    )

    model = LinearProbe(feature_dim=x_tr.shape[1], num_classes=num_classes).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = -1.0
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx_d = bx.to(dev)
            by_d = by.to(dev)
            optimizer.zero_grad()
            logits: Tensor = model(bx_d)
            loss: Tensor = F.cross_entropy(logits, by_d)
            torch.autograd.backward(loss)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits: Tensor = model(x_v.to(dev))
            val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
        val_m = compute_classification_metrics(y_v.numpy(), val_probs, num_classes=num_classes)

        if val_m.macro_f1 > best_f1:
            best_f1 = val_m.macro_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_out: Tensor = model(x_v.to(dev))
        test_out: Tensor = model(x_te.to(dev))
        val_probs = F.softmax(val_out, dim=-1).cpu().numpy()
        test_probs = F.softmax(test_out, dim=-1).cpu().numpy()

    return LinearProbeResult(
        model=model,
        val_metrics=compute_classification_metrics(y_v.numpy(), val_probs, num_classes=num_classes),
        test_metrics=compute_classification_metrics(y_te.numpy(), test_probs, num_classes=num_classes),
        best_epoch=best_epoch,
    )

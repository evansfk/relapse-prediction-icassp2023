"""Classification metrics for the Person Identification proxy task (Table 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

if TYPE_CHECKING:
    from numpy.typing import NDArray

_BINARY_CLASSES = 2


@dataclass(frozen=True)
class ClassificationMetrics:
    """Summary metrics for multi-class person identification."""

    accuracy: float
    macro_f1: float
    macro_auprc: float
    macro_precision: float
    macro_recall: float
    macro_auroc: float | None = None


def compute_classification_metrics(
    y_true: NDArray[np.integer[Any]], y_probs: NDArray[np.floating[Any]], /, *, num_classes: int | None = None
) -> ClassificationMetrics:
    """Compute macro-averaged classification metrics matching Table 2 of the paper.

    Args:
        y_true: Integer class targets of shape ``(N,)``.
        y_probs: Predicted class probability matrix of shape ``(N, K)``.
        num_classes: Total number of classes ``K`` (inferred from ``y_probs.shape[1]`` if omitted).

    Returns:
        :class:`ClassificationMetrics` with accuracy, macro-F1, macro-AUPRC, precision,
        recall, and macro-AUROC (``None`` if a class is absent in ``y_true``).
    """
    y_t = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probs = np.asarray(y_probs, dtype=np.float64)
    k = num_classes if num_classes is not None else probs.shape[1]
    y_pred = np.argmax(probs, axis=1)

    present_classes = np.unique(y_t)
    acc = float(accuracy_score(y_t, y_pred))
    macro_f1 = float(f1_score(y_t, y_pred, labels=present_classes, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_t, y_pred, labels=present_classes, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_t, y_pred, labels=present_classes, average="macro", zero_division=0))

    if k == _BINARY_CLASSES:
        if len(present_classes) > 1:
            ap_pos = float(average_precision_score((y_t == 1).astype(np.int64), probs[:, 1]))
            ap_neg = float(average_precision_score((y_t == 0).astype(np.int64), probs[:, 0]))
            macro_auprc = 0.5 * (ap_pos + ap_neg)
            macro_auroc: float | None = float(roc_auc_score((y_t == 1).astype(np.int64), probs[:, 1]))
        else:
            macro_auprc = 0.0
            macro_auroc = None
    else:
        y_onehot = label_binarize(y_t, classes=np.arange(k))
        per_class_ap = [float(average_precision_score(y_onehot[:, c], probs[:, c])) for c in present_classes]
        macro_auprc = float(np.mean(per_class_ap)) if per_class_ap else 0.0
        if len(present_classes) == k:
            macro_auroc = float(roc_auc_score(y_onehot, probs, average="macro", multi_class="ovr"))
        else:
            macro_auroc = None

    return ClassificationMetrics(
        accuracy=acc,
        macro_f1=macro_f1,
        macro_auprc=macro_auprc,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        macro_auroc=macro_auroc,
    )

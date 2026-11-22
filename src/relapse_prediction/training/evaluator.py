"""Downstream evaluation runners for Person Identification (Table 2) and Survival Analysis (Figs. 2-3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from relapse_prediction.config import SurvivalConfig
from relapse_prediction.evaluation.survival_metrics import evaluate_survival_model
from relapse_prediction.models.linear_probe import train_linear_probe
from relapse_prediction.survival.deepsurv import DeepSurvModel
from relapse_prediction.survival.features import assemble_survival_features
from relapse_prediction.survival.neural_mtlr import NeuralMTLRModel
from relapse_prediction.survival.tree_models import ExtraSurvivalTreesModel, RandomSurvivalForestModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch

    from numpy.typing import NDArray

    from relapse_prediction.config import WindowHours
    from relapse_prediction.survival.base import BaseSurvivalModel

_MIN_CLASSES = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class PersonIDCVSummary:
    """Aggregated Person Identification metrics across folds/seeds (Table 2)."""

    n_users_retained: int
    n_samples_retained: int
    n_splits: int
    macro_f1_mean: float
    macro_f1_std: float
    macro_auprc_mean: float
    macro_auprc_std: float
    per_split_f1: list[float]
    per_split_auprc: list[float]
    accuracy_mean: float
    accuracy_std: float
    macro_precision_mean: float
    macro_precision_std: float
    macro_recall_mean: float
    macro_recall_std: float
    macro_auroc_mean: float
    macro_auroc_std: float


def run_person_id_evaluation(
    embeddings: NDArray[np.floating] | Sequence[NDArray[np.floating]],
    user_ids: NDArray[np.integer] | NDArray[np.str_] | NDArray[np.object_],
    /,
    *,
    min_samples_per_user: int = 100,
    n_splits: int = 5,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> PersonIDCVSummary:
    """Evaluate frozen embeddings on the Person Identification proxy task (Table 2).

    Supports both a single ``(S, D)`` embedding array and a sequence of ``n_splits``
    per-initialization embedding arrays (matching Paper Section 4's two-step 5-fold CV
    where each SSL seed pairs with an independent 75/15/10% stratified split).

    Args:
        embeddings: Single ``(S, D)`` matrix or sequence of ``n_splits`` ``(S, D)`` matrices.
        user_ids: Raw user identifiers of shape ``(S,)``.
        min_samples_per_user: Minimum sample count to retain a user class (100 in paper).
        n_splits: Number of independent stratified train/val/test iterations.
        train_ratio: Training split fraction.
        val_ratio: Validation split fraction (test fraction is ``1 - train_ratio - val_ratio``).
        epochs: Max training epochs for each linear probe.
        batch_size: Probe mini-batch size.
        lr: Probe Adam learning rate.
        patience: Validation macro-F1 early-stopping patience.
        device: PyTorch device.
        seed: Base random seed.

    Returns:
        :class:`PersonIDCVSummary` with mean and std macro-F1 and macro-AUPRC.
    """
    u_arr = np.asarray(user_ids)
    counts = pd.Series(u_arr).value_counts()
    valid_users = counts[counts >= min_samples_per_user].index.to_numpy()
    if len(valid_users) < _MIN_CLASSES:
        msg = f"Fewer than {_MIN_CLASSES} users have >= {min_samples_per_user} samples (found {len(valid_users)})"
        raise ValueError(msg)

    mask = np.isin(u_arr, valid_users)
    y_filt = np.asarray(LabelEncoder().fit_transform(u_arr[mask]), dtype=np.int64)
    num_classes = len(np.unique(y_filt))

    if isinstance(embeddings, np.ndarray):
        per_fold_embeddings = [np.asarray(embeddings, dtype=np.float32)[mask]] * n_splits
    else:
        if len(embeddings) != n_splits:
            msg = f"Expected {n_splits} per-fold embedding matrices, got {len(embeddings)}"
            raise ValueError(msg)
        per_fold_embeddings = [np.asarray(emb, dtype=np.float32)[mask] for emb in embeddings]

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        msg = f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0"
        raise ValueError(msg)

    f1_scores: list[float] = []
    auprc_scores: list[float] = []
    acc_scores: list[float] = []
    prec_scores: list[float] = []
    rec_scores: list[float] = []
    auroc_scores: list[float] = []

    for split_idx in range(n_splits):
        split_seed = seed + split_idx
        x_filt = per_fold_embeddings[split_idx]
        x_tr, x_tmp, y_tr, y_tmp = train_test_split(
            x_filt, y_filt, test_size=1.0 - train_ratio, stratify=y_filt, random_state=split_seed
        )
        inner_counts = np.bincount(y_tmp, minlength=num_classes)
        inner_strat = y_tmp if int(inner_counts.min()) >= _MIN_CLASSES else None
        x_val, x_te, y_val, y_te = train_test_split(
            x_tmp,
            y_tmp,
            test_size=test_ratio / (test_ratio + val_ratio),
            stratify=inner_strat,
            random_state=split_seed,
        )

        probe_res = train_linear_probe(
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_val,
            y_val=y_val,
            x_test=x_te,
            y_test=y_te,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            device=device,
            seed=split_seed,
        )
        m = probe_res.test_metrics
        f1_scores.append(m.macro_f1)
        auprc_scores.append(m.macro_auprc)
        acc_scores.append(m.accuracy)
        prec_scores.append(m.macro_precision)
        rec_scores.append(m.macro_recall)
        if m.macro_auroc is not None:
            auroc_scores.append(m.macro_auroc)

    ddof = 1 if len(f1_scores) > 1 else 0
    ddof_auroc = 1 if len(auroc_scores) > 1 else 0
    return PersonIDCVSummary(
        n_users_retained=num_classes,
        n_samples_retained=len(y_filt),
        n_splits=n_splits,
        macro_f1_mean=float(np.mean(f1_scores)),
        macro_f1_std=float(np.std(f1_scores, ddof=ddof)),
        macro_auprc_mean=float(np.mean(auprc_scores)),
        macro_auprc_std=float(np.std(auprc_scores, ddof=ddof)),
        per_split_f1=f1_scores,
        per_split_auprc=auprc_scores,
        accuracy_mean=float(np.mean(acc_scores)),
        accuracy_std=float(np.std(acc_scores, ddof=ddof)),
        macro_precision_mean=float(np.mean(prec_scores)),
        macro_precision_std=float(np.std(prec_scores, ddof=ddof)),
        macro_recall_mean=float(np.mean(rec_scores)),
        macro_recall_std=float(np.std(rec_scores, ddof=ddof)),
        macro_auroc_mean=float(np.mean(auroc_scores)) if auroc_scores else 0.0,
        macro_auroc_std=float(np.std(auroc_scores, ddof=ddof_auroc)) if auroc_scores else 0.0,
    )


def person_id_summaries_to_frame(summaries: Mapping[str, PersonIDCVSummary], /) -> pd.DataFrame:
    """Convert multi-model :class:`PersonIDCVSummary` mappings to a long metrics DataFrame.

    Matches the schema expected by ``plot_person_id_metrics`` (``identification.pdf``):
    columns ``["model", "Metric", "mean", "std"]`` across ``F1``, ``Recall``,
    ``Precision``, ``AUPRC``, ``Accuracy``, and ``AUROC``.
    """
    rows: list[dict[str, object]] = []
    for model_name, s in summaries.items():
        metric_pairs = (
            ("F1", s.macro_f1_mean, s.macro_f1_std),
            ("Recall", s.macro_recall_mean, s.macro_recall_std),
            ("Precision", s.macro_precision_mean, s.macro_precision_std),
            ("AUPRC", s.macro_auprc_mean, s.macro_auprc_std),
            ("Accuracy", s.accuracy_mean, s.accuracy_std),
            ("AUROC", s.macro_auroc_mean, s.macro_auroc_std),
        )
        for metric_name, mean_val, std_val in metric_pairs:
            rows.append({"model": model_name, "Metric": metric_name, "mean": mean_val, "std": std_val})
    return pd.DataFrame(rows)


def format_person_id_table(summaries: Mapping[str, PersonIDCVSummary], /, *, decimals: int = 4) -> pd.DataFrame:
    """Format aggregated Person ID results matching Paper Table 2 (AUPRC and macro-F1)."""
    rows: list[dict[str, str]] = []
    for model_name, s in summaries.items():
        rows.append(
            {
                "model": model_name,
                "AUPRC": f"{s.macro_auprc_mean:.{decimals}f} (+/- {s.macro_auprc_std:.{decimals}f})",
                "macro-F1": f"{s.macro_f1_mean:.{decimals}f} (+/- {s.macro_f1_std:.{decimals}f})",
            }
        )
    return pd.DataFrame(rows).set_index("model")


def compute_ablation_relative_change(
    results: pd.DataFrame, /, *, baseline_label: str, label_col: str = "ablation"
) -> pd.DataFrame:
    """Compute relative percentage change vs. baseline matching Paper Table 3.

    Reproduces the Paper Table 3 relative change formula ``((ablation - base) / base) * 100``
    for ``c_index`` and ``neg_log_ibs``, excluding the baseline row and sorted ascending
    by ``c_index_rel_change_pct``.

    Args:
        results: DataFrame containing ``label_col``, ``"c_index"``, and ``"neg_log_ibs"``.
        baseline_label: Row label in ``label_col`` used as the 0% reference baseline.
        label_col: Column naming each ablation configuration.

    Returns:
        DataFrame with ``label_col``, ``"c_index_rel_change_pct"``, and
        ``"neg_log_ibs_rel_change_pct"``.

    Raises:
        ValueError: If ``baseline_label`` does not appear exactly once or has zero value.
    """
    base_rows = results[results[label_col] == baseline_label]
    if len(base_rows) != 1:
        msg = f"Expected exactly 1 baseline row for {baseline_label!r} in column {label_col!r}, found {len(base_rows)}"
        raise ValueError(msg)

    base_c = float(base_rows["c_index"].iloc[0])
    base_nli = float(base_rows["neg_log_ibs"].iloc[0])
    if base_c == 0.0 or base_nli == 0.0:
        msg = "Baseline c_index and neg_log_ibs must be non-zero to compute relative percentage change"
        raise ValueError(msg)

    ablation_rows = results[results[label_col] != baseline_label].copy()
    ablation_rows["c_index_rel_change_pct"] = ((ablation_rows["c_index"] - base_c) / base_c) * 100.0
    ablation_rows["neg_log_ibs_rel_change_pct"] = ((ablation_rows["neg_log_ibs"] - base_nli) / base_nli) * 100.0
    return (
        ablation_rows[[label_col, "c_index_rel_change_pct", "neg_log_ibs_rel_change_pct"]]
        .sort_values(by="c_index_rel_change_pct", ascending=True)
        .reset_index(drop=True)
    )


def build_default_survival_models(config: SurvivalConfig, /, *, device: str | torch.device) -> list[BaseSurvivalModel]:
    """Instantiate the four survival models benchmarked in Section 4 (Figs. 2-3)."""
    return [
        RandomSurvivalForestModel(
            n_estimators=config.n_trees,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            max_samples=config.max_samples,
            n_jobs=1,
            seed=config.seed,
        ),
        ExtraSurvivalTreesModel(
            n_estimators=config.n_trees,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            max_samples=config.max_samples,
            n_jobs=1,
            seed=config.seed,
        ),
        NeuralMTLRModel(
            hidden_units=config.mtlr_hidden_units,
            bins=config.mtlr_bins,
            binning=config.mtlr_binning,
            l2_smooth=config.mtlr_l2_smooth,
            activation=config.mtlr_activation,
            dropout=config.mtlr_dropout,
            lr=config.mtlr_lr,
            weight_decay=config.mtlr_weight_decay,
            epochs=config.mtlr_epochs,
            batch_size=config.mtlr_batch_size,
            device=device,
            seed=config.seed,
        ),
        DeepSurvModel(
            hidden_units=config.deepsurv_hidden_units,
            activation=config.deepsurv_activation,
            dropout=config.deepsurv_dropout,
            lr=config.deepsurv_lr,
            weight_decay=config.deepsurv_weight_decay,
            epochs=config.deepsurv_epochs,
            batch_size=config.deepsurv_batch_size,
            device=device,
            seed=config.seed,
        ),
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class SurvivalBenchmarkResult:
    """Results DataFrame, fitted models, split indices, and assembled train/test covariates."""

    results: pd.DataFrame
    fitted_models: dict[str, BaseSurvivalModel]
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    x_train: NDArray[np.float64]
    x_test: NDArray[np.float64]


def run_survival_benchmark(
    embeddings: NDArray[np.floating],
    /,
    *,
    metadata: pd.DataFrame,
    time_days: NDArray[np.floating],
    event: NDArray[np.bool_],
    embedding_name: str = "TFC",
    window_hours: WindowHours | None = None,
    static_features: pd.DataFrame | None = None,
    config: SurvivalConfig | None = None,
    models: list[BaseSurvivalModel] | None = None,
    device: str | torch.device = "cpu",
) -> SurvivalBenchmarkResult:
    """Run the 60/40 survival train/test evaluation across survival models (Section 4).

    Args:
        embeddings: Extracted downstream embeddings ``(S, D)``.
        metadata: Sample metadata DataFrame containing ``user_id`` and past-episode columns.
        time_days: Time-to-relapse / censoring times in days ``(S,)``.
        event: Boolean relapse indicators ``(S,)``.
        embedding_name: Label for the embedding source (e.g., ``"TFC"``, ``"Mixing-Up"``).
        window_hours: Optional window duration (``4``, ``8``, or ``12``) recorded in ``sample_length``.
        static_features: Optional patient-level static covariate DataFrame indexed by ``user_id``.
        config: :class:`SurvivalConfig` controlling split, covariates, and model hyperparameters.
        models: Optional explicit list of survival models (defaults to the 4 paper models).
        device: PyTorch device for DeepSurv and Neural MTLR.

    Returns:
        :class:`SurvivalBenchmarkResult` containing metrics table, fitted models, and split covariates.
    """
    cfg = config or SurvivalConfig()
    n_samples = len(embeddings)
    e_arr = np.asarray(event, dtype=bool).reshape(-1)
    t_arr = np.asarray(time_days, dtype=np.float64).reshape(-1)
    if len(metadata) != n_samples or len(t_arr) != n_samples or len(e_arr) != n_samples:
        msg = (
            f"Row count mismatch: embeddings={n_samples}, metadata={len(metadata)}, "
            f"time_days={len(t_arr)}, event={len(e_arr)}"
        )
        raise ValueError(msg)
    all_indices = np.arange(n_samples, dtype=np.int64)

    stratify_arr = e_arr if (np.sum(e_arr) >= _MIN_CLASSES and np.sum(~e_arr) >= _MIN_CLASSES) else None
    train_idx, test_idx = train_test_split(
        all_indices, test_size=cfg.test_size, stratify=stratify_arr, random_state=cfg.seed
    )

    assembled = assemble_survival_features(
        emb_train=embeddings[train_idx],
        emb_test=embeddings[test_idx],
        meta_train=metadata.iloc[train_idx].reset_index(drop=True),
        meta_test=metadata.iloc[test_idx].reset_index(drop=True),
        include_user_id=cfg.include_user_id,
        include_past_episodes=cfg.include_past_episodes,
        include_event_free_days=cfg.include_event_free_days,
        static_features=static_features,
        include_static=cfg.include_static,
        drop_first=cfg.drop_first_onehot,
    )

    model_list = models if models is not None else build_default_survival_models(cfg, device=device)
    rows: list[dict[str, object]] = []
    fitted: dict[str, BaseSurvivalModel] = {}

    for model in model_list:
        model.fit(assembled.x_train, time=t_arr[train_idx], event=e_arr[train_idx])
        metrics = evaluate_survival_model(
            model,
            assembled.x_test,
            time_train=t_arr[train_idx],
            event_train=e_arr[train_idx],
            time_test=t_arr[test_idx],
            event_test=e_arr[test_idx],
            ibs_mode=cfg.ibs_mode,
        )
        fitted[model.name] = model
        row: dict[str, object] = {
            "embedding": embedding_name,
            "survival_model": model.name,
            "c_index": metrics.c_index,
            "ibs": metrics.ibs,
            "neg_log_ibs": metrics.neg_log_ibs,
            "n_features": assembled.x_train.shape[1],
        }
        if window_hours is not None:
            row["sample_length"] = f"{window_hours}-hour Dataset"
        rows.append(row)

    return SurvivalBenchmarkResult(
        results=pd.DataFrame(rows),
        fitted_models=fitted,
        train_indices=np.asarray(train_idx, dtype=np.int64),
        test_indices=np.asarray(test_idx, dtype=np.int64),
        x_train=assembled.x_train,
        x_test=assembled.x_test,
    )

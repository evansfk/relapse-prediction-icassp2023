r"""Run downstream Survival Analysis (DeepSurv, Neural MTLR, ExtraSurvivalTrees, RSF).

Example:
    uv run python scripts/run_survival_analysis.py --data-dir outputs/synthetic_4h \
        --embeddings outputs/runs/tfc_4h/downstream_embeddings.npy --past-episodes
"""

from __future__ import annotations

import argparse

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from relapse_prediction.config import SurvivalConfig
from relapse_prediction.data.synthetic import generate_synthetic_wearable_dataset
from relapse_prediction.evaluation.survival_metrics import stratify_risk_groups
from relapse_prediction.models.minirocket import MiniRocketExtractor
from relapse_prediction.training.evaluator import SurvivalBenchmarkResult, run_survival_benchmark
from relapse_prediction.utils.paper_figures import (
    plot_risk_classification_timeline,
    plot_risk_group_histogram,
    plot_risk_group_survival_curves,
    plot_survival_time_distribution,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from relapse_prediction.survival.base import BaseSurvivalModel


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for downstream survival analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with dataset.npz and metadata.csv.")
    parser.add_argument("--embeddings", type=Path, default=None, help="Path to .npy embeddings (omit for MiniRocket).")
    parser.add_argument("--embedding-name", type=str, default="TFC")
    parser.add_argument("--config", type=Path, default=Path("configs/survival/default.yaml"))
    parser.add_argument(
        "--minirocket-features",
        type=int,
        default=9996,
        help="MiniRocket feature count when --embeddings is omitted (9996 in paper).",
    )
    parser.add_argument(
        "--user-id",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config include_user_id (use --no-user-id for Table 3 ablation).",
    )
    parser.add_argument(
        "--static",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config include_static (+st in Table 3).",
    )
    parser.add_argument(
        "--past-episodes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config include_past_episodes (+ps, p_ep, ll in Table 3).",
    )
    parser.add_argument(
        "--event-free-days",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config include_event_free_days (opt-in dynamic covariate).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override neural survival training epochs.")
    parser.add_argument("--n-trees", type=int, default=None, help="Override forest ensemble tree count.")
    parser.add_argument("--seed", type=int, default=None, help="Override config random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save-figures-dir",
        type=Path,
        default=None,
        help="Optional directory to export publication PDFs (survival distribution, Fig. 4 risk timeline, curves).",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        default=None,
        help="Optional CSV path to persist the survival benchmark metrics table.",
    )
    return parser.parse_args()


def _apply_cli_overrides(cfg: SurvivalConfig, args: argparse.Namespace, /) -> SurvivalConfig:
    """Apply explicit CLI overrides without clobbering YAML defaults."""
    if args.user_id is not None:
        cfg.include_user_id = args.user_id
    if args.static is not None:
        cfg.include_static = args.static
    if args.past_episodes is not None:
        cfg.include_past_episodes = args.past_episodes
    if args.event_free_days is not None:
        cfg.include_event_free_days = args.event_free_days
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.deepsurv_epochs = args.epochs
        cfg.mtlr_epochs = args.epochs
    if args.n_trees is not None:
        cfg.n_trees = args.n_trees
    return cfg


def _export_survival_figures(
    fig_dir: Path,
    /,
    *,
    best_model: BaseSurvivalModel,
    benchmark: SurvivalBenchmarkResult,
    metadata: pd.DataFrame,
    relapses_df: pd.DataFrame | None,
    time_days: NDArray[np.floating],
    event: NDArray[np.bool_],
    test_risk: NDArray[np.float64],
    risk_tiers: NDArray[np.str_],
    cutoffs: tuple[float, float],
    seed: int,
) -> None:
    """Export survival distribution, Fig. 4 risk timeline, and risk-tier curve PDFs."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_dist = plot_survival_time_distribution(
        time_train=time_days[benchmark.train_indices],
        event_train=event[benchmark.train_indices],
        time_test=time_days[benchmark.test_indices],
        event_test=event[benchmark.test_indices],
        output_path=fig_dir / "survival_distribution.pdf",
    )
    plt.close(fig_dist)

    fig_hist = plot_risk_group_histogram(
        test_risk, risk_tiers, cutoffs=cutoffs, output_path=fig_dir / "risk_groups_histogram.pdf"
    )
    plt.close(fig_hist)

    fig_curves = plot_risk_group_survival_curves(
        best_model,
        benchmark.x_test,
        time_test=time_days[benchmark.test_indices],
        event_test=event[benchmark.test_indices],
        risk_labels=risk_tiers,
        seed=seed,
        output_path=fig_dir / "risk_survival_curves.pdf",
    )
    plt.close(fig_curves)

    if relapses_df is not None:
        plot_risk_classification_timeline(
            metadata.iloc[benchmark.test_indices].reset_index(drop=True),
            relapses_df,
            risk_tiers,
            seed=seed,
            output_path=fig_dir / "final_risk_classification.pdf",
        )
    print(f"Saved publication figures -> {fig_dir}")


def main() -> None:
    """Run downstream survival benchmark and print 3-tier risk stratification summary."""
    args = parse_args()
    default_cfg_path = Path("configs/survival/default.yaml")
    if args.config != default_cfg_path and not args.config.exists():
        msg = f"Survival config file not found: {args.config}"
        raise FileNotFoundError(msg)
    cfg = SurvivalConfig.from_yaml(args.config) if args.config.exists() else SurvivalConfig()
    cfg = _apply_cli_overrides(cfg, args)

    if args.data_dir is not None:
        loaded = np.load(args.data_dir / "dataset.npz")
        event = loaded["event"]
        time_days = loaded["time_days"]
        metadata = pd.read_csv(args.data_dir / "metadata.csv")
        static_path = args.data_dir / "static_features.csv"
        static_df = pd.read_csv(static_path, index_col="user_id") if static_path.exists() else None
        relapses_path = args.data_dir / "relapse_episodes.csv"
        relapses_df = pd.read_csv(relapses_path) if relapses_path.exists() else None
        pretext_x = loaded["pretext_x"] if args.embeddings is None else np.empty((0, 0, 0), dtype=np.float32)
        downstream_x = loaded["downstream_x"] if args.embeddings is None else np.empty((0, 0, 0), dtype=np.float32)
    else:
        bundle = generate_synthetic_wearable_dataset(window_hours=4, samples_per_user=30, seed=cfg.seed)
        pretext_x = bundle.pretext_x
        downstream_x = bundle.downstream_x
        event = bundle.event
        time_days = bundle.time_days
        metadata = bundle.metadata
        static_df = bundle.static_features
        relapses_df = bundle.relapse_episodes

    if args.embeddings is not None:
        embeddings = np.asarray(np.load(args.embeddings), dtype=np.float32)
        emb_name = args.embedding_name
    else:
        # Fit MiniRocket strictly on unlabeled pretext cohort (zero downstream test leakage).
        embeddings = (
            MiniRocketExtractor(num_features=args.minirocket_features, device=args.device, seed=cfg.seed)
            .fit(pretext_x)
            .transform(downstream_x)
        )
        emb_name = "MiniRocket"

    benchmark = run_survival_benchmark(
        embeddings,
        metadata=metadata,
        time_days=time_days,
        event=event,
        embedding_name=emb_name,
        static_features=static_df,
        config=cfg,
        device=args.device,
    )

    print("\nDownstream Survival Analysis Results:")
    print(benchmark.results.to_string(index=False))

    # Fit 3-tier risk cutoffs strictly on training-set risk scores, then stratify test set (Fig. 4).
    best_model = benchmark.fitted_models["DeepSurv"]
    train_risk = best_model.predict_risk(benchmark.x_train)
    _, train_cutoffs = stratify_risk_groups(train_risk, quantiles=cfg.risk_quantiles)
    test_risk = best_model.predict_risk(benchmark.x_test)
    risk_tiers, (low_cut, high_cut) = stratify_risk_groups(test_risk, cutoffs=train_cutoffs)
    tier_counts = pd.Series(risk_tiers).value_counts().reindex(["low", "medium", "high"], fill_value=0)
    print(
        f"\nDeepSurv Risk Stratification (train-fitted cutoffs: {low_cut:.2f}, {high_cut:.2f}): "
        f"low={tier_counts['low']}, medium={tier_counts['medium']}, high={tier_counts['high']}"
    )

    if args.save_results is not None:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        benchmark.results.to_csv(args.save_results, index=False)
        print(f"Saved metrics table -> {args.save_results}")

    if args.save_figures_dir is not None:
        _export_survival_figures(
            args.save_figures_dir,
            best_model=best_model,
            benchmark=benchmark,
            metadata=metadata,
            relapses_df=relapses_df,
            time_days=time_days,
            event=event,
            test_risk=test_risk,
            risk_tiers=risk_tiers,
            cutoffs=(low_cut, high_cut),
            seed=cfg.seed,
        )


if __name__ == "__main__":
    main()

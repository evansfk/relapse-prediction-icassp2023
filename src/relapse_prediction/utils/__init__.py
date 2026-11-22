"""Utility helpers for reproducibility and publication-grade plotting."""

from .paper_figures import (
    ieee_plot_style,
    plot_cohort_relapse_timeline,
    plot_per_embedding_comparison,
    plot_per_survival_model_comparison,
    plot_person_id_metrics,
    plot_risk_classification_timeline,
    plot_risk_group_histogram,
    plot_risk_group_survival_curves,
    plot_survival_time_distribution,
    plot_user_sample_distribution,
    save_altair_chart,
    save_matplotlib_figure,
)
from .plotting import plot_mts, plot_survival_comparison
from .seed import seed_everything

__all__ = [
    "ieee_plot_style",
    "plot_cohort_relapse_timeline",
    "plot_mts",
    "plot_per_embedding_comparison",
    "plot_per_survival_model_comparison",
    "plot_person_id_metrics",
    "plot_risk_classification_timeline",
    "plot_risk_group_histogram",
    "plot_risk_group_survival_curves",
    "plot_survival_comparison",
    "plot_survival_time_distribution",
    "plot_user_sample_distribution",
    "save_altair_chart",
    "save_matplotlib_figure",
    "seed_everything",
]

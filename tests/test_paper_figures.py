"""Unit tests for IEEE ICASSP 2023 publication figures (Altair & SciencePlots) and Tables 2-3."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from relapse_prediction.data.synthetic import generate_synthetic_wearable_dataset
from relapse_prediction.evaluation.survival_metrics import stratify_risk_groups
from relapse_prediction.survival.deepsurv import DeepSurvModel
from relapse_prediction.training.evaluator import (
    PersonIDCVSummary,
    compute_ablation_relative_change,
    format_person_id_table,
    person_id_summaries_to_frame,
)
from relapse_prediction.utils.paper_figures import (
    plot_cohort_relapse_timeline,
    plot_per_embedding_comparison,
    plot_per_survival_model_comparison,
    plot_person_id_metrics,
    plot_risk_classification_timeline,
    plot_risk_group_histogram,
    plot_risk_group_survival_curves,
    plot_survival_time_distribution,
    plot_user_sample_distribution,
)

if TYPE_CHECKING:
    from pathlib import Path


def _assert_valid_pdf(path: Path, /) -> None:
    assert path.exists(), f"Expected PDF at {path}"
    raw = path.read_bytes()
    assert len(raw) > 500
    assert raw.startswith(b"%PDF-")


def test_table2_and_table3_formatters_match_paper_formulas() -> None:
    summaries = {
        "MiniRocket": PersonIDCVSummary(
            n_users_retained=14,
            n_samples_retained=500,
            n_splits=5,
            macro_f1_mean=0.6737,
            macro_f1_std=0.0307,
            macro_auprc_mean=0.7091,
            macro_auprc_std=0.0296,
            per_split_f1=[0.67],
            per_split_auprc=[0.71],
            accuracy_mean=0.68,
            accuracy_std=0.02,
            macro_precision_mean=0.69,
            macro_precision_std=0.02,
            macro_recall_mean=0.66,
            macro_recall_std=0.02,
            macro_auroc_mean=0.92,
            macro_auroc_std=0.01,
        ),
        "Mixing-up": PersonIDCVSummary(
            n_users_retained=14,
            n_samples_retained=500,
            n_splits=5,
            macro_f1_mean=0.7694,
            macro_f1_std=0.0160,
            macro_auprc_mean=0.7451,
            macro_auprc_std=0.0086,
            per_split_f1=[0.77],
            per_split_auprc=[0.75],
            accuracy_mean=0.79,
            accuracy_std=0.01,
            macro_precision_mean=0.78,
            macro_precision_std=0.01,
            macro_recall_mean=0.76,
            macro_recall_std=0.01,
            macro_auroc_mean=0.95,
            macro_auroc_std=0.01,
        ),
    }

    frame = person_id_summaries_to_frame(summaries)
    assert set(frame["Metric"]) == {"F1", "Recall", "Precision", "AUPRC", "Accuracy", "AUROC"}
    assert len(frame) == 12

    table2 = format_person_id_table(summaries)
    assert list(table2.columns) == ["AUPRC", "macro-F1"]
    assert table2.loc["Mixing-up", "macro-F1"] == "0.7694 (+/- 0.0160)"

    ablation_raw = pd.DataFrame(
        [
            {"ablation": "baseline", "c_index": 0.75, "neg_log_ibs": 2.00},
            {"ablation": "- user's-ID", "c_index": 0.60, "neg_log_ibs": 1.60},
            {"ablation": "+ st, ps, p_ep, ll", "c_index": 0.84, "neg_log_ibs": 2.30},
        ]
    )
    table3 = compute_ablation_relative_change(ablation_raw, baseline_label="baseline")
    assert list(table3["ablation"]) == ["- user's-ID", "+ st, ps, p_ep, ll"]
    assert table3["c_index_rel_change_pct"].iloc[0] == pytest.approx(-20.0)
    assert table3["neg_log_ibs_rel_change_pct"].iloc[0] == pytest.approx(-20.0)
    assert table3["c_index_rel_change_pct"].iloc[1] == pytest.approx(12.0)
    assert table3["neg_log_ibs_rel_change_pct"].iloc[1] == pytest.approx(15.0)

    with pytest.raises(ValueError, match="Expected exactly 1 baseline row"):
        compute_ablation_relative_change(ablation_raw, baseline_label="missing")


def test_all_nine_paper_and_report_figures_export_valid_pdfs(tmp_path: Path) -> None:
    bundle = generate_synthetic_wearable_dataset(
        window_hours=4, seq_len=48, n_pretext_users=10, n_downstream_users=8, samples_per_user=16, seed=42
    )
    assert "next_episode" in bundle.metadata.columns

    # 1. Paper Fig. 1 (chart.pdf - Altair)
    chart_fig1 = plot_cohort_relapse_timeline(
        bundle.metadata, bundle.relapse_episodes, max_samples=80, seed=42, output_path=tmp_path / "chart.pdf"
    )
    assert chart_fig1 is not None
    _assert_valid_pdf(tmp_path / "chart.pdf")

    # 2. Paper Fig. 2 (per_embedding.pdf) & Fig. 3 (per_smodel.pdf)
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for dur in ("4-hour Dataset", "8-hour Dataset", "12-hour Dataset"):
        for emb in ("TS-TCC", "TFC", "MiniRocket", "Mixing-up"):
            for sm in ("RandomSurvivalForest", "ExtraSurvivalTrees", "NeuralMTLR", "DeepSurv"):
                c_idx = float(rng.uniform(0.62, 0.76))
                ibs = float(rng.uniform(0.11, 0.22))
                rows.append(
                    {
                        "embedding": emb,
                        "survival_model": sm,
                        "sample_length": dur,
                        "c_index": c_idx,
                        "ibs": ibs,
                        "neg_log_ibs": float(-np.log(ibs)),
                    }
                )
    surv_df = pd.DataFrame(rows)

    fig2 = plot_per_embedding_comparison(surv_df, output_path=tmp_path / "per_embedding.pdf")
    plt.close(fig2)
    _assert_valid_pdf(tmp_path / "per_embedding.pdf")

    fig3 = plot_per_survival_model_comparison(
        surv_df,
        sample_length="8-hour Dataset",
        c_index_ylim=(0.5, 0.8),
        neg_log_ibs_ylim=(0.6, 2.3),
        output_path=tmp_path / "per_smodel.pdf",
    )
    plt.close(fig3)
    _assert_valid_pdf(tmp_path / "per_smodel.pdf")

    # 3. Report Figs: identification.pdf & identification_dist.pdf
    pid_rows = [
        {"model": m, "Metric": met, "mean": float(rng.uniform(0.45, 0.92)), "std": 0.02}
        for m in ("TF-C", "MiniRocket", "TS-TCC", "Mixing-up")
        for met in ("F1", "Recall", "Precision", "AUPRC", "Accuracy", "AUROC")
    ]
    fig_id = plot_person_id_metrics(pd.DataFrame(pid_rows), output_path=tmp_path / "identification.pdf")
    plt.close(fig_id)
    _assert_valid_pdf(tmp_path / "identification.pdf")

    fig_udist = plot_user_sample_distribution(
        bundle.downstream_user_ids, min_samples_threshold=10, output_path=tmp_path / "identification_dist.pdf"
    )
    plt.close(fig_udist)
    _assert_valid_pdf(tmp_path / "identification_dist.pdf")

    # 4. Report Fig: survival_distribution.pdf
    split = len(bundle.time_days) // 2
    fig_sdist = plot_survival_time_distribution(
        time_train=bundle.time_days[:split],
        event_train=bundle.event[:split],
        time_test=bundle.time_days[split:],
        event_test=bundle.event[split:],
        output_path=tmp_path / "survival_distribution.pdf",
    )
    plt.close(fig_sdist)
    _assert_valid_pdf(tmp_path / "survival_distribution.pdf")

    # 5. Paper Fig. 4 (final_risk_classification.pdf) + risk diagnostics
    x_feat = bundle.downstream_x.mean(axis=-1).astype(np.float64)
    model = DeepSurvModel(hidden_units=16, epochs=10, batch_size=32, seed=42).fit(
        x_feat[:split], time=bundle.time_days[:split], event=bundle.event[:split]
    )
    train_risk = model.predict_risk(x_feat[:split])
    _, cutoffs = stratify_risk_groups(train_risk)
    test_risk = model.predict_risk(x_feat[split:])
    risk_tiers, _ = stratify_risk_groups(test_risk, cutoffs=cutoffs)

    chart_fig4 = plot_risk_classification_timeline(
        bundle.metadata.iloc[split:].reset_index(drop=True),
        bundle.relapse_episodes,
        risk_tiers,
        max_users=5,
        seed=42,
        output_path=tmp_path / "final_risk_classification.pdf",
    )
    assert chart_fig4 is not None
    _assert_valid_pdf(tmp_path / "final_risk_classification.pdf")

    fig_hist = plot_risk_group_histogram(
        test_risk, risk_tiers, cutoffs=cutoffs, output_path=tmp_path / "risk_groups_histogram.pdf"
    )
    plt.close(fig_hist)
    _assert_valid_pdf(tmp_path / "risk_groups_histogram.pdf")

    fig_curves = plot_risk_group_survival_curves(
        model,
        x_feat[split:],
        time_test=bundle.time_days[split:],
        event_test=bundle.event[split:],
        risk_labels=risk_tiers,
        seed=42,
        output_path=tmp_path / "risk_survival_curves.pdf",
    )
    plt.close(fig_curves)
    _assert_valid_pdf(tmp_path / "risk_survival_curves.pdf")

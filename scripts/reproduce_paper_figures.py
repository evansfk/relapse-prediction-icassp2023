r"""Reproduce all IEEE ICASSP 2023 paper figures, report diagnostics, and LaTeX tables.

Runs the full multi-duration (4h, 8h, 12h), multi-embedding (TS-TCC, TFC, MiniRocket,
Mixing-up), and multi-survival-model pipeline on synthetic wearable data (or custom
dataset directories conforming to the documented schema) and exports every publication
figure in the exact SciencePlots IEEE / Altair ``sfmono`` visual style:

- ``chart.pdf`` (Paper Fig. 1 — Altair cohort timeline of relapses & right-censoring)
- ``per_embedding.pdf`` (Paper Fig. 2 — Multi-duration per-embedding boxplot)
- ``per_smodel.pdf`` (Paper Fig. 3 — 8-hour per-survival-model grouped barplot)
- ``final_risk_classification.pdf`` (Paper Fig. 4 — Altair 5-user 3-tier risk timeline)
- ``identification.pdf`` (Person Identification grouped metric barplot)
- ``identification_dist.pdf`` (Per-user sample count distribution)
- ``survival_distribution.pdf`` (60/40 train/test survival time histograms)
- ``risk_groups_histogram.pdf`` & ``risk_survival_curves.pdf`` (Risk tier diagnostics)
- ``table2_person_id.{csv,tex}`` & ``table3_ablations.{csv,tex}`` (Paper Tables 2 & 3)

Example:
    uv run python scripts/reproduce_paper_figures.py --output-dir outputs/paper_figures --fast
"""

from __future__ import annotations

import argparse
import dataclasses

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from relapse_prediction.config import (
    MixingUpConfig,
    SurvivalConfig,
    TFCConfig,
    TSTCCConfig,
    WindowHours,
    select_channel_indices,
)
from relapse_prediction.data.synthetic import SyntheticWearableDataset, generate_synthetic_wearable_dataset
from relapse_prediction.evaluation.survival_metrics import stratify_risk_groups
from relapse_prediction.models.minirocket import MiniRocketExtractor
from relapse_prediction.survival.deepsurv import DeepSurvModel
from relapse_prediction.training.evaluator import (
    PersonIDCVSummary,
    compute_ablation_relative_change,
    format_person_id_table,
    person_id_summaries_to_frame,
    run_person_id_evaluation,
    run_survival_benchmark,
)
from relapse_prediction.training.ssl_trainer import (
    extract_embeddings,
    pretrain_mixing_up,
    pretrain_tfc,
    pretrain_tstcc,
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
    from numpy.typing import NDArray

_WINDOW_DURATIONS: tuple[WindowHours, ...] = (4, 8, 12)
_DEFAULT_PAPER_DURATION: WindowHours = 8
_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for end-to-end paper figure & table reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use compact synthetic sequence lengths and low epoch counts for quick end-to-end verification.",
    )
    parser.add_argument("--n-pretext-users", type=int, default=20)
    parser.add_argument("--n-downstream-users", type=int, default=16)
    parser.add_argument("--samples-per-user", type=int, default=28)
    parser.add_argument("--ssl-epochs", type=int, default=None, help="SSL pretraining epochs per model.")
    parser.add_argument("--probe-epochs", type=int, default=None, help="Person ID linear probe epochs.")
    parser.add_argument("--probe-splits", type=int, default=None, help="Person ID CV split count (5 in paper).")
    parser.add_argument("--survival-epochs", type=int, default=None, help="DeepSurv / Neural MTLR epochs.")
    parser.add_argument("--n-trees", type=int, default=None, help="Survival forest tree count.")
    parser.add_argument("--minirocket-features", type=int, default=None, help="MiniRocket feature count.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_ssl_configs(
    hours: WindowHours, /, *, n_channels: int, seq_len: int, fast: bool, ssl_epochs: int | None, seed: int
) -> tuple[TFCConfig, TSTCCConfig, MixingUpConfig]:
    """Load tuned YAML SSL configs (preserving 4h lr=3e-4, tc_timesteps=9, batch_size=32) with optional overrides."""
    tfc_yaml = _CONFIGS_DIR / "ssl" / f"tfc_{hours}h.yaml"
    tstcc_yaml = _CONFIGS_DIR / "ssl" / f"tstcc_{hours}h.yaml"
    mix_yaml = _CONFIGS_DIR / "ssl" / "mixing_up.yaml"

    tfc_cfg = TFCConfig.from_yaml(tfc_yaml) if tfc_yaml.exists() else TFCConfig()
    tstcc_cfg = TSTCCConfig.from_yaml(tstcc_yaml) if tstcc_yaml.exists() else TSTCCConfig()
    mix_cfg = MixingUpConfig.from_yaml(mix_yaml) if mix_yaml.exists() else MixingUpConfig()

    tfc_cfg.input_channels = n_channels
    tfc_cfg.seq_len = seq_len
    tfc_cfg.seed = seed

    tstcc_cfg.input_channels = n_channels
    tstcc_cfg.seq_len = seq_len
    tstcc_cfg.seed = seed

    mix_cfg.input_channels = n_channels
    mix_cfg.seq_len = seq_len
    mix_cfg.seed = seed

    if fast:
        tfc_cfg.epochs = 2
        tstcc_cfg.epochs = 2
        tstcc_cfg.tc_timesteps = 3
        mix_cfg.epochs = 2
    if ssl_epochs is not None:
        tfc_cfg.epochs = ssl_epochs
        tstcc_cfg.epochs = ssl_epochs
        mix_cfg.epochs = ssl_epochs
    return tfc_cfg, tstcc_cfg, mix_cfg


def _extract_all_four_embeddings(
    pretext_x: NDArray[np.float32],
    downstream_x: NDArray[np.float32],
    /,
    *,
    hours: WindowHours,
    fast: bool,
    ssl_epochs: int | None,
    minirocket_features: int,
    seed: int,
    device: str,
) -> dict[str, NDArray[np.float32]]:
    """Train TS-TCC, TFC, Mixing-up, and fit MiniRocket on ``pretext_x``; extract downstream embeddings."""
    n_channels = int(pretext_x.shape[1])
    seq_len = int(pretext_x.shape[2])
    tfc_cfg, tstcc_cfg, mix_cfg = _load_ssl_configs(
        hours, n_channels=n_channels, seq_len=seq_len, fast=fast, ssl_epochs=ssl_epochs, seed=seed
    )

    tfc_res = pretrain_tfc(pretext_x, tfc_cfg, device=device)
    emb_tfc = extract_embeddings(tfc_res.encoder, downstream_x, device=device)

    tstcc_res = pretrain_tstcc(pretext_x, tstcc_cfg, device=device)
    emb_tstcc = extract_embeddings(tstcc_res.encoder, downstream_x, device=device)

    mix_res = pretrain_mixing_up(pretext_x, mix_cfg, device=device)
    emb_mixup = extract_embeddings(mix_res.encoder, downstream_x, device=device)

    emb_mr = (
        MiniRocketExtractor(num_features=minirocket_features, device=device, seed=seed)
        .fit(pretext_x)
        .transform(downstream_x)
    )

    return {"TS-TCC": emb_tstcc, "TFC": emb_tfc, "MiniRocket": emb_mr, "Mixing-up": emb_mixup}


def _run_table3_ablations(
    bundle: SyntheticWearableDataset,
    base_tfc_emb: NDArray[np.float32],
    /,
    *,
    fast: bool,
    ssl_epochs: int | None,
    surv_cfg: SurvivalConfig,
    seed: int,
    device: str,
) -> pd.DataFrame:
    """Reproduce Paper Table 3 feature importance & static/past-episode covariate ablations."""
    rows: list[dict[str, object]] = []

    def _eval_deepsurv(label: str, emb: NDArray[np.float32], /, *, cfg: SurvivalConfig) -> None:
        model = DeepSurvModel(
            hidden_units=cfg.deepsurv_hidden_units,
            activation=cfg.deepsurv_activation,
            dropout=cfg.deepsurv_dropout,
            lr=cfg.deepsurv_lr,
            weight_decay=cfg.deepsurv_weight_decay,
            epochs=cfg.deepsurv_epochs,
            batch_size=cfg.deepsurv_batch_size,
            device=device,
            seed=cfg.seed,
        )
        res = run_survival_benchmark(
            emb,
            metadata=bundle.metadata,
            time_days=bundle.time_days,
            event=bundle.event,
            embedding_name="TFC",
            static_features=bundle.static_features,
            config=cfg,
            models=[model],
            device=device,
        )
        r0 = res.results.iloc[0]
        rows.append(
            {
                "ablation": label,
                "c_index": float(r0["c_index"]),
                "ibs": float(r0["ibs"]),
                "neg_log_ibs": float(r0["neg_log_ibs"]),
            }
        )

    base_cfg = dataclasses.replace(
        surv_cfg, include_user_id=True, include_static=False, include_past_episodes=False, seed=seed
    )
    _eval_deepsurv("baseline", base_tfc_emb, cfg=base_cfg)

    no_id_cfg = dataclasses.replace(base_cfg, include_user_id=False)
    _eval_deepsurv("- user's-ID", base_tfc_emb, cfg=no_id_cfg)

    modality_specs = (
        ("- heart rate", "heart_rate"),
        ("- hour-of-day", "hour_of_day"),
        ("- accelerometer", "accelerometer"),
        ("- gyroscope", "gyroscope"),
    )
    for display_label, group_name in modality_specs:
        keep_idx = select_channel_indices([group_name])
        sub_pretext = bundle.pretext_x[:, keep_idx, :]
        sub_downstream = bundle.downstream_x[:, keep_idx, :]
        cfg_sub, _, _ = _load_ssl_configs(
            12,
            n_channels=len(keep_idx),
            seq_len=int(sub_pretext.shape[2]),
            fast=fast,
            ssl_epochs=ssl_epochs,
            seed=seed,
        )
        res_sub = pretrain_tfc(sub_pretext, cfg_sub, device=device)
        emb_sub = extract_embeddings(res_sub.encoder, sub_downstream, device=device)
        _eval_deepsurv(display_label, emb_sub, cfg=base_cfg)

    covariate_specs = (("+ st", True, False), ("+ ps, p_ep, ll", False, True), ("+ st, ps, p_ep, ll", True, True))
    for display_label, use_static, use_past in covariate_specs:
        cov_cfg = dataclasses.replace(base_cfg, include_static=use_static, include_past_episodes=use_past)
        _eval_deepsurv(display_label, base_tfc_emb, cfg=cov_cfg)

    return compute_ablation_relative_change(pd.DataFrame(rows), baseline_label="baseline")


def _export_final_risk_and_distribution_figures(
    bundle: SyntheticWearableDataset,
    tfc_embeddings: NDArray[np.float32],
    /,
    *,
    final_cfg: SurvivalConfig,
    out_dir: Path,
    device: str,
) -> None:
    """Run best-config survival analysis (static + past episodes) and export Fig. 4 & diagnostics."""
    final_bench = run_survival_benchmark(
        tfc_embeddings,
        metadata=bundle.metadata,
        time_days=bundle.time_days,
        event=bundle.event,
        embedding_name="TFC",
        window_hours=12,
        static_features=bundle.static_features,
        config=final_cfg,
        device=device,
    )
    best_model = final_bench.fitted_models["DeepSurv"]
    train_risk = best_model.predict_risk(final_bench.x_train)
    _, train_cutoffs = stratify_risk_groups(train_risk, quantiles=final_cfg.risk_quantiles)
    test_risk = best_model.predict_risk(final_bench.x_test)
    risk_tiers, (low_cut, high_cut) = stratify_risk_groups(test_risk, cutoffs=train_cutoffs)

    fig_surv_dist = plot_survival_time_distribution(
        time_train=bundle.time_days[final_bench.train_indices],
        event_train=bundle.event[final_bench.train_indices],
        time_test=bundle.time_days[final_bench.test_indices],
        event_test=bundle.event[final_bench.test_indices],
        output_path=out_dir / "survival_distribution.pdf",
    )
    plt.close(fig_surv_dist)

    plot_risk_classification_timeline(
        bundle.metadata.iloc[final_bench.test_indices].reset_index(drop=True),
        bundle.relapse_episodes,
        risk_tiers,
        max_users=5,
        seed=final_cfg.seed,
        output_path=out_dir / "final_risk_classification.pdf",
    )

    fig_risk_hist = plot_risk_group_histogram(
        test_risk, risk_tiers, cutoffs=(low_cut, high_cut), output_path=out_dir / "risk_groups_histogram.pdf"
    )
    plt.close(fig_risk_hist)

    fig_risk_curves = plot_risk_group_survival_curves(
        best_model,
        final_bench.x_test,
        time_test=bundle.time_days[final_bench.test_indices],
        event_test=bundle.event[final_bench.test_indices],
        risk_labels=risk_tiers,
        seed=final_cfg.seed,
        output_path=out_dir / "risk_survival_curves.pdf",
    )
    plt.close(fig_risk_curves)


def _export_person_id_artifacts(
    bundle_4h: SyntheticWearableDataset,
    embeddings_4h: dict[str, NDArray[np.float32]],
    /,
    *,
    fast: bool,
    ssl_epochs: int | None,
    minirocket_features: int,
    min_samples: int,
    probe_splits: int,
    probe_epochs: int,
    seed: int,
    device: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Run 4-hour Person Identification proxy benchmark and export Table 2 & ``identification.pdf``."""
    pid_summaries: dict[str, PersonIDCVSummary] = {}
    n_ch = int(bundle_4h.pretext_x.shape[1])
    seq_len = int(bundle_4h.pretext_x.shape[2])

    for emb_name in list(embeddings_4h):
        emb_matrix = embeddings_4h[emb_name]
        folds: list[NDArray[np.float32]] | None = None
        if not fast and probe_splits > 1:
            folds = [emb_matrix]
            for i in range(1, probe_splits):
                tfc_c, tstcc_c, mix_c = _load_ssl_configs(
                    4, n_channels=n_ch, seq_len=seq_len, fast=fast, ssl_epochs=ssl_epochs, seed=seed + i
                )
                if emb_name == "MiniRocket":
                    folds.append(
                        MiniRocketExtractor(num_features=minirocket_features, device=device, seed=seed + i)
                        .fit(bundle_4h.pretext_x)
                        .transform(bundle_4h.downstream_x)
                    )
                elif emb_name == "TFC":
                    r_t = pretrain_tfc(bundle_4h.pretext_x, tfc_c, device=device)
                    folds.append(extract_embeddings(r_t.encoder, bundle_4h.downstream_x, device=device))
                elif emb_name == "TS-TCC":
                    r_ts = pretrain_tstcc(bundle_4h.pretext_x, tstcc_c, device=device)
                    folds.append(extract_embeddings(r_ts.encoder, bundle_4h.downstream_x, device=device))
                else:
                    r_m = pretrain_mixing_up(bundle_4h.pretext_x, mix_c, device=device)
                    folds.append(extract_embeddings(r_m.encoder, bundle_4h.downstream_x, device=device))
            pid_input: NDArray[np.float32] | list[NDArray[np.float32]] = folds
        else:
            pid_input = emb_matrix
        summary = run_person_id_evaluation(
            pid_input,
            bundle_4h.downstream_user_ids,
            min_samples_per_user=min_samples,
            n_splits=probe_splits,
            epochs=probe_epochs,
            device=device,
            seed=seed,
        )
        pid_summaries[emb_name] = summary
        if folds is not None and summary.per_split_f1:
            best_fold = int(np.argmax(summary.per_split_f1))
            embeddings_4h[emb_name] = folds[best_fold]
    table2_df = format_person_id_table(pid_summaries)
    table2_df.to_csv(out_dir / "table2_person_id.csv")
    (out_dir / "table2_person_id.tex").write_text(table2_df.to_latex(), encoding="utf-8")

    pid_metrics_df = person_id_summaries_to_frame(pid_summaries)
    fig_pid = plot_person_id_metrics(pid_metrics_df, output_path=out_dir / "identification.pdf")
    plt.close(fig_pid)
    return table2_df


def main() -> None:
    """Run the full synthetic benchmark and export all IEEE ICASSP 2023 figures and tables."""
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_epochs = args.probe_epochs if args.probe_epochs is not None else (15 if args.fast else 100)
    probe_splits = args.probe_splits if args.probe_splits is not None else (3 if args.fast else 5)
    surv_epochs = args.survival_epochs if args.survival_epochs is not None else (30 if args.fast else 1000)
    n_trees = args.n_trees if args.n_trees is not None else (25 if args.fast else 200)
    mr_features = args.minirocket_features if args.minirocket_features is not None else (336 if args.fast else 9996)
    hidden_units = 48 if args.fast else 150
    mtlr_bins = 20 if args.fast else 150
    surv_lr = 1e-3 if args.fast else 1e-4
    seq_len_map: dict[WindowHours, int | None] = (
        {4: 96, 8: 144, 12: 192} if args.fast else {4: None, 8: None, 12: None}
    )

    surv_cfg = SurvivalConfig(
        include_user_id=True,
        include_static=False,
        include_past_episodes=False,
        deepsurv_hidden_units=hidden_units,
        deepsurv_lr=surv_lr,
        deepsurv_epochs=surv_epochs,
        mtlr_hidden_units=hidden_units,
        mtlr_bins=mtlr_bins,
        mtlr_lr=surv_lr,
        mtlr_epochs=surv_epochs,
        n_trees=n_trees,
        seed=args.seed,
    )

    bundles: dict[WindowHours, SyntheticWearableDataset] = {}
    embeddings_by_duration: dict[WindowHours, dict[str, NDArray[np.float32]]] = {}
    all_survival_frames: list[pd.DataFrame] = []

    for hours in _WINDOW_DURATIONS:
        print(f"[{hours}h] Generating synthetic wearable cohort and pretraining all 4 embeddings...")
        bundle = generate_synthetic_wearable_dataset(
            window_hours=hours,
            seq_len=seq_len_map[hours],
            n_pretext_users=args.n_pretext_users,
            n_downstream_users=args.n_downstream_users,
            samples_per_user=args.samples_per_user,
            seed=args.seed + int(hours),
        )
        bundles[hours] = bundle
        embeddings_by_duration[hours] = _extract_all_four_embeddings(
            bundle.pretext_x,
            bundle.downstream_x,
            hours=hours,
            fast=args.fast,
            ssl_epochs=args.ssl_epochs,
            minirocket_features=mr_features,
            seed=args.seed,
            device=args.device,
        )

    table2_df = _export_person_id_artifacts(
        bundles[4],
        embeddings_by_duration[4],
        fast=args.fast,
        ssl_epochs=args.ssl_epochs,
        minirocket_features=mr_features,
        min_samples=max(6, int(args.samples_per_user * 0.4)),
        probe_splits=probe_splits,
        probe_epochs=probe_epochs,
        seed=args.seed,
        device=args.device,
        out_dir=out_dir,
    )

    for hours in _WINDOW_DURATIONS:
        bundle = bundles[hours]
        for emb_name, emb_matrix in embeddings_by_duration[hours].items():
            bench = run_survival_benchmark(
                emb_matrix,
                metadata=bundle.metadata,
                time_days=bundle.time_days,
                event=bundle.event,
                embedding_name=emb_name,
                window_hours=hours,
                static_features=bundle.static_features,
                config=surv_cfg,
                device=args.device,
            )
            all_survival_frames.append(bench.results)

    combined_results = pd.concat(all_survival_frames, ignore_index=True)
    combined_results.to_csv(out_dir / "survival_results_all_durations.csv", index=False)

    bundle_8h = bundles[_DEFAULT_PAPER_DURATION]
    plot_cohort_relapse_timeline(
        bundle_8h.metadata, bundle_8h.relapse_episodes, seed=args.seed, output_path=out_dir / "chart.pdf"
    )
    plt.close(
        plot_user_sample_distribution(bundle_8h.downstream_user_ids, output_path=out_dir / "identification_dist.pdf")
    )

    plt.close(plot_per_embedding_comparison(combined_results, output_path=out_dir / "per_embedding.pdf"))
    plt.close(
        plot_per_survival_model_comparison(
            combined_results, sample_length="8-hour Dataset", output_path=out_dir / "per_smodel.pdf"
        )
    )

    bundle_12h = bundles[12]
    table3_df = _run_table3_ablations(
        bundle_12h,
        embeddings_by_duration[12]["TFC"],
        fast=args.fast,
        ssl_epochs=args.ssl_epochs,
        surv_cfg=surv_cfg,
        seed=args.seed,
        device=args.device,
    )
    table3_df.to_csv(out_dir / "table3_ablations.csv", index=False)
    (out_dir / "table3_ablations.tex").write_text(
        table3_df.set_index("ablation").round(2).to_latex(), encoding="utf-8"
    )

    final_cfg = dataclasses.replace(surv_cfg, include_static=True, include_past_episodes=True)
    _export_final_risk_and_distribution_figures(
        bundle_12h, embeddings_by_duration[12]["TFC"], final_cfg=final_cfg, out_dir=out_dir, device=args.device
    )

    print(f"\nExported all publication figures & tables to {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        print(f"  - {p.name}")
    print("\nTable 2 (Synthetic Person Identification):")
    print(table2_df.to_string())
    print("\nTable 3 (Synthetic DeepSurv Relative Change %):")
    print(table3_df.to_string(index=False))


if __name__ == "__main__":
    main()

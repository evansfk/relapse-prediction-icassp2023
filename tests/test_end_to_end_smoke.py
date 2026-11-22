"""End-to-end smoke tests proving the full ICASSP 2023 pipeline runs on synthetic data."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from relapse_prediction.config import MixingUpConfig, SurvivalConfig, TFCConfig, TSTCCConfig, select_channel_indices
from relapse_prediction.data.datasets import compute_fft_magnitude
from relapse_prediction.data.synthetic import SyntheticWearableDataset, generate_synthetic_wearable_dataset
from relapse_prediction.models.minirocket import MiniRocketExtractor
from relapse_prediction.training.evaluator import run_person_id_evaluation, run_survival_benchmark
from relapse_prediction.training.ssl_trainer import (
    extract_embeddings,
    pretrain_mixing_up,
    pretrain_tfc,
    pretrain_tstcc,
)
from relapse_prediction.utils.plotting import plot_mts, plot_survival_comparison


@pytest.fixture(scope="module")
def synthetic_bundle() -> SyntheticWearableDataset:
    return generate_synthetic_wearable_dataset(
        window_hours=4, seq_len=120, n_pretext_users=10, n_downstream_users=8, samples_per_user=16, seed=42
    )


def test_end_to_end_tfc_person_id_and_survival_pipeline(synthetic_bundle: SyntheticWearableDataset) -> None:
    # 1. Pretrain TFC on unlabeled pretext cohort (with Table 3 modality subset option).
    keep_idx = select_channel_indices(["gyroscope"])
    pretext_sub = synthetic_bundle.pretext_x[:, keep_idx, :]
    downstream_sub = synthetic_bundle.downstream_x[:, keep_idx, :]

    tfc_cfg = TFCConfig(input_channels=len(keep_idx), seq_len=pretext_sub.shape[2], batch_size=32, epochs=4, seed=42)
    pretrain_res = pretrain_tfc(pretext_sub, tfc_cfg, device="cpu")
    assert len(pretrain_res.epoch_losses) == 4
    assert all(np.isfinite(loss) and loss > 0.0 for loss in pretrain_res.epoch_losses)

    # Regression C1: TFC consistency objective must pull z_t and z_f together (positive cosine similarity).
    pretrain_res.encoder.eval()
    with torch.no_grad():
        x_eval = torch.from_numpy(downstream_sub[:32])
        _, z_t, _, z_f = pretrain_res.encoder(x_eval, compute_fft_magnitude(x_eval))
        mean_cos = float(F.cosine_similarity(z_t, z_f, dim=1).mean().item())
    assert mean_cos > 0.0, f"TFC cross-domain projections collapsed antipodally: cos={mean_cos:.3f}"

    # Verify clamped_margin mode also keeps loss finite and positive.
    tfc_margin_res = pretrain_tfc(
        pretext_sub,
        TFCConfig(
            input_channels=len(keep_idx),
            seq_len=pretext_sub.shape[2],
            batch_size=32,
            epochs=2,
            consistency_mode="clamped_margin",
            seed=42,
        ),
        device="cpu",
    )
    assert all(np.isfinite(loss) and loss > 0.0 for loss in tfc_margin_res.epoch_losses)

    # 2. Extract frozen 256-dim TFC embeddings (and verify empty-input handling).
    embeddings = extract_embeddings(pretrain_res.encoder, downstream_sub, batch_size=64, device="cpu")
    assert embeddings.shape == (len(downstream_sub), 256)
    assert np.isfinite(embeddings).all()
    empty_emb = extract_embeddings(pretrain_res.encoder, downstream_sub[:0], device="cpu")
    assert empty_emb.shape == (0, 256)

    # 3. Proxy Task: Person Identification (both single-matrix and multi-seed Table 2 protocol).
    pid_summary = run_person_id_evaluation(
        [embeddings, embeddings],
        synthetic_bundle.downstream_user_ids,
        min_samples_per_user=6,
        n_splits=2,
        epochs=10,
        batch_size=32,
        seed=42,
    )
    assert pid_summary.n_users_retained >= 2
    assert 0.0 <= pid_summary.macro_f1_mean <= 1.0
    assert 0.0 <= pid_summary.macro_auprc_mean <= 1.0

    # 4. Downstream Survival Analysis across all 4 models with static & past-episode features.
    surv_cfg = SurvivalConfig(
        include_user_id=True,
        include_static=True,
        include_past_episodes=True,
        deepsurv_hidden_units=32,
        deepsurv_epochs=15,
        mtlr_hidden_units=32,
        mtlr_bins=15,
        mtlr_epochs=15,
        n_trees=20,
        seed=42,
    )
    benchmark = run_survival_benchmark(
        embeddings,
        metadata=synthetic_bundle.metadata,
        time_days=synthetic_bundle.time_days,
        event=synthetic_bundle.event,
        embedding_name="TFC",
        static_features=synthetic_bundle.static_features,
        config=surv_cfg,
        device="cpu",
    )
    assert set(benchmark.results["survival_model"]) == {
        "DeepSurv",
        "NeuralMTLR",
        "ExtraSurvivalTrees",
        "RandomSurvivalForest",
    }
    assert (benchmark.results["c_index"] >= 0.0).all()
    assert (benchmark.results["c_index"] <= 1.0).all()
    assert np.isfinite(benchmark.results["neg_log_ibs"]).all()

    # 5. Plotting smoke test (non-interactive Agg backend).
    fig_mts = plot_mts([synthetic_bundle.pretext_x[0]], titles=["Synthetic 4h"], show=False)
    fig_surv = plot_survival_comparison(benchmark.results, show=False)
    plt.close(fig_mts)
    plt.close(fig_surv)


def test_end_to_end_tstcc_mixingup_and_minirocket_smoke(synthetic_bundle: SyntheticWearableDataset) -> None:
    pretext = synthetic_bundle.pretext_x
    downstream = synthetic_bundle.downstream_x
    seq_len = pretext.shape[2]

    # TS-TCC pretraining + embedding extraction.
    tstcc_cfg = TSTCCConfig(input_channels=10, seq_len=seq_len, batch_size=32, epochs=2, tc_timesteps=3, seed=42)
    tstcc_res = pretrain_tstcc(pretext, tstcc_cfg, device="cpu")
    tstcc_emb = extract_embeddings(tstcc_res.encoder, downstream, device="cpu")
    assert tstcc_emb.shape[0] == len(downstream)
    assert np.isfinite(tstcc_emb).all()

    # Mixing-Up pretraining + embedding extraction.
    mix_cfg = MixingUpConfig(input_channels=10, seq_len=seq_len, batch_size=32, epochs=2, seed=42)
    mix_res = pretrain_mixing_up(pretext, mix_cfg, device="cpu")
    mix_emb = extract_embeddings(mix_res.encoder, downstream, device="cpu")
    assert mix_emb.shape == (len(downstream), 128)
    assert np.isfinite(mix_emb).all()

    # MiniRocket baseline fitted strictly on unlabeled pretext data (no downstream leakage).
    mr_emb = MiniRocketExtractor(num_features=252, seed=42).fit(pretext).transform(downstream)
    assert mr_emb.shape == (len(downstream), 252)
    assert np.isfinite(mr_emb).all()

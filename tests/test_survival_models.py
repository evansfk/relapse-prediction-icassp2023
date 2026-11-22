"""Unit tests for survival models, survival/classification metrics, features, and risk groups."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from sksurv.ensemble import ExtraSurvivalTrees

from relapse_prediction.evaluation.classification import compute_classification_metrics
from relapse_prediction.evaluation.survival_metrics import (
    _valid_eval_times,
    compute_c_index,
    compute_ibs_and_brier,
    evaluate_survival_model,
    stratify_risk_groups,
    to_structured_survival,
)
from relapse_prediction.models.linear_probe import train_linear_probe
from relapse_prediction.survival.deepsurv import DeepSurvModel
from relapse_prediction.survival.features import assemble_survival_features
from relapse_prediction.survival.neural_mtlr import NeuralMTLRModel
from relapse_prediction.survival.tree_models import ExtraSurvivalTreesModel, RandomSurvivalForestModel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from relapse_prediction.survival.base import BaseSurvivalModel


def _make_correlated_survival_data(
    *, n: int, d: int, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.bool_]]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    true_risk = 1.5 * x[:, 0] - 1.2 * x[:, 1] + 0.6 * x[:, 2]
    event_time = rng.exponential(scale=np.exp(-0.5 * true_risk) * 40.0) + 1.0
    censor_time = rng.uniform(20.0, 90.0, size=n)
    event = event_time <= censor_time
    obs_time = np.minimum(event_time, censor_time)
    return x, obs_time, event


@pytest.mark.parametrize(
    "model",
    [
        DeepSurvModel(hidden_units=32, lr=1e-3, epochs=40, batch_size=64, seed=42),
        DeepSurvModel(hidden_units=32, lr=2e-3, epochs=60, batch_size=None, seed=42),
        NeuralMTLRModel(hidden_units=32, bins=20, binning="uniform", lr=2e-3, epochs=60, batch_size=None, seed=42),
        NeuralMTLRModel(hidden_units=32, bins=20, binning="quantile", lr=1e-3, epochs=40, batch_size=64, seed=42),
        ExtraSurvivalTreesModel(n_estimators=40, max_depth=5, min_samples_leaf=5, max_samples=0.63, n_jobs=1, seed=42),
        RandomSurvivalForestModel(
            n_estimators=40, max_depth=5, min_samples_leaf=5, max_samples=0.63, n_jobs=1, seed=42
        ),
    ],
)
def test_survival_model_learns_signal_and_respects_boundary_conditions(model: BaseSurvivalModel) -> None:
    x, t, e = _make_correlated_survival_data(n=240, d=8, seed=42)
    x_tr, x_te = x[:150], x[150:]
    t_tr, t_te = t[:150], t[150:]
    e_tr, e_te = e[:150], e[150:]

    fitted = model.fit(x_tr, time=t_tr, event=e_tr)
    assert fitted is model

    # Regression C9: strictly prior to earliest event time (t = 0), S(0)=1 and H(0)=0 for ALL models.
    t_zero = np.array([0.0])
    s0 = model.predict_survival(x_te[:5], times=t_zero)
    h0 = model.predict_cumulative_hazard(x_te[:5], times=t_zero)
    np.testing.assert_allclose(s0, 1.0, atol=1e-6)
    np.testing.assert_allclose(h0, 0.0, atol=1e-6)

    eval_grid = np.linspace(float(t_te.min()) + 1.0, float(t_te.max()) - 1.0, 15)
    surv = model.predict_survival(x_te, times=eval_grid)

    assert surv.shape == (len(x_te), len(eval_grid))
    assert np.all((surv >= 0.0) & (surv <= 1.0))
    assert np.all(np.diff(surv, axis=1) <= 1e-6)

    metrics = evaluate_survival_model(
        model,
        x_te,
        time_train=t_tr,
        event_train=e_tr,
        time_test=t_te,
        event_test=e_te,
        num_time_points=20,
        ibs_mode="paper",
    )
    assert 0.60 < metrics.c_index <= 1.0, f"{model.name} achieved weak C-index {metrics.c_index:.3f}"
    assert 0.0 < metrics.ibs < 0.30
    assert np.isfinite(metrics.neg_log_ibs)

    metrics_q = evaluate_survival_model(
        model,
        x_te,
        time_train=t_tr,
        event_train=e_tr,
        time_test=t_te,
        event_test=e_te,
        num_time_points=20,
        ibs_mode="quantile",
    )
    assert 0.0 < metrics_q.ibs < 0.30


def test_c_index_valid_eval_times_and_risk_stratification() -> None:
    times = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    events = np.array([True, True, True, False, True, True])
    perfect_risk = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    c_idx, _, _, _ = compute_c_index(time=times, event=events, risk_scores=perfect_risk)
    assert np.isclose(c_idx, 1.0)

    tiers, cutoffs = stratify_risk_groups(perfect_risk)
    assert cutoffs[0] < cutoffs[1]
    assert set(tiers.tolist()) == {"low", "medium", "high"}

    # Applying pre-fitted train cutoffs & handling constant/tied risk scores cleanly.
    applied_tiers, _ = stratify_risk_groups(np.array([1.0, 3.5, 6.0]), cutoffs=cutoffs)
    assert applied_tiers.tolist() == ["low", "medium", "high"]
    tied_tiers, _ = stratify_risk_groups(np.full(5, 2.5))
    assert set(tied_tiers.tolist()) == {"medium"}

    # Regression C8: _valid_eval_times fallback respects training time upper bound.
    t_tr = np.array([5.0, 10.0, 15.0, 20.0])
    t_te = np.array([19.5, 25.0, 30.0])
    grid = _valid_eval_times(t_tr, t_te, num_points=10, ibs_mode="paper")
    assert float(grid.min()) > float(t_tr.min())
    assert float(grid.max()) < float(t_tr.max())


def test_classification_metrics_binary_macro_auprc_and_linear_probe_determinism() -> None:
    # Regression C14: binary macro-AUPRC must average class 0 and class 1 AP on imbalanced data.
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    probs = np.column_stack([np.linspace(0.9, 0.1, 10), np.linspace(0.1, 0.9, 10)])
    m = compute_classification_metrics(y_true, probs, num_classes=2)
    assert np.isclose(m.macro_auprc, 1.0)

    rng = np.random.default_rng(7)
    x = rng.standard_normal((60, 8)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    r1 = train_linear_probe(
        x_train=x[:36],
        y_train=y[:36],
        x_val=x[36:48],
        y_val=y[36:48],
        x_test=x[48:],
        y_test=y[48:],
        num_classes=2,
        epochs=15,
        seed=123,
    )
    r2 = train_linear_probe(
        x_train=x[:36],
        y_train=y[:36],
        x_val=x[36:48],
        y_val=y[36:48],
        x_test=x[48:],
        y_test=y[48:],
        num_classes=2,
        epochs=15,
        seed=123,
    )
    assert np.isclose(r1.test_metrics.macro_f1, r2.test_metrics.macro_f1)
    assert np.isclose(r1.test_metrics.macro_auprc, r2.test_metrics.macro_auprc)


def test_assemble_survival_features_table3_strict_and_edge_cases() -> None:
    emb_tr = np.ones((6, 16), dtype=np.float32)
    emb_te = np.ones((4, 16), dtype=np.float32)
    meta_tr = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2, 2],  # int user IDs vs str static index
            "past_episodes": [0, 1, 0, 2, 1, 1],
            "last_episode_level": [0, 2, 0, 3, 1, 1],
            "last_episode_psychotic": [0, 2, 0, 2, 1, 1],
            "event_free_days": [0.0, 12.0, 0.0, 5.0, 30.0, 45.0],
        }
    )
    meta_te = pd.DataFrame(
        {
            "user_id": [0, 1, 2, 99],
            "past_episodes": [1, 2, 0, 0],
            "last_episode_level": [2, 3, 0, 0],
            "last_episode_psychotic": [2, 2, 0, 0],
            "event_free_days": [20.0, 10.0, 0.0, 0.0],
        }
    )
    static_df = pd.DataFrame(
        {
            "age": [25.0, 34.0, 41.0],
            "all_nan_col": [np.nan, np.nan, np.nan],
            "diagnosis": ["A", "B", "A"],
            "smoker": pd.Series([True, False, True], index=["0", "1", "2"], dtype="boolean"),
        },
        index=pd.Index(["0", "1", "2"], name="user_id"),
    )

    # Regression C7: event_free_days is excluded by default for Table 3 fidelity.
    feats_table3 = assemble_survival_features(
        emb_train=emb_tr,
        emb_test=emb_te,
        meta_train=meta_tr,
        meta_test=meta_te,
        include_user_id=True,
        include_past_episodes=True,
        static_features=static_df,
        include_static=True,
    )
    assert "event_free_days" not in feats_table3.feature_names
    assert {"past_episodes", "last_episode_level", "last_episode_psychotic"}.issubset(feats_table3.feature_names)
    assert feats_table3.x_train.shape[1] == feats_table3.x_test.shape[1] == len(feats_table3.feature_names)
    assert not np.isnan(feats_table3.x_train).any()
    assert not np.isnan(feats_table3.x_test).any()

    # Opt-in event_free_days and drop_first=False work when requested.
    feats_with_efd = assemble_survival_features(
        emb_train=emb_tr,
        emb_test=emb_te,
        meta_train=meta_tr,
        meta_test=meta_te,
        include_user_id=True,
        include_past_episodes=True,
        include_event_free_days=True,
        drop_first=False,
    )
    assert "event_free_days" in feats_with_efd.feature_names
    assert sum(name.startswith("uid_") for name in feats_table3.feature_names) == 2  # K - 1 = 2 with drop_first=True
    assert sum(name.startswith("uid_") for name in feats_with_efd.feature_names) == 3  # K = 3 with drop_first=False


def test_compute_ibs_handles_test_times_exceeding_max_train_time() -> None:
    t_tr = np.array([5.0, 10.0, 20.0, 35.0, 50.0])
    e_tr = np.array([True, False, True, False, True])
    # Test contains an uncensored event strictly beyond max(t_tr) = 50.0.
    t_te = np.array([8.0, 15.0, 30.0, 65.0])
    e_te = np.array([True, False, True, True])
    eval_t = _valid_eval_times(t_tr, t_te, num_points=10, ibs_mode="paper")
    surv_probs = np.full((len(t_te), len(eval_t)), 0.5, dtype=np.float64)

    ibs, neg_log_ibs, curve = compute_ibs_and_brier(
        time_train=t_tr,
        event_train=e_tr,
        time_test=t_te,
        event_test=e_te,
        survival_probs=surv_probs,
        eval_times=eval_t,
    )
    assert 0.0 < ibs < 1.0
    assert np.isfinite(neg_log_ibs)
    assert curve.shape == eval_t.shape

    # Regression B-H5: all-censored test split and administrative tail censoring succeed without sksurv crash.
    ibs_cens, _, curve_cens = compute_ibs_and_brier(
        time_train=t_tr,
        event_train=e_tr,
        time_test=np.array([12.0, 25.0, 55.0, 70.0]),
        event_test=np.array([False, False, False, False]),
        survival_probs=surv_probs,
        eval_times=eval_t,
    )
    assert 0.0 < ibs_cens < 1.0
    assert np.isfinite(curve_cens).all()

    # Regression CORR-A-H2: forest predict_risk matches scikit-survival estimator.predict (event-only CHF sum).
    x_f, t_f, e_f = _make_correlated_survival_data(n=80, d=6, seed=11)
    est = ExtraSurvivalTreesModel(n_estimators=25, max_depth=4, min_samples_leaf=3, n_jobs=1, seed=11).fit(
        x_f[:50], time=t_f[:50], event=e_f[:50]
    )
    ref_est = ExtraSurvivalTrees(
        n_estimators=25,
        max_depth=4,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        bootstrap=True,
        max_samples=0.63,
        n_jobs=1,
        random_state=11,
    ).fit(x_f[:50].astype(np.float64), to_structured_survival(time=t_f[:50], event=e_f[:50]))
    np.testing.assert_allclose(est.predict_risk(x_f[50:]), ref_est.predict(x_f[50:].astype(np.float64)))

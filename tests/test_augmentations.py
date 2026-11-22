"""Unit tests for augmentations, preprocessing, windowing, and right-censoring labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from torch.utils.data import DataLoader

from relapse_prediction.config import CHANNEL_NAMES, select_channel_indices
from relapse_prediction.data.augmentations import (
    JitterTransform,
    PermutationTransform,
    ScalingTransform,
    TFCFrequencyAugmentation,
    TFCTimeAugmentation,
    TSTCCAugmentation,
    add_frequency,
    remove_frequency,
)
from relapse_prediction.data.datasets import TSTCCDataset
from relapse_prediction.data.preprocessing import (
    append_time_of_day_features,
    build_survival_labels,
    clean_wearable_signals,
    encode_cyclical_time,
    extract_consecutive_windows,
    standardize_per_sample,
)


def test_time_domain_transforms_preserve_shape_and_handle_2d() -> None:
    rng = np.random.default_rng(0)
    x_3d = rng.standard_normal((8, 10, 60)).astype(np.float32)
    x_2d = x_3d[0]

    for tfm in [
        JitterTransform(sigma=0.1, seed=1),
        ScalingTransform(sigma=0.2, loc=1.0, seed=2),
        PermutationTransform(max_segments=5, seed=3),
    ]:
        out_3d = tfm(x_3d)
        assert out_3d.shape == x_3d.shape
        assert out_3d.dtype == np.float32
        assert not np.allclose(out_3d, x_3d)

        out_2d = tfm(x_2d)
        assert out_2d.shape == (1, 10, 60)

    with pytest.raises(TypeError):
        JitterTransform()([[1.0, 2.0]])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Expected 2D or 3D"):
        JitterTransform()(np.zeros((4,)))


def test_permutation_preserves_channel_multiset() -> None:
    x = np.arange(2 * 3 * 40, dtype=np.float32).reshape(2, 3, 40)
    permuted = PermutationTransform(max_segments=6, seed=7)(x)
    assert permuted.shape == x.shape
    for b in range(2):
        for c in range(3):
            np.testing.assert_array_equal(np.sort(permuted[b, c]), np.sort(x[b, c]))


def test_tstcc_and_tfc_augmentation_banks_and_batch_independence() -> None:
    x = torch.randn(4, 10, 64)
    weak, strong = TSTCCAugmentation(seed=42)(x)
    assert weak.shape == x.shape
    assert strong.shape == x.shape

    x_t_aug = TFCTimeAugmentation(seed=42)(x)
    assert x_t_aug.shape == x.shape

    x_f = torch.fft.fft(x, dim=-1).abs() / x.shape[-1]
    assert remove_frequency(x_f, mask_ratio=0.1).shape == x_f.shape
    assert add_frequency(x_f, perturb_ratio=0.1).shape == x_f.shape

    # Regression C13: add_frequency must scale per sample/channel, not across the full batch.
    x_f_perturbed = x_f.clone()
    x_f_perturbed[0] *= 1e5
    gen_a = torch.Generator().manual_seed(99)
    gen_b = torch.Generator().manual_seed(99)
    out_a = add_frequency(x_f, perturb_ratio=0.5, generator=gen_a)
    out_b = add_frequency(x_f_perturbed, perturb_ratio=0.5, generator=gen_b)
    torch.testing.assert_close(out_a[1:], out_b[1:])

    # Deterministic TFCFrequencyAugmentation when seeded.
    f1 = TFCFrequencyAugmentation(seed=11)(x_f)
    f2 = TFCFrequencyAugmentation(seed=11)(x_f)
    torch.testing.assert_close(f1, f2)

    # Regression CORR-A-H1 / B-H2: multi-worker DataLoader produces distinct augmentations across workers AND epochs.
    ds = TSTCCDataset(
        torch.ones(8, 4, 32), jitter_scale_ratio=1.1, scale_loc=1.0, jitter_ratio=0.3, max_seg=6, seed=42
    )
    gen = torch.Generator().manual_seed(7)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2, generator=gen)
    ep1 = torch.cat([w for _, w, _ in loader], dim=0)
    ep2 = torch.cat([w for _, w, _ in loader], dim=0)
    # Worker 0 (first batch) and Worker 1 (second batch) must not draw identical random scales:
    assert not torch.allclose(ep1[:2], ep1[2:4])
    # Epoch 2 must not replay Epoch 1's exact random draws:
    assert not torch.allclose(ep1, ep2)


def test_cyclical_time_encoding_and_channel_selection() -> None:
    idx = pd.date_range("2022-01-01 00:00:00", periods=24, freq="h", tz="UTC")
    sin_h, cos_h = encode_cyclical_time(idx, freq="hour")
    assert sin_h.shape == (24,)
    np.testing.assert_allclose(sin_h**2 + cos_h**2, np.ones(24), atol=1e-6)
    assert np.isclose(sin_h[0], 0.0, atol=1e-6)
    assert np.isclose(cos_h[0], 1.0, atol=1e-6)
    assert np.isclose(sin_h[6], 1.0, atol=1e-6)

    # Regression B-M3: local circadian wall-clock hour is preserved (midnight local -> sin=0, cos=1).
    local_idx = pd.date_range("2022-01-01 00:00:00", periods=4, freq="h", tz="Europe/Athens")
    sin_loc, cos_loc = encode_cyclical_time(local_idx, freq="hour")
    assert np.isclose(sin_loc[0], 0.0, atol=1e-6)
    assert np.isclose(cos_loc[0], 1.0, atol=1e-6)

    df = pd.DataFrame({"heart_rate": np.full(24, 70.0)}, index=idx)
    enriched = append_time_of_day_features(df)
    assert {"hour_sin", "hour_cos"}.issubset(enriched.columns)

    no_gyr = select_channel_indices(["gyroscope"])
    assert no_gyr == [0, 1, 2, 6, 7, 8, 9]
    assert len(select_channel_indices()) == len(CHANNEL_NAMES)
    with pytest.raises(ValueError, match="Unknown channel groups"):
        select_channel_indices(["invalid_sensor"])


def test_clean_wearable_signals_gap_isolation_multirate_rr_and_windowing() -> None:
    # 9 hours: [00:00..04:00) valid, [04:00..05:00) charging gap (dropped), [05:00..09:00) valid.
    idx1 = pd.date_range("2022-01-01 00:00:00", periods=240, freq="min")
    idx2 = pd.date_range("2022-01-01 05:00:00", periods=240, freq="min")
    idx = idx1.append(idx2)
    rng = np.random.default_rng(12)
    raw = pd.DataFrame(
        {
            "linacc_x": rng.normal(0.0, 1.0, size=len(idx)),
            "linacc_y": rng.normal(0.0, 1.0, size=len(idx)),
            "linacc_z": rng.normal(0.0, 1.0, size=len(idx)),
            "gyr_x": rng.normal(0.0, 10.0, size=len(idx)),
            "gyr_y": rng.normal(0.0, 10.0, size=len(idx)),
            "gyr_z": rng.normal(0.0, 10.0, size=len(idx)),
            "heart_rate": rng.normal(72.0, 4.0, size=len(idx)),
            "rr_interval": rng.normal(830.0, 30.0, size=len(idx)),
        },
        index=idx,
    )
    # Inject single-point outliers that should be filtered and linearly interpolated.
    raw.loc[idx[5], "linacc_x"] = 999.0
    raw.loc[idx[10], "rr_interval"] = 50.0  # < 300 ms

    cleaned = clean_wearable_signals(raw, min_valid_minutes_per_hour=54, max_interp_gap_minutes=5)
    # Regression C4: the entire dropped charging hour [04:00..05:00) must stay 100% NaN (zero edge bleed).
    charging_hour = cleaned.loc["2022-01-01 04:00:00":"2022-01-01 04:59:00"]
    assert charging_hour.isna().all().all()

    windows, starts = extract_consecutive_windows(cleaned, window_hours=4, resample_minutes=1)
    assert windows.shape == (2, 10, 240)
    assert len(starts) == 2
    assert not np.isnan(windows).any()

    # Regression C15: exclude_channels=(8, 9) preserves raw sin/cos unit circle.
    std_keep_time = standardize_per_sample(windows, exclude_channels=(8, 9))
    np.testing.assert_allclose(std_keep_time[:, :8, :].mean(axis=-1), 0.0, atol=1e-4)
    np.testing.assert_allclose(std_keep_time[:, 8, :] ** 2 + std_keep_time[:, 9, :] ** 2, 1.0, atol=1e-5)

    # Regression B-H1: constant non-zero float32 channel (833.3 ms RR) maps strictly to 0.0, not +/-1.0.
    const_x = np.full((2, 1, 480), 833.3, dtype=np.float32)
    np.testing.assert_array_equal(standardize_per_sample(const_x), np.zeros_like(const_x))

    # Regression C4: a 10-min gap must NOT be bridged when max_interp_gap_minutes=5.
    gap_df = raw.iloc[:60].copy()
    gap_df.loc[idx[20:30], "linacc_x"] = np.nan
    cleaned_gap = clean_wearable_signals(gap_df, min_valid_minutes_per_hour=54, max_interp_gap_minutes=5)
    assert cleaned_gap["linacc_x"].iloc[20:30].isna().all()

    # Regression C4: multi-rate (interleaved NaNs) consecutive identical RR values are removed.
    mr_idx = pd.date_range("2022-01-01 00:00:00", periods=10, freq="50ms")
    mr_df = pd.DataFrame(
        {
            "linacc_x": np.zeros(10),
            "heart_rate": np.full(10, 70.0),
            "rr_interval": [800.0, np.nan, np.nan, np.nan, 800.0, np.nan, np.nan, np.nan, 900.0, np.nan],
        },
        index=mr_idx,
    )
    cleaned_mr = clean_wearable_signals(mr_df, min_valid_minutes_per_hour=0, max_interp_gap_minutes=0)
    # Duplicate 800.0 is dropped before averaging, so mean is (800 + 900) / 2 = 850 (not 833.33).
    assert float(cleaned_mr["rr_interval"].iloc[0]) == pytest.approx(850.0)

    # Regression C4: resample_minutes=2 still counts valid coverage in exact minutes.
    cleaned_2m = clean_wearable_signals(raw.iloc[:60], min_valid_minutes_per_hour=54, resample_minutes=2)
    assert cleaned_2m["heart_rate"].notna().all()

    # Regression C11: extract_consecutive_windows never stitches across omitted timestamps.
    omitted_df = raw.copy()  # has an unpadded 1-hour jump between 03:59 and 05:00
    w_omitted, _ = extract_consecutive_windows(omitted_df, window_hours=5, resample_minutes=1)
    assert w_omitted.shape[0] == 0


def test_build_survival_labels_excludes_overlapping_relapses_and_marks_censoring() -> None:
    samples = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u2"],
            "timestamp": [
                "2021-01-01 00:00:00",  # 10 days before relapse -> event=True, time=10
                "2021-01-10 23:00:00",  # starts 1h before relapse; with window_hours=4 -> excluded!
                "2021-01-12 00:00:00",  # inside relapse [Jan 11, Jan 15] -> excluded
                "2021-01-20 00:00:00",  # after relapse, last sample Jan 30 -> event=False, time=10, past=1
                "2021-01-30 00:00:00",  # last sample -> event=False
                "2021-01-05 00:00:00",  # u2 has no relapse -> excluded from downstream
            ],
        }
    )
    episodes = pd.DataFrame(
        {
            "user_id": ["u1"],
            "start": ["2021-01-11 00:00:00"],
            "end": ["2021-01-15 00:00:00"],
            "severity": ["severe"],
            "psychotic": [True],
        }
    )

    res = build_survival_labels(samples, episodes, window_hours=4.0)
    assert list(res.retained_indices) == [0, 3, 4]
    assert list(res.event) == [True, False, False]
    assert np.isclose(res.time_days[0], 10.0)
    assert np.isclose(res.time_days[1], 10.0)
    assert list(res.metadata["past_episodes"]) == [0, 1, 1]
    assert list(res.metadata["last_episode_level"]) == [0, 3, 3]
    # 0 = no prior episode, 2 = psychotic (distinct from 1 = depressive)
    assert list(res.metadata["last_episode_psychotic"]) == [0, 2, 2]

    # Verify string "False" is parsed as depressive (1), not psychotic (2).
    episodes_dep = episodes.assign(psychotic="False")
    res_dep = build_survival_labels(samples, episodes_dep, window_hours=4.0)
    assert list(res_dep.metadata["last_episode_psychotic"]) == [0, 1, 1]

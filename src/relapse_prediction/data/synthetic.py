"""Synthetic wearable time-series and right-censored survival data generator.

Generates realistic multi-patient synthetic data matching the exact ``(S, 10, L)``
tensor contract of Fekas et al. (IEEE ICASSP 2023) without exposing any real patient
records. Synthetic relapse times correlate with simulated circadian & heart-rate drift
so end-to-end pretraining, person identification, and survival evaluation can be
verified deterministically out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from relapse_prediction.config import CHANNEL_NAMES, WINDOW_HOURS_TO_LENGTH, WindowHours

from .preprocessing import build_survival_labels, standardize_per_sample

if TYPE_CHECKING:
    from numpy.typing import NDArray

_HOURS_PER_DAY = 24.0
_MINUTES_PER_HOUR = 60.0
_RR_BOUNDS_SYNTHETIC = (320.0, 1950.0)


@dataclass
class SyntheticWearableDataset:
    """Container holding synthetic pretext and downstream datasets matching Table 1."""

    pretext_x: NDArray[np.float32]
    pretext_user_ids: NDArray[np.int64]
    downstream_x: NDArray[np.float32]
    downstream_user_ids: NDArray[np.int64]
    event: NDArray[np.bool_]
    time_days: NDArray[np.float64]
    metadata: pd.DataFrame
    static_features: pd.DataFrame
    relapse_episodes: pd.DataFrame
    channel_names: tuple[str, ...] = CHANNEL_NAMES


def _simulate_wearable_window(
    *,
    rng: np.random.Generator,
    start_ts: pd.Timestamp,
    t_minutes: NDArray[np.float32],
    base_hr: float,
    base_rr: float,
    circadian_phase: float,
    acc_amp: float,
    gyr_amp: float,
    freq_mult: float,
    proximity: float,
) -> NDArray[np.float32]:
    """Synthesize a single 10-channel wearable window ``(10, L)``."""
    length = len(t_minutes)
    continuous_hours = (start_ts.hour + t_minutes / _MINUTES_PER_HOUR) % _HOURS_PER_DAY
    continuous_rad = 2.0 * np.pi * continuous_hours / _HOURS_PER_DAY

    # Match preprocessing.encode_cyclical_time (integer hour encoding) on channels 8-9.
    discrete_hour_rad = 2.0 * np.pi * np.floor(continuous_hours) / _HOURS_PER_DAY
    hour_sin = np.sin(discrete_hour_rad)
    hour_cos = np.cos(discrete_hour_rad)

    circadian = np.sin(continuous_rad + circadian_phase + 0.6 * proximity)
    micro = np.sin(2.0 * np.pi * freq_mult * t_minutes / max(length / 4.0, 1.0))

    linacc_x = acc_amp * (0.7 * circadian + 0.3 * micro) + rng.normal(0.0, 0.25, size=length)
    linacc_y = acc_amp * (0.5 * np.cos(continuous_rad + circadian_phase) - 0.2 * proximity) + rng.normal(
        0.0, 0.25, size=length
    )
    linacc_z = acc_amp * 0.4 * micro + rng.normal(0.0, 0.25, size=length)

    gyr_x = gyr_amp * circadian + rng.normal(0.0, 2.0, size=length)
    gyr_y = gyr_amp * micro + rng.normal(0.0, 2.0, size=length)
    gyr_z = gyr_amp * 0.5 * np.cos(continuous_rad) + rng.normal(0.0, 2.0, size=length)

    hr = base_hr + 8.0 * circadian + 14.0 * proximity + rng.normal(0.0, 2.0, size=length)
    rr_noise = rng.normal(0.0, 25.0 * (1.1 - 0.5 * proximity), size=length)
    rr = np.clip(
        base_rr - 70.0 * circadian - 130.0 * proximity + rr_noise, _RR_BOUNDS_SYNTHETIC[0], _RR_BOUNDS_SYNTHETIC[1]
    )

    return np.stack([linacc_x, linacc_y, linacc_z, gyr_x, gyr_y, gyr_z, hr, rr, hour_sin, hour_cos], axis=0).astype(
        np.float32
    )


def _sample_user_episodes(
    rng: np.random.Generator, user_key: str, base_date: pd.Timestamp, /
) -> tuple[list[float], list[dict[str, object]]]:
    """Sample 1-2 simulated relapse episodes for a downstream patient."""
    severities = ("low", "moderate", "severe")
    n_eps = int(rng.integers(1, 3))
    ep_days = (
        [float(rng.uniform(70.0, 130.0))]
        if n_eps == 1
        else [float(rng.uniform(45.0, 80.0)), float(rng.uniform(115.0, 150.0))]
    )
    rows: list[dict[str, object]] = []
    for ep_start in ep_days:
        ep_dur = float(rng.uniform(5.0, 12.0))
        rows.append(
            {
                "user_id": user_key,
                "start": base_date + pd.Timedelta(days=ep_start),
                "end": base_date + pd.Timedelta(days=ep_start + ep_dur),
                "severity": severities[int(rng.integers(0, len(severities)))],
                "psychotic": bool(rng.integers(0, 2)),
            }
        )
    return ep_days, rows


def generate_synthetic_wearable_dataset(
    *,
    window_hours: WindowHours = 4,
    seq_len: int | None = None,
    n_pretext_users: int = 20,
    n_downstream_users: int = 14,
    samples_per_user: int = 24,
    standardize: bool = True,
    seed: int = 42,
) -> SyntheticWearableDataset:
    """Generate synthetic wearable windows ``(S, 10, L)`` and right-censored relapse labels.

    Args:
        window_hours: Duration in hours (``4``, ``8``, or ``12`` -> ``240``, ``480``, ``720`` steps).
        seq_len: Optional override for sequence length ``L`` (useful for fast smoke tests).
        n_pretext_users: Total simulated patients in the unlabeled pretext cohort.
        n_downstream_users: Subset of patients with observed relapse timelines.
        samples_per_user: Approximate number of windows generated per user.
        standardize: Whether to apply per-sample mean-variance standardization.
        seed: Random seed for reproducibility.

    Returns:
        :class:`SyntheticWearableDataset` matching the pipeline's expected schema.
    """
    if n_downstream_users > n_pretext_users:
        msg = f"n_downstream_users ({n_downstream_users}) cannot exceed n_pretext_users ({n_pretext_users})"
        raise ValueError(msg)

    length = seq_len if seq_len is not None else WINDOW_HOURS_TO_LENGTH[window_hours]
    rng = np.random.default_rng(seed)

    base_date = pd.Timestamp("2020-01-01 00:00:00")
    t_minutes = np.arange(length, dtype=np.float32)

    all_windows: list[NDArray[np.float32]] = []
    all_user_ids: list[int] = []
    sample_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    static_rows: list[dict[str, object]] = []

    for u in range(n_pretext_users):
        user_key = f"synthetic_user_{u:02d}"
        base_hr = float(rng.uniform(58.0, 88.0))
        base_rr = 60000.0 / base_hr
        circadian_phase = float(rng.uniform(-np.pi / 3, np.pi / 3))
        acc_amp = float(rng.uniform(0.6, 2.4))
        gyr_amp = float(rng.uniform(8.0, 35.0))
        freq_mult = float(rng.uniform(0.8, 1.4))

        static_rows.append(
            {
                "user_id": user_key,
                "age": float(rng.integers(20, 55)),
                "illness_duration_years": float(rng.uniform(0.5, 12.0)),
                "treatment_compliance": float(rng.uniform(0.4, 1.0)),
                "diagnosis_group": int(rng.integers(0, 3)),
            }
        )

        day_offsets = np.sort(rng.uniform(0.0, 180.0, size=samples_per_user))
        if u < n_downstream_users:
            relapse_starts, user_ep_rows = _sample_user_episodes(rng, user_key, base_date)
            episode_rows.extend(user_ep_rows)
        else:
            relapse_starts = []

        for day_offset in day_offsets:
            start_ts = base_date + pd.Timedelta(days=float(day_offset))
            future_gaps = [ep - float(day_offset) for ep in relapse_starts if ep > float(day_offset)]
            proximity = float(np.exp(-min(future_gaps) / 25.0)) if future_gaps else 0.0
            sample = _simulate_wearable_window(
                rng=rng,
                start_ts=start_ts,
                t_minutes=t_minutes,
                base_hr=base_hr,
                base_rr=base_rr,
                circadian_phase=circadian_phase,
                acc_amp=acc_amp,
                gyr_amp=gyr_amp,
                freq_mult=freq_mult,
                proximity=proximity,
            )
            all_windows.append(sample)
            all_user_ids.append(u)
            sample_rows.append({"user_id": user_key, "user_idx": u, "timestamp": start_ts})

    pretext_x = np.stack(all_windows, axis=0)
    if standardize:
        pretext_x = standardize_per_sample(pretext_x)
    pretext_user_ids = np.asarray(all_user_ids, dtype=np.int64)

    sample_df = pd.DataFrame(sample_rows)
    episodes_df = pd.DataFrame(episode_rows)
    static_df = pd.DataFrame(static_rows).set_index("user_id")

    labeled = build_survival_labels(sample_df[["user_id", "timestamp"]], episodes_df, window_hours=float(window_hours))
    idx = labeled.retained_indices

    return SyntheticWearableDataset(
        pretext_x=pretext_x,
        pretext_user_ids=pretext_user_ids,
        downstream_x=pretext_x[idx],
        downstream_user_ids=pretext_user_ids[idx],
        event=labeled.event,
        time_days=labeled.time_days,
        metadata=labeled.metadata,
        static_features=static_df,
        relapse_episodes=episodes_df,
    )

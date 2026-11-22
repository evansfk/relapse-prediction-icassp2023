"""Wearable signal preprocessing, time encoding, windowing, and right-censoring labels.

Reconstructs the Section 2 preprocessing pipeline from Fekas et al. (IEEE ICASSP 2023)
without requiring access to the private clinical *e-Prevention* database or internal
``eprevention`` package. Any dataset conforming to the documented DataFrame schemas can
be passed directly to these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

SEVERITY_MAP: dict[str, int] = {"low": 1, "moderate": 2, "severe": 3}

_MINUTES_PER_HOUR = 60
_SECONDS_PER_DAY = 86400.0
_MIN_SURVIVAL_DAYS = 1e-3

_SURVIVAL_META_COLUMNS: tuple[str, ...] = (
    "orig_idx",
    "user_id",
    "timestamp",
    "next_episode",
    "event",
    "time_days",
    "past_episodes",
    "last_episode_level",
    "last_episode_psychotic",
    "event_free_days",
)

_FREQ_PERIODS: dict[str, float] = {
    "day": 30.436875,
    "dayofweek": 7.0,
    "dayofyear": 365.2425,
    "year": 10.0,
    "hour": 24.0,
    "month": 12.0,
}


def encode_cyclical_time(
    timestamps: pd.Series[Any] | pd.DatetimeIndex, /, *, freq: str = "hour", fractional_hour: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Encode timestamps into cyclical sine and cosine features.

    For ``freq="hour"``, computes the paper's Section 2 time-of-day variables:
    ``x_sin = sin(2 * pi * hour / 24)`` and ``x_cos = cos(2 * pi * hour / 24)``.

    Args:
        timestamps: Datetime-like Series or DatetimeIndex.
        freq: One of ``"hour"``, ``"day"``, ``"dayofweek"``, ``"dayofyear"``, ``"month"``, ``"year"``.
        fractional_hour: If ``True`` and ``freq == "hour"``, include fractional minutes
            (``hour + minute / 60``); defaults to ``False`` matching ``time_encoding.py``.

    Returns:
        Tuple ``(sin_values, cos_values)`` as 1D ``float64`` NumPy arrays.

    Raises:
        ValueError: If ``freq`` is unsupported.
    """
    if freq not in _FREQ_PERIODS:
        msg = f"Unsupported frequency {freq!r}; expected one of {sorted(_FREQ_PERIODS)}"
        raise ValueError(msg)

    dt = pd.to_datetime(timestamps, utc=True)
    dt_series = dt.to_series() if isinstance(dt, pd.DatetimeIndex) else dt

    values: NDArray[np.float64]
    if freq == "hour":
        values = np.asarray(dt_series.dt.hour, dtype=np.float64)
        if fractional_hour:
            values = values + np.asarray(dt_series.dt.minute, dtype=np.float64) / float(_MINUTES_PER_HOUR)
    elif freq == "day":
        values = np.asarray(dt_series.dt.day - 1, dtype=np.float64)
    elif freq == "dayofweek":
        values = np.asarray(dt_series.dt.dayofweek, dtype=np.float64)
    elif freq == "dayofyear":
        values = np.asarray(dt_series.dt.dayofyear - 1, dtype=np.float64)
    elif freq == "month":
        values = np.asarray(dt_series.dt.month - 1, dtype=np.float64)
    else:
        decade = _FREQ_PERIODS["year"]
        years = np.asarray(dt_series.dt.year, dtype=np.float64)
        values = years - (years // decade) * decade

    angle = (values / _FREQ_PERIODS[freq]) * 2.0 * np.pi
    return np.sin(angle), np.cos(angle)


def append_time_of_day_features(df: pd.DataFrame, /, *, freqs: Sequence[str] = ("hour",)) -> pd.DataFrame:
    """Append cyclical sine/cosine columns derived from a ``DatetimeIndex``.

    Args:
        df: DataFrame indexed by timestamps.
        freqs: Cyclical periods to encode (default ``("hour",)``).

    Returns:
        Copy of ``df`` with ``{freq}_sin`` and ``{freq}_cos`` columns appended.
    """
    dt_index = df.index
    if not isinstance(dt_index, pd.DatetimeIndex):
        msg = "DataFrame index must be a pandas DatetimeIndex"
        raise TypeError(msg)

    out = df.copy()
    for freq in freqs:
        sin_vals, cos_vals = encode_cyclical_time(dt_index, freq=freq)
        out[f"{freq}_sin"] = sin_vals
        out[f"{freq}_cos"] = cos_vals
    return out


def _interpolate_short_interior_gaps(df: pd.DataFrame, /, *, max_gap_steps: int) -> pd.DataFrame:
    """Linearly interpolate only strictly interior NaN runs of length ``<= max_gap_steps``.

    Unlike bare ``df.interpolate(limit=k, limit_direction="both")``, this never fills edges
    of long charging gaps and never bridges gaps longer than ``max_gap_steps``.
    """
    if max_gap_steps <= 0:
        return df

    interpolated = df.interpolate(method="linear", limit_area="inside")
    for col in df.columns:
        is_na = df[col].isna()
        if not is_na.any():
            continue
        run_ids = (is_na != is_na.shift(1)).cumsum()
        run_lengths = is_na.groupby(run_ids).transform("sum")
        keep_na = is_na & (run_lengths > max_gap_steps)
        interpolated.loc[keep_na, col] = np.nan
    return interpolated


def clean_wearable_signals(
    df: pd.DataFrame,
    /,
    *,
    acc_cols: Sequence[str] = ("linacc_x", "linacc_y", "linacc_z"),
    gyr_cols: Sequence[str] = ("gyr_x", "gyr_y", "gyr_z"),
    hr_col: str = "heart_rate",
    rr_col: str = "rr_interval",
    acc_limit: float = 19.6,
    gyr_limit: float = 573.0,
    rr_bounds_ms: tuple[float, float] = (300.0, 2000.0),
    min_valid_minutes_per_hour: int = 54,
    resample_minutes: int = 1,
    max_interp_gap_minutes: int = 5,
) -> pd.DataFrame:
    """Apply Section 2 wearable outlier removal, filtering, and resampling.

    Steps matching Fekas et al. (IEEE ICASSP 2023, Sec. 2):
    1. Exclude kinetic sensor values exceeding hardware limits.
    2. Exclude identical consecutive non-null RR-intervals (multi-rate safe) and RR-intervals
       outside ``[300, 2000]`` ms.
    3. Resample to ``resample_minutes`` averages (1-minute low-pass filter by default).
    4. Linearly interpolate strictly interior short gaps of length ``<= max_interp_gap_minutes``
       *before* dropping invalid hours (never filling edges of multi-hour charging gaps).
    5. Blank out any 1-hour block with fewer than ``min_valid_minutes_per_hour`` (default 54,
       i.e. 90%) valid heart-rate minutes.

    Args:
        df: Raw sensor DataFrame indexed by ``DatetimeIndex``.
        acc_cols: Linear accelerometer column names.
        gyr_cols: Gyroscope column names.
        hr_col: Heart rate (BPM) column name.
        rr_col: RR-interval (ms) column name.
        acc_limit: Symmetric absolute limit for linear acceleration.
        gyr_limit: Symmetric absolute limit for angular velocity.
        rr_bounds_ms: Valid ``(min_ms, max_ms)`` range for RR-intervals.
        min_valid_minutes_per_hour: Minimum valid heart-rate minutes required per hour.
        resample_minutes: Resampling / moving-average interval in minutes.
        max_interp_gap_minutes: Maximum contiguous missing minutes to linearly interpolate.

    Returns:
        Cleaned and resampled DataFrame indexed at ``resample_minutes`` intervals.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        msg = "Input DataFrame must have a DatetimeIndex"
        raise TypeError(msg)

    cleaned = df.sort_index().copy()
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")]

    for col in acc_cols:
        if col in cleaned.columns:
            cleaned.loc[cleaned[col].abs() > acc_limit, col] = np.nan

    for col in gyr_cols:
        if col in cleaned.columns:
            cleaned.loc[cleaned[col].abs() > gyr_limit, col] = np.nan

    if rr_col in cleaned.columns:
        rr = cleaned[rr_col]
        rr_non_null = rr.dropna()
        dup_rr_index = rr_non_null.index[rr_non_null.diff() == 0.0]
        out_of_bounds = (rr < rr_bounds_ms[0]) | (rr > rr_bounds_ms[1])
        cleaned.loc[out_of_bounds, rr_col] = np.nan
        cleaned.loc[dup_rr_index, rr_col] = np.nan

    if hr_col in cleaned.columns:
        cleaned.loc[cleaned[hr_col] <= 0, hr_col] = np.nan

    # Count exact 1-minute valid HR coverage per hour prior to resampling / interpolation.
    invalid_hours: pd.DatetimeIndex = pd.DatetimeIndex([])
    if hr_col in cleaned.columns and min_valid_minutes_per_hour > 0:
        one_min_valid = cleaned[hr_col].resample("1min").count() > 0
        hr_buckets = pd.DatetimeIndex(one_min_valid.index).floor("h")
        valid_min_per_hour = one_min_valid.groupby(hr_buckets).sum()
        invalid_hours = pd.DatetimeIndex(valid_min_per_hour[valid_min_per_hour < min_valid_minutes_per_hour].index)

    resampled = cleaned.resample(f"{resample_minutes}min").mean()
    max_gap_steps = max_interp_gap_minutes // max(resample_minutes, 1)
    resampled = _interpolate_short_interior_gaps(resampled, max_gap_steps=max_gap_steps)

    # Mask invalid (< 54 min/hr) hours *after* short-gap interpolation so dropped hours stay NaN.
    if len(invalid_hours) > 0:
        resampled_hours = pd.DatetimeIndex(resampled.index).floor("h")
        resampled.loc[resampled_hours.isin(invalid_hours), :] = np.nan

    return resampled


def extract_consecutive_windows(
    df: pd.DataFrame, /, *, window_hours: int = 8, resample_minutes: int = 1, add_time_encoding: bool = True
) -> tuple[NDArray[np.float32], list[pd.Timestamp]]:
    """Segment a cleaned sensor DataFrame into non-overlapping gap-free windows.

    Never stitches across either explicit ``NaN`` gaps or omitted timestamps.

    Args:
        df: Cleaned sensor DataFrame indexed by ``DatetimeIndex``.
        window_hours: Window duration in hours (4, 8, or 12 in the paper).
        resample_minutes: Expected step between consecutive rows in minutes (default 1).
        add_time_encoding: Whether to append ``hour_sin`` and ``hour_cos`` channels.

    Returns:
        Tuple ``(windows, start_timestamps)`` where ``windows`` has shape ``(N, C, L)``
        and ``start_timestamps`` has length ``N``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        msg = "Input DataFrame must have a DatetimeIndex"
        raise TypeError(msg)

    window_len = (window_hours * _MINUTES_PER_HOUR) // resample_minutes
    feature_df = append_time_of_day_features(df, freqs=("hour",)) if add_time_encoding else df

    is_missing = feature_df.isna().any(axis=1)
    expected_step = pd.Timedelta(minutes=resample_minutes)
    time_jump = feature_df.index.to_series().diff().ne(expected_step).fillna(value=False)
    # Start a new contiguous run whenever missingness toggles OR a timestamp step is skipped.
    run_break = (is_missing != is_missing.shift(1)) | time_jump
    run_ids = run_break.cumsum()

    windows: list[NDArray[np.float32]] = []
    starts: list[pd.Timestamp] = []

    valid_df = feature_df[~is_missing]
    for _, group in valid_df.groupby(run_ids[~is_missing], sort=False):
        n_full_windows = len(group) // window_len
        for w in range(n_full_windows):
            chunk = group.iloc[w * window_len : (w + 1) * window_len]
            windows.append(chunk.to_numpy(dtype=np.float32).T)
            starts.append(pd.Timestamp(chunk.index[0]))

    if not windows:
        n_channels = feature_df.shape[1]
        return np.empty((0, n_channels, window_len), dtype=np.float32), []

    return np.stack(windows, axis=0), starts


def standardize_per_sample(
    x: NDArray[np.floating[Any]], /, *, eps: float = 1e-8, exclude_channels: Sequence[int] = ()
) -> NDArray[np.float32]:
    """Apply per-sample, per-channel zero-mean unit-variance standardization.

    Equivalent to ``tslearn.preprocessing.TimeSeriesScalerMeanVariance`` used in the
    original experiments. Pass ``exclude_channels=(8, 9)`` to preserve raw ``[-1, 1]``
    cyclical ``hour_sin`` / ``hour_cos`` offsets.

    Args:
        x: Array of shape ``(S, C, L)``.
        eps: Numerical floor for near-constant channels.
        exclude_channels: Optional channel indices left unstandardized.

    Returns:
        Standardized ``float32`` array of shape ``(S, C, L)``.
    """
    arr = np.asarray(x, dtype=np.float32)
    mean = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1, keepdims=True)
    safe_std = np.where(std < eps, np.float32(1.0), std)
    standardized: NDArray[np.float32] = np.asarray((arr - mean) / safe_std, dtype=np.float32)
    if exclude_channels:
        out = standardized.copy()
        idx = list(exclude_channels)
        out[:, idx, :] = arr[:, idx, :]
        return out
    return standardized


def _parse_psychotic_flag(raw: object) -> int:
    """Encode past-episode psychotic indicator as ``1`` (depressive) or ``2`` (psychotic)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return 1
    if isinstance(raw, (bool, np.bool_)):
        return 2 if bool(raw) else 1
    if isinstance(raw, (int, np.integer)):
        return 2 if int(raw) != 0 else 1
    text = str(raw).strip().lower()
    if text in {"true", "1", "yes", "psychotic"}:
        return 2
    return 1


@dataclass(frozen=True)
class SurvivalLabelResult:
    """Structured output of right-censored survival labeling."""

    retained_indices: NDArray[np.int64]
    event: NDArray[np.bool_]
    time_days: NDArray[np.float64]
    metadata: pd.DataFrame


def build_survival_labels(
    sample_records: pd.DataFrame, relapse_episodes: pd.DataFrame, /, *, window_hours: float = 0.0
) -> SurvivalLabelResult:
    """Compute right-censored time-to-relapse labels matching ``load_survival_data.py``.

    Expected schemas:
        - ``sample_records``: columns ``["user_id", "timestamp"]`` (window start per row).
        - ``relapse_episodes``: columns ``["user_id", "start", "end"]`` and optionally
          ``"severity"`` (``"low"``/``"moderate"``/``"severe"``) and ``"psychotic"`` (bool).

    Rules (Section 3, Fig. 1):
        - Keep only users present in ``relapse_episodes``.
        - Exclude any window ``[timestamp, timestamp + window_hours]`` overlapping an active
          relapse interval ``[start, end]``.
        - If a future relapse starts after the window ends, set ``event=True`` and
          ``time_days = (next_relapse_start - timestamp)`` in days.
        - Otherwise (no future relapse observed), mark right-censored ``event=False`` and
          ``time_days = (user_last_timestamp - timestamp)`` in days.
        - Past-episode encoding: ``last_episode_level`` in ``{0: none, 1: low, 2: moderate, 3: severe}``
          and ``last_episode_psychotic`` in ``{0: no prior episode, 1: depressive, 2: psychotic}``.

    Args:
        sample_records: Per-window user ID and timestamp metadata.
        relapse_episodes: Clinical relapse intervals per patient.
        window_hours: Window duration in hours used to exclude partial overlap with active relapses.

    Returns:
        :class:`SurvivalLabelResult` containing kept indices, event flags, survival times,
        and an enriched metadata DataFrame including past-episode history features.
    """
    records = sample_records.copy()
    records["orig_idx"] = np.arange(len(records), dtype=np.int64)
    records["timestamp"] = pd.to_datetime(records["timestamp"])

    episodes = relapse_episodes.copy()
    episodes["start"] = pd.to_datetime(episodes["start"])
    episodes["end"] = pd.to_datetime(episodes["end"])

    window_delta = pd.Timedelta(hours=window_hours)
    rows: list[dict[str, object]] = []

    for user_id, user_samples in records.groupby("user_id", sort=False):
        user_eps = episodes[episodes["user_id"] == user_id].sort_values("start")
        if user_eps.empty:
            continue

        sorted_samples = user_samples.sort_values("timestamp")
        last_sample_time = pd.Timestamp(sorted_samples["timestamp"].max())

        for _, row in sorted_samples.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            ts_end = ts + window_delta
            overlaps_episode = bool(((user_eps["start"] <= ts_end) & (ts <= user_eps["end"])).any())
            if overlaps_episode:
                continue

            future_eps = user_eps[user_eps["start"] > ts_end]
            past_eps = user_eps[user_eps["end"] < ts]

            if not future_eps.empty:
                next_episode = pd.Timestamp(future_eps["start"].iloc[0])
                event = True
                time_days = max(float((next_episode - ts).total_seconds() / _SECONDS_PER_DAY), _MIN_SURVIVAL_DAYS)
            else:
                next_episode = last_sample_time
                event = False
                time_days = max(float((last_sample_time - ts).total_seconds() / _SECONDS_PER_DAY), _MIN_SURVIVAL_DAYS)

            n_past = len(past_eps)
            if n_past > 0:
                last_ep = past_eps.iloc[-1]
                raw_sev = str(last_ep.get("severity", "")).strip().lower()
                last_level = SEVERITY_MAP.get(raw_sev, 0)
                last_psychotic = _parse_psychotic_flag(last_ep.get("psychotic", False))
                last_end = pd.Timestamp(last_ep["end"])
                event_free_days = max(float((ts - last_end).total_seconds() / _SECONDS_PER_DAY), 0.0)
            else:
                last_level = 0
                last_psychotic = 0
                event_free_days = 0.0

            rows.append(
                {
                    "orig_idx": int(row["orig_idx"]),
                    "user_id": user_id,
                    "timestamp": ts,
                    "next_episode": next_episode,
                    "event": event,
                    "time_days": time_days,
                    "past_episodes": n_past,
                    "last_episode_level": last_level,
                    "last_episode_psychotic": last_psychotic,
                    "event_free_days": event_free_days,
                }
            )

    if not rows:
        empty_meta = pd.DataFrame(columns=list(_SURVIVAL_META_COLUMNS))
        return SurvivalLabelResult(
            retained_indices=np.empty(0, dtype=np.int64),
            event=np.empty(0, dtype=bool),
            time_days=np.empty(0, dtype=np.float64),
            metadata=empty_meta,
        )

    meta = pd.DataFrame(rows).sort_values("orig_idx").reset_index(drop=True)
    return SurvivalLabelResult(
        retained_indices=meta["orig_idx"].to_numpy(dtype=np.int64),
        event=meta["event"].to_numpy(dtype=bool),
        time_days=meta["time_days"].to_numpy(dtype=np.float64),
        metadata=meta,
    )

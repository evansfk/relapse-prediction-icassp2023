"""Downstream feature assembly combining SSL embeddings, user IDs, static & past-event features.

Implements the covariate combinations evaluated in Section 4 and Table 3 of Fekas et al.
(IEEE ICASSP 2023):
- Pretrained SSL embeddings (``TFC``, ``Mixing-Up``, ``TS-TCC``, or ``MiniRocket``)
- One-hot encoded user-ID (``user_id``)
- Static clinical covariates (``st``)
- Past-episode history features (``p_ep``, ``ll``, ``ps``) changing only after an event
  (plus an optional opt-in ``event_free_days`` feature off by default for Table 3 fidelity).
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Strictly matches Table 3 (+ ps, p_ep, ll): features that change only after an event occurs.
PAST_EPISODE_COLUMNS: tuple[str, ...] = ("past_episodes", "last_episode_level", "last_episode_psychotic")


@dataclass(frozen=True, slots=True, kw_only=True)
class AssembledSurvivalFeatures:
    """Train/test covariate matrices with explicit column names."""

    x_train: NDArray[np.float64]
    x_test: NDArray[np.float64]
    feature_names: list[str]


def _normalize_static_frame(static_features: pd.DataFrame, /) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate index uniqueness and split static covariates into numeric/boolean vs. categorical."""
    aligned = static_features.copy()
    aligned.index = aligned.index.astype(str)
    if aligned.index.has_duplicates:
        msg = "static_features index contains duplicate user_id entries"
        raise ValueError(msg)

    num_cols: list[str] = []
    cat_cols: list[str] = []
    for col in aligned.columns:
        dtype = aligned[col].dtype
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            aligned[col] = aligned[col].astype("float64")
            num_cols.append(str(col))
        else:
            cat_cols.append(str(col))
    return aligned, num_cols, cat_cols


def assemble_survival_features(
    *,
    emb_train: NDArray[np.floating],
    emb_test: NDArray[np.floating],
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    include_user_id: bool = True,
    include_past_episodes: bool = False,
    include_event_free_days: bool = False,
    static_features: pd.DataFrame | None = None,
    include_static: bool = False,
    drop_first: bool = True,
) -> AssembledSurvivalFeatures:
    """Assemble train and test covariate matrices without data leakage.

    Encoders (one-hot user ID and categorical static attributes) and median imputers are
    fitted strictly on ``meta_train`` and applied to ``meta_test``.

    Args:
        emb_train: Training SSL embeddings ``(N_train, D)``.
        emb_test: Test SSL embeddings ``(N_test, D)``.
        meta_train: Training metadata containing ``user_id`` and optionally past-episode columns.
        meta_test: Test metadata matching ``meta_train`` schema.
        include_user_id: Concatenate one-hot user-ID indicators (Section 4).
        include_past_episodes: Concatenate ``past_episodes``, ``last_episode_level``, and
            ``last_episode_psychotic`` (Table 3: ``+ ps, p_ep, ll``).
        include_event_free_days: Opt-in dynamic feature ``event_free_days`` (off by default to
            match Paper Table 3 strictly and avoid temporal-shortcut leakage under random splits).
        static_features: Optional DataFrame indexed by ``user_id`` containing static attributes.
        include_static: Concatenate static patient covariates (Table 3: ``+ st``).
        drop_first: Drop first category per one-hot variable (default ``True``, matching
            ``pd.get_dummies(..., drop_first=True)`` and ``sksurv.OneHotEncoder`` in ``2.0-5.0``;
            set ``False`` when evaluating cold-start unseen test users).

    Returns:
        :class:`AssembledSurvivalFeatures` containing ``x_train``, ``x_test``, and ``feature_names``.
    """
    if include_static and static_features is None:
        msg = "include_static=True requires a non-None static_features DataFrame"
        raise ValueError(msg)

    train_blocks: list[NDArray[np.float64]] = [np.asarray(emb_train, dtype=np.float64)]
    test_blocks: list[NDArray[np.float64]] = [np.asarray(emb_test, dtype=np.float64)]
    names: list[str] = [f"emb_{i}" for i in range(train_blocks[0].shape[1])]
    drop_mode = "first" if drop_first else None

    if include_user_id:
        ohe = OneHotEncoder(drop=drop_mode, sparse_output=False, handle_unknown="ignore")
        id_tr = ohe.fit_transform(meta_train[["user_id"]].astype(str))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            id_te = ohe.transform(meta_test[["user_id"]].astype(str))
        train_blocks.append(np.asarray(id_tr, dtype=np.float64))
        test_blocks.append(np.asarray(id_te, dtype=np.float64))
        names.extend([f"uid_{name.removeprefix('user_id_')}" for name in ohe.get_feature_names_out(["user_id"])])

    past_cols = list(PAST_EPISODE_COLUMNS) if include_past_episodes else []
    if include_event_free_days:
        past_cols.append("event_free_days")

    if past_cols:
        missing_past = [col for col in past_cols if col not in meta_train.columns or col not in meta_test.columns]
        if missing_past:
            msg = f"Missing requested past-episode metadata columns: {missing_past}"
            raise ValueError(msg)
        train_blocks.append(meta_train[past_cols].fillna(0.0).to_numpy(dtype=np.float64))
        test_blocks.append(meta_test[past_cols].fillna(0.0).to_numpy(dtype=np.float64))
        names.extend(past_cols)

    if include_static and static_features is not None:
        aligned_static, num_cols, cat_cols = _normalize_static_frame(static_features)
        st_tr = aligned_static.reindex(meta_train["user_id"].astype(str)).reset_index(drop=True)
        st_te = aligned_static.reindex(meta_test["user_id"].astype(str)).reset_index(drop=True)

        if num_cols:
            medians = st_tr[num_cols].median()
            train_blocks.append(st_tr[num_cols].fillna(medians).fillna(0.0).to_numpy(dtype=np.float64))
            test_blocks.append(st_te[num_cols].fillna(medians).fillna(0.0).to_numpy(dtype=np.float64))
            names.extend([f"st_{c}" for c in num_cols])

        if cat_cols:
            cat_ohe = OneHotEncoder(drop=drop_mode, sparse_output=False, handle_unknown="ignore")
            c_tr = cat_ohe.fit_transform(st_tr[cat_cols].fillna("missing").astype(str))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                c_te = cat_ohe.transform(st_te[cat_cols].fillna("missing").astype(str))
            train_blocks.append(np.asarray(c_tr, dtype=np.float64))
            test_blocks.append(np.asarray(c_te, dtype=np.float64))
            names.extend([f"st_{name}" for name in cat_ohe.get_feature_names_out(cat_cols)])

    return AssembledSurvivalFeatures(
        x_train=np.concatenate(train_blocks, axis=1), x_test=np.concatenate(test_blocks, axis=1), feature_names=names
    )

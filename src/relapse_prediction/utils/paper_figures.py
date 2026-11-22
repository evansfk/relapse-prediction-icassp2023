"""Publication-grade figure generators reproducing IEEE ICASSP 2023 plots and report figures.

Reconstructs the exact Matplotlib/Seaborn (`science` + `ieee` + `grid`) and Altair (`sfmono`)
visualizations from Fekas et al. (IEEE ICASSP 2023) and accompanying diagnostic figures
so users can reproduce every paper and evaluation plot cleanly from either
synthetic runs or private clinical datasets:

- **Paper Fig. 1 (`chart.pdf`):** :func:`plot_cohort_relapse_timeline` (Altair)
- **Paper Fig. 2 (`per_embedding.pdf`):** :func:`plot_per_embedding_comparison` (Seaborn + IEEE)
- **Paper Fig. 3 (`per_smodel.pdf`):** :func:`plot_per_survival_model_comparison` (Seaborn + IEEE)
- **Paper Fig. 4 (`final_risk_classification.pdf`):** :func:`plot_risk_classification_timeline` (Altair)
- **Report Fig. (`identification.pdf`):** :func:`plot_person_id_metrics` (Seaborn + IEEE)
- **Report Fig. (`identification_dist.pdf`):** :func:`plot_user_sample_distribution` (Seaborn + IEEE)
- **Report Fig. (`survival_distribution.pdf`):** :func:`plot_survival_time_distribution` (Seaborn + IEEE)
- **Risk Diagnostics:** :func:`plot_risk_group_histogram` & :func:`plot_risk_group_survival_curves`
"""

from __future__ import annotations

import contextlib
import re
import shutil

from pathlib import Path
from typing import TYPE_CHECKING, Any

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401 - registers SciencePlots matplotlib styles on import
import seaborn as sns

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from relapse_prediction.survival.base import BaseSurvivalModel

# Desaturated SciencePlots IEEE palette matching Paper Figs. 2-3 and person-ID plots.
_IEEE_PALETTE_3: tuple[str, ...] = ("#000000", "#DF2020", "#2020DF")
_IEEE_PALETTE_4: tuple[str, ...] = ("#000000", "#DF2020", "#2020DF", "#107010")
_HIST_PALETTE_2: tuple[str, ...] = ("#555555", "#FF5555")
_RISK_TIER_COLORS: dict[str, str] = {"low": "#4C78A8", "medium": "#F58518", "high": "#E45756"}
_SEVERITY_COLORS: tuple[str, ...] = ("#4C78A8", "#F58518", "#E45756")
_SEVERITY_DOMAIN: tuple[str, ...] = ("low", "moderate", "severe")
_RISK_DOMAIN: tuple[str, ...] = ("low", "medium", "high")
_RISK_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
_CENSORED_DOMAIN: tuple[str, ...] = ("false", "true")
_CENSORED_COLORS: tuple[str, ...] = ("#000000", "#8C8C8C")

_DEFAULT_EMBEDDING_ORDER: tuple[str, ...] = ("TS-TCC", "TFC", "MiniRocket", "Mixing-up")
_DEFAULT_SURVIVAL_ORDER: tuple[str, ...] = ("RandomSurvivalForest", "ExtraSurvivalTrees", "NeuralMTLR", "DeepSurv")
_PAPER_SURVIVAL_RENAME: dict[str, str] = {
    "RandomSurvivalForest": "C-SurvivalForest",
    "ConditionalSurvivalForest": "C-SurvivalForest",
    "NeuralMTLR": "NeuralMultiTask",
    "NonLinearCoxPH": "DeepSurv",
}
_DEFAULT_METRIC_ORDER: tuple[str, ...] = ("F1", "Recall", "Precision", "AUPRC", "Accuracy", "AUROC")
_NEG_LOG_IBS_LABEL = r"$-\log\left(\mathrm{IBS}\right)$"
_SFMONO_FONT = "SF Mono, DejaVu Sans Mono, Liberation Mono, monospace"
_SFMONO_THEME_NAME = "icassp2023_sfmono"
_SFMONO_REGISTERED = False


def _ensure_altair_sfmono_theme() -> None:
    """Register and enable the ``sfmono`` Altair theme matching Paper Figs. 1 and 4."""
    global _SFMONO_REGISTERED  # noqa: PLW0603
    alt.data_transformers.disable_max_rows()
    if not _SFMONO_REGISTERED:

        @alt.theme.register(_SFMONO_THEME_NAME, enable=True)
        def _sfmono_theme() -> alt.theme.ThemeConfig:
            return {
                "config": {
                    "title": {"font": _SFMONO_FONT},
                    "axis": {"labelFont": _SFMONO_FONT, "titleFont": _SFMONO_FONT, "labelFontSize": 19, "title": ""},
                    "header": {"labelFont": _SFMONO_FONT, "titleFont": _SFMONO_FONT},
                    "legend": {
                        "labelFont": _SFMONO_FONT,
                        "titleFont": _SFMONO_FONT,
                        "labelFontSize": 19,
                        "titleFontSize": 19,
                    },
                }
            }

        _SFMONO_REGISTERED = True
    else:
        alt.theme.enable(_SFMONO_THEME_NAME)


@contextlib.contextmanager
def ieee_plot_style(
    *, font_size: int = 14, figsize: tuple[float, float] = (8.0, 6.0), use_latex: bool | None = None
) -> Iterator[None]:
    """Context manager applying SciencePlots ``["science", "ieee", "grid"]`` styling.

    Automatically appends ``"no-latex"`` with STIX mathtext and Times-compatible serif
    fonts when TeX is unavailable so plots render cleanly on any machine while staying
    visually faithful to the IEEE ICASSP 2023 figures.

    Args:
        font_size: Base font size in points (``14`` for Figs. 2-3, ``16`` for ID plots).
        figsize: ``(width, height)`` in inches.
        use_latex: Force LaTeX text rendering (``True``), disable (``False``), or
            auto-detect ``latex`` on ``PATH`` (``None``).
    """
    enable_tex = (shutil.which("latex") is not None) if use_latex is None else use_latex
    styles = ["science", "ieee", "grid"] if enable_tex else ["science", "ieee", "grid", "no-latex"]
    with (
        plt.style.context(styles),
        plt.rc_context(
            {
                "font.size": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "legend.fontsize": font_size,
                "figure.figsize": figsize,
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
                "mathtext.fontset": "stix",
                "grid.linestyle": "--",
                "grid.linewidth": 0.5,
                "grid.alpha": 0.7,
            }
        ),
    ):
        yield


def save_matplotlib_figure(fig: Figure, output_path: str | Path, /, *, dpi: int = 300) -> Path:
    """Persist a Matplotlib figure with tight bounding box and create parent directories."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    return path


def save_altair_chart(chart: alt.TopLevelMixin, output_path: str | Path, /, *, ppi: int = 300) -> Path:
    """Persist an Altair chart to ``.pdf``, ``.png``, ``.svg``, or ``.html`` via ``vl-convert``."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".png":
        chart.save(str(path), ppi=ppi)
    else:
        chart.save(str(path))
    return path


def _natural_duration_key(label: str) -> tuple[int, str]:
    """Sort duration labels such as ``'4-hour Dataset'``, ``'8-hour Dataset'``, ``'12-hour Dataset'``."""
    match = re.search(r"\d+", label)
    return (int(match.group(0)) if match is not None else 0, label)


def _format_user_labels(raw_ids: Sequence[object]) -> tuple[dict[str, str], dict[str, int]]:
    """Map raw user identifiers deterministically to ``'User 1' .. 'User N'`` and numeric order."""
    unique_ids: list[str] = []
    seen: set[str] = set()
    for uid in raw_ids:
        key = str(uid)
        if key not in seen:
            seen.add(key)
            unique_ids.append(key)

    label_map: dict[str, str] = {}
    order_map: dict[str, int] = {}
    for idx, key in enumerate(unique_ids, start=1):
        label = f"User {idx}"
        label_map[key] = label
        order_map[label] = idx
    return label_map, order_map


def _build_connector_arrows(sample_df: pd.DataFrame, /) -> pd.DataFrame:
    """Construct horizontal line segments connecting each sample dot to ``next_episode``."""
    rows: list[dict[str, object]] = []
    for seg_id, (_, row) in enumerate(sample_df.iterrows()):
        user_label = str(row["index"])
        user_num = int(row["User"])
        rows.append({"x": row["Days"], "y": seg_id, "class": user_label, "User": user_num})
        rows.append({"x": row["next_episode"], "y": seg_id, "class": user_label, "User": user_num})
    return pd.DataFrame(rows)


def _prepare_timeline_frames(
    metadata: pd.DataFrame,
    relapse_episodes: pd.DataFrame,
    /,
    *,
    selected_users: Sequence[str | int] | None,
    max_samples: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize metadata & relapse intervals into the exact schema used by Paper Figs. 1 and 4."""
    meta = metadata.copy()
    if "timestamp" not in meta.columns or "user_id" not in meta.columns:
        msg = "metadata must contain 'user_id' and 'timestamp' columns"
        raise ValueError(msg)

    meta["user_key"] = meta["user_id"].astype(str)
    meta["Days"] = pd.to_datetime(meta["timestamp"])
    if "next_episode" in meta.columns:
        meta["next_episode"] = pd.to_datetime(meta["next_episode"])
    elif "time_days" in meta.columns:
        meta["next_episode"] = meta["Days"] + pd.to_timedelta(meta["time_days"].astype(float), unit="D")
    else:
        msg = "metadata must contain either 'next_episode' or 'time_days'"
        raise ValueError(msg)

    eps = relapse_episodes.copy()
    if not {"user_id", "start", "end"}.issubset(eps.columns):
        msg = "relapse_episodes must contain 'user_id', 'start', and 'end' columns"
        raise ValueError(msg)
    eps["user_key"] = eps["user_id"].astype(str)
    eps["From"] = pd.to_datetime(eps["start"])
    eps["To"] = pd.to_datetime(eps["end"])
    if "severity" in eps.columns:
        eps["Level"] = eps["severity"].astype(str).str.strip().str.lower()
    else:
        eps["Level"] = "moderate"

    label_map, order_map = _format_user_labels(list(meta["user_key"].tolist()) + list(eps["user_key"].tolist()))
    meta["index"] = meta["user_key"].map(label_map)
    meta["User"] = meta["index"].map(order_map).astype(int)
    eps["index"] = eps["user_key"].map(label_map)
    eps["User"] = eps["index"].map(order_map).astype(int)

    if selected_users is not None:
        keep_keys: set[str] = set()
        keep_nums: set[int] = set()
        for item in selected_users:
            if isinstance(item, int):
                keep_nums.add(item)
            else:
                keep_keys.add(str(item))
        meta = meta[meta["user_key"].isin(keep_keys) | meta["index"].isin(keep_keys) | meta["User"].isin(keep_nums)]
        eps = eps[eps["user_key"].isin(keep_keys) | eps["index"].isin(keep_keys) | eps["User"].isin(keep_nums)]

    if max_samples is not None and len(meta) > max_samples:
        meta = meta.sample(n=max_samples, random_state=seed)

    return meta.reset_index(drop=True), eps.reset_index(drop=True)


def _normalize_survival_results_frame(
    results: pd.DataFrame, /, *, use_paper_model_names: bool
) -> tuple[pd.DataFrame, str, str, str, str, str]:
    """Normalize survival benchmark columns from either repo or original notebook schemas."""
    df = results.copy()

    emb_col = "embeddings" if "embeddings" in df.columns else "embedding"
    model_col = "survival-model" if "survival-model" in df.columns else "survival_model"
    c_col = "c-index" if "c-index" in df.columns else "c_index"
    nli_col = _NEG_LOG_IBS_LABEL if _NEG_LOG_IBS_LABEL in df.columns else "neg_log_ibs"
    len_col = "sample length" if "sample length" in df.columns else "sample_length"

    if nli_col not in df.columns and "ibs" in df.columns:
        df[nli_col] = -np.log(np.clip(df["ibs"].astype(float), 1e-12, None))

    df[emb_col] = df[emb_col].astype(str).replace({"minirocket": "MiniRocket", "Mixing-Up": "Mixing-up"})
    if use_paper_model_names:
        df[model_col] = df[model_col].astype(str).replace(_PAPER_SURVIVAL_RENAME)

    return df, emb_col, model_col, c_col, nli_col, len_col


def _apply_top_shadow_legend(ax: Axes, /, *, ncol: int, y_anchor: float = 1.24) -> None:
    """Place the centered shadowed fancybox legend above the top subplot matching Figs. 2-3."""
    handles, labels = ax.get_legend_handles_labels()
    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_legend.remove()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, y_anchor),
            fancybox=True,
            shadow=True,
            frameon=True,
            facecolor="white",
            edgecolor="#CCCCCC",
            ncol=ncol,
        )


def plot_cohort_relapse_timeline(
    metadata: pd.DataFrame,
    relapse_episodes: pd.DataFrame,
    /,
    *,
    selected_users: Sequence[str | int] | None = None,
    max_samples: int | None = 1000,
    width: int = 1300,
    height: int = 500,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> alt.LayerChart:
    """Reproduce Paper Fig. 1 (``chart.pdf``): Altair timeline of relapse severity & censoring.

    Each patient is a horizontal track with severity-colored relapse intervals (``Level``),
    sample dots colored black (``Censored = false``) or gray (``Censored = true``), and
    thin horizontal connector lines extending from each sample to ``next_episode``.

    Args:
        metadata: Sample metadata containing ``user_id``, ``timestamp``, ``event``, and
            ``next_episode`` (or ``time_days``).
        relapse_episodes: Relapse interval table with ``user_id``, ``start``, ``end``,
            and optional ``severity`` (``"low"``/``"moderate"``/``"severe"``).
        selected_users: Optional subset of user IDs, ``"User k"`` labels, or 1-based user numbers.
        max_samples: Max sample dots to display (``1000`` in paper; ``None`` plots all).
        width: Chart width in pixels (``1300`` in paper).
        height: Chart height in pixels (``500`` in paper).
        seed: Random seed when downsampling to ``max_samples``.
        output_path: Optional file path (e.g., ``chart.pdf``) to save directly.

    Returns:
        Layered Altair chart.
    """
    _ensure_altair_sfmono_theme()
    sample_df, eps_df = _prepare_timeline_frames(
        metadata, relapse_episodes, selected_users=selected_users, max_samples=max_samples, seed=seed
    )
    if "event" not in sample_df.columns:
        msg = "metadata must contain an 'event' boolean column for censoring indicators"
        raise ValueError(msg)
    sample_df["Censored"] = np.where(sample_df["event"].astype(bool), "false", "true")
    arrows_df = _build_connector_arrows(sample_df)

    y_sort = alt.EncodingSortField(field="User", order="ascending")
    relapses = (
        alt.Chart(eps_df)
        .mark_bar()
        .encode(
            x=alt.X("From:T", title=""),
            x2=alt.X2("To:T"),
            color=alt.Color(
                "Level:N",
                sort=list(_SEVERITY_DOMAIN),
                scale=alt.Scale(domain=list(_SEVERITY_DOMAIN), range=list(_SEVERITY_COLORS)),
            ),
            y=alt.Y("index:N", sort=y_sort, title=""),
        )
        .properties(width=width, height=height)
    )

    samples = (
        alt.Chart(sample_df)
        .mark_circle(size=200)
        .encode(
            x=alt.X("Days:T", title=""),
            y=alt.Y("index:N", sort=y_sort, title=""),
            color=alt.Color(
                "Censored:N",
                sort=list(_CENSORED_DOMAIN),
                scale=alt.Scale(domain=list(_CENSORED_DOMAIN), range=list(_CENSORED_COLORS)),
            ),
        )
        .properties(width=width, height=height)
    )

    arrows = (
        alt.Chart(arrows_df)
        .mark_line()
        .encode(
            x=alt.X("x:T", title=""),
            y=alt.Y("class:N", sort=y_sort, title=""),
            detail=alt.Detail("y:N"),
            color=alt.value("#000000"),
            strokeWidth=alt.value(0.3),
        )
        .properties(width=width, height=height)
    )

    chart: alt.LayerChart = alt.layer(relapses, samples, arrows).resolve_scale(color="independent")
    if output_path is not None:
        save_altair_chart(chart, output_path)
    return chart


def plot_per_embedding_comparison(
    results: pd.DataFrame,
    /,
    *,
    embedding_order: Sequence[str] | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    font_size: int = 14,
    use_latex: bool | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Reproduce Paper Fig. 2 (``per_embedding.pdf``): 2-row boxplot across durations & embeddings.

    Top panel displays ``c-index`` and bottom panel displays ``-log(IBS)`` across survival
    models for each pretrained embedding grouped by window duration
    (``4-hour Dataset``, ``8-hour Dataset``, ``12-hour Dataset``).

    Args:
        results: DataFrame containing ``embedding``/``embeddings``, ``sample_length``/``sample length``,
            ``c_index``/``c-index``, and ``neg_log_ibs`` (or ``ibs``).
        embedding_order: Explicit x-axis ordering (defaults to paper order when present).
        figsize: Figure size in inches (``(8, 6)`` in paper).
        font_size: Base IEEE font size (``14`` in paper).
        use_latex: Optional override for Matplotlib ``text.usetex``.
        output_path: Optional path (e.g., ``per_embedding.pdf``) to save directly.

    Returns:
        Matplotlib :class:`Figure`.
    """
    df, emb_col, _, c_col, nli_col, len_col = _normalize_survival_results_frame(results, use_paper_model_names=False)
    if len_col not in df.columns:
        msg = "results must contain a 'sample_length' or 'sample length' column"
        raise ValueError(msg)

    present_embs = list(dict.fromkeys(df[emb_col].astype(str).tolist()))
    order = (
        list(embedding_order)
        if embedding_order is not None
        else [e for e in _DEFAULT_EMBEDDING_ORDER if e in present_embs]
        + [e for e in present_embs if e not in _DEFAULT_EMBEDDING_ORDER]
    )
    hue_order = sorted(df[len_col].astype(str).unique().tolist(), key=_natural_duration_key)
    palette = list(_IEEE_PALETTE_3[: len(hue_order)]) if len(hue_order) <= len(_IEEE_PALETTE_3) else None

    flier_style: dict[str, object] = {
        "marker": "d",
        "markerfacecolor": "#000000",
        "markeredgecolor": "#000000",
        "markersize": 5,
    }

    with ieee_plot_style(font_size=font_size, figsize=figsize, use_latex=use_latex):
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=figsize)
        ax_top, ax_bot = axes[0], axes[1]

        sns.boxplot(
            data=df,
            x=emb_col,
            y=c_col,
            hue=len_col,
            order=order,
            hue_order=hue_order,
            palette=palette,
            flierprops=flier_style,
            ax=ax_top,
        )
        ax_top.set_ylabel("c-index")
        ax_top.set_xlabel("")
        ax_top.tick_params(labelbottom=False)
        ax_top.grid(visible=True, axis="y", linestyle="--", linewidth=0.5)

        sns.boxplot(
            data=df,
            x=emb_col,
            y=nli_col,
            hue=len_col,
            order=order,
            hue_order=hue_order,
            palette=palette,
            flierprops=flier_style,
            ax=ax_bot,
        )
        if ax_bot.get_legend() is not None:
            ax_bot.get_legend().remove()
        ax_bot.set_ylabel(_NEG_LOG_IBS_LABEL)
        ax_bot.set_xlabel("embeddings")
        ax_bot.grid(visible=True, axis="y", linestyle="--", linewidth=0.5)

        _apply_top_shadow_legend(ax_top, ncol=max(1, min(len(hue_order), 3)), y_anchor=1.22)
        fig.subplots_adjust(hspace=0.04)

    if output_path is not None:
        save_matplotlib_figure(fig, output_path)
    return fig


def plot_per_survival_model_comparison(
    results: pd.DataFrame,
    /,
    *,
    sample_length: str | None = None,
    use_paper_model_names: bool = True,
    model_order: Sequence[str] | None = None,
    embedding_order: Sequence[str] | None = None,
    c_index_ylim: tuple[float, float] | None = None,
    neg_log_ibs_ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    font_size: int = 14,
    use_latex: bool | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Reproduce Paper Fig. 3 (``per_smodel.pdf``): 2-row grouped barplot per survival model.

    Top panel displays ``c-index`` and bottom panel displays ``-log(IBS)`` across the four
    survival models grouped by pretrained embedding.

    Args:
        results: DataFrame containing ``embedding``/``embeddings``, ``survival_model``/``survival-model``,
            ``c_index``/``c-index``, and ``neg_log_ibs`` (or ``ibs``).
        sample_length: Optional duration filter (e.g., ``"8-hour Dataset"``).
        use_paper_model_names: Map ``RandomSurvivalForest -> C-SurvivalForest`` and
            ``NeuralMTLR -> NeuralMultiTask`` to match the printed paper labels.
        model_order: Optional explicit x-axis ordering of survival models.
        embedding_order: Optional explicit hue ordering of embedding methods.
        c_index_ylim: Optional y-axis limits for ``c-index`` (paper used ``(0.5, 0.8)``).
        neg_log_ibs_ylim: Optional y-axis limits for ``-log(IBS)`` (paper used ``(0.6, 2.3)``).
        figsize: Figure size in inches (``(8, 6)`` in paper).
        font_size: Base IEEE font size (``14`` in paper).
        use_latex: Optional override for Matplotlib ``text.usetex``.
        output_path: Optional path (e.g., ``per_smodel.pdf``) to save directly.

    Returns:
        Matplotlib :class:`Figure`.
    """
    df, emb_col, model_col, c_col, nli_col, len_col = _normalize_survival_results_frame(
        results, use_paper_model_names=use_paper_model_names
    )
    if sample_length is not None and len_col in df.columns:
        df = df[df[len_col] == sample_length].copy()
        if df.empty:
            msg = f"No survival results matched sample_length={sample_length!r}"
            raise ValueError(msg)

    default_models = (
        tuple(_PAPER_SURVIVAL_RENAME.get(m, m) for m in _DEFAULT_SURVIVAL_ORDER)
        if use_paper_model_names
        else _DEFAULT_SURVIVAL_ORDER
    )
    present_models = list(dict.fromkeys(df[model_col].astype(str).tolist()))
    x_order = (
        list(model_order)
        if model_order is not None
        else [m for m in default_models if m in present_models]
        + [m for m in present_models if m not in default_models]
    )
    present_embs = list(dict.fromkeys(df[emb_col].astype(str).tolist()))
    hue_order = (
        list(embedding_order)
        if embedding_order is not None
        else [e for e in _DEFAULT_EMBEDDING_ORDER if e in present_embs]
        + [e for e in present_embs if e not in _DEFAULT_EMBEDDING_ORDER]
    )
    palette = list(_IEEE_PALETTE_4[: len(hue_order)]) if len(hue_order) <= len(_IEEE_PALETTE_4) else None

    with ieee_plot_style(font_size=font_size, figsize=figsize, use_latex=use_latex):
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=figsize)
        ax_top, ax_bot = axes[0], axes[1]

        sns.barplot(
            data=df,
            x=model_col,
            y=c_col,
            hue=emb_col,
            order=x_order,
            hue_order=hue_order,
            palette=palette,
            errorbar=None,
            ax=ax_top,
        )
        ax_top.set_ylabel("c-index")
        ax_top.set_xlabel("")
        ax_top.tick_params(labelbottom=False)
        if c_index_ylim is not None:
            ax_top.set_ylim(*c_index_ylim)
        ax_top.grid(visible=True, axis="y", linestyle="--", linewidth=0.5)

        sns.barplot(
            data=df,
            x=model_col,
            y=nli_col,
            hue=emb_col,
            order=x_order,
            hue_order=hue_order,
            palette=palette,
            errorbar=None,
            ax=ax_bot,
        )
        if ax_bot.get_legend() is not None:
            ax_bot.get_legend().remove()
        ax_bot.set_ylabel(_NEG_LOG_IBS_LABEL)
        ax_bot.set_xlabel("survival-model")
        if neg_log_ibs_ylim is not None:
            ax_bot.set_ylim(*neg_log_ibs_ylim)
        ax_bot.tick_params(axis="x", labelrotation=0, labelsize=max(10, font_size - 2))
        ax_bot.grid(visible=True, axis="y", linestyle="--", linewidth=0.5)

        _apply_top_shadow_legend(ax_top, ncol=max(1, min(len(hue_order), 4)), y_anchor=1.22)
        fig.subplots_adjust(hspace=0.04)

    if output_path is not None:
        save_matplotlib_figure(fig, output_path)
    return fig


def plot_risk_classification_timeline(
    metadata_test: pd.DataFrame,
    relapse_episodes: pd.DataFrame,
    risk_labels: NDArray[np.str_] | Sequence[str],
    /,
    *,
    selected_users: Sequence[str | int] | None = None,
    max_users: int = 5,
    max_samples: int | None = 500,
    width: int = 600,
    height: int = 250,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> alt.LayerChart:
    """Reproduce Paper Fig. 4 (``final_risk_classification.pdf``): Altair 3-tier risk timeline.

    Displays solid black relapse bars, sample dots colored by predicted risk tier
    (``low``, ``medium``, ``high``), and thin black connector lines pointing from each
    recording to its next relapse / censoring time.

    Args:
        metadata_test: Test-set sample metadata aligned row-for-row with ``risk_labels``.
        relapse_episodes: Relapse interval table with ``user_id``, ``start``, ``end``.
        risk_labels: Predicted ``"low"`` / ``"medium"`` / ``"high"`` tier per test sample.
        selected_users: Optional subset of users to show (paper displays 5 diverse users).
        max_users: Maximum number of users displayed when ``selected_users`` is ``None``.
        max_samples: Max sample dots to display (``500`` in paper; ``None`` plots all).
        width: Chart width in pixels (``600`` in paper).
        height: Chart height in pixels (``250`` in paper).
        seed: Random seed when downsampling to ``max_samples``.
        output_path: Optional path (e.g., ``final_risk_classification.pdf``) to save directly.

    Returns:
        Layered Altair chart.
    """
    _ensure_altair_sfmono_theme()
    labels_list = [str(lbl).strip().lower() for lbl in risk_labels]
    if len(labels_list) != len(metadata_test):
        msg = f"risk_labels length ({len(labels_list)}) must match metadata_test rows ({len(metadata_test)})"
        raise ValueError(msg)

    enriched = metadata_test.copy().reset_index(drop=True)
    enriched["Risk"] = labels_list
    enriched["risk_sort"] = enriched["Risk"].map(_RISK_ORDER).fillna(1).astype(int)

    chosen = selected_users
    if chosen is None:
        unique_users = list(dict.fromkeys(enriched["user_id"].astype(str).tolist()))
        chosen = unique_users[:max_users]

    sample_df, eps_df = _prepare_timeline_frames(
        enriched, relapse_episodes, selected_users=chosen, max_samples=max_samples, seed=seed
    )
    arrows_df = _build_connector_arrows(sample_df)

    y_sort = alt.EncodingSortField(field="User", order="ascending")
    relapses = (
        alt.Chart(eps_df)
        .mark_bar()
        .encode(
            x=alt.X("From:T", title=""),
            x2=alt.X2("To:T"),
            color=alt.value("#000000"),
            y=alt.Y("index:N", sort=y_sort, title=""),
        )
        .properties(width=width, height=height)
    )

    samples = (
        alt.Chart(sample_df)
        .mark_circle(size=200)
        .encode(
            x=alt.X("Days:T", title=""),
            y=alt.Y("index:N", sort=y_sort, title=""),
            color=alt.Color(
                "Risk:N",
                sort=list(_RISK_DOMAIN),
                scale=alt.Scale(domain=list(_RISK_DOMAIN), range=[_RISK_TIER_COLORS[k] for k in _RISK_DOMAIN]),
            ),
        )
        .properties(width=width, height=height)
    )

    arrows = (
        alt.Chart(arrows_df)
        .mark_line()
        .encode(
            x=alt.X("x:T", title=""),
            y=alt.Y("class:N", sort=y_sort, title=""),
            detail=alt.Detail("y:N"),
            color=alt.value("#000000"),
            strokeWidth=alt.value(0.3),
        )
        .properties(width=width, height=height)
    )

    chart: alt.LayerChart = alt.layer(relapses, samples, arrows).resolve_scale(color="independent")
    if output_path is not None:
        save_altair_chart(chart, output_path)
    return chart


def plot_person_id_metrics(
    metrics_df: pd.DataFrame,
    /,
    *,
    value_col: str = "mean",
    metric_order: Sequence[str] | None = None,
    model_order: Sequence[str] | None = None,
    figsize: tuple[float, float] = (8.0, 4.0),
    font_size: int = 16,
    use_latex: bool | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Reproduce ``identification.pdf``: Person Identification grouped metric barplot.

    Compares ``F1``, ``Recall``, ``Precision``, ``AUPRC``, ``Accuracy``, and ``AUROC``
    across each embedding method (Section 4, Table 2).

    Args:
        metrics_df: Long-format DataFrame with columns ``["Metric", "model", value_col]``.
        value_col: Column to plot on the y-axis (``"mean"`` or ``"std"``).
        metric_order: Optional explicit x-axis metric order.
        model_order: Optional explicit legend/hue model order.
        figsize: Figure size in inches (``(8, 4)`` in notebook).
        font_size: Base IEEE font size (``16`` in notebook).
        use_latex: Optional override for Matplotlib ``text.usetex``.
        output_path: Optional path (e.g., ``identification.pdf``) to save directly.

    Returns:
        Matplotlib :class:`Figure`.
    """
    present_metrics = list(dict.fromkeys(metrics_df["Metric"].astype(str).tolist()))
    x_order = (
        list(metric_order)
        if metric_order is not None
        else [m for m in _DEFAULT_METRIC_ORDER if m in present_metrics]
        + [m for m in present_metrics if m not in _DEFAULT_METRIC_ORDER]
    )
    present_models = list(dict.fromkeys(metrics_df["model"].astype(str).tolist()))
    hue_order = list(model_order) if model_order is not None else present_models
    palette = list(_IEEE_PALETTE_4[: len(hue_order)]) if len(hue_order) <= len(_IEEE_PALETTE_4) else None

    with ieee_plot_style(font_size=font_size, figsize=figsize, use_latex=use_latex):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.barplot(
            data=metrics_df,
            x="Metric",
            y=value_col,
            hue="model",
            order=x_order,
            hue_order=hue_order,
            palette=palette,
            errorbar=None,
            ax=ax,
        )
        ax.set_xlabel("Metric")
        ax.set_ylabel(value_col)
        ax.grid(visible=True, axis="y", linestyle="--", linewidth=0.5)
        _apply_top_shadow_legend(ax, ncol=max(1, min(len(hue_order), 4)), y_anchor=1.22)
        fig.tight_layout()

    if output_path is not None:
        save_matplotlib_figure(fig, output_path)
    return fig


def plot_user_sample_distribution(
    user_ids: NDArray[np.integer[Any]] | NDArray[np.str_] | NDArray[np.object_] | Sequence[object],
    /,
    *,
    min_samples_threshold: int | None = None,
    figsize: tuple[float, float] = (8.0, 4.0),
    font_size: int = 16,
    use_latex: bool | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Reproduce ``identification_dist.pdf``: per-user recording count barplot.

    Descending royalblue bar chart highlighting the per-patient sample imbalance motivating
    the ``>= 100``-sample retention filter (Section 4).

    Args:
        user_ids: Per-sample user identifier array or sequence.
        min_samples_threshold: Optional dashed horizontal line marking the retention cutoff.
        figsize: Figure size in inches (``(8, 4)`` in notebook).
        font_size: Base IEEE font size (``16`` in notebook).
        use_latex: Optional override for Matplotlib ``text.usetex``.
        output_path: Optional path (e.g., ``identification_dist.pdf``) to save directly.

    Returns:
        Matplotlib :class:`Figure`.
    """
    raw = [str(u) for u in np.asarray(user_ids).tolist()]
    label_map, _ = _format_user_labels(raw)
    counts = pd.Series([label_map[u] for u in raw]).value_counts().reset_index()
    counts.columns = ["users", "counts"]

    with ieee_plot_style(font_size=font_size, figsize=figsize, use_latex=use_latex):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.barplot(data=counts, x="users", y="counts", color="royalblue", ax=ax)
        if min_samples_threshold is not None:
            ax.axhline(min_samples_threshold, color="#DF2020", linestyle="--", linewidth=1.2)
        ax.set_xlabel("users")
        ax.set_ylabel("counts")
        ax.tick_params(axis="x", labelrotation=90)
        ax.grid(visible=True, axis="y", linestyle="--", linewidth=0.5)
        fig.tight_layout()

    if output_path is not None:
        save_matplotlib_figure(fig, output_path)
    return fig


def plot_survival_time_distribution(
    *,
    time_train: NDArray[np.floating[Any]],
    event_train: NDArray[np.bool_],
    time_test: NDArray[np.floating[Any]],
    event_test: NDArray[np.bool_],
    bins: int | str = "auto",
    figsize: tuple[float, float] = (14.0, 7.0),
    font_size: int = 16,
    use_latex: bool | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Reproduce ``survival_distribution.pdf``: train vs. test survival time histograms.

    Stacked 2-row shared-x histogram comparing survival time distributions between the 60%
    training split and 40% test split, colored by censoring status (``censored = True`` vs. ``False``).

    Args:
        time_train: Training survival/censoring times in days.
        event_train: Training event indicators (``True`` = relapse occurred, ``False`` = right-censored).
        time_test: Test survival/censoring times in days.
        event_test: Test event indicators (``True`` = relapse occurred, ``False`` = right-censored).
        bins: Histogram bin specification passed to ``seaborn.histplot``.
        figsize: Figure size in inches.
        font_size: Base IEEE font size.
        use_latex: Optional override for Matplotlib ``text.usetex``.
        output_path: Optional path (e.g., ``survival_distribution.pdf``) to save directly.

    Returns:
        Matplotlib :class:`Figure`.
    """
    train_df = pd.DataFrame(
        {"survival time": np.asarray(time_train, dtype=np.float64), "censored": ~np.asarray(event_train, dtype=bool)}
    )
    test_df = pd.DataFrame(
        {"survival time": np.asarray(time_test, dtype=np.float64), "censored": ~np.asarray(event_test, dtype=bool)}
    )

    with ieee_plot_style(font_size=font_size, figsize=figsize, use_latex=use_latex):
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=figsize)
        ax_tr, ax_te = axes[0], axes[1]

        sns.histplot(
            data=train_df,
            x="survival time",
            hue="censored",
            hue_order=[False, True],
            palette=list(_HIST_PALETTE_2),
            bins=bins,
            ax=ax_tr,
        )
        ax_tr.set_title("Training set")
        ax_tr.set_ylabel("Count")
        ax_tr.grid(visible=True, linestyle="--", linewidth=0.5)

        sns.histplot(
            data=test_df,
            x="survival time",
            hue="censored",
            hue_order=[False, True],
            palette=list(_HIST_PALETTE_2),
            bins=bins,
            ax=ax_te,
        )
        ax_te.set_title("Test set")
        ax_te.set_xlabel("survival time")
        ax_te.set_ylabel("Count")
        ax_te.grid(visible=True, linestyle="--", linewidth=0.5)
        fig.tight_layout()

    if output_path is not None:
        save_matplotlib_figure(fig, output_path)
    return fig


def plot_risk_group_histogram(
    risk_scores: NDArray[np.floating[Any]],
    risk_labels: NDArray[np.str_] | Sequence[str],
    /,
    *,
    cutoffs: tuple[float, float],
    num_bins: int = 30,
    figsize: tuple[float, float] = (12.0, 4.0),
    font_size: int = 14,
    use_latex: bool | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Reproduce ``pysurvival.utils.display.create_risk_groups`` risk-score distribution plot.

    Shades the predicted risk-score histogram by ``low`` / ``medium`` / ``high`` tiers
    with vertical dashed lines at the train-fitted tier boundaries (Section 4).

    Args:
        risk_scores: 1D predicted risk scores ``(N,)``.
        risk_labels: Per-sample ``"low"`` / ``"medium"`` / ``"high"`` tier labels.
        cutoffs: ``(low_cutoff, high_cutoff)`` thresholds separating risk tiers.
        num_bins: Number of histogram bins across the full risk range.
        figsize: Figure size in inches.
        font_size: Base IEEE font size.
        use_latex: Optional override for Matplotlib ``text.usetex``.
        output_path: Optional path to save the figure directly.

    Returns:
        Matplotlib :class:`Figure`.
    """
    scores = np.asarray(risk_scores, dtype=np.float64)
    labels = np.asarray([str(lbl).strip().lower() for lbl in risk_labels])
    lo, hi = float(np.min(scores)), float(np.max(scores))
    bin_edges = np.linspace(lo, hi + 1e-9, num_bins + 1) if hi > lo else np.linspace(lo - 0.5, hi + 0.5, num_bins + 1)

    with ieee_plot_style(font_size=font_size, figsize=figsize, use_latex=use_latex):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        for tier in _RISK_DOMAIN:
            tier_scores = scores[labels == tier]
            if len(tier_scores) > 0:
                ax.hist(
                    tier_scores,
                    bins=bin_edges.tolist(),
                    color=_RISK_TIER_COLORS[tier],
                    alpha=0.75,
                    edgecolor="#222222",
                    linewidth=0.5,
                    label=f"{tier} risk (n={len(tier_scores)})",
                )
        ax.axvline(cutoffs[0], color="#222222", linestyle="--", linewidth=1.2)
        ax.axvline(cutoffs[1], color="#222222", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Predicted risk score")
        ax.set_ylabel("Count")
        ax.set_title("Risk Score Distribution by Stratification Tier")
        ax.grid(visible=True, axis="y", linestyle="--", linewidth=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC")
        fig.tight_layout()

    if output_path is not None:
        save_matplotlib_figure(fig, output_path)
    return fig


def plot_risk_group_survival_curves(
    model: BaseSurvivalModel,
    x_test: NDArray[np.floating[Any]],
    /,
    *,
    time_test: NDArray[np.floating[Any]],
    event_test: NDArray[np.bool_],
    risk_labels: NDArray[np.str_] | Sequence[str],
    seed: int = 42,
    figsize: tuple[float, float] = (10.0, 5.5),
    font_size: int = 14,
    use_latex: bool | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot representative per-tier predicted survival curves against observed relapse time.

    Selects one uncensored test sample per risk tier (``low``, ``medium``, ``high``), plots
    its predicted survival curve ``S(t)``, and marks its observed time-to-relapse ``T`` with
    a matching vertical dashed line.

    Args:
        model: Fitted :class:`BaseSurvivalModel`.
        x_test: Test covariate matrix ``(N, F)``.
        time_test: Observed test survival times in days ``(N,)``.
        event_test: Test event indicators ``(N,)``.
        risk_labels: Per-sample ``"low"`` / ``"medium"`` / ``"high"`` tier labels.
        seed: Random seed for choosing one representative uncensored sample per tier.
        figsize: Figure size in inches.
        font_size: Base IEEE font size.
        use_latex: Optional override for Matplotlib ``text.usetex``.
        output_path: Optional path to save the figure directly.

    Returns:
        Matplotlib :class:`Figure`.
    """
    rng = np.random.default_rng(seed)
    x_arr = np.asarray(x_test, dtype=np.float64)
    t_arr = np.asarray(time_test, dtype=np.float64)
    e_arr = np.asarray(event_test, dtype=bool)
    labels = np.asarray([str(lbl).strip().lower() for lbl in risk_labels])
    times = model.times

    with ieee_plot_style(font_size=font_size, figsize=figsize, use_latex=use_latex):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        plotted_groups: list[str] = []

        for idx, tier in enumerate(_RISK_DOMAIN):
            candidates = np.flatnonzero((labels == tier) & e_arr)
            if len(candidates) == 0:
                candidates = np.flatnonzero(labels == tier)
            if len(candidates) == 0:
                continue
            chosen = int(rng.choice(candidates))
            surv_curve = model.predict_survival(x_arr[chosen : chosen + 1], times=times).ravel()
            color = _RISK_TIER_COLORS[tier]
            ax.plot(times, surv_curve, color=color, linewidth=2.0, label=f"{tier} risk")
            t_obs = float(t_arr[chosen])
            ax.axvline(x=t_obs, color=color, linestyle="--", linewidth=1.3)
            ax.annotate(
                f"T={t_obs:.1f}",
                xy=(t_obs, min(0.95, 0.45 + 0.15 * idx)),
                xytext=(t_obs, min(0.95, 0.45 + 0.15 * idx)),
                color=color,
                fontsize=max(10, font_size - 2),
            )
            plotted_groups.append(tier)

        groups_str = ", ".join(plotted_groups)
        ax.set_title(f"Comparing Survival Functions ({groups_str} risk grades)")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival probability S(t)")
        ax.set_ylim(0.0, 1.05)
        ax.grid(visible=True, linestyle="--", linewidth=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC")
        fig.tight_layout()

    if output_path is not None:
        save_matplotlib_figure(fig, output_path)
    return fig

#!/usr/bin/env python3
"""Human vs NN next-fixation-generation comparison figure (2 rows x 3 columns).

Row 1 = Human, Row 2 = Network.

Column 1: Subject median fixation advantage (horizontal orientation,
           symmetric x-limits around 0).
Column 2: Bidirectional template + observed transition heatmaps (left),
           bidirectional delta-similarity scatter (right).
Column 3: Delta fraction of transitions by run length (all fixations).

Also generates two supplementary figures:
  - TransitionSupplement: 2 rows x 2 cols showing forward/backward templates
    (same layout as columns 2-3 of main figure, but for forward and backward).
  - AdvantageSupplement: Fixation advantage broken down by Relevant vs
    Irrelevant, with Humans and Network side by side.

Sweep transition data is cached to disk so that subsequent runs skip the
expensive shuffle computation.

Example
-------
conda run -n analysis python metarnn/plot_NN_H_next_fixation_gen.py \
  --nn-root metarnn/simulations/human_like_04_04_input5 \
  --tag 04_04_input5 \
  --n-sims 1000 \
  --no-show
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.lib.analyze_choice_fixation_sweeps import (  # noqa: E402
    PANEL_ALL,
    list_subjects,
    mean_ci95,
    nanmean_safe,
    sweep_template_matrices,
)
from analysis.lib.plot_fixation_advantage_violin import (  # noqa: E402
    COLORS as FA_COLORS,
    compute_fixation_advantages,
    load_all_fixations,
)
from metarnn.lib.plot_NN_sweep_transitions import (  # noqa: E402
    MAX_RUN_BIN,
    _ensure_dir,
    compute_sweep_transition_data,
)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(
    base_dir: Path, buffer: int, n_sims: int, seed: int,
    exclude_subjects: Tuple[str, ...], collapse_null_shuffles: bool = True,
) -> str:
    """Deterministic MD5-based cache key from computation parameters."""
    raw = f"{base_dir.resolve()}|{buffer}|{n_sims}|{seed}|{sorted(exclude_subjects)}"
    if collapse_null_shuffles:
        raw += "|collapseNull=1"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _sweep_cache_path(base_dir: Path, cache_key: str) -> Path:
    return base_dir / "output" / "next_fixation_gen" / "cache" / f"sweep_data_{cache_key}.pkl"


def _fa_cache_key(
    data_dir: Path, excluded_subjects: Tuple[str, ...],
    other_items_mode: str, advantage_timepoint: str,
) -> str:
    """Deterministic cache key for a fixation-advantage dataframe."""
    raw = (
        f"{data_dir.resolve()}|{sorted(excluded_subjects)}"
        f"|{other_items_mode}|{advantage_timepoint}"
    )
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _fa_cache_path(cache_root: Path, cache_key: str) -> Path:
    return cache_root / "output" / "next_fixation_gen" / "cache" / f"fa_data_{cache_key}.pkl"


def _load_or_compute_fa(
    *,
    cache_root: Path,
    data_dir: Path,
    excluded_subjects: Tuple[str, ...],
    other_items_mode: str = "all",
    advantage_timepoint: str = "pre",
    label: str = "",
) -> pd.DataFrame:
    """Load cached or compute and cache a fixation-advantage dataframe."""
    key = _fa_cache_key(
        data_dir, excluded_subjects, other_items_mode, advantage_timepoint,
    )
    cache_path = _fa_cache_path(cache_root, key)

    if cache_path.exists():
        print(f"Loading cached {label} fixation advantage from {cache_path} ...", flush=True)
        return pd.read_pickle(cache_path)

    print(f"Computing {label} fixation advantage ...", flush=True)
    fix = load_all_fixations(data_dir, excluded_subjects=excluded_subjects)
    adv = compute_fixation_advantages(
        fix, other_items_mode=other_items_mode, advantage_timepoint=advantage_timepoint,
    )

    _ensure_dir(cache_path.parent)
    adv.to_pickle(cache_path)
    print(f"  Cached {label} fixation advantage to {cache_path}", flush=True)
    return adv


def _derive_subject_ids(
    base_dir: Path,
    *,
    buffer: int = 50,
    exclude_subjects: Tuple[str, ...] = (),
    data: Dict[str, Any],
) -> Dict[str, list]:
    """Re-derive subject IDs for cached sweep data that lacks them."""
    data_root = base_dir / "data"
    sids = list_subjects(data_root)
    exclude_set = set(str(s) for s in exclude_subjects)
    sids = [s for s in sids if s not in exclude_set]

    valid_sids: list = []
    for sid in sids:
        log_path = data_root / sid / f"{sid}_MAIN_logfile_7.csv"
        fix_path = data_root / sid / f"{sid}_fixations_df_original_buffer_{buffer}.csv"
        if not fix_path.exists():
            fix_path = data_root / sid / f"{sid}_fixations_df_original.csv"
        if log_path.exists() and fix_path.exists():
            valid_sids.append(sid)
    sids = sorted(valid_sids)

    # Validate against array shapes for PANEL_ALL.
    n_sweep = data["transprop_obs"][PANEL_ALL].shape[0]
    if len(sids) != n_sweep:
        print(
            f"  WARNING: derived subject count ({len(sids)}) != "
            f"sweep array size ({n_sweep}) for PANEL_ALL",
            flush=True,
        )
    return {PANEL_ALL: sids}


def _load_or_compute_sweep(
    base_dir: Path,
    *,
    buffer: int = 50,
    n_sims: int = 1000,
    seed: int = 123,
    exclude_subjects: Tuple[str, ...] = ("107", "131"),
    label: str = "",
    collapse_null_shuffles: bool = True,
) -> Dict[str, Any]:
    """Load sweep data from cache, or compute and cache it."""
    key = _cache_key(
        base_dir, buffer, n_sims, seed, exclude_subjects,
        collapse_null_shuffles=collapse_null_shuffles,
    )
    cache_path = _sweep_cache_path(base_dir, key)

    if cache_path.exists():
        print(f"Loading cached {label} sweep data from {cache_path} ...", flush=True)
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        # Verify parameters match
        meta = data.get("_cache_meta", {})
        if (
            meta.get("n_sims") == n_sims
            and meta.get("seed") == seed
            and meta.get("buffer") == buffer
            and tuple(meta.get("exclude_subjects", ())) == tuple(exclude_subjects)
            and bool(meta.get("collapse_null_shuffles", False)) == bool(collapse_null_shuffles)
        ):
            # Backfill subject_ids if missing from older cache files.
            if "subject_ids" not in data:
                data["subject_ids"] = _derive_subject_ids(
                    base_dir, buffer=buffer,
                    exclude_subjects=exclude_subjects, data=data,
                )
            return data
        print("  Cache parameters mismatch — recomputing.", flush=True)

    print(f"Computing {label} sweep transition data ...", flush=True)
    data = compute_sweep_transition_data(
        base_dir=base_dir,
        buffer=buffer,
        n_sims=n_sims,
        seed=seed,
        exclude_subjects=exclude_subjects,
        collapse_null_shuffles=collapse_null_shuffles,
    )

    # Attach metadata and save
    data["_cache_meta"] = {
        "base_dir": str(base_dir.resolve()),
        "buffer": buffer,
        "n_sims": n_sims,
        "seed": seed,
        "exclude_subjects": list(exclude_subjects),
        "collapse_null_shuffles": bool(collapse_null_shuffles),
    }
    _ensure_dir(cache_path.parent)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached to {cache_path}", flush=True)

    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subset_advantage(df: pd.DataFrame, item_subset: str) -> pd.DataFrame:
    """Subset fixation advantage data by item relevance category."""
    if item_subset == "all":
        out = df.copy()
        out["fixation_advantage"] = out["fixation_advantage_all"]
        return out
    if item_subset == "relevant":
        out = df[df["is_relevant"] == 1].copy()
        out["fixation_advantage"] = out["fixation_advantage_relevant"]
        return out
    if item_subset == "irrelevant":
        out = df[df["is_relevant"] == 0].copy()
        out["fixation_advantage"] = out["fixation_advantage_irrelevant"]
        return out
    raise ValueError(f"Invalid item_subset: {item_subset}")


def _sem(x: np.ndarray) -> float:
    """Standard error of the mean."""
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _fa_subject_medians_with_ids(
    df_all: pd.DataFrame,
    time_scale: float = 1.0,
) -> pd.DataFrame:
    """Per-subject median fixation advantage, preserving subject IDs."""
    vals = pd.to_numeric(df_all["fixation_advantage"], errors="coerce")
    vals = vals[np.isfinite(vals)]
    sub_col = df_all.loc[vals.index, "subject"]
    medians = vals.groupby(sub_col).median() * time_scale
    return medians.reset_index().rename(
        columns={"fixation_advantage": "median_fixation_advantage"}
    )


def _save_stats_csvs(
    *,
    human_adv: pd.DataFrame,
    nn_adv: pd.DataFrame,
    human_sweep: Dict[str, Any],
    nn_sweep: Dict[str, Any],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Save per-subject CSVs for statistical testing in R."""
    stats_dir = out_dir / "stats"
    _ensure_dir(stats_dir)
    suffix = f"_{tag}" if tag else ""

    # --- CSV 1: Fixation advantage subject medians ---
    conditions = ["all", "relevant", "irrelevant"]
    group_configs = [
        {"group": "human", "adv": human_adv, "time_scale": 1.0 / 1000.0},
        {"group": "nn", "adv": nn_adv, "time_scale": 1.0},
    ]
    fa_rows: list = []
    for gcfg in group_configs:
        for cond in conditions:
            subset = _subset_advantage(gcfg["adv"], cond)
            meds_df = _fa_subject_medians_with_ids(subset, gcfg["time_scale"])
            meds_df["group"] = gcfg["group"]
            meds_df["condition"] = cond
            fa_rows.append(meds_df)
    fa_df = pd.concat(fa_rows, ignore_index=True)
    fa_df = fa_df[["subject", "group", "condition", "median_fixation_advantage"]]
    fa_path = stats_dir / f"fixation_advantage_subject_medians{suffix}.csv"
    fa_df.to_csv(fa_path, index=False)
    print(f"Saved: {fa_path} ({len(fa_df)} rows)")

    # --- CSV 1b: Human fixation-level advantage (for hierarchical model) ---
    time_scale_human = 1.0 / 1000.0
    fa_fix_rows: list = []
    for cond in conditions:
        subset = _subset_advantage(human_adv, cond)
        vals = pd.to_numeric(subset["fixation_advantage"], errors="coerce")
        keep = np.isfinite(vals)
        fa_fix_rows.append(pd.DataFrame({
            "subject": subset.loc[keep.index[keep], "subject"],
            "condition": cond,
            "fixation_advantage": vals[keep] * time_scale_human,
        }))
    fa_fix_df = pd.concat(fa_fix_rows, ignore_index=True)
    fa_fix_path = stats_dir / f"fixation_advantage_human_fixlevel{suffix}.csv"
    fa_fix_df.to_csv(fa_fix_path, index=False)
    print(f"Saved: {fa_fix_path} ({len(fa_fix_df)} rows)")

    # --- CSV 2: Delta similarity by template ---
    template_names = ["bidirectional", "forward", "backward"]
    dsim_rows: list = []
    for group_label, sweep in [("human", human_sweep), ("nn", nn_sweep)]:
        sids = sweep["subject_ids"][PANEL_ALL]
        for tname in template_names:
            delta_arr = sweep["delta_similarity"][PANEL_ALL][tname]
            for i, sid in enumerate(sids):
                dsim_rows.append({
                    "subject": sid,
                    "group": group_label,
                    "template": tname,
                    "delta_similarity": float(delta_arr[i]),
                })
    dsim_df = pd.DataFrame(dsim_rows)
    dsim_path = stats_dir / f"delta_similarity_subject{suffix}.csv"
    dsim_df.to_csv(dsim_path, index=False)
    print(f"Saved: {dsim_path} ({len(dsim_df)} rows)")

    # --- CSV 3: Sequence length delta ---
    seqlen_rows: list = []
    for group_label, sweep in [("human", human_sweep), ("nn", nn_sweep)]:
        sids = sweep["subject_ids"][PANEL_ALL]
        obs = np.asarray(sweep["transprop_obs"][PANEL_ALL], dtype=float)
        nul = np.asarray(sweep["transprop_null"][PANEL_ALL], dtype=float)
        delta = obs - nul
        bin_labels = sweep["run_bin_labels"]
        for i, sid in enumerate(sids):
            for j, bl in enumerate(bin_labels):
                seqlen_rows.append({
                    "subject": sid,
                    "group": group_label,
                    "seq_length": bl,
                    "delta_proportion": float(delta[i, j]),
                })
    seqlen_df = pd.DataFrame(seqlen_rows)
    seqlen_path = stats_dir / f"seq_length_delta_subject{suffix}.csv"
    seqlen_df.to_csv(seqlen_path, index=False)
    print(f"Saved: {seqlen_path} ({len(seqlen_df)} rows)")


# ---------------------------------------------------------------------------
# Column 1: Vertical fixation advantage (All items only)
# ---------------------------------------------------------------------------

def _fa_subject_medians(
    df_all: pd.DataFrame,
    time_scale: float = 1.0,
) -> np.ndarray:
    """Compute per-subject median fixation advantage (scaled)."""
    vals = pd.to_numeric(df_all["fixation_advantage"], errors="coerce")
    vals = vals[np.isfinite(vals)]
    sub_col = df_all.loc[vals.index, "subject"]
    return vals.groupby(sub_col).median().to_numpy(dtype=float) * time_scale


def _plot_fixation_advantage_vertical(
    ax: plt.Axes,
    df_all: pd.DataFrame,
    *,
    time_unit_label: str = "s",
    time_scale: float = 1.0,
    ylim: Optional[Tuple[float, float]] = None,
    fill_color: str = ".7",
) -> None:
    """Vertical bar + strip of subject median fixation advantage.

    *time_scale* is applied to the raw values before plotting (e.g. 1/1000
    to convert ms -> s for human data).
    *ylim* overrides the automatic symmetric limits if provided.
    *fill_color* sets the bar fill color (default gray).
    """
    rng = np.random.default_rng(42)

    sub_medians = _fa_subject_medians(df_all, time_scale)

    n = len(sub_medians)
    jitter = rng.uniform(-0.08, 0.08, size=n)

    group_mean = float(np.mean(sub_medians))
    sem_val = _sem(sub_medians)

    _edge_gray = ".7"
    _bar_w = 0.5

    # Fill bar (behind dots)
    ax.bar(0, group_mean, width=_bar_w,
           color=fill_color, edgecolor="none", linewidth=0, zorder=2)
    # Black outline bar (on top of dots)
    ax.bar(0, group_mean, width=_bar_w,
           color="none", edgecolor="black", linewidth=1.5, zorder=4)

    # Individual subject dots (semi-transparent white fill, gray edge)
    for i in range(n):
        ax.scatter(
            jitter[i], sub_medians[i],
            s=6 ** 2,
            facecolor=(1, 1, 1, 0.5),
            edgecolor=_edge_gray,
            linewidth=1,
            zorder=3,
        )

    # SEM error bar (no caps, matching overview style)
    ax.errorbar(
        0, group_mean,
        yerr=sem_val,
        fmt="none",
        ecolor="black",
        linewidth=1.5,
        capsize=0,
        zorder=5,
    )

    ax.axhline(0, color="black", linestyle="-", linewidth=1.5, zorder=1)
    ax.set_xticks([])
    ax.set_xlim(-0.7, 0.7)  # white space around bar
    ax.set_ylabel(f"Time advantage ({time_unit_label})")
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Symmetric y-limits around 0
        max_abs = float(np.nanmax(np.abs(sub_medians))) if len(sub_medians) > 0 else 1.0
        margin = max_abs * 0.1
        ax.set_ylim(-(max_abs + margin), max_abs + margin)


# ---------------------------------------------------------------------------
# Column 2: Heatmap + bidirectional delta-similarity
# ---------------------------------------------------------------------------

def _plot_single_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    *,
    cmap="viridis",
    title: str = "",
    cbar: bool = False,
    cbar_ax: Optional[plt.Axes] = None,
    cbar_kws: Optional[Dict[str, Any]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    mask_diagonal: bool = False,
    border: bool = True,
    cell_lines: bool = True,
) -> None:
    """Plot a single 6x6 heatmap.

    If ``mask_diagonal=True``, the main diagonal is treated as missing and
    rendered as the axes background (white) rather than a colored cell.
    When ``border`` is True, a black box is drawn around the heatmap; when
    ``cell_lines`` is True, each cell gets a thin outline.
    """
    n_items = 6

    # Cells to hide: NaN inputs plus (optionally) the main diagonal.
    mask = ~np.isfinite(mat)
    if mask_diagonal:
        mask = mask | np.eye(mat.shape[0], dtype=bool)

    if vmin is None:
        finite = mat[np.isfinite(mat) & ~(mask_diagonal & np.eye(mat.shape[0], dtype=bool))]
        vmin = 0.0
        vmax = max(1e-6, float(np.nanmax(finite))) if finite.size > 0 else 1.0

    if cbar_kws is None:
        cbar_kws = {"shrink": 0.8} if (cbar and cbar_ax is None) else None

    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        square=True,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        mask=mask,
        linewidths=1.5 if cell_lines else 0,
        linecolor="black" if cell_lines else "none",
        xticklabels=[str(j) for j in range(1, n_items + 1)],
        yticklabels=[str(j) for j in range(1, n_items + 1)],
    )
    if mask_diagonal:
        for i in range(mat.shape[0]):
            ax.add_patch(plt.Rectangle(
                (i, i), 1, 1,
                facecolor="white",
                hatch="/////",
                edgecolor="black",
                linewidth=1.5,
                zorder=3,
            ))
    if border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
    if cbar and cbar_ax is not None:
        for spine in cbar_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
    if title:
        ax.set_title(title, fontsize=22)
    ax.set_xlabel("Next pos." if show_xlabel else "", fontsize=14)
    ax.set_ylabel("Current pos." if show_ylabel else "", fontsize=14)
    ax.tick_params(length=0)


def _plot_single_delta_similarity(
    ax: plt.Axes,
    delta_vals: np.ndarray,
    *,
    ylabel: str = "Similarity above chance",
    ylim: Optional[Tuple[float, float]] = None,
    yticks: Optional[Sequence[float]] = None,
) -> None:
    """Bar + strip of per-subject bidirectional delta-similarity (styled like fixation advantage)."""
    y = np.asarray(delta_vals, dtype=float)
    y = y[np.isfinite(y)]

    rng = np.random.default_rng(0)
    _gray = ".7"
    _bar_w = 0.5

    m = float(nanmean_safe(y))
    sem_val = _sem(y)

    # Gray fill bar (behind dots)
    ax.bar(0, m, width=_bar_w,
           color=_gray, edgecolor="none", linewidth=0, zorder=2)
    # Black outline bar (on top of dots)
    ax.bar(0, m, width=_bar_w,
           color="none", edgecolor="black", linewidth=1.5, zorder=4)

    # Individual subject dots
    if len(y) > 0:
        jitter = rng.uniform(-0.08, 0.08, size=len(y))
        for i in range(len(y)):
            ax.scatter(
                jitter[i], y[i],
                s=6 ** 2,
                facecolor=(1, 1, 1, 0.5),
                edgecolor=_gray,
                linewidth=1,
                zorder=3,
            )

    # SEM error bar
    ax.errorbar(
        0, m, yerr=sem_val,
        fmt="none", ecolor="black",
        linewidth=1.5, capsize=0, zorder=5,
    )

    ax.set_xticks([])
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)


# ---------------------------------------------------------------------------
# Column 3: Run-length delta bars
# ---------------------------------------------------------------------------

def _plot_single_runlength_delta(
    ax: plt.Axes,
    transprop_obs_all: np.ndarray,
    transprop_null_all: np.ndarray,
    bin_labels: Sequence[str],
    *,
    ylabel: str = "Proportion above chance",
    ylim: Optional[Tuple[float, float]] = None,
    yticks: Optional[Sequence[float]] = None,
) -> None:
    """Delta (observed - null) bar plot per run-length bin for PANEL_ALL."""
    k = len(bin_labels)
    xs = np.arange(k, dtype=float)
    _gray = ".7"
    _bar_w = 0.8

    obs = np.asarray(transprop_obs_all, dtype=float)
    nul = np.asarray(transprop_null_all, dtype=float)
    n = min(obs.shape[0], nul.shape[0])

    if n == 0:
        mean_val = np.full(k, np.nan)
        ci = np.full(k, np.nan)
    else:
        delta = obs[:n, :] - nul[:n, :]
        mean_val, ci = mean_ci95(delta, axis=0)

    # Gray fill bars (behind)
    ax.bar(xs, mean_val, _bar_w,
           color=_gray, edgecolor="none", linewidth=0, zorder=2)
    # Black outline bars (on top)
    ax.bar(xs, mean_val, _bar_w,
           color="none", edgecolor="black", linewidth=1.5, zorder=4)
    # Error bars
    ax.errorbar(xs, mean_val, yerr=ci,
                fmt="none", ecolor="black", capsize=0, linewidth=1.5, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(bin_labels))
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)


# ---------------------------------------------------------------------------
# Supplement 1: Transition Supplement (forward / backward)
# ---------------------------------------------------------------------------

def _create_transition_supplement(
    *,
    human_sweep: Dict[str, Any],
    nn_sweep: Dict[str, Any],
    templates: Dict[str, np.ndarray],
    out_dir: Path,
    tag: str = "",
    show: bool = True,
) -> Path:
    """2 rows (Humans/Network) x 2 cols (Forward/Backward) transition figure."""

    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    with plt.rc_context({
        "font.family": "Arial",
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "hatch.linewidth": 1.5,
    }):
        fig = plt.figure(figsize=(12, 8.5))
        gs_outer = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.55)

        # Shared heatmap color scale
        all_vals = []
        for tname in ("forward", "backward"):
            t = templates[tname]
            finite = t[np.isfinite(t)]
            if finite.size > 0:
                all_vals.append(finite)
        hm_vmax = max(1e-6, float(np.nanmax(np.concatenate(all_vals)))) if all_vals else 1.0

        col_templates = ["forward", "backward"]
        col_titles = ["Forward", "Backward"]

        row_configs = [
            {"label": "Humans", "sweep": human_sweep,
             "ds_ylim": (0, 1), "ds_yticks": [0, 0.5, 1]},
            {"label": "Network", "sweep": nn_sweep,
             "ds_ylim": (0, 0.2), "ds_yticks": [0, 0.1, 0.2]},
        ]

        for r, cfg in enumerate(row_configs):
            sweep = cfg["sweep"]
            row_axes = []
            for c, (tname, ctitle) in enumerate(zip(col_templates, col_titles)):
                gs_cell = gs_outer[r, c].subgridspec(
                    1, 2, wspace=1.2, width_ratios=[1, 0.4],
                )
                ax_template = fig.add_subplot(gs_cell[0, 0])
                ax_delta = fig.add_subplot(gs_cell[0, 1])

                divider_tmpl = make_axes_locatable(ax_template)
                ax_cbar_tmpl = divider_tmpl.append_axes("right", size="5%", pad=0.08)
                _plot_single_heatmap(
                    ax_template, templates[tname],
                    cmap="viridis",
                    title=ctitle if r == 0 else "",
                    vmin=0.0, vmax=0.5,
                    mask_diagonal=True,
                    cbar=True,
                    cbar_ax=ax_cbar_tmpl,
                )
                ax_cbar_tmpl.set_yticks([0.0, 0.25, 0.5])
                ax_cbar_tmpl.set_yticklabels(["0.00", "0.25", "0.50"])
                ax_cbar_tmpl.tick_params(labelsize=14)

                delta = sweep["delta_similarity"][PANEL_ALL][tname]
                _plot_single_delta_similarity(
                    ax_delta, delta,
                    ylabel="Sim. (Obs. \u2212 Chance)",
                    ylim=cfg["ds_ylim"], yticks=cfg["ds_yticks"],
                )
                row_axes.extend([ax_template, ax_delta])
            cfg["_axes"] = row_axes

        # Row titles
        fig.canvas.draw()
        for cfg in row_configs:
            all_ax = cfg["_axes"]
            positions = [a.get_position() for a in all_ax]
            x_center = (min(p.x0 for p in positions) + max(p.x1 for p in positions)) / 2
            y_top = max(p.y1 for p in positions) + 0.04
            fig.text(x_center, y_top, cfg["label"],
                     fontsize=26, fontweight="normal",
                     ha="center", va="bottom", transform=fig.transFigure)

        suffix = f"_{tag}" if tag else ""
        out_path = out_dir / f"FigureTransitionSupplement{suffix}.pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Supplement 2: Advantage Supplement (Relevant / Irrelevant)
# ---------------------------------------------------------------------------

def _create_advantage_supplement(
    *,
    human_adv: pd.DataFrame,
    nn_adv: pd.DataFrame,
    out_dir: Path,
    tag: str = "",
    show: bool = True,
) -> Path:
    """Side-by-side Relevant/Irrelevant fixation advantage (Humans | Network)."""

    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    categories = ["Relevant", "Irrelevant"]
    cat_colors = [FA_COLORS["Relevant"], FA_COLORS["Irrelevant"]]

    group_configs = [
        {"label": "Humans", "adv": human_adv,
         "time_unit": "s", "time_scale": 1.0 / 1000.0},
        {"label": "Network", "adv": nn_adv,
         "time_unit": "steps", "time_scale": 1.0},
    ]

    # Compute shared y-limits from human data
    h_all_medians = []
    for cat in categories:
        subset = _subset_advantage(human_adv, cat.lower())
        meds = _fa_subject_medians(subset, 1.0 / 1000.0)
        h_all_medians.append(meds)
    h_combined = np.concatenate(h_all_medians)
    max_abs = float(np.nanmax(np.abs(h_combined))) if len(h_combined) > 0 else 1.0
    margin = max_abs * 0.1
    shared_ylim = (-(max_abs + margin), max_abs + margin)

    with plt.rc_context({
        "font.family": "Arial",
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(9, 5))
        fig.subplots_adjust(wspace=0.4)

        rng = np.random.default_rng(42)
        _bar_w = 0.5
        _edge_gray = ".7"
        xs = np.arange(len(categories), dtype=float)

        for ax, gcfg in zip(axes, group_configs):
            for i, (cat, color) in enumerate(zip(categories, cat_colors)):
                subset = _subset_advantage(gcfg["adv"], cat.lower())
                sub_medians = _fa_subject_medians(subset, gcfg["time_scale"])
                n = len(sub_medians)
                group_mean = float(np.mean(sub_medians))
                sem_val = _sem(sub_medians)

                jitter = rng.uniform(-0.08, 0.08, size=n)

                # Fill bar
                ax.bar(xs[i], group_mean, width=_bar_w,
                       color=color, edgecolor="none", linewidth=0, zorder=2)
                # Black outline bar
                ax.bar(xs[i], group_mean, width=_bar_w,
                       color="none", edgecolor="black", linewidth=1.5, zorder=4)
                # Subject dots
                for j in range(n):
                    ax.scatter(
                        xs[i] + jitter[j], sub_medians[j],
                        s=6 ** 2,
                        facecolor=(1, 1, 1, 0.5),
                        edgecolor=_edge_gray,
                        linewidth=1,
                        zorder=3,
                    )
                # SEM error bar
                ax.errorbar(
                    xs[i], group_mean, yerr=sem_val,
                    fmt="none", ecolor="black",
                    linewidth=1.5, capsize=0, zorder=5,
                )

            ax.axhline(0, color="black", linestyle="-", linewidth=1.5, zorder=1)
            ax.set_xticks(xs)
            ax.set_xticklabels(categories)
            ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
            ax.set_ylim(shared_ylim)
            ax.set_ylabel(f"Time advantage ({gcfg['time_unit']})")
            ax.set_title(gcfg["label"], fontsize=26, fontweight="normal")
            ax.spines[["top", "right"]].set_visible(False)

        suffix = f"_{tag}" if tag else ""
        out_path = out_dir / f"FigureAdvantageSupplement{suffix}.pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def create_figure(
    *,
    nn_root: Path,
    tag: str = "",
    n_sims: int = 1000,
    seed: int = 123,
    show: bool = True,
    out_dir: Optional[Path] = None,
    collapse_null_shuffles: bool = True,
) -> Path:
    if out_dir is None:
        out_dir = nn_root / "output" / "next_fixation_gen"
    _ensure_dir(out_dir)

    # When opting out of the (now default) collapsed null, append a suffix
    # to the tag so outputs don't overwrite the default-null run.
    if (not collapse_null_shuffles) and "uncollapsedNull" not in tag:
        tag = f"{tag}_uncollapsedNull" if tag else "uncollapsedNull"

    # ------------------------------------------------------------------
    # Step 1: Compute (or load cached) sweep transition data
    # ------------------------------------------------------------------
    human_sweep = _load_or_compute_sweep(
        _REPO_ROOT,
        buffer=50, n_sims=n_sims, seed=seed,
        exclude_subjects=("107", "131"),
        label="human",
        collapse_null_shuffles=collapse_null_shuffles,
    )

    nn_sweep = _load_or_compute_sweep(
        nn_root,
        buffer=50, n_sims=n_sims, seed=seed,
        exclude_subjects=(),
        label="NN",
        collapse_null_shuffles=collapse_null_shuffles,
    )

    # ------------------------------------------------------------------
    # Step 2: Compute (or load cached) fixation advantage
    # ------------------------------------------------------------------
    human_adv = _load_or_compute_fa(
        cache_root=_REPO_ROOT,
        data_dir=_REPO_ROOT / "data",
        excluded_subjects=("107", "131"),
        label="human",
    )
    h_all = _subset_advantage(human_adv, "all")

    nn_data_dir = nn_root / "data"
    if not nn_data_dir.exists():
        nn_data_dir = nn_root / "output"
    nn_adv = _load_or_compute_fa(
        cache_root=nn_root,
        data_dir=nn_data_dir,
        excluded_subjects=(),
        label="NN",
    )
    n_all = _subset_advantage(nn_adv, "all")

    # ------------------------------------------------------------------
    # Step 3: Get bidirectional template
    # ------------------------------------------------------------------
    templates = sweep_template_matrices(n_items=6)
    bi_template = templates["bidirectional"]

    # ------------------------------------------------------------------
    # Step 4: Save per-subject CSVs for statistical testing
    # ------------------------------------------------------------------
    _save_stats_csvs(
        human_adv=human_adv,
        nn_adv=nn_adv,
        human_sweep=human_sweep,
        nn_sweep=nn_sweep,
        out_dir=out_dir,
        tag=tag,
    )

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    print("Building figure ...", flush=True)

    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    with plt.rc_context({
        "font.family": "Arial",
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "hatch.linewidth": 1.5,
    }):
        fig = plt.figure(figsize=(17.5, 12))
        # Outer grid: [heatmaps+run-length | fixation_advantage]. Figure is
        # widened to give room for (a) the added colorbar next to each
        # observed heatmap, (b) a wider run-length column, and (c) a bit
        # more breathing room between the middle and final columns.
        gs_outer = fig.add_gridspec(
            2, 2,
            wspace=0.0,
            hspace=0.45,
            width_ratios=[0.31, 3.0],
        )

        # Heatmap color scale: per-row sequential range [vmin, vmax] fitted
        # to the *off-diagonal* observed cells. The diagonal is always 0 by
        # construction (sequences are pre-collapsed), so including it in the
        # scale compresses the color gradient onto the top of the colormap
        # and washes out the off-diagonal pattern. By setting vmin to the
        # off-diagonal minimum instead of 0, the full colormap range is
        # used for the actual data variation. The diagonal cells clip to
        # the darkest color, and the template's 0.5 neighbor cells clip to
        # the brightest — both clear visually.
        def _obs_offdiag_range(sweep_data: Dict[str, Any]) -> Tuple[float, float]:
            m = sweep_data["obs_trans_by_panel"][PANEL_ALL]
            if m.size == 0:
                return 0.0, 1.0
            off_diag_mask = ~np.eye(m.shape[0], dtype=bool) & np.isfinite(m)
            vals = m[off_diag_mask]
            if vals.size == 0:
                return 0.0, 1.0
            lo = max(0.0, float(vals.min()))
            hi = max(lo + 1e-6, float(vals.max()))
            return lo, hi

        hm_vmin_human, hm_vmax_human = _obs_offdiag_range(human_sweep)
        hm_vmin_nn, hm_vmax_nn = _obs_offdiag_range(nn_sweep)
        hm_cmap = "viridis"

        # Pre-compute shared y-limits from human data
        human_sub_medians = _fa_subject_medians(h_all, 1.0 / 1000.0)
        max_abs_human = float(np.nanmax(np.abs(human_sub_medians))) if len(human_sub_medians) > 0 else 1.0
        margin_human = max_abs_human * 0.1
        shared_fa_ylim = (-(max_abs_human + margin_human), max_abs_human + margin_human)

        row_configs = [
            {
                "label": "Humans",
                "panel_prefix": ["A", "B", "C"],
                "time_unit": "s",
                "time_scale": 1.0 / 1000.0,  # ms -> s
                "fa_data": h_all,
                "sweep": human_sweep,
                "hm_vmin": hm_vmin_human,
                "hm_vmax": hm_vmax_human,
                "hm_cmap": hm_cmap,
                "ds_ylim": (0, 0.6),
                "ds_yticks": [0, 0.2, 0.4, 0.6],
                "rl_ylim": (0, 0.06),
                "rl_yticks": [0, 0.03, 0.06],
            },
            {
                "label": "Network",
                "panel_prefix": ["D", "E", "F"],
                "time_unit": "steps",
                "time_scale": 1.0,
                "fa_data": n_all,
                "sweep": nn_sweep,
                "hm_vmin": hm_vmin_nn,
                "hm_vmax": hm_vmax_nn,
                "hm_cmap": hm_cmap,
                "ds_ylim": (0, 0.3),
                "ds_yticks": [0, 0.1, 0.2, 0.3],
                "rl_ylim": (0, 0.02),
                "rl_yticks": [0, 0.01, 0.02],
            },
        ]

        for r, cfg in enumerate(row_configs):
            sweep = cfg["sweep"]
            # --- Column 1: Fixation Advantage (vertical, All only) ---
            ax_col1 = fig.add_subplot(gs_outer[r, 0])
            _plot_fixation_advantage_vertical(
                ax_col1, cfg["fa_data"],
                time_unit_label=cfg["time_unit"],
                time_scale=cfg["time_scale"],
                ylim=shared_fa_ylim,
            )

            # --- Right side: [heatmaps+delta | run-length] with wider gap ---
            # Ratios give the heatmaps+delta-similarity sub-column room for
            # its new colorbar without overlapping the similarity y-label,
            # while also bumping the run-length column up slightly so it
            # doesn't visually shrink relative to the widened heatmap area.
            gs_right = gs_outer[r, 1].subgridspec(
                1, 2,
                wspace=0.45,
                width_ratios=[2.25, 1.2],
            )

            # --- Column 2: Compound panel (heatmaps + delta similarity) ---
            gs_col2 = gs_right[0, 0].subgridspec(
                2, 2,
                wspace=0.05,
                hspace=0.55,
                width_ratios=[1, 0.3],
            )
            ax_template = fig.add_subplot(gs_col2[0, 0])
            ax_obs_trans = fig.add_subplot(gs_col2[1, 0])
            ax_delta_sim = fig.add_subplot(gs_col2[:, 1])

            # Template heatmap (top-left) — viridis binary reference.
            # Diagonal cells are rendered solid black to mark that
            # self-transitions are excluded by construction.
            divider_tmpl = make_axes_locatable(ax_template)
            ax_cbar_tmpl = divider_tmpl.append_axes("right", size="5%", pad=0.08)
            _plot_single_heatmap(
                ax_template, bi_template,
                cmap="viridis",
                title="Template",
                vmin=0.0, vmax=0.5,
                show_xlabel=False, show_ylabel=False,
                mask_diagonal=True,
                cbar=True,
                cbar_ax=ax_cbar_tmpl,
            )
            ax_cbar_tmpl.set_yticks([0.0, 0.25, 0.5])
            ax_cbar_tmpl.set_yticklabels(["0.00", "0.25", "0.50"])
            ax_cbar_tmpl.tick_params(labelsize=14)

            # Observed transition heatmap (bottom-left): row-normalized
            # conditional transition probabilities. The diagonal is always
            # zero by construction (sequences are pre-collapsed) and is
            # masked out (rendered white) so the color scale can fit the
            # off-diagonal variation without being pulled toward zero.
            # A colorbar is attached to the right of the observed axes.
            obs_mat = sweep["obs_trans_by_panel"][PANEL_ALL]
            divider = make_axes_locatable(ax_obs_trans)
            ax_cbar = divider.append_axes("right", size="5%", pad=0.08)
            _plot_single_heatmap(
                ax_obs_trans, obs_mat,
                cmap=cfg["hm_cmap"],
                title="Observed",
                vmin=cfg["hm_vmin"], vmax=cfg["hm_vmax"],
                mask_diagonal=True,
                cbar=True,
                cbar_ax=ax_cbar,
            )
            # Colorbar ticks at min / middle / max, two decimal places.
            cb_vmin = cfg["hm_vmin"]
            cb_vmax = cfg["hm_vmax"]
            cb_mid = (cb_vmin + cb_vmax) / 2
            ax_cbar.set_yticks([cb_vmin, cb_mid, cb_vmax])
            ax_cbar.set_yticklabels([
                f"{cb_vmin:.2f}", f"{cb_mid:.2f}", f"{cb_vmax:.2f}",
            ])
            ax_cbar.tick_params(labelsize=14)

            # Bidirectional delta-similarity (right, full height)
            delta_bi = sweep["delta_similarity"][PANEL_ALL]["bidirectional"]
            _plot_single_delta_similarity(
                ax_delta_sim, delta_bi,
                ylim=cfg["ds_ylim"], yticks=cfg["ds_yticks"],
            )

            # --- Column 3: Run-Length Delta Bars ---
            ax_col3 = fig.add_subplot(gs_right[0, 1])
            _plot_single_runlength_delta(
                ax_col3,
                sweep["transprop_obs"][PANEL_ALL],
                sweep["transprop_null"][PANEL_ALL],
                sweep["run_bin_labels"],
                ylim=cfg["rl_ylim"], yticks=cfg["rl_yticks"],
            )

            # Store axes for row title placement after layout
            cfg["_ax_left"] = ax_col1
            cfg["_ax_right"] = ax_col3

        # Force layout so get_position() returns final coords
        fig.canvas.draw()

        for cfg in row_configs:
            pos_left = cfg["_ax_left"].get_position()
            pos_right = cfg["_ax_right"].get_position()
            # Center over the full row extent including labels
            x_center = (pos_left.x0 + pos_right.x1) / 2
            y_top = max(pos_left.y1, pos_right.y1) + 0.04
            fig.text(
                x_center, y_top, cfg["label"],
                fontsize=26, fontweight="normal",
                ha="center", va="bottom",
                transform=fig.transFigure,
            )

        suffix = f"_{tag}" if tag else ""
        out_path = out_dir / f"FigureNN_H_next_fixation_gen{suffix}.pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Supplement figures
    # ------------------------------------------------------------------
    print("Building TransitionSupplement ...", flush=True)
    trans_supp_path = _create_transition_supplement(
        human_sweep=human_sweep,
        nn_sweep=nn_sweep,
        templates=templates,
        out_dir=out_dir,
        tag=tag,
        show=show,
    )

    print("Building AdvantageSupplement ...", flush=True)
    adv_supp_path = _create_advantage_supplement(
        human_adv=human_adv,
        nn_adv=nn_adv,
        out_dir=out_dir,
        tag=tag,
        show=show,
    )

    # Sidecar metadata
    meta_path = out_dir / f"FigureNN_H_next_fixation_gen{suffix}.meta.txt"
    meta_path.write_text(
        "\n".join([
            f"nn_root={nn_root}",
            f"tag={tag}",
            f"n_sims={n_sims}",
            f"seed={seed}",
            f"human_exclude_subjects=['107','131']",
            f"trans_supplement={trans_supp_path}",
            f"advantage_supplement={adv_supp_path}",
        ])
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human vs NN next-fixation-generation comparison (2x3 figure)."
    )
    parser.add_argument(
        "--nn-root", type=str, required=True,
        help="Path to NN simulation directory.",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Output filename suffix.",
    )
    parser.add_argument(
        "--n-sims", type=int, default=1000,
        help="Number of shuffle simulations per subject.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display interactive plots.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory override.",
    )
    parser.add_argument(
        "--no-collapse-null-shuffles", action="store_true",
        help=(
            "Opt out of collapsing consecutive repeats in shuffled null "
            "sequences. By default, the null collapses consecutive repeats "
            "so it mirrors the observed pipeline (no self-transitions). "
            "When this flag is set, outputs get an '_uncollapsedNull' "
            "suffix so they don't overwrite the default run."
        ),
    )

    args = parser.parse_args()

    create_figure(
        nn_root=Path(args.nn_root).resolve(),
        tag=str(args.tag),
        n_sims=int(args.n_sims),
        show=(not bool(args.no_show)),
        out_dir=Path(args.out_dir).resolve() if args.out_dir else None,
        collapse_null_shuffles=(not bool(args.no_collapse_null_shuffles)),
    )


if __name__ == "__main__":
    main()

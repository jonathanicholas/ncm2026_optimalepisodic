#!/usr/bin/env python3
"""Fixation-transition structure: humans vs. prior-memory network (Figure S6).

A 1 row x 2 column figure showing chance-corrected proportion of transitions
by sequence (run) length:
  Column 1 (Humans):  Delta fraction of transitions by run length.
  Column 2 (Network): Delta fraction of transitions by run length.

(The transition-matrix heatmaps and delta-similarity bars previously shown
here have moved to the main Figure 5.)

Sweep transition data is cached to disk so that subsequent runs skip the
expensive shuffle computation.

Example
-------
conda run -n analysis python \
  supplemental_analysis/fixation_transitions/plot_fixation_transitions.py \
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.lib.analyze_choice_fixation_sweeps import (  # noqa: E402
    PANEL_ALL,
    list_subjects,
    mean_ci95,
    nanmean_safe,
    sweep_template_matrices,
)
from metarnn.lib.plot_NN_sweep_transitions import (  # noqa: E402
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

def _sem(x: np.ndarray) -> float:
    """Standard error of the mean."""
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _save_stats_csvs(
    *,
    human_sweep: Dict[str, Any],
    nn_sweep: Dict[str, Any],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Save per-subject CSVs for statistical testing in R."""
    stats_dir = out_dir / "stats"
    _ensure_dir(stats_dir)
    suffix = f"_{tag}" if tag else ""

    # --- CSV 1: Delta similarity by template ---
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

    # --- CSV 2: Sequence length delta ---
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
# Column 1: Heatmap + bidirectional delta-similarity
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

    # Position labels: only the two endpoints (0° at position 1, 300° at
    # position 6); intermediate positions (2-5) are left blank. The angular
    # labels emphasise that the six positions sit on a 60°-spaced ring.
    pos_labels = ["0°", "", "", "", "", "300°"]
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
        xticklabels=pos_labels,
        yticklabels=pos_labels,
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
    ax.set_xlabel("Next pos." if show_xlabel else "", fontsize=18)
    ax.set_ylabel("Current pos." if show_ylabel else "", fontsize=18)
    ax.tick_params(length=0)
    # Explicit per-label styling — sns.heatmap can apply its own tick-label
    # sizes/rotations, so we override after the heatmap has been drawn:
    # match the axis-label fontsize, and keep the y-axis degree labels
    # upright rather than rotated 90 degrees.
    for label in ax.get_xticklabels():
        label.set_fontsize(18)
    for label in ax.get_yticklabels():
        label.set_fontsize(18)
        label.set_rotation(0)


def _plot_single_delta_similarity(
    ax: plt.Axes,
    delta_vals: np.ndarray,
    *,
    ylabel: str = "Similarity above chance",
    ylim: Optional[Tuple[float, float]] = None,
    yticks: Optional[Sequence[float]] = None,
) -> None:
    """Bar + strip of per-subject bidirectional delta-similarity."""
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
           color="none", edgecolor="black", linewidth=3.0, zorder=4)

    # Individual subject dots — sized to match the per-subject dots in the
    # forest plot (s=90) so the two panels feel visually consistent.
    if len(y) > 0:
        jitter = rng.uniform(-0.08, 0.08, size=len(y))
        for i in range(len(y)):
            ax.scatter(
                jitter[i], y[i],
                s=90,
                facecolor=(1, 1, 1, 0.5),
                edgecolor=_gray,
                linewidth=1,
                zorder=3,
            )

    # SEM error bar
    ax.errorbar(
        0, m, yerr=sem_val,
        fmt="none", ecolor="black",
        linewidth=3.0, capsize=0, zorder=5,
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
# Column 2: Run-length delta bars
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
    # Step 2: Save per-subject CSVs for statistical testing
    # ------------------------------------------------------------------
    _save_stats_csvs(
        human_sweep=human_sweep,
        nn_sweep=nn_sweep,
        out_dir=out_dir,
        tag=tag,
    )

    # ------------------------------------------------------------------
    # Build figure: 1 row x 2 columns of run-length delta bars only.
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
        fig = plt.figure(figsize=(11, 5))
        gs = fig.add_gridspec(1, 2, wspace=0.4)

        col_configs = [
            {
                "label": "Humans",
                "sweep": human_sweep,
                "rl_ylim": (0, 0.06),
                "rl_yticks": [0, 0.03, 0.06],
            },
            {
                "label": "RNN",
                "sweep": nn_sweep,
                "rl_ylim": (0, 0.02),
                "rl_yticks": [0, 0.01, 0.02],
            },
        ]

        for c, cfg in enumerate(col_configs):
            sweep = cfg["sweep"]
            ax = fig.add_subplot(gs[0, c])
            _plot_single_runlength_delta(
                ax,
                sweep["transprop_obs"][PANEL_ALL],
                sweep["transprop_null"][PANEL_ALL],
                sweep["run_bin_labels"],
                ylim=cfg["rl_ylim"], yticks=cfg["rl_yticks"],
            )
            cfg["_ax"] = ax

        # Force layout so get_position() returns final coords for title placement.
        fig.canvas.draw()

        for cfg in col_configs:
            pos = cfg["_ax"].get_position()
            x_center = (pos.x0 + pos.x1) / 2
            y_top = pos.y1 + 0.04
            fig.text(
                x_center, y_top, cfg["label"],
                fontsize=26, fontweight="normal",
                ha="center", va="bottom",
                transform=fig.transFigure,
            )

        suffix = f"_{tag}" if tag else ""
        out_path = out_dir / f"FigureFixationTransitions{suffix}.pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)

        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixation-transition structure: humans vs. network (Figure S6)."
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

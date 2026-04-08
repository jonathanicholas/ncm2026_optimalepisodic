#!/usr/bin/env python3
"""Create an overview figure summarizing NN behavior + eyegaze.

Composes an 8-panel figure (Panels A-D: behavior, E-H: eyegaze) from
cached or freshly computed NN analysis outputs.

Example:
  conda run -n analysis python metarnn/lib/plot_NN_overview.py \\
    --root metarnn/simulations/human_like_04_04_input5 \\
    --out-dir metarnn/simulations/human_like_04_04_input5/output/overview \\
    --tag 04_04_input5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.tools import add_constant
from statsmodels.api import OLS

# Allow running as a script (e.g., `python metarnn/lib/plot_NN_overview.py`) by
# ensuring the repository root is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import metarnn.lib.analyze_NN_behavior as nn_beh
import metarnn.lib.create_eyeplot_NN as nn_eye


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ci95_mean(values: np.ndarray, bootstrap: bool = False, n_boot: int = 10000,
              rng_seed: int = 42) -> Tuple[float, float]:
    """Return (mean, half_ci) for *values*, ignoring NaNs.

    If *bootstrap* is False (default), half_ci = 1.96 * SEM.
    If *bootstrap* is True, half_ci is half the width of the BCa 95% bootstrap CI.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    m = float(np.mean(values))
    if not bootstrap:
        se = float(np.std(values, ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return (m, 1.96 * se)
    ci_lo, ci_hi = _bootstrap_ci(values, n_boot=n_boot, rng_seed=rng_seed)
    half_ci = (ci_hi - ci_lo) / 2.0
    return (m, half_ci)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                  rng_seed: int = 42) -> Tuple[float, float]:
    """Return (lo, hi) of a percentile bootstrap 95% CI for the mean."""
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = values[rng.integers(0, n, size=n)].mean()
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def fit_logistic(x: np.ndarray, y: np.ndarray):
    """Fit a GLM with Binomial family (logistic regression).

    Returns the fitted model, or None on failure.
    """
    if len(np.unique(y)) < 2:
        return None
    X = add_constant(pd.Series(x))
    try:
        return GLM(y, X, family=Binomial()).fit()
    except Exception:
        return None


def fit_ols(x: np.ndarray, y: np.ndarray):
    """Fit an OLS regression. Returns the fitted model, or None on failure."""
    if len(x) < 2:
        return None
    X = add_constant(pd.Series(x))
    try:
        return OLS(y, X).fit()
    except Exception:
        return None


def _panel_strip_mean(
    ax: plt.Axes,
    values: np.ndarray,
    ylabel: str,
    panel_label: str,
    chance: Optional[float] = None,
):
    """Strip plot with mean +/- 95% CI overlay."""
    m, err = ci95_mean(values)
    # Place seed points at jittered offsets away from x=0 to avoid overlap with group mean.
    # Alternate signs deterministically so points split across both sides of the group mean
    # (e.g., for n=5 this gives 3 on the left and 2 on the right).
    n = len(values)
    rng = np.random.default_rng(42)
    magnitudes = rng.uniform(0.015, 0.045, size=n)
    signs = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    offsets = magnitudes * signs
    ax.scatter(
        offsets, values,
        s=12**2, facecolor="gray", edgecolor="gray", alpha=0.5, linewidth=0, zorder=2,
    )
    ax.errorbar([0], [m], yerr=[err], fmt="none", ecolor="black", capsize=0, zorder=3)
    ax.scatter([0], [m], s=14**2, facecolor=".5", edgecolor="black", linewidth=2.5, zorder=3)
    if chance is not None:
        ax.plot((-0.075, 0.075), (chance, chance), "k:")
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_xlim(-0.075, 0.075)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if panel_label:
        ax.text(-0.5, 1.1, panel_label, transform=ax.transAxes, fontsize=26, fontweight="bold", ha="left", va="top")


def plot_fix_time_vs_count(ax: plt.Axes, df: pd.DataFrame, count_col: str, xlabel: str, panel_label: str):
    """Scatter of mean total fixation time (+/- 95% CI) by item count, with OLS fits."""
    d = df.dropna(subset=[count_col, "total_fix_time"]).copy()
    if len(d) == 0:
        ax.axis("off")
        return

    # Per-subject regression lines (no intercept — forced through origin)
    # for _sid, sdf in d.groupby("subject"):
    #     x_s = sdf[count_col].values.astype(float)
    #     y_s = sdf["total_fix_time"].values.astype(float)
    #     if len(x_s) < 2 or np.allclose(x_s, x_s[0]):
    #         continue
    #     try:
    #         m_subj = OLS(y_s, x_s).fit()
    #     except Exception:
    #         continue
    #     x0, x1 = float(np.min(x_s)), float(np.max(x_s))
    #     x_line = np.array([x0, x1])
    #     y_line = m_subj.predict(x_line)
    #     ax.plot(x_line, y_line, color="0.7", alpha=0.6, linewidth=1)

    # Group-level regression line (no intercept — forced through origin)
    x_all = d[count_col].values.astype(float)
    y_all = d["total_fix_time"].values.astype(float)
    mask = np.isfinite(x_all) & np.isfinite(y_all)
    xs_unique = np.array(sorted(d[count_col].dropna().unique()))
    # try:
    #     gmodel = OLS(y_all[mask], x_all[mask]).fit()
    # except Exception:
    #     gmodel = None
    # if gmodel is not None and len(xs_unique) > 0:
    #     yhat = gmodel.predict(xs_unique)
    #     ax.plot(xs_unique, yhat, color="black", linewidth=4)

    # Binned means with error bars
    means: list = []
    errs: list = []
    for v in xs_unique:
        vals = d.loc[d[count_col] == v, "total_fix_time"].values
        m_v, err_v = ci95_mean(vals)
        means.append(m_v)
        errs.append(err_v)

    if len(xs_unique) > 0:
        ax.errorbar(xs_unique, means, yerr=errs, fmt="none", ecolor="black", capsize=0)
        ax.scatter(xs_unique, means, s=14**2, facecolor=".5", edgecolor="black", linewidth=2.5, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fixation Time (steps)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if panel_label:
        ax.text(-0.25, 1.1, panel_label, transform=ax.transAxes, fontsize=26, fontweight="bold", ha="left", va="top")


def plot_fix_time_vs_offer_value(ax: plt.Axes, df: pd.DataFrame, panel_label: str):
    """Scatter of mean total fixation time by true offer value, with quadratic fit."""
    d = df.dropna(subset=["true_offer_value", "total_fix_time"]).copy()
    if len(d) == 0:
        ax.axis("off")
        return

    d["true_offer_value"] = pd.to_numeric(d["true_offer_value"], errors="coerce")
    d = d.dropna(subset=["true_offer_value"]).copy()
    if len(d) == 0:
        ax.axis("off")
        return

    rows = []
    for ov, g in d.groupby("true_offer_value"):
        vals = g["total_fix_time"].astype(float).values
        m, err = ci95_mean(vals)
        rows.append({"true_offer_value": float(ov), "mean": float(m), "err": float(err), "n": int(len(vals))})
    summ = pd.DataFrame(rows).sort_values("true_offer_value")

    ax.errorbar(
        summ["true_offer_value"].values,
        summ["mean"].values,
        yerr=summ["err"].values,
        fmt="none",
        ecolor="black",
        capsize=0,
        zorder=2,
    )
    ax.scatter(
        summ["true_offer_value"].values,
        summ["mean"].values,
        s=14**2,
        facecolor=".5",
        edgecolor="black",
        linewidth=2.5,
        zorder=3,
    )

    # Quadratic fit to binned means
    x = summ["true_offer_value"].values.astype(float)
    y = summ["mean"].values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(np.unique(x)) >= 3:
        coefs = np.polyfit(x, y, deg=2)
        xg = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        yg = np.polyval(coefs, xg)
        ax.plot(xg, yg, color="black", linewidth=4, zorder=1)

    ax.set_xlabel("True Offer Value")
    ax.set_ylabel("Fixation Time (steps)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if panel_label:
        ax.text(-0.25, 1.1, panel_label, transform=ax.transAxes, fontsize=26, fontweight="bold", ha="left", va="top")


# ---------------------------------------------------------------------------


def _pick_latest(paths):
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))


def _auto_find_behavior_csv(root: str) -> Optional[str]:
    """Try to find an existing NN trial-level behavior CSV under <root>/output or <root>/figures."""
    candidates = glob.glob(os.path.join(root, "output", "**", "nn_trial_level_behavior*.csv"), recursive=True)
    candidates += glob.glob(os.path.join(root, "figures", "**", "nn_trial_level_behavior*.csv"), recursive=True)
    return _pick_latest(candidates)


def _auto_find_prop_time_pred_summary(root: str) -> Optional[str]:
    """Find the cached summary CSV produced by predict_choice_from_item_prop_time_interactions.py."""
    candidates = glob.glob(os.path.join(root, "output", "eyegaze", "stats", "summary_prop_time_*.csv"), recursive=True)
    candidates += glob.glob(os.path.join(root, "figures", "**", "summary_prop_time_*.csv"), recursive=True)
    candidates = [c for c in candidates if c and os.path.exists(c) and "_drop" not in os.path.basename(os.path.dirname(c))]
    if not candidates:
        return None

    # Prefer the file matching the settings used for the overview panels.
    # If multiple summaries exist (e.g., true vs recalled values), prefer recalled.
    scored = []
    for path in candidates:
        score = 0.0
        try:
            df = pd.read_csv(path)
            if "value_source" in df.columns:
                vs = df["value_source"].astype(str)
                if (vs == "recalled").any():
                    score += 100.0
                elif (vs == "true").any():
                    score += 50.0
            if "feature_set" in df.columns and (df["feature_set"].astype(str) == "location_interactions").any():
                score += 10.0
            if "visit_type" in df.columns and (df["visit_type"].astype(str) == "all").any():
                score += 5.0
            if "visit_normalization" in df.columns and (df["visit_normalization"].astype(str) == "within").any():
                score += 1.0
        except Exception:
            # If the CSV can't be parsed, just rely on recency.
            pass

        # Tie-breaker: newest file wins.
        try:
            score += float(os.path.getmtime(path)) * 1e-9
        except Exception:
            pass
        scored.append((score, path))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1] if scored else None


def _auto_find_prop_time_coef_table(root: str) -> Optional[str]:
    """Find the cached coefficient table CSV produced by predict_choice_from_item_prop_time_interactions.py."""
    candidates = glob.glob(os.path.join(root, "output", "eyegaze", "stats", "coef_table_prop_time_*.csv"), recursive=True)
    candidates += glob.glob(os.path.join(root, "figures", "**", "coef_table_prop_time_*.csv"), recursive=True)
    candidates = [c for c in candidates if c and os.path.exists(c) and "_drop" not in os.path.basename(os.path.dirname(c))]
    if not candidates:
        return None

    scored = []
    for path in candidates:
        score = 0.0
        try:
            df = pd.read_csv(path)
            if "value_source" in df.columns:
                vs = df["value_source"].astype(str)
                if (vs == "recalled").any():
                    score += 100.0
                elif (vs == "true").any():
                    score += 50.0
            if "feature_set" in df.columns and (df["feature_set"].astype(str) == "location_interactions").any():
                score += 10.0
            if "visit_type" in df.columns and (df["visit_type"].astype(str) == "all").any():
                score += 5.0
            if "visit_normalization" in df.columns and (df["visit_normalization"].astype(str) == "within").any():
                score += 1.0
        except Exception:
            pass

        try:
            score += float(os.path.getmtime(path)) * 1e-9
        except Exception:
            pass
        scored.append((score, path))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1] if scored else None


def _load_or_build_behavior_df(
    *,
    root: str,
    behavior_csv: Optional[str],
    cache_csv: str,
    use_cache: bool,
) -> Tuple[pd.DataFrame, str]:
    """Return (df_all, source_label)."""

    if behavior_csv is not None:
        behavior_csv = os.path.abspath(behavior_csv)
        if not os.path.exists(behavior_csv):
            raise FileNotFoundError(f"Behavior CSV not found: {behavior_csv}")
        return pd.read_csv(behavior_csv), "behavior_csv_arg"

    if use_cache and os.path.exists(cache_csv):
        return pd.read_csv(cache_csv), "behavior_cache"

    auto = _auto_find_behavior_csv(root)
    if auto is not None and os.path.exists(auto):
        return pd.read_csv(auto), "behavior_autofind"

    # Fallback: rebuild trial-level dataset from NN human_like export.
    data_root = os.path.join(root, "data")
    subjects = nn_beh.list_subjects(data_root)
    if len(subjects) == 0:
        raise RuntimeError(f"No subjects found under {data_root}")

    all_rows = []
    for sid in subjects:
        df = nn_beh.build_subject_trial_dataset(sid, root)
        if df is None or len(df) == 0:
            continue
        all_rows.append(df)

    if len(all_rows) == 0:
        raise RuntimeError("No NN trial-level datasets could be built.")

    df_all = pd.concat(all_rows, ignore_index=True)
    _ensure_dir(os.path.dirname(cache_csv) or ".")
    df_all.to_csv(cache_csv, index=False)
    return df_all, "behavior_rebuilt"


def _add_panel_label(ax: plt.Axes, label: str, *, dx: float = -55, dy: float = 12) -> None:
    """Add a panel label with a fixed offset from the top-left of the axes."""

    ax.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=26,
        fontweight="bold",
        ha="left",
        va="top",
        annotation_clip=False,
    )


def _panel_single_point_sem(
    ax: plt.Axes,
    *,
    mean: float,
    sem: float,
    ylabel: str,
    chance: Optional[float] = None,
) -> None:
    """Single-point panel styled like analyze_NN_behavior.py strip+mean panels."""

    ax.errorbar([0], [mean], yerr=[sem], fmt="none", ecolor="black", capsize=0)
    ax.scatter(
        [0],
        [mean],
        s=14**2,
        facecolor=".5",
        edgecolor="black",
        linewidth=2.5,
        zorder=3,
    )
    if chance is not None:
        ax.plot((-0.075, 0.075), (chance, chance), "k--")
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_xlim(-0.075, 0.075)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


_HUMAN_COLOR = ".5"

# Very light gray for NN data (almost white, keeps black outlines visible)
_NN_FILL = "white"
_NN_SUBJECT = "white"


def _lighten_color(color, amount: float = 0.55):
    """Blend *color* toward white.  amount=0 returns the original, 1 returns white."""
    import matplotlib.colors as mcolors
    c = np.array(mcolors.to_rgb(color))
    return tuple(1 - amount * (1 - c))


def _add_human_hline(
    ax: plt.Axes,
    mean: float,
    sem: float,
    *,
    color: str = _HUMAN_COLOR,
    alpha: float = 0.4,
) -> None:
    """Add a horizontal line with SEM error band for human reference."""
    ax.axhline(mean, color=color, linewidth=2, linestyle="-", zorder=4)
    ax.axhspan(mean - sem, mean + sem, color=color, alpha=alpha, zorder=1)


def _add_human_hline_ci(
    ax: plt.Axes,
    mean: float,
    ci_lo: float,
    ci_hi: float,
    *,
    color: str = _HUMAN_COLOR,
    alpha: float = 0.25,
    zorder: int = 0,
) -> None:
    """Add a horizontal line with 95% CI band for human reference."""
    ax.axhspan(ci_lo, ci_hi, color=color, alpha=alpha, linewidth=0, zorder=zorder)


def _load_human_benchmarks(human_data_dir: str, *, recalled_valence: bool = False,
                           bootstrap_ci: bool = False) -> dict:
    """Load human group-level stats for overlay on NN panels.

    Parameters
    ----------
    human_data_dir : str
        Path to the human ``output/`` directory (repo-level).
    recalled_valence : bool
        If True, load take/leave valence stats computed from recalled
        (rather than true) reward valence.

    Returns
    -------
    dict with available keys:
        accuracy_mean, accuracy_sem,
        rel_fix_mean, rel_fix_sem,
        cv_mean, cv_sem,
        take_leave_stats (DataFrame with decision_label, valence_label, mean, sem).
    """
    result: dict = {}
    human_data_dir = os.path.abspath(human_data_dir)

    # 1. Accuracy (all 43 subjects, matching analyze_behavior.py)
    acc_path = os.path.join(human_data_dir, "behavior", "stats", "subject_behavior_summary.csv")
    if os.path.exists(acc_path):
        df = pd.read_csv(acc_path)
        vals = df["choice_accuracy"].dropna().to_numpy(dtype=float)
        if len(vals) > 0:
            result["accuracy_mean"] = float(np.nanmean(vals))
            if bootstrap_ci:
                lo, hi = _bootstrap_ci(vals)
                result["accuracy_ci_lo"] = lo
                result["accuracy_ci_hi"] = hi
            else:
                result["accuracy_sem"] = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
    else:
        print(f"[human overlay] accuracy CSV not found: {acc_path}")

    # 2. Relevant fixation time
    rel_path = os.path.join(
        human_data_dir,
        "eyegaze", "stats",
        "choice_fixation_relevance_subject_means_relevant_only_duration.csv",
    )
    if os.path.exists(rel_path):
        df = pd.read_csv(rel_path)
        vals = df["mean_prop"].dropna().to_numpy(dtype=float)
        if len(vals) > 0:
            result["rel_fix_mean"] = float(np.nanmean(vals))
            if bootstrap_ci:
                lo, hi = _bootstrap_ci(vals)
                result["rel_fix_ci_lo"] = lo
                result["rel_fix_ci_hi"] = hi
            else:
                result["rel_fix_sem"] = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
    else:
        print(f"[human overlay] relevant-fix CSV not found: {rel_path}")

    # 3. CV accuracy (prop-time interaction model)
    cv_summary = os.path.join(
        human_data_dir,
        "eyegaze", "stats",
        "summary_prop_time_location_interactions_recalled_all_norm-within.csv",
    )
    if os.path.exists(cv_summary):
        df = pd.read_csv(cv_summary)
        if "feature_set" in df.columns:
            df = df[df["feature_set"].astype(str) == "location_interactions"]
        if "visit_type" in df.columns:
            df = df[df["visit_type"].astype(str) == "all"]
        if len(df) > 0 and "cv_mean" in df.columns and "cv_sem" in df.columns:
            result["cv_mean"] = float(df.iloc[0]["cv_mean"])
            result["cv_sem"] = float(df.iloc[0]["cv_sem"])
    else:
        print(f"[human overlay] CV summary CSV not found: {cv_summary}")

    # 4. Take/leave proportions
    # The human pipeline already uses recalled valence by default, so the
    # standard filename contains recalled-valence data regardless of the flag.
    tl_filename = "choice_fixation_relsign4_relevant_subject_means_duration.csv"
    tl_path = os.path.join(human_data_dir, "eyegaze", "stats", tl_filename)
    if os.path.exists(tl_path):
        df = pd.read_csv(tl_path)
        stats = (
            df.groupby(["decision_label", "valence_label"])["mean_prop"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        stats["sem"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
        if bootstrap_ci:
            ci_los, ci_his = [], []
            for _, row in stats.iterrows():
                vals = df.loc[
                    (df["decision_label"] == row["decision_label"])
                    & (df["valence_label"] == row["valence_label"]),
                    "mean_prop",
                ].dropna().to_numpy(dtype=float)
                lo, hi = _bootstrap_ci(vals) if len(vals) > 1 else (row["mean"], row["mean"])
                ci_los.append(lo)
                ci_his.append(hi)
            stats["ci_lo"] = ci_los
            stats["ci_hi"] = ci_his
        result["take_leave_stats"] = stats
    else:
        print(f"[human overlay] take/leave CSV not found: {tl_path}")

    # 5. Regression coefficients (prop-time interaction model)
    cv_coef = os.path.join(
        human_data_dir,
        "eyegaze", "stats",
        "coef_table_prop_time_location_interactions_recalled_all_norm-within.csv",
    )
    if os.path.exists(cv_coef):
        df = pd.read_csv(cv_coef)
        if "feature_set" in df.columns:
            df = df[df["feature_set"].astype(str) == "location_interactions"]
        if "visit_type" in df.columns:
            df = df[df["visit_type"].astype(str) == "all"]
        result["coef_table"] = df
    else:
        print(f"[human overlay] coef table CSV not found: {cv_coef}")

    return result


def create_nn_overview_figure(
    *,
    root: str,
    out_dir: str,
    tag: str = "",
    metric: str = "duration",
    behavior_csv: Optional[str] = None,
    no_cache: bool = False,
    human_data_dir: Optional[str] = None,
    drop_fix_pred_dir: Optional[str] = None,
    human_recalled_valence: bool = False,
    bootstrap_ci: bool = False,
) -> str:
    root = os.path.abspath(root)
    out_dir = os.path.abspath(out_dir)
    _ensure_dir(out_dir)

    cache_dir = os.path.join(out_dir, "cache")
    _ensure_dir(cache_dir)
    behavior_cache = os.path.join(cache_dir, "nn_trial_level_behavior_cached.csv")
    eye_trial_cache = os.path.join(cache_dir, f"Figure3_NN_trial_level_{metric}.csv")

    pred_summary_csv = _auto_find_prop_time_pred_summary(root)
    pred_coef_table_csv = _auto_find_prop_time_coef_table(root)

    def _pick_value_source_from_summary(df: pd.DataFrame) -> Optional[str]:
        if "value_source" not in df.columns:
            return None
        vs = df["value_source"].astype(str)
        if (vs == "recalled").any():
            return "recalled"
        if (vs == "true").any():
            return "true"
        return None

    def _pick_value_source_from_files(summary_path: Optional[str], coef_table_path: Optional[str]) -> Optional[str]:
        """Prefer recalled if either file contains it; else true if present."""
        for path in (summary_path, coef_table_path):
            if not path or not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            vs = _pick_value_source_from_summary(df)
            if vs == "recalled":
                return "recalled"
        for path in (summary_path, coef_table_path):
            if not path or not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            vs = _pick_value_source_from_summary(df)
            if vs == "true":
                return "true"
        return None

    df_all, behavior_source = _load_or_build_behavior_df(
        root=root,
        behavior_csv=behavior_csv,
        cache_csv=behavior_cache,
        use_cache=(not no_cache),
    )

    # Compute/load the eyegaze trial table and derive the two panel tables.
    _, subj_rel, subj_val_long = nn_eye.compute_eyeplot_nn_tables(
        root=root,
        metric=metric,
        cache_trial_csv=eye_trial_cache,
        use_cache=(not no_cache),
    )

    sns.set_context("poster")
    with plt.rc_context(
        {
            "font.family": "Arial",
            "axes.titlesize": 24,
            "axes.labelsize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
        }
    ):
        fig = plt.figure(figsize=(18, 12.5))
        gs = fig.add_gridspec(2, 1, hspace=0.3)

        # Row 1: flat gridspec with ratios preserving Figure2/Figure3 panel
        # proportions and uniform wspace so all inter-panel gaps are equal.
        gs_top = gs[0].subgridspec(1, 4, width_ratios=[0.265, 0.758, 0.25, 1.0], wspace=0.59)
        axA = fig.add_subplot(gs_top[0, 0])
        axB = fig.add_subplot(gs_top[0, 1])
        axE = fig.add_subplot(gs_top[0, 2])
        axF = fig.add_subplot(gs_top[0, 3])

        # Row 2: nested gridspec so panel C matches Figure2 row 2 panel 1 width
        gs_bot = gs[1].subgridspec(1, 2, width_ratios=[0.94, 1.0], wspace=0.37)
        gs_bot_left = gs_bot[0, 0].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.8)
        gs_bot_right = gs_bot[0, 1].subgridspec(1, 2, width_ratios=[0.35, 1.0], wspace=0.0)
        axC = fig.add_subplot(gs_bot_left[0, 0])
        axD = fig.add_subplot(gs_bot_left[0, 1])
        axG = fig.add_subplot(gs_bot_right[0, 0])
        axH = fig.add_subplot(gs_bot_right[0, 1])

        # Panel A: NN choice accuracy (Figure2_NN A)
        subj_acc = df_all.groupby("subject")["correct"].mean()
        _panel_strip_mean(
            axA,
            subj_acc.values.astype(float),
            ylabel="Choice Accuracy",
            panel_label="",
            chance=0.5,
        )
        axA.set_ylim(0, 1.05)
        axA.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axA.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        #_add_panel_label(axA, "A", dx=-90, dy=16)

        # Panel B: logistic choice ~ true offer value (styled to match human figure)
        _d_b = df_all.dropna(subset=["true_offer_value", "choice"]).copy()
        if len(_d_b) > 0:
            _mu_b = float(_d_b["true_offer_value"].mean())
            _sd_b = float(_d_b["true_offer_value"].std(ddof=0))
            _d_b["true_z"] = 0.0 if _sd_b <= 0 else (_d_b["true_offer_value"] - _mu_b) / _sd_b
            # Use human z-range for the grid so curves span the same extent
            # Human offer values range from -17 to +16; z-score with
            # human mu~0.06, sd~4.39 gives z in approx [-3.89, 3.63].
            # Compute it from the standard human range to stay exact.
            _human_offer_min, _human_offer_max = -17.0, 16.0
            _z_min_b = (_human_offer_min - _mu_b) / _sd_b if _sd_b > 0 else -4.0
            _z_max_b = (_human_offer_max - _mu_b) / _sd_b if _sd_b > 0 else 4.0
            _grid_b = np.linspace(_z_min_b, _z_max_b, 100)

            for _sid, _sdf in _d_b.groupby("subject"):
                _m = fit_logistic(_sdf["true_z"].values, _sdf["choice"].values)
                if _m is None:
                    continue
                _pred = _m.predict(add_constant(pd.Series(_grid_b)))
                axB.plot(_grid_b, _pred, color="gray", alpha=0.25, linewidth=1, zorder=0)

            _gm_b = fit_logistic(_d_b["true_z"].values, _d_b["choice"].values)
            if _gm_b is not None:
                _gpred_b = _gm_b.predict(add_constant(pd.Series(_grid_b)))
                _bg = _grid_b.copy()
                _bg[0] = _grid_b[0] + 0.025
                _bg[-1] = _grid_b[-1] - 0.025
                axB.plot(_bg, _gpred_b, color="k", linewidth=6)
                axB.plot(_grid_b, _gpred_b, color=".5", linewidth=4)

            # Add scatter plots for individual choices (subsample 10%)
            _choices_1 = _d_b[_d_b["choice"] == 1].sample(frac=0.1, random_state=42)
            _choices_0 = _d_b[_d_b["choice"] == 0].sample(frac=0.1, random_state=42)
            axB.scatter(_choices_1["true_z"],
                        np.random.uniform(1.02, 1.05, size=len(_choices_1)),
                        color="gray", alpha=0.03, s=25, linewidth=0)
            axB.scatter(_choices_0["true_z"],
                        np.random.uniform(-0.05, -0.02, size=len(_choices_0)),
                        color="gray", alpha=0.03, s=25, linewidth=0)

            axB.set_ylabel("Proportion Offers Taken")
            axB.set_xlabel("Standardized Offer Value (z)")
            axB.set_ylim(-0.07, 1.07)
            axB.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            axB.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
            axB.set_xlim(-5, 5)
            axB.set_xticks([-5, -2.5, 0, 2.5, 5])
            axB.set_xticklabels([-5, -2.5, 0, 2.5, 5])
            axB.spines["right"].set_visible(False)
            axB.spines["top"].set_visible(False)
        #_add_panel_label(axB, "B", dx=-65, dy=16)

        # Panel C: total fixation time vs # unique items fixated (Figure2_NN E)
        plot_fix_time_vs_count(
            axC,
            df_all,
            count_col="unique_items_fixated",
            xlabel="# of Items Fixated",
            panel_label="",
        )
        axC.set_xticks([1, 2, 3, 4, 5, 6])
        axC.set_xticklabels([1, 2, 3, 4, 5, 6])
        axC.set_xlim(0.5, 6.5)
        axC.set_ylim(0, 25)
        #_add_panel_label(axC, "C", dx=-90, dy=16)

        # Panel D: signed offer value quintile bar chart for total fixation time
        # Styled to match Figure2's quintile bar chart (double-bar technique)
        _n_bins_d = 5
        _bin_labels_d = ["1", "2", "3", "4", "5"]
        _bar_width_d = 1.0
        _gray_color = ".7"

        _d_d = df_all.dropna(subset=["true_offer_value", "total_fix_time"]).copy()
        _d_d = _d_d[_d_d["total_fix_time"] > 0]
        if len(_d_d) > 0:
            _mu_d = float(_d_d["true_offer_value"].mean())
            _sd_d = float(_d_d["true_offer_value"].std(ddof=0))
            if _sd_d > 0:
                _d_d["z_val"] = (_d_d["true_offer_value"] - _mu_d) / _sd_d
            else:
                _d_d["z_val"] = 0.0
            _d_d["bin"] = pd.qcut(_d_d["z_val"], q=_n_bins_d, labels=_bin_labels_d, duplicates="drop")
            _subj_means_d = _d_d.groupby(["subject", "bin"], observed=True)["total_fix_time"].mean().reset_index()

            _rng_d = np.random.default_rng(42)
            _x_pos_d = np.arange(_n_bins_d)

            # Compute all bin means and SEs first
            _grp_means_d, _grp_ses_d = [], []
            for bi, bl in enumerate(_bin_labels_d):
                vals = _subj_means_d.loc[_subj_means_d["bin"] == bl, "total_fix_time"].values.astype(float)
                m_v = float(np.nanmean(vals))
                se_v = float(np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals)))) if len(vals) > 1 else 0.0
                _grp_means_d.append(m_v)
                _grp_ses_d.append(se_v)

                # Subject dots (styled like Figure2)
                jitter = _rng_d.uniform(-0.15, 0.15, size=len(vals))
                for vi, v in enumerate(vals):
                    axD.scatter(_x_pos_d[bi] + jitter[vi], v,
                                s=6**2, facecolor=(1, 1, 1, 0.5), edgecolor=_gray_color,
                                linewidth=1, zorder=3)

            _grp_means_d = np.array(_grp_means_d)
            _grp_ses_d = np.array(_grp_ses_d)

            # Color fill bars (behind dots)
            axD.bar(_x_pos_d, _grp_means_d, _bar_width_d,
                    color=_gray_color, edgecolor="none", linewidth=0, zorder=2)
            # Black outline bars (on top of dots)
            axD.bar(_x_pos_d, _grp_means_d, _bar_width_d,
                    color="none", edgecolor="black", linewidth=2.5, zorder=4)
            # Error bars on top
            axD.errorbar(_x_pos_d, _grp_means_d, yerr=_grp_ses_d,
                         fmt="none", ecolor="black", capsize=0, linewidth=2.5, zorder=5)

            axD.set_xlabel("Offer Value Quintile")
            axD.set_ylabel("Fixation Time (steps)")
            axD.set_xticks(_x_pos_d)
            axD.set_xticklabels(_bin_labels_d)
            axD.spines["right"].set_visible(False)
            axD.spines["top"].set_visible(False)
            axD.set_ylim(6, 10.5)
            axD.set_yticks([6, 7, 8, 9, 10])
        else:
            axD.axis("off")

        # Panel E: prop relevant fixation time (Figure3_NN A)
        nn_eye._panel_relevant_only(
            axE, subj_rel,
            strip_color="gray",
            mean_facecolor=".5",
            chance_style="k:",
            chance_linewidth=2.5,
            strip_zorder=2,
            strip_edgecolor=None,
        )
        #_add_panel_label(axE, "E", dx=-90, dy=16)

        # Panel F: prop relevant fixation time split by take/leave and valence (Figure3_NN B)
        nn_eye._panel_take_leave_valence(
            axF, subj_val_long,
            lighten_amount=0,
            subject_line_color="0.7",
            figure3_legend=True,
        )
        #_add_panel_label(axF, "F", dx=-65, dy=16)

        chosen_vs = _pick_value_source_from_files(pred_summary_csv, pred_coef_table_csv)

        # Panel G: classification accuracy from prop-time interaction model (visit_type=all)
        # Styled as a bar chart matching Figure3's Panel D (cv accuracy bar)
        if pred_summary_csv is None:
            axG.axis("off")
            axG.text(0.5, 0.5, "Missing summary_prop_time_*.csv", ha="center", va="center", transform=axG.transAxes)
        else:
            summ = pd.read_csv(pred_summary_csv)
            summ = summ.copy()
            if chosen_vs is not None and "value_source" in summ.columns:
                summ = summ[summ["value_source"].astype(str) == chosen_vs]
            if "feature_set" in summ.columns:
                summ = summ[summ["feature_set"].astype(str) == "location_interactions"]
            summ = summ[summ["visit_type"].astype(str) == "all"]

            if len(summ) == 0 or ("cv_mean" not in summ.columns) or ("cv_sem" not in summ.columns):
                axG.axis("off")
                axG.text(0.5, 0.5, "No CV accuracy in summary", ha="center", va="center", transform=axG.transAxes)
            else:
                row = summ.iloc[0]
                _cv_mean = float(row["cv_mean"])
                _cv_sem = float(row["cv_sem"])
                _bar_w = 0.6
                # Gray bar fill (NN)
                axG.bar([0], [_cv_mean], _bar_w,
                        color=".7", edgecolor="none", linewidth=0, zorder=2)
                # Black outline on top
                axG.bar([0], [_cv_mean], _bar_w,
                        color="none", edgecolor="black", linewidth=2.5, zorder=4)
                # Error bar on top
                axG.errorbar([0], [_cv_mean], yerr=[_cv_sem],
                             fmt="none", ecolor="black", capsize=0, linewidth=2.5, zorder=5)
                axG.axhline(0.5, color="black", linestyle="--", linewidth=1, zorder=1)
                axG.set_ylabel("Choice Prediction Accuracy")
                axG.set_xticks([])
                axG.set_xlim(-0.75, 0.75)
                axG.set_ylim(0.5, 0.75)
                axG.set_yticks([0.5, 0.6, 0.7])
                axG.spines["right"].set_visible(False)
                axG.spines["top"].set_visible(False)

        # Panel H: polar/radar plot of regression coefficients, matching Figure3 Panel E
        _outer_radius = 530
        if pred_coef_table_csv is None:
            axH.axis("off")
            axH.text(0.5, 0.5, "Missing coef_table_prop_time_*.csv", ha="center", va="center", transform=axH.transAxes)
        else:
            coef_df = pd.read_csv(pred_coef_table_csv)
            coef_df = coef_df.copy()
            if chosen_vs is not None and "value_source" in coef_df.columns:
                coef_df = coef_df[coef_df["value_source"].astype(str) == chosen_vs]
            if "feature_set" in coef_df.columns:
                coef_df = coef_df[coef_df["feature_set"].astype(str) == "location_interactions"]
            if "visit_type" in coef_df.columns:
                coef_df = coef_df[coef_df["visit_type"].astype(str) == "all"]

            def _get_loc_term(term: str) -> Optional[pd.DataFrame]:
                rows = []
                for loc in range(1, 7):
                    feat = f"loc{loc}_{term}"
                    sub = coef_df[coef_df["feature"].astype(str) == feat]
                    if len(sub) == 0:
                        return None
                    r = sub.iloc[0]
                    rows.append({
                        "loc": int(loc),
                        "coef": float(r["coef"]),
                        "lo": float(r["lo"]),
                        "hi": float(r["hi"]),
                    })
                return pd.DataFrame(rows)

            reward = _get_loc_term("pt_x_val")
            reward_x_rel = _get_loc_term("pt_x_val_x_rel")

            if reward is None or reward_x_rel is None:
                axH.axis("off")
                axH.text(0.5, 0.5, "Missing reward terms in coef table", ha="center", va="center", transform=axH.transAxes)
            else:
                palette = ["#ba7caf", "#6fc7eb"]
                nn_palette = palette

                # --- Polar coordinate helpers ---
                _coef_min, _coef_max = -0.5, 0.75

                def _coef_to_radius(c):
                    return (c - _coef_min) / (_coef_max - _coef_min) * _outer_radius

                def _polar_to_xy(r, angle_deg):
                    rad = np.deg2rad(angle_deg)
                    return r * np.cos(rad), r * np.sin(rad)

                item_angles = {1: 90, 2: 30, 3: -30, 4: -90, 5: -150, 6: 150}
                loc_order = [5, 4, 3, 2, 1, 6]  # CCW angular order

                # --- Set up axes ---
                _lim = _outer_radius + 5
                axH.set_aspect("equal", adjustable="box")
                axH.set_xlim(-_lim, _lim)
                axH.set_ylim(-_lim, _lim)
                for spine in axH.spines.values():
                    spine.set_visible(False)
                axH.set_xticks([])
                axH.set_yticks([])
                axH.set_title("Regression Coefficients", fontsize=27)

                # --- Concentric gridline circles ---
                for tv in [0.0, 0.25, 0.5, 0.75]:
                    r = _coef_to_radius(tv)
                    if tv == 0.0:
                        style = dict(fill=False, edgecolor="black", linewidth=2.5, linestyle=":")
                    else:
                        style = dict(fill=False, edgecolor="black", linewidth=2)
                    axH.add_patch(Circle((0, 0), r, **style, zorder=1))

                # --- Full-diameter sector lines ---
                for i in range(3):
                    angle_deg = i * 60 - 30
                    rad = np.deg2rad(90 - angle_deg)
                    x_end = _outer_radius * np.cos(rad)
                    y_end = _outer_radius * np.sin(rad)
                    axH.plot([-x_end, x_end], [-y_end, y_end],
                             color="black", linewidth=2, zorder=1)

                # --- Tick labels ---
                for tv in [0.0, 0.5]:
                    r = _coef_to_radius(tv)
                    axH.text(r, 0, f"{tv:g}", fontsize=24, ha="center", va="center",
                             color="black", zorder=2,
                             bbox=dict(facecolor="white", edgecolor="none", pad=3))

                # --- Plot NN coefficients as dots only (no connecting lines) ---
                for df_term, color in [(reward, nn_palette[0]), (reward_x_rel, nn_palette[1])]:
                    for _, row in df_term.iterrows():
                        loc = int(row["loc"])
                        r = _coef_to_radius(row["coef"])
                        mx, my = _polar_to_xy(r, item_angles[loc])
                        axH.scatter(mx, my, s=14**2, facecolor=color,
                                    edgecolor="black", linewidth=2.5, zorder=4)

                # --- Legend ---
                legend_handles = [
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=nn_palette[0],
                           markeredgecolor="black", markeredgewidth=2.5, markersize=14,
                           label="Irrelevant x Reward"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=nn_palette[1],
                           markeredgecolor="black", markeredgewidth=2.5, markersize=14,
                           label="Relevant x Reward"),
                ]
                axH.legend(handles=legend_handles, frameon=True, fontsize=20,
                           loc="upper center", handletextpad=0.1, ncol=1,
                           bbox_to_anchor=(0.5, 0),
                           facecolor="white", edgecolor="none")

        # --- Drop-fixation overlay on Panels G and H ---
        if drop_fix_pred_dir is not None:
            drop_fix_pred_dir = os.path.abspath(drop_fix_pred_dir)
            # Find summary and coef table in the drop-fix directory.
            drop_summ_candidates = glob.glob(os.path.join(drop_fix_pred_dir, "summary_prop_time_*.csv"))
            drop_coef_candidates = glob.glob(os.path.join(drop_fix_pred_dir, "coef_table_prop_time_*.csv"))
            drop_summ_csv = _pick_latest(drop_summ_candidates)
            drop_coef_csv = _pick_latest(drop_coef_candidates)

            # Panel G overlay: CV accuracy as X marker.
            if drop_summ_csv is not None:
                ds = pd.read_csv(drop_summ_csv)
                if "feature_set" in ds.columns:
                    ds = ds[ds["feature_set"].astype(str) == "location_interactions"]
                if "visit_type" in ds.columns:
                    ds = ds[ds["visit_type"].astype(str) == "all"]
                if len(ds) > 0 and "cv_mean" in ds.columns and "cv_sem" in ds.columns:
                    dr = ds.iloc[0]
                    axG.scatter(
                        [0], [float(dr["cv_mean"])],
                        s=14**2, marker="X", facecolor="white", edgecolor="black",
                        linewidth=2.5, zorder=7,
                    )
                    axG.errorbar(
                        [0], [float(dr["cv_mean"])], yerr=[float(dr["cv_sem"])],
                        fmt="none", ecolor="black", capsize=0, zorder=6,
                    )
            else:
                print(f"[drop-fix overlay] No summary CSV found in {drop_fix_pred_dir}")

            # Panel H overlay: coefficients as X markers (polar coordinates).
            if drop_coef_csv is not None:
                dc = pd.read_csv(drop_coef_csv)
                if "feature_set" in dc.columns:
                    dc = dc[dc["feature_set"].astype(str) == "location_interactions"]
                if "visit_type" in dc.columns:
                    dc = dc[dc["visit_type"].astype(str) == "all"]

                def _get_loc_term_drop(term: str) -> Optional[pd.DataFrame]:
                    rows = []
                    for loc in range(1, 7):
                        feat = f"loc{loc}_{term}"
                        sub = dc[dc["feature"].astype(str) == feat]
                        if len(sub) == 0:
                            return None
                        r = sub.iloc[0]
                        rows.append({"loc": int(loc), "coef": float(r["coef"]), "lo": float(r["lo"]), "hi": float(r["hi"])})
                    return pd.DataFrame(rows)

                _item_angles_drop = {1: 90, 2: 30, 3: -30, 4: -90, 5: -150, 6: 150}
                _coef_min_d, _coef_max_d = -0.5, 0.75

                def _coef_to_r_drop(c):
                    return (c - _coef_min_d) / (_coef_max_d - _coef_min_d) * _outer_radius

                def _p2xy_drop(r, a):
                    return r * np.cos(np.deg2rad(a)), r * np.sin(np.deg2rad(a))

                palette = ["#ba7caf", "#6fc7eb"]
                for term, color in [("pt_x_val", palette[0]), ("pt_x_val_x_rel", palette[1])]:
                    dt = _get_loc_term_drop(term)
                    if dt is not None:
                        for _, rr in dt.iterrows():
                            loc = int(rr["loc"])
                            r = _coef_to_r_drop(rr["coef"])
                            mx, my = _p2xy_drop(r, _item_angles_drop[loc])
                            axH.scatter(
                                mx, my, s=14**2, marker="X",
                                facecolor=color, edgecolor="black", linewidth=2.5,
                                zorder=5,
                            )
            else:
                print(f"[drop-fix overlay] No coef table CSV found in {drop_fix_pred_dir}")

        # --- Human group-level overlays ---
        if human_data_dir is not None:
            hb = _load_human_benchmarks(human_data_dir, recalled_valence=human_recalled_valence,
                                        bootstrap_ci=bootstrap_ci)

            # Panel A: accuracy
            if "accuracy_mean" in hb:
                m = hb["accuracy_mean"]
                if "accuracy_ci_lo" in hb:
                    _add_human_hline_ci(axA, m, hb["accuracy_ci_lo"], hb["accuracy_ci_hi"])
                else:
                    se = hb["accuracy_sem"]
                    _add_human_hline_ci(axA, m, m - 1.96 * se, m + 1.96 * se)

            # Panel E: relevant fixation time
            if "rel_fix_mean" in hb:
                m = hb["rel_fix_mean"]
                if "rel_fix_ci_lo" in hb:
                    _add_human_hline_ci(axE, m, hb["rel_fix_ci_lo"], hb["rel_fix_ci_hi"])
                else:
                    se = hb["rel_fix_sem"]
                    _add_human_hline_ci(axE, m, m - 1.96 * se, m + 1.96 * se)

            # Panel G: CV accuracy (CI behind bar)
            if "cv_mean" in hb:
                m, se = hb["cv_mean"], hb["cv_sem"]
                _add_human_hline_ci(axG, m, m - 1.96 * se, m + 1.96 * se, zorder=0)

            # Panel H: human regression coefficients as radar lines + CI bands
            if "coef_table" in hb:
                hcoef = hb["coef_table"]
                palette = ["#ba7caf", "#6fc7eb"]
                _item_angles_h = {1: 90, 2: 30, 3: -30, 4: -90, 5: -150, 6: 150}
                _loc_order_h = [5, 4, 3, 2, 1, 6]
                _coef_min_h, _coef_max_h = -0.5, 0.75

                def _coef_to_r_h(c):
                    return (c - _coef_min_h) / (_coef_max_h - _coef_min_h) * _outer_radius

                def _p2xy_h(r, a):
                    return r * np.cos(np.deg2rad(a)), r * np.sin(np.deg2rad(a))

                for term, color in [("pt_x_val", palette[0]), ("pt_x_val_x_rel", palette[1])]:
                    h_coefs, h_los, h_his = {}, {}, {}
                    for loc in range(1, 7):
                        feat = f"loc{loc}_{term}"
                        sub = hcoef[hcoef["feature"].astype(str) == feat]
                        if len(sub) > 0:
                            h_coefs[loc] = float(sub.iloc[0]["coef"])
                            h_los[loc] = float(sub.iloc[0]["lo"])
                            h_his[loc] = float(sub.iloc[0]["hi"])
                    if len(h_coefs) == 6:
                        # Gather in angular order
                        angles_ord = [_item_angles_h[loc] for loc in _loc_order_h]
                        r_means = [_coef_to_r_h(h_coefs[loc]) for loc in _loc_order_h]
                        r_los = [_coef_to_r_h(h_los[loc]) for loc in _loc_order_h]
                        r_his = [_coef_to_r_h(h_his[loc]) for loc in _loc_order_h]

                        mean_xy = [_p2xy_h(r, a) for r, a in zip(r_means, angles_ord)]
                        lo_xy = [_p2xy_h(r, a) for r, a in zip(r_los, angles_ord)]
                        hi_xy = [_p2xy_h(r, a) for r, a in zip(r_his, angles_ord)]

                        # CI band as quads
                        for i in range(len(_loc_order_h)):
                            j = (i + 1) % len(_loc_order_h)
                            quad = np.array([hi_xy[i], hi_xy[j], lo_xy[j], lo_xy[i]])
                            axH.add_patch(Polygon(quad, closed=True, facecolor=color,
                                                  alpha=0.45, edgecolor="none", zorder=2))

            # Panel F: take/leave proportion band (behind NN data)
            if "take_leave_stats" in hb:
                tl = hb["take_leave_stats"]
                decision_order = [d for d in ["take", "leave"] if d in tl["decision_label"].values]
                valence_order = [v for v in ["positive", "negative"] if v in tl["valence_label"].values]
                x_index = {d: i for i, d in enumerate(decision_order)}
                offsets = [-0.32, 0.32]
                val_to_offset = {v: offsets[j] for j, v in enumerate(valence_order)}

                for dlab in decision_order:
                    xs, ys, ci_los, ci_his = [], [], [], []
                    for vlab in valence_order:
                        row = tl[
                            (tl["decision_label"] == dlab)
                            & (tl["valence_label"] == vlab)
                        ]
                        if not row.empty:
                            xs.append(x_index[dlab] + val_to_offset[vlab])
                            m_val = float(row["mean"].values[0])
                            ys.append(m_val)
                            if "ci_lo" in tl.columns:
                                ci_los.append(float(row["ci_lo"].values[0]))
                                ci_his.append(float(row["ci_hi"].values[0]))
                            else:
                                s = float(row["sem"].values[0])
                                ci_los.append(m_val - 1.96 * s)
                                ci_his.append(m_val + 1.96 * s)
                    if len(xs) >= 2:
                        xs_arr = np.array(xs)
                        axF.fill_between(xs_arr, np.array(ci_los), np.array(ci_his),
                                         color=_HUMAN_COLOR, alpha=0.25, linewidth=0, zorder=0)

        fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.12)

        suffix = f"_{tag}" if tag else ""
        out_path = os.path.join(out_dir, f"FigureNN_overview{suffix}.pdf")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    # Sidecar note for reproducibility/debugging.
    meta_path = os.path.join(out_dir, f"FigureNN_overview{('_' + tag) if tag else ''}.meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"root={root}\n")
        f.write(f"out_dir={out_dir}\n")
        f.write(f"tag={tag}\n")
        f.write(f"metric={metric}\n")
        f.write(f"behavior_source={behavior_source}\n")
        f.write(f"behavior_cache={behavior_cache}\n")
        f.write(f"eye_trial_cache={eye_trial_cache}\n")
        f.write(f"prop_time_pred_summary={pred_summary_csv or ''}\n")
        f.write(f"prop_time_pred_coef_table={pred_coef_table_csv or ''}\n")
        f.write(f"human_data_dir={human_data_dir or ''}\n")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NN overview (8-panel) figure.")
    parser.add_argument(
        "--root",
        default="metarnn/simulations/human_like",
        help="NN human_like root containing data/ and output/ subfolders.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for the overview PDF (default: <root>/output/overview).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag appended to output filename.",
    )
    parser.add_argument(
        "--metric",
        choices=["duration", "count"],
        default="duration",
        help="Eyegaze metric for Figure3-derived panels.",
    )
    parser.add_argument(
        "--behavior-csv",
        default=None,
        help="Optional precomputed NN trial-level behavior CSV (avoids rebuilding).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore any cached intermediate CSVs and recompute.",
    )
    parser.add_argument(
        "--human-data-dir",
        default=None,
        help="Path to human output/ directory. Adds horizontal reference lines showing human group-level means on panels A, E, F, G.",
    )
    parser.add_argument(
        "--drop-fix-pred-dir",
        default=None,
        help="Directory with prop-time prediction results from dropped-fixation run. Overlays on panels G and H with X markers.",
    )
    parser.add_argument(
        "--human-recalled-valence",
        action="store_true",
        help="Use recalled reward valence (instead of true) for the human Panel F overlay.",
    )
    parser.add_argument(
        "--bootstrap-ci",
        action="store_true",
        help="Use bootstrapped 95%% CIs (percentile method, 10k resamples) instead of 1.96*SEM for human overlays.",
    )

    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(root, "output", "overview")

    out_path = create_nn_overview_figure(
        root=root,
        out_dir=out_dir,
        tag=args.tag,
        metric=args.metric,
        behavior_csv=args.behavior_csv,
        no_cache=bool(args.no_cache),
        human_data_dir=args.human_data_dir,
        drop_fix_pred_dir=args.drop_fix_pred_dir,
        human_recalled_valence=bool(args.human_recalled_valence),
        bootstrap_ci=bool(args.bootstrap_ci),
    )
    print(f"Saved NN overview figure to {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_behavior import (
    build_recalled_values_map,
    extract_game_items,
    list_subjects,
    load_main_logfile,
    load_valuerecall,
)
from prepare_choice_fixations import is_image_name


EYETRACK_EXCLUDE_SUBJECTS = {"107", "131"}  # per repo instructions


def _write_summary_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_coef_table_csv(
    path: str,
    *,
    coef_summary: CoefSummary,
    meta: dict,
) -> None:
    """Write a long-form coefficient table.

    This is used by composite figures (e.g., NN overview panel H) to recreate
    the exact coefficient panels without re-fitting.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(
        {
            "feature": list(coef_summary.feature_names),
            "coef": np.asarray(coef_summary.coef, dtype=float),
            "lo": np.asarray(coef_summary.lo, dtype=float),
            "hi": np.asarray(coef_summary.hi, dtype=float),
        }
    )
    for k, v in meta.items():
        df[k] = v
    df.to_csv(path, index=False)


def _derived_reward_effects_location_interactions(
    *,
    coef_dist: CoefDist,
) -> Tuple[dict, dict]:
    """Compute derived reward effects for location_interactions feature set.

    Interpretable mapping:
      - "Irrelevant × Reward" uses only the reward term (since rel=0 => reward×relevance term drops out)
      - "Relevant × Reward" uses reward + reward×relevance (since rel=1)

    Because the model uses per-location terms, we aggregate by taking the mean across
    the 6 locations, and quantify uncertainty via the subject-level bootstrap draws.

    Returns:
      irr: dict with keys mean, lo, hi
      rel: dict with keys mean, lo, hi
    """

    feature_names = list(coef_dist.summary.feature_names)
    idx_val = [feature_names.index(f"loc{loc}_pt_x_val") for loc in range(1, 7)]
    idx_val_x_rel = [feature_names.index(f"loc{loc}_pt_x_val_x_rel") for loc in range(1, 7)]

    base = np.asarray(coef_dist.summary.coef, dtype=float)
    irr_mean = float(np.nanmean(base[idx_val]))
    rel_mean = float(np.nanmean(base[idx_val] + base[idx_val_x_rel]))

    boot = np.asarray(coef_dist.samples, dtype=float)
    irr_s = np.nanmean(boot[:, idx_val], axis=1)
    rel_s = np.nanmean(boot[:, idx_val] + boot[:, idx_val_x_rel], axis=1)

    irr = {
        "mean": irr_mean,
        "lo": float(np.nanpercentile(irr_s, 2.5)),
        "hi": float(np.nanpercentile(irr_s, 97.5)),
    }
    rel = {
        "mean": rel_mean,
        "lo": float(np.nanpercentile(rel_s, 2.5)),
        "hi": float(np.nanpercentile(rel_s, 97.5)),
    }
    return irr, rel


def _extract_game_items_flexible(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Extract (game, image, outcome) rows for encoding.

    Human logs store values on rows where (phase==encoding, event==value).
    NN "human_like" exports often store the outcome on (phase==encoding, event==image).
    """

    enc = df[(df["phase"] == "encoding") & (df["event"] == "value")]
    if enc.empty:
        enc = df[(df["phase"] == "encoding") & (df["event"] == "image")]
    enc = enc[["game", "image", "outcome"]].dropna(subset=["game", "image"]).copy()

    game_items: Dict[int, pd.DataFrame] = {}
    for g, gdf in enc.groupby("game"):
        gdf = gdf.drop_duplicates(subset=["image"])  # defensive
        game_items[int(g)] = gdf.reset_index(drop=True)
    return game_items


def _has_any_valuerecall(data_root: str, subjects: List[str]) -> bool:
    for sub in subjects:
        path = os.path.join(data_root, sub, f"{sub}_valuerecall.csv")
        if os.path.exists(path):
            return True
    return False


@dataclass(frozen=True)
class CVSummary:
    mean: float
    sem: float
    fold_acc: np.ndarray
    n_rows: int


@dataclass(frozen=True)
class CoefSummary:
    coef: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    feature_names: List[str]


@dataclass(frozen=True)
class CoefDist:
    summary: CoefSummary
    samples: np.ndarray  # shape (n_samples, n_features)


@dataclass(frozen=True)
class ChanceAccSummary:
    mean: float
    sd: float
    lo: float
    hi: float
    p95: float
    p_perm: float


def _safe_sem(x: Sequence[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return float("nan")
    return float(np.nanstd(x, ddof=1) / np.sqrt(len(x)))


def _zscore_train_apply(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = float(np.nanmean(x_train))
    sd = float(np.nanstd(x_train, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(x_train, dtype=float), np.zeros_like(x_test, dtype=float)
    return (x_train - mu) / sd, (x_test - mu) / sd


def _kfold_indices(n: int, n_folds: int, rng: np.random.Generator) -> List[np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    return [a for a in np.array_split(idx, n_folds) if len(a) > 0]


def _fit_predict_proba_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    regularization: str,
) -> Optional[np.ndarray]:
    """Fit logistic regression and return predicted probabilities.

    regularization:
      - "l2": L2/ridge regularization
      - "none": unregularized MLE (can be unstable under separation)
    """
    if len(np.unique(y_train)) < 2:
        return None

    if regularization not in {"l2", "none"}:
        raise ValueError("regularization must be 'l2' or 'none'")

    # Try sklearn first
    try:
        from sklearn.linear_model import LogisticRegression

        if regularization == "l2":
            clf = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
            )
        else:
            clf = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=2000,
            )
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)[:, 1]
        return np.asarray(p, dtype=float)
    except Exception:
        pass

    # Fallback: statsmodels GLM
    try:
        import statsmodels.api as sm

        Xtr = sm.add_constant(pd.DataFrame(X_train), has_constant="add")
        Xte = sm.add_constant(pd.DataFrame(X_test), has_constant="add")
        model = sm.GLM(y_train, Xtr, family=sm.families.Binomial())
        if regularization == "l2":
            res = model.fit_regularized(alpha=1.0, L1_wt=0.0)
        else:
            res = model.fit()
        p = res.predict(Xte)
        return np.asarray(p, dtype=float)
    except Exception:
        pass

    # Final fallback: pure-numpy ridge logistic regression
    l2 = 1.0 if regularization == "l2" else 0.0
    beta = _fit_logistic_ridge_numpy(X_train, y_train, l2=l2)
    if beta is None:
        return None
    return _predict_proba_numpy(X_test, beta)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def _fit_logistic_ridge_numpy(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Optional[np.ndarray]:
    """Fit ridge-penalized logistic regression via IRLS.

    Returns beta including intercept as beta[0]. Intercept is not penalized.
    """
    if len(np.unique(y)) < 2:
        return None

    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    beta = np.zeros(p + 1, dtype=float)

    penalty = np.diag([0.0] + [l2] * p)

    for _ in range(max_iter):
        eta = Xb @ beta
        mu = _sigmoid(eta)
        w = mu * (1.0 - mu)
        w = np.clip(w, 1e-9, np.inf)
        z = eta + (y - mu) / w

        Xw = Xb * w[:, None]
        H = Xb.T @ Xw + penalty
        rhs = Xw.T @ z
        try:
            beta_new = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(H, rhs, rcond=None)[0]

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return beta


def _predict_proba_numpy(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    return _sigmoid(Xb @ beta)


def _fit_logistic_coef(X: np.ndarray, y: np.ndarray, regularization: str) -> Optional[np.ndarray]:
    """Fit logistic regression and return coefficients (no intercept)."""
    if len(np.unique(y)) < 2:
        return None

    if regularization not in {"l2", "none"}:
        raise ValueError("regularization must be 'l2' or 'none'")

    try:
        from sklearn.linear_model import LogisticRegression

        if regularization == "l2":
            clf = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=5000,
            )
        else:
            clf = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=5000,
            )
        clf.fit(X, y)
        return np.asarray(clf.coef_[0], dtype=float)
    except Exception:
        pass

    try:
        import statsmodels.api as sm

        Xdf = sm.add_constant(pd.DataFrame(X), has_constant="add")
        model = sm.GLM(y, Xdf, family=sm.families.Binomial())
        if regularization == "l2":
            res = model.fit_regularized(alpha=1.0, L1_wt=0.0)
        else:
            res = model.fit()
        params = np.asarray(res.params, dtype=float)
        if len(params) == X.shape[1] + 1:
            return params[1:]
        return params[-X.shape[1] :]
    except Exception:
        pass

    l2 = 1.0 if regularization == "l2" else 0.0
    beta = _fit_logistic_ridge_numpy(X, y, l2=l2)
    if beta is None:
        return None
    return beta[1:]


def _zscore_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    return (X - mu) / sd, mu, sd


def bootstrap_coef_summary(
    df: pd.DataFrame,
    x_cols: Sequence[str],
    y_col: str,
    cluster_col: str,
    n_boot: int,
    seed: int,
    regularization: str,
    return_samples: bool = False,
) -> CoefSummary | CoefDist:
    d = df.dropna(subset=list(x_cols) + [y_col, cluster_col]).copy()
    X = np.column_stack([pd.to_numeric(d[c], errors="coerce").values.astype(float) for c in x_cols])
    y = pd.to_numeric(d[y_col], errors="coerce").values.astype(int)
    Xz, mu, sd = _zscore_columns(X)

    base = _fit_logistic_coef(Xz, y, regularization=regularization)
    if base is None:
        base = np.full(X.shape[1], np.nan, dtype=float)

    rng = np.random.default_rng(seed)
    clusters = d[cluster_col].astype(str).values
    unique_clusters = np.unique(clusters)

    # Pre-build cluster -> row indices mapping to avoid np.isin per iteration.
    cluster_to_idx: Dict[str, np.ndarray] = {}
    for i, c in enumerate(clusters):
        cluster_to_idx.setdefault(c, []).append(i)
    cluster_to_idx = {k: np.array(v, dtype=int) for k, v in cluster_to_idx.items()}

    boot = np.full((n_boot, X.shape[1]), np.nan, dtype=float)
    for b in range(n_boot):
        draw = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        # Concatenate pre-computed index arrays for selected clusters.
        sel = np.concatenate([cluster_to_idx[c] for c in draw])
        Xb = X[sel]
        yb = y[sel]
        # standardize using the original full-sample scaling for comparability
        Xbz = (Xb - mu) / sd
        coef_b = _fit_logistic_coef(Xbz, yb, regularization=regularization)
        if coef_b is None or len(coef_b) != X.shape[1]:
            continue
        boot[b, :] = coef_b

    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    summary = CoefSummary(coef=base, lo=lo, hi=hi, feature_names=list(x_cols))
    if return_samples:
        return CoefDist(summary=summary, samples=boot)
    return summary


def _coef_delta_from_samples(
    obs_samples: np.ndarray,
    null_samples: np.ndarray,
    feature_names: List[str],
    seed: int,
) -> CoefSummary:
    """Compute delta = obs - null and 95% CI via sampling differences.

    We randomly pair bootstrap draws with permutation draws (with replacement if needed).
    """
    rng = np.random.default_rng(seed)
    if obs_samples.ndim != 2 or null_samples.ndim != 2:
        raise ValueError("Samples must be 2D arrays")
    if obs_samples.shape[1] != null_samples.shape[1]:
        raise ValueError("Observed and null samples must have same #features")

    n_obs = obs_samples.shape[0]
    n_null = null_samples.shape[0]
    n = int(min(n_obs, n_null))
    if n <= 1:
        delta = np.full(obs_samples.shape[1], np.nan, dtype=float)
        return CoefSummary(coef=delta, lo=delta, hi=delta, feature_names=feature_names)

    obs_idx = rng.integers(0, n_obs, size=n, endpoint=False)
    null_idx = rng.integers(0, n_null, size=n, endpoint=False)
    diffs = obs_samples[obs_idx, :] - null_samples[null_idx, :]
    delta_mean = np.nanmean(diffs, axis=0)
    delta_lo = np.nanpercentile(diffs, 2.5, axis=0)
    delta_hi = np.nanpercentile(diffs, 97.5, axis=0)
    return CoefSummary(coef=delta_mean, lo=delta_lo, hi=delta_hi, feature_names=feature_names)


def plot_location_delta_panels(
    delta_summary: CoefSummary,
    feature_names: List[str],
    value_source: str,
    out_path: str,
    title: Optional[str] = None,
):
    delta_summary = CoefSummary(
        coef=delta_summary.coef,
        lo=delta_summary.lo,
        hi=delta_summary.hi,
        feature_names=feature_names,
    )

    def _get(term: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = [f"loc{loc}_{term}" for loc in range(1, 7)]
        idx = [delta_summary.feature_names.index(c) for c in cols]
        return (
            delta_summary.coef[idx],
            delta_summary.lo[idx],
            delta_summary.hi[idx],
        )

    terms = [
        ("pt_x_val", "reward"),
        ("pt_x_rel", "relevance"),
        ("pt_x_val_x_rel", "reward×relevance"),
    ]

    y_all = []
    for term, _lab in terms:
        c, lo, hi = _get(term)
        y_all += [c, lo, hi]
    y_min = float(np.nanmin(np.concatenate(y_all)))
    y_max = float(np.nanmax(np.concatenate(y_all)))
    pad = 0.05 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 0.1

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    x = np.arange(1, 7)
    for ax, (term, lab) in zip(axes, terms):
        c, lo, hi = _get(term)
        lo2 = np.minimum(lo, c)
        hi2 = np.maximum(hi, c)
        yerr = np.vstack([c - lo2, hi2 - c])
        ax.errorbar(x, c, yerr=yerr, fmt="o", color="black", ecolor="black", capsize=0)
        ax.axhline(0, color="0.7", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_xlabel("Location")
        ax.set_title(lab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim((y_min - pad, y_max + pad))
    axes[0].set_ylabel("Observed − perm-null coefficient")
    if title is None:
        title = f"Coef difference vs perm-null ({value_source} values)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_rank_delta_panels(
    delta_summary: CoefSummary,
    feature_names: List[str],
    value_source: str,
    out_path: str,
    title: Optional[str] = None,
):
    delta_summary = CoefSummary(
        coef=delta_summary.coef,
        lo=delta_summary.lo,
        hi=delta_summary.hi,
        feature_names=feature_names,
    )

    def _get(term: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = [f"rank{rk}_{term}" for rk in range(1, 7)]
        idx = [delta_summary.feature_names.index(c) for c in cols]
        return (
            delta_summary.coef[idx],
            delta_summary.lo[idx],
            delta_summary.hi[idx],
        )

    terms = [
        ("pt", "reward (rank bins)"),
        ("pt_x_rel", "relevance (rank bins)"),
    ]

    y_all = []
    for term, _lab in terms:
        c, lo, hi = _get(term)
        y_all += [c, lo, hi]
    y_min = float(np.nanmin(np.concatenate(y_all)))
    y_max = float(np.nanmax(np.concatenate(y_all)))
    pad = 0.05 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 0.1

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    x = np.arange(1, 7)
    for ax, (term, lab) in zip(axes, terms):
        c, lo, hi = _get(term)
        lo2 = np.minimum(lo, c)
        hi2 = np.maximum(hi, c)
        yerr = np.vstack([c - lo2, hi2 - c])
        ax.errorbar(x, c, yerr=yerr, fmt="o", color="black", ecolor="black", capsize=0)
        ax.axhline(0, color="0.7", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_xlabel("Value rank (1=most positive)")
        ax.set_title(lab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim((y_min - pad, y_max + pad))
    axes[0].set_ylabel("Observed − perm-null coefficient")
    if title is None:
        title = f"Coef difference vs perm-null ({value_source} values)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_rank_rel_irr_delta_panels(
    delta_summary: CoefSummary,
    feature_names: List[str],
    value_source: str,
    out_path: str,
    title: Optional[str] = None,
):
    delta_summary = CoefSummary(
        coef=delta_summary.coef,
        lo=delta_summary.lo,
        hi=delta_summary.hi,
        feature_names=feature_names,
    )

    def _get(term: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = [f"rank{rk}_{term}" for rk in range(1, 7)]
        idx = [delta_summary.feature_names.index(c) for c in cols]
        return (
            delta_summary.coef[idx],
            delta_summary.lo[idx],
            delta_summary.hi[idx],
        )

    terms = [
        ("pt_rel", "relevant time (rank bins)"),
        ("pt_irr", "irrelevant time (rank bins)"),
    ]

    y_all = []
    for term, _lab in terms:
        c, lo, hi = _get(term)
        y_all += [c, lo, hi]
    y_min = float(np.nanmin(np.concatenate(y_all)))
    y_max = float(np.nanmax(np.concatenate(y_all)))
    pad = 0.05 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 0.1

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    x = np.arange(1, 7)
    for ax, (term, lab) in zip(axes, terms):
        c, lo, hi = _get(term)
        lo2 = np.minimum(lo, c)
        hi2 = np.maximum(hi, c)
        yerr = np.vstack([c - lo2, hi2 - c])
        ax.errorbar(x, c, yerr=yerr, fmt="o", color="black", ecolor="black", capsize=0)
        ax.axhline(0, color="0.7", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_xlabel("Value rank (1=most positive)")
        ax.set_title(lab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim((y_min - pad, y_max + pad))
    axes[0].set_ylabel("Observed − perm-null coefficient")
    if title is None:
        title = f"Coef difference vs perm-null ({value_source} values)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_location_coef_panels(
    coef_summary: CoefSummary,
    value_source: str,
    out_path: str,
    title: Optional[str] = None,
    null_summary: Optional[CoefSummary] = None,
):
    def _get(summary: CoefSummary, term: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = [f"loc{loc}_{term}" for loc in range(1, 7)]
        idx = [summary.feature_names.index(c) for c in cols]
        return (
            summary.coef[idx],
            summary.lo[idx],
            summary.hi[idx],
        )

    terms = [
        ("pt_x_val", "reward"),
        ("pt_x_rel", "relevance"),
        ("pt_x_val_x_rel", "reward×relevance"),
    ]

    y_all = []
    for term, _lab in terms:
        c, lo, hi = _get(coef_summary, term)
        y_all += [c, lo, hi]
        if null_summary is not None:
            cn, lon, hin = _get(null_summary, term)
            y_all += [cn, lon, hin]
    y_min = float(np.nanmin(np.concatenate(y_all)))
    y_max = float(np.nanmax(np.concatenate(y_all)))
    pad = 0.05 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 0.1

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    x = np.arange(1, 7)
    for ax, (term, lab) in zip(axes, terms):
        c, lo, hi = _get(coef_summary, term)
        lo2 = np.minimum(lo, c)
        hi2 = np.maximum(hi, c)
        yerr = np.vstack([c - lo2, hi2 - c])
        ax.errorbar(x - 0.12, c, yerr=yerr, fmt="o", color="black", ecolor="black", capsize=0, label="observed")
        if null_summary is not None:
            cn, lon, hin = _get(null_summary, term)
            lon2 = np.minimum(lon, cn)
            hin2 = np.maximum(hin, cn)
            yerrn = np.vstack([cn - lon2, hin2 - cn])
            ax.errorbar(
                x + 0.12,
                cn,
                yerr=yerrn,
                fmt="s",
                color="0.5",
                ecolor="0.5",
                capsize=0,
                label="perm null",
            )
        ax.axhline(0, color="0.7", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_xlabel("Location")
        ax.set_title(lab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim((y_min - pad, y_max + pad))

    axes[0].set_ylabel("Coefficient (z-scored predictors)")
    if title is None:
        title = f"Prop-time interaction coefficients ({value_source} values)"
    fig.suptitle(title)
    if null_summary is not None:
        axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_visit_type_accuracy_comparison(
    observed: Dict[str, CVSummary],
    chance: Dict[str, Optional[ChanceAccSummary]],
    out_path: str,
    title: str,
):
    """Plot observed CV accuracy (mean±SEM) and permutation p95 threshold for visit types."""

    order = ["all", "first", "revisit"]
    labels = {"all": "all", "first": "first", "revisit": "revisit"}

    x = np.arange(len(order), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))

    # observed
    obs_mean = [observed[v].mean for v in order]
    obs_sem = [observed[v].sem for v in order]
    ax.errorbar(
        x,
        obs_mean,
        yerr=obs_sem,
        fmt="o",
        color="black",
        ecolor="black",
        capsize=0,
        label="observed (CV mean±SEM)",
    )

    # permutation baseline: dashed line at one-sided 95th percentile (p95)
    # Draw a short horizontal segment centered on each category (not an offset marker).
    seg_half_width = 0.20
    p95_label_used = False
    for xi, v in zip(x, order):
        s = chance.get(v)
        if s is None or not np.isfinite(s.p95):
            continue
        ax.hlines(
            s.p95,
            xi - seg_half_width,
            xi + seg_half_width,
            colors="0.5",
            linestyles="--",
            linewidth=1.5,
            label=None if p95_label_used else "perm 95th percentile",
        )
        p95_label_used = True

    ax.set_xticks(x)
    ax.set_xticklabels([labels[v] for v in order])
    ax.set_xlabel("Fixation component")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_visit_type_accuracy_delta_from_p95(
    observed: Dict[str, CVSummary],
    chance: Dict[str, Optional[ChanceAccSummary]],
    out_path: str,
    title: str,
):
    """Plot delta accuracy = observed CV mean - perm 95th percentile (one-sided).

    Error bars reflect the SEM of the observed CV accuracy across folds.
    """

    order = ["all", "first", "revisit"]
    labels = {"all": "all", "first": "first", "revisit": "revisit"}

    x = np.arange(len(order), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))

    delta = []
    sem = []
    for v in order:
        obs = observed[v].mean
        se = observed[v].sem
        ch = chance.get(v)
        p95 = np.nan if ch is None else ch.p95
        delta.append(obs - p95)
        sem.append(se)

    delta = np.asarray(delta, dtype=float)
    sem = np.asarray(sem, dtype=float)

    ax.errorbar(x, delta, yerr=sem, fmt="o", color="black", ecolor="black", capsize=0)
    ax.axhline(0, color="0.7", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[v] for v in order])
    ax.set_xlabel("Fixation component")
    ax.set_ylabel("CV accuracy − perm 95th percentile")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_visit_type_permutation_accuracy_distributions(
    perm_acc: Dict[str, np.ndarray],
    observed: Dict[str, CVSummary],
    chance: Dict[str, Optional[ChanceAccSummary]],
    out_path: str,
    title: str,
):
    """Plot permutation accuracy distributions by visit type.

    Shows histograms of permutation (null) CV accuracy along with vertical dotted
    lines for the one-sided permutation 95th percentile and the observed CV mean.
    """

    order = ["all", "first", "revisit"]
    labels = {"all": "All", "first": "First", "revisit": "Revisit"}

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    # choose a common x-range for comparability
    all_vals: List[float] = []
    for vt in order:
        sims = perm_acc.get(vt)
        if sims is not None:
            sims = np.asarray(sims, dtype=float)
            sims = sims[np.isfinite(sims)]
            all_vals.extend(sims.tolist())
        if vt in observed and np.isfinite(observed[vt].mean):
            all_vals.append(float(observed[vt].mean))
        s = chance.get(vt)
        if s is not None and np.isfinite(s.p95):
            all_vals.append(float(s.p95))

    if all_vals:
        x_min = float(np.nanmin(all_vals))
        x_max = float(np.nanmax(all_vals))
        pad = 0.01
        xlim = (x_min - pad, x_max + pad)
    else:
        xlim = None

    p95_label_used = False
    obs_label_used = False
    for ax, vt in zip(axes, order):
        sims = perm_acc.get(vt)
        if sims is None:
            ax.set_axis_off()
            continue

        sims = np.asarray(sims, dtype=float)
        sims = sims[np.isfinite(sims)]
        if len(sims) == 0:
            ax.set_axis_off()
            continue

        ax.hist(
            sims,
            bins=30,
            density=True,
            color="0.85",
            edgecolor="0.55",
            linewidth=0.6,
        )

        s = chance.get(vt)
        if s is not None and np.isfinite(s.p95):
            ax.axvline(
                s.p95,
                color="0.4",
                linestyle=":",
                linewidth=1.8,
                label=None if p95_label_used else "perm 95th percentile",
            )
            p95_label_used = True

        if vt in observed and np.isfinite(observed[vt].mean):
            ax.axvline(
                observed[vt].mean,
                color="black",
                linestyle=":",
                linewidth=1.8,
                label=None if obs_label_used else "observed",
            )
            obs_label_used = True

        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_title(labels[vt])
        ax.set_xlabel("CV accuracy")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Density")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def plot_first_n_permutation_accuracy_distributions(
    perm_acc: Dict[int, np.ndarray],
    observed: Dict[int, CVSummary],
    chance: Dict[int, Optional[ChanceAccSummary]],
    out_path: str,
    title: str,
):
    """Plot permutation accuracy distributions across first-N fixation settings.

    Creates a grid of histograms (one per N) of permutation (null) CV accuracies.
    Each panel includes vertical dotted lines for the permutation 95th percentile
    and the observed CV mean.
    """

    ns = sorted(perm_acc.keys())
    if len(ns) == 0:
        raise ValueError("No permutation samples provided")

    # Choose common x-limits for comparability
    all_vals: List[float] = []
    for n in ns:
        sims = np.asarray(perm_acc.get(n), dtype=float)
        sims = sims[np.isfinite(sims)]
        all_vals.extend(sims.tolist())
        if n in observed and np.isfinite(observed[n].mean):
            all_vals.append(float(observed[n].mean))
        s = chance.get(n)
        if s is not None and np.isfinite(s.p95):
            all_vals.append(float(s.p95))

    if all_vals:
        x_min = float(np.nanmin(all_vals))
        x_max = float(np.nanmax(all_vals))
        pad = 0.01
        xlim = (x_min - pad, x_max + pad)
    else:
        xlim = None

    n_panels = len(ns)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.5, 3.2 * n_rows), sharey=True)
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    p95_label_used = False
    obs_label_used = False
    for i, n in enumerate(ns):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]

        sims = np.asarray(perm_acc.get(n), dtype=float)
        sims = sims[np.isfinite(sims)]
        if len(sims) == 0:
            ax.set_axis_off()
            continue

        ax.hist(
            sims,
            bins=30,
            density=True,
            color="0.85",
            edgecolor="0.55",
            linewidth=0.6,
        )

        s = chance.get(n)
        if s is not None and np.isfinite(s.p95):
            ax.axvline(
                s.p95,
                color="0.4",
                linestyle=":",
                linewidth=1.8,
                label=None if p95_label_used else "perm 95th percentile",
            )
            p95_label_used = True

        if n in observed and np.isfinite(observed[n].mean):
            ax.axvline(
                observed[n].mean,
                color="black",
                linestyle=":",
                linewidth=1.8,
                label=None if obs_label_used else "observed",
            )
            obs_label_used = True

        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_title(f"N={n}")
        ax.set_xlabel("CV accuracy")
        ax.spines[["top", "right"]].set_visible(False)

    # turn off any unused axes
    for j in range(n_panels, n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        axes[r, c].set_axis_off()

    axes[0, 0].set_ylabel("Density")
    fig.suptitle(title)
    # put legend once, if anything was labeled
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


def plot_location_coef_panels_by_visit_type(
    coef_by_visit: Dict[str, CoefSummary],
    value_source: str,
    out_path: str,
    title: str,
):
    """Overlay coefficient panels with hues for all/first/revisit.

    Only implemented for location_interactions feature set.
    """

    order = ["all", "first", "revisit"]
    colors = {"all": "black", "first": "tab:blue", "revisit": "tab:orange"}
    markers = {"all": "o", "first": "^", "revisit": "s"}

    def _get(summary: CoefSummary, term: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = [f"loc{loc}_{term}" for loc in range(1, 7)]
        idx = [summary.feature_names.index(c) for c in cols]
        return (
            summary.coef[idx],
            summary.lo[idx],
            summary.hi[idx],
        )

    terms = [
        ("pt_x_val", "reward"),
        ("pt_x_rel", "relevance"),
        ("pt_x_val_x_rel", "reward×relevance"),
    ]

    # y-limits across all visit types + terms
    y_all = []
    for v in order:
        s = coef_by_visit.get(v)
        if s is None:
            continue
        for term, _lab in terms:
            c, lo, hi = _get(s, term)
            y_all += [c, lo, hi]
    y_min = float(np.nanmin(np.concatenate(y_all)))
    y_max = float(np.nanmax(np.concatenate(y_all)))
    pad = 0.05 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 0.1

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    x = np.arange(1, 7)

    offsets = {"all": -0.20, "first": 0.0, "revisit": 0.20}
    for ax, (term, lab) in zip(axes, terms):
        for v in order:
            s = coef_by_visit.get(v)
            if s is None:
                continue
            c, lo, hi = _get(s, term)
            lo2 = np.minimum(lo, c)
            hi2 = np.maximum(hi, c)
            yerr = np.vstack([c - lo2, hi2 - c])
            ax.errorbar(
                x + offsets[v],
                c,
                yerr=yerr,
                fmt=markers[v],
                color=colors[v],
                ecolor=colors[v],
                capsize=0,
                label=v if term == terms[0][0] else None,
            )
        ax.axhline(0, color="0.7", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_xlabel("Location")
        ax.set_title(lab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim((y_min - pad, y_max + pad))

    axes[0].set_ylabel("Coefficient (z-scored predictors)")
    fig.suptitle(title)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_first_n_accuracy_comparison(
    observed: Dict[int, CVSummary],
    chance: Dict[int, Optional[ChanceAccSummary]],
    out_path: str,
    title: str,
):
    """Plot observed CV accuracy (mean±SEM) across first-N fixation settings.

    If chance summaries are provided (i.e., permutations were run), overlays the
    one-sided permutation 95th percentile (p95) as short dashed segments.
    """

    ns = sorted(observed.keys())
    x = np.asarray(ns, dtype=float)
    y = np.asarray([observed[n].mean for n in ns], dtype=float)
    yerr = np.asarray([observed[n].sem for n in ns], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.4))
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        color="black",
        ecolor="black",
        capsize=0,
        label="observed (CV mean±SEM)",
    )

    seg_half_width = 0.18
    p95_label_used = False
    for n in ns:
        s = chance.get(n)
        if s is None or not np.isfinite(s.p95):
            continue
        ax.hlines(
            s.p95,
            n - seg_half_width,
            n + seg_half_width,
            colors="0.5",
            linestyles="--",
            linewidth=1.5,
            label=None if p95_label_used else "perm 95th percentile",
        )
        p95_label_used = True

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(n)) for n in x])
    ax.set_xlabel("First N fixations")
    ax.set_ylabel("Accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_location_coef_panels_by_first_n(
    coef_by_n: Dict[int, CoefSummary],
    value_source: str,
    out_path: str,
    title: str,
):
    """Overlay location_interactions coefficient panels across N settings."""

    ns = sorted(coef_by_n.keys())
    if len(ns) == 0:
        raise ValueError("No coefficient summaries provided")

    cmap = plt.get_cmap("viridis", len(ns))
    colors = {n: cmap(i) for i, n in enumerate(ns)}
    markers = ["o", "^", "s", "D", "v", "P", "X"]
    marker_by_n = {n: markers[i % len(markers)] for i, n in enumerate(ns)}
    offsets = np.linspace(-0.25, 0.25, num=len(ns))
    offset_by_n = {n: float(off) for n, off in zip(ns, offsets)}

    def _get(summary: CoefSummary, term: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = [f"loc{loc}_{term}" for loc in range(1, 7)]
        idx = [summary.feature_names.index(c) for c in cols]
        return (
            summary.coef[idx],
            summary.lo[idx],
            summary.hi[idx],
        )

    terms = [
        ("pt_x_val", "reward"),
        ("pt_x_rel", "relevance"),
        ("pt_x_val_x_rel", "reward×relevance"),
    ]

    y_all = []
    for n in ns:
        s = coef_by_n.get(n)
        if s is None:
            continue
        for term, _lab in terms:
            c, lo, hi = _get(s, term)
            y_all += [c, lo, hi]
    y_min = float(np.nanmin(np.concatenate(y_all)))
    y_max = float(np.nanmax(np.concatenate(y_all)))
    pad = 0.05 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 0.1

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    x = np.arange(1, 7)
    for ax, (term, lab) in zip(axes, terms):
        for n in ns:
            s = coef_by_n.get(n)
            if s is None:
                continue
            c, lo, hi = _get(s, term)
            lo2 = np.minimum(lo, c)
            hi2 = np.maximum(hi, c)
            yerr = np.vstack([c - lo2, hi2 - c])
            ax.errorbar(
                x + offset_by_n[n],
                c,
                yerr=yerr,
                fmt=marker_by_n[n],
                color=colors[n],
                ecolor=colors[n],
                capsize=0,
                label=f"N={n}" if term == terms[0][0] else None,
            )
        ax.axhline(0, color="0.7", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_xlabel("Location")
        ax.set_title(lab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim((y_min - pad, y_max + pad))

    axes[0].set_ylabel("Coefficient (z-scored predictors)")
    fig.suptitle(title)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_rank_coef_panels(
    coef_summary: CoefSummary,
    value_source: str,
    out_path: str,
    title: Optional[str] = None,
    null_summary: Optional[CoefSummary] = None,
):
    def _get(summary: CoefSummary, term: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = [f"rank{rk}_{term}" for rk in range(1, 7)]
        idx = [summary.feature_names.index(c) for c in cols]
        return (
            summary.coef[idx],
            summary.lo[idx],
            summary.hi[idx],
        )

    terms = [
        ("pt", "reward (rank bins)"),
        ("pt_x_rel", "relevance (rank bins)"),
    ]

    y_all = []
    for term, _lab in terms:
        c, lo, hi = _get(coef_summary, term)
        y_all += [c, lo, hi]
        if null_summary is not None:
            cn, lon, hin = _get(null_summary, term)
            y_all += [cn, lon, hin]
    y_min = float(np.nanmin(np.concatenate(y_all)))
    y_max = float(np.nanmax(np.concatenate(y_all)))
    pad = 0.05 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 0.1

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    x = np.arange(1, 7)
    for ax, (term, lab) in zip(axes, terms):
        c, lo, hi = _get(coef_summary, term)
        lo2 = np.minimum(lo, c)
        hi2 = np.maximum(hi, c)
        yerr = np.vstack([c - lo2, hi2 - c])
        ax.errorbar(x - 0.12, c, yerr=yerr, fmt="o", color="black", ecolor="black", capsize=0, label="observed")
        if null_summary is not None:
            cn, lon, hin = _get(null_summary, term)
            lon2 = np.minimum(lon, cn)
            hin2 = np.maximum(hin, cn)
            yerrn = np.vstack([cn - lon2, hin2 - cn])
            ax.errorbar(
                x + 0.12,
                cn,
                yerr=yerrn,
                fmt="s",
                color="0.5",
                ecolor="0.5",
                capsize=0,
                label="perm null",
            )
        ax.axhline(0, color="0.7", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_xlabel("Value rank (1=most positive)")
        ax.set_title(lab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim((y_min - pad, y_max + pad))

    axes[0].set_ylabel("Coefficient (z-scored predictors)")
    if title is None:
        title = f"Rank-based prop-time coefficients ({value_source} values)"
    fig.suptitle(title)
    if null_summary is not None:
        axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_rank_rel_irr_panels(
    coef_summary: CoefSummary,
    value_source: str,
    out_path: str,
    title: Optional[str] = None,
    null_summary: Optional[CoefSummary] = None,
):
    def _get(summary: CoefSummary, term: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = [f"rank{rk}_{term}" for rk in range(1, 7)]
        idx = [summary.feature_names.index(c) for c in cols]
        return (
            summary.coef[idx],
            summary.lo[idx],
            summary.hi[idx],
        )

    terms = [
        ("pt_rel", "relevant time (rank bins)"),
        ("pt_irr", "irrelevant time (rank bins)"),
    ]

    y_all = []
    for term, _lab in terms:
        c, lo, hi = _get(coef_summary, term)
        y_all += [c, lo, hi]
        if null_summary is not None:
            cn, lon, hin = _get(null_summary, term)
            y_all += [cn, lon, hin]
    y_min = float(np.nanmin(np.concatenate(y_all)))
    y_max = float(np.nanmax(np.concatenate(y_all)))
    pad = 0.05 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 0.1

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    x = np.arange(1, 7)
    for ax, (term, lab) in zip(axes, terms):
        c, lo, hi = _get(coef_summary, term)
        lo2 = np.minimum(lo, c)
        hi2 = np.maximum(hi, c)
        yerr = np.vstack([c - lo2, hi2 - c])
        ax.errorbar(x - 0.12, c, yerr=yerr, fmt="o", color="black", ecolor="black", capsize=0, label="observed")
        if null_summary is not None:
            cn, lon, hin = _get(null_summary, term)
            lon2 = np.minimum(lon, cn)
            hin2 = np.maximum(hin, cn)
            yerrn = np.vstack([cn - lon2, hin2 - cn])
            ax.errorbar(
                x + 0.12,
                cn,
                yerr=yerrn,
                fmt="s",
                color="0.5",
                ecolor="0.5",
                capsize=0,
                label="perm null",
            )
        ax.axhline(0, color="0.7", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_xlabel("Value rank (1=most positive)")
        ax.set_title(lab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim((y_min - pad, y_max + pad))

    axes[0].set_ylabel("Coefficient (z-scored predictors)")
    if title is None:
        title = f"Rank rel/irr prop-time coefficients ({value_source} values)"
    fig.suptitle(title)
    if null_summary is not None:
        axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def cross_validated_accuracy(
    df: pd.DataFrame,
    x_cols: Sequence[str],
    y_col: str,
    n_folds: int,
    seed: int,
    regularization: str,
) -> CVSummary:
    d = df.dropna(subset=list(x_cols) + [y_col]).copy()
    if len(d) < max(50, n_folds * 5):
        return CVSummary(mean=float("nan"), sem=float("nan"), fold_acc=np.full(n_folds, np.nan), n_rows=len(d))

    X = np.column_stack([pd.to_numeric(d[c], errors="coerce").values.astype(float) for c in x_cols])
    y = pd.to_numeric(d[y_col], errors="coerce").values.astype(int)

    rng = np.random.default_rng(seed)
    folds = _kfold_indices(len(d), n_folds, rng)

    accs: List[float] = []
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(len(d)), test_idx)
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # z-score each column using training stats
        X_train_z = np.zeros_like(X_train)
        X_test_z = np.zeros_like(X_test)
        for j in range(X.shape[1]):
            X_train_z[:, j], X_test_z[:, j] = _zscore_train_apply(X_train[:, j], X_test[:, j])

        p = _fit_predict_proba_logistic(X_train_z, y_train, X_test_z, regularization=regularization)
        if p is None:
            accs.append(np.nan)
            continue
        yhat = (p >= 0.5).astype(int)
        accs.append(float(np.mean(yhat == y_test)))

    if len(accs) < n_folds:
        accs = accs + [np.nan] * (n_folds - len(accs))
    fold_acc = np.asarray(accs, dtype=float)
    return CVSummary(mean=float(np.nanmean(fold_acc)), sem=_safe_sem(fold_acc), fold_acc=fold_acc, n_rows=len(d))


def _clean_fixation_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Some files have a stray newline in fix_y
    if "fix_y\n" in df.columns and "fix_y" not in df.columns:
        df = df.rename(columns={"fix_y\n": "fix_y"})
    return df


def _ensure_fix_start(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a numeric `fix_start` column exists for within-trial ordering.

    Most processed fixation files include `fix_start` (EDF timestamps). If missing,
    we fall back to the row order as an approximate time ordering.
    """
    if "fix_start" not in df.columns:
        df = df.copy()
        df["fix_start"] = np.arange(len(df), dtype=float)
        return df
    out = df.copy()
    out["fix_start"] = pd.to_numeric(out["fix_start"], errors="coerce")
    # if everything is missing, fall back to row order
    if not np.any(np.isfinite(out["fix_start"].to_numpy(dtype=float, na_value=np.nan))):
        out["fix_start"] = np.arange(len(out), dtype=float)
    return out


def _infer_location_ids_from_roi_centers(df_choice_fix: pd.DataFrame) -> Dict[Tuple[float, float], int]:
    """Map rounded ROI centers -> loc_id in {1..6} using angle ordering."""
    rx = pd.to_numeric(df_choice_fix["roi_x"], errors="coerce")
    ry = pd.to_numeric(df_choice_fix["roi_y"], errors="coerce")
    keep = np.isfinite(rx) & np.isfinite(ry)
    rx = rx[keep].to_numpy(dtype=float)
    ry = ry[keep].to_numpy(dtype=float)

    centers = sorted(set(zip(np.round(rx, 1), np.round(ry, 1))))
    if len(centers) != 6:
        raise ValueError(f"Expected 6 unique ROI centers, found {len(centers)}")

    cx = float(np.mean([c[0] for c in centers]))
    cy = float(np.mean([c[1] for c in centers]))
    angles = []
    for (x, y) in centers:
        ang = float(np.arctan2((y - cy), (x - cx)))
        angles.append((ang, (x, y)))

    # sort counter-clockwise by angle
    angles.sort(key=lambda t: t[0])
    return {xy: i + 1 for i, (_a, xy) in enumerate(angles)}


def _build_location_image_map(df_choice_fix: pd.DataFrame, loc_map: Dict[Tuple[float, float], int]) -> Dict[Tuple[int, int], str]:
    """Return mapping (game, loc_id) -> image_name based on modal roi_content."""
    tmp = df_choice_fix.copy()
    tmp["roi_xr"] = pd.to_numeric(tmp["roi_x"], errors="coerce").round(1)
    tmp["roi_yr"] = pd.to_numeric(tmp["roi_y"], errors="coerce").round(1)
    tmp = tmp.dropna(subset=["roi_xr", "roi_yr"]).copy()
    tmp["loc_id"] = [loc_map.get((x, y), np.nan) for x, y in zip(tmp["roi_xr"], tmp["roi_yr"])]
    tmp = tmp.dropna(subset=["loc_id"]).copy()
    tmp["loc_id"] = tmp["loc_id"].astype(int)

    mapping: Dict[Tuple[int, int], str] = {}
    for (g, loc), gdf in tmp.groupby(["game", "loc_id"], sort=False):
        # mode image name
        mode = gdf["roi_content"].astype(str).mode(dropna=True)
        if len(mode) == 0:
            continue
        mapping[(int(g), int(loc))] = str(mode.iloc[0]).strip()
    return mapping


def build_prop_time_interaction_dataset(
    subid: str,
    data_root: str,
    output_root: str,
    value_source: str,
    feature_set: str = "location_interactions",
    visit_type: str = "all",
    visit_normalization: str = "within",
    first_n_fixations: Optional[int] = None,
    drop_frac: float = 0.0,
    drop_seed: int = 0,
) -> Optional[pd.DataFrame]:
    if value_source not in {"true", "recalled"}:
        raise ValueError("value_source must be 'true' or 'recalled'")

    if visit_type not in {"all", "first", "revisit", "first_n"}:
        raise ValueError("visit_type must be one of: all, first, revisit, first_n")

    if visit_type == "first_n":
        if first_n_fixations is None:
            raise ValueError("first_n_fixations must be provided when visit_type='first_n'")
        if int(first_n_fixations) <= 0:
            raise ValueError("first_n_fixations must be a positive integer")
        first_n_fixations = int(first_n_fixations)

    if visit_normalization not in {"within", "total"}:
        raise ValueError("visit_normalization must be one of: within, total")

    df_main = load_main_logfile(subid, data_root)
    if df_main is None:
        return None

    game_items = _extract_game_items_flexible(df_main)
    if len(game_items) == 0:
        return None

    recalled_map: Dict[int, Dict[str, float]] = {}
    if value_source == "recalled":
        df_val = load_valuerecall(subid, data_root)
        recalled_map = build_recalled_values_map(df_main, df_val) if df_val is not None else {}

    fix_file = os.path.join(output_root, subid, f"{subid}_fixations_df_original_buffer_50.csv")
    if not os.path.exists(fix_file):
        return None

    fix = pd.read_csv(fix_file)
    fix = _clean_fixation_columns(fix)
    fix["phase"] = fix.get("phase").astype(str).str.strip()
    fix["event"] = fix.get("event").astype(str).str.strip()

    if "roi_content" not in fix.columns:
        return None

    # We'll use two views:
    # - fix_choice: choice-phase fixations (for prop_time)
    # - fix_all_img: all fixations to image ROIs (for stable location inference + loc->image mapping)
    fix_all_img = fix[fix["roi_content"].apply(is_image_name)].copy()
    fix_choice = fix[(fix["phase"] == "choice") & (fix["event"] == "choice")].copy()
    fix_choice = fix_choice[fix_choice["roi_content"].apply(is_image_name)].copy()

    req = ["game", "trial_number", "option", "roi_content", "fix_duration_bounded", "roi_x", "roi_y"]
    missing = [c for c in req if c not in fix_all_img.columns]
    if missing:
        raise ValueError(f"Missing required columns in fixation file for {subid}: {missing}")

    for dfx in (fix_all_img, fix_choice):
        dfx["game"] = pd.to_numeric(dfx["game"], errors="coerce")
        dfx["trial_number"] = pd.to_numeric(dfx["trial_number"], errors="coerce")
        dfx["fix_dur"] = pd.to_numeric(dfx["fix_duration_bounded"], errors="coerce")
        dfx["option"] = dfx.get("option").astype(str).str.strip()
        dfx["roi_content"] = dfx["roi_content"].astype(str).str.strip()

    fix_choice = _ensure_fix_start(fix_choice)

    fix_choice = fix_choice.dropna(subset=["game", "trial_number", "fix_dur"]).copy()
    fix_choice = fix_choice[fix_choice["fix_dur"] > 0].copy()
    fix_choice["game"] = fix_choice["game"].astype(int)
    fix_choice["trial_number"] = fix_choice["trial_number"].astype(int)

    # infer stable 6 locations using all image-ROI rows (choice-only can miss unfixated locations)
    fix_for_centers = fix_all_img.dropna(subset=["roi_x", "roi_y"]).copy()
    loc_map = _infer_location_ids_from_roi_centers(fix_for_centers)

    # build (game, loc)->image mapping from all image-ROI fixations
    fix_all_img = fix_all_img.dropna(subset=["game"]).copy()
    fix_all_img["game"] = fix_all_img["game"].astype(int)
    loc_img_map = _build_location_image_map(fix_all_img, loc_map)

    # Build per-trial prop_time per location
    fix_choice["roi_xr"] = pd.to_numeric(fix_choice["roi_x"], errors="coerce").round(1)
    fix_choice["roi_yr"] = pd.to_numeric(fix_choice["roi_y"], errors="coerce").round(1)
    fix_choice = fix_choice.dropna(subset=["roi_xr", "roi_yr"]).copy()
    fix_choice["loc_id"] = [loc_map.get((x, y), np.nan) for x, y in zip(fix_choice["roi_xr"], fix_choice["roi_yr"])]
    fix_choice = fix_choice.dropna(subset=["loc_id"]).copy()
    fix_choice["loc_id"] = fix_choice["loc_id"].astype(int)

    # Optionally drop a fraction of fixations (randomly, per trial).
    if drop_frac > 0.0:
        rng_drop = np.random.default_rng(drop_seed)
        keep_mask = rng_drop.random(len(fix_choice)) >= drop_frac
        fix_choice = fix_choice[keep_mask].copy()
        if fix_choice.empty:
            return None

    # Split fixation time by (a) the first fixation to each location within the trial vs (b) revisits
    # (subsequent fixations to that same location during the same trial).
    fix_choice = fix_choice.sort_values(["game", "trial_number", "option", "fix_start"]).copy()

    # total duration per location
    dur_total = (
        fix_choice.groupby(["game", "trial_number", "option", "loc_id"], sort=False)["fix_dur"]
        .sum()
        .reset_index(name="dur_total")
    )
    # first fixation duration per location
    first_rows = fix_choice.groupby(["game", "trial_number", "option", "loc_id"], sort=False).head(1)
    dur_first = (
        first_rows.groupby(["game", "trial_number", "option", "loc_id"], sort=False)["fix_dur"]
        .sum()
        .reset_index(name="dur_first")
    )

    dur = dur_total.merge(dur_first, how="left", on=["game", "trial_number", "option", "loc_id"])
    dur["dur_first"] = pd.to_numeric(dur["dur_first"], errors="coerce").fillna(0.0)
    dur["dur_total"] = pd.to_numeric(dur["dur_total"], errors="coerce").fillna(0.0)
    dur["dur_revisit"] = (dur["dur_total"] - dur["dur_first"]).clip(lower=0.0)

    def _wide(col: str) -> pd.DataFrame:
        w = dur.pivot_table(
            index=["game", "trial_number", "option"],
            columns="loc_id",
            values=col,
            fill_value=0.0,
            aggfunc="sum",
        )
        for loc in range(1, 7):
            if loc not in w.columns:
                w[loc] = 0.0
        return w[[1, 2, 3, 4, 5, 6]].copy()

    wide_total = _wide("dur_total")
    wide_first = _wide("dur_first")
    wide_revisit = _wide("dur_revisit")

    wide_first_n: Optional[pd.DataFrame] = None
    denom_first_n: Optional[pd.Series] = None
    if visit_type == "first_n":
        first_n_rows = (
            fix_choice.groupby(["game", "trial_number", "option"], sort=False)
            .head(int(first_n_fixations))
            .copy()
        )
        dur_first_n = (
            first_n_rows.groupby(["game", "trial_number", "option", "loc_id"], sort=False)["fix_dur"]
            .sum()
            .reset_index(name="dur_first_n")
        )
        wide_first_n = dur_first_n.pivot_table(
            index=["game", "trial_number", "option"],
            columns="loc_id",
            values="dur_first_n",
            fill_value=0.0,
            aggfunc="sum",
        )
        for loc in range(1, 7):
            if loc not in wide_first_n.columns:
                wide_first_n[loc] = 0.0
        wide_first_n = wide_first_n[[1, 2, 3, 4, 5, 6]].copy()
        denom_first_n = wide_first_n.sum(axis=1).astype(float)

    # Denominators for proportion time
    denom_total = wide_total.sum(axis=1).astype(float)
    denom_first = wide_first.sum(axis=1).astype(float)
    denom_revisit = wide_revisit.sum(axis=1).astype(float)

    if visit_type == "all":
        wide_use = wide_total
        denom_use = denom_total
    elif visit_type == "first":
        wide_use = wide_first
        denom_use = denom_first if visit_normalization == "within" else denom_total
    elif visit_type == "revisit":
        wide_use = wide_revisit
        denom_use = denom_revisit if visit_normalization == "within" else denom_total
    else:
        assert wide_first_n is not None and denom_first_n is not None
        wide_use = wide_first_n
        denom_use = denom_first_n if visit_normalization == "within" else denom_total

    prop = wide_use.div(denom_use.replace({0.0: np.nan}), axis=0)
    prop.columns = [f"prop_time_loc{loc}" for loc in prop.columns]

    # Always compute revisit fraction (based on *all* fixations), regardless of visit_type
    tot_all = wide_total.sum(axis=1).astype(float)
    rev_all = wide_revisit.sum(axis=1).astype(float)
    revisit_fraction = (rev_all / tot_all.replace({0.0: np.nan})).rename("revisit_fraction")

    # Build choice outcome table
    choices = df_main[(df_main["phase"] == "choice") & (df_main["event"] == "choice")].copy()
    if "trial_number" not in choices.columns:
        return None
    choices["game"] = pd.to_numeric(choices["game"], errors="coerce")
    choices["trial_number"] = pd.to_numeric(choices["trial_number"], errors="coerce")
    choices = choices.dropna(subset=["game", "trial_number", "choice", "option"]).copy()
    choices["game"] = choices["game"].astype(int)
    choices["trial_number"] = choices["trial_number"].astype(int)
    choices["option"] = choices["option"].astype(str).str.strip()
    # 1->take, 2->leave
    choices["choice_bin"] = choices["choice"].replace({2: 0, 1: 1}).astype(float)

    # merge prop times onto choices
    prop_df = prop.reset_index()
    prop_df = prop_df.merge(revisit_fraction.reset_index(), how="left", on=["game", "trial_number", "option"])
    merged = choices.merge(prop_df, how="left", on=["game", "trial_number", "option"])

    if feature_set not in {"location_interactions", "rank_relevance", "rank_rel_irr"}:
        raise ValueError("feature_set must be one of: location_interactions, rank_relevance, rank_rel_irr")

    # add per-location values and relevance, then build predictors
    rows = []
    for _, r in merged.iterrows():
        g = int(r["game"])
        t = int(r["trial_number"])
        option = str(r["option"]).strip()

        # gather prop times
        props: List[float] = []
        for loc in range(1, 7):
            props.append(float(r.get(f"prop_time_loc{loc}", np.nan)))

        props_arr = np.asarray(props, dtype=float)
        if not np.any(np.isfinite(props_arr)):
            # no item ROI fixations -> cannot define prop_time
            continue

        # build value maps for the 6 items using the location->image mapping
        # values are stable within game
        items_df = game_items.get(g)
        if items_df is None or len(items_df) == 0:
            continue
        outcome_by_image = pd.Series(
            pd.to_numeric(items_df["outcome"], errors="coerce").values,
            index=items_df["image"].astype(str).values,
        ).to_dict()
        rec_vals_map = recalled_map.get(g, {})

        # Build per-location value + relevance arrays
        vals = np.full(6, np.nan, dtype=float)
        rels = np.full(6, np.nan, dtype=float)
        for loc in range(1, 7):
            img = loc_img_map.get((g, loc))
            if img is None:
                continue
            if value_source == "true":
                v = outcome_by_image.get(img, np.nan)
            else:
                v = rec_vals_map.get(img, np.nan)
            rel = 1.0 if option in str(img).split("_") else 0.0
            vals[loc - 1] = float(v) if np.isfinite(v) else np.nan
            rels[loc - 1] = float(rel)

        feats: Dict[str, float] = {}
        # Always include raw per-location props/values/relevance so we can efficiently build
        # permutation-based chance baselines without re-reading fixations.
        for loc in range(1, 7):
            feats[f"prop_time_loc{loc}"] = float(props_arr[loc - 1])
            feats[f"loc{loc}_val"] = float(vals[loc - 1]) if np.isfinite(vals[loc - 1]) else np.nan
            feats[f"loc{loc}_rel"] = float(rels[loc - 1]) if np.isfinite(rels[loc - 1]) else np.nan

        if feature_set == "location_interactions":
            for loc in range(1, 7):
                p = float(props_arr[loc - 1])
                if p == 0.0 or (np.isfinite(p) and abs(p) < 1e-12):
                    # if a location wasn't fixated, its contribution is 0 regardless of unknown mapping
                    feats[f"loc{loc}_pt_x_val"] = 0.0
                    feats[f"loc{loc}_pt_x_rel"] = 0.0
                    feats[f"loc{loc}_pt_x_val_x_rel"] = 0.0
                    continue
                v = vals[loc - 1]
                rel = rels[loc - 1]
                feats[f"loc{loc}_pt_x_val"] = p * float(v) if np.isfinite(v) else np.nan
                feats[f"loc{loc}_pt_x_rel"] = p * float(rel) if np.isfinite(rel) else np.nan
                feats[f"loc{loc}_pt_x_val_x_rel"] = (
                    p * float(v) * float(rel) if (np.isfinite(v) and np.isfinite(rel)) else np.nan
                )
        elif feature_set == "rank_relevance":
            # Rank-based: rank items by value within the trial (1=most positive).
            # Ties share the same rank (dense ranking).
            if not np.all(np.isfinite(vals)) or not np.all(np.isfinite(rels)):
                # cannot define ranks without a complete value ordering
                for rk in range(1, 7):
                    feats[f"rank{rk}_pt"] = np.nan
                    feats[f"rank{rk}_pt_x_rel"] = np.nan
            else:
                uniq = sorted(np.unique(vals), reverse=True)
                rank_of_val = {v: i + 1 for i, v in enumerate(uniq)}
                ranks = np.asarray([rank_of_val[v] for v in vals], dtype=int)

                rank_pt = {rk: 0.0 for rk in range(1, 7)}
                rank_pt_x_rel = {rk: 0.0 for rk in range(1, 7)}
                for loc in range(1, 7):
                    p = float(props_arr[loc - 1])
                    if not np.isfinite(p) or p == 0.0:
                        continue
                    rk = int(ranks[loc - 1])
                    rank_pt[rk] += p
                    rank_pt_x_rel[rk] += p * float(rels[loc - 1])

                for rk in range(1, 7):
                    feats[f"rank{rk}_pt"] = float(rank_pt[rk])
                    feats[f"rank{rk}_pt_x_rel"] = float(rank_pt_x_rel[rk])
        else:
            # Rank-based, clean separation: separate predictors for time on relevant vs irrelevant items
            # within each value rank bin.
            if not np.all(np.isfinite(vals)) or not np.all(np.isfinite(rels)):
                for rk in range(1, 7):
                    feats[f"rank{rk}_pt_rel"] = np.nan
                    feats[f"rank{rk}_pt_irr"] = np.nan
            else:
                uniq = sorted(np.unique(vals), reverse=True)
                rank_of_val = {v: i + 1 for i, v in enumerate(uniq)}
                ranks = np.asarray([rank_of_val[v] for v in vals], dtype=int)

                rank_pt_rel = {rk: 0.0 for rk in range(1, 7)}
                rank_pt_irr = {rk: 0.0 for rk in range(1, 7)}
                for loc in range(1, 7):
                    p = float(props_arr[loc - 1])
                    if not np.isfinite(p) or p == 0.0:
                        continue
                    rk = int(ranks[loc - 1])
                    if float(rels[loc - 1]) >= 0.5:
                        rank_pt_rel[rk] += p
                    else:
                        rank_pt_irr[rk] += p

                for rk in range(1, 7):
                    feats[f"rank{rk}_pt_rel"] = float(rank_pt_rel[rk])
                    feats[f"rank{rk}_pt_irr"] = float(rank_pt_irr[rk])

        row = {
            "subject": subid,
            "game": g,
            "trial_number": t,
            "option": option,
            "choice": float(r["choice_bin"]),
            "revisit_fraction": float(r.get("revisit_fraction", np.nan)),
        }
        row.update(feats)
        rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows)


def _dense_ranks_desc(vals: np.ndarray) -> np.ndarray:
    """Dense ranks in descending order: 1=largest. Ties share rank."""
    uniq = sorted(np.unique(vals), reverse=True)
    rank_of_val = {v: i + 1 for i, v in enumerate(uniq)}
    return np.asarray([rank_of_val[v] for v in vals], dtype=int)


def _features_from_props_vals_rels(
    props: np.ndarray,
    vals: np.ndarray,
    rels: np.ndarray,
    feature_set: str,
) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    if feature_set == "location_interactions":
        for loc in range(1, 7):
            p = float(props[loc - 1])
            v = vals[loc - 1]
            rel = rels[loc - 1]
            if not np.isfinite(p) or p == 0.0:
                feats[f"loc{loc}_pt_x_val"] = 0.0
                feats[f"loc{loc}_pt_x_rel"] = 0.0
                feats[f"loc{loc}_pt_x_val_x_rel"] = 0.0
            else:
                feats[f"loc{loc}_pt_x_val"] = p * float(v) if np.isfinite(v) else np.nan
                feats[f"loc{loc}_pt_x_rel"] = p * float(rel) if np.isfinite(rel) else np.nan
                feats[f"loc{loc}_pt_x_val_x_rel"] = (
                    p * float(v) * float(rel) if (np.isfinite(v) and np.isfinite(rel)) else np.nan
                )
        return feats

    if not (np.all(np.isfinite(vals)) and np.all(np.isfinite(rels)) and np.all(np.isfinite(props))):
        # can't define rank features without complete value+relevance arrays
        if feature_set == "rank_relevance":
            for rk in range(1, 7):
                feats[f"rank{rk}_pt"] = np.nan
                feats[f"rank{rk}_pt_x_rel"] = np.nan
        else:
            for rk in range(1, 7):
                feats[f"rank{rk}_pt_rel"] = np.nan
                feats[f"rank{rk}_pt_irr"] = np.nan
        return feats

    ranks = _dense_ranks_desc(vals)

    if feature_set == "rank_relevance":
        rank_pt = {rk: 0.0 for rk in range(1, 7)}
        rank_pt_x_rel = {rk: 0.0 for rk in range(1, 7)}
        for loc in range(1, 7):
            p = float(props[loc - 1])
            if p == 0.0:
                continue
            rk = int(ranks[loc - 1])
            rank_pt[rk] += p
            rank_pt_x_rel[rk] += p * float(rels[loc - 1])
        for rk in range(1, 7):
            feats[f"rank{rk}_pt"] = float(rank_pt[rk])
            feats[f"rank{rk}_pt_x_rel"] = float(rank_pt_x_rel[rk])
        return feats

    # rank_rel_irr
    rank_pt_rel = {rk: 0.0 for rk in range(1, 7)}
    rank_pt_irr = {rk: 0.0 for rk in range(1, 7)}
    for loc in range(1, 7):
        p = float(props[loc - 1])
        if p == 0.0:
            continue
        rk = int(ranks[loc - 1])
        if float(rels[loc - 1]) >= 0.5:
            rank_pt_rel[rk] += p
        else:
            rank_pt_irr[rk] += p
    for rk in range(1, 7):
        feats[f"rank{rk}_pt_rel"] = float(rank_pt_rel[rk])
        feats[f"rank{rk}_pt_irr"] = float(rank_pt_irr[rk])
    return feats


def _get_x_cols(feature_set: str) -> List[str]:
    if feature_set == "location_interactions":
        cols: List[str] = []
        for loc in range(1, 7):
            cols += [
                f"loc{loc}_pt_x_val",
                f"loc{loc}_pt_x_rel",
                f"loc{loc}_pt_x_val_x_rel",
            ]
        return cols
    if feature_set == "rank_relevance":
        cols = []
        for rk in range(1, 7):
            cols += [
                f"rank{rk}_pt",
                f"rank{rk}_pt_x_rel",
            ]
        return cols
    if feature_set == "rank_rel_irr":
        cols = []
        for rk in range(1, 7):
            cols += [
                f"rank{rk}_pt_rel",
                f"rank{rk}_pt_irr",
            ]
        return cols
    raise ValueError("Unknown feature_set")


def simulate_chance_accuracy(
    df: pd.DataFrame,
    feature_set: str,
    n_sims: int,
    n_folds: int,
    seed: int,
    regularization: str,
) -> Tuple[np.ndarray, float, float, float, float, CoefSummary, np.ndarray]:
    """Permutation-based chance baseline.

    Keeps prop_time across locations fixed per trial, but permutes the mapping of (value, relevance)
    pairs across the 6 locations within each trial. This preserves gaze allocation and the set of
    items/features within the game, while breaking the gaze->value/relevance alignment.
    """
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=["choice"]).copy()
    # We'll handle NaNs in vals/rels inside feature construction and by dropna in CV

    x_cols = _get_x_cols(feature_set)

    # observed scaling (to keep coef units comparable across permutations)
    d_obs = df.dropna(subset=list(x_cols) + ["choice"]).copy()
    X_obs = np.column_stack([pd.to_numeric(d_obs[c], errors="coerce").values.astype(float) for c in x_cols])
    _X_obs_z, mu_obs, sd_obs = _zscore_columns(X_obs)

    # Pre-extract arrays for vectorized permutation (avoids iterrows).
    n_trials = len(d)
    props_all = np.column_stack(
        [pd.to_numeric(d.get(f"prop_time_loc{loc}", np.nan), errors="coerce").values.astype(float) for loc in range(1, 7)]
    )  # (n_trials, 6)
    vals_all = np.column_stack(
        [pd.to_numeric(d.get(f"loc{loc}_val", np.nan), errors="coerce").values.astype(float) for loc in range(1, 7)]
    )
    rels_all = np.column_stack(
        [pd.to_numeric(d.get(f"loc{loc}_rel", np.nan), errors="coerce").values.astype(float) for loc in range(1, 7)]
    )
    choice_all = pd.to_numeric(d["choice"], errors="coerce").values.astype(float)

    sim_acc = np.full(n_sims, np.nan, dtype=float)
    sim_coef = np.full((n_sims, len(x_cols)), np.nan, dtype=float)
    for s in range(n_sims):
        # Generate all permutations at once: argsort of uniform random keys gives
        # a uniform random permutation per row, equivalent to but much faster than
        # a Python-level loop of rng.permutation(6) calls.
        keys = rng.random((n_trials, 6))
        perms = np.argsort(keys, axis=1)  # (n_trials, 6)
        row_idx = np.arange(n_trials)[:, None]
        vals_p = vals_all[row_idx, perms]
        rels_p = rels_all[row_idx, perms]

        if feature_set == "location_interactions":
            # Vectorized feature construction for location_interactions.
            # Zero out prop_time where not finite or zero.
            p_valid = np.where(np.isfinite(props_all) & (props_all != 0.0), props_all, 0.0)

            pt_x_val = p_valid * np.where(np.isfinite(vals_p), vals_p, np.nan)
            pt_x_rel = p_valid * np.where(np.isfinite(rels_p), rels_p, np.nan)
            pt_x_val_x_rel = p_valid * np.where(
                np.isfinite(vals_p) & np.isfinite(rels_p),
                vals_p * rels_p,
                np.nan,
            )
            # Where prop was zero/non-finite, features are 0.
            pt_x_val = np.where(np.isfinite(pt_x_val), pt_x_val, np.nan)
            pt_x_rel = np.where(np.isfinite(pt_x_rel), pt_x_rel, np.nan)
            pt_x_val_x_rel = np.where(np.isfinite(pt_x_val_x_rel), pt_x_val_x_rel, np.nan)
            # But where original prop was zero/non-finite, set to 0 (matching scalar logic).
            zero_mask = ~(np.isfinite(props_all) & (props_all != 0.0))
            pt_x_val[zero_mask] = 0.0
            pt_x_rel[zero_mask] = 0.0
            pt_x_val_x_rel[zero_mask] = 0.0

            # Interleave columns: loc1_pt_x_val, loc1_pt_x_rel, loc1_pt_x_val_x_rel, loc2_...
            X_sim = np.empty((n_trials, 18), dtype=float)
            for loc in range(6):
                X_sim[:, loc * 3] = pt_x_val[:, loc]
                X_sim[:, loc * 3 + 1] = pt_x_rel[:, loc]
                X_sim[:, loc * 3 + 2] = pt_x_val_x_rel[:, loc]

            # Build minimal DataFrame for CV (which expects column names).
            sim_data = {"choice": choice_all}
            for j, col in enumerate(x_cols):
                sim_data[col] = X_sim[:, j]
            sim_df = pd.DataFrame(sim_data)
        else:
            # Fallback for rank-based feature sets: use per-row construction.
            feats_rows = []
            for i in range(n_trials):
                feats = _features_from_props_vals_rels(
                    props_all[i], vals_p[i], rels_p[i], feature_set
                )
                out = {"choice": choice_all[i]}
                out.update(feats)
                feats_rows.append(out)
            sim_df = pd.DataFrame(feats_rows)

        cv = cross_validated_accuracy(
            sim_df,
            x_cols,
            "choice",
            n_folds=n_folds,
            seed=seed + 1000 + s,
            regularization=regularization,
        )
        sim_acc[s] = cv.mean

        sim_fit = sim_df.dropna(subset=list(x_cols) + ["choice"]).copy()
        if len(sim_fit) > 20:
            Xp = np.column_stack([pd.to_numeric(sim_fit[c], errors="coerce").values.astype(float) for c in x_cols])
            yp = pd.to_numeric(sim_fit["choice"], errors="coerce").values.astype(int)
            Xpz = (Xp - mu_obs) / sd_obs
            coef_p = _fit_logistic_coef(Xpz, yp, regularization=regularization)
            if coef_p is not None and len(coef_p) == len(x_cols):
                sim_coef[s, :] = coef_p

    lo = float(np.nanpercentile(sim_acc, 2.5))
    hi = float(np.nanpercentile(sim_acc, 97.5))
    mu = float(np.nanmean(sim_acc))
    p95 = float(np.nanpercentile(sim_acc, 95.0))

    coef_mean = np.nanmean(sim_coef, axis=0)
    coef_lo = np.nanpercentile(sim_coef, 2.5, axis=0)
    coef_hi = np.nanpercentile(sim_coef, 97.5, axis=0)
    null_summary = CoefSummary(coef=coef_mean, lo=coef_lo, hi=coef_hi, feature_names=list(x_cols))

    return sim_acc, mu, lo, hi, p95, null_summary, sim_coef


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict choice from prop-time features. Default uses per-location interactions: prop_time*value, "
            "prop_time*relevance, prop_time*value*relevance for each of 6 item locations (18 predictors). "
            "Alternative feature set bins items by within-trial value rank (ties allowed)."
        )
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="data")
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Optional dataset root containing data/ subfolder. "
            "Useful for NN simulations compiled into a human-like layout, e.g. metarnn/simulations/human_like_<SIM_TAG>. "
            "If provided, overrides --data-root/--output-root to <root>/data."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Number of subject-level bootstrap resamples for coefficient CIs.",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=0,
        help="Number of chance-baseline simulations (0 disables).",
    )
    parser.add_argument(
        "--value-source",
        choices=["true", "recalled", "both"],
        default="both",
        help="Whether to use encoded outcomes, recalled values, or run both.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["location_interactions", "rank_relevance", "rank_rel_irr"],
        default="location_interactions",
        help=(
            "Which predictors to use. location_interactions: 18 per-location interaction features. "
            "rank_relevance: 12 features based on within-trial value rank bins (rank_pt and rank_pt_x_rel for ranks 1..6). "
            "rank_rel_irr: 12 features splitting rank-binned time into relevant vs irrelevant (rank_pt_rel and rank_pt_irr for ranks 1..6)."
        ),
    )
    parser.add_argument(
        "--regularization",
        choices=["l2", "none"],
        default="l2",
        help="Whether to use L2 (ridge) regularization or unregularized logistic regression.",
    )
    parser.add_argument(
        "--visit-type",
        choices=["all", "first", "revisit", "first_n", "compare", "first_n_compare"],
        default="all",
        help=(
            "Which fixation-time component to use when computing per-location prop_time within each trial. "
            "all: all fixations (current behavior). first: only the first fixation to each location within the trial. "
            "revisit: only revisits (all subsequent fixations to a location after its first fixation in that trial). "
            "first_n: only the first N fixations within the trial (across all locations; requires --first-n-fixations). "
            "compare: run all/first/revisit and generate comparison plots. "
            "first_n_compare: run first_n for N in --first-n-values and generate comparison plots."
        ),
    )
    parser.add_argument(
        "--first-n-fixations",
        type=int,
        default=None,
        help=(
            "Used only when --visit-type first_n. Keeps only the first N fixations (sorted by fix_start) "
            "within each choice trial (across all locations) when computing prop_time_loc*."
        ),
    )
    parser.add_argument(
        "--first-n-values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help=(
            "Used only when --visit-type first_n_compare. List of N values to run (e.g., 1 2 3 4 5 6)."
        ),
    )
    parser.add_argument(
        "--visit-normalization",
        choices=["within", "total"],
        default="within",
        help=(
            "How to normalize prop_time when using --visit-type first or revisit. "
            "within: divide by total FIRST (or REVISIT) time, so prop_time sums to 1 within that component (may drop trials with zero revisits). "
            "total: divide by total fixation time (all item-ROI fixations), so first+revisit fractions are on the same scale and revisit-only keeps trials with zero revisits (all zeros)."
        ),
    )
    parser.add_argument(
        "--drop-frac",
        type=float,
        default=0.0,
        help="Fraction of fixations to randomly drop before computing prop_time (0.0 = keep all).",
    )
    parser.add_argument(
        "--drop-seed",
        type=int,
        default=0,
        help="RNG seed for fixation dropping (only used when --drop-frac > 0).",
    )
    args = parser.parse_args()

    # Allow running on NN 'human_like' roots that contain their own data/ and output/ folders.
    default_out_dir = "figures"
    if args.root:
        args.data_root = os.path.join(args.root, "data")
        args.output_root = os.path.join(args.root, "data")
        if args.out_dir == default_out_dir:
            args.out_dir = os.path.join(args.root, default_out_dir)

    if args.visit_type == "first_n":
        if args.first_n_fixations is None or int(args.first_n_fixations) <= 0:
            raise ValueError("--first-n-fixations must be a positive integer when --visit-type first_n")
    if args.visit_type == "first_n_compare":
        if not args.first_n_values or any(int(n) <= 0 for n in args.first_n_values):
            raise ValueError("--first-n-values must be one or more positive integers when --visit-type first_n_compare")

    subs = [s for s in list_subjects(args.data_root) if s not in EYETRACK_EXCLUDE_SUBJECTS]

    # NN simulations typically lack value-recall transcripts. When recalled/both is specified,
    # fall back to true values if no valuerecall files are present.
    if args.value_source in {"recalled", "both"}:
        if not _has_any_valuerecall(args.data_root, subs):
            print(
                "No valuerecall files found under data/. Falling back to --value-source true "
                "(useful for NN simulations)."
            )
            args.value_source = "true"

    def _run_one(
        value_source: str,
        visit_type: str,
        first_n_fixations: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, CVSummary, CoefDist, Optional[ChanceAccSummary], Optional[np.ndarray], Optional[np.ndarray]]:
        dfs = []
        used = []
        for sub in subs:
            df_sub = build_prop_time_interaction_dataset(
                sub,
                data_root=args.data_root,
                output_root=args.output_root,
                value_source=value_source,
                feature_set=args.feature_set,
                visit_type=visit_type,
                visit_normalization=args.visit_normalization,
                first_n_fixations=first_n_fixations if visit_type == "first_n" else args.first_n_fixations,
                drop_frac=args.drop_frac,
                drop_seed=args.drop_seed,
            )
            if df_sub is None or len(df_sub) == 0:
                continue
            dfs.append(df_sub)
            used.append(sub)

        if not dfs:
            raise RuntimeError(f"No usable data for value_source={value_source}")

        df = pd.concat(dfs, ignore_index=True)
        x_cols = _get_x_cols(args.feature_set)

        cv = cross_validated_accuracy(
            df,
            x_cols,
            "choice",
            n_folds=args.n_folds,
            seed=args.seed,
            regularization=args.regularization,
        )

        coef_dist = bootstrap_coef_summary(
            df,
            x_cols=x_cols,
            y_col="choice",
            cluster_col="subject",
            n_boot=args.n_bootstrap,
            seed=args.seed,
            regularization=args.regularization,
            return_samples=True,
        )
        assert isinstance(coef_dist, CoefDist)
        coef_sum = coef_dist.summary

        null_coef_samples = None
        sim_acc_samps = None
        chance_summary = None
        if args.n_sims and args.n_sims > 0:
            sim_acc, mu, lo, hi, p95, null_coef, null_coef_samples = simulate_chance_accuracy(
                df,
                feature_set=args.feature_set,
                n_sims=args.n_sims,
                n_folds=args.n_folds,
                seed=args.seed,
                regularization=args.regularization,
            )
            sim_acc_samps = sim_acc
            # one-sided permutation p-value (proportion of permuted accuracies
            # at or above the observed accuracy; uncorrected empirical estimate)
            obs = cv.mean
            p_perm = float(np.mean(sim_acc >= obs))
            sd = float(np.nanstd(sim_acc))
            chance_summary = ChanceAccSummary(mean=mu, sd=sd, lo=lo, hi=hi, p95=p95, p_perm=p_perm)

        # Cache a compact summary CSV for downstream composite figures.
        summary_rows: List[dict] = []
        base_row = {
            "value_source": str(value_source),
            "feature_set": str(args.feature_set),
            "visit_type": str(visit_type),
            "visit_normalization": str(args.visit_normalization),
            "regularization": str(args.regularization),
            "n_folds": int(args.n_folds),
            "seed": int(args.seed),
            "n_bootstrap": int(args.n_bootstrap),
            "n_sims": int(args.n_sims),
            "n_subjects": int(len(set(df["subject"]))),
            "n_rows_total": int(len(df)),
            "n_rows_cv": int(cv.n_rows),
            "cv_mean": float(cv.mean),
            "cv_sem": float(cv.sem),
        }
        if chance_summary is not None:
            base_row.update(
                {
                    "perm_mean": float(chance_summary.mean),
                    "perm_sd": float(chance_summary.sd),
                    "perm_lo": float(chance_summary.lo),
                    "perm_hi": float(chance_summary.hi),
                    "perm_p95": float(chance_summary.p95),
                    "perm_p": float(chance_summary.p_perm),
                }
            )

        if args.feature_set == "location_interactions":
            irr, rel = _derived_reward_effects_location_interactions(coef_dist=coef_dist)
            summary_rows.append(
                {
                    **base_row,
                    "term": "Irrelevant x Reward",
                    "coef_mean": float(irr["mean"]),
                    "coef_lo": float(irr["lo"]),
                    "coef_hi": float(irr["hi"]),
                }
            )
            summary_rows.append(
                {
                    **base_row,
                    "term": "Relevant x Reward",
                    "coef_mean": float(rel["mean"]),
                    "coef_lo": float(rel["lo"]),
                    "coef_hi": float(rel["hi"]),
                }
            )
        else:
            summary_rows.append(
                {
                    **base_row,
                    "term": "(not_available)",
                    "coef_mean": float("nan"),
                    "coef_lo": float("nan"),
                    "coef_hi": float("nan"),
                }
            )

        vt_tag = visit_type
        if visit_type == "first_n":
            n_tag = int(first_n_fixations) if first_n_fixations is not None else int(args.first_n_fixations)
            vt_tag = f"first_n{n_tag}"

        summary_path = os.path.join(
            args.out_dir,
            f"summary_prop_time_{args.feature_set}_{value_source}_{vt_tag}_norm-{args.visit_normalization}.csv",
        )
        _write_summary_csv(summary_path, summary_rows)
        print(f"Saved summary CSV: {summary_path}")

        if sim_acc_samps is not None:
            perm_null_path = os.path.join(
                args.out_dir,
                f"perm_null_prop_time_{args.feature_set}_{value_source}_{vt_tag}_norm-{args.visit_normalization}.csv",
            )
            pd.DataFrame({
                "perm": np.arange(len(sim_acc_samps), dtype=int),
                "cv_accuracy": np.asarray(sim_acc_samps, dtype=float),
            }).to_csv(perm_null_path, index=False)
            print(f"Saved permutation null CSV: {perm_null_path}")

        coef_table_path = os.path.join(
            args.out_dir,
            f"coef_table_prop_time_{args.feature_set}_{value_source}_{vt_tag}_norm-{args.visit_normalization}.csv",
        )
        _write_coef_table_csv(
            coef_table_path,
            coef_summary=coef_sum,
            meta={
                "value_source": str(value_source),
                "feature_set": str(args.feature_set),
                "visit_type": str(visit_type),
                "visit_normalization": str(args.visit_normalization),
                "regularization": str(args.regularization),
                "n_folds": int(args.n_folds),
                "seed": int(args.seed),
                "n_bootstrap": int(args.n_bootstrap),
                "n_sims": int(args.n_sims),
            },
        )
        print(f"Saved coef table CSV: {coef_table_path}")

        header_vt = visit_type
        if visit_type == "first_n":
            n_head = int(first_n_fixations) if first_n_fixations is not None else int(args.first_n_fixations)
            header_vt = f"first_n (N={n_head})"
        print(f"\n=== value_source={value_source} | visit_type={header_vt} | visit_norm={args.visit_normalization} ===")
        print(f"Subjects used (n={len(set(df['subject']))}): {', '.join(sorted(set(used)))}")
        print(f"Rows total: {len(df)}")
        print(f"Rows used in CV (after dropna): {cv.n_rows}")
        print(f"Mean {args.n_folds}-fold CV accuracy: {cv.mean:.4f} (SEM {cv.sem:.4f})")

        # Quantify first-vs-revisit at a descriptive level (based on all fixations)
        if "revisit_fraction" in df.columns:
            sub_means = (
                df.dropna(subset=["revisit_fraction"])\
                  .groupby("subject", sort=False)["revisit_fraction"]\
                  .mean()
            )
            if len(sub_means) > 0:
                grand_mean = float(np.nanmean(sub_means.values))
                grand_sem = _safe_sem(sub_means.values)
                print(
                    f"Mean revisit fraction (per-subject mean across trials): {grand_mean:.4f} (SEM {grand_sem:.4f})"
                )

        if args.n_sims and args.n_sims > 0:
            print(f"Chance baseline (permute item identity across locations): mean {mu:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")
            print(f"Permutation 95th percentile (one-sided): {p95:.4f}")
            print(f"Permutation p (acc >= observed): {p_perm:.4g}")
        return df, cv, coef_dist, chance_summary, null_coef_samples, sim_acc_samps

    def _run_value_source(value_source: str):
        if args.visit_type not in {"compare", "first_n_compare"}:
            if args.visit_type == "first_n":
                _run_one(value_source, "first_n", first_n_fixations=int(args.first_n_fixations))
            else:
                _run_one(value_source, args.visit_type)
            return

        # compare modes: (a) all/first/revisit or (b) first_n sweep
        if args.feature_set != "location_interactions":
            raise ValueError("Compare modes currently only support --feature-set location_interactions")

        if args.visit_type == "compare":
            visit_types = ["all", "first", "revisit"]
            cv_by_vt: Dict[str, CVSummary] = {}
            chance_by_vt: Dict[str, Optional[ChanceAccSummary]] = {}
            coef_by_vt: Dict[str, CoefSummary] = {}
            perm_acc_by_vt: Dict[str, np.ndarray] = {}

            for vt in visit_types:
                _df, cv, coef_dist, chance_sum, _null_samples, sim_acc = _run_one(value_source, vt)
                cv_by_vt[vt] = cv
                chance_by_vt[vt] = chance_sum
                coef_by_vt[vt] = coef_dist.summary
                if sim_acc is not None:
                    perm_acc_by_vt[vt] = sim_acc

            acc_out = os.path.join(
                args.out_dir,
                f"Figure_prop_time_{args.feature_set}_visit_type_accuracy_{value_source}_norm-{args.visit_normalization}.pdf",
            )
            plot_visit_type_accuracy_comparison(
                observed=cv_by_vt,
                chance=chance_by_vt,
                out_path=acc_out,
                title=f"CV accuracy by visit type ({value_source}; norm={args.visit_normalization})",
            )
            print(f"Saved visit-type accuracy plot: {acc_out}")

            acc_delta_out = os.path.join(
                args.out_dir,
                f"Figure_prop_time_{args.feature_set}_visit_type_accuracy_delta_from_p95_{value_source}_norm-{args.visit_normalization}.pdf",
            )
            plot_visit_type_accuracy_delta_from_p95(
                observed=cv_by_vt,
                chance=chance_by_vt,
                out_path=acc_delta_out,
                title=f"CV accuracy − perm 95th percentile ({value_source}; norm={args.visit_normalization})",
            )
            print(f"Saved visit-type accuracy delta plot: {acc_delta_out}")

            coef_out = os.path.join(
                args.out_dir,
                f"Figure_prop_time_{args.feature_set}_visit_type_coefs_{value_source}_norm-{args.visit_normalization}.pdf",
            )
            plot_location_coef_panels_by_visit_type(
                coef_by_visit=coef_by_vt,
                value_source=value_source,
                out_path=coef_out,
                title=f"Coefficients by visit type ({value_source}; norm={args.visit_normalization})",
            )
            print(f"Saved visit-type coef plot: {coef_out}")

            # Permutation distribution panels (requires permutations)
            if len(perm_acc_by_vt) > 0:
                perm_out = os.path.join(
                    args.out_dir,
                    f"Figure_prop_time_{args.feature_set}_visit_type_perm_accuracy_distributions_{value_source}_norm-{args.visit_normalization}.pdf",
                )
                plot_visit_type_permutation_accuracy_distributions(
                    perm_acc=perm_acc_by_vt,
                    observed=cv_by_vt,
                    chance=chance_by_vt,
                    out_path=perm_out,
                    title=f"Permutation accuracy distributions ({value_source}; norm={args.visit_normalization})",
                )
                print(f"Saved visit-type permutation distributions plot: {perm_out}")
            return

        # first_n_compare mode
        ns = [int(n) for n in args.first_n_values]
        ns = sorted(list(dict.fromkeys(ns)))  # unique, preserve order then sort

        cv_by_n: Dict[int, CVSummary] = {}
        chance_by_n: Dict[int, Optional[ChanceAccSummary]] = {}
        coef_by_n: Dict[int, CoefSummary] = {}
        perm_acc_by_n: Dict[int, np.ndarray] = {}

        for n in ns:
            _df, cv, coef_dist, chance_sum, _null_samples, sim_acc = _run_one(value_source, "first_n", first_n_fixations=n)
            cv_by_n[n] = cv
            chance_by_n[n] = chance_sum
            coef_by_n[n] = coef_dist.summary
            if sim_acc is not None:
                perm_acc_by_n[n] = sim_acc

        acc_out = os.path.join(
            args.out_dir,
            f"Figure_prop_time_{args.feature_set}_first_n_accuracy_{value_source}_norm-{args.visit_normalization}.pdf",
        )
        plot_first_n_accuracy_comparison(
            observed=cv_by_n,
            chance=chance_by_n,
            out_path=acc_out,
            title=f"CV accuracy by first N fixations ({value_source}; norm={args.visit_normalization})",
        )
        print(f"Saved first-N accuracy plot: {acc_out}")

        coef_out = os.path.join(
            args.out_dir,
            f"Figure_prop_time_{args.feature_set}_first_n_coefs_{value_source}_norm-{args.visit_normalization}.pdf",
        )
        plot_location_coef_panels_by_first_n(
            coef_by_n=coef_by_n,
            value_source=value_source,
            out_path=coef_out,
            title=f"Coefficients by first N fixations ({value_source}; norm={args.visit_normalization})",
        )
        print(f"Saved first-N coef plot: {coef_out}")

        if len(perm_acc_by_n) > 0:
            perm_out = os.path.join(
                args.out_dir,
                f"Figure_prop_time_{args.feature_set}_first_n_perm_accuracy_distributions_{value_source}_norm-{args.visit_normalization}.pdf",
            )
            plot_first_n_permutation_accuracy_distributions(
                perm_acc=perm_acc_by_n,
                observed=cv_by_n,
                chance=chance_by_n,
                out_path=perm_out,
                title=f"Permutation accuracy distributions by first N fixations ({value_source}; norm={args.visit_normalization})",
            )
            print(f"Saved first-N permutation distributions plot: {perm_out}")
        return

    if args.value_source in {"true", "both"}:
        _run_value_source("true")
    if args.value_source in {"recalled", "both"}:
        _run_value_source("recalled")


if __name__ == "__main__":
    main()

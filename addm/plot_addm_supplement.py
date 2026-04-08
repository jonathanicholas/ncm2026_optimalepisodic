"""aDDM Supplement Figure.

Combines PPC, fit parameters, and parameter recovery into a single
2-row x 3-column supplementary figure (18 x 12 inches).

Layout:
  Row 1: [PPC (choice | RT | LL bar)] [Fit parameter bars (theta | d | sigma)]
  Row 2: [d recovery] [theta recovery] [sigma recovery]

Data variants:
  - PPC and fit params use rtTrans
  - Parameter recovery uses fix

Run from the repository root:
    python -m addm.plot_addm_supplement
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _sem(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size <= 1:
        return 0.0
    return float(np.std(v, ddof=1) / np.sqrt(v.size))


def _normalize_subject_id(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def _identity_limits(x: np.ndarray, y: np.ndarray, pad: float = 0.05) -> Tuple[float, float]:
    vals = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    span = hi - lo
    lo -= pad * span
    hi += pad * span
    return lo, hi


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _zscore(series: pd.Series) -> pd.Series:
    mu = float(series.mean())
    sd = float(series.std(ddof=0))
    if sd <= 0:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sd


# ---------------------------------------------------------------------------
# PPC quintile panel
# ---------------------------------------------------------------------------


def _plot_ppc_quintile_panel(
    ax: plt.Axes,
    *,
    human_subj: pd.DataFrame,
    sim_addm_subj: pd.DataFrame,
    sim_ddm_subj: pd.DataFrame,
    y_col: str,
    y_label: str,
    ylim: Tuple[float, float],
    show_legend: bool,
    rng: np.random.Generator,
) -> None:
    """Draw one PPC quintile panel: bars for data, lines for simulations."""
    bin_labels = ["1", "2", "3", "4", "5"]
    x_pos = np.arange(5)
    bar_width = 1.0

    # --- Human data: bars + subject dots ---
    grp_means, grp_ses = [], []
    for bi, bl in enumerate(bin_labels):
        vals = human_subj.loc[human_subj["bin"] == bl, y_col].values.astype(float)
        m_v = float(np.nanmean(vals))
        se_v = float(np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals)))) if len(vals) > 1 else 0.0
        grp_means.append(m_v)
        grp_ses.append(se_v)

        # Subject dots
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        for vi, v in enumerate(vals):
            ax.scatter(
                x_pos[bi] + jitter[vi], v,
                s=6**2, facecolor=(1, 1, 1, 0.5), edgecolor="0.5",
                linewidth=1, zorder=3,
            )

    grp_means = np.array(grp_means)
    grp_ses = np.array(grp_ses)

    # Color fill bar
    ax.bar(x_pos, grp_means, bar_width,
           color="0.5", edgecolor="none", linewidth=0, zorder=2)
    # Black outline bar
    ax.bar(x_pos, grp_means, bar_width,
           color="none", edgecolor="black", linewidth=2.5, zorder=4)
    # SEM error bars
    ax.errorbar(x_pos, grp_means, yerr=grp_ses,
                fmt="none", ecolor="black", capsize=0, linewidth=2.5, zorder=5)

    # --- Simulation data: connected lines ---
    for sim_subj, color, label in [
        (sim_addm_subj, "C0", "aDDM"),
        (sim_ddm_subj, "C1", "DDM"),
    ]:
        sim_means, sim_ses = [], []
        for bl in bin_labels:
            vals = sim_subj.loc[sim_subj["bin"] == bl, y_col].values.astype(float)
            sim_means.append(float(np.nanmean(vals)) if len(vals) > 0 else float("nan"))
            sim_ses.append(_sem(vals) if len(vals) > 1 else 0.0)
        sim_means = np.array(sim_means)
        sim_ses = np.array(sim_ses)

        mask = np.isfinite(sim_means)
        ax.plot(x_pos[mask], sim_means[mask], color=color, linewidth=3, zorder=6)
        if np.any(mask & np.isfinite(sim_ses)):
            ax.fill_between(
                x_pos, sim_means - sim_ses, sim_means + sim_ses,
                where=mask & np.isfinite(sim_ses),
                color=color, alpha=0.18, linewidth=0, zorder=1,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.set_ylim(*ylim)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show_legend:
        handles = [
            Patch(facecolor="0.5", edgecolor="black", linewidth=1.5, label="Data"),
            Line2D([0], [0], color="C0", linewidth=3, label="aDDM"),
            Line2D([0], [0], color="C1", linewidth=3, label="DDM"),
        ]
        ax.legend(handles=handles, frameon=True, facecolor="white",
                  edgecolor="none", fontsize=16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="aDDM Supplement Figure.")
    parser.add_argument("--out-dir", type=str, default="output/addm")
    parser.add_argument("--exclude", nargs="*", default=["107", "131"])
    args = parser.parse_args()

    cwd = Path(".").resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = {str(x).strip() for x in args.exclude}

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    # PPC data (rtTrans)
    trial = pd.read_csv(cwd / "output/addm/ppc/trial_level_recalled_offer_choice_rt.csv")
    trial["subject_id"] = _normalize_subject_id(trial["subject_id"])
    trial = trial[~trial["subject_id"].isin(exclude)].copy()

    n_subj = int(trial["subject_id"].nunique())
    if n_subj != 41:
        raise ValueError(f"Expected N=41 eyetracking subjects after exclusions; found N={n_subj}.")
    print(f"N subjects: {n_subj}")

    sim_addm = pd.read_csv(
        cwd / "output/addm/ppc/addm_trialsim_rtTrans_recalled_final_addm_kfold_fit_summary_free3-rtTrans_nsim1000_seed123_dt1.csv"
    )
    sim_addm["subject_id"] = _normalize_subject_id(sim_addm["subject_id"])
    sim_addm = sim_addm[~sim_addm["subject_id"].isin(exclude)].copy()
    sim_addm["time_s_sim_mean"] = pd.to_numeric(sim_addm["time_ms_sim_mean"], errors="coerce") / 1000.0

    sim_ddm = pd.read_csv(
        cwd / "output/addm/ppc/addm_trialsim_rtTrans_recalled_final_addm_kfold_fit_summary_theta1-rtTrans_nsim1000_seed123_dt1.csv"
    )
    sim_ddm["subject_id"] = _normalize_subject_id(sim_ddm["subject_id"])
    sim_ddm = sim_ddm[~sim_ddm["subject_id"].isin(exclude)].copy()
    sim_ddm["time_s_sim_mean"] = pd.to_numeric(sim_ddm["time_ms_sim_mean"], errors="coerce") / 1000.0

    # LL comparison (rtTrans)
    ll_df = pd.read_csv(cwd / "output/addm/kfold_compare/rtTrans_recalled_final/cv_compare_by_game_wide.csv")

    # Fit parameters (rtTrans)
    fit_df = pd.read_csv(cwd / "output/addm/kfold/rtTrans_recalled_final/addm_kfold_fit_summary_free3-rtTrans.csv")

    # Parameter recovery (fix)
    sweep_df = pd.read_csv(cwd / "output/addm/parameter_recovery_sweep/fix_recalled_final/sweep_runs.csv")

    # ------------------------------------------------------------------
    # Quintile binning
    # ------------------------------------------------------------------

    # Human data: z-score and bin into 5 quintiles
    d_h = trial.dropna(subset=["recalled_offer_value", "rt_s", "accept"]).copy()
    d_h = d_h[d_h["rt_s"] > 0].copy()
    d_h["z_val"] = _zscore(d_h["recalled_offer_value"])
    d_h["bin"], bin_edges = pd.qcut(d_h["z_val"], q=5, labels=["1", "2", "3", "4", "5"], retbins=True, duplicates="drop")

    # Map z-score bin edges back to raw recalled_offer_value edges
    rov = d_h["recalled_offer_value"]
    z_val = d_h["z_val"]
    mu_z = float(rov.mean())
    sd_z = float(rov.std(ddof=0))
    raw_edges = mu_z + bin_edges * sd_z

    # Compute per-subject bin means for human data
    human_subj_choice = d_h.groupby(["subject_id", "bin"], observed=True)["accept"].mean().reset_index()
    human_subj_rt = d_h.groupby(["subject_id", "bin"], observed=True)["rt_s"].mean().reset_index()

    # Bin simulation data using same raw edges
    for sim_df in [sim_addm, sim_ddm]:
        sim_df["bin"] = pd.cut(
            sim_df["v_offer"],
            bins=[-np.inf] + list(raw_edges[1:-1]) + [np.inf],
            labels=["1", "2", "3", "4", "5"],
        )

    # Compute per-subject bin means for simulation data
    sim_addm_subj_choice = sim_addm.dropna(subset=["bin"]).groupby(
        ["subject_id", "bin"], observed=True
    )["accept_sim_mean"].mean().reset_index().rename(columns={"accept_sim_mean": "accept"})

    sim_addm_subj_rt = sim_addm.dropna(subset=["bin"]).groupby(
        ["subject_id", "bin"], observed=True
    )["time_s_sim_mean"].mean().reset_index().rename(columns={"time_s_sim_mean": "rt_s"})

    sim_ddm_subj_choice = sim_ddm.dropna(subset=["bin"]).groupby(
        ["subject_id", "bin"], observed=True
    )["accept_sim_mean"].mean().reset_index().rename(columns={"accept_sim_mean": "accept"})

    sim_ddm_subj_rt = sim_ddm.dropna(subset=["bin"]).groupby(
        ["subject_id", "bin"], observed=True
    )["time_s_sim_mean"].mean().reset_index().rename(columns={"time_s_sim_mean": "rt_s"})

    # ------------------------------------------------------------------
    # LL delta
    # ------------------------------------------------------------------
    delta = (
        pd.to_numeric(ll_df["free3-rtTrans"], errors="coerce")
        - pd.to_numeric(ll_df["theta1-rtTrans"], errors="coerce")
    )
    delta = delta[np.isfinite(delta)]
    ll_mean = float(np.mean(delta.values))
    ll_sem_val = _sem(delta.values)

    # ------------------------------------------------------------------
    # Fit parameter summaries
    # ------------------------------------------------------------------
    fit_rows = fit_df[fit_df["mode"].astype(str) == "fit"].copy() if "mode" in fit_df.columns else fit_df.copy()
    param_summaries = []
    for p, label in [("theta", r"$\theta$"), ("d", "d"), ("sigma", r"$\sigma$")]:
        vals = pd.to_numeric(fit_rows[p], errors="coerce").dropna().values
        param_summaries.append({
            "param": p,
            "label": label,
            "mean": float(np.mean(vals)),
            "sem": _sem(vals),
        })

    # ------------------------------------------------------------------
    # Parameter recovery aggregation
    # ------------------------------------------------------------------
    sweep_cols = ["d_true", "theta_true", "sigma_true", "d_hat", "theta_hat", "sigma_hat"]
    sweep_agg = sweep_df.groupby("combo", dropna=False)[sweep_cols].mean().reset_index()

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    sns.set_context("poster")
    sns.set_style("ticks")
    with plt.rc_context({
        "font.family": "Arial",
        "axes.titlesize": 24,
        "axes.labelsize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
    }):
        fig = plt.figure(figsize=(23, 12))

        # Outer grid: 2 rows
        gs_outer = fig.add_gridspec(2, 1, hspace=0.35)

        # Row 1: PPC (2/3 width) | Fit params (1/3 width)
        gs_row1 = gs_outer[0].subgridspec(1, 2, width_ratios=[2, 1], wspace=0.15)

        # PPC subpanels: choice | RT | LL bar
        gs_ppc = gs_row1[0].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.67], wspace=0.55)
        ax_choice = fig.add_subplot(gs_ppc[0])
        ax_rt = fig.add_subplot(gs_ppc[1])
        ax_ll = fig.add_subplot(gs_ppc[2])

        # Fit parameter sub-axes
        gs_params = gs_row1[1].subgridspec(1, 3, wspace=0.8)
        ax_theta = fig.add_subplot(gs_params[0])
        ax_d = fig.add_subplot(gs_params[1])
        ax_sigma = fig.add_subplot(gs_params[2])

        # Row 2: 3 recovery panels
        gs_row2 = gs_outer[1].subgridspec(1, 3, wspace=0.3)
        ax_rec_d = fig.add_subplot(gs_row2[0])
        ax_rec_theta = fig.add_subplot(gs_row2[1])
        ax_rec_sigma = fig.add_subplot(gs_row2[2])

        # --- PPC: Proportion Offers Taken ---
        rng = np.random.default_rng(42)
        _plot_ppc_quintile_panel(
            ax_choice,
            human_subj=human_subj_choice,
            sim_addm_subj=sim_addm_subj_choice,
            sim_ddm_subj=sim_ddm_subj_choice,
            y_col="accept",
            y_label="Proportion Offers Taken",
            ylim=(0, 1.05),
            show_legend=True,
            rng=rng,
        )
        ax_choice.set_yticks([0.0, 0.5, 1.0])
        ax_choice.set_yticklabels(["0", "0.5", "1"])

        # --- PPC: Response Time ---
        _plot_ppc_quintile_panel(
            ax_rt,
            human_subj=human_subj_rt,
            sim_addm_subj=sim_addm_subj_rt,
            sim_ddm_subj=sim_ddm_subj_rt,
            y_col="rt_s",
            y_label="Response Time (s)",
            ylim=(0, 15),
            show_legend=False,
            rng=rng,
        )
        ax_rt.set_yticks([0.0, 5.0, 10.0, 15.0])
        ax_rt.set_yticklabels(["0", "5", "10", "15"])

        # --- PPC: LL difference bar ---
        ax_ll.bar(
            [0], [ll_mean], width=0.55,
            color="0.5", edgecolor="black", linewidth=2, zorder=1,
        )
        ax_ll.errorbar(
            [0], [ll_mean], yerr=[ll_sem_val],
            fmt="_", markersize=16, color="black", linewidth=1.3, capsize=0, zorder=3,
        )
        ax_ll.axhline(0.0, linestyle="--", color="0.25", linewidth=1)
        ax_ll.set_xlim(-0.8, 0.8)
        ax_ll.set_xticks([])
        ax_ll.set_ylabel(r"$\Delta$ Held-out LL (aDDM $-$ DDM)", fontsize=18)
        ax_ll.set_ylim(0, 20)
        ax_ll.set_yticks([0, 10, 20])
        ax_ll.grid(False)
        ax_ll.spines["right"].set_visible(False)
        ax_ll.spines["top"].set_visible(False)

        # --- Fit parameter bars ---
        for ax_p, row in zip([ax_theta, ax_d, ax_sigma], param_summaries):
            ax_p.bar(
                [0], [row["mean"]], width=0.55,
                color="0.5", edgecolor="black", linewidth=2, zorder=1,
            )
            ax_p.errorbar(
                [0], [row["mean"]], yerr=[row["sem"]],
                fmt="_", markersize=16, color="black", linewidth=1.3, capsize=0, zorder=3,
            )
            ax_p.axhline(0.0, linestyle="--", color="0.25", linewidth=1)
            ax_p.set_xlim(-0.8, 0.8)
            ax_p.set_xticks([])
            ax_p.set_ylabel(row["label"])
            ax_p.spines["right"].set_visible(False)
            ax_p.spines["top"].set_visible(False)

            # Scientific notation for d and sigma
            if row["param"] in ("d", "sigma"):
                ax_p.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                ax_p.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # --- Parameter recovery scatter plots ---
        recovery_params = [
            (ax_rec_d, "d", r"$d$", r"$\hat{d}$"),
            (ax_rec_theta, "theta", r"$\theta$", r"$\hat{\theta}$"),
            (ax_rec_sigma, "sigma", r"$\sigma$", r"$\hat{\sigma}$"),
        ]
        for ax_r, p, true_label, hat_label in recovery_params:
            x = sweep_agg[f"{p}_true"].to_numpy(dtype=float)
            y = sweep_agg[f"{p}_hat"].to_numpy(dtype=float)

            ax_r.scatter(x, y, s=28, alpha=0.7, color="black", edgecolors="none", zorder=2)

            lo, hi = _identity_limits(x, y)
            ax_r.plot([lo, hi], [lo, hi], color="black", linewidth=2.5, zorder=1)
            ax_r.set_xlim(lo, hi)
            ax_r.set_ylim(lo, hi)

            r = _corr(x, y)
            ax_r.text(
                0.05, 0.95, f"r = {r:.2f}",
                transform=ax_r.transAxes, fontsize=28,
                va="top", ha="left",
            )

            ax_r.set_xlabel(true_label)
            ax_r.set_ylabel(hat_label)
            ax_r.grid(False)
            ax_r.spines["right"].set_visible(False)
            ax_r.spines["top"].set_visible(False)

            # Scientific notation for d axes
            if p == "d":
                ax_r.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                ax_r.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                ax_r.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

        # Shared x-label for the choice and RT PPC panels
        fig.canvas.draw()
        choice_pos = ax_choice.get_position()
        rt_pos = ax_rt.get_position()
        x_center = (choice_pos.x0 + rt_pos.x1) / 2
        y_below = min(choice_pos.y0, rt_pos.y0) - 0.04
        fig.text(
            x_center, y_below,
            "Recalled Offer Value Quintile",
            ha="center", va="top", fontsize=28,
        )

        out_pdf = out_dir / "FigureS3.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        supp_dir = Path.cwd() / "output" / "figures" / "supplementary"
        supp_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(supp_dir / "FigureS3.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()

"""K-fold cross-validation for the adapted aDDM.

This is a trial-level K-fold alternative to leave-one-game-out (LOGO).

Design
------
- We split *trials* into K folds (default K=10).
- For each fold k:
  - Build likelihood components (V_offer bins, gaze stats, time-bin edges) on
    training trials only (and fixations belonging to those trials).
  - Fit parameters on the training set (mode=fit) using pyBADS.
  - Evaluate held-out log-likelihood on the test fold.

Fold assignment
---------------
By default we assign folds *within subject* (stratified by subject), so every
fold contains data from every participant (roughly balanced). Alternative
schemes are provided via --split-by.

Two modes
---------
1) fit (default): fit parameters separately for each fold using pyBADS.
2) eval-only: skip fitting and only evaluate held-out log-likelihood for a
   fixed parameter set.

Example
--------
Full 10-fold:
    python -m addm.lib.run_kfold_cv --output-dir output --n-folds 10 --n-sim-per-vbin 500 --setting free3-rtTrans

"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .adapted_addm_simulation import ADDMParameters
    from .addm_fitting import (
        BinningConfig,
        FitConfig,
        SimulationConfig,
        build_fitting_components,
        compute_binned_loglik_given_components,
        fit_one_fold_pybads,
        load_group_fixations,
        load_group_trial_templates,
        sigma_from_mu,
    )
except ImportError:  # pragma: no cover
    from adapted_addm_simulation import ADDMParameters
    from addm_fitting import (
        BinningConfig,
        FitConfig,
        SimulationConfig,
        build_fitting_components,
        compute_binned_loglik_given_components,
        fit_one_fold_pybads,
        load_group_fixations,
        load_group_trial_templates,
        sigma_from_mu,
    )


# ---- multiprocessing globals (initialized per worker) ----
_G: Dict[str, Any] = {}


def _assign_folds(
    df_trials: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
    split_by: str,
) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError("--n-folds must be >= 2")

    rng = np.random.default_rng(int(seed))
    df = df_trials.copy()

    fold = pd.Series(-1, index=df.index, dtype=int)

    if split_by == "global":
        idx = df.index.to_numpy()
        rng.shuffle(idx)
        fold_ids = np.arange(idx.size, dtype=int) % int(n_folds)
        fold.loc[idx] = fold_ids
    elif split_by == "subject":
        for _, df_g in df.groupby(["subject"], sort=False):
            idx = df_g.index.to_numpy()
            rng.shuffle(idx)
            fold_ids = np.arange(idx.size, dtype=int) % int(n_folds)
            fold.loc[idx] = fold_ids
    elif split_by == "subject_game":
        for _, df_g in df.groupby(["subject", "game"], sort=False):
            idx = df_g.index.to_numpy()
            rng.shuffle(idx)
            fold_ids = np.arange(idx.size, dtype=int) % int(n_folds)
            fold.loc[idx] = fold_ids
    elif split_by == "subject_game_block":
        # Assign *blocks* (subject, game) to folds, keeping all trials within a
        # subject×game in the same fold. This supports n_folds > trials-per-game
        # (e.g., 10 folds when each game has 8 choice trials).
        blocks = df[["subject", "game"]].drop_duplicates().reset_index(drop=True)
        block_idx = np.arange(len(blocks), dtype=int)
        rng.shuffle(block_idx)
        blocks = blocks.iloc[block_idx].reset_index(drop=True)
        blocks["fold"] = (np.arange(len(blocks), dtype=int) % int(n_folds)).astype(int)
        df = df.merge(blocks, on=["subject", "game"], how="left", validate="many_to_one")
        if df["fold"].isna().any():
            raise RuntimeError("Block fold assignment failed for some rows")
        df["fold"] = df["fold"].astype(int)
        return df
    else:
        raise ValueError(f"Unknown split_by: {split_by}")

    if int((fold < 0).sum()) > 0:
        raise RuntimeError("Fold assignment failed for some rows")

    df["fold"] = fold.astype(int)
    return df


def _filter_subjects(df: pd.DataFrame, *, exclude_subjects: List[str]) -> pd.DataFrame:
    if not exclude_subjects:
        return df
    exclude_set = {str(s) for s in exclude_subjects}
    if "subject" not in df.columns:
        return df
    return df[~df["subject"].astype(str).isin(exclude_set)].copy()


def _init_worker(
    output_dir: str,
    fold_assignments_csv: str,
    time_col: str,
    include_transition: bool,
    n_v_bins: int,
    rt_bins_max: int,
    rt_bins_fixed: int,
    min_trials_per_rt_bin: int,
    n_sim_per_vbin: int,
    alpha: float,
    seed: int,
    irrelevant_mode: str,
    include_center_fixations: bool,
    center_gaze_mode: str,
    center_mode: str,
    reward_scale: float,
    reward_source: str,
    data_dir: str,
    exclude_subjects: tuple[str, ...],
    fixation_data_dir: str = "",
) -> None:
    """Load group data once per process and cache config + fold assignments."""

    out = Path(output_dir).resolve()
    fix_dir = Path(fixation_data_dir).resolve() if fixation_data_dir else None
    df_trials_all = load_group_trial_templates(
        out,
        include_center_fixations=bool(include_center_fixations),
        reward_scale=float(reward_scale),
        reward_source=str(reward_source),
        data_dir=Path(str(data_dir)).resolve(),
        exclude_subjects=tuple(str(s) for s in (exclude_subjects or ())),
        fixation_data_dir=fix_dir,
    )
    df_fix_all = load_group_fixations(
        out,
        include_center_fixations=bool(include_center_fixations),
        exclude_subjects=tuple(str(s) for s in (exclude_subjects or ())),
        fixation_data_dir=fix_dir,
    )

    df_assign = pd.read_csv(Path(fold_assignments_csv))
    keys = ["subject", "game", "trial_number"]
    if not set(keys + ["fold"]).issubset(set(df_assign.columns)):
        raise ValueError("Fold assignments CSV missing required columns")
    # Normalize merge key types to avoid dtype mismatch across processes.
    df_trials_all = df_trials_all.copy()
    df_trials_all["subject"] = df_trials_all["subject"].astype(str)
    df_trials_all["game"] = pd.to_numeric(df_trials_all["game"], errors="coerce")
    df_trials_all["trial_number"] = pd.to_numeric(df_trials_all["trial_number"], errors="coerce")
    df_trials_all = df_trials_all.dropna(subset=["game", "trial_number"]).copy()
    df_trials_all["trial_number"] = df_trials_all["trial_number"].astype(int)

    df_assign = df_assign[keys + ["fold"]].copy()
    df_assign["subject"] = df_assign["subject"].astype(str)
    df_assign["game"] = pd.to_numeric(df_assign["game"], errors="coerce")
    df_assign["trial_number"] = pd.to_numeric(df_assign["trial_number"], errors="coerce")
    df_assign["fold"] = pd.to_numeric(df_assign["fold"], errors="coerce")
    df_assign = df_assign.dropna(subset=["game", "trial_number", "fold"]).copy()
    df_assign["trial_number"] = df_assign["trial_number"].astype(int)
    df_assign["fold"] = df_assign["fold"].astype(int)

    df_trials_all = df_trials_all.merge(df_assign, on=keys, how="inner")
    if df_trials_all.empty:
        raise ValueError("No trials matched fold assignments")

    config = FitConfig(
        binning=BinningConfig(
            n_v_offer_bins=int(n_v_bins),
            rt_bins_max=int(rt_bins_max),
            rt_bins_fixed=int(rt_bins_fixed),
            min_trials_per_rt_bin=int(min_trials_per_rt_bin),
        ),
        sim=SimulationConfig(
            dt_ms=1.0,
            n_sim_per_vbin=int(n_sim_per_vbin),
            alpha_smoothing=float(alpha),
            seed=int(seed),
            include_transition_time=bool(include_transition),
            irrelevant_mode=str(irrelevant_mode),
            include_center_fixations=bool(include_center_fixations),
            center_gaze_mode=str(center_gaze_mode),
            center_mode=str(center_mode),
        ),
        time_col=str(time_col),
    )

    _G.clear()
    _G.update(
        {
            "output_dir": out,
            "df_trials_all": df_trials_all,
            "df_fix_all": df_fix_all,
            "config": config,
            "keys": keys,
        }
    )


def _run_fold_task(task: Dict[str, Any]) -> Dict[str, object]:
    """Run one K-fold split; return a summary row."""

    df_trials_all: pd.DataFrame = _G["df_trials_all"]
    df_fix_all: pd.DataFrame = _G["df_fix_all"]
    config: FitConfig = _G["config"]
    keys: List[str] = _G["keys"]

    fold_id = int(task["fold"])
    mode = str(task["mode"])
    split_by = str(task["split_by"])
    n_folds = int(task["n_folds"])
    noise_param = str(task["noise_param"])
    fixed_sigma = task.get("fixed_sigma")
    fixed_mu = task.get("fixed_mu")
    fixed_theta = task.get("fixed_theta")
    fit_phi_center = bool(task.get("fit_phi_center", False))

    df_trials_test = df_trials_all[df_trials_all["fold"] == int(fold_id)].copy()
    df_trials_train = df_trials_all[df_trials_all["fold"] != int(fold_id)].copy()

    df_key = df_trials_all[keys + ["fold"]].copy()
    df_train_key = df_key[df_key["fold"] != int(fold_id)][keys].drop_duplicates()
    df_fix_train = df_fix_all.merge(df_train_key, on=keys, how="inner")

    fold_seed = int(task["seed"])

    df_trials_train_b, _, components = build_fitting_components(
        df_trials_train=df_trials_train,
        df_fix_train=df_fix_train,
        config=config,
    )

    if mode == "fit":
        best_x, ll_train = fit_one_fold_pybads(
            df_trials_train=df_trials_train,
            df_fix_train=df_fix_train,
            config=config,
            seed=fold_seed,
            noise_param=noise_param,
            bounds=task["bounds"],
            x0=task["x0"],
            max_iter=int(task["max_iter"]),
            max_fun_evals=int(task["max_fun_evals"]),
            fixed_sigma=fixed_sigma,
            fixed_mu=fixed_mu,
            fixed_theta=fixed_theta,
            fit_phi_center=bool(fit_phi_center),
            df_trials_train_b=df_trials_train_b,
            components=components,
        )

        best_x = np.asarray(best_x, dtype=float)
        idx = 0
        best_d = float(best_x[idx])
        idx += 1

        if fixed_theta is not None:
            best_theta = float(fixed_theta)
        else:
            best_theta = float(best_x[idx])
            idx += 1

        if fixed_sigma is not None:
            best_sigma = float(fixed_sigma)
            best_mu = float(best_sigma) / max(best_d, 1e-300)
        elif fixed_mu is not None:
            if noise_param != "mu":
                raise ValueError("fixed_mu is only valid when noise_param == 'mu'")
            best_mu = float(fixed_mu)
            best_sigma = sigma_from_mu(best_d, best_mu)
        else:
            best_noise = float(best_x[idx])
            idx += 1
            best_mu = best_noise if noise_param == "mu" else float(best_noise) / max(best_d, 1e-300)
            best_sigma = sigma_from_mu(best_d, best_noise) if noise_param == "mu" else float(best_noise)

        if fit_phi_center:
            best_phi_center = float(best_x[idx])
        else:
            best_phi_center = 1.0

        params = ADDMParameters(d=best_d, theta=best_theta, sigma=best_sigma, phi_center=best_phi_center)
    else:
        fixed_params = task.get("fixed_params")
        if fixed_params is None:
            raise ValueError("eval-only task missing fixed_params")
        params = fixed_params
        best_mu = float(params.sigma) / float(params.d)
        ll_train = compute_binned_loglik_given_components(
            df_trials_eval=df_trials_train_b,
            params=params,
            config=config,
            components=components,
            rng=np.random.default_rng(fold_seed),
        )

    ll_test = compute_binned_loglik_given_components(
        df_trials_eval=df_trials_test,
        params=params,
        config=config,
        components=components,
        rng=np.random.default_rng(fold_seed),
    )

    res: Dict[str, object] = dict(task["meta"])  # copy
    res.update(
        {
            "setting": str(task["setting"]),
            "heldout_game": float(fold_id),
            "fold": int(fold_id),
            "mode": mode,
            "split_by": split_by,
            "n_folds": n_folds,
            "n_train_trials": int(len(df_trials_train)),
            "n_test_trials": int(len(df_trials_test)),
            "d": float(params.d),
            "theta": float(params.theta),
            "sigma": float(params.sigma),
            "mu": float(best_mu),
            "phi_center": float(getattr(params, "phi_center", 1.0)),
            "noise_param": str(noise_param),
            "loglik_train": float(ll_train),
            "loglik_test": float(ll_test),
            "seed": int(fold_seed),
        }
    )
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold CV for the adapted aDDM.")

    parser.add_argument("--mode", choices=["fit", "eval-only"], default="fit")
    parser.add_argument("--output-dir", type=str, default="output")

    parser.add_argument(
        "--setting",
        type=str,
        default="kfold",
        help="Label used in output filename, mirroring LOGO 'setting' convention.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="",
        help="Optional subdirectory under output/addm/kfold/ to write outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set and an output summary already exists, skip folds already completed.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel processes (folds in parallel). Use 0 for all cores.",
    )

    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument(
        "--split-by",
        choices=["subject", "subject_game", "subject_game_block", "global"],
        default="subject",
        help=(
            "How to assign folds. 'subject_game' assigns folds *within* each subject×game "
            "(will only use up to the number of trials in a game, e.g. 8). "
            "'subject_game_block' assigns whole subject×game blocks to folds (supports 10 folds)."
        ),
    )
    parser.add_argument(
        "--exclude-subjects",
        type=str,
        nargs="*",
        default=["107", "131"],
        help="Subjects to exclude (default excludes 107 and 131 for eyetracking analyses).",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "Seed used ONLY for fold assignment (train/test split). If not provided, uses --seed. "
            "This allows comparing models across identical folds while sweeping --seed for fitting randomness."
        ),
    )
    parser.add_argument("--max-folds", type=int, default=None, help="If set, only run the first N folds.")
    parser.add_argument("--tag", type=str, default="", help="Optional tag appended to output filenames.")

    # What time variable are we fitting?
    parser.add_argument("--time-col", choices=["fix_ms", "rt_ms"], default="fix_ms")
    parser.add_argument(
        "--include-transition",
        action="store_true",
        help="If set, adds sampled transition time to simulated decision time (use with --time-col rt_ms).",
    )

    # Binning config
    parser.add_argument("--n-v-bins", type=int, default=7)
    parser.add_argument("--rt-bins-max", type=int, default=15)
    parser.add_argument(
        "--rt-bins-fixed",
        type=int,
        default=0,
        help="If >0, force a fixed number of RT bins per (V_offer bin, choice) cell (subject to ties/data).",
    )
    parser.add_argument("--min-trials-per-rt-bin", type=int, default=25)

    # Simulation config
    parser.add_argument("--n-sim-per-vbin", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0)

    # Model variants / gaze handling (match LOGO runner)
    parser.add_argument(
        "--irrelevant-mode",
        choices=["zero", "theta_sumrel", "sumrel"],
        default="zero",
        help="How drift behaves during irrelevant item fixations.",
    )
    parser.add_argument(
        "--include-center-fixations",
        action="store_true",
        help="If set, use the *with-center* modeling dataset (roi_content=='fixation').",
    )
    parser.add_argument(
        "--center-gaze-mode",
        choices=["separate", "merge_with_irrelevant"],
        default="separate",
        help="How the gaze generator treats center fixations.",
    )
    parser.add_argument(
        "--center-mode",
        choices=["same_as_irrelevant", "zero", "theta_sumrel", "sumrel", "phi_sumrel"],
        default="same_as_irrelevant",
        help="How drift behaves during center fixations.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Multiply all encoded rewards by this factor before fitting (default 1.0).",
    )

    parser.add_argument(
        "--reward-source",
        choices=["true", "recalled"],
        default="true",
        help=(
            "Which item rewards to use when computing V_offer and evidence accumulation. "
            "'true' uses encoding outcomes; 'recalled' uses value-recall transcriptions."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help=(
            "Path to the repo data directory containing <subid>/valuerecall/<subid>_valuerecall.csv. "
            "Only used when --reward-source recalled."
        ),
    )
    parser.add_argument(
        "--fixation-data-dir",
        type=str,
        default=None,
        help=(
            "If set, read per-subject fixation CSV files from this directory "
            "instead of --output-dir. Subject discovery also uses this directory."
        ),
    )

    # Optional center gain (only used when --center-mode phi_sumrel)
    parser.add_argument("--phi-center-min", type=float, default=0.0)
    parser.add_argument("--phi-center-max", type=float, default=10.0)
    parser.add_argument(
        "--phi-center0",
        type=float,
        default=5.0,
        help="Initial guess for phi_center (only used when --center-mode phi_sumrel).",
    )
    parser.add_argument(
        "--phi-center",
        type=float,
        default=1.0,
        help="phi_center used for --mode eval-only (only relevant when --center-mode phi_sumrel).",
    )

    # Optional pyBADS budget controls (useful for smoke tests)
    parser.add_argument("--max-iter", type=int, default=0, help="If >0, sets pyBADS max_iter.")
    parser.add_argument("--max-fun-evals", type=int, default=0, help="If >0, sets pyBADS max_fun_evals.")

    # Bounds / init for fitting
    parser.add_argument("--d-min", type=float, default=1e-4)
    parser.add_argument("--d-max", type=float, default=0.05)
    parser.add_argument("--theta-min", type=float, default=0.01)
    parser.add_argument("--theta-max", type=float, default=0.99)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--sigma-max", type=float, default=0.1)

    parser.add_argument(
        "--noise-param",
        choices=["sigma", "mu"],
        default="mu",
        help="Noise parameterization: fit sigma directly, or fit mu with sigma=d*mu.",
    )
    parser.add_argument(
        "--mu-min",
        type=float,
        default=None,
        help="Lower bound on mu (internal units). If unset and --noise-param mu, defaults to sigma_min/d_max.",
    )
    parser.add_argument(
        "--mu-max",
        type=float,
        default=None,
        help="Upper bound on mu (internal units). If unset and --noise-param mu, defaults to sigma_max/d_max.",
    )

    parser.add_argument("--d0", type=float, default=0.0005)
    parser.add_argument("--theta0", type=float, default=0.5)
    parser.add_argument("--sigma0", type=float, default=0.001)
    parser.add_argument(
        "--fix-sigma",
        type=float,
        default=float("nan"),
        help="If set (finite), fixes sigma to this value and fits only (d, theta). Units are the internal per-sqrt(ms) sigma used by the model.",
    )
    parser.add_argument(
        "--fix-mu",
        type=float,
        default=float("nan"),
        help=(
            "If set (finite) and --noise-param mu, fixes mu to this value and fits only (d, theta). "
            "Note: sigma is not fixed in this case; it is deterministically set as sigma=d*mu for each candidate d."
        ),
    )
    parser.add_argument(
        "--fix-theta",
        type=float,
        default=float("nan"),
        help="If set (finite), fixes theta to this value and drops theta from optimization.",
    )
    parser.add_argument(
        "--mu0",
        type=float,
        default=None,
        help="Initial guess for mu (internal units). If unset and --noise-param mu, defaults to sigma0/d0.",
    )

    # Eval-only params
    parser.add_argument(
        "--params",
        type=float,
        nargs=3,
        metavar=("D", "THETA", "NOISE"),
        default=None,
        help="Only used for --mode eval-only. Format: d theta sigma (if --noise-param sigma) or d theta mu (if --noise-param mu).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()

    include_center_fixations = bool(args.include_center_fixations)
    reward_scale = float(args.reward_scale)
    reward_source = str(args.reward_source)
    data_dir = Path(str(args.data_dir)).resolve()
    exclude_subjects = tuple(str(s) for s in (args.exclude_subjects or []))
    fixation_data_dir = Path(args.fixation_data_dir).resolve() if args.fixation_data_dir is not None else None

    df_trials_all = load_group_trial_templates(
        output_dir,
        include_center_fixations=include_center_fixations,
        reward_scale=reward_scale,
        reward_source=reward_source,
        data_dir=data_dir,
        exclude_subjects=exclude_subjects,
        fixation_data_dir=fixation_data_dir,
    )
    df_fix_all = load_group_fixations(
        output_dir,
        include_center_fixations=include_center_fixations,
        exclude_subjects=exclude_subjects,
        fixation_data_dir=fixation_data_dir,
    )

    # (Exclusions are already applied during loading via exclude_subjects.)

    if df_trials_all.empty:
        raise ValueError("No trials left after exclusions")

    split_seed = int(args.seed) if args.split_seed is None else int(args.split_seed)

    df_trials_all = _assign_folds(
        df_trials_all,
        n_folds=int(args.n_folds),
        seed=int(split_seed),
        split_by=str(args.split_by),
    )

    # Worker processes build config; main only needs fold assignments + bounds.

    noise_param = str(args.noise_param)
    fixed_sigma = float(args.fix_sigma) if np.isfinite(float(args.fix_sigma)) else None
    if fixed_sigma is not None and fixed_sigma <= 0:
        raise ValueError("--fix-sigma must be > 0")

    fixed_mu = float(args.fix_mu) if np.isfinite(float(args.fix_mu)) else None
    if fixed_mu is not None:
        if fixed_mu <= 0:
            raise ValueError("--fix-mu must be > 0")
        if noise_param != "mu":
            raise ValueError("--fix-mu is only valid when --noise-param mu")
        if fixed_sigma is not None:
            raise ValueError("Do not set both --fix-mu and --fix-sigma")

    fixed_theta = float(args.fix_theta) if np.isfinite(float(args.fix_theta)) else None
    if fixed_theta is not None:
        if not (0.0 <= float(fixed_theta) <= 1.0):
            raise ValueError("--fix-theta must be in [0, 1]")
        if not (float(args.theta_min) <= float(fixed_theta) <= float(args.theta_max)):
            raise ValueError("--fix-theta must be within [--theta-min, --theta-max]")

    if args.mode == "eval-only":
        if args.params is None:
            raise ValueError("--mode eval-only requires --params d theta noise")
        d_eval = float(args.params[0])
        theta_eval = float(fixed_theta) if fixed_theta is not None else float(args.params[1])
        noise_eval = float(args.params[2])
        if noise_param == "mu":
            sigma_eval = sigma_from_mu(d_eval, noise_eval)
        elif noise_param == "sigma":
            sigma_eval = noise_eval
        else:
            raise ValueError(f"Unknown noise_param: {noise_param}")
        phi_center_eval = float(args.phi_center) if str(args.center_mode) == "phi_sumrel" else 1.0
        fixed_params = ADDMParameters(d=d_eval, theta=theta_eval, sigma=sigma_eval, phi_center=phi_center_eval)
    else:
        fixed_params = None

    if noise_param == "mu":
        mu_min = float(args.mu_min) if args.mu_min is not None else float(args.sigma_min) / float(args.d_max)
        mu_max = float(args.mu_max) if args.mu_max is not None else float(args.sigma_max) / float(args.d_max)
        mu0 = float(args.mu0) if args.mu0 is not None else float(args.sigma0) / float(args.d0)

        margin = 1e-3
        mu0 = float(np.clip(mu0, mu_min * (1.0 + margin), mu_max * (1.0 - margin)))
        if not (np.isfinite(mu0) and mu_min < mu0 < mu_max):
            mu0 = float(np.sqrt(mu_min * mu_max))
            mu0 = float(np.clip(mu0, mu_min * (1.0 + margin), mu_max * (1.0 - margin)))
        lb3, ub3, x03 = mu_min, mu_max, mu0
        plb3 = max(mu_min, mu0 / 10)
        pub3 = min(mu_max, mu0 * 10)
    elif noise_param == "sigma":
        lb3, ub3, x03 = float(args.sigma_min), float(args.sigma_max), float(args.sigma0)
        plb3 = max(float(args.sigma_min), float(args.sigma0) / 10)
        pub3 = min(float(args.sigma_max), float(args.sigma0) * 10)
    else:
        raise ValueError(f"Unknown noise_param: {noise_param}")

    # Build bounds for (d, [theta], noise) first.
    # If fixed_mu is set (only valid when noise_param=='mu'), we treat the noise dimension as fixed.
    noise_is_fixed = (fixed_sigma is not None) or (fixed_mu is not None)

    if noise_is_fixed and fixed_theta is not None:
        lb = np.array([args.d_min], dtype=float)
        ub = np.array([args.d_max], dtype=float)
        x0 = np.array([args.d0], dtype=float)
        plb = np.array([max(args.d_min, args.d0 / 10)], dtype=float)
        pub = np.array([min(args.d_max, args.d0 * 10)], dtype=float)
        bounds = (lb, ub, plb, pub)
    elif noise_is_fixed and fixed_theta is None:
        lb = np.array([args.d_min, args.theta_min], dtype=float)
        ub = np.array([args.d_max, args.theta_max], dtype=float)
        x0 = np.array([args.d0, args.theta0], dtype=float)
        plb = np.array([max(args.d_min, args.d0 / 10), max(args.theta_min, 0.05)], dtype=float)
        pub = np.array([min(args.d_max, args.d0 * 10), min(args.theta_max, 0.95)], dtype=float)
        bounds = (lb, ub, plb, pub)
    elif (not noise_is_fixed) and fixed_theta is not None:
        lb = np.array([args.d_min, lb3], dtype=float)
        ub = np.array([args.d_max, ub3], dtype=float)
        x0 = np.array([args.d0, x03], dtype=float)
        plb = np.array([max(args.d_min, args.d0 / 10), plb3], dtype=float)
        pub = np.array([min(args.d_max, args.d0 * 10), pub3], dtype=float)
        bounds = (lb, ub, plb, pub)
    else:
        lb = np.array([args.d_min, args.theta_min, lb3], dtype=float)
        ub = np.array([args.d_max, args.theta_max, ub3], dtype=float)
        x0 = np.array([args.d0, args.theta0, x03], dtype=float)
        plb = np.array([max(args.d_min, args.d0 / 10), max(args.theta_min, 0.05), plb3], dtype=float)
        pub = np.array([min(args.d_max, args.d0 * 10), min(args.theta_max, 0.95), pub3], dtype=float)
        bounds = (lb, ub, plb, pub)

    # Optionally append phi_center bounds if this run uses it.
    fit_phi_center_global = bool(str(args.center_mode) == "phi_sumrel")
    if fit_phi_center_global and args.mode == "fit":
        phi_lb = float(args.phi_center_min)
        phi_ub = float(args.phi_center_max)
        phi0 = float(args.phi_center0)
        if not (np.isfinite(phi_lb) and np.isfinite(phi_ub) and phi_lb < phi_ub):
            raise ValueError("Invalid phi_center bounds")
        margin = 1e-3
        phi0 = float(np.clip(phi0, phi_lb * (1.0 + margin), phi_ub * (1.0 - margin)))
        if not (phi_lb < phi0 < phi_ub):
            phi0 = float(np.sqrt(max(phi_lb, 1e-12) * phi_ub))

        lb, ub, plb, pub = bounds
        lb = np.concatenate([lb, np.array([phi_lb], dtype=float)])
        ub = np.concatenate([ub, np.array([phi_ub], dtype=float)])
        x0 = np.concatenate([x0, np.array([phi0], dtype=float)])
        plb = np.concatenate([plb, np.array([max(phi_lb, phi0 / 10)], dtype=float)])
        pub = np.concatenate([pub, np.array([min(phi_ub, phi0 * 10)], dtype=float)])
        bounds = (lb, ub, plb, pub)

    folds = list(range(int(args.n_folds)))
    if args.max_folds is not None:
        folds = folds[: int(args.max_folds)]

    results: List[Dict[str, object]] = []

    out_dir = output_dir / "addm" / "kfold"
    if str(args.out_subdir).strip():
        out_dir = out_dir / str(args.out_subdir).strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = ("_" + str(args.tag)) if str(args.tag).strip() else ""
    setting = str(args.setting).strip() if str(args.setting).strip() else "kfold"

    out_path = out_dir / f"addm_kfold_fit_summary_{setting}{tag}.csv"
    assign_path = out_dir / f"addm_kfold_fold_assignments_{setting}{tag}.csv"

    done_folds: set[int] = set()
    if bool(args.resume) and out_path.exists():
        try:
            df_prev = pd.read_csv(out_path)
            if "fold" in df_prev.columns:
                done_folds = set(pd.to_numeric(df_prev["fold"], errors="coerce").dropna().astype(int).tolist())
            results = df_prev.to_dict("records")  # type: ignore[assignment]
            print(f"[resume] Found existing summary with {len(done_folds)} completed folds: {sorted(done_folds)}")
        except Exception:
            print("[resume] Existing summary found but could not be read; starting fresh")
            results = []
            done_folds = set()

    keys = ["subject", "game", "trial_number"]
    df_key = df_trials_all[keys + ["fold"]].copy()
    # Normalize key types for stable joins in worker processes.
    df_key["subject"] = df_key["subject"].astype(str)
    df_key["game"] = pd.to_numeric(df_key["game"], errors="coerce")
    df_key["trial_number"] = pd.to_numeric(df_key["trial_number"], errors="coerce")
    df_key["fold"] = pd.to_numeric(df_key["fold"], errors="coerce")
    df_key = df_key.dropna(subset=["game", "trial_number", "fold"]).copy()
    df_key["trial_number"] = df_key["trial_number"].astype(int)
    df_key["fold"] = df_key["fold"].astype(int)
    df_key.drop_duplicates().to_csv(assign_path, index=False)

    # Build metadata (written into each row).
    meta: Dict[str, object] = {
        "time_col": str(args.time_col),
        "include_transition": int(bool(args.include_transition)),
        "irrelevant_mode": str(args.irrelevant_mode),
        "include_center_fixations": int(bool(include_center_fixations)),
        "center_gaze_mode": str(args.center_gaze_mode),
        "center_mode": str(args.center_mode),
        "reward_scale": float(reward_scale),
        "n_v_bins": int(args.n_v_bins),
        "rt_bins_max": int(args.rt_bins_max),
        "rt_bins_fixed": int(args.rt_bins_fixed),
        "min_trials_per_rt_bin": int(args.min_trials_per_rt_bin),
        "n_sim_per_vbin": int(args.n_sim_per_vbin),
        "alpha": float(args.alpha),
        "d_min": float(args.d_min),
        "d_max": float(args.d_max),
        "theta_min": float(args.theta_min),
        "theta_max": float(args.theta_max),
        "sigma_min": float(args.sigma_min),
        "sigma_max": float(args.sigma_max),
        "mu_min": float(args.mu_min) if args.mu_min is not None else float("nan"),
        "mu_max": float(args.mu_max) if args.mu_max is not None else float("nan"),
        "fix_mu": float(fixed_mu) if fixed_mu is not None else float("nan"),
        "phi_center_min": float(args.phi_center_min),
        "phi_center_max": float(args.phi_center_max),
    }

    # Build tasks.
    tasks: List[Dict[str, Any]] = []
    for i, fold_id in enumerate(folds):
        if int(fold_id) in done_folds:
            continue

        fold_seed = int(args.seed + 1000 * i)
        t: Dict[str, Any] = {
            "setting": setting,
            "fold": int(fold_id),
            "mode": str(args.mode),
            "split_by": str(args.split_by),
            "n_folds": int(args.n_folds),
            "noise_param": str(noise_param),
            "fixed_sigma": fixed_sigma,
            "fixed_mu": fixed_mu,
            "fixed_theta": fixed_theta,
            "fit_phi_center": bool(fit_phi_center_global) and str(args.mode) == "fit",
            "seed": int(fold_seed),
            "max_iter": int(args.max_iter),
            "max_fun_evals": int(args.max_fun_evals),
            "meta": meta,
        }
        if str(args.mode) == "fit":
            t["bounds"] = bounds
            t["x0"] = np.asarray(x0, dtype=float)
        else:
            t["fixed_params"] = fixed_params
        tasks.append(t)

    if not tasks:
        print("[OK] Nothing to do (all folds complete).")
        return

    n_jobs = int(args.n_jobs)
    if n_jobs <= 0:
        n_jobs = int(os.cpu_count() or 1)
    n_jobs = max(1, min(n_jobs, len(tasks)))

    print(f"[INFO] Folds to run: {len(tasks)} | jobs={n_jobs} | output={out_path}")

    from multiprocessing import get_context

    ctx = get_context("fork")
    t0 = time.time()

    initargs = (
        str(output_dir),
        str(assign_path),
        str(args.time_col),
        bool(args.include_transition),
        int(args.n_v_bins),
        int(args.rt_bins_max),
        int(args.rt_bins_fixed),
        int(args.min_trials_per_rt_bin),
        int(args.n_sim_per_vbin),
        float(args.alpha),
        int(args.seed),
        str(args.irrelevant_mode),
        bool(include_center_fixations),
        str(args.center_gaze_mode),
        str(args.center_mode),
        float(reward_scale),
        str(reward_source),
        str(data_dir),
        tuple(exclude_subjects),
        str(fixation_data_dir) if fixation_data_dir is not None else "",
    )

    pending: List[Dict[str, object]] = []

    if n_jobs == 1:
        _init_worker(*initargs)
        it = map(_run_fold_task, tasks)
    else:
        pool = ctx.Pool(processes=n_jobs, initializer=_init_worker, initargs=initargs)
        it = pool.imap_unordered(_run_fold_task, tasks, chunksize=1)

    try:
        for k, res in enumerate(it, start=1):
            pending.append(res)

            df_new = pd.DataFrame(pending)
            if out_path.exists():
                try:
                    df_old = pd.read_csv(out_path)
                    df_all = pd.concat([df_old, df_new], ignore_index=True)
                except Exception:
                    df_all = df_new
            else:
                df_all = df_new
            df_all = df_all.drop_duplicates(subset=["fold"], keep="last")
            df_all.to_csv(out_path, index=False)
            pending.clear()

            param_str = f"d={res['d']:.4g}, theta={res['theta']:.3g}, sigma={res['sigma']:.4g}, mu={res['mu']:.4g}"
            if str(res.get("center_mode", "")) == "phi_sumrel":
                param_str += f", phi_center={res.get('phi_center', 1.0):.3g}"
            print(f"[fold {k}/{len(tasks)}] fold={int(res['fold'])} ll_test={res['loglik_test']:.2f} params=({param_str})")

            if k % 2 == 0:
                elapsed = time.time() - t0
                rate = k / max(1e-9, elapsed)
                eta = (len(tasks) - k) / max(1e-9, rate)
                print(f"[PROGRESS] {k}/{len(tasks)} done | {rate:.2f} folds/s | ETA {eta/60:.1f} min")
    finally:
        if n_jobs != 1:
            pool.close()
            pool.join()

    print("[OK] Wrote K-fold summary:", out_path)
    print("[OK] Wrote fold assignments:", assign_path)


if __name__ == "__main__":
    main()

"""Large-scale parameter recovery sweep for the adapted aDDM.

This script samples many parameter combinations from a specified space,
then for each combination:
1) Simulates a synthetic dataset (one draw per trial; n_sim_per_trial=1 recommended).
2) Fits parameters back using the same binned simulation-based likelihood.
3) Stores true vs recovered parameters.

It parallelizes across CPU cores and supports resuming where left off.

"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from pybads import BADS  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("pyBADS is required. Install it with `pip install pybads`.") from e

try:
    from .adapted_addm_simulation import ADDMParameters
    from .addm_fitting import (
        BinningConfig,
        FitConfig,
        FittingComponents,
        SimulationConfig,
        assign_v_offer_bins_from_edges,
        build_rt_bins_per_cell,
        compute_binned_loglik_given_components,
        load_group_fixations,
        load_group_trial_templates,
        build_fitting_components,
        simulate_trials_generative,
    )
except ImportError:  # pragma: no cover
    from adapted_addm_simulation import ADDMParameters
    from addm_fitting import (
        BinningConfig,
        FitConfig,
        FittingComponents,
        SimulationConfig,
        assign_v_offer_bins_from_edges,
        build_rt_bins_per_cell,
        compute_binned_loglik_given_components,
        load_group_fixations,
        load_group_trial_templates,
        build_fitting_components,
        simulate_trials_generative,
    )


# ---- multiprocessing globals (initialized per worker) ----
_G: Dict[str, Any] = {}


def _sigma_from_mu(*, d: float, mu: float) -> float:
    return float(d) * float(mu)


def _convert_units_to_internal_ms_for_mu(*, d: float, mu: float, units: str) -> Tuple[float, float]:
    """Convert (d, mu) in 'ms' or 's' to internal per-ms mu units.

    If user uses per-second parameters such that sigma_s = d_s * mu_s, then
    after conversion to internal units (dt_ms=1):
      d_ms = d_s / 1000
      sigma_ms = sigma_s / sqrt(1000)
      mu_ms = sigma_ms / d_ms = mu_s * sqrt(1000)
    """
    u = str(units).lower().strip()
    if u == "ms":
        return float(d), float(mu)
    if u == "s":
        ms_per_s = 1000.0
        sqrt_ms_per_s = float(np.sqrt(ms_per_s))
        return float(d) / ms_per_s, float(mu) * sqrt_ms_per_s
    raise ValueError(f"units must be 'ms' or 's', got {units!r}")


def _convert_units_to_internal_ms(*, d: float, sigma: float, units: str) -> Tuple[float, float]:
    """Convert user params (ms or s) to internal per-ms/per-sqrt(ms)."""
    u = str(units).lower().strip()
    if u == "ms":
        return float(d), float(sigma)
    if u == "s":
        ms_per_s = 1000.0
        sqrt_ms_per_s = float(np.sqrt(ms_per_s))
        return float(d) / ms_per_s, float(sigma) / sqrt_ms_per_s
    raise ValueError(f"units must be 'ms' or 's', got {units!r}")


def _sample_parameter_combos(
    *,
    n: int,
    seed: int,
    d_min: float,
    d_max: float,
    theta_min: float,
    theta_max: float,
    sigma_min: float,
    sigma_max: float,
    mu_min: float,
    mu_max: float,
    noise_param: str,
    log_uniform: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))

    def logu(lo: float, hi: float, size: int) -> np.ndarray:
        lo = float(lo)
        hi = float(hi)
        if lo <= 0 or hi <= 0:
            raise ValueError("log-uniform sampling requires positive bounds")
        a = np.log(lo)
        b = np.log(hi)
        return np.exp(rng.uniform(a, b, size=size))

    d = logu(d_min, d_max, n) if log_uniform else rng.uniform(float(d_min), float(d_max), size=n)
    theta = rng.uniform(float(theta_min), float(theta_max), size=n)

    npmode = str(noise_param).lower().strip()
    if npmode not in ("sigma", "mu"):
        raise ValueError(f"noise_param must be 'sigma' or 'mu', got {noise_param!r}")

    if npmode == "sigma":
        sigma = logu(sigma_min, sigma_max, n) if log_uniform else rng.uniform(float(sigma_min), float(sigma_max), size=n)
        mu = sigma / np.maximum(d, 1e-300)
    else:
        mu = logu(mu_min, mu_max, n) if log_uniform else rng.uniform(float(mu_min), float(mu_max), size=n)
        sigma = d * mu

    df = pd.DataFrame(
        {
            "combo": np.arange(1, n + 1, dtype=int),
            "d": d,
            "theta": theta,
            "sigma": sigma,
            "mu": mu,
        }
    )
    return df


def _init_worker(
    output_dir: str,
    reward_source: str,
    data_dir: str,
    time_col: str,
    include_transition: bool,
    irrelevant_mode: str,
    include_center_fixations: bool,
    center_gaze_mode: str,
    center_mode: str,
    n_v_bins: int,
    rt_bins_max: int,
    rt_bins_fixed: int,
    min_trials_per_rt_bin: int,
    n_sim_per_vbin: int,
    alpha: float,
    seed: int,
    fixation_data_dir: str = "",
) -> None:
    """Load data once per process."""
    out = Path(output_dir).resolve()
    fix_dir = Path(fixation_data_dir).resolve() if fixation_data_dir else None

    reward_source_s = str(reward_source).strip().lower()
    data_dir_p: Optional[Path] = None
    if reward_source_s == "recalled":
        data_dir_p = Path(str(data_dir)).resolve()

    df_trials = load_group_trial_templates(
        out,
        include_center_fixations=bool(include_center_fixations),
        reward_source=reward_source_s,
        data_dir=data_dir_p,
        fixation_data_dir=fix_dir,
    )
    df_fix = load_group_fixations(out, include_center_fixations=bool(include_center_fixations), fixation_data_dir=fix_dir)

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

    # Build once to get v_edges + gaze stats. We will rebuild time-bin edges per synthetic dataset.
    df_trials_b, _, base_components = build_fitting_components(
        df_trials_train=df_trials,
        df_fix_train=df_fix,
        config=config,
    )

    _G.clear()
    _G.update(
        {
            "output_dir": out,
            "config": config,
            "df_trials_b": df_trials_b,
            "v_edges": base_components.v_edges,
            "gaze_by_bin": base_components.gaze_by_bin,
        }
    )


def _build_synth_components(*, df_synth: pd.DataFrame) -> FittingComponents:
    config: FitConfig = _G["config"]

    # Ensure v_bin exists and is consistent with the global v_edges.
    if "v_bin" not in df_synth.columns:
        df_synth = df_synth.copy()
        df_synth["v_bin"] = assign_v_offer_bins_from_edges(
            df_synth["v_offer"].to_numpy(dtype=float),
            _G["v_edges"],
        )

    rt_edges_by_cell = build_rt_bins_per_cell(
        df_synth,
        v_bin_col="v_bin",
        rt_col=str(config.time_col),
        rt_bins_max=config.binning.rt_bins_max,
        min_trials_per_rt_bin=config.binning.min_trials_per_rt_bin,
        rt_bins_fixed=getattr(config.binning, "rt_bins_fixed", 0),
    )

    max_time_ms = float(np.nanmax(pd.to_numeric(df_synth[str(config.time_col)], errors="coerce").to_numpy(dtype=float)))

    return FittingComponents(
        v_edges=_G["v_edges"],
        gaze_by_bin=_G["gaze_by_bin"],
        rt_edges_by_cell=rt_edges_by_cell,
        max_time_ms=max_time_ms,
    )


def _fit_one(
    *,
    df_synth: pd.DataFrame,
    seed_fit: int,
    bounds: Dict[str, float],
    init: Dict[str, float],
    noise_param: str,
    max_iter: int,
    max_fun_evals: int,
    fixed_sigma: float | None,
    fixed_theta: float | None,
) -> Tuple[ADDMParameters, float, float]:
    config: FitConfig = _G["config"]

    synth_components = _build_synth_components(df_synth=df_synth)

    npmode = str(noise_param).lower().strip()
    if npmode not in ("sigma", "mu"):
        raise ValueError(f"noise_param must be 'sigma' or 'mu', got {noise_param!r}")

    if fixed_sigma is not None and float(fixed_sigma) <= 0:
        raise ValueError("fixed_sigma must be > 0")

    if fixed_theta is not None:
        if not (0.0 <= float(fixed_theta) <= 1.0):
            raise ValueError("fixed_theta must be in [0, 1]")

    # Deterministic objective with fixed RNG seed.
    def objective(x: np.ndarray) -> float:
        d = float(x[0])
        if fixed_theta is not None:
            theta = float(fixed_theta)
            noise_idx = 1
        else:
            theta = float(x[1])
            noise_idx = 2
        if fixed_sigma is not None:
            sigma = float(fixed_sigma)
        else:
            if npmode == "sigma":
                sigma = float(x[noise_idx])
            else:
                mu = float(x[noise_idx])
                sigma = _sigma_from_mu(d=d, mu=mu)
        p = ADDMParameters(d=d, theta=theta, sigma=sigma)
        ll = compute_binned_loglik_given_components(
            df_trials_eval=df_synth,
            params=p,
            config=config,
            components=synth_components,
            rng=np.random.default_rng(int(seed_fit)),
        )
        return -float(ll)

    # Build hard bounds / init vector in the appropriate parameterization.
    # Dimensionality depends on which parameters are fixed.
    dim_names: list[str]
    if fixed_sigma is not None and fixed_theta is not None:
        dim_names = ["d"]
        lb = np.array([bounds["d_min"]], dtype=float)
        ub = np.array([bounds["d_max"]], dtype=float)
        raw_x0 = np.array([init["d0"]], dtype=float)
    elif fixed_sigma is not None and fixed_theta is None:
        dim_names = ["d", "theta"]
        lb = np.array([bounds["d_min"], bounds["theta_min"]], dtype=float)
        ub = np.array([bounds["d_max"], bounds["theta_max"]], dtype=float)
        raw_x0 = np.array([init["d0"], init["theta0"]], dtype=float)
    elif fixed_sigma is None and fixed_theta is not None:
        dim_names = ["d", "noise"]
        n_lo = bounds["sigma_min"] if npmode == "sigma" else bounds["mu_min"]
        n_hi = bounds["sigma_max"] if npmode == "sigma" else bounds["mu_max"]
        n0 = init["sigma0"] if npmode == "sigma" else init["mu0"]
        lb = np.array([bounds["d_min"], n_lo], dtype=float)
        ub = np.array([bounds["d_max"], n_hi], dtype=float)
        raw_x0 = np.array([init["d0"], n0], dtype=float)
    else:
        dim_names = ["d", "theta", "noise"]
        if npmode == "sigma":
            lb = np.array([bounds["d_min"], bounds["theta_min"], bounds["sigma_min"]], dtype=float)
            ub = np.array([bounds["d_max"], bounds["theta_max"], bounds["sigma_max"]], dtype=float)
            raw_x0 = np.array([init["d0"], init["theta0"], init["sigma0"]], dtype=float)
        else:
            lb = np.array([bounds["d_min"], bounds["theta_min"], bounds["mu_min"]], dtype=float)
            ub = np.array([bounds["d_max"], bounds["theta_max"], bounds["mu_max"]], dtype=float)
            raw_x0 = np.array([init["d0"], init["theta0"], init["mu0"]], dtype=float)

    # Clip x0 safely inside hard bounds (and not too close to the edges) to avoid
    # pyBADS warnings and brittle starts.
    width = ub - lb
    # Use a relative margin (1% of range) so x0 is well inside (lb, ub).
    eps = np.maximum(1e-12, 0.01 * width)
    x0 = np.clip(raw_x0, lb + eps, ub - eps)

    # Construct plausible bounds that:
    # - contain x0
    # - are not too close to the hard bounds
    # - have a minimum width relative to (ub-lb) to reduce mesh overflow.
    plb = np.empty_like(lb, dtype=float)
    pub = np.empty_like(ub, dtype=float)
    for i, name in enumerate(dim_names):
        if name == "theta":
            plb[i] = max(float(lb[i] + eps[i]), float(x0[i]) - 0.45)
            pub[i] = min(float(ub[i] - eps[i]), float(x0[i]) + 0.45)
        else:
            plb[i] = max(float(lb[i] + eps[i]), float(x0[i]) / 10.0)
            pub[i] = min(float(ub[i] - eps[i]), float(x0[i]) * 10.0)

    min_width = 0.2 * width
    for i in range(int(lb.size)):
        if not np.isfinite(min_width[i]) or min_width[i] <= 0:
            plb[i] = lb[i]
            pub[i] = ub[i]
            continue

        if (pub[i] - plb[i]) < min_width[i]:
            half = 0.5 * min_width[i]
            plb[i] = max(lb[i], float(x0[i]) - half)
            pub[i] = min(ub[i], float(x0[i]) + half)

        # Ensure x0 is within the plausible bounds.
        plb[i] = min(plb[i], float(x0[i]))
        pub[i] = max(pub[i], float(x0[i]))

        # Final fallback if still too narrow due to proximity to hard bounds.
        if (pub[i] - plb[i]) < 0.05 * width[i]:
            plb[i] = lb[i]
            pub[i] = ub[i]

    bads = BADS(objective, x0, lb, ub, plb, pub)
    if int(max_iter) > 0:
        try:
            bads.options["max_iter"] = int(max_iter)
        except Exception:
            pass
    if int(max_fun_evals) > 0:
        try:
            bads.options["max_fun_evals"] = int(max_fun_evals)
        except Exception:
            pass

    res = bads.optimize()
    x_hat = np.asarray(res.x, dtype=float).ravel()
    rec_d = float(x_hat[0])
    if fixed_theta is not None:
        rec_theta = float(fixed_theta)
        noise_idx = 1
    else:
        rec_theta = float(x_hat[1])
        noise_idx = 2

    if fixed_sigma is not None:
        rec_sigma = float(fixed_sigma)
        rec_mu = float(rec_sigma) / max(float(rec_d), 1e-300)
    else:
        if npmode == "sigma":
            rec_sigma = float(x_hat[noise_idx])
            rec_mu = float(rec_sigma) / max(float(rec_d), 1e-300)
        else:
            rec_mu = float(x_hat[noise_idx])
            rec_sigma = _sigma_from_mu(d=float(rec_d), mu=float(rec_mu))

    rec = ADDMParameters(d=float(rec_d), theta=float(rec_theta), sigma=float(rec_sigma))
    fval = float(getattr(res, "fval", np.nan))
    return rec, rec_mu, fval


def _run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """One (combo, rep) recovery run."""
    config: FitConfig = _G["config"]
    df_trials_b: pd.DataFrame = _G["df_trials_b"]

    combo = int(task["combo"])
    rep = int(task["rep"])
    seed_fit = int(task["seed_fit"])
    seed_sim = int(task["seed_sim"])

    true_params = ADDMParameters(d=float(task["d_true"]), theta=float(task["theta_true"]), sigma=float(task["sigma_true"]))
    true_mu = float(task.get("mu_true", float(true_params.sigma) / max(float(true_params.d), 1e-300)))

    # Censoring horizon should match the time metric we are modeling.
    time_col = str(config.time_col)
    max_time_ms = float(np.nanmax(pd.to_numeric(df_trials_b[time_col], errors="coerce").to_numpy(dtype=float)))

    df_sim = simulate_trials_generative(
        df_trials=df_trials_b,
        params=true_params,
        config=config,
        components=FittingComponents(
            v_edges=_G["v_edges"],
            gaze_by_bin=_G["gaze_by_bin"],
            rt_edges_by_cell={},
            max_time_ms=max_time_ms,
        ),
        n_sim_per_trial=1,
        seed=seed_sim,
    )

    # Merge simulated outcomes back into templates.
    df_synth = df_trials_b.merge(
        df_sim[["subject", "game", "trial_number", "accept_sim", "rt_ms_sim"]],
        on=["subject", "game", "trial_number"],
        how="inner",
    ).copy()

    df_synth["accept"] = df_synth["accept_sim"].astype(int)
    df_synth["rt_ms"] = df_synth["rt_ms_sim"].astype(float)

    # If fitting fixation time, ensure that column is set from the simulated time.
    if str(config.time_col) != "rt_ms":
        df_synth[str(config.time_col)] = df_synth["rt_ms"].astype(float)

    bounds = task["bounds"]
    init = task["init"]

    # Optional randomization of the starting point to reduce corner/local-optimum failures.
    # Sample away from the hard bounds to avoid brittle initialization.
    if bool(task.get("randomize_init", False)):
        rrng = np.random.default_rng(int(seed_fit) + 202)

        def _interior(lo: float, hi: float) -> tuple[float, float]:
            lo = float(lo)
            hi = float(hi)
            width = hi - lo
            if not np.isfinite(width) or width <= 0:
                return lo, hi
            m = 0.01 * width
            return lo + m, hi - m

        d_lo, d_hi = _interior(bounds["d_min"], bounds["d_max"])
        t_lo, t_hi = _interior(bounds["theta_min"], bounds["theta_max"])
        nmode = str(task.get("noise_param", "sigma")).lower().strip()
        if nmode == "mu":
            s_lo, s_hi = _interior(bounds["mu_min"], bounds["mu_max"])
        else:
            s_lo, s_hi = _interior(bounds["sigma_min"], bounds["sigma_max"])

        # Sample d0/sigma0 log-uniform; theta0 uniform.
        def _logu(lo: float, hi: float) -> float:
            lo = float(lo)
            hi = float(hi)
            if lo <= 0 or hi <= 0:
                return float(rrng.uniform(lo, hi))
            return float(np.exp(rrng.uniform(np.log(lo), np.log(hi))))

        init = {"d0": _logu(d_lo, d_hi), "theta0": float(rrng.uniform(t_lo, t_hi))}
        if nmode == "mu":
            init["mu0"] = _logu(s_lo, s_hi)
        else:
            init["sigma0"] = _logu(s_lo, s_hi)

    start = time.time()
    try:
        rec, rec_mu, fval = _fit_one(
            df_synth=df_synth,
            seed_fit=seed_fit,
            bounds=bounds,
            init=init,
            noise_param=str(task.get("noise_param", "sigma")),
            max_iter=int(task["max_iter"]),
            max_fun_evals=int(task["max_fun_evals"]),
            fixed_sigma=(float(task["fixed_sigma"]) if task.get("fixed_sigma") is not None else None),
            fixed_theta=(float(task["fixed_theta"]) if task.get("fixed_theta") is not None else None),
        )
    except Exception as exc:
        import warnings
        warnings.warn(f"combo={combo} rep={rep} failed: {exc!r}")
        elapsed = float(time.time() - start)
        ms_per_s = 1000.0
        sqrt_ms_per_s = float(np.sqrt(ms_per_s))
        return {
            "combo": combo,
            "rep": rep,
            "seed_fit": seed_fit,
            "seed_sim": seed_sim,
            "noise_param": str(task.get("noise_param", "sigma")),
            "d_true": float(true_params.d),
            "theta_true": float(true_params.theta),
            "sigma_true": float(true_params.sigma),
            "mu_true": float(true_mu),
            "d_hat": float("nan"),
            "theta_hat": float("nan"),
            "sigma_hat": float("nan"),
            "mu_hat": float("nan"),
            "d_true_per_s": float(true_params.d) * ms_per_s,
            "sigma_true_per_s": float(true_params.sigma) * sqrt_ms_per_s,
            "d_hat_per_s": float("nan"),
            "sigma_hat_per_s": float("nan"),
            "fit_fval": float("nan"),
            "elapsed_s": elapsed,
            "hit_d_min": 0,
            "hit_d_max": 0,
            "hit_theta_min": 0,
            "hit_theta_max": 0,
            "hit_sigma_min": 0,
            "hit_sigma_max": 0,
        }
    elapsed = float(time.time() - start)

    # Convenience per-second values for d/sigma.
    ms_per_s = 1000.0
    sqrt_ms_per_s = float(np.sqrt(ms_per_s))

    # Bound-hit flags for diagnostics
    d_min = float(bounds["d_min"])
    d_max = float(bounds["d_max"])
    theta_min = float(bounds["theta_min"])
    theta_max = float(bounds["theta_max"])

    tol_rel = 2e-3
    tol_abs_d = 1e-8
    tol_abs_theta = 1e-8

    d_span = max(d_max - d_min, 0.0)
    theta_span = max(theta_max - theta_min, 0.0)
    d_tol = max(tol_abs_d, tol_rel * d_span)
    theta_tol = max(tol_abs_theta, tol_rel * theta_span)

    hit_d_min = bool(rec.d <= d_min + d_tol)
    hit_d_max = bool(rec.d >= d_max - d_tol)
    if task.get("fixed_theta") is not None:
        hit_theta_min = False
        hit_theta_max = False
    else:
        hit_theta_min = bool(rec.theta <= theta_min + theta_tol)
        hit_theta_max = bool(rec.theta >= theta_max - theta_tol)
    if task.get("fixed_sigma") is not None:
        hit_sigma_min = False
        hit_sigma_max = False
    else:
        nmode = str(task.get("noise_param", "sigma")).lower().strip()
        if nmode == "mu":
            mu_min = float(bounds["mu_min"])
            mu_max = float(bounds["mu_max"])
            mu_span = max(mu_max - mu_min, 0.0)
            tol_abs_mu = 1e-8
            mu_tol = max(tol_abs_mu, tol_rel * mu_span)
            hit_sigma_min = bool(rec_mu <= mu_min + mu_tol)
            hit_sigma_max = bool(rec_mu >= mu_max - mu_tol)
        else:
            sigma_min = float(bounds["sigma_min"])
            sigma_max = float(bounds["sigma_max"])
            sigma_span = max(sigma_max - sigma_min, 0.0)
            tol_abs_sigma = 1e-8
            sigma_tol = max(tol_abs_sigma, tol_rel * sigma_span)
            hit_sigma_min = bool(rec.sigma <= sigma_min + sigma_tol)
            hit_sigma_max = bool(rec.sigma >= sigma_max - sigma_tol)

    return {
        "combo": combo,
        "rep": rep,
        "seed_fit": seed_fit,
        "seed_sim": seed_sim,
        "noise_param": str(task.get("noise_param", "sigma")),
        "d_true": float(true_params.d),
        "theta_true": float(true_params.theta),
        "sigma_true": float(true_params.sigma),
        "mu_true": float(true_mu),
        "d_hat": float(rec.d),
        "theta_hat": float(rec.theta),
        "sigma_hat": float(rec.sigma),
        "mu_hat": float(rec_mu),
        "d_true_per_s": float(true_params.d) * ms_per_s,
        "sigma_true_per_s": float(true_params.sigma) * sqrt_ms_per_s,
        "d_hat_per_s": float(rec.d) * ms_per_s,
        "sigma_hat_per_s": float(rec.sigma) * sqrt_ms_per_s,
        "fit_fval": float(fval),
        "elapsed_s": elapsed,
        "hit_d_min": int(hit_d_min),
        "hit_d_max": int(hit_d_max),
        "hit_theta_min": int(hit_theta_min),
        "hit_theta_max": int(hit_theta_max),
        "hit_sigma_min": int(hit_sigma_min),
        "hit_sigma_max": int(hit_sigma_max),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Large-scale parameter recovery sweep.")

    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional subdirectory name under output/addm/parameter_recovery_sweep to avoid mixing runs.",
    )

    parser.add_argument("--time-col", choices=("fix_ms", "rt_ms"), default="fix_ms")
    parser.add_argument("--include-transition", action="store_true")

    parser.add_argument(
        "--reward-source",
        choices=("true", "recalled"),
        default="true",
        help=(
            "Which item rewards to use when computing V_offer and evidence accumulation. "
            "'true' uses encoding outcomes; 'recalled' uses value-recall transcriptions (missing→true fallback)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to repo-level data directory (only used when --reward-source recalled).",
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

    # Fixation-handling variants (match modeling.run_leave_one_game_out)
    parser.add_argument(
        "--irrelevant-mode",
        choices=("zero", "theta_sumrel", "sumrel"),
        default="zero",
        help="How evidence accumulates during offer-irrelevant item fixations.",
    )
    parser.add_argument(
        "--include-center-fixations",
        action="store_true",
        help=(
            "If set, include center fixations (roi_content=='fixation') as an explicit channel in gaze generation. "
            "Requires *_fixations_for_modeling_withcenter.csv (run: python -m modeling.prepare_fixations_for_modeling --output-dir output --with-center)."
        ),
    )
    parser.add_argument(
        "--center-gaze-mode",
        choices=("separate", "merge_with_irrelevant"),
        default="separate",
        help="Whether to treat center fixations as a separate gaze channel or merge them into the irrelevant channel.",
    )
    parser.add_argument(
        "--center-mode",
        choices=("same_as_irrelevant", "zero", "theta_sumrel", "sumrel"),
        default="same_as_irrelevant",
        help="How evidence accumulates during center fixations (item_index==-1).",
    )

    parser.add_argument("--units", choices=("ms", "s"), default="ms")

    # Sampling space (in --units).
    parser.add_argument("--n-combos", type=int, default=1000)
    parser.add_argument("--reps-per-combo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--d-min", type=float, default=1e-5)
    parser.add_argument("--d-max", type=float, default=5e-3)
    parser.add_argument("--theta-min", type=float, default=0.01)
    parser.add_argument("--theta-max", type=float, default=0.99)
    parser.add_argument("--sigma-min", type=float, default=1e-5)
    parser.add_argument("--sigma-max", type=float, default=5e-3)

    parser.add_argument(
        "--noise-param",
        choices=("sigma", "mu"),
        default="mu",
        help="Parameterize noise as either absolute sigma, or mu where sigma = d*mu.",
    )
    parser.add_argument(
        "--mu-min",
        type=float,
        default=float("nan"),
        help="Sampling bound for mu (only used when --noise-param mu). If unset, derived from sigma_min/d_max.",
    )
    parser.add_argument(
        "--mu-max",
        type=float,
        default=float("nan"),
        help="Sampling bound for mu (only used when --noise-param mu). If unset, derived from sigma_max/d_max.",
    )

    parser.add_argument(
        "--no-log-uniform",
        action="store_true",
        help="If set, sample d and sigma uniformly instead of log-uniform.",
    )

    # Fitting config
    parser.add_argument("--n-v-bins", type=int, default=7)
    parser.add_argument("--rt-bins-max", type=int, default=15)
    parser.add_argument(
        "--rt-bins-fixed",
        type=int,
        default=0,
        help="If >0, force a fixed number of RT bins per (V_offer bin, choice) cell (subject to ties/data).",
    )
    parser.add_argument("--min-trials-per-rt-bin", type=int, default=25)

    parser.add_argument("--n-sim-per-vbin", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1.0)

    # Bounds / init (in --units)
    parser.add_argument("--fit-d-min", type=float, default=1e-5)
    parser.add_argument("--fit-d-max", type=float, default=5e-3)
    parser.add_argument("--fit-theta-min", type=float, default=0.01)
    parser.add_argument("--fit-theta-max", type=float, default=0.99)
    parser.add_argument("--fit-sigma-min", type=float, default=1e-5)
    parser.add_argument("--fit-sigma-max", type=float, default=5e-3)
    parser.add_argument(
        "--fix-sigma",
        type=float,
        default=float("nan"),
        help="If set (finite), fixes sigma to this value during fitting and fits only (d, theta). Units follow --units.",
    )
    parser.add_argument(
        "--fix-theta",
        type=float,
        default=float("nan"),
        help="If set (finite), fixes theta to this value during fitting and drops theta from optimization.",
    )

    parser.add_argument(
        "--fit-mu-min",
        type=float,
        default=float("nan"),
        help="Fit bound for mu (only used when --noise-param mu). If unset, derived from fit_sigma_min/fit_d_max.",
    )
    parser.add_argument(
        "--fit-mu-max",
        type=float,
        default=float("nan"),
        help="Fit bound for mu (only used when --noise-param mu). If unset, derived from fit_sigma_max/fit_d_max.",
    )

    parser.add_argument("--d0", type=float, default=5e-4)
    parser.add_argument("--theta0", type=float, default=0.5)
    parser.add_argument("--sigma0", type=float, default=5e-4)
    parser.add_argument(
        "--mu0",
        type=float,
        default=float("nan"),
        help="Initial guess for mu (only used when --noise-param mu). If unset, derived from sigma0/d0.",
    )

    parser.add_argument("--max-iter", type=int, default=0)
    parser.add_argument("--max-fun-evals", type=int, default=0)

    parser.add_argument("--n-jobs", type=int, default=0, help="Number of parallel processes. 0 = use all cores.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--randomize-init",
        action="store_true",
        help="If set, draws a different (d0, theta0, sigma0) per task within the fit bounds (can reduce local-optimum corner solutions).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    reward_source = str(args.reward_source).strip().lower()
    if reward_source not in {"true", "recalled"}:
        raise ValueError("--reward-source must be one of {'true','recalled'}")
    data_dir = Path(str(args.data_dir)).resolve()
    fixation_data_dir = Path(args.fixation_data_dir).resolve() if args.fixation_data_dir is not None else None
    if reward_source == "recalled" and not data_dir.exists():
        raise FileNotFoundError(f"--data-dir does not exist: {data_dir}")

    out_dir = output_dir / "addm" / "parameter_recovery_sweep"
    if str(args.run_tag).strip():
        out_dir = out_dir / str(args.run_tag).strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_param = str(args.noise_param).lower().strip()
    if noise_param not in ("sigma", "mu"):
        raise ValueError(f"--noise-param must be sigma or mu, got {args.noise_param!r}")

    # Convert sampling bounds into internal units.
    d_min_i, sigma_min_i = _convert_units_to_internal_ms(d=float(args.d_min), sigma=float(args.sigma_min), units=str(args.units))
    d_max_i, sigma_max_i = _convert_units_to_internal_ms(d=float(args.d_max), sigma=float(args.sigma_max), units=str(args.units))

    # Convert provided mu bounds if any.
    d_min_i_for_mu, mu_min_i_in = _convert_units_to_internal_ms_for_mu(d=float(args.d_min), mu=float(args.mu_min), units=str(args.units))
    d_max_i_for_mu, mu_max_i_in = _convert_units_to_internal_ms_for_mu(d=float(args.d_max), mu=float(args.mu_max), units=str(args.units))
    _ = d_min_i_for_mu, d_max_i_for_mu

    # Default mu bounds: keep sigma within [sigma_min, sigma_max] at the top drift (d_max).
    mu_min_i = float(mu_min_i_in) if np.isfinite(float(args.mu_min)) else float(sigma_min_i) / max(float(d_max_i), 1e-300)
    mu_max_i = float(mu_max_i_in) if np.isfinite(float(args.mu_max)) else float(sigma_max_i) / max(float(d_max_i), 1e-300)

    # Theta is unitless.
    theta_min = float(args.theta_min)
    theta_max = float(args.theta_max)

    combos = _sample_parameter_combos(
        n=int(args.n_combos),
        seed=int(args.seed),
        d_min=d_min_i,
        d_max=d_max_i,
        theta_min=theta_min,
        theta_max=theta_max,
        sigma_min=sigma_min_i,
        sigma_max=sigma_max_i,
        mu_min=mu_min_i,
        mu_max=mu_max_i,
        noise_param=noise_param,
        log_uniform=(not bool(args.no_log_uniform)),
    )

    # Convert fit bounds/init into internal units.
    fit_d_min_i, fit_sigma_min_i = _convert_units_to_internal_ms(d=float(args.fit_d_min), sigma=float(args.fit_sigma_min), units=str(args.units))
    fit_d_max_i, fit_sigma_max_i = _convert_units_to_internal_ms(d=float(args.fit_d_max), sigma=float(args.fit_sigma_max), units=str(args.units))
    d0_i, sigma0_i = _convert_units_to_internal_ms(d=float(args.d0), sigma=float(args.sigma0), units=str(args.units))

    # Convert fixed sigma (if any) into internal units.
    _, fixed_sigma_i_in = _convert_units_to_internal_ms(d=float(args.d0), sigma=float(args.fix_sigma), units=str(args.units))
    fixed_sigma_i = float(fixed_sigma_i_in) if np.isfinite(float(args.fix_sigma)) else None
    if fixed_sigma_i is not None and float(fixed_sigma_i) <= 0:
        raise ValueError("--fix-sigma must be > 0")

    fixed_theta = float(args.fix_theta) if np.isfinite(float(args.fix_theta)) else None
    if fixed_theta is not None:
        if not (0.0 <= float(fixed_theta) <= 1.0):
            raise ValueError("--fix-theta must be in [0, 1]")
        if not (float(args.fit_theta_min) <= float(fixed_theta) <= float(args.fit_theta_max)):
            raise ValueError("--fix-theta must be within [--fit-theta-min, --fit-theta-max]")

    # Convert provided mu fit bounds/init if any.
    _, fit_mu_min_i_in = _convert_units_to_internal_ms_for_mu(d=float(args.fit_d_min), mu=float(args.fit_mu_min), units=str(args.units))
    _, fit_mu_max_i_in = _convert_units_to_internal_ms_for_mu(d=float(args.fit_d_max), mu=float(args.fit_mu_max), units=str(args.units))
    _, mu0_i_in = _convert_units_to_internal_ms_for_mu(d=float(args.d0), mu=float(args.mu0), units=str(args.units))

    fit_mu_min_i = float(fit_mu_min_i_in) if np.isfinite(float(args.fit_mu_min)) else float(fit_sigma_min_i) / max(float(fit_d_max_i), 1e-300)
    fit_mu_max_i = float(fit_mu_max_i_in) if np.isfinite(float(args.fit_mu_max)) else float(fit_sigma_max_i) / max(float(fit_d_max_i), 1e-300)
    mu0_i = float(mu0_i_in) if np.isfinite(float(args.mu0)) else float(sigma0_i) / max(float(d0_i), 1e-300)

    if noise_param == "mu":
        # Avoid starting exactly on bounds (common when sigma0/d0 equals sigma_max/d_max).
        margin = 1e-3
        mu0_i = float(np.clip(mu0_i, float(fit_mu_min_i) * (1.0 + margin), float(fit_mu_max_i) * (1.0 - margin)))
        mu_span = float(fit_mu_max_i) - float(fit_mu_min_i)
        # If mu0 is still hugging a bound, bias it back toward the middle.
        if np.isfinite(mu_span) and mu_span > 0:
            frac = (float(mu0_i) - float(fit_mu_min_i)) / mu_span
            if frac < 0.05 or frac > 0.95:
                mu0_i = float(np.sqrt(float(fit_mu_min_i) * float(fit_mu_max_i)))
                mu0_i = float(
                    np.clip(mu0_i, float(fit_mu_min_i) * (1.0 + margin), float(fit_mu_max_i) * (1.0 - margin))
                )

        if not (np.isfinite(mu0_i) and float(fit_mu_min_i) < float(mu0_i) < float(fit_mu_max_i)):
            mu0_i = float(np.sqrt(float(fit_mu_min_i) * float(fit_mu_max_i)))
            mu0_i = float(np.clip(mu0_i, float(fit_mu_min_i) * (1.0 + margin), float(fit_mu_max_i) * (1.0 - margin)))

    bounds = {
        "d_min": float(fit_d_min_i),
        "d_max": float(fit_d_max_i),
        "theta_min": float(args.fit_theta_min),
        "theta_max": float(args.fit_theta_max),
        "sigma_min": float(fit_sigma_min_i),
        "sigma_max": float(fit_sigma_max_i),
        "mu_min": float(fit_mu_min_i),
        "mu_max": float(fit_mu_max_i),
    }
    init = {"d0": float(d0_i), "theta0": float(args.theta0), "sigma0": float(sigma0_i), "mu0": float(mu0_i)}

    reps_per_combo = int(args.reps_per_combo)
    if reps_per_combo < 1:
        raise ValueError("--reps-per-combo must be >= 1")

    # Prepare tasks.
    tasks: List[Dict[str, Any]] = []

    existing: Optional[pd.DataFrame] = None
    out_csv = out_dir / "sweep_runs.csv"
    done_keys: set[Tuple[int, int]] = set()
    if bool(args.resume) and out_csv.exists():
        try:
            existing = pd.read_csv(out_csv)
            for _, r in existing.iterrows():
                done_keys.add((int(r["combo"]), int(r["rep"])))
        except Exception:
            done_keys = set()

    base_seed = int(args.seed)
    for _, row in combos.iterrows():
        combo_id = int(row["combo"])
        for rep in range(1, reps_per_combo + 1):
            if (combo_id, rep) in done_keys:
                continue
            # deterministic per-task seeds
            seed_fit = base_seed + combo_id * 10_000 + rep * 101
            seed_sim = seed_fit + 999
            tasks.append(
                {
                    "combo": combo_id,
                    "rep": rep,
                    "seed_fit": int(seed_fit),
                    "seed_sim": int(seed_sim),
                    "d_true": float(row["d"]),
                    "theta_true": float(row["theta"]),
                    "sigma_true": float(row["sigma"]),
                    "mu_true": float(row.get("mu", float(row["sigma"]) / max(float(row["d"]), 1e-300))),
                    "noise_param": noise_param,
                    "fixed_sigma": fixed_sigma_i,
                    "fixed_theta": fixed_theta,
                    "bounds": bounds,
                    "init": init,
                    "max_iter": int(args.max_iter),
                    "max_fun_evals": int(args.max_fun_evals),
                    "randomize_init": bool(args.randomize_init),
                }
            )

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_tag": str(args.run_tag),
        "n_combos": int(args.n_combos),
        "reps_per_combo": int(args.reps_per_combo),
        "units": str(args.units),
        "sampling_bounds": {
            "d_min": float(args.d_min),
            "d_max": float(args.d_max),
            "theta_min": float(args.theta_min),
            "theta_max": float(args.theta_max),
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "noise_param": noise_param,
            "mu_min": None if not np.isfinite(float(args.mu_min)) else float(args.mu_min),
            "mu_max": None if not np.isfinite(float(args.mu_max)) else float(args.mu_max),
        },
        "sampling_bounds_effective_internal": {
            "d_min": float(d_min_i),
            "d_max": float(d_max_i),
            "theta_min": float(theta_min),
            "theta_max": float(theta_max),
            "sigma_min": float(sigma_min_i),
            "sigma_max": float(sigma_max_i),
            "noise_param": noise_param,
            "mu_min": float(mu_min_i),
            "mu_max": float(mu_max_i),
        },
        "fit_bounds": {
            "d_min": float(args.fit_d_min),
            "d_max": float(args.fit_d_max),
            "theta_min": float(args.fit_theta_min),
            "theta_max": float(args.fit_theta_max),
            "sigma_min": float(args.fit_sigma_min),
            "sigma_max": float(args.fit_sigma_max),
            "fit_mu_min": None if not np.isfinite(float(args.fit_mu_min)) else float(args.fit_mu_min),
            "fit_mu_max": None if not np.isfinite(float(args.fit_mu_max)) else float(args.fit_mu_max),
            "fix_sigma": None if not np.isfinite(float(args.fix_sigma)) else float(args.fix_sigma),
            "fix_theta": None if not np.isfinite(float(args.fix_theta)) else float(args.fix_theta),
        },
        "fit_bounds_effective_internal": {
            "d_min": float(fit_d_min_i),
            "d_max": float(fit_d_max_i),
            "theta_min": float(args.fit_theta_min),
            "theta_max": float(args.fit_theta_max),
            "sigma_min": float(fit_sigma_min_i),
            "sigma_max": float(fit_sigma_max_i),
            "noise_param": noise_param,
            "mu_min": float(fit_mu_min_i),
            "mu_max": float(fit_mu_max_i),
            "fixed_sigma": float(fixed_sigma_i) if fixed_sigma_i is not None else None,
            "fixed_theta": float(fixed_theta) if fixed_theta is not None else None,
        },
        "fit_init": {
            "d0": float(args.d0),
            "theta0": float(args.theta0),
            "sigma0": float(args.sigma0),
            "mu0": None if not np.isfinite(float(args.mu0)) else float(args.mu0),
        },
        "fit_init_effective_internal": {
            "d0": float(d0_i),
            "theta0": float(args.theta0),
            "sigma0": float(sigma0_i),
            "mu0": float(mu0_i),
        },
        "config": {
            "time_col": str(args.time_col),
            "include_transition": bool(args.include_transition),
            "reward_source": str(reward_source),
            "data_dir": str(data_dir),
            "irrelevant_mode": str(args.irrelevant_mode),
            "include_center_fixations": bool(args.include_center_fixations),
            "center_gaze_mode": str(args.center_gaze_mode),
            "center_mode": str(args.center_mode),
            "n_v_bins": int(args.n_v_bins),
            "rt_bins_max": int(args.rt_bins_max),
            "rt_bins_fixed": int(args.rt_bins_fixed),
            "min_trials_per_rt_bin": int(args.min_trials_per_rt_bin),
            "n_sim_per_vbin": int(args.n_sim_per_vbin),
            "alpha": float(args.alpha),
        },
    }
    (out_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2))

    if not tasks:
        print("[OK] Nothing to do (all tasks complete).")
        return

    n_jobs = int(args.n_jobs)
    if n_jobs <= 0:
        n_jobs = int(os.cpu_count() or 1)

    print(f"[INFO] Tasks to run: {len(tasks)} | jobs={n_jobs} | output={out_dir}")

    # Run parallel.
    from multiprocessing import get_context

    ctx = get_context("fork")
    results: List[Dict[str, Any]] = []

    t0 = time.time()
    initargs = (
        str(output_dir),
        str(reward_source),
        str(data_dir),
        str(args.time_col),
        bool(args.include_transition),
        str(args.irrelevant_mode),
        bool(args.include_center_fixations),
        str(args.center_gaze_mode),
        str(args.center_mode),
        int(args.n_v_bins),
        int(args.rt_bins_max),
        int(args.rt_bins_fixed),
        int(args.min_trials_per_rt_bin),
        int(args.n_sim_per_vbin),
        float(args.alpha),
        int(args.seed),
        str(fixation_data_dir) if fixation_data_dir is not None else "",
    )

    with ctx.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=initargs,
    ) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_task, tasks, chunksize=1), start=1):
            results.append(res)
            # incremental write every ~10 results
            if i % 10 == 0 or i == len(tasks):
                df_new = pd.DataFrame(results)
                if out_csv.exists():
                    try:
                        df_old = pd.read_csv(out_csv)
                        df_all = pd.concat([df_old, df_new], ignore_index=True)
                    except Exception:
                        df_all = df_new
                else:
                    df_all = df_new
                # Drop duplicates on (combo, rep) keeping last.
                df_all = df_all.drop_duplicates(subset=["combo", "rep"], keep="last")
                df_all.to_csv(out_csv, index=False)
                results.clear()

            if i % 25 == 0:
                elapsed = time.time() - t0
                rate = i / max(1e-9, elapsed)
                eta = (len(tasks) - i) / max(1e-9, rate)
                print(f"[PROGRESS] {i}/{len(tasks)} done | {rate:.2f} tasks/s | ETA {eta/60:.1f} min")

    print("[OK] Sweep complete")
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()

"""Plot results from parameter_recovery_sweep.

Creates a true vs recovered scatter for each parameter.

Usage
-----

    python -m addm.plot_recovery_sweep \
      --sweep-csv output/addm/parameter_recovery_sweep/sweep_runs.csv

Aggregate across repetitions per combo:

    python -m addm.plot_recovery_sweep \
      --sweep-csv output/addm/parameter_recovery_sweep/sweep_runs.csv \
      --aggregate mean

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise RuntimeError("matplotlib is required for plotting. Install it with `pip install matplotlib`.") from e


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot parameter recovery sweep results.")
    parser.add_argument("--sweep-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--aggregate", choices=("none", "mean", "median"), default="mean")
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument(
        "--log-d-sigma",
        action="store_true",
        help="If set, plot d and sigma on log10 scale (recommended when sampling log-uniform).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small constant for log scaling / clipping.",
    )
    args = parser.parse_args()

    sweep_csv = Path(args.sweep_csv).resolve()
    if not sweep_csv.exists():
        raise FileNotFoundError(str(sweep_csv))

    df = pd.read_csv(sweep_csv)

    has_mu = all(c in df.columns for c in ("mu_true", "mu_hat"))

    if str(args.aggregate) != "none":
        agg = str(args.aggregate)
        g = df.groupby("combo", dropna=False)
        if agg == "mean":
            cols = ["d_true", "theta_true", "sigma_true", "d_hat", "theta_hat", "sigma_hat"]
            if has_mu:
                cols += ["mu_true", "mu_hat"]
            dfp = g[cols].mean().reset_index()
        else:
            cols = ["d_true", "theta_true", "sigma_true", "d_hat", "theta_hat", "sigma_hat"]
            if has_mu:
                cols += ["mu_true", "mu_hat"]
            dfp = g[cols].median().reset_index()
        df = dfp

    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else sweep_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine if bound-hit flags are present.
    has_hits = all(c in df.columns for c in ("hit_d_max", "hit_sigma_min", "hit_theta_min"))

    def maybe_log(p: str, arr: np.ndarray) -> np.ndarray:
        if not bool(args.log_d_sigma):
            return arr
        if p in ("d", "sigma"):
            return np.log10(np.clip(arr, float(args.eps), np.inf))
        return arr

    # --- True vs recovered ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, p in zip(axes, ["d", "theta", "sigma"]):
        x = df[f"{p}_true"].to_numpy(dtype=float)
        y = df[f"{p}_hat"].to_numpy(dtype=float)

        x_plot = maybe_log(p, x)
        y_plot = maybe_log(p, y)

        if has_hits and p == "d":
            m_bad = df["hit_d_max"].to_numpy(dtype=int) == 1
            ax.scatter(x_plot[~m_bad], y_plot[~m_bad], s=float(args.s), alpha=float(args.alpha))
            ax.scatter(x_plot[m_bad], y_plot[m_bad], s=float(args.s), alpha=0.9, color="tab:red", label="hit_d_max")
            ax.legend(loc="lower right")
        elif has_hits and p == "sigma":
            m_bad = df["hit_sigma_min"].to_numpy(dtype=int) == 1
            ax.scatter(x_plot[~m_bad], y_plot[~m_bad], s=float(args.s), alpha=float(args.alpha))
            ax.scatter(x_plot[m_bad], y_plot[m_bad], s=float(args.s), alpha=0.9, color="tab:red", label="hit_sigma_min")
            ax.legend(loc="lower right")
        else:
            ax.scatter(x_plot, y_plot, s=float(args.s), alpha=float(args.alpha))

        lo, hi = _identity_limits(x_plot, y_plot)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        r = _corr(x_plot, y_plot)
        suffix = " (log10)" if (bool(args.log_d_sigma) and p in ("d", "sigma")) else ""
        ax.set_title(f"{p}: true vs recovered (r={r:.3f}){suffix}")
        ax.set_xlabel(f"{p}_true" + suffix)
        ax.set_ylabel(f"{p}_hat" + suffix)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out1 = out_dir / f"recovery_true_vs_hat_{args.aggregate}{'_log' if args.log_d_sigma else ''}.{args.format}"
    fig.savefig(out1, dpi=200)
    plt.close(fig)

    # Print bound-hit summary if available.
    if has_hits:
        hit_summary = {
            "hit_d_max_frac": float(df["hit_d_max"].mean()) if "hit_d_max" in df.columns else float("nan"),
            "hit_sigma_min_frac": float(df["hit_sigma_min"].mean()) if "hit_sigma_min" in df.columns else float("nan"),
            "hit_theta_min_frac": float(df["hit_theta_min"].mean()) if "hit_theta_min" in df.columns else float("nan"),
        }
        print("[INFO] Bound-hit fractions:", hit_summary)

    print("[OK] Wrote:", out1)


if __name__ == "__main__":
    main()

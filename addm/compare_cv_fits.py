"""Compare cross-validation fits across settings.

This script reads one or more CV summary CSVs (typically written by
``run_kfold_cv.py`` and then renamed to encode the setting).

By convention in this project, the *setting* is the suffix after the last
underscore in the filename stem, e.g.:
  addm_kfold_fit_summary_3free.csv  -> setting=3free
  addm_kfold_fit_summary_theta1.csv -> setting=theta1

It merges per-fold held-out log-likelihoods across settings and writes
summary tables (per-setting means and a wide per-fold table).

Statistical comparison of models is handled separately by the brms script
``addm/run_mixed_effects_addm_comparison.R``.

The fold identifier column is ``heldout_game``.

Higher log-likelihood is better.

Example
-------
python -m addm.compare_cv_fits \
  --cv-dir output/addm/kfold \
  --out-dir output/addm/kfold_compare

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _parse_setting_from_filename(path: Path) -> str:
    # User convention: setting is suffix after last underscore in the stem.
    parts = path.stem.split("_")
    return parts[-1] if parts else path.stem


def _read_cv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"heldout_game", "loglik_test"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    # Normalize types.
    out = df.copy()
    out["heldout_game"] = pd.to_numeric(out["heldout_game"], errors="coerce")
    out["loglik_test"] = pd.to_numeric(out["loglik_test"], errors="coerce")
    if "n_test_trials" in out.columns:
        out["n_test_trials"] = pd.to_numeric(out["n_test_trials"], errors="coerce")
    if "mode" in out.columns:
        out["mode"] = out["mode"].astype(str)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare held-out log-likelihoods across model settings.")
    parser.add_argument("--cv-dir", "--logo-dir", type=str, default="output/addm/kfold", help="Directory containing CV summary CSVs")
    parser.add_argument(
        "--glob",
        type=str,
        default="*.csv",
        help="Glob pattern under --cv-dir for candidate CSVs (default: *.csv)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fit",
        choices=("fit", "eval-only", "any"),
        help="Which mode rows to include (default: fit).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/addm/kfold_compare",
        help="Output directory for comparison tables/plots",
    )

    args = parser.parse_args()

    cv_dir = Path(args.cv_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(cv_dir.glob(str(args.glob)))
    if not paths:
        raise FileNotFoundError(f"No files match {args.glob!r} in {cv_dir}")

    dfs: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = _read_cv_csv(p)
        except Exception:
            continue

        setting = _parse_setting_from_filename(p)
        df = df.copy()
        df["setting"] = setting

        if args.mode != "any" and "mode" in df.columns:
            df = df[df["mode"].astype(str) == str(args.mode)].copy()

        # Keep one row per held-out game.
        df = (
            df.groupby(["setting", "heldout_game"], as_index=False)
            .agg(
                loglik_test=("loglik_test", "mean"),
                n_test_trials=("n_test_trials", "mean") if "n_test_trials" in df.columns else ("loglik_test", "size"),
            )
            .copy()
        )

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No readable CV summary CSVs found (must have heldout_game, loglik_test columns).")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["loglik_per_trial"] = df_all["loglik_test"] / df_all["n_test_trials"].replace(0, np.nan)

    # Order settings by performance.
    means = df_all.groupby("setting")["loglik_test"].mean().sort_values(ascending=False)
    setting_order = means.index.astype(str).tolist()

    # Wide table: rows=game, cols=settings (ordered best→worst).
    df_wide = df_all.pivot(index="heldout_game", columns="setting", values="loglik_test").reset_index()
    ordered_cols = ["heldout_game"] + [c for c in setting_order if c in set(df_wide.columns)]
    df_wide = df_wide[ordered_cols].copy()

    # Summaries per setting.
    summary_rows: List[Dict[str, float | str]] = []
    for setting in sorted(df_all["setting"].unique().tolist()):
        d = df_all[df_all["setting"] == setting]
        ll = d["loglik_test"].to_numpy(dtype=float)
        ll_pt = d["loglik_per_trial"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "setting": setting,
                "n_games": int(d.shape[0]),
                "mean_loglik_test": float(np.nanmean(ll)),
                "mean_loglik_per_trial": float(np.nanmean(ll_pt)),
            }
        )

    df_summary = pd.DataFrame(summary_rows).sort_values("mean_loglik_test", ascending=False)
    df_summary.to_csv(out_dir / "cv_compare_summary.csv", index=False)

    # Write the wide per-game table.
    df_wide.to_csv(out_dir / "cv_compare_by_game_wide.csv", index=False)

    # Console output.
    print("[OK] Read settings (best→worst):", ", ".join([s for s in setting_order if s in set(df_all["setting"].unique().tolist())]))
    print("\nSummary (mean held-out LL):")
    print(df_summary.to_string(index=False))

    print("\nWrote:")
    for name in [
        "cv_compare_summary.csv",
        "cv_compare_by_game_wide.csv",
    ]:
        p = out_dir / name
        if p.exists():
            print(" -", p)


if __name__ == "__main__":
    main()

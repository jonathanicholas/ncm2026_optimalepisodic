import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.tools import add_constant
from statsmodels.api import OLS


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_subjects(data_root: str) -> List[str]:
    subs = []
    for d in os.listdir(data_root):
        if not os.path.isdir(os.path.join(data_root, d)):
            continue
        # subject folders look numeric (e.g., 101, 102, ...)
        if re.fullmatch(r"\d+", d):
            subs.append(d)
    subs.sort()
    return subs


def load_main_logfile(subid: str, data_root: str) -> Optional[pd.DataFrame]:
    path = os.path.join(data_root, subid, f"{subid}_MAIN_logfile_7.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Normalize types
    for col in ["game", "outcome", "choice", "correct", "rt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # String normalize
    for col in ["phase", "event", "image", "option"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def extract_game_items(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    enc = df[(df["phase"] == "encoding") & (df["event"] == "value")]
    enc = enc[["game", "image", "outcome"]].dropna(subset=["game", "image"]).copy()
    game_items: Dict[int, pd.DataFrame] = {}
    for g, gdf in enc.groupby("game"):
        # keep first occurrence per image
        gdf = gdf.drop_duplicates(subset=["image"])  # defensive
        game_items[int(g)] = gdf.reset_index(drop=True)
    return game_items


def extract_memory_value_order(df: pd.DataFrame) -> Dict[int, List[str]]:
    mem = df[(df["phase"] == "memory") & (df["event"] == "value_recall")]
    mem = mem[["game", "image", "onset"]].dropna(subset=["game"]).copy()
    # Sort by onset within game to preserve order
    mem = mem.sort_values(["game", "onset"]) if "onset" in mem.columns else mem
    order: Dict[int, List[str]] = {}
    for g, gdf in mem.groupby("game"):
        order[int(g)] = list(gdf["image"].astype(str))
    return order


def load_freerecall(subid: str, data_root: str) -> Optional[pd.DataFrame]:
    path = os.path.join(data_root, subid, f"{subid}_freerecall.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for col in ["game"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["item"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_valuerecall(subid: str, data_root: str) -> Optional[pd.DataFrame]:
    path = os.path.join(data_root, subid, f"{subid}_valuerecall.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for col in ["game"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # normalize tokens in original_item and item columns
    for col in ["original_item", "item"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def parse_sign_token(tok: str) -> Optional[int]:
    if tok is None:
        return None
    t = str(tok).strip().lower()
    if t in {"plus", "+", "positive", "pos"}:
        return 1
    if t in {"minus", "-", "negative", "neg"}:
        return -1
    # Sometimes the first row could be the magnitude accidentally
    if t.isdigit():
        return None
    return None


def parse_magnitude_token(tok: str) -> Optional[int]:
    if tok is None:
        return None
    t = str(tok).strip()
    if t.isdigit():
        v = int(t)
        if 1 <= v <= 9:
            return v
    return None


def build_recalled_values_map(
    df_main: pd.DataFrame,
    df_val: pd.DataFrame
) -> Dict[int, Dict[str, Optional[int]]]:
    order = extract_memory_value_order(df_main)
    recalled_map: Dict[int, Dict[str, Optional[int]]] = {}
    for g, gdf in df_val.groupby("game"):
        g = int(g)
        # Expect 12 rows per game: sign, magnitude alternating for 6 items.
        rows = gdf.reset_index(drop=True)
        vals: List[Optional[int]] = []
        i = 0
        while i < len(rows):
            sign_tok = rows.loc[i, "original_item"] if "original_item" in rows.columns else rows.loc[i, "item"]
            mag_tok = None
            if i + 1 < len(rows):
                mag_tok = rows.loc[i + 1, "original_item"] if "original_item" in rows.columns else rows.loc[i + 1, "item"]
            sign = parse_sign_token(sign_tok)
            mag = parse_magnitude_token(mag_tok) if mag_tok is not None else None
            if sign is not None and mag is not None:
                vals.append(sign * mag)
                i += 2
            else:
                # Handle missing or malformed pairs: try reverse (mag first then sign)
                sign_rev = parse_sign_token(mag_tok) if mag_tok is not None else None
                mag_rev = parse_magnitude_token(sign_tok)
                if sign_rev is not None and mag_rev is not None:
                    vals.append(sign_rev * mag_rev)
                    i += 2
                else:
                    # Single empty row -> missing
                    vals.append(np.nan)
                    i += 1
        # Align with memory order images
        images = order.get(g, [])
        img_to_val: Dict[str, Optional[int]] = {}
        for idx, img in enumerate(images):
            v = vals[idx] if idx < len(vals) else np.nan
            img_to_val[str(img)] = v
        recalled_map[g] = img_to_val
    return recalled_map


def compute_true_offer_value(option: str, game_items_df: pd.DataFrame) -> int:
    token = str(option).strip()
    # Sum outcomes for items whose image contains the token among underscore-separated parts
    mask = game_items_df["image"].astype(str).str.split("_").apply(lambda parts: token in parts)
    return int(np.nansum(game_items_df.loc[mask, "outcome"].astype(float)))


def build_subject_choice_dataset(
    subid: str,
    data_root: str
) -> Optional[pd.DataFrame]:
    df_main = load_main_logfile(subid, data_root)
    if df_main is None:
        return None
    game_items = extract_game_items(df_main)
    df_free = load_freerecall(subid, data_root)
    df_val = load_valuerecall(subid, data_root)
    recalled_map = build_recalled_values_map(df_main, df_val) if df_val is not None else {}

    # Filter free recall to items actually shown in game
    free_by_game: Dict[int, List[str]] = {}
    if df_free is not None:
        for g, gdf in df_free.groupby("game"):
            g = int(g)
            shown = set(game_items.get(g, pd.DataFrame({"image": []}))[["image"]].astype(str)["image"].tolist())
            # Keep only items from this game, and deduplicate in case of repeated recalls
            recalled_items = [itm for itm in gdf["item"].astype(str) if itm in shown]
            free_by_game[g] = sorted(set(recalled_items))

    choices = df_main[(df_main["phase"] == "choice") & (df_main["event"] == "choice")].copy()
    # Normalize choice: 1->1 (take), 2->0 (leave)
    choices["choice_bin"] = choices["choice"].replace({2: 0, 1: 1}).astype(float)
    # Some rows may have missing rt/correct
    choices["rt"] = pd.to_numeric(choices["rt"], errors="coerce")
    choices["correct"] = pd.to_numeric(choices["correct"], errors="coerce")
    rows = []
    for _, row in choices.iterrows():
        g = int(row["game"]) if not pd.isna(row["game"]) else None
        option = row.get("option", None)
        if g is None or option is None or pd.isna(option):
            continue
        items_df = game_items.get(g)
        if items_df is None:
            continue
        true_val = compute_true_offer_value(option, items_df)
        # recalled offer value: sum of recalled values of free-recalled items relevant to offer
        recalled_total = len(free_by_game.get(g, []))
        # Offer-relevant recalled items (deduplicated by construction), should be <= 3 by design
        recalled_offer_items = [
            itm for itm in free_by_game.get(g, []) if option in str(itm).split("_")
        ]
        # Cap to 3 (max offer-relevant items per trial by design)
        if len(recalled_offer_items) > 3:
            recalled_offer_items = recalled_offer_items[:3]
        # Sum recalled values for those items when available
        rec_vals_map = recalled_map.get(g, {})
        rec_vals = [rec_vals_map.get(itm, np.nan) for itm in recalled_offer_items]
        rec_vals = [v for v in rec_vals if not pd.isna(v)]
        recalled_offer_val = np.nan if len(rec_vals) == 0 else float(np.sum(rec_vals))
        rows.append({
            "subject": subid,
            "game": g,
            "option": str(option),
            "choice": float(row["choice_bin"]),
            "correct": float(row["correct"]) if not pd.isna(row["correct"]) else np.nan,
            "rt": float(row["rt"]) if not pd.isna(row["rt"]) else np.nan,
            "true_offer_value": float(true_val),
            "recalled_offer_value": recalled_offer_val,
            "recalled_total_count": int(recalled_total),
            "recalled_offer_count": int(len(recalled_offer_items)),
        })
    return pd.DataFrame(rows)


def build_subject_item_values_dataset(
    subid: str,
    data_root: str
) -> Optional[pd.DataFrame]:
    df_main = load_main_logfile(subid, data_root)
    if df_main is None:
        return None
    df_val = load_valuerecall(subid, data_root)
    if df_val is None:
        return None
    game_items = extract_game_items(df_main)
    recalled_map = build_recalled_values_map(df_main, df_val)
    rows = []
    for g, items_df in game_items.items():
        for _, r in items_df.iterrows():
            img = str(r["image"])
            true_val = float(r["outcome"]) if not pd.isna(r["outcome"]) else np.nan
            rec_val = recalled_map.get(g, {}).get(img, np.nan)
            rows.append({
                "subject": subid,
                "game": int(g),
                "image": img,
                "true_value": true_val,
                "recalled_value": float(rec_val) if not pd.isna(rec_val) else np.nan,
            })
    return pd.DataFrame(rows)


def build_subject_spatial_recall_dataset(
    subid: str,
    data_root: str
) -> Optional[pd.DataFrame]:
    df_main = load_main_logfile(subid, data_root)
    if df_main is None:
        return None
    d = df_main[(df_main["phase"] == "memory") & (df_main["event"] == "spatial_recall")].copy()
    if len(d) == 0:
        return pd.DataFrame([])
    d = d.dropna(subset=["true_position", "recalled_position"]).copy()
    if len(d) == 0:
        return pd.DataFrame([])
    d["spatial_correct"] = (d["true_position"] == d["recalled_position"]).astype(float)
    rows = []
    for _, row in d.iterrows():
        g = int(row["game"]) if "game" in d.columns and not pd.isna(row.get("game")) else np.nan
        rows.append({
            "subject": subid,
            "game": g,
            "spatial_correct": float(row["spatial_correct"]),
        })
    return pd.DataFrame(rows)


def fit_logistic(x: np.ndarray, y: np.ndarray) -> Optional[GLM]:
    # Require variability
    if len(np.unique(y)) < 2:
        return None
    X = add_constant(pd.Series(x))
    try:
        model = GLM(y, X, family=Binomial()).fit()
        return model
    except Exception:
        return None


def fit_ols(x: np.ndarray, y: np.ndarray) -> Optional[OLS]:
    if len(x) < 2:
        return None
    X = add_constant(pd.Series(x))
    try:
        model = OLS(y, X).fit()
        return model
    except Exception:
        return None


def ci95_mean(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    m = float(np.mean(values))
    se = float(np.std(values, ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return (m, 1.96 * se)


def export_stats_dataframes(
    df_all: pd.DataFrame,
    df_items_all: Optional[pd.DataFrame],
    df_spatial_all: Optional[pd.DataFrame],
    stats_dir: str,
):
    """Write all data needed for R mixed-effects fits to CSVs in stats_dir."""
    # Full choices dataset
    df_all.to_csv(os.path.join(stats_dir, "choices_all.csv"), index=False)

    # 1) accuracy dataset
    d1 = df_all.dropna(subset=["correct"]).copy()
    d1["correct_bin"] = d1["correct"].astype(int)
    d1[["subject", "correct_bin", "game", "rt", "true_offer_value", "recalled_offer_value", "recalled_total_count", "recalled_offer_count"]].to_csv(
        os.path.join(stats_dir, "accuracy_df.csv"), index=False
    )

    # 2) true offer value logistic
    d2 = df_all.dropna(subset=["choice", "true_offer_value"]).copy()
    d2["choice_bin"] = d2["choice"].astype(int)
    d2[["subject", "choice_bin", "true_offer_value", "game", "rt", "recalled_total_count", "recalled_offer_count"]].to_csv(
        os.path.join(stats_dir, "choice_true_value_df.csv"), index=False
    )

    # 3) recalled offer value logistic
    d3 = df_all.dropna(subset=["choice", "recalled_offer_value"]).copy()
    d3["choice_bin"] = d3["choice"].astype(int)
    d3[["subject", "choice_bin", "recalled_offer_value", "game", "rt", "recalled_total_count", "recalled_offer_count"]].to_csv(
        os.path.join(stats_dir, "choice_recalled_value_df.csv"), index=False
    )

    # 4a) RT vs total recall
    d4a = df_all.dropna(subset=["rt", "recalled_total_count"]).copy()
    d4a = d4a[d4a["rt"] > 0]
    d4a["log_rt"] = np.log(d4a["rt"])
    d4a[["subject", "log_rt", "recalled_total_count", "game"]].to_csv(
        os.path.join(stats_dir, "rt_total_recall_df.csv"), index=False
    )

    # 4b) RT vs offer-relevant recall
    d4b = df_all.dropna(subset=["rt", "recalled_offer_count"]).copy()
    d4b = d4b[d4b["rt"] > 0]
    d4b["log_rt"] = np.log(d4b["rt"])
    d4b[["subject", "log_rt", "recalled_offer_count", "game", "option"]].to_csv(
        os.path.join(stats_dir, "rt_offer_recall_df.csv"), index=False
    )

    # 5) recall proportion centered per game
    d5 = df_all.groupby(["subject", "game"]).agg({"recalled_total_count": "first"}).reset_index()
    d5["recall_prop_centered"] = d5["recalled_total_count"].astype(float) / 6.0 - (6.0 / 16.0)
    d5[["subject", "game", "recall_prop_centered", "recalled_total_count"]].to_csv(
        os.path.join(stats_dir, "recall_prop_df.csv"), index=False
    )

    # 6) item-level true vs recalled values
    if df_items_all is not None and len(df_items_all) > 0:
        df_items_all[["subject", "game", "image", "true_value", "recalled_value"]].to_csv(
            os.path.join(stats_dir, "item_values_df.csv"), index=False
        )

    # 7) spatial recall accuracy (location accuracy) per trial
    if df_spatial_all is not None and len(df_spatial_all) > 0:
        d_sp = df_spatial_all.copy()
        d_sp["spatial_correct_bin"] = d_sp["spatial_correct"].astype(int)
        d_sp[["subject", "game", "spatial_correct_bin"]].to_csv(
            os.path.join(stats_dir, "spatial_accuracy_df.csv"), index=False
        )


def export_summary_stats(df_all: pd.DataFrame, df_spatial_all: Optional[pd.DataFrame], stats_dir: str):
    """Compute and save simple summary stats (group mean and SE) for key measures."""

    rows = []

    # Choice accuracy
    subj_acc = df_all.groupby("subject")["correct"].mean()
    vals = subj_acc.values.astype(float)
    vals = vals[~np.isnan(vals)]
    if len(vals) > 0:
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        rows.append({
            "measure": "choice_accuracy",
            "group_mean": m,
            "group_se": se,
            "n_subjects": int(len(vals)),
        })

    # Item recall rate (proportion of 6 items)
    sub_game_counts = df_all.groupby(["subject", "game"]).agg({"recalled_total_count": "first"}).reset_index()
    subj_avg = sub_game_counts.groupby("subject")["recalled_total_count"].mean()
    prop = subj_avg / 6.0
    vals = prop.values.astype(float)
    vals = vals[~np.isnan(vals)]
    if len(vals) > 0:
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        rows.append({
            "measure": "item_recall_rate",
            "group_mean": m,
            "group_se": se,
            "n_subjects": int(len(vals)),
        })

    # Location recall rate (spatial accuracy), if available
    if df_spatial_all is not None and len(df_spatial_all) > 0:
        subj_spatial = df_spatial_all.groupby("subject")["spatial_correct"].mean()
        vals = subj_spatial.values.astype(float)
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            rows.append({
                "measure": "location_recall_rate",
                "group_mean": m,
                "group_se": se,
                "n_subjects": int(len(vals)),
            })

    if len(rows) > 0:
        summary_df = pd.DataFrame(rows)
        out_path = os.path.join(stats_dir, "summary_stats.csv")
        summary_df.to_csv(out_path, index=False)
        print("\nSummary statistics (group mean ± SE):")
        print(summary_df.to_string(index=False))


def compile_figure2(
    df_all: pd.DataFrame,
    df_spatial_all: Optional[pd.DataFrame],
    df_items_all: Optional[pd.DataFrame],
    out_dir: str,
):
    """Create Figure 2 (2x2) and save as PDF.

    Layout:
    Row 1: A) choice accuracy,  B) logistic choice ~ standardized offer values
    Row 2: C) RT vs # memories, D) RT vs standardized offer value
    """

    sns.set_context("poster")
    with plt.rc_context({
        "font.family": "Arial",
        "axes.titlesize": 24,
        "axes.labelsize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
    }):

        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(2, 1)

        # Top row: A (narrow), B logistic (medium), new plot (narrow)
        gs_top = gs[0].subgridspec(1, 3, width_ratios=[0.35, 1.0, 0.3], wspace=0.8) #0.6
        # Bottom row: C (medium), D (wider) — unchanged
        gs_bot = gs[1].subgridspec(1, 2, width_ratios=[1, 1.8], wspace=0.4)

        # --- Panel A: choice accuracy ---
        ax1 = fig.add_subplot(gs_top[0, 0])
        subj_acc = df_all.groupby("subject")["correct"].mean()
        m1, err1 = ci95_mean(subj_acc.values)
        sns.stripplot(
            x=["Accuracy"] * len(subj_acc),
            y=subj_acc.values,
            color="gray",
            alpha=0.5,
            size=12,
            jitter=0.05,
            ax=ax1,
            zorder=0
        )
        ax1.errorbar(
            [0],
            [m1],
            yerr=[err1],
            fmt="none",
            ecolor="black",
            capsize=0,
        )
        ax1.scatter(
            [0],
            [m1],
            s=14**2,
            facecolor=".5",
            edgecolor="black",
            linewidth=2.5,
            zorder=3,
        )
        ax1.set_ylabel("Choice Accuracy")
        ax1.set_ylim(0, 1.05)
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_xticks([])
        ax1.set_xlim(-0.075, 0.075)
        ax1.plot((-0.075, 0.075), (0.5, 0.5), 'k:')
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        # --- Panel B: logistic choice ~ standardized offer values ---
        ax2 = fig.add_subplot(gs_top[0, 1])

        true_color = "#81bf78"
        rec_color = "#b08dc1"

        d_true = df_all.dropna(subset=["true_offer_value", "choice"]).copy()
        d_rec = df_all.dropna(subset=["recalled_offer_value", "choice"]).copy()

        if len(d_true) > 0:
            mu_true = float(d_true["true_offer_value"].mean())
            sd_true = float(d_true["true_offer_value"].std(ddof=0))
            d_true["true_z"] = 0.0 if sd_true <= 0 else (d_true["true_offer_value"] - mu_true) / sd_true

        if len(d_rec) > 0:
            mu_rec = float(d_rec["recalled_offer_value"].mean())
            sd_rec = float(d_rec["recalled_offer_value"].std(ddof=0))
            d_rec["rec_z"] = 0.0 if sd_rec <= 0 else (d_rec["recalled_offer_value"] - mu_rec) / sd_rec

        z_mins = []
        z_maxs = []
        if len(d_true) > 0:
            z_mins.append(float(d_true["true_z"].min()))
            z_maxs.append(float(d_true["true_z"].max()))
        if len(d_rec) > 0:
            z_mins.append(float(d_rec["rec_z"].min()))
            z_maxs.append(float(d_rec["rec_z"].max()))

        if len(z_mins) > 0:
            z_min = min(z_mins)
            z_max = max(z_maxs)
            grid = np.linspace(z_min, z_max, 100)

            if len(d_true) > 0:
                for sid, sdf in d_true.groupby("subject"):
                    model = fit_logistic(sdf["true_z"].values, sdf["choice"].values)
                    if model is None:
                        continue
                    pred = model.predict(add_constant(pd.Series(grid)))
                    ax2.plot(grid, pred, color=true_color, alpha=0.2, linewidth=1, zorder=0)

                gmodel_t = fit_logistic(d_true["true_z"].values, d_true["choice"].values)
                if gmodel_t is not None:
                    gpred_t = gmodel_t.predict(add_constant(pd.Series(grid)))
                    background_grid = grid.copy()
                    background_grid[0] = grid[0] + 0.025
                    background_grid[-1] = grid[-1] - 0.025
                    ax2.plot(background_grid, gpred_t, color='k', linewidth=6)
                    ax2.plot(grid, gpred_t, color=true_color, linewidth=4)

            if len(d_rec) > 0:
                for sid, sdf in d_rec.groupby("subject"):
                    model = fit_logistic(sdf["rec_z"].values, sdf["choice"].values)
                    if model is None:
                        continue
                    pred = model.predict(add_constant(pd.Series(grid)))
                    ax2.plot(grid, pred, color=rec_color, alpha=0.2, linewidth=1, zorder=0)

                gmodel_r = fit_logistic(d_rec["rec_z"].values, d_rec["choice"].values)
                if gmodel_r is not None:
                    gpred_r = gmodel_r.predict(add_constant(pd.Series(grid)))
                    background_grid = grid.copy()
                    background_grid[0] = grid[0] + 0.025
                    background_grid[-1] = grid[-1] - 0.025
                    ax2.plot(background_grid, gpred_r, color='k', linewidth=6)
                    ax2.plot(grid, gpred_r, color=rec_color, linewidth=4)

        # Add scatter plots for individual choices
        # For choices = 1 (top of plot)
        choices_1_true = d_true[d_true["choice"] == 1]
        choices_1_rec = d_rec[d_rec["choice"] == 1]
        # Recalled closer to 1, true further above
        ax2.scatter(choices_1_rec["rec_z"],
                    np.random.uniform(1.02, 1.05, size=len(choices_1_rec)),
                    color=rec_color, alpha=0.1, s=25, linewidth=0)
        ax2.scatter(choices_1_true["true_z"],
                    np.random.uniform(1.07, 1.1, size=len(choices_1_true)),
                    color=true_color, alpha=0.1, s=25, linewidth=0)

        # For choices = 0 (bottom of plot)
        choices_0_true = d_true[d_true["choice"] == 0]
        choices_0_rec = d_rec[d_rec["choice"] == 0]
        # Recalled closer to 0, true further below
        ax2.scatter(choices_0_rec["rec_z"],
                    np.random.uniform(-0.05, -0.02, size=len(choices_0_rec)),
                    color=rec_color, alpha=0.1, s=25, linewidth=0)
        ax2.scatter(choices_0_true["true_z"],
                    np.random.uniform(-0.1, -0.07, size=len(choices_0_true)),
                    color=true_color, alpha=0.1, s=25, linewidth=0)

        ax2.set_ylabel("Proportion Offers Taken")
        ax2.set_xlabel("Standardized Offer Value (z)")
        ax2.set_ylim(-0.12, 1.12)
        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax2.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax2.set_xlim((-5, 5))
        ax2.set_xticks([-5, -2.5, 0, 2.5, 5])
        ax2.set_xticklabels([-5, -2.5, 0, 2.5, 5])

        legend_elems = [
            Line2D([0], [0], color=true_color, lw=4, label="True"),
            Line2D([0], [0], color=rec_color, lw=4, label="Recalled"),
        ]
        ax2.legend(handles=legend_elems, loc="lower right", frameon=True,
                   edgecolor="black", fancybox=False,
                   bbox_to_anchor=(1.03, 0.07))

        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        # --- Panel C: ELPD difference (recalled - true) ---
        ax_c_new = fig.add_subplot(gs_top[0, 2])
        elpd_csv = os.path.join(out_dir, "stats", "choice_elpd_diff_brms.csv")
        elpd_df = pd.read_csv(elpd_csv)
        elpd_diff = float(elpd_df["elpd_diff"].iloc[0])
        elpd_se = float(elpd_df["se_diff"].iloc[0])

        ax_c_new.bar(0, elpd_diff, width=0.6,
                     color=rec_color, edgecolor="none", linewidth=0, zorder=2)
        ax_c_new.bar(0, elpd_diff, width=0.6,
                     color="none", edgecolor="black", linewidth=2.5, zorder=4)
        ax_c_new.errorbar(0, elpd_diff, yerr=elpd_se,
                          fmt="none", ecolor="black", capsize=0, linewidth=2.5, zorder=5)

        ax_c_new.set_ylabel("ELPD (Recalled - True)")
        ax_c_new.set_ylim(0, 150)
        ax_c_new.set_yticks([0, 50, 100, 150])
        ax_c_new.set_xticks([])
        ax_c_new.set_xlim(-0.6, 0.6)
        ax_c_new.spines["right"].set_visible(False)
        ax_c_new.spines["top"].set_visible(False)

        # --- Panel D: RT vs number of memories ---
        ax3 = fig.add_subplot(gs_bot[0, 0])
        d_rt = df_all.dropna(subset=["rt", "recalled_total_count"]).copy()
        d_rt = d_rt[d_rt["rt"] > 0]
        d_rt["log_rt"] = np.log(d_rt["rt"])

        x_rt = d_rt["recalled_total_count"].values
        y_rt = d_rt["rt"].values
        mask_rt = ~np.isnan(x_rt) & ~np.isnan(y_rt)

        for sid, sdf in d_rt.groupby("subject"):
            x_s = sdf["recalled_total_count"].values
            y_s = sdf["rt"].values
            if len(x_s) < 2 or np.allclose(x_s, x_s[0]):
                continue
            m_subj = fit_ols(x_s, y_s)
            if m_subj is None:
                continue
            x0, x1 = float(np.min(x_s)), float(np.max(x_s))
            x_line = np.array([x0, x1])
            y_line = m_subj.predict(add_constant(pd.Series(x_line)))
            ax3.plot(x_line, y_line, color="0.7", alpha=0.6, linewidth=1)

        gmodel_rt = fit_ols(x_rt[mask_rt], y_rt[mask_rt])
        xs_rt = np.array(sorted(d_rt["recalled_total_count"].dropna().unique()))
        if gmodel_rt is not None and len(xs_rt) > 0:
            yhat_rt = gmodel_rt.predict(add_constant(pd.Series(xs_rt)))
            ax3.plot(xs_rt, yhat_rt, color="black", linewidth=4)
        means_rt = []
        errs_rt = []
        xs_rt_plot = []
        for v in xs_rt:
            vals = d_rt.loc[d_rt["recalled_total_count"] == v, "rt"].values
            if len(vals) < 10:
                continue
            m_v, err_v = ci95_mean(vals)
            xs_rt_plot.append(v)
            means_rt.append(m_v)
            errs_rt.append(err_v)
        xs_rt_plot = np.array(xs_rt_plot)
        if len(xs_rt_plot) > 0:
            ax3.errorbar(
                xs_rt_plot,
                means_rt,
                yerr=errs_rt,
                fmt="none",
                ecolor="black",
                capsize=0,
            )
            ax3.scatter(
                xs_rt_plot,
                means_rt,
                s=14**2,
                facecolor=".5",
                edgecolor="black",
                linewidth=2.5,
                zorder=3,
            )
        ax3.set_xlabel("Number of Memories")
        ax3.set_ylabel("Response Time (s)")
        ax3.set_ylim(0, 21.5)
        ax3.set_xticks([1, 2, 3, 4, 5, 6])
        ax3.set_xticklabels([1, 2, 3, 4, 5, 6])
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)

        # --- Panel E: RT vs offer value quintile (split into True / Recalled subpanels) ---
        gs_d = gs_bot[0, 1].subgridspec(1, 2, wspace=0.4)
        ax4_true = fig.add_subplot(gs_d[0])
        ax4_rec = fig.add_subplot(gs_d[1])

        n_bins_d = 5
        bin_labels = ["1", "2", "3", "4", "5"]
        true_color_d = "#81bf78"
        rec_color_d = "#b08dc1"

        # Helper: z-score (not absolute value)
        def _zscore(series):
            mu = float(series.mean())
            sd = float(series.std(ddof=0))
            if sd <= 0:
                return pd.Series(0.0, index=series.index)
            return (series - mu) / sd

        # --- Compute per-subject bin means for both series ---
        def _bin_subject_means(d, val_col):
            d = d.copy()
            d["z_val"] = _zscore(d[val_col])
            d["bin"] = pd.qcut(d["z_val"], q=n_bins_d, labels=bin_labels, duplicates="drop")
            subj_means = d.groupby(["subject", "bin"], observed=True)["rt"].mean().reset_index()
            return subj_means, d

        d_true_d = df_all.dropna(subset=["true_offer_value", "rt"]).copy()
        d_true_d = d_true_d[d_true_d["rt"] > 0]
        subj_true, d_true_d = _bin_subject_means(d_true_d, "true_offer_value")

        d_rec_d = df_all.dropna(subset=["recalled_offer_value", "rt"]).copy()
        d_rec_d = d_rec_d[d_rec_d["rt"] > 0]
        subj_rec, d_rec_d = _bin_subject_means(d_rec_d, "recalled_offer_value")

        x_pos_d = np.arange(n_bins_d)
        rng = np.random.default_rng(42)
        bar_width_d = 1.0

        for ax_d, subj_df, color, title in [
            (ax4_true, subj_true, true_color_d, "True"),
            (ax4_rec, subj_rec, rec_color_d, "Recalled"),
        ]:
            grp_means, grp_ses = [], []
            for bi, bl in enumerate(bin_labels):
                vals = subj_df.loc[subj_df["bin"] == bl, "rt"].values.astype(float)
                m_v = float(np.nanmean(vals))
                se_v = float(np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals)))) if len(vals) > 1 else 0.0
                grp_means.append(m_v)
                grp_ses.append(se_v)

                # Subject dots
                jitter = rng.uniform(-0.15, 0.15, size=len(vals))
                for vi, v in enumerate(vals):
                    ax_d.scatter(x_pos_d[bi] + jitter[vi], v,
                                 s=6**2, facecolor=(1, 1, 1, 0.5), edgecolor=color,
                                 linewidth=1, zorder=3)

            grp_means = np.array(grp_means)
            grp_ses = np.array(grp_ses)

            # Bars (color fill behind dots)
            ax_d.bar(x_pos_d, grp_means, bar_width_d,
                     color=color, edgecolor="none", linewidth=0, zorder=2)
            # Bar outline on top of dots
            ax_d.bar(x_pos_d, grp_means, bar_width_d,
                     color="none", edgecolor="black", linewidth=2.5, zorder=4)
            # Error bars on top
            ax_d.errorbar(x_pos_d, grp_means, yerr=grp_ses,
                          fmt="none", ecolor="black", capsize=0, linewidth=2.5, zorder=5)

            ax_d.set_title(title)
            ax_d.set_xlabel("")
            ax_d.set_ylim(0, 21.5)
            ax_d.set_xticks(x_pos_d)
            ax_d.set_xticklabels(bin_labels)
            ax_d.spines["right"].set_visible(False)
            ax_d.spines["top"].set_visible(False)

        ax4_true.set_ylabel("Response Time (s)")
        # Shared x label centered between the two subpanels
        fig.tight_layout()
        # Shared x label centered between the two subpanels, aligned with Panel C xlabel
        ax3_xlabel_y = ax3.get_position().y0 - 0.048
        fig.text(
            (ax4_true.get_position().x0 + ax4_rec.get_position().x1) / 2,
            ax3_xlabel_y,
            "Offer Value Quintile", ha="center", va="top", fontsize=28,
        )
        fig.savefig(os.path.join(out_dir, "Figure1.pdf"))
        fig_dir = os.path.join(os.getcwd(), "output", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, "Figure1.pdf"))
        plt.close(fig)


def compile_figure2_supplement(
    df_all: pd.DataFrame,
    df_spatial_all: Optional[pd.DataFrame],
    df_items_all: Optional[pd.DataFrame],
    out_dir: str,
):
    """Create Figure 2 Supplement (1x3) and save as PDF.

    Layout: A) item recall rate, B) recalled vs true rewards, C) location recall rate
    """

    sns.set_context("poster")
    with plt.rc_context({
        "font.family": "Arial",
        "axes.titlesize": 24,
        "axes.labelsize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
    }):

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.35, 1.0, 0.35], wspace=1)

        # --- Panel A: item recall rate ---
        ax1 = fig.add_subplot(gs[0, 0])
        sub_game_counts = df_all.groupby(["subject", "game"]).agg({"recalled_total_count": "first"}).reset_index()
        subj_avg = sub_game_counts.groupby("subject")["recalled_total_count"].mean()
        prop = subj_avg / 6.0
        m1, err1 = ci95_mean(prop.values)
        sns.stripplot(
            x=["Proportion"] * len(prop),
            y=prop.values,
            color="gray",
            alpha=0.5,
            size=12,
            jitter=0.05,
            ax=ax1,
            zorder=0,
        )
        ax1.errorbar(
            [0],
            [m1],
            yerr=[err1],
            fmt="none",
            ecolor="black",
            capsize=0,
        )
        ax1.scatter(
            [0],
            [m1],
            s=14**2,
            facecolor=".5",
            edgecolor="black",
            linewidth=2.5,
            zorder=3,
        )
        ax1.set_ylabel("Item Recall Rate")
        ax1.set_ylim(0, 1.05)
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_xticks([])
        ax1.set_xlim(-0.075, 0.075)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.text(-1.2, 1, "A", transform=ax1.transAxes,
            fontsize=26, fontweight="bold", ha="left", va="bottom")

        # --- Panel B: recalled vs true rewards ---
        ax2 = fig.add_subplot(gs[0, 1])
        if df_items_all is not None and len(df_items_all) > 0:
            d_items = df_items_all.dropna(subset=["true_value", "recalled_value"]).copy()
            if len(d_items) > 0:
                sns.scatterplot(data=d_items, x="true_value", y="recalled_value", color="gray",
                                alpha=0.6, ax=ax2, s=30)
                if len(d_items) >= 2:
                    model_items = fit_ols(d_items["true_value"].values, d_items["recalled_value"].values)
                    if model_items is not None:
                        xs_items = np.linspace(np.nanmin(d_items["true_value"]), np.nanmax(d_items["true_value"]), 100)
                        yhat_items = model_items.predict(add_constant(pd.Series(xs_items)))
                        ax2.plot(xs_items, yhat_items, color="black", linewidth=4, zorder=101)
                if len(d_items) > 0:
                    lims = [
                        float(np.nanmin(d_items["true_value"])),
                        float(np.nanmax(d_items["true_value"]))
                    ]
                    ax2.plot(lims, lims, linestyle="--", color=".25", alpha=1, zorder=100)
        ax2.set_xlabel("True Reward")
        ax2.set_ylabel("Recalled Reward")
        ax2.set_xticks([-9, 0, 9])
        ax2.set_xticklabels(["-9", "0", "+9"])
        ax2.set_yticks([-9, 0, 9])
        ax2.set_yticklabels(["-9", "0", "+9"])
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.text(-0.4, 1, "B", transform=ax2.transAxes,
            fontsize=26, fontweight="bold", ha="left", va="bottom")

        # --- Panel C: location recall rate ---
        ax3 = fig.add_subplot(gs[0, 2])
        if df_spatial_all is not None and len(df_spatial_all) > 0:
            subj_spatial = df_spatial_all.groupby("subject")["spatial_correct"].mean()
            m_s, err_s = ci95_mean(subj_spatial.values)
            sns.stripplot(
                x=["Accuracy"] * len(subj_spatial),
                y=subj_spatial.values,
                color="gray",
                alpha=0.5,
                size=12,
                jitter=0.05,
                ax=ax3,
                zorder=0,
            )
            ax3.errorbar(
                [0],
                [m_s],
                yerr=[err_s],
                fmt="none",
                ecolor="black",
                capsize=0,
            )
            ax3.scatter(
                [0],
                [m_s],
                s=14**2,
                facecolor=".5",
                edgecolor="black",
                linewidth=2.5,
                zorder=3,
            )
        ax3.set_ylabel("Location Recall Rate")
        ax3.set_ylim(0, 1.05)
        ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax3.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax3.set_xticks([])
        ax3.set_xlim(-0.075, 0.075)
        ax3.plot((-0.075, 0.075), (1/6, 1/6), 'k:')
        ax3.spines["bottom"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.text(-1.1, 1, "C", transform=ax3.transAxes,
            fontsize=26, fontweight="bold", ha="left", va="bottom")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "FigureS2.pdf"), bbox_inches="tight")
        supp_dir = os.path.join(os.getcwd(), "output", "figures", "supplementary")
        os.makedirs(supp_dir, exist_ok=True)
        fig.savefig(os.path.join(supp_dir, "FigureS2.pdf"), bbox_inches="tight")
        plt.close(fig)


def main():
    data_root = os.path.join(os.getcwd(), "data")
    out_dir = os.path.join(os.getcwd(), "output", "behavior")
    ensure_output_dir(out_dir)

    subjects = list_subjects(data_root)

    # Containers for group-level data
    all_choice_rows = []
    all_item_rows = []
    all_spatial_rows = []
    df_items_all = None
    df_spatial_all = None

    for sid in subjects:
        df_choices = build_subject_choice_dataset(sid, data_root)
        if df_choices is None or len(df_choices) == 0:
            continue

        all_choice_rows.append(df_choices)

        # Item-level data
        df_items = build_subject_item_values_dataset(sid, data_root)
        if df_items is not None and len(df_items) > 0:
            all_item_rows.append(df_items)

        # Spatial recall data
        df_spatial = build_subject_spatial_recall_dataset(sid, data_root)
        if df_spatial is not None and len(df_spatial) > 0:
            all_spatial_rows.append(df_spatial)

    if len(all_choice_rows) == 0:
        print("No choice data found.")
        return

    df_all = pd.concat(all_choice_rows, ignore_index=True)

    if len(all_spatial_rows) > 0:
        df_spatial_all = pd.concat(all_spatial_rows, ignore_index=True)

    if len(all_item_rows) > 0:
        df_items_all = pd.concat(all_item_rows, ignore_index=True)

    # Figure 2 (2x2: accuracy, logistic, RT vs memories, RT vs recalled offer value)
    compile_figure2(
        df_all=df_all,
        df_spatial_all=df_spatial_all,
        df_items_all=df_items_all,
        out_dir=out_dir,
    )

    # Figure 2 Supplement (1x3: item recall, true vs recalled rewards, location recall)
    compile_figure2_supplement(
        df_all=df_all,
        df_spatial_all=df_spatial_all,
        df_items_all=df_items_all,
        out_dir=out_dir,
    )

    # Export stats-ready CSVs for R mixed-effects models
    stats_dir = os.path.join(out_dir, "stats")
    ensure_output_dir(stats_dir)

    # Per-subject summary table
    subj_rows = []
    for sid in sorted(df_all["subject"].unique()):
        sub_df = df_all[df_all["subject"] == sid]
        acc = float(sub_df["correct"].mean()) if "correct" in sub_df.columns and len(sub_df) > 0 else np.nan

        rt_series = pd.to_numeric(sub_df.get("rt"), errors="coerce") if "rt" in sub_df.columns else pd.Series([], dtype=float)
        rt_series = rt_series[(~rt_series.isna()) & (rt_series > 0)]
        rt_mean = float(rt_series.mean()) if len(rt_series) > 0 else np.nan
        rt_median = float(rt_series.median()) if len(rt_series) > 0 else np.nan

        spatial_rate = np.nan
        if df_spatial_all is not None and len(df_spatial_all) > 0:
            sp_sub = df_spatial_all[df_spatial_all["subject"] == sid]
            if len(sp_sub) > 0 and "spatial_correct" in sp_sub.columns:
                spatial_rate = float(sp_sub["spatial_correct"].mean())

        subj_rows.append({
            "subject": sid,
            "choice_accuracy": acc,
            "rt_mean": rt_mean,
            "rt_median": rt_median,
            "spatial_recall_rate": spatial_rate,
        })

    subj_summary_df = pd.DataFrame(subj_rows)
    subj_summary_df.to_csv(os.path.join(stats_dir, "subject_behavior_summary.csv"), index=False)
    export_stats_dataframes(
        df_all,
        df_items_all,
        df_spatial_all,
        stats_dir,
    )
    export_summary_stats(df_all, df_spatial_all, stats_dir)

    print(f"Saved Figure1.pdf, FigureS2.pdf, subject summary, and stats CSVs to {out_dir}")


if __name__ == "__main__":
    main()

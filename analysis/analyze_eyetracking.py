
'''
Use this command to create the eye movement summary figure
python analysis/analyze_eyetracking.py --buffer-ms 50 --out-dir output/eyegaze
'''

import os
import sys
import argparse
from pathlib import Path

# Add lib/ to sys.path so we can import helper modules
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from visualize_recall_fixation_wedges_group import get_cluster_heatmap_data, draw_full_roi
from visualize_choice_fixation_wedges_group import (
    get_choice_contrast_heatmap_data,
    get_choice_interaction_heatmap_data,
)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def add_panel_A_time_course(ax, roi_type: str = "original", buffer_size: int = 50) -> None:
    """Populate panel A with the clean group time course trace.

    This reads the precomputed summary CSV from analyze_recall_group:
      output/eyegaze/recall/group_time_course_{roi_type}_buffer_{buffer_size}.csv

    and recreates the clean black line + SEM band + significance bar,
    matching create_group_time_course_plot_clean but without redoing
    any statistics or smoothing.
    """

    base_output_dir = os.path.join(os.getcwd(), "output")
    recall_dir = os.path.join(base_output_dir, "eyegaze", "recall")
    csv_path = os.path.join(recall_dir, f"group_time_course_{roi_type}_buffer_{buffer_size}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected time course CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "time_point" not in df.columns:
        raise ValueError("CSV missing 'time_point' column")

    time_points = df["time_point"].to_numpy()
    # Prefer smoothed series if available, otherwise fall back
    if "group_duration_mean_smooth" in df.columns and not df["group_duration_mean_smooth"].isna().all():
        means = df["group_duration_mean_smooth"].to_numpy(dtype=float)
    else:
        means = df["group_duration_mean"].to_numpy(dtype=float)

    if "group_duration_sem_smooth" in df.columns and not df["group_duration_sem_smooth"].isna().all():
        sems = df["group_duration_sem_smooth"].to_numpy(dtype=float)
    else:
        sems = df["group_duration_sem"].to_numpy(dtype=float)

    chance_level = float(df["chance_level"].iloc[0]) if "chance_level" in df.columns else 1.0 / 6.0
    sig = df.get("significant", pd.Series(np.zeros(len(df), dtype=bool))).astype(bool).to_numpy()

    # Plot main data line and SEM band in black
    valid_mask = ~np.isnan(means)
    if np.any(valid_mask):
        t_valid = time_points[valid_mask]
        m_valid = means[valid_mask]
        s_valid = sems[valid_mask]

        ax.plot(t_valid, m_valid, "-", linewidth=4, color="black", zorder=10)
        ax.fill_between(
            t_valid,
            m_valid - s_valid,
            m_valid + s_valid,
            alpha=0.35,
            color=".4",
            zorder=5,
            linewidth=0,
        )

    # Reference lines in black
    ax.axhline(y=chance_level, color="black", linestyle=":", linewidth=2.5, alpha=1)
    ax.axvline(x=0, color="black", linestyle="-", alpha=1)

    # Significance bar as in create_group_time_course_plot_clean
    if np.any(sig):
        sig_indices = np.where(sig)[0]
        if len(sig_indices) > 0:
            sig_y_position = 0.6

            segments = []
            current_segment = [sig_indices[0]]
            for i in range(1, len(sig_indices)):
                if sig_indices[i] == sig_indices[i - 1] + 1:
                    current_segment.append(sig_indices[i])
                else:
                    segments.append(current_segment)
                    current_segment = [sig_indices[i]]
            segments.append(current_segment)

            for segment in segments:
                start_time = time_points[segment[0]]
                end_time = time_points[segment[-1]]
                ax.plot(
                    [start_time, end_time],
                    [sig_y_position, sig_y_position],
                    color="black",
                    linewidth=16,
                    alpha=1.0,
                )

    # Match axis limits and labels from the clean plot
    ax.set_ylim(0, 0.6)
    ax.set_xlim(-3000, 750)
    ax.set_xticks([-3000, -2000, -1000, 0, 750])
    ax.set_xticklabels([-3, -2, -1, 0, 0.75])
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6])
    #ax.set_ylabel("Proportion of Fixation Time on Recalled Item Location")
    ax.set_ylabel("Proportion Fixation Time")
    ax.set_xlabel("Time Relative to Verbal Recall (s)")

    # Clean look: no top/right spines handled later by layout code
    sns.despine(ax=ax)


def add_panel_A_time_course_subject(
    ax,
    subject_id: str,
    roi_type: str = "original",
    buffer_size: int = 50,
) -> None:
    """Per-subject version of panel A: fixation time course for a single subject.

    Reads
      output/{subject_id}/{subject_id}_fixation_time_course_{roi_type}_buffer_{buffer_size}.csv

    and plots the subject's proportion of fixation duration on the recalled
    item location over time, with a chance-level reference line.
    """

    base_data_dir = os.path.join(os.getcwd(), "data")
    subj_dir = os.path.join(base_data_dir, str(subject_id))
    csv_path = os.path.join(
        subj_dir,
        f"{subject_id}_fixation_time_course_{roi_type}_buffer_{buffer_size}.csv",
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected subject time course CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "time_point" not in df.columns:
        raise ValueError("CSV missing 'time_point' column for subject time course")

    time_points = df["time_point"].to_numpy()
    if "proportion_duration" in df.columns:
        vals = df["proportion_duration"].to_numpy(dtype=float)
    elif "proportion_count" in df.columns:
        vals = df["proportion_count"].to_numpy(dtype=float)
    else:
        raise ValueError(
            "Subject time course CSV missing 'proportion_duration' and 'proportion_count' columns"
        )

    chance_level = float(df["chance_level"].iloc[0]) if "chance_level" in df.columns else 1.0 / 6.0

    valid_mask = ~np.isnan(vals)
    if np.any(valid_mask):
        t_valid = time_points[valid_mask]
        v_valid = vals[valid_mask]
        ax.plot(t_valid, v_valid, "-", linewidth=3, color="black", zorder=10)

    ax.axhline(y=chance_level, color="black", linestyle=":", linewidth=2.5, alpha=1)
    ax.axvline(x=0, color="black", linestyle="-", alpha=1)

    ax.set_ylim(0, 0.6)
    ax.set_xlim(-3000, 750)
    ax.set_xticks([-3000, -2000, -1000, 0, 750])
    ax.set_xticklabels([-3, -2, -1, 0, 0.75])
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6])
    ax.set_ylabel("Proportion Fixation Time")
    ax.set_xlabel("Time Relative to Verbal Recall (s)")

    sns.despine(ax=ax)


def add_panel_B_relevant_only(
    ax,
    metric: str = "duration",
    highlight_subject: str | int | None = None,
) -> None:
    """Populate panel B with the relevant-only styled fixation plot.

    Reads the precomputed subject-level relevant-only means from
      output/choice_fixation_relevance_subject_means_relevant_only_{metric}.csv

    and recreates the jittered subject points plus group mean+SEM,
    matching plot_relevant_only_styled in choice_fixation_proportions.py.
    """

    stats_dir = os.path.join(os.getcwd(), "output", "eyegaze", "stats")
    csv_path = os.path.join(
        stats_dir,
        f"choice_fixation_relevance_subject_means_relevant_only_{metric}.csv",
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected relevant-only CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "mean_prop" not in df.columns:
        raise ValueError("CSV missing 'mean_prop' column for relevant-only plot")

    df["subject"] = df["subject"].astype(str)
    vals = df["mean_prop"].astype(float).to_numpy()

    # Style to match panel A in behavior summary (stripplot + big mean dot + CI)
    x_label = "Relevant"
    x_coords = [0] * len(vals)

    # Individual subject points (strip-style jittered around x=0)
    sns.stripplot(
        x=x_coords,
        y=vals,
        #color="tab:green",
        color="gray",
        alpha=0.5,
        size=12,
        jitter=0.05,
        ax=ax,
        zorder=0,
    )

    # Group mean and 95% CI using same ci95_mean helper logic
    mean_val = float(np.nanmean(vals)) if len(vals) > 0 else np.nan

    if len(vals) > 1:
        se = float(np.nanstd(vals, ddof=1)) / np.sqrt(np.sum(~np.isnan(vals)))
        ci = 1.96 * se
    else:
        ci = 0.0

    # Draw CI bar only
    ax.errorbar(
        [0],
        [mean_val],
        yerr=[ci],
        fmt="none",
        ecolor="black",
        capsize=0,
    )
    # Draw large filled marker with black edge
    ax.scatter(
        [0],
        [mean_val],
        s=14 ** 2,
        #facecolor="tab:green",  # .5
        facecolor=".5",  # .5
        edgecolor="black",
        linewidth=2.5,
        zorder=3,
    )

    # Optionally highlight a specific subject's point
    if highlight_subject is not None:
        sid = str(highlight_subject)
        row = df[df["subject"] == sid]
        if not row.empty:
            sval = float(row["mean_prop"].iloc[0])
            ax.scatter(
                [0],
                [sval],
                s=16 ** 2,
                facecolor="none",
                edgecolor="black",
                linewidth=3.0,
                zorder=4,
            )

    # Y-axis and reference line styled like panel A
    ax.set_ylim(0.2, 0.8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([0.2, 0.4, 0.6, 0.8])
    ylabel = (
        "Prop. Relevant Fix. Time"
        if metric == "duration"
        else "Subject Mean Count Proportion"
    )
    ax.set_ylabel(ylabel)

    # No x tick labels beyond a blank category, consistent with panel A
    ax.set_xticks([])
    ax.set_xlim(-0.075, 0.075)
    ax.plot((-0.075, 0.075), (0.5, 0.5), "k:", linewidth=2.5)

    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def add_panel_C_relsign4_relevant(
    ax,
    metric: str = "duration",
    highlight_subject: str | int | None = None,
) -> None:
    """Populate panel C with the relsign4 relevant styled subject-mean plot.

    Reads precomputed subject-level relsign4 means from
      output/choice_fixation_relsign4_relevant_subject_means_{metric}.csv

    and recreates the styled plot from plot_subject_means_styled in
    choice_fixation_proportions.py (decision × valence with subject lines,
    group means, and SEM error bars).
    """

    stats_dir = os.path.join(os.getcwd(), "output", "eyegaze", "stats")
    csv_path = os.path.join(
        stats_dir,
        f"choice_fixation_relsign4_relevant_subject_means_{metric}.csv",
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected relsign4 relevant CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"subject", "decision_label", "valence_label", "mean_prop"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"CSV for panel C missing columns: {missing}")

    decision_order = [
        d for d in ["take", "leave"] if d in df["decision_label"].unique()
    ] or sorted(df["decision_label"].unique())
    valence_order = [
        v for v in ["positive", "negative"] if v in df["valence_label"].unique()
    ] or sorted(df["valence_label"].unique())

    palette = sns.color_palette("deep", n_colors=len(valence_order))
    x_index = {d: i for i, d in enumerate(decision_order)}

    offsets_map = {1: [0.0], 2: [-0.32, 0.32], 3: [-0.2, 0.0, 0.2]}
    offsets = offsets_map.get(len(valence_order), np.linspace(-0.3, 0.3, len(valence_order)))
    val_to_offset = {vlab: offsets[j] for j, vlab in enumerate(valence_order)}

    df["subject"] = df["subject"].astype(str)

    # Thin subject lines connecting valences within each decision
    for subj in df["subject"].unique():
        for dlab in decision_order:
            xs, ys = [], []
            for vlab in valence_order:
                row = df[
                    (df["subject"] == subj)
                    & (df["decision_label"] == dlab)
                    & (df["valence_label"] == vlab)
                ]
                if not row.empty:
                    xs.append(x_index[dlab] + val_to_offset[vlab])
                    ys.append(float(row["mean_prop"].values[0]))
            if len(xs) >= 2:
                lw = 0.8
                alpha = 0.8
                color = "0.7"
                if highlight_subject is not None and subj == str(highlight_subject):
                    lw = 2.0
                    alpha = 1.0
                    color = "black"
                ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw, zorder=1)

    # Group means and SEM with thick black connector, as in plot_subject_means_styled
    stats_df = (
        df.groupby(["decision_label", "valence_label"])["mean_prop"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats_df["sem"] = stats_df["std"] / np.sqrt(stats_df["count"].clip(lower=1))
    color_map = {vlab: palette[j] for j, vlab in enumerate(valence_order)}

    for dlab in decision_order:
        rows = stats_df[stats_df["decision_label"] == dlab]
        rows = rows.set_index("valence_label").reindex(valence_order).reset_index()
        xs = [x_index[dlab] + val_to_offset[v] for v in rows["valence_label"]]
        ys = rows["mean"].values.astype(float)
        ses = rows["sem"].values.astype(float)
        # Thick black line connecting means
        ax.plot(xs, ys, color="black", linewidth=2.5, zorder=5)
        # Mean points with SEM, styled to match panel B group mean
        for xi, yi, sei, vlab in zip(xs, ys, ses, rows["valence_label"]):
            ax.errorbar(
                xi,
                yi,
                yerr=sei * 2,
                fmt="o",
                ms=14,
                mfc=color_map[vlab],
                mec="black",
                mew=2.5,
                ecolor="black",
                elinewidth=2.5,
                capsize=0,
                zorder=6,
            )

    # Axis labels and ticks as in the original styled plot
    ylabel = (
        "Proportion Fixation Time"
        if metric == "duration"
        else "Subject Mean Count Proportion"
    )
    # Let global rc_context control axis label font sizes for consistency
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[v],
            markeredgecolor="black",
            markeredgewidth=2.5,
            markersize=14,
            linewidth=0,
        )
        for v in valence_order
    ]
    label_map = {"positive": "Positive Reward", "negative": "Negative Reward"}
    ax.legend(
        legend_handles,
        [label_map.get(v, v) for v in valence_order],
        fontsize=20,
        handletextpad=0.3,
        loc="upper center",
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )

    dec_label_map = {"take": "Take", "leave": "Leave"}
    ax.set_xticks([x_index[d] for d in decision_order])
    ax.set_xticklabels([dec_label_map.get(d, d) for d in decision_order])

    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_ylim(0, 0.63)

    sns.despine(ax=ax)


def add_panel_B_cluster_heatmap(
    ax,
    roi_type: str = "original",
    buffer_size: int = 50,
    outer_radius: int = 530,
    central_radius: int = 20,
    circle_radius: int = 380,
    data_dir: str | None = None,
) -> None:
    """Populate panel B by recomputing the cluster-wide wedge heatmap.

    Uses the same data and plotting logic as compute_group_cluster_heatmap
    in visualize_recall_fixation_wedges_group, but draws directly into
    the provided axes instead of loading a saved PNG.
    """

    base_output_dir = os.path.join(os.getcwd(), "output")
    data = get_cluster_heatmap_data(
        output_base_dir=base_output_dir,
        roi_type=roi_type,
        buffer_size=buffer_size,
        outer_radius=outer_radius,
        central_radius=central_radius,
        circle_radius=circle_radius,
        data_dir=data_dir,
    )
    if data is None:
        raise RuntimeError("Could not compute cluster heatmap data for panel B.")

    group_H_smooth = data["group_H_smooth"]
    x_range = data["x_range"]
    y_range = data["y_range"]

    ax.figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    gray_hot = sns.color_palette("BuPu", as_cmap=True)
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    im = ax.imshow(
        group_H_smooth.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=gray_hot,
        vmin=0,
    )

    clip_circle = Circle((0, 0), outer_radius, transform=ax.transData)
    im.set_clip_path(clip_circle)

    cax = inset_axes(
        ax,
        width="3%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    cbar = plt.colorbar(im, cax=cax)
    data_max = float(np.nanmax(group_H_smooth)) if np.any(group_H_smooth) else 0.0
    im.set_clim(0.0, data_max)
    if data_max > 0:
        ticks = np.linspace(0.0, 0.0025, 6)
    else:
        ticks = np.array([0.0])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t * 1e3:g}" for t in ticks])
    # Put the 10^-3 factor using two Arial text elements to mimic a superscript.
    # First draw the base "×10", then measure its extent to place "-3".
    base_text = cbar.ax.text(
        0.9,
        1.0,
        "×10",
        transform=cbar.ax.transAxes,
        fontfamily="Arial",
        ha="left",
        va="bottom",
        fontsize=12,
    )
    # Ensure positions are realized before measuring
    cbar.ax.figure.canvas.draw()
    renderer = cbar.ax.figure.canvas.get_renderer()
    bbox = base_text.get_window_extent(renderer=renderer)
    # Convert the right edge of the base text back into axes coordinates (x only)
    inv = cbar.ax.transAxes.inverted()
    (x_right, _y_dummy) = inv.transform((bbox.x1, bbox.y1))
    # Place a smaller "-3" slightly to the right of the base text,
    # with a fixed y inside the axes so vertical tweaks are visible.
    superscript_y = 1.025  # adjust this value to move the superscript up/down
    cbar.ax.text(
        x_right,# + 0.005,
        superscript_y,
        "-3",
        transform=cbar.ax.transAxes,
        fontfamily="Arial",
        ha="left",
        va="bottom",
        fontsize=9,
    )

    draw_full_roi(ax, outer_radius=outer_radius, central_radius=central_radius)

    rect_size = 160
    half_size = rect_size / 2.0
    rect_center_x = 0.0
    rect_center_y = circle_radius
    rect = Rectangle(
        (rect_center_x - half_size, rect_center_y - half_size),
        rect_size,
        rect_size,
        fill=False,
        edgecolor="black",
        linestyle="-",
        linewidth=2,
    )
    ax.add_patch(rect)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Proportion Fixation Time", fontsize=27, y=1.02)


def _plot_choice_contrast_into_axes(
    ax,
    data: dict,
    title: str,
    cbar_label: str,
    vmin: float = -0.0003,
    vmax: float = 0.0003,
    ticks: np.ndarray | None = None,
) -> None:
    """Generic helper to draw a choice contrast heatmap into an existing axes.

    Matches the styling of plot_choice_contrast_heatmap in
    visualize_choice_fixation_wedges_group but draws into the provided
    axes instead of creating a new figure.
    """

    if data is None:
        raise RuntimeError("No data provided for choice contrast panel.")

    group_D_smooth = data["group_D_smooth"]
    x_range = data["x_range"]
    y_range = data["y_range"]
    outer_radius = data["outer_radius"]
    central_radius = data["central_radius"]
    circle_radius = data["circle_radius"]

    fig = ax.figure
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    diverge = sns.diverging_palette(240, 10, as_cmap=True)

    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

    im = ax.imshow(
        group_D_smooth.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=diverge,
        vmin=vmin,
        vmax=vmax,
    )

    clip_circle = Circle((0, 0), outer_radius, transform=ax.transData)
    im.set_clip_path(clip_circle)

    cax = inset_axes(
        ax,
        width="3%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    cbar = plt.colorbar(im, cax=cax)
    if ticks is None:
        ticks = np.linspace(vmin, vmax, 7)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t * 1e3:g}" for t in ticks])
    if cbar_label:
        cbar.set_label(cbar_label)

    # Add a "×10^-3" scale factor above the colorbar using the same
    # styling as the recall wedge panel.
    base_text = cbar.ax.text(
        0.9,
        1.0,
        "×10",
        transform=cbar.ax.transAxes,
        fontfamily="Arial",
        ha="left",
        va="bottom",
        fontsize=12,
    )
    cbar.ax.figure.canvas.draw()
    renderer = cbar.ax.figure.canvas.get_renderer()
    bbox = base_text.get_window_extent(renderer=renderer)
    inv = cbar.ax.transAxes.inverted()
    (x_right, _y_dummy) = inv.transform((bbox.x1, bbox.y1))
    superscript_y = 1.025
    cbar.ax.text(
        x_right,
        superscript_y,
        "-3",
        transform=cbar.ax.transAxes,
        fontfamily="Arial",
        ha="left",
        va="bottom",
        fontsize=9,
    )

    draw_full_roi(ax, outer_radius=outer_radius, central_radius=central_radius)

    rect_size = 160
    half_size = rect_size / 2.0
    rect_center_x = 0.0
    rect_center_y = circle_radius
    rect = Rectangle(
        (rect_center_x - half_size, rect_center_y - half_size),
        rect_size,
        rect_size,
        fill=False,
        edgecolor="black",
        linestyle="-",
        linewidth=2,
    )
    ax.add_patch(rect)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=27, pad=10)


def add_panel_choice_relevance_contrast(
    ax,
    roi_type: str = "original",
    buffer_size: int = 50,
    data_dir: str | None = None,
) -> None:
    """Populate a panel with the relevant-minus-irrelevant choice contrast.

    Uses get_choice_contrast_heatmap_data and draws the resulting map
    into the provided axes, with a fixed symmetric color scale of
    ±0.0003 (displayed as ±0.3 on the ×10⁻³ colorbar).
    """

    base_output_dir = os.path.join(os.getcwd(), "output")
    task_path = os.path.join(os.getcwd(), "task")
    data = get_choice_contrast_heatmap_data(
        output_base_dir=base_output_dir,
        task_path=task_path,
        roi_type=roi_type,
        buffer_size=buffer_size,
        data_dir=data_dir,
    )
    if data is None:
        raise RuntimeError("Could not compute relevant-minus-irrelevant choice contrast data.")

    _plot_choice_contrast_into_axes(
        ax,
        data=data,
        title="Relevant - Irrelevant",
        cbar_label="", #Prop. Fix. Time Difference
        vmin=-0.0003,
        vmax=0.0003,
    )


def add_panel_choice_interaction_relevant(
    ax,
    roi_type: str = "original",
    buffer_size: int = 50,
    data_dir: str | None = None,
) -> None:
    """Populate a panel with the valence × decision interaction (relevant items).

    Uses get_choice_interaction_heatmap_data with anchor_type="relevant"
    and draws the resulting interaction map into the provided axes,
    using the same fixed ±0.0003 color scale as the relevance contrast.
    """

    base_output_dir = os.path.join(os.getcwd(), "output")
    task_path = os.path.join(os.getcwd(), "task")
    data = get_choice_interaction_heatmap_data(
        output_base_dir=base_output_dir,
        task_path=task_path,
        roi_type=roi_type,
        buffer_size=buffer_size,
        anchor_type="relevant",
        data_dir=data_dir,
    )
    if data is None:
        raise RuntimeError("Could not compute interaction contrast data for relevant items.")

    _plot_choice_contrast_into_axes(
        ax,
        data=data,
        title="Choice x Valence",
        cbar_label="",#Prop. Fix. Time Difference",
        vmin=-0.0003,
        vmax=0.0003,
    )



def _load_prop_time_summary() -> pd.DataFrame:
    """Load the prop-time interaction model summary CSV for human data.

    Expects output/eyegaze/stats/summary_prop_time_location_interactions_recalled_all_norm-within.csv
    """
    base = os.path.join(os.getcwd(), "output", "eyegaze", "stats")
    path = os.path.join(base, "summary_prop_time_location_interactions_recalled_all_norm-within.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prop-time summary CSV not found: {path}")
    df = pd.read_csv(path)
    if "feature_set" in df.columns:
        df = df[df["feature_set"].astype(str) == "location_interactions"]
    if "visit_type" in df.columns:
        df = df[df["visit_type"].astype(str) == "all"]
    return df


def _load_prop_time_coef_table() -> pd.DataFrame:
    """Load the prop-time interaction model coefficient table for human data."""
    base = os.path.join(os.getcwd(), "output", "eyegaze", "stats")
    path = os.path.join(base, "coef_table_prop_time_location_interactions_recalled_all_norm-within.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prop-time coef table CSV not found: {path}")
    df = pd.read_csv(path)
    if "feature_set" in df.columns:
        df = df[df["feature_set"].astype(str) == "location_interactions"]
    if "visit_type" in df.columns:
        df = df[df["visit_type"].astype(str) == "all"]
    return df


def add_panel_D_cv_accuracy(ax) -> None:
    """Panel D: CV accuracy from prop-time interaction model.

    Single gray dot with SEM error bar and 0.5 dashed chance line,
    styled identically to Panel G of the NN overview figure.
    """
    summ = _load_prop_time_summary()
    if len(summ) == 0 or "cv_mean" not in summ.columns or "cv_sem" not in summ.columns:
        raise ValueError("Prop-time summary CSV missing cv_mean or cv_sem")

    row = summ.iloc[0]
    mean = float(row["cv_mean"])
    sem = float(row["cv_sem"])

    bar_width = 0.6
    # Gray bar fill
    ax.bar([0], [mean], bar_width,
           color=".7", edgecolor="none", linewidth=0, zorder=2)
    # Black outline on top
    ax.bar([0], [mean], bar_width,
           color="none", edgecolor="black", linewidth=2.5, zorder=4)
    # Error bar on top
    ax.errorbar([0], [mean], yerr=[sem],
                fmt="none", ecolor="black", capsize=0, linewidth=2.5, zorder=5)
    ax.set_ylabel("Choice Prediction Accuracy")
    ax.set_xticks([])
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0.5, 0.75)
    ax.set_yticks([0.5, 0.6, 0.7])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def add_panel_E_coef_by_item(ax) -> None:
    """Panel E: regression coefficients by item position (1-6).

    Two series — Irrelevant x Reward (pt_x_val) and Relevant x Reward
    (pt_x_val_x_rel) — with dots and shaded 95% CI bands, styled
    identically to Panel H of the NN overview figure.
    """
    coef_df = _load_prop_time_coef_table()

    palette = ["#ba7caf", "#6fc7eb"]

    def _get_loc_term(term: str):
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
        raise ValueError("Missing reward terms in coef table")

    for df_term, label, color in [
        (reward, "Irrelevant x Reward", palette[0]),
        (reward_x_rel, "Relevant x Reward", palette[1]),
    ]:
        x = df_term["loc"].to_numpy(dtype=float)
        c = df_term["coef"].to_numpy(dtype=float)
        lo = df_term["lo"].to_numpy(dtype=float)
        hi = df_term["hi"].to_numpy(dtype=float)
        # CI error bars (asymmetric: lo to hi)
        yerr_lo = c - lo
        yerr_hi = hi - c
        ax.errorbar(
            x, c, yerr=[yerr_lo, yerr_hi],
            fmt="none", ecolor="black", capsize=0, zorder=2,
        )
        # Points with black outline, matching B and D style
        ax.scatter(
            x, c,
            s=14**2, facecolor=color, edgecolor="black",
            linewidth=2.5, zorder=3, label=label,
        )

    ax.axhline(0, color="0", linewidth=1)
    ax.set_xticks(np.arange(1, 7))
    ax.set_xlabel("Item")
    ax.set_ylabel("Regression Coefficient")
    ax.set_ylim(-0.5, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="best")


_FOREST_PALETTE = {"irr": "#ba7caf", "rel": "#6fc7eb"}
_FOREST_TERM_FOR = {"irr": "pt_x_val", "rel": "pt_x_val_x_rel"}


def _per_location_coef_rows(coef_df: pd.DataFrame, term: str) -> list:
    rows = []
    for loc in range(1, 7):
        feat = f"loc{loc}_{term}"
        sub = coef_df[coef_df["feature"].astype(str) == feat]
        if len(sub) == 0:
            raise ValueError(f"Coef table missing feature={feat}")
        r = sub.iloc[0]
        rows.append({
            "loc": int(loc),
            "coef": float(r["coef"]),
            "lo": float(r["lo"]),
            "hi": float(r["hi"]),
        })
    return rows


def add_panel_coef_forest_per_location(ax) -> None:
    """Horizontal forest plot of per-location logistic-regression coefficients.

    Per-location: 6 rows (loc1 at top, loc6 at bottom), each holding two colored
    dots side-by-side with horizontal 95% bootstrap CIs:
      - Irrelevant x Reward (purple #ba7caf): loc{i}_pt_x_val
      - Relevant x Reward (light blue #6fc7eb): loc{i}_pt_x_val_x_rel
    Each term's mean across the 6 locations is overlaid as a solid colored
    vertical line. Y-tick labels are the item angles (0 deg .. 300 deg).
    """
    coef_df = _load_prop_time_coef_table()

    loc_y = {loc: float(7 - loc) for loc in range(1, 7)}  # loc1 -> y=6, loc6 -> y=1
    offset = 0.18
    y_offset = {"irr": +offset, "rel": -offset}

    per_loc_marker_size = 14 ** 2
    per_loc_elw = 2.5

    per_term_means: dict = {}

    for key in ("irr", "rel"):
        color = _FOREST_PALETTE[key]
        term = _FOREST_TERM_FOR[key]
        rows = _per_location_coef_rows(coef_df, term)
        per_term_means[key] = float(np.mean([r["coef"] for r in rows]))
        for r in rows:
            y = loc_y[r["loc"]] + y_offset[key]
            ax.errorbar(
                [r["coef"]], [y],
                xerr=[[r["coef"] - r["lo"]], [r["hi"] - r["coef"]]],
                fmt="none", ecolor="black", elinewidth=per_loc_elw,
                capsize=0, zorder=3,
            )
            ax.scatter(
                [r["coef"]], [y],
                s=per_loc_marker_size, facecolor=color, edgecolor="black",
                linewidth=2.5, zorder=4,
            )

    for key in ("irr", "rel"):
        ax.axvline(
            per_term_means[key], color=_FOREST_PALETTE[key], linewidth=3.5,
            linestyle="-", zorder=2, alpha=0.9,
        )

    ax.axvline(0, color="black", linewidth=2.5, linestyle=":", zorder=1)

    yticks = [loc_y[loc] for loc in range(1, 7)]
    yticklabels = [f"{(loc - 1) * 60}°" for loc in range(1, 7)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=20)
    ax.set_ylim(min(loc_y.values()) - 0.5, max(loc_y.values()) + 0.5)

    ax.set_xlim(-0.75, 0.75)
    ax.set_xlabel("Effect on Choice (log odds)", fontsize=26)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_FOREST_PALETTE["irr"],
               markeredgecolor="black", markeredgewidth=2.5, markersize=14,
               label="Irrelevant x Reward"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_FOREST_PALETTE["rel"],
               markeredgecolor="black", markeredgewidth=2.5, markersize=14,
               label="Relevant x Reward"),
    ]
    ax.legend(handles=legend_handles, frameon=True, fontsize=16,
              loc="upper center", handletextpad=0.2, ncol=2,
              columnspacing=1.0, borderaxespad=0.2,
              bbox_to_anchor=(0.5, -0.32),
              facecolor="white", edgecolor="none")


def create_eyeplot_layout(
    out_dir: str,
    buffer_ms: int = 50,
) -> None:
    """Create a 2-row eyetracking summary layout.

    Row 1 (top):
        - Col 1 (A): recall time course + recall wedge cluster heatmap
        - Col 2 (B): proportion relevant fixation time

    Row 2 (bottom):
        - Col 1 (C): relsign4 relevant subject means
        - Col 2 (E): choice prediction accuracy (CV bar)
        - Col 3 (D): reward and interaction coefficients by item (forest)

    Styling is matched to the summary_panel figure in analyze_behavior.py
    (seaborn "poster" context, Arial fonts, similar font sizes).
    """

    ensure_output_dir(out_dir)

    sns.set_context("poster")
    with plt.rc_context({
        "font.family": "Arial",
        "axes.titlesize": 24,
        "axes.labelsize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
    }):
        fig = plt.figure(figsize=(16, 12))

        # Two rows with independent column widths.
        outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.37)

        # Row 1: [A, heatmap] block (tight) + B (quarter-width, separated)
        row1 = outer[0].subgridspec(1, 2, width_ratios=[2, 0.25], wspace=0.6)
        row1_left = row1[0, 0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
        ax_A  = fig.add_subplot(row1_left[0, 0])   # A: recall time course
        ax_hm = fig.add_subplot(row1_left[0, 1])   # recall wedge cluster heatmap
        ax_B  = fig.add_subplot(row1[0, 1])         # B: prop. relevant fixation time

        # Row 2: C (separated) + [E, D] block (tight)
        row2 = outer[1].subgridspec(1, 2, width_ratios=[1, 1.35], wspace=0.4) #0.5
        ax_C = fig.add_subplot(row2[0, 0])    # C: relsign4 relevant
        row2_right = row2[0, 1].subgridspec(1, 2, width_ratios=[0.35, 1], wspace=0.5)
        ax_E = fig.add_subplot(row2_right[0, 0])    # E: CV accuracy
        # Reserve a strip at the bottom of the coef cell for its legend so the
        # legend doesn't dangle below the other panels in this row.
        coef_cell = row2_right[0, 1].subgridspec(
            2, 1, height_ratios=[1, 0.32], hspace=0.0,
        )
        ax_D = fig.add_subplot(coef_cell[0, 0])    # D: regression coefficients forest

        # Panel A: recall time course
        try:
            add_panel_A_time_course(ax_A, buffer_size=buffer_ms)
        except Exception as e:
            print(f"Warning: failed to populate panel A (time course): {e}")

        # Per-subject data lives in data/ (not output/)
        base_data_dir = os.path.join(os.getcwd(), "data")

        # Recall cluster heatmap (next to panel A)
        try:
            add_panel_B_cluster_heatmap(ax_hm, roi_type="original", buffer_size=50, data_dir=base_data_dir)
        except Exception as e:
            print(f"Warning: failed to populate recall cluster heatmap: {e}")

        # Panel B: relevant-only fixation proportions
        try:
            add_panel_B_relevant_only(ax_B, metric="duration")
        except Exception as e:
            print(f"Warning: failed to populate panel B (relevant-only fixation): {e}")

        # Panel C: relsign4 relevant styled subject means
        try:
            add_panel_C_relsign4_relevant(ax_C, metric="duration")
        except Exception as e:
            print(f"Warning: failed to populate panel C (relsign4 relevant): {e}")

        # Panel D: regression coefficients by item (horizontal forest plot)
        try:
            add_panel_coef_forest_per_location(ax_D)
        except Exception as e:
            print(f"Warning: failed to populate panel D (coef by item): {e}")

        # Panel E: CV accuracy from prop-time interaction model, under B
        try:
            add_panel_D_cv_accuracy(ax_E)
        except Exception as e:
            print(f"Warning: failed to populate panel E (CV accuracy): {e}")

        # Hide top and right spines on all panels
        for ax in [ax_A, ax_hm, ax_B, ax_C, ax_D, ax_E]:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        fig.subplots_adjust(left=0.08, right=0.94, top=0.95, bottom=0.08)

        out_path = os.path.join(out_dir, "Figure2.pdf")
        fig.savefig(out_path)
        fig_dir = os.path.join(os.getcwd(), "output", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, "Figure2.pdf"))
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create group and individual eyeplots.")
    parser.add_argument(
        "--buffer-ms",
        type=int,
        default=50,
        help="Temporal buffer (ms) used in preprocessing (for file naming).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("output", "eyegaze"),
        help="Directory to save eyeplot figures.",
    )
    args = parser.parse_args()

    base_out_dir = os.path.join(os.getcwd(), args.out_dir)
    os.makedirs(base_out_dir, exist_ok=True)

    # Group eyeplot
    create_eyeplot_layout(
        base_out_dir,
        buffer_ms=args.buffer_ms,
    )



if __name__ == "__main__":
    main()

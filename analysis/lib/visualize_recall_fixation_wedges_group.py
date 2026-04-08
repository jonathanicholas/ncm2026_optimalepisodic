import os
import argparse
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter

def generate_circle_positions(radius: int = 380, n_positions: int = 6):
    """Generate n evenly spaced positions around a circle (same as subject script).

    Starting from top (90 degrees) and going clockwise.
    Returns
    -------
    positions : list of (x, y)
    angles   : list of float (degrees)
    """
    positions = []
    angles = []
    angle_order = [90, 30, -30, -90, -150, 150]

    for angle_deg in angle_order:
        angle = np.deg2rad(angle_deg)
        x = int(radius * np.cos(angle))
        y = int(radius * np.sin(angle))
        positions.append((x, y))
        angles.append(angle_deg)

    return positions, angles


def draw_full_roi(ax, outer_radius: int, central_radius: int) -> None:
    """Draw full circular ROI (outer/central circles and 6 sector lines).

    Coordinates are in the rotated frame with center at (0, 0), y up.
    """
    # Outer and central circles
    outer = Circle((0, 0), outer_radius, fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(outer)

    # Sector lines every 60 degrees, rotated by -30 degrees (matching ROI layout)
    for i in range(6):
        angle_deg = i * 60 - 30
        rad = math.radians(90 - angle_deg)
        x_end = outer_radius * math.cos(rad)
        y_end = outer_radius * math.sin(rad)
        ax.plot([0, x_end], [0, y_end], color="black", linewidth=2)
    center = Circle((0, 0), central_radius, fill=True, edgecolor="black", facecolor='white',linewidth=2,zorder=100)
    ax.add_patch(center)


def find_subjects_with_wedge_data(output_base_dir: str, roi_type: str, buffer_size: int, data_dir: str | None = None):
    """Return list of subjects that have wedge-aligned fixation CSVs for given config.

    If *data_dir* is provided, per-subject data is looked up there instead of
    *output_base_dir*.
    """
    search_dir = data_dir if data_dir is not None else output_base_dir
    subjects = []
    for entry in os.listdir(search_dir):
        subj_dir = os.path.join(search_dir, entry)
        if not os.path.isdir(subj_dir):
            continue
        # Exclude group-style dirs by convention
        if entry.startswith("group"):
            continue
        csv_path = os.path.join(
            subj_dir,
            f"{entry}_wedge_aligned_fixations_{roi_type}_buffer_{buffer_size}.csv",
        )
        if os.path.exists(csv_path):
            subjects.append(entry)
    subjects.sort()
    return subjects


def load_subject_wedge_data(output_base_dir: str, subj: str, roi_type: str, buffer_size: int, data_dir: str | None = None) -> pd.DataFrame:
    """Load per-subject wedge-aligned fixation data produced by the subject script.

    If *data_dir* is provided, per-subject data is loaded from there instead of
    *output_base_dir*.
    """
    search_dir = data_dir if data_dir is not None else output_base_dir
    subj_dir = os.path.join(search_dir, subj)
    csv_path = os.path.join(
        subj_dir,
        f"{subj}_wedge_aligned_fixations_{roi_type}_buffer_{buffer_size}.csv",
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing wedge-aligned data for subject {subj}: {csv_path}")
    return pd.read_csv(csv_path)


def compute_group_wedge_heatmaps(
    output_base_dir: str,
    roi_type: str,
    buffer_size: int,
    bin_size: int,
    outer_radius: int = 530,
    central_radius: int = 20,
    circle_radius: int = 380,
):
    """Compute group-level, per-time-bin wedge heatmaps.

    For each subject and each time bin, compute a duration-weighted 2D histogram
    over (x_rot, y_rot), then normalize within that subject+bin so the map
    represents a *proportion* of that subject's fixation duration.

    Group map per bin is the mean of these subject-normalized maps.
    """
    subjects = find_subjects_with_wedge_data(output_base_dir, roi_type, buffer_size)
    if not subjects:
        print("No subjects with wedge-aligned fixation data found.")
        return

    print(f"Found {len(subjects)} subjects with wedge-aligned data: {subjects}")

    # Group output directory
    group_dir = os.path.join(
        output_base_dir,
        "group_wedge_heatmaps",
        f"{roi_type}_buffer_{buffer_size}",
    )
    plots_dir = os.path.join(group_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Histogram grid (must match subject-level visualization for alignment; full circle)
    x_range = (-outer_radius+5, outer_radius+5)
    y_range = (-outer_radius+5, outer_radius+5)
    n_bins_x = 50
    n_bins_y = 50

    # Collect all bin centers that have any data across subjects
    all_bin_centers = set()
    subj_data = {}
    for subj in subjects:
        df = load_subject_wedge_data(output_base_dir, subj, roi_type, buffer_size)
        # Ensure required columns exist
        required = {"bin_center", "x_rot", "y_rot", "duration"}
        missing = required - set(df.columns)
        if missing:
            print(f"Skipping {subj}: missing columns {missing}")
            continue
        subj_data[subj] = df
        all_bin_centers.update(df["bin_center"].unique())

    if not subj_data:
        print("No valid subject data after column checks.")
        return

    bin_centers = sorted(all_bin_centers)

    # For summary CSV
    summary_rows = []

    for bc in bin_centers:
        subject_maps = []
        contributing_subjects = []

        for subj, df in subj_data.items():
            bin_df = df[df["bin_center"] == bc]
            if bin_df.empty:
                continue

            durations = bin_df["duration"].values
            if durations.sum() <= 0:
                continue

            H, xedges, yedges = np.histogram2d(
                bin_df["x_rot"].values,
                bin_df["y_rot"].values,
                bins=[n_bins_x, n_bins_y],
                range=[x_range, y_range],
                weights=durations,
            )

            total = H.sum()
            if total <= 0:
                continue

            H_norm = H / total  # subject-level proportion map for this bin
            subject_maps.append(H_norm)
            contributing_subjects.append(subj)

        if not subject_maps:
            continue

        # Group mean proportion map
        group_H = np.mean(np.stack(subject_maps, axis=0), axis=0)

        # Smooth slightly to reduce blockiness, allowing smoothing into
        # regions that were previously zero.
        if np.any(group_H):
            group_H_smooth = gaussian_filter(group_H, sigma=1)
        else:
            group_H_smooth = group_H

        # Plot group heatmap with a colormap where 0 maps to gray, then
        # clip to the circular ROI so outside the circle is pure white.
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        gray_hot = sns.color_palette("light:b", as_cmap=True)

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

        # Colorbar placed just to the right of the axes, with height
        # matched to the axes (roughly the circle height) and fully
        # inside the figure.
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
        # Limit colorbar to the actual data range
        data_max = float(np.nanmax(group_H_smooth)) if np.any(group_H_smooth) else 0.0
        im.set_clim(0.0, data_max)
        # Choose ticks between 0 and data_max
        if data_max > 0:
            ticks = np.linspace(0.0, data_max, 5)
        else:
            ticks = np.array([0.0])
        cbar.set_ticks(ticks)
        # Manually scale tick labels by 1e3 and annotate units in label
        cbar.set_ticklabels([f"{t * 1e3:g}" for t in ticks])
        cbar.set_label("Mean proportion of fixation duration (per subject) (×10⁻³)")

        # Draw full circular ROI (all wedges) and item rectangle
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
            edgecolor="cyan",
            linewidth=2,
        )
        ax.add_patch(rect)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel("X (rotated, px)")
        ax.set_ylabel("Y (rotated, px; up)")

        n_contrib = len(contributing_subjects)
        # ax.set_title(
        #     f"Group wedge heatmap (ROI={roi_type}, buffer={buffer_size}px)\n"
        #     f"Bin center {int(bc)} ms (N={n_contrib} subjects)"
        # )

        out_png = os.path.join(
            plots_dir,
            f"group_wedge_heatmap_bin_{int(bc)}ms_{roi_type}_buffer_{buffer_size}.png",
        )
        out_pdf = os.path.join(
            plots_dir,
            f"group_wedge_heatmap_bin_{int(bc)}ms_{roi_type}_buffer_{buffer_size}.pdf",
        )
        # Leave extra margin on the right so the colorbar label
        # has some white space before the edge of the figure.
        fig.tight_layout(rect=(0, 0, 0.85, 1))
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
        plt.close(fig)

        summary_rows.append(
            {
                "bin_center": float(bc),
                "n_subjects": int(n_contrib),
                # Store the PNG path relative to the group directory
                "plot_file": os.path.relpath(out_png, group_dir),
            }
        )

    # Save summary CSV
    if summary_rows:
        os.makedirs(group_dir, exist_ok=True)
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(
            group_dir,
            f"group_wedge_heatmap_summary_{roi_type}_buffer_{buffer_size}.csv",
        )
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved group wedge heatmap summary to {summary_csv}")
    else:
        print("No group heatmaps were generated (no overlapping data across subjects).")


def compute_group_cluster_heatmap(
    output_base_dir: str,
    roi_type: str,
    buffer_size: int,
    outer_radius: int = 530,
    central_radius: int = 20,
    circle_radius: int = 380,
    data_dir: str | None = None,
):
    """Compute and save a group-level wedge heatmap over the significant time cluster.

    This is a plotting wrapper around get_cluster_heatmap_data: it computes
    the cluster-averaged fixation map and then renders it to its own figure
    with colorbar, ROI overlays, and saved PNG/PDF files.
    """
    data = get_cluster_heatmap_data(
        output_base_dir=output_base_dir,
        roi_type=roi_type,
        buffer_size=buffer_size,
        outer_radius=outer_radius,
        central_radius=central_radius,
        circle_radius=circle_radius,
        data_dir=data_dir,
    )
    if data is None:
        return

    t_start = data["t_start"]
    t_end = data["t_end"]
    group_H_smooth = data["group_H_smooth"]
    x_range = data["x_range"]
    y_range = data["y_range"]

    # Output paths aligned with per-bin maps
    group_dir = os.path.join(
        output_base_dir,
        "group_wedge_heatmaps",
        f"{roi_type}_buffer_{buffer_size}",
    )
    plots_dir = os.path.join(group_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")
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
    cbar.set_label("Mean proportion fixation time (×10⁻³)")

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
        linestyle="--",
        linewidth=2,
    )
    ax.add_patch(rect)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    out_png = os.path.join(
        plots_dir,
        f"group_wedge_heatmap_cluster_{int(t_start)}to{int(t_end)}ms_{roi_type}_buffer_{buffer_size}.png",
    )
    out_pdf = os.path.join(
        plots_dir,
        f"group_wedge_heatmap_cluster_{int(t_start)}to{int(t_end)}ms_{roi_type}_buffer_{buffer_size}.pdf",
    )
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved cluster-wide group heatmap to {out_png} and {out_pdf}")


def get_cluster_heatmap_data(
    output_base_dir: str,
    roi_type: str,
    buffer_size: int,
    outer_radius: int = 530,
    central_radius: int = 20,
    circle_radius: int = 380,
    data_dir: str | None = None,
):
    """Compute cluster-averaged fixation map and metadata without plotting.

    Returns a dict with keys:
      t_start, t_end, group_H_smooth, x_range, y_range, outer_radius,
      central_radius, circle_radius.
    If no valid cluster or subject data are found, returns None.

    If *data_dir* is provided, per-subject wedge data is loaded from there
    instead of *output_base_dir*. Group-level cluster JSON is always read
    from *output_base_dir*.
    """
    cluster_path = os.path.join(
        output_base_dir,
        "eyegaze",
        "recall",
        f"group_time_course_clusters_{roi_type}_buffer_{buffer_size}.json",
    )
    if not os.path.exists(cluster_path):
        print(f"No cluster JSON found at {cluster_path}; skipping cluster-wide map.")
        return None

    with open(cluster_path, "r") as f:
        cluster_info = json.load(f)

    clusters = cluster_info.get("clusters", [])
    if not clusters:
        print("Cluster JSON has no clusters; skipping cluster-wide map.")
        return None

    # Pick first significant cluster by p < 0.05 (if p_value present)
    sig_clusters = [c for c in clusters if c.get("p_value") is not None and c["p_value"] < 0.05]
    if not sig_clusters:
        print("No significant clusters (p < 0.05) found; skipping cluster-wide map.")
        return None

    chosen = sig_clusters[0]
    t_start = float(chosen["start_time"])
    t_end = float(chosen["end_time"])
    print(
        f"Using significant cluster from {t_start:.1f} ms to {t_end:.1f} ms "
        f"(p={chosen.get('p_value'):.3g})."
    )

    subjects = find_subjects_with_wedge_data(output_base_dir, roi_type, buffer_size, data_dir=data_dir)
    if not subjects:
        print("No subjects with wedge-aligned fixation data found.")
        return None

    # Histogram grid (full circle, same as per-bin maps)
    x_range = (-outer_radius-5, outer_radius+5)
    y_range = (-outer_radius-5, outer_radius+5)
    n_bins_x = 50
    n_bins_y = 50

    subject_maps = []
    contributing_subjects = []

    for subj in subjects:
        df = load_subject_wedge_data(output_base_dir, subj, roi_type, buffer_size, data_dir=data_dir)
        required = {"time_rel", "x_rot", "y_rot", "duration"}
        missing = required - set(df.columns)
        if missing:
            print(f"Skipping {subj} for cluster map: missing columns {missing}")
            continue

        # Restrict to the chosen cluster window
        bin_df = df[(df["time_rel"] >= t_start) & (df["time_rel"] <= t_end)]
        if bin_df.empty:
            continue

        durations = bin_df["duration"].values
        if durations.sum() <= 0:
            continue

        H, xedges, yedges = np.histogram2d(
            bin_df["x_rot"].values,
            bin_df["y_rot"].values,
            bins=[n_bins_x, n_bins_y],
            range=[x_range, y_range],
            weights=durations,
        )

        total = H.sum()
        if total <= 0:
            continue

        H_norm = H / total
        subject_maps.append(H_norm)
        contributing_subjects.append(subj)

    if not subject_maps:
        print("No subject contributed data within the significant cluster window; skipping map.")
        return None

    group_H = np.mean(np.stack(subject_maps, axis=0), axis=0)

    # Smooth cluster map to reduce blockiness, including regions that were
    # originally zero.
    if np.any(group_H):
        group_H_smooth = gaussian_filter(group_H, sigma=2.0)
    else:
        group_H_smooth = group_H

    return {
        "t_start": t_start,
        "t_end": t_end,
        "group_H_smooth": group_H_smooth,
        "x_range": x_range,
        "y_range": y_range,
        "outer_radius": outer_radius,
        "central_radius": central_radius,
        "circle_radius": circle_radius,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create group-level wedge-aligned fixation heatmaps from subject-level "
            "wedge-aligned fixation CSVs. Each map is normalized per subject and "
            "time bin to represent a proportion of fixation duration, then averaged "
            "across subjects."
        )
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="../data",
        help="Base output directory containing per-subject wedge CSVs",
    )
    parser.add_argument(
        "--roi_type",
        type=str,
        default="original",
        choices=["original", "equal_area"],
        help="ROI type used to generate the subject-level fixation data",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=50,
        help="Buffer size used in the subject-level fixation processing",
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=100,
        help="Time bin size in ms (for reporting titles only; must match subject script)",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("GROUP WEDGE-ALIGNED FIXATION HEATMAPS")
    print("=" * 50)
    print(f"Output base dir: {args.output_base_dir}")
    print(f"ROI type: {args.roi_type}")
    print(f"Buffer size: {args.buffer_size}px")
    print(f"Assumed bin size (subject script): {args.bin_size} ms")

    compute_group_wedge_heatmaps(
        output_base_dir=args.output_base_dir,
        roi_type=args.roi_type,
        buffer_size=args.buffer_size,
        bin_size=args.bin_size,
    )

    # Also compute a single map over the first significant time cluster, if available
    compute_group_cluster_heatmap(
        output_base_dir=args.output_base_dir,
        roi_type=args.roi_type,
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    main()

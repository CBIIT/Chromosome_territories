# Author: Adib Keikhosravi, Ph.D.
# Staff Scientist, Laboratory of Receptor Biology and Gene Expression, CCR, NCI
# National Institutes of Health
# Email: adib.keikhosravi@nih.gov
# License: MIT License

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def savefig(path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _read_inputs(results_dir: Path, df: pd.DataFrame | None, population_summary: pd.DataFrame | None):
    per_nucleus_csv = results_dir / "series7_chrX_arm_measurements_per_nucleus.csv"
    pop_csv = results_dir / "series7_chrX_arm_measurements_population_summary.csv"
    if df is None:
        df = pd.read_csv(per_nucleus_csv)
    if population_summary is None:
        population_summary = pd.read_csv(pop_csv) if pop_csv.exists() else pd.DataFrame()
    return df, population_summary, per_nucleus_csv, pop_csv


def _boxplot(data, labels, **kwargs):
    try:
        return plt.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return plt.boxplot(data, labels=labels, **kwargs)


def create_plots(
    results_dir: str | Path,
    df: pd.DataFrame | None = None,
    population_summary: pd.DataFrame | None = None,
) -> Path:
    """Create Fig. 2-like and QC plots from a per-nucleus measurement table."""
    results_dir = Path(results_dir)
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    df, pop, per_nucleus_csv, pop_csv = _read_inputs(results_dir, df, population_summary)

    if df.empty:
        with open(plot_dir / "plot_summary.txt", "w", encoding="utf-8") as f:
            f.write("No nuclei were measured; no plots were generated.\n")
        return plot_dir

    plt.figure(figsize=(6, 5))
    data = [df["p_fraction_of_nucleus"].dropna(), df["q_fraction_of_nucleus"].dropna()]
    _boxplot(data, ["P arm", "Q arm"], showfliers=False)
    plt.ylabel("Fraction of nuclear volume")
    plt.title("Arm volume as fraction of nucleus")
    savefig(plot_dir / "fig2c_like_arm_fraction_boxplot.png")

    plt.figure(figsize=(6, 5))
    data = [df["pq_overlap_fraction_of_p"].dropna(), df["pq_overlap_fraction_of_q"].dropna()]
    _boxplot(data, ["Overlap / P", "Overlap / Q"], showfliers=False)
    plt.ylabel("Overlap fraction")
    plt.title("P-Q overlap normalized to each arm")
    savefig(plot_dir / "fig2d_like_overlap_boxplot.png")

    plt.figure(figsize=(5, 4))
    contact_freq = float(df["pq_contact"].mean()) if len(df) else np.nan
    plt.bar(["No contact", "Contact"], [1 - contact_freq, contact_freq])
    plt.ylabel("Fraction of nuclei")
    plt.title("P-Q contact frequency")
    savefig(plot_dir / "fig2e_like_contact_frequency.png")

    # Radial shell plot intentionally omitted from the streamlined result set.

    plt.figure(figsize=(6, 4))
    plt.hist(df["pq_overlap_fraction_of_p"].dropna(), bins=20, alpha=0.7, label="Overlap / P")
    plt.hist(df["pq_overlap_fraction_of_q"].dropna(), bins=20, alpha=0.7, label="Overlap / Q")
    plt.xlabel("Overlap fraction")
    plt.ylabel("Number of nuclei")
    plt.title("Distribution of P-Q overlap")
    plt.legend()
    savefig(plot_dir / "overlap_histograms.png")

    plt.figure(figsize=(6, 4))
    plt.hist(df["pq_min_edge_distance_um"].dropna(), bins=20)
    plt.xlabel("Minimum edge-to-edge distance (um)")
    plt.ylabel("Number of nuclei")
    plt.title("Distribution of P-Q minimum edge distance")
    savefig(plot_dir / "edge_distance_histogram.png")

    if "p_fish_signal_count" in df.columns and "q_fish_signal_count" in df.columns:
        p_counts = df["p_fish_signal_count"].fillna(0).astype(int)
        q_counts = df["q_fish_signal_count"].fillna(0).astype(int)
        max_count = int(max(p_counts.max() if len(p_counts) else 0, q_counts.max() if len(q_counts) else 0, 1))
        x = np.arange(max_count + 1)
        p_hist = np.array([(p_counts == i).sum() for i in x], dtype=float)
        q_hist = np.array([(q_counts == i).sum() for i in x], dtype=float)
        width = 0.38
        plt.figure(figsize=(6, 4))
        plt.bar(x - width / 2, p_hist, width=width, label="P arm")
        plt.bar(x + width / 2, q_hist, width=width, label="Q arm")
        plt.xlabel("Number of final FISH signal compartments per nucleus")
        plt.ylabel("Number of nuclei")
        plt.title("P/Q FISH signal count per nucleus")
        plt.xticks(x)
        plt.legend()
        savefig(plot_dir / "fish_signal_count_distribution.png")

    plt.figure(figsize=(5.5, 5))
    plt.scatter(df["p_volume_um3"], df["q_volume_um3"], alpha=0.8)
    plt.xlabel("P-arm volume (um^3)")
    plt.ylabel("Q-arm volume (um^3)")
    plt.title("P vs Q arm volume")
    savefig(plot_dir / "p_vs_q_volume_scatter.png")

    plt.figure(figsize=(5.5, 5))
    mean_arm_vol = 0.5 * (df["p_volume_um3"] + df["q_volume_um3"])
    plt.scatter(mean_arm_vol, df["pq_overlap_volume_um3"], alpha=0.8)
    plt.xlabel("Mean arm volume (um^3)")
    plt.ylabel("P-Q overlap volume (um^3)")
    plt.title("Mean arm volume vs overlap")
    savefig(plot_dir / "mean_arm_volume_vs_overlap_scatter.png")

    dz = df["p_centroid_z_um"] - df["q_centroid_z_um"]
    dy = df["p_centroid_y_um"] - df["q_centroid_y_um"]
    dx = df["p_centroid_x_um"] - df["q_centroid_x_um"]
    centroid_dist = np.sqrt(dz**2 + dy**2 + dx**2)
    plt.figure(figsize=(6, 4))
    plt.hist(centroid_dist.dropna(), bins=20)
    plt.xlabel("Centroid-to-centroid distance (um)")
    plt.ylabel("Number of nuclei")
    plt.title("Distribution of P-Q centroid separation")
    savefig(plot_dir / "centroid_distance_histogram.png")

    # Optional 3D shape and radial-position QC plots.
    if "p_radial_centroid" in df.columns and "q_radial_centroid" in df.columns:
        plt.figure(figsize=(6, 5))
        data = [df["p_radial_centroid"].dropna(), df["q_radial_centroid"].dropna()]
        _boxplot(data, ["P centroid", "Q centroid"], showfliers=False)
        plt.ylabel("Radial coordinate (0=center, 1=surface)")
        plt.title("Centroid radial position")
        savefig(plot_dir / "ct_centroid_radial_position_boxplot.png")

    if "p_shape_sphericity" in df.columns and "q_shape_sphericity" in df.columns:
        plt.figure(figsize=(6, 5))
        data = [df["p_shape_sphericity"].dropna(), df["q_shape_sphericity"].dropna()]
        _boxplot(data, ["P arm", "Q arm"], showfliers=False)
        plt.ylabel("Sphericity")
        plt.title("Arm 3D sphericity")
        savefig(plot_dir / "ct_sphericity_boxplot.png")

    if "nucleus_shape_sphericity" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df["nucleus_shape_sphericity"].dropna(), bins=20)
        plt.xlabel("Nucleus sphericity")
        plt.ylabel("Number of nuclei")
        plt.title("Distribution of nuclear sphericity")
        savefig(plot_dir / "nucleus_sphericity_histogram.png")

    # Internal presence-gate diagnostic plots are intentionally omitted from the streamlined public output.

    with open(plot_dir / "plot_summary.txt", "w", encoding="utf-8") as f:
        f.write("Plots created from:\n")
        f.write(f"  {per_nucleus_csv}\n")
        if pop_csv.exists():
            f.write(f"  {pop_csv}\n")
        f.write("\nPopulation summary:\n")
        if len(pop):
            f.write(pop.to_string(index=False))
            f.write("\n")
        f.write("\nOutputs:\n")
        for p in sorted(plot_dir.glob("*.png")):
            f.write(f"  {p.name}\n")

    return plot_dir

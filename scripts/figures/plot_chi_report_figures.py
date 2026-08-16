#!/usr/bin/env python3
"""Generate ACM CHI-style figures for the final CVS project report.

Outputs are sized for the full width of the final two-column ACM layout and
exported as vector PDF/SVG plus 600-dpi PNG. Numerical figures read canonical
analysis CSVs directly; a manifest records source hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cvs_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cvs_font_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
ANALYSIS = PROJECT / "05_analysis_outputs"
OUTPUT = PROJECT / "06_figures" / "chi_report"

INTENSITY_DIR = ANALYSIS / "own_data_intensity_edge_validation"
ML_DIR = ANALYSIS / "final_ml_validation"

INPUTS = [
    INTENSITY_DIR / "intensity_edge_null_distributions.csv",
    INTENSITY_DIR / "intensity_edge_null_summary.csv",
    INTENSITY_DIR / "intensity_edge_split_half_reliability.csv",
    ML_DIR / "nested_ml_performance.csv",
    ML_DIR / "incremental_feature_block_tests.csv",
    ML_DIR / "ridge_whole_pipeline_permutations.csv",
]

# Okabe-Ito colorblind-safe palette. Patterns and marker shapes preserve meaning
# when printed in grayscale.
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
CHARCOAL = "#252525"
MIDGRAY = "#777777"
LIGHTGRAY = "#E5E5E5"
PALE = "#F7F7F7"

FULL_WIDTH = 7.16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", color=CHARCOAL)


def finish_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.7, color=CHARCOAL)
    if grid_axis:
        ax.grid(axis=grid_axis, color=LIGHTGRAY, linewidth=0.6, zorder=0)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if extension == "png" else {}
        fig.savefig(OUTPUT / f"{stem}.{extension}", **kwargs)
    plt.close(fig)


def add_workflow_box(
    ax: plt.Axes,
    x: float,
    width: float,
    title: str,
    lines: list[str],
    color: str,
) -> None:
    y, height, header = 14, 69, 15
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.0,rounding_size=1.8",
        facecolor="white", edgecolor="#BDBDBD", linewidth=0.9,
    )
    ax.add_patch(box)
    header_patch = Rectangle(
        (x, y + height - header), width, header,
        facecolor=color, edgecolor="none",
    )
    header_patch.set_clip_path(box)
    ax.add_patch(header_patch)
    ax.text(x + width / 2, y + height - header / 2, title,
            ha="center", va="center", color="white", fontsize=7.4,
            fontweight="bold")
    start_y = y + height - header - 7
    spacing = 9.2 if len(lines) <= 4 else (7.7 if len(lines) <= 5 else 6.5)
    for index, line in enumerate(lines):
        ax.text(x + 2.2, start_y - index * spacing, line,
                ha="left", va="top", fontsize=5.45, color=CHARCOAL,
                linespacing=1.12)


def figure_workflow() -> None:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.05))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.02, top=0.98)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 88)
    ax.axis("off")

    width = 18.0
    xs = [0, 20.5, 41.0, 61.5, 82.0]
    stages = [
        ("1  STUDY", [
            "24 unfamiliar dyads",
            "Piper / Cloudy",
            "2 films x 5 prompts",
            "46 discussions",
            "~10 min each",
        ], BLUE),
        ("2  CAPTURE", [
            "Corrected turns",
            "WhisperX timing",
            "Shared-mic audio",
            "Fixed-camera video",
            "Connection survey",
        ], ORANGE),
        ("3  FEATURES", [
            "Transcript",
            "  structure + semantics",
            "Acoustic",
            "  500-ms handoff edges",
            "  F0 / intensity / rate",
            "Video",
            "  pose + head proxies",
        ], GREEN),
        ("4  QC + AGG.", [
            "Turn -> session -> dyad",
            "Pseudo-pair null",
            "Time-shift null",
            "Split-half reliability",
            "Tracking + mapping",
        ], PURPLE),
        ("5  ANALYSIS", [
            "Hierarchical delta R2",
            "Permutation tests",
            "CANDOR replication",
            "  N = 1,653",
            "Nested LODO ML",
        ], VERMILION),
    ]
    for x, (title, lines, color) in zip(xs, stages):
        add_workflow_box(ax, x, width, title, lines, color)
    for left, right in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch(
            (left + width + 0.5, 48.5), (right - 0.5, 48.5),
            arrowstyle="-|>", mutation_scale=10, linewidth=1.1,
            color=MIDGRAY, shrinkA=0, shrinkB=0,
        ))

    # A compact visual definition of the key acoustic handoff measurement.
    ax.plot([42.8, 47.2], [7.0, 7.0], color=BLUE, linewidth=3.2,
            solid_capstyle="butt")
    ax.plot([47.2, 51.6], [7.0, 7.0], color=ORANGE, linewidth=3.2,
            solid_capstyle="butt")
    ax.axvline(47.2, ymin=0.045, ymax=0.13, color=CHARCOAL,
               linewidth=0.8, linestyle="--")
    ax.text(45.0, 2.4, "A", ha="center", va="top",
            fontsize=5.7, color=CHARCOAL)
    ax.text(49.4, 2.4, "B", ha="center", va="top",
            fontsize=5.7, color=CHARCOAL)
    ax.text(47.2, 10.0, "floor transfer", ha="center", va="bottom",
            fontsize=6.3, fontweight="bold", color=CHARCOAL)
    save_figure(fig, "fig1_methods_workflow")


def plot_null_panel(
    ax: plt.Axes,
    distributions: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
) -> None:
    pseudo = distributions[f"pseudo_{metric}_mean"].to_numpy()
    shifted = distributions[f"time_shift_{metric}_mean"].to_numpy()
    observed = float(summary.loc[
        (summary.metric == metric) & (summary["null"] == "pseudo"), "real_mean"
    ].iloc[0])
    combined = np.concatenate([pseudo, shifted])
    bins = np.linspace(combined.min(), max(combined.max(), observed) * 1.03, 34)
    ax.hist(pseudo, bins=bins, density=True, color=SKY, alpha=0.55,
            edgecolor=BLUE, linewidth=0.5, label="Pseudo-pair null")
    _, _, shifted_patches = ax.hist(
        shifted, bins=bins, density=True, histtype="stepfilled",
        color="white", edgecolor=ORANGE, linewidth=0.8, hatch="////",
        label="Time-shift null",
    )
    for patch in shifted_patches:
        patch.set_facecolor("none")
    ax.axvline(observed, color=VERMILION, linewidth=2.0,
               label="Real adjacency")
    ax.text(0.98, 0.94, "one-sided p = .001",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=CHARCOAL)
    ax.set_xlabel(f"Mean intensity-edge {metric}")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    finish_axes(ax)


def figure_acoustic_validation() -> None:
    distributions = pd.read_csv(INPUTS[0])
    summary = pd.read_csv(INPUTS[1])
    reliability = pd.read_csv(INPUTS[2])

    fig, axes = plt.subplots(
        1, 3, figsize=(FULL_WIDTH, 2.55),
        gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]},
    )
    plot_null_panel(axes[0], distributions, summary, "proximity")
    plot_null_panel(axes[1], distributions, summary, "synchrony")
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.38, -0.02),
               ncol=3, frameon=False, handlelength=2.2)

    ax = axes[2]
    order = ["proximity", "synchrony", "composite"]
    rel = reliability.set_index("metric").loc[order]
    y = np.arange(len(order))[::-1]
    ax.axvline(0, color=MIDGRAY, linewidth=0.8, linestyle="--", zorder=1)
    ax.scatter(rel.split_half_pearson_r, y, s=38, color=PURPLE,
               edgecolor=CHARCOAL, linewidth=0.6, marker="D", zorder=3)
    for value, ypos in zip(rel.split_half_pearson_r, y):
        ax.text(value - 0.012 if value < 0 else value + 0.012, ypos,
                f"r = {value:.2f}", va="center",
                ha="right" if value < 0 else "left", fontsize=7)
    ax.set_yticks(y, ["Proximity", "Synchrony", "Composite"])
    ax.set_xlim(-0.28, 0.16)
    ax.set_ylim(-0.7, 2.7)
    ax.set_xlabel("Odd-even split-half correlation")
    ax.text(0.5, 0.02, "Spearman-Brown reliability = 0 for all",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8)
    finish_axes(ax, "x")
    panel_label(ax, "C")
    fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.27, wspace=0.42)
    save_figure(fig, "fig2_acoustic_measurement_validation")


def short_block(name: str) -> str:
    return {
        "transcript": "Transcript",
        "transcript_acoustic": "+ Acoustic",
        "transcript_acoustic_video": "+ Video",
        "full_exploratory_llm": "+ LLM proxies",
    }[name]


def figure_predictive_validation() -> None:
    performance = pd.read_csv(INPUTS[3])
    incremental = pd.read_csv(INPUTS[4])
    permutations = pd.read_csv(INPUTS[5])
    block_order = [
        "transcript", "transcript_acoustic", "transcript_acoustic_video",
        "full_exploratory_llm",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.65),
                             gridspec_kw={"width_ratios": [0.92, 1.2]})
    ax = axes[0]
    x = np.arange(len(block_order))
    for algorithm, offset, color, marker, label in [
        ("ridge", -0.09, BLUE, "o", "Ridge"),
        ("elastic_net", 0.09, ORANGE, "s", "Elastic Net"),
    ]:
        values = (performance[performance.algorithm == algorithm]
                  .set_index("feature_block").loc[block_order, "cv_r2"])
        ax.plot(x + offset, values, color=color, marker=marker, markersize=5,
                linewidth=1.0, label=label)
    ax.axhline(0, color=CHARCOAL, linewidth=0.8, linestyle="--")
    ax.set_xticks(x, [short_block(block) for block in block_order], rotation=25,
                  ha="right")
    ax.set_ylabel("Leave-one-dyad-out CV $R^2$")
    ax.legend(frameon=False, loc="lower left")
    finish_axes(ax, "y")
    panel_label(ax, "A")

    ax = axes[1]
    permutation_wide = permutations.pivot(
        index="permutation", columns="feature_block", values="cv_rmse"
    )
    comparisons = [
        ("transcript", "transcript_acoustic", "Add acoustic"),
        ("transcript_acoustic", "transcript_acoustic_video", "Add video"),
        ("transcript_acoustic_video", "full_exploratory_llm", "Add LLM proxies"),
    ]
    y_positions = np.arange(len(comparisons))[::-1]
    for ypos, (base, expanded, label) in zip(y_positions, comparisons):
        null_gain = permutation_wide[base] - permutation_wide[expanded]
        low, high = np.quantile(null_gain, [0.025, 0.975])
        row = incremental[
            (incremental.base_block == base)
            & (incremental.expanded_block == expanded)
        ].iloc[0]
        ax.plot([low, high], [ypos, ypos], color=MIDGRAY, linewidth=5,
                alpha=0.55, solid_capstyle="round", zorder=1)
        ax.scatter(row.observed_rmse_gain, ypos, s=45, color=GREEN,
                   edgecolor=CHARCOAL, linewidth=0.6, zorder=3)
        ax.text(high + 0.008, ypos, f"p = {row.permutation_p:.3f}",
                va="center", ha="left", fontsize=7)
    ax.axvline(0, color=CHARCOAL, linewidth=0.8, linestyle="--")
    ax.set_yticks(y_positions, [item[2] for item in comparisons])
    ax.set_xlabel("Observed RMSE gain (positive = better prediction)")
    ax.set_xlim(-0.14, 0.21)
    finish_axes(ax, "x")
    panel_label(ax, "B")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.27, wspace=0.38)
    save_figure(fig, "fig3_predictive_validation")


def write_manifest() -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).relative_to(PROJECT)),
        "script_sha256": sha256(Path(__file__)),
        "inputs": [
            {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
            for path in INPUTS
        ],
        "outputs": sorted(
            path.name for path in OUTPUT.iterdir()
            if path.suffix in {".pdf", ".svg", ".png"}
        ),
        "design": {
            "target_width_inches": FULL_WIDTH,
            "raster_dpi": 600,
            "vector_formats": ["pdf", "svg"],
            "palette": "Okabe-Ito colorblind-safe",
            "grayscale_redundancy": "marker shapes, line styles, and hatching",
        },
    }
    (OUTPUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    missing = [path for path in INPUTS if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required inputs: {missing}")
    configure_style()
    figure_workflow()
    figure_acoustic_validation()
    figure_predictive_validation()
    write_manifest()
    print(f"Wrote CHI report figures to {OUTPUT}")


if __name__ == "__main__":
    main()

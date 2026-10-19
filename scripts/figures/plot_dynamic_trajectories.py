"""Plot within-conversation dynamic trajectory figures.

Generates:
1. Spaghetti plot: per-dyad feature trajectories colored by partner_eval
2. Coefficient forest plot: static vs dynamic predictor betas with CIs
3. Scatter: strongest dynamic feature vs partner_eval

Inputs
------
- 05_analysis_outputs/dynamic_trajectory/windowed_features.csv
- 05_analysis_outputs/dynamic_trajectory/trajectory_summaries.csv
- 05_analysis_outputs/dynamic_trajectory/model_results.csv

Outputs
-------
- 06_figures/fig_dynamic_trajectory_spaghetti.{pdf,png}
- 06_figures/fig_dynamic_trajectory_coefficients.{pdf,png}
- 06_figures/fig_dynamic_trajectory_scatter.{pdf,png}

Example
-------
    python scripts/figures/plot_dynamic_trajectories.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

ANALYSIS = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs"))
FIGURES = Path(os.environ.get("CVS_FIGURES", PROJECT / "06_figures"))
IN_DIR = ANALYSIS / "dynamic_trajectory"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.{png,pdf}", FIGURES / stem)


def z(v: np.ndarray) -> np.ndarray:
    sd = np.nanstd(v, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.full_like(v, np.nan)
    return (v - np.nanmean(v)) / sd


# ---------------------------------------------------------------------------
# Figure 1: Spaghetti plot
# ---------------------------------------------------------------------------

def fig_spaghetti(
    windowed: pd.DataFrame,
    traj: pd.DataFrame,
    feature: str = "mean_turn_dur",
    window_type: str = "question",
) -> None:
    """Per-dyad trajectory lines colored by partner_eval."""
    wf = windowed[windowed["window_type"] == window_type].copy()
    if len(wf) == 0:
        logger.warning("No windowed data for %s, skipping spaghetti.", window_type)
        return

    # Aggregate to dyad level per window: average across speakers and conditions
    dyad_win = (
        wf.groupby(["pair_id", "window"])[feature]
        .mean()
        .reset_index()
    )

    # Get partner_eval per dyad
    pe = traj[traj["window_type"] == window_type][["pair_id", "partner_eval"]].drop_duplicates()
    dyad_win = dyad_win.merge(pe, on="pair_id", how="left")

    if dyad_win["partner_eval"].isna().all():
        logger.warning("No partner_eval data, skipping spaghetti.")
        return

    # Median split for coloring
    med = dyad_win["partner_eval"].median()

    fig, ax = plt.subplots(figsize=(6, 4))

    for pid, g in dyad_win.groupby("pair_id"):
        g = g.sort_values("window")
        pe_val = g["partner_eval"].iloc[0]
        color = "#d62728" if pe_val >= med else "#1f77b4"
        alpha = 0.5
        ax.plot(g["window"], g[feature], color=color, alpha=alpha, linewidth=1)

    # Group means
    for label, mask_fn, color, ls in [
        ("High connection", lambda x: x >= med, "#d62728", "-"),
        ("Low connection", lambda x: x < med, "#1f77b4", "--"),
    ]:
        sub = dyad_win[mask_fn(dyad_win["partner_eval"])]
        means = sub.groupby("window")[feature].mean()
        ax.plot(means.index, means.values, color=color, linewidth=2.5,
                linestyle=ls, label=label, zorder=5)

    ax.set_xlabel(f"{'Question' if window_type == 'question' else 'Time bin'}")
    ax.set_ylabel(feature.replace("_", " ").title())
    ax.set_title(f"{feature.replace('_', ' ').title()} Trajectory by Connection Level")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    if window_type == "question":
        ax.set_xticks([1, 2, 3, 4, 5])

    save(fig, "fig_dynamic_trajectory_spaghetti")


# ---------------------------------------------------------------------------
# Figure 2: Coefficient forest plot
# ---------------------------------------------------------------------------

def fig_coefficients(results: pd.DataFrame, window_type: str = "question") -> None:
    """Forest plot comparing static vs dynamic predictor betas."""
    res = results[results["window_type"] == window_type].copy()
    if len(res) == 0:
        logger.warning("No results for %s, skipping coefficient plot.", window_type)
        return

    # Select static-only and the three dynamic-only models
    mask = res["model"].isin(["static_only", "dynamic_only_slope",
                               "dynamic_only_early_late_diff", "dynamic_only_sd"])
    plot_data = res[mask].copy()
    if len(plot_data) == 0:
        return

    # Create readable labels
    model_labels = {
        "static_only": "Mean",
        "dynamic_only_slope": "Slope",
        "dynamic_only_early_late_diff": "Early-Late Diff",
        "dynamic_only_sd": "SD",
    }
    plot_data["model_label"] = plot_data["model"].map(model_labels)

    # Sort by feature then model type
    plot_data = plot_data.sort_values(["feature", "model"])

    # Create combined label
    plot_data["label"] = plot_data["feature"].str.replace("_", " ") + " — " + plot_data["model_label"]

    fig, ax = plt.subplots(figsize=(7, max(4, len(plot_data) * 0.28)))

    y_pos = np.arange(len(plot_data))
    colors = plot_data["model"].map({
        "static_only": "#1f77b4",
        "dynamic_only_slope": "#ff7f0e",
        "dynamic_only_early_late_diff": "#2ca02c",
        "dynamic_only_sd": "#9467bd",
    })

    ax.barh(y_pos, plot_data["beta"], color=colors, alpha=0.7, height=0.6)
    ax.errorbar(
        plot_data["beta"], y_pos,
        xerr=[
            plot_data["beta"] - plot_data["ci95_lo"],
            plot_data["ci95_hi"] - plot_data["beta"],
        ],
        fmt="none", color="black", linewidth=0.8, capsize=2,
    )

    # Mark significant results
    for i, (_, row) in enumerate(plot_data.iterrows()):
        if row["p_perm"] < 0.05:
            ax.text(
                max(row["ci95_hi"], row["beta"]) + 0.02, i, "*",
                fontsize=12, fontweight="bold", va="center",
            )

    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["label"], fontsize=8)
    ax.set_xlabel("Standardized Beta (→ partner_eval)")
    ax.set_title("Static vs Dynamic Predictors of Connection")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    save(fig, "fig_dynamic_trajectory_coefficients")


# ---------------------------------------------------------------------------
# Figure 3: Scatter of strongest dynamic predictor
# ---------------------------------------------------------------------------

def fig_scatter(
    traj: pd.DataFrame,
    results: pd.DataFrame,
    window_type: str = "question",
) -> None:
    """Scatter of strongest dynamic predictor vs partner_eval."""
    res = results[
        (results["window_type"] == window_type)
        & (results["model"].str.startswith("dynamic_only"))
    ].sort_values("p_perm")

    if len(res) == 0:
        logger.warning("No dynamic results, skipping scatter.")
        return

    best = res.iloc[0]
    predictor_col = best["predictor"]
    feat_label = predictor_col.replace("_", " ")

    sub = traj[traj["window_type"] == window_type].copy()
    if predictor_col not in sub.columns:
        logger.warning("Column %s not in traj, skipping scatter.", predictor_col)
        return

    mask = sub[predictor_col].notna() & sub["partner_eval"].notna()
    sub = sub[mask]
    if len(sub) < 3:
        return

    x = sub[predictor_col].values
    y = sub["partner_eval"].values

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, alpha=0.7, edgecolors="white", linewidth=0.5, s=50)

    # Regression line
    mask_finite = np.isfinite(x) & np.isfinite(y)
    if mask_finite.sum() > 2:
        slope, intercept = np.polyfit(x[mask_finite], y[mask_finite], 1)
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#d62728",
                linewidth=1.5, linestyle="--")

    ax.set_xlabel(feat_label.title())
    ax.set_ylabel("Partner Evaluation (dyad mean)")
    ax.set_title(
        f"r = {best['r']:.2f}, p_perm = {best['p_perm']:.3f}",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "fig_dynamic_trajectory_scatter")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load data
    windowed_path = IN_DIR / "windowed_features.csv"
    traj_path = IN_DIR / "trajectory_summaries.csv"
    results_path = IN_DIR / "model_results.csv"

    for p in (windowed_path, traj_path, results_path):
        if not p.exists():
            sys.exit(f"Missing input: {p}\nRun build_dynamic_trajectory_features.py "
                     f"and analyze_dynamic_trajectories.py first.")

    windowed = pd.read_csv(windowed_path)
    traj = pd.read_csv(traj_path)
    results = pd.read_csv(results_path)

    logger.info("windowed: %d rows, traj: %d rows, results: %d rows",
                len(windowed), len(traj), len(results))

    # Determine primary window type
    wtype = "question" if "question" in results["window_type"].values else results["window_type"].iloc[0]

    fig_spaghetti(windowed, traj, feature="mean_turn_dur", window_type=wtype)
    fig_coefficients(results, window_type=wtype)
    fig_scatter(traj, results, window_type=wtype)

    logger.info("All figures written to %s", FIGURES)


if __name__ == "__main__":
    main()

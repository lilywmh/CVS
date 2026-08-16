"""Paper figures for the multimodal video-behavior stage.

Purpose
-------
Three figures supporting the multimodal dissociation result
(05_analysis_outputs/multimodal_video_incremental/):

1. fig_video_distinctness.(png|pdf)      — predictor correlation heatmap
   (turn composite, vocal intensity-edge index, 4 video variables).
2. fig_video_speaker_validity.(png|pdf)  — per-participant speaker vs listener
   wrist speed (paired slope plot): behavioral validity of the video measures.
3. fig_video_incremental.(png|pdf)       — incremental ΔR² in the connection
   composite: vocal index over turn-level model vs each video variable over
   the turn+vocal model, annotated with permutation p.

Inputs
------
- 05_analysis_outputs/multimodal_video_incremental/{distinctness_matrix.csv,
  incremental_results.csv}
- <turn-run>/results/speaker_listener_arm_comparison.csv
- 04_data tables via scripts.models.analyze_multimodal_video_incremental.load_inputs
  (used to recompute the vocal-over-turn ΔR² with the identical pipeline).

Example
-------
    python scripts/figures/plot_multimodal_video_figures.py

How to check correctness
------------------------
- Slope plot: most lines should rise from listener to speaker (≈88%).
- ΔR² bars for video variables must match incremental_results.csv
  (delta_r2_overM2 column) exactly.
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

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.models.analyze_multimodal_video_incremental import (  # noqa: E402
    CONTROLS,
    TURN_BLOCK,
    VIDEO_VARS,
    VOCAL_INDEX_COLS,
    incremental_test,
    load_inputs,
    pc1,
    zscore,
)

FIGURES = Path(os.environ.get("CVS_FIGURES", PROJECT / "06_figures"))
RESULTS = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs")) / "multimodal_video_incremental"
TURN_RUN = PROJECT / "data/interim/video_behavior/turn_runs/turn_aligned_video_20260707_120008"

LABELS = {
    "turn_composite": "Turn-level composite",
    "vocal_intensity_edge": "Vocal intensity-edge",
    "video_delta_wrist_speed_px_s": "Video: speaker−listener\nwrist speed",
    "video_delta_active_motion_rate": "Video: speaker−listener\nactive motion",
    "video_speaker_arm_openness": "Video: arm openness\n(speaking)",
    "video_mutual_partner_orientation": "Video: mutual partner\norientation",
}

logger = logging.getLogger("plot_multimodal_video_figures")


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.{png,pdf}", FIGURES / stem)


def fig_distinctness() -> None:
    path = RESULTS / "distinctness_matrix.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run analyze_multimodal_video_incremental.py first.")
    m = pd.read_csv(path, index_col=0)
    labels = [LABELS.get(c, c) for c in m.columns]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(m.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = m.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Predictor distinctness (N = 24 dyads)", fontsize=11)
    save(fig, "fig_video_distinctness")


def fig_speaker_validity() -> None:
    path = TURN_RUN / "results" / "speaker_listener_arm_comparison.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}.")
    df = pd.read_csv(path)
    person = df.groupby(["dyad_id", "participant_id"])[
        ["speaker_mean_wrist_speed_px_s", "listener_mean_wrist_speed_px_s"]
    ].mean().reset_index()
    pos_share = (person.speaker_mean_wrist_speed_px_s
                 > person.listener_mean_wrist_speed_px_s).mean()

    fig, ax = plt.subplots(figsize=(4.6, 5.4))
    for _, row in person.iterrows():
        up = row.speaker_mean_wrist_speed_px_s > row.listener_mean_wrist_speed_px_s
        ax.plot([0, 1],
                [row.listener_mean_wrist_speed_px_s, row.speaker_mean_wrist_speed_px_s],
                color="tab:blue" if up else "tab:red", alpha=0.45, lw=1.2)
    means = [person.listener_mean_wrist_speed_px_s.mean(),
             person.speaker_mean_wrist_speed_px_s.mean()]
    ax.plot([0, 1], means, color="black", lw=3, marker="o", ms=7, zorder=5,
            label="Sample mean")
    ax.set_xticks([0, 1], ["While listening", "While speaking"], fontsize=10)
    ax.set_ylabel("Mean wrist speed (px/s)", fontsize=10)
    ax.set_title(
        f"Behavioral validity: {pos_share:.0%} of participants\n"
        f"move more while speaking (n = {len(person)})", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_video_speaker_validity")


def fig_incremental() -> None:
    inc_path = RESULTS / "incremental_results.csv"
    if not inc_path.exists():
        raise SystemExit(f"Missing {inc_path}; run analyze_multimodal_video_incremental.py first.")
    inc = pd.read_csv(inc_path)
    primary = inc[inc.outcome == "connection_composite"].set_index("video_variable")

    # Recompute vocal-over-turn ΔR² with the identical pipeline (seed fixed).
    df, _ = load_inputs()
    rng = np.random.default_rng(42)
    turn_c, _, _ = pc1(df[TURN_BLOCK], anchor="total_turns")
    vocal_index = zscore(np.mean([zscore(df[c]) for c in VOCAL_INDEX_COLS], axis=0))
    controls = np.column_stack([zscore(df[c]) for c in CONTROLS])
    y = np.column_stack([zscore(df[c]) for c in
                         ["dyad_partner_eval_mean", "dyad_shared_reality_mean",
                          "dyad_enjoyment_mean"]]).mean(axis=1)
    m1 = np.column_stack([turn_c, controls])
    vocal_res = incremental_test(y, m1, vocal_index, 5000, rng)

    names = ["Vocal intensity-edge\n(over turn-level)"] + [
        LABELS[v].replace("Video: ", "") + "\n(over turn + vocal)" for v in VIDEO_VARS]
    values = [vocal_res["delta_r2"]] + [primary.loc[v, "delta_r2_overM2"] for v in VIDEO_VARS]
    pvals = [vocal_res["p_perm"]] + [primary.loc[v, "p_perm_overM2"] for v in VIDEO_VARS]
    colors = ["tab:orange"] + ["tab:blue"] * len(VIDEO_VARS)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.85)
    for bar, p in zip(bars, pvals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"p_perm = {p:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_ylabel("ΔR² in connection composite", fontsize=10)
    ax.set_title("Incremental validity: vocal alignment vs video behavior (N = 24)",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="black", lw=0.8)
    save(fig, "fig_video_incremental")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fig_distinctness()
    fig_speaker_validity()
    fig_incremental()


if __name__ == "__main__":
    main()

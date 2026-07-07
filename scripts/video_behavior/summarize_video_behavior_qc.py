#!/usr/bin/env python3
"""
Summarize turn-aligned video behavior outputs for QC and first-pass descriptive
analysis.

This script consumes a completed run from run_turn_aligned_video_analysis.py and
creates analysis-ready QC summaries, descriptive tables, exploratory
condition/outcome summaries, and simple figures. It does not modify the source
run outputs.

Usage:
  python scripts/video_behavior/summarize_video_behavior_qc.py
  python scripts/video_behavior/summarize_video_behavior_qc.py --run-dir data/interim/video_behavior/turn_runs/<run_id>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
ANALYSIS_OUTPUTS = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs"))
DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
TURN_RUNS = PROJECT / "data" / "interim" / "video_behavior" / "turn_runs"

FEATURE_COLS = [
    "pose_detection_rate",
    "hand_detection_rate",
    "mean_wrist_speed_px_s",
    "max_wrist_speed_px_s",
    "mean_hand_to_face_distance_px",
    "mean_hand_to_shoulder_distance_px",
    "gesture_frame_rate",
]

OUTCOME_COLS = [
    "dyad_partner_eval_mean",
    "dyad_shared_reality_mean",
    "dyad_enjoyment_mean",
    "dyad_solo_mean",
]


def latest_run_dir() -> Path:
    runs = sorted(p for p in TURN_RUNS.glob("turn_aligned_video_*") if (p / "results").exists())
    if not runs:
        raise SystemExit(f"No completed video run found under {TURN_RUNS}")
    return runs[-1]


def read_run_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name, rel in {
        "turn": "results/per_turn_video_features.csv",
        "question": "results/per_question_summary.csv",
        "speaker": "results/per_speaker_summary.csv",
        "session": "results/per_session_summary.csv",
        "turn_qc": "qc/turn_level_qc.csv",
    }.items():
        path = run_dir / rel
        if not path.exists():
            raise SystemExit(f"Missing expected run output: {path}")
        tables[name] = pd.read_csv(path)
    return tables


def finite_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in FEATURE_COLS if col in df.columns]


def qc_counts(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    counts = (
        df.groupby(group_cols + ["qc_status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["PASS", "WARNING", "FAIL"]:
        if col not in counts:
            counts[col] = 0
    counts["n_units"] = counts[["PASS", "WARNING", "FAIL"]].sum(axis=1)
    counts["fail_rate"] = counts["FAIL"] / counts["n_units"].replace(0, np.nan)
    counts["warning_or_fail_rate"] = (counts["WARNING"] + counts["FAIL"]) / counts["n_units"].replace(0, np.nan)
    return counts


def detection_quality_by_session(session: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dyad_id", "session_id", "video_id", "number_of_turns", "number_of_questions",
        "qc_status", "mean_pose_detection_rate", "mean_hand_detection_rate",
        "mean_missingness_rate", "mean_gesture_frame_rate",
        "mean_mean_wrist_speed_px_s",
    ]
    return session[[c for c in cols if c in session.columns]].sort_values(["qc_status", "mean_hand_detection_rate"])


def detection_quality_by_speaker(speaker: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dyad_id", "session_id", "video_id", "speaker_label", "participant_id",
        "number_of_turns", "total_speaking_time_sec", "qc_status",
        "mean_pose_detection_rate", "mean_hand_detection_rate",
        "mean_missingness_rate", "mean_gesture_frame_rate",
        "mean_mean_wrist_speed_px_s",
    ]
    return speaker[[c for c in cols if c in speaker.columns]].sort_values(["qc_status", "mean_hand_detection_rate"])


def feature_descriptives(turn: pd.DataFrame, question: pd.DataFrame, speaker: pd.DataFrame, session: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, df in [("turn", turn), ("question", question), ("speaker", speaker), ("session", session)]:
        cols = [c for c in df.columns if any(f in c for f in FEATURE_COLS) and pd.api.types.is_numeric_dtype(df[c])]
        for col in cols:
            s = pd.to_numeric(df[col], errors="coerce")
            rows.append({
                "level": level,
                "feature": col,
                "n": int(s.notna().sum()),
                "missing_rate": float(s.isna().mean()),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "sd": float(s.std()) if s.notna().sum() > 1 else np.nan,
                "median": float(s.median()) if s.notna().any() else np.nan,
                "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
                "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
            })
    return pd.DataFrame(rows)


def condition_summary(session: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in session.columns if c.startswith("mean_") and pd.api.types.is_numeric_dtype(session[c])]
    rows = []
    for condition, group in session.groupby("session_id", dropna=False):
        row = {"condition": condition, "n_sessions": len(group)}
        for col in feature_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_sd"] = group[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


def load_outcomes() -> pd.DataFrame | None:
    candidates = [DATA / "dyad_level_dataset.csv", DATA / "outcomes.csv"]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "pair_id" in df.columns or "dyad_id" in df.columns:
                return df
    return None


def session_to_dyad_features(session: pd.DataFrame) -> pd.DataFrame:
    df = session.copy()
    if "pair_id" not in df.columns:
        df["pair_id"] = df["dyad_id"].astype(str).str.extract(r"dyad(\d+)_")[0].astype(float)
    feature_cols = [
        c for c in df.columns
        if (c.startswith("mean_") or c in ["mean_pose_detection_rate", "mean_hand_detection_rate"])
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df.groupby("pair_id", dropna=False)[feature_cols].mean().reset_index()


def outcome_correlations(session: pd.DataFrame) -> pd.DataFrame:
    outcomes = load_outcomes()
    if outcomes is None:
        return pd.DataFrame()
    dyad_video = session_to_dyad_features(session)
    if "pair_id" not in outcomes.columns and "dyad_id" in outcomes.columns:
        outcomes = outcomes.copy()
        outcomes["pair_id"] = outcomes["dyad_id"].astype(str).str.extract(r"dyad(\d+)_")[0].astype(float)
    merged = dyad_video.merge(outcomes, on="pair_id", how="inner")
    if merged.empty:
        return pd.DataFrame()
    feature_cols = [c for c in dyad_video.columns if c != "pair_id"]
    outcome_cols = [c for c in OUTCOME_COLS if c in merged.columns]
    rows = []
    for feature in feature_cols:
        for outcome in outcome_cols:
            sub = merged[[feature, outcome]].dropna()
            if len(sub) < 5 or sub[feature].std() == 0 or sub[outcome].std() == 0:
                r = p = np.nan
            else:
                from scipy import stats
                r, p = stats.pearsonr(sub[feature], sub[outcome])
            rows.append({
                "feature": feature,
                "outcome": outcome,
                "n": len(sub),
                "r": r,
                "p": p,
            })
    return pd.DataFrame(rows)


def write_figures(out_dir: Path, session: pd.DataFrame, speaker: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    heat = session.pivot_table(
        index="dyad_id",
        columns="session_id",
        values="mean_hand_detection_rate",
        aggfunc="mean",
    )
    plt.figure(figsize=(7, max(4, 0.22 * len(heat))))
    sns.heatmap(heat, annot=False, cmap="viridis", vmin=0, vmax=1)
    plt.title("Mean Hand Detection Rate By Session")
    plt.tight_layout()
    plt.savefig(fig_dir / "qc_hand_detection_heatmap.png", dpi=150)
    plt.close()

    if "mean_gesture_frame_rate" in session.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=session, x="session_id", y="mean_gesture_frame_rate")
        sns.stripplot(data=session, x="session_id", y="mean_gesture_frame_rate", color="black", alpha=0.5)
        plt.title("Gesture Activity By Condition")
        plt.tight_layout()
        plt.savefig(fig_dir / "gesture_activity_by_condition.png", dpi=150)
        plt.close()

    if "mean_hand_detection_rate" in speaker.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=speaker, x="participant_id", y="mean_hand_detection_rate")
        sns.stripplot(data=speaker, x="participant_id", y="mean_hand_detection_rate", color="black", alpha=0.5)
        plt.title("Hand Detection Rate By Participant Role")
        plt.tight_layout()
        plt.savefig(fig_dir / "hand_detection_by_role.png", dpi=150)
        plt.close()


def write_markdown_report(
    path: Path,
    run_dir: Path,
    session_qc: pd.DataFrame,
    speaker_qc: pd.DataFrame,
    descriptives: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    lines = [
        "# Video Behavior QC Summary",
        "",
        f"Source run: `{run_dir}`",
        "",
        "## Session QC",
        "",
        session_qc[["dyad_id", "session_id", "qc_status", "mean_pose_detection_rate", "mean_hand_detection_rate"]]
        .head(15).to_markdown(index=False),
        "",
        "## Speaker QC",
        "",
        speaker_qc[["dyad_id", "session_id", "participant_id", "qc_status", "mean_pose_detection_rate", "mean_hand_detection_rate"]]
        .head(15).to_markdown(index=False),
        "",
        "## Feature Descriptives",
        "",
        descriptives.head(25).to_markdown(index=False),
    ]
    if not correlations.empty:
        lines.extend([
            "",
            "## Exploratory Outcome Correlations",
            "",
            correlations.sort_values("p").head(20).to_markdown(index=False),
        ])
    atomic_write_text(path, "\n".join(lines) + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_OUTPUTS / "video_behavior_qc_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = read_run_tables(run_dir)
    turn = tables["turn"]
    question = tables["question"]
    speaker = tables["speaker"]
    session = tables["session"]

    session_qc = detection_quality_by_session(session)
    speaker_qc = detection_quality_by_speaker(speaker)
    turn_qc_counts = qc_counts(turn, ["dyad_id", "session_id", "video_id"])
    descriptives = feature_descriptives(turn, question, speaker, session)
    by_condition = condition_summary(session)
    correlations = outcome_correlations(session)

    session_qc.to_csv(out_dir / "video_detection_quality_by_session.csv", index=False)
    speaker_qc.to_csv(out_dir / "video_detection_quality_by_speaker.csv", index=False)
    turn_qc_counts.to_csv(out_dir / "video_turn_qc_counts_by_session.csv", index=False)
    descriptives.to_csv(out_dir / "video_feature_descriptives.csv", index=False)
    by_condition.to_csv(out_dir / "video_feature_by_condition.csv", index=False)
    correlations.to_csv(out_dir / "video_feature_outcome_correlations.csv", index=False)
    write_figures(out_dir, session, speaker)
    write_markdown_report(out_dir / "video_behavior_qc_summary.md", run_dir, session_qc, speaker_qc, descriptives, correlations)
    print(f"Wrote video QC/descriptive summary -> {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Completed turn-aligned video run directory. Defaults to latest.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to 05_analysis_outputs/video_behavior_qc_summary.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""
First-pass video behavior analysis for the CVS conversation study.

This script consumes a completed turn-aligned video run and asks four practical
questions before any larger model is attempted:

1. Which video features are reliable enough to interpret?
2. Do piper/cloudy sessions differ in gesture/visibility features?
3. Is there systematic speaker-role asymmetry?
4. Are dyad-level video features descriptively associated with outcomes?

The analyses are exploratory and small-N. Outputs are intended for QC and
feature-selection decisions, not confirmatory claims.

Usage:
  python scripts/models/analyze_video_behavior_features.py
  python scripts/models/analyze_video_behavior_features.py --run-dir data/interim/video_behavior/turn_runs/<run_id>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
ANALYSIS_OUTPUTS = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs"))
TURN_RUNS = PROJECT / "data" / "interim" / "video_behavior" / "turn_runs"

SESSION_FEATURES = {
    "pose_detection": "mean_pose_detection_rate",
    "hand_detection": "mean_hand_detection_rate",
    "gesture_activity": "mean_gesture_frame_rate",
    "wrist_speed": "mean_mean_wrist_speed_px_s",
    "hand_to_face_distance": "mean_mean_hand_to_face_distance_px",
    "hand_to_shoulder_distance": "mean_mean_hand_to_shoulder_distance_px",
}

OUTCOMES = [
    "dyad_partner_eval_mean",
    "dyad_shared_reality_mean",
    "dyad_enjoyment_mean",
    "dyad_solo_mean",
]


def latest_run_dir() -> Path:
    runs = sorted(p for p in TURN_RUNS.glob("turn_aligned_video_*") if (p / "results").exists())
    if not runs:
        raise SystemExit(f"No completed video runs found under {TURN_RUNS}")
    return runs[-1]


def load_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session = pd.read_csv(run_dir / "results" / "per_session_summary.csv")
    speaker = pd.read_csv(run_dir / "results" / "per_speaker_summary.csv")
    question = pd.read_csv(run_dir / "results" / "per_question_summary.csv")
    return session, speaker, question


def add_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pair_id" not in df.columns:
        df["pair_id"] = df["dyad_id"].astype(str).str.extract(r"dyad(\d+)_")[0].astype(float)
    return df


def feature_reliability_screen(session: pd.DataFrame, speaker: pd.DataFrame, question: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_name, col in SESSION_FEATURES.items():
        if col not in session:
            continue
        s = pd.to_numeric(session[col], errors="coerce")
        q = pd.to_numeric(question[col], errors="coerce") if col in question else pd.Series(dtype=float)
        sp = pd.to_numeric(speaker[col], errors="coerce") if col in speaker else pd.Series(dtype=float)
        missing_rate = float(s.isna().mean())
        mean = float(s.mean()) if s.notna().any() else np.nan
        sd = float(s.std()) if s.notna().sum() > 1 else np.nan
        role_sd = speaker_role_sd(speaker, col)

        if feature_name == "pose_detection":
            recommended = "qc_only"
            rationale = "Near-ceiling detection quality; useful for QC, not behavior."
        elif feature_name == "hand_detection":
            recommended = "qc_covariate_or_sensitivity"
            rationale = "Hand visibility is partly behavior and partly tracking quality."
        elif missing_rate > 0.30:
            recommended = "not_primary"
            rationale = "Too much missingness for primary interpretation."
        elif mean_hand_detection_low(session):
            recommended = "exploratory_with_hand_qc"
            rationale = "Use only with hand-detection QC because hand tracking varies by session/speaker."
        else:
            recommended = "candidate_behavior_feature"
            rationale = "Feature has usable coverage at session level."

        rows.append({
            "feature": feature_name,
            "column": col,
            "session_n": int(s.notna().sum()),
            "session_missing_rate": missing_rate,
            "session_mean": mean,
            "session_sd": sd,
            "question_missing_rate": float(q.isna().mean()) if len(q) else np.nan,
            "speaker_missing_rate": float(sp.isna().mean()) if len(sp) else np.nan,
            "speaker_role_sd": role_sd,
            "recommended_use": recommended,
            "rationale": rationale,
        })
    return pd.DataFrame(rows)


def speaker_role_sd(speaker: pd.DataFrame, col: str) -> float:
    if col not in speaker or "participant_id" not in speaker:
        return np.nan
    means = speaker.groupby("participant_id")[col].mean()
    return float(means.std()) if len(means) > 1 else np.nan


def mean_hand_detection_low(session: pd.DataFrame) -> bool:
    if "mean_hand_detection_rate" not in session:
        return True
    return float(session["mean_hand_detection_rate"].mean()) < 0.80


def paired_condition_comparison(session: pd.DataFrame) -> pd.DataFrame:
    df = add_pair_id(session)
    rows = []
    for feature, col in SESSION_FEATURES.items():
        if col not in df:
            continue
        wide = df.pivot_table(index="pair_id", columns="session_id", values=col, aggfunc="mean")
        if not {"piper", "cloudy"}.issubset(wide.columns):
            continue
        paired = wide[["piper", "cloudy"]].dropna()
        diff = paired["piper"] - paired["cloudy"]
        if len(diff) >= 3 and diff.std() > 0:
            t, p = stats.ttest_rel(paired["piper"], paired["cloudy"])
            dz = diff.mean() / diff.std(ddof=1)
            w, wp = stats.wilcoxon(diff) if len(diff) >= 5 and (diff != 0).any() else (np.nan, np.nan)
        else:
            t = p = dz = w = wp = np.nan
        rows.append({
            "feature": feature,
            "column": col,
            "n_pairs": len(paired),
            "piper_mean": paired["piper"].mean(),
            "cloudy_mean": paired["cloudy"].mean(),
            "mean_diff_piper_minus_cloudy": diff.mean(),
            "paired_t": t,
            "paired_p": p,
            "cohens_dz": dz,
            "wilcoxon_stat": w,
            "wilcoxon_p": wp,
        })
    return pd.DataFrame(rows)


def speaker_asymmetry(speaker: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_pair_id(speaker)
    role_rows = []
    balance_rows = []
    for feature, col in SESSION_FEATURES.items():
        if col not in df:
            continue
        role = df.groupby("participant_id")[col].agg(["count", "mean", "std"]).reset_index()
        role.insert(0, "feature", feature)
        role.insert(1, "column", col)
        role_rows.append(role)

        wide = df.pivot_table(index=["pair_id", "session_id"], columns="participant_id", values=col, aggfunc="mean")
        if {"A", "B"}.issubset(wide.columns):
            tmp = wide[["A", "B"]].dropna().reset_index()
            tmp["feature"] = feature
            tmp["column"] = col
            tmp["A_minus_B"] = tmp["A"] - tmp["B"]
            denom = tmp["A"].abs() + tmp["B"].abs()
            tmp["asymmetry_index"] = np.where(denom > 0, (tmp["A"] - tmp["B"]) / denom, np.nan)
            balance_rows.append(tmp)
    role_summary = pd.concat(role_rows, ignore_index=True) if role_rows else pd.DataFrame()
    balance = pd.concat(balance_rows, ignore_index=True) if balance_rows else pd.DataFrame()
    return role_summary, balance


def dyad_video_features(session: pd.DataFrame) -> pd.DataFrame:
    df = add_pair_id(session)
    cols = [col for col in SESSION_FEATURES.values() if col in df]
    out = df.groupby("pair_id", dropna=False)[cols].mean().reset_index()
    out = out.rename(columns={col: f"video_{name}" for name, col in SESSION_FEATURES.items() if col in out})
    return out


def outcome_correlations(session: pd.DataFrame) -> pd.DataFrame:
    outcomes_path = DATA / "dyad_level_dataset.csv"
    if not outcomes_path.exists():
        return pd.DataFrame()
    outcomes = pd.read_csv(outcomes_path)
    video = dyad_video_features(session)
    merged = video.merge(outcomes, on="pair_id", how="inner")
    rows = []
    feature_cols = [c for c in video.columns if c != "pair_id"]
    for feature in feature_cols:
        for outcome in [c for c in OUTCOMES if c in merged]:
            sub = merged[[feature, outcome]].dropna()
            if len(sub) >= 5 and sub[feature].std() > 0 and sub[outcome].std() > 0:
                pearson_r, pearson_p = stats.pearsonr(sub[feature], sub[outcome])
                spearman_r, spearman_p = stats.spearmanr(sub[feature], sub[outcome])
            else:
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
            rows.append({
                "feature": feature,
                "outcome": outcome,
                "n": len(sub),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            })
    return pd.DataFrame(rows)


def write_figures(out_dir: Path, session: pd.DataFrame, asymmetry: pd.DataFrame, correlations: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_cols = [
        ("gesture_activity", "mean_gesture_frame_rate"),
        ("wrist_speed", "mean_mean_wrist_speed_px_s"),
        ("hand_detection", "mean_hand_detection_rate"),
    ]
    for name, col in plot_cols:
        if col not in session:
            continue
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=session, x="session_id", y=col)
        sns.stripplot(data=session, x="session_id", y=col, color="black", alpha=0.55)
        plt.title(f"{name.replace('_', ' ').title()} By Condition")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{name}_by_condition.png", dpi=150)
        plt.close()

    if not asymmetry.empty:
        sub = asymmetry[asymmetry["feature"].isin(["gesture_activity", "wrist_speed", "hand_detection"])]
        if not sub.empty:
            plt.figure(figsize=(7, 4))
            sns.boxplot(data=sub, x="feature", y="asymmetry_index")
            plt.axhline(0, color="black", linewidth=1)
            plt.xticks(rotation=20, ha="right")
            plt.title("A vs B Video Behavior Asymmetry")
            plt.tight_layout()
            plt.savefig(fig_dir / "speaker_asymmetry_indices.png", dpi=150)
            plt.close()

    if not correlations.empty:
        corr = correlations.pivot_table(index="feature", columns="outcome", values="pearson_r")
        plt.figure(figsize=(8, max(4, 0.35 * len(corr))))
        sns.heatmap(corr, annot=True, cmap="vlag", center=0, vmin=-1, vmax=1, fmt=".2f")
        plt.title("Exploratory Video Feature Outcome Correlations")
        plt.tight_layout()
        plt.savefig(fig_dir / "video_outcome_correlation_heatmap.png", dpi=150)
        plt.close()


def write_report(out_dir: Path, run_dir: Path, reliability: pd.DataFrame, condition: pd.DataFrame, role: pd.DataFrame, correlations: pd.DataFrame) -> None:
    lines = [
        "# Video Behavior Feature Analysis",
        "",
        f"Source run: `{run_dir}`",
        "",
        "These analyses are exploratory and intended to decide which video features are usable.",
        "",
        "## Feature Reliability Screen",
        "",
        reliability.to_markdown(index=False),
        "",
        "## Paired Condition Comparison",
        "",
        condition.to_markdown(index=False),
        "",
        "## Speaker Role Summary",
        "",
        role.to_markdown(index=False) if not role.empty else "No role summary available.",
        "",
        "## Exploratory Outcome Correlations",
        "",
        correlations.sort_values("pearson_p").head(30).to_markdown(index=False) if not correlations.empty else "No outcome correlations available.",
        "",
        "## Recommended Next Step",
        "",
        "- Treat pose detection primarily as QC.",
        "- Use hand detection as a QC/sensitivity covariate.",
        "- Prioritize session/speaker/question-level gesture activity over raw turn-level wrist speed.",
        "- Keep outcome correlations exploratory because N is small.",
    ]
    path = out_dir / "video_behavior_feature_analysis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_OUTPUTS / "video_behavior_feature_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    session, speaker, question = load_tables(run_dir)
    reliability = feature_reliability_screen(session, speaker, question)
    condition = paired_condition_comparison(session)
    role_summary, asymmetry = speaker_asymmetry(speaker)
    correlations = outcome_correlations(session)

    reliability.to_csv(out_dir / "video_feature_reliability_screen.csv", index=False)
    condition.to_csv(out_dir / "video_condition_comparison.csv", index=False)
    role_summary.to_csv(out_dir / "video_speaker_role_summary.csv", index=False)
    asymmetry.to_csv(out_dir / "video_speaker_asymmetry_by_session.csv", index=False)
    dyad_video_features(session).to_csv(out_dir / "dyad_video_features.csv", index=False)
    correlations.to_csv(out_dir / "video_outcome_correlations_exploratory.csv", index=False)
    write_figures(out_dir, session, asymmetry, correlations)
    write_report(out_dir, run_dir, reliability, condition, role_summary, correlations)
    print(f"Wrote video behavior feature analysis -> {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Completed turn-aligned video run directory. Defaults to latest.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to 05_analysis_outputs/video_behavior_feature_analysis.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

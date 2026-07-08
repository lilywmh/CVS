#!/usr/bin/env python3
"""
Create ROI/role QC outputs for a completed turn-aligned video run.

Lab standard (hard-coded for every video): participant A (door side) occupies
the RIGHT half of the frame and participant B (window side) the LEFT half.
This script makes the ROI/role assignment auditable by exporting sampled frame
overlays and summary tables. When auditing an old run, it reads that run's
config_used.yaml, so overlays reflect what the run actually used (runs before
2026-07-07 used the reversed, incorrect mapping). It does not modify raw
videos or the source run outputs.

Usage:
  python scripts/video_behavior/check_roi_role_mapping.py
  python scripts/video_behavior/check_roi_role_mapping.py --run-dir data/interim/video_behavior/turn_runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
TURN_RUNS = PROJECT / "data" / "interim" / "video_behavior" / "turn_runs"
ANALYSIS_OUTPUTS = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs"))

# Hard-coded lab standard: A = door side = video RIGHT, B = window side = LEFT.
DEFAULT_ROIS = {
    "A": [0.5, 0.0, 1.0, 1.0],
    "B": [0.0, 0.0, 0.5, 1.0],
}


def latest_run_dir() -> Path:
    runs = sorted(p for p in TURN_RUNS.glob("turn_aligned_video_*") if (p / "results" / "per_turn_video_features.csv").exists())
    if not runs:
        raise SystemExit(f"No completed video runs found under {TURN_RUNS}")
    return runs[-1]


def load_rois(run_dir: Path) -> dict[str, list[float]]:
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        return DEFAULT_ROIS
    try:
        import yaml
    except ImportError:
        return DEFAULT_ROIS
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    rois = {}
    for participant in config.get("participants", []):
        participant_id = str(participant.get("participant_id", "")).strip()
        roi = participant.get("roi")
        if participant_id and isinstance(roi, list) and len(roi) == 4:
            rois[participant_id] = [float(x) for x in roi]
    return rois or DEFAULT_ROIS


def load_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "turn": run_dir / "results" / "per_turn_video_features.csv",
        "speaker": run_dir / "results" / "per_speaker_summary.csv",
        "video": run_dir / "metadata" / "video_metadata_checked.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing expected run output(s):\n" + "\n".join(missing))
    return (
        pd.read_csv(required["turn"]),
        pd.read_csv(required["speaker"]),
        pd.read_csv(required["video"]),
    )


def roi_side(roi: list[float]) -> str:
    center_x = (float(roi[0]) + float(roi[2])) / 2.0
    if center_x < 0.45:
        return "left"
    if center_x > 0.55:
        return "right"
    return "center"


def roi_pixels(roi: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        int(round(roi[0] * width)),
        int(round(roi[1] * height)),
        int(round(roi[2] * width)),
        int(round(roi[3] * height)),
    )


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def valid_turns(turn: pd.DataFrame) -> pd.DataFrame:
    df = turn.copy()
    for col in ["turn_start_sec", "turn_end_sec", "turn_duration_sec", "hand_detection_rate", "pose_detection_rate"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[
        (df["turn_end_sec"] > df["turn_start_sec"])
        & (df["turn_duration_sec"] >= 0.50)
        & df["video_id"].notna()
        & df["participant_id"].notna()
    ].copy()


def choose_review_samples(turn: pd.DataFrame, max_per_participant_video: int) -> pd.DataFrame:
    df = valid_turns(turn)
    samples = []
    group_cols = ["video_id", "participant_id"]
    for (video_id, participant_id), group in df.groupby(group_cols, dropna=False):
        group = group.sort_values(["question_id", "turn_id"])
        high = group[group["hand_detection_rate"] >= 0.80]
        low = group[group["hand_detection_rate"] <= 0.20]
        mid = group[(group["hand_detection_rate"] > 0.20) & (group["hand_detection_rate"] < 0.80)]

        choices = []
        if not high.empty:
            choices.append(("high_hand_detection", high.iloc[len(high) // 2]))
        if not low.empty:
            choices.append(("low_hand_detection", low.iloc[len(low) // 2]))
        if not mid.empty and len(choices) < max_per_participant_video:
            choices.append(("mid_hand_detection", mid.iloc[len(mid) // 2]))
        if not choices and not group.empty:
            choices.append(("available_turn", group.iloc[len(group) // 2]))

        for reason, row in choices[:max_per_participant_video]:
            record = row.to_dict()
            record["sample_reason"] = reason
            samples.append(record)
    return pd.DataFrame(samples)


def role_detection_summary(speaker: pd.DataFrame, rois: dict[str, list[float]]) -> pd.DataFrame:
    df = speaker.copy()
    for participant_id, roi in rois.items():
        mask = df["participant_id"].astype(str) == str(participant_id)
        df.loc[mask, "expected_roi"] = json.dumps(roi)
        df.loc[mask, "expected_side"] = roi_side(roi)
    keep = [
        "dyad_id", "session_id", "video_id", "speaker_label", "participant_id",
        "expected_side", "number_of_turns", "total_speaking_time_sec", "qc_status",
        "mean_pose_detection_rate", "mean_hand_detection_rate", "mean_missingness_rate",
        "mean_gesture_frame_rate", "mean_mean_wrist_speed_px_s",
    ]
    return df[[c for c in keep if c in df]].sort_values(["video_id", "participant_id"])


def asymmetry_flags(speaker_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = {
        "hand_detection": "mean_hand_detection_rate",
        "gesture_activity": "mean_gesture_frame_rate",
        "wrist_speed": "mean_mean_wrist_speed_px_s",
    }
    for (dyad_id, session_id, video_id), group in speaker_summary.groupby(["dyad_id", "session_id", "video_id"], dropna=False):
        row = {"dyad_id": dyad_id, "session_id": session_id, "video_id": video_id}
        for feature, col in features.items():
            a_values = pd.to_numeric(group.loc[group["participant_id"].astype(str) == "A", col], errors="coerce") if col in group else pd.Series(dtype=float)
            b_values = pd.to_numeric(group.loc[group["participant_id"].astype(str) == "B", col], errors="coerce") if col in group else pd.Series(dtype=float)
            a = float(a_values.mean()) if a_values.notna().any() else np.nan
            b = float(b_values.mean()) if b_values.notna().any() else np.nan
            row[f"A_{feature}"] = a
            row[f"B_{feature}"] = b
            row[f"A_minus_B_{feature}"] = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
        hand_gap = abs(safe_float(row.get("A_minus_B_hand_detection")))
        gesture_gap = abs(safe_float(row.get("A_minus_B_gesture_activity")))
        row["manual_review_priority"] = (
            "HIGH" if hand_gap >= 0.30 or gesture_gap >= 0.20
            else "MEDIUM" if hand_gap >= 0.15 or gesture_gap >= 0.10
            else "LOW"
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["manual_review_priority", "video_id"], ascending=[True, True])


def draw_text_box(cv2: Any, image: np.ndarray, lines: list[str], x: int, y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    line_h = 24
    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    box_w = max(widths, default=0) + 18
    box_h = line_h * len(lines) + 14
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    for idx, line in enumerate(lines):
        cv2.putText(image, line, (x + 9, y + 24 + idx * line_h), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def write_overlay_image(
    cv2: Any,
    frame: np.ndarray,
    row: pd.Series,
    rois: dict[str, list[float]],
    out_path: Path,
    max_width: int,
) -> None:
    height, width = frame.shape[:2]
    participant_id = str(row.get("participant_id", ""))
    colors = {"A": (30, 180, 255), "B": (120, 220, 80)}
    for pid, roi in rois.items():
        x0, y0, x1, y1 = roi_pixels(roi, width, height)
        color = colors.get(str(pid), (200, 200, 200))
        thickness = 6 if str(pid) == participant_id else 3
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
        cv2.putText(frame, f"{pid} expected {roi_side(roi)}", (x0 + 14, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

    hand = safe_float(row.get("hand_detection_rate"))
    pose = safe_float(row.get("pose_detection_rate"))
    lines = [
        f"{row.get('video_id', '')} | role {participant_id} | {row.get('speaker_label', '')}",
        f"q{row.get('question_id', '')} turn {row.get('turn_id', '')} | {safe_float(row.get('turn_start_sec')):.2f}-{safe_float(row.get('turn_end_sec')):.2f}s",
        f"pose={pose:.2f} hand={hand:.2f} | {row.get('sample_reason', '')}",
    ]
    draw_text_box(cv2, frame, lines, 18, max(18, height - 104))

    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frame):
        raise RuntimeError(f"Could not write overlay image: {out_path}")


def extract_review_images(
    samples: pd.DataFrame,
    video_meta: pd.DataFrame,
    rois: dict[str, list[float]],
    out_dir: Path,
    max_width: int,
) -> pd.DataFrame:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required for ROI/role image QC. Install opencv-python in the active environment.") from exc

    meta = video_meta.set_index("video_id")
    rows = []
    for video_id, group in samples.groupby("video_id", dropna=False):
        if video_id not in meta.index:
            continue
        video_path = PROJECT / str(meta.loc[video_id, "video_path"])
        if not video_path.exists():
            video_path = Path(str(meta.loc[video_id, "video_path"]))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            for _, row in group.iterrows():
                record = row.to_dict()
                record["review_image"] = ""
                record["review_status"] = "video_unreadable"
                rows.append(record)
            continue
        fps = safe_float(meta.loc[video_id, "fps"], safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0))
        for _, row in group.iterrows():
            mid_sec = (safe_float(row.get("turn_start_sec")) + safe_float(row.get("turn_end_sec"))) / 2.0
            frame_index = max(0, int(round(mid_sec * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            record = row.to_dict()
            if not ok:
                record["review_image"] = ""
                record["review_status"] = "frame_unreadable"
                rows.append(record)
                continue
            file_name = (
                f"{video_id}__{row.get('participant_id')}__q{int(safe_float(row.get('question_id'), 0))}"
                f"_t{int(safe_float(row.get('turn_id'), 0))}__{row.get('sample_reason')}.jpg"
            )
            rel_path = Path("review_frames") / file_name
            write_overlay_image(cv2, frame, row, rois, out_dir / rel_path, max_width)
            record["review_image"] = str(rel_path)
            record["review_status"] = "written"
            rows.append(record)
        cap.release()
    return pd.DataFrame(rows)


def write_report(
    out_path: Path,
    run_dir: Path,
    rois: dict[str, list[float]],
    summary: pd.DataFrame,
    flags: pd.DataFrame,
    samples: pd.DataFrame,
) -> None:
    high = flags[flags["manual_review_priority"] == "HIGH"] if "manual_review_priority" in flags else pd.DataFrame()
    role_means = (
        summary.groupby("participant_id", dropna=False)[["mean_pose_detection_rate", "mean_hand_detection_rate", "mean_gesture_frame_rate"]]
        .mean(numeric_only=True)
        .reset_index()
        if not summary.empty else pd.DataFrame()
    )
    lines = [
        "# ROI/Role Mapping QC",
        "",
        f"Source run: `{run_dir}`",
        "",
        "## Assumed ROIs",
        "",
    ]
    for participant_id, roi in rois.items():
        lines.append(f"- `{participant_id}`: `{roi}` ({roi_side(roi)})")
    lines.extend([
        "",
        "## What To Check",
        "",
        "Open the sampled images in `review_frames/` and confirm that the highlighted role matches the expected person on screen.",
        "If A/B is reversed for a video, update the speaker log or participant ROI config before interpreting hand and gesture features.",
        "",
        "## Role Means",
        "",
        markdown_table(role_means) if not role_means.empty else "No role summary available.",
        "",
        "## High-Priority Sessions",
        "",
        markdown_table(high.head(20)) if not high.empty else "No high-priority asymmetry flags.",
        "",
        "## Review Samples",
        "",
        markdown_table(
            samples[["video_id", "participant_id", "speaker_label", "question_id", "turn_id", "sample_reason", "review_image", "review_status"]]
            .head(40)
        ) if not samples.empty else "No review samples were written.",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            display[col] = display[col].fillna("").astype(str)
    headers = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        cells = [str(row[col]).replace("\n", " ") for col in display.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="Completed turn-aligned video run directory.")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory. Default: 05_analysis_outputs/video_roi_role_qc/<source_run_id>/ "
             "(run-scoped so frames from different runs are never mixed).")
    parser.add_argument("--max-per-participant-video", type=int, default=2, help="Sampled frames per participant per video.")
    parser.add_argument("--max-width", type=int, default=1280, help="Maximum saved overlay width in pixels.")
    parser.add_argument("--no-images", action="store_true", help="Only write tables and report, without frame overlays.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_dir = args.run_dir or latest_run_dir()
    # Run-scoped default output dir: frames from different source runs (which
    # may use different ROI mappings) must never be mixed in one folder.
    out_dir = args.out_dir or (ANALYSIS_OUTPUTS / "video_roi_role_qc" / run_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    turn, speaker, video_meta = load_tables(run_dir)
    rois = load_rois(run_dir)
    summary = role_detection_summary(speaker, rois)
    flags = asymmetry_flags(summary)
    samples = choose_review_samples(turn, max(1, args.max_per_participant_video))
    if args.no_images:
        review = samples.copy()
        review["review_image"] = ""
        review["review_status"] = "not_requested"
    else:
        review = extract_review_images(samples, video_meta, rois, out_dir, args.max_width)

    summary.to_csv(out_dir / "roi_role_detection_summary.csv", index=False)
    flags.to_csv(out_dir / "roi_role_asymmetry_flags.csv", index=False)
    review.to_csv(out_dir / "roi_role_review_samples.csv", index=False)
    write_report(out_dir / "roi_role_qc_report.md", run_dir, rois, summary, flags, review)
    print(f"Wrote ROI/role QC outputs to {out_dir}")
    print(f"Review report: {out_dir / 'roi_role_qc_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

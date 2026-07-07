#!/usr/bin/env python3
"""
Render speaker-to-visual-participant mapping overlays for QC.

This renderer is paired with scripts/video_behavior/check_speaker_visual_mapping.py.
It reads the check script outputs, especially:
  intermediate/per_turn_mouth_mapping_qc.csv
  intermediate/per_turn_participant_mouth_metrics.csv
and draws A/B participant ROIs plus turn-level speaker/visual mapping decisions.

Example:
  python scripts/video_behavior/render_speaker_visual_mapping_qc.py \
    --mapping-run 05_analysis_outputs/speaker_visual_mapping_qc/speaker_visual_mapping_smoke \
    --source-run data/interim/video_behavior/turn_runs/turn_aligned_video_20260703_205901 \
    --run-id speaker_visual_mapping_frame_qc_sample \
    --limit-videos 1 \
    --max-turns-per-video 20 \
    --every-frame \
    --write-video \
    --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.video_behavior.render_arm_skeleton_qc import (  # noqa: E402
    draw_label,
    load_source_config,
    resolve_video_path,
    sanitize_token,
)
from scripts.video_behavior.run_turn_aligned_video_analysis import (  # noqa: E402
    participant_roi_lookup,
    roi_to_pixels,
    sample_frame_indices,
)
from scripts.video_behavior.run_video_analysis import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    file_record,
    get_software_versions,
)


OUTPUT_ROOT = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs")) / "video_speaker_visual_mapping_qc"
CHECK_OUTPUT_ROOT = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs")) / "speaker_visual_mapping_qc"

FRAME_COLUMNS = [
    "video_id",
    "dyad_id",
    "session_id",
    "condition",
    "question_id",
    "turn_id",
    "turn_start_sec",
    "turn_end_sec",
    "timestamp_sec",
    "frame_index",
    "speaker_label",
    "expected_participant_id",
    "predicted_visual_participant_id",
    "predicted_visual_side",
    "prediction_status",
    "prediction_confidence",
    "expected_matches_visual",
    "left_mouth_motion_score",
    "right_mouth_motion_score",
    "left_face_detection_rate",
    "right_face_detection_rate",
    "transcript_text",
    "output_frame",
    "output_video",
]

SUMMARY_COLUMNS = [
    "video_id",
    "n_turns",
    "n_frames",
    "n_informative",
    "n_ambiguous",
    "n_insufficient_face_detection",
    "n_match",
    "n_conflict",
    "mean_prediction_confidence",
]

PARTICIPANT_COLORS = {
    "A": (0, 180, 255),
    "B": (110, 220, 90),
}
STATUS_COLORS = {
    "informative": (70, 220, 70),
    "ambiguous": (0, 200, 255),
    "insufficient_face_detection": (40, 40, 255),
    "missing": (190, 190, 190),
}
PID_TO_SIDE = {"A": "left", "B": "right", "a": "left", "b": "right"}
SIDE_TO_PID = {"left": "A", "right": "B"}


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def latest_check_run() -> Path:
    runs = sorted(
        p for p in CHECK_OUTPUT_ROOT.glob("*")
        if p.is_dir() and (p / "intermediate" / "per_turn_mouth_mapping_qc.csv").exists()
    )
    if not runs:
        raise SystemExit(f"No completed speaker visual mapping QC run found under {CHECK_OUTPUT_ROOT}")
    return runs[-1]


def resolve_mapping_run(path: Path | None) -> Path:
    if path is None:
        return latest_check_run()
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Mapping run does not exist: {path}")

    # User may pass the parent output root, e.g. 05_analysis_outputs/speaker_visual_mapping_qc.
    if (path / "intermediate" / "per_turn_mouth_mapping_qc.csv").exists():
        return path
    if (path / "per_turn_mouth_mapping_qc.csv").exists():
        return path

    child_runs = sorted(
        p for p in path.glob("*")
        if p.is_dir() and (p / "intermediate" / "per_turn_mouth_mapping_qc.csv").exists()
    )
    if child_runs:
        chosen = child_runs[-1]
        print(f"Using latest mapping run inside parent directory: {chosen}")
        return chosen

    raise SystemExit(
        "Could not find check_speaker_visual_mapping.py outputs in mapping run.\n"
        f"Given path: {path}\n"
        "Expected one of:\n"
        "  <mapping-run>/intermediate/per_turn_mouth_mapping_qc.csv\n"
        "  <mapping-run>/per_turn_mouth_mapping_qc.csv\n"
        "Or pass the specific run directory under 05_analysis_outputs/speaker_visual_mapping_qc/<run-id>."
    )


def infer_source_run(mapping_run: Path, args: argparse.Namespace) -> Path:
    if args.source_run:
        source_run = Path(args.source_run)
        if not source_run.exists():
            raise SystemExit(f"--source-run does not exist: {source_run}")
        return source_run

    config_used = read_json_if_exists(mapping_run / "config_used.json")
    value = config_used.get("source_run")
    if value and Path(value).exists():
        return Path(value)

    manifest = read_json_if_exists(mapping_run / "run_manifest.json")
    value = manifest.get("source_run")
    if value and Path(value).exists():
        return Path(value)
    for record in manifest.get("inputs", []):
        candidate = Path(str(record.get("path", "")))
        if candidate.exists() and candidate.name in {"video_metadata_checked.csv", "labeled_turns_checked.csv"}:
            maybe = candidate.parent.parent
            if (maybe / "metadata" / "video_metadata_checked.csv").exists():
                return maybe

    raise SystemExit(
        "Could not infer source turn-aligned run from mapping run. Pass it explicitly, for example:\n"
        "  --source-run data/interim/video_behavior/turn_runs/turn_aligned_video_20260703_205901"
    )


def load_video_metadata(source_run: Path) -> pd.DataFrame:
    candidates = [
        source_run / "metadata" / "video_metadata_checked.csv",
        source_run / "metadata" / "video_metadata.csv",
        source_run / "video_metadata_checked.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise SystemExit("Missing source video metadata. Expected one of:\n" + "\n".join(f"  {p}" for p in candidates))


def load_turn_text(source_run: Path) -> pd.DataFrame:
    candidates = [
        source_run / "metadata" / "labeled_turns_checked.csv",
        source_run / "metadata" / "turns_checked.csv",
        source_run / "metadata" / "labeled_turns.csv",
    ]
    for path in candidates:
        if path.exists():
            turns = pd.read_csv(path)
            keep_cols = [c for c in ["video_id", "question_id", "turn_id", "utterance_text", "transcript_text", "text"] if c in turns.columns]
            if {"video_id", "turn_id"}.issubset(keep_cols):
                return turns[keep_cols].drop_duplicates()
    return pd.DataFrame()


def first_existing_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower_to_actual = {str(col).lower(): str(col) for col in df.columns}
    for alias in aliases:
        if alias.lower() in lower_to_actual:
            return lower_to_actual[alias.lower()]
    return None


def normalize_turn_table(raw: pd.DataFrame, mapping_csv: Path) -> pd.DataFrame:
    aliases = {
        "video_id": ["video_id", "video", "recording_id"],
        "dyad_id": ["dyad_id", "dyad"],
        "session_id": ["session_id", "session"],
        "condition": ["condition", "video_condition"],
        "question_id": ["question_id", "question", "question_label"],
        "turn_id": ["turn_id", "window_id", "segment_id", "utterance_id"],
        "speaker_label": ["speaker_label", "audio_speaker", "speaker", "diarized_speaker", "diarization_speaker"],
        "expected_participant_id": ["expected_participant_id", "participant_id", "participant_role", "expected_visual_participant_id"],
        "turn_start_sec": ["turn_start_sec", "start_time_sec", "start_sec", "start_time", "start", "onset_sec", "window_start_sec"],
        "turn_end_sec": ["turn_end_sec", "end_time_sec", "end_sec", "end_time", "end", "offset_sec", "window_end_sec"],
        "predicted_visual_participant_id": ["predicted_visual_participant_id", "predicted_participant_id", "mapped_participant_id", "visual_participant_id"],
        "prediction_status": ["prediction_status", "qc_status", "status", "mapping_status", "decision"],
        "prediction_confidence": ["prediction_confidence", "confidence", "mapping_confidence", "score", "best_score", "mouth_score_margin"],
        "expected_matches_visual": ["expected_matches_visual", "matches_visual", "match", "is_match"],
        "transcript_text": ["transcript_text", "utterance_text", "text", "transcript", "sentence"],
    }
    out = pd.DataFrame()
    missing = []
    for target, names in aliases.items():
        source = first_existing_column(raw, names)
        if source is None:
            if target in ["video_id", "turn_id", "turn_start_sec", "turn_end_sec", "speaker_label"]:
                missing.append((target, names))
            out[target] = ""
        else:
            out[target] = raw[source]
    if missing:
        pretty = "\n".join(f"  {target}: tried {names}" for target, names in missing)
        raise SystemExit(
            f"Mapping turn CSV is missing required columns: {mapping_csv}\n"
            f"Available columns: {list(raw.columns)}\n"
            f"Expected aliases:\n{pretty}"
        )

    out["turn_start_sec"] = pd.to_numeric(out["turn_start_sec"], errors="coerce")
    out["turn_end_sec"] = pd.to_numeric(out["turn_end_sec"], errors="coerce")
    out["prediction_confidence"] = pd.to_numeric(out["prediction_confidence"], errors="coerce")
    bad = out["turn_start_sec"].isna() | out["turn_end_sec"].isna()
    if bad.any():
        raise SystemExit(
            f"Mapping turn CSV contains invalid turn_start_sec / turn_end_sec rows: {mapping_csv}\n"
            f"Bad rows: {out[bad].head(10).to_dict(orient='records')}"
        )

    out["predicted_visual_participant_id"] = out["predicted_visual_participant_id"].astype(str).replace({"nan": "", "None": ""})
    out["predicted_visual_side"] = out["predicted_visual_participant_id"].map(PID_TO_SIDE).fillna("")
    out["prediction_status"] = out["prediction_status"].astype(str).str.lower().replace({"nan": "", "none": ""})
    out.loc[out["prediction_status"].astype(str).str.len().eq(0), "prediction_status"] = "missing"
    return out


def load_participant_metrics(mapping_run: Path) -> pd.DataFrame:
    candidates = [
        mapping_run / "intermediate" / "per_turn_participant_mouth_metrics.csv",
        mapping_run / "per_turn_participant_mouth_metrics.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def merge_participant_metrics(turns: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    out = turns.copy()
    for col in ["left_mouth_motion_score", "right_mouth_motion_score", "left_face_detection_rate", "right_face_detection_rate"]:
        out[col] = ""
    if metrics.empty:
        return out

    required = {"video_id", "question_id", "turn_id", "participant_id", "mouth_motion_score", "face_detection_rate"}
    if not required.issubset(metrics.columns):
        print(f"Participant metrics found but missing expected columns. Available: {list(metrics.columns)}")
        return out

    m = metrics.copy()
    m["side"] = m["participant_id"].astype(str).map(PID_TO_SIDE)
    pivot_score = m.pivot_table(
        index=["video_id", "question_id", "turn_id"],
        columns="side",
        values="mouth_motion_score",
        aggfunc="mean",
    ).reset_index()
    pivot_face = m.pivot_table(
        index=["video_id", "question_id", "turn_id"],
        columns="side",
        values="face_detection_rate",
        aggfunc="mean",
    ).reset_index()
    pivot_score = pivot_score.rename(columns={"left": "left_mouth_motion_score", "right": "right_mouth_motion_score"})
    pivot_face = pivot_face.rename(columns={"left": "left_face_detection_rate", "right": "right_face_detection_rate"})
    keys = ["video_id", "question_id", "turn_id"]
    out = out.merge(pivot_score[[c for c in keys + ["left_mouth_motion_score", "right_mouth_motion_score"] if c in pivot_score.columns]], on=keys, how="left", suffixes=("", "_new"))
    out = out.merge(pivot_face[[c for c in keys + ["left_face_detection_rate", "right_face_detection_rate"] if c in pivot_face.columns]], on=keys, how="left", suffixes=("", "_new"))
    for col in ["left_mouth_motion_score", "right_mouth_motion_score", "left_face_detection_rate", "right_face_detection_rate"]:
        new = f"{col}_new"
        if new in out.columns:
            out[col] = out[new].combine_first(out[col])
            out = out.drop(columns=[new])
    return out


def add_transcript_text(turns: pd.DataFrame, source_run: Path) -> pd.DataFrame:
    if turns["transcript_text"].astype(str).str.len().gt(0).any():
        return turns
    text_df = load_turn_text(source_run)
    if text_df.empty:
        return turns
    text_col = first_existing_column(text_df, ["transcript_text", "utterance_text", "text"])
    if text_col is None:
        return turns
    keys = ["video_id", "turn_id"]
    if "question_id" in text_df.columns and "question_id" in turns.columns:
        keys = ["video_id", "question_id", "turn_id"]
    small = text_df[keys + [text_col]].drop_duplicates().rename(columns={text_col: "_transcript_text"})
    merged = turns.merge(small, on=keys, how="left")
    merged["transcript_text"] = merged["_transcript_text"].fillna(merged["transcript_text"])
    return merged.drop(columns=["_transcript_text"])


def load_mapping_run(mapping_run: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, Path, Path, Path | None]:
    mapping_run = resolve_mapping_run(mapping_run)
    mapping_csv_candidates = [
        mapping_run / "intermediate" / "per_turn_mouth_mapping_qc.csv",
        mapping_run / "per_turn_mouth_mapping_qc.csv",
    ]
    mapping_csv = next((p for p in mapping_csv_candidates if p.exists()), None)
    if mapping_csv is None:
        raise SystemExit(f"Missing per-turn mapping CSV in {mapping_run}")
    raw = pd.read_csv(mapping_csv)
    turns = normalize_turn_table(raw, mapping_csv)
    source_run = infer_source_run(mapping_run, args)
    turns = merge_participant_metrics(turns, load_participant_metrics(mapping_run))
    turns = add_transcript_text(turns, source_run)
    metrics_path = mapping_run / "intermediate" / "per_turn_participant_mouth_metrics.csv"
    return turns, mapping_csv, source_run, metrics_path if metrics_path.exists() else None


def select_turns(mapping: pd.DataFrame, video_meta: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = mapping.copy()
    videos = video_meta.copy()
    if args.video_id:
        requested = set(args.video_id)
        selected = selected[selected["video_id"].astype(str).isin(requested)].copy()
        videos = videos[videos["video_id"].astype(str).isin(requested)].copy()
    if args.question_id:
        requested_questions = {str(value) for value in args.question_id}
        selected = selected[selected["question_id"].astype(str).isin(requested_questions)].copy()
    if args.turn_id:
        requested_turns = set(args.turn_id)
        selected = selected[selected["turn_id"].astype(str).isin(requested_turns)].copy()
    if args.speaker_label:
        requested_speakers = set(args.speaker_label)
        selected = selected[selected["speaker_label"].astype(str).isin(requested_speakers)].copy()
    if args.status:
        requested_statuses = set(args.status)
        selected = selected[selected["prediction_status"].astype(str).isin(requested_statuses)].copy()

    videos = videos[videos["video_id"].astype(str).isin(set(selected["video_id"].astype(str)))].copy()
    if args.limit_videos is not None:
        keep = sorted(selected["video_id"].astype(str).unique())[: args.limit_videos]
        selected = selected[selected["video_id"].astype(str).isin(keep)].copy()
        videos = videos[videos["video_id"].astype(str).isin(keep)].copy()
    if args.max_turns_per_video is not None:
        selected = (
            selected.sort_values(["video_id", "turn_start_sec", "turn_end_sec"])
            .groupby("video_id", dropna=False)
            .head(args.max_turns_per_video)
            .reset_index(drop=True)
        )
    videos = videos[videos["video_id"].astype(str).isin(set(selected["video_id"].astype(str)))].copy()
    return selected.reset_index(drop=True), videos.reset_index(drop=True)


def frame_indices_for_turn(turn: pd.Series, fps: float, args: argparse.Namespace) -> list[int]:
    start = max(0.0, float(turn["turn_start_sec"]) - float(args.context_sec))
    end = max(start, float(turn["turn_end_sec"]) + float(args.context_sec))
    if args.every_frame:
        start_frame = max(0, int(round(start * fps)))
        end_frame = max(start_frame, int(round(end * fps)))
        indices = list(range(start_frame, end_frame + 1))
        if args.max_frames_per_turn and args.max_frames_per_turn > 0:
            indices = indices[: args.max_frames_per_turn]
        return indices
    return sample_frame_indices(
        start,
        end,
        fps,
        float(args.sample_fps),
        int(args.max_frames_per_turn) if args.max_frames_per_turn else None,
    )


def format_value(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    text = str(value)
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def wrap_text(text: str, width: int = 78, max_lines: int = 2) -> list[str]:
    text = " ".join(str(text).split())
    if not text or text.lower() in {"nan", "none"}:
        return []
    return textwrap.wrap(text, width=width)[:max_lines]


def draw_multiline_text(cv2: Any, frame: Any, lines: list[str], x: int, y: int, color: tuple[int, int, int]) -> None:
    for i, line in enumerate(lines):
        draw_label(cv2, frame, line, x, y + i * 24, color)


def draw_rois(cv2: Any, frame: Any, participant_rois: dict[str, Any]) -> dict[str, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for participant_id, roi in participant_rois.items():
        pid = str(participant_id)
        x0, y0, x1, y1 = roi_to_pixels(roi, width, height)
        color = PARTICIPANT_COLORS.get(pid, (220, 220, 220))
        side = PID_TO_SIDE.get(pid, "")
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3)
        draw_label(cv2, frame, f"participant {pid} ({side})", x0 + 8, y0 + 32, color)
        boxes[pid] = (x0, y0, x1, y1)
    return boxes


def draw_mapping_overlay(cv2: Any, frame: Any, turn: pd.Series, participant_rois: dict[str, Any]) -> None:
    boxes = draw_rois(cv2, frame, participant_rois)
    status = str(turn.get("prediction_status", "missing")).lower()
    status_color = STATUS_COLORS.get(status, STATUS_COLORS["missing"])
    predicted_pid = format_value(turn.get("predicted_visual_participant_id", ""))
    predicted_side = PID_TO_SIDE.get(predicted_pid, "")
    expected_pid = format_value(turn.get("expected_participant_id", ""))
    match = format_value(turn.get("expected_matches_visual", ""))

    if predicted_pid in boxes:
        x0, y0, x1, y1 = boxes[predicted_pid]
        cv2.rectangle(frame, (x0 + 5, y0 + 5), (x1 - 5, y1 - 5), status_color, 5)
        draw_label(cv2, frame, f"PREDICTED VISUAL SPEAKER: {predicted_pid} / {predicted_side}", x0 + 8, y0 + 62, status_color)

    start = format_value(turn.get("turn_start_sec"), 2)
    end = format_value(turn.get("turn_end_sec"), 2)
    conf = format_value(turn.get("prediction_confidence"), 3)
    left_score = format_value(turn.get("left_mouth_motion_score"), 4)
    right_score = format_value(turn.get("right_mouth_motion_score"), 4)
    left_face = format_value(turn.get("left_face_detection_rate"), 2)
    right_face = format_value(turn.get("right_face_detection_rate"), 2)

    lines = [
        f"video={turn.get('video_id', '')} q={turn.get('question_id', '')} turn={turn.get('turn_id', '')}",
        f"time={start}-{end}s speaker_label={turn.get('speaker_label', '')} expected={expected_pid}",
        f"predicted_visual={predicted_pid}/{predicted_side} confidence={conf} status={status} match={match}",
    ]
    if left_score or right_score:
        lines.append(f"mouth motion: left/A={left_score} right/B={right_score}")
    if left_face or right_face:
        lines.append(f"face detection: left/A={left_face} right/B={right_face}")
    lines.extend(wrap_text(str(turn.get("transcript_text", "") or "")))
    draw_multiline_text(cv2, frame, lines, 20, 35, status_color)


def frame_row(turn: pd.Series, frame_index: int, timestamp_sec: float, output_frame: str, output_video: str) -> dict[str, Any]:
    return {
        "video_id": turn.get("video_id", ""),
        "dyad_id": turn.get("dyad_id", ""),
        "session_id": turn.get("session_id", ""),
        "condition": turn.get("condition", ""),
        "question_id": turn.get("question_id", ""),
        "turn_id": turn.get("turn_id", ""),
        "turn_start_sec": turn.get("turn_start_sec", ""),
        "turn_end_sec": turn.get("turn_end_sec", ""),
        "timestamp_sec": timestamp_sec,
        "frame_index": int(frame_index),
        "speaker_label": turn.get("speaker_label", ""),
        "expected_participant_id": turn.get("expected_participant_id", ""),
        "predicted_visual_participant_id": turn.get("predicted_visual_participant_id", ""),
        "predicted_visual_side": turn.get("predicted_visual_side", ""),
        "prediction_status": turn.get("prediction_status", ""),
        "prediction_confidence": turn.get("prediction_confidence", ""),
        "expected_matches_visual": turn.get("expected_matches_visual", ""),
        "left_mouth_motion_score": turn.get("left_mouth_motion_score", ""),
        "right_mouth_motion_score": turn.get("right_mouth_motion_score", ""),
        "left_face_detection_rate": turn.get("left_face_detection_rate", ""),
        "right_face_detection_rate": turn.get("right_face_detection_rate", ""),
        "transcript_text": turn.get("transcript_text", ""),
        "output_frame": output_frame,
        "output_video": output_video,
    }


def failure_row(turn: pd.Series, frame_index: int = -1, reason: str = "") -> dict[str, Any]:
    row = frame_row(turn, frame_index, -1, "", "")
    row["prediction_status"] = "missing"
    row["transcript_text"] = f"RENDER FAILURE: {reason}"
    return row


def process_video(
    video_meta: pd.Series,
    turns: pd.DataFrame,
    config: dict[str, Any],
    args: argparse.Namespace,
    frames_dir: Path,
    videos_dir: Path,
) -> list[dict[str, Any]]:
    import cv2

    video_id = str(video_meta["video_id"])
    rows: list[dict[str, Any]] = []
    participant_rois = participant_roi_lookup(config)
    if not participant_rois:
        raise RuntimeError("participant_roi_lookup(config) returned no ROIs. Check config participants[].roi.")

    try:
        video_path = resolve_video_path(video_meta["video_path"])
    except Exception as exc:
        return [failure_row(turn, reason=f"Could not resolve video path: {exc}") for _, turn in turns.iterrows()]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [failure_row(turn, reason=f"Could not open video: {video_path}") for _, turn in turns.iterrows()]

    writer = None
    output_video = ""
    try:
        fps = float(video_meta.get("fps", 0) or cap.get(cv2.CAP_PROP_FPS) or 0)
        if fps <= 0:
            raise RuntimeError(f"Invalid fps for video {video_id}: {fps}")
        for _, turn in turns.sort_values("turn_start_sec").iterrows():
            turn_frame_dir = frames_dir / sanitize_token(video_id) / sanitize_token(str(turn["turn_id"]))
            turn_frame_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in frame_indices_for_turn(turn, fps, args):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_bgr = cap.read()
                if not ok:
                    rows.append(failure_row(turn, frame_index=frame_index, reason="Could not read frame"))
                    continue
                frame_out = frame_bgr.copy()
                height, width = frame_out.shape[:2]
                draw_mapping_overlay(cv2, frame_out, turn, participant_rois)

                frame_name = f"frame{int(frame_index):06d}.jpg"
                frame_path = turn_frame_dir / frame_name
                cv2.imwrite(str(frame_path), frame_out)

                if args.write_video:
                    if writer is None:
                        videos_dir.mkdir(parents=True, exist_ok=True)
                        video_name = f"{sanitize_token(video_id)}_speaker_visual_mapping_qc.mp4"
                        output_video = str(videos_dir / video_name)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_video, fourcc, float(args.output_fps), (width, height))
                    writer.write(frame_out)

                rows.append(frame_row(turn, int(frame_index), frame_index / fps, str(frame_path), output_video))
    finally:
        if writer is not None:
            writer.release()
        cap.release()
    return rows


def summarize(frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = frame_df.copy()
    df["prediction_confidence"] = pd.to_numeric(df["prediction_confidence"], errors="coerce")
    by_turn = df.drop_duplicates(["video_id", "turn_id"])
    rows = []
    for video_id, group in by_turn.groupby("video_id", dropna=False):
        frame_group = df[df["video_id"].astype(str) == str(video_id)]
        statuses = group["prediction_status"].astype(str).str.lower()
        matches = group["expected_matches_visual"].astype(str).str.lower()
        rows.append({
            "video_id": video_id,
            "n_turns": len(group),
            "n_frames": len(frame_group),
            "n_informative": int((statuses == "informative").sum()),
            "n_ambiguous": int((statuses == "ambiguous").sum()),
            "n_insufficient_face_detection": int((statuses == "insufficient_face_detection").sum()),
            "n_match": int(matches.isin(["true", "1", "1.0"]).sum()),
            "n_conflict": int(matches.isin(["false", "0", "0.0"]).sum()),
            "mean_prediction_confidence": group["prediction_confidence"].mean(skipna=True),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def write_manifest(
    run_dir: Path,
    mapping_run: Path,
    mapping_csv: Path,
    metrics_path: Path | None,
    source_run: Path,
    output_paths: list[Path],
    config: dict[str, Any],
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mapping_run": str(mapping_run),
        "mapping_csv": str(mapping_csv),
        "participant_metrics_csv": str(metrics_path) if metrics_path else "",
        "source_run": str(source_run),
        "software_versions": get_software_versions(config),
        "outputs": [file_record(path, "speaker_visual_mapping_render_qc_output") for path in output_paths if path.exists()],
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)


def run(args: argparse.Namespace) -> int:
    mapping_run = resolve_mapping_run(Path(args.mapping_run) if args.mapping_run else None)
    mapping, mapping_csv, source_run, metrics_path = load_mapping_run(mapping_run, args)
    video_meta = load_video_metadata(source_run)
    config = load_source_config(source_run)
    selected_turns, selected_videos = select_turns(mapping, video_meta, args)
    if selected_turns.empty:
        raise SystemExit("No mapping turns selected. Check filters such as --video-id, --turn-id, --speaker-label, or --status.")

    run_id = args.run_id or f"speaker_visual_mapping_render_qc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_root) if args.output_root else OUTPUT_ROOT
    run_dir = output_root / run_id
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                "Output run directory already exists:\n"
                f"  {run_dir}\n"
                "Use a new --run-id or pass --overwrite for generated QC output."
            )
        shutil.rmtree(run_dir)
    frames_dir = run_dir / "frames"
    videos_dir = run_dir / "overlay_videos"
    frames_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    for _, video in selected_videos.iterrows():
        video_id = str(video["video_id"])
        video_turns = selected_turns[selected_turns["video_id"].astype(str) == video_id].copy()
        if video_turns.empty:
            continue
        print(f"Rendering {video_id}: {len(video_turns)} turns")
        rows.extend(process_video(video, video_turns, config, args, frames_dir, videos_dir))

    frame_df = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    summary_df = summarize(frame_df)
    frame_path = run_dir / "speaker_visual_mapping_frame_manifest.csv"
    summary_path = run_dir / "speaker_visual_mapping_render_qc_summary.csv"
    selected_turns_path = run_dir / "selected_mapping_turns.csv"
    atomic_write_csv(frame_path, frame_df)
    atomic_write_csv(summary_path, summary_df)
    atomic_write_csv(selected_turns_path, selected_turns)
    write_manifest(run_dir, mapping_run, mapping_csv, metrics_path, source_run, [frame_path, summary_path, selected_turns_path, videos_dir], config)
    print(f"Speaker visual mapping render QC written to: {run_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render speaker visual mapping QC overlays.")
    parser.add_argument("--mapping-run", help="Completed check_speaker_visual_mapping.py run, or parent speaker_visual_mapping_qc directory. Defaults to latest run.")
    parser.add_argument("--source-run", help="Optional source turn-aligned run if not recorded in the mapping run.")
    parser.add_argument("--run-id", help="Output run id under the render QC output root.")
    parser.add_argument("--output-root", help="Directory for render QC outputs. Defaults to CVS_ANALYSIS_OUTPUTS/video_speaker_visual_mapping_qc.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated QC run directory.")
    parser.add_argument("--video-id", nargs="*", help="Optional video_id filter.")
    parser.add_argument("--question-id", nargs="*", help="Optional question_id filter.")
    parser.add_argument("--turn-id", nargs="*", help="Optional turn_id filter.")
    parser.add_argument("--speaker-label", nargs="*", help="Optional speaker label filter, e.g. A, B, SPEAKER_00.")
    parser.add_argument("--status", nargs="*", choices=["informative", "ambiguous", "insufficient_face_detection", "missing"], help="Optional prediction status filter.")
    parser.add_argument("--limit-videos", type=int, default=1, help="Maximum videos to render; omit for all selected videos.")
    parser.add_argument("--max-turns-per-video", type=int, default=20, help="Maximum turns per video; omit for all selected turns.")
    parser.add_argument("--context-sec", type=float, default=0.25, help="Seconds before/after each turn to include.")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Sampling rate when --every-frame is not set.")
    parser.add_argument("--max-frames-per-turn", type=int, default=30, help="Maximum rendered frames per turn; 0 means no cap.")
    parser.add_argument("--every-frame", action="store_true", help="Render every native frame in selected turns.")
    parser.add_argument("--write-video", action="store_true", help="Also write an MP4 overlay video from rendered frames.")
    parser.add_argument("--output-fps", type=float, default=10.0, help="Overlay-video frame rate.")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_arg_parser().parse_args()))

#!/usr/bin/env python3
"""
QC speaker-label to visual-participant mapping with mouth movement.

This script checks whether the participant labeled as speaker in audio-aligned
turns is also the participant whose mouth moves more in the video. It uses
MediaPipe FaceMesh mouth-open dynamics inside configured participant ROIs. It
does not relabel source data; it writes auditable QC tables.

Usage:
  python scripts/video_behavior/check_speaker_visual_mapping.py \
    --source-run data/interim/video_behavior/turn_runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.video_behavior.extract_video_behavior_features import extract_face_features  # noqa: E402
from scripts.video_behavior.render_arm_skeleton_qc import (  # noqa: E402
    load_source_config,
    resolve_video_path,
)
from scripts.video_behavior.run_question_side_video_analysis import (  # noqa: E402
    import_face_mesh_solution,
    latest_turn_run,
    read_source_run,
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


OUTPUT_ROOT = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs")) / "speaker_visual_mapping_qc"

TURN_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "question_id",
    "turn_id", "speaker_label", "expected_participant_id", "turn_start_sec",
    "turn_end_sec", "turn_duration_sec", "n_sampled_frames",
    "predicted_visual_participant_id", "prediction_status",
    "prediction_confidence", "expected_matches_visual",
]
PARTICIPANT_METRIC_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "question_id",
    "turn_id", "speaker_label", "participant_id", "face_detection_rate",
    "mouth_open_mean", "mouth_open_sd", "mouth_open_delta_mean",
    "mouth_motion_score", "n_face_frames",
]
VIDEO_MAPPING_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "speaker_label",
    "expected_participant_id", "predicted_visual_participant_id",
    "n_informative_turns", "n_matching_turns", "match_rate",
    "visual_prediction_rate", "mapping_status", "qc_notes",
]
CONFLICT_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "question_id",
    "turn_id", "speaker_label", "expected_participant_id",
    "predicted_visual_participant_id", "prediction_confidence",
    "turn_start_sec", "turn_end_sec", "turn_duration_sec",
]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def select_inputs(video_meta: pd.DataFrame, turns: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos = video_meta.copy()
    selected_turns = turns.copy()
    if args.video_id:
        requested = set(args.video_id)
        videos = videos[videos["video_id"].astype(str).isin(requested)].copy()
        selected_turns = selected_turns[selected_turns["video_id"].astype(str).isin(requested)].copy()
    if args.limit_videos is not None:
        keep = videos["video_id"].head(args.limit_videos).astype(str).tolist()
        videos = videos[videos["video_id"].astype(str).isin(keep)].copy()
        selected_turns = selected_turns[selected_turns["video_id"].astype(str).isin(keep)].copy()
    if args.question_id:
        requested_questions = {str(value) for value in args.question_id}
        selected_turns = selected_turns[selected_turns["question_id"].astype(str).isin(requested_questions)].copy()
    selected_turns = selected_turns.copy()
    selected_turns["turn_duration_sec"] = pd.to_numeric(
        selected_turns.get("turn_duration_sec", selected_turns["end_time_sec"] - selected_turns["start_time_sec"]),
        errors="coerce",
    )
    selected_turns = selected_turns[
        (pd.to_numeric(selected_turns["end_time_sec"], errors="coerce") > pd.to_numeric(selected_turns["start_time_sec"], errors="coerce"))
        & (selected_turns["turn_duration_sec"] >= float(args.min_turn_duration_sec))
    ].copy()
    if args.max_turns_per_video is not None:
        selected_turns = (
            selected_turns.sort_values(["video_id", "question_id", "turn_id"])
            .groupby("video_id", dropna=False)
            .head(args.max_turns_per_video)
            .reset_index(drop=True)
        )
    videos = videos[videos["video_id"].astype(str).isin(set(selected_turns["video_id"].astype(str)))].copy()
    return videos.reset_index(drop=True), selected_turns.reset_index(drop=True)


def mouth_metrics(values: list[float], n_expected: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "face_detection_rate": 0.0 if n_expected else np.nan,
            "mouth_open_mean": np.nan,
            "mouth_open_sd": np.nan,
            "mouth_open_delta_mean": np.nan,
            "mouth_motion_score": np.nan,
            "n_face_frames": 0,
        }
    delta = np.abs(np.diff(finite))
    delta_mean = float(delta.mean()) if delta.size else 0.0
    sd = float(finite.std(ddof=0)) if finite.size else np.nan
    score = sd + delta_mean
    return {
        "face_detection_rate": float(finite.size / n_expected) if n_expected else np.nan,
        "mouth_open_mean": float(finite.mean()),
        "mouth_open_sd": sd,
        "mouth_open_delta_mean": delta_mean,
        "mouth_motion_score": float(score),
        "n_face_frames": int(finite.size),
    }


def process_video(
    video_meta: pd.Series,
    turns: pd.DataFrame,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import cv2

    face_solution = import_face_mesh_solution()
    if face_solution is None:
        raise RuntimeError("MediaPipe FaceMesh solution is unavailable in this environment.")

    video_path = resolve_video_path(video_meta["video_path"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    face_model = face_solution.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=float(args.face_min_detection_confidence),
    )
    participant_rois = participant_roi_lookup(config)
    turn_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []

    try:
        fps = float(video_meta["fps"])
        for _, turn in turns.sort_values(["question_id", "turn_id"]).iterrows():
            frame_indices = sample_frame_indices(
                float(turn["start_time_sec"]),
                float(turn["end_time_sec"]),
                fps,
                float(args.sample_fps),
                int(args.max_frames_per_turn) if args.max_frames_per_turn else None,
            )
            if not frame_indices:
                continue
            mouth_by_participant = {participant_id: [] for participant_id in participant_rois}
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_bgr = cap.read()
                if not ok:
                    for participant_id in participant_rois:
                        mouth_by_participant[participant_id].append(np.nan)
                    continue
                height, width = frame_bgr.shape[:2]
                for participant_id, roi in participant_rois.items():
                    x0, y0, x1, y1 = roi_to_pixels(roi, width, height)
                    crop = frame_bgr[y0:y1, x0:x1]
                    if crop.size == 0:
                        mouth_by_participant[participant_id].append(np.nan)
                        continue
                    crop_h, crop_w = crop.shape[:2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    result = face_model.process(crop_rgb)
                    faces = result.multi_face_landmarks or []
                    if not faces:
                        mouth_by_participant[participant_id].append(np.nan)
                        continue
                    features = extract_face_features(faces[0], crop_w, crop_h)
                    mouth_by_participant[participant_id].append(features.get("mouth_open", np.nan))

            turn_metric_rows = []
            for participant_id, values in mouth_by_participant.items():
                metrics = mouth_metrics(values, len(frame_indices))
                row = {
                    "dyad_id": turn["dyad_id"],
                    "session_id": turn["session_id"],
                    "video_id": turn["video_id"],
                    "condition": video_meta.get("condition", ""),
                    "question_id": turn["question_id"],
                    "turn_id": turn["turn_id"],
                    "speaker_label": turn["speaker_label"],
                    "participant_id": participant_id,
                    **metrics,
                }
                participant_rows.append(row)
                turn_metric_rows.append(row)

            informative = [
                row for row in turn_metric_rows
                if np.isfinite(row["mouth_motion_score"])
                and row["face_detection_rate"] >= float(args.min_face_detection_rate)
            ]
            if len(informative) >= 2:
                ordered = sorted(informative, key=lambda row: row["mouth_motion_score"], reverse=True)
                best = ordered[0]
                second = ordered[1]
                score_gap = best["mouth_motion_score"] - second["mouth_motion_score"]
                denom = best["mouth_motion_score"] + second["mouth_motion_score"] + 1e-9
                confidence = float(score_gap / denom)
                if score_gap >= float(args.min_score_gap) and confidence >= float(args.min_confidence):
                    predicted = best["participant_id"]
                    status = "informative"
                else:
                    predicted = ""
                    status = "ambiguous"
            else:
                predicted = ""
                confidence = np.nan
                status = "insufficient_face_detection"

            expected = str(turn.get("participant_id", "") or turn.get("participant_role", "")).strip()
            match = bool(predicted and expected and predicted == expected)
            turn_rows.append({
                "dyad_id": turn["dyad_id"],
                "session_id": turn["session_id"],
                "video_id": turn["video_id"],
                "condition": video_meta.get("condition", ""),
                "question_id": turn["question_id"],
                "turn_id": turn["turn_id"],
                "speaker_label": turn["speaker_label"],
                "expected_participant_id": expected,
                "turn_start_sec": float(turn["start_time_sec"]),
                "turn_end_sec": float(turn["end_time_sec"]),
                "turn_duration_sec": float(turn["end_time_sec"]) - float(turn["start_time_sec"]),
                "n_sampled_frames": int(len(frame_indices)),
                "predicted_visual_participant_id": predicted,
                "prediction_status": status,
                "prediction_confidence": confidence,
                "expected_matches_visual": match if predicted else np.nan,
            })
    finally:
        face_model.close()
        cap.release()
    return turn_rows, participant_rows


def aggregate_video_mapping(turn_df: pd.DataFrame) -> pd.DataFrame:
    if turn_df.empty:
        return pd.DataFrame(columns=VIDEO_MAPPING_COLUMNS)
    informative = turn_df[turn_df["prediction_status"] == "informative"].copy()
    rows = []
    group_cols = ["dyad_id", "session_id", "video_id", "condition", "speaker_label"]
    for keys, group in informative.groupby(group_cols, dropna=False):
        predicted_counts = group["predicted_visual_participant_id"].value_counts()
        predicted = str(predicted_counts.idxmax()) if not predicted_counts.empty else ""
        expected_values = group["expected_participant_id"].dropna().astype(str)
        expected = str(expected_values.mode().iloc[0]) if not expected_values.empty else ""
        n = int(len(group))
        n_matching = int((group["expected_matches_visual"] == True).sum())  # noqa: E712
        visual_rate = float(predicted_counts.max() / n) if n and not predicted_counts.empty else np.nan
        match_rate = float(n_matching / n) if n else np.nan
        notes = []
        if n < 3:
            notes.append("Few informative turns.")
        if expected and predicted and expected != predicted:
            notes.append("Visual mouth-motion mapping conflicts with expected speaker log role.")
        if np.isfinite(visual_rate) and visual_rate < 0.70:
            notes.append("Low visual prediction agreement.")
        status = "PASS" if not notes else "WARNING"
        rows.append({
            "dyad_id": keys[0],
            "session_id": keys[1],
            "video_id": keys[2],
            "condition": keys[3],
            "speaker_label": keys[4],
            "expected_participant_id": expected,
            "predicted_visual_participant_id": predicted,
            "n_informative_turns": n,
            "n_matching_turns": n_matching,
            "match_rate": match_rate,
            "visual_prediction_rate": visual_rate,
            "mapping_status": status,
            "qc_notes": " ".join(notes),
        })
    return pd.DataFrame(rows, columns=VIDEO_MAPPING_COLUMNS)


def conflict_rows(turn_df: pd.DataFrame) -> pd.DataFrame:
    if turn_df.empty:
        return pd.DataFrame(columns=CONFLICT_COLUMNS)
    conflicts = turn_df[
        (turn_df["prediction_status"] == "informative")
        & (turn_df["expected_matches_visual"] == False)  # noqa: E712
    ].copy()
    return conflicts[[col for col in CONFLICT_COLUMNS if col in conflicts.columns]]


def write_manifest(run_dir: Path, source_run: Path, output_paths: list[Path], config: dict[str, Any]) -> None:
    manifest = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_run": str(source_run),
        "software_versions": get_software_versions({"openface": {"enabled": False}, "mediapipe": {"enabled": True}}),
        "config": config,
        "inputs": [
            file_record(source_run / "metadata" / "video_metadata_checked.csv", "source_run_metadata"),
            file_record(source_run / "metadata" / "labeled_turns_checked.csv", "source_run_turns"),
        ],
        "outputs": [file_record(path, "pipeline_output") for path in output_paths if path.exists()],
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)


def run(args: argparse.Namespace) -> int:
    source_run = args.source_run or latest_turn_run()
    config = load_source_config(source_run)
    video_meta, turns = read_source_run(source_run)
    videos, selected_turns = select_inputs(video_meta, turns, args)

    run_id = args.run_id or f"speaker_visual_mapping_qc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / run_id
    if run_dir.exists() and not args.overwrite:
        raise SystemExit(
            "Output run directory already exists:\n"
            f"  {run_dir}\n"
            "Use a new --run-id or pass --overwrite for generated outputs."
        )
    for subdir in ["metadata", "intermediate", "results", "qc"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    config_used = json_ready(vars(args).copy())
    config_used["source_run"] = str(source_run)
    config_used["note"] = "FaceMesh mouth-movement QC for speaker-label to visual-participant mapping."
    atomic_write_json(run_dir / "config_used.json", config_used)
    atomic_write_csv(run_dir / "metadata" / "selected_turns.csv", selected_turns)
    if args.dry_run:
        outputs = [run_dir / "metadata" / "selected_turns.csv"]
        write_manifest(run_dir, source_run, outputs, config_used)
        print(f"Dry run complete: {run_dir}")
        return 0

    turn_parts = []
    participant_parts = []
    failed_rows = []
    for _, video in videos.iterrows():
        video_turns = selected_turns[selected_turns["video_id"].astype(str) == str(video["video_id"])].copy()
        if video_turns.empty:
            continue
        print(f"Processing {video['video_id']}: {len(video_turns)} speaker turns")
        try:
            turn_rows, participant_rows = process_video(video, video_turns, config, args)
            turn_parts.extend(turn_rows)
            participant_parts.extend(participant_rows)
        except Exception as exc:
            failed_rows.append({
                "video_id": video.get("video_id", ""),
                "video_path": video.get("video_path", ""),
                "stage": "mouth_movement_qc",
                "error": str(exc),
                "suggested_fix": "Check video readability and FaceMesh availability in the active environment.",
            })
            print(f"ERROR processing {video.get('video_id', '')}: {exc}")
            if args.fail_fast:
                break

    turn_df = pd.DataFrame(turn_parts, columns=TURN_COLUMNS)
    participant_df = pd.DataFrame(participant_parts, columns=PARTICIPANT_METRIC_COLUMNS)
    mapping_df = aggregate_video_mapping(turn_df)
    conflicts_df = conflict_rows(turn_df)
    failed_df = pd.DataFrame(failed_rows, columns=["video_id", "video_path", "stage", "error", "suggested_fix"])
    outputs = [
        run_dir / "intermediate" / "per_turn_mouth_mapping_qc.csv",
        run_dir / "intermediate" / "per_turn_participant_mouth_metrics.csv",
        run_dir / "results" / "speaker_visual_mapping_summary.csv",
        run_dir / "qc" / "speaker_visual_mapping_conflicts.csv",
        run_dir / "qc" / "failed_videos.csv",
    ]
    atomic_write_csv(outputs[0], turn_df)
    atomic_write_csv(outputs[1], participant_df)
    atomic_write_csv(outputs[2], mapping_df)
    atomic_write_csv(outputs[3], conflicts_df)
    atomic_write_csv(outputs[4], failed_df)
    write_manifest(run_dir, source_run, outputs + [run_dir / "metadata" / "selected_turns.csv"], config_used)
    print(f"Speaker visual mapping QC complete: {run_dir}")
    return 1 if failed_rows and not turn_parts else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=None, help="Completed turn-aligned run to check.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--max-frames-per-turn", type=int, default=80)
    parser.add_argument("--min-turn-duration-sec", type=float, default=0.75)
    parser.add_argument("--min-face-detection-rate", type=float, default=0.40)
    parser.add_argument("--min-score-gap", type=float, default=0.003)
    parser.add_argument("--min-confidence", type=float, default=0.10)
    parser.add_argument("--face-min-detection-confidence", type=float, default=0.50)
    parser.add_argument("--video-id", nargs="*", help="Optional video_id filter.")
    parser.add_argument("--question-id", nargs="*", help="Optional question_id filter.")
    parser.add_argument("--limit-videos", type=int, default=None, help="Debug/smoke-test video limit.")
    parser.add_argument("--max-turns-per-video", type=int, default=None, help="Debug/smoke-test turn limit.")
    parser.add_argument("--dry-run", action="store_true", help="Write selected turns only.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_arg_parser().parse_args()))

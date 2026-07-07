#!/usr/bin/env python3
"""
Time-window upper-limb role analysis for CVS discussion videos.

This runner complements turn-level speaker/listener analyses by assigning each
participant in fixed time windows to speaker, listener, silent, or overlap
using audio-aligned turn timestamps. Arm posture/movement features still come
from MediaPipe Pose shoulder-elbow-wrist landmarks.

Usage:
  python scripts/video_behavior/run_time_window_arm_role_analysis.py \
    --source-run data/interim/video_behavior/turn_runs/<run_id>

  python scripts/video_behavior/run_time_window_arm_role_analysis.py \
    --source-run data/interim/video_behavior/turn_runs/<run_id> \
    --run-id time_window_arm_smoke \
    --limit-videos 1 --max-questions-per-video 1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.video_behavior.run_question_side_video_analysis import (  # noqa: E402
    latest_turn_run,
    make_question_windows,
    read_source_run,
)
from scripts.video_behavior.run_turn_aligned_video_analysis import (  # noqa: E402
    DEFAULT_CONFIG,
    attach_speaker_log_roles,
    deep_merge,
    import_video_stack,
    participant_roi_lookup,
    roi_to_pixels,
    sample_frame_indices,
)
from scripts.video_behavior.run_video_analysis import (  # noqa: E402
    POSE_LANDMARK_NAMES,
    atomic_write_csv,
    atomic_write_json,
    file_record,
    get_software_versions,
)
from scripts.video_behavior.upper_limb_features import (  # noqa: E402
    ARM_FRAME_COLUMNS,
    compute_upper_limb_frame_row,
    finite_max,
    finite_mean,
    join_notes,
    summarize_frame_group,
    weighted_mean,
)


OUTPUT_ROOT = Path(
    os.environ.get("CVS_VIDEO_OUTPUTS", PROJECT / "data" / "interim" / "video_behavior" / "time_window_arm_runs")
)

EXTRA_FRAME_COLUMNS = [
    "window_id", "window_start_sec", "window_end_sec", "window_duration_sec",
    "floor_state", "active_speaker_count", "speech_overlap_sec",
    "speech_overlap_ratio", "total_speech_overlap_sec",
]
TIME_WINDOW_FRAME_COLUMNS = ARM_FRAME_COLUMNS + EXTRA_FRAME_COLUMNS
TIME_WINDOW_ROLE_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "question_id",
    "window_id", "window_start_sec", "window_end_sec", "window_duration_sec",
    "participant_id", "conversational_role", "floor_state",
    "active_speaker_count", "speech_overlap_sec", "speech_overlap_ratio",
    "total_speech_overlap_sec", "n_sampled_frames", "pose_detection_rate",
    "arm_detection_rate", "mean_wrist_speed_px_s", "max_wrist_speed_px_s",
    "active_motion_frame_rate", "hold_frame_rate", "near_face_frame_rate",
    "raised_frame_rate", "open_frame_rate", "closed_frame_rate",
    "resting_low_frame_rate", "mean_arm_openness_index",
    "mean_arm_asymmetry_index", "dominant_upper_limb_behavior_state",
    "dominant_upper_limb_posture_state", "behavior_state_entropy",
    "qc_status", "qc_notes",
]
PARTICIPANT_ROLE_SUMMARY_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "participant_id",
    "conversational_role", "n_windows", "total_window_time_sec",
    "total_speech_overlap_sec", "mean_speech_overlap_ratio",
    "mean_pose_detection_rate", "mean_arm_detection_rate",
    "mean_wrist_speed_px_s", "max_wrist_speed_px_s",
    "active_motion_frame_rate", "hold_frame_rate", "near_face_frame_rate",
    "raised_frame_rate", "open_frame_rate", "closed_frame_rate",
    "resting_low_frame_rate", "mean_arm_openness_index",
    "mean_arm_asymmetry_index", "qc_status", "qc_notes",
]


def load_source_config(source_run: Path) -> dict[str, Any]:
    config_path = source_run / "config_used.yaml"
    if not config_path.exists():
        return DEFAULT_CONFIG
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read config_used.yaml.") from exc
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config file is not a mapping: {config_path}")
    return deep_merge(DEFAULT_CONFIG, loaded)


def load_source_tables(source_run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    video_meta, turns = read_source_run(source_run)
    speaker_path = source_run / "metadata" / "speaker_logs_checked.csv"
    if speaker_path.exists():
        speakers = pd.read_csv(speaker_path)
        turns = attach_speaker_log_roles(turns, speakers)
    return video_meta, turns, make_question_windows(turns)


def build_time_windows(
    question_windows: pd.DataFrame,
    window_sec: float,
    stride_sec: float,
) -> pd.DataFrame:
    rows = []
    for _, question in question_windows.iterrows():
        start = float(question["question_start_sec"])
        end = float(question["question_end_sec"])
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        window_idx = 0
        t = start
        while t < end:
            window_end = min(t + window_sec, end)
            if window_end > t:
                rows.append({
                    **question.to_dict(),
                    "window_id": f"q{question['question_id']}_w{window_idx:04d}",
                    "window_start_sec": t,
                    "window_end_sec": window_end,
                    "window_duration_sec": window_end - t,
                })
            window_idx += 1
            t += stride_sec
    return pd.DataFrame(rows)


def select_inputs(
    video_meta: pd.DataFrame,
    question_windows: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos = video_meta.copy()
    questions = question_windows.copy()
    if args.video_id:
        requested = set(args.video_id)
        videos = videos[videos["video_id"].astype(str).isin(requested)].copy()
        questions = questions[questions["video_id"].astype(str).isin(requested)].copy()
    if args.limit_videos is not None:
        keep = videos["video_id"].head(args.limit_videos).astype(str).tolist()
        videos = videos[videos["video_id"].astype(str).isin(keep)].copy()
        questions = questions[questions["video_id"].astype(str).isin(keep)].copy()
    if args.question_id:
        requested_questions = {str(value) for value in args.question_id}
        questions = questions[questions["question_id"].astype(str).isin(requested_questions)].copy()
    if args.max_questions_per_video is not None:
        questions = (
            questions.sort_values(["video_id", "question_id"])
            .groupby("video_id", dropna=False)
            .head(args.max_questions_per_video)
            .reset_index(drop=True)
        )
    videos = videos[videos["video_id"].astype(str).isin(set(questions["video_id"].astype(str)))].copy()
    return videos.reset_index(drop=True), questions.reset_index(drop=True)


def role_rows_for_windows(
    windows: pd.DataFrame,
    turns: pd.DataFrame,
    participants: list[str],
    min_speech_overlap_sec: float,
    min_speech_overlap_ratio: float,
) -> pd.DataFrame:
    turns = turns.copy()
    for col in ["start_time_sec", "end_time_sec"]:
        turns[col] = pd.to_numeric(turns[col], errors="coerce")
    if "participant_id" not in turns and "participant_role" in turns:
        turns["participant_id"] = turns["participant_role"]
    rows = []
    for _, window in windows.iterrows():
        duration = float(window["window_duration_sec"])
        relevant = turns[
            (turns["video_id"].astype(str) == str(window["video_id"]))
            & (turns["question_id"].astype(str) == str(window["question_id"]))
            & (turns["end_time_sec"] > float(window["window_start_sec"]))
            & (turns["start_time_sec"] < float(window["window_end_sec"]))
        ].copy()
        overlap_by_participant = {participant: 0.0 for participant in participants}
        if not relevant.empty:
            starts = np.maximum(relevant["start_time_sec"].to_numpy(dtype=float), float(window["window_start_sec"]))
            ends = np.minimum(relevant["end_time_sec"].to_numpy(dtype=float), float(window["window_end_sec"]))
            relevant["overlap_sec"] = np.maximum(0.0, ends - starts)
            for participant_id, group in relevant.groupby("participant_id", dropna=False):
                participant = str(participant_id)
                overlap_by_participant[participant] = overlap_by_participant.get(participant, 0.0) + float(group["overlap_sec"].sum())

        active = [
            participant for participant, overlap in overlap_by_participant.items()
            if overlap >= min_speech_overlap_sec
            or (duration > 0 and overlap / duration >= min_speech_overlap_ratio)
        ]
        active_speaker_count = len(active)
        floor_state = "silent" if active_speaker_count == 0 else "overlap" if active_speaker_count > 1 else "single_speaker"
        total_overlap = float(sum(overlap_by_participant.values()))
        for participant in participants:
            participant_overlap = float(overlap_by_participant.get(participant, 0.0))
            if floor_state == "silent":
                role = "silent"
            elif floor_state == "overlap" and participant in active:
                role = "overlap"
            elif participant in active:
                role = "speaker"
            else:
                role = "listener"
            rows.append({
                **window.to_dict(),
                "participant_id": participant,
                "conversational_role": role,
                "floor_state": floor_state,
                "active_speaker_count": active_speaker_count,
                "speech_overlap_sec": participant_overlap,
                "speech_overlap_ratio": participant_overlap / duration if duration > 0 else np.nan,
                "total_speech_overlap_sec": total_overlap,
            })
    return pd.DataFrame(rows)


def pose_points_from_result(
    pose_result: Any,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
) -> dict[str, tuple[float, float, float]]:
    pose_points: dict[str, tuple[float, float, float]] = {}
    if not pose_result.pose_landmarks:
        return pose_points
    for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
        name = POSE_LANDMARK_NAMES[idx] if idx < len(POSE_LANDMARK_NAMES) else f"pose_{idx}"
        pose_points[name] = (
            x0 + landmark.x * crop_w,
            y0 + landmark.y * crop_h,
            float(getattr(landmark, "visibility", np.nan)),
        )
    return pose_points


def process_video(
    video_meta: pd.Series,
    windows: pd.DataFrame,
    role_windows: pd.DataFrame,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    import cv2

    cv2_module, pose_solution, _ = import_video_stack()
    del cv2_module
    video_path = PROJECT / str(video_meta["video_path"])
    if not video_path.exists():
        video_path = Path(str(video_meta["video_path"]))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    pose_model = pose_solution.Pose(
        static_image_mode=True,
        model_complexity=int(args.pose_model_complexity),
        enable_segmentation=False,
        min_detection_confidence=float(args.pose_min_detection_confidence),
    )
    participant_rois = participant_roi_lookup(config)
    role_lookup = {
        (str(row["window_id"]), str(row["participant_id"])): row.to_dict()
        for _, row in role_windows.iterrows()
    }

    rows = []
    try:
        fps = float(video_meta["fps"])
        for _, window in windows.sort_values(["question_id", "window_start_sec"]).iterrows():
            previous_wrist = {participant_id: {} for participant_id in participant_rois}
            frame_indices = sample_frame_indices(
                float(window["window_start_sec"]),
                float(window["window_end_sec"]),
                fps,
                float(args.sample_fps),
                args.max_frames_per_window,
            )
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                height, width = frame_bgr.shape[:2]
                timestamp_sec = frame_index / fps if fps > 0 else np.nan
                for participant_id, roi in participant_rois.items():
                    role_row = role_lookup.get((str(window["window_id"]), str(participant_id)))
                    if role_row is None:
                        continue
                    x0, y0, x1, y1 = roi_to_pixels(roi, width, height)
                    crop = frame_bgr[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    crop_h, crop_w = crop.shape[:2]
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pose_result = pose_model.process(rgb)
                    pose_points = pose_points_from_result(pose_result, x0, y0, crop_w, crop_h)
                    base = {
                        "dyad_id": window["dyad_id"],
                        "session_id": window["session_id"],
                        "video_id": window["video_id"],
                        "condition": video_meta.get("condition", ""),
                        "question_id": window["question_id"],
                        "turn_id": window["window_id"],
                        "speaker_label": role_row["floor_state"],
                        "participant_id": participant_id,
                        "conversational_role": role_row["conversational_role"],
                        "utterance_text": "",
                        "turn_start_sec": float(window["window_start_sec"]),
                        "turn_end_sec": float(window["window_end_sec"]),
                        "turn_duration_sec": float(window["window_duration_sec"]),
                        "frame_index": int(frame_index),
                        "timestamp_sec": float(timestamp_sec),
                    }
                    arm_row = compute_upper_limb_frame_row(
                        base,
                        pose_points,
                        previous_wrist.setdefault(participant_id, {}),
                        config,
                    )
                    arm_row.update({
                        "window_id": window["window_id"],
                        "window_start_sec": float(window["window_start_sec"]),
                        "window_end_sec": float(window["window_end_sec"]),
                        "window_duration_sec": float(window["window_duration_sec"]),
                        "floor_state": role_row["floor_state"],
                        "active_speaker_count": int(role_row["active_speaker_count"]),
                        "speech_overlap_sec": float(role_row["speech_overlap_sec"]),
                        "speech_overlap_ratio": float(role_row["speech_overlap_ratio"]),
                        "total_speech_overlap_sec": float(role_row["total_speech_overlap_sec"]),
                    })
                    rows.append(arm_row)
    finally:
        cap.release()
        pose_model.close()
    return pd.DataFrame(rows, columns=TIME_WINDOW_FRAME_COLUMNS)


def aggregate_window_role(frame_df: pd.DataFrame, role_windows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame(columns=TIME_WINDOW_ROLE_COLUMNS)
    rows = []
    group_cols = ["video_id", "question_id", "window_id", "participant_id"]
    role_index = role_windows.set_index(group_cols)
    for keys, group in frame_df.groupby(group_cols, dropna=False):
        role_row = role_index.loc[keys]
        if isinstance(role_row, pd.DataFrame):
            role_row = role_row.iloc[0]
        summary = summarize_frame_group(group, config, include_prefix=False)
        first = group.iloc[0]
        rows.append({
            "dyad_id": first["dyad_id"],
            "session_id": first["session_id"],
            "video_id": keys[0],
            "condition": first["condition"],
            "question_id": keys[1],
            "window_id": keys[2],
            "window_start_sec": role_row["window_start_sec"],
            "window_end_sec": role_row["window_end_sec"],
            "window_duration_sec": role_row["window_duration_sec"],
            "participant_id": keys[3],
            "conversational_role": role_row["conversational_role"],
            "floor_state": role_row["floor_state"],
            "active_speaker_count": role_row["active_speaker_count"],
            "speech_overlap_sec": role_row["speech_overlap_sec"],
            "speech_overlap_ratio": role_row["speech_overlap_ratio"],
            "total_speech_overlap_sec": role_row["total_speech_overlap_sec"],
            **summary,
        })
    return pd.DataFrame(rows, columns=TIME_WINDOW_ROLE_COLUMNS)


def aggregate_participant_role(window_role_df: pd.DataFrame) -> pd.DataFrame:
    if window_role_df.empty:
        return pd.DataFrame(columns=PARTICIPANT_ROLE_SUMMARY_COLUMNS)
    rows = []
    group_cols = ["dyad_id", "session_id", "video_id", "condition", "participant_id", "conversational_role"]
    for keys, group in window_role_df.groupby(group_cols, dropna=False):
        weights = pd.to_numeric(group["n_sampled_frames"], errors="coerce").fillna(0)
        rows.append({
            **{col: value for col, value in zip(group_cols, keys, strict=False)},
            "n_windows": int(len(group)),
            "total_window_time_sec": float(pd.to_numeric(group["window_duration_sec"], errors="coerce").sum()),
            "total_speech_overlap_sec": float(pd.to_numeric(group["speech_overlap_sec"], errors="coerce").sum()),
            "mean_speech_overlap_ratio": finite_mean(group["speech_overlap_ratio"]),
            "mean_pose_detection_rate": weighted_mean(group["pose_detection_rate"], weights),
            "mean_arm_detection_rate": weighted_mean(group["arm_detection_rate"], weights),
            "mean_wrist_speed_px_s": weighted_mean(group["mean_wrist_speed_px_s"], weights),
            "max_wrist_speed_px_s": finite_max(group["max_wrist_speed_px_s"]),
            "active_motion_frame_rate": weighted_mean(group["active_motion_frame_rate"], weights),
            "hold_frame_rate": weighted_mean(group["hold_frame_rate"], weights),
            "near_face_frame_rate": weighted_mean(group["near_face_frame_rate"], weights),
            "raised_frame_rate": weighted_mean(group["raised_frame_rate"], weights),
            "open_frame_rate": weighted_mean(group["open_frame_rate"], weights),
            "closed_frame_rate": weighted_mean(group["closed_frame_rate"], weights),
            "resting_low_frame_rate": weighted_mean(group["resting_low_frame_rate"], weights),
            "mean_arm_openness_index": weighted_mean(group["mean_arm_openness_index"], weights),
            "mean_arm_asymmetry_index": weighted_mean(group["mean_arm_asymmetry_index"], weights),
            "qc_status": "WARNING" if (group["qc_status"] == "WARNING").any() else "PASS",
            "qc_notes": join_notes(group["qc_notes"]),
        })
    return pd.DataFrame(rows, columns=PARTICIPANT_ROLE_SUMMARY_COLUMNS)


def role_counts(role_windows: pd.DataFrame) -> pd.DataFrame:
    if role_windows.empty:
        return pd.DataFrame()
    return (
        role_windows.groupby(["dyad_id", "session_id", "video_id", "condition", "participant_id", "conversational_role"], dropna=False)
        .agg(
            n_windows=("window_id", "count"),
            total_window_time_sec=("window_duration_sec", "sum"),
            total_speech_overlap_sec=("speech_overlap_sec", "sum"),
        )
        .reset_index()
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def write_manifest(run_dir: Path, source_run: Path, outputs: list[Path], config_used: dict[str, Any]) -> None:
    manifest = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_run": str(source_run),
        "software_versions": get_software_versions({"openface": {"enabled": False}, "mediapipe": {"enabled": True}}),
        "config": config_used,
        "inputs": [
            file_record(source_run / "metadata" / "video_metadata_checked.csv", "source_run_metadata"),
            file_record(source_run / "metadata" / "labeled_turns_checked.csv", "source_run_turns"),
        ],
        "outputs": [file_record(path, "pipeline_output") for path in outputs if path.exists()],
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)


def run(args: argparse.Namespace) -> int:
    source_run = args.source_run or latest_turn_run()
    config = load_source_config(source_run)
    video_meta, turns, question_windows = load_source_tables(source_run)
    video_meta, question_windows = select_inputs(video_meta, question_windows, args)
    windows = build_time_windows(question_windows, float(args.window_sec), float(args.stride_sec))
    participants = list(participant_roi_lookup(config).keys())
    role_windows = role_rows_for_windows(
        windows,
        turns,
        participants,
        float(args.min_speech_overlap_sec),
        float(args.min_speech_overlap_ratio),
    )

    run_id = args.run_id or f"time_window_arm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / run_id
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                "Output run directory already exists:\n"
                f"  {run_dir}\n"
                "Use a new --run-id or pass --overwrite for generated outputs."
            )
        shutil.rmtree(run_dir)
    for subdir in ["metadata", "intermediate", "results", "qc"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    config_used = json_ready(vars(args).copy())
    config_used["source_run"] = str(source_run)
    config_used["participants"] = participants
    config_used["note"] = "Fixed-window arm features with speaker/listener/silent/overlap roles assigned from audio-aligned turn timestamps."
    atomic_write_json(run_dir / "config_used.json", config_used)
    atomic_write_csv(run_dir / "metadata" / "question_windows.csv", question_windows)
    atomic_write_csv(run_dir / "metadata" / "time_windows.csv", windows)
    atomic_write_csv(run_dir / "metadata" / "time_window_roles.csv", role_windows)
    if args.dry_run:
        outputs = [
            run_dir / "metadata" / "question_windows.csv",
            run_dir / "metadata" / "time_windows.csv",
            run_dir / "metadata" / "time_window_roles.csv",
        ]
        write_manifest(run_dir, source_run, outputs, config_used)
        print(f"Dry run complete: {run_dir}")
        return 0

    frame_parts = []
    failed_rows = []
    for _, video in video_meta.iterrows():
        video_id = str(video["video_id"])
        video_windows = windows[windows["video_id"].astype(str) == video_id].copy()
        video_role_windows = role_windows[role_windows["video_id"].astype(str) == video_id].copy()
        if video_windows.empty:
            continue
        print(f"Processing {video_id}: {len(video_windows)} time windows")
        try:
            frame_parts.append(process_video(video, video_windows, video_role_windows, config, args))
        except Exception as exc:
            failed_rows.append({
                "video_id": video.get("video_id", ""),
                "video_path": video.get("video_path", ""),
                "stage": "main_video_analysis",
                "error": str(exc),
                "suggested_fix": "Check video readability and the active MediaPipe/OpenCV environment.",
            })
            print(f"ERROR processing {video_id}: {exc}")
            if args.fail_fast:
                break

    frame_df = pd.concat(frame_parts, ignore_index=True) if frame_parts else pd.DataFrame(columns=TIME_WINDOW_FRAME_COLUMNS)
    window_role_df = aggregate_window_role(frame_df, role_windows, config)
    participant_role_df = aggregate_participant_role(window_role_df)
    count_df = role_counts(role_windows)
    failed_df = pd.DataFrame(failed_rows, columns=["video_id", "video_path", "stage", "error", "suggested_fix"])

    outputs = [
        run_dir / "intermediate" / "per_frame_time_window_arm_features.csv",
        run_dir / "results" / "per_time_window_role_features.csv",
        run_dir / "results" / "per_participant_time_window_role_summary.csv",
        run_dir / "results" / "time_window_role_counts.csv",
        run_dir / "qc" / "failed_videos.csv",
    ]
    atomic_write_csv(outputs[0], frame_df)
    atomic_write_csv(outputs[1], window_role_df)
    atomic_write_csv(outputs[2], participant_role_df)
    atomic_write_csv(outputs[3], count_df)
    atomic_write_csv(outputs[4], failed_df)
    write_manifest(
        run_dir,
        source_run,
        outputs + [
            run_dir / "metadata" / "question_windows.csv",
            run_dir / "metadata" / "time_windows.csv",
            run_dir / "metadata" / "time_window_roles.csv",
        ],
        config_used,
    )
    print(f"Time-window arm role run complete: {run_dir}")
    return 1 if failed_rows and not frame_parts else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=None, help="Completed turn-aligned run to reuse for metadata and audio-aligned turns.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Reuse an existing generated run directory.")
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames-per-window", type=int, default=1)
    parser.add_argument("--min-speech-overlap-sec", type=float, default=0.20)
    parser.add_argument("--min-speech-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--pose-model-complexity", type=int, default=2)
    parser.add_argument("--pose-min-detection-confidence", type=float, default=0.50)
    parser.add_argument("--video-id", nargs="*", help="Optional video_id filter.")
    parser.add_argument("--question-id", nargs="*", help="Optional question_id filter.")
    parser.add_argument("--limit-videos", type=int, default=None, help="Debug/smoke-test video limit.")
    parser.add_argument("--max-questions-per-video", type=int, default=None, help="Debug/smoke-test question limit.")
    parser.add_argument("--dry-run", action="store_true", help="Write time-window role metadata only.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first video-processing error.")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_arg_parser().parse_args()))

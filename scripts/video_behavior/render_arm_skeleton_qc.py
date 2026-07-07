#!/usr/bin/env python3
"""
Render shoulder-elbow-wrist skeleton overlays for upper-limb QC.

This visual audit helper reads a completed turn-aligned video run, samples
turn frames, and draws participant ROIs plus connected MediaPipe Pose arm
landmarks. Outputs are written to analysis-output folders, not raw video
locations.

Usage:
  python scripts/video_behavior/render_arm_skeleton_qc.py \
    --source-run data/interim/video_behavior/turn_runs/<run_id>

  python scripts/video_behavior/render_arm_skeleton_qc.py \
    --source-run data/interim/video_behavior/turn_runs/<run_id> \
    --video-id dyad12_250928_piper --question-id 1 --write-video
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.video_behavior.run_question_side_video_analysis import (  # noqa: E402
    latest_turn_run,
    read_source_run,
)
from scripts.video_behavior.run_turn_aligned_video_analysis import (  # noqa: E402
    DEFAULT_CONFIG,
    deep_merge,
    import_video_stack,
    participant_roi_lookup,
    roi_to_pixels,
    sample_frame_indices,
    speaker_lookup_from_config,
)
from scripts.video_behavior.run_video_analysis import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    file_record,
    get_software_versions,
)
from scripts.video_behavior.upper_limb_features import (  # noqa: E402
    compute_upper_limb_frame_row,
    frame_base_from_turn,
)


OUTPUT_ROOT = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs")) / "video_arm_skeleton_qc"

FRAME_COLUMNS = [
    "video_id", "dyad_id", "session_id", "condition", "question_id",
    "turn_id", "speaker_label", "participant_id", "conversational_role",
    "frame_index", "timestamp_sec", "pose_detected", "n_detected_arms",
    "upper_limb_behavior_state", "upper_limb_posture_state", "output_frame",
    "output_video",
]
SUMMARY_COLUMNS = [
    "video_id", "participant_id", "conversational_role", "n_rows",
    "pose_detection_rate", "arm_detection_rate",
]

ARM_CONNECTIONS = [
    ("left_shoulder", "left_elbow", (0, 190, 255)),
    ("left_elbow", "left_wrist", (0, 190, 255)),
    ("right_shoulder", "right_elbow", (255, 150, 0)),
    ("right_elbow", "right_wrist", (255, 150, 0)),
]
ARM_DOTS = {
    "left_shoulder": (0, 190, 255),
    "left_elbow": (0, 190, 255),
    "left_wrist": (0, 190, 255),
    "right_shoulder": (255, 150, 0),
    "right_elbow": (255, 150, 0),
    "right_wrist": (255, 150, 0),
}


def sanitize_token(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "missing"


def resolve_video_path(path_value: Any) -> Path:
    path = Path(str(path_value))
    if path.exists():
        return path
    project_path = PROJECT / path
    if project_path.exists():
        return project_path
    return path


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


def select_videos(video_meta: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = video_meta.copy()
    if args.video_id:
        requested = set(args.video_id)
        selected = selected[selected["video_id"].astype(str).isin(requested)].copy()
        missing = sorted(requested - set(selected["video_id"].astype(str)))
        if missing:
            raise SystemExit("Requested video_id not found in source run: " + ", ".join(missing))
    if args.limit_videos and args.limit_videos > 0:
        selected = selected.head(args.limit_videos).copy()
    return selected.reset_index(drop=True)


def select_turns(turns: pd.DataFrame, video_ids: set[str], args: argparse.Namespace) -> pd.DataFrame:
    selected = turns[turns["video_id"].astype(str).isin(video_ids)].copy()
    if args.question_id:
        requested_questions = {str(value) for value in args.question_id}
        selected = selected[selected["question_id"].astype(str).isin(requested_questions)].copy()
    if args.turn_id:
        requested_turns = {str(value) for value in args.turn_id}
        selected = selected[selected["turn_id"].astype(str).isin(requested_turns)].copy()
    if args.max_turns_per_video and args.max_turns_per_video > 0:
        selected = (
            selected.sort_values(["video_id", "turn_id"])
            .groupby("video_id", dropna=False)
            .head(args.max_turns_per_video)
            .reset_index(drop=True)
        )
    return selected.reset_index(drop=True)


def draw_label(cv2: Any, frame_bgr: Any, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    y = max(28, y)
    width = max(180, min(520, 12 * len(text)))
    cv2.rectangle(frame_bgr, (x, y - 24), (x + width, y + 8), (20, 20, 20), -1)
    cv2.putText(frame_bgr, text, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def draw_arm_skeleton(cv2: Any, frame_bgr: Any, pose_points: dict[str, tuple[float, float, float]]) -> None:
    for start_name, end_name, color in ARM_CONNECTIONS:
        start = pose_points.get(start_name)
        end = pose_points.get(end_name)
        if start is None or end is None:
            continue
        cv2.line(
            frame_bgr,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            color,
            4,
            cv2.LINE_AA,
        )
    for name, color in ARM_DOTS.items():
        point = pose_points.get(name)
        if point is None:
            continue
        cv2.circle(frame_bgr, (int(round(point[0])), int(round(point[1]))), 7, color, -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, (int(round(point[0])), int(round(point[1]))), 8, (10, 10, 10), 1, cv2.LINE_AA)


def frame_indices_for_turn(turn: pd.Series, fps: float, args: argparse.Namespace) -> list[int]:
    start = float(turn["start_time_sec"])
    end = float(turn["end_time_sec"])
    if args.every_frame:
        start_frame = max(0, int(start * fps))
        end_frame = max(start_frame, int(end * fps))
        indices = list(range(start_frame, end_frame + 1))
        if args.max_frames_per_turn and args.max_frames_per_turn > 0:
            indices = indices[:args.max_frames_per_turn]
        return indices
    return sample_frame_indices(
        start,
        end,
        fps,
        float(args.sample_fps),
        int(args.max_frames_per_turn) if args.max_frames_per_turn else None,
    )


def pose_points_from_result(
    pose_result: Any,
    landmark_names: list[str],
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
) -> dict[str, tuple[float, float, float]]:
    pose_points: dict[str, tuple[float, float, float]] = {}
    if not pose_result.pose_landmarks:
        return pose_points
    for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
        name = landmark_names[idx] if idx < len(landmark_names) else f"pose_{idx}"
        pose_points[name] = (
            x0 + landmark.x * crop_w,
            y0 + landmark.y * crop_h,
            float(getattr(landmark, "visibility", 1.0)),
        )
    return pose_points


def process_video(
    video_meta: pd.Series,
    turns: pd.DataFrame,
    config: dict[str, Any],
    args: argparse.Namespace,
    frames_dir: Path,
    videos_dir: Path,
) -> list[dict[str, Any]]:
    cv2, pose_solution, _ = import_video_stack()
    from scripts.video_behavior.run_video_analysis import POSE_LANDMARK_NAMES

    video_path = resolve_video_path(video_meta["video_path"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    pose_model = pose_solution.Pose(
        static_image_mode=True,
        model_complexity=int(config["mediapipe"]["pose_model_complexity"]),
        enable_segmentation=False,
        min_detection_confidence=float(config["mediapipe"]["pose_min_detection_confidence"]),
    )
    participant_rois = participant_roi_lookup(config)
    speaker_to_participant = speaker_lookup_from_config(config)
    writer = None
    output_video = ""
    rows: list[dict[str, Any]] = []

    try:
        fps = float(video_meta["fps"])
        video_turns = turns.sort_values(["question_id", "turn_id"]).copy()
        for _, turn in video_turns.iterrows():
            speaker_participant = (
                speaker_to_participant.get(str(turn["speaker_label"]))
                or speaker_to_participant.get(str(turn.get("participant_role", "")))
                or str(turn.get("participant_role", "")).strip()
                or str(turn["speaker_label"])
            )
            previous_wrist = {participant_id: {} for participant_id in participant_rois}
            for frame_index in frame_indices_for_turn(turn, fps, args):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                frame_out = frame_bgr.copy()
                height, width = frame_out.shape[:2]
                frame_participant_rows = []
                for participant_id, roi in participant_rois.items():
                    x0, y0, x1, y1 = roi_to_pixels(roi, width, height)
                    crop = frame_bgr[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    crop_h, crop_w = crop.shape[:2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pose_result = pose_model.process(crop_rgb)
                    pose_points = pose_points_from_result(
                        pose_result,
                        POSE_LANDMARK_NAMES,
                        x0,
                        y0,
                        crop_w,
                        crop_h,
                    )
                    role = "speaker" if participant_id == speaker_participant else "listener"
                    base = frame_base_from_turn(
                        turn,
                        video_meta,
                        participant_id,
                        role,
                        int(frame_index),
                        frame_index / fps if fps > 0 else 0.0,
                    )
                    arm_row = compute_upper_limb_frame_row(
                        base,
                        pose_points,
                        previous_wrist.setdefault(participant_id, {}),
                        config,
                    )

                    color = (0, 180, 255) if role == "speaker" else (170, 220, 90)
                    cv2.rectangle(frame_out, (x0, y0), (x1, y1), color, 3)
                    draw_arm_skeleton(cv2, frame_out, pose_points)
                    label = (
                        f"{participant_id} {role} {arm_row['upper_limb_behavior_state']} "
                        f"| {arm_row['speaker_label']} t{arm_row['turn_id']}"
                    )
                    draw_label(cv2, frame_out, label, x0 + 8, y0 + 32, color)
                    frame_participant_rows.append(arm_row)

                frame_name = (
                    f"{sanitize_token(video_meta['video_id'])}_"
                    f"q{sanitize_token(turn['question_id'])}_"
                    f"t{sanitize_token(turn['turn_id'])}_"
                    f"frame{int(frame_index):06d}.jpg"
                )
                frame_path = frames_dir / frame_name
                cv2.imwrite(str(frame_path), frame_out)

                if args.write_video:
                    if writer is None:
                        videos_dir.mkdir(parents=True, exist_ok=True)
                        video_name = f"{sanitize_token(video_meta['video_id'])}_arm_skeleton_qc.mp4"
                        output_video = str(videos_dir / video_name)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_video, fourcc, float(args.output_fps), (width, height))
                    writer.write(frame_out)

                for arm_row in frame_participant_rows:
                    rows.append({
                        "video_id": arm_row["video_id"],
                        "dyad_id": arm_row["dyad_id"],
                        "session_id": arm_row["session_id"],
                        "condition": arm_row["condition"],
                        "question_id": arm_row["question_id"],
                        "turn_id": arm_row["turn_id"],
                        "speaker_label": arm_row["speaker_label"],
                        "participant_id": arm_row["participant_id"],
                        "conversational_role": arm_row["conversational_role"],
                        "frame_index": arm_row["frame_index"],
                        "timestamp_sec": arm_row["timestamp_sec"],
                        "pose_detected": arm_row["pose_detected"],
                        "n_detected_arms": arm_row["n_detected_arms"],
                        "upper_limb_behavior_state": arm_row["upper_limb_behavior_state"],
                        "upper_limb_posture_state": arm_row["upper_limb_posture_state"],
                        "output_frame": str(frame_path),
                        "output_video": output_video,
                    })
    finally:
        if writer is not None:
            writer.release()
        pose_model.close()
        cap.release()
    return rows


def summarize_rows(frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = frame_df.copy()
    df["pose_detected"] = df["pose_detected"].astype(bool)
    df["arm_detected"] = pd.to_numeric(df["n_detected_arms"], errors="coerce").fillna(0) > 0
    return (
        df.groupby(["video_id", "participant_id", "conversational_role"], dropna=False)
        .agg(
            n_rows=("frame_index", "count"),
            pose_detection_rate=("pose_detected", "mean"),
            arm_detection_rate=("arm_detected", "mean"),
        )
        .reset_index()
    )


def write_manifest(run_dir: Path, source_run: Path, output_paths: list[Path], config: dict[str, Any]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_run": str(source_run),
        "software_versions": get_software_versions(config),
        "outputs": [file_record(path, "arm_skeleton_qc_output") for path in output_paths if path.exists()],
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)


def run(args: argparse.Namespace) -> int:
    source_run = Path(args.source_run) if args.source_run else latest_turn_run()
    video_meta, turns = read_source_run(source_run)
    config = load_source_config(source_run)
    selected_videos = select_videos(video_meta, args)
    selected_turns = select_turns(turns, set(selected_videos["video_id"].astype(str)), args)

    run_id = args.run_id or f"arm_skeleton_qc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_root) if args.output_root else OUTPUT_ROOT
    run_dir = output_root / run_id
    frames_dir = run_dir / "frames"
    videos_dir = run_dir / "overlay_videos"
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                "Output run directory already exists:\n"
                f"  {run_dir}\n"
                "Use a new --run-id, or rerun with --overwrite if you want to replace this generated QC output."
            )
        shutil.rmtree(run_dir)
    frames_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    for _, video in selected_videos.iterrows():
        video_turns = selected_turns[selected_turns["video_id"].astype(str) == str(video["video_id"])].copy()
        if video_turns.empty:
            continue
        rows.extend(process_video(video, video_turns, config, args, frames_dir, videos_dir))

    frame_df = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    summary_df = summarize_rows(frame_df)
    frame_path = run_dir / "arm_skeleton_frame_manifest.csv"
    summary_path = run_dir / "arm_skeleton_qc_summary.csv"
    atomic_write_csv(frame_path, frame_df)
    atomic_write_csv(summary_path, summary_df)
    write_manifest(run_dir, source_run, [frame_path, summary_path, videos_dir], config)
    print(f"Arm skeleton QC written to: {run_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render connected upper-limb skeleton overlays for CVS video QC.")
    parser.add_argument("--source-run", help="Completed turn-aligned run directory. Defaults to the latest run.")
    parser.add_argument("--run-id", help="Output run id under the QC output root.")
    parser.add_argument("--output-root", help="Directory for QC outputs. Defaults to CVS_ANALYSIS_OUTPUTS/video_arm_skeleton_qc.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated QC run directory with the same run id.")
    parser.add_argument("--video-id", nargs="*", help="Optional video_id filter.")
    parser.add_argument("--question-id", nargs="*", help="Optional question_id filter.")
    parser.add_argument("--turn-id", nargs="*", help="Optional turn_id filter.")
    parser.add_argument("--limit-videos", type=int, default=1, help="Maximum videos to render; 0 means all selected videos.")
    parser.add_argument("--max-turns-per-video", type=int, default=5, help="Maximum turns per video; 0 means all selected turns.")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frame sampling rate when --every-frame is not set.")
    parser.add_argument("--max-frames-per-turn", type=int, default=20, help="Maximum rendered frames per turn; 0 means no cap.")
    parser.add_argument("--every-frame", action="store_true", help="Render every native frame in selected turns.")
    parser.add_argument("--write-video", action="store_true", help="Also write an MP4 overlay video from rendered frames.")
    parser.add_argument("--output-fps", type=float, default=10.0, help="Overlay-video frame rate.")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_arg_parser().parse_args()))

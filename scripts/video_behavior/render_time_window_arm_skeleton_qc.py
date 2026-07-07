#!/usr/bin/env python3
"""
Render time-window arm-role skeleton overlays for frame-by-frame QC.

This visual helper reads a completed run_time_window_arm_role_analysis.py run
and draws participant ROIs plus connected shoulder-elbow-wrist landmarks. The
labels come from audio-aligned time-window role assignment: speaker, listener,
silent, or overlap.

Usage:
  python scripts/video_behavior/render_time_window_arm_skeleton_qc.py \
    --time-window-run data/interim/video_behavior/time_window_arm_runs/time_window_arm_full \
    --run-id time_window_arm_frame_qc \
    --limit-videos 1 --max-windows-per-video 20 --every-frame --write-video
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

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.video_behavior.render_arm_skeleton_qc import (  # noqa: E402
    draw_arm_skeleton,
    draw_label,
    load_source_config,
    pose_points_from_result,
    resolve_video_path,
    sanitize_token,
)
from scripts.video_behavior.run_turn_aligned_video_analysis import (  # noqa: E402
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
    compute_upper_limb_frame_row,
)


OUTPUT_ROOT = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs")) / "video_time_window_arm_skeleton_qc"
TIME_WINDOW_RUNS = PROJECT / "data" / "interim" / "video_behavior" / "time_window_arm_runs"

FRAME_COLUMNS = [
    "video_id", "dyad_id", "session_id", "condition", "question_id",
    "window_id", "window_start_sec", "window_end_sec", "participant_id",
    "conversational_role", "floor_state", "frame_index", "timestamp_sec",
    "n_detected_arms", "upper_limb_behavior_state",
    "upper_limb_posture_state", "output_frame", "output_video",
]
SUMMARY_COLUMNS = [
    "video_id", "participant_id", "conversational_role", "floor_state",
    "n_rows", "arm_detection_rate",
]

ROLE_COLORS = {
    "speaker": (0, 180, 255),
    "listener": (110, 220, 90),
    "silent": (190, 190, 190),
    "overlap": (245, 120, 255),
}


def latest_time_window_run() -> Path:
    runs = sorted(
        p for p in TIME_WINDOW_RUNS.glob("time_window_arm_*")
        if (p / "metadata" / "time_window_roles.csv").exists()
    )
    if not runs:
        raise SystemExit(f"No completed time-window arm run found under {TIME_WINDOW_RUNS}")
    return runs[-1]


def load_time_window_run(time_window_run: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    roles_path = time_window_run / "metadata" / "time_window_roles.csv"
    config_path = time_window_run / "config_used.json"
    missing = [str(path) for path in [roles_path, config_path] if not path.exists()]
    if missing:
        raise SystemExit("Missing expected time-window run file(s):\n" + "\n".join(missing))
    roles = pd.read_csv(roles_path)
    config_used = json.loads(config_path.read_text(encoding="utf-8"))
    source_run = Path(config_used.get("source_run", ""))
    if not source_run.exists():
        raise SystemExit(f"Source turn-aligned run from config_used.json does not exist: {source_run}")
    video_meta_path = source_run / "metadata" / "video_metadata_checked.csv"
    if not video_meta_path.exists():
        raise SystemExit(f"Missing source video metadata: {video_meta_path}")
    return roles, pd.read_csv(video_meta_path), source_run


def select_roles(roles: pd.DataFrame, video_meta: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = roles.copy()
    videos = video_meta.copy()
    if args.video_id:
        requested = set(args.video_id)
        selected = selected[selected["video_id"].astype(str).isin(requested)].copy()
        videos = videos[videos["video_id"].astype(str).isin(requested)].copy()
    if args.question_id:
        requested_questions = {str(value) for value in args.question_id}
        selected = selected[selected["question_id"].astype(str).isin(requested_questions)].copy()
    if args.window_id:
        requested_windows = set(args.window_id)
        selected = selected[selected["window_id"].astype(str).isin(requested_windows)].copy()
    if args.role:
        requested_roles = set(args.role)
        selected = selected[selected["conversational_role"].astype(str).isin(requested_roles)].copy()
    if args.floor_state:
        requested_floor = set(args.floor_state)
        selected = selected[selected["floor_state"].astype(str).isin(requested_floor)].copy()
    videos = videos[videos["video_id"].astype(str).isin(set(selected["video_id"].astype(str)))].copy()
    if args.limit_videos is not None:
        keep = sorted(selected["video_id"].astype(str).unique())[: args.limit_videos]
        selected = selected[selected["video_id"].astype(str).isin(keep)].copy()
        videos = videos[videos["video_id"].astype(str).isin(keep)].copy()
    if args.max_windows_per_video is not None:
        window_keys = (
            selected[["video_id", "question_id", "window_id", "window_start_sec"]]
            .drop_duplicates()
            .sort_values(["video_id", "question_id", "window_start_sec"])
            .groupby("video_id", dropna=False)
            .head(args.max_windows_per_video)
        )
        selected = selected.merge(
            window_keys[["video_id", "question_id", "window_id"]],
            on=["video_id", "question_id", "window_id"],
            how="inner",
        )
    videos = videos[videos["video_id"].astype(str).isin(set(selected["video_id"].astype(str)))].copy()
    return selected.reset_index(drop=True), videos.reset_index(drop=True)


def frame_indices_for_window(window: pd.Series, fps: float, args: argparse.Namespace) -> list[int]:
    start = float(window["window_start_sec"])
    end = float(window["window_end_sec"])
    if args.every_frame:
        start_frame = max(0, int(round(start * fps)))
        end_frame = max(start_frame, int(round(end * fps)))
        indices = list(range(start_frame, end_frame + 1))
        if args.max_frames_per_window and args.max_frames_per_window > 0:
            indices = indices[:args.max_frames_per_window]
        return indices
    return sample_frame_indices(
        start,
        end,
        fps,
        float(args.sample_fps),
        int(args.max_frames_per_window) if args.max_frames_per_window else None,
    )


def process_video(
    video_meta: pd.Series,
    roles: pd.DataFrame,
    config: dict[str, Any],
    args: argparse.Namespace,
    frames_dir: Path,
    videos_dir: Path,
) -> list[dict[str, Any]]:
    import cv2

    _, pose_solution, _ = import_video_stack()
    video_path = resolve_video_path(video_meta["video_path"])
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
    writer = None
    output_video = ""
    rows: list[dict[str, Any]] = []

    try:
        fps = float(video_meta["fps"])
        windows = (
            roles[["dyad_id", "session_id", "video_id", "condition", "question_id", "window_id", "window_start_sec", "window_end_sec"]]
            .drop_duplicates()
            .sort_values(["question_id", "window_start_sec"])
        )
        for _, window in windows.iterrows():
            window_roles = roles[
                (roles["video_id"].astype(str) == str(window["video_id"]))
                & (roles["window_id"].astype(str) == str(window["window_id"]))
            ].copy()
            previous_wrist = {participant_id: {} for participant_id in participant_rois}
            for frame_index in frame_indices_for_window(window, fps, args):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                frame_out = frame_bgr.copy()
                height, width = frame_out.shape[:2]
                frame_rows = []
                for _, role_row in window_roles.iterrows():
                    participant_id = str(role_row["participant_id"])
                    roi = participant_rois.get(participant_id)
                    if roi is None:
                        continue
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
                    base = {
                        "dyad_id": role_row["dyad_id"],
                        "session_id": role_row["session_id"],
                        "video_id": role_row["video_id"],
                        "condition": role_row.get("condition", video_meta.get("condition", "")),
                        "question_id": role_row["question_id"],
                        "turn_id": role_row["window_id"],
                        "speaker_label": role_row["floor_state"],
                        "participant_id": participant_id,
                        "conversational_role": role_row["conversational_role"],
                        "utterance_text": "",
                        "turn_start_sec": float(role_row["window_start_sec"]),
                        "turn_end_sec": float(role_row["window_end_sec"]),
                        "turn_duration_sec": float(role_row["window_end_sec"]) - float(role_row["window_start_sec"]),
                        "frame_index": int(frame_index),
                        "timestamp_sec": frame_index / fps if fps > 0 else 0.0,
                    }
                    arm_row = compute_upper_limb_frame_row(
                        base,
                        pose_points,
                        previous_wrist.setdefault(participant_id, {}),
                        config,
                    )
                    role = str(role_row["conversational_role"])
                    color = ROLE_COLORS.get(role, (220, 220, 220))
                    cv2.rectangle(frame_out, (x0, y0), (x1, y1), color, 3)
                    draw_arm_skeleton(cv2, frame_out, pose_points)
                    label = (
                        f"{participant_id} {role} {arm_row['upper_limb_behavior_state']} "
                        f"| {role_row['floor_state']} {role_row['window_id']}"
                    )
                    draw_label(cv2, frame_out, label, x0 + 8, y0 + 32, color)
                    frame_rows.append({
                        "video_id": role_row["video_id"],
                        "dyad_id": role_row["dyad_id"],
                        "session_id": role_row["session_id"],
                        "condition": role_row.get("condition", video_meta.get("condition", "")),
                        "question_id": role_row["question_id"],
                        "window_id": role_row["window_id"],
                        "window_start_sec": role_row["window_start_sec"],
                        "window_end_sec": role_row["window_end_sec"],
                        "participant_id": participant_id,
                        "conversational_role": role,
                        "floor_state": role_row["floor_state"],
                        "frame_index": int(frame_index),
                        "timestamp_sec": frame_index / fps if fps > 0 else 0.0,
                        "n_detected_arms": arm_row["n_detected_arms"],
                        "upper_limb_behavior_state": arm_row["upper_limb_behavior_state"],
                        "upper_limb_posture_state": arm_row["upper_limb_posture_state"],
                    })

                frame_name = (
                    f"{sanitize_token(video_meta['video_id'])}_"
                    f"{sanitize_token(window['window_id'])}_"
                    f"frame{int(frame_index):06d}.jpg"
                )
                frame_path = frames_dir / frame_name
                cv2.imwrite(str(frame_path), frame_out)
                if args.write_video:
                    if writer is None:
                        videos_dir.mkdir(parents=True, exist_ok=True)
                        video_name = f"{sanitize_token(video_meta['video_id'])}_time_window_arm_qc.mp4"
                        output_video = str(videos_dir / video_name)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_video, fourcc, float(args.output_fps), (width, height))
                    writer.write(frame_out)
                for row in frame_rows:
                    row["output_frame"] = str(frame_path)
                    row["output_video"] = output_video
                    rows.append(row)
    finally:
        if writer is not None:
            writer.release()
        pose_model.close()
        cap.release()
    return rows


def summarize(frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = frame_df.copy()
    df["arm_detected"] = pd.to_numeric(df["n_detected_arms"], errors="coerce").fillna(0) > 0
    return (
        df.groupby(["video_id", "participant_id", "conversational_role", "floor_state"], dropna=False)
        .agg(
            n_rows=("frame_index", "count"),
            arm_detection_rate=("arm_detected", "mean"),
        )
        .reset_index()
    )


def write_manifest(run_dir: Path, time_window_run: Path, source_run: Path, output_paths: list[Path], config: dict[str, Any]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "time_window_run": str(time_window_run),
        "source_run": str(source_run),
        "software_versions": get_software_versions(config),
        "outputs": [file_record(path, "time_window_arm_skeleton_qc_output") for path in output_paths if path.exists()],
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)


def run(args: argparse.Namespace) -> int:
    time_window_run = Path(args.time_window_run) if args.time_window_run else latest_time_window_run()
    roles, video_meta, source_run = load_time_window_run(time_window_run)
    config = load_source_config(source_run)
    selected_roles, selected_videos = select_roles(roles, video_meta, args)

    run_id = args.run_id or f"time_window_arm_skeleton_qc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        video_roles = selected_roles[selected_roles["video_id"].astype(str) == str(video["video_id"])].copy()
        if video_roles.empty:
            continue
        print(f"Rendering {video['video_id']}: {video_roles['window_id'].nunique()} windows")
        rows.extend(process_video(video, video_roles, config, args, frames_dir, videos_dir))

    frame_df = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    summary_df = summarize(frame_df)
    frame_path = run_dir / "time_window_arm_skeleton_frame_manifest.csv"
    summary_path = run_dir / "time_window_arm_skeleton_qc_summary.csv"
    atomic_write_csv(frame_path, frame_df)
    atomic_write_csv(summary_path, summary_df)
    write_manifest(run_dir, time_window_run, source_run, [frame_path, summary_path, videos_dir], config)
    print(f"Time-window arm skeleton QC written to: {run_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render frame-by-frame time-window arm role overlays.")
    parser.add_argument("--time-window-run", help="Completed time-window arm run. Defaults to the latest run.")
    parser.add_argument("--run-id", help="Output run id under the QC output root.")
    parser.add_argument("--output-root", help="Directory for QC outputs. Defaults to CVS_ANALYSIS_OUTPUTS/video_time_window_arm_skeleton_qc.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated QC run directory.")
    parser.add_argument("--video-id", nargs="*", help="Optional video_id filter.")
    parser.add_argument("--question-id", nargs="*", help="Optional question_id filter.")
    parser.add_argument("--window-id", nargs="*", help="Optional window_id filter, for example q1_w0003.")
    parser.add_argument("--role", nargs="*", choices=["speaker", "listener", "silent", "overlap"], help="Optional conversational-role filter.")
    parser.add_argument("--floor-state", nargs="*", choices=["single_speaker", "silent", "overlap"], help="Optional floor-state filter.")
    parser.add_argument("--limit-videos", type=int, default=1, help="Maximum videos to render; omit for all selected videos.")
    parser.add_argument("--max-windows-per-video", type=int, default=20, help="Maximum unique windows per video; omit for all selected windows.")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Sampling rate when --every-frame is not set.")
    parser.add_argument("--max-frames-per-window", type=int, default=30, help="Maximum rendered frames per window; 0 means no cap.")
    parser.add_argument("--every-frame", action="store_true", help="Render every native frame in selected windows.")
    parser.add_argument("--write-video", action="store_true", help="Also write an MP4 overlay video from rendered frames.")
    parser.add_argument("--output-fps", type=float, default=10.0, help="Overlay-video frame rate.")
    parser.add_argument("--pose-model-complexity", type=int, default=2)
    parser.add_argument("--pose-min-detection-confidence", type=float, default=0.50)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_arg_parser().parse_args()))

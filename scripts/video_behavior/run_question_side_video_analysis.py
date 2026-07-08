#!/usr/bin/env python3
"""
Question-level left/right video behavior analysis.

This runner is intentionally independent of speaker labels. It reuses a
completed turn-aligned run only to get checked video metadata and question
time windows, then extracts MediaPipe features from fixed left/right ROIs.

Outputs are generated in a new timestamped run directory. Raw videos and the
source turn-aligned run are never modified.

Usage:
  python scripts/video_behavior/run_question_side_video_analysis.py \
    --source-run data/interim/video_behavior/turn_runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import math
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
from scripts.video_behavior.run_turn_aligned_video_analysis import (  # noqa: E402
    finite_max,
    finite_mean,
    import_video_stack,
    roi_to_pixels,
    sample_frame_indices,
)
from scripts.video_behavior.run_video_analysis import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    file_record,
    get_software_versions,
)


TURN_RUNS = PROJECT / "data" / "interim" / "video_behavior" / "turn_runs"
OUTPUT_ROOT = Path(
    os.environ.get("CVS_VIDEO_OUTPUTS", PROJECT / "data" / "interim" / "video_behavior" / "side_question_runs")
)

# Side-to-identity mapping (hard-coded lab standard, every video):
# left half of the screen = participant B (window side),
# right half of the screen = participant A (door side).
SIDE_ROIS = {
    "left": [0.0, 0.0, 0.5, 1.0],
    "right": [0.5, 0.0, 1.0, 1.0],
}
SIDE_TO_PARTICIPANT = {"left": "B", "right": "A"}

POSE_HEAD_LANDMARK_IDS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
}
POSE_HEAD_REFERENCE_IDS = [
    POSE_HEAD_LANDMARK_IDS["left_eye_inner"],
    POSE_HEAD_LANDMARK_IDS["left_eye"],
    POSE_HEAD_LANDMARK_IDS["left_eye_outer"],
    POSE_HEAD_LANDMARK_IDS["right_eye_inner"],
    POSE_HEAD_LANDMARK_IDS["right_eye"],
    POSE_HEAD_LANDMARK_IDS["right_eye_outer"],
    POSE_HEAD_LANDMARK_IDS["left_ear"],
    POSE_HEAD_LANDMARK_IDS["right_ear"],
]


def latest_turn_run() -> Path:
    runs = sorted(p for p in TURN_RUNS.glob("turn_aligned_video_*") if (p / "metadata" / "labeled_turns_checked.csv").exists())
    if not runs:
        raise SystemExit(f"No completed turn-aligned video run found under {TURN_RUNS}")
    return runs[-1]


def read_source_run(source_run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    video_path = source_run / "metadata" / "video_metadata_checked.csv"
    turns_path = source_run / "metadata" / "labeled_turns_checked.csv"
    missing = [str(path) for path in [video_path, turns_path] if not path.exists()]
    if missing:
        raise SystemExit("Missing expected source-run file(s):\n" + "\n".join(missing))
    return pd.read_csv(video_path), pd.read_csv(turns_path)


def make_question_windows(turns: pd.DataFrame) -> pd.DataFrame:
    df = turns.copy()
    for col in ["start_time_sec", "end_time_sec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["valid_interval"] = df["end_time_sec"] > df["start_time_sec"]
    rows = []
    group_cols = ["dyad_id", "session_id", "video_id", "condition", "question_id"]
    for keys, group in df.groupby(group_cols, dropna=False):
        valid = group[group["valid_interval"]]
        if valid.empty:
            continue
        start = float(valid["start_time_sec"].min())
        end = float(valid["end_time_sec"].max())
        rows.append({
            "dyad_id": keys[0],
            "session_id": keys[1],
            "video_id": keys[2],
            "condition": keys[3],
            "question_id": keys[4],
            "question_start_sec": start,
            "question_end_sec": end,
            "question_duration_sec": end - start,
            "n_turns_total": int(len(group)),
            "n_turns_valid": int(len(valid)),
            "n_turns_invalid": int((~group["valid_interval"]).sum()),
        })
    return pd.DataFrame(rows).sort_values(["video_id", "question_id"]).reset_index(drop=True)


def import_face_mesh_solution() -> Any | None:
    try:
        import mediapipe as mp
        return mp.solutions.face_mesh
    except Exception:
        try:
            import importlib
            return importlib.import_module("mediapipe.python.solutions.face_mesh")
        except Exception:
            return None


def side_feature_row(
    base: dict[str, Any],
    side: str,
    frame_index: int,
    fps: float,
    frame_features: dict[str, Any],
) -> dict[str, Any]:
    return {
        **base,
        "side": side,
        "frame_index": frame_index,
        "timestamp_sec": frame_index / fps if fps > 0 else np.nan,
        **frame_features,
    }


def pose_landmark_xy(
    pose_landmarks: Any,
    landmark_id: int,
    crop_w: int,
    crop_h: int,
    min_visibility: float,
) -> tuple[float, float] | None:
    if pose_landmarks is None or landmark_id >= len(pose_landmarks.landmark):
        return None
    landmark = pose_landmarks.landmark[landmark_id]
    visibility = float(getattr(landmark, "visibility", 1.0))
    presence = float(getattr(landmark, "presence", 1.0))
    if visibility < min_visibility or presence < min_visibility:
        return None
    return landmark.x * crop_w, landmark.y * crop_h


def pose_head_yaw_proxy(
    pose_landmarks: Any,
    crop_w: int,
    crop_h: int,
    min_visibility: float,
) -> float:
    nose = pose_landmark_xy(
        pose_landmarks,
        POSE_HEAD_LANDMARK_IDS["nose"],
        crop_w,
        crop_h,
        min_visibility,
    )
    if nose is None:
        return np.nan
    references = [
        pose_landmark_xy(pose_landmarks, landmark_id, crop_w, crop_h, min_visibility)
        for landmark_id in POSE_HEAD_REFERENCE_IDS
    ]
    references = [point for point in references if point is not None]
    if len(references) < 2:
        return np.nan
    xs = np.array([point[0] for point in references], dtype=float)
    span = float(xs.max() - xs.min())
    if not np.isfinite(span) or span < 2.0:
        return np.nan
    center_x = float((xs.max() + xs.min()) / 2.0)
    return float((nose[0] - center_x) / span)


def classify_look_target(partner_orientation_proxy: float, threshold: float) -> str:
    if not np.isfinite(partner_orientation_proxy):
        return "unclear"
    if partner_orientation_proxy > threshold:
        return "partner"
    if partner_orientation_proxy < -threshold:
        return "away"
    return "front_screen"


def look_indicator(target: str, expected: str) -> float:
    if target == "unclear":
        return np.nan
    return float(target == expected)


def process_crop(
    crop_bgr: np.ndarray,
    x0: int,
    y0: int,
    frame_index: int,
    fps: float,
    previous_wrist: tuple[float, float, float] | None,
    pose_model: Any,
    hands_model: Any,
    face_model: Any | None,
    side: str,
    partner_threshold: float,
    head_landmark_min_visibility: float,
) -> tuple[dict[str, Any], tuple[float, float, float] | None]:
    import cv2

    crop_h, crop_w = crop_bgr.shape[:2]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pose_points = {}
    pose_detected = False
    pose_head_yaw = np.nan
    hand_detected = False
    wrist_speed = np.nan
    wrist_xy: tuple[float, float] | None = None

    if pose_model is not None:
        pose_result = pose_model.process(rgb)
        if pose_result.pose_landmarks:
            pose_detected = True
            pose_head_yaw = pose_head_yaw_proxy(
                pose_result.pose_landmarks,
                crop_w,
                crop_h,
                head_landmark_min_visibility,
            )
            for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
                pose_points[idx] = (x0 + landmark.x * crop_w, y0 + landmark.y * crop_h)

    if hands_model is not None:
        hand_result = hands_model.process(rgb)
        hands = hand_result.multi_hand_landmarks or []
        if hands:
            hand_detected = True
            wrist = hands[0].landmark[0]
            wrist_xy = (x0 + wrist.x * crop_w, y0 + wrist.y * crop_h)
            now_sec = frame_index / fps if fps > 0 else np.nan
            if previous_wrist is not None and np.isfinite(now_sec):
                dt = now_sec - previous_wrist[2]
                if dt > 0:
                    wrist_speed = math.hypot(wrist_xy[0] - previous_wrist[0], wrist_xy[1] - previous_wrist[1]) / dt
            if np.isfinite(now_sec):
                previous_wrist = (wrist_xy[0], wrist_xy[1], now_sec)

    hand_to_face = np.nan
    hand_to_shoulder = np.nan
    if wrist_xy is not None and pose_points:
        nose = pose_points.get(0)
        if nose is not None:
            hand_to_face = math.hypot(wrist_xy[0] - nose[0], wrist_xy[1] - nose[1])
        shoulders = [pose_points.get(11), pose_points.get(12)]
        shoulders = [pt for pt in shoulders if pt is not None]
        if shoulders:
            hand_to_shoulder = min(math.hypot(wrist_xy[0] - pt[0], wrist_xy[1] - pt[1]) for pt in shoulders)

    face_detected = np.nan
    face_head_yaw = np.nan
    head_yaw_proxy = pose_head_yaw
    orientation_source = "pose" if np.isfinite(pose_head_yaw) else ""
    partner_orientation_proxy = np.nan
    if face_model is not None:
        face_detected = 0.0
        face_result = face_model.process(rgb)
        faces = face_result.multi_face_landmarks or []
        if faces:
            face_detected = 1.0
            face_features = extract_face_features(faces[0], crop_w, crop_h)
            face_head_yaw = face_features.get("head_yaw_proxy", np.nan)
            if np.isfinite(face_head_yaw):
                head_yaw_proxy = face_head_yaw
                orientation_source = "facemesh"
    if np.isfinite(head_yaw_proxy):
        partner_orientation_proxy = head_yaw_proxy if side == "left" else -head_yaw_proxy
    look_target = classify_look_target(partner_orientation_proxy, partner_threshold)

    return {
        "pose_detected": float(pose_detected),
        "hand_detected": float(hand_detected),
        "face_detected": face_detected,
        "head_orientation_detected": float(look_target != "unclear"),
        "orientation_source": orientation_source,
        "look_target_proxy": look_target,
        "wrist_speed_px_s": wrist_speed,
        "hand_to_face_distance_px": hand_to_face,
        "hand_to_shoulder_distance_px": hand_to_shoulder,
        "pose_head_yaw_proxy": pose_head_yaw,
        "face_head_yaw_proxy": face_head_yaw,
        "head_yaw_proxy": head_yaw_proxy,
        "partner_orientation_proxy": partner_orientation_proxy,
        "looking_partner_proxy": look_indicator(look_target, "partner"),
        "looking_front_screen_proxy": look_indicator(look_target, "front_screen"),
        "looking_away_proxy": look_indicator(look_target, "away"),
    }, previous_wrist


def process_video(
    video_meta: pd.Series,
    windows: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    import cv2

    cv2_module, pose_solution, hands_solution = import_video_stack()
    del cv2_module
    face_solution = import_face_mesh_solution() if args.face_orientation else None

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
    hands_model = hands_solution.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=float(args.hands_min_detection_confidence),
    )
    face_model = None
    if face_solution is not None:
        face_model = face_solution.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False)

    fps = float(video_meta["fps"])
    rows = []
    try:
        for _, window in windows.iterrows():
            frame_indices = sample_frame_indices(
                float(window["question_start_sec"]),
                float(window["question_end_sec"]),
                fps,
                float(args.sample_fps),
                args.max_frames_per_question,
            )
            previous = {"left": None, "right": None}
            base = window.to_dict()
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                height, width = frame_bgr.shape[:2]
                for side, roi in SIDE_ROIS.items():
                    x0, y0, x1, y1 = roi_to_pixels(roi, width, height)
                    crop = frame_bgr[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    features, previous[side] = process_crop(
                        crop,
                        x0,
                        y0,
                        frame_index,
                        fps,
                        previous[side],
                        pose_model,
                        hands_model,
                        face_model,
                        side,
                        float(args.partner_orientation_threshold),
                        float(args.head_landmark_min_visibility),
                    )
                    features["gesture_active"] = float(
                        np.isfinite(features["wrist_speed_px_s"])
                        and features["wrist_speed_px_s"] > float(args.motion_speed_threshold_px_s)
                    )
                    rows.append(side_feature_row(base, side, frame_index, fps, features))
    finally:
        cap.release()
        pose_model.close()
        hands_model.close()
        if face_model is not None:
            face_model.close()
    return pd.DataFrame(rows)


def aggregate_side(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dyad_id", "session_id", "video_id", "condition", "question_id", "side"]
    for keys, group in frame_df.groupby(group_cols, dropna=False):
        wrist_speed = pd.to_numeric(group["wrist_speed_px_s"], errors="coerce").to_numpy(dtype=float)
        orientation_detected = pd.to_numeric(group.get("head_orientation_detected", np.nan), errors="coerce")
        status = "PASS"
        notes = []
        hand_rate = float(group["hand_detected"].mean()) if len(group) else 0.0
        if hand_rate < 0.20:
            status = "WARNING"
            notes.append("Low hand detection rate.")
        orientation_rate = finite_mean(orientation_detected.to_numpy(dtype=float))
        if np.isfinite(orientation_rate) and orientation_rate < 0.50:
            status = "WARNING"
            notes.append("Low head-orientation detection rate.")
        rows.append({
            "dyad_id": keys[0],
            "session_id": keys[1],
            "video_id": keys[2],
            "condition": keys[3],
            "question_id": keys[4],
            "side": keys[5],
            "question_start_sec": float(group["question_start_sec"].iloc[0]),
            "question_end_sec": float(group["question_end_sec"].iloc[0]),
            "question_duration_sec": float(group["question_duration_sec"].iloc[0]),
            "n_sampled_frames": int(len(group)),
            "pose_detection_rate": float(group["pose_detected"].mean()) if len(group) else 0.0,
            "hand_detection_rate": hand_rate,
            "face_detection_rate": float(group["face_detected"].mean()) if "face_detected" in group else np.nan,
            "head_orientation_detection_rate": orientation_rate,
            "gesture_activity_rate": float(group["gesture_active"].mean()) if len(group) else np.nan,
            "mean_wrist_speed_px_s": finite_mean(wrist_speed),
            "max_wrist_speed_px_s": finite_max(wrist_speed),
            "mean_hand_to_face_distance_px": finite_mean(group["hand_to_face_distance_px"].to_numpy(dtype=float)),
            "mean_hand_to_shoulder_distance_px": finite_mean(group["hand_to_shoulder_distance_px"].to_numpy(dtype=float)),
            "mean_head_yaw_proxy": finite_mean(group["head_yaw_proxy"].to_numpy(dtype=float)),
            "mean_partner_orientation_proxy": finite_mean(group["partner_orientation_proxy"].to_numpy(dtype=float)),
            "looking_partner_proxy_rate": finite_mean(group["looking_partner_proxy"].to_numpy(dtype=float)),
            "looking_front_screen_proxy_rate": finite_mean(group["looking_front_screen_proxy"].to_numpy(dtype=float)),
            "looking_away_proxy_rate": finite_mean(group["looking_away_proxy"].to_numpy(dtype=float)),
            "qc_status": status,
            "qc_notes": " ".join(notes),
        })
    return pd.DataFrame(rows).sort_values(["video_id", "question_id", "side"]).reset_index(drop=True)


def aggregate_interaction(frame_df: pd.DataFrame, side_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dyad_id", "session_id", "video_id", "condition", "question_id"]
    for keys, group in frame_df.groupby(group_cols, dropna=False):
        row = {
            "dyad_id": keys[0],
            "session_id": keys[1],
            "video_id": keys[2],
            "condition": keys[3],
            "question_id": keys[4],
        }
        side_group = side_df[
            (side_df["video_id"] == keys[2])
            & (side_df["question_id"].astype(str) == str(keys[4]))
        ]
        for feature in [
            "hand_detection_rate",
            "gesture_activity_rate",
            "mean_wrist_speed_px_s",
            "head_orientation_detection_rate",
            "looking_partner_proxy_rate",
            "looking_front_screen_proxy_rate",
            "looking_away_proxy_rate",
        ]:
            wide = side_group.pivot_table(index="question_id", columns="side", values=feature, aggfunc="mean")
            left = wide["left"].iloc[0] if "left" in wide else np.nan
            right = wide["right"].iloc[0] if "right" in wide else np.nan
            row[f"left_{feature}"] = left
            row[f"right_{feature}"] = right
            row[f"left_minus_right_{feature}"] = left - right if np.isfinite(left) and np.isfinite(right) else np.nan
        look_targets = group.pivot_table(index="frame_index", columns="side", values="look_target_proxy", aggfunc="first")
        if {"left", "right"}.issubset(look_targets.columns):
            paired_targets = look_targets[["left", "right"]].dropna()
            paired_targets = paired_targets[
                (paired_targets["left"] != "unclear")
                & (paired_targets["right"] != "unclear")
            ]
            row["n_orientation_pair_frames"] = int(len(paired_targets))
            if len(paired_targets):
                row["mutual_partner_orientation_rate"] = float(
                    ((paired_targets["left"] == "partner") & (paired_targets["right"] == "partner")).mean()
                )
                row["both_front_screen_orientation_rate"] = float(
                    ((paired_targets["left"] == "front_screen") & (paired_targets["right"] == "front_screen")).mean()
                )
                row["any_partner_orientation_rate"] = float(
                    ((paired_targets["left"] == "partner") | (paired_targets["right"] == "partner")).mean()
                )
            else:
                row["mutual_partner_orientation_rate"] = np.nan
                row["both_front_screen_orientation_rate"] = np.nan
                row["any_partner_orientation_rate"] = np.nan
        else:
            row["n_orientation_pair_frames"] = 0
            row["mutual_partner_orientation_rate"] = np.nan
            row["both_front_screen_orientation_rate"] = np.nan
            row["any_partner_orientation_rate"] = np.nan
        signal = group.pivot_table(index="frame_index", columns="side", values="wrist_speed_px_s", aggfunc="mean")
        if {"left", "right"}.issubset(signal.columns):
            paired = signal[["left", "right"]].replace([np.inf, -np.inf], np.nan).dropna()
            row["wrist_speed_synchrony_r"] = float(paired["left"].corr(paired["right"])) if len(paired) >= 5 else np.nan
            row["n_synchrony_frames"] = int(len(paired))
        else:
            row["wrist_speed_synchrony_r"] = np.nan
            row["n_synchrony_frames"] = 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["video_id", "question_id"]).reset_index(drop=True)


def aggregate_session(side_df: pd.DataFrame, interaction_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    side_summary = (
        side_df.groupby(["dyad_id", "session_id", "video_id", "condition", "side"], dropna=False)
        .agg(
            n_questions=("question_id", "nunique"),
            mean_pose_detection_rate=("pose_detection_rate", "mean"),
            mean_hand_detection_rate=("hand_detection_rate", "mean"),
            mean_face_detection_rate=("face_detection_rate", "mean"),
            mean_head_orientation_detection_rate=("head_orientation_detection_rate", "mean"),
            mean_gesture_activity_rate=("gesture_activity_rate", "mean"),
            mean_wrist_speed_px_s=("mean_wrist_speed_px_s", "mean"),
            mean_looking_partner_proxy_rate=("looking_partner_proxy_rate", "mean"),
            mean_looking_front_screen_proxy_rate=("looking_front_screen_proxy_rate", "mean"),
            mean_looking_away_proxy_rate=("looking_away_proxy_rate", "mean"),
        )
        .reset_index()
    )
    interaction_summary = (
        interaction_df.groupby(["dyad_id", "session_id", "video_id", "condition"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    return side_summary, interaction_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=None, help="Completed turn-aligned run to reuse for metadata/question windows.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames-per-question", type=int, default=None)
    parser.add_argument("--motion-speed-threshold-px-s", type=float, default=40.0)
    parser.add_argument(
        "--face-orientation",
        action="store_true",
        help="Also use FaceMesh when available; pose head landmarks are used as the default orientation source.",
    )
    parser.add_argument(
        "--partner-orientation-threshold",
        type=float,
        default=0.03,
        help="Head-orientation threshold: toward partner above this, away below the negative threshold, otherwise front/screen.",
    )
    parser.add_argument(
        "--head-landmark-min-visibility",
        type=float,
        default=0.20,
        help="Minimum MediaPipe Pose visibility/presence for nose, eye, and ear landmarks used in head-orientation fallback.",
    )
    parser.add_argument("--pose-model-complexity", type=int, default=2)
    parser.add_argument("--pose-min-detection-confidence", type=float, default=0.50)
    parser.add_argument("--hands-min-detection-confidence", type=float, default=0.50)
    parser.add_argument("--limit-videos", type=int, default=None, help="Debug/smoke-test limit.")
    parser.add_argument("--max-questions-per-video", type=int, default=None, help="Debug/smoke-test limit.")
    parser.add_argument("--dry-run", action="store_true", help="Write question windows and manifest only.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first video-processing error.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    source_run = args.source_run or latest_turn_run()
    run_id = args.run_id or f"side_question_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / run_id
    for subdir in ["metadata", "intermediate", "results", "qc"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    video_meta, turns = read_source_run(source_run)
    question_windows = make_question_windows(turns)
    if args.limit_videos is not None:
        keep_videos = video_meta["video_id"].head(args.limit_videos).tolist()
        video_meta = video_meta[video_meta["video_id"].isin(keep_videos)].copy()
        question_windows = question_windows[question_windows["video_id"].isin(keep_videos)].copy()
    if args.max_questions_per_video is not None:
        question_windows = (
            question_windows.sort_values(["video_id", "question_id"])
            .groupby("video_id", dropna=False)
            .head(args.max_questions_per_video)
            .reset_index(drop=True)
        )

    atomic_write_csv(run_dir / "metadata" / "question_windows.csv", question_windows)
    config_used = json_ready(vars(args).copy())
    config_used["source_run"] = str(source_run)
    config_used["side_rois"] = SIDE_ROIS
    config_used["note"] = "Speaker-independent left/right question-level movement and head-orientation video features."
    atomic_write_json(run_dir / "config_used.json", config_used)
    if args.dry_run:
        write_manifest(run_dir, source_run, [run_dir / "metadata" / "question_windows.csv"], config_used)
        print(f"Dry run complete: {run_dir}")
        return 0

    frame_parts = []
    failed_rows = []
    for _, meta in video_meta.iterrows():
        windows = question_windows[question_windows["video_id"] == meta["video_id"]].copy()
        if windows.empty:
            continue
        print(f"Processing {meta['video_id']} ({len(windows)} question windows)")
        try:
            frame_parts.append(process_video(meta, windows, args))
        except Exception as exc:
            failed_rows.append({
                "video_id": meta.get("video_id", ""),
                "video_path": meta.get("video_path", ""),
                "stage": "main_video_analysis",
                "error": str(exc),
                "suggested_fix": "Check video readability and run inside the same Python environment that passed check_video_dependencies.py.",
            })
            print(f"ERROR processing {meta.get('video_id', '')}: {exc}")
            if args.fail_fast:
                break
    frame_df = pd.concat(frame_parts, ignore_index=True) if frame_parts else pd.DataFrame()
    side_df = aggregate_side(frame_df) if not frame_df.empty else pd.DataFrame()
    interaction_df = aggregate_interaction(frame_df, side_df) if not frame_df.empty else pd.DataFrame()
    session_side, session_interaction = aggregate_session(side_df, interaction_df) if not side_df.empty else (pd.DataFrame(), pd.DataFrame())

    outputs = [
        run_dir / "intermediate" / "per_frame_side_features.csv",
        run_dir / "results" / "per_question_side_features.csv",
        run_dir / "results" / "per_question_interaction_features.csv",
        run_dir / "results" / "per_session_side_summary.csv",
        run_dir / "results" / "per_session_interaction_summary.csv",
        run_dir / "qc" / "failed_videos.csv",
    ]
    atomic_write_csv(outputs[0], frame_df)
    atomic_write_csv(outputs[1], side_df)
    atomic_write_csv(outputs[2], interaction_df)
    atomic_write_csv(outputs[3], session_side)
    atomic_write_csv(outputs[4], session_interaction)
    failed_columns = ["video_id", "video_path", "stage", "error", "suggested_fix"]
    atomic_write_csv(outputs[5], pd.DataFrame(failed_rows, columns=failed_columns))
    write_manifest(run_dir, source_run, outputs + [run_dir / "metadata" / "question_windows.csv"], config_used)
    print(f"Question-level side video run complete: {run_dir}")
    return 1 if failed_rows and not frame_parts else 0


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


if __name__ == "__main__":
    raise SystemExit(main())

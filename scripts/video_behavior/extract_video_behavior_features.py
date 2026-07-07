#!/usr/bin/env python3
"""
extract_video_behavior_features.py
==================================
Video-behavior feature extraction for the CVS conversation study.

The script extracts auditable movement and facial-expression proxies from
private discussion videos. It does not save images, frames, or cropped faces.

For each video:
  1. sample frames at a fixed rate,
  2. compute whole-frame optical-flow motion,
  3. detect up to two faces with MediaPipe Face Mesh,
  4. assign detected faces to stable left/right visual tracks,
  5. derive facial-expression and facial-motion proxies from landmarks,
  6. write frame-level and session-level CSV outputs.

Facial-expression features are landmark proxies, not emotion classifications:
smile aperture, mouth opening, eye opening, brow raise, and head-pose proxies.

Dependencies:
  opencv-python, mediapipe, numpy, pandas, tqdm

Usage:
  python scripts/video_behavior/extract_video_behavior_features.py
  python scripts/video_behavior/extract_video_behavior_features.py --video-dir data/raw/videos
  python scripts/video_behavior/extract_video_behavior_features.py --limit 2 --sample-fps 1
"""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
VIDEO_DIR = Path(os.environ.get("CVS_VIDEO_DIR", PROJECT / "01_pipeline" / "video"))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
DYAD_RE = re.compile(r"(dyad\d+_\d+)", re.IGNORECASE)
CONDITION_RE = re.compile(r"(piper|cloudy)", re.IGNORECASE)

FACE_LANDMARKS = {
    "left_mouth": 61,
    "right_mouth": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "left_brow": 105,
    "right_brow": 334,
    "nose_tip": 1,
    "face_left": 234,
    "face_right": 454,
    "forehead": 10,
    "chin": 152,
}


def parse_video_id(path: Path) -> dict[str, str]:
    """Infer dyad and condition from path/filename when possible."""
    text = str(path)
    dyad = DYAD_RE.search(text)
    condition = CONDITION_RE.search(text)
    return {
        "video_id": path.stem,
        "dyad_id": dyad.group(1).lower() if dyad else path.stem,
        "condition": condition.group(1).lower() if condition else "unknown",
    }


def iter_videos(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        return []
    return sorted(
        p for p in video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def point(landmarks: Any, idx: int, width: int, height: int) -> np.ndarray:
    lm = landmarks.landmark[idx]
    return np.array([lm.x * width, lm.y * height], dtype=float)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def face_bbox(landmarks: Any, width: int, height: int) -> tuple[float, float, float, float]:
    xs = np.array([lm.x * width for lm in landmarks.landmark], dtype=float)
    ys = np.array([lm.y * height for lm in landmarks.landmark], dtype=float)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def extract_face_features(landmarks: Any, width: int, height: int) -> dict[str, float]:
    pts = {name: point(landmarks, idx, width, height) for name, idx in FACE_LANDMARKS.items()}
    x0, y0, x1, y1 = face_bbox(landmarks, width, height)
    face_width = max(distance(pts["face_left"], pts["face_right"]), x1 - x0, 1.0)
    face_height = max(distance(pts["forehead"], pts["chin"]), y1 - y0, 1.0)
    eye_mid_y = float(np.mean([
        pts["left_eye_top"][1],
        pts["left_eye_bottom"][1],
        pts["right_eye_top"][1],
        pts["right_eye_bottom"][1],
    ]))
    brow_y = float(np.mean([pts["left_brow"][1], pts["right_brow"][1]]))
    face_center = (pts["face_left"] + pts["face_right"]) / 2.0

    return {
        "face_center_x": float((x0 + x1) / 2.0),
        "face_center_y": float((y0 + y1) / 2.0),
        "face_width_px": float(face_width),
        "face_height_px": float(face_height),
        "smile_width": safe_div(distance(pts["left_mouth"], pts["right_mouth"]), face_width),
        "mouth_open": safe_div(distance(pts["upper_lip"], pts["lower_lip"]), face_width),
        "left_eye_open": safe_div(distance(pts["left_eye_top"], pts["left_eye_bottom"]), face_width),
        "right_eye_open": safe_div(distance(pts["right_eye_top"], pts["right_eye_bottom"]), face_width),
        "eye_open": safe_div(
            distance(pts["left_eye_top"], pts["left_eye_bottom"])
            + distance(pts["right_eye_top"], pts["right_eye_bottom"]),
            2.0 * face_width,
        ),
        "brow_raise": safe_div(eye_mid_y - brow_y, face_width),
        "head_yaw_proxy": safe_div(pts["nose_tip"][0] - face_center[0], face_width),
        "head_pitch_proxy": safe_div(pts["nose_tip"][1] - face_center[1], face_height),
    }


def assign_tracks(face_rows: list[dict[str, float]]) -> list[tuple[str, dict[str, float]]]:
    """Assign detected faces to left/right visual tracks by horizontal position."""
    ordered = sorted(face_rows, key=lambda row: row["face_center_x"])
    labels = ["left", "right"]
    return [(labels[i], row) for i, row in enumerate(ordered[:2])]


def dense_motion(prev_gray: np.ndarray | None, gray: np.ndarray) -> float:
    if prev_gray is None:
        return np.nan
    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.nanmean(mag))


def add_face_motion(rows: list[dict[str, Any]]) -> None:
    previous: dict[str, dict[str, Any]] = {}
    for row in rows:
        track = row.get("face_track")
        if not track:
            continue
        prev = previous.get(track)
        if prev is None:
            row["face_motion_px"] = np.nan
        else:
            dt = row["time_s"] - prev["time_s"]
            displacement = math.hypot(
                row["face_center_x"] - prev["face_center_x"],
                row["face_center_y"] - prev["face_center_y"],
            )
            row["face_motion_px"] = safe_div(displacement, dt)
        previous[track] = row


def summarize_session(frame_df: pd.DataFrame, video_meta: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        **video_meta,
        "duration_s": frame_df["time_s"].max() if not frame_df.empty else np.nan,
        "sampled_frames": int(frame_df["frame_index"].nunique()) if not frame_df.empty else 0,
        "face_observations": int(frame_df["face_track"].notna().sum()) if "face_track" in frame_df else 0,
        "mean_frame_motion": frame_df["frame_motion"].mean() if "frame_motion" in frame_df else np.nan,
        "sd_frame_motion": frame_df["frame_motion"].std() if "frame_motion" in frame_df else np.nan,
    }
    if frame_df.empty or "face_track" not in frame_df:
        return summary

    for track in ["left", "right"]:
        sub = frame_df[frame_df["face_track"] == track]
        prefix = f"{track}_face"
        summary[f"{prefix}_frames"] = int(len(sub))
        summary[f"{prefix}_detection_rate"] = safe_div(len(sub), frame_df["frame_index"].nunique())
        for col in [
            "face_motion_px",
            "smile_width",
            "mouth_open",
            "eye_open",
            "brow_raise",
            "head_yaw_proxy",
            "head_pitch_proxy",
        ]:
            summary[f"{prefix}_{col}_mean"] = sub[col].mean() if col in sub else np.nan
            summary[f"{prefix}_{col}_sd"] = sub[col].std() if col in sub else np.nan

    left = frame_df[frame_df["face_track"] == "left"].set_index("frame_index")
    right = frame_df[frame_df["face_track"] == "right"].set_index("frame_index")
    common = left.index.intersection(right.index)
    summary["both_faces_frames"] = int(len(common))
    summary["both_faces_rate"] = safe_div(len(common), frame_df["frame_index"].nunique())
    if len(common) >= 3:
        for col in ["face_motion_px", "smile_width", "mouth_open", "eye_open", "brow_raise"]:
            summary[f"left_right_{col}_corr"] = left.loc[common, col].corr(right.loc[common, col])
    return summary


def process_video(
    video_path: Path,
    sample_fps: float,
    max_faces: int,
    min_detection_confidence: float,
    max_frames: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(native_fps / sample_fps)))
    meta = {
        **parse_video_id(video_path),
        "video_path": str(video_path),
        "native_fps": float(native_fps),
        "total_frames": total_frames,
        "sample_fps": sample_fps,
    }

    rows: list[dict[str, Any]] = []
    prev_gray = None
    sampled = 0

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
    )

    try:
        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue
            sampled += 1
            if max_frames is not None and sampled > max_frames:
                break

            height, width = frame_bgr.shape[:2]
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_motion = dense_motion(prev_gray, gray)
            prev_gray = gray

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            time_s = float(frame_index / native_fps)
            base = {
                **meta,
                "frame_index": frame_index,
                "time_s": time_s,
                "frame_motion": frame_motion,
                "detected_faces": 0,
                "face_track": None,
            }

            faces = result.multi_face_landmarks or []
            face_features = [extract_face_features(face, width, height) for face in faces]
            if not face_features:
                rows.append(base)
            else:
                for track, features in assign_tracks(face_features):
                    rows.append({
                        **base,
                        "detected_faces": len(face_features),
                        "face_track": track,
                        **features,
                    })
            frame_index += 1
    finally:
        face_mesh.close()
        cap.release()

    add_face_motion(rows)
    frame_df = pd.DataFrame(rows)
    summary = summarize_session(frame_df, meta)
    return frame_df, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract movement and facial-expression proxies from private CVS videos."
    )
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--frames-out", type=Path, default=DATA / "video_behavior_frames.csv")
    parser.add_argument("--summary-out", type=Path, default=DATA / "video_behavior_dyad.csv")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-faces", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos.")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug limit per video.")
    return parser


def run(args: argparse.Namespace) -> None:
    videos = iter_videos(args.video_dir)
    if args.limit is not None:
        videos = videos[:args.limit]
    if not videos:
        raise FileNotFoundError(
            f"No videos found under {args.video_dir}. "
            "Set CVS_VIDEO_DIR or pass --video-dir with private video files."
        )

    all_frames = []
    summaries = []
    for video_path in tqdm(videos, desc="video files"):
        frame_df, summary = process_video(
            video_path=video_path,
            sample_fps=args.sample_fps,
            max_faces=args.max_faces,
            min_detection_confidence=args.min_detection_confidence,
            max_frames=args.max_frames,
        )
        all_frames.append(frame_df)
        summaries.append(summary)

    args.frames_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_frames, ignore_index=True).to_csv(args.frames_out, index=False)
    pd.DataFrame(summaries).to_csv(args.summary_out, index=False)
    print(f"Wrote frame-level video features: {args.frames_out}")
    print(f"Wrote session-level video features: {args.summary_out}")


if __name__ == "__main__":
    run(build_arg_parser().parse_args())

#!/usr/bin/env python3
"""
Workflow-ready video behavior pipeline for the CVS conversation study.

This runner is designed for private lab video data. It never modifies raw
videos. Each execution writes a timestamped run directory containing config,
logs, metadata, intermediate outputs, final analysis tables, QC artifacts, and
manifests.

Command examples:
  python scripts/video_behavior/run_video_analysis.py --config config/video_analysis.example.yml
  python scripts/video_behavior/run_video_analysis.py --config config/video_analysis.example.yml --video data/raw/videos/example.mp4
  python scripts/video_behavior/run_video_analysis.py --config config/video_analysis.example.yml --dry-run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
DEFAULT_VIDEO_DIR = Path(os.environ.get("CVS_VIDEO_DIR", PROJECT / "01_pipeline" / "video"))
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("CVS_VIDEO_OUTPUTS", PROJECT / "data" / "interim" / "video_behavior" / "runs")
)

ACCEPTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
RUN_STATUS_FILE = "RUN_STATUS.json"

POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]

HAND_LANDMARK_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

REQUIRED_FINAL_COLUMNS = {
    "facial": [
        "video_id", "participant_id", "frame_index", "timestamp_ms",
        "feature_name", "value", "confidence", "source",
    ],
    "pose": [
        "video_id", "participant_id", "frame_index", "timestamp_ms",
        "keypoint_index", "keypoint_name", "x_px", "y_px", "z",
        "visibility", "confidence", "source",
    ],
    "hands": [
        "video_id", "participant_id", "frame_index", "timestamp_ms",
        "hand_label", "keypoint_index", "keypoint_name", "x_px", "y_px",
        "z", "confidence", "source",
    ],
    "gesture": [
        "video_id", "participant_id", "frame_index", "timestamp_ms",
        "feature_name", "value", "unit", "confidence", "source",
    ],
}


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "video_dir": str(DEFAULT_VIDEO_DIR),
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "accepted_video_extensions": sorted(ACCEPTED_VIDEO_EXTENSIONS),
    },
    "run": {
        "run_id": None,
        "overwrite": False,
        "expected_fps": None,
        "fps_tolerance": 0.05,
        "continue_on_warning": True,
        "fail_fast": False,
        "allow_empty_optional_outputs": True,
    },
    "sampling": {
        "sample_fps": 2.0,
        "max_frames_per_video": None,
    },
    "participants": [
        {"participant_id": "left", "roi": [0.0, 0.0, 0.5, 1.0]},
        {"participant_id": "right", "roi": [0.5, 0.0, 1.0, 1.0]},
    ],
    "openface": {
        "enabled": False,
        "required": False,
        "executable": "FeatureExtraction",
        "input_csv_dir": None,
        "confidence_min": 0.80,
        "success_required": True,
    },
    "mediapipe": {
        "enabled": True,
        "pose_enabled": True,
        "hands_enabled": True,
        "pose_model_complexity": 2,
        "pose_min_detection_confidence": 0.50,
        "pose_min_tracking_confidence": 0.50,
        "hands_max_num_hands": 2,
        "hands_min_detection_confidence": 0.50,
        "hands_min_tracking_confidence": 0.50,
    },
    "features": {
        "motion_speed_threshold_px_s": 40.0,
        "hold_speed_threshold_px_s": 10.0,
        "min_stroke_duration_s": 0.20,
        "max_post_stroke_hold_s": 2.0,
    },
    "qc": {
        "save_overlay_frames": True,
        "max_overlay_frames": 8,
        "save_plots": True,
        "pose_detection_rate_warning": 0.50,
        "hand_detection_rate_warning": 0.20,
        "coordinate_margin_px": 25,
    },
}


class PipelineError(Exception):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        likely_cause: str,
        suggested_fix: str,
        affected_file: str | None = None,
        can_continue: bool = True,
        level: str = "ERROR",
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.likely_cause = likely_cause
        self.suggested_fix = suggested_fix
        self.affected_file = affected_file
        self.can_continue = can_continue
        self.level = level

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "stage": self.stage,
            "likely_cause": self.likely_cause,
            "suggested_fix": self.suggested_fix,
            "affected_file": self.affected_file,
            "can_continue": self.can_continue,
            "level": self.level,
        }


@dataclass
class RunDirs:
    root: Path
    config: Path
    logs: Path
    metadata: Path
    openface: Path
    mediapipe: Path
    qc: Path
    final: Path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise PipelineError(
            "PyYAML is required to read YAML config files.",
            stage="config_loading",
            likely_cause="The active environment does not include PyYAML.",
            suggested_fix="Install PyYAML or recreate the conda environment from environment.yml.",
            affected_file=str(path),
            can_continue=False,
            level="CRITICAL",
        ) from exc

    if not path.exists():
        raise PipelineError(
            "Config file does not exist.",
            stage="config_loading",
            likely_cause="The --config path is wrong or the file was moved.",
            suggested_fix="Pass a valid config path, for example config/video_analysis.example.yml.",
            affected_file=str(path),
            can_continue=False,
            level="CRITICAL",
        )
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise PipelineError(
            "Config root must be a mapping.",
            stage="config_loading",
            likely_cause="The YAML file contains a list or scalar at the top level.",
            suggested_fix="Use key/value YAML with sections such as project, run, and sampling.",
            affected_file=str(path),
            can_continue=False,
            level="CRITICAL",
        )
    return deep_merge(DEFAULT_CONFIG, loaded)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))


def validate_config(config: dict[str, Any]) -> None:
    sample_fps = config["sampling"].get("sample_fps")
    if not isinstance(sample_fps, (int, float)) or sample_fps <= 0:
        raise PipelineError(
            "sampling.sample_fps must be a positive number.",
            stage="config_validation",
            likely_cause="Invalid sampling rate in the config.",
            suggested_fix="Set sampling.sample_fps to a value such as 1, 2, or 5.",
            can_continue=False,
            level="CRITICAL",
        )
    participants = config.get("participants", [])
    if not participants:
        raise PipelineError(
            "At least one participant ROI is required.",
            stage="config_validation",
            likely_cause="participants is empty.",
            suggested_fix="Define left/right participant ROIs in normalized coordinates.",
            can_continue=False,
            level="CRITICAL",
        )
    for participant in participants:
        roi = participant.get("roi")
        if not isinstance(roi, list) or len(roi) != 4:
            raise PipelineError(
                "Each participant ROI must be [x0, y0, x1, y1].",
                stage="config_validation",
                likely_cause="A participant ROI is missing or malformed.",
                suggested_fix="Use normalized ROI coordinates between 0 and 1.",
                affected_file=str(participant),
                can_continue=False,
                level="CRITICAL",
            )
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise PipelineError(
                "Participant ROI values must be normalized and ordered.",
                stage="config_validation",
                likely_cause="ROI coordinates are outside [0, 1] or x0/x1 are reversed.",
                suggested_fix="Use [0.0, 0.0, 0.5, 1.0] for the left half as a template.",
                affected_file=str(participant),
                can_continue=False,
                level="CRITICAL",
            )


def build_run_dirs(output_root: Path, run_id: str, resume: Path | None) -> RunDirs:
    root = resume if resume else output_root / run_id
    return RunDirs(
        root=root,
        config=root / "config",
        logs=root / "logs",
        metadata=root / "metadata",
        openface=root / "openface",
        mediapipe=root / "mediapipe",
        qc=root / "qc",
        final=root / "final",
    )


def ensure_run_dirs(dirs: RunDirs, *, overwrite: bool, resume: bool) -> None:
    if dirs.root.exists() and not resume and not overwrite:
        raise PipelineError(
            "Run directory already exists.",
            stage="run_setup",
            likely_cause="The configured run_id was already used.",
            suggested_fix="Use a new run_id, enable resume, or choose a different output_root.",
            affected_file=str(dirs.root),
            can_continue=False,
            level="CRITICAL",
        )
    for path in [
        dirs.config, dirs.logs, dirs.metadata, dirs.openface, dirs.mediapipe,
        dirs.qc, dirs.final,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("video_behavior")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(stage)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_pipeline_error(logger: logging.Logger, error: PipelineError) -> None:
    extra = {"stage": error.stage}
    message = (
        f"{error}; likely cause: {error.likely_cause}; "
        f"suggested fix: {error.suggested_fix}"
    )
    if error.level == "CRITICAL":
        logger.critical(message, extra=extra)
    elif error.level == "WARNING":
        logger.warning(message, extra=extra)
    else:
        logger.error(message, extra=extra)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True))


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        df.to_csv(tmp, index=False)
    os.replace(tmp_path, path)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "role": role,
        "size_bytes": stat.st_size,
        "modified_time": stat.st_mtime,
        "sha256": sha256_file(path) if path.is_file() and stat.st_size > 0 else None,
    }


def get_git_info() -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT,
                text=True,
                capture_output=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def get_software_versions(config: dict[str, Any]) -> dict[str, Any]:
    packages = ["numpy", "pandas", "opencv-python", "mediapipe", "matplotlib", "PyYAML", "scipy"]
    versions: dict[str, Any] = {
        "python": sys.version,
        "git": get_git_info(),
        "packages": {},
        "openface": {},
    }
    for package in packages:
        try:
            versions["packages"][package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions["packages"][package] = None

    executable = config.get("openface", {}).get("executable")
    if executable:
        path = shutil.which(executable)
        versions["openface"]["executable"] = path
        if path:
            try:
                result = subprocess.run([path, "-version"], text=True, capture_output=True, timeout=10)
                versions["openface"]["version_output"] = (result.stdout or result.stderr).strip()
            except Exception as exc:
                versions["openface"]["version_output"] = f"unavailable: {exc}"
    return versions


def find_videos(config: dict[str, Any], explicit_video: Path | None) -> list[Path]:
    accepted = {ext.lower() for ext in config["project"].get("accepted_video_extensions", [])}
    if explicit_video:
        videos = [explicit_video]
    else:
        video_dir = Path(config["project"]["video_dir"])
        if not video_dir.exists():
            raise PipelineError(
                "Video directory does not exist.",
                stage="input_validation",
                likely_cause="CVS_VIDEO_DIR or project.video_dir points to the wrong place.",
                suggested_fix="Set CVS_VIDEO_DIR or project.video_dir to the private video folder.",
                affected_file=str(video_dir),
                can_continue=False,
                level="CRITICAL",
            )
        videos = sorted(p for p in video_dir.rglob("*") if p.is_file())

    valid = []
    for video in videos:
        if not video.exists():
            raise PipelineError(
                "Input video file does not exist.",
                stage="input_validation",
                likely_cause="The video path is wrong or the file was moved.",
                suggested_fix="Check the path or regenerate the input manifest.",
                affected_file=str(video),
                can_continue=True,
            )
        if video.suffix.lower() not in accepted:
            raise PipelineError(
                "Unsupported video format.",
                stage="input_validation",
                likely_cause=f"Extension {video.suffix} is not in the accepted format list.",
                suggested_fix="Convert the video to mp4/mov/m4v/avi/mkv or update accepted_video_extensions.",
                affected_file=str(video),
                can_continue=True,
                level="WARNING",
            )
        valid.append(video)
    if not valid:
        raise PipelineError(
            "No valid input videos found.",
            stage="input_validation",
            likely_cause="The input folder is empty or all files have unsupported extensions.",
            suggested_fix="Place private videos in the configured video directory.",
            can_continue=False,
            level="CRITICAL",
        )
    return valid


def video_id_from_path(path: Path) -> str:
    return path.stem.replace(" ", "_")


def inspect_video(video_path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise PipelineError(
            "OpenCV is required to inspect and process videos.",
            stage="dependency_check",
            likely_cause="opencv-python is not installed in the active environment.",
            suggested_fix="Install opencv-python or recreate the conda environment.",
            affected_file=str(video_path),
            can_continue=False,
            level="CRITICAL",
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError(
            "Could not open video.",
            stage="metadata_inspection",
            likely_cause="The file is corrupted, uses an unsupported codec, or permissions block reading.",
            suggested_fix="Try opening the file in a media player or re-export as H.264 mp4.",
            affected_file=str(video_path),
            can_continue=True,
        )
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise PipelineError(
            "Video metadata are invalid.",
            stage="metadata_inspection",
            likely_cause="The video header is incomplete or the codec is unreadable.",
            suggested_fix="Re-export the raw video and verify FPS/frame count before analysis.",
            affected_file=str(video_path),
            can_continue=True,
        )
    return {
        "video_id": video_id_from_path(video_path),
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": frame_count / fps,
        "extension": video_path.suffix.lower(),
    }


def expected_sample_indices(frame_count: int, fps: float, sample_fps: float, max_frames: int | None) -> list[int]:
    step = max(1, int(round(fps / sample_fps)))
    indices = list(range(0, frame_count, step))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def validate_metadata(meta: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = []
    expected_fps = config["run"].get("expected_fps")
    tolerance = float(config["run"].get("fps_tolerance", 0.05))
    if expected_fps is not None:
        rel_diff = abs(meta["fps"] - float(expected_fps)) / float(expected_fps)
        if rel_diff > tolerance:
            warnings.append({
                "stage": "metadata_validation",
                "message": "Video FPS differs from configured expected FPS.",
                "observed_fps": meta["fps"],
                "expected_fps": expected_fps,
                "relative_difference": rel_diff,
            })
    duration_from_frames = meta["frame_count"] / meta["fps"]
    if abs(duration_from_frames - meta["duration_s"]) > 0.01:
        warnings.append({
            "stage": "metadata_validation",
            "message": "Duration is inconsistent with frame_count / fps.",
            "duration_s": meta["duration_s"],
            "computed_duration_s": duration_from_frames,
        })
    return warnings


def import_mediapipe_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise PipelineError(
            "MediaPipe/OpenCV dependency is missing.",
            stage="dependency_check",
            likely_cause="mediapipe or opencv-python is not installed.",
            suggested_fix="Recreate the conda environment or disable the mediapipe stage in config.",
            can_continue=False,
            level="CRITICAL",
        ) from exc
    return cv2, mp


def roi_to_pixels(roi: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(round(roi[0] * width))
    y0 = int(round(roi[1] * height))
    x1 = int(round(roi[2] * width))
    y1 = int(round(roi[3] * height))
    return x0, y0, x1, y1


def process_mediapipe_video(
    video_path: Path,
    meta: dict[str, Any],
    config: dict[str, Any],
    dirs: RunDirs,
    logger: logging.Logger,
) -> dict[str, Path]:
    cv2, mp = import_mediapipe_dependencies()
    video_id = meta["video_id"]
    video_dir = dirs.mediapipe / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = dirs.qc / video_id
    overlay_dir = qc_dir / "sampled_overlay_frames"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    pose_rows: list[dict[str, Any]] = []
    hand_rows: list[dict[str, Any]] = []
    overlay_count = 0
    sample_indices = set(expected_sample_indices(
        meta["frame_count"],
        meta["fps"],
        float(config["sampling"]["sample_fps"]),
        config["sampling"].get("max_frames_per_video"),
    ))
    if not sample_indices:
        raise PipelineError(
            "No sampled frames were selected for processing.",
            stage="preprocessing",
            likely_cause="The video has zero frames or max_frames_per_video is set to 0.",
            suggested_fix="Check video metadata and sampling config.",
            affected_file=str(video_path),
            can_continue=True,
        )

    mediapipe_config = config["mediapipe"]
    pose_model = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=int(mediapipe_config["pose_model_complexity"]),
        enable_segmentation=False,
        min_detection_confidence=float(mediapipe_config["pose_min_detection_confidence"]),
    ) if mediapipe_config.get("pose_enabled", True) else None
    hands_model = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=int(mediapipe_config["hands_max_num_hands"]),
        min_detection_confidence=float(mediapipe_config["hands_min_detection_confidence"]),
    ) if mediapipe_config.get("hands_enabled", True) else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError(
            "Could not reopen video for MediaPipe processing.",
            stage="main_video_analysis",
            likely_cause="The video became unreadable after metadata inspection.",
            suggested_fix="Check filesystem permissions and video integrity.",
            affected_file=str(video_path),
            can_continue=True,
        )

    logger.info(f"Processing {video_id} with MediaPipe on {len(sample_indices)} sampled frames.", extra={"stage": "main_video_analysis"})
    try:
        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index not in sample_indices:
                frame_index += 1
                continue
            timestamp_ms = frame_index / meta["fps"] * 1000.0
            overlay = frame_bgr.copy()
            height, width = frame_bgr.shape[:2]

            for participant in config["participants"]:
                participant_id = participant["participant_id"]
                x0, y0, x1, y1 = roi_to_pixels(participant["roi"], width, height)
                crop = frame_bgr[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                crop_h, crop_w = crop.shape[:2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 210, 0), 2)
                cv2.putText(
                    overlay,
                    participant_id,
                    (x0 + 8, max(y0 + 24, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 210, 0),
                    2,
                    cv2.LINE_AA,
                )

                if pose_model is not None:
                    pose_result = pose_model.process(crop_rgb)
                    if pose_result.pose_landmarks:
                        for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
                            x_px = x0 + landmark.x * crop_w
                            y_px = y0 + landmark.y * crop_h
                            visibility = float(getattr(landmark, "visibility", np.nan))
                            pose_rows.append({
                                "video_id": video_id,
                                "participant_id": participant_id,
                                "frame_index": frame_index,
                                "timestamp_ms": timestamp_ms,
                                "keypoint_index": idx,
                                "keypoint_name": POSE_LANDMARK_NAMES[idx] if idx < len(POSE_LANDMARK_NAMES) else f"pose_{idx}",
                                "x_px": x_px,
                                "y_px": y_px,
                                "z": float(landmark.z),
                                "visibility": visibility,
                                "confidence": visibility,
                                "source": "mediapipe_pose_heavy",
                            })
                            if idx in {0, 11, 12, 15, 16} and np.isfinite(x_px) and np.isfinite(y_px):
                                cv2.circle(overlay, (int(x_px), int(y_px)), 4, (0, 255, 0), -1)

                if hands_model is not None:
                    hand_result = hands_model.process(crop_rgb)
                    landmarks = hand_result.multi_hand_landmarks or []
                    handedness = hand_result.multi_handedness or []
                    for hand_i, hand_landmarks in enumerate(landmarks):
                        label = "unknown"
                        confidence = np.nan
                        if hand_i < len(handedness) and handedness[hand_i].classification:
                            classification = handedness[hand_i].classification[0]
                            label = classification.label.lower()
                            confidence = float(classification.score)
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            x_px = x0 + landmark.x * crop_w
                            y_px = y0 + landmark.y * crop_h
                            hand_rows.append({
                                "video_id": video_id,
                                "participant_id": participant_id,
                                "frame_index": frame_index,
                                "timestamp_ms": timestamp_ms,
                                "hand_label": label,
                                "keypoint_index": idx,
                                "keypoint_name": HAND_LANDMARK_NAMES[idx] if idx < len(HAND_LANDMARK_NAMES) else f"hand_{idx}",
                                "x_px": x_px,
                                "y_px": y_px,
                                "z": float(landmark.z),
                                "confidence": confidence,
                                "source": "mediapipe_hands",
                            })
                            if idx in {0, 4, 8, 12, 16, 20} and np.isfinite(x_px) and np.isfinite(y_px):
                                cv2.circle(overlay, (int(x_px), int(y_px)), 3, (0, 165, 255), -1)

            if config["qc"].get("save_overlay_frames", True) and overlay_count < int(config["qc"]["max_overlay_frames"]):
                overlay_path = overlay_dir / f"{video_id}_frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(overlay_path), overlay)
                overlay_count += 1
            frame_index += 1
    finally:
        cap.release()
        if pose_model is not None:
            pose_model.close()
        if hands_model is not None:
            hands_model.close()

    pose_df = pd.DataFrame(pose_rows, columns=REQUIRED_FINAL_COLUMNS["pose"])
    hand_df = pd.DataFrame(hand_rows, columns=REQUIRED_FINAL_COLUMNS["hands"])
    pose_path = video_dir / "pose_keypoints_long.csv"
    hand_path = video_dir / "hand_keypoints_long.csv"
    atomic_write_csv(pose_path, pose_df)
    atomic_write_csv(hand_path, hand_df)
    gesture_path = video_dir / "gesture_features_long.csv"
    gesture_df = compute_gesture_features(hand_df, pose_df, config)
    atomic_write_csv(gesture_path, gesture_df)
    return {"pose": pose_path, "hands": hand_path, "gesture": gesture_path}


def compute_derivatives(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values(["frame_index", "timestamp_ms"]).copy()
    t = group["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    x = group["x_px"].to_numpy(dtype=float)
    y = group["y_px"].to_numpy(dtype=float)
    dt = np.diff(t, prepend=np.nan)
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)
    speed = np.sqrt(dx ** 2 + dy ** 2) / dt
    accel = np.diff(speed, prepend=np.nan) / dt
    jerk = np.diff(accel, prepend=np.nan) / dt
    group["speed_px_s"] = speed
    group["acceleration_px_s2"] = accel
    group["jerk_px_s3"] = jerk
    return group


def compute_gesture_features(hand_df: pd.DataFrame, pose_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if hand_df.empty:
        return pd.DataFrame(rows, columns=REQUIRED_FINAL_COLUMNS["gesture"])

    hand_df = hand_df.copy()
    derivative_groups = []
    for _, group in hand_df.groupby(["video_id", "participant_id", "hand_label", "keypoint_index"], dropna=False):
        derivative_groups.append(compute_derivatives(group))
    hand_df = pd.concat(derivative_groups, ignore_index=True)

    for _, row in hand_df.iterrows():
        for feature_name, value, unit in [
            ("keypoint_velocity", row.get("speed_px_s"), "px/s"),
            ("keypoint_acceleration", row.get("acceleration_px_s2"), "px/s2"),
            ("keypoint_jerk", row.get("jerk_px_s3"), "px/s3"),
        ]:
            rows.append({
                "video_id": row["video_id"],
                "participant_id": row["participant_id"],
                "frame_index": row["frame_index"],
                "timestamp_ms": row["timestamp_ms"],
                "feature_name": f"{feature_name}_{row['hand_label']}_{row['keypoint_name']}",
                "value": value,
                "unit": unit,
                "confidence": row.get("confidence"),
                "source": "derived_from_mediapipe_hands",
            })

    wrist = hand_df[hand_df["keypoint_name"] == "wrist"].copy()
    if not wrist.empty:
        rows.extend(compute_spatial_gesture_rows(wrist, pose_df))
        rows.extend(compute_temporal_gesture_rows(wrist, config))
    return pd.DataFrame(rows, columns=REQUIRED_FINAL_COLUMNS["gesture"])


def compute_spatial_gesture_rows(wrist: pd.DataFrame, pose_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if pose_df.empty:
        return rows
    pose_lookup = pose_df.set_index(["video_id", "participant_id", "frame_index", "keypoint_name"])
    for _, hand in wrist.iterrows():
        base = {
            "video_id": hand["video_id"],
            "participant_id": hand["participant_id"],
            "frame_index": hand["frame_index"],
            "timestamp_ms": hand["timestamp_ms"],
            "confidence": hand.get("confidence"),
            "source": "derived_from_mediapipe_pose_hands",
        }
        for target_name, feature_name in [
            ("nose", "hand_to_face_distance"),
            ("left_shoulder", "hand_to_left_shoulder_distance"),
            ("right_shoulder", "hand_to_right_shoulder_distance"),
        ]:
            key = (hand["video_id"], hand["participant_id"], hand["frame_index"], target_name)
            if key not in pose_lookup.index:
                continue
            target = pose_lookup.loc[key]
            if isinstance(target, pd.DataFrame):
                target = target.iloc[0]
            value = math.hypot(hand["x_px"] - target["x_px"], hand["y_px"] - target["y_px"])
            rows.append({**base, "feature_name": f"{feature_name}_{hand['hand_label']}", "value": value, "unit": "px"})
    return rows


def compute_temporal_gesture_rows(wrist: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    speed_threshold = float(config["features"]["motion_speed_threshold_px_s"])
    hold_threshold = float(config["features"]["hold_speed_threshold_px_s"])
    min_duration = float(config["features"]["min_stroke_duration_s"])
    max_hold = float(config["features"]["max_post_stroke_hold_s"])

    for (video_id, participant_id, hand_label), group in wrist.groupby(["video_id", "participant_id", "hand_label"], dropna=False):
        group = group.sort_values("timestamp_ms")
        moving = group["speed_px_s"].fillna(0).to_numpy() > speed_threshold
        times = group["timestamp_ms"].to_numpy(dtype=float) / 1000.0
        strokes: list[tuple[float, float]] = []
        start = None
        for i, is_moving in enumerate(moving):
            if is_moving and start is None:
                start = times[i]
            if (not is_moving or i == len(moving) - 1) and start is not None:
                end = times[i] if is_moving and i == len(moving) - 1 else times[max(i - 1, 0)]
                if end - start >= min_duration:
                    strokes.append((start, end))
                start = None
        duration_min = max((times[-1] - times[0]) / 60.0, np.nan) if len(times) >= 2 else np.nan
        gesture_rate = len(strokes) / duration_min if np.isfinite(duration_min) and duration_min > 0 else np.nan
        base = {
            "video_id": video_id,
            "participant_id": participant_id,
            "frame_index": -1,
            "timestamp_ms": np.nan,
            "confidence": group["confidence"].mean(),
            "source": "derived_from_mediapipe_hands",
        }
        rows.append({**base, "feature_name": f"gesture_rate_{hand_label}", "value": gesture_rate, "unit": "strokes/min"})
        if strokes:
            rows.append({
                **base,
                "feature_name": f"mean_stroke_duration_{hand_label}",
                "value": float(np.mean([end - start for start, end in strokes])),
                "unit": "s",
            })
        holds = []
        for _, end in strokes:
            after = group[group["timestamp_ms"] / 1000.0 > end]
            if after.empty:
                continue
            low = after[after["speed_px_s"].fillna(np.inf) < hold_threshold]
            if low.empty:
                continue
            hold = min((low.iloc[0]["timestamp_ms"] / 1000.0) - end, max_hold)
            if hold >= 0:
                holds.append(hold)
        if holds:
            rows.append({
                **base,
                "feature_name": f"mean_post_stroke_hold_{hand_label}",
                "value": float(np.mean(holds)),
                "unit": "s",
            })
    rows.extend(compute_bilateral_symmetry_rows(wrist))
    rows.extend(compute_gesture_space_rows(wrist))
    return rows


def compute_bilateral_symmetry_rows(wrist: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (video_id, participant_id), group in wrist.groupby(["video_id", "participant_id"], dropna=False):
        pivot = group.pivot_table(
            index="frame_index",
            columns="hand_label",
            values="speed_px_s",
            aggfunc="mean",
        )
        if {"left", "right"}.issubset(set(pivot.columns)) and len(pivot.dropna()) >= 3:
            left = pivot["left"]
            right = pivot["right"]
            symmetry = 1.0 - (left - right).abs() / (left.abs() + right.abs() + 1e-9)
            value = float(symmetry.replace([np.inf, -np.inf], np.nan).mean())
        else:
            value = np.nan
        rows.append({
            "video_id": video_id,
            "participant_id": participant_id,
            "frame_index": -1,
            "timestamp_ms": np.nan,
            "feature_name": "bilateral_symmetry_index",
            "value": value,
            "unit": "unitless",
            "confidence": group["confidence"].mean(),
            "source": "derived_from_mediapipe_hands",
        })
    return rows


def compute_gesture_space_rows(wrist: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        ConvexHull = None
    for (video_id, participant_id, hand_label), group in wrist.groupby(["video_id", "participant_id", "hand_label"], dropna=False):
        coords2 = group[["x_px", "y_px"]].dropna().to_numpy(dtype=float)
        area = np.nan
        if ConvexHull is not None and len(coords2) >= 3:
            try:
                area = float(ConvexHull(coords2).volume)
            except Exception:
                area = np.nan
        rows.append({
            "video_id": video_id,
            "participant_id": participant_id,
            "frame_index": -1,
            "timestamp_ms": np.nan,
            "feature_name": f"gesture_space_area_{hand_label}",
            "value": area,
            "unit": "px2",
            "confidence": group["confidence"].mean(),
            "source": "derived_from_mediapipe_hands",
        })
    return rows


def run_openface_video(
    video_path: Path,
    meta: dict[str, Any],
    config: dict[str, Any],
    dirs: RunDirs,
    logger: logging.Logger,
) -> Path | None:
    openface_config = config["openface"]
    if not openface_config.get("enabled", False):
        return None
    video_id = meta["video_id"]
    out_dir = dirs.openface / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv_dir = openface_config.get("input_csv_dir")
    csv_path = None
    if input_csv_dir:
        candidate = Path(input_csv_dir) / f"{video_path.stem}.csv"
        if candidate.exists():
            csv_path = candidate
        else:
            raise PipelineError(
                "Configured OpenFace CSV is missing.",
                stage="openface_facial_analysis",
                likely_cause="input_csv_dir does not contain a matching OpenFace CSV.",
                suggested_fix="Run OpenFace externally or correct openface.input_csv_dir.",
                affected_file=str(candidate),
                can_continue=not openface_config.get("required", False),
            )
    else:
        executable = shutil.which(openface_config.get("executable", "FeatureExtraction"))
        if not executable:
            raise PipelineError(
                "OpenFace executable was not found.",
                stage="openface_facial_analysis",
                likely_cause="OpenFace 2.x is not installed or FeatureExtraction is not on PATH.",
                suggested_fix="Install OpenFace and set openface.executable, or disable openface.enabled.",
                affected_file=openface_config.get("executable"),
                can_continue=not openface_config.get("required", False),
                level="CRITICAL" if openface_config.get("required", False) else "WARNING",
            )
        cmd = [
            executable,
            "-f", str(video_path),
            "-out_dir", str(out_dir),
            "-aus",
            "-gaze",
            "-pose",
        ]
        logger.info(f"Running OpenFace for {video_id}.", extra={"stage": "openface_facial_analysis"})
        result = subprocess.run(cmd, text=True, capture_output=True)
        atomic_write_text(out_dir / "openface_stdout.log", result.stdout)
        atomic_write_text(out_dir / "openface_stderr.log", result.stderr)
        if result.returncode != 0:
            raise PipelineError(
                "OpenFace returned a non-zero exit status.",
                stage="openface_facial_analysis",
                likely_cause="OpenFace could not process the video or its dependencies are missing.",
                suggested_fix="Inspect openface_stderr.log and try a small test video.",
                affected_file=str(video_path),
                can_continue=not openface_config.get("required", False),
            )
        generated = sorted(out_dir.glob("*.csv"))
        if generated:
            csv_path = generated[0]

    if csv_path is None or not csv_path.exists():
        raise PipelineError(
            "OpenFace did not produce a CSV output.",
            stage="openface_facial_analysis",
            likely_cause="OpenFace finished without a parseable CSV.",
            suggested_fix="Check OpenFace logs and video readability.",
            affected_file=str(video_path),
            can_continue=not openface_config.get("required", False),
        )
    facial_df = parse_openface_csv(csv_path, meta, config)
    facial_path = out_dir / "facial_features_long.csv"
    atomic_write_csv(facial_path, facial_df)
    return facial_path


def parse_openface_csv(csv_path: Path, meta: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw.columns = [col.strip() for col in raw.columns]
    confidence_min = float(config["openface"]["confidence_min"])
    if "confidence" in raw.columns:
        raw = raw[raw["confidence"] >= confidence_min]
    if config["openface"].get("success_required", True) and "success" in raw.columns:
        raw = raw[raw["success"] == 1]

    feature_cols = [
        col for col in raw.columns
        if col.startswith("AU") or col.startswith("gaze_") or col.startswith("pose_R")
    ]
    rows = []
    for _, row in raw.iterrows():
        frame_index = int(row["frame"]) if "frame" in row and pd.notna(row["frame"]) else -1
        timestamp_ms = float(row["timestamp"]) * 1000.0 if "timestamp" in row and pd.notna(row["timestamp"]) else frame_index / meta["fps"] * 1000.0
        face_id = str(row["face_id"]) if "face_id" in row else "face_0"
        confidence = float(row["confidence"]) if "confidence" in row and pd.notna(row["confidence"]) else np.nan
        for col in feature_cols:
            rows.append({
                "video_id": meta["video_id"],
                "participant_id": face_id,
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "feature_name": col,
                "value": row[col],
                "confidence": confidence,
                "source": "openface_2x",
            })
    return pd.DataFrame(rows, columns=REQUIRED_FINAL_COLUMNS["facial"])


def validate_csv_schema(path: Path, required_columns: list[str], allow_empty: bool = False) -> list[dict[str, Any]]:
    issues = []
    if not path.exists():
        return [{"level": "FAIL", "message": "Expected output file is missing.", "path": str(path)}]
    if path.stat().st_size == 0:
        return [{"level": "FAIL", "message": "Expected output file is empty.", "path": str(path)}]
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return [{"level": "FAIL", "message": f"CSV could not be read: {exc}", "path": str(path)}]
    if df.empty and not allow_empty:
        issues.append({"level": "FAIL", "message": "CSV has no data rows.", "path": str(path)})
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        issues.append({"level": "FAIL", "message": f"CSV is missing required columns: {missing}", "path": str(path)})
    return issues


def validate_time_series(path: Path, id_columns: list[str]) -> list[dict[str, Any]]:
    issues = []
    if not path.exists() or path.stat().st_size == 0:
        return issues
    df = pd.read_csv(path)
    if df.empty:
        return issues
    for _, group in df.groupby(id_columns, dropna=False):
        frame_times = group[["frame_index", "timestamp_ms"]].drop_duplicates().sort_values("frame_index")
        if not frame_times["frame_index"].is_monotonic_increasing:
            issues.append({"level": "FAIL", "message": "Frame indices are not increasing.", "path": str(path)})
        if not frame_times["timestamp_ms"].is_monotonic_increasing:
            issues.append({"level": "FAIL", "message": "Timestamps are not increasing.", "path": str(path)})
    return issues


def validate_spatial_bounds(path: Path, meta: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    issues = []
    if not path.exists() or path.stat().st_size == 0:
        return issues
    df = pd.read_csv(path)
    if df.empty or not {"x_px", "y_px"}.issubset(df.columns):
        return issues
    margin = float(config["qc"]["coordinate_margin_px"])
    bad = df[
        (df["x_px"] < -margin)
        | (df["x_px"] > meta["width"] + margin)
        | (df["y_px"] < -margin)
        | (df["y_px"] > meta["height"] + margin)
    ]
    if len(bad) > 0:
        issues.append({
            "level": "WARNING",
            "message": "Some detected keypoints fall outside expected image bounds.",
            "path": str(path),
            "n_bad_rows": int(len(bad)),
        })
    return issues


def validate_video_outputs(
    video_id: str,
    meta: dict[str, Any],
    paths: dict[str, Path | None],
    config: dict[str, Any],
) -> dict[str, Any]:
    allow_empty = bool(config["run"].get("allow_empty_optional_outputs", True))
    issues: list[dict[str, Any]] = []
    if paths.get("facial") is not None:
        issues.extend(validate_csv_schema(paths["facial"], REQUIRED_FINAL_COLUMNS["facial"], allow_empty=allow_empty))
        issues.extend(validate_time_series(paths["facial"], ["video_id", "participant_id"]))
    if paths.get("pose") is not None:
        issues.extend(validate_csv_schema(paths["pose"], REQUIRED_FINAL_COLUMNS["pose"], allow_empty=allow_empty))
        issues.extend(validate_time_series(paths["pose"], ["video_id", "participant_id"]))
        issues.extend(validate_spatial_bounds(paths["pose"], meta, config))
    if paths.get("hands") is not None:
        issues.extend(validate_csv_schema(paths["hands"], REQUIRED_FINAL_COLUMNS["hands"], allow_empty=allow_empty))
        issues.extend(validate_time_series(paths["hands"], ["video_id", "participant_id", "hand_label"]))
        issues.extend(validate_spatial_bounds(paths["hands"], meta, config))
    if paths.get("gesture") is not None:
        issues.extend(validate_csv_schema(paths["gesture"], REQUIRED_FINAL_COLUMNS["gesture"], allow_empty=True))

    detection = compute_detection_summary(video_id, meta, paths, config)
    issues.extend(detection["issues"])
    if any(issue["level"] == "FAIL" for issue in issues):
        status = "FAIL"
    elif issues:
        status = "WARNING"
    else:
        status = "PASS"
    return {
        "video_id": video_id,
        "status": status,
        "issues": issues,
        "detection_summary": detection,
    }


def compute_detection_summary(
    video_id: str,
    meta: dict[str, Any],
    paths: dict[str, Path | None],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_n = len(expected_sample_indices(
        meta["frame_count"],
        meta["fps"],
        float(config["sampling"]["sample_fps"]),
        config["sampling"].get("max_frames_per_video"),
    ))
    summary: dict[str, Any] = {"expected_sampled_frames": expected_n, "issues": []}
    if paths.get("pose") and paths["pose"].exists():
        pose = pd.read_csv(paths["pose"])
        by_participant = pose.groupby("participant_id")["frame_index"].nunique().to_dict() if not pose.empty else {}
        summary["pose_detected_frames_by_participant"] = by_participant
        for participant in config["participants"]:
            participant_id = participant["participant_id"]
            rate = by_participant.get(participant_id, 0) / expected_n if expected_n else 0
            summary[f"pose_detection_rate_{participant_id}"] = rate
            if rate < float(config["qc"]["pose_detection_rate_warning"]):
                summary["issues"].append({
                    "level": "WARNING",
                    "message": f"Low pose detection rate for {participant_id}.",
                    "rate": rate,
                    "threshold": config["qc"]["pose_detection_rate_warning"],
                })
    if paths.get("hands") and paths["hands"].exists():
        hands = pd.read_csv(paths["hands"])
        by_participant = hands.groupby("participant_id")["frame_index"].nunique().to_dict() if not hands.empty else {}
        summary["hand_detected_frames_by_participant"] = by_participant
        for participant in config["participants"]:
            participant_id = participant["participant_id"]
            rate = by_participant.get(participant_id, 0) / expected_n if expected_n else 0
            summary[f"hand_detection_rate_{participant_id}"] = rate
            if rate < float(config["qc"]["hand_detection_rate_warning"]):
                summary["issues"].append({
                    "level": "WARNING",
                    "message": f"Low hand detection rate for {participant_id}.",
                    "rate": rate,
                    "threshold": config["qc"]["hand_detection_rate_warning"],
                })
    return summary


def generate_qc_outputs(
    video_id: str,
    meta: dict[str, Any],
    paths: dict[str, Path | None],
    qc: dict[str, Any],
    dirs: RunDirs,
    config: dict[str, Any],
) -> None:
    qc_dir = dirs.qc / video_id
    qc_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(qc_dir / "qc_report.json", qc)
    lines = [
        f"# QC Report: {video_id}",
        "",
        f"Status: {qc['status']}",
        f"FPS: {meta['fps']}",
        f"Frame count: {meta['frame_count']}",
        f"Duration seconds: {meta['duration_s']:.3f}",
        "",
        "## Issues",
    ]
    if qc["issues"]:
        for issue in qc["issues"]:
            lines.append(f"- {issue.get('level', 'WARNING')}: {issue.get('message')}")
    else:
        lines.append("- None")
    atomic_write_text(qc_dir / "qc_report.md", "\n".join(lines) + "\n")
    if config["qc"].get("save_plots", True):
        generate_qc_plot(qc_dir / "qc_plots.png", paths)


def generate_qc_plot(path: Path, paths: dict[str, Path | None]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if paths.get("pose") and paths["pose"].exists():
        pose = pd.read_csv(paths["pose"])
        if not pose.empty:
            counts = pose.groupby("participant_id")["frame_index"].nunique()
            counts.plot(kind="bar", ax=axes[0], title="Pose detected frames")
    axes[0].set_ylabel("frames")
    if paths.get("hands") and paths["hands"].exists():
        hands = pd.read_csv(paths["hands"])
        if not hands.empty:
            counts = hands.groupby("participant_id")["frame_index"].nunique()
            counts.plot(kind="bar", ax=axes[1], title="Hand detected frames")
    axes[1].set_ylabel("frames")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def concatenate_outputs(video_results: list[dict[str, Any]], dirs: RunDirs) -> dict[str, Path]:
    final_paths = {
        "facial": dirs.final / "video_facial_features_long.csv",
        "pose": dirs.final / "video_pose_keypoints_long.csv",
        "hands": dirs.final / "video_hand_keypoints_long.csv",
        "gesture": dirs.final / "video_gesture_features_long.csv",
    }
    for key, path in final_paths.items():
        frames = []
        for result in video_results:
            source = result["outputs"].get(key)
            if source is not None and Path(source).exists():
                frames.append(pd.read_csv(source))
        if frames:
            atomic_write_csv(path, pd.concat(frames, ignore_index=True))
        else:
            atomic_write_csv(path, pd.DataFrame(columns=REQUIRED_FINAL_COLUMNS[key]))
    return final_paths


def build_summary_tables(video_results: list[dict[str, Any]], dirs: RunDirs) -> tuple[Path, Path]:
    rows = []
    qc_rows = []
    for result in video_results:
        meta = result["metadata"]
        qc = result["qc"]
        rows.append({
            "video_id": meta["video_id"],
            "video_path": meta["video_path"],
            "fps": meta["fps"],
            "frame_count": meta["frame_count"],
            "duration_s": meta["duration_s"],
            "width": meta["width"],
            "height": meta["height"],
            "qc_status": qc["status"],
        })
        detection = qc.get("detection_summary", {})
        qc_rows.append({
            "video_id": meta["video_id"],
            "qc_status": qc["status"],
            "n_issues": len(qc.get("issues", [])),
            "expected_sampled_frames": detection.get("expected_sampled_frames"),
            **{
                key: value
                for key, value in detection.items()
                if key.startswith("pose_detection_rate_") or key.startswith("hand_detection_rate_")
            },
        })
    summary_path = dirs.final / "video_behavior_summary.csv"
    qc_path = dirs.final / "video_qc_summary.csv"
    atomic_write_csv(summary_path, pd.DataFrame(rows))
    atomic_write_csv(qc_path, pd.DataFrame(qc_rows))
    return summary_path, qc_path


def write_manifests(
    input_videos: list[Path],
    dirs: RunDirs,
    video_results: list[dict[str, Any]],
    final_paths: dict[str, Path],
    summary_paths: tuple[Path, Path],
) -> None:
    input_records = [file_record(path, "raw_video_input") for path in input_videos]
    atomic_write_csv(dirs.metadata / "input_manifest.csv", pd.DataFrame(input_records))

    output_files = []
    for result in video_results:
        for path in result["outputs"].values():
            if path is not None and Path(path).exists():
                output_files.append(Path(path))
        qc_dir = dirs.qc / result["metadata"]["video_id"]
        output_files.extend(sorted(p for p in qc_dir.rglob("*") if p.is_file()))
    output_files.extend(final_paths.values())
    output_files.extend(summary_paths)
    output_files.extend(sorted(p for p in dirs.metadata.glob("*") if p.is_file()))
    output_files.extend(sorted(p for p in dirs.config.glob("*") if p.is_file()))
    output_records = [file_record(path, "pipeline_output") for path in sorted(set(output_files))]
    atomic_write_csv(dirs.metadata / "output_manifest.csv", pd.DataFrame(output_records))
    manifest = {
        "run_dir": str(dirs.root),
        "inputs": input_records,
        "outputs": output_records,
    }
    atomic_write_json(dirs.final / "processing_manifest.json", manifest)


def write_run_status(dirs: RunDirs, status: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(dirs.root),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(dirs.root / RUN_STATUS_FILE, payload)


def detect_incomplete_run(run_dir: Path) -> dict[str, Any] | None:
    status_path = run_dir / RUN_STATUS_FILE
    if not status_path.exists():
        return {"status": "UNKNOWN", "message": "Run status file is missing."}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "CORRUPT", "message": "Run status file could not be parsed."}
    if status.get("status") != "COMPLETE":
        return status
    return None


def clean_failed_run(run_dir: Path) -> None:
    incomplete = detect_incomplete_run(run_dir)
    if incomplete is None:
        raise PipelineError(
            "Refusing to clean a completed run.",
            stage="rollback",
            likely_cause="The requested run is marked COMPLETE.",
            suggested_fix="Create a new run instead of cleaning this one.",
            affected_file=str(run_dir),
            can_continue=False,
            level="CRITICAL",
        )
    if not run_dir.exists():
        return
    shutil.rmtree(run_dir)


def process_one_video(
    video_path: Path,
    meta: dict[str, Any],
    config: dict[str, Any],
    dirs: RunDirs,
    logger: logging.Logger,
) -> dict[str, Any]:
    video_id = meta["video_id"]
    paths: dict[str, Path | None] = {"facial": None, "pose": None, "hands": None, "gesture": None}
    warnings = validate_metadata(meta, config)
    for warning in warnings:
        logger.warning(warning["message"], extra={"stage": warning["stage"]})

    try:
        facial_path = run_openface_video(video_path, meta, config, dirs, logger)
        paths["facial"] = facial_path
    except PipelineError as exc:
        if not exc.can_continue:
            raise
        log_pipeline_error(logger, exc)

    if config["mediapipe"].get("enabled", True):
        mp_paths = process_mediapipe_video(video_path, meta, config, dirs, logger)
        paths.update(mp_paths)

    qc = validate_video_outputs(video_id, meta, paths, config)
    qc["issues"].extend(warnings)
    if qc["status"] == "PASS" and warnings:
        qc["status"] = "WARNING"
    generate_qc_outputs(video_id, meta, paths, qc, dirs, config)
    return {"metadata": meta, "outputs": paths, "qc": qc}


def run_pipeline(args: argparse.Namespace) -> int:
    try:
        config = load_yaml_config(Path(args.config))
        if args.fail_fast:
            config["run"]["fail_fast"] = True
        if args.continue_on_warning:
            config["run"]["continue_on_warning"] = True
        validate_config(config)
        if args.clean_failed_run:
            clean_failed_run(Path(args.clean_failed_run))
            print(f"Removed failed run directory: {args.clean_failed_run}")
            return 0

        run_id = config["run"].get("run_id") or f"video_behavior_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_root = Path(config["project"]["output_root"])
        resume_dir = Path(args.resume) if args.resume else None
        if resume_dir:
            incomplete = detect_incomplete_run(resume_dir)
            if incomplete is None:
                raise PipelineError(
                    "Requested resume run is already complete.",
                    stage="rollback",
                    likely_cause="The resume directory is marked COMPLETE.",
                    suggested_fix="Start a new run or choose an incomplete run directory.",
                    affected_file=str(resume_dir),
                    can_continue=False,
                    level="CRITICAL",
                )
        dirs = build_run_dirs(output_root, run_id, resume_dir)
        ensure_run_dirs(dirs, overwrite=bool(config["run"].get("overwrite")), resume=bool(resume_dir))
        logger = setup_logging(dirs.logs / "run.log", args.verbose)
        write_run_status(dirs, "RUNNING")
        save_yaml(dirs.config / "config_used.yaml", config)
        atomic_write_json(dirs.config / "software_versions.json", get_software_versions(config))

        videos = find_videos(config, Path(args.video) if args.video else None)
        metadata_rows = []
        for video in videos:
            try:
                metadata_rows.append(inspect_video(video))
            except PipelineError as exc:
                log_pipeline_error(logger, exc)
                if config["run"].get("fail_fast") or not exc.can_continue:
                    raise
        if not metadata_rows:
            raise PipelineError(
                "No videos passed metadata inspection.",
                stage="metadata_inspection",
                likely_cause="All input videos were unreadable or invalid.",
                suggested_fix="Check codecs and paths, then rerun.",
                can_continue=False,
                level="CRITICAL",
            )
        atomic_write_csv(dirs.metadata / "video_metadata.csv", pd.DataFrame(metadata_rows))

        planned = {
            "run_id": run_id,
            "run_dir": str(dirs.root),
            "n_videos": len(metadata_rows),
            "videos": [row["video_path"] for row in metadata_rows],
        }
        if args.dry_run:
            atomic_write_json(dirs.metadata / "dry_run_plan.json", planned)
            write_run_status(dirs, "DRY_RUN_COMPLETE", planned)
            logger.info(f"Dry run complete. Planned run directory: {dirs.root}", extra={"stage": "dry_run"})
            return 0

        video_results = []
        for meta in metadata_rows:
            video_path = Path(meta["video_path"])
            try:
                result = process_one_video(video_path, meta, config, dirs, logger)
                video_results.append(result)
                status = result["qc"]["status"]
                logger.info(f"{meta['video_id']} QC status: {status}", extra={"stage": "output_validation"})
                if status == "FAIL" and config["run"].get("fail_fast"):
                    raise PipelineError(
                        "Video failed QC and fail-fast is enabled.",
                        stage="output_validation",
                        likely_cause="One or more required output checks failed.",
                        suggested_fix="Inspect the per-video QC report.",
                        affected_file=meta["video_path"],
                        can_continue=False,
                    )
                if status == "WARNING" and not config["run"].get("continue_on_warning", True):
                    raise PipelineError(
                        "Video produced warnings and continue-on-warning is disabled.",
                        stage="output_validation",
                        likely_cause="QC warnings were generated.",
                        suggested_fix="Inspect the per-video QC report or enable continue_on_warning.",
                        affected_file=meta["video_path"],
                        can_continue=False,
                    )
            except PipelineError as exc:
                log_pipeline_error(logger, exc)
                if config["run"].get("fail_fast") or not exc.can_continue:
                    raise

        final_paths = concatenate_outputs(video_results, dirs)
        summary_paths = build_summary_tables(video_results, dirs)
        write_manifests([Path(row["video_path"]) for row in metadata_rows], dirs, video_results, final_paths, summary_paths)
        run_summary = {
            "run_id": run_id,
            "run_dir": str(dirs.root),
            "n_videos": len(video_results),
            "n_pass": sum(1 for result in video_results if result["qc"]["status"] == "PASS"),
            "n_warning": sum(1 for result in video_results if result["qc"]["status"] == "WARNING"),
            "n_fail": sum(1 for result in video_results if result["qc"]["status"] == "FAIL"),
        }
        atomic_write_json(dirs.metadata / "run_summary.json", run_summary)
        write_run_status(dirs, "COMPLETE", run_summary)
        logger.info(f"Run complete: {dirs.root}", extra={"stage": "final_report"})
        return 0
    except PipelineError as exc:
        try:
            log_pipeline_error(logging.getLogger("video_behavior"), exc)
        finally:
            return 1
    except PermissionError as exc:
        print(
            "CRITICAL [write_error] Permission or write error. "
            f"Likely cause: output folder is not writable. Suggested fix: choose a writable output_root. Details: {exc}",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("ERROR [interrupted] Run interrupted by user. Re-run with --resume if outputs were partially written.", file=sys.stderr)
        return 130


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CVS lab-grade video behavior pipeline.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--video", help="Optional single video path.")
    parser.add_argument("--resume", help="Resume an incomplete run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/inputs and write a plan without processing frames.")
    parser.add_argument("--verbose", action="store_true", help="Print debug-level logs to the terminal.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first video error or QC failure.")
    parser.add_argument("--continue-on-warning", action="store_true", help="Continue when a video produces QC warnings.")
    parser.add_argument("--clean-failed-run", help="Remove a previous incomplete run directory created by this pipeline.")
    return parser


if __name__ == "__main__":
    raise SystemExit(run_pipeline(build_arg_parser().parse_args()))

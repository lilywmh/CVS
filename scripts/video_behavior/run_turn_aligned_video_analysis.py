#!/usr/bin/env python3
"""
Turn-aligned video behavior pipeline for CVS conversation recordings.

This project-specific runner treats the output of
scripts/acoustic_alignment/align_manual_labels_to_whisperx.py and discussion
speaker logs as authoritative. It does not infer question boundaries or
speaker turns from video. The primary analysis unit is:

    dyad_id x session_id x question_id x speaker_turn

Raw videos, transcript-alignment files, and speaker logs are read-only inputs.
Each execution writes a timestamped run directory with analysis-ready outputs,
checked metadata, QC files, logs, and a manifest.

Example:
  python scripts/video_behavior/run_turn_aligned_video_analysis.py \
      --config config/video_turn_analysis.example.yml
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.video_behavior.run_video_analysis import (  # noqa: E402
    ACCEPTED_VIDEO_EXTENSIONS,
    HAND_LANDMARK_NAMES,
    POSE_LANDMARK_NAMES,
    PipelineError,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    deep_merge,
    file_record,
    get_software_versions,
    inspect_video,
    log_pipeline_error,
    save_yaml,
    setup_logging,
)
from scripts.video_behavior.upper_limb_features import (  # noqa: E402
    ARM_FRAME_COLUMNS,
    ARM_TURN_ROLE_COLUMNS,
    PARTICIPANT_ARM_SUMMARY_COLUMNS,
    PARTICIPANT_ROLE_ARM_SUMMARY_COLUMNS,
    SPEAKER_LISTENER_COMPARISON_COLUMNS,
    aggregate_participant_arm_summary,
    aggregate_participant_role_arm_summary,
    compute_upper_limb_frame_row,
    frame_base_from_turn,
    speaker_listener_arm_comparison,
    summarize_turn_role_arm_features,
)


DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("CVS_VIDEO_OUTPUTS", PROJECT / "data" / "interim" / "video_behavior" / "turn_runs")
)

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "video_dir": str(PROJECT / "01_pipeline" / "recording_videos"),
        "video_metadata_path": None,
        "video_view_preference": ["discussion", "1st_view", "2nd_view"],
        "aligned_turns_path": str(PROJECT / "04_data" / "labeled_turns.csv"),
        "speaker_log_path": str(PROJECT / "04_data" / "Discussion Transcription Log - Sheet1.csv"),
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "accepted_video_extensions": sorted(ACCEPTED_VIDEO_EXTENSIONS),
    },
    "run": {
        "run_id": None,
        "overwrite": False,
        "fail_fast": False,
        "continue_on_warning": True,
    },
    "schema": {
        "video_metadata_required": ["dyad_id", "session_id", "video_id", "video_path"],
        "aligned_turns_required": [
            "dyad_id", "question", "turn_idx", "speaker", "t_start", "t_end",
        ],
        "speaker_log_required": ["dyad_id", "session_id", "speaker_label"],
    },
    "validation": {
        "expected_sessions_per_dyad": 2,
        "expected_question_ids": None,
        "allow_overlapping_turns": False,
        "min_turn_duration_sec": 0.20,
        "max_turn_duration_sec": 300.0,
        "empty_utterance_warning": True,
        "timestamp_tolerance_sec": 0.05,
        "speaker_log_question_column": "question_id",
        "speaker_log_validity_column": "segment_valid",
        "speaker_log_notes_column": "notes",
        "speaker_log_participant_column": "participant_id",
        "speaker_log_role_column": "participant_role",
    },
    "sampling": {
        "sample_fps": 2.0,
        "max_frames_per_turn": None,
    },
    # Lab standard (fixed camera setup, hard-coded for every video):
    # participant A = door side = RIGHT half of the frame,
    # participant B = window side = LEFT half of the frame.
    # participant_roi_lookup() enforces this; configs that contradict it fail.
    "participants": [
        {"participant_id": "A", "speaker_label": "A", "roi": [0.5, 0.0, 1.0, 1.0]},
        {"participant_id": "B", "speaker_label": "B", "roi": [0.0, 0.0, 0.5, 1.0]},
    ],
    "mediapipe": {
        "enabled": True,
        "pose_enabled": True,
        "hands_enabled": True,
        "pose_model_complexity": 2,
        "pose_min_detection_confidence": 0.50,
        "hands_max_num_hands": 2,
        "hands_min_detection_confidence": 0.50,
    },
    "openface": {
        "enabled": False,
        "input_csv_dir": None,
        "confidence_min": 0.80,
        "success_required": True,
    },
    "features": {
        "motion_speed_threshold_px_s": 40.0,
        "hold_speed_threshold_px_s": 10.0,
    },
    "upper_limb": {
        "enabled": True,
        "visibility_min": 0.30,
        "movement_speed_threshold_px_s": 60.0,
        "hold_speed_threshold_px_s": 12.0,
        "near_face_threshold_norm": 0.75,
        "raised_wrist_threshold_norm": -0.15,
        "resting_low_threshold_norm": 0.75,
        "open_openness_threshold": 0.85,
        "closed_openness_threshold": 0.45,
        "contracted_extension_threshold": 0.55,
    },
    "qc": {
        "pose_detection_rate_warning": 0.50,
        "hand_detection_rate_warning": 0.20,
        "arm_detection_rate_warning": 0.50,
        "high_missingness_warning": 0.50,
        "save_html_report": True,
    },
}


TURN_OUTPUT_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "question_id", "turn_id",
    "speaker_label", "participant_id", "utterance_text", "turn_start_sec",
    "turn_end_sec", "turn_duration_sec", "n_sampled_frames", "pose_detection_rate",
    "hand_detection_rate", "mean_wrist_speed_px_s", "max_wrist_speed_px_s",
    "mean_hand_to_face_distance_px", "mean_hand_to_shoulder_distance_px",
    "gesture_frame_rate", "openface_mean_AU12_r", "openface_mean_AU06_r",
    "openface_mean_AU04_r", "openface_mean_pose_Rx", "openface_mean_pose_Ry",
    "openface_mean_pose_Rz", "missingness_rate", "qc_status", "qc_notes",
]


@dataclass
class TurnRunDirs:
    root: Path
    results: Path
    qc: Path
    logs: Path
    metadata: Path


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise PipelineError(
            "PyYAML is required to read the config file.",
            stage="config_loading",
            likely_cause="The active environment does not include PyYAML.",
            suggested_fix="Install PyYAML or recreate the conda environment.",
            affected_file=str(path),
            can_continue=False,
            level="CRITICAL",
        ) from exc
    if not path.exists():
        raise PipelineError(
            "Config file does not exist.",
            stage="config_loading",
            likely_cause="The --config path is wrong.",
            suggested_fix="Use config/video_turn_analysis.example.yml or a private copy.",
            affected_file=str(path),
            can_continue=False,
            level="CRITICAL",
        )
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise PipelineError(
            "Config root must be a mapping.",
            stage="config_loading",
            likely_cause="The YAML file top level is not key/value sections.",
            suggested_fix="Use the provided example config as a template.",
            affected_file=str(path),
            can_continue=False,
            level="CRITICAL",
        )
    return deep_merge(DEFAULT_CONFIG, loaded)


def build_run_dirs(config: dict[str, Any]) -> TurnRunDirs:
    run_id = config["run"].get("run_id") or f"turn_aligned_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = Path(config["project"]["output_root"]) / run_id
    return TurnRunDirs(
        root=root,
        results=root / "results",
        qc=root / "qc",
        logs=root / "logs",
        metadata=root / "metadata",
    )


def ensure_run_dirs(dirs: TurnRunDirs, overwrite: bool) -> None:
    if dirs.root.exists() and not overwrite:
        raise PipelineError(
            "Run directory already exists.",
            stage="run_setup",
            likely_cause="The configured run_id was already used.",
            suggested_fix="Use a new run_id or set run.overwrite only for generated run folders.",
            affected_file=str(dirs.root),
            can_continue=False,
            level="CRITICAL",
        )
    for path in [dirs.results, dirs.qc, dirs.logs, dirs.metadata]:
        path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise PipelineError(
            "Required input table is missing.",
            stage="input_loading",
            likely_cause="A configured metadata/transcript/log path is wrong.",
            suggested_fix="Update the config path or generate the required source table.",
            affected_file=str(path),
            can_continue=False,
            level="CRITICAL",
        )
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise PipelineError(
        "Unsupported table format.",
        stage="input_loading",
        likely_cause=f"{suffix} is not supported for project tables.",
        suggested_fix="Use csv, tsv, json, or jsonl.",
        affected_file=str(path),
        can_continue=False,
        level="CRITICAL",
    )


def normalize_ids(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def require_columns(df: pd.DataFrame, columns: list[str], name: str, path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise PipelineError(
            f"{name} is missing required columns: {missing}",
            stage="schema_validation",
            likely_cause="The input table does not match the expected project schema.",
            suggested_fix="Regenerate labeled_turns.csv with align_manual_labels_to_whisperx.py or update the config schema.",
            affected_file=str(path),
            can_continue=False,
            level="CRITICAL",
        )


def aligned_turns_path_from_config(config: dict[str, Any]) -> Path:
    return Path(
        config["project"].get("aligned_turns_path")
        or config["project"].get("aligned_transcript_path")
        or DEFAULT_CONFIG["project"]["aligned_turns_path"]
    )


def video_metadata_path_from_config(config: dict[str, Any]) -> Path | None:
    path = config["project"].get("video_metadata_path")
    return Path(path) if path else None


def source_input_paths_from_config(config: dict[str, Any]) -> list[Path]:
    video_metadata_path = video_metadata_path_from_config(config)
    video_source = video_metadata_path if video_metadata_path is not None else Path(config["project"]["video_dir"])
    return [
        video_source,
        aligned_turns_path_from_config(config),
        Path(config["project"]["speaker_log_path"]),
    ]


def build_video_metadata_from_directory(config: dict[str, Any]) -> pd.DataFrame:
    """Build dyad/session video metadata from recording_videos filenames."""
    video_dir = Path(config["project"].get("video_dir") or "")
    if not video_dir.exists():
        raise PipelineError(
            "Video metadata table is missing and video_dir does not exist.",
            stage="input_loading",
            likely_cause="Neither project.video_metadata_path nor project.video_dir points to usable video metadata.",
            suggested_fix="Set project.video_dir to 01_pipeline/recording_videos or provide video_metadata_path.",
            affected_file=str(video_dir),
            can_continue=False,
            level="CRITICAL",
        )
    accepted = {ext.lower() for ext in config["project"].get("accepted_video_extensions", [])}
    preference = [str(x).lower() for x in config["project"].get("video_view_preference", ["discussion"])]
    pattern = re.compile(r"(?P<dyad>dyad\d+_\d+)_(?P<condition>piper|cloudy)_(?P<view>.+)$", re.IGNORECASE)
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in sorted(video_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in accepted:
            continue
        match = pattern.match(path.stem)
        if not match:
            continue
        dyad_id = match.group("dyad")
        condition = match.group("condition").lower()
        view = match.group("view").strip("_").lower()
        rank = preference.index(view) if view in preference else len(preference)
        candidates.setdefault((dyad_id, condition), []).append({
            "dyad_id": dyad_id,
            "session_id": condition,
            "video_id": f"{dyad_id}_{condition}",
            "video_path": str(path),
            "condition": condition,
            "view": view,
            "view_rank": rank,
        })
    rows = []
    for key, group in sorted(candidates.items()):
        rows.append(sorted(group, key=lambda row: (row["view_rank"], row["video_path"]))[0])
    if not rows:
        raise PipelineError(
            "No video metadata rows could be generated from video_dir.",
            stage="input_loading",
            likely_cause="No accepted video filenames matched dyad_condition_view naming.",
            suggested_fix="Check filenames or provide an explicit video_metadata_path.",
            affected_file=str(video_dir),
            can_continue=False,
            level="CRITICAL",
        )
    return pd.DataFrame(rows).drop(columns=["view_rank"])


def load_project_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    video_path = video_metadata_path_from_config(config)
    turns_path = aligned_turns_path_from_config(config)
    speaker_path = Path(config["project"]["speaker_log_path"])
    videos = read_table(video_path) if video_path and video_path.exists() else build_video_metadata_from_directory(config)
    transcript = read_table(turns_path)
    speakers = read_table(speaker_path)
    require_columns(videos, config["schema"]["video_metadata_required"], "video metadata", video_path or Path(config["project"]["video_dir"]))
    required_turn_cols = config["schema"].get("aligned_turns_required") or config["schema"].get("aligned_transcript_required")
    require_columns(transcript, required_turn_cols, "aligned turns", turns_path)
    transcript = normalize_aligned_turns(transcript)
    speakers = normalize_speaker_log(speakers)
    require_columns(speakers, config["schema"]["speaker_log_required"], "speaker log", speaker_path)
    transcript = attach_speaker_log_roles(transcript, speakers)

    id_cols = ["dyad_id", "session_id", "video_id", "question_id", "turn_id", "speaker_label"]
    videos = normalize_ids(videos, [col for col in id_cols if col in videos.columns])
    transcript = normalize_ids(transcript, [col for col in id_cols if col in transcript.columns])
    speakers = normalize_ids(speakers, [col for col in id_cols if col in speakers.columns])
    transcript["start_time_sec"] = pd.to_numeric(transcript["start_time_sec"], errors="coerce")
    transcript["end_time_sec"] = pd.to_numeric(transcript["end_time_sec"], errors="coerce")
    return videos, transcript, speakers


def attach_speaker_log_roles(transcript: pd.DataFrame, speakers: pd.DataFrame) -> pd.DataFrame:
    """Attach speaker-log participant roles to turns.

    The aligned turns carry timing and speaker labels. The speaker log is the
    authoritative source for which participant role a diarized speaker label
    refers to in each video/session.
    """
    role_cols = [col for col in ["participant_id", "participant_role"] if col in speakers.columns]
    if not role_cols:
        return transcript

    possible_keys = [
        ["video_id", "speaker_label", "question_id"],
        ["video_id", "speaker_label"],
        ["dyad_id", "session_id", "speaker_label", "question_id"],
        ["dyad_id", "session_id", "speaker_label"],
    ]
    key_cols = next(
        (
            keys for keys in possible_keys
            if all(key in transcript.columns for key in keys)
            and all(key in speakers.columns for key in keys)
        ),
        None,
    )
    if key_cols is None:
        return transcript

    mapping = speakers[key_cols + role_cols].dropna(subset=["speaker_label"]).drop_duplicates(key_cols)
    if mapping.empty:
        return transcript
    mapping = mapping.rename(columns={col: f"speaker_log_{col}" for col in role_cols})
    merged = transcript.merge(mapping, on=key_cols, how="left")
    for col in role_cols:
        log_col = f"speaker_log_{col}"
        if log_col not in merged:
            continue
        if col not in merged:
            merged[col] = merged[log_col]
        else:
            log_values = merged[log_col].astype("string").str.strip()
            use_log = log_values.notna() & (log_values != "")
            merged.loc[use_log, col] = merged.loc[use_log, log_col]
        merged = merged.drop(columns=[log_col])
    return merged


def normalize_speaker_log(speakers: pd.DataFrame) -> pd.DataFrame:
    """Normalize either a long speaker log or the CVS transcription log."""
    speakers = speakers.copy()
    if {"Dyad ID", "Piper_00", "Piper_01", "Cloudy_00", "Cloudy_01"}.issubset(speakers.columns):
        rows = []
        for _, row in speakers.iterrows():
            dyad_id = str(row["Dyad ID"]).strip()
            if not dyad_id or dyad_id.lower() == "nan":
                continue
            for condition, c0, c1, status_col, notes_col in [
                ("piper", "Piper_00", "Piper_01", "Discussion_Piper", "Notes"),
                ("cloudy", "Cloudy_00", "Cloudy_01", "Discussion_Cloudy", "Note"),
            ]:
                for speaker_label, role_col in [("SPEAKER_00", c0), ("SPEAKER_01", c1)]:
                    role = str(row.get(role_col, "")).strip().upper()
                    rows.append({
                        "dyad_id": dyad_id,
                        "session_id": condition,
                        "video_id": f"{dyad_id}_{condition}",
                        "speaker_label": speaker_label,
                        "participant_id": role,
                        "participant_role": role,
                        "segment_valid": str(row.get(status_col, "")).strip().lower() not in {"bad", "exclude", "invalid", "no", "false", "0"},
                        "notes": row.get(notes_col, ""),
                        "Order": row.get("Order", ""),
                    })
        return pd.DataFrame(rows)
    rename_map = {
        "Dyad ID": "dyad_id",
        "Session": "session_id",
        "Condition": "session_id",
        "Speaker": "speaker_label",
        "Role": "participant_role",
        "Participant": "participant_id",
    }
    for old, new in rename_map.items():
        if old in speakers.columns and new not in speakers.columns:
            speakers = speakers.rename(columns={old: new})
    if "video_id" not in speakers.columns and {"dyad_id", "session_id"}.issubset(speakers.columns):
        speakers["video_id"] = speakers["dyad_id"].astype(str) + "_" + speakers["session_id"].astype(str)
    return speakers


def normalize_aligned_turns(turns: pd.DataFrame) -> pd.DataFrame:
    """Normalize align_manual_labels_to_whisperx.py output to video-turn schema."""
    turns = turns.copy()
    rename_map = {
        "question": "question_id",
        "turn_idx": "turn_id",
        "speaker": "speaker_label",
        "text": "utterance_text",
        "t_start": "start_time_sec",
        "t_end": "end_time_sec",
        "dur": "turn_duration_sec",
        "coverage": "alignment_confidence",
        "role": "participant_role",
    }
    for old, new in rename_map.items():
        if old in turns.columns and new not in turns.columns:
            turns = turns.rename(columns={old: new})
    if "session_id" not in turns.columns and "condition" in turns.columns:
        turns["session_id"] = turns["condition"]
    if "video_id" not in turns.columns and {"dyad_id", "session_id"}.issubset(turns.columns):
        turns["video_id"] = turns["dyad_id"].astype(str) + "_" + turns["session_id"].astype(str)
    if "utterance_text" not in turns.columns:
        turns["utterance_text"] = ""
    if "alignment_status" not in turns.columns:
        if "alignment_confidence" in turns.columns:
            turns["alignment_status"] = np.where(
                pd.to_numeric(turns["alignment_confidence"], errors="coerce") >= 0.5,
                "aligned",
                "low_coverage",
            )
        else:
            turns["alignment_status"] = "unknown"
    canonical_required = [
        "dyad_id", "session_id", "video_id", "question_id", "turn_id",
        "speaker_label", "start_time_sec", "end_time_sec",
    ]
    missing = [col for col in canonical_required if col not in turns.columns]
    if missing:
        raise PipelineError(
            f"Aligned turns could not be normalized to required columns: {missing}",
            stage="schema_validation",
            likely_cause="labeled_turns.csv is missing native alignment columns or video/session identifiers.",
            suggested_fix="Regenerate labeled_turns.csv with align_manual_labels_to_whisperx.py or add video metadata fields.",
            can_continue=False,
            level="CRITICAL",
        )
    return turns


def speaker_lookup_from_config(config: dict[str, Any]) -> dict[str, str]:
    mapping = {}
    for participant in config["participants"]:
        mapping[str(participant["speaker_label"])] = str(participant["participant_id"])
    return mapping


# Fixed camera setup, identical in every recording (hard-coded lab standard):
# on screen, participant A (door side) is on the RIGHT and participant B
# (window side) is on the LEFT. Confirmed by mouth-motion QC
# (see scripts/video_behavior/build_participant_side_map.py).
PARTICIPANT_SIDE_STANDARD = {"A": "right", "B": "left"}


def roi_screen_side(roi: list[float]) -> str:
    """Classify an ROI as the left or right half of the frame by its x-center."""
    x_center = (float(roi[0]) + float(roi[2])) / 2.0
    return "right" if x_center >= 0.5 else "left"


def participant_roi_lookup(config: dict[str, Any]) -> dict[str, list[float]]:
    rois = {str(p["participant_id"]): p["roi"] for p in config["participants"]}
    for participant_id, expected_side in PARTICIPANT_SIDE_STANDARD.items():
        roi = rois.get(participant_id)
        if roi is None:
            continue
        actual_side = roi_screen_side(roi)
        if actual_side != expected_side:
            raise PipelineError(
                f"Participant {participant_id} ROI is on the video {actual_side}, "
                f"but the lab standard fixes {participant_id} on the video "
                f"{expected_side} (A = door side = right, B = window side = left).",
                stage="config_loading",
                likely_cause="The config participants section contradicts the fixed camera setup.",
                suggested_fix=(
                    f"Set participant {participant_id} roi to "
                    f"{[0.5, 0.0, 1.0, 1.0] if expected_side == 'right' else [0.0, 0.0, 0.5, 1.0]} "
                    "or remove the participants section to use the enforced default."
                ),
                affected_file="config participants section",
                can_continue=False,
                level="CRITICAL",
            )
    return rois


def validate_project_inputs(
    videos: pd.DataFrame,
    transcript: pd.DataFrame,
    speakers: pd.DataFrame,
    checked_video_meta: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qc_rows: list[dict[str, Any]] = []
    videos = videos.copy()
    transcript = transcript.copy()
    speakers = speakers.copy()

    video_ids = set(videos["video_id"])
    transcript_ids = set(transcript["video_id"])
    speaker_video_ids = set(speakers["video_id"]) if "video_id" in speakers.columns else video_ids
    for video_id in sorted(video_ids):
        if video_id not in transcript_ids:
            qc_rows.append(issue("FAIL", "video", video_id, "Video has no matching transcript alignment."))
        if "video_id" in speakers.columns and video_id not in speaker_video_ids:
            qc_rows.append(issue("FAIL", "video", video_id, "Video has no matching speaker log rows."))
    for video_id in sorted(transcript_ids - video_ids):
        qc_rows.append(issue("FAIL", "video", video_id, "Transcript references a video missing from metadata."))

    expected_sessions = config["validation"].get("expected_sessions_per_dyad")
    if expected_sessions:
        for dyad_id, group in videos.groupby("dyad_id", dropna=False):
            n_sessions = group["session_id"].nunique()
            if n_sessions != int(expected_sessions):
                qc_rows.append(issue(
                    "WARNING", "dyad", dyad_id,
                    f"Dyad has {n_sessions} sessions; expected {expected_sessions}.",
                ))

    expected_questions = config["validation"].get("expected_question_ids")
    if expected_questions:
        expected_questions = {str(q) for q in expected_questions}
        for keys, group in transcript.groupby(["dyad_id", "session_id"], dropna=False):
            observed = set(group["question_id"].astype(str))
            missing = sorted(expected_questions - observed)
            if missing:
                qc_rows.append(issue(
                    "WARNING", "session", "|".join(keys),
                    f"Session is missing expected questions: {missing}.",
                ))

    meta_by_video = checked_video_meta.set_index("video_id").to_dict("index") if not checked_video_meta.empty else {}
    for idx, row in transcript.iterrows():
        unit_id = turn_unit_id(row)
        start = row["start_time_sec"]
        end = row["end_time_sec"]
        if not np.isfinite(start) or not np.isfinite(end):
            qc_rows.append(issue("FAIL", "turn", unit_id, "Turn has missing/non-numeric timestamp."))
            continue
        if start >= end:
            qc_rows.append(issue("FAIL", "turn", unit_id, "Turn start time is not before end time."))
        duration = end - start
        if duration < float(config["validation"]["min_turn_duration_sec"]):
            qc_rows.append(issue("WARNING", "turn", unit_id, "Turn duration is very short."))
        if duration > float(config["validation"]["max_turn_duration_sec"]):
            qc_rows.append(issue("WARNING", "turn", unit_id, "Turn duration is unusually long."))
        meta = meta_by_video.get(row["video_id"])
        if meta and end > float(meta["duration_s"]) + float(config["validation"]["timestamp_tolerance_sec"]):
            qc_rows.append(issue("FAIL", "turn", unit_id, "Turn timestamp extends past video duration."))
        utterance = row.get("utterance_text", "")
        if config["validation"].get("empty_utterance_warning", True) and not str(utterance).strip():
            qc_rows.append(issue("WARNING", "turn", unit_id, "Turn has empty utterance_text."))

    if not config["validation"].get("allow_overlapping_turns", False):
        for keys, group in transcript.sort_values("start_time_sec").groupby(["video_id", "question_id"], dropna=False):
            prev_end = None
            prev_turn = None
            for _, row in group.iterrows():
                if prev_end is not None and row["start_time_sec"] < prev_end:
                    qc_rows.append(issue(
                        "FAIL", "question", "|".join(keys),
                        f"Transcript turns overlap: {prev_turn} and {row['turn_id']}.",
                    ))
                prev_end = row["end_time_sec"]
                prev_turn = row["turn_id"]

    known_speakers = speaker_set(speakers)
    for idx, row in transcript.iterrows():
        if row["speaker_label"] not in known_speakers:
            qc_rows.append(issue("WARNING", "turn", turn_unit_id(row), "Speaker label is not present in speaker logs."))

    q_col = config["validation"].get("speaker_log_question_column")
    if q_col and q_col in speakers.columns:
        speaker_question_keys = set(
            tuple(x)
            for x in speakers[["dyad_id", "session_id", "speaker_label", q_col]]
            .dropna()
            .astype(str)
            .itertuples(index=False, name=None)
        )
        for _, row in transcript.iterrows():
            key = (str(row["dyad_id"]), str(row["session_id"]), str(row["speaker_label"]), str(row["question_id"]))
            if key not in speaker_question_keys:
                qc_rows.append(issue(
                    "WARNING", "turn", turn_unit_id(row),
                    "Transcript speaker/question combination is not present in speaker logs.",
                ))

    validity_col = config["validation"].get("speaker_log_validity_column")
    if validity_col and validity_col in speakers.columns:
        invalid = speakers[
            speakers[validity_col].astype(str).str.lower().isin({"false", "0", "no", "invalid", "exclude"})
        ]
        for _, row in invalid.iterrows():
            unit = "|".join([str(row.get("dyad_id", "")), str(row.get("session_id", "")), str(row.get("speaker_label", ""))])
            note_col = config["validation"].get("speaker_log_notes_column")
            note = row.get(note_col, "") if note_col in speakers.columns else ""
            qc_rows.append(issue("WARNING", "speaker_log", unit, f"Speaker log marks segment invalid. {note}".strip()))

    for col_name, label in [
        (config["validation"].get("speaker_log_participant_column"), "participant identifier"),
        (config["validation"].get("speaker_log_role_column"), "participant role"),
    ]:
        if col_name and col_name in speakers.columns:
            missing = speakers[speakers[col_name].isna() | (speakers[col_name].astype(str).str.strip() == "")]
            for _, row in missing.iterrows():
                unit = "|".join([str(row.get("dyad_id", "")), str(row.get("session_id", "")), str(row.get("speaker_label", ""))])
                qc_rows.append(issue("WARNING", "speaker_log", unit, f"Speaker log is missing {label}."))

    qc_df = pd.DataFrame(qc_rows)
    return videos, transcript, speakers, qc_df


def issue(status: str, unit_type: str, unit_id: str, note: str) -> dict[str, Any]:
    return {"qc_status": status, "unit_type": unit_type, "unit_id": unit_id, "qc_notes": note}


def turn_unit_id(row: pd.Series) -> str:
    return "|".join([
        str(row.get("dyad_id", "")),
        str(row.get("session_id", "")),
        str(row.get("video_id", "")),
        str(row.get("question_id", "")),
        str(row.get("turn_id", "")),
        str(row.get("speaker_label", "")),
    ])


def speaker_set(speakers: pd.DataFrame) -> set[str]:
    return set(speakers["speaker_label"].dropna().astype(str).str.strip())


def inspect_videos(videos: pd.DataFrame, config: dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    rows = []
    accepted = {ext.lower() for ext in config["project"].get("accepted_video_extensions", [])}
    for _, row in videos.iterrows():
        video_path = Path(row["video_path"])
        if video_path.suffix.lower() not in accepted:
            logger.warning(
                f"Unsupported video extension for {video_path}.",
                extra={"stage": "metadata_inspection"},
            )
            continue
        meta = inspect_video(video_path)
        meta.update({
            "dyad_id": row["dyad_id"],
            "session_id": row["session_id"],
            "video_id": row["video_id"],
            "condition": row.get("condition", ""),
            "participant_A": row.get("participant_A", ""),
            "participant_B": row.get("participant_B", ""),
        })
        rows.append(meta)
    return pd.DataFrame(rows)


def sample_frame_indices(start_sec: float, end_sec: float, fps: float, sample_fps: float, max_frames: int | None) -> list[int]:
    start_frame = max(0, int(math.floor(start_sec * fps)))
    end_frame = max(start_frame, int(math.ceil(end_sec * fps)))
    step = max(1, int(round(fps / sample_fps)))
    indices = list(range(start_frame, end_frame + 1, step))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def import_video_stack() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise PipelineError(
            "MediaPipe/OpenCV dependency is missing.",
            stage="dependency_check",
            likely_cause="mediapipe or opencv-python is not installed.",
            suggested_fix="Recreate the conda environment before processing video.",
            can_continue=False,
            level="CRITICAL",
        ) from exc
    try:
        pose_solution = mp.solutions.pose
        hands_solution = mp.solutions.hands
    except AttributeError:
        try:
            pose_solution = importlib.import_module("mediapipe.python.solutions.pose")
            hands_solution = importlib.import_module("mediapipe.python.solutions.hands")
        except Exception as exc:
            version = getattr(mp, "__version__", "unknown")
            raise PipelineError(
                "MediaPipe imported, but the Pose/Hands solutions API is unavailable.",
                stage="dependency_check",
                likely_cause=(
                    f"The installed mediapipe package/version ({version}) does not expose "
                    "mp.solutions or mediapipe.python.solutions."
                ),
                suggested_fix=(
                    "Install a MediaPipe build with the legacy solutions API, for example "
                    "`python -m pip install --upgrade --force-reinstall mediapipe`, or run "
                    "the pipeline in a Python 3.10/3.11 conda environment where MediaPipe "
                    "solutions are supported."
                ),
                can_continue=False,
                level="CRITICAL",
            ) from exc
    return cv2, pose_solution, hands_solution


def roi_to_pixels(roi: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        int(round(roi[0] * width)),
        int(round(roi[1] * height)),
        int(round(roi[2] * width)),
        int(round(roi[3] * height)),
    )


def process_video_turns(
    video_meta: pd.Series,
    turns: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del logger
    if not config["mediapipe"].get("enabled", True):
        return (
            empty_feature_rows(turns, "MediaPipe disabled in config."),
            pd.DataFrame(columns=ARM_FRAME_COLUMNS),
        )

    cv2, pose_solution, hands_solution = import_video_stack()
    video_path = Path(video_meta["video_path"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError(
            "Could not open video for turn-aligned processing.",
            stage="main_video_analysis",
            likely_cause="The video is unreadable or codec support is missing.",
            suggested_fix="Re-export as H.264 mp4 or check file permissions.",
            affected_file=str(video_path),
            can_continue=True,
        )

    participant_rois = participant_roi_lookup(config)
    speaker_to_participant = speaker_lookup_from_config(config)
    pose_model = None
    hands_model = None
    if config["mediapipe"].get("pose_enabled", True):
        pose_model = pose_solution.Pose(
            static_image_mode=True,
            model_complexity=int(config["mediapipe"]["pose_model_complexity"]),
            enable_segmentation=False,
            min_detection_confidence=float(config["mediapipe"]["pose_min_detection_confidence"]),
        )
    if config["mediapipe"].get("hands_enabled", True):
        hands_model = hands_solution.Hands(
            static_image_mode=True,
            max_num_hands=int(config["mediapipe"]["hands_max_num_hands"]),
            min_detection_confidence=float(config["mediapipe"]["hands_min_detection_confidence"]),
        )

    turn_rows = []
    arm_rows = []
    upper_limb_enabled = bool(config.get("upper_limb", {}).get("enabled", True)) and pose_model is not None
    try:
        for _, turn in turns.iterrows():
            participant_id = (
                speaker_to_participant.get(str(turn["speaker_label"]))
                or speaker_to_participant.get(str(turn.get("participant_role", "")))
                or str(turn.get("participant_role", "")).strip()
                or str(turn["speaker_label"])
            )
            if float(turn["start_time_sec"]) >= float(turn["end_time_sec"]):
                turn_rows.append(base_turn_row(
                    turn,
                    video_meta,
                    participant_id,
                    "FAIL",
                    "Invalid turn interval from labeled_turns.csv; video features were not extracted.",
                ))
                continue
            roi = participant_rois.get(participant_id)
            if roi is None:
                turn_rows.append(base_turn_row(turn, video_meta, participant_id, "FAIL", "No participant ROI for speaker_label."))
                continue
            frame_indices = sample_frame_indices(
                float(turn["start_time_sec"]),
                float(turn["end_time_sec"]),
                float(video_meta["fps"]),
                float(config["sampling"]["sample_fps"]),
                config["sampling"].get("max_frames_per_turn"),
            )
            speaker_frame_features = []
            speaker_previous_wrist: tuple[float, float, float] | None = None
            arm_previous_wrist: dict[str, dict[str, tuple[float, float, float]]] = {
                participant_key: {} for participant_key in participant_rois
            }
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                height, width = frame_bgr.shape[:2]
                timestamp_sec = frame_index / float(video_meta["fps"])
                participant_items = participant_rois.items() if upper_limb_enabled else [(participant_id, roi)]
                for frame_participant_id, frame_roi in participant_items:
                    x0, y0, x1, y1 = roi_to_pixels(frame_roi, width, height)
                    crop = frame_bgr[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    crop_h, crop_w = crop.shape[:2]
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pose_points: dict[str, tuple[float, float, float]] = {}
                    pose_detected = False

                    if pose_model is not None:
                        pose_result = pose_model.process(rgb)
                        if pose_result.pose_landmarks:
                            pose_detected = True
                            for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
                                name = POSE_LANDMARK_NAMES[idx] if idx < len(POSE_LANDMARK_NAMES) else f"pose_{idx}"
                                pose_points[name] = (
                                    x0 + landmark.x * crop_w,
                                    y0 + landmark.y * crop_h,
                                    float(getattr(landmark, "visibility", np.nan)),
                                )

                    if upper_limb_enabled:
                        conversational_role = "speaker" if frame_participant_id == participant_id else "listener"
                        base = frame_base_from_turn(
                            turn,
                            video_meta,
                            frame_participant_id,
                            conversational_role,
                            int(frame_index),
                            timestamp_sec,
                        )
                        arm_rows.append(compute_upper_limb_frame_row(
                            base,
                            pose_points,
                            arm_previous_wrist.setdefault(frame_participant_id, {}),
                            config,
                        ))

                    if frame_participant_id != participant_id:
                        continue

                    hand_detected = False
                    wrist_speeds = []
                    hand_face_distances = []
                    hand_shoulder_distances = []
                    if hands_model is not None:
                        hand_result = hands_model.process(rgb)
                        for hand_landmarks in hand_result.multi_hand_landmarks or []:
                            hand_detected = True
                            wrist = hand_landmarks.landmark[0]
                            wrist_x = x0 + wrist.x * crop_w
                            wrist_y = y0 + wrist.y * crop_h
                            if speaker_previous_wrist is not None:
                                dt = timestamp_sec - speaker_previous_wrist[2]
                                if dt > 0:
                                    wrist_speeds.append(math.hypot(wrist_x - speaker_previous_wrist[0], wrist_y - speaker_previous_wrist[1]) / dt)
                            speaker_previous_wrist = (wrist_x, wrist_y, timestamp_sec)
                            if "nose" in pose_points:
                                hand_face_distances.append(math.hypot(wrist_x - pose_points["nose"][0], wrist_y - pose_points["nose"][1]))
                            shoulders = [pose_points.get("left_shoulder"), pose_points.get("right_shoulder")]
                            shoulder_dist = [
                                math.hypot(wrist_x - s[0], wrist_y - s[1])
                                for s in shoulders if s is not None
                            ]
                            if shoulder_dist:
                                hand_shoulder_distances.append(float(np.mean(shoulder_dist)))

                    speaker_frame_features.append({
                        "pose_detected": pose_detected,
                        "hand_detected": hand_detected,
                        "wrist_speed": np.nanmean(wrist_speeds) if wrist_speeds else np.nan,
                        "hand_to_face": np.nanmean(hand_face_distances) if hand_face_distances else np.nan,
                        "hand_to_shoulder": np.nanmean(hand_shoulder_distances) if hand_shoulder_distances else np.nan,
                    })
            turn_rows.append(summarize_turn_features(turn, video_meta, participant_id, frame_indices, speaker_frame_features, config))
    finally:
        cap.release()
        if pose_model is not None:
            pose_model.close()
        if hands_model is not None:
            hands_model.close()
    return (
        pd.DataFrame(turn_rows, columns=TURN_OUTPUT_COLUMNS),
        pd.DataFrame(arm_rows, columns=ARM_FRAME_COLUMNS),
    )


def empty_feature_rows(turns: pd.DataFrame, note: str) -> pd.DataFrame:
    rows = []
    for _, turn in turns.iterrows():
        rows.append(base_turn_row(turn, pd.Series(dtype=object), str(turn["speaker_label"]), "WARNING", note))
    return pd.DataFrame(rows, columns=TURN_OUTPUT_COLUMNS)


def base_turn_row(
    turn: pd.Series,
    video_meta: pd.Series,
    participant_id: str,
    qc_status: str,
    qc_notes: str,
) -> dict[str, Any]:
    start = float(turn["start_time_sec"])
    end = float(turn["end_time_sec"])
    return {
        "dyad_id": turn["dyad_id"],
        "session_id": turn["session_id"],
        "video_id": turn["video_id"],
        "condition": video_meta.get("condition", ""),
        "question_id": turn["question_id"],
        "turn_id": turn["turn_id"],
        "speaker_label": turn["speaker_label"],
        "participant_id": participant_id,
        "utterance_text": turn.get("utterance_text", ""),
        "turn_start_sec": start,
        "turn_end_sec": end,
        "turn_duration_sec": end - start,
        "n_sampled_frames": 0,
        "pose_detection_rate": np.nan,
        "hand_detection_rate": np.nan,
        "mean_wrist_speed_px_s": np.nan,
        "max_wrist_speed_px_s": np.nan,
        "mean_hand_to_face_distance_px": np.nan,
        "mean_hand_to_shoulder_distance_px": np.nan,
        "gesture_frame_rate": np.nan,
        "openface_mean_AU12_r": np.nan,
        "openface_mean_AU06_r": np.nan,
        "openface_mean_AU04_r": np.nan,
        "openface_mean_pose_Rx": np.nan,
        "openface_mean_pose_Ry": np.nan,
        "openface_mean_pose_Rz": np.nan,
        "missingness_rate": 1.0,
        "qc_status": qc_status,
        "qc_notes": qc_notes,
    }


def summarize_turn_features(
    turn: pd.Series,
    video_meta: pd.Series,
    participant_id: str,
    frame_indices: list[int],
    frame_features: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    row = base_turn_row(turn, video_meta, participant_id, "PASS", "")
    n_expected = len(frame_indices)
    n_observed = len(frame_features)
    pose_rate = np.mean([f["pose_detected"] for f in frame_features]) if frame_features else 0.0
    hand_rate = np.mean([f["hand_detected"] for f in frame_features]) if frame_features else 0.0
    wrist_speed = np.array([f["wrist_speed"] for f in frame_features], dtype=float) if frame_features else np.array([])
    hand_to_face = np.array([f["hand_to_face"] for f in frame_features], dtype=float) if frame_features else np.array([])
    hand_to_shoulder = np.array([f["hand_to_shoulder"] for f in frame_features], dtype=float) if frame_features else np.array([])
    gesture_frame_rate = np.mean(wrist_speed > float(config["features"]["motion_speed_threshold_px_s"])) if wrist_speed.size else np.nan
    missingness = 1.0 - (n_observed / n_expected) if n_expected else 1.0
    notes = []
    status = "PASS"
    if pose_rate < float(config["qc"]["pose_detection_rate_warning"]):
        status = "WARNING"
        notes.append("Low pose detection rate.")
    if hand_rate < float(config["qc"]["hand_detection_rate_warning"]):
        status = "WARNING"
        notes.append("Low hand detection rate.")
    if missingness > float(config["qc"]["high_missingness_warning"]):
        status = "WARNING"
        notes.append("High sampled-frame missingness.")
    row.update({
        "n_sampled_frames": n_observed,
        "pose_detection_rate": pose_rate,
        "hand_detection_rate": hand_rate,
        "mean_wrist_speed_px_s": finite_mean(wrist_speed),
        "max_wrist_speed_px_s": finite_max(wrist_speed),
        "mean_hand_to_face_distance_px": finite_mean(hand_to_face),
        "mean_hand_to_shoulder_distance_px": finite_mean(hand_to_shoulder),
        "gesture_frame_rate": gesture_frame_rate,
        "missingness_rate": missingness,
        "qc_status": status,
        "qc_notes": " ".join(notes),
    })
    return row


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def finite_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else np.nan


def attach_openface_features(turn_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if not config["openface"].get("enabled", False):
        return turn_df
    input_dir = config["openface"].get("input_csv_dir")
    if not input_dir:
        return turn_df.assign(qc_notes=turn_df["qc_notes"].fillna("") + " OpenFace enabled but input_csv_dir not configured.")
    updated = turn_df.copy()
    for video_id, group in updated.groupby("video_id", dropna=False):
        csv_path = Path(input_dir) / f"{video_id}.csv"
        if not csv_path.exists():
            mask = updated["video_id"] == video_id
            updated.loc[mask, "qc_status"] = updated.loc[mask, "qc_status"].replace("PASS", "WARNING")
            updated.loc[mask, "qc_notes"] = updated.loc[mask, "qc_notes"].fillna("") + " Missing OpenFace CSV."
            continue
        raw = pd.read_csv(csv_path)
        raw.columns = [c.strip() for c in raw.columns]
        if "confidence" in raw.columns:
            raw = raw[raw["confidence"] >= float(config["openface"]["confidence_min"])]
        if config["openface"].get("success_required", True) and "success" in raw.columns:
            raw = raw[raw["success"] == 1]
        if "timestamp" not in raw.columns:
            continue
        for idx, turn in group.iterrows():
            segment = raw[(raw["timestamp"] >= turn["turn_start_sec"]) & (raw["timestamp"] <= turn["turn_end_sec"])]
            for col, out_col in [
                ("AU12_r", "openface_mean_AU12_r"),
                ("AU06_r", "openface_mean_AU06_r"),
                ("AU04_r", "openface_mean_AU04_r"),
                ("pose_Rx", "openface_mean_pose_Rx"),
                ("pose_Ry", "openface_mean_pose_Ry"),
                ("pose_Rz", "openface_mean_pose_Rz"),
            ]:
                if col in segment.columns:
                    updated.loc[idx, out_col] = segment[col].mean()
    return updated


def merge_turn_qc(turn_df: pd.DataFrame, input_qc: pd.DataFrame) -> pd.DataFrame:
    if input_qc.empty:
        return turn_df
    turn_df = turn_df.copy()
    turn_qc = input_qc[input_qc["unit_type"] == "turn"].copy()
    notes_by_unit = turn_qc.groupby("unit_id")["qc_notes"].apply(lambda x: " ".join(x)).to_dict()
    fail_units = set(turn_qc[turn_qc["qc_status"] == "FAIL"]["unit_id"])
    warning_units = set(turn_qc[turn_qc["qc_status"] == "WARNING"]["unit_id"])
    for idx, row in turn_df.iterrows():
        unit = turn_unit_id(pd.Series({
            "dyad_id": row["dyad_id"],
            "session_id": row["session_id"],
            "video_id": row["video_id"],
            "question_id": row["question_id"],
            "turn_id": row["turn_id"],
            "speaker_label": row["speaker_label"],
        }))
        if unit in notes_by_unit:
            turn_df.loc[idx, "qc_notes"] = (str(row.get("qc_notes", "")) + " " + notes_by_unit[unit]).strip()
        if unit in fail_units:
            turn_df.loc[idx, "qc_status"] = "FAIL"
        elif unit in warning_units and row["qc_status"] == "PASS":
            turn_df.loc[idx, "qc_status"] = "WARNING"
    return turn_df


def aggregate_question(turn_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_cols = video_feature_columns(turn_df)
    for keys, group in turn_df.groupby(["dyad_id", "session_id", "video_id", "question_id"], dropna=False):
        row = {
            "dyad_id": keys[0],
            "session_id": keys[1],
            "video_id": keys[2],
            "question_id": keys[3],
            "question_start_sec": group["turn_start_sec"].min(),
            "question_end_sec": group["turn_end_sec"].max(),
            "question_duration_sec": group["turn_end_sec"].max() - group["turn_start_sec"].min(),
            "number_of_turns": len(group),
            "number_of_speakers": group["speaker_label"].nunique(),
            "speaker_turn_balance": speaker_balance(group["speaker_label"]),
            "mean_missingness_rate": group["missingness_rate"].mean(),
            "qc_status": combine_status(group["qc_status"]),
            "qc_notes": " ".join(sorted(set(" ".join(group["qc_notes"].fillna("")).split(". ")))).strip(),
        }
        for col in feature_cols:
            row[f"mean_{col}"] = group[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_speaker(turn_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = video_feature_columns(turn_df)
    rows = []
    for keys, group in turn_df.groupby(["dyad_id", "session_id", "video_id", "speaker_label", "participant_id"], dropna=False):
        row = {
            "dyad_id": keys[0],
            "session_id": keys[1],
            "video_id": keys[2],
            "speaker_label": keys[3],
            "participant_id": keys[4],
            "number_of_turns": len(group),
            "total_speaking_time_sec": group["turn_duration_sec"].sum(),
            "mean_missingness_rate": group["missingness_rate"].mean(),
            "qc_status": combine_status(group["qc_status"]),
            "qc_notes": " ".join(sorted(set(" ".join(group["qc_notes"].fillna("")).split(". ")))).strip(),
        }
        for col in feature_cols:
            row[f"mean_{col}"] = group[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_session(turn_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = video_feature_columns(turn_df)
    rows = []
    for keys, group in turn_df.groupby(["dyad_id", "session_id", "video_id"], dropna=False):
        row = {
            "dyad_id": keys[0],
            "session_id": keys[1],
            "video_id": keys[2],
            "number_of_questions": group["question_id"].nunique(),
            "number_of_turns": len(group),
            "number_of_speakers": group["speaker_label"].nunique(),
            "session_start_sec": group["turn_start_sec"].min(),
            "session_end_sec": group["turn_end_sec"].max(),
            "session_duration_sec": group["turn_end_sec"].max() - group["turn_start_sec"].min(),
            "mean_missingness_rate": group["missingness_rate"].mean(),
            "qc_status": combine_status(group["qc_status"]),
            "qc_notes": " ".join(sorted(set(" ".join(group["qc_notes"].fillna("")).split(". ")))).strip(),
        }
        for col in feature_cols:
            row[f"mean_{col}"] = group[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def cross_session_comparison(turn_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = video_feature_columns(turn_df)
    grouped = turn_df.groupby(["dyad_id", "session_id", "question_id", "speaker_label"], dropna=False)[feature_cols].mean().reset_index()
    rows = []
    for keys, group in grouped.groupby(["dyad_id", "question_id", "speaker_label"], dropna=False):
        sessions = sorted(group["session_id"].unique())
        if len(sessions) < 2:
            continue
        first = group[group["session_id"] == sessions[0]].iloc[0]
        second = group[group["session_id"] == sessions[1]].iloc[0]
        row = {
            "dyad_id": keys[0],
            "question_id": keys[1],
            "speaker_label": keys[2],
            "session_1": sessions[0],
            "session_2": sessions[1],
        }
        for col in feature_cols:
            row[f"{col}_session_1"] = first[col]
            row[f"{col}_session_2"] = second[col]
            row[f"{col}_change"] = second[col] - first[col]
        rows.append(row)
    return pd.DataFrame(rows)


def video_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "dyad_id", "session_id", "video_id", "condition", "question_id", "turn_id",
        "speaker_label", "participant_id", "utterance_text", "qc_status", "qc_notes",
    }
    return [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def speaker_balance(labels: pd.Series) -> float:
    counts = labels.value_counts()
    if len(counts) <= 1 or counts.sum() == 0:
        return 0.0
    proportions = counts / counts.sum()
    return float(1.0 - proportions.max())


def combine_status(statuses: pd.Series) -> str:
    values = set(statuses.dropna())
    if "FAIL" in values:
        return "FAIL"
    if "WARNING" in values:
        return "WARNING"
    return "PASS"


def build_qc_tables(turn_df: pd.DataFrame, question_df: pd.DataFrame, speaker_df: pd.DataFrame, input_qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    turn_qc = turn_df[[
        "dyad_id", "session_id", "video_id", "question_id", "turn_id",
        "speaker_label", "qc_status", "qc_notes", "pose_detection_rate",
        "hand_detection_rate", "missingness_rate",
    ]].copy()
    if not input_qc.empty:
        input_turn = input_qc[input_qc["unit_type"] == "turn"]
        if not input_turn.empty:
            turn_qc = pd.concat([turn_qc, input_turn], ignore_index=True, sort=False)
    question_qc = question_df[[
        "dyad_id", "session_id", "video_id", "question_id", "qc_status",
        "qc_notes", "mean_missingness_rate",
    ]].copy()
    speaker_qc = speaker_df[[
        "dyad_id", "session_id", "video_id", "speaker_label", "participant_id",
        "qc_status", "qc_notes", "mean_missingness_rate",
    ]].copy()
    return turn_qc, question_qc, speaker_qc


def write_alignment_review(dirs: TurnRunDirs, transcript: pd.DataFrame, coverage_threshold: float = 0.5) -> Path:
    """Write a compact review file for turns likely needing label/alignment checks."""
    df = transcript.copy()
    df["alignment_confidence"] = pd.to_numeric(df.get("alignment_confidence", np.nan), errors="coerce")
    bad_mask = (
        (df["start_time_sec"] >= df["end_time_sec"])
        | (df["alignment_confidence"].fillna(1.0) < coverage_threshold)
        | (df.get("alignment_status", "").astype(str).str.lower() == "low_coverage")
    )
    review = df.loc[bad_mask].copy()
    if review.empty:
        review = pd.DataFrame(columns=[
            "dyad_id", "session_id", "video_id", "question_id", "turn_id",
            "speaker_label", "participant_role", "start_time_sec", "end_time_sec",
            "turn_duration_sec", "alignment_confidence", "alignment_status",
            "utterance_text", "previous_text", "next_text", "review_reason",
        ])
    else:
        review = review.sort_values(["video_id", "turn_id"]).copy()
        previous_text = []
        next_text = []
        for idx, row in review.iterrows():
            same_video = df[df["video_id"] == row["video_id"]].sort_values("turn_id")
            turn_id = str(row["turn_id"])
            positions = same_video.index[same_video["turn_id"].astype(str) == turn_id].tolist()
            if not positions:
                previous_text.append("")
                next_text.append("")
                continue
            pos = same_video.index.get_loc(positions[0])
            previous_text.append(same_video.iloc[pos - 1]["utterance_text"] if pos > 0 else "")
            next_text.append(same_video.iloc[pos + 1]["utterance_text"] if pos < len(same_video) - 1 else "")
        review["previous_text"] = previous_text
        review["next_text"] = next_text
        reasons = []
        for _, row in review.iterrows():
            row_reasons = []
            if row["start_time_sec"] >= row["end_time_sec"]:
                row_reasons.append("start_time_sec >= end_time_sec")
            if pd.notna(row["alignment_confidence"]) and row["alignment_confidence"] < coverage_threshold:
                row_reasons.append(f"alignment_confidence < {coverage_threshold}")
            if str(row.get("alignment_status", "")).lower() == "low_coverage":
                row_reasons.append("alignment_status low_coverage")
            reasons.append("; ".join(row_reasons))
        review["review_reason"] = reasons
        keep = [
            "dyad_id", "session_id", "video_id", "question_id", "turn_id",
            "speaker_label", "participant_role", "start_time_sec", "end_time_sec",
            "turn_duration_sec", "alignment_confidence", "alignment_status",
            "utterance_text", "previous_text", "next_text", "review_reason",
        ]
        review = review[[col for col in keep if col in review.columns]]
    path = dirs.qc / "alignment_label_review.csv"
    atomic_write_csv(path, review)
    return path


def write_qc_report(path: Path, turn_qc: pd.DataFrame, question_qc: pd.DataFrame, speaker_qc: pd.DataFrame, input_qc: pd.DataFrame) -> None:
    def table(df: pd.DataFrame, max_rows: int = 50) -> str:
        if df.empty:
            return "<p>No rows.</p>"
        return df.head(max_rows).to_html(index=False, escape=True)

    parts = [
        "<html><head><meta charset='utf-8'><title>Video QC Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px} table{border-collapse:collapse;font-size:12px} th,td{border:1px solid #ccc;padding:4px 6px} th{background:#eee}</style>",
        "</head><body>",
        "<h1>Turn-Aligned Video QC Report</h1>",
        "<h2>Status Counts</h2>",
        table(pd.DataFrame({
            "table": ["turn", "question", "speaker"],
            "PASS": [
                int((turn_qc["qc_status"] == "PASS").sum()) if "qc_status" in turn_qc else 0,
                int((question_qc["qc_status"] == "PASS").sum()) if "qc_status" in question_qc else 0,
                int((speaker_qc["qc_status"] == "PASS").sum()) if "qc_status" in speaker_qc else 0,
            ],
            "WARNING": [
                int((turn_qc["qc_status"] == "WARNING").sum()) if "qc_status" in turn_qc else 0,
                int((question_qc["qc_status"] == "WARNING").sum()) if "qc_status" in question_qc else 0,
                int((speaker_qc["qc_status"] == "WARNING").sum()) if "qc_status" in speaker_qc else 0,
            ],
            "FAIL": [
                int((turn_qc["qc_status"] == "FAIL").sum()) if "qc_status" in turn_qc else 0,
                int((question_qc["qc_status"] == "FAIL").sum()) if "qc_status" in question_qc else 0,
                int((speaker_qc["qc_status"] == "FAIL").sum()) if "qc_status" in speaker_qc else 0,
            ],
        })),
        "<h2>Input Validation Issues</h2>",
        table(input_qc),
        "<h2>Problematic Turns</h2>",
        table(turn_qc[turn_qc["qc_status"].isin(["WARNING", "FAIL"])] if "qc_status" in turn_qc else turn_qc),
        "<h2>Problematic Questions</h2>",
        table(question_qc[question_qc["qc_status"].isin(["WARNING", "FAIL"])] if "qc_status" in question_qc else question_qc),
        "<h2>Problematic Speakers</h2>",
        table(speaker_qc[speaker_qc["qc_status"].isin(["WARNING", "FAIL"])] if "qc_status" in speaker_qc else speaker_qc),
        "</body></html>",
    ]
    atomic_write_text(path, "\n".join(parts))


def write_manifest(dirs: TurnRunDirs, config_path: Path, input_paths: list[Path], output_paths: list[Path], config: dict[str, Any]) -> None:
    manifest = {
        "run_dir": str(dirs.root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "software_versions": get_software_versions(config),
        "inputs": [file_record(p, "source_input") for p in input_paths if p.exists()],
        "outputs": [file_record(p, "pipeline_output") for p in output_paths if p.exists()],
    }
    atomic_write_json(dirs.root / "run_manifest.json", manifest)


def run_pipeline(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = load_yaml_config(config_path)
        if args.fail_fast:
            config["run"]["fail_fast"] = True
        if args.continue_on_warning:
            config["run"]["continue_on_warning"] = True
        dirs = build_run_dirs(config)
        ensure_run_dirs(dirs, overwrite=bool(config["run"].get("overwrite")))
        logger = setup_logging(dirs.logs / "run.log", args.verbose)
        save_yaml(dirs.root / "config_used.yaml", config)

        videos, transcript, speakers = load_project_inputs(config)
        logger.info("Loaded video metadata, labeled turns, and speaker logs.", extra={"stage": "input_loading"})
        checked_meta = inspect_videos(videos, config, logger)
        videos, transcript, speakers, input_qc = validate_project_inputs(videos, transcript, speakers, checked_meta, config)
        atomic_write_csv(dirs.metadata / "video_metadata_checked.csv", checked_meta)
        atomic_write_csv(dirs.metadata / "labeled_turns_checked.csv", transcript)
        atomic_write_csv(dirs.metadata / "speaker_logs_checked.csv", speakers)

        if args.dry_run:
            atomic_write_csv(dirs.qc / "input_validation_qc.csv", input_qc)
            alignment_review_path = write_alignment_review(dirs, transcript)
            write_manifest(
                dirs,
                config_path,
                source_input_paths_from_config(config),
                [
                    dirs.qc / "input_validation_qc.csv",
                    alignment_review_path,
                    dirs.metadata / "video_metadata_checked.csv",
                    dirs.metadata / "labeled_turns_checked.csv",
                    dirs.metadata / "speaker_logs_checked.csv",
                ],
                config,
            )
            logger.info(f"Dry run complete: {dirs.root}", extra={"stage": "dry_run"})
            return 0

        per_video_turns = []
        per_video_arm_frames = []
        for _, video_meta in checked_meta.iterrows():
            video_turns = transcript[transcript["video_id"] == video_meta["video_id"]].copy()
            if video_turns.empty:
                continue
            logger.info(f"Processing turn intervals for {video_meta['video_id']}.", extra={"stage": "main_video_analysis"})
            try:
                turn_part, arm_frame_part = process_video_turns(video_meta, video_turns, config, logger)
                per_video_turns.append(turn_part)
                per_video_arm_frames.append(arm_frame_part)
            except PipelineError as exc:
                log_pipeline_error(logger, exc)
                if config["run"].get("fail_fast") or not exc.can_continue:
                    raise
        if per_video_turns:
            turn_df = pd.concat(per_video_turns, ignore_index=True)
        else:
            turn_df = pd.DataFrame(columns=TURN_OUTPUT_COLUMNS)
        if per_video_arm_frames:
            arm_frame_df = pd.concat(per_video_arm_frames, ignore_index=True)
        else:
            arm_frame_df = pd.DataFrame(columns=ARM_FRAME_COLUMNS)
        turn_df = attach_openface_features(turn_df, config)
        turn_df = merge_turn_qc(turn_df, input_qc)

        arm_turn_role_df = summarize_turn_role_arm_features(arm_frame_df, config)
        participant_arm_df = aggregate_participant_arm_summary(arm_frame_df, config)
        participant_role_arm_df = aggregate_participant_role_arm_summary(arm_turn_role_df)
        speaker_listener_arm_df = speaker_listener_arm_comparison(participant_role_arm_df)
        question_df = aggregate_question(turn_df)
        speaker_df = aggregate_speaker(turn_df)
        session_df = aggregate_session(turn_df)
        cross_session_df = cross_session_comparison(turn_df)
        turn_qc, question_qc, speaker_qc = build_qc_tables(turn_df, question_df, speaker_df, input_qc)
        alignment_review_path = write_alignment_review(dirs, transcript)

        per_turn_path = dirs.results / "per_turn_video_features.csv"
        question_path = dirs.results / "per_question_summary.csv"
        speaker_path = dirs.results / "per_speaker_summary.csv"
        session_path = dirs.results / "per_session_summary.csv"
        cross_session_path = dirs.results / "cross_session_comparison.csv"
        arm_frame_path = dirs.results / "arm_frame_features.csv"
        arm_turn_role_path = dirs.results / "per_turn_arm_role_features.csv"
        participant_arm_path = dirs.results / "per_participant_arm_summary.csv"
        participant_role_arm_path = dirs.results / "per_participant_role_arm_summary.csv"
        speaker_listener_arm_path = dirs.results / "speaker_listener_arm_comparison.csv"
        turn_qc_path = dirs.qc / "turn_level_qc.csv"
        question_qc_path = dirs.qc / "question_level_qc.csv"
        speaker_qc_path = dirs.qc / "speaker_level_qc.csv"
        qc_report_path = dirs.qc / "qc_report.html"
        output_paths = [
            per_turn_path,
            question_path,
            speaker_path,
            session_path,
            cross_session_path,
            arm_frame_path,
            arm_turn_role_path,
            participant_arm_path,
            participant_role_arm_path,
            speaker_listener_arm_path,
            turn_qc_path,
            question_qc_path,
            speaker_qc_path,
            alignment_review_path,
            qc_report_path,
            dirs.root / "config_used.yaml",
            dirs.root / "run_manifest.json",
            dirs.metadata / "video_metadata_checked.csv",
            dirs.metadata / "labeled_turns_checked.csv",
            dirs.metadata / "speaker_logs_checked.csv",
        ]
        atomic_write_csv(per_turn_path, turn_df)
        atomic_write_csv(question_path, question_df)
        atomic_write_csv(speaker_path, speaker_df)
        atomic_write_csv(session_path, session_df)
        atomic_write_csv(cross_session_path, cross_session_df)
        atomic_write_csv(arm_frame_path, arm_frame_df)
        atomic_write_csv(arm_turn_role_path, arm_turn_role_df)
        atomic_write_csv(participant_arm_path, participant_arm_df)
        atomic_write_csv(participant_role_arm_path, participant_role_arm_df)
        atomic_write_csv(speaker_listener_arm_path, speaker_listener_arm_df)
        atomic_write_csv(turn_qc_path, turn_qc)
        atomic_write_csv(question_qc_path, question_qc)
        atomic_write_csv(speaker_qc_path, speaker_qc)
        if config["qc"].get("save_html_report", True):
            write_qc_report(qc_report_path, turn_qc, question_qc, speaker_qc, input_qc)
        write_manifest(
            dirs,
            config_path,
            source_input_paths_from_config(config),
            output_paths,
            config,
        )
        logger.info(f"Turn-aligned video run complete: {dirs.root}", extra={"stage": "final_report"})
        return 0
    except PipelineError as exc:
        log_pipeline_error(logging.getLogger("video_behavior"), exc)
        return 1
    except PermissionError as exc:
        print(
            "CRITICAL [write_error] Permission or write error. "
            f"Suggested fix: choose a writable output_root. Details: {exc}",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("ERROR [interrupted] Run interrupted. No raw inputs were modified.", file=sys.stderr)
        return 130


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run turn-aligned CVS video behavior analysis.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate source tables and metadata without video feature extraction.")
    parser.add_argument("--verbose", action="store_true", help="Print debug-level logs to the terminal.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first critical video/unit error.")
    parser.add_argument("--continue-on-warning", action="store_true", help="Continue when QC warnings occur.")
    return parser


if __name__ == "__main__":
    raise SystemExit(run_pipeline(build_arg_parser().parse_args()))

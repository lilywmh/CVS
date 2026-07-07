#!/usr/bin/env python3
"""
Upper-limb posture and movement proxies for CVS discussion videos.

The functions in this module intentionally produce transparent behavioral
proxies from MediaPipe Pose landmarks. They classify shoulder-elbow-wrist
geometry and wrist movement into auditable states; they do not infer emotion,
intent, or psychological traits.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


ARM_SIDES = ("left", "right")
ARM_STATE_NAMES = (
    "active_motion",
    "near_face",
    "raised",
    "open",
    "closed",
    "resting_low",
    "neutral",
)
POSTURE_STATE_NAMES = (
    "near_face",
    "raised",
    "open",
    "closed",
    "resting_low",
    "neutral",
)

DEFAULT_UPPER_LIMB_CONFIG: dict[str, float] = {
    "visibility_min": 0.30,
    "movement_speed_threshold_px_s": 60.0,
    "hold_speed_threshold_px_s": 12.0,
    "near_face_threshold_norm": 0.75,
    "raised_wrist_threshold_norm": -0.15,
    "resting_low_threshold_norm": 0.75,
    "open_openness_threshold": 0.85,
    "closed_openness_threshold": 0.45,
    "contracted_extension_threshold": 0.55,
    "asymmetry_speed_floor_px_s": 10.0,
}

ARM_FRAME_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "question_id",
    "turn_id", "speaker_label", "participant_id", "conversational_role",
    "utterance_text", "turn_start_sec", "turn_end_sec", "turn_duration_sec",
    "frame_index", "timestamp_sec", "pose_detected", "n_detected_arms",
    "body_scale_px", "shoulder_mid_x", "shoulder_mid_y",
    "left_arm_detected", "right_arm_detected",
    "left_elbow_angle_deg", "right_elbow_angle_deg",
    "left_wrist_speed_px_s", "right_wrist_speed_px_s",
    "mean_wrist_speed_px_s", "max_wrist_speed_px_s",
    "left_wrist_to_nose_norm", "right_wrist_to_nose_norm",
    "left_wrist_to_shoulder_norm", "right_wrist_to_shoulder_norm",
    "left_wrist_y_relative_shoulder_norm", "right_wrist_y_relative_shoulder_norm",
    "left_arm_extension_norm", "right_arm_extension_norm",
    "left_arm_state", "right_arm_state",
    "arm_openness_index", "arm_asymmetry_index",
    "any_wrist_near_face", "any_wrist_raised",
    "upper_limb_posture_state", "upper_limb_movement_state",
    "upper_limb_behavior_state", "qc_notes",
]

ARM_TURN_ROLE_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "question_id",
    "turn_id", "speaker_label", "participant_id", "conversational_role",
    "utterance_text", "turn_start_sec", "turn_end_sec", "turn_duration_sec",
    "n_sampled_frames", "pose_detection_rate", "arm_detection_rate",
    "mean_wrist_speed_px_s", "max_wrist_speed_px_s",
    "active_motion_frame_rate", "hold_frame_rate",
    "near_face_frame_rate", "raised_frame_rate", "open_frame_rate",
    "closed_frame_rate", "resting_low_frame_rate",
    "mean_arm_openness_index", "mean_arm_asymmetry_index",
    "dominant_upper_limb_behavior_state", "dominant_upper_limb_posture_state",
    "behavior_state_entropy", "qc_status", "qc_notes",
]

PARTICIPANT_ARM_SUMMARY_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "participant_id",
    "n_sampled_frames", "first_timestamp_sec", "last_timestamp_sec",
    "pose_detection_rate", "arm_detection_rate", "mean_wrist_speed_px_s",
    "max_wrist_speed_px_s", "active_motion_frame_rate", "hold_frame_rate",
    "near_face_frame_rate", "raised_frame_rate", "open_frame_rate",
    "closed_frame_rate", "resting_low_frame_rate",
    "mean_arm_openness_index", "mean_arm_asymmetry_index",
    "dominant_upper_limb_behavior_state", "dominant_upper_limb_posture_state",
    "behavior_state_entropy", "qc_status", "qc_notes",
]

PARTICIPANT_ROLE_ARM_SUMMARY_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "participant_id",
    "conversational_role", "n_turn_role_rows", "total_role_time_sec",
    "mean_pose_detection_rate", "mean_arm_detection_rate",
    "mean_wrist_speed_px_s", "max_wrist_speed_px_s",
    "active_motion_frame_rate", "hold_frame_rate",
    "near_face_frame_rate", "raised_frame_rate", "open_frame_rate",
    "closed_frame_rate", "resting_low_frame_rate",
    "mean_arm_openness_index", "mean_arm_asymmetry_index",
    "dominant_upper_limb_behavior_state", "dominant_upper_limb_posture_state",
    "qc_status", "qc_notes",
]

SPEAKER_LISTENER_COMPARISON_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "participant_id",
    "speaker_time_sec", "listener_time_sec",
    "speaker_active_motion_frame_rate", "listener_active_motion_frame_rate",
    "delta_active_motion_frame_rate",
    "speaker_near_face_frame_rate", "listener_near_face_frame_rate",
    "delta_near_face_frame_rate",
    "speaker_open_frame_rate", "listener_open_frame_rate",
    "delta_open_frame_rate",
    "speaker_closed_frame_rate", "listener_closed_frame_rate",
    "delta_closed_frame_rate",
    "speaker_mean_wrist_speed_px_s", "listener_mean_wrist_speed_px_s",
    "delta_mean_wrist_speed_px_s",
    "speaker_mean_arm_openness_index", "listener_mean_arm_openness_index",
    "delta_mean_arm_openness_index",
    "speaker_arm_detection_rate", "listener_arm_detection_rate",
    "qc_status", "qc_notes",
]


def upper_limb_config(config: dict[str, Any]) -> dict[str, float]:
    values = dict(DEFAULT_UPPER_LIMB_CONFIG)
    values.update(config.get("upper_limb", {}) or {})
    features = config.get("features", {}) or {}
    if "motion_speed_threshold_px_s" in features and "movement_speed_threshold_px_s" not in (config.get("upper_limb", {}) or {}):
        values["movement_speed_threshold_px_s"] = float(features["motion_speed_threshold_px_s"])
    if "hold_speed_threshold_px_s" in features and "hold_speed_threshold_px_s" not in (config.get("upper_limb", {}) or {}):
        values["hold_speed_threshold_px_s"] = float(features["hold_speed_threshold_px_s"])
    return {key: float(value) for key, value in values.items()}


def finite_mean(values: list[float] | np.ndarray | pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else np.nan


def finite_max(values: list[float] | np.ndarray | pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.max()) if array.size else np.nan


def safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def point_xy(point: tuple[float, ...] | None) -> np.ndarray | None:
    if point is None or len(point) < 2:
        return None
    x = float(point[0])
    y = float(point[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return np.array([x, y], dtype=float)


def visible_point(
    pose_points: dict[str, tuple[float, float, float]],
    name: str,
    visibility_min: float,
) -> np.ndarray | None:
    point = pose_points.get(name)
    if point is None:
        return None
    visibility = float(point[2]) if len(point) > 2 else 1.0
    if np.isfinite(visibility) and visibility < visibility_min:
        return None
    return point_xy(point)


def distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return np.nan
    return float(np.linalg.norm(a - b))


def angle_degrees(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float:
    if a is None or b is None or c is None:
        return np.nan
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def frame_base_from_turn(
    turn: pd.Series,
    video_meta: pd.Series,
    participant_id: str,
    conversational_role: str,
    frame_index: int,
    timestamp_sec: float,
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
        "conversational_role": conversational_role,
        "utterance_text": turn.get("utterance_text", ""),
        "turn_start_sec": start,
        "turn_end_sec": end,
        "turn_duration_sec": end - start,
        "frame_index": int(frame_index),
        "timestamp_sec": float(timestamp_sec),
    }


def compute_upper_limb_frame_row(
    base: dict[str, Any],
    pose_points: dict[str, tuple[float, float, float]],
    previous_wrist: dict[str, tuple[float, float, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = upper_limb_config(config)
    visibility_min = cfg["visibility_min"]
    timestamp_sec = float(base["timestamp_sec"])

    shoulders = {
        side: visible_point(pose_points, f"{side}_shoulder", visibility_min)
        for side in ARM_SIDES
    }
    elbows = {
        side: visible_point(pose_points, f"{side}_elbow", visibility_min)
        for side in ARM_SIDES
    }
    wrists = {
        side: visible_point(pose_points, f"{side}_wrist", visibility_min)
        for side in ARM_SIDES
    }
    nose = visible_point(pose_points, "nose", visibility_min)
    shoulder_width = distance(shoulders["left"], shoulders["right"])
    detected_shoulders = [point for point in shoulders.values() if point is not None]
    if np.isfinite(shoulder_width) and shoulder_width > 2:
        shoulder_mid = (shoulders["left"] + shoulders["right"]) / 2.0
        body_scale = shoulder_width
    elif detected_shoulders:
        shoulder_mid = np.mean(np.vstack(detected_shoulders), axis=0)
        body_scale = np.nan
    else:
        shoulder_mid = np.array([np.nan, np.nan], dtype=float)
        body_scale = np.nan

    row = dict(base)
    row.update({
        "pose_detected": bool(pose_points),
        "body_scale_px": body_scale,
        "shoulder_mid_x": float(shoulder_mid[0]) if np.isfinite(shoulder_mid[0]) else np.nan,
        "shoulder_mid_y": float(shoulder_mid[1]) if np.isfinite(shoulder_mid[1]) else np.nan,
        "qc_notes": "",
    })

    detected_arm_count = 0
    wrist_speeds: list[float] = []
    wrist_midline_distances: list[float] = []
    extension_values: list[float] = []
    near_face_flags: list[bool] = []
    raised_flags: list[bool] = []
    low_flags: list[bool] = []
    arm_states: dict[str, str] = {}

    for side in ARM_SIDES:
        shoulder = shoulders[side]
        elbow = elbows[side]
        wrist = wrists[side]
        arm_detected = shoulder is not None and elbow is not None and wrist is not None
        if arm_detected:
            detected_arm_count += 1

        speed = np.nan
        if wrist is not None:
            prev = previous_wrist.get(side)
            if prev is not None:
                dt = timestamp_sec - float(prev[2])
                if dt > 0:
                    speed = distance(wrist, np.array([prev[0], prev[1]], dtype=float)) / dt
            previous_wrist[side] = (float(wrist[0]), float(wrist[1]), timestamp_sec)
        if np.isfinite(speed):
            wrist_speeds.append(float(speed))

        elbow_angle = angle_degrees(shoulder, elbow, wrist)
        wrist_to_nose = safe_div(distance(wrist, nose), body_scale)
        wrist_to_shoulder = safe_div(distance(wrist, shoulder), body_scale)
        wrist_y_relative = safe_div(float(wrist[1] - shoulder[1]), body_scale) if wrist is not None and shoulder is not None else np.nan
        extension = wrist_to_shoulder
        if np.isfinite(extension):
            extension_values.append(extension)
        if wrist is not None and np.isfinite(body_scale):
            wrist_midline_distances.append(abs(float(wrist[0] - shoulder_mid[0])) / body_scale)

        near_face = bool(np.isfinite(wrist_to_nose) and wrist_to_nose <= cfg["near_face_threshold_norm"])
        raised = bool(np.isfinite(wrist_y_relative) and wrist_y_relative <= cfg["raised_wrist_threshold_norm"])
        resting_low = bool(np.isfinite(wrist_y_relative) and wrist_y_relative >= cfg["resting_low_threshold_norm"])
        near_face_flags.append(near_face)
        raised_flags.append(raised)
        low_flags.append(resting_low)

        if not arm_detected:
            arm_state = "undetected"
        elif near_face:
            arm_state = "near_face"
        elif raised:
            arm_state = "raised"
        elif resting_low and (not np.isfinite(speed) or speed <= cfg["hold_speed_threshold_px_s"]):
            arm_state = "resting_low"
        elif np.isfinite(extension) and extension <= cfg["contracted_extension_threshold"]:
            arm_state = "closed"
        elif np.isfinite(elbow_angle) and elbow_angle >= 130 and np.isfinite(extension) and extension >= 0.80:
            arm_state = "open"
        else:
            arm_state = "neutral"
        arm_states[side] = arm_state

        row.update({
            f"{side}_arm_detected": arm_detected,
            f"{side}_elbow_angle_deg": elbow_angle,
            f"{side}_wrist_speed_px_s": speed,
            f"{side}_wrist_to_nose_norm": wrist_to_nose,
            f"{side}_wrist_to_shoulder_norm": wrist_to_shoulder,
            f"{side}_wrist_y_relative_shoulder_norm": wrist_y_relative,
            f"{side}_arm_extension_norm": extension,
            f"{side}_arm_state": arm_state,
        })

    openness = finite_mean(wrist_midline_distances)
    speed_mean = finite_mean(wrist_speeds)
    speed_max = finite_max(wrist_speeds)
    speed_values = [value for value in wrist_speeds if np.isfinite(value)]
    if len(speed_values) >= 2 and max(speed_values) >= cfg["asymmetry_speed_floor_px_s"]:
        asymmetry = abs(speed_values[0] - speed_values[1]) / (abs(speed_values[0]) + abs(speed_values[1]) + 1e-9)
    else:
        asymmetry = np.nan

    if detected_arm_count == 0:
        posture_state = "undetected"
    elif any(near_face_flags):
        posture_state = "near_face"
    elif any(raised_flags):
        posture_state = "raised"
    elif np.isfinite(openness) and openness >= cfg["open_openness_threshold"] and finite_mean(extension_values) >= 0.70:
        posture_state = "open"
    elif (
        np.isfinite(openness)
        and openness <= cfg["closed_openness_threshold"]
    ) or (
        np.isfinite(finite_mean(extension_values))
        and finite_mean(extension_values) <= cfg["contracted_extension_threshold"]
    ):
        posture_state = "closed"
    elif all(low_flags) and low_flags:
        posture_state = "resting_low"
    else:
        posture_state = "neutral"

    if detected_arm_count == 0:
        movement_state = "undetected"
    elif np.isfinite(speed_max) and speed_max >= cfg["movement_speed_threshold_px_s"]:
        movement_state = "active_motion"
    elif np.isfinite(speed_max) and speed_max <= cfg["hold_speed_threshold_px_s"]:
        movement_state = "hold"
    else:
        movement_state = "subtle_motion"

    behavior_state = "active_motion" if movement_state == "active_motion" else posture_state
    row.update({
        "n_detected_arms": detected_arm_count,
        "mean_wrist_speed_px_s": speed_mean,
        "max_wrist_speed_px_s": speed_max,
        "arm_openness_index": openness,
        "arm_asymmetry_index": asymmetry,
        "any_wrist_near_face": bool(any(near_face_flags)),
        "any_wrist_raised": bool(any(raised_flags)),
        "upper_limb_posture_state": posture_state,
        "upper_limb_movement_state": movement_state,
        "upper_limb_behavior_state": behavior_state,
    })
    if detected_arm_count == 0 and row["pose_detected"]:
        row["qc_notes"] = "Pose detected but shoulder-elbow-wrist arm chain was incomplete."
    elif not row["pose_detected"]:
        row["qc_notes"] = "Pose not detected."
    return {col: row.get(col, np.nan) for col in ARM_FRAME_COLUMNS}


def summarize_turn_role_arm_features(frame_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame(columns=ARM_TURN_ROLE_COLUMNS)
    rows = []
    group_cols = [
        "dyad_id", "session_id", "video_id", "condition", "question_id",
        "turn_id", "speaker_label", "participant_id", "conversational_role",
    ]
    for keys, group in frame_df.groupby(group_cols, dropna=False):
        base = {col: value for col, value in zip(group_cols, keys, strict=False)}
        base.update({
            "utterance_text": first_value(group, "utterance_text"),
            "turn_start_sec": float(group["turn_start_sec"].iloc[0]),
            "turn_end_sec": float(group["turn_end_sec"].iloc[0]),
            "turn_duration_sec": float(group["turn_duration_sec"].iloc[0]),
        })
        rows.append({
            **base,
            **summarize_frame_group(group, config, include_prefix=False),
        })
    return pd.DataFrame(rows, columns=ARM_TURN_ROLE_COLUMNS)


def aggregate_participant_arm_summary(frame_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame(columns=PARTICIPANT_ARM_SUMMARY_COLUMNS)
    rows = []
    group_cols = ["dyad_id", "session_id", "video_id", "condition", "participant_id"]
    for keys, group in frame_df.groupby(group_cols, dropna=False):
        base = {col: value for col, value in zip(group_cols, keys, strict=False)}
        summary = summarize_frame_group(group, config, include_prefix=False)
        rows.append({
            **base,
            "n_sampled_frames": summary["n_sampled_frames"],
            "first_timestamp_sec": finite_min(group["timestamp_sec"]),
            "last_timestamp_sec": finite_max(group["timestamp_sec"]),
            "pose_detection_rate": summary["pose_detection_rate"],
            "arm_detection_rate": summary["arm_detection_rate"],
            "mean_wrist_speed_px_s": summary["mean_wrist_speed_px_s"],
            "max_wrist_speed_px_s": summary["max_wrist_speed_px_s"],
            "active_motion_frame_rate": summary["active_motion_frame_rate"],
            "hold_frame_rate": summary["hold_frame_rate"],
            "near_face_frame_rate": summary["near_face_frame_rate"],
            "raised_frame_rate": summary["raised_frame_rate"],
            "open_frame_rate": summary["open_frame_rate"],
            "closed_frame_rate": summary["closed_frame_rate"],
            "resting_low_frame_rate": summary["resting_low_frame_rate"],
            "mean_arm_openness_index": summary["mean_arm_openness_index"],
            "mean_arm_asymmetry_index": summary["mean_arm_asymmetry_index"],
            "dominant_upper_limb_behavior_state": summary["dominant_upper_limb_behavior_state"],
            "dominant_upper_limb_posture_state": summary["dominant_upper_limb_posture_state"],
            "behavior_state_entropy": summary["behavior_state_entropy"],
            "qc_status": summary["qc_status"],
            "qc_notes": summary["qc_notes"],
        })
    return pd.DataFrame(rows, columns=PARTICIPANT_ARM_SUMMARY_COLUMNS)


def aggregate_participant_role_arm_summary(turn_role_df: pd.DataFrame) -> pd.DataFrame:
    if turn_role_df.empty:
        return pd.DataFrame(columns=PARTICIPANT_ROLE_ARM_SUMMARY_COLUMNS)
    rows = []
    group_cols = ["dyad_id", "session_id", "video_id", "condition", "participant_id", "conversational_role"]
    for keys, group in turn_role_df.groupby(group_cols, dropna=False):
        base = {col: value for col, value in zip(group_cols, keys, strict=False)}
        weights = group["n_sampled_frames"].astype(float).clip(lower=0)
        rows.append({
            **base,
            "n_turn_role_rows": int(len(group)),
            "total_role_time_sec": float(group["turn_duration_sec"].sum()),
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
            "dominant_upper_limb_behavior_state": dominant_state(group["dominant_upper_limb_behavior_state"]),
            "dominant_upper_limb_posture_state": dominant_state(group["dominant_upper_limb_posture_state"]),
            "qc_status": combine_status(group["qc_status"]),
            "qc_notes": join_notes(group["qc_notes"]),
        })
    return pd.DataFrame(rows, columns=PARTICIPANT_ROLE_ARM_SUMMARY_COLUMNS)


def speaker_listener_arm_comparison(role_df: pd.DataFrame) -> pd.DataFrame:
    if role_df.empty:
        return pd.DataFrame(columns=SPEAKER_LISTENER_COMPARISON_COLUMNS)
    rows = []
    group_cols = ["dyad_id", "session_id", "video_id", "condition", "participant_id"]
    for keys, group in role_df.groupby(group_cols, dropna=False):
        base = {col: value for col, value in zip(group_cols, keys, strict=False)}
        speaker = group[group["conversational_role"] == "speaker"]
        listener = group[group["conversational_role"] == "listener"]
        row = {
            **base,
            "speaker_time_sec": finite_sum(speaker["total_role_time_sec"]) if not speaker.empty else np.nan,
            "listener_time_sec": finite_sum(listener["total_role_time_sec"]) if not listener.empty else np.nan,
        }
        for col in [
            "active_motion_frame_rate",
            "near_face_frame_rate",
            "open_frame_rate",
            "closed_frame_rate",
            "mean_wrist_speed_px_s",
            "mean_arm_openness_index",
        ]:
            speaker_value = first_numeric(speaker, col)
            listener_value = first_numeric(listener, col)
            row[f"speaker_{col}"] = speaker_value
            row[f"listener_{col}"] = listener_value
            row[f"delta_{col}"] = speaker_value - listener_value if np.isfinite(speaker_value) and np.isfinite(listener_value) else np.nan
        row["speaker_arm_detection_rate"] = first_numeric(speaker, "mean_arm_detection_rate")
        row["listener_arm_detection_rate"] = first_numeric(listener, "mean_arm_detection_rate")
        row["qc_status"] = combine_status(group["qc_status"])
        row["qc_notes"] = join_notes(group["qc_notes"])
        rows.append(row)
    return pd.DataFrame(rows, columns=SPEAKER_LISTENER_COMPARISON_COLUMNS)


def summarize_frame_group(group: pd.DataFrame, config: dict[str, Any], include_prefix: bool) -> dict[str, Any]:
    del include_prefix
    qc = config.get("qc", {}) or {}
    n = int(len(group))
    pose_rate = float(group["pose_detected"].mean()) if n else np.nan
    arm_detected = group["n_detected_arms"].fillna(0).astype(float) > 0
    arm_rate = float(arm_detected.mean()) if n else np.nan
    valid = group[group["upper_limb_behavior_state"] != "undetected"].copy()
    denominator = len(valid)
    summary = {
        "n_sampled_frames": n,
        "pose_detection_rate": pose_rate,
        "arm_detection_rate": arm_rate,
        "mean_wrist_speed_px_s": finite_mean(group["mean_wrist_speed_px_s"]),
        "max_wrist_speed_px_s": finite_max(group["max_wrist_speed_px_s"]),
        "active_motion_frame_rate": state_rate(valid, "upper_limb_behavior_state", "active_motion", denominator),
        "hold_frame_rate": state_rate(group, "upper_limb_movement_state", "hold", n),
        "near_face_frame_rate": state_rate(valid, "upper_limb_posture_state", "near_face", denominator),
        "raised_frame_rate": state_rate(valid, "upper_limb_posture_state", "raised", denominator),
        "open_frame_rate": state_rate(valid, "upper_limb_posture_state", "open", denominator),
        "closed_frame_rate": state_rate(valid, "upper_limb_posture_state", "closed", denominator),
        "resting_low_frame_rate": state_rate(valid, "upper_limb_posture_state", "resting_low", denominator),
        "mean_arm_openness_index": finite_mean(group["arm_openness_index"]),
        "mean_arm_asymmetry_index": finite_mean(group["arm_asymmetry_index"]),
        "dominant_upper_limb_behavior_state": dominant_state(valid["upper_limb_behavior_state"]),
        "dominant_upper_limb_posture_state": dominant_state(valid["upper_limb_posture_state"]),
        "behavior_state_entropy": state_entropy(valid["upper_limb_behavior_state"]),
    }
    notes = []
    status = "PASS"
    if np.isfinite(pose_rate) and pose_rate < float(qc.get("pose_detection_rate_warning", 0.50)):
        status = "WARNING"
        notes.append("Low pose detection rate.")
    if np.isfinite(arm_rate) and arm_rate < float(qc.get("arm_detection_rate_warning", qc.get("pose_detection_rate_warning", 0.50))):
        status = "WARNING"
        notes.append("Low arm-chain detection rate.")
    summary["qc_status"] = status
    summary["qc_notes"] = " ".join(notes)
    return summary


def state_rate(group: pd.DataFrame, column: str, state: str, denominator: int) -> float:
    if denominator <= 0 or column not in group:
        return np.nan
    return float((group[column] == state).sum() / denominator)


def dominant_state(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    values = values[values != "undetected"]
    if values.empty:
        return "undetected"
    return str(values.value_counts().idxmax())


def state_entropy(values: pd.Series) -> float:
    values = values.dropna().astype(str)
    values = values[values != "undetected"]
    if values.empty:
        return np.nan
    probs = values.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(probs * np.log(probs)).sum())


def first_value(group: pd.DataFrame, column: str) -> Any:
    if column not in group or group.empty:
        return ""
    return group[column].iloc[0]


def first_numeric(group: pd.DataFrame, column: str) -> float:
    if group.empty or column not in group:
        return np.nan
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def finite_min(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(array.min()) if array.size else np.nan


def finite_sum(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(array.sum()) if array.size else np.nan


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weight_array = pd.to_numeric(weights, errors="coerce").fillna(0).to_numpy(dtype=float)
    mask = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    if not mask.any():
        return finite_mean(value_array)
    return float(np.average(value_array[mask], weights=weight_array[mask]))


def combine_status(statuses: pd.Series) -> str:
    values = set(statuses.dropna().astype(str))
    if "FAIL" in values:
        return "FAIL"
    if "WARNING" in values:
        return "WARNING"
    return "PASS"


def join_notes(notes: pd.Series) -> str:
    parts = []
    for note in notes.dropna().astype(str):
        for piece in note.split("."):
            piece = piece.strip()
            if piece and piece not in parts:
                parts.append(piece)
    return ". ".join(parts)

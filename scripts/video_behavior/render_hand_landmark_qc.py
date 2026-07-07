#!/usr/bin/env python3
"""
Render MediaPipe hand-landmark overlays for question-level left/right QC.

This is a visual audit helper. It reuses a completed turn-aligned run for
checked video metadata and question windows, then saves sampled frames with
left/right ROI boxes and connected 21-point hand landmarks.

Usage:
  python scripts/video_behavior/render_hand_landmark_qc.py \
    --source-run data/interim/video_behavior/turn_runs/<run_id>
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.video_behavior.run_question_side_video_analysis import (  # noqa: E402
    SIDE_ROIS,
    latest_turn_run,
    make_question_windows,
    read_source_run,
)
from scripts.video_behavior.run_turn_aligned_video_analysis import (  # noqa: E402
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


OUTPUT_ROOT = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs")) / "video_hand_landmark_qc"
FRAME_COLUMNS = [
    "video_id",
    "dyad_id",
    "session_id",
    "condition",
    "question_id",
    "question_start_sec",
    "question_end_sec",
    "side",
    "frame_index",
    "timestamp_sec",
    "n_hands_detected",
    "output_frame",
]
SUMMARY_COLUMNS = [
    "video_id",
    "question_id",
    "side",
    "n_sampled_frames",
    "hand_detection_rate",
    "mean_hands_detected",
]


def import_drawing_tools() -> tuple[Any, Any | None]:
    try:
        import mediapipe as mp

        return mp.solutions.drawing_utils, getattr(mp.solutions, "drawing_styles", None)
    except Exception:
        import importlib

        drawing_utils = importlib.import_module("mediapipe.python.solutions.drawing_utils")
        try:
            drawing_styles = importlib.import_module("mediapipe.python.solutions.drawing_styles")
        except Exception:
            drawing_styles = None
        return drawing_utils, drawing_styles


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


def select_videos(video_meta: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = video_meta.copy()
    if args.video_id:
        requested = set(args.video_id)
        selected = selected[selected["video_id"].astype(str).isin(requested)].copy()
        found = set(selected["video_id"].astype(str))
        missing = sorted(requested - found)
        if missing:
            raise SystemExit("Requested video_id not found in source run: " + ", ".join(missing))
    if args.limit_videos and args.limit_videos > 0:
        selected = selected.head(args.limit_videos).copy()
    return selected.reset_index(drop=True)


def select_windows(windows: pd.DataFrame, video_ids: list[str], args: argparse.Namespace) -> pd.DataFrame:
    selected = windows[windows["video_id"].astype(str).isin(video_ids)].copy()
    if args.question_id:
        requested = {str(value) for value in args.question_id}
        selected = selected[selected["question_id"].astype(str).isin(requested)].copy()
    if args.max_questions_per_video and args.max_questions_per_video > 0:
        selected = (
            selected.sort_values(["video_id", "question_id"])
            .groupby("video_id", dropna=False)
            .head(args.max_questions_per_video)
            .reset_index(drop=True)
        )
    return selected.reset_index(drop=True)


def draw_label(cv2: Any, frame_bgr: Any, side: str, n_hands: int, x0: int, y0: int, x1: int, y1: int) -> None:
    color = (0, 180, 255) if side == "left" else (255, 170, 0)
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 3)
    label = f"{side}: hands={n_hands}"
    label_y = max(28, y0 + 32)
    cv2.rectangle(frame_bgr, (x0 + 8, label_y - 24), (x0 + 210, label_y + 8), (20, 20, 20), -1)
    cv2.putText(frame_bgr, label, (x0 + 16, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_hand_landmarks(
    crop_bgr: Any,
    hands: list[Any],
    hands_solution: Any,
    drawing_utils: Any,
    drawing_styles: Any | None,
) -> None:
    landmark_style = None
    connection_style = None
    if drawing_styles is not None:
        landmark_style_fn = getattr(drawing_styles, "get_default_hand_landmarks_style", None)
        connection_style_fn = getattr(drawing_styles, "get_default_hand_connections_style", None)
        landmark_style = landmark_style_fn() if callable(landmark_style_fn) else None
        connection_style = connection_style_fn() if callable(connection_style_fn) else None

    for hand_landmarks in hands:
        drawing_utils.draw_landmarks(
            crop_bgr,
            hand_landmarks,
            hands_solution.HAND_CONNECTIONS,
            landmark_style,
            connection_style,
        )


def process_video(
    video_meta: pd.Series,
    windows: pd.DataFrame,
    args: argparse.Namespace,
    frames_dir: Path,
    cv2: Any,
    hands_solution: Any,
    drawing_utils: Any,
    drawing_styles: Any | None,
) -> list[dict[str, Any]]:
    video_path = resolve_video_path(video_meta["video_path"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    hands_model = hands_solution.Hands(
        static_image_mode=True,
        max_num_hands=int(args.max_num_hands),
        min_detection_confidence=float(args.min_detection_confidence),
    )

    fps = float(video_meta["fps"])
    rows: list[dict[str, Any]] = []
    try:
        for _, window in windows.sort_values("question_id").iterrows():
            frame_indices = sample_frame_indices(
                float(window["question_start_sec"]),
                float(window["question_end_sec"]),
                fps,
                float(args.sample_fps),
                int(args.max_frames_per_question),
            )
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_bgr = cap.read()
                if not ok:
                    continue

                height, width = frame_bgr.shape[:2]
                side_counts: dict[str, int] = {}
                for side, roi in SIDE_ROIS.items():
                    x0, y0, x1, y1 = roi_to_pixels(roi, width, height)
                    crop_bgr = frame_bgr[y0:y1, x0:x1]
                    if crop_bgr.size == 0:
                        side_counts[side] = 0
                        continue
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    result = hands_model.process(crop_rgb)
                    hands = result.multi_hand_landmarks or []
                    side_counts[side] = len(hands)
                    if hands:
                        draw_hand_landmarks(crop_bgr, hands, hands_solution, drawing_utils, drawing_styles)
                    draw_label(cv2, frame_bgr, side, len(hands), x0, y0, x1, y1)

                question_token = sanitize_token(window["question_id"])
                frame_name = f"{sanitize_token(video_meta['video_id'])}_q{question_token}_frame{int(frame_index):06d}.jpg"
                frame_path = frames_dir / frame_name
                cv2.imwrite(str(frame_path), frame_bgr)

                for side in SIDE_ROIS:
                    rows.append({
                        "video_id": video_meta.get("video_id", ""),
                        "dyad_id": video_meta.get("dyad_id", ""),
                        "session_id": video_meta.get("session_id", ""),
                        "condition": video_meta.get("condition", ""),
                        "question_id": window.get("question_id", ""),
                        "question_start_sec": window.get("question_start_sec", ""),
                        "question_end_sec": window.get("question_end_sec", ""),
                        "side": side,
                        "frame_index": int(frame_index),
                        "timestamp_sec": frame_index / fps if fps > 0 else "",
                        "n_hands_detected": side_counts.get(side, 0),
                        "output_frame": str(frame_path),
                    })
    finally:
        hands_model.close()
        cap.release()
    return rows


def summarize_frames(frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = frame_df.copy()
    df["hand_detected"] = df["n_hands_detected"] > 0
    summary = (
        df.groupby(["video_id", "question_id", "side"], dropna=False)
        .agg(
            n_sampled_frames=("frame_index", "nunique"),
            hand_detection_rate=("hand_detected", "mean"),
            mean_hands_detected=("n_hands_detected", "mean"),
        )
        .reset_index()
    )
    return summary[SUMMARY_COLUMNS]


def write_report(run_dir: Path, frame_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Hand Landmark QC",
        "",
        f"Sampled side-frame rows: {len(frame_df)}",
        f"Annotated image files: {frame_df['output_frame'].nunique() if not frame_df.empty else 0}",
        "",
        "## Outputs",
        "",
        "- `review_frames/`: sampled frames with left/right ROI boxes and connected hand landmarks",
        "- `hand_landmark_qc_frames.csv`: one row per sampled frame side",
        "- `hand_landmark_qc_summary.csv`: hand-detection rate by video, question, and side",
        "",
    ]
    if not summary_df.empty:
        lines.extend(["## Summary", ""])
        preview = summary_df.head(20).to_string(index=False)
        lines.extend(["```text", preview, "```", ""])
    (run_dir / "hand_landmark_qc_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(run_dir: Path, source_run: Path, outputs: list[Path], args: argparse.Namespace) -> None:
    manifest = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_run": str(source_run),
        "software_versions": get_software_versions({"openface": {"enabled": False}, "mediapipe": {"enabled": True}}),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "inputs": [
            file_record(source_run / "metadata" / "video_metadata_checked.csv", "source_run_metadata"),
            file_record(source_run / "metadata" / "labeled_turns_checked.csv", "source_run_turns"),
        ],
        "outputs": [file_record(path, "qc_output") for path in outputs if path.exists()],
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=None, help="Completed turn-aligned run to reuse.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--video-id", action="append", help="Optional video_id to render. Repeat for multiple videos.")
    parser.add_argument("--question-id", action="append", help="Optional question_id to render. Repeat for multiple questions.")
    parser.add_argument("--limit-videos", type=int, default=1, help="Number of videos to render by default. Use 0 for all selected videos.")
    parser.add_argument(
        "--max-questions-per-video",
        type=int,
        default=1,
        help="Number of questions per video to render by default. Use 0 for all selected questions.",
    )
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--max-frames-per-question", type=int, default=12)
    parser.add_argument("--max-num-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.50)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    source_run = args.source_run or latest_turn_run()
    run_id = args.run_id or f"hand_landmark_qc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / run_id
    frames_dir = run_dir / "review_frames"

    video_meta, turns = read_source_run(source_run)
    video_meta = select_videos(video_meta, args)
    if video_meta.empty:
        raise SystemExit("No videos selected for hand-landmark QC.")
    question_windows = select_windows(make_question_windows(turns), video_meta["video_id"].astype(str).tolist(), args)
    if question_windows.empty:
        raise SystemExit("No question windows selected for hand-landmark QC.")

    cv2, _, hands_solution = import_video_stack()
    drawing_utils, drawing_styles = import_drawing_tools()
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for _, meta in video_meta.iterrows():
        windows = question_windows[question_windows["video_id"].astype(str) == str(meta["video_id"])].copy()
        if windows.empty:
            continue
        print(f"Rendering {meta['video_id']} ({len(windows)} question window(s))")
        rows.extend(process_video(meta, windows, args, frames_dir, cv2, hands_solution, drawing_utils, drawing_styles))

    frame_df = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    summary_df = summarize_frames(frame_df)
    outputs = [
        run_dir / "hand_landmark_qc_frames.csv",
        run_dir / "hand_landmark_qc_summary.csv",
        run_dir / "hand_landmark_qc_report.md",
        run_dir / "run_manifest.json",
    ]
    atomic_write_csv(outputs[0], frame_df)
    atomic_write_csv(outputs[1], summary_df)
    write_report(run_dir, frame_df, summary_df)
    write_manifest(run_dir, source_run, outputs, args)
    print(f"Hand-landmark QC complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

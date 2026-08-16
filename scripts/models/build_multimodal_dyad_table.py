"""Build analysis-ready merged transcript + vocal + video dyad tables.

Purpose
-------
Produce the two tables the statistical analysis and paper will use:

1. ``multimodal_dyad_session_table.csv`` — one row per dyad x session
   (condition), combining the four selected video-behavior variables
   (see 07_writeups/video_variable_selection_2026-07-09.md), vocal-alignment
   floor-transfer metrics, transcript structural/semantic features, and QC
   flags.
2. ``multimodal_dyad_table.csv`` — one row per dyad (sessions averaged),
   additionally merged with connection outcomes and optional covariates.

No new constructs are computed here: every column is copied or averaged from
an upstream, documented source. Composite construction belongs to the
analysis step, not this merge.

Inputs (validated; script fails with a clear error if missing)
------
- <turn-run>/results/speaker_listener_arm_comparison.csv
  (default turn run: data/interim/video_behavior/turn_runs/turn_aligned_video_20260707_120008)
- <side-run>/results/per_question_interaction_features.csv
  (default side run: data/interim/video_behavior/side_question_runs/side_question_video_20260708_231724)
- 04_data/vocal_alignment_floor_transfer_dyad.csv  (pair_id x condition)
- 04_data/structural_dyad_analysis_mapped.csv      (Dyad ID x condition)
- 04_data/scientific_dyad_analysis_results.csv     (Dyad ID x condition)
- 04_data/outcomes.csv                             (pair_id)
- 04_data/covariates_dyad.csv                      (pair_id; optional)

Outputs
-------
- ${CVS_DATA:-04_data}/multimodal_dyad_session_table.csv
- ${CVS_DATA:-04_data}/multimodal_dyad_table.csv
- ${CVS_DATA:-04_data}/multimodal_dyad_table_manifest.json (inputs + sha256 +
  row counts + merge diagnostics)

Example
-------
    python scripts/models/build_multimodal_dyad_table.py
    python scripts/models/build_multimodal_dyad_table.py --overwrite

How to check correctness
------------------------
- Session table should have 46 rows (24 dyads x 2 sessions, minus dyad10
  piper whose video is missing); dyad table 24 rows.
- The manifest lists any dyad/session present in one source but not another
  (``merge_diagnostics``) — every asymmetry there should be explainable
  (dyad10 piper missing; outcomes pair 11 has no transcripts/recordings).
- Video QC columns (``video_arm_qc_n_warning``, ``video_min_arm_detection``,
  ``video_head_orientation_detection``) let analyses exclude or sensitivity-
  test weak sessions.

Common failure cases
--------------------
- Missing selected-variable columns -> the source run predates the corrected
  mapping or schema; regenerate the run.
- Duplicate dyad x session rows in a source -> upstream table corrupted;
  inspect the file listed in the error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_VERSION = "1.0.0 (2026-07-09)"
PROJECT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))

DEFAULT_TURN_RUN = PROJECT / "data/interim/video_behavior/turn_runs/turn_aligned_video_20260707_120008"
DEFAULT_SIDE_RUN = PROJECT / "data/interim/video_behavior/side_question_runs/side_question_video_20260708_231724"

ARM_COLUMNS = [
    "dyad_id", "session_id", "participant_id",
    "delta_mean_wrist_speed_px_s", "delta_active_motion_frame_rate",
    "speaker_mean_arm_openness_index",
    "speaker_arm_detection_rate", "listener_arm_detection_rate", "qc_status",
]
SIDE_COLUMNS = [
    "dyad_id", "session_id", "question_id", "mutual_partner_orientation_rate",
    "left_head_orientation_detection_rate", "right_head_orientation_detection_rate",
]

logger = logging.getLogger("build_multimodal_dyad_table")


def fail(message: str) -> None:
    logger.error(message)
    sys.exit(1)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_required(path: Path, required_columns: list[str], label: str) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing required input for {label}: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        fail(f"{path} lacks required columns {missing} for {label}. "
             "Regenerate the upstream output before merging.")
    return df


def pair_id_from_dyad_id(series: pd.Series, source: str) -> pd.Series:
    extracted = series.astype(str).str.extract(r"dyad(\d+)_", expand=False)
    if extracted.isna().any():
        bad = series[extracted.isna()].unique()[:5]
        fail(f"Could not parse pair_id from dyad_id values {list(bad)} in {source}.")
    return extracted.astype(int)


def assert_unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
    dup = df.duplicated(subset=keys)
    if dup.any():
        fail(f"Duplicate {keys} rows in {label}: {df.loc[dup, keys].to_dict('records')[:5]}")


def build_video_session_table(turn_run: Path, side_run: Path) -> pd.DataFrame:
    arm = read_required(
        turn_run / "results" / "speaker_listener_arm_comparison.csv", ARM_COLUMNS,
        "speaker-listener arm comparison (turn run)")
    side = read_required(
        side_run / "results" / "per_question_interaction_features.csv", SIDE_COLUMNS,
        "question-level interaction features (side run)")

    arm_sess = (
        arm.groupby(["dyad_id", "session_id"])
        .agg(
            video_delta_wrist_speed_px_s=("delta_mean_wrist_speed_px_s", "mean"),
            video_delta_active_motion_rate=("delta_active_motion_frame_rate", "mean"),
            video_speaker_arm_openness=("speaker_mean_arm_openness_index", "mean"),
            video_min_arm_detection=("speaker_arm_detection_rate", "min"),
            video_arm_qc_n_warning=("qc_status", lambda s: int((s != "PASS").sum())),
            video_arm_n_participants=("participant_id", "nunique"),
        )
        .reset_index()
    )
    incomplete = arm_sess[arm_sess.video_arm_n_participants != 2]
    if not incomplete.empty:
        logger.warning(
            "%d sessions have arm data for only one participant (video features "
            "there average a single person): %s — check turn-run QC before "
            "interpreting them.",
            len(incomplete), incomplete[["dyad_id", "session_id"]].to_dict("records"))

    side["head_orientation_detection_min"] = side[
        ["left_head_orientation_detection_rate", "right_head_orientation_detection_rate"]
    ].min(axis=1)
    side_sess = (
        side.groupby(["dyad_id", "session_id"])
        .agg(
            video_mutual_partner_orientation=("mutual_partner_orientation_rate", "mean"),
            video_head_orientation_detection=("head_orientation_detection_min", "mean"),
            video_n_questions=("question_id", "nunique"),
        )
        .reset_index()
    )

    merged = arm_sess.merge(side_sess, on=["dyad_id", "session_id"], how="outer", indicator=True)
    asymmetry = merged[merged["_merge"] != "both"]
    if not asymmetry.empty:
        logger.warning("Sessions present in only one video source: %s",
                       asymmetry[["dyad_id", "session_id", "_merge"]].to_dict("records"))
    merged = merged.drop(columns=["_merge"])
    merged["pair_id"] = pair_id_from_dyad_id(merged["dyad_id"], "video tables")
    merged = merged.rename(columns={"session_id": "condition"})
    assert_unique(merged, ["pair_id", "condition"], "video session table")
    return merged


def build_session_table(video: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    diagnostics: dict[str, list] = {}

    vocal = read_required(DATA / "vocal_alignment_floor_transfer_dyad.csv",
                          ["pair_id", "condition"], "vocal alignment")
    assert_unique(vocal, ["pair_id", "condition"], "vocal alignment")

    structural = read_required(DATA / "structural_dyad_analysis_mapped.csv",
                               ["Dyad ID", "condition"], "structural transcript features")
    structural = structural.rename(columns={"Dyad ID": "dyad_id", "Order": "order"})
    structural["pair_id"] = pair_id_from_dyad_id(structural["dyad_id"], "structural table")
    structural = structural.drop(columns=["dyad_id"])
    assert_unique(structural, ["pair_id", "condition"], "structural table")

    semantic = read_required(DATA / "scientific_dyad_analysis_results.csv",
                             ["Dyad ID", "condition"], "semantic transcript features")
    semantic["pair_id"] = pair_id_from_dyad_id(semantic["Dyad ID"], "semantic table")
    semantic = semantic[["pair_id", "condition", "semantic_similarity",
                         "sent_alignment", "sentiment_synchrony", "mean_sentiment"]]
    assert_unique(semantic, ["pair_id", "condition"], "semantic table")

    table = video.merge(vocal, on=["pair_id", "condition"], how="outer", indicator="m_vocal")
    diagnostics["video_vs_vocal_mismatch"] = table.loc[
        table["m_vocal"] != "both", ["pair_id", "condition", "m_vocal"]
    ].astype(str).to_dict("records")
    table = table.drop(columns=["m_vocal"])

    for name, other in (("structural", structural), ("semantic", semantic)):
        table = table.merge(other, on=["pair_id", "condition"], how="left", indicator=f"m_{name}")
        missing = table.loc[table[f"m_{name}"] != "both", ["pair_id", "condition"]]
        diagnostics[f"missing_{name}"] = missing.astype(str).to_dict("records")
        table = table.drop(columns=[f"m_{name}"])

    id_cols = ["pair_id", "dyad_id", "condition", "order"]
    other_cols = [c for c in table.columns if c not in id_cols]
    return table[[c for c in id_cols if c in table.columns] + other_cols], diagnostics


def build_dyad_table(session_table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    diagnostics: dict[str, list] = {}
    numeric = session_table.select_dtypes("number").columns.difference(["pair_id", "order"])
    dyad = session_table.groupby("pair_id")[list(numeric)].mean().reset_index()
    dyad.insert(1, "n_sessions", session_table.groupby("pair_id").size().values)

    outcomes = read_required(DATA / "outcomes.csv", ["pair_id"], "outcomes")
    outcomes = outcomes.drop(columns=[c for c in outcomes.columns if c.startswith("Unnamed")])
    assert_unique(outcomes, ["pair_id"], "outcomes")
    merged = dyad.merge(outcomes, on="pair_id", how="left", indicator=True)
    diagnostics["dyads_without_outcomes"] = merged.loc[
        merged["_merge"] != "both", "pair_id"].tolist()
    diagnostics["outcome_pairs_without_video"] = sorted(
        set(outcomes.pair_id) - set(dyad.pair_id))
    merged = merged.drop(columns=["_merge"])

    covariates_path = DATA / "covariates_dyad.csv"
    if covariates_path.exists():
        covariates = pd.read_csv(covariates_path)
        assert_unique(covariates, ["pair_id"], "covariates")
        merged = merged.merge(covariates, on="pair_id", how="left")
    else:
        logger.warning("Optional covariates file not found (%s); dyad table has no covariates.",
                       covariates_path)
    return merged, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turn-run", type=Path, default=DEFAULT_TURN_RUN)
    parser.add_argument("--side-run", type=Path, default=DEFAULT_SIDE_RUN)
    parser.add_argument("--out-dir", type=Path, default=DATA)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    session_out = args.out_dir / "multimodal_dyad_session_table.csv"
    dyad_out = args.out_dir / "multimodal_dyad_table.csv"
    for path in (session_out, dyad_out):
        if path.exists() and not args.overwrite:
            fail(f"{path} already exists. Use --overwrite to replace it.")

    video = build_video_session_table(args.turn_run, args.side_run)
    session_table, session_diag = build_session_table(video)
    dyad_table, dyad_diag = build_dyad_table(session_table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session_table.to_csv(session_out, index=False)
    dyad_table.to_csv(dyad_out, index=False)

    inputs = [
        args.turn_run / "results" / "speaker_listener_arm_comparison.csv",
        args.side_run / "results" / "per_question_interaction_features.csv",
        DATA / "vocal_alignment_floor_transfer_dyad.csv",
        DATA / "structural_dyad_analysis_mapped.csv",
        DATA / "scientific_dyad_analysis_results.csv",
        DATA / "outcomes.csv",
    ]
    manifest = {
        "script": "scripts/models/build_multimodal_dyad_table.py",
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_turn_run": str(args.turn_run),
        "video_side_run": str(args.side_run),
        "inputs": [{"path": str(p), "sha256": sha256_of(p)} for p in inputs if p.exists()],
        "outputs": {
            "session_table": {"path": str(session_out), "rows": int(len(session_table))},
            "dyad_table": {"path": str(dyad_out), "rows": int(len(dyad_table))},
        },
        "merge_diagnostics": {**session_diag, **dyad_diag},
    }
    (args.out_dir / "multimodal_dyad_table_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Session table: %d rows x %d cols -> %s",
                len(session_table), session_table.shape[1], session_out)
    logger.info("Dyad table: %d rows x %d cols -> %s",
                len(dyad_table), dyad_table.shape[1], dyad_out)
    for key, value in {**session_diag, **dyad_diag}.items():
        if value:
            logger.warning("merge_diagnostics.%s: %s", key, value)


if __name__ == "__main__":
    main()

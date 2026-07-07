"""Build a per-session participant A/B <-> visual left/right side map.

Purpose
-------
All current video runners assume a single global ROI mapping (A = left half,
B = right half). The lab standard is the opposite (A = door side, appearing on
the RIGHT of the video; B = window side, appearing on the LEFT), and the
FaceMesh mouth-motion QC contradicts the global assumption in most sessions.
This script produces an inspectable, per-session side-map candidate so the
mapping can be manually confirmed BEFORE any participant-attributed video
feature is interpreted.

The script is purely additive: it reads existing QC outputs and writes a new
candidate table. It does not modify any pipeline input, config, or prior run.

Inputs
------
1. --mapping-run: a completed check_speaker_visual_mapping.py run directory.
   Required file: results/speaker_visual_mapping_summary.csv with columns
   dyad_id, session_id, video_id, condition, speaker_label,
   expected_participant_id, predicted_visual_participant_id,
   n_informative_turns, match_rate.
   NOTE: in that file, predicted_visual_participant_id refers to the
   CONFIGURED ROI BOX ("A" = left half, "B" = right half under the default
   config), i.e. it encodes a SCREEN SIDE, not a verified identity.
2. --speaker-log: the discussion transcription log
   (default 04_data/Discussion Transcription Log - Sheet1.csv), used only to
   enumerate expected sessions and report missing ones.

Outputs (written to --out-dir, default
05_analysis_outputs/participant_side_map/<UTC timestamp not used; fixed name>)
-------
- participant_side_map_candidate.csv : one row per dyad_id x session_id with
  the decided side for A/B, evidence pattern, confidence, and review flags.
- participant_side_map_summary.md    : human-readable QC summary.
- run_manifest.json                  : inputs, parameters, script version.

Decision logic
--------------
For each session, score two hypotheses using mouth-motion evidence:
  H_A_left  : A sits on the visual left  (current config default)
  H_A_right : A sits on the visual right (lab standard)
Score(H) = sum over speakers of match_rate-under-H weighted by
n_informative_turns. Under H_A_right the predicted box is compared against
the FLIPPED expected participant. The decided hypothesis is the higher score;
margin = |score difference| / total weight.

Review flags (needs_manual_review = True) when any of:
- evidence_pattern == "degenerate" (both speakers predicted to the same box,
  usually because one face is rarely detected);
- decided side contradicts the lab-standard prior;
- margin < --min-margin (default 0.30);
- fewer than --min-turns informative turns in total (default 10);
- session present in speaker log but absent from mouth QC.

Example
-------
    python scripts/video_behavior/build_participant_side_map.py \
        --mapping-run 05_analysis_outputs/speaker_visual_mapping_qc/speaker_visual_mapping_full

How to check correctness
------------------------
Open review frames from 05_analysis_outputs/video_roi_role_qc/review_frames/
(or any raw video frame) for every row with needs_manual_review = True and
fill the manual_side_A column ("left" or "right"), then set
review_status = "confirmed". Downstream runners should only consume rows with
review_status = confirmed (or accepted default-standard rows).

Common failure cases
--------------------
- Missing mapping-run summary CSV -> the QC run was not completed; rerun
  check_speaker_visual_mapping.py first.
- Missing required columns -> the mapping-run predates the current QC schema;
  regenerate it.
- Sessions listed as missing_from_mouth_qc -> video absent or QC skipped;
  these fall back to the lab-standard prior and always require manual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_VERSION = "1.0.0 (2026-07-07)"
PROJECT = Path(__file__).resolve().parents[2]

REQUIRED_SUMMARY_COLUMNS = [
    "dyad_id", "session_id", "video_id", "condition", "speaker_label",
    "expected_participant_id", "predicted_visual_participant_id",
    "n_informative_turns", "match_rate",
]

# Lab standard: A = door side, appears on the RIGHT of the video;
# B = window side, appears on the LEFT.
STANDARD_A_SIDE = "right"

logger = logging.getLogger("build_participant_side_map")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    logger.error(message)
    sys.exit(1)


def load_mapping_summary(mapping_run: Path) -> pd.DataFrame:
    summary_path = mapping_run / "results" / "speaker_visual_mapping_summary.csv"
    if not summary_path.exists():
        fail(
            f"Missing {summary_path}. The mouth-motion QC run is incomplete; "
            "rerun scripts/video_behavior/check_speaker_visual_mapping.py first."
        )
    df = pd.read_csv(summary_path)
    missing = [c for c in REQUIRED_SUMMARY_COLUMNS if c not in df.columns]
    if missing:
        fail(
            f"{summary_path} lacks required columns {missing}. "
            "Regenerate the QC run with the current check_speaker_visual_mapping.py."
        )
    return df


def load_expected_sessions(speaker_log: Path) -> pd.DataFrame:
    if not speaker_log.exists():
        fail(f"Speaker log not found: {speaker_log}")
    log = pd.read_csv(speaker_log)
    if "Dyad ID" not in log.columns:
        fail(f"Speaker log {speaker_log} lacks 'Dyad ID' column.")
    log = log[log["Dyad ID"].notna() & (log["Dyad ID"].astype(str).str.strip() != "")]
    rows = []
    for _, r in log.iterrows():
        dyad = str(r["Dyad ID"]).strip()
        for condition, status_col in (("piper", "Discussion_Piper"), ("cloudy", "Discussion_Cloudy")):
            status = str(r.get(status_col, "")).strip().lower()
            rows.append({"dyad_id": dyad, "session_id": condition, "log_status": status})
    return pd.DataFrame(rows)


def score_session(group: pd.DataFrame) -> dict:
    """Score A-left vs A-right hypotheses for one dyad x session."""
    flip = {"A": "B", "B": "A"}
    weight_total = 0.0
    score_left = 0.0
    score_right = 0.0
    predicted_boxes = []
    for _, row in group.iterrows():
        n = float(row["n_informative_turns"])
        if n <= 0 or pd.isna(row["match_rate"]):
            continue
        weight_total += n
        match_left = float(row["match_rate"])  # config default: expected box == expected participant
        score_left += n * match_left
        # Under A-right, the expected box is the flipped participant. The QC's
        # per-turn prediction is binary A-box/B-box, so match under the flipped
        # hypothesis is the complement of the recorded match rate.
        score_right += n * (1.0 - match_left)
        predicted_boxes.append(str(row["predicted_visual_participant_id"]))

    n_speakers = int(len(group))
    if weight_total == 0:
        pattern = "no_evidence"
    elif n_speakers >= 2 and len(set(predicted_boxes)) == 1:
        pattern = "degenerate"  # both speakers dominated by the same box
    elif n_speakers < 2:
        pattern = "single_speaker"
    else:
        pattern = "two_sided"

    if weight_total > 0:
        margin = abs(score_right - score_left) / weight_total
        evidence_a_side = "right" if score_right >= score_left else "left"
    else:
        margin = 0.0
        evidence_a_side = None
    return {
        "evidence_pattern": pattern,
        "n_informative_turns_total": int(weight_total),
        "score_a_left": round(score_left, 2),
        "score_a_right": round(score_right, 2),
        "evidence_margin": round(margin, 3),
        "evidence_a_side": evidence_a_side,
    }


def build_side_map(summary: pd.DataFrame, expected: pd.DataFrame,
                   min_margin: float, min_turns: int) -> pd.DataFrame:
    records = []
    grouped = dict(tuple(summary.groupby(["dyad_id", "session_id"], sort=True)))
    all_keys = sorted(set(grouped) | {
        (r.dyad_id, r.session_id)
        for r in expected.itertuples()
        if r.log_status == "good"
    })
    for dyad_id, session_id in all_keys:
        group = grouped.get((dyad_id, session_id))
        base = {
            "dyad_id": dyad_id,
            "session_id": session_id,
            "video_id": group["video_id"].iloc[0] if group is not None else "",
            "standard_a_side": STANDARD_A_SIDE,
        }
        if group is None:
            base.update({
                "evidence_pattern": "missing_from_mouth_qc",
                "n_informative_turns_total": 0,
                "score_a_left": 0.0, "score_a_right": 0.0,
                "evidence_margin": 0.0, "evidence_a_side": None,
            })
        else:
            base.update(score_session(group))

        evidence = base["evidence_a_side"]
        usable = (
            base["evidence_pattern"] == "two_sided"
            and base["n_informative_turns_total"] >= min_turns
            and base["evidence_margin"] >= min_margin
        )
        if usable:
            decided, decided_by = evidence, "mouth_qc"
        else:
            decided, decided_by = STANDARD_A_SIDE, "default_standard"

        agrees = (evidence == STANDARD_A_SIDE) if evidence else None
        needs_review = (
            not usable
            or (usable and decided != STANDARD_A_SIDE)
        )
        base.update({
            "evidence_agrees_with_standard": agrees,
            "candidate_a_side": decided,
            "candidate_b_side": "left" if decided == "right" else "right",
            "decided_by": decided_by,
            "needs_manual_review": bool(needs_review),
            "manual_side_A": "",
            "review_status": "pending" if needs_review else "auto_ok",
            "notes": "",
        })
        records.append(base)
    return pd.DataFrame(records)


def write_summary_md(out_path: Path, side_map: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Participant Side Map — Candidate QC Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')} · script v{SCRIPT_VERSION}",
        f"Mouth-QC source: `{args.mapping_run}`",
        f"Lab standard prior: A = {STANDARD_A_SIDE} (door side), B = "
        f"{'left' if STANDARD_A_SIDE == 'right' else 'right'} (window side)",
        "",
        "## Counts",
        "",
        side_map["evidence_pattern"].value_counts().rename_axis("evidence_pattern")
        .to_frame("n_sessions").to_markdown(),
        "",
        side_map["decided_by"].value_counts().rename_axis("decided_by")
        .to_frame("n_sessions").to_markdown(),
        "",
        f"Sessions needing manual review: "
        f"{int(side_map['needs_manual_review'].sum())} / {len(side_map)}",
        "",
        "## How to review",
        "",
        "1. For each `needs_manual_review = True` row, open a sample frame for that",
        "   session (e.g. `05_analysis_outputs/video_roi_role_qc/review_frames/` or the",
        "   raw video) and note which side participant A (door side) actually occupies.",
        "2. Fill `manual_side_A` with `left` or `right` and set `review_status` to",
        "   `confirmed`.",
        "3. Only a fully reviewed table should be promoted to",
        "   `04_data/participant_side_map.csv` for downstream runners.",
        "",
        "## Sessions needing review",
        "",
        side_map[side_map["needs_manual_review"]][
            ["dyad_id", "session_id", "evidence_pattern", "n_informative_turns_total",
             "evidence_margin", "evidence_a_side", "candidate_a_side", "decided_by"]
        ].to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mapping-run", type=Path,
        default=PROJECT / "05_analysis_outputs/speaker_visual_mapping_qc/speaker_visual_mapping_full",
        help="Completed check_speaker_visual_mapping.py run directory.")
    parser.add_argument(
        "--speaker-log", type=Path,
        default=PROJECT / "04_data/Discussion Transcription Log - Sheet1.csv")
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT / "05_analysis_outputs/participant_side_map")
    parser.add_argument("--min-margin", type=float, default=0.30,
                        help="Minimum evidence margin to trust mouth QC over the prior.")
    parser.add_argument("--min-turns", type=int, default=10,
                        help="Minimum total informative turns to trust mouth QC.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting an existing candidate table.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_csv = args.out_dir / "participant_side_map_candidate.csv"
    if out_csv.exists() and not args.overwrite:
        fail(f"{out_csv} already exists. Re-run with --overwrite to replace it "
             "(the previous file is not versioned by this script).")

    summary = load_mapping_summary(args.mapping_run)
    expected = load_expected_sessions(args.speaker_log)
    side_map = build_side_map(summary, expected, args.min_margin, args.min_turns)

    if side_map.empty:
        fail("No sessions found in mouth QC or speaker log — nothing to write.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    side_map.to_csv(out_csv, index=False)
    write_summary_md(args.out_dir / "participant_side_map_summary.md", side_map, args)

    summary_path = args.mapping_run / "results" / "speaker_visual_mapping_summary.csv"
    manifest = {
        "script": "scripts/video_behavior/build_participant_side_map.py",
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": {
            "min_margin": args.min_margin,
            "min_turns": args.min_turns,
            "standard_a_side": STANDARD_A_SIDE,
        },
        "inputs": [
            {"path": str(summary_path), "sha256": sha256_of(summary_path)},
            {"path": str(args.speaker_log), "sha256": sha256_of(args.speaker_log)},
        ],
        "outputs": [str(out_csv), str(args.out_dir / "participant_side_map_summary.md")],
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    n_review = int(side_map["needs_manual_review"].sum())
    n_flip = int((side_map["evidence_a_side"] == "right").sum())
    logger.info("Wrote %s (%d sessions).", out_csv, len(side_map))
    logger.info("Evidence favors A=right (lab standard) in %d sessions.", n_flip)
    if n_review:
        logger.warning(
            "%d sessions need manual review (degenerate/weak/contradictory evidence). "
            "See participant_side_map_summary.md. Output is usable as a CANDIDATE only; "
            "do not feed it to runners until reviewed.", n_review)


if __name__ == "__main__":
    main()

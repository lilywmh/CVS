#!/usr/bin/env python3
"""Build participant- and conversation-level CANDOR survey outcome tables.

The script reads every ``<conversation UUID>/survey.csv`` under an extracted
CANDOR root. It keeps the published survey constructs as separate outcomes and
also creates an explicitly exploratory connection composite by standardizing
six participant-level outcomes before averaging them.

Example on CARC:
    python scripts/models/build_candor_survey_outcomes.py \
      --input /scratch1/$USER/betterup_raw_media/extracted \
      --out-dir /scratch1/$USER/betterup_raw_media/outputs/candor_full_resumable \
      --vocal /scratch1/$USER/betterup_raw_media/outputs/candor_full_resumable/vocal_alignment_floor_transfer_backbiter.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUTCOMES = [
    "how_enjoyable",
    "i_like_you",
    "you_like_me",
    "in_common",
    "i_felt_close_to_my_partner",
    "i_would_like_to_become_friends",
    "interested_in_exchanging_contact_info",
    "shared_reality",
    "responsive",
]

# This is an analysis-defined composite, not a CANDOR-provided scale.
EXPLORATORY_CONNECTION_ITEMS = [
    "how_enjoyable",
    "i_like_you",
    "in_common",
    "i_felt_close_to_my_partner",
    "shared_reality",
    "responsive",
]

COVARIATES = [
    "pre_affect",
    "pre_arousal",
    "technical_quality",
    "conv_length",
    "survey_duration_in_seconds",
    "know_partner_1",
    "age",
]

ID_COLUMNS = ["convo_id", "user_id", "partner_id"]


def _read_surveys(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("survey.csv"))
    if not paths:
        raise SystemExit(f"No <conversation>/survey.csv files found under {root}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = set(ID_COLUMNS) - set(frame.columns)
        if missing:
            raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
        frame["source_conversation_dir"] = path.parent.name
        mismatch = frame["convo_id"].astype(str).ne(path.parent.name)
        if mismatch.any():
            bad = frame.loc[mismatch, "convo_id"].astype(str).unique()[:3]
            raise SystemExit(
                f"{path}: convo_id does not match directory UUID: {list(bad)}"
            )
        frames.append(frame)

    surveys = pd.concat(frames, ignore_index=True, sort=False)
    keys = ["convo_id", "user_id"]
    duplicate = surveys.duplicated(keys, keep=False)
    if duplicate.any():
        duplicate_rows = surveys.loc[duplicate].sort_values(keys)
        conflicting = duplicate_rows.groupby(keys, dropna=False).size().gt(1)
        # Duplicate extracted copies are safe only when their survey rows match.
        keep_columns = [c for c in surveys.columns if c != "source_conversation_dir"]
        for key in conflicting[conflicting].index:
            group = duplicate_rows[
                duplicate_rows["convo_id"].eq(key[0])
                & duplicate_rows["user_id"].eq(key[1])
            ][keep_columns]
            if len(group.drop_duplicates()) != 1:
                raise SystemExit(f"Conflicting duplicate survey rows for {key}")
        surveys = surveys.drop_duplicates(keys, keep="first")
    return surveys


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def build_participant_table(surveys: pd.DataFrame) -> pd.DataFrame:
    missing_outcomes = [column for column in OUTCOMES if column not in surveys]
    if missing_outcomes:
        raise SystemExit(f"CANDOR survey files lack outcomes: {missing_outcomes}")

    keep = ID_COLUMNS + ["date", "sex"] + OUTCOMES + COVARIATES
    participant = surveys[[c for c in keep if c in surveys]].copy()
    _numeric(participant, OUTCOMES + COVARIATES)

    z_columns = []
    for item in EXPLORATORY_CONNECTION_ITEMS:
        std = participant[item].std(ddof=0)
        z_column = f"z_{item}"
        participant[z_column] = (
            (participant[item] - participant[item].mean()) / std
            if np.isfinite(std) and std > 0 else np.nan
        )
        z_columns.append(z_column)
    participant["connection_composite_z_exploratory"] = participant[z_columns].mean(
        axis=1, skipna=True
    )
    participant["connection_composite_n_items"] = participant[z_columns].notna().sum(axis=1)
    participant.loc[
        participant["connection_composite_n_items"] < 4,
        "connection_composite_z_exploratory",
    ] = np.nan
    return participant


def build_dyad_table(participant: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = OUTCOMES + COVARIATES + ["connection_composite_z_exploratory"]
    for convo_id, group in participant.groupby("convo_id", sort=True):
        row: dict[str, object] = {
            "dyad_id": convo_id,
            "n_survey_respondents": int(group["user_id"].nunique()),
        }
        for column in numeric:
            if column not in group:
                continue
            values = group[column].dropna()
            row[f"candor_{column}_mean"] = values.mean() if len(values) else np.nan
            row[f"candor_{column}_abs_diff"] = (
                abs(values.iloc[0] - values.iloc[1]) if len(values) == 2 else np.nan
            )
        if "technical_quality" in group:
            row["candor_technical_quality_min"] = group["technical_quality"].min()
        reported_sex = group.get("sex", pd.Series(dtype=str)).dropna().astype(str)
        sexes = sorted(reported_sex.unique())
        row["sex_composition"] = "+".join(sexes) if sexes else np.nan
        row["same_reported_sex"] = (
            bool(len(sexes) == 1) if len(reported_sex) == 2 else np.nan
        )
        participant_ids = sorted(group["user_id"].dropna().astype(str).unique())
        row["participant_id_1"] = participant_ids[0] if participant_ids else np.nan
        row["participant_id_2"] = participant_ids[1] if len(participant_ids) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def merge_vocal(dyad: pd.DataFrame, vocal_path: Path) -> pd.DataFrame:
    vocal = pd.read_csv(vocal_path)
    if "dyad_id" not in vocal:
        raise SystemExit(
            f"{vocal_path} has no dyad_id. Re-run the final alignment aggregation "
            "with the updated compute_vocal_alignment_floor_transfer.py; acoustic "
            "feature extraction does not need to be repeated."
        )
    if vocal.duplicated(["dyad_id"]).any():
        if "condition" not in vocal:
            raise SystemExit(f"Duplicate dyad_id rows in {vocal_path}")
        non_candor = vocal[~vocal["condition"].astype(str).str.startswith("candor_")]
        if not non_candor.empty or vocal.duplicated(["dyad_id", "condition"]).any():
            raise SystemExit(f"Ambiguous duplicate CANDOR rows in {vocal_path}")
    merged = dyad.merge(vocal, on="dyad_id", how="outer", indicator=True)
    counts = merged["_merge"].value_counts().to_dict()
    print(f"Survey-vocal merge: {counts}")
    return merged.drop(columns="_merge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Root containing extracted <UUID>/survey.csv files")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--vocal", type=Path,
                        help="Optional vocal-alignment table containing dyad_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    surveys = _read_surveys(args.input)
    participant = build_participant_table(surveys)
    dyad = build_dyad_table(participant)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    participant_path = args.out_dir / "candor_survey_participant.csv"
    dyad_path = args.out_dir / "candor_survey_dyad.csv"
    participant.to_csv(participant_path, index=False)
    dyad.to_csv(dyad_path, index=False)
    print(f"Participant survey table: {len(participant)} rows -> {participant_path}")
    print(f"Dyad survey table: {len(dyad)} conversations -> {dyad_path}")

    if args.vocal:
        merged = merge_vocal(dyad, args.vocal)
        merged_path = args.out_dir / "candor_survey_vocal_merged.csv"
        merged.to_csv(merged_path, index=False)
        print(f"Merged survey-vocal table: {len(merged)} rows -> {merged_path}")


if __name__ == "__main__":
    main()

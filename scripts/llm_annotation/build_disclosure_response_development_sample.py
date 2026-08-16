#!/usr/bin/env python3
"""Build an outcome-blind disclosure-response human-coding sample."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


FIRST_PERSON = re.compile(
    r"\b(i|i'm|im|i've|ive|i'd|id|me|my|mine|myself|we|we're|we've|our|ours)\b",
    flags=re.IGNORECASE,
)
PERSONAL_CUES = re.compile(
    r"\b(feel|felt|experience|remember|grew up|family|friend|relationship|"
    r"school|work|job|afraid|scared|anxious|sad|happy|struggle|hard for me|"
    r"happened to me|in my life|when i was)\b",
    flags=re.IGNORECASE,
)

LABEL_COLUMNS = [
    "disclosure_depth",
    "response_type",
    "topically_linked",
    "coder_confidence",
    "coder_notes",
]


def _heuristic_stratum(text: str) -> str:
    words = str(text).split()
    if len(words) >= 6 and FIRST_PERSON.search(str(text)) and PERSONAL_CUES.search(
        str(text)
    ):
        return "likely_personal"
    if len(words) >= 6 and FIRST_PERSON.search(str(text)):
        return "first_person_other"
    return "unlikely_personal"


def build_pairs(turns: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dyad_id", "condition", "speaker", "turn_idx", "t_start", "t_end", "text"
    }
    missing = required - set(turns.columns)
    if missing:
        raise SystemExit(f"turn table missing columns: {sorted(missing)}")

    rows = []
    keys = ["dyad_id", "condition"]
    for key, group in turns.groupby(keys, sort=True):
        ordered = group.sort_values("t_start", kind="mergesort").reset_index(drop=True)
        for index in range(1, len(ordered)):
            disclosure, response = ordered.iloc[index - 1], ordered.iloc[index]
            if str(disclosure["speaker"]) == str(response["speaker"]):
                continue
            disclosure_text = "" if pd.isna(disclosure["text"]) else str(disclosure["text"])
            response_text = "" if pd.isna(response["text"]) else str(response["text"])
            rows.append(
                {
                    "sequence_id": (
                        f"{key[0]}::{key[1]}::{disclosure['turn_idx']}::"
                        f"{response['turn_idx']}"
                    ),
                    "dyad_id": key[0],
                    "condition": key[1],
                    "disclosure_speaker": disclosure["speaker"],
                    "disclosure_role": disclosure.get("role", np.nan),
                    "disclosure_turn_idx": disclosure["turn_idx"],
                    "disclosure_text": disclosure_text,
                    "response_speaker": response["speaker"],
                    "response_role": response.get("role", np.nan),
                    "response_turn_idx": response["turn_idx"],
                    "response_text": response_text,
                    "turn_interval_s": float(response["t_start"] - disclosure["t_end"]),
                    "heuristic_stratum": _heuristic_stratum(disclosure_text),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs["sequence_id"].duplicated().any():
        raise RuntimeError("sequence_id is not unique")
    return pairs


def balanced_sample(pairs: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = pairs.copy()
    pairs["_random"] = rng.random(len(pairs))
    targets = {
        "likely_personal": int(round(n * 0.40)),
        "first_person_other": int(round(n * 0.35)),
        "unlikely_personal": n - int(round(n * 0.40)) - int(round(n * 0.35)),
    }
    selected = []
    for stratum, target in targets.items():
        candidates = pairs[pairs["heuristic_stratum"].eq(stratum)].copy()
        # Round-robin ordering spreads early selections across sessions.
        candidates["_within_session_rank"] = candidates.groupby(
            ["dyad_id", "condition"]
        )["_random"].rank(method="first")
        candidates = candidates.sort_values(["_within_session_rank", "_random"])
        selected.append(candidates.head(target))
    sample = pd.concat(selected, ignore_index=True)
    if len(sample) < n:
        remaining = pairs[~pairs["sequence_id"].isin(sample["sequence_id"])].sort_values(
            "_random"
        )
        sample = pd.concat([sample, remaining.head(n - len(sample))], ignore_index=True)
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample.insert(0, "coding_order", np.arange(1, len(sample) + 1))
    return sample.drop(columns=["_random", "_within_session_rank"], errors="ignore")


def run(args: argparse.Namespace) -> None:
    turns = pd.read_csv(args.turns)
    pairs = build_pairs(turns)
    sample = balanced_sample(pairs, args.n, args.seed)
    for column in LABEL_COLUMNS:
        sample[column] = ""

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs.drop(columns=[], errors="ignore").to_csv(
        out / "all_original_adjacent_switches.csv", index=False
    )
    sample.to_csv(out / "development_sample_master.csv", index=False)
    for coder, random_state in (("A", args.seed + 1), ("B", args.seed + 2)):
        coding_sheet = sample.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )
        coding_sheet["coding_order"] = np.arange(1, len(coding_sheet) + 1)
        coding_sheet.to_csv(out / f"coder_{coder}.csv", index=False)
    summary = sample.groupby(["heuristic_stratum", "condition"]).size().rename("n")
    summary.to_csv(out / "sample_composition.csv")
    print(f"Candidate switches: {len(pairs)}")
    print(f"Coding sample: {len(sample)} -> {out}")
    print(summary.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", default="04_data/labeled_turns.csv")
    parser.add_argument(
        "--out-dir", default="05_analysis_outputs/disclosure_response_annotation"
    )
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

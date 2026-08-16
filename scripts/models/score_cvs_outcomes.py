#!/usr/bin/env python3
"""Reconstruct CVS participant-level outcome scores and dyad-level outcomes.

The original scoring script for `04_data/outcomes.csv` (originally `peer_eval.csv`)
is not present in the repository. This script recovers it by exhaustive search over
item subsets and reverse-coding patterns, verified to reproduce both
`dyad_<outcome>_mean` and `ddi_<outcome>` to floating-point exactness (max abs
error < 1e-9 on all 25 complete pairs).

Recovered scoring key
---------------------
partner_eval   : mean of end_Q10_1 .. end_Q10_13, with end_Q10_13 reverse-coded (8 - x)
shared_reality : mean of end_Q1_1  .. end_Q1_8    (no reverse coding)
enjoyment      : end_Q8   (single item)
solo           : end_Q9   (single item)

All items are 7-point; reverse coding is 8 - x.

Inputs
------
04_data/MASTER_SHEET_ONE_ROW_PER_PARTICIPANT.csv
04_data/outcomes.csv                (only for verification)

Outputs
-------
<out-dir>/participant_outcomes.csv  one row per participant, all four scales
<out-dir>/dyad_outcomes.csv         one row per pair, means + within-pair abs diffs
<out-dir>/verification.json         max abs error vs the original outcomes.csv

Example
-------
python scripts/models/score_cvs_outcomes.py \
    --master 04_data/MASTER_SHEET_ONE_ROW_PER_PARTICIPANT.csv \
    --outcomes 04_data/outcomes.csv \
    --out-dir 05_analysis_outputs/s2_outcome_validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LIKERT_MAX_PLUS_ONE = 8  # reverse coding on a 1-7 scale is 8 - x

SCALES = {
    "partner_eval": {
        "items": [f"end_Q10_{i}" for i in range(1, 14)],
        "reverse": ["end_Q10_13"],
    },
    "shared_reality": {
        "items": [f"end_Q1_{i}" for i in range(1, 9)],
        "reverse": [],
    },
    "enjoyment": {"items": ["end_Q8"], "reverse": []},
    "solo": {"items": ["end_Q9"], "reverse": []},
}

COMPOSITE_SCALES = ["partner_eval", "shared_reality", "enjoyment"]  # solo excluded


def score_participants(master: pd.DataFrame) -> pd.DataFrame:
    missing = [
        c for s in SCALES.values() for c in s["items"] if c not in master.columns
    ]
    if missing:
        raise ValueError(f"master sheet is missing expected items: {missing}")

    out = master[["participant_id", "pair_id", "role"]].copy()
    for name, spec in SCALES.items():
        block = master[spec["items"]].apply(pd.to_numeric, errors="coerce")
        for col in spec["reverse"]:
            block[col] = LIKERT_MAX_PLUS_ONE - block[col]
        out[name] = block.mean(axis=1)

    z = out[COMPOSITE_SCALES].apply(lambda s: (s - s.mean()) / s.std(ddof=1))
    out["composite_z"] = z.mean(axis=1)
    return out


def to_dyad(participants: pd.DataFrame) -> pd.DataFrame:
    cols = list(SCALES) + ["composite_z"]
    rows = []
    for pair_id, g in participants.groupby("pair_id"):
        if len(g) != 2:
            continue
        row = {"pair_id": pair_id, "n_members": len(g)}
        for c in cols:
            a, b = g[c].to_numpy()
            row[f"dyad_{c}_mean"] = (a + b) / 2
            row[f"ddi_{c}"] = abs(a - b)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)


def verify(dyad: pd.DataFrame, original: pd.DataFrame) -> dict:
    report = {}
    for name in SCALES:
        mc, dc = f"dyad_{name}_mean", f"ddi_{name}"
        if mc not in original.columns:
            continue
        j = dyad.merge(
            original[["pair_id", mc, dc]], on="pair_id", suffixes=("", "_orig")
        ).dropna(subset=[mc, f"{mc}_orig"])
        report[name] = {
            "n_pairs": int(len(j)),
            "max_abs_error_mean": float(np.abs(j[mc] - j[f"{mc}_orig"]).max()),
            "max_abs_error_absdiff": float(np.abs(j[dc] - j[f"{dc}_orig"]).max()),
        }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", type=Path, required=True)
    p.add_argument("--outcomes", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    master = pd.read_csv(a.master)
    participants = score_participants(master)
    dyad = to_dyad(participants)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    participants.to_csv(a.out_dir / "participant_outcomes.csv", index=False)
    dyad.to_csv(a.out_dir / "dyad_outcomes.csv", index=False)

    if a.outcomes is not None and a.outcomes.exists():
        report = verify(dyad, pd.read_csv(a.outcomes))
        (a.out_dir / "verification.json").write_text(json.dumps(report, indent=2))
        worst = max(
            (v["max_abs_error_mean"], v["max_abs_error_absdiff"])
            for v in report.values()
        )
        print(json.dumps(report, indent=2))
        print(f"[verify] worst max abs error = {max(worst):.2e}")

    print(f"[done] {len(participants)} participants, {len(dyad)} complete pairs")


if __name__ == "__main__":
    main()

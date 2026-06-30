#!/usr/bin/env python3
"""
build_covariates.py
======================
Build a DYAD-LEVEL covariate table from the participant master sheet, for use
as controls / exploratory moderators in
test_vocal_alignment_incremental_validity.py.

SOLID (use directly):
  - age           : 2026 - birth-year (Q18); dyad mean + within-dyad difference
  - gender_mixed  : 1 if the two partners differ on gender (Q23), else 0
                    (Q23 is coded 1/2; we don't assume which is which, so we
                     report same-vs-mixed, which is the usual dyad covariate)

DRAFT — VERIFY CODING DIRECTION before trusting (scored from raw items with
published keys; the 1..k response direction in Qualtrics must be confirmed):
  - ucla_loneliness : UCLA v3, 20 items, reverse 1,5,6,9,10,15,16,19,20
  - phq9            : sum of available PHQ-9 items (depression)
  - aq10            : AQ-10 autism-trait score (agree/disagree -> 1 pt rule)
  - self_monitoring : version unclear (1..5 + code 9) -> NOT scored, flagged

Each scale is aggregated to dyad level as the MEAN of the two partners (and the
absolute difference, in case dissimilarity matters).

Output: 04_data/covariates_dyad.csv (one row per pair_id)

Usage:
  python scripts/models/build_covariates.py --master /path/to/master.csv

If --master is omitted, the script looks for common private/master-sheet names
under data/private, data/raw, and 04_data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/
CURRENT_YEAR = 2026

MASTER_CANDIDATES = [
    PROJECT / "data" / "private" / "master_sheet_one_row_per_participant.csv",
    PROJECT / "data" / "private" / "MASTER_SHEET_ONE_ROW_PER_PARTICIPANT.csv",
    PROJECT / "data" / "private" / "participant_master.csv",
    PROJECT / "data" / "raw" / "master_sheet_one_row_per_participant.csv",
    PROJECT / "data" / "raw" / "participant_master.csv",
    PROJECT / "04_data" / "master_sheet_one_row_per_participant.csv",
    PROJECT / "04_data" / "MASTER_SHEET_ONE_ROW_PER_PARTICIPANT.csv",
    PROJECT / "04_data" / "participant_master.csv",
]

# ---- scale item keys (response scales confirmed from survey screenshots,
#      reverse-key lists provided by the researcher) --------------------------
# UCLA Loneliness (1978 ORIGINAL version), 20 items, Never/Rarely/Sometimes/
# Often = 0..3. VERIFIED from item texts (prescreen export): ALL 20 items are
# worded in the lonely direction (no positively-worded items) -> NO reverse
# scoring. (The 1980 revision's reverse keys 1,5,6,9,10,15,16,19,20 do NOT
# apply here and were removed after checking the actual item wording.)
UCLA_ITEMS = [f"UCLA Loneliness_{i}" for i in range(1, 21)]
UCLA_REVERSE = set()          # 1978 unidirectional version -> straight sum
UCLA_LO, UCLA_HI = 0, 3

# PHQ depression: Not at all..Nearly every day = 0..3, no reverse items.
# Only 8 symptom items are present (item 9 omitted) -> effectively PHQ-8.
# Q6 is the functional-impairment item and is NOT summed.
PHQ_ITEMS = [f"PHQ-9_{i}" for i in range(1, 10)]

# AQ-10, 1=Definitely disagree .. 4=Definitely agree. Researcher's Likert-sum
# approach: reverse items 2,3,4,5,6,9 then sum all 10 (range 10..40).
AQ_ITEMS = [f"AQ-10_{i}" for i in range(1, 11)]
AQ_REVERSE = {2, 3, 4, 5, 6, 9}
AQ_LO, AQ_HI = 1, 4

# Self-monitoring (Lennox & Wolfe Revised SMS), 13 items, 6-point; code 9=missing.
# VERIFIED from item texts: only items 9 ("I have trouble changing my behavior")
# and 12 ("I have difficulty putting up a front") are negatively worded ->
# reverse ONLY those two. The other 11 items are positively keyed (ability/
# sensitivity) and are NOT reversed. (The earlier list 1,5,7,9,11,12,13 reversed
# positively-worded items and was corrected after checking the wording.)
SM_ITEMS = [f"self-monitoring_{i}" for i in range(1, 14)]
SM_REVERSE = {9, 12}
SM_MISSING = 9


def num(s):
    return pd.to_numeric(s, errors="coerce")


def _reverse_sum(df, items, reverse_idx, lo, hi, missing=None, min_present=None):
    """Reverse-code the listed items (new = lo+hi-x) then row-sum.

    If lo/hi is None they are detected from the data (per scale)."""
    cols = [c for c in items if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    X = df[cols].apply(num)
    if missing is not None:
        X = X.mask(X == missing)
    if lo is None or hi is None:
        flat = X.values[np.isfinite(X.values)]
        lo, hi = (float(np.nanmin(flat)), float(np.nanmax(flat))) if flat.size else (0, 1)
    for i, c in enumerate(cols, start=1):
        if i in reverse_idx:
            X[c] = (lo + hi) - X[c]
    mc = min_present if min_present is not None else max(1, len(cols) - 2)
    return X.sum(axis=1, min_count=mc)


def score_ucla(df):
    return _reverse_sum(df, UCLA_ITEMS, UCLA_REVERSE, UCLA_LO, UCLA_HI,
                        min_present=17)


def score_phq(df):
    cols = [c for c in PHQ_ITEMS if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(num).sum(axis=1, min_count=max(1, len(cols) - 1))


def score_aq(df):
    return _reverse_sum(df, AQ_ITEMS, AQ_REVERSE, AQ_LO, AQ_HI, min_present=8)


def score_sm(df):
    # 6-point scale; detect lo/hi from data after dropping the 9=missing code
    return _reverse_sum(df, SM_ITEMS, SM_REVERSE, None, None,
                        missing=SM_MISSING, min_present=11)


def run(cfg):
    m = pd.read_csv(cfg["master"])
    need = {"pair_id", "role"}
    if need - set(m.columns):
        raise SystemExit(f"master sheet missing {need - set(m.columns)}")
    m = m[m["pair_id"].notna()].copy()
    m["pair_id"] = m["pair_id"].astype(int)

    # --- person-level derived vars ---
    m["age"] = CURRENT_YEAR - num(m["Q18"]) if "Q18" in m else np.nan
    m["gender"] = num(m["Q23"]) if "Q23" in m else np.nan   # 1/2, direction agnostic
    m["ucla_loneliness"] = score_ucla(m)
    m["phq9"] = score_phq(m)
    m["aq10"] = score_aq(m)
    m["self_monitoring"] = score_sm(m)

    person_scales = ["age", "ucla_loneliness", "phq9", "aq10", "self_monitoring"]

    # --- aggregate to dyad ---
    rows = []
    for pid, g in m.groupby("pair_id"):
        r = {"pair_id": pid, "n_participants": len(g)}
        # gender composition (same vs mixed), direction-agnostic
        gv = g["gender"].dropna().tolist()
        r["gender_mixed"] = int(len(set(gv)) > 1) if len(gv) == 2 else np.nan
        for s in person_scales:
            vals = g[s].dropna()
            r[f"{s}_mean"] = float(vals.mean()) if len(vals) else np.nan
            r[f"{s}_diff"] = float(vals.max() - vals.min()) if len(vals) == 2 else np.nan
        rows.append(r)
    cov = pd.DataFrame(rows).sort_values("pair_id")

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    cov.to_csv(out, index=False)

    print(f"Wrote dyad covariates for {len(cov)} pairs -> {out}")
    print(f"  age_mean: {cov['age_mean'].mean():.1f} (sd {cov['age_mean'].std():.1f}), "
          f"range {cov['age_mean'].min():.0f}-{cov['age_mean'].max():.0f}")
    print(f"  gender_mixed dyads: {int(cov['gender_mixed'].sum(skipna=True))} / "
          f"{cov['gender_mixed'].notna().sum()}")
    for s in ["ucla_loneliness", "phq9", "aq10", "self_monitoring"]:
        c = f"{s}_mean"
        print(f"  {s}: n={cov[c].notna().sum()}, "
              f"mean {cov[c].mean():.1f}, range {cov[c].min():.0f}-{cov[c].max():.0f}")
    print("\n[scoring] keys VERIFIED against item texts (prescreen export). "
          "UCLA 0-3, 1978 unidirectional -> NO reverse, straight sum of 20; "
          "PHQ summed 8 items (item 9 absent, Q6 excluded); "
          "AQ-10 1-4 Likert-sum (reverse 2,3,4,5,6,9); "
          "self-monitoring 6pt, code 9->missing (reverse ONLY 9,12).")
    return cov


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default=None,
                   help="Participant master CSV. If omitted, common private/master-sheet paths are searched.")
    p.add_argument("--out", default=str(PROJECT / "04_data" / "covariates_dyad.csv"))
    a = p.parse_args()

    master = Path(a.master).expanduser() if a.master else None
    if master is None:
        master = next((p for p in MASTER_CANDIDATES if p.exists()), None)
    if master is None or not master.exists():
        candidates = "\n".join(f"  - {p}" for p in MASTER_CANDIDATES)
        raise SystemExit(
            "No participant master sheet was provided or found.\n\n"
            "Run with:\n"
            "  python scripts/models/build_covariates.py --master /path/to/master.csv\n\n"
            "Or place one of these files in the repository:\n"
            f"{candidates}\n\n"
            "Expected columns include pair_id, role, Q18, Q23, and the scale item columns."
        )
    return {"master": str(master), "out": a.out}


if __name__ == "__main__":
    run(parse_args())

"""Build within-conversation dynamic trajectory features.

Computes per-window (question or time-bin) conversation features and
trajectory summaries (slope, early/late shift, variability) for predicting
connection outcomes.

Inputs
------
- 04_data/acoustic_turns.csv   — turn-level acoustic features
- 04_data/outcomes.csv         — dyad-level outcomes (partner_eval)

Outputs
-------
- 05_analysis_outputs/dynamic_trajectory/windowed_features.csv
- 05_analysis_outputs/dynamic_trajectory/trajectory_summaries.csv
- 05_analysis_outputs/dynamic_trajectory/manifest.json

Example
-------
    python scripts/models/build_dynamic_trajectory_features.py
    python scripts/models/build_dynamic_trajectory_features.py --window time_bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
ANALYSIS = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs"))
OUT = ANALYSIS / "dynamic_trajectory"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKCHANNELS = {
    "yeah", "yea", "yep", "yes", "right", "okay", "ok", "mm", "mhm", "mmhm",
    "uh-huh", "uhhuh", "hmm", "hm", "sure", "totally", "definitely", "exactly",
    "true", "nice", "wow", "oh", "ah", "gotcha", "cool", "agreed",
}

STOP_WORDS = set(
    "a an the and or but so if then of to in on at for with about as is are "
    "was were be been being it its it's this that these those i you he she we "
    "they me him her us them my your our their mine yours do does did done have "
    "has had not no yes yeah just like really very too also there here what which "
    "who whom when where why how all any some can could would should will im i'm "
    "you're that's dont don't think know kind sort one get got go going thing "
    "things lot um uh oh okay".split()
)

ACOUSTIC_FEATURES = ["f0_mean", "int_mean", "rate_sylps"]

N_BINS = 5  # number of equal-duration time bins

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def zscore(s: pd.Series) -> pd.Series:
    """Z-score a series; returns NaN if constant."""
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (s - s.mean()) / sd


def _words(text: str) -> list[str]:
    """Lowercase word tokens."""
    return re.findall(r"[a-z']+", str(text).lower())


def _content_words(text: str) -> list[str]:
    return [w for w in _words(text) if w not in STOP_WORDS and len(w) > 2]


def _is_backchannel(text: str, max_words: int = 3) -> bool:
    """True if every word in a short turn is a backchannel token."""
    ws = _words(str(text))
    if not ws or len(ws) > max_words:
        return False
    return all(w in BACKCHANNELS for w in ws)


def _lexical_overlap(current_text: str, prev_partner_text: str | None) -> float:
    """Content-word overlap with the previous partner turn."""
    if prev_partner_text is None:
        return np.nan
    curr = _content_words(current_text)
    prev = set(_content_words(prev_partner_text))
    if not curr or not prev:
        return np.nan
    return sum(w in prev for w in curr) / len(curr)


# ---------------------------------------------------------------------------
# Step 0: Load & validate
# ---------------------------------------------------------------------------

def load_turns(path: Path) -> pd.DataFrame:
    """Load acoustic_turns.csv and validate required columns."""
    df = pd.read_csv(path)
    need = {
        "dyad_id", "pair_id", "condition", "turn_idx", "speaker", "role",
        "question", "t_start", "t_end", "dur", "n_words", "text",
        "f0_mean", "int_mean", "rate_sylps",
    }
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"acoustic_turns.csv is missing columns: {sorted(missing)}")
    df = df.sort_values(["dyad_id", "condition", "t_start"]).reset_index(drop=True)
    # coerce numeric
    for c in ["t_start", "t_end", "dur", "n_words"] + ACOUSTIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_outcomes(path: Path) -> pd.DataFrame:
    """Load outcomes.csv and return pair_id + partner_eval."""
    df = pd.read_csv(path)
    if "dyad_partner_eval_mean" not in df.columns:
        sys.exit("outcomes.csv is missing dyad_partner_eval_mean")
    return df[["pair_id", "dyad_partner_eval_mean"]].rename(
        columns={"dyad_partner_eval_mean": "partner_eval"}
    )


# ---------------------------------------------------------------------------
# Step 1: Compute per-turn derived columns
# ---------------------------------------------------------------------------

def add_turn_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add backchannel flag, response latency, and lexical overlap."""
    df = df.copy()
    df["text"] = df["text"].fillna("")

    # backchannel flag
    df["is_backchannel"] = df["text"].apply(_is_backchannel)

    # response latency (gap to next turn within the same session)
    df["response_latency"] = np.nan
    for _, g in df.groupby(["dyad_id", "condition"]):
        idx = g.index
        gaps = g["t_start"].shift(-1).values - g["t_end"].values
        df.loc[idx, "response_latency"] = gaps

    # lexical overlap with prior partner turn
    df["lexical_overlap"] = np.nan
    for _, g in df.groupby(["dyad_id", "condition"]):
        idx = g.index.tolist()
        texts = df.loc[idx, "text"].values
        speakers = df.loc[idx, "speaker"].values
        for k in range(1, len(idx)):
            # find most recent turn by a different speaker
            for j in range(k - 1, -1, -1):
                if speakers[j] != speakers[k]:
                    df.loc[idx[k], "lexical_overlap"] = _lexical_overlap(
                        texts[k], texts[j]
                    )
                    break
    return df


# ---------------------------------------------------------------------------
# Step 2: Window & aggregate
# ---------------------------------------------------------------------------

def assign_time_bins(df: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """Assign each turn to an equal-duration time bin within its session."""
    df = df.copy()
    df["time_bin"] = np.nan
    for _, g in df.groupby(["dyad_id", "condition"]):
        idx = g.index
        t0 = g["t_start"].min()
        t1 = g["t_end"].max()
        dur = t1 - t0
        if dur <= 0:
            continue
        mid = (g["t_start"].values + g["t_end"].values) / 2
        bins = np.clip(((mid - t0) / dur * n_bins).astype(int), 0, n_bins - 1) + 1
        df.loc[idx, "time_bin"] = bins
    df["time_bin"] = df["time_bin"].astype("Int64")
    return df


def compute_window_features(
    df: pd.DataFrame, window_col: str
) -> pd.DataFrame:
    """Aggregate turn-level data to per-window, per-speaker features.

    Returns a long-form DataFrame with one row per (dyad_id, condition,
    window, speaker).
    """
    rows: list[dict] = []
    group_keys = ["dyad_id", "pair_id", "condition", window_col]

    for (dyad_id, pair_id, cond, win), wg in df.groupby(group_keys):
        if pd.isna(win):
            continue
        win_dur_s = wg["t_end"].max() - wg["t_start"].min()
        win_dur_min = win_dur_s / 60 if win_dur_s > 0 else np.nan
        total_talk = wg["dur"].sum()

        # dyadic features
        n_turns_total = len(wg)
        turn_rate = n_turns_total / win_dur_min if win_dur_min and win_dur_min > 0 else np.nan

        # floor share imbalance
        speaker_durs = wg.groupby("speaker")["dur"].sum()
        if len(speaker_durs) == 2 and total_talk > 0:
            shares = speaker_durs / total_talk
            imbalance = float(2 * abs(shares.iloc[0] - 0.5))
        else:
            imbalance = np.nan

        # per-speaker features
        for spk, sg in wg.groupby("speaker"):
            role = sg["role"].iloc[0] if "role" in sg.columns else ""
            spk_dur = sg["dur"].sum()
            n_turns = len(sg)

            row = {
                "dyad_id": dyad_id,
                "pair_id": pair_id,
                "condition": cond,
                "window": int(win),
                "window_type": window_col,
                "speaker": spk,
                "role": role,
                # timing
                "n_turns": n_turns,
                "window_dur_min": win_dur_min,
                "turn_rate": turn_rate,
                # per-person
                "floor_share": spk_dur / total_talk if total_talk > 0 else np.nan,
                "floor_share_imbalance": imbalance,
                "mean_turn_dur": sg["dur"].mean(),
                "backchannel_rate": sg["is_backchannel"].mean(),
                "response_latency": sg["response_latency"].mean(),
                "lexical_overlap": sg["lexical_overlap"].mean(),
            }
            # acoustic means
            for feat in ACOUSTIC_FEATURES:
                row[feat] = sg[feat].mean()

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: Trajectory summaries
# ---------------------------------------------------------------------------

TRAJECTORY_FEATURES = [
    "mean_turn_dur", "floor_share", "floor_share_imbalance",
    "backchannel_rate", "response_latency", "lexical_overlap",
    "turn_rate", "f0_mean", "int_mean", "rate_sylps",
]

# Features that are inherently per-person (not dyadic)
PERSON_FEATURES = [
    "mean_turn_dur", "floor_share", "backchannel_rate",
    "response_latency", "lexical_overlap",
    "f0_mean", "int_mean", "rate_sylps",
]

# Features that are dyadic (same for both speakers in a window)
DYADIC_FEATURES = ["floor_share_imbalance", "turn_rate"]


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of y ~ x via OLS; returns NaN if < 2 finite pairs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return float(stats.linregress(x[mask], y[mask]).slope)


def compute_trajectory_summaries(
    wf: pd.DataFrame,
) -> pd.DataFrame:
    """Compute slope, early/late, SD summaries per feature per person per session.

    Parameters
    ----------
    wf : DataFrame
        Windowed features (output of compute_window_features).

    Returns
    -------
    DataFrame with one row per (dyad_id, pair_id, condition, speaker)
    containing static + dynamic summaries for each feature.
    """
    rows: list[dict] = []

    for keys, sg in wf.groupby(["dyad_id", "pair_id", "condition", "speaker", "window_type"]):
        dyad_id, pair_id, cond, spk, wtype = keys
        sg = sg.sort_values("window")
        role = sg["role"].iloc[0]
        windows = sg["window"].values.astype(float)

        row: dict = {
            "dyad_id": dyad_id,
            "pair_id": pair_id,
            "condition": cond,
            "speaker": spk,
            "role": role,
            "n_windows": len(sg),
            "window_type": wtype,
        }

        for feat in TRAJECTORY_FEATURES:
            vals = sg[feat].values.astype(float)
            row[f"{feat}_mean"] = float(np.nanmean(vals))
            row[f"{feat}_sd"] = float(np.nanstd(vals, ddof=1)) if np.sum(np.isfinite(vals)) > 1 else np.nan

            # early (windows 1-2), late (windows 4-5)
            early_mask = np.isin(sg["window"].values, [1, 2])
            late_mask = np.isin(sg["window"].values, [4, 5])
            early_val = float(np.nanmean(vals[early_mask])) if early_mask.any() else np.nan
            late_val = float(np.nanmean(vals[late_mask])) if late_mask.any() else np.nan
            row[f"{feat}_early"] = early_val
            row[f"{feat}_late"] = late_val
            row[f"{feat}_early_late_diff"] = late_val - early_val if np.isfinite(early_val) and np.isfinite(late_val) else np.nan

            # OLS slope across window positions
            row[f"{feat}_slope"] = _ols_slope(windows, vals)

        rows.append(row)

    return pd.DataFrame(rows)


def collapse_to_dyad(traj: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Average person-level trajectory summaries to dyad level and merge outcome.

    For person-level features: average across the two speakers within each
    (dyad, condition) session, then average across sessions within each dyad.
    For dyadic features: they're already the same for both speakers.
    """
    # get all summary columns
    summary_cols = [c for c in traj.columns if any(
        c.endswith(s) for s in ("_mean", "_sd", "_early", "_late", "_early_late_diff", "_slope")
    )]

    # session level: average across speakers
    session = traj.groupby(["dyad_id", "pair_id", "condition", "window_type"])[
        summary_cols
    ].mean().reset_index()

    # dyad level: average across sessions (conditions)
    dyad = session.groupby(["pair_id", "window_type"])[
        summary_cols
    ].mean().reset_index()

    # merge outcome
    dyad = dyad.merge(outcomes, on="pair_id", how="left")

    return dyad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", choices=["question", "time_bin", "both"],
                    default="both",
                    help="Windowing scheme (default: both)")
    ap.add_argument("--turns", type=Path,
                    default=DATA / "acoustic_turns.csv",
                    help="Path to acoustic_turns.csv")
    ap.add_argument("--outcomes", type=Path,
                    default=DATA / "outcomes.csv",
                    help="Path to outcomes.csv")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="Output directory")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    turns = load_turns(args.turns)
    outcomes = load_outcomes(args.outcomes)
    n_sessions = turns.groupby(["dyad_id", "condition"]).ngroups
    print(f"  turns: {len(turns)} rows, "
          f"{n_sessions} sessions, "
          f"{turns['pair_id'].nunique()} dyads")
    print(f"  outcomes: {len(outcomes)} dyads")

    print("Computing turn-level derived features...")
    turns = add_turn_derived(turns)

    # Assign time bins regardless (needed if --window includes time_bin)
    turns = assign_time_bins(turns)

    windowed_frames = []
    windows_to_run = []
    if args.window in ("question", "both"):
        windows_to_run.append("question")
    if args.window in ("time_bin", "both"):
        windows_to_run.append("time_bin")

    for wtype in windows_to_run:
        print(f"Computing per-window features ({wtype})...")
        wf = compute_window_features(turns, wtype)
        print(f"  {len(wf)} rows (speaker × window × session)")
        windowed_frames.append(wf)

    windowed = pd.concat(windowed_frames, ignore_index=True)

    # Save windowed features
    windowed.to_csv(args.out / "windowed_features.csv", index=False)
    print(f"Wrote {args.out / 'windowed_features.csv'} ({len(windowed)} rows)")

    # Trajectory summaries
    print("Computing trajectory summaries...")
    traj = compute_trajectory_summaries(windowed)
    traj.to_csv(args.out / "trajectory_summaries_person.csv", index=False)
    print(f"  person-level: {len(traj)} rows")

    # Dyad-level with outcome
    dyad = collapse_to_dyad(traj, outcomes)
    dyad.to_csv(args.out / "trajectory_summaries.csv", index=False)
    print(f"  dyad-level: {len(dyad)} rows")

    # Manifest
    manifest = {
        "status": "COMPLETE",
        "script": str(Path(__file__).resolve().relative_to(PROJECT)),
        "created": datetime.now().isoformat(timespec="seconds"),
        "windows": windows_to_run,
        "inputs": {
            "turns": {"path": str(args.turns), "sha256": sha256(args.turns),
                      "n_rows": len(turns)},
            "outcomes": {"path": str(args.outcomes), "sha256": sha256(args.outcomes),
                         "n_rows": len(outcomes)},
        },
        "outputs": {
            "windowed_features": {
                "path": str(args.out / "windowed_features.csv"),
                "n_rows": len(windowed),
                "n_sessions": int(windowed.groupby(["dyad_id", "condition"]).ngroups),
            },
            "trajectory_summaries_person": {
                "path": str(args.out / "trajectory_summaries_person.csv"),
                "n_rows": len(traj),
            },
            "trajectory_summaries": {
                "path": str(args.out / "trajectory_summaries.csv"),
                "n_rows": len(dyad),
                "n_dyads": int(dyad["pair_id"].nunique()),
            },
        },
        "features": TRAJECTORY_FEATURES,
        "dynamic_summaries": ["mean", "sd", "early", "late", "early_late_diff", "slope"],
        "python": sys.version,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out / 'manifest.json'}")
    print("Done.")


if __name__ == "__main__":
    main()

"""CANDOR replication of within-conversation dynamic trajectory analysis.

Adapts the CVS dynamic-trajectory pipeline to CANDOR data:
- Equal-duration time bins only (CANDOR lacks the 5-question structure)
- Outcome: how_enjoyable (dyad mean)
- Participant-level survey → dyad-level aggregation
- Larger N (~1,656 conversations)

Inputs
------
- CANDOR acoustic turns (e.g., acoustic_turns_backbiter.csv)
- CANDOR participant survey (e.g., candor_survey_participant.csv)

Outputs
-------
- <out>/windowed_features.csv
- <out>/trajectory_summaries.csv
- <out>/model_results.csv
- <out>/summary.md
- <out>/manifest.json

Example (CARC)
--------------
    python scripts/models/replicate_candor_dynamic_trajectory.py \
        --turns  "$CANDOR_OUT_DIR/acoustic_turns_backbiter.csv" \
        --survey "$CANDOR_OUT_DIR/candor_survey_participant.csv" \
        --out    "$RESULT_DIR" \
        --outcome how_enjoyable \
        --nperm 5000 --nboot 2000 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
_models_dir = str(Path(__file__).resolve().parent)
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# Reuse CVS trajectory functions (same directory, no __init__.py)
from build_dynamic_trajectory_features import (  # type: ignore[import-untyped]
    ACOUSTIC_FEATURES,
    BACKCHANNELS,
    N_BINS,
    TRAJECTORY_FEATURES,
    _is_backchannel,
    _lexical_overlap,
    _words,
    assign_time_bins,
    compute_trajectory_summaries,
    compute_window_features,
    sha256,
    zscore,
)
from analyze_dynamic_trajectories import (  # type: ignore[import-untyped]
    DYNAMIC_SUFFIXES,
    bh_fdr,
    run_models,
    write_summary,
)

# ---------------------------------------------------------------------------
# CANDOR data loading
# ---------------------------------------------------------------------------

REQUIRED_TURN_COLS = {
    "dyad_id", "pair_id", "speaker", "t_start", "t_end", "dur",
}


def load_candor_turns(path: Path) -> pd.DataFrame:
    """Load CANDOR acoustic turns and adapt to CVS column conventions."""
    df = pd.read_csv(path)
    missing = REQUIRED_TURN_COLS - set(df.columns)
    if missing:
        sys.exit(f"CANDOR turns missing columns: {sorted(missing)}")

    # CANDOR uses convo_id/user_id in some files; normalize
    if "convo_id" in df.columns and "dyad_id" not in df.columns:
        df["dyad_id"] = df["convo_id"]
    if "user_id" in df.columns and "speaker" not in df.columns:
        df["speaker"] = df["user_id"]

    # CANDOR has no question column; add a placeholder condition
    if "condition" not in df.columns:
        df["condition"] = df.get("condition", "candor")
    if "role" not in df.columns:
        df["role"] = ""
    if "n_words" not in df.columns:
        df["n_words"] = np.nan

    # Exclude explicit backchannels if tagged
    if "candor_turn_kind" in df.columns:
        bc_mask = df["candor_turn_kind"].astype(str).str.lower().eq("backchannel")
        n_bc = bc_mask.sum()
        df = df.loc[~bc_mask].copy()
        print(f"  excluded {n_bc} explicit CANDOR backchannels")

    df = df.sort_values(["dyad_id", "t_start"]).reset_index(drop=True)
    for c in ["t_start", "t_end", "dur"] + [f for f in ACOUSTIC_FEATURES if f in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_candor_survey(path: Path, outcome: str) -> pd.DataFrame:
    """Load CANDOR participant survey and aggregate to dyad level."""
    sv = pd.read_csv(path)

    # Normalize ID columns
    if "convo_id" in sv.columns:
        sv["dyad_id"] = sv["convo_id"].astype(str)
    elif "dyad_id" in sv.columns:
        sv["dyad_id"] = sv["dyad_id"].astype(str)
    else:
        sys.exit("Survey missing convo_id or dyad_id")

    if outcome not in sv.columns:
        sys.exit(f"Survey missing outcome column: {outcome}")

    sv[outcome] = pd.to_numeric(sv[outcome], errors="coerce")

    # Aggregate to dyad level
    dyad = (
        sv.groupby("dyad_id")
        .agg(
            n_respondents=(outcome, "count"),
            outcome_mean=(outcome, "mean"),
        )
        .reset_index()
    )
    # Keep only dyads with both respondents
    dyad = dyad[dyad["n_respondents"] >= 2].copy()
    dyad = dyad.rename(columns={"outcome_mean": outcome})

    print(f"  survey: {len(sv)} participants → {len(dyad)} dyads with ≥2 respondents")
    return dyad


def add_turn_derived_candor(df: pd.DataFrame) -> pd.DataFrame:
    """Add backchannel flag, response latency, and lexical overlap for CANDOR."""
    df = df.copy()

    # text may not be available (data rules)
    has_text = "text" in df.columns
    if has_text:
        df["text"] = df["text"].fillna("")
        df["is_backchannel"] = df["text"].apply(_is_backchannel)
    else:
        df["is_backchannel"] = False

    # response latency
    df["response_latency"] = np.nan
    for _, g in df.groupby("dyad_id"):
        idx = g.index
        gaps = g["t_start"].shift(-1).values - g["t_end"].values
        df.loc[idx, "response_latency"] = gaps

    # lexical overlap
    df["lexical_overlap"] = np.nan
    if has_text:
        for _, g in df.groupby("dyad_id"):
            idx = g.index.tolist()
            texts = df.loc[idx, "text"].values
            speakers = df.loc[idx, "speaker"].values
            for k in range(1, len(idx)):
                for j in range(k - 1, -1, -1):
                    if speakers[j] != speakers[k]:
                        df.loc[idx[k], "lexical_overlap"] = _lexical_overlap(
                            texts[k], texts[j]
                        )
                        break

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--turns", type=Path, required=True,
                    help="CANDOR acoustic turns CSV")
    ap.add_argument("--survey", type=Path, required=True,
                    help="CANDOR participant survey CSV")
    ap.add_argument("--outcome", default="how_enjoyable",
                    help="Survey outcome column (default: how_enjoyable)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory")
    ap.add_argument("--nperm", type=int, default=5000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nbins", type=int, default=5,
                    help="Number of equal-duration time bins (default: 5)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # -- Load data --
    print("Loading CANDOR turns...")
    turns = load_candor_turns(args.turns)
    n_convos = turns["dyad_id"].nunique()
    print(f"  {len(turns)} turns, {n_convos} conversations")

    print("Loading CANDOR survey...")
    survey = load_candor_survey(args.survey, args.outcome)

    # Map pair_id from turns to survey (pair_id is integer in turns)
    # Build dyad_id → pair_id lookup from turns
    dyad_pair = turns[["dyad_id", "pair_id"]].drop_duplicates()
    survey = survey.merge(dyad_pair, on="dyad_id", how="inner")
    print(f"  matched: {len(survey)} dyads with turns + survey")

    # -- Derived features --
    print("Computing turn-level derived features...")
    turns = add_turn_derived_candor(turns)

    # -- Time bins (CANDOR has no question structure) --
    print(f"Assigning {args.nbins} equal-duration time bins...")
    turns = assign_time_bins(turns, n_bins=args.nbins)

    # -- Window features --
    print("Computing per-window features...")
    wf = compute_window_features(turns, "time_bin")
    print(f"  {len(wf)} rows")
    wf.to_csv(args.out / "windowed_features.csv", index=False)

    # -- Trajectory summaries --
    print("Computing trajectory summaries...")
    traj = compute_trajectory_summaries(wf)
    traj.to_csv(args.out / "trajectory_summaries_person.csv", index=False)
    print(f"  person-level: {len(traj)} rows")

    # Collapse to dyad level
    summary_cols = [c for c in traj.columns if any(
        c.endswith(s) for s in ("_mean", "_sd", "_early", "_late", "_early_late_diff", "_slope")
    )]
    session = traj.groupby(["dyad_id", "pair_id", "window_type"])[
        summary_cols
    ].mean().reset_index()
    # CANDOR has one conversation per dyad, so session = dyad
    dyad = session.copy()
    dyad = dyad.merge(
        survey[["pair_id", args.outcome]],
        on="pair_id", how="left",
    )
    # Rename outcome to partner_eval for compatibility with analyze functions
    dyad = dyad.rename(columns={args.outcome: "partner_eval"})
    dyad.to_csv(args.out / "trajectory_summaries.csv", index=False)
    n_with_outcome = dyad["partner_eval"].notna().sum()
    print(f"  dyad-level: {len(dyad)} rows, {n_with_outcome} with outcome")

    # -- Models --
    print(f"\nRunning models (nperm={args.nperm}, nboot={args.nboot})...")
    results = run_models(dyad, "time_bin", args.nperm, args.nboot, rng)

    if len(results) == 0:
        sys.exit("No model results produced.")

    # FDR correction
    perm_mask = results["p_perm"].notna()
    if perm_mask.any():
        results.loc[perm_mask, "q_perm_BH"] = bh_fdr(
            results.loc[perm_mask, "p_perm"].values
        )
    delta_mask = results["p_delta_r2"].notna()
    if delta_mask.any():
        results.loc[delta_mask, "q_delta_r2_BH"] = bh_fdr(
            results.loc[delta_mask, "p_delta_r2"].values
        )

    results = results.round(4)
    results.to_csv(args.out / "model_results.csv", index=False)
    print(f"Wrote model_results.csv ({len(results)} rows)")

    # -- Summary --
    write_summary(results, args.out)
    print(f"Wrote summary.md")

    # -- Print headlines --
    static = results[results["model"] == "static_only"].sort_values("p_perm")
    if len(static) > 0:
        best_s = static.iloc[0]
        print(f"\nStrongest static predictor: {best_s['predictor']}: "
              f"r = {best_s['r']:.3f}, p = {best_s['p_perm']:.4f}")

    dyn = results[results["model"].str.startswith("dynamic_only")].sort_values("p_perm")
    if len(dyn) > 0:
        best_d = dyn.iloc[0]
        print(f"Strongest dynamic predictor: {best_d['predictor']}: "
              f"r = {best_d['r']:.3f}, p = {best_d['p_perm']:.4f}")

    incr = results[results["model"] == "static_plus_slope"].sort_values("p_delta_r2")
    if len(incr) > 0:
        best_i = incr.iloc[0]
        print(f"Best incremental: {best_i['feature']}: "
              f"delta-R2 = {best_i['delta_r2']:.3f}, p = {best_i['p_delta_r2']:.4f}")

    # -- Manifest --
    manifest = {
        "status": "COMPLETE",
        "script": str(Path(__file__).resolve().relative_to(PROJECT)),
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": "CANDOR",
        "inputs": {
            "turns": {"path": str(args.turns), "sha256": sha256(args.turns),
                      "n_rows": len(turns), "n_conversations": n_convos},
            "survey": {"path": str(args.survey)},
        },
        "outcome": args.outcome,
        "n_bins": args.nbins,
        "n_dyads_with_outcome": int(n_with_outcome),
        "n_perm": args.nperm,
        "n_boot": args.nboot,
        "seed": args.seed,
        "n_model_rows": len(results),
        "python": sys.version,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest.json")
    print("Done.")


if __name__ == "__main__":
    main()

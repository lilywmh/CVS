#!/usr/bin/env python3
"""
qc_vocal_measure_validity.py
============================
Measurement-validity QC for the vocal-alignment features, run BEFORE any
incremental-validity test against survey outcomes.

Motivation
----------
The CANDOR processed-audio run found essentially zero association between
intensity-edge vocal alignment and survey connection (r = .011). That null is
only interpretable if the vocal measure itself carries real, reliable,
dyad-specific signal. The existing whole-turn surrogate test found intensity
synchrony at chance (real .0091 vs pseudo .0082, p = .54), which means the
primary theoretical measure has NOT yet been shown to measure anything
dyad-specific. This script supplies the two missing pieces of evidence:

  1. SPLIT-HALF RELIABILITY
     Each session's floor-transfer turn sequence is split into two halves,
     every vocal-alignment feature is recomputed independently on each half,
     and the two halves are correlated across sessions. Spearman-Brown gives
     the implied full-length reliability. A feature with near-zero reliability
     cannot correlate with anything, so a null outcome association is
     uninformative about theory.

  2. EDGE-SPECIFIC SURROGATE (chance baseline)
     compute_vocal_alignment_floor_transfer.py only runs the real-vs-pseudo
     dyad test on WHOLE-TURN synchrony. The primary theoretical measures are
     the EDGE variants (speaker A's turn-offset -> speaker B's turn-onset).
     This script runs the same real-vs-pseudo-dyad test on edge synchrony and
     edge proximity, which is what a reviewer will actually ask for.

  3. ATTENUATION CEILING
     Given the estimated reliability of the vocal measure (and, optionally, the
     reliability of the outcome from qc_candor_outcome_reliability.py), reports
     the maximum observable correlation sqrt(r_xx * r_yy). This converts "we
     found nothing" into a quantified statement about what could have been
     found at all.

Estimators are IMPORTED from compute_vocal_alignment_floor_transfer.py rather
than reimplemented, so split halves are scored with exactly the same code as
the canonical run. Filtering (backchannel classification and floor-transfer
exclusion flags) also reuses that module.

Inputs
------
  --in    acoustic turn table produced by extract_acoustic_features.py
          (CANDOR: acoustic_turns_backbiter.csv). Must contain
          dyad_id/pair_id, condition, speaker, t_start, t_end and the prosody
          columns f0_mean/int_mean/rate_sylps plus onset_*/offset_* columns.

Outputs (timestamped run directory, never overwritten)
------------------------------------------------------
  split_half_reliability.csv   per-feature r_half, Spearman-Brown r_xx, n
  split_half_session_scores.csv  per-session half-A / half-B feature values
  edge_surrogate.csv           per-feature real vs pseudo-dyad edge stats
  attenuation_ceiling.csv      max observable r given reliabilities
  qc_report.md                 human-readable summary with pass/fail flags
  manifest.json                inputs, config, git commit, row counts

Example (CARC)
--------------
  python scripts/acoustic_alignment/qc_vocal_measure_validity.py \
    --in  /scratch1/$USER/betterup_raw_media/outputs/candor_full_processed_resumable/acoustic_turns_backbiter.csv \
    --out-root /scratch1/$USER/betterup_raw_media/outputs/vocal_measure_qc \
    --n-surrogate 2000 --seed 42

Example (local pilot)
---------------------
  python scripts/acoustic_alignment/qc_vocal_measure_validity.py \
    --in data/derived/candor/acoustic_turns_backbiter.csv \
    --out-root data/interim/vocal_measure_qc
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the canonical estimators so the halves are scored identically to the
# full-length run. Any change to the main script propagates here automatically.
import compute_vocal_alignment_floor_transfer as vaft  # noqa: E402


# Reliability thresholds used only to print an advisory verdict in the report.
# They are NOT gates on the analysis; the numbers themselves are the result.
RELIABILITY_ADEQUATE = 0.60
RELIABILITY_MARGINAL = 0.40


# --------------------------------------------------------------------------
# split-half construction
# --------------------------------------------------------------------------
def assign_halves(turns: pd.DataFrame, scheme: str, n_blocks: int) -> pd.Series:
    """Return a Series of 'A'/'B' half labels for one session's turns.

    scheme='blocks'  (default, recommended)
        Split the time-ordered turn sequence into ``n_blocks`` contiguous
        blocks and assign alternating blocks to half A and half B. Contiguity
        preserves local turn adjacency (which every estimator depends on)
        while alternation balances the two halves over conversation time, so
        the estimate is not contaminated by warm-up or fatigue drift.

    scheme='time'
        First half of turns vs second half. Simpler and closer to a classic
        test-retest split, but conflates reliability with genuine within-
        conversation change (which is exactly what 'convergence' measures), so
        it will understate reliability for the convergence family.
    """
    ordered = turns.sort_values("t_start")
    n = len(ordered)
    if scheme == "time":
        labels = np.where(np.arange(n) < n // 2, "A", "B")
    elif scheme == "blocks":
        block_id = np.minimum((np.arange(n) * n_blocks) // max(n, 1), n_blocks - 1)
        labels = np.where(block_id % 2 == 0, "A", "B")
    else:
        raise SystemExit(f"Unknown split scheme: {scheme}")
    return pd.Series(labels, index=ordered.index).reindex(turns.index)


def split_half_scores(df: pd.DataFrame, cfg: dict, scheme: str,
                      n_blocks: int) -> pd.DataFrame:
    """Compute every vocal-alignment feature separately on each session half."""
    session_keys = (["dyad_id", "pair_id", "condition"]
                    if "dyad_id" in df.columns else ["pair_id", "condition"])
    rows = []
    # Independent RNG per half so the proximity baselines are not shared, but
    # seeded deterministically from the session index for reproducibility.
    for session_index, (key, grp) in enumerate(df.groupby(session_keys, sort=True)):
        key = key if isinstance(key, tuple) else (key,)
        halves = assign_halves(grp, scheme, n_blocks)
        row = dict(zip(session_keys, key))
        usable = True
        for half in ("A", "B"):
            sub = grp[halves == half]
            rng = np.random.default_rng(cfg["seed"] + 1000 * session_index
                                        + (0 if half == "A" else 1))
            feats = vaft.process_session(sub, cfg, rng)
            if feats.get("n_switch", 0) < cfg["min_switches"]:
                usable = False
            for name, value in feats.items():
                row[f"{name}__{half}"] = value
        row["both_halves_usable"] = usable
        rows.append(row)
    return pd.DataFrame(rows)


def _spearman_brown(r: float) -> float:
    """Full-length reliability implied by a correlation between two halves."""
    if not np.isfinite(r):
        return np.nan
    denominator = 1.0 + r
    if denominator <= 0:
        # A negative half-half correlation implies no reliable common variance.
        return 0.0
    return float(np.clip(2.0 * r / denominator, 0.0, 1.0))


def summarise_split_half(scores: pd.DataFrame) -> pd.DataFrame:
    """Correlate half A against half B across sessions, per feature."""
    usable = scores[scores["both_halves_usable"]]
    feature_names = sorted({
        c[:-3] for c in scores.columns
        if c.endswith("__A") and not c.startswith(("n_turns", "n_switch"))
    })
    rows = []
    for feature in feature_names:
        a = pd.to_numeric(usable.get(f"{feature}__A"), errors="coerce")
        b = pd.to_numeric(usable.get(f"{feature}__B"), errors="coerce")
        if a is None or b is None:
            continue
        mask = a.notna() & b.notna()
        n = int(mask.sum())
        if n < 10 or a[mask].std(ddof=0) == 0 or b[mask].std(ddof=0) == 0:
            rows.append({"feature": feature, "n_sessions": n,
                         "r_half_half": np.nan, "reliability_spearman_brown": np.nan,
                         "mean_half_a": np.nan, "mean_half_b": np.nan,
                         "sd_half_a": np.nan, "sd_half_b": np.nan,
                         "note": "insufficient usable sessions or zero variance"})
            continue
        r = float(np.corrcoef(a[mask], b[mask])[0, 1])
        rows.append({
            "feature": feature,
            "n_sessions": n,
            "r_half_half": round(r, 4),
            "reliability_spearman_brown": round(_spearman_brown(r), 4),
            "mean_half_a": round(float(a[mask].mean()), 5),
            "mean_half_b": round(float(b[mask].mean()), 5),
            "sd_half_a": round(float(a[mask].std(ddof=1)), 5),
            "sd_half_b": round(float(b[mask].std(ddof=1)), 5),
            "note": "",
        })
    out = pd.DataFrame(rows)
    return out.sort_values("reliability_spearman_brown", ascending=False,
                           na_position="last").reset_index(drop=True)


# --------------------------------------------------------------------------
# edge-specific surrogate test
# --------------------------------------------------------------------------
def _edge_series(df: pd.DataFrame, cfg: dict):
    """Per-session (offset_A, onset_B) series for each edge-capable feature.

    Mirrors vaft.edge_alignment's pairing and within-speaker z-scoring exactly,
    but keeps the raw paired series so they can be recombined across dyads.
    """
    session_keys = (["dyad_id", "pair_id", "condition"]
                    if "dyad_id" in df.columns else ["pair_id", "condition"])
    series = {}
    for feature, (_whole, onset, offset) in vaft.FEATURES.items():
        if onset is None or offset is None:
            continue  # rate has no meaningful sub-second edge
        series[feature] = []

    for _key, grp in df.groupby(session_keys, sort=True):
        if sum(1 for _ in vaft._adjacent_pairs(grp) ) < cfg["min_switches"]:
            continue
        for feature, (_whole, onset, offset) in vaft.FEATURES.items():
            if onset is None or offset is None:
                continue
            if onset not in grp.columns or offset not in grp.columns:
                continue
            t = grp.copy()
            if feature in ("f0", "int"):
                t[onset] = vaft.zscore_within_speaker(t, onset)
                t[offset] = vaft.zscore_within_speaker(t, offset)
            pairs = [(a[offset], b[onset]) for a, b in vaft._adjacent_pairs(t)
                     if np.isfinite(a[offset]) and np.isfinite(b[onset])]
            if len(pairs) >= 4:
                off = np.array([p[0] for p in pairs], dtype=float)
                on = np.array([p[1] for p in pairs], dtype=float)
                series[feature].append((off, on))
    return series


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 4 or np.std(a[:n]) == 0 or np.std(b[:n]) == 0:
        return np.nan
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 4:
        return np.nan
    return float(np.mean(np.abs(a[:n] - b[:n])))


def edge_surrogate_test(df: pd.DataFrame, cfg: dict,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Real vs pseudo-dyad test on turn-EDGE coordination.

    Real     : offset series and onset series from the SAME conversation.
    Pseudo   : speaker A's offset series from conversation i paired with
               speaker B's onset series from a DIFFERENT conversation j. These
               two people never interacted, so any apparent coordination is
               attributable to shared population structure rather than to the
               dyad.

    Two metrics are reported per feature:
      synchrony  mean within-session Pearson r (higher = coordinated)
      proximity  mean |offset - onset| expressed as (pseudo - real), so that
                 positive = edges are closer together than chance
    """
    series = _edge_series(df, cfg)
    n_batches = max(1, int(np.ceil(cfg["n_surrogate"] / 50)))
    rows = []

    for feature, sessions in series.items():
        if len(sessions) < 3:
            rows.append({"feature": feature, "metric": "synchrony",
                         "n_sessions": len(sessions), "real_mean": np.nan,
                         "pseudo_mean": np.nan, "p_vs_pseudo": np.nan,
                         "note": "fewer than 3 usable sessions"})
            continue

        real_sync = float(np.nanmean([_corr(off, on) for off, on in sessions]))
        real_absdiff = float(np.nanmean([_mean_abs_diff(off, on)
                                         for off, on in sessions]))

        pseudo_sync_means, pseudo_absdiff_means = [], []
        for batch in range(n_batches):
            batch_size = max(1, min(50, cfg["n_surrogate"] - batch * 50))
            sync_vals, diff_vals = [], []
            for _ in range(batch_size):
                i, j = rng.integers(0, len(sessions), size=2)
                if i == j:
                    continue
                sync_vals.append(_corr(sessions[i][0], sessions[j][1]))
                diff_vals.append(_mean_abs_diff(sessions[i][0], sessions[j][1]))
            if sync_vals:
                pseudo_sync_means.append(np.nanmean(sync_vals))
            if diff_vals:
                pseudo_absdiff_means.append(np.nanmean(diff_vals))

        pseudo_sync_means = np.asarray(pseudo_sync_means, dtype=float)
        pseudo_absdiff_means = np.asarray(pseudo_absdiff_means, dtype=float)

        # Two-sided permutation p for synchrony (same convention as the
        # whole-turn surrogate test in the main script).
        p_sync = ((np.sum(np.abs(pseudo_sync_means) >= abs(real_sync)) + 1)
                  / (len(pseudo_sync_means) + 1))
        rows.append({
            "feature": feature, "metric": "edge_synchrony",
            "n_sessions": len(sessions),
            "real_mean": round(real_sync, 4),
            "pseudo_mean": round(float(np.nanmean(pseudo_sync_means)), 4),
            "p_vs_pseudo": round(float(p_sync), 4),
            "note": "",
        })

        # Proximity: real should have SMALLER |offset - onset| than pseudo.
        # Report as pseudo - real so that positive = better than chance, and
        # test one-sided in that direction.
        real_prox = float(np.nanmean(pseudo_absdiff_means) - real_absdiff)
        pseudo_prox = pseudo_absdiff_means - np.nanmean(pseudo_absdiff_means)
        p_prox = ((np.sum(pseudo_prox >= real_prox) + 1)
                  / (len(pseudo_prox) + 1))
        rows.append({
            "feature": feature, "metric": "edge_proximity",
            "n_sessions": len(sessions),
            "real_mean": round(real_prox, 5),
            "pseudo_mean": 0.0,
            "p_vs_pseudo": round(float(p_prox), 4),
            "note": "real_mean = pseudo mean|off-on| minus real mean|off-on|; "
                    "positive = closer than chance",
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# attenuation ceiling
# --------------------------------------------------------------------------
def attenuation_table(reliability: pd.DataFrame,
                      outcome_reliability: float | None) -> pd.DataFrame:
    """Max observable correlation given measurement error in x (and y).

    r_observed_max = sqrt(r_xx * r_yy). If the outcome reliability is unknown,
    report the ceiling under a perfectly reliable outcome, which is the most
    generous case: sqrt(r_xx).
    """
    r_yy = outcome_reliability if outcome_reliability is not None else 1.0
    rows = []
    for _, row in reliability.iterrows():
        r_xx = row["reliability_spearman_brown"]
        ceiling = (float(np.sqrt(max(r_xx, 0.0) * r_yy))
                   if np.isfinite(r_xx) else np.nan)
        rows.append({
            "feature": row["feature"],
            "reliability_vocal": r_xx,
            "reliability_outcome_assumed": r_yy,
            "max_observable_r": None if not np.isfinite(ceiling) else round(ceiling, 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
PRIMARY_FEATURES = ["va_int_edge_proximity", "va_int_edge_synchrony"]


def verdict(r_xx: float) -> str:
    if not np.isfinite(r_xx):
        return "UNKNOWN"
    if r_xx >= RELIABILITY_ADEQUATE:
        return "ADEQUATE"
    if r_xx >= RELIABILITY_MARGINAL:
        return "MARGINAL"
    return "INADEQUATE"


def write_report(out_dir: Path, reliability: pd.DataFrame,
                 edge_surr: pd.DataFrame, attenuation: pd.DataFrame,
                 cfg: dict, n_sessions: int, n_turns_primary: int) -> Path:
    lines = [
        "# Vocal-measure validity QC",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Input: `{cfg['in']}`",
        f"Split scheme: `{cfg['split_scheme']}` "
        f"(n_blocks={cfg['n_blocks']}), seed={cfg['seed']}, "
        f"n_surrogate={cfg['n_surrogate']}",
        f"Sessions scored: {n_sessions}; primary floor-transfer turns: {n_turns_primary}",
        "",
    ]
    if n_sessions < 100:
        lines += [
            f"> **WARNING: only {n_sessions} sessions.** Reliability and",
            "> surrogate estimates below are unstable at this sample size and",
            "> should be treated as a smoke test, not as evidence. Re-run on the",
            "> full corpus before drawing any conclusion.",
            "",
        ]
    lines += [
        "## Why this exists",
        "",
        "A null association between vocal alignment and connection is only",
        "interpretable if the vocal measure is reliable and captures",
        "dyad-specific coordination. If reliability is near zero, or if the",
        "measure does not beat a pseudo-dyad baseline, the null is a",
        "measurement result rather than a theoretical one.",
        "",
        "## 1. Split-half reliability",
        "",
        "| feature | n | r(half A, half B) | Spearman-Brown r_xx | verdict |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in reliability.iterrows():
        lines.append(
            f"| `{row['feature']}` | {row['n_sessions']} | "
            f"{row['r_half_half']} | {row['reliability_spearman_brown']} | "
            f"{verdict(row['reliability_spearman_brown'])} |"
        )

    lines += [
        "",
        "## 2. Edge-specific surrogate (real vs pseudo-dyad)",
        "",
        "| feature | metric | n sessions | real | pseudo | p |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in edge_surr.iterrows():
        lines.append(
            f"| `{row['feature']}` | {row['metric']} | {row['n_sessions']} | "
            f"{row['real_mean']} | {row['pseudo_mean']} | {row['p_vs_pseudo']} |"
        )

    lines += [
        "",
        "## 3. Attenuation ceiling",
        "",
        "Maximum correlation this measure could show with a connection outcome,",
        "given its own unreliability: `sqrt(r_xx * r_yy)`.",
        "",
        "| feature | r_xx (vocal) | r_yy (outcome assumed) | max observable r |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in attenuation.iterrows():
        lines.append(
            f"| `{row['feature']}` | {row['reliability_vocal']} | "
            f"{row['reliability_outcome_assumed']} | {row['max_observable_r']} |"
        )

    # Headline block for the two primary theoretical measures.
    lines += ["", "## 4. Verdict on the primary theoretical measures", ""]
    for feature in PRIMARY_FEATURES:
        rel_row = reliability[reliability["feature"] == feature]
        surr_row = edge_surr[
            (edge_surr["feature"] == feature.split("_")[1])
            & (edge_surr["metric"] == ("edge_synchrony" if "synchrony" in feature
                                       else "edge_proximity"))
        ]
        r_xx = (rel_row["reliability_spearman_brown"].iloc[0]
                if len(rel_row) else np.nan)
        p_surr = surr_row["p_vs_pseudo"].iloc[0] if len(surr_row) else np.nan
        ceiling = attenuation.loc[attenuation["feature"] == feature,
                                  "max_observable_r"]
        ceiling = ceiling.iloc[0] if len(ceiling) else np.nan
        lines += [
            f"### `{feature}`",
            "",
            f"- Reliability r_xx = **{r_xx}** ({verdict(r_xx)})",
            f"- Beats pseudo-dyad chance: p = **{p_surr}**",
            f"- Max observable correlation with a perfectly measured outcome: **{ceiling}**",
            "",
        ]
    lines += [
        "## How to read this",
        "",
        "- If r_xx is INADEQUATE **or** the surrogate p is non-significant for",
        "  the primary measures, the observed null association with connection",
        "  does NOT support a theoretical claim. Fix measurement first; do not",
        "  spend LLM annotation budget on the incremental-validity test yet.",
        "- If r_xx is adequate AND the surrogate is significant, the null is a",
        "  genuine finding and the incremental-validity test is worth running.",
        "- Compare `max observable r` against the observed r = .011. If the",
        "  ceiling is itself small, report the analysis as an equivalence test",
        "  rather than as evidence of absence.",
        "",
    ]
    path = out_dir / "qc_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
def run(cfg: dict) -> Path:
    in_path = Path(cfg["in"])
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    required = {"condition", "speaker", "t_start"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{in_path} missing required columns: {sorted(missing)}")
    if "dyad_id" not in df.columns and "pair_id" not in df.columns:
        raise SystemExit(f"{in_path} has neither dyad_id nor pair_id")

    edge_cols = [c for f in vaft.FEATURES.values()
                 for c in (f[1], f[2]) if c is not None]
    present_edge = [c for c in edge_cols if c in df.columns]
    if not present_edge:
        raise SystemExit(
            f"{in_path} has no onset/offset columns {edge_cols}; the edge "
            "surrogate test cannot run. Re-run extract_acoustic_features.py "
            "with edge extraction enabled."
        )

    print(f"Loaded {len(df):,} turns from {in_path}")
    df = vaft.classify_backchannels(df, cfg["bc_max_words"])
    df = vaft.add_primary_floor_transfer_flags(df, cfg)
    primary = df[df["is_primary_floor_transfer"]].copy()
    print(f"Primary floor-transfer turns: {len(primary):,} / {len(df):,} "
          f"({100 * len(primary) / max(len(df), 1):.1f}%)")

    out_root = Path(cfg["out_root"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"vocal_measure_qc_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a run
    print(f"Run directory: {out_dir}")

    print("[1/3] split-half reliability ...")
    scores = split_half_scores(primary, cfg, cfg["split_scheme"], cfg["n_blocks"])
    scores.to_csv(out_dir / "split_half_session_scores.csv", index=False)
    reliability = summarise_split_half(scores)
    reliability.to_csv(out_dir / "split_half_reliability.csv", index=False)
    n_usable = int(scores["both_halves_usable"].sum())
    print(f"      {n_usable} / {len(scores)} sessions usable in both halves")

    print("[2/3] edge-specific surrogate ...")
    rng = np.random.default_rng(cfg["seed"])
    edge_surr = edge_surrogate_test(primary, cfg, rng)
    edge_surr.to_csv(out_dir / "edge_surrogate.csv", index=False)

    print("[3/3] attenuation ceiling ...")
    attenuation = attenuation_table(reliability, cfg["outcome_reliability"])
    attenuation.to_csv(out_dir / "attenuation_ceiling.csv", index=False)

    report = write_report(out_dir, reliability, edge_surr, attenuation, cfg,
                          n_sessions=len(scores), n_turns_primary=len(primary))

    manifest = {
        "script": str(Path(__file__).resolve().relative_to(PROJECT)),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in cfg.items()},
        "n_turns_input": int(len(df)),
        "n_turns_primary": int(len(primary)),
        "n_sessions_scored": int(len(scores)),
        "n_sessions_usable_both_halves": n_usable,
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")

    print("\n--- primary theoretical measures ---")
    for feature in PRIMARY_FEATURES:
        row = reliability[reliability["feature"] == feature]
        if len(row):
            r_xx = row["reliability_spearman_brown"].iloc[0]
            print(f"  {feature:28s} r_xx = {r_xx}  [{verdict(r_xx)}]")
        else:
            print(f"  {feature:28s} not present in input")
    if not edge_surr.empty:
        print("--- edge surrogate ---")
        for _, row in edge_surr.iterrows():
            flag = " *" if (np.isfinite(row["p_vs_pseudo"])
                            and row["p_vs_pseudo"] < 0.05) else ""
            print(f"  {row['feature']:5s} {row['metric']:15s} "
                  f"real {row['real_mean']} vs pseudo {row['pseudo_mean']} "
                  f"p={row['p_vs_pseudo']}{flag}")
    print(f"\nReport: {report}")
    return out_dir


def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True,
                   help="acoustic turn table (acoustic_turns*.csv)")
    p.add_argument("--out-root", required=True,
                   help="parent directory; a timestamped run dir is created inside")
    p.add_argument("--split-scheme", choices=["blocks", "time"], default="blocks",
                   help="how to split each session into halves (default: blocks)")
    p.add_argument("--n-blocks", type=int, default=10,
                   help="number of alternating contiguous blocks for --split-scheme blocks")
    p.add_argument("--outcome-reliability", type=float, default=None,
                   help="r_yy from qc_candor_outcome_reliability.py; "
                        "if omitted the ceiling assumes a perfect outcome")
    p.add_argument("--n-surrogate", type=int, default=vaft.CONFIG["n_surrogate"])
    p.add_argument("--n-baseline-shuffles", type=int,
                   default=vaft.CONFIG["n_baseline_shuffles"])
    p.add_argument("--min-switches", type=int, default=vaft.CONFIG["min_switches"],
                   help="minimum speaker switches required WITHIN EACH HALF")
    p.add_argument("--bc-max-words", type=int, default=vaft.CONFIG["bc_max_words"])
    p.add_argument("--min-turn-s", type=float, default=vaft.CONFIG["min_turn_s"])
    p.add_argument("--min-coverage", type=float, default=vaft.CONFIG["min_coverage"])
    p.add_argument("--seed", type=int, default=vaft.CONFIG["seed"])
    a = p.parse_args()

    cfg = dict(vaft.CONFIG)
    cfg.update({
        "in": a.inp,
        "out_root": a.out_root,
        "split_scheme": a.split_scheme,
        "n_blocks": a.n_blocks,
        "outcome_reliability": a.outcome_reliability,
        "n_surrogate": a.n_surrogate,
        "n_baseline_shuffles": a.n_baseline_shuffles,
        "min_switches": a.min_switches,
        "bc_max_words": a.bc_max_words,
        "min_turn_s": a.min_turn_s,
        "min_coverage": a.min_coverage,
        "seed": a.seed,
    })
    return cfg


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""
compute_vocal_alignment.py
=============================
Turn-adjacent vocal entrainment metrics for the CVS study.

Consumes the per-turn prosody table from extract_acoustic_features.py
(acoustic_turns.csv) and produces dyad-level vocal-alignment features
(vocal_alignment_dyad.csv) keyed by pair_id + condition.

Why turn-adjacent (and not continuous synchrony): the recordings are a single
shared microphone, so two voices are never separable at the same instant. We
therefore use the established Levitan & Hirschberg (2011) family of measures,
computed on prosodic features at SPEAKER SWITCHES:

  PROXIMITY   how close partner's turn-prosody is at adjacent turns, relative
              to a non-adjacent (shuffled-partner) baseline. Higher = more
              entrained. Operationalised here as:
                  proximity = |baseline_diff| - |adjacent_diff|
  CONVERGENCE do the two speakers grow MORE similar over the session? Slope of
              |A_t - B_t| regressed on turn-pair index; we report -slope so
              that higher = converging.
  SYNCHRONY   do the two speakers' turn-by-turn feature values co-vary? Pearson
              r between speaker-A turn series and the temporally-adjacent
              speaker-B turn series. Higher = more synchronous.

Features entrained on: f0 (pitch), intensity (energy), rate (speaking rate).
f0 and intensity are z-scored WITHIN speaker first, so we capture coordination
of contour rather than baseline voice differences.

Edge-aware variant: alignment is also computed on turn-EDGE prosody
(speaker A's turn-offset -> speaker B's turn-onset), which is the most
defensible "sub-second" entrainment signal at the transition point.

Output: one row per (pair_id, condition) with columns like
  va_f0_proximity, va_f0_convergence, va_f0_synchrony,
  va_int_*, va_rate_*, plus edge_* variants and n_switch.

NOTE on condition: piper/cloudy are kept as SEPARATE rows (condition is an
experimental manipulation; do NOT average it away).
test_vocal_alignment_incremental_validity.py enters condition as a covariate.

Usage:
  python scripts/acoustic_alignment/compute_vocal_alignment.py
  python scripts/acoustic_alignment/compute_vocal_alignment.py --in 04_data/acoustic_turns.csv --out ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/

CONFIG = {
    "in": PROJECT / "04_data" / "acoustic_turns.csv",
    "out": PROJECT / "04_data" / "vocal_alignment_dyad.csv",
    "n_baseline_shuffles": 200,   # for proximity baseline
    "min_switches": 5,            # need at least this many speaker switches
    "n_surrogate": 2000,          # cross-dyad pseudo-pairs for the chance test
    "bc_max_words": 2,            # backchannel: at most this many words
    "seed": 42,
}

# feature -> (whole-turn column, onset column, offset column)
FEATURES = {
    "f0":   ("f0_mean",  "onset_f0",  "offset_f0"),
    "int":  ("int_mean", "onset_int", "offset_int"),
    "rate": ("rate_sylps", None, None),  # rate has no meaningful sub-second edge
}

# continuer / backchannel lexicon (listener feedback that does not take the floor)
BACKCHANNELS = {
    "yeah", "yea", "yep", "yes", "right", "okay", "ok", "mm", "mhm", "mmhm",
    "uh-huh", "uhhuh", "hmm", "hm", "sure", "totally", "definitely", "exactly",
    "true", "nice", "wow", "oh", "ah", "gotcha", "cool", "agreed",
}


def _norm_text(s):
    import re
    return re.sub(r"[^a-z' ]", "", str(s).lower()).strip()


def classify_backchannels(df, max_words):
    """Flag turns that are backchannels: short continuers that DON'T take the
    floor (the OTHER speaker holds it both before and after).

    Rule (per session, time-ordered):
      is_backchannel = (<= max_words) AND (all words in BACKCHANNELS lexicon)
                       AND (previous and next turn are the SAME other speaker)
    The floor-not-transferred clause is what separates a true continuer
    ("A... [B: yeah] ...A") from an agreement opener that starts a real
    exchange. Returns df with a boolean 'is_backchannel' column.
    """
    df = df.copy()
    df["is_backchannel"] = False
    for _, idx in df.groupby(["dyad_id", "condition"]).groups.items():
        g = df.loc[idx].sort_values("t_start")
        order = g.index.tolist()
        for k, i in enumerate(order):
            words = _norm_text(df.at[i, "text"]).split() if "text" in df else []
            if not words or len(words) > max_words:
                continue
            if not all(w in BACKCHANNELS for w in words):
                continue
            spk = df.at[i, "speaker"]
            prev_spk = df.at[order[k - 1], "speaker"] if k > 0 else None
            next_spk = df.at[order[k + 1], "speaker"] if k < len(order) - 1 else None
            # floor stays with the OTHER speaker on both sides (or at an edge)
            neigh = [s for s in (prev_spk, next_spk) if s is not None]
            if neigh and all(s != spk for s in neigh):
                df.at[i, "is_backchannel"] = True
    return df


def zscore_within_speaker(df, col):
    out = df.groupby("speaker")[col].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else np.nan)
    )
    return out


def _adjacent_pairs(turns):
    """Yield (prev_row, next_row) for every speaker switch, in time order."""
    turns = turns.sort_values("t_start").reset_index(drop=True)
    for i in range(1, len(turns)):
        a, b = turns.iloc[i - 1], turns.iloc[i]
        if a["speaker"] != b["speaker"]:
            yield a, b


def proximity(turns, col, rng, n_shuffle):
    """|baseline_diff| - |adjacent_diff| averaged over switches.

    Baseline: pair each B-turn with a RANDOM A-turn (shuffled), repeat n_shuffle
    times -> expected absolute difference if turns were not locally coordinated.
    """
    adj = [abs(a[col] - b[col]) for a, b in _adjacent_pairs(turns)
           if np.isfinite(a[col]) and np.isfinite(b[col])]
    if len(adj) < 2:
        return np.nan
    adj_mean = float(np.mean(adj))

    vals = turns[["speaker", col]].dropna()
    spk = vals["speaker"].unique()
    if len(spk) != 2:
        return np.nan
    a_vals = vals[vals["speaker"] == spk[0]][col].to_numpy()
    b_vals = vals[vals["speaker"] == spk[1]][col].to_numpy()
    if a_vals.size < 2 or b_vals.size < 2:
        return np.nan
    base = []
    for _ in range(n_shuffle):
        ia = rng.integers(0, a_vals.size, size=min(a_vals.size, b_vals.size))
        ib = rng.integers(0, b_vals.size, size=min(a_vals.size, b_vals.size))
        base.append(np.mean(np.abs(a_vals[ia] - b_vals[ib])))
    return float(np.mean(base) - adj_mean)  # >0 => closer than chance


def convergence(turns, col):
    """-slope of |A_t - B_t| over adjacent-pair index. Higher => converging."""
    diffs = [abs(a[col] - b[col]) for a, b in _adjacent_pairs(turns)
             if np.isfinite(a[col]) and np.isfinite(b[col])]
    if len(diffs) < 4:
        return np.nan
    x = np.arange(len(diffs), dtype=float)
    slope = np.polyfit(x, diffs, 1)[0]
    return float(-slope)


def synchrony(turns, col):
    """Pearson r between the A-side and B-side series of adjacent pairs."""
    a_series, b_series = [], []
    for a, b in _adjacent_pairs(turns):
        if np.isfinite(a[col]) and np.isfinite(b[col]):
            a_series.append(a[col])
            b_series.append(b[col])
    if len(a_series) < 4:
        return np.nan
    if np.std(a_series) == 0 or np.std(b_series) == 0:
        return np.nan
    return float(np.corrcoef(a_series, b_series)[0, 1])


def edge_alignment(turns, offset_col, onset_col, rng, n_shuffle):
    """Entrainment at the transition point: A's turn-OFFSET prosody ->
    B's turn-ONSET prosody. Returns (proximity, convergence, synchrony)."""
    pairs = [(a[offset_col], b[onset_col]) for a, b in _adjacent_pairs(turns)
             if np.isfinite(a[offset_col]) and np.isfinite(b[onset_col])]
    if len(pairs) < 4:
        return np.nan, np.nan, np.nan
    off = np.array([p[0] for p in pairs])
    on = np.array([p[1] for p in pairs])
    adj = np.abs(off - on)
    # proximity vs shuffled baseline
    base = [np.mean(np.abs(off - rng.permutation(on))) for _ in range(n_shuffle)]
    prox = float(np.mean(base) - adj.mean())
    # convergence
    x = np.arange(len(adj), dtype=float)
    conv = float(-np.polyfit(x, adj, 1)[0])
    # synchrony
    sync = (float(np.corrcoef(off, on)[0, 1])
            if np.std(off) and np.std(on) else np.nan)
    return prox, conv, sync


def process_session(turns, cfg, rng):
    """All vocal-alignment features for one (pair_id, condition)."""
    n_switch = sum(1 for _ in _adjacent_pairs(turns))
    row = {"n_turns": len(turns), "n_switch": n_switch}
    if n_switch < cfg["min_switches"]:
        return row  # too sparse; downstream can drop on n_switch

    for feat, (whole, onset, offset) in FEATURES.items():
        if whole not in turns:
            continue
        t = turns.copy()
        # z-score within speaker for f0 / intensity (not rate)
        if feat in ("f0", "int"):
            t[whole] = zscore_within_speaker(t, whole)
        row[f"va_{feat}_proximity"] = proximity(t, whole, rng, cfg["n_baseline_shuffles"])
        row[f"va_{feat}_convergence"] = convergence(t, whole)
        row[f"va_{feat}_synchrony"] = synchrony(t, whole)
        # edge-based (sub-second) variant where available
        if onset and offset and onset in t and offset in t:
            if feat in ("f0", "int"):
                t[onset] = zscore_within_speaker(t, onset)
                t[offset] = zscore_within_speaker(t, offset)
            p, c, s = edge_alignment(t, offset, onset, rng, cfg["n_baseline_shuffles"])
            row[f"va_{feat}_edge_proximity"] = p
            row[f"va_{feat}_edge_convergence"] = c
            row[f"va_{feat}_edge_synchrony"] = s
    return row


def _compute_all_sessions(df, cfg, rng):
    rows = []
    for (pid, cond), grp in df.groupby(["pair_id", "condition"]):
        feats = process_session(grp, cfg, rng)
        feats.update({"pair_id": pid, "condition": cond})
        rows.append(feats)
    out = pd.DataFrame(rows)
    lead = ["pair_id", "condition", "n_turns", "n_switch"]
    return out[lead + [c for c in out.columns if c not in lead]]


def surrogate_test(df, cfg, rng):
    """Are REAL dyads more synchronised than PSEUDO dyads (chance)?

    For each whole-turn feature, build each session's adjacency series
    (A-side values, B-side values). Real synchrony = corr within a session.
    Pseudo synchrony = corr of speaker-A's series from one session with
    speaker-B's series from a DIFFERENT session (never interacted). We compare
    the mean real synchrony against a null distribution of mean pseudo
    synchrony. This is the standard 'above-chance entrainment' check that a
    reviewer will ask for (proximity already has its own shuffled baseline).
    """
    feats = {k: v[0] for k, v in FEATURES.items()}
    # collect per-session A/B adjacency series for each feature
    series = {f: [] for f in feats}
    for (_pid, _cond), grp in df.groupby(["pair_id", "condition"]):
        if sum(1 for _ in _adjacent_pairs(grp)) < cfg["min_switches"]:
            continue
        for f, col in feats.items():
            if col not in grp:
                continue
            t = grp.copy()
            if f in ("f0", "int"):
                t[col] = zscore_within_speaker(t, col)
            a_s, b_s = [], []
            for a, b in _adjacent_pairs(t):
                if np.isfinite(a[col]) and np.isfinite(b[col]):
                    a_s.append(a[col]); b_s.append(b[col])
            if len(a_s) >= 4:
                series[f].append((np.array(a_s), np.array(b_s)))

    def _corr(a, b):
        n = min(len(a), len(b))
        if n < 4 or np.std(a[:n]) == 0 or np.std(b[:n]) == 0:
            return np.nan
        return np.corrcoef(a[:n], b[:n])[0, 1]

    rows = []
    for f, lst in series.items():
        if len(lst) < 3:
            continue
        real = np.nanmean([_corr(a, b) for a, b in lst])
        pseudo_means = []
        for _ in range(cfg["n_surrogate"] // 50):
            vals = []
            for _ in range(50):
                i, j = rng.integers(0, len(lst), size=2)
                if i == j:
                    continue
                vals.append(_corr(lst[i][0], lst[j][1]))  # A_i vs B_j
            if vals:
                pseudo_means.append(np.nanmean(vals))
        pseudo_means = np.array(pseudo_means)
        p = (np.sum(np.abs(pseudo_means) >= abs(real)) + 1) / (len(pseudo_means) + 1)
        rows.append({"feature": f, "metric": "synchrony",
                     "real_mean": round(float(real), 4),
                     "pseudo_mean": round(float(np.nanmean(pseudo_means)), 4),
                     "p_vs_pseudo": round(float(p), 4),
                     "n_sessions": len(lst)})
    return pd.DataFrame(rows)


def run(cfg):
    df = pd.read_csv(cfg["in"])
    needed = {"pair_id", "condition", "speaker", "t_start"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"{cfg['in']} missing columns: {missing}")

    df = classify_backchannels(df, cfg["bc_max_words"])
    n_bc = int(df["is_backchannel"].sum())
    print(f"Backchannels flagged: {n_bc} / {len(df)} turns "
          f"({100*n_bc/len(df):.1f}%).")

    rng = np.random.default_rng(cfg["seed"])
    Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)

    # --- MAIN: exclude backchannels from the turn-adjacency sequence ----------
    main = _compute_all_sessions(df[~df["is_backchannel"]], cfg, rng)
    main.to_csv(cfg["out"], index=False)
    print(f"[main, backchannels EXCLUDED] {len(main)} sessions x {main.shape[1]} "
          f"cols -> {cfg['out']}")

    # --- SENSITIVITY: include backchannels -----------------------------------
    sens_path = str(cfg["out"]).replace(".csv", "_with_bc.csv")
    sens = _compute_all_sessions(df, cfg, rng)
    sens.to_csv(sens_path, index=False)
    print(f"[sensitivity, backchannels INCLUDED] -> {sens_path}")

    # --- SURROGATE: real vs pseudo-dyad synchrony (chance test) ---------------
    surr_path = str(cfg["out"]).replace(".csv", "_surrogate.csv")
    surr = surrogate_test(df[~df["is_backchannel"]], cfg, rng)
    surr.to_csv(surr_path, index=False)
    print(f"[surrogate, real vs pseudo-dyad] -> {surr_path}")
    if not surr.empty:
        print("  real-vs-chance synchrony:")
        for _, r in surr.iterrows():
            flag = " *" if r["p_vs_pseudo"] < 0.05 else ""
            print(f"    {r['feature']:5s}: real {r['real_mean']:+.3f} vs "
                  f"pseudo {r['pseudo_mean']:+.3f}  p={r['p_vs_pseudo']:.3f}{flag}")

    low = (main["n_switch"] < cfg["min_switches"]).sum()
    if low:
        print(f"[warn] {low} sessions below min_switches={cfg['min_switches']}.")
    return main


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default=str(CONFIG["in"]))
    p.add_argument("--out", default=str(CONFIG["out"]))
    p.add_argument("--n-baseline-shuffles", type=int, default=CONFIG["n_baseline_shuffles"])
    p.add_argument("--min-switches", type=int, default=CONFIG["min_switches"])
    p.add_argument("--n-surrogate", type=int, default=CONFIG["n_surrogate"])
    p.add_argument("--bc-max-words", type=int, default=CONFIG["bc_max_words"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    a = p.parse_args()
    cfg = dict(CONFIG)
    cfg.update({"in": a.inp, "out": a.out,
                "n_baseline_shuffles": a.n_baseline_shuffles,
                "min_switches": a.min_switches, "n_surrogate": a.n_surrogate,
                "bc_max_words": a.bc_max_words, "seed": a.seed})
    return cfg


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""
07_align_manual_labels_to_whisperx.py
=====================================
Put your MANUAL (corrected) speaker labels onto WhisperX WORD TIMESTAMPS,
without re-running pyannote diarization.

The problem this solves
-----------------------
  * Manual .txt (all_srt/<cond>/*.txt)  -> CORRECT speaker + turn boundaries,
                                           but NO timestamps.
  * WhisperX .srt (outputs/.../*.srt)   -> accurate word timestamps,
                                           but WRONG (pyannote) speaker labels.

We align the two word sequences (they are the same words, just segmented
differently) and copy each WhisperX word's start/end time onto the matching
word in your manual transcript. Each manual turn then gets
[t_start, t_end] = [first word start, last word end], keeping YOUR speaker
label. pyannote is never trusted for "who".

This is forced alignment by text, and it is non-destructive: your .txt files
are read-only inputs and are never modified.

Output: labeled_turns.csv with schema compatible with
08_extract_acoustic_features.py
  dyad_id, pair_id, condition, Order, turn_idx, speaker, role,
  t_start, t_end, dur, n_words, text, coverage

`coverage` = fraction of the turn's words that received a timestamp from
alignment (a per-turn QC flag; low coverage -> treat that turn's prosody
cautiously).

Then run acoustic extraction in transfer mode:
  python scripts/03_acoustic_alignment/08_extract_acoustic_features.py \
      --turns-csv 04_data/labeled_turns.csv \
      --out 04_data/acoustic_turns.csv

Dependencies: numpy, pandas (stdlib difflib for alignment)
Usage:
  python scripts/03_acoustic_alignment/07_align_manual_labels_to_whisperx.py
  python scripts/03_acoustic_alignment/07_align_manual_labels_to_whisperx.py --limit 1
"""
from __future__ import annotations

import argparse
import difflib
import glob
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/

CONFIG = {
    "manual_dirs": {
        "piper": PROJECT / "01_pipeline" / "all_srt" / "piper",
        "cloudy": PROJECT / "01_pipeline" / "all_srt" / "cloudy",
    },
    # WhisperX SRTs (word-highlighted) live in per-dyad output dirs
    "outputs_dir": PROJECT / "01_pipeline" / "outputs",
    "log_path": PROJECT / "04_data" / "Discussion Transcription Log - Sheet1.csv",
    "out": PROJECT / "04_data" / "labeled_turns.csv",
    "min_turn_words": 1,
}

SPK_RE = re.compile(r"^\[?(SPEAKER_\d+)\]?\s*:\s*(.*)", re.IGNORECASE)
Q_RE = re.compile(r"^\s*Question[_\s]*(\d+)", re.IGNORECASE)
TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
U_RE = re.compile(r"<u>(.*?)</u>")


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


# ─── manual transcript -> turns (with word lists) ─────────────────────────────
def parse_manual_txt(path):
    turns = []
    curr_q = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mq = Q_RE.match(line)
            if mq:
                curr_q = int(mq.group(1))
                continue
            m = SPK_RE.match(line)
            if not m:
                continue
            spk, txt = m.group(1).upper(), m.group(2).strip()
            words = [w for w in txt.split() if w]
            if len(words) >= 1:
                turns.append({"question": curr_q, "speaker": spk,
                              "text": txt, "words": words})
    return turns


# ─── WhisperX srt -> word timeline ────────────────────────────────────────────
def _ts(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt_word_timeline(path):
    """Return [(word, start, end)] using the <u>underlined</u> token per cue.

    WhisperX word-highlight SRTs advance one underlined word per cue; that
    word's timestamp is the cue's [start,end]. Falls back to splitting the
    cue text evenly if no <u> tag is present.
    """
    timeline = []
    start = end = None
    text_lines = []

    def flush():
        nonlocal start, end, text_lines
        if start is None or not text_lines:
            start = end = None
            text_lines = []
            return
        raw = " ".join(text_lines)
        raw = SPK_RE.sub(lambda m: m.group(2), raw)  # strip speaker prefix
        # ONLY <u>highlighted</u> cues carry real per-word onsets. The
        # interleaved no-highlight cues are intermediate frames repeating the
        # whole sentence with a ~20 ms span -- using them would cluster many
        # words at one timestamp, so we skip them entirely.
        for w in U_RE.findall(raw):
            nw = norm(w)
            if nw:
                timeline.append((nw, start, end))
        start = end = None
        text_lines = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                flush()
                continue
            if line.isdigit():
                continue
            m = TS_RE.match(line)
            if m:
                start = _ts(*m.group(1, 2, 3, 4))
                end = _ts(*m.group(5, 6, 7, 8))
                continue
            text_lines.append(line)
    flush()
    # de-duplicate consecutive identical (word,start) from repeated cues
    dedup = []
    for w, s, e in timeline:
        if dedup and dedup[-1][0] == w and abs(dedup[-1][1] - s) < 1e-3:
            continue
        dedup.append([w, s, e])
    # The word-highlight SRT has ~20 ms cue durations, so per-cue END times are
    # unreliable. Reconstruct each word's END as the NEXT word's START (last word
    # keeps a small default). This yields realistic per-word spans for prosody.
    for i in range(len(dedup) - 1):
        nxt_start = dedup[i + 1][1]
        if nxt_start > dedup[i][1]:
            dedup[i][2] = nxt_start
    if dedup:
        dedup[-1][2] = max(dedup[-1][2], dedup[-1][1] + 0.30)
    return [tuple(x) for x in dedup]


# ─── align manual words <-> srt word timeline ─────────────────────────────────
def align_turns(manual_turns, srt_timeline):
    """Assign timestamps to each manual turn via difflib word alignment.

    Returns list of turns augmented with t_start, t_end, coverage.
    """
    manual_words = []
    owner = []  # turn index for each manual word
    for ti, t in enumerate(manual_turns):
        for w in t["words"]:
            nw = norm(w)
            if nw:
                manual_words.append(nw)
                owner.append(ti)

    srt_words = [w for (w, _, _) in srt_timeline]
    sm = difflib.SequenceMatcher(a=manual_words, b=srt_words, autojunk=False)

    # per manual-word timestamp (None if unmatched)
    wt = [None] * len(manual_words)
    for a0, b0, size in sm.get_matching_blocks():
        for k in range(size):
            _, s, e = srt_timeline[b0 + k]
            wt[a0 + k] = (s, e)

    # collapse to per-turn spans
    per_turn = {ti: [] for ti in range(len(manual_turns))}
    for i, ts in enumerate(wt):
        per_turn[owner[i]].append(ts)

    out = []
    last_end = 0.0
    for ti, t in enumerate(manual_turns):
        stamps = [x for x in per_turn[ti] if x is not None]
        n_words = len(per_turn[ti])
        coverage = len(stamps) / n_words if n_words else 0.0
        if stamps:
            t_start = min(s for s, _ in stamps)
            t_end = max(e for _, e in stamps)
            last_end = t_end
        else:
            # no matched words: place just after previous turn (flagged by coverage=0)
            t_start = last_end
            t_end = last_end
        out.append({**t, "t_start": round(t_start, 3),
                    "t_end": round(t_end, 3), "coverage": round(coverage, 3)})
    return out


# ─── plumbing (reused conventions from 02/03) ─────────────────────────────────
def meta_from_name(name):
    dyad_id = (re.search(r"(dyad\d+_\d+)", name) or [None, None])[1] \
        if re.search(r"(dyad\d+_\d+)", name) else None
    m = re.search(r"(dyad\d+_\d+)", name)
    dyad_id = m.group(1) if m else None
    mp = re.search(r"dyad(\d+)_", name)
    pid = int(mp.group(1)) if mp else None
    return dyad_id, pid


def load_log(log_path):
    if not Path(log_path).exists():
        return {}, {}
    log = pd.read_csv(log_path).dropna(subset=["Dyad ID"])
    log["Dyad ID"] = log["Dyad ID"].astype(str).str.strip()
    role_map, order_map = {}, {}
    for _, r in log.iterrows():
        d = r["Dyad ID"]
        order_map[d] = r.get("Order")
        for cond, c0, c1 in [("piper", "Piper_00", "Piper_01"),
                             ("cloudy", "Cloudy_00", "Cloudy_01")]:
            if c0 in r and c1 in r:
                role_map[(d, cond)] = {"SPEAKER_00": str(r[c0]).strip().upper(),
                                       "SPEAKER_01": str(r[c1]).strip().upper()}
    return role_map, order_map


def find_srt(outputs_dir, stem):
    hits = glob.glob(str(Path(outputs_dir) / f"{stem}*" / "*.srt"))
    if hits:
        return hits[0]
    hits = glob.glob(str(Path(outputs_dir) / "**" / f"{stem}*.srt"), recursive=True)
    return hits[0] if hits else None


def run(cfg):
    role_map, order_map = load_log(cfg["log_path"])
    rows = []
    sessions = []
    for cond, d in cfg["manual_dirs"].items():
        for txt in sorted(glob.glob(str(Path(d) / "*.txt"))):
            sessions.append((cond, txt))
    if cfg.get("limit"):
        sessions = sessions[: cfg["limit"]]

    for cond, txt in sessions:
        stem = Path(txt).stem  # e.g. dyad2_250429_cloudy_discussion
        dyad_id, pid = meta_from_name(stem)
        manual = parse_manual_txt(txt)
        srt = find_srt(cfg["outputs_dir"], stem)
        if srt is None:
            print(f"[warn] no WhisperX SRT for {stem}; cannot timestamp -> skipped")
            continue
        timeline = parse_srt_word_timeline(srt)
        if not timeline:
            print(f"[warn] empty timeline for {stem}; skipped")
            continue
        turns = align_turns(manual, timeline)
        roles = role_map.get((dyad_id, cond), {})
        order = order_map.get(dyad_id)
        for i, t in enumerate(turns):
            rows.append({
                "dyad_id": dyad_id, "pair_id": pid, "condition": cond,
                "Order": order, "turn_idx": i, "speaker": t["speaker"],
                "role": roles.get(t["speaker"]),
                "t_start": t["t_start"], "t_end": t["t_end"],
                "dur": round(t["t_end"] - t["t_start"], 3),
                "n_words": len(t["words"]), "text": t["text"],
                "coverage": t["coverage"], "question": t["question"],
            })
        cov = sum(t["coverage"] for t in turns) / max(1, len(turns))
        print(f"  {stem}: {len(turns)} turns, mean word coverage {cov:.2f}")

    if not rows:
        raise SystemExit("No turns produced.")
    df = pd.DataFrame(rows)
    # sanity: monotonic time within session (warn on inversions)
    bad = 0
    for (pid, cond), g in df.groupby(["pair_id", "condition"]):
        if not g["t_start"].is_monotonic_increasing:
            bad += 1
    if bad:
        print(f"[warn] {bad} sessions have non-monotonic turn starts "
              f"(low-coverage turns) -- inspect before prosody extraction.")
    Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["out"], index=False)
    print(f"\nWrote {len(df)} labeled turns -> {cfg['out']}")
    lowcov = (df["coverage"] < 0.5).sum()
    print(f"[qc] {lowcov} turns have <50% word coverage "
          f"({100*lowcov/len(df):.1f}%).")
    return df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", default=str(CONFIG["outputs_dir"]))
    p.add_argument("--log-path", default=str(CONFIG["log_path"]))
    p.add_argument("--out", default=str(CONFIG["out"]))
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    cfg = dict(CONFIG)
    cfg.update({"outputs_dir": a.outputs_dir, "log_path": a.log_path,
                "out": a.out, "limit": a.limit})
    return cfg


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""
Inspect one low-coverage labeled turn against the manual transcript and
WhisperX word timeline.

This is a read-only diagnostic helper for turns flagged in
04_data/labeled_turns.csv, especially rows with coverage == 0 or t_start >=
t_end. It does not modify transcripts, SRTs, or derived CSVs.

Examples:
  python scripts/acoustic_alignment/inspect_turn_alignment.py \
      --dyad-id dyad12_250928 --condition piper --turn-idx 13

  python scripts/acoustic_alignment/inspect_turn_alignment.py \
      --dyad-id dyad22_251022 --condition cloudy --turn-idx 80 --window-sec 20
"""
from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path

import pandas as pd

from align_manual_labels_to_whisperx import (
    CONFIG,
    find_srt,
    norm,
    parse_manual_txt,
    parse_srt_word_timeline,
)


PROJECT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
SRT_ROOT = Path(os.environ.get("CVS_SRT_ROOT", PROJECT / "01_pipeline" / "all_srt"))
WHISPERX_OUTPUTS = Path(
    os.environ.get("CVS_WHISPERX_OUTPUTS", PROJECT / "01_pipeline" / "outputs")
)


def find_manual_txt(dyad_id: str, condition: str, srt_root: Path) -> Path:
    expected = srt_root / condition / f"{dyad_id}_{condition}_discussion.txt"
    if expected.exists():
        return expected
    hits = sorted((srt_root / condition).glob(f"{dyad_id}_{condition}*.txt"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No manual transcript found for {dyad_id} {condition} under {srt_root}")


def normalized_words(text: str) -> list[str]:
    return [norm(w) for w in str(text).split() if norm(w)]


def print_manual_context(manual_turns: list[dict], turn_idx: int, context: int) -> None:
    print("\nMANUAL TRANSCRIPT CONTEXT")
    lo = max(0, turn_idx - context)
    hi = min(len(manual_turns), turn_idx + context + 1)
    for idx in range(lo, hi):
        turn = manual_turns[idx]
        marker = ">>" if idx == turn_idx else "  "
        print(
            f"{marker} turn={idx} q={turn['question']} speaker={turn['speaker']} "
            f"words={len(turn['words'])}: {turn['text']}"
        )
        if idx == turn_idx:
            print(f"   normalized: {' '.join(normalized_words(turn['text']))}")


def srt_words_in_window(timeline: list[tuple[str, float, float]], center: float, window_sec: float) -> list[tuple[str, float, float]]:
    lo = center - window_sec
    hi = center + window_sec
    return [(w, s, e) for w, s, e in timeline if lo <= s <= hi]


def print_srt_time_context(timeline: list[tuple[str, float, float]], center: float, window_sec: float) -> None:
    words = srt_words_in_window(timeline, center, window_sec)
    print(f"\nWHISPERX SRT WORDS AROUND {center:.3f}s +/- {window_sec:.1f}s")
    if not words:
        print("No WhisperX words found in this time window.")
        return
    chunks = []
    for w, s, _ in words:
        chunks.append(f"{w}({s:.1f})")
    print(" ".join(chunks))


def best_srt_text_matches(
    timeline: list[tuple[str, float, float]],
    target_words: list[str],
    n: int = 5,
) -> list[tuple[float, float, float, str]]:
    if not target_words:
        return []
    srt_words = [w for w, _, _ in timeline]
    target_len = len(target_words)
    min_len = max(1, target_len - 3)
    max_len = min(len(srt_words), target_len + 5)
    candidates = []
    for win_len in range(min_len, max_len + 1):
        for start in range(0, max(0, len(srt_words) - win_len + 1)):
            window = srt_words[start:start + win_len]
            ratio = difflib.SequenceMatcher(a=target_words, b=window, autojunk=False).ratio()
            if ratio <= 0:
                continue
            t0 = timeline[start][1]
            t1 = timeline[start + win_len - 1][2]
            candidates.append((ratio, t0, t1, " ".join(window)))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:n]


def run(args: argparse.Namespace) -> None:
    labeled = pd.read_csv(args.labeled_turns)
    row = labeled[
        (labeled["dyad_id"].astype(str) == args.dyad_id)
        & (labeled["condition"].astype(str) == args.condition)
        & (labeled["turn_idx"].astype(int) == args.turn_idx)
    ]
    if row.empty:
        raise SystemExit(f"No labeled turn found for {args.dyad_id} {args.condition} turn_idx={args.turn_idx}")
    row = row.iloc[0]

    manual_path = find_manual_txt(args.dyad_id, args.condition, args.srt_root)
    stem = manual_path.stem
    srt_path = find_srt(args.outputs_dir, stem)
    if not srt_path:
        raise SystemExit(f"No WhisperX SRT found for stem {stem} under {args.outputs_dir}")
    srt_path = Path(srt_path)

    manual_turns = parse_manual_txt(str(manual_path))
    timeline = parse_srt_word_timeline(str(srt_path))
    target_text = str(row.get("text", ""))
    target_words = normalized_words(target_text)

    print("FILES")
    print(f"manual: {manual_path}")
    print(f"whisperx: {srt_path}")
    print(f"labeled_turns: {args.labeled_turns}")
    print("\nLABELED TURN")
    print(row.to_string())

    print_manual_context(manual_turns, args.turn_idx, args.context_turns)
    center = float(row["t_start"]) if pd.notna(row["t_start"]) else 0.0
    print_srt_time_context(timeline, center, args.window_sec)

    print("\nBEST TEXT MATCHES IN WHISPERX WORD TIMELINE")
    matches = best_srt_text_matches(timeline, target_words, n=args.top_n)
    if not matches:
        print("No text matches found.")
    for ratio, t0, t1, text in matches:
        print(f"ratio={ratio:.3f} time={t0:.3f}-{t1:.3f}: {text}")

    print("\nWHAT TO CHECK")
    print("- If the best match is strong but at a different time, the turn boundary may need timestamp correction.")
    print("- If the best match is weak, compare manual wording with WhisperX wording and fix transcript text or alignment source.")
    print("- If this is a short backchannel, consider merging with neighboring turn or accepting it as un-timestampable.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one manual/WhisperX alignment turn.")
    parser.add_argument("--dyad-id", required=True)
    parser.add_argument("--condition", required=True, choices=["piper", "cloudy"])
    parser.add_argument("--turn-idx", required=True, type=int)
    parser.add_argument("--labeled-turns", type=Path, default=DATA / "labeled_turns.csv")
    parser.add_argument("--srt-root", type=Path, default=SRT_ROOT)
    parser.add_argument("--outputs-dir", type=Path, default=WHISPERX_OUTPUTS)
    parser.add_argument("--context-turns", type=int, default=3)
    parser.add_argument("--window-sec", type=float, default=15.0)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""
extract_acoustic_features.py
===============================
Per-turn prosodic feature extraction for the CVS vocal-alignment study.

DESIGN NOTE (read me): the discussion recordings are a SINGLE shared microphone
(the stereo .mp4s are duplicated mono; L - R = -inf dB). There is therefore NO
per-speaker audio stream and no way to measure two voices at the same instant.
Speaker identity comes only from WhisperX diarization ([SPEAKER_00]/[SPEAKER_01]
in the .srt). Consequently this script measures prosody PER TURN (per diarized
segment), and compute_vocal_alignment.py measures TURN-ADJACENT entrainment
(Levitan & Hirschberg style) rather than continuous synchrony.

Pipeline
--------
  for each discussion .srt in OUTPUTS_DIR:
    1. parse cues -> merge consecutive same-speaker cues into turns (IPUs)
    2. load the matching 16k mono WAV
    3. for each turn [t_start, t_end]:
         - whole-turn prosody: f0 (mean/sd/range), intensity (mean/sd),
           speaking rate (syllable proxy / words-per-sec)
         - EDGE windows needed for entrainment:
             onset window  = first  EDGE_WIN_S seconds of the turn
             offset window = last   EDGE_WIN_S seconds of the turn
           (mean f0 + mean intensity within each)
    4. map SPEAKER_xx -> experimental role A/B using the transcription log
  -> write one tidy row per turn to OUTPUT_CSV (acoustic_turns.csv)

f0 is z-scored WITHIN speaker downstream (in 04) so we measure coordination of
contour, not raw male/female pitch differences.

Dependencies:  praat-parselmouth, librosa, numpy, pandas, tqdm, soundfile
  pip install praat-parselmouth librosa soundfile pandas numpy tqdm --break-system-packages

Usage:
  python scripts/acoustic_alignment/extract_acoustic_features.py
  python scripts/acoustic_alignment/extract_acoustic_features.py --outputs-dir ... --wav-dir ... --out ...
  python scripts/acoustic_alignment/extract_acoustic_features.py --limit 1
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── CONFIG (override via CLI) ────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/
DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
WHISPERX_OUTPUTS = Path(
    os.environ.get("CVS_WHISPERX_OUTPUTS", PROJECT / "01_pipeline" / "outputs")
)
WAV_DIR = Path(os.environ.get("CVS_WAV_DIR", PROJECT / "01_pipeline" / "_wav"))

CONFIG = {
    # Defaults point to the local private-data layout used during development.
    # These folders are intentionally ignored by Git. Replicators can either
    # recreate this layout locally or pass --outputs-dir/--wav-dir/--log-path/--out.
    # WhisperX per-dyad output dirs, each containing one *_discussion_16k.srt
    "outputs_dir": WHISPERX_OUTPUTS,
    # 16 kHz mono WAVs produced by the WhisperX pipeline
    "wav_dir": WAV_DIR,
    # experimental log w/ SPEAKER_xx -> role (A/B) mapping + Order
    "log_path": DATA / "Discussion Transcription Log - Sheet1.csv",
    "out": DATA / "acoustic_turns.csv",
    # turn segmentation
    "merge_gap_s": 0.30,   # cues by same speaker within this gap -> one turn
    "min_turn_s": 0.25,    # drop turns shorter than this (unreliable f0)
    # edge windows for entrainment
    "edge_win_s": 0.50,    # length of onset/offset prosody window
    # f0 extraction range (Hz); widen if speakers are very low/high
    "f0_min": 75.0,
    "f0_max": 400.0,
}

SPEAKER_RE = re.compile(r"^\[?(SPEAKER_\d+)\]?\s*:?\s*(.*)", re.IGNORECASE)
TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


# ─── SRT PARSING ──────────────────────────────────────────────────────────────
def _ts_to_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt_cues(srt_path: str):
    """Return list of {start, end, speaker, text} cues in file order.

    The WhisperX SRTs here are word-highlighted: the same sentence repeats with
    one word <u>underlined</u> per cue. We strip tags and keep every cue; turns
    are reconstructed in merge_cues_to_turns().
    """
    cues = []
    start = end = None
    text_lines: list[str] = []

    def flush():
        nonlocal start, end, text_lines
        if start is None:
            return
        raw = " ".join(text_lines).strip()
        raw = re.sub(r"</?u>", "", raw)  # drop underline tags
        m = SPEAKER_RE.match(raw)
        if m:
            spk, txt = m.group(1).upper(), m.group(2).strip()
        else:
            spk, txt = None, raw
        cues.append({"start": start, "end": end, "speaker": spk, "text": txt})
        start = end = None
        text_lines = []

    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                flush()
                continue
            if line.isdigit():  # cue index
                continue
            m = TS_RE.match(line)
            if m:
                start = _ts_to_sec(*m.group(1, 2, 3, 4))
                end = _ts_to_sec(*m.group(5, 6, 7, 8))
                continue
            text_lines.append(line)
    flush()
    return cues


def merge_cues_to_turns(cues, merge_gap_s: float, min_turn_s: float):
    """Collapse consecutive same-speaker cues into turns (IPUs).

    Because word-highlight cues overlap/repeat, a 'turn' = maximal run of cues
    with the same speaker where each cue starts within merge_gap_s of the
    running turn end. Text is de-duplicated by taking the longest cue text seen
    in the run (the fully-revealed sentence).
    """
    turns = []
    cur = None
    for c in cues:
        if c["speaker"] is None or c["start"] is None:
            continue
        if (
            cur is not None
            and c["speaker"] == cur["speaker"]
            and c["start"] <= cur["end"] + merge_gap_s
        ):
            cur["end"] = max(cur["end"], c["end"])
            if len(c["text"]) > len(cur["text"]):
                cur["text"] = c["text"]
        else:
            if cur is not None:
                turns.append(cur)
            cur = {"speaker": c["speaker"], "start": c["start"], "end": c["end"],
                   "text": c["text"]}
    if cur is not None:
        turns.append(cur)
    # filter ultra-short turns
    turns = [t for t in turns if (t["end"] - t["start"]) >= min_turn_s]
    return turns


# ─── ACOUSTIC FEATURES ────────────────────────────────────────────────────────
@dataclass
class TurnAcoustics:
    f0_mean: float
    f0_sd: float
    f0_range: float
    int_mean: float
    int_sd: float
    rate_sylps: float        # syllable nuclei per second (librosa proxy)
    rate_wps: float          # words per second (from SRT text)
    onset_f0: float          # mean f0 in first edge_win_s
    onset_int: float
    offset_f0: float         # mean f0 in last edge_win_s
    offset_int: float


def _safe(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    return arr


def extract_turn_acoustics(sound, t0, t1, text, cfg) -> "TurnAcoustics":
    """Prosody for one turn slice [t0,t1] of a parselmouth Sound.

    Imports parselmouth/librosa lazily so the file can be imported (and the SRT
    logic unit-tested) on machines without the audio stack installed.
    """
    import parselmouth
    from parselmouth.praat import call

    seg = sound.extract_part(from_time=t0, to_time=t1, preserve_times=False)

    # --- pitch (f0) ---
    # Praat needs a window of >= ~3/pitch_floor seconds; very short turns raise.
    # Guard so a sub-window turn yields NaN instead of crashing the whole run.
    try:
        if (t1 - t0) < (3.0 / cfg["f0_min"]):
            raise ValueError("segment too short for pitch")
        pitch = seg.to_pitch(pitch_floor=cfg["f0_min"], pitch_ceiling=cfg["f0_max"])
        f0 = _safe(pitch.selected_array["frequency"])
        f0 = f0[f0 > 0]
        f0_mean = float(np.mean(f0)) if f0.size else np.nan
        f0_sd = float(np.std(f0)) if f0.size else np.nan
        f0_range = float(np.percentile(f0, 95) - np.percentile(f0, 5)) if f0.size else np.nan
    except Exception:
        f0_mean = f0_sd = f0_range = np.nan

    # --- intensity (dB) ---
    try:
        intensity = seg.to_intensity(minimum_pitch=cfg["f0_min"])
        iv = _safe(intensity.values.T.ravel())
        int_mean = float(np.mean(iv)) if iv.size else np.nan
        int_sd = float(np.std(iv)) if iv.size else np.nan
    except Exception:
        int_mean = int_sd = np.nan

    # --- speaking rate ---
    dur = max(1e-6, t1 - t0)
    rate_wps = len(str(text).split()) / dur
    rate_sylps = _syllable_rate(seg, dur, cfg)

    # --- edge windows (turn-onset / turn-offset prosody) ---
    w = min(cfg["edge_win_s"], (t1 - t0) / 2)
    onset_f0, onset_int = _window_means(sound, t0, t0 + w, cfg)
    offset_f0, offset_int = _window_means(sound, t1 - w, t1, cfg)

    return TurnAcoustics(
        f0_mean, f0_sd, f0_range, int_mean, int_sd, rate_sylps, rate_wps,
        onset_f0, onset_int, offset_f0, offset_int,
    )


def _window_means(sound, a, b, cfg):
    """Mean f0 (Hz) and intensity (dB) within [a,b] of the full Sound."""
    import numpy as np
    if b - a < 0.05:
        return np.nan, np.nan
    try:
        seg = sound.extract_part(from_time=max(0, a), to_time=b, preserve_times=False)
        pitch = seg.to_pitch(pitch_floor=cfg["f0_min"], pitch_ceiling=cfg["f0_max"])
        f0 = _safe(pitch.selected_array["frequency"])
        f0 = f0[f0 > 0]
        f0m = float(np.mean(f0)) if f0.size else np.nan
        try:
            iv = _safe(seg.to_intensity(minimum_pitch=cfg["f0_min"]).values.T.ravel())
            im = float(np.mean(iv)) if iv.size else np.nan
        except Exception:
            im = np.nan
        return f0m, im
    except Exception:
        return np.nan, np.nan


def _syllable_rate(seg, dur, cfg):
    """Validated syllable-nuclei rate (De Jong & Wempe, 2009 method).

    Counts syllable nuclei as intensity PEAKS that are (1) loud enough relative
    to the segment max (silence threshold), (2) separated from the previous peak
    by an intensity DIP of >= min_dip dB, and (3) VOICED (pitch defined at the
    peak). Returns nuclei per second. This is the publication-grade replacement
    for the old librosa-onset proxy.
    """
    try:
        from parselmouth.praat import call
        if dur < 0.10:
            return np.nan
        intensity = seg.to_intensity(minimum_pitch=cfg["f0_min"])
        t = intensity.xs()
        v = np.asarray(intensity.values).ravel()
        finite = np.isfinite(v)
        t, v = t[finite], v[finite]
        if v.size < 3:
            return np.nan
        int_max = float(np.max(v))
        silence_db = cfg.get("syl_silence_db", -25.0)   # peaks below max+(-25) ignored
        min_dip = cfg.get("syl_min_dip_db", 2.0)
        thr = int_max + silence_db

        # candidate local maxima above threshold
        peaks = [i for i in range(1, len(v) - 1)
                 if v[i] > v[i - 1] and v[i] >= v[i + 1] and v[i] > thr]

        # require a dip of >= min_dip dB between consecutive accepted peaks
        pitch = seg.to_pitch(pitch_floor=cfg["f0_min"], pitch_ceiling=cfg["f0_max"])
        nuclei = 0
        last = None
        for i in peaks:
            if last is not None:
                dip = v[last:i + 1].min()
                if (v[i] - dip) < min_dip or (v[last] - dip) < min_dip:
                    if v[i] <= v[last]:
                        continue
            # voicing check at the peak time
            f0_here = pitch.get_value_at_time(float(t[i]))
            if not (f0_here and f0_here > 0):
                continue
            nuclei += 1
            last = i
        return nuclei / max(1e-6, dur)
    except Exception:
        return np.nan


# ─── DYAD / ROLE PLUMBING ─────────────────────────────────────────────────────
def dyad_meta_from_name(name: str):
    """'dyad10_250916_cloudy_discussion' -> (dyad_id, pair_id, condition)."""
    dyad_id = None
    m = re.search(r"(dyad\d+_\d+)", name)
    if m:
        dyad_id = m.group(1)
    pid = None
    mp = re.search(r"dyad(\d+)_", name)
    if mp:
        pid = int(mp.group(1))
    cond = "piper" if "piper" in name.lower() else ("cloudy" if "cloud" in name.lower() else None)
    return dyad_id, pid, cond


def load_role_map(log_path):
    """Return {(dyad_id, condition): {'SPEAKER_00': 'A'/'B', 'SPEAKER_01': ...}, ...}
    plus {(dyad_id): Order}."""
    if not Path(log_path).exists():
        print(f"[warn] log not found: {log_path} -- roles will be NaN")
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
                role_map[(d, cond)] = {
                    "SPEAKER_00": str(r[c0]).strip().upper(),
                    "SPEAKER_01": str(r[c1]).strip().upper(),
                }
    return role_map, order_map


def find_wav(wav_dir, srt_stem):
    """Match an SRT stem to its 16k WAV (handles the *_16k suffix)."""
    base = srt_stem.replace("_16k", "")
    for cand in [f"{base}_16k.wav", f"{srt_stem}.wav", f"{base}.wav"]:
        p = Path(wav_dir) / cand
        if p.exists():
            return str(p)
    hits = glob.glob(str(Path(wav_dir) / f"{base}*.wav"))
    return hits[0] if hits else None


# ─── DRIVER ───────────────────────────────────────────────────────────────────
def process_dyad(srt_path, cfg, role_map, order_map):
    name = Path(srt_path).parent.name or Path(srt_path).stem
    dyad_id, pair_id, cond = dyad_meta_from_name(name)
    cues = parse_srt_cues(srt_path)
    turns = merge_cues_to_turns(cues, cfg["merge_gap_s"], cfg["min_turn_s"])
    if not turns:
        return []

    roles = role_map.get((dyad_id, cond), {})
    order = order_map.get(dyad_id)

    # lazy audio load (skipped in --dry-run)
    sound = None
    if not cfg.get("dry_run"):
        import parselmouth
        wav = find_wav(cfg["wav_dir"], Path(srt_path).stem)
        if wav is None:
            print(f"[warn] no WAV for {name}; skipping audio (text-only rows)")
        else:
            sound = parselmouth.Sound(wav)

    rows = []
    for i, t in enumerate(turns):
        row = {
            "dyad_id": dyad_id, "pair_id": pair_id, "condition": cond,
            "Order": order, "turn_idx": i, "speaker": t["speaker"],
            "role": roles.get(t["speaker"]), "t_start": round(t["start"], 3),
            "t_end": round(t["end"], 3), "dur": round(t["end"] - t["start"], 3),
            "n_words": len(str(t["text"]).split()), "text": t["text"],
        }
        if sound is not None:
            ac = extract_turn_acoustics(sound, t["start"], t["end"], t["text"], cfg)
            row.update(asdict(ac))
        rows.append(row)
    return rows


def process_turns_csv(cfg):
    """TRANSFER MODE: turns already timestamped by align_manual_labels_to_whisperx.py
    (manual speaker labels + WhisperX word timestamps). We only add prosody.

    This is the RECOMMENDED path: it uses your corrected diarization and never
    trusts pyannote for 'who spoke'.
    """
    turns_df = pd.read_csv(cfg["turns_csv"])
    rows = []
    import parselmouth
    sound_cache = {}
    for (dyad_id, cond), grp in tqdm(turns_df.groupby(["dyad_id", "condition"]),
                                     desc="sessions"):
        sound = None
        if not cfg.get("dry_run"):
            key = f"{dyad_id}_{cond}"
            if key not in sound_cache:
                stem = f"{dyad_id}_{cond}_discussion"
                wav = find_wav(cfg["wav_dir"], stem)
                sound_cache[key] = parselmouth.Sound(wav) if wav else None
                if wav is None:
                    print(f"[warn] no WAV for {stem}")
            sound = sound_cache[key]
        n_fail = 0
        for _, t in grp.iterrows():
            row = t.to_dict()
            if sound is not None and t["t_end"] > t["t_start"]:
                try:
                    ac = extract_turn_acoustics(sound, t["t_start"], t["t_end"],
                                                t.get("text", ""), cfg)
                    row.update(asdict(ac))
                except Exception as e:  # never let one turn kill the session
                    n_fail += 1
                    row["_acoustic_error"] = str(e)[:80]
            rows.append(row)
        if n_fail:
            print(f"[warn] {dyad_id}_{cond}: {n_fail} turns failed acoustic "
                  f"extraction (left as NaN).")
    df = pd.DataFrame(rows)
    Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["out"], index=False)
    n_ok = df["f0_mean"].notna().sum() if "f0_mean" in df else 0
    print(f"\nWrote {len(df)} turns (transfer mode), {n_ok} with f0 -> {cfg['out']}")
    return df


def run(cfg):
    if cfg.get("turns_csv"):
        return process_turns_csv(cfg)
    role_map, order_map = load_role_map(cfg["log_path"])
    srts = sorted(glob.glob(str(Path(cfg["outputs_dir"]) / "*discussion*" / "*.srt")))
    if cfg.get("limit"):
        srts = srts[: cfg["limit"]]
    if not srts:
        raise SystemExit(f"No SRTs under {cfg['outputs_dir']}")

    all_rows = []
    for s in tqdm(srts, desc="dyads"):
        try:
            all_rows.extend(process_dyad(s, cfg, role_map, order_map))
        except Exception as e:
            print(f"[error] {s}: {e}")
    if not all_rows:
        raise SystemExit("No turns extracted.")

    df = pd.DataFrame(all_rows)
    Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["out"], index=False)
    print(f"\nWrote {len(df)} turns from {df['dyad_id'].nunique()} dyad-sessions "
          f"-> {cfg['out']}")
    miss = df[df["role"].isna()]["dyad_id"].nunique() if "role" in df else 0
    if miss:
        print(f"[warn] {miss} dyad-sessions had unmapped roles (check the log).")
    return df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", default=str(CONFIG["outputs_dir"]))
    p.add_argument("--wav-dir", default=str(CONFIG["wav_dir"]))
    p.add_argument("--log-path", default=str(CONFIG["log_path"]))
    p.add_argument("--out", default=str(CONFIG["out"]))
    p.add_argument("--merge-gap-s", type=float, default=CONFIG["merge_gap_s"])
    p.add_argument("--min-turn-s", type=float, default=CONFIG["min_turn_s"])
    p.add_argument("--edge-win-s", type=float, default=CONFIG["edge_win_s"])
    p.add_argument("--f0-min", type=float, default=CONFIG["f0_min"])
    p.add_argument("--f0-max", type=float, default=CONFIG["f0_max"])
    p.add_argument("--limit", type=int, default=None, help="process only N dyads")
    p.add_argument("--dry-run", action="store_true",
                   help="parse SRT + segment turns only, skip audio (no parselmouth)")
    p.add_argument("--turns-csv", default=None,
                   help="TRANSFER MODE: timestamped turns from align_manual_labels_to_whisperx.py "
                        "(uses your manual diarization instead of WhisperX SRT labels)")
    a = p.parse_args()
    cfg = dict(CONFIG)
    cfg.update({
        "outputs_dir": a.outputs_dir, "wav_dir": a.wav_dir, "log_path": a.log_path,
        "out": a.out, "merge_gap_s": a.merge_gap_s, "min_turn_s": a.min_turn_s,
        "edge_win_s": a.edge_win_s, "f0_min": a.f0_min, "f0_max": a.f0_max,
        "limit": a.limit, "dry_run": a.dry_run, "turns_csv": a.turns_csv,
    })
    return cfg


if __name__ == "__main__":
    run(parse_args())

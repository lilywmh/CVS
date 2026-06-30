#!/usr/bin/env python3
"""
08b_validate_speaker_enrollment.py
==================================
Enrollment-based speaker verification / supervised diarization.

WHAT THIS IS (your idea, formalised)
------------------------------------
Instead of blind clustering (pyannote, which mislabeled your audio), we ENROLL
a voiceprint per speaker from a section you trust, then label the rest of the
audio by matching each segment to the nearest voiceprint:

  1. ENROLL  : from a known-good section (default: Question 1 turns in your
               manual labels), pull each speaker's clean speech, extract a
               speaker EMBEDDING (ECAPA-TDNN x-vector) per turn, average into
               one centroid per speaker.
  2. SCORE   : for every other turn, extract its embedding and take cosine
               similarity to each centroid. Predicted speaker = nearest
               centroid; `margin` = sim(best) - sim(2nd) (confidence).
  3. COMPARE : against your manual label -> agreement rate + a list of
               DISAGREEMENTS and LOW-MARGIN turns to eyeball.

Two intended uses (NOT a replacement for your manual labels):
  --mode validate  (default) : QC your manual diarization with an independent
                               embedding method; report agreement + flags.
  --mode scale               : for NEW dyads, enroll from a clean section and
                               auto-label the rest (emit predicted_speaker).

Inputs
------
  labeled_turns.csv  (from 07_align_manual_labels_to_whisperx.py)
  the 16 kHz mono WAVs

Outputs
-------
  enroll_validation.csv : per-turn predicted_speaker, sim_00, sim_01, margin,
                          manual speaker, agree (validate mode)
  prints per-session agreement + overall summary

Embedding backend
-----------------
Default: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb), the
standard 2-speaker-separable x-vector. Swappable (NeMo TitaNet, pyannote
embedding) via the `embed_segment` hook. Heavy deps are imported lazily so the
plumbing can be unit-tested without torch:

  pip install speechbrain torchaudio soundfile numpy pandas scikit-learn \
      --break-system-packages

Caveats baked into the design
-----------------------------
  * Single shared mic -> overlapping speech is the main error source; we skip
    very short turns (<MIN_EMBED_S) and low-coverage turns for enrollment.
  * Enrollment quality hinges on clean, unmixed speech in the enroll section.
  * 2 speakers in a quiet room are highly separable -> expect high agreement;
    investigate sessions that come back low.

Usage
-----
  python scripts/03_acoustic_alignment/08b_validate_speaker_enrollment.py
  python scripts/03_acoustic_alignment/08b_validate_speaker_enrollment.py --enroll-question 1 --limit 2
  python scripts/03_acoustic_alignment/08b_validate_speaker_enrollment.py --mode scale --turns-csv new_dyads_turns.csv
  python scripts/03_acoustic_alignment/08b_validate_speaker_enrollment.py --self-test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/

CONFIG = {
    "turns_csv": PROJECT / "04_data" / "labeled_turns.csv",
    "wav_dir": PROJECT / "01_pipeline" / "_wav",
    "out": PROJECT / "04_data" / "enroll_validation.csv",
    "enroll_question": 1,      # which Question block to enroll from
    "min_embed_s": 1.0,        # skip turns shorter than this (unreliable x-vector)
    "min_coverage": 0.6,       # skip low-alignment turns for enrollment
    "min_enroll_turns": 3,     # need >= this many enroll turns per speaker
    "low_margin": 0.05,        # flag turns whose cosine margin < this
    "mode": "validate",        # validate | scale
    "seed": 42,
}


# ─── embedding backend (lazy / swappable) ─────────────────────────────────────
_EMBED_MODEL = None


def _get_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _EMBED_MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
    return _EMBED_MODEL


def embed_segment(sound, t0, t1):
    """Return a unit-norm speaker embedding for [t0,t1] of a parselmouth Sound.

    Swap this function to use NeMo TitaNet or pyannote embeddings instead.
    """
    import torch
    seg = sound.extract_part(from_time=t0, to_time=t1, preserve_times=False)
    wav = np.asarray(seg.values).ravel().astype("float32")
    sr = int(seg.sampling_frequency)
    if sr != 16000:  # ECAPA expects 16 kHz
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    model = _get_model()
    with torch.no_grad():
        emb = model.encode_batch(torch.tensor(wav).unsqueeze(0)).squeeze().cpu().numpy()
    n = np.linalg.norm(emb)
    return emb / n if n else emb


# ─── core math (backend-agnostic; unit-testable) ──────────────────────────────
def cosine(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else np.nan


def build_centroids(enroll_embs):
    """enroll_embs: {speaker: [emb, ...]} -> {speaker: unit centroid}."""
    cents = {}
    for spk, embs in enroll_embs.items():
        if not embs:
            continue
        c = np.mean(np.vstack(embs), axis=0)
        n = np.linalg.norm(c)
        cents[spk] = c / n if n else c
    return cents


def assign(emb, centroids):
    """Return (pred_speaker, sims_dict, margin)."""
    sims = {spk: cosine(emb, c) for spk, c in centroids.items()}
    ordered = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    pred = ordered[0][0]
    margin = (ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else np.nan
    return pred, sims, margin


# ─── per-session driver ───────────────────────────────────────────────────────
def process_session(grp, sound, cfg, embed_fn):
    """grp: turns for one (dyad_id, condition). Returns list of per-turn dicts."""
    speakers = sorted(grp["speaker"].dropna().unique())
    if len(speakers) != 2:
        return [], f"expected 2 speakers, got {len(speakers)}"

    # --- enrollment pool: known-good section, long + high-coverage turns ---
    eq = cfg["enroll_question"]
    pool = grp[(grp.get("question") == eq)
               & (grp["dur"] >= cfg["min_embed_s"])
               & (grp.get("coverage", 1.0) >= cfg["min_coverage"])]
    enroll_idx = set(pool.index)
    enroll_embs = {s: [] for s in speakers}
    for _, t in pool.iterrows():
        enroll_embs[t["speaker"]].append(embed_fn(sound, t["t_start"], t["t_end"]))

    for s in speakers:
        if len(enroll_embs[s]) < cfg["min_enroll_turns"]:
            return [], (f"speaker {s} has {len(enroll_embs[s])} enroll turns "
                        f"in Q{eq} (<{cfg['min_enroll_turns']}); widen enroll section")
    centroids = build_centroids(enroll_embs)

    # --- score the remaining turns (hold enrollment turns out in validate) ---
    rows = []
    for idx, t in grp.iterrows():
        if t["dur"] < cfg["min_embed_s"]:
            continue
        is_enroll = idx in enroll_idx
        if cfg["mode"] == "validate" and is_enroll:
            continue  # held out to avoid circularity
        emb = embed_fn(sound, t["t_start"], t["t_end"])
        pred, sims, margin = assign(emb, centroids)
        rows.append({
            "dyad_id": t["dyad_id"], "condition": t["condition"],
            "turn_idx": t["turn_idx"], "t_start": t["t_start"], "t_end": t["t_end"],
            "dur": t["dur"], "manual_speaker": t["speaker"],
            "predicted_speaker": pred,
            "sim_" + speakers[0].split("_")[-1]: round(sims[speakers[0]], 4),
            "sim_" + speakers[1].split("_")[-1]: round(sims[speakers[1]], 4),
            "margin": round(margin, 4),
            "agree": (pred == t["speaker"]),
            "low_margin": (margin < cfg["low_margin"]),
        })
    return rows, None


def run(cfg, embed_fn=None):
    embed_fn = embed_fn or embed_segment
    df = pd.read_csv(cfg["turns_csv"])
    if "question" not in df.columns:
        print("[warn] no 'question' column; enrollment will use earliest turns "
              "as a fallback section.")
        df["question"] = (df.groupby(["dyad_id", "condition"]).cumcount()
                          < 12).map({True: cfg["enroll_question"], False: -1})

    import parselmouth  # noqa: lazy; skipped under --self-test (own embed_fn)
    sessions = list(df.groupby(["dyad_id", "condition"]))
    if cfg.get("limit"):
        sessions = sessions[: cfg["limit"]]

    all_rows, summary = [], []
    for (dyad_id, cond), grp in sessions:
        stem = f"{dyad_id}_{cond}_discussion"
        wav = _find_wav(cfg["wav_dir"], stem)
        if wav is None and embed_fn is embed_segment:
            print(f"[warn] no WAV for {stem}; skipped")
            continue
        sound = parselmouth.Sound(wav) if (wav and embed_fn is embed_segment) else None
        rows, err = process_session(grp, sound, cfg, embed_fn)
        if err:
            print(f"[skip] {stem}: {err}")
            continue
        all_rows.extend(rows)
        if rows:
            agree = np.mean([r["agree"] for r in rows])
            lowm = np.mean([r["low_margin"] for r in rows])
            summary.append((stem, len(rows), agree, lowm))
            print(f"  {stem}: n={len(rows):3d}  agreement={agree:5.1%}  "
                  f"low-margin={lowm:4.1%}")

    if not all_rows:
        raise SystemExit("No turns scored (check WAVs / enrollment thresholds).")
    out = pd.DataFrame(all_rows)
    Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg["out"], index=False)

    overall = out["agree"].mean()
    print(f"\nOverall agreement with manual labels: {overall:.1%} "
          f"over {len(out)} turns, {len(summary)} sessions.")
    flags = out[(~out["agree"]) | (out["low_margin"])]
    print(f"{len(flags)} turns flagged (disagree or low-margin) -> inspect these.")
    print(f"Wrote -> {cfg['out']}")
    if cfg["mode"] == "validate" and overall < 0.9:
        print("[note] <90% agreement: either manual labels need a second look, "
              "or enrollment audio is noisy/overlapping. Check flagged sessions.")
    return out


def _find_wav(wav_dir, stem):
    import glob
    base = stem.replace("_16k", "")
    for c in [f"{base}_16k.wav", f"{stem}.wav", f"{base}.wav"]:
        p = Path(wav_dir) / c
        if p.exists():
            return str(p)
    hits = glob.glob(str(Path(wav_dir) / f"{base}*.wav"))
    return hits[0] if hits else None


# ─── self-test: exercise plumbing without torch/audio ─────────────────────────
def _self_test(cfg):
    """Inject deterministic pseudo-embeddings so we can validate the
    enrollment -> cosine -> assignment -> agreement logic on real turn tables
    with no GPU, torch, or audio. Each speaker gets a distinct base vector +
    noise, so agreement should be high (sanity of the pipeline, not of ECAPA)."""
    rng = np.random.default_rng(cfg["seed"])
    base = {}  # (session, speaker) -> base vector

    def fake_embed(_sound, t0, t1, _ctx={}):
        return None  # replaced below via closure per session

    df = pd.read_csv(cfg["turns_csv"])
    if "question" not in df.columns:
        df["question"] = (df.groupby(["dyad_id", "condition"]).cumcount() < 12)\
            .map({True: cfg["enroll_question"], False: -1})

    sessions = list(df.groupby(["dyad_id", "condition"]))
    if cfg.get("limit"):
        sessions = sessions[: cfg["limit"]]

    all_rows = []
    for (dyad_id, cond), grp in sessions:
        spk = sorted(grp["speaker"].dropna().unique())
        vecs = {s: rng.normal(0, 1, 32) for s in spk}

        def emb_fn(_s, t0, t1, _spk=None, _grp=grp, _vecs=vecs):
            # look up which speaker this turn belongs to (cheat: by t_start)
            r = _grp[(abs(_grp.t_start - t0) < 1e-6)]
            s = r["speaker"].iloc[0] if len(r) else spk[0]
            return _vecs[s] + rng.normal(0, 0.25, 32)

        rows, err = process_session(grp, None, cfg, emb_fn)
        if err:
            print(f"[skip] {dyad_id}_{cond}: {err}")
            continue
        all_rows.extend(rows)
        if rows:
            print(f"  {dyad_id}_{cond}: n={len(rows)} agreement="
                  f"{np.mean([r['agree'] for r in rows]):.1%}")
    if all_rows:
        out = pd.DataFrame(all_rows)
        print(f"\n[self-test] {len(out)} turns, overall agreement "
              f"{out['agree'].mean():.1%} (expect high: distinct fake voices). "
              f"Plumbing OK.")
    else:
        print("[self-test] no rows produced.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--turns-csv", default=str(CONFIG["turns_csv"]))
    p.add_argument("--wav-dir", default=str(CONFIG["wav_dir"]))
    p.add_argument("--out", default=str(CONFIG["out"]))
    p.add_argument("--enroll-question", type=int, default=CONFIG["enroll_question"])
    p.add_argument("--min-embed-s", type=float, default=CONFIG["min_embed_s"])
    p.add_argument("--min-coverage", type=float, default=CONFIG["min_coverage"])
    p.add_argument("--min-enroll-turns", type=int, default=CONFIG["min_enroll_turns"])
    p.add_argument("--low-margin", type=float, default=CONFIG["low_margin"])
    p.add_argument("--mode", choices=["validate", "scale"], default=CONFIG["mode"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--self-test", action="store_true",
                   help="exercise logic with fake embeddings (no torch/audio)")
    a = p.parse_args()
    cfg = dict(CONFIG)
    cfg.update({"turns_csv": a.turns_csv, "wav_dir": a.wav_dir, "out": a.out,
                "enroll_question": a.enroll_question, "min_embed_s": a.min_embed_s,
                "min_coverage": a.min_coverage, "min_enroll_turns": a.min_enroll_turns,
                "low_margin": a.low_margin, "mode": a.mode, "limit": a.limit})
    return cfg, a.self_test


if __name__ == "__main__":
    cfg, self_test = parse_args()
    if self_test:
        _self_test(cfg)
    else:
        run(cfg)

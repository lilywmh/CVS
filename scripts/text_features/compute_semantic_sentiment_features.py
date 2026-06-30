import os
import re
import glob
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

"""
compute_semantic_sentiment_features.py
==============
Compute semantic and sentiment-alignment features from corrected dyadic
conversation transcripts.

For each dyad-condition session, the script:
  1. reads corrected transcript text files from 01_pipeline/all_srt,
  2. scores each turn with a sentiment model,
  3. computes per-question semantic similarity between the two speakers,
  4. computes per-question sentiment alignment and a supplementary cross-turn
     sentiment synchrony score,
  5. merges session-level features with the transcription log.

Output:
  04_data/scientific_dyad_analysis_results.csv
"""

# ─── CPU OPTIMIZATION ────────────────────────────────────────────────────────
torch.set_num_threads(min(8, os.cpu_count() or 1))

# ─── PATHS (resolved relative to this script; override via env vars) ──────────
PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/
DATA       = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
SRT_ROOT   = Path(os.environ.get("CVS_SRT_ROOT", PROJECT / "01_pipeline" / "all_srt"))
PIPER_DIR  = str(SRT_ROOT / "piper") + "/"
CLOUDY_DIR = str(SRT_ROOT / "cloudy") + "/"
LOG_PATH   = str(DATA / "Discussion Transcription Log - Sheet1.csv")
(DATA / "caches").mkdir(parents=True, exist_ok=True)
CACHE_PATH = str(DATA / "caches" / "processed_turns_cache.csv")
OUTPUT_CSV = str(DATA / "scientific_dyad_analysis_results.csv")

# ─── MODELS ──────────────────────────────────────────────────────────────────
SENTIMENT_MODEL  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SIMILARITY_MODEL = 'all-MiniLM-L6-v2'

# ─── INITIALIZATION ──────────────────────────────────────────────────────────
print("Initializing models...")
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL, use_safetensors=True)
sent_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
sem_model = SentenceTransformer(SIMILARITY_MODEL, device='cpu')

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def parse_transcript(filepath, condition):
    filename     = os.path.basename(filepath)
    id_match     = re.search(r"(dyad\d+_\d+)", filename)
    dyad_full_id = id_match.group(1) if id_match else "unknown"

    turns         = []
    curr_q        = 0
    speaker_regex = re.compile(r"^([A-Z0-9_\s]+):\s*(.*)", re.IGNORECASE)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "Question" in line:
                m = re.search(r"(\d+)", line)
                curr_q = int(m.group(1)) if m else curr_q
                continue
            match = speaker_regex.match(line)
            if match:
                turns.append({
                    "Dyad ID":   dyad_full_id,
                    "condition": condition,
                    "question":  curr_q,
                    "speaker":   match.group(1).strip(),
                    "text":      match.group(2).strip(),
                })
    return turns


def per_question_semantic_similarity(grp, spks):
    """
    Compute semantic similarity separately for each discussion question, then
    average across questions.

    To stay consistent with the Part 1 analysis, all turns from the same
    speaker within a question are concatenated before computing cosine
    similarity.
    """
    q_sims = []
    for _, q_grp in grp.groupby("question"):
        txt1 = " ".join(q_grp[q_grp["speaker"] == spks[0]]["text"]).strip()
        txt2 = " ".join(q_grp[q_grp["speaker"] == spks[1]]["text"]).strip()
        if not txt1 or not txt2:
            continue
        e1, e2 = sem_model.encode([txt1, txt2])
        q_sims.append(util.cos_sim(e1, e2).item())
    return float(np.mean(q_sims)) if q_sims else np.nan


def per_question_sentiment_alignment(grp, spks):
    """
    Compute per-question sentiment-valence alignment, consistent with Part 1:
      sent_align = 1 - |score_A - score_B| / 2

    The mean across questions is used as the dyad-level sentiment-alignment
    feature. The older sentiment_synchrony metric, a cross-turn correlation, is
    retained as a supplementary feature.
    """
    # Primary metric: per-question alignment, matched to the Part 1 analysis.
    q_aligns = []
    for _, q_grp in grp.groupby("question"):
        s1 = q_grp[q_grp["speaker"] == spks[0]]["sentiment_score"]
        s2 = q_grp[q_grp["speaker"] == spks[1]]["sentiment_score"]
        if s1.empty or s2.empty:
            continue
        # Average all sentiment scores within the question before alignment.
        m1, m2 = s1.mean(), s2.mean()
        q_aligns.append(1 - abs(m1 - m2) / 2)
    sent_alignment = float(np.mean(q_aligns)) if q_aligns else np.nan

    # Supplementary metric: the previous cross-turn sentiment correlation.
    s1_all = grp[grp["speaker"] == spks[0]]["sentiment_score"].reset_index(drop=True)
    s2_all = grp[grp["speaker"] == spks[1]]["sentiment_score"].reset_index(drop=True)
    min_l  = min(len(s1_all), len(s2_all))
    sync   = float(np.corrcoef(s1_all[:min_l], s2_all[:min_l])[0, 1]) if min_l > 3 else np.nan

    return sent_alignment, sync


# ─── EXECUTION ───────────────────────────────────────────────────────────────

def run():
    # 1. Load cache
    if os.path.exists(CACHE_PATH):
        cache_df = pd.read_csv(CACHE_PATH)
        id_col   = 'Dyad ID' if 'Dyad ID' in cache_df.columns else 'dyad_id'
        processed = cache_df[id_col].astype(str).unique().tolist() if not cache_df.empty else []
        cache_df  = cache_df.rename(columns={id_col: 'Dyad ID'})
    else:
        cache_df, processed = pd.DataFrame(), []

    # 2. Identify new files
    to_process = []
    for d, lbl in [(PIPER_DIR, 'piper'), (CLOUDY_DIR, 'cloudy')]:
        if not os.path.exists(d):
            continue
        for f in glob.glob(os.path.join(d, "*.txt")):
            id_m = re.search(r"(dyad\d+_\d+)", os.path.basename(f))
            if id_m and id_m.group(1) not in processed:
                to_process.append((f, lbl))

    # 3. Sentiment scoring (RoBERTa)
    if to_process:
        print(f"Processing {len(to_process)} new files...")
        new_data = []
        for f, lbl in to_process:
            new_data.extend(parse_transcript(f, lbl))
        new_df = pd.DataFrame(new_data)

        if not new_df.empty:
            lbl_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            scores  = []
            for i in tqdm(range(0, len(new_df), 8)):
                batch = new_df["text"].iloc[i:i+8].tolist()
                out   = sent_pipe(batch, truncation=True, max_length=512)
                scores.extend([lbl_map[r["label"]] * r["score"] for r in out])
            new_df["sentiment_score"] = scores
            full_df = pd.concat([cache_df, new_df], ignore_index=True)
            full_df.to_csv(CACHE_PATH, index=False)
        else:
            full_df = cache_df
    else:
        full_df = cache_df

    if full_df.empty:
        return

    # 4. Compute dyadic features
    print("Computing dyadic features (per-question)...")
    results = []
    for (dyad, cond), grp in tqdm(full_df.groupby(["Dyad ID", "condition"])):
        grp  = grp.reset_index(drop=True)
        spks = sorted(grp["speaker"].unique())
        if len(spks) < 2:
            continue

        # A. Mean per-question semantic similarity.
        sem_sim = per_question_semantic_similarity(grp, spks)

        # B. Mean per-question sentiment alignment + cross-turn sentiment synchrony.
        sent_align, sent_sync = per_question_sentiment_alignment(grp, spks)

        results.append({
            "Dyad ID":             dyad,
            "condition":           cond,
            # Primary metrics: per-question means aligned with Part 1.
            "semantic_similarity": sem_sim,
            "sent_alignment":      sent_align,
            # Supplementary metric retained from the earlier global calculation.
            "sentiment_synchrony": sent_sync,
            "mean_sentiment":      grp["sentiment_score"].mean(),
        })

    # 5. Merge with experimental log
    final_df = pd.DataFrame(results)
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH).dropna(subset=["Dyad ID"])
        final_df["Dyad ID"] = final_df["Dyad ID"].astype(str)
        log_df["Dyad ID"]   = log_df["Dyad ID"].astype(str)
        final_df = final_df.merge(log_df, on="Dyad ID", how="left")

    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Complete! Results saved to: {OUTPUT_CSV}")
    print(f"  Columns: {list(final_df.columns)}")


if __name__ == "__main__":
    run()

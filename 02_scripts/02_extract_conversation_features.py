import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

"""
02_extract_conversation_features.py
===================================
Extract structural conversation features from corrected dyadic transcripts.

For each dyad-condition session, the script:
  1. reads corrected transcript text files from 01_pipeline/all_srt,
  2. maps diarized speaker labels to experimental roles A/B using the
     transcription log,
  3. computes turn-taking, participation balance, lexical diversity, pronoun
     rates, question counts, and backchannel rates,
  4. writes one row per dyad-condition session.

Output:
  04_data/structural_dyad_analysis_mapped.csv
"""

# ─── CONFIGURATION & CACHE CLEANUP ───────────────────────────────────────────
# Set this to True to force a full re-processing of all files
CLEAN_CACHE = True

# ─── PATHS (resolved relative to this script; override via env vars) ──────────
PROJECT    = Path(__file__).resolve().parent.parent  # cvs_conversation/
DATA       = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
SRT_ROOT   = Path(os.environ.get("CVS_SRT_ROOT", PROJECT / "01_pipeline" / "all_srt"))
PIPER_DIR  = str(SRT_ROOT / "piper") + "/"
CLOUDY_DIR = str(SRT_ROOT / "cloudy") + "/"
LOG_PATH   = str(DATA / "Discussion Transcription Log - Sheet1.csv")
(DATA / "caches").mkdir(parents=True, exist_ok=True)
CACHE_PATH = str(DATA / "caches" / "structural_turns_cache.csv")
OUTPUT_CSV = str(DATA / "structural_dyad_analysis_mapped.csv")

# Cleanup logic: Deletes the cache if CLEAN_CACHE is True
if CLEAN_CACHE and os.path.exists(CACHE_PATH):
    try:
        os.remove(CACHE_PATH)
        print("!!! Cache cleaned. Performing a fresh run.")
    except OSError as e:
        print(f"[warn] could not remove cache ({e}); will overwrite instead.")

# ─── LINGUISTIC PARAMETERS ───────────────────────────────────────────────────
BACKCHANNELS = {"yeah", "yes", "right", "okay", "ok", "definitely", "totally", "sure", "mm", "mhm"}
I_PRONOUNS = {"i", "me", "my", "mine", "myself"}
WE_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}

# ─── CORE SCIENTIFIC FUNCTIONS ───────────────────────────────────────────────

def calculate_gini(x):
    """Measures participation inequality (0 = equal, 0.5+ = one person dominates)."""
    if len(x) < 2 or sum(x) == 0: return 0
    x = np.array(x, dtype=float)
    return np.abs(np.subtract.outer(x, x)).sum() / (2 * len(x) * x.sum())

def calculate_ttr(text_list):
    """Measures Lexical Diversity (Type-Token Ratio)."""
    words = " ".join(text_list).lower().split()
    return len(set(words)) / len(words) if words else 0

def parse_transcript(filepath, condition):
    filename = os.path.basename(filepath)
    id_match = re.search(r"(dyad\d+_\d+)", filename)
    dyad_id = id_match.group(1) if id_match else "unknown"
    
    turns = []
    curr_q = 0
    speaker_regex = re.compile(r"^([A-Z0-9_\s]+):\s*(.*)", re.IGNORECASE)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if "Question" in line:
                m = re.search(r"(\d+)", line)
                if m: curr_q = int(m.group(1))
                continue
            match = speaker_regex.match(line)
            if match:
                turns.append({
                    "Dyad ID": dyad_id, "condition": condition, 
                    "question": curr_q, "speaker": match.group(1).strip().upper(), 
                    "text": match.group(2).strip()
                })
    return turns

# ─── EXECUTION ───────────────────────────────────────────────────────────────

def run():
    # 1. Load Experimental Log for Role Mapping
    print("Loading Experimental Log...")
    log_df = pd.read_csv(LOG_PATH).dropna(subset=['Dyad ID'])
    log_df['Dyad ID'] = log_df['Dyad ID'].astype(str)

    # 2. Process Files (or Load Cache)
    if os.path.exists(CACHE_PATH):
        print("Loading existing turns from cache...")
        df = pd.read_csv(CACHE_PATH)
    else:
        print("Parsing transcript files...")
        all_data = []
        for d, lbl in [(PIPER_DIR, 'piper'), (CLOUDY_DIR, 'cloudy')]:
            if not os.path.exists(d): continue
            for f in glob.glob(os.path.join(d, "*.txt")):
                all_data.extend(parse_transcript(f, lbl))
        df = pd.DataFrame(all_data)
        df.to_csv(CACHE_PATH, index=False)

    df['word_count'] = df['text'].apply(lambda t: len(str(t).split()))

    # 3. Analyze Dyads with Role Mapping
    print("Extracting Structural Features...")
    results = []
    for (dyad, cond), grp in tqdm(df.groupby(['Dyad ID', 'condition'])):
        log_row = log_df[log_df['Dyad ID'] == dyad]
        if log_row.empty: continue

        # Determine Role Mapping (A or B) from Log
        col_00 = 'Piper_00' if cond == 'piper' else 'Cloudy_00'
        col_01 = 'Piper_01' if cond == 'piper' else 'Cloudy_01'
        
        role_map = {
            'SPEAKER_00': str(log_row[col_00].values[0]).strip().upper(),
            'SPEAKER_01': str(log_row[col_01].values[0]).strip().upper()
        }
        grp['role'] = grp['speaker'].map(role_map)

        # Split Data by Experimental Role
        data_a = grp[grp['role'] == 'A']
        data_b = grp[grp['role'] == 'B']
        if data_a.empty or data_b.empty: continue

        # Calculate Word Counts and Pronoun Rates
        words_a = " ".join(data_a['text']).lower().split()
        words_b = " ".join(data_b['text']).lower().split()
        
        results.append({
            'Dyad ID': dyad, 
            'condition': cond,
            'Order': log_row['Order'].values[0],
            'total_turns': len(grp),
            'participation_gini': calculate_gini([len(words_a), len(words_b)]),
            'turn_taking_density': sum(1 for i in range(1, len(grp)) if grp.iloc[i]['speaker'] != grp.iloc[i-1]['speaker']) / len(grp),
            # Role A Features
            'ttr_A': calculate_ttr(data_a['text']),
            'i_rate_A': sum(1 for w in words_a if w in I_PRONOUNS) / max(1, len(words_a)),
            'we_rate_A': sum(1 for w in words_a if w in WE_PRONOUNS) / max(1, len(words_a)),
            'q_count_A': sum(1 for t in data_a['text'] if "?" in str(t)),
            'bc_rate_A': sum(1 for t in data_a['text'] if str(t).lower().strip(".,!?") in BACKCHANNELS) / len(data_a),
            # Role B Features
            'ttr_B': calculate_ttr(data_b['text']),
            'i_rate_B': sum(1 for w in words_b if w in I_PRONOUNS) / max(1, len(words_b)),
            'we_rate_B': sum(1 for w in words_b if w in WE_PRONOUNS) / max(1, len(words_b)),
            'q_count_B': sum(1 for t in data_b['text'] if "?" in str(t)),
            'bc_rate_B': sum(1 for t in data_b['text'] if str(t).lower().strip(".,!?") in BACKCHANNELS) / len(data_b),
        })

    # 4. Final Save
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Scientific Structural Analysis Complete! Results: {OUTPUT_CSV}")

if __name__ == "__main__":
    run()

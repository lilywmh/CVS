# Recording Processing Pipeline

Record of processing steps for CVS recording → transcription workflow.

---

## Overview

| Step | Description        | Input              | Output                    | Script / Tool   |
|------|--------------------|--------------------|---------------------------|-----------------|
| 1a   | Semantic/sentiment features | corrected transcripts | `04_data/scientific_dyad_analysis_results.csv` | `02_scripts/01_semantic.py` |
| 1b   | Structural conversation features | corrected transcripts + transcription log | `04_data/structural_dyad_analysis_mapped.csv` | `02_scripts/02_extract_conversation_features.py` |
| 1c   | LLM annotation features | corrected transcripts + API access | `05_analysis_outputs/llm_annotation_output/` | `02_scripts/test_annotation.py` |
| 1d   | Poster multivariate model | semantic + structural + LLM + outcomes | `05_analysis_outputs/multivariate_output/` | `02_scripts/poster_analysis_pipeline.py` |
| 1    | Convert to WAV     | Recording (any)    | `_wav/<stem>_16k.wav`     | ffmpeg          |
| 2    | Transcribe + diarize | 16k mono WAV     | `outputs/<stem>/` (srt, json, …) | WhisperX (`run_whisperx.sh`) |
| 3a   | Transfer manual labels → timestamps | manual `.txt` + WhisperX `.srt` | `04_data/labeled_turns.csv` | `02_scripts/03a_transfer_labels.py` |
| 3    | Per-turn prosody   | `labeled_turns.csv` + WAV | `04_data/acoustic_turns.csv` | `02_scripts/03_acoustic_features.py` |
| 4    | Vocal alignment (entrainment) | `acoustic_turns.csv` | `04_data/vocal_alignment_dyad.csv` | `02_scripts/04_vocal_alignment.py` |
| 5    | Dissociation test  | turn-level + vocal CSVs | `05_analysis_outputs/dissociation_results.{csv,json}` | `02_scripts/05_dissociation_test.py` |
| 6    | Publication/poster figures | analysis CSVs | `06_figures/` and plot PDFs/PNGs | `02_scripts/plot_*.py`, `02_scripts/poster_plots.py` |
| 3b   | *(optional)* Diarization validation/scaling | `labeled_turns.csv` + WAV | `04_data/enroll_validation.csv` | `02_scripts/03b_enroll_diarize.py` |

---

## Step 1: Convert to stable WAV

- **Purpose:** Normalize input to 16 kHz mono for WhisperX.
- **Tool:** `ffmpeg`
- **Command (from script):**
  ```bash
  ffmpeg -hide_banner -y -i "<input>" -ac 1 -ar 16000 -c:a pcm_s16le "<stem>_16k.wav"
  ```
- **Output dir:** `_wav/`
- **Skip if:** `_wav/<stem>_16k.wav` already exists.

---

## Step 2: WhisperX (transcription + diarization)

- **Purpose:** ASR + speaker diarization, word-level timing.
- **Tool:** WhisperX (via `run_whisperx.sh`)
- **Config (defaults in script):**
  - Model: `small`
  - Device: `cpu`
  - Compute type: `int8`
  - Batch size: `2`
  - Language: `en`
  - Speakers: `min_speakers=2`, `max_speakers=2`
  - Output formats: SRT, plus JSON/TSV/VTT/TXT in output dir
- **Output dir:** `outputs/<stem>/`
- **Logs:** `logs/<stem>.log`
- **Requires:** `HF_TOKEN` (Hugging Face) set in environment.

---

## How to run

From `recording/`:

```bash
export HF_TOKEN='hf_...'
./run_whisperx.sh file1.m4a file2.m4a ...
```

Parallel jobs: set `JOBS` (default 6), e.g. `JOBS=4 ./run_whisperx.sh ...`

---

## Vocal-alignment pipeline (Steps 3a → 3 → 4 → 5)

This branch tests the core proposal question: does **vocal alignment**
(sub-second prosodic coordination) capture a distinct dimension of coordination
that predicts social connection **beyond turn-level dynamics**?

### Design constraint (read first)

The discussion recordings are a **single shared microphone** — the `.mp4`s are
stereo in name only (L − R = −∞ dB, i.e. duplicated mono). There is therefore
**no per-speaker audio stream** and two voices are never separable at the same
instant. Consequences baked into the scripts:

- Speaker identity comes from the **manual `.txt` labels** (corrected by hand),
  never from pyannote. WhisperX is used only for word **timestamps**.
- Vocal alignment is measured as **turn-adjacent entrainment** (Levitan &
  Hirschberg style: speaker A's turn-offset prosody → speaker B's turn-onset
  prosody at speaker switches), **not** continuous simultaneous synchrony.

### Dependencies

```bash
# Steps 3a / 4 / 5 (light)
pip install numpy pandas scipy scikit-learn tqdm --break-system-packages
# Step 3 (prosody)
pip install praat-parselmouth librosa soundfile --break-system-packages
# Step 3b only (optional, speaker embeddings)
pip install speechbrain torchaudio --break-system-packages
```

### Step 3a — Transfer manual labels onto WhisperX timestamps

- **Why:** manual `.txt` has correct speakers/turns but **no timestamps**;
  WhisperX `.srt` has accurate word timestamps but **wrong** pyannote labels.
  This aligns the two word sequences and copies each word's time onto your
  manual turn, keeping your label. Non-destructive: `.txt` files are read-only.
- **Note:** uses only the `<u>highlighted</u>` SRT cues (the intermediate
  no-highlight cues repeat the whole sentence in ~20 ms and would cluster
  words). Emits a per-turn `coverage` QC column (fraction of words timestamped).
- **Run:**
  ```bash
  cd 02_scripts
  python 03a_transfer_labels.py            # all sessions
  python 03a_transfer_labels.py --limit 1  # smoke-test one session
  ```

### Step 3 — Per-turn prosodic features

- **Extracts:** f0 (mean/sd/range), intensity (mean/sd), speaking rate
  (syllable proxy + words/sec), plus **turn-edge windows** (onset/offset f0 &
  intensity) used for entrainment. f0/intensity are z-scored within speaker
  downstream so we capture contour coordination, not baseline pitch.
- **Run (transfer mode = recommended, uses your manual labels):**
  ```bash
  python 03_acoustic_features.py --turns-csv ../04_data/labeled_turns.csv \
                                 --out ../04_data/acoustic_turns.csv
  python 03_acoustic_features.py --dry-run --limit 1   # parse turns, skip audio
  ```

### Step 4 — Vocal alignment (entrainment metrics)

- **Computes**, per (pair_id, condition), for f0 / intensity / rate:
  **proximity** (closer than a shuffled-partner baseline), **convergence**
  (growing more similar over the session), **synchrony** (turn-by-turn
  co-variation), plus edge-based (sub-second) variants.
- **Condition is kept, not averaged** (piper/cloudy stay separate rows).
- **Run:**
  ```bash
  python 04_vocal_alignment.py   # in: acoustic_turns.csv  out: vocal_alignment_dyad.csv
  ```

### Step 5 — Dissociation test (the hypothesis)

- Reduces each construct to **one pre-specified composite** (PC1 of the
  turn-level block; PC1 of the vocal block) *before* touching any outcome — no
  outcome-driven feature selection (avoids the small-N leakage flagged in the
  code review). Per outcome it reports: distinctness correlation, hierarchical
  R² → **ΔR²** for adding vocal alignment, nested-F **and** permutation p,
  partial correlations (Pearson + Spearman), a commonality split
  (unique-turn / unique-vocal / shared), and **BH-FDR** across outcomes.
- **Condition handling:** outcomes are dyad-level, so the vocal block is
  averaged to `pair_id` for the primary test (switch `--unit session` if/when
  per-condition outcomes exist).
- **Run:**
  ```bash
  python 05_dissociation_test.py --n-perm 5000 --seed 42
  ```

### Step 3b — *(optional)* Diarization validation / scaling

- **Speaker enrollment:** build a voiceprint per speaker from a known-good
  section (default Question 1) using ECAPA-TDNN x-vectors, then label the rest
  by cosine similarity. `--mode validate` (default) cross-checks your manual
  labels and flags disagreements; `--mode scale` auto-labels *new* dyads.
- Not a replacement for manual labels — a QC/scaling arm.
- **Run:**
  ```bash
  python 03b_enroll_diarize.py                 # validate, enroll on Q1
  python 03b_enroll_diarize.py --self-test     # plumbing only (no torch/audio)
  ```

### Full run order

```bash
cd 02_scripts
python 03a_transfer_labels.py
python 03_acoustic_features.py --turns-csv ../04_data/labeled_turns.csv \
                               --out ../04_data/acoustic_turns.csv
python 04_vocal_alignment.py
python 05_dissociation_test.py
# optional QC:
python 03b_enroll_diarize.py
```

---

## Semantic, Structural, LLM, and Poster Analysis Pipeline

These scripts form the main text-based replication path. They can be run from
the repository root after the corrected transcript text files and required CSVs
are in place.

```bash
python 02_scripts/01_semantic.py
python 02_scripts/02_extract_conversation_features.py

# Requires ANTHROPIC_API_KEY. Use qwen_annotation.py instead with
# OPENROUTER_API_KEY if you want the Qwen/OpenRouter comparison workflow.
python 02_scripts/test_annotation.py

python 02_scripts/llm_regression.py
python 02_scripts/poster_analysis_pipeline.py
python 02_scripts/poster_plots.py
```

Poster and figure helper scripts should be treated as downstream visualization
steps. They do not define the primary statistical test unless explicitly stated
in their headers.

```bash
python 02_scripts/plot_audio_alignment.py
python 02_scripts/plot_highlow_compare.py
python 02_scripts/plot_vocal_outcome_heatmap.py
```

---

## Directory layout

```
cvs_conversation/
├── PIPELINE.md          # this file
├── 01_pipeline/
│   ├── run_whisperx.sh  # transcription/diarization script
│   ├── _wav/            # 16 kHz mono WAVs (intermediate)
│   ├── outputs/         # per-file WhisperX outputs (srt, json, tsv, vtt, txt)
│   ├── all_srt/         # manual .txt (corrected labels) + .srt per condition
│   └── logs/            # per-file run logs
├── 02_scripts/          # 01_semantic … 05_dissociation_test.py
├── 04_data/             # *.csv inputs/outputs (labeled_turns, acoustic_turns, …)
└── 05_analysis_outputs/ # dissociation_results.{csv,json}, etc.
```

Input recordings live in: `REC_DIR` (script default: `/Users/Meihui/Downloads/sync/CVS/recording`). Pass filenames (relative to that dir) to `run_whisperx.sh`.

# Recording Processing Pipeline

Record of processing steps for CVS recording → transcription workflow.

---

## Overview

| Step | Description        | Input              | Output                    | Script / Tool   |
|------|--------------------|--------------------|---------------------------|-----------------|
| 1    | Convert to WAV     | Recording (any)    | `_wav/<stem>_16k.wav`     | ffmpeg          |
| 2    | Transcribe + diarize | 16k mono WAV     | `outputs/<stem>/` (srt, json, …) | WhisperX (`run_whisperx.sh`) |

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

## Optional / future steps

*(Add new steps here as you extend the pipeline.)*

| Step | Description        | Input   | Output   | Notes |
|------|--------------------|---------|----------|--------|
| …    | …                  | …       | …        | …     |

---

## Directory layout

```
recording/
├── PIPELINE.md          # this file
├── run_whisperx.sh      # main processing script
├── _wav/                # 16 kHz mono WAVs (intermediate)
├── outputs/             # per-file WhisperX outputs (srt, json, tsv, vtt, txt)
└── logs/                # per-file run logs
```

Input recordings live in: `REC_DIR` (script default: `/Users/Meihui/Downloads/sync/CVS/recording`). Pass filenames (relative to that dir) to `run_whisperx.sh`.

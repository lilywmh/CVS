# Data Layout

This folder is the intended location for data files in a clean replication
checkout. Most data are ignored by Git because they may be large, generated, or
restricted.

## Subfolders

| Folder | Purpose | Commit to Git? |
| --- | --- | --- |
| `raw/` | Raw recordings, raw exports, original participant files | No |
| `interim/` | Intermediate files such as converted audio, WhisperX output, caches | No |
| `derived/` | De-identified derived CSVs used for analysis replication | Optional |
| `private/` | Restricted master sheets or participant-level data | No |

## Key Files Used By Scripts

| File | Current legacy path | Used by | Share status |
| --- | --- | --- | --- |
| Corrected transcripts | `01_pipeline/all_srt/{piper,cloudy}/*.txt` | text, LLM, label-transfer scripts | Restricted unless de-identified |
| WhisperX SRTs | `01_pipeline/outputs/<session>/*.srt` | `07_align_manual_labels_to_whisperx.py` | Restricted/intermediate |
| 16 kHz WAV files | `01_pipeline/_wav/*_16k.wav` | acoustic scripts | Restricted |
| Transcription log | `04_data/Discussion Transcription Log - Sheet1.csv` | role mapping | Check for identifiers before sharing |
| Outcomes | `04_data/outcomes.csv` | modeling scripts | Share if de-identified |
| Semantic features | `04_data/scientific_dyad_analysis_results.csv` | model/plot scripts | Share if de-identified |
| Structural features | `04_data/structural_dyad_analysis_mapped.csv` | model/plot scripts | Share if de-identified |
| LLM features | `05_analysis_outputs/llm_annotation_output/dyad_features.csv` | LLM/model scripts | Share if text-free and de-identified |
| Acoustic turns | `04_data/acoustic_turns.csv` | vocal-alignment scripts | Share if de-identified |
| Vocal alignment | `04_data/vocal_alignment_dyad.csv` | dissociation/plot scripts | Share if de-identified |

For public replication, prefer sharing de-identified derived tables rather than
raw recordings or full transcripts.

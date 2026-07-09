# Data Layout

This folder is the intended location for data files in a clean replication
checkout. Most data are ignored by Git because they may be large, generated, or
restricted.

The public repository does not need the complete private folder tree. Scripts
use repository-relative local defaults for the original development checkout,
but private replications can use the cleaner paths below with environment
variables such as `CVS_DATA`, `CVS_SRT_ROOT`, `CVS_WHISPERX_OUTPUTS`, and
`CVS_WAV_DIR`. Figure scripts also respect `CVS_FIGURES`.

## Subfolders

| Folder | Purpose | Commit to Git? |
| --- | --- | --- |
| `raw/` | Raw recordings, raw exports, original participant files | No |
| `interim/` | Intermediate files such as converted audio, WhisperX output, caches | No |
| `derived/` | De-identified derived CSVs used for analysis replication | No by default; force-add only if approved for public sharing |
| `private/` | Restricted master sheets or participant-level data | No |

## Key Files Used By Scripts

| File | Current legacy path | Used by | Share status |
| --- | --- | --- | --- |
| Corrected transcripts | `01_pipeline/all_srt/{piper,cloudy}/*.txt` | text, LLM, label-transfer scripts | Restricted unless de-identified |
| WhisperX SRTs | `01_pipeline/outputs/<session>/*.srt` | `align_manual_labels_to_whisperx.py` | Restricted/intermediate |
| 16 kHz WAV files | `01_pipeline/_wav/*_16k.wav` | acoustic scripts | Restricted |
| Discussion videos | `01_pipeline/video/*.{mp4,mov,m4v,avi,mkv}` | video-behavior scripts | Restricted |
| Video behavior run directories | `data/interim/video_behavior/runs/<run_id>/` | `run_video_analysis.py` | Generated; do not commit |
| Turn-aligned video behavior run directories | `data/interim/video_behavior/turn_runs/<run_id>/` | `run_turn_aligned_video_analysis.py` | Generated; do not commit |
| Transcription log | `04_data/Discussion Transcription Log - Sheet1.csv` | role mapping | Check for identifiers before sharing |
| Outcomes | `04_data/outcomes.csv` | modeling scripts | Share if de-identified |
| Semantic features | `04_data/scientific_dyad_analysis_results.csv` | model/plot scripts | Share if de-identified |
| Structural features | `04_data/structural_dyad_analysis_mapped.csv` | model/plot scripts | Share if de-identified |
| LLM features | `05_analysis_outputs/llm_annotation_output/dyad_features.csv` | LLM/model scripts | Share if text-free and de-identified |
| Acoustic turns | `04_data/acoustic_turns.csv` | vocal-alignment scripts | Share if de-identified |
| Vocal alignment | `04_data/vocal_alignment_dyad.csv` | dissociation/plot scripts | Share if de-identified |
| Primary floor-transfer vocal alignment | `04_data/vocal_alignment_floor_transfer_dyad.csv` | vocal-alignment/model scripts | Share if de-identified |
| Floor-transfer excluded turns | `04_data/vocal_alignment_floor_transfer_excluded_turns.csv` | vocal-alignment QC | Check text snippets before sharing |
| Video behavior | `04_data/video_behavior_dyad.csv` | video-behavior/model scripts | Share if de-identified and face images are not included |
| Video behavior final outputs | `data/interim/video_behavior/runs/<run_id>/final/*.csv` | video-behavior/model scripts | Share only after de-identification review |
| Turn-aligned video behavior outputs | `data/interim/video_behavior/turn_runs/<run_id>/results/*.csv` | video-behavior/model scripts | Share only after de-identification review |

For public replication, prefer sharing de-identified derived tables rather than
raw recordings or full transcripts.

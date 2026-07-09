# CVS Conversation Analysis

Reproducible code for a dyadic co-viewing conversation study. The repository
extracts semantic, structural, LLM-coded, and vocal-alignment features from
paired discussion transcripts, then tests their association with social
connection outcomes.

Private recordings, transcripts, API outputs, and participant-level data are
not included in this repository.

## Repository Structure

```text
scripts/
  text_features/          transcript-level semantic and structural features
  llm_annotation/         LLM-assisted turn and conversation annotation
  acoustic_alignment/     timestamp alignment, acoustic features, vocal alignment
  video_behavior/         movement and facial-expression proxies from private videos
  models/                 statistical analyses and optional covariates
  figures/                publication and exploratory figures
data/                     data dictionary and expected input layout
config/                   example local path configuration
PIPELINE.md               reproducible run order
environment.yml           Conda environment specification
setup_conda_env.sh        environment setup helper
```

Generated or restricted local folders such as `01_pipeline/`, `04_data/`,
`05_analysis_outputs/`, and `06_figures/` are ignored by Git.

## Setup

```bash
bash setup_conda_env.sh
conda activate cvs-conversation
```

API keys, if needed, should be supplied through environment variables.

```bash
export ANTHROPIC_API_KEY="..."
export OPENROUTER_API_KEY="..."
export HF_TOKEN="..."
```

## Data Policy

The public repository contains methods code only. To run the analyses locally,
place private inputs in the documented local paths:

- corrected transcripts and WhisperX files under `01_pipeline/all_srt/`
- raw/intermediate audio under `01_pipeline/_wav/` or the configured recording path
- derived analysis tables under `04_data/`
- participant master sheet at `04_data/MASTER_SHEET_ONE_ROW_PER_PARTICIPANT.csv`

These are local defaults, not public repository contents. If you keep private
data in a cleaner layout, set path variables before running scripts:

```bash
export CVS_SRT_ROOT=data/raw/all_srt
export CVS_WHISPERX_OUTPUTS=data/interim/whisperx_outputs
export CVS_WAV_DIR=data/interim/wav_16k
export CVS_VIDEO_DIR=data/raw/videos
export CVS_DATA=data/derived
export CVS_ANALYSIS_OUTPUTS=outputs
export CVS_FIGURES=figures
```

See `data/README.md` for the expected files and privacy notes.

## Reproduction

Run scripts from the repository root. Use `PIPELINE.md` as the canonical run
order; filenames are descriptive rather than numbered.

Core text/annotation/model path:

```bash
python scripts/text_features/compute_semantic_sentiment_features.py
python scripts/text_features/compute_structural_conversation_features.py
python scripts/llm_annotation/annotate_turns_claude.py
python scripts/models/analyze_multivariate_connection_models.py
python scripts/figures/plot_dyad_feature_outcome_associations.py
```

Vocal-alignment path:

```bash
python scripts/acoustic_alignment/align_manual_labels_to_whisperx.py
python scripts/acoustic_alignment/extract_acoustic_features.py \
    --turns-csv "${CVS_DATA:-04_data}/labeled_turns.csv" \
    --out "${CVS_DATA:-04_data}/acoustic_turns.csv"
python scripts/acoustic_alignment/compute_vocal_alignment_floor_transfer.py
python scripts/models/test_vocal_alignment_incremental_validity.py \
    --vocal "${CVS_DATA:-04_data}/vocal_alignment_floor_transfer_dyad.csv" \
    --out-csv "${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/floor_transfer_dissociation_results.csv" \
    --out-json "${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/floor_transfer_dissociation_results.json"
python scripts/figures/plot_vocal_handoff_alignment.py
python scripts/figures/compare_high_low_connection_handoffs.py
python scripts/figures/plot_vocal_alignment_outcome_correlations.py
```

Turn-aligned video-behavior path:

```bash
python scripts/video_behavior/run_turn_aligned_video_analysis.py \
    --config config/video_turn_analysis.example.yml
```

This project-specific runner treats
`scripts/acoustic_alignment/align_manual_labels_to_whisperx.py` output
(`${CVS_DATA:-04_data}/labeled_turns.csv`) and
`04_data/Discussion Transcription Log - Sheet1.csv` as authoritative. By
default it generates video metadata from `01_pipeline/recording_videos`,
preferring `*_discussion.*` files when multiple camera views exist. It writes
timestamped run directories under
`${CVS_VIDEO_OUTPUTS:-data/interim/video_behavior/turn_runs}` with turn-level,
question-level, speaker-level, session-level, cross-session, and upper-limb
speaker/listener outputs. The upper-limb tables use shoulder-elbow-wrist Pose
landmarks to summarize whole-arm movement and posture proxies.
Use `scripts/video_behavior/summarize_video_behavior_qc.py` after a completed
run to generate QC/descriptive tables and figures under
`${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/video_behavior_qc_summary/`.
Then run `scripts/video_behavior/check_roi_role_mapping.py` to generate sampled
left/right ROI overlays under
`${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/video_roi_role_qc/` before
interpreting hand or gesture features.
For frame-by-frame arm visual checks, run:

```bash
python scripts/video_behavior/render_arm_skeleton_qc.py \
    --source-run data/interim/video_behavior/turn_runs/<run_id> \
    --write-video
```

To analyze arm behavior during speaking, listening, silence, and overlapping
speech, run the fixed-window role analysis:

```bash
python scripts/video_behavior/run_time_window_arm_role_analysis.py \
    --source-run data/interim/video_behavior/turn_runs/<run_id>
```

For the primary movement analysis, use question-level left/right windows rather
than short speaker turns:

```bash
python scripts/video_behavior/run_question_side_video_analysis.py \
    --source-run data/interim/video_behavior/turn_runs/<run_id>
```

This writes speaker-independent side-level outputs under
`${CVS_VIDEO_OUTPUTS:-data/interim/video_behavior/side_question_runs}`.
The side-level outputs include rough head-orientation rates for looking toward
the partner, facing the screen/front, and looking away; these are QC-dependent
orientation proxies rather than calibrated eye-gaze labels.

## Outputs

Local runs write derived tables and figures to ignored folders, primarily
`04_data/`, `05_analysis_outputs/`, and `06_figures/`. These outputs can be
regenerated from the private inputs and scripts.

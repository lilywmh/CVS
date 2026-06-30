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
python scripts/acoustic_alignment/compute_vocal_alignment.py
python scripts/models/test_vocal_alignment_incremental_validity.py
python scripts/figures/plot_vocal_handoff_alignment.py
python scripts/figures/compare_high_low_connection_handoffs.py
python scripts/figures/plot_vocal_alignment_outcome_correlations.py
```

## Outputs

Local runs write derived tables and figures to ignored folders, primarily
`04_data/`, `05_analysis_outputs/`, and `06_figures/`. These outputs can be
regenerated from the private inputs and scripts.

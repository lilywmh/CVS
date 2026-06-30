# CVS Conversation Analysis

Reproducible code for a dyadic co-viewing conversation study. The pipeline
extracts semantic, structural, LLM-coded, and vocal-alignment features from
paired discussion transcripts, then tests their association with social
connection outcomes.

Private recordings, transcripts, API outputs, and participant-level data are
not included in this repository.

## Repository Structure

```text
scripts/              canonical numbered analysis scripts
data/                 data dictionary and expected input layout
config/               example local path configuration
PIPELINE.md           concise end-to-end run order
environment.yml       Conda environment specification
setup_conda_env.sh    environment setup helper
```

Generated or restricted local folders such as `01_pipeline/`, `04_data/`,
`05_analysis_outputs/`, and `06_figures/` are ignored by Git.

## Setup

```bash
bash setup_conda_env.sh
conda activate cvs-conversation
```

The setup script creates or updates the Conda environment from
`environment.yml`. API keys, if needed, should be supplied through environment
variables.

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

See `data/README.md` for the expected files and privacy notes.

## Reproduction

Run scripts from the repository root. The primary text, annotation, modeling,
acoustic-alignment, and figure steps are listed in `PIPELINE.md`; individual
script purposes are indexed in `scripts/README.md`.

Typical analysis entry points:

```bash
python scripts/01_text_features/01_compute_semantic_sentiment_features.py
python scripts/01_text_features/02_compute_structural_conversation_features.py
python scripts/02_llm_annotation/03_annotate_turns_claude.py
python scripts/04_models/05_poster_multivariate_analysis.py
python scripts/05_figures/06_make_poster_plots.py
```

Vocal-alignment analyses require timestamped turns and audio:

```bash
python scripts/03_acoustic_alignment/07_align_manual_labels_to_whisperx.py
python scripts/03_acoustic_alignment/08_extract_acoustic_features.py \
    --turns-csv 04_data/labeled_turns.csv \
    --out 04_data/acoustic_turns.csv
python scripts/03_acoustic_alignment/09_compute_vocal_alignment.py
python scripts/04_models/10_test_vocal_alignment_incremental_validity.py
python scripts/05_figures/12_plot_audio_alignment.py
python scripts/05_figures/13_plot_highlow_compare.py
python scripts/05_figures/14_plot_vocal_outcome_heatmap.py
```

## Outputs

Local runs write derived tables and figures to ignored folders, primarily
`04_data/`, `05_analysis_outputs/`, and `06_figures/`. These outputs can be
regenerated from the private inputs and scripts.

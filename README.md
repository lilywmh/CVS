# CVS Conversation Analysis

This repository contains the reproducible analysis code for a dyadic co-viewing
conversation study. The pipeline extracts semantic, structural, LLM-coded, and
vocal-alignment features from paired discussion transcripts and tests their
association with social connection outcomes.

The repository is organized as a research compendium: scripts are numbered in
the order they should be run, generated outputs are excluded from version
control, and environment setup is captured in `environment.yml`.

## Repository Layout

```text
01_pipeline/              # raw transcription pipeline outputs; ignored by Git
02_scripts/               # canonical replication scripts
03_notebooks/             # exploratory notebooks; not the primary pipeline
04_data/                  # small input/derived CSVs used by scripts
05_analysis_outputs/      # generated tables and model outputs; ignored by Git
06_figures/               # generated figures; ignored by Git
PIPELINE.md               # step-by-step processing and analysis pipeline
environment.yml           # Conda environment specification
setup_conda_env.sh        # helper to create/update the Conda environment
```

## Quick Start

```bash
bash setup_conda_env.sh
conda activate cvs-conversation
```

Then follow the script order in `02_scripts/README.md` or the detailed workflow
in `PIPELINE.md`.

## Data Requirements

Large raw/intermediate materials are intentionally not tracked:

- raw audio/video recordings
- WhisperX outputs
- per-file transcript folders under `01_pipeline/all_srt`
- generated model outputs and figures

To replicate from raw data, place inputs in the paths documented in
`PIPELINE.md`. To replicate the statistical analyses from derived tables, place
the required CSVs in `04_data/`.

## Reproducibility Notes

- Use the numbered scripts in `02_scripts/` as the canonical pipeline.
- Treat notebooks as exploratory or diagnostic unless explicitly referenced.
- Analysis scripts resolve paths relative to the repository root, so they should
  run from any working directory.
- LLM annotation scripts require API keys in environment variables, not in code.

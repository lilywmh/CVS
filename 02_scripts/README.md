# Script Index

This folder contains the canonical scripts for reproducing the CVS conversation
analysis. Scripts are numbered by pipeline order when they are part of the main
replication path. Unnumbered files are supplementary analyses or plotting
helpers.

## Main Replication Pipeline

| Step | Script | Purpose | Main output |
| --- | --- | --- | --- |
| 01 | `01_semantic.py` | Semantic similarity and sentiment alignment from corrected transcripts | `04_data/scientific_dyad_analysis_results.csv` |
| 02 | `02_extract_conversation_features.py` | Structural conversation features, role mapping, turn-taking, pronouns, lexical diversity | `04_data/structural_dyad_analysis_mapped.csv` |
| 03a | `03a_transfer_labels.py` | Transfer corrected speaker labels onto WhisperX word timestamps | `04_data/labeled_turns.csv` |
| 03 | `03_acoustic_features.py` | Per-turn prosodic features from timestamped turns and WAV files | `04_data/acoustic_turns.csv` |
| 04 | `04_vocal_alignment.py` | Turn-adjacent vocal entrainment metrics | `04_data/vocal_alignment_dyad.csv` |
| 05 | `05_dissociation_test.py` | Pre-specified incremental/discriminant validity tests for vocal alignment | `05_analysis_outputs/dissociation_results.csv` and `.json` |
| 06 | `06_build_covariates.py` | Optional dyad-level covariates from participant master sheet | `04_data/covariates_dyad.csv` |

## LLM Annotation Pipeline

| Script | Purpose |
| --- | --- |
| `test_annotation.py` | Claude/Anthropic turn-level and conversation-level annotations |
| `qwen_annotation.py` | Qwen/OpenRouter version of the same annotation workflow |
| `llm_regression.py` | Regression screen for LLM-derived dyad features |

Set API keys via environment variables:

```bash
export ANTHROPIC_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## Poster and Plotting Scripts

| Script | Purpose |
| --- | --- |
| `poster_analysis_pipeline.py` | Full poster-oriented multivariate model, diagnostics, tables, and figures |
| `poster_plots.py` | Compact poster plots from semantic/structural features |
| `plot_audio_alignment.py` | Vocal-alignment illustration plots |
| `plot_highlow_compare.py` | High- vs low-connection exemplar comparison plots |
| `plot_vocal_outcome_heatmap.py` | Exploratory heatmap of vocal metrics vs outcomes |

Plotting scripts are downstream of the main analysis outputs. They should be
run after the relevant tables have been generated.

## Suggested Run Order

```bash
python 02_scripts/01_semantic.py
python 02_scripts/02_extract_conversation_features.py
python 02_scripts/test_annotation.py
python 02_scripts/llm_regression.py
python 02_scripts/poster_analysis_pipeline.py

python 02_scripts/03a_transfer_labels.py
python 02_scripts/03_acoustic_features.py
python 02_scripts/04_vocal_alignment.py
python 02_scripts/05_dissociation_test.py

python 02_scripts/poster_plots.py
python 02_scripts/plot_audio_alignment.py
python 02_scripts/plot_highlow_compare.py
python 02_scripts/plot_vocal_outcome_heatmap.py
```

Some scripts require raw audio or API access. See `PIPELINE.md` for input
placement, expected file naming, and methodological notes.

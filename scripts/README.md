# Script Index

This folder contains the canonical replication scripts, grouped by analysis
stage. Run commands below assume the repository root as the working directory.

## 01 Text Features

| Step | Script | Purpose | Main output |
| --- | --- | --- | --- |
| 01 | `01_text_features/01_compute_semantic_sentiment_features.py` | Semantic similarity and sentiment alignment from corrected transcripts | `04_data/scientific_dyad_analysis_results.csv` |
| 02 | `01_text_features/02_compute_structural_conversation_features.py` | Structural conversation features, role mapping, turn-taking, pronouns, lexical diversity | `04_data/structural_dyad_analysis_mapped.csv` |

## 02 LLM Annotation

| Step | Script | Purpose | Main output |
| --- | --- | --- | --- |
| 03 | `02_llm_annotation/03_annotate_turns_claude.py` | Claude/Anthropic turn-level and conversation-level annotation | `05_analysis_outputs/llm_annotation_output/` |
| 03b | `02_llm_annotation/03b_annotate_turns_qwen.py` | Qwen/OpenRouter version of the same annotation workflow | `05_analysis_outputs/qwen_annotation_output/` |

Set API keys through environment variables:

```bash
export ANTHROPIC_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## 03 Acoustic Alignment

| Step | Script | Purpose | Main output |
| --- | --- | --- | --- |
| 07 | `03_acoustic_alignment/07_align_manual_labels_to_whisperx.py` | Transfer corrected speaker labels onto WhisperX word timestamps | `04_data/labeled_turns.csv` |
| 08 | `03_acoustic_alignment/08_extract_acoustic_features.py` | Per-turn prosodic features from timestamped turns and WAV files | `04_data/acoustic_turns.csv` |
| 08b | `03_acoustic_alignment/08b_validate_speaker_enrollment.py` | Optional enrollment-based diarization validation/scaling | `04_data/enroll_validation.csv` |
| 09 | `03_acoustic_alignment/09_compute_vocal_alignment.py` | Turn-adjacent vocal entrainment metrics | `04_data/vocal_alignment_dyad.csv` |

## 04 Models

| Step | Script | Purpose | Main output |
| --- | --- | --- | --- |
| 04 | `04_models/04_analyze_llm_features.py` | Regression screen for LLM-derived dyad features | `05_analysis_outputs/llm_regression_output/` |
| 05 | `04_models/05_poster_multivariate_analysis.py` | Poster-oriented multivariate model, diagnostics, tables, and figures | `05_analysis_outputs/multivariate_output/` |
| 10 | `04_models/10_test_vocal_alignment_incremental_validity.py` | Pre-specified incremental/discriminant validity tests for vocal alignment | `05_analysis_outputs/dissociation_results.csv` and `.json` |
| 11 | `04_models/11_build_covariates.py` | Optional dyad-level covariates from participant master sheet | `04_data/covariates_dyad.csv` |

`04_models/legacy_dyad_analysis.py` is retained for comparison with earlier
analysis code; it is not the preferred replication entry point.

## 05 Figures

| Script | Purpose |
| --- | --- |
| `05_figures/06_make_poster_plots.py` | Compact poster plots from semantic/structural features |
| `05_figures/plot_audio_alignment.py` | Vocal-alignment illustration plots |
| `05_figures/plot_highlow_compare.py` | High- vs low-connection exemplar comparison plots |
| `05_figures/plot_vocal_outcome_heatmap.py` | Exploratory heatmap of vocal metrics vs outcomes |

## Suggested Run Order

```bash
python scripts/01_text_features/01_compute_semantic_sentiment_features.py
python scripts/01_text_features/02_compute_structural_conversation_features.py

python scripts/02_llm_annotation/03_annotate_turns_claude.py
python scripts/04_models/04_analyze_llm_features.py
python scripts/04_models/05_poster_multivariate_analysis.py

python scripts/03_acoustic_alignment/07_align_manual_labels_to_whisperx.py
python scripts/03_acoustic_alignment/08_extract_acoustic_features.py
python scripts/03_acoustic_alignment/09_compute_vocal_alignment.py
python scripts/04_models/10_test_vocal_alignment_incremental_validity.py

python scripts/05_figures/06_make_poster_plots.py
python scripts/05_figures/plot_audio_alignment.py
python scripts/05_figures/plot_highlow_compare.py
python scripts/05_figures/plot_vocal_outcome_heatmap.py
```

Some scripts require raw audio, restricted transcript data, or API access. See
`../PIPELINE.md` and `../data/README.md` for input placement and caveats.

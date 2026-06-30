# Script Index

Scripts are grouped by analysis stage. Filenames are descriptive; the canonical
execution order is documented in `../PIPELINE.md`.

## Text Features

| Script | Purpose | Main output |
| --- | --- | --- |
| `text_features/compute_semantic_sentiment_features.py` | Semantic similarity and sentiment alignment from corrected transcripts | `04_data/scientific_dyad_analysis_results.csv` |
| `text_features/compute_structural_conversation_features.py` | Turn-taking, pronouns, lexical diversity, role mapping, and other structural features | `04_data/structural_dyad_analysis_mapped.csv` |

## LLM Annotation

| Script | Purpose | Main output |
| --- | --- | --- |
| `llm_annotation/annotate_turns_claude.py` | Claude/Anthropic turn-level and conversation-level annotation | `05_analysis_outputs/llm_annotation_output/` |
| `llm_annotation/annotate_turns_qwen.py` | Qwen/OpenRouter version of the same annotation workflow | `05_analysis_outputs/qwen_annotation_output/` |

Set API keys through environment variables:

```bash
export ANTHROPIC_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## Acoustic Alignment

| Script | Purpose | Main output |
| --- | --- | --- |
| `acoustic_alignment/align_manual_labels_to_whisperx.py` | Transfer corrected speaker labels onto WhisperX word timestamps | `04_data/labeled_turns.csv` |
| `acoustic_alignment/extract_acoustic_features.py` | Per-turn prosodic features from timestamped turns and WAV files | `04_data/acoustic_turns.csv` |
| `acoustic_alignment/compute_vocal_alignment.py` | Turn-adjacent vocal entrainment metrics | `04_data/vocal_alignment_dyad.csv` |
| `acoustic_alignment/validate_speaker_enrollment.py` | Optional enrollment-based diarization validation/scaling | `04_data/enroll_validation.csv` |

## Models

| Script | Purpose | Main output |
| --- | --- | --- |
| `models/analyze_llm_features.py` | Regression screen for LLM-derived dyad features | `05_analysis_outputs/llm_regression_output/` |
| `models/analyze_multivariate_connection_models.py` | Multivariate models of dyad-level connection outcomes, diagnostics, tables, and figures | `05_analysis_outputs/multivariate_output/` |
| `models/test_vocal_alignment_incremental_validity.py` | Pre-specified incremental/discriminant validity tests for vocal alignment | `05_analysis_outputs/dissociation_results.csv` and `.json` |
| `models/build_covariates.py` | Optional dyad-level covariates from participant master sheet | `04_data/covariates_dyad.csv` |

`models/legacy/legacy_dyad_analysis.py` is retained for comparison with earlier
analysis code; it is not the preferred replication entry point.

## Figures

| Script | Purpose |
| --- | --- |
| `figures/plot_dyad_feature_outcome_associations.py` | Dyad-level feature-outcome association figures |
| `figures/plot_vocal_handoff_alignment.py` | Vocal handoff-alignment example and dataset-level scatter |
| `figures/compare_high_low_connection_handoffs.py` | High- vs low-connection handoff comparison figures |
| `figures/plot_vocal_alignment_outcome_correlations.py` | Exploratory vocal-alignment metric by outcome correlation heatmap |

Some scripts require raw audio, restricted transcript data, or API access. See
`../PIPELINE.md` and `../data/README.md` for input placement and caveats.

# Replication Guide

This guide describes what another researcher needs to reproduce the analysis
from this repository.

## 1. Environment

Create the Conda environment:

```bash
bash setup_conda_env.sh
conda activate cvs-conversation
```

The helper installs a Jupyter kernel named `Python (cvs-conversation)`.

## 2. Inputs

The code expects the repository root to contain:

```text
01_pipeline/all_srt/piper/*.txt
01_pipeline/all_srt/cloudy/*.txt
01_pipeline/outputs/<session>/*.srt
01_pipeline/_wav/*_16k.wav
04_data/Discussion Transcription Log - Sheet1.csv
04_data/outcomes.csv
```

For analyses from already-derived tables, the most important `04_data` files
are:

- `scientific_dyad_analysis_results.csv`
- `structural_dyad_analysis_mapped.csv`
- `dyad_level_dataset.csv`
- `labeled_turns.csv`
- `acoustic_turns.csv`
- `vocal_alignment_dyad.csv`
- `outcomes.csv`

Raw recordings and generated outputs are not committed because they are large
and may contain sensitive participant material.

## 3. Canonical Pipeline

Use the grouped numbered scripts in `scripts/`.

1. `scripts/01_text_features/01_compute_semantic_sentiment_features.py`: semantic similarity and sentiment alignment.
2. `scripts/01_text_features/02_compute_structural_conversation_features.py`: structural conversation features.
3. `scripts/02_llm_annotation/03_annotate_turns_claude.py`: LLM turn/conversation annotation, if API access is available.
4. `scripts/04_models/05_poster_multivariate_analysis.py`: poster-oriented multivariate analysis.
5. `scripts/03_acoustic_alignment/07_align_manual_labels_to_whisperx.py`: align corrected speaker labels to WhisperX timestamps.
6. `scripts/03_acoustic_alignment/08_extract_acoustic_features.py`: extract acoustic/prosodic features.
7. `scripts/03_acoustic_alignment/09_compute_vocal_alignment.py`: compute vocal entrainment metrics.
8. `scripts/04_models/10_test_vocal_alignment_incremental_validity.py`: test whether vocal alignment adds incremental variance.
9. `scripts/05_figures/`: generate figures after tables are produced.

## 4. Methodological Cautions

- The recordings use a single shared microphone. Vocal features are therefore
  turn-adjacent entrainment measures, not simultaneous two-channel synchrony.
- Small-N regression results should be treated as pilot evidence. The scripts
  emphasize pre-specified composites, leave-one-out checks, bootstrap intervals,
  and FDR correction where appropriate.
- LLM annotation outputs depend on model/provider version. Keep raw prompts,
  model names, and output CSVs with any archived analysis.

## 5. GitHub Hygiene

Recommended version-control policy:

- Commit scripts, documentation, environment files, and small non-sensitive
  derived CSVs needed for reproduction.
- Do not commit raw audio/video, WhisperX directories, API keys, caches, or
  generated plot/model output folders.
- Use releases or an external archive such as OSF/Zenodo for any larger
  replication package that includes approved data.

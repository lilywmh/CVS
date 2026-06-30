# Analysis Pipeline

Run commands from the repository root after activating the Conda environment:

```bash
conda activate cvs-conversation
```

## Required Inputs

| Input | Local path |
| --- | --- |
| Corrected transcript text and matching WhisperX SRT files | `01_pipeline/all_srt/` |
| 16 kHz mono WAV files for acoustic analyses | `01_pipeline/_wav/` |
| Outcome and derived feature tables | `04_data/` |
| Participant master sheet for optional covariates | `04_data/MASTER_SHEET_ONE_ROW_PER_PARTICIPANT.csv` |

These data are private and intentionally excluded from Git.

## Main Text and Annotation Pipeline

| Step | Command | Main output |
| --- | --- | --- |
| 01 Semantic/sentiment features | `python scripts/01_text_features/01_compute_semantic_sentiment_features.py` | `04_data/scientific_dyad_analysis_results.csv` |
| 02 Structural features | `python scripts/01_text_features/02_compute_structural_conversation_features.py` | `04_data/structural_dyad_analysis_mapped.csv` |
| 03 LLM annotation | `python scripts/02_llm_annotation/03_annotate_turns_claude.py` | `05_analysis_outputs/llm_annotation_output/` |
| 04 LLM feature analysis | `python scripts/04_models/04_analyze_llm_features.py` | `05_analysis_outputs/llm_regression_output/` |
| 05 Poster multivariate model | `python scripts/04_models/05_poster_multivariate_analysis.py` | `05_analysis_outputs/multivariate_output/` |
| 06 Poster plots | `python scripts/05_figures/06_make_poster_plots.py` | `06_figures/` |

Use `scripts/02_llm_annotation/03b_annotate_turns_qwen.py` for the optional
Qwen/OpenRouter annotation workflow.

## Acoustic and Vocal-Alignment Pipeline

The recordings use a shared microphone, so the vocal pipeline estimates
turn-adjacent entrainment rather than simultaneous speaker separation.

| Step | Command | Main output |
| --- | --- | --- |
| 07 Label/timestamp alignment | `python scripts/03_acoustic_alignment/07_align_manual_labels_to_whisperx.py` | `04_data/labeled_turns.csv` |
| 08 Acoustic turn features | `python scripts/03_acoustic_alignment/08_extract_acoustic_features.py --turns-csv 04_data/labeled_turns.csv --out 04_data/acoustic_turns.csv` | `04_data/acoustic_turns.csv` |
| 09 Vocal alignment | `python scripts/03_acoustic_alignment/09_compute_vocal_alignment.py` | `04_data/vocal_alignment_dyad.csv` |
| 10 Incremental-validity model | `python scripts/04_models/10_test_vocal_alignment_incremental_validity.py --n-perm 5000 --seed 42` | `05_analysis_outputs/dissociation_results.csv` |

Optional covariate construction:

```bash
python scripts/04_models/11_build_covariates.py \
    --master 04_data/MASTER_SHEET_ONE_ROW_PER_PARTICIPANT.csv
```

## Figure Scripts

| Step | Command | Purpose |
| --- | --- | --- |
| 12 | `python scripts/05_figures/12_plot_audio_alignment.py` | Vocal-alignment illustration plots |
| 13 | `python scripts/05_figures/13_plot_highlow_compare.py` | High- vs low-connection exemplar comparison plots |
| 14 | `python scripts/05_figures/14_plot_vocal_outcome_heatmap.py` | Exploratory heatmap of vocal metrics vs outcomes |

These are downstream visualization steps and should be interpreted relative to
the model outputs they consume.

## Optional and Legacy Scripts

Optional speaker-enrollment validation:

```bash
python scripts/03_acoustic_alignment/08b_validate_speaker_enrollment.py
```

Legacy comparison analysis:

```bash
python scripts/04_models/legacy/legacy_dyad_analysis.py
```

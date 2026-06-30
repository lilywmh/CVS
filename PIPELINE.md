# Analysis Pipeline

Run commands from the repository root after activating the Conda environment:

```bash
conda activate cvs-conversation
```

The script folders define the analysis stage. The tables below define the
canonical execution order.

## Required Inputs

| Input | Local path |
| --- | --- |
| Corrected transcript text and matching WhisperX SRT files | `01_pipeline/all_srt/` |
| 16 kHz mono WAV files for acoustic analyses | `01_pipeline/_wav/` |
| Outcome and derived feature tables | `04_data/` |
| Participant master sheet for optional covariates | `04_data/MASTER_SHEET_ONE_ROW_PER_PARTICIPANT.csv` |

These data are private and intentionally excluded from Git.

The paths above are local defaults from the development checkout. For a clean
private replication folder, set path variables before running scripts:

```bash
export CVS_SRT_ROOT=data/raw/all_srt
export CVS_WHISPERX_OUTPUTS=data/interim/whisperx_outputs
export CVS_WAV_DIR=data/interim/wav_16k
export CVS_DATA=data/derived
export CVS_ANALYSIS_OUTPUTS=outputs
export CVS_FIGURES=figures
```

## Core Text Pipeline

| Order | Command | Main output |
| --- | --- | --- |
| 1 | `python scripts/text_features/compute_semantic_sentiment_features.py` | `04_data/scientific_dyad_analysis_results.csv` |
| 2 | `python scripts/text_features/compute_structural_conversation_features.py` | `04_data/structural_dyad_analysis_mapped.csv` |
| 3 | `python scripts/llm_annotation/annotate_turns_claude.py` | `05_analysis_outputs/llm_annotation_output/` |
| 4 | `python scripts/models/analyze_llm_features.py` | `05_analysis_outputs/llm_regression_output/` |
| 5 | `python scripts/models/analyze_multivariate_connection_models.py` | `05_analysis_outputs/multivariate_output/` |
| 6 | `python scripts/figures/plot_dyad_feature_outcome_associations.py` | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/dyad_feature_outcome_figures/` |

Use `scripts/llm_annotation/annotate_turns_qwen.py` for the optional
Qwen/OpenRouter annotation workflow.

## Vocal-Alignment Pipeline

The recordings use a shared microphone, so this pipeline estimates
turn-adjacent entrainment rather than simultaneous speaker separation.

| Order | Command | Main output |
| --- | --- | --- |
| 1 | `python scripts/acoustic_alignment/align_manual_labels_to_whisperx.py` | `${CVS_DATA:-04_data}/labeled_turns.csv` |
| 2 | `python scripts/acoustic_alignment/extract_acoustic_features.py --turns-csv "${CVS_DATA:-04_data}/labeled_turns.csv" --out "${CVS_DATA:-04_data}/acoustic_turns.csv"` | `${CVS_DATA:-04_data}/acoustic_turns.csv` |
| 3 | `python scripts/acoustic_alignment/compute_vocal_alignment.py` | `04_data/vocal_alignment_dyad.csv` |
| 4 | `python scripts/models/test_vocal_alignment_incremental_validity.py --n-perm 5000 --seed 42` | `05_analysis_outputs/dissociation_results.csv` |
| 5 | `python scripts/figures/plot_vocal_handoff_alignment.py` | `${CVS_FIGURES:-06_figures}/vocal_handoff_alignment_example.*`, `${CVS_FIGURES:-06_figures}/vocal_handoff_intensity_scatter.*` |
| 6 | `python scripts/figures/compare_high_low_connection_handoffs.py` | `${CVS_FIGURES:-06_figures}/high_low_connection_*.{png,pdf}` |
| 7 | `python scripts/figures/plot_vocal_alignment_outcome_correlations.py` | `${CVS_FIGURES:-06_figures}/vocal_alignment_outcome_correlations.*`, `${CVS_DATA:-04_data}/vocal_alignment_outcome_correlations.csv` |

## Optional Scripts

Covariate construction:

```bash
python scripts/models/build_covariates.py \
    --master "${CVS_DATA:-04_data}/MASTER_SHEET_ONE_ROW_PER_PARTICIPANT.csv"
```

Speaker-enrollment validation:

```bash
python scripts/acoustic_alignment/validate_speaker_enrollment.py
```

Legacy comparison analysis:

```bash
python scripts/models/legacy/legacy_dyad_analysis.py
```

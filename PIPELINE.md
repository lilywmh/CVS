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
| Discussion video files for movement/facial analyses | `01_pipeline/video/` |
| Outcome and derived feature tables | `04_data/` |
| Participant master sheet for optional covariates | `04_data/MASTER_SHEET_ONE_ROW_PER_PARTICIPANT.csv` |

These data are private and intentionally excluded from Git.

The paths above are local defaults from the development checkout. For a clean
private replication folder, set path variables before running scripts:

```bash
export CVS_SRT_ROOT=data/raw/all_srt
export CVS_WHISPERX_OUTPUTS=data/interim/whisperx_outputs
export CVS_WAV_DIR=data/interim/wav_16k
export CVS_VIDEO_DIR=data/raw/videos
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
| 3 | `python scripts/acoustic_alignment/compute_vocal_alignment_floor_transfer.py` | `${CVS_DATA:-04_data}/vocal_alignment_floor_transfer_dyad.csv`, `${CVS_DATA:-04_data}/vocal_alignment_floor_transfer_excluded_turns.csv` |
| 4 | `python scripts/models/test_vocal_alignment_incremental_validity.py --vocal "${CVS_DATA:-04_data}/vocal_alignment_floor_transfer_dyad.csv" --out-csv "${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/floor_transfer_dissociation_results.csv" --out-json "${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/floor_transfer_dissociation_results.json" --n-perm 5000 --seed 42` | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/floor_transfer_dissociation_results.csv` |
| 5 | `python scripts/figures/plot_vocal_handoff_alignment.py` | `${CVS_FIGURES:-06_figures}/vocal_handoff_alignment_example.*`, `${CVS_FIGURES:-06_figures}/vocal_handoff_intensity_scatter.*` |
| 6 | `python scripts/figures/compare_high_low_connection_handoffs.py` | `${CVS_FIGURES:-06_figures}/high_low_connection_*.{png,pdf}` |
| 7 | `python scripts/figures/plot_vocal_alignment_outcome_correlations.py` | `${CVS_FIGURES:-06_figures}/vocal_alignment_outcome_correlations.*`, `${CVS_DATA:-04_data}/vocal_alignment_outcome_correlations.csv` |

## Video-Behavior Pipeline

This workflow processes private discussion videos in timestamped run
directories. The project-specific entry point treats
`scripts/acoustic_alignment/align_manual_labels_to_whisperx.py` output
(`${CVS_DATA:-04_data}/labeled_turns.csv`) and
`04_data/Discussion Transcription Log - Sheet1.csv` as authoritative. By
default it generates video metadata from `01_pipeline/recording_videos`,
preferring `*_discussion.*` files when multiple camera views exist. It does
not infer question boundaries or speaker turns from video.

For speaker-dependent exploratory checks, the turn-aligned unit is
`dyad_id x session_id x question_id x speaker_turn`. Because speaker turns can
be very short, the preferred movement-analysis unit is now the
speaker-independent question window:

```text
dyad_id x session_id x question_id x side
```

where `side` is `left` or `right`. This avoids relying on diarization or A/B
role mapping when estimating hand movement, gesture activity, left/right
movement synchrony, and rough partner-vs-front/screen head orientation.

The turn-aligned runner also writes upper-limb posture/movement proxies from
MediaPipe Pose shoulder-elbow-wrist landmarks. These tables support two stages:
whole-discussion participant summaries (`per_participant_arm_summary.csv`) and
within-person speaker-vs-listener comparisons
(`speaker_listener_arm_comparison.csv`).

| Order | Command | Main output |
| --- | --- | --- |
| 1 | `python scripts/video_behavior/run_turn_aligned_video_analysis.py --config config/video_turn_analysis.example.yml` | `${CVS_VIDEO_OUTPUTS:-data/interim/video_behavior/turn_runs}/<run_id>/results/` |
| 2 | `python scripts/video_behavior/summarize_video_behavior_qc.py` | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/video_behavior_qc_summary/` |
| 3 | `python scripts/video_behavior/check_roi_role_mapping.py --run-dir data/interim/video_behavior/turn_runs/<run_id>` | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/video_roi_role_qc/` |
| 4 | `python scripts/video_behavior/render_arm_skeleton_qc.py --source-run data/interim/video_behavior/turn_runs/<run_id>` | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/video_arm_skeleton_qc/` |
| 5 | `python scripts/video_behavior/run_time_window_arm_role_analysis.py --source-run data/interim/video_behavior/turn_runs/<run_id>` | `${CVS_VIDEO_OUTPUTS:-data/interim/video_behavior/time_window_arm_runs}/<run_id>/results/` |
| 6 | `python scripts/video_behavior/run_question_side_video_analysis.py --source-run data/interim/video_behavior/turn_runs/<run_id>` | `${CVS_VIDEO_OUTPUTS:-data/interim/video_behavior/side_question_runs}/<run_id>/results/` |
| 7 | `python scripts/models/analyze_video_behavior_features.py` | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/video_behavior_feature_analysis/` |

Participant side mapping is HARD-CODED for every video (fixed camera setup):
**A = door side = video RIGHT half; B = window side = video LEFT half.**
`run_turn_aligned_video_analysis.py` enforces this and fails on any config
that contradicts it. Turn runs created before 2026-07-07 used the reversed
mapping, so their participant-attributed (A/B) outputs are invalid — rerun
step 1 before using `per_speaker_summary`, `per_participant_arm_summary`,
`speaker_listener_arm_comparison`, or `per_turn_arm_role_features`.
Side-level (left/right) question outputs are unaffected.
`scripts/video_behavior/build_participant_side_map.py` remains available as
verification QC that the hard-coded standard holds in each session.

Useful modes:

```bash
python scripts/video_behavior/run_turn_aligned_video_analysis.py --config config/video_turn_analysis.example.yml --dry-run
python scripts/video_behavior/run_turn_aligned_video_analysis.py --config config/video_turn_analysis.example.yml --fail-fast
python scripts/video_behavior/summarize_video_behavior_qc.py --run-dir data/interim/video_behavior/turn_runs/<run_id>
python scripts/video_behavior/check_roi_role_mapping.py --run-dir data/interim/video_behavior/turn_runs/<run_id>
python scripts/video_behavior/render_arm_skeleton_qc.py --source-run data/interim/video_behavior/turn_runs/<run_id> --write-video
python scripts/video_behavior/run_time_window_arm_role_analysis.py --source-run data/interim/video_behavior/turn_runs/<run_id> --dry-run
python scripts/video_behavior/run_time_window_arm_role_analysis.py --source-run data/interim/video_behavior/turn_runs/<run_id>
python scripts/video_behavior/run_question_side_video_analysis.py --source-run data/interim/video_behavior/turn_runs/<run_id>
python scripts/video_behavior/run_question_side_video_analysis.py --source-run data/interim/video_behavior/turn_runs/<run_id> --face-orientation
```

`run_video_analysis.py` remains available as a lower-level video-processing
utility. `extract_video_behavior_features.py` is retained as an exploratory
extractor.

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

Gender/acoustic confound check:

```bash
python scripts/models/analyze_gender_acoustic_confound.py
```

Legacy comparison analysis:

```bash
python scripts/models/legacy/legacy_dyad_analysis.py
```

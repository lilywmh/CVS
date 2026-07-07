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
| `acoustic_alignment/compute_vocal_alignment_floor_transfer.py` | Primary floor-transfer vocal entrainment metrics excluding invalid, low-coverage, very short, and backchannel turns | `04_data/vocal_alignment_floor_transfer_dyad.csv`, `04_data/vocal_alignment_floor_transfer_excluded_turns.csv` |
| `acoustic_alignment/compute_vocal_alignment.py` | Turn-adjacent vocal entrainment metrics | `04_data/vocal_alignment_dyad.csv` |
| `acoustic_alignment/validate_speaker_enrollment.py` | Optional enrollment-based diarization validation/scaling | `04_data/enroll_validation.csv` |

## Video Behavior

| Script | Purpose | Main output |
| --- | --- | --- |
| `video_behavior/run_turn_aligned_video_analysis.py` | Project-specific turn-aligned video analysis using `acoustic_alignment/align_manual_labels_to_whisperx.py` output and discussion speaker logs as authoritative inputs | `${CVS_VIDEO_OUTPUTS:-data/interim/video_behavior/turn_runs}/<run_id>/results/` |
| `video_behavior/summarize_video_behavior_qc.py` | QC and first-pass descriptive summaries for completed turn-aligned video runs | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/video_behavior_qc_summary/` |
| `video_behavior/check_roi_role_mapping.py` | Sample-frame overlays and tables for auditing A/B role mapping to left/right participant ROIs | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/video_roi_role_qc/` |
| `video_behavior/check_speaker_visual_mapping.py` | FaceMesh mouth-movement QC for auditing `SPEAKER_00`/`SPEAKER_01` to visual participant mapping | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/speaker_visual_mapping_qc/` |
| `video_behavior/render_arm_skeleton_qc.py` | Frame overlays and optional MP4s connecting shoulder-elbow-wrist landmarks for upper-limb posture/movement QC | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/video_arm_skeleton_qc/` |
| `video_behavior/render_hand_landmark_qc.py` | Sample-frame overlays showing left/right ROI boxes and connected MediaPipe hand landmarks | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/video_hand_landmark_qc/` |
| `video_behavior/run_time_window_arm_role_analysis.py` | Fixed-window upper-limb analysis assigning participant roles as speaker, listener, silent, or overlap from audio-aligned turn timestamps | `${CVS_VIDEO_OUTPUTS:-data/interim/video_behavior/time_window_arm_runs}/<run_id>/results/` |
| `video_behavior/run_question_side_video_analysis.py` | Speaker-independent question-level left/right movement, gesture activity, synchrony, and partner/front-screen head-orientation proxy extraction | `${CVS_VIDEO_OUTPUTS:-data/interim/video_behavior/side_question_runs}/<run_id>/results/` |
| `video_behavior/run_video_analysis.py` | Workflow-ready video behavior pipeline with config, logs, manifests, QC reports, OpenFace parsing/wrapping, and MediaPipe Pose/Hands features | `${CVS_VIDEO_OUTPUTS:-data/interim/video_behavior/runs}/<run_id>/final/` |
| `video_behavior/extract_video_behavior_features.py` | Movement and facial-expression proxy extraction from private discussion videos | `04_data/video_behavior_frames.csv`, `04_data/video_behavior_dyad.csv` |
| `video_behavior/build_participant_side_map.py` | Per-session candidate mapping of participants A/B to visual left/right sides from mouth-motion QC plus the lab-standard prior (A = door side = video right), with manual-review flags | `05_analysis_outputs/participant_side_map/` |

## Models

| Script | Purpose | Main output |
| --- | --- | --- |
| `models/analyze_llm_features.py` | Regression screen for LLM-derived dyad features | `05_analysis_outputs/llm_regression_output/` |
| `models/analyze_multivariate_connection_models.py` | Multivariate models of dyad-level connection outcomes, diagnostics, tables, and figures | `05_analysis_outputs/multivariate_output/` |
| `models/test_vocal_alignment_incremental_validity.py` | Pre-specified incremental/discriminant validity tests for vocal alignment | `05_analysis_outputs/dissociation_results.csv` and `.json` |
| `models/analyze_video_behavior_features.py` | Exploratory video-feature reliability, condition comparison, speaker asymmetry, and outcome-correlation summaries | `05_analysis_outputs/video_behavior_feature_analysis/` |
| `models/analyze_gender_acoustic_confound.py` | Exploratory check for gender composition and raw pitch-difference as acoustic confounds | `05_analysis_outputs/gender_mixed_analysis.csv` and `gender_pitch_confound_check.csv` |
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

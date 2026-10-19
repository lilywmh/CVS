# Script Index

Scripts are grouped by analysis stage. Filenames are descriptive; the canonical
execution order is documented in `../PIPELINE.md`.

## Primary Multimodal Analysis

```bash
python -m scripts.multimodal_analysis.run_pipeline
```

The `multimodal_analysis/` package is the canonical integration layer for
transcript, acoustic, and visual question-level features. It combines cell
construction, measurement reliability, outcome associations, nested held-out
integration, figures, and the reproducibility manifest in one workflow.

## Text Features

| Script | Purpose | Main output |
| --- | --- | --- |
| `text_features/compute_semantic_sentiment_features.py` | Semantic similarity and sentiment alignment from corrected transcripts | `04_data/scientific_dyad_analysis_results.csv` |
| `text_features/compute_structural_conversation_features.py` | Turn-taking, pronouns, lexical diversity, role mapping, and other structural features | `04_data/structural_dyad_analysis_mapped.csv` |
| `text_features/build_naturalturn_units.py` | Prospective NaturalTurn adapter with source-turn lineage, timing QC, and a non-overwriting run manifest | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/topic_episode_pipeline/<run_id>/` |

## LLM Annotation

| Script | Purpose | Main output |
| --- | --- | --- |
| `llm_annotation/annotate_turns_claude.py` | Claude/Anthropic turn-level and conversation-level annotation | `05_analysis_outputs/llm_annotation_output/` |
| `llm_annotation/annotate_turns_qwen.py` | Qwen/OpenRouter version of the same annotation workflow | `05_analysis_outputs/qwen_annotation_output/` |
| `llm_annotation/prepare_topic_episode_annotation.py` | Prepare Def-DTS-derived topic prompts plus blinded NaturalTurn/topic validation sheets; makes no API calls | `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/topic_episode_pipeline/<run_id>/` |

Set API keys through environment variables:

```bash
export ANTHROPIC_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## Acoustic Alignment

| Script | Purpose | Main output |
| --- | --- | --- |
| `acoustic_alignment/align_manual_labels_to_whisperx.py` | Transfer corrected speaker labels onto WhisperX word timestamps | `04_data/labeled_turns.csv` |
| `acoustic_alignment/prepare_candor_for_vocal_alignment.py` | Convert CANDOR transcript CSVs and synchronized processed speaker media into this repo's timestamped-turn + WAV layout | `data/derived/candor/labeled_turns_backbiter.csv`, `data/interim/candor_wav/` |
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
| `models/build_candor_survey_outcomes.py` | Aggregate CANDOR post-conversation survey responses and optionally merge them with UUID-keyed vocal alignment | `candor_survey_participant.csv`, `candor_survey_dyad.csv`, `candor_survey_vocal_merged.csv` |
| `models/validate_candor_intensity_edge.py` | Canonical PREDICTOR-side measurement validity: original-adjacent non-overlap intensity-edge proximity and synchrony, participant-disjoint bijective pseudo pairs, time shifts, and independently normalized split-half reliability | `intensity_edge_validation/` |
| `models/qc_candor_outcome_reliability.py` | OUTCOME-side measurement validity: Cronbach alpha, ceiling/range restriction, partner ICC, reliability of the dyad mean, and the joint attenuation ceiling with the intensity-edge reliability | `<out-root>/outcome_qc_<timestamp>/` |
| `models/analyze_candor_survey_vocal.py` | Test the pre-specified CANDOR intensity-edge association, secondary outcomes, and covariate robustness | `survey_vocal_analysis/` |
| `models/replicate_candor_turn_taking.py` | Frozen directional external replication of CVS switch count, switch density, and turn-length findings in CANDOR | `turn_taking_replication/` |
| `models/analyze_candor_turn_duration_estimands.py` | Post-outcome CANDOR follow-up separating conversation-level turn duration from participant-relative duration across transcript segmentations | `turn_duration_estimands/run_<jobid>/` |
| `models/analyze_candor_floor_exchange_evaluation.py` | Harmonizes the CVS floor-exchange finding with CANDOR `i_like_you`, tests exact Backbiter switches, and checks outcome, floor-imbalance, segmentation, nonlinear, and influence sensitivity | `05_analysis_outputs/candor_floor_exchange_evaluation/` |
| `models/analyze_candor_floor_time_distribution.py` | Tests shared experience, experience inequality, and mutually adjusted actor/partner associations for turn duration; segmentation contrasts are secondary | `candor_floor_time_distribution/` |
| `models/analyze_shared_reality_floor_share.py` | Runs the prespecified dyad-mean, participant, and shared-reality-disagreement floor-share models separately for three CANDOR segmentations, then performs a 24-dyad CVS validation with structure-preserving permutation inference | `shared_reality_floor_share/` |
| `models/analyze_candor_liking_metaperception.py` | Tests whether relative speaking time and dyadic floor imbalance predict liking bias, calibration error, and actual liking received | `candor_liking_metaperception/` |
| `models/analyze_candor_dyadic_truth_bias.py` | Multilevel dyadic Truth-and-Bias sensitivity model separating direct tracking from assumed similarity, with crossed perceiver/evaluator/conversation effects | `candor_dyadic_truth_bias/` |
| `models/analyze_multivariate_connection_models.py` | Multivariate models of dyad-level connection outcomes, diagnostics, tables, and figures | `05_analysis_outputs/multivariate_output/` |
| `models/test_vocal_alignment_incremental_validity.py` | Pre-specified incremental/discriminant validity tests for vocal alignment | `05_analysis_outputs/dissociation_results.csv` and `.json` |
| `models/analyze_video_behavior_features.py` | Exploratory video-feature reliability, condition comparison, speaker asymmetry, and outcome-correlation summaries | `05_analysis_outputs/video_behavior_feature_analysis/` |
| `models/analyze_gender_acoustic_confound.py` | Exploratory check for gender composition and raw pitch-difference as acoustic confounds | `05_analysis_outputs/gender_mixed_analysis.csv` and `gender_pitch_confound_check.csv` |
| `models/build_covariates.py` | Optional dyad-level covariates from participant master sheet | `04_data/covariates_dyad.csv` |
| `models/build_multimodal_dyad_table.py` | Analysis-ready merged transcript + vocal + video tables (dyad x session and dyad level) with QC flags and merge diagnostics | `04_data/multimodal_dyad_session_table.csv`, `04_data/multimodal_dyad_table.csv` |
| `models/analyze_multimodal_video_incremental.py` | Pre-specified incremental-validity tests for the four selected video variables over turn-level + vocal predictors (permutation p, BH-FDR, QC-gated sensitivity) | `05_analysis_outputs/multimodal_video_incremental/` |
| `models/build_dynamic_trajectory_features.py` | Within-conversation dynamic trajectory features: per-window (question or time-bin) aggregation and trajectory summaries (slope, early/late shift, variability) | `05_analysis_outputs/dynamic_trajectory/` |
| `models/analyze_dynamic_trajectories.py` | Tests whether trajectory summaries predict partner_eval beyond static means; dyad-level regressions with permutation inference and bootstrap CIs | `05_analysis_outputs/dynamic_trajectory/` |
| `models/replicate_candor_dynamic_trajectory.py` | CANDOR replication of dynamic trajectory analysis using equal-duration time bins and `how_enjoyable` as outcome | `<CANDOR_OUT_DIR>/dynamic_trajectory/` |

`models/legacy/legacy_dyad_analysis.py` is retained for comparison with earlier
analysis code; it is not the preferred replication entry point.

## Figures

| Script | Purpose |
| --- | --- |
| `figures/plot_dyad_feature_outcome_associations.py` | Dyad-level feature-outcome association figures |
| `figures/plot_vocal_handoff_alignment.py` | Vocal handoff-alignment example and dataset-level scatter |
| `figures/compare_high_low_connection_handoffs.py` | High- vs low-connection handoff comparison figures |
| `figures/plot_vocal_alignment_outcome_correlations.py` | Exploratory vocal-alignment metric by outcome correlation heatmap |
| `figures/plot_multimodal_video_figures.py` | Video-stage paper figures: predictor distinctness heatmap, speaker-vs-listener validity slope plot, vocal-vs-video incremental ΔR² comparison |
| `figures/plot_dynamic_trajectories.py` | Dynamic trajectory figures: spaghetti trajectories by connection level, static vs dynamic coefficient forest plot, strongest-predictor scatter |

Some scripts require raw audio, restricted transcript data, or API access. See
`../PIPELINE.md` and `../data/README.md` for input placement and caveats.

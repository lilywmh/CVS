# Video Behavior Pipeline

`run_turn_aligned_video_analysis.py` is the project-specific entry point for
private CVS discussion videos. It treats
`scripts/acoustic_alignment/align_manual_labels_to_whisperx.py` output
(`${CVS_DATA:-04_data}/labeled_turns.csv`) and
`04_data/Discussion Transcription Log - Sheet1.csv` as authoritative. It does
not infer question boundaries or speaker turns from video.

`run_video_analysis.py` remains available as a lower-level whole-video utility.

## Raw Input Boundary

Raw videos are read-only inputs. The pipeline never modifies, renames, moves,
clips, or overwrites raw recordings.

## Environment Check

The MediaPipe Pose/Hands stage requires a Python 3.10/3.11 environment with a
Solutions-compatible MediaPipe build. Before running the full pipeline:

```bash
conda env create -f environment.yml
conda activate cvs-conversation
python scripts/video_behavior/check_video_dependencies.py
```

If the check reports Python 3.13 or missing `mediapipe_solutions`, recreate or
switch to the `cvs-conversation` environment instead of the incompatible env.

Accepted formats:

- `.mp4`
- `.mov`
- `.m4v`
- `.avi`
- `.mkv`

Default raw input path:

```bash
${CVS_VIDEO_DIR:-01_pipeline/video}
```

## Run Directory Boundary

Each run writes to:

```bash
${CVS_VIDEO_OUTPUTS:-data/interim/video_behavior/runs}/<run_id>/
```

Run IDs default to:

```text
video_behavior_YYYYMMDD_HHMMSS
```

Completed run directories are not overwritten. Incomplete runs can be resumed
with `--resume` or removed with `--clean-failed-run`.

## Output Structure

For the project-specific turn-aligned workflow:

```text
<run_id>/
  config_used.yaml
  run_manifest.json
  logs/
    run.log
  metadata/
    video_metadata_checked.csv
    labeled_turns_checked.csv
    speaker_logs_checked.csv
  results/
    per_turn_video_features.csv
    arm_frame_features.csv
    per_turn_arm_role_features.csv
    per_participant_arm_summary.csv
    per_participant_role_arm_summary.csv
    speaker_listener_arm_comparison.csv
    per_question_summary.csv
    per_speaker_summary.csv
    per_session_summary.csv
    cross_session_comparison.csv
  qc/
    turn_level_qc.csv
    question_level_qc.csv
    speaker_level_qc.csv
    qc_report.html
```

The main analysis unit is:

```text
dyad_id x session_id x question_id x turn_id x speaker_label
```

The upper-limb extension also writes two complementary arm-analysis units:

```text
dyad_id x session_id x question_id x turn_id x participant_id x conversational_role
dyad_id x session_id x video_id x participant_id
```

`conversational_role` is `speaker` for the participant who owns the current
transcript turn and `listener` for the other configured participant ROI. These
arm features are descriptive posture/movement proxies from shoulder-elbow-wrist
geometry; they are not emotion or intent labels.

The default source inputs are:

- videos in `01_pipeline/recording_videos`
- labeled turns in `04_data/labeled_turns.csv`
- speaker-role mappings in `04_data/Discussion Transcription Log - Sheet1.csv`

If `project.video_metadata_path` is blank, video metadata is generated from
filenames such as `dyad12_250928_piper_discussion.mp4`. The generated
`video_id` is `dyad_id_condition`, for example `dyad12_250928_piper`.

The native `labeled_turns.csv` output is accepted directly. It is normalized
internally from:

- `question` to `question_id`
- `turn_idx` to `turn_id`
- `speaker` to `speaker_label`
- `text` to `utterance_text`
- `t_start` and `t_end` to `start_time_sec` and `end_time_sec`
- `coverage` to `alignment_confidence`

After normalization, the turn table must include:

- `dyad_id`
- `session_id`
- `video_id`
- `question_id`
- `turn_id`
- `speaker_label`
- `utterance_text`
- `start_time_sec`
- `end_time_sec`

For the lower-level whole-video workflow:

```text
<run_id>/
  config/
    config_used.yaml
    software_versions.json
  logs/
    run.log
  metadata/
    input_manifest.csv
    output_manifest.csv
    video_metadata.csv
    run_summary.json
  openface/
    <video_id>/facial_features_long.csv
  mediapipe/
    <video_id>/pose_keypoints_long.csv
    <video_id>/hand_keypoints_long.csv
    <video_id>/gesture_features_long.csv
  qc/
    <video_id>/qc_report.json
    <video_id>/qc_report.md
    <video_id>/qc_plots.png
    <video_id>/sampled_overlay_frames/
  final/
    video_facial_features_long.csv
    video_pose_keypoints_long.csv
    video_hand_keypoints_long.csv
    video_gesture_features_long.csv
    video_behavior_summary.csv
    video_qc_summary.csv
    processing_manifest.json
```

## Units and Coordinates

- `frame_index`: zero-based native video frame index.
- `timestamp_ms`: milliseconds from video start.
- Pixel coordinates: origin at top-left, `x` increases rightward, `y`
  increases downward.
- Participant ROIs: normalized `[x0, y0, x1, y1]` coordinates.
- MediaPipe `z`: detector-native relative depth, not metric depth.
- Velocity: pixels per second.
- Acceleration: pixels per second squared.
- Jerk: pixels per second cubed.

## Validation

The runner validates:

- every video has transcript-alignment rows
- every video has speaker-log rows when speaker logs include `video_id`
- every dyad has the expected number of sessions
- every session has configured expected questions, when provided
- every question has transcript turns
- transcript timestamps are within video duration
- turn start/end times are ordered
- impossible overlaps are flagged unless explicitly allowed
- very short, very long, or empty turns are flagged
- unknown speaker labels are flagged
- video-derived features can be matched back to transcript turns
- input paths and formats
- video FPS, frame count, dimensions, and duration
- config schema
- dependency availability for enabled stages
- output file existence and non-empty status
- required CSV columns
- frame index and timestamp ordering
- keypoint spatial bounds
- missing/empty detection results
- pose and hand detection rates

Each video receives a QC status:

- `PASS`: required outputs exist and QC thresholds are met.
- `WARNING`: outputs exist but detection rates or metadata checks raise concerns.
- `FAIL`: unreadable input, invalid outputs, or critical schema failures.

## Scientific Tool Roles

OpenFace 2.x is the preferred facial-behavior source for:

- AU intensity/presence
- AU12 smile
- AU6 + AU12 Duchenne marker inputs
- AU4 brow furrow
- gaze vectors
- head pose yaw/pitch/roll

MediaPipe is used for gesture/body behavior:

- fixed left/right participant ROI processing
- 33-point Pose Heavy keypoints
- 21-point hand keypoints
- shoulder-elbow-wrist upper-limb posture and movement states
- hand velocity, acceleration, and jerk
- hand-to-face and hand-to-shoulder distances
- gesture rate and simple stroke/hold heuristics
- bilateral symmetry index

For two-person fixed-camera videos, the default config processes left and right
halves separately. Map `left` and `right` to private participant IDs in a
restricted metadata table outside the public repo.

For the turn-aligned workflow, update `participants` in
`config/video_turn_analysis.example.yml` so each `speaker_label` maps to the
correct participant ROI or role label used in the aligned transcript and
discussion logs.

## ROI/Role QC

After a completed turn-aligned run, audit the assumed A/B-to-left/right mapping
before interpreting hand or gesture features:

```bash
python scripts/video_behavior/check_roi_role_mapping.py \
  --run-dir data/interim/video_behavior/turn_runs/<run_id>
```

This writes to:

```text
05_analysis_outputs/video_roi_role_qc/
  roi_role_qc_report.md
  roi_role_detection_summary.csv
  roi_role_asymmetry_flags.csv
  roi_role_review_samples.csv
  review_frames/
```

Use the sampled frame overlays to confirm that the highlighted role matches the
expected person on screen. Large A/B differences in hand detection can reflect
real behavior, but they can also reflect ROI assignment, occlusion, arms-crossed
posture, or one participant being harder to detect.

## Upper-Limb Skeleton QC

Render connected shoulder-elbow-wrist overlays for frame-by-frame visual review:

```bash
python scripts/video_behavior/render_arm_skeleton_qc.py \
  --source-run data/interim/video_behavior/turn_runs/<run_id>
```

This writes to:

```text
05_analysis_outputs/video_arm_skeleton_qc/
  arm_skeleton_qc_<timestamp>/
    frames/
    arm_skeleton_frame_manifest.csv
    arm_skeleton_qc_summary.csv
    run_manifest.json
```

Use `--write-video` to also create an MP4 overlay video from the rendered
frames. Use `--every-frame` only for narrow selections because full native-frame
rendering can be large.

## Speaker Visual Mapping QC

When speaker/listener labels look wrong, audit the
`SPEAKER_00`/`SPEAKER_01` to visual-participant mapping with mouth movement:

```bash
python scripts/video_behavior/check_speaker_visual_mapping.py \
  --source-run data/interim/video_behavior/turn_runs/<run_id>
```

This uses MediaPipe FaceMesh inside each configured participant ROI and
compares per-turn mouth-open dynamics across participants. It writes:

```text
${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/speaker_visual_mapping_qc/<run_id>/
  intermediate/
    per_turn_mouth_mapping_qc.csv
    per_turn_participant_mouth_metrics.csv
  results/
    speaker_visual_mapping_summary.csv
  qc/
    speaker_visual_mapping_conflicts.csv
    failed_videos.csv
```

Use it as QC evidence before interpreting speaker/listener arm behavior. It
does not modify source labels or rewrite the speaker log.

## Participant Side Map (A/B to left/right)

The runners currently assume a single global ROI mapping (A = left half,
B = right half). The lab standard is the opposite: **A = door side, appearing
on the video RIGHT; B = window side, appearing on the video LEFT.** The
mouth-motion QC contradicts the global default in most sessions, so the
mapping must be resolved per session before any participant-attributed (A/B)
feature is interpreted.

```bash
python scripts/video_behavior/build_participant_side_map.py \
  --mapping-run 05_analysis_outputs/speaker_visual_mapping_qc/speaker_visual_mapping_full
```

Writes to `${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/participant_side_map/`:

```text
participant_side_map_candidate.csv   one row per dyad x session
participant_side_map_summary.md      review instructions and flagged sessions
run_manifest.json                    inputs (sha256), parameters, version
```

Decision logic: mouth-motion evidence scores the A-left vs A-right hypotheses
per session; strong two-sided evidence decides the side, otherwise the lab
standard (A = right) is used as the prior and the session is flagged
`needs_manual_review`. Review flagged sessions against
`video_roi_role_qc/review_frames/` or raw video, fill `manual_side_A` and set
`review_status = confirmed`, then promote the confirmed table to
`04_data/participant_side_map.csv`. Downstream runners should consume only the
confirmed table (integration patch pending).

This script is additive QC: it never modifies configs, source labels, or
existing run outputs.

## Time-Window Speaker/Listener/Silent Roles

Turn-level arm analysis only covers transcript turns. To include silent gaps
and overlapping speech, run the fixed-window role analysis:

```bash
python scripts/video_behavior/run_time_window_arm_role_analysis.py \
  --source-run data/interim/video_behavior/turn_runs/<run_id>
```

Each participant in each time window is assigned from audio-aligned turn
overlap:

- `speaker`: this participant has speech in the window
- `listener`: the partner has speech and this participant does not
- `silent`: no configured participant has speech in the window
- `overlap`: more than one participant has speech in the window

Outputs are written to:

```text
${CVS_VIDEO_OUTPUTS:-data/interim/video_behavior/time_window_arm_runs}/<run_id>/
  metadata/
    question_windows.csv
    time_windows.csv
    time_window_roles.csv
  intermediate/
    per_frame_time_window_arm_features.csv
  results/
    per_time_window_role_features.csv
    per_participant_time_window_role_summary.csv
    time_window_role_counts.csv
  qc/
    failed_videos.csv
```

Use `--dry-run` to inspect role counts before running MediaPipe, and adjust
`--window-sec`, `--stride-sec`, `--min-speech-overlap-sec`, and
`--min-speech-overlap-ratio` for sensitivity checks.

## Hand Landmark QC

To inspect how MediaPipe is finding hands, render sampled frames with connected
hand landmarks:

```bash
python scripts/video_behavior/render_hand_landmark_qc.py \
  --source-run data/interim/video_behavior/turn_runs/<run_id>
```

By default this renders one question from one video so the output is quick to
inspect. To target a specific video/question:

```bash
python scripts/video_behavior/render_hand_landmark_qc.py \
  --source-run data/interim/video_behavior/turn_runs/<run_id> \
  --video-id dyad10_250916_cloudy \
  --question-id 1
```

This writes annotated images and detection summaries to:

```text
05_analysis_outputs/video_hand_landmark_qc/<run_id>/
  hand_landmark_qc_report.md
  hand_landmark_qc_frames.csv
  hand_landmark_qc_summary.csv
  review_frames/
```

## Question-Level Left/Right Analysis

For the primary movement analysis, prefer speaker-independent question windows
over short speaker turns:

```bash
python scripts/video_behavior/run_question_side_video_analysis.py \
  --source-run data/interim/video_behavior/turn_runs/<run_id>
```

This reuses the checked video metadata and labeled-turn table from an existing
turn-aligned run to define question start/end times, then extracts left/right
ROI features without using `SPEAKER_00`, `SPEAKER_01`, A, or B. The same run
also derives a rough head-orientation target for each side:

- `partner`: head oriented inward, toward the other participant
- `front_screen`: head approximately front-facing, consistent with looking at
  the screen/camera/front of room
- `away`: head oriented outward, away from the other participant
- `unclear`: head orientation was not detected cleanly

Main outputs:

```text
data/interim/video_behavior/side_question_runs/<run_id>/
  config_used.json
  run_manifest.json
  metadata/question_windows.csv
  intermediate/per_frame_side_features.csv
  results/per_question_side_features.csv
  results/per_question_interaction_features.csv
  results/per_session_side_summary.csv
  results/per_session_interaction_summary.csv
```

Key orientation columns:

- `looking_partner_proxy_rate`
- `looking_front_screen_proxy_rate`
- `looking_away_proxy_rate`
- `head_orientation_detection_rate`
- `mutual_partner_orientation_rate`
- `both_front_screen_orientation_rate`

These are not calibrated eye-gaze labels. They are rough head/face-orientation
proxies and should be validated against sampled frames before scientific
interpretation. Use `--face-orientation` to add FaceMesh when available; Pose
head landmarks are used as the default fallback because they are more stable in
the current fixed-camera recordings.

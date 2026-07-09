# CLAUDE.md — Project context for Claude Code

## What this project is

Dyadic co-viewing conversation study (N≈24 dyads, strangers). Each dyad watches
two animated shorts (`piper`, `cloudy`) and discusses each; discussions are
recorded (single shared mic + fixed camera). Goal: a publishable undergraduate
research project and a 10–20 page manuscript-style writing sample on
transcript, vocal, and video-behavior coordination vs. social connection
outcomes. Reproducibility, QC, and interpretability take priority over
advanced methods.

## Canonical docs (read before changing anything)

- `PIPELINE.md` — canonical run order for all stages.
- `scripts/README.md` — script index.
- `scripts/video_behavior/README.md` — video-stage details and QC tools.
- `07_writeups/AUDIT_2026-07-07_video_stage.md` — recent audit + key decisions.
- `07_writeups/methods_results_draft.md` — methods/results draft (vocal stage).

## Hard rules (do not violate)

1. **Participant side mapping is HARD-CODED for every video:** on screen,
   participant **A (door side) = RIGHT half; B (window side) = LEFT half**.
   Enforced by `participant_roi_lookup()` in
   `scripts/video_behavior/run_turn_aligned_video_analysis.py`
   (`PARTICIPANT_SIDE_STANDARD`). No per-session exceptions.
2. The manually corrected aligned/review SRT speaker labels are authoritative.
   FaceMesh mouth-motion QC is advisory only — do not re-open the side-mapping
   question based on it.
3. Raw videos (`01_pipeline/recording_videos/`) are read-only. Never modify,
   move, or overwrite them.
4. Do not overwrite completed run directories; runners write new timestamped
   dirs under `data/interim/video_behavior/`.
5. Do not break existing transcript (`04_data/labeled_turns.csv`) or acoustic
   outputs — they are upstream dependencies of the video stage.
6. `SPEAKER_00/01` → A/B mapping comes from
   `04_data/Discussion Transcription Log - Sheet1.csv` per session.
   Never assume SPEAKER_00 is the same participant across videos.

## Canonical runs (as of 2026-07-09)

- Turn-aligned video (CORRECT mapping):
  `data/interim/video_behavior/turn_runs/turn_aligned_video_20260707_120008`
- Time-window arm roles (correct, completed 2026-07-08 23:17):
  `data/interim/video_behavior/time_window_arm_runs/time_window_arm_20260708_214626`
- Question-side (+face orientation), completed 2026-07-09 11:14:
  `side_question_runs/side_question_video_20260708_231724`
  (46 sessions, 410 rows, 0 failed videos).
- All runs from before 2026-07-07 used a REVERSED A/B mapping; their
  participant-attributed outputs are invalid. Stale ROI QC frames are in
  `05_analysis_outputs/video_roi_role_qc/review_frames_STALE_reversed_mapping_run_20260701/`.

## Environment

- `conda activate cvs-conversation` (Python 3.10/3.11; MediaPipe Solutions API
  required — check with `python scripts/video_behavior/check_video_dependencies.py`).
- Long video jobs: run with `caffeinate -i` so macOS doesn't sleep mid-run.

## Current status / next steps

Video pipeline fully regenerated on the corrected mapping as of 2026-07-09
14:51 (turn run, time-window roles, question-side features, ROI QC, QC
summary, feature analysis all trace to run `20260707_120008`). Next:

1. Select 3–5 reliable, interpretable video variables (Pose/arm features are
   robust; hand detection is unreliable in many sessions — QC-gate it).
2. Build analysis-ready merged transcript+voice+video tables.
3. Draft analysis plan, figures, and the 10–20 page writing sample.

## Engineering contract (summary)

Validate inputs and fail with informative errors; make every output traceable
to dyad/session/question/turn/speaker + script + run; write manifests and
logs; safe output dirs (timestamped, no overwrites); deterministic where
possible; minimal targeted changes; no style-only refactors; QC outputs for
every major step; document purpose/inputs/outputs/example command for every
new script.

#!/usr/bin/env bash
set -euo pipefail

# Application script for CANDOR vocal-alignment replication on CARC.
# Run from the repository root after activating the cvs-conversation env, or
# call this from scripts/carc/candor_backbiter.sbatch.

CANDOR_INPUT="${CANDOR_INPUT:-/scratch1/$USER/candor/raw_media_part_*.zip}"
CANDOR_OUT_DIR="${CANDOR_OUT_DIR:-data/derived/candor}"
CANDOR_WAV_DIR="${CANDOR_WAV_DIR:-data/interim/candor_wav}"
CANDOR_LIMIT="${CANDOR_LIMIT:-}"

mkdir -p "$CANDOR_OUT_DIR" "$CANDOR_WAV_DIR"

limit_args=()
if [[ -n "$CANDOR_LIMIT" ]]; then
  limit_args=(--limit "$CANDOR_LIMIT")
fi

python scripts/acoustic_alignment/prepare_candor_for_vocal_alignment.py \
  --input "$CANDOR_INPUT" \
  --out "$CANDOR_OUT_DIR/labeled_turns_backbiter.csv" \
  --wav-dir "$CANDOR_WAV_DIR" \
  --extract-audio \
  "${limit_args[@]}"

python scripts/acoustic_alignment/extract_acoustic_features.py \
  --turns-csv "$CANDOR_OUT_DIR/labeled_turns_backbiter.csv" \
  --out "$CANDOR_OUT_DIR/acoustic_turns_backbiter.csv"

python scripts/acoustic_alignment/compute_vocal_alignment_floor_transfer.py \
  --in "$CANDOR_OUT_DIR/acoustic_turns_backbiter.csv" \
  --out "$CANDOR_OUT_DIR/vocal_alignment_floor_transfer_backbiter.csv" \
  --excluded-turns-out "$CANDOR_OUT_DIR/vocal_alignment_floor_transfer_excluded_turns_backbiter.csv"

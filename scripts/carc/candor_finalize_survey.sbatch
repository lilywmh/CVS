#!/bin/bash

#SBATCH --job-name=candor_finalize
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/candor_finalize_%j.out
#SBATCH --error=logs/candor_finalize_%j.err

# Rebuild only the final alignment tables with conversation UUIDs, then merge
# them with CANDOR survey outcomes. Acoustic chunks and WAVs are not touched.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate cvs-conversation

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

CANDOR_INPUT="${CANDOR_INPUT:-/scratch1/$USER/betterup_raw_media/extracted}"
CANDOR_OUT_DIR="${CANDOR_OUT_DIR:-/scratch1/$USER/betterup_raw_media/outputs/candor_full_resumable}"

acoustic="$CANDOR_OUT_DIR/acoustic_turns_backbiter.csv"
alignment="$CANDOR_OUT_DIR/vocal_alignment_floor_transfer_backbiter.csv"
excluded="$CANDOR_OUT_DIR/vocal_alignment_floor_transfer_excluded_turns_backbiter.csv"
staging="$CANDOR_OUT_DIR/.candor_finalize_${SLURM_JOB_ID}"

if [[ ! -s "$acoustic" ]]; then
  echo "Missing acoustic input: $acoustic" >&2
  exit 1
fi

mkdir -p "$staging"

python scripts/acoustic_alignment/compute_vocal_alignment_floor_transfer.py \
  --in "$acoustic" \
  --out "$staging/$(basename "$alignment")" \
  --excluded-turns-out "$staging/$(basename "$excluded")"

python scripts/models/build_candor_survey_outcomes.py \
  --input "$CANDOR_INPUT" \
  --out-dir "$staging" \
  --vocal "$staging/$(basename "$alignment")"

mv "$staging/$(basename "$alignment")" "$alignment"
mv "$staging/vocal_alignment_floor_transfer_backbiter_with_bc.csv" \
  "$CANDOR_OUT_DIR/vocal_alignment_floor_transfer_backbiter_with_bc.csv"
mv "$staging/vocal_alignment_floor_transfer_backbiter_surrogate.csv" \
  "$CANDOR_OUT_DIR/vocal_alignment_floor_transfer_backbiter_surrogate.csv"
mv "$staging/$(basename "$excluded")" "$excluded"
mv "$staging/candor_survey_participant.csv" "$CANDOR_OUT_DIR/candor_survey_participant.csv"
mv "$staging/candor_survey_dyad.csv" "$CANDOR_OUT_DIR/candor_survey_dyad.csv"
mv "$staging/candor_survey_vocal_merged.csv" "$CANDOR_OUT_DIR/candor_survey_vocal_merged.csv"
rmdir "$staging"

echo "CANDOR survey + vocal finalization complete: $CANDOR_OUT_DIR"

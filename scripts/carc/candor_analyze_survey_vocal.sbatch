#!/bin/bash

#SBATCH --job-name=candor_survey_va
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/candor_survey_va_%j.out
#SBATCH --error=logs/candor_survey_va_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate cvs-conversation

CANDOR_OUT_DIR="${CANDOR_OUT_DIR:-/scratch1/$USER/betterup_raw_media/outputs/candor_full_resumable}"
input="$CANDOR_OUT_DIR/candor_survey_vocal_merged.csv"
output="$CANDOR_OUT_DIR/survey_vocal_analysis"

if [[ ! -s "$input" ]]; then
  echo "Missing merged survey-vocal input: $input" >&2
  exit 1
fi

python scripts/models/analyze_candor_survey_vocal.py \
  --input "$input" \
  --out-dir "$output" \
  --n-perm "${CANDOR_N_PERM:-10000}" \
  --seed "${CANDOR_SEED:-42}"

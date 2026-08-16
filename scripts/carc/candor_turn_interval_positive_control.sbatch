#!/bin/bash
#SBATCH --job-name=candor_turn_pc
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/candor_turn_pc_%j.out
#SBATCH --error=logs/candor_turn_pc_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch1/wulily/betterup_raw_media/code/cvs_conversation}"
CANDOR_OUT_DIR="${CANDOR_OUT_DIR:-/scratch1/wulily/betterup_raw_media/outputs/candor_full_processed_resumable}"
RESULT_DIR="${RESULT_DIR:-$CANDOR_OUT_DIR/turn_interval_positive_control}"

cd "$PROJECT_DIR"
mkdir -p logs "$RESULT_DIR"

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate cvs-conversation

python scripts/models/replicate_candor_turn_interval_enjoyment.py \
  --turns "$CANDOR_OUT_DIR/labeled_turns_backbiter.csv" \
  --survey "$CANDOR_OUT_DIR/candor_survey_participant.csv" \
  --out-dir "$RESULT_DIR"

echo "Positive-control report: $RESULT_DIR/turn_interval_positive_control_report.md"

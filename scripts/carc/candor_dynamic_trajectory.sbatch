#!/bin/bash
#SBATCH --job-name=candor_dyn_traj
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/candor_dyn_traj_%j.out
#SBATCH --error=logs/candor_dyn_traj_%j.err
#SBATCH --account=elisab_1216

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch1/wulily/betterup_raw_media/code/cvs_conversation}"
CANDOR_OUT_DIR="${CANDOR_OUT_DIR:-/scratch1/wulily/betterup_raw_media/outputs/candor_full_processed_resumable}"
RESULT_DIR="${RESULT_DIR:-$CANDOR_OUT_DIR/dynamic_trajectory/run_${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs "$RESULT_DIR"

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate cvs-conversation

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

turns="$CANDOR_OUT_DIR/acoustic_turns_backbiter.csv"
survey="$CANDOR_OUT_DIR/candor_survey_participant.csv"

if [[ ! -s "$turns" || ! -s "$survey" ]]; then
  echo "Missing required input: $turns or $survey" >&2
  exit 1
fi

python scripts/models/replicate_candor_dynamic_trajectory.py \
  --turns  "$turns" \
  --survey "$survey" \
  --out    "$RESULT_DIR" \
  --outcome how_enjoyable \
  --nperm 5000 \
  --nboot 2000 \
  --seed 42

echo "CANDOR dynamic trajectory replication: $RESULT_DIR"

#!/bin/bash

#SBATCH --job-name=candor_backbiter_resume
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/candor_backbiter_resume_%j.out
#SBATCH --error=logs/candor_backbiter_resume_%j.err

# If you have multiple CARC project accounts, uncomment and set this:
##SBATCH --account=<project_id>

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate cvs-conversation

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

bash scripts/carc/run_candor_backbiter_resumable.sh

#!/bin/bash
#SBATCH --job-name=candor_edge_qc
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/candor_edge_qc_%j.out
#SBATCH --error=logs/candor_edge_qc_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch1/wulily/betterup_raw_media/code/cvs_conversation}"
CANDOR_OUT_DIR="${CANDOR_OUT_DIR:-/scratch1/wulily/betterup_raw_media/outputs/candor_full_processed_resumable}"
VALIDATION_DIR="${VALIDATION_DIR:-$CANDOR_OUT_DIR/intensity_edge_validation}"

cd "$PROJECT_DIR"
mkdir -p logs "$VALIDATION_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    module load conda
  fi
  eval "$(conda shell.bash hook)"
  conda activate cvs-conversation
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c "import numpy, pandas, scipy; print('Python:', __import__('sys').executable)"

NULL_ARGS=()
if [[ "${RESTART_NULL:-0}" == "1" ]]; then
  NULL_ARGS=(--restart-null)
fi
"$PYTHON_BIN" scripts/models/validate_candor_intensity_edge.py \
  --turns "$CANDOR_OUT_DIR/acoustic_turns_backbiter.csv" \
  --out-dir "$VALIDATION_DIR" \
  --n-permutations "${N_PERMUTATIONS:-1000}" \
  --seed "${SEED:-42}" \
  "${NULL_ARGS[@]}"

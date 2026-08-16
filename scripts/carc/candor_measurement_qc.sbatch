#!/bin/bash
#
# candor_measurement_qc.sbatch
# ============================
# Runs BOTH sides of measurement validity before any LLM-annotation spend:
#
#   1. PREDICTOR side  scripts/models/validate_candor_intensity_edge.py
#                      (pseudo-dyad null, time-shift null, split-half
#                      reliability of the intensity-edge construct)
#   2. OUTCOME side    scripts/models/qc_candor_outcome_reliability.py
#                      (Cronbach alpha, ceiling/range restriction, partner ICC,
#                      reliability of the dyad mean)
#
# Step 2 consumes step 1's reliability table so the attenuation ceiling
# reflects measurement error on both sides: max observable r = sqrt(r_xx*r_yy).
#
# Decision rule: continue to LLM annotation only if the intensity-edge
# construct beats both nulls AND the outcome has usable partner agreement. A
# survey null is not interpretable as evidence against the theory when either
# side fails.
#
# scripts/carc/candor_validate_intensity_edge.sbatch runs step 1 alone; use
# this wrapper when you want both reports in one job.
#
# Submit with:
#   sbatch scripts/carc/candor_measurement_qc.sbatch
#
# CPU-only; reads existing tables, no audio is reprocessed.

#SBATCH --job-name=candor_measurement_qc
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/candor_measurement_qc_%j.out
#SBATCH --error=logs/candor_measurement_qc_%j.err

# If you have multiple CARC project accounts, uncomment and set this:
##SBATCH --account=<project_id>

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch1/wulily/betterup_raw_media/code/cvs_conversation}"
CANDOR_OUT_DIR="${CANDOR_OUT_DIR:-/scratch1/wulily/betterup_raw_media/outputs/candor_full_processed_resumable}"
VALIDATION_DIR="${VALIDATION_DIR:-$CANDOR_OUT_DIR/intensity_edge_validation}"
OUTCOME_QC_ROOT="${OUTCOME_QC_ROOT:-$CANDOR_OUT_DIR/outcome_qc}"

cd "$PROJECT_DIR"
mkdir -p logs "$VALIDATION_DIR" "$OUTCOME_QC_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    module load conda
  fi
  eval "$(conda shell.bash hook)"
  conda activate cvs-conversation
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c "import numpy, pandas, scipy; print('Python:', __import__('sys').executable)"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ACOUSTIC_TURNS="$CANDOR_OUT_DIR/acoustic_turns_backbiter.csv"
PARTICIPANT_SURVEY="$CANDOR_OUT_DIR/candor_survey_participant.csv"

for f in "$ACOUSTIC_TURNS" "$PARTICIPANT_SURVEY"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required input not found: $f" >&2
    echo "       candor_survey_participant.csv comes from build_candor_survey_outcomes.py" >&2
    exit 1
  fi
done

echo "=== [1/2] predictor side: intensity-edge construct validation ==="
NULL_ARGS=()
if [[ "${RESTART_NULL:-0}" == "1" ]]; then
  NULL_ARGS=(--restart-null)
fi
"$PYTHON_BIN" scripts/models/validate_candor_intensity_edge.py \
  --turns   "$ACOUSTIC_TURNS" \
  --out-dir "$VALIDATION_DIR" \
  --n-permutations      "${N_PERMUTATIONS:-1000}" \
  --seed "${SEED:-42}" \
  "${NULL_ARGS[@]}"

VOCAL_RELIABILITY="$VALIDATION_DIR/intensity_edge_split_half_reliability.csv"

echo
echo "=== [2/2] outcome side: survey reliability, ceiling, partner ICC ==="
SCALE_ARGS=()
if [[ -n "${SURVEY_SCALE_MAX:-}" ]]; then
  SCALE_ARGS=(--scale-max "$SURVEY_SCALE_MAX")
fi
if [[ -f "$VOCAL_RELIABILITY" ]]; then
  "$PYTHON_BIN" scripts/models/qc_candor_outcome_reliability.py \
    --participant "$PARTICIPANT_SURVEY" \
    --out-root    "$OUTCOME_QC_ROOT" \
    --vocal-reliability-csv "$VOCAL_RELIABILITY" \
    "${SCALE_ARGS[@]}"
else
  echo "[warn] $VOCAL_RELIABILITY missing; reporting outcome-only ceiling" >&2
  "$PYTHON_BIN" scripts/models/qc_candor_outcome_reliability.py \
    --participant "$PARTICIPANT_SURVEY" \
    --out-root    "$OUTCOME_QC_ROOT" \
    "${SCALE_ARGS[@]}"
fi

echo
echo "=== QC complete. Read these two reports before spending on annotation: ==="
echo "  $VALIDATION_DIR/intensity_edge_measurement_report.md"
ls -1d "$OUTCOME_QC_ROOT"/outcome_qc_*/qc_report.md | sort | tail -1

#!/bin/bash
#SBATCH --job-name=topic_drift
#SBATCH --account=elisab_1216
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/topic_drift_%j.out
#SBATCH --error=logs/topic_drift_%j.err
#
# Topic-drift analysis (CPU only; sentence-embedding models are small).
# Requires: sentence-transformers, scikit-learn (for k-means topic clustering).
# Runs, in order:
#   1. human data, main embedding model + TF-IDF robustness
#   2. compare for every simulation run dir given in RUNS (space-separated),
#      main model + TF-IDF robustness
#
# Inputs : 04_data/acoustic_turns.csv, 04_data/outcomes.csv, the run dirs in RUNS
# Outputs: 05_analysis_outputs/llm_sim_partner/drift_* (one dir per analysis)
#          Per run: summary.md, windows/segments/items CSVs, walk_trajectory.png
#
# Examples:
#   sbatch --export=ALL,RUNS="05_analysis_outputs/llm_sim_partner/run_X_topic_jobN 05_analysis_outputs/llm_sim_partner/run_Y_turn_jobM" \
#          scripts/carc/topic_drift.sbatch
#   sbatch --export=ALL,SKIP_HUMAN=1,RUNS="<run dir>" scripts/carc/topic_drift.sbatch
#   # robustness: second model, wider window
#   sbatch --export=ALL,MODEL=sentence-transformers/all-mpnet-base-v2,WINDOW=5,RUNS="<run dirs>" \
#          scripts/carc/topic_drift.sbatch

source /etc/profile
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch1/$USER/cvs_conversation}"
MODEL="${MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
WINDOW="${WINDOW:-3}"
RUNS="${RUNS:-}"
SKIP_HUMAN="${SKIP_HUMAN:-0}"
export HF_HOME="${HF_HOME:-/scratch1/$USER/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "$PROJECT_DIR"
mkdir -p logs
module purge
module load conda
eval "$(conda shell.bash hook)"
# A conda env active in the submitting shell is inherited (sbatch --export=ALL) and makes
# "conda activate" a no-op after module purge, leaving the base python first on PATH.
# Clear any inherited activation, activate fresh, and verify the interpreter.
for _ in 1 2 3 4; do conda deactivate 2> /dev/null || true; done
conda activate cvs-vllm
case "$(command -v python)" in
  */envs/cvs-vllm/*) ;;
  *) echo "python is $(command -v python), not the cvs-vllm env; aborting" >&2; exit 1 ;;
esac
# keep BLAS/OpenMP thread pools to the cores Slurm gave us
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
echo "python: $(which python)"
python -c 'import numpy, pandas, sentence_transformers, sklearn; print("env ok: numpy", numpy.__version__, "sklearn", sklearn.__version__)' || { echo "cvs-vllm environment is broken; see the error above" >&2; exit 1; }

PY=(python scripts/llm_simulation/topic_drift.py)
ST=(--backend st --model "$MODEL" --device cpu --window "$WINDOW")
TF=(--backend tfidf --window "$WINDOW")

if [[ "$SKIP_HUMAN" != "1" ]]; then
  "${PY[@]}" human "${ST[@]}"
  "${PY[@]}" human "${TF[@]}"
fi
for r in $RUNS; do
  [[ -f "$r/actor.jsonl" ]] || { echo "skip $r: no actor.jsonl" >&2; continue; }
  "${PY[@]}" compare --run "$r" "${ST[@]}"
  "${PY[@]}" compare --run "$r" "${TF[@]}"
done
echo "[$(date)] done; results in 05_analysis_outputs/llm_sim_partner/drift_*"

#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-environment.yml}"
ENV_NAME="${CONDA_ENV_NAME:-cvs-conversation}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda or Anaconda, then re-run this script."
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo "Environment ready. Activate it with: conda activate $ENV_NAME"

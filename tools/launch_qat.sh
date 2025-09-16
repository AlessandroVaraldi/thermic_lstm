#!/usr/bin/env bash
[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_HOME="${CONDA_HOME:-$HOME/anaconda3}"
ENV_NAME="${ENV_NAME:-pytorch}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# scegli device automaticamente
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  DEVICE_OPT="--device cuda"
else
  DEVICE_OPT="--device cpu"
fi

"$CONDA_HOME/bin/conda" run --no-capture-output -n "$ENV_NAME" \
  python -u -m tools.qat      \
  $DEVICE_OPT                 \
  "$@"

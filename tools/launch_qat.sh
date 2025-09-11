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
  python -u -m tools.qat \
  --augment 2 --batch 16 --epochs 100 --lr 1e-3 \
  --amp 1 --amp-dtype bf16 --accum 2 \
  --workers 8 --pin-memory 1 --persist 1 --prefetch 4 \
  --val-interval 1 --val-max-batches 0 \
  --ckpt 0 --tbptt-k 0 --compile 0 --fused-adam 1 \
  --mp-time 1 --mp-tau-thr 0.08 --mp-scale-mul 1.5 --mp-rshift-delta -1 \
  $DEVICE_OPT \
  "$@"

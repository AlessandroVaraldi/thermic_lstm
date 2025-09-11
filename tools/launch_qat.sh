#!/usr/bin/env bash
set -euo pipefail

# Optional: attiva l'ambiente
# source ~/.bashrc
# conda activate torch-gpu  # se lo usi

# Torna alla root del progetto anche se lanciato da altrove
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Memoria CUDA più robusta
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u tools/qat.py \
  --augment 2 --batch 16 --epochs 100 --lr 1e-3 \
  --amp 1 --amp-dtype bf16 --accum 2 \
  --workers 8 --pin-memory 1 --persist 1 --prefetch 4 \
  --val-interval 1 --val-max-batches 0 \
  --ckpt 0 --tbptt-k 0 --compile 0 --fused-adam 1 \
  --mp-time 1 --mp-tau-thr 0.08 --mp-scale-mul 1.5 --mp-rshift-delta -1 \
  "$@"

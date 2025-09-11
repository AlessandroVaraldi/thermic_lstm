#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python qlstm_train.py \
  --augment 2 --batch 16 \
  --amp 1 --amp-dtype bf16 \
  --accum 1 \
  --workers 8 --pin-memory 1 --persist 1 --prefetch 4 \
  --val-interval 1 --val-max-batches 0 \
  --ckpt 0 --tbptt-k 0 \
  --fused-adam 1 --compile 0 \
  --mp-time 1 --mp-tau-thr 0.08 --mp-scale-mul 1.5 --mp-rshift-delta -1 \
  "$@" | tee "logs/qat_$(date +%F_%H%M).log"

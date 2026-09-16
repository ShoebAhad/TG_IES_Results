#!/usr/bin/env bash
# Secondary cell: DenseNet-121, CIFAR-10, SGD(exp), seed 0, all 4 methods.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"

run() {
  local method=$1
  local name="${method}_cifar10_densenet121_SGD_exponential_learning_rate_seed0"
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name"; return 0
  fi
  echo "=== RUN: $name $(date) ==="
  python train.py --method "$method" --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model densenet121 \
    --optimizer SGD_exponential_learning_rate --seed 0 --out_dir "$OUT" \
    --recalib_every 30 \
    > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

for m in baseline ies_shipped ies_alg1 tgies; do run "$m"; done
echo "RUN_DENSENET_ALL_DONE $(date)"

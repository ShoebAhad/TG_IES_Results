#!/usr/bin/env bash
# Secondary cell: ResNet-18, CIFAR-10, Adam, seed 0, all 4 methods.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"

run() {
  local method=$1
  local name="${method}_cifar10_resnet18_Adam_seed0"
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name"; return 0
  fi
  echo "=== RUN: $name $(date) ==="
  python train.py --method "$method" --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer Adam --seed 0 --out_dir "$OUT" \
    --recalib_every 30 \
    > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

for m in baseline ies_shipped ies_alg1 tgies; do run "$m"; done
echo "RUN_ADAM_ALL_DONE $(date)"

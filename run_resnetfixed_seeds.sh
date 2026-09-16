#!/usr/bin/env bash
# Seed-average the ResNet-18/fixed-LR secondary cell: seeds 1,2, all 4 methods.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"

run() {
  local method=$1 seed=$2
  local name="${method}_cifar10_resnet18_SGD_fixed_learning_rate_seed${seed}"
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name"; return 0
  fi
  echo "=== RUN: $name $(date) ==="
  python train.py --method "$method" --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_fixed_learning_rate --seed "$seed" --out_dir "$OUT" \
    --recalib_every 30 \
    > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

for seed in 1 2; do
  for m in baseline ies_shipped ies_alg1 tgies; do run "$m" "$seed"; done
done
echo "RUN_RESNETFIXEDSEEDS_ALL_DONE $(date)"

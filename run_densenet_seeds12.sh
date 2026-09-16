#!/usr/bin/env bash
# NEXT_STEPS.md item 1 (other half): backfill DenseNet-121/CIFAR-10/ResNet-18-matched
# SGD-exponential recipe to 3 seeds (seed 0 already complete) by adding seeds 1-2, 4 core
# methods. Concurrency is fine here: only wall-clock-timing numbers are corrupted by
# running jobs in parallel, and this table's metrics are accuracy/backprop-saved-pct.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"
MAXJOBS=3

run() {
  local method=$1 seed=$2
  local name="${method}_cifar10_densenet121_SGD_exponential_learning_rate_seed${seed}"
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name"
    return 0
  fi
  echo "=== LAUNCH: $name $(date) ==="
  python train.py --method "$method" --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 8 --model densenet121 \
    --optimizer SGD_exponential_learning_rate --seed "$seed" --out_dir "$OUT" \
    --recalib_every 30 \
    > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

throttle() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do
    wait -n
  done
}

echo "=== run_densenet_seeds12.sh START $(date) (MAXJOBS=$MAXJOBS) ==="
for seed in 1 2; do
  for m in baseline ies_shipped ies_alg1 tgies; do
    throttle
    run "$m" "$seed" &
  done
done
wait
echo "RUN_DENSENET_SEEDS12_ALL_DONE $(date)"

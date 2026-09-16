#!/usr/bin/env bash
# Scoped resume of NEXT_STEPS.md items 8 and 11 ONLY (12/13 explicitly deferred).
# Sequential (MAXJOBS=1): the shared GPU has only ~2.8GB free (an unrelated VLLM
# process holds the rest), and each run needs ~1.5GB, so no safe concurrency margin.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"

run() {
  local name=$1; shift
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name"
    return 0
  fi
  echo "=== LAUNCH: $name $(date) ==="
  python train.py "$@" --out_dir "$OUT" > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

echo "=== run_items_8_11.sh START $(date) ==="

# ---- Item 8: Selective Backprop, primary cell, 3 seeds ----
for seed in 0 1 2; do
  name="selective_backprop_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}"
  run "$name" --method selective_backprop --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed "$seed"
done
echo "=== ITEM_8_DONE $(date) ==="

# ---- Item 11: reactivation probe, primary-cell tgies (self-calibrated), 3 seeds ----
for seed in 0 1 2; do
  name="tgies_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}_reactprobe"
  run "$name" --method tgies --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed "$seed" --recalib_every 30 \
    --reactivation_probe_every 20 --reactivation_probe_n 300 --tag reactprobe
done
echo "=== ITEM_11_DONE $(date) ==="

echo "=== run_items_8_11.sh ALL_DONE $(date) ==="

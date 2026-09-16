#!/usr/bin/env bash
# Priority 3 (LR-schedule dependency): ResNet-18/CIFAR-10, all 4 methods, across LR
# schedules not yet covered by the existing matrix. "Constant" (SGD_fixed_learning_rate,
# lr=1e-3) and "Exponential" (SGD_exponential_learning_rate, lr0=0.1, gamma=0.96) already
# have full 3-seed data (tab:secondary, tab:primary) -- this adds Linear (schedule already
# implemented in common.py but never run), Step-decay, and Cosine-decay (both newly added
# to common.py 2026-08-27) to test whether TG-IES's savings advantage is specific to
# decaying-LR schedules in general or to the exponential schedule in particular.
# Single seed to start (skip-if-complete; extend to seeds 1-2 later if time allows).
# Strictly SEQUENTIAL (not throttled): the reviewer's ask explicitly wants wall-clock
# time as one of the reported columns for this comparison, which concurrent jobs would
# corrupt (unlike the earlier trimmed seed-extension pass, which didn't need clean timing).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"

run() {
  local name=$1; shift
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name $(date)"
    return 0
  fi
  echo "=== RUN: $name $(date) ==="
  python train.py "$@" --out_dir "$OUT" > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

echo "=== run_lr_sweep.sh START $(date) ==="

for sched in SGD_linear_learning_rate SGD_step_learning_rate SGD_cosine_learning_rate; do
  echo "########## $sched ##########"
  for m in baseline ies_shipped ies_alg1 tgies; do
    name="${m}_cifar10_resnet18_${sched}_seed0"
    run "$name" --method "$m" --dataset cifar10 --root_dir cifar-10 \
      --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
      --optimizer "$sched" --seed 0 --recalib_every 30
  done
done

echo "RUN_LR_SWEEP_ALL_DONE $(date)"

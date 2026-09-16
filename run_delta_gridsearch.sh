#!/usr/bin/env bash
# Naive delta grid search, no theory: primary cell, K*=1 fixed (matching the
# "delta only" ablation row's setting), seed 0, to test whether an untheorized
# line search over delta lands on savings/accuracy comparable to the
# theory-calibrated delta=0.0077 (already have that point + delta=0.001 from
# the existing ablation table -- this fills in the rest of the curve).
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

echo "=== run_delta_gridsearch.sh START $(date) ==="
for delta in 0.003 0.005 0.015 0.03 0.05; do
  name="tgies_cifar10_resnet18_SGD_exponential_learning_rate_seed0_deltagrid${delta}"
  run "$name" --method tgies --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed 0 \
    --tgies_fix_delta "$delta" --tgies_fix_kstar 1 --tag "deltagrid${delta}"
done
echo "=== run_delta_gridsearch.sh ALL_DONE $(date) ==="

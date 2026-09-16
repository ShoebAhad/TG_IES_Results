#!/usr/bin/env bash
# Ablation: isolate delta-calibration vs kstar-calibration (2x2 grid; the other
# two corners are the existing ies_alg1 and full-tgies runs).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"

run() {
  local tag=$1; shift
  local name="tgies_cifar10_resnet18_SGD_exponential_learning_rate_seed0_${tag}"
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name"; return 0
  fi
  echo "=== RUN: $name $(date) ==="
  python train.py --method tgies --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed 0 --out_dir "$OUT" \
    --recalib_every 30 --tag "$tag" "$@" \
    > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

run "ablation_deltaonly" --tgies_fix_kstar 1
run "ablation_kstaronly" --tgies_fix_delta 0.001

echo "RUN_ABLATION_ALL_DONE $(date)"

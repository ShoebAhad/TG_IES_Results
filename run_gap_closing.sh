#!/usr/bin/env bash
# Closes explicitly-flagged single-seed empirical-scope gaps (paper Sec. "Empirical
# Scope" / "Future work" item, and Sec. 6.6 "we do not yet have 3-seed data for the
# new baselines on VGG-16/CIFAR-100"):
#   1. DenseNet-121/CIFAR-10/SGD-exp: seeds 1,2 for all 4 core methods (was
#      deliberately deferred as single-seed in run_densenet.sh).
#   2. InfoBatch / BLS-InfoBatch / AlignPrune on VGG-16/CIFAR-10 and
#      CIFAR-100/ResNet-18: seeds 1,2 (currently only seed 0 exists for these).
# Skip-if-complete, safe to re-run/resume. Strictly sequential (one train.py at a
# time) to keep wall-clock timing measurements clean, matching this repo's other
# single-job drivers rather than run_new_baselines_and_sensitivity.sh's MAXJOBS=3.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"

run() {
  local name=$1; shift
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name $(date)"; return 0
  fi
  echo "=== RUN: $name $(date) ==="
  python train.py "$@" --out_dir "$OUT" > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

echo "=== run_gap_closing.sh START $(date) ==="

echo "########## DenseNet-121, CIFAR-10, SGD(exp), seeds 1,2 ##########"
for seed in 1 2; do
  for m in baseline ies_shipped ies_alg1 tgies; do
    name="${m}_cifar10_densenet121_SGD_exponential_learning_rate_seed${seed}"
    run "$name" --method "$m" --dataset cifar10 --root_dir cifar-10 \
      --batch_size 64 --epochs 200 --num_workers 4 --model densenet121 \
      --optimizer SGD_exponential_learning_rate --seed "$seed" --recalib_every 30
  done
done
echo "DENSENET_SEEDS_DONE $(date)"

echo "########## New baselines, VGG-16/CIFAR-10/SGD-exp, seeds 1,2 ##########"
for seed in 1 2; do
  for method in infobatch bls_infobatch alignprune; do
    name="${method}_cifar10_vgg16_SGD_exponential_learning_rate_seed${seed}"
    run "$name" --method "$method" --dataset cifar10 --root_dir cifar-10 \
      --batch_size 64 --epochs 200 --num_workers 4 --model vgg16 \
      --optimizer SGD_exponential_learning_rate --seed "$seed"
  done
done
echo "NEWBASELINES_VGG16_SEEDS_DONE $(date)"

echo "########## New baselines, CIFAR-100/ResNet-18/SGD-exp, seeds 1,2 ##########"
for seed in 1 2; do
  for method in infobatch bls_infobatch alignprune; do
    name="${method}_cifar100_resnet18_SGD_exponential_learning_rate_seed${seed}"
    run "$name" --method "$method" --dataset cifar100 --root_dir cifar-100 \
      --batch_size 64 --epochs 200 --num_workers 8 --model resnet18 \
      --optimizer SGD_exponential_learning_rate --seed "$seed"
  done
done
echo "NEWBASELINES_CIFAR100_SEEDS_DONE $(date)"

echo "RUN_GAP_CLOSING_ALL_DONE $(date)"

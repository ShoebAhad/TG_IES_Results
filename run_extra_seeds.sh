#!/usr/bin/env bash
# Extends the Adam/ResNet-18 and CIFAR-100/ResNet-18/SGD-exp single-seed cells to 3
# seeds by running seeds 1 and 2 for all four methods, matching the exact
# hyperparameters used for each cell's existing seed 0 run (run_adam.sh /
# run_cifar100.sh). Skip-if-complete, so it is safe to re-run or resume after an
# interruption.
#
# DenseNet-121/SGD-exp is deliberately NOT included here: at ~35-52 min/run x 8 runs
# it is the slowest of the three single-seed cells and was deprioritized (kept
# single-seed, deferred to future work) rather than run alongside the other two.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"

run() {
  local method=$1 dataset=$2 root_dir=$3 model=$4 opt=$5 seed=$6 workers=$7
  local name="${method}_${dataset}_${model}_${opt}_seed${seed}"
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name $(date)"; return 0
  fi
  echo "=== RUN: $name $(date) ==="
  python train.py --method "$method" --dataset "$dataset" --root_dir "$root_dir" \
    --batch_size 64 --epochs 200 --num_workers "$workers" --model "$model" \
    --optimizer "$opt" --seed "$seed" --out_dir "$OUT" \
    --recalib_every 30 \
    > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

echo "########## ResNet-18, CIFAR-10, Adam, seeds 1,2 ##########"
for seed in 1 2; do
  for m in baseline ies_shipped ies_alg1 tgies; do
    run "$m" cifar10 cifar-10 resnet18 Adam "$seed" 4
  done
done

echo "########## ResNet-18, CIFAR-100, SGD(exp), seeds 1,2 ##########"
for seed in 1 2; do
  for m in baseline ies_shipped ies_alg1 tgies; do
    run "$m" cifar100 cifar-100 resnet18 SGD_exponential_learning_rate "$seed" 8
  done
done

echo "RUN_EXTRA_SEEDS_ALL_DONE $(date)"

#!/usr/bin/env bash
# Adds RHO-LOSS (Mindermann et al. 2022) as a fourth modern dynamic-pruning
# baseline alongside InfoBatch/BLS-InfoBatch/AlignPrune (Sec. "Comparison
# Against Modern Dynamic-Pruning Baselines"), closing Future-work item (4)
# ("RHO-LOSS and RL-style dynamic pruning... We view all four as next steps").
# Matches the SAME 3-seed x 3-setting matrix the other new baselines have
# after run_gap_closing.sh completes (primary ResNet-18/CIFAR-10, VGG-16/
# CIFAR-10, CIFAR-100/ResNet-18), for parity rather than a single-seed
# afterthought comparison.
#
# Stage 1 pretrains the frozen irreducible-loss model once per (dataset,
# model) pair (pretrain_il_model.py; ~40 short epochs on a 10% holdout split,
# cheap, shared across all 3 seeds of that pair -- skip-if-cached).
# Stage 2 runs the 9 main train.py --method rho_loss jobs (skip-if-complete).
# Strictly sequential, matching this repo's other single-job drivers.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs; ILDIR=il_cache
mkdir -p "$OUT" "$LOGDIR" "$ILDIR"

echo "=== run_rho_loss.sh START $(date) ==="

echo "########## Stage 1: pretrain irreducible-loss models ##########"
pretrain() {
  local dataset=$1 root_dir=$2 model=$3
  echo "=== IL PRETRAIN: ${dataset}_${model} $(date) ==="
  python pretrain_il_model.py --dataset "$dataset" --root_dir "$root_dir" --model "$model" \
    --out_dir "$ILDIR" --num_workers 4 \
    > "$LOGDIR/il_pretrain_${dataset}_${model}.log" 2>&1
  echo "=== IL DONE: ${dataset}_${model} exit=$? $(date) ==="
}
pretrain cifar10 cifar-10 resnet18
pretrain cifar10 cifar-10 vgg16
pretrain cifar100 cifar-100 resnet18
echo "IL_PRETRAIN_ALL_DONE $(date)"

run() {
  local name=$1; shift
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name $(date)"; return 0
  fi
  echo "=== RUN: $name $(date) ==="
  python train.py "$@" --out_dir "$OUT" --il_cache_dir "$ILDIR" > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

echo "########## Stage 2: rho_loss main runs ##########"
echo "---- primary cell: ResNet-18/CIFAR-10/SGD-exp, seeds 0-2 ----"
for seed in 0 1 2; do
  name="rho_loss_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}"
  run "$name" --method rho_loss --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed "$seed"
done

echo "---- VGG-16/CIFAR-10/SGD-exp, seeds 0-2 ----"
for seed in 0 1 2; do
  name="rho_loss_cifar10_vgg16_SGD_exponential_learning_rate_seed${seed}"
  run "$name" --method rho_loss --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model vgg16 \
    --optimizer SGD_exponential_learning_rate --seed "$seed"
done

echo "---- CIFAR-100/ResNet-18/SGD-exp, seeds 0-2 ----"
for seed in 0 1 2; do
  name="rho_loss_cifar100_resnet18_SGD_exponential_learning_rate_seed${seed}"
  run "$name" --method rho_loss --dataset cifar100 --root_dir cifar-100 \
    --batch_size 64 --epochs 200 --num_workers 8 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed "$seed"
done

echo "RUN_RHO_LOSS_ALL_DONE $(date)"

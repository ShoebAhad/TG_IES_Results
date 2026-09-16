#!/usr/bin/env bash
# Trimmed + parallelized replacement for the run_extra_seeds -> run_gap_closing ->
# run_rho_loss sequential chain (that chain estimated ~19 GPU-hours; this cuts
# scope and throttles concurrency to bring it down substantially). Kept vs cut:
#   KEEP (full 2-seed extension, top priority -- already in progress / interrupted
#     work): Adam seeds 1-2, CIFAR-100 seeds 1-2, all 4 core methods each.
#   TRIM (2 seeds instead of 3, i.e. seed 1 only added): InfoBatch/BLS-InfoBatch/
#     AlignPrune on VGG-16 and CIFAR-100 -- still improves on single-seed, cheaper
#     than the full 3-seed extension.
#   CUT: DenseNet-121 seeds 1-2 (the single biggest time cost, ~5-6h alone, for
#     the least-novel finding -- already an accepted, explicitly-documented
#     single-seed limitation in the paper).
#   TRIM: rho_loss to the primary cell only (ResNet-18/CIFAR-10, 3 seeds),
#     matching the ORIGINAL single-seed-elsewhere scope the other new baselines
#     shipped with, instead of the full 3-setting matrix.
# Runs up to MAXJOBS concurrently (same throttle pattern as
# run_new_baselines_and_sensitivity.sh, which already established this GPU
# handles a few CIFAR-scale jobs at once). Skip-if-complete throughout.
#
# NOTE: because jobs run concurrently, per-run wall-clock (epoch_time_s /
# total_time_s) for everything in this script is NOT reliable for timing
# comparisons -- these cells don't feed the paper's wall-clock-timing section
# anyway (that section only covers the already-completed primary/secondary
# cells), only the accuracy / backprop-saved-pct numbers, which are unaffected
# by concurrency.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs; ILDIR=il_cache
mkdir -p "$OUT" "$LOGDIR" "$ILDIR"
MAXJOBS=3

run() {
  local name=$1; shift
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name $(date)"
    return 0
  fi
  echo "=== LAUNCH: $name $(date) ==="
  python train.py "$@" --out_dir "$OUT" > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

throttle() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do
    wait -n
  done
}

echo "=== run_remaining_fast.sh START $(date) (MAXJOBS=$MAXJOBS) ==="

echo "########## KEEP: Adam seeds 1-2, all 4 methods ##########"
for seed in 1 2; do
  for m in baseline ies_shipped ies_alg1 tgies; do
    name="${m}_cifar10_resnet18_Adam_seed${seed}"
    throttle
    run "$name" --method "$m" --dataset cifar10 --root_dir cifar-10 \
      --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
      --optimizer Adam --seed "$seed" --recalib_every 30 &
  done
done
wait

echo "########## KEEP: CIFAR-100 seeds 1-2, all 4 methods ##########"
for seed in 1 2; do
  for m in baseline ies_shipped ies_alg1 tgies; do
    name="${m}_cifar100_resnet18_SGD_exponential_learning_rate_seed${seed}"
    throttle
    run "$name" --method "$m" --dataset cifar100 --root_dir cifar-100 \
      --batch_size 64 --epochs 200 --num_workers 8 --model resnet18 \
      --optimizer SGD_exponential_learning_rate --seed "$seed" --recalib_every 30 &
  done
done
wait
echo "KEEP_STAGE_DONE $(date)"

echo "########## TRIM: new baselines, VGG-16/CIFAR-10, seed 1 only ##########"
for method in infobatch bls_infobatch alignprune; do
  name="${method}_cifar10_vgg16_SGD_exponential_learning_rate_seed1"
  throttle
  run "$name" --method "$method" --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model vgg16 \
    --optimizer SGD_exponential_learning_rate --seed 1 &
done
wait

echo "########## TRIM: new baselines, CIFAR-100/ResNet-18, seed 1 only ##########"
for method in infobatch bls_infobatch alignprune; do
  name="${method}_cifar100_resnet18_SGD_exponential_learning_rate_seed1"
  throttle
  run "$name" --method "$method" --dataset cifar100 --root_dir cifar-100 \
    --batch_size 64 --epochs 200 --num_workers 8 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed 1 &
done
wait
echo "TRIM_STAGE_DONE $(date)"

echo "########## TRIM: rho_loss, primary cell only (ResNet-18/CIFAR-10, seeds 0-2) ##########"
if [ ! -f "$ILDIR/il_losses_cifar10_resnet18.npy" ]; then
  echo "=== IL PRETRAIN: cifar10_resnet18 $(date) ==="
  python pretrain_il_model.py --dataset cifar10 --root_dir cifar-10 --model resnet18 \
    --out_dir "$ILDIR" --num_workers 4 > "$LOGDIR/il_pretrain_cifar10_resnet18.log" 2>&1
  echo "=== IL DONE: cifar10_resnet18 exit=$? $(date) ==="
fi
for seed in 0 1 2; do
  name="rho_loss_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}"
  throttle
  run "$name" --method rho_loss --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed "$seed" --il_cache_dir "$ILDIR" &
done
wait

echo "RUN_REMAINING_FAST_ALL_DONE $(date)"

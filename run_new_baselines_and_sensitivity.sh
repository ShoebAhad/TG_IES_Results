#!/usr/bin/env bash
# New-baseline comparison matrix (InfoBatch / BLS-InfoBatch / AlignPrune) +
# gradient-norm-floor percentile sensitivity sweep (VGG-16 + ResNet-18 contrast).
# Runs up to MAXJOBS training jobs concurrently (GPU has ample VRAM/compute
# headroom for CIFAR-scale models; see run log for the utilization check).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"
MAXJOBS=3

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

throttle() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do
    wait -n
  done
}

echo "=== run_new_baselines_and_sensitivity.sh START $(date) (MAXJOBS=$MAXJOBS) ==="

# ---- New-baseline matrix: primary cell (ResNet-18/CIFAR-10/SGD-exp, seeds 0-2) ----
for method in infobatch bls_infobatch alignprune; do
  for seed in 0 1 2; do
    name="${method}_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}"
    throttle
    run "$name" --method "$method" --dataset cifar10 --root_dir cifar-10 \
      --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
      --optimizer SGD_exponential_learning_rate --seed "$seed" &
  done
done
wait

# ---- New-baseline matrix: VGG-16/CIFAR-10/SGD-exp seed0 (TG-IES failure case) ----
for method in infobatch bls_infobatch alignprune; do
  name="${method}_cifar10_vgg16_SGD_exponential_learning_rate_seed0"
  throttle
  run "$name" --method "$method" --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model vgg16 \
    --optimizer SGD_exponential_learning_rate --seed 0 &
done
wait

# ---- New-baseline matrix: CIFAR-100/ResNet-18/SGD-exp seed0 ----
for method in infobatch bls_infobatch alignprune; do
  name="${method}_cifar100_resnet18_SGD_exponential_learning_rate_seed0"
  throttle
  run "$name" --method "$method" --dataset cifar100 --root_dir cifar-100 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed 0 &
done
wait

echo "=== NEW_BASELINES_ALL_DONE $(date) ==="

# ---- Gradient-norm-floor percentile sensitivity sweep ----
for pct in 5 10 20 35 50 80; do
  for model in vgg16 resnet18; do
    name="tgies_cifar10_${model}_SGD_exponential_learning_rate_seed0_epspct${pct}"
    throttle
    run "$name" --method tgies --dataset cifar10 --root_dir cifar-10 \
      --batch_size 64 --epochs 200 --num_workers 4 --model "$model" \
      --optimizer SGD_exponential_learning_rate --seed 0 --recalib_every 30 \
      --eps_percentile "$pct" --tag "epspct${pct}" &
  done
done
wait

echo "=== SENSITIVITY_SWEEP_ALL_DONE $(date) ==="
echo "=== run_new_baselines_and_sensitivity.sh ALL_DONE $(date) ==="

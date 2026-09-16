#!/usr/bin/env bash
# NEXT_STEPS.md items 8, 11, 12, 13 -- the remaining GPU-budget items after item 7.
#
# Item 8  [#11]  Selective Backprop baseline, primary cell, 3 seeds (3 runs).
# Item 11 [#23]  Reactivation probe: primary-cell tgies reruns (self-calibrated K*, NOT
#                --tgies_fix_kstar) with --reactivation_probe_every, tagged _reactprobe
#                so they don't collide with the existing headline primary-cell results
#                (which lack the probe instrumentation). 3 runs.
# Item 12 [#22]  Class imbalance, one ratio (10:1), CIFAR-100, 4 core methods, 3 seeds
#                (12 runs).
# Item 13 [#21]  Noisy labels, two levels (20%, 40%), CIFAR-10/ResNet-18, 4 core
#                methods, 3 seeds (24 runs).
#
# batch_size_sweep.py (item 10) is NOT included here -- it needs an uncontended GPU
# for its nvidia-smi memory/utilization/power polling to mean anything, so it must run
# alone, before or after this throttled batch, never concurrently with it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"
MAXJOBS=6

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

echo "=== run_remaining_items.sh START $(date) (MAXJOBS=$MAXJOBS) ==="

# ---- Item 8: Selective Backprop, primary cell, 3 seeds ----
for seed in 0 1 2; do
  name="selective_backprop_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}"
  throttle
  run "$name" --method selective_backprop --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed "$seed" &
done

# ---- Item 11: reactivation probe, primary-cell tgies (self-calibrated), 3 seeds ----
for seed in 0 1 2; do
  name="tgies_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}_reactprobe"
  throttle
  run "$name" --method tgies --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed "$seed" --recalib_every 30 \
    --reactivation_probe_every 20 --reactivation_probe_n 300 --tag reactprobe &
done

echo "=== ITEM_8_11_LAUNCHED $(date) ==="
wait
echo "=== ITEM_8_11_DONE $(date) ==="

# ---- Item 12: class imbalance 10:1, CIFAR-100, 4 methods, 3 seeds ----
for method in baseline ies_shipped ies_alg1 tgies; do
  for seed in 0 1 2; do
    name="${method}_cifar100_resnet18_SGD_exponential_learning_rate_seed${seed}_imbal10"
    throttle
    run "$name" --method "$method" --dataset cifar100 --root_dir cifar-100 \
      --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
      --optimizer SGD_exponential_learning_rate --seed "$seed" --recalib_every 30 \
      --imbalance_ratio 10.0 --tag imbal10 &
  done
done
wait
echo "=== ITEM_12_DONE $(date) ==="

# ---- Item 13: noisy labels 20%/40%, CIFAR-10/ResNet-18, 4 methods, 3 seeds ----
for noise_pct in 20 40; do
  noise_frac=$(python3 -c "print($noise_pct/100)")
  for method in baseline ies_shipped ies_alg1 tgies; do
    for seed in 0 1 2; do
      name="${method}_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}_noise${noise_pct}"
      throttle
      run "$name" --method "$method" --dataset cifar10 --root_dir cifar-10 \
        --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
        --optimizer SGD_exponential_learning_rate --seed "$seed" --recalib_every 30 \
        --label_noise "$noise_frac" --tag "noise${noise_pct}" &
    done
  done
done
wait
echo "=== ITEM_13_DONE $(date) ==="

echo "=== run_remaining_items.sh ALL_DONE $(date) ==="

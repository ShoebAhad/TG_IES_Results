#!/usr/bin/env bash
# NEXT_STEPS.md items 12 (class imbalance) and 13 (label noise) -- both scoped and
# smoke-tested in an earlier session (2026-09-03) but never launched (user paused
# before this stage). Launching now since GPU time is no longer the binding
# constraint. Item 12: CIFAR-100, 4 core methods, 3 seeds, imbalance_ratio=10.
# Item 13: CIFAR-10/ResNet-18, 4 core methods, 3 seeds, label_noise in {0.2, 0.4}.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/milgpt/AI_Research/anaconda3/etc/profile.d/conda.sh
conda activate research
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"
MAXJOBS=4

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

echo "=== run_items_12_13.sh START $(date) ==="

# ---- Item 12: class imbalance, CIFAR-100, 4 methods, 3 seeds, ratio=10 ----
for method in baseline ies_shipped ies_alg1 tgies; do
  for seed in 0 1 2; do
    name="${method}_cifar100_resnet18_SGD_exponential_learning_rate_seed${seed}_imbalance10"
    run "$name" --method "$method" --dataset cifar100 --root_dir cifar-100 \
      --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
      --optimizer SGD_exponential_learning_rate --seed "$seed" --recalib_every 30 \
      --imbalance_ratio 10 --tag imbalance10 &
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
  done
done
wait
echo "=== ITEM_12_DONE $(date) ==="

# ---- Item 13: label noise, CIFAR-10/ResNet-18, 4 methods, 3 seeds, 2 levels ----
for rate in 0.2 0.4; do
  tag="noise$(python3 -c "print(int(${rate}*100))")"
  for method in baseline ies_shipped ies_alg1 tgies; do
    for seed in 0 1 2; do
      name="${method}_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}_${tag}"
      run "$name" --method "$method" --dataset cifar10 --root_dir cifar-10 \
        --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
        --optimizer SGD_exponential_learning_rate --seed "$seed" --recalib_every 30 \
        --label_noise "$rate" --tag "$tag" &
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
    done
  done
done
wait
echo "=== ITEM_13_DONE $(date) ==="

echo "=== run_items_12_13.sh ALL_DONE $(date) ==="

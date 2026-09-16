#!/usr/bin/env bash
# Secondary confirmatory cell: CIFAR-100, resnet18, SGD(exp), seed 0, all 4 methods.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate

OUT=results
LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"

run() {
  local method=$1 model=$2 opt=$3 seed=$4
  local name="${method}_cifar100_${model}_${opt}_seed${seed}"
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name"
    return 0
  fi
  echo "=== RUN: method=$method model=$model opt=$opt seed=$seed $(date) ==="
  python train.py --method "$method" --dataset cifar100 --root_dir cifar-100 \
    --batch_size 64 --epochs 200 --num_workers 8 --model "$model" \
    --optimizer "$opt" --seed "$seed" --out_dir "$OUT" \
    --recalib_every 30 \
    > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

echo "########## CIFAR-100 secondary cell: resnet18 / SGD(E) / seed 0 ##########"
for m in baseline ies_shipped ies_alg1 tgies; do
  run "$m" resnet18 SGD_exponential_learning_rate 0
done

echo "RUN_CIFAR100_ALL_DONE $(date)"

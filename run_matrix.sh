#!/usr/bin/env bash
# Full comparison matrix: baseline / ies_shipped / ies_alg1 / tgies
# Primary: ResNet-18, CIFAR-10, SGD(E), 200 epochs, seeds 0,1,2   (12 runs)
# Secondary A: VGG-16,   CIFAR-10, SGD(E), 200 epochs, seed 0     (4 runs)
# Secondary B: ResNet-18, CIFAR-10, SGD(F), 200 epochs, seed 0    (4 runs)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate

OUT=results
LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR"

run() {
  local method=$1 model=$2 opt=$3 seed=$4 tag=$5
  local name="${method}_cifar10_${model}_${opt}_seed${seed}${tag:+_$tag}"
  local csv="$OUT/${name}.csv"
  # resume support: a run is "done" once its CSV has 201 lines (header + 200 epochs)
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (already complete): $name"
    return 0
  fi
  echo "=== RUN: method=$method model=$model opt=$opt seed=$seed tag=$tag $(date) ==="
  python train.py --method "$method" --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 8 --model "$model" \
    --optimizer "$opt" --seed "$seed" --out_dir "$OUT" \
    --recalib_every 30 ${tag:+--tag "$tag"} \
    > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date) ==="
}

METHODS="baseline ies_shipped ies_alg1 tgies"

echo "########## PRIMARY: resnet18 / SGD(E) / seeds 0,1,2 ##########"
for seed in 0 1 2; do
  for m in $METHODS; do
    run "$m" resnet18 SGD_exponential_learning_rate "$seed" ""
  done
done

echo "########## SECONDARY A: vgg16 / SGD(E) / seed 0 ##########"
for m in $METHODS; do
  run "$m" vgg16 SGD_exponential_learning_rate 0 ""
done

echo "########## SECONDARY B: resnet18 / SGD(F) / seed 0 ##########"
for m in $METHODS; do
  run "$m" resnet18 SGD_fixed_learning_rate 0 ""
done

echo "RUN_MATRIX_ALL_DONE $(date)"

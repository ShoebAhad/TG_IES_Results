#!/usr/bin/env bash
# NEXT_STEPS.md item 9 [#5,#6] (scoped): CIFAR-native ViT-Tiny transformer
# experiment. 4 core methods x {CIFAR-10, CIFAR-100} x 3 seeds = 24 runs.
#
# This is the CIFAR-scale version only. The TinyImageNet/ImageNet matrix in
# NEXT_STEPS.md section A (~1350-1450 GPU-hours) is explicitly NOT in scope and
# train.py cannot run it anyway (--dataset accepts only cifar10/cifar100).
#
# num_workers is held at 4 for EVERY run in this cell: it changes the
# augmentation RNG stream, so varying it across methods would break the
# same-seed pairing that the paper's paired statistics depend on.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs
mkdir -p "$OUT" "$LOGDIR" checkpoints
MAXJOBS=${MAXJOBS:-3}
WORKERS=4

run() {
  local method=$1 dataset=$2 seed=$3
  local root; [ "$dataset" = "cifar10" ] && root=cifar-10 || root=cifar-100
  local name="${method}_${dataset}_vit_tiny_cifar_AdamW_warmup_cosine_seed${seed}"
  local csv="$OUT/${name}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 201 ]; then
    echo "SKIP (complete): $name"; return 0
  fi
  echo "=== LAUNCH: $name $(date -u +%FT%TZ) ==="
  python -u train.py --method "$method" --dataset "$dataset" --root_dir "$root" \
    --model vit_tiny_cifar --optimizer AdamW_warmup_cosine \
    --epochs 200 --warmup_epochs 10 --batch_size 64 --num_workers "$WORKERS" \
    --seed "$seed" --out_dir "$OUT" --recalib_every 30 --ckpt_every 10 \
    > "$LOGDIR/${name}.log" 2>&1
  echo "=== DONE: $name exit=$? $(date -u +%FT%TZ) ==="
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done; }

echo "=== run_vit.sh START $(date -u +%FT%TZ) (MAXJOBS=$MAXJOBS WORKERS=$WORKERS) ==="
# baseline first within each seed so a bad baseline surfaces early
for seed in 0 1 2; do
  for dataset in cifar10 cifar100; do
    for m in baseline ies_shipped ies_alg1 tgies; do
      throttle
      run "$m" "$dataset" "$seed" &
    done
  done
done
wait
echo "RUN_VIT_ALL_DONE $(date -u +%FT%TZ)"

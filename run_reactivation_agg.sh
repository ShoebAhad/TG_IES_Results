#!/usr/bin/env bash
# P0 items #2/#6: rerun the reactivation probe (primary cell, 3 seeds) with the new
# agg_grad_norm instrumentation (||mean_i g_i|| vs mean_i ||g_i||) added to train.py's
# --reactivation_probe_every path. Tagged "reactprobe2" (not "reactprobe") so the
# existing 2026-09-06 per-instance-violation results are preserved untouched; this run
# adds the aggregate/population-level check on top, does not replace it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/milgpt/AI_Research/anaconda3/etc/profile.d/conda.sh
conda activate research
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

echo "=== run_reactivation_agg.sh START $(date) ==="

for seed in 0 1 2; do
  name="tgies_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}_reactprobe2"
  run "$name" --method tgies --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed "$seed" --recalib_every 30 \
    --reactivation_probe_every 20 --reactivation_probe_n 300 --tag reactprobe2 &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
done
wait

echo "=== run_reactivation_agg.sh ALL_DONE $(date) ==="

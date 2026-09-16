#!/usr/bin/env bash
# NEXT_STEPS.md item 7 [#20]: K*-sweep + trajectory-divergence instrumentation.
#
# Phase 1: rerun the primary-cell baseline (3 seeds) with --dump_traj_every, writing
# flattened-parameter snapshots to traj_ref/ every 20 epochs. Tagged "traj" so these
# reruns don't collide with (or get skipped in favor of) the existing untagged baseline
# CSVs already used elsewhere in the paper.
#
# Phase 2: sweep K* in {1,2,4,8,15,30} (--tgies_fix_kstar) x 3 seeds = 18 runs, each also
# dumping traj snapshots and, since method != baseline, comparing against Phase 1's
# reference to log results/traj_dist_<run_name>.csv (L2 param distance from the shared
# full-data trajectory). K=15 already has an (uninstrumented) accuracy/savings data point
# from the primary cell's self-calibrated tgies runs (K* saturates at max_kstar=15 in
# every run in this paper, per the rounding-direction remark in sec:practical) but is
# rerun here anyway so its trajectory-divergence curve exists too.
#
# Must run Phase 1 to completion before Phase 2 starts: Phase 2 jobs read snapshot files
# Phase 1 writes, at matching epochs, so a Phase 2 job outrunning an incomplete Phase 1
# reference would silently skip comparisons (logged as a warning, not a crash -- see
# train.py) rather than produce wrong numbers, but we avoid it entirely with the `wait`
# between phases.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
OUT=results; LOGDIR=logs; TRAJREF=traj_ref
mkdir -p "$OUT" "$LOGDIR" "$TRAJREF"
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

echo "=== run_kstar_trajectory.sh START $(date) (MAXJOBS=$MAXJOBS) ==="

# ---- Phase 1: baseline + trajectory reference snapshots (primary cell, 3 seeds) ----
for seed in 0 1 2; do
  name="baseline_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}_traj"
  run "$name" --method baseline --dataset cifar10 --root_dir cifar-10 \
    --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
    --optimizer SGD_exponential_learning_rate --seed "$seed" --tag traj \
    --dump_traj_every 20 --traj_ref_dir "$TRAJREF" &
done
wait

echo "=== PHASE1_BASELINE_TRAJ_DONE $(date) ==="

# ---- Phase 2: K*-sweep, primary cell, 3 seeds, trajectory distance vs Phase 1 ----
for kstar in 1 2 4 8 15 30; do
  for seed in 0 1 2; do
    name="tgies_cifar10_resnet18_SGD_exponential_learning_rate_seed${seed}_kstar${kstar}"
    throttle
    run "$name" --method tgies --dataset cifar10 --root_dir cifar-10 \
      --batch_size 64 --epochs 200 --num_workers 4 --model resnet18 \
      --optimizer SGD_exponential_learning_rate --seed "$seed" --recalib_every 30 \
      --tgies_fix_kstar "$kstar" --tag "kstar${kstar}" \
      --dump_traj_every 20 --traj_ref_dir "$TRAJREF" &
  done
done
wait

echo "=== PHASE2_KSTAR_SWEEP_DONE $(date) ==="
echo "=== run_kstar_trajectory.sh ALL_DONE $(date) ==="

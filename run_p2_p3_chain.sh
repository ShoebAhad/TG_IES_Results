#!/usr/bin/env bash
# Resumable P2/P3 chain: finishes run_lr_sweep.sh (P3), then runs the three
# drift_term_run.py instrumentation passes (P2: exponential/fixed/Adam, matching
# sec:lrmechanism's existing three schedules). Meant to be launched detached
# (nohup ... &, disown) so it survives the launching shell/session ending --
# this chain has died mid-run twice before with the GPU idle and no error,
# consistent with being killed on session exit rather than a real crash.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== run_p2_p3_chain.sh START $(date) ===" >> run_p2_p3_driver.log

bash run_lr_sweep.sh >> run_p2_p3_driver.log 2>&1

source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
for sched in SGD_exponential_learning_rate SGD_fixed_learning_rate Adam; do
  echo "=== RUN: drift_term_run.py --schedule $sched $(date) ===" >> run_p2_p3_driver.log
  python drift_term_run.py --schedule "$sched" --root_dir cifar-10 --model resnet18 \
    --seed 0 --epochs 200 --subset_size 300 --out_dir results \
    > "logs/drift_${sched}.log" 2>&1
  echo "=== DONE: drift_term_run.py --schedule $sched exit=$? $(date) ===" >> run_p2_p3_driver.log
done

echo "P2_P3_CHAIN_ALL_DONE $(date)" >> run_p2_p3_driver.log

#!/usr/bin/env bash
# NEXT_STEPS.md item 2 remainder (item #17): rerun the 3 drift_term_run.py passes now
# that the script also saves ||g_i^(t)|| (drift_<tag>_norms.npy), needed for the
# ||g_i|| vs |Delta^2 L_i| calibration-bound plot the reviewer asked for. Overwrites
# the existing drift_<tag>_{losses,inner,subset_idx,eta}.npy with identical data (same
# seed/schedule/subset selection) plus the new norms file. Strictly sequential, same as
# the original P2/P3 chain (drift_term_run.py has no mid-run checkpoint).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate

echo "=== run_drift_norms.sh START $(date) ==="
for sched in SGD_exponential_learning_rate SGD_fixed_learning_rate Adam; do
  tag=$(echo "$sched" | sed -e 's/^SGD_//' -e 's/_learning_rate$//')
  norms_file="results/drift_${tag}_norms.npy"
  if [ -f "$norms_file" ]; then
    echo "SKIP (norms already present): drift_${tag}"
    continue
  fi
  echo "=== RUN: drift_term_run.py --schedule $sched $(date) ==="
  python drift_term_run.py --schedule "$sched" --root_dir cifar-10 --model resnet18 \
    --seed 0 --epochs 200 --subset_size 300 --out_dir results \
    > "logs/drift_${sched}_norms.log" 2>&1
  echo "=== DONE: drift_term_run.py --schedule $sched exit=$? $(date) ==="
done
echo "RUN_DRIFT_NORMS_ALL_DONE $(date)"

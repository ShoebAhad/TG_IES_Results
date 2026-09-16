#!/usr/bin/env bash
# Chains: (1) drift-norm rerun (item 2 remainder, #17, ~1.5h) then (2) CIFAR-100
# seed1-2 backfill (item 1 part, ~2h). Launched detached so it survives session end.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== run_p2_item1_chain.sh START $(date) ===" >> run_p2_item1_driver.log
bash run_drift_norms.sh >> run_p2_item1_driver.log 2>&1
bash run_cifar100_seeds12.sh >> run_p2_item1_driver.log 2>&1
echo "P2_ITEM1_CHAIN_ALL_DONE $(date)" >> run_p2_item1_driver.log

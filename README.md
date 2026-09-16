<h2 align="center">TG-IES: Theory-Guided Instance-Dependent Early Stopping</h2>
<p align="center">Code and result artifacts for <i>"Theory-Guided Instance-Dependent Early Stopping via Second-Order Loss Dynamics"</i>, extending <a href="https://openreview.net/pdf?id=P42DbV2nuV">Instance-dependent Early Stopping (IES), ICLR 2025 Spotlight</a>.</p>

### What this repo contains
- **`paper_iclr2026/`** — the current paper source (`main.tex`) and compiled PDF.
- **`paper/`** — an earlier ICML-format draft of the same paper.
- **`train.py`**, **`common.py`** — the unified training script implementing every method compared in the paper (`baseline`, `ies_shipped`, `ies_alg1`, `tgies`, `infobatch`, `bls_infobatch`, `alignprune`, `rho_loss`, `selective_backprop`).
- **`cifar_main.py`**, **`resnet.py`** — the original IES authors' code this project forked from (see citation below).
- **`*_analysis.py`** — scripts that turn raw `results/*.csv` into the paper's figures and reported statistics (drift-term validation, noise-autocorrelation, K*-trajectory, reactivation probe, equal-compute comparison, etc.).
- **`run_*.sh`** — driver scripts for the experiment matrix; each skips a cell whose output CSV already has a full run (`>= 201` lines), so they are safe to re-invoke to resume interrupted sweeps. Their `*_driver.log` files are the literal console output from the runs behind the paper's tables, kept for audit purposes.
- **`results/`** — per-run CSV/`.npy` outputs (one row per epoch) underlying every table and figure in the paper.
- **`figures/`** — the paper's generated figures (PDF + PNG), and the `make_*_figure.py` scripts that produce them from `results/`.
- **`logs/`** — raw stdout logs from individual training runs.
- **`EXPERIMENT_LOG.md`**, **`NEXT_STEPS.md`** — working notes documenting what was run, why, and what remains open; kept for transparency into the paper's revision history.

**Not included** (regenerable, not needed to inspect reported results): raw CIFAR-10/CIFAR-100 data (downloaded automatically by `train.py`/`cifar_main.py` on first run), the ~1.3GB `traj_ref/` reference-trajectory checkpoints used only to re-derive the K*-sweep trajectory-divergence figure, and Python/LaTeX build byproducts.

### Reproducing a run
```bash
pip install -r requirements.txt
# TG-IES, primary setting:
python3 train.py --method tgies --dataset cifar10 --model resnet18 --seed 0
# No-removal baseline:
python3 train.py --method baseline --dataset cifar10 --model resnet18 --seed 0
# IES exactly as shipped by the original authors:
python3 train.py --method ies_shipped --dataset cifar10 --model resnet18 --seed 0 --threshold 1e-3
```
Each run writes `results/<method>_<dataset>_<model>_<optimizer>_seed<k>.csv`, one row per epoch (test accuracy, cumulative backprop instances, and — for `tgies` — the recalibrated `delta`/`K*`). See `run_matrix.sh` and `NEXT_STEPS.md` for the full experiment matrix and how each reported number in the paper was produced.

### Citation
This project builds directly on the original IES codebase and paper:
```bibtex
@inproceedings{
yuan2025instancedependent,
title={Instance-dependent Early Stopping},
author={Suqin Yuan and Runqi Lin and Lei Feng and Bo Han and Tongliang Liu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}
```
Original IES code: [github.com/tmllab/2025_ICLR_IES](https://github.com/tmllab/2025_ICLR_IES). Original authors: <a href="https://suqinyuan.github.io">Suqin Yuan</a>, <a href="https://runqilin.github.io">Runqi Lin</a>, <a href="https://lfeng1995.github.io">Lei Feng</a>, <a href="https://bhanml.github.io">Bo Han</a>, <a href="https://tongliang-liu.github.io">Tongliang Liu</a>.

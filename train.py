"""Unified training entry point comparing four methods on identical data/model/optimizer infra:

  baseline    - No Removal (standard training on the full dataset every epoch).
  ies_shipped - The IES method exactly as shipped in cifar_main.py (moving-average+k
                smoothing on |delta^2 L|, train-mode forward for active samples but
                eval-mode forward for excluded samples, mean() gradient normalization).
  ies_alg1    - IES as literally specified by Algorithm 1 of Yuan et al. (ICLR 2025):
                single-epoch |delta^2 L_i(w^t)| < delta check (no smoothing/persistence),
                full-dataset forward pass every epoch in a CONSISTENT train() mode (no_grad
                for excluded samples) so L_i is a pure function of theta^t, mean() normalization.
  tgies       - Theory-Guided IES (this work): same consistent-forward-pass discipline as
                ies_alg1, plus (a) delta calibrated online via Corollary 4.5 (noise-based,
                Gaussian threshold whose per-check rate is backed out from a target COMPOUND
                miss budget over the whole K*-window, so it does not compound to near-zero
                survival when K* is large), (b) K* consecutive-epoch patience via Corollary
                6.4 (closed-form from a per-instance
                gradient-norm floor estimate, a tolerance tau and a Lipschitz proxy beta_test),
                and (c) the 1/N-scaled active-set gradient (eq. 14) implemented per mini-batch
                as sum(active_losses)/batch_size, which reduces exactly to standard training
                when nothing has been excluded and shrinks proportionally (not renormalizes
                upward) as instances are excluded -- the deliberate trade of IES's "larger
                gradient norm" speed heuristic for the provable stability bound of Theorem 6.2.
"""
import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

from common import (CifarDataset, device, get_cifar10_dataset, get_cifar100_dataset, get_model,
                     get_optimizer_and_scheduler, setup_seed)

parser = argparse.ArgumentParser(description="Unified IES / TG-IES training script")
parser.add_argument("--method", type=str, required=True,
                     choices=["baseline", "ies_shipped", "ies_alg1", "tgies",
                              "infobatch", "bls_infobatch", "alignprune", "rho_loss",
                              "selective_backprop"])
parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100"])
parser.add_argument("--root_dir", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--model", type=str, default="resnet18",
                     choices=["resnet18", "resnet34", "resnet50", "resnet101", "vgg16", "densenet121",
                              "vit_tiny_cifar"])
parser.add_argument("--optimizer", type=str, default="SGD_exponential_learning_rate",
                     choices=["Adam", "SGD_fixed_learning_rate", "SGD_exponential_learning_rate",
                              "Adam_W", "SGD_linear_learning_rate", "SGD_step_learning_rate",
                              "SGD_cosine_learning_rate", "AdamW_warmup_cosine"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", type=str, default="results")
parser.add_argument("--tag", type=str, default="", help="extra suffix for output filenames")
parser.add_argument("--warmup_epochs", type=int, default=10,
                     help="AdamW_warmup_cosine: linear warmup length in epochs")
parser.add_argument("--ckpt_every", type=int, default=10,
                     help="write a resume checkpoint every N epochs (0 disables checkpointing)")
parser.add_argument("--ckpt_dir", type=str, default="checkpoints",
                     help="directory for resume checkpoints")
parser.add_argument("--no_resume", action="store_true",
                     help="ignore any existing checkpoint and start from epoch 0")

# ies_shipped / ies_alg1
parser.add_argument("--threshold", type=float, default=1e-3, help="delta for ies_shipped/ies_alg1")
parser.add_argument("--moving_average_rate", type=int, default=3, help="ies_shipped only")
parser.add_argument("--k", type=int, default=1, help="ies_shipped only: trailing MA windows summed")

# tgies (Corollary 4.5 / Corollary 6.4)
parser.add_argument("--fp_rate_per_check", type=float, default=0.3,
                     help="p_budget: target COMPOUND false-negative-for-mastery probability over "
                          "the whole K*-consecutive-checks confirmation window (not a per-check "
                          "rate -- the per-check rate is backed out via 1-(1-p)^kstar so it does "
                          "not compound to near-zero survival for large kstar)")
parser.add_argument("--tau", type=float, default=0.02, help="tolerance on test-loss deviation for K*")
parser.add_argument("--beta_test", type=float, default=1.0, help="Lipschitz proxy for test loss")
parser.add_argument("--eps_percentile", type=float, default=20.0,
                     help="percentile of the per-instance grad-norm proxy used as epsilon0")
parser.add_argument("--recalib_every", type=int, default=30,
                     help="epoch at which delta/K* are calibrated once from Cor. 4.5/6.4 and then "
                          "held fixed for the rest of training (not a repeating interval). A "
                          "shrinking-noise-floor re-tracking delta never lets convergence 'grow "
                          "into' a stable bar the way the original paper's fixed delta does, and "
                          "each recalibration event also resets any in-progress K*-window "
                          "climbs, so repeated recalibration was found to strangle savings.")
parser.add_argument("--recalib_repeat", action="store_true",
                     help="opt back into the original repeating-recalibration behavior (every "
                          "recalib_every epochs) instead of a single one-time calibration")
parser.add_argument("--min_kstar", type=int, default=1)
parser.add_argument("--max_kstar", type=int, default=15)

# ablation: hold one of tgies's two calibrated quantities fixed instead of
# calibrating it, to isolate which mechanism drives tgies's gains
parser.add_argument("--tgies_fix_delta", type=float, default=None,
                     help="ablation: skip Corollary 4.5 delta calibration, use this fixed delta "
                          "for tgies throughout (K* still calibrated via Corollary 6.4)")
parser.add_argument("--tgies_fix_kstar", type=int, default=None,
                     help="ablation: skip Corollary 6.4 kstar calibration, use this fixed kstar "
                          "for tgies throughout (delta still calibrated via Corollary 4.5)")

# trajectory-divergence instrumentation (rem:shared-traj / K*-sweep, NEXT_STEPS item 7):
# quantifies how far a removal method's actual parameter trajectory drifts from the
# shared full-data (baseline) reference trajectory thm:safe's certificate assumes.
# method=="baseline" WRITES the reference snapshots; every other method READS them and
# logs the L2 distance -- baseline must therefore be run first for a given
# (dataset, model, optimizer, seed) cell before any comparison run reads from it.
parser.add_argument("--dump_traj_every", type=int, default=0,
                     help="0 disables. Otherwise, every N epochs (and at the final epoch): "
                          "baseline writes a flattened-parameter snapshot to --traj_ref_dir; "
                          "any other method loads the matching snapshot and logs the L2 "
                          "distance to results/traj_dist_<run_name>.csv")
parser.add_argument("--traj_ref_dir", type=str, default="traj_ref",
                     help="directory for baseline's shared-reference parameter snapshots")

# reactivation probe (reviewer item #23): tgies only, self-calibrated mode only (no
# effect under --tgies_fix_kstar, since eps0 is never computed there). Every N epochs,
# takes a random subsample of the CURRENTLY PASSIVE (excluded) set and computes each
# instance's TRUE per-instance gradient norm ||g_i|| via an individual batch-size-1
# backward pass -- the expensive, exact quantity cor:bridgeprob is actually about,
# not the cheap last-layer proxy used for calibration -- and logs what fraction
# already exceed the last-calibrated eps0 floor that justified their exclusion.
parser.add_argument("--reactivation_probe_every", type=int, default=0,
                     help="0 disables. Otherwise, every N epochs, probe a subsample of "
                          "the passive set's true gradient norms and log to "
                          "results/reactivation_<run_name>.csv")
parser.add_argument("--reactivation_probe_n", type=int, default=300,
                     help="max number of passive instances probed per check")

# infobatch / bls_infobatch / alignprune: soft (probabilistic) pruning with
# unbiased-in-expectation gradient rescaling (Qin et al. 2024, arXiv:2303.04947),
# scored either by raw per-sample loss (infobatch), an EMA-smoothed per-sample
# loss (bls_infobatch, per arXiv:2604.04681 "Batch Loss Score"), or a
# loss-trajectory-vs-reference correlation (alignprune, per arXiv:2604.07306
# "AlignPrune"/DAS). Unlike the IES family, these methods do NOT forward
# pruned instances at all (forward+backward both skipped), so they are a
# strictly cheaper-per-epoch mechanism than IES-style mastery tracking, which
# always forwards every instance to evaluate the Delta^2 criterion.
parser.add_argument("--ib_prune_ratio", type=float, default=0.5,
                     help="r: probability of pruning a below-mean-score sample each epoch")
parser.add_argument("--ib_anneal_frac", type=float, default=0.875,
                     help="pruning only active for the first ib_anneal_frac fraction of "
                          "epochs; the remainder trains on the full dataset (annealing)")
parser.add_argument("--bls_alpha", type=float, default=0.7,
                     help="bls_infobatch: EMA decay for the smoothed per-sample score")
parser.add_argument("--align_window", type=int, default=25,
                     help="alignprune: number of most-recent observed epochs used to "
                          "compute the Dynamic Alignment Score (Pearson correlation "
                          "between a sample's loss trajectory and the population-mean "
                          "loss trajectory at the same epoch indices -- a documented "
                          "simplification since no clean-labeled reference subset exists "
                          "in this CIFAR setup)")

# rho_loss (Mindermann et al. 2022, arXiv:2206.07137): reuses the exact same
# soft-pruning/anneal scaffold as infobatch/bls_infobatch/alignprune above (a
# documented simplification of the original's hard top-b-of-large-batch
# selection, matching this paper's existing adaptation of BLS-InfoBatch and
# AlignPrune into the same mechanism for a fair, mechanism-isolated
# comparison), scored by reducible loss = current per-instance training loss
# minus a FROZEN per-instance "irreducible loss" read from a separately
# pretrained holdout model (see pretrain_il_model.py). The holdout split used
# to pretrain that model is permanently excluded from the main run's active
# pool (never forwarded or backpropped here), matching the original method;
# see pretrain_il_model.py's docstring for the pretraining recipe and cost
# accounting.
parser.add_argument("--il_cache_dir", type=str, default="il_cache",
                     help="rho_loss: directory holding il_losses_{dataset}_{model}.npy "
                          "and il_holdout_idx_{dataset}_{model}.npy from pretrain_il_model.py")

# selective_backprop (Jiang et al. 2019, "Accelerating Deep Learning by Focusing on the
# Biggest Losers", arXiv:1910.00762): probabilistic accept/reject on relative loss
# percentile, biased toward high-loss instances, NO gradient rescaling (unlike
# infobatch/bls_infobatch/alignprune above, SB is an explicitly biased sampler by
# design, not an unbiased-in-expectation one) and no annealing back to the full
# dataset. Reuses the infobatch-family scaffold: a persistent per-instance score
# array (here, each instance's most-recently-observed loss) ranked into percentiles
# every epoch, in place of the original paper's streaming rolling-history reservoir
# -- a documented simplification consistent with this codebase's other adapted
# baselines (rho_loss, bls_infobatch, alignprune all note similar simplifications
# above) and with every method here forwarding/re-scoring the full dataset each epoch.
parser.add_argument("--sb_beta", type=float, default=3.0,
                     help="selective_backprop: selectivity exponent. Accept probability "
                          "for an instance at loss-percentile rank p in [0,1] is p^(beta-1); "
                          "beta=1 reduces to uniform (no selection), higher beta concentrates "
                          "backprop more sharply on the highest-loss instances")

# class imbalance / label noise (reviewer items #22, #21): applied once, right after
# the dataset loads and before N is fixed, using the RNG state setup_seed(args.seed)
# already established -- so all four methods sharing a --seed see the identical
# imbalanced subset / identical corrupted labels, preserving the paired-seed design
# app:stats's statistics depend on.
parser.add_argument("--imbalance_ratio", type=float, default=1.0,
                     help="1.0 = no imbalance (default). Otherwise, an exponential "
                          "per-class subsampling profile with this ratio between the "
                          "most- and least-frequent class after subsampling (step "
                          "imbalance is the common alternative; exponential is used here "
                          "since it does not require picking an arbitrary head/tail split)")
parser.add_argument("--label_noise", type=float, default=0.0,
                     help="fraction in [0,1] of TRAIN labels uniformly reassigned to a "
                          "different class (never the original), applied once before "
                          "training starts -- symmetric label noise, the standard setting "
                          "in this literature")

args = None  # populated in __main__; kept as a module global for brevity elsewhere in this file


def pearson_rowwise(vals, ref_vals, valid_mask, min_obs=3):
    """Row-wise Pearson correlation between vals[i,:] and ref_vals[i,:] over the
    columns where valid_mask[i,:] is True. Returns NaN for rows with < min_obs
    valid columns (not enough history yet to score)."""
    n = valid_mask.sum(axis=1).astype(np.float64)
    enough = n >= min_obs
    n_safe = np.maximum(n, 1.0)
    mean_x = (vals * valid_mask).sum(axis=1) / n_safe
    mean_y = (ref_vals * valid_mask).sum(axis=1) / n_safe
    dx = (vals - mean_x[:, None]) * valid_mask
    dy = (ref_vals - mean_y[:, None]) * valid_mask
    cov = (dx * dy).sum(axis=1) / n_safe
    var_x = (dx * dx).sum(axis=1) / n_safe
    var_y = (dy * dy).sum(axis=1) / n_safe
    denom = np.sqrt(np.maximum(var_x * var_y, 1e-12))
    das = cov / denom
    das[~enough] = np.nan
    return das


def phi_inv(q):
    # Phi^-1(q) = sqrt(2) * erfinv(2q - 1)
    return math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * q - 1.0)).item()


def build_transform(dataset):
    if dataset == "cifar10":
        return Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(),
                         Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        return Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(),
                         Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])


def sliding_window_mean(buf, window):
    # buf: (N, L) numpy array of |delta^2 L| history, most recent last. Returns (N, L-window+1).
    if buf.shape[1] < window:
        return np.zeros((buf.shape[0], 0), dtype=buf.dtype)
    shape = (buf.shape[0], buf.shape[1] - window + 1, window)
    strides = (buf.strides[0], buf.strides[1], buf.strides[1])
    windows = np.lib.stride_tricks.as_strided(buf, shape=shape, strides=strides)
    return windows.mean(axis=2)


def main():
    start_time = time.time()
    setup_seed(args.seed)

    if args.dataset == "cifar10":
        num_classes = 10
        threshold = args.threshold
        get_dataset = get_cifar10_dataset
        root_dir = args.root_dir or "./cifar-10"
    else:
        num_classes = 100
        threshold = args.threshold * 2
        get_dataset = get_cifar100_dataset
        root_dir = args.root_dir or "./cifar-100"

    transform = build_transform(args.dataset)
    train_dataset, test_dataset = get_dataset(root_dir, transform)

    if args.imbalance_ratio != 1.0:
        # Exponential per-class subsampling profile (reviewer item #22): the c-th class
        # (in a seed-determined random class order, so "most frequent" isn't tied to the
        # dataset's own label indexing) keeps a fraction imbalance_ratio^(-c/(C-1)) of
        # its original count, giving exactly args.imbalance_ratio between the most- and
        # least-kept classes. Uses np.random directly (already seeded by setup_seed
        # above), so all four methods sharing --seed see the identical imbalanced subset.
        labels_full = train_dataset.labels
        class_order = np.random.permutation(num_classes)
        keep_idx = []
        for rank, c in enumerate(class_order):
            cls_idx = np.nonzero(labels_full == c)[0]
            frac = args.imbalance_ratio ** (-rank / max(num_classes - 1, 1))
            n_keep = max(1, int(round(len(cls_idx) * frac)))
            keep_idx.append(np.random.choice(cls_idx, size=n_keep, replace=False))
        keep_idx = np.sort(np.concatenate(keep_idx))
        train_dataset = CifarDataset(train_dataset.data, train_dataset.labels, keep_idx.tolist(), transform)
        counts = np.bincount(labels_full[keep_idx], minlength=num_classes)
        print(f"[imbalance] ratio={args.imbalance_ratio}: kept {len(keep_idx)}/{len(labels_full)} "
              f"instances, per-class counts min={counts[counts > 0].min()} max={counts.max()}",
              flush=True)

    if args.label_noise > 0.0:
        # Symmetric label noise (reviewer item #21): each kept training instance
        # independently has probability args.label_noise of having its label uniformly
        # reassigned to a DIFFERENT class. Corrupts train_dataset.labels in place, only
        # at index_list's positions, so combining with --imbalance_ratio corrupts only
        # the already-kept subset.
        idx_arr = np.array(train_dataset.index_list)
        corrupt_mask = np.random.rand(len(idx_arr)) < args.label_noise
        corrupt_positions = idx_arr[corrupt_mask]
        orig_labels = train_dataset.labels[corrupt_positions]
        offset = np.random.randint(1, num_classes, size=len(corrupt_positions))
        train_dataset.labels[corrupt_positions] = (orig_labels + offset) % num_classes
        print(f"[label_noise] rate={args.label_noise}: corrupted {int(corrupt_mask.sum())}/"
              f"{len(idx_arr)} training labels", flush=True)

    N = len(train_dataset)
    data_arr, labels_arr = train_dataset.data, train_dataset.labels

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, drop_last=False)

    model = get_model(args.model, num_classes).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(args.optimizer, model.parameters(), args)
    ce_none = nn.CrossEntropyLoss(reduction='none')
    ce_mean = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"{args.method}_{args.dataset}_{args.model}_{args.optimizer}_seed{args.seed}{tag}"
    csv_path = os.path.join(args.out_dir, run_name + ".csv")
    CSV_HEADER = ['epoch', 'active_set_size', 'avg_train_loss', 'avg_test_loss', 'test_correct',
                  'test_acc_pct', 'backprop_instances_this_epoch', 'cum_backprop_instances',
                  'total_excluded_now', 'newly_excluded_this_epoch', 'delta', 'kstar', 'epoch_time_s']
    ckpt_path = os.path.join(args.ckpt_dir, run_name + ".ckpt")

    # ---- bookkeeping state (numpy, indexed by global sample id 0..N-1) ----
    L_prev = np.full(N, np.nan, dtype=np.float64)
    L_prevprev = np.full(N, np.nan, dtype=np.float64)
    active = np.ones(N, dtype=bool)

    # ies_shipped ring buffer of |delta^2 L| history
    ma, kk = args.moving_average_rate, args.k
    buf_width = ma + kk
    diff_buf = np.zeros((N, 0), dtype=np.float64)  # grows up to buf_width then rolls

    # tgies persistence
    cnt = np.zeros(N, dtype=np.int64)
    delta_tgies = threshold  # will be recalibrated after warmup
    kstar = args.min_kstar
    eps0 = None  # set on first self-calibration; stays None under --tgies_fix_kstar

    # infobatch / bls_infobatch score (raw or EMA-smoothed per-sample loss);
    # init 0.0 for all so mean==0 and nothing is "below mean" in epoch 1 --
    # the whole dataset trains once to seed real scores, matching InfoBatch's
    # own warmup-by-construction behavior.
    ib_score = np.zeros(N, dtype=np.float64)
    il_losses = None
    holdout_mask = np.zeros(N, dtype=bool)
    if args.method == "rho_loss":
        il_path = os.path.join(args.il_cache_dir, f"il_losses_{args.dataset}_{args.model}.npy")
        holdout_path = os.path.join(args.il_cache_dir, f"il_holdout_idx_{args.dataset}_{args.model}.npy")
        if not os.path.exists(il_path) or not os.path.exists(holdout_path):
            raise FileNotFoundError(
                f"rho_loss needs a pretrained irreducible-loss model: run "
                f"`python pretrain_il_model.py --dataset {args.dataset} --model {args.model}` first "
                f"(expected {il_path} and {holdout_path})")
        il_losses = np.load(il_path).astype(np.float64)
        holdout_idx = np.load(holdout_path)
        holdout_mask[holdout_idx] = True
        # tie every score at "reducible loss == 0" for the warmup epoch, matching
        # infobatch's own all-zero-tie trick (see comment above) so the first epoch
        # trains the (non-holdout) pool once before any pruning decision is made
        ib_score = il_losses.copy()

    # alignprune ring buffers: last align_window (epoch_idx, loss) observations
    # per sample, plus the population-mean-loss trajectory needed as reference.
    W = args.align_window
    align_loss_hist = np.zeros((N, W), dtype=np.float64)
    align_epoch_hist = np.zeros((N, W), dtype=np.int64)
    align_valid = np.zeros((N, W), dtype=bool)
    align_pos = np.zeros(N, dtype=np.int64)
    pop_mean_traj = np.zeros(args.epochs + 1, dtype=np.float64)

    cum_backprop = 0
    best_test_acc = 0

    # ---- resume support -------------------------------------------------
    # Config knobs that change the science. A checkpoint written under a
    # different fingerprint is NOT resumable: erroring out beats silently
    # restarting a multi-hour cell or, worse, continuing it with mismatched
    # semantics. Driver scripts run each cell in its own process, so a hard
    # exit here skips just this cell.
    fingerprint = {
        "method": args.method, "dataset": args.dataset, "model": args.model,
        "optimizer": args.optimizer, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "threshold": args.threshold,
        "moving_average_rate": args.moving_average_rate, "k": args.k,
        "recalib_every": args.recalib_every, "recalib_repeat": args.recalib_repeat,
        "fp_rate_per_check": args.fp_rate_per_check, "tau": args.tau,
        "beta_test": args.beta_test, "eps_percentile": args.eps_percentile,
        "min_kstar": args.min_kstar, "max_kstar": args.max_kstar,
        "tgies_fix_delta": args.tgies_fix_delta, "tgies_fix_kstar": args.tgies_fix_kstar,
        "ib_prune_ratio": args.ib_prune_ratio, "ib_anneal_frac": args.ib_anneal_frac,
        "bls_alpha": args.bls_alpha, "align_window": args.align_window,
        "warmup_epochs": args.warmup_epochs, "sb_beta": args.sb_beta,
        "imbalance_ratio": args.imbalance_ratio, "label_noise": args.label_noise,
    }

    start_epoch = 0
    elapsed_prior = 0.0
    if args.ckpt_every > 0 and not args.no_resume and os.path.exists(ckpt_path):
        # map to CPU, not `device`: torch.set_rng_state/set_rng_state_all require
        # CPU ByteTensors, and load_state_dict moves params/optimizer state onto
        # the right device itself. Loading onto CUDA here crashes the RNG restore.
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("fingerprint") != fingerprint:
            raise SystemExit(
                f"[{run_name}] checkpoint {ckpt_path} was written with a different config; "
                f"refusing to resume. Delete it to start fresh.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        L_prev, L_prevprev, active = ck["L_prev"], ck["L_prevprev"], ck["active"]
        diff_buf = ck["diff_buf"]
        cnt, delta_tgies, kstar = ck["cnt"], ck["delta_tgies"], ck["kstar"]
        eps0 = ck.get("eps0")  # absent in checkpoints written before this key existed
        ib_score = ck["ib_score"]
        align_loss_hist, align_epoch_hist = ck["align_loss_hist"], ck["align_epoch_hist"]
        align_valid, align_pos = ck["align_valid"], ck["align_pos"]
        pop_mean_traj = ck["pop_mean_traj"]
        cum_backprop, best_test_acc = ck["cum_backprop"], ck["best_test_acc"]
        start_epoch, elapsed_prior = ck["epoch"], ck["elapsed"]
        # RNG: without this the augmentation/shuffle stream diverges after a
        # resume, breaking the same-seed pairing across methods that the
        # paper's paired statistics rely on.
        torch.set_rng_state(ck["rng_torch"])
        if ck["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        np.random.set_state(ck["rng_numpy"])
        random.setstate(ck["rng_python"])
        # The CSV may hold rows past the checkpoint (rows are appended every
        # epoch, checkpoints only every ckpt_every). Truncate to the
        # checkpointed epoch so resumed rows neither duplicate nor gap.
        if os.path.exists(csv_path):
            with open(csv_path, newline='') as f:
                rows = list(csv.reader(f))
            with open(csv_path, 'w', newline='') as f:
                csv.writer(f).writerows(rows[:1 + start_epoch])
        else:
            # checkpoint survived but the CSV did not: keep going rather than
            # discard GPU-hours. The row set will be short, so the driver's
            # >=201-line completeness check re-runs this cell later.
            print(f"[{run_name}] WARNING: resuming without {csv_path}; "
                  f"per-epoch rows before epoch {start_epoch} are lost", flush=True)
            with open(csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(CSV_HEADER)
        print(f"[{run_name}] RESUMED from epoch {start_epoch}/{args.epochs} "
              f"({elapsed_prior:.0f}s already spent)", flush=True)

    if start_epoch == 0:
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADER)

    def save_ckpt(next_epoch):
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        blob = {
            "fingerprint": fingerprint,
            "epoch": next_epoch,
            "elapsed": elapsed_prior + (time.time() - start_time),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "L_prev": L_prev, "L_prevprev": L_prevprev, "active": active,
            "diff_buf": diff_buf,
            "cnt": cnt, "delta_tgies": delta_tgies, "kstar": kstar, "eps0": eps0,
            "ib_score": ib_score,
            "align_loss_hist": align_loss_hist, "align_epoch_hist": align_epoch_hist,
            "align_valid": align_valid, "align_pos": align_pos,
            "pop_mean_traj": pop_mean_traj,
            "cum_backprop": cum_backprop, "best_test_acc": best_test_acc,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
        }
        # atomic: a crash mid-write must not leave a corrupt checkpoint
        tmp = ckpt_path + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, ckpt_path)

    for epoch in range(start_epoch, args.epochs):
        t_epoch0 = time.time()
        model.train()
        running_loss = 0.0
        backprop_this_epoch = 0

        if args.method == "baseline":
            loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, drop_last=True)
            for inputs, labels, indices in loader:
                if inputs.size(0) == 1:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                losses = ce_none(outputs, labels)
                loss = losses.mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                backprop_this_epoch += inputs.size(0)
            active_size = N
            excluded_this_epoch = 0
            newly_excluded_count = 0
            delta_report, kstar_report = float('nan'), 0

        elif args.method in ("infobatch", "bls_infobatch", "alignprune", "rho_loss", "selective_backprop"):
            prune_active = epoch < int(round(args.ib_anneal_frac * args.epochs))

            if args.method == "alignprune":
                pop_ref = pop_mean_traj[align_epoch_hist]  # (N, W) gather
                score = pearson_rowwise(align_loss_hist, pop_ref, align_valid)
                # DAS: LOW correlation with the population trend is the "prune" signal
                # (misaligned trajectory), same direction as loss-based below-mean pruning.
            elif args.method == "rho_loss":
                score = ib_score - il_losses  # reducible loss; low (or negative) => prune
            else:  # infobatch, bls_infobatch, selective_backprop all rank the same persistent score
                score = ib_score

            valid_score = ~np.isnan(score) & ~holdout_mask  # holdout entries never compete for keep/prune

            if args.method == "selective_backprop":
                # Percentile-rank accept probability, biased toward high-loss instances,
                # NO rescaling and no annealing (see the docstring above --sb_beta): an
                # instance never yet scored (valid_score False, early epochs) is always
                # accepted once, both to bootstrap its score and since "unknown loss"
                # should not be read as "low loss".
                valid_idx = np.nonzero(valid_score)[0]
                ranks = np.zeros(N, dtype=np.float64)
                if len(valid_idx) > 0:
                    order = np.argsort(score[valid_idx])
                    ranks[valid_idx[order]] = (np.arange(len(valid_idx)) + 1) / len(valid_idx)
                accept_prob = np.where(valid_score, ranks ** (args.sb_beta - 1.0), 1.0)
                coin = np.random.rand(N)
                keep_mask = coin < accept_prob
                below_mean = np.zeros(N, dtype=bool)  # unused by SB; keeps the rescale branch below inert
            else:
                mean_score = np.nanmean(score[valid_score]) if valid_score.any() else 0.0
                below_mean = valid_score & (score < mean_score)
                if prune_active:
                    coin = np.random.rand(N)
                    keep_mask = (~below_mean) | (coin >= args.ib_prune_ratio)
                else:
                    keep_mask = np.ones(N, dtype=bool)

            if args.method == "rho_loss":
                keep_mask &= ~holdout_mask  # holdout split is never trained on in the main run

            sampled_idx = np.nonzero(keep_mask)[0]
            sampled_ds = CifarDataset(data_arr, labels_arr, sampled_idx.tolist(), transform)
            drop_last_sampled = len(sampled_idx) >= args.batch_size
            sampled_loader = DataLoader(sampled_ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=drop_last_sampled)

            rescale = 1.0 / (1.0 - args.ib_prune_ratio)
            new_losses_epoch = np.full(N, np.nan, dtype=np.float64)

            for inputs, labels, indices in sampled_loader:
                if inputs.size(0) == 1:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                idx_np = indices.numpy()
                optimizer.zero_grad()
                outputs = model(inputs)
                losses = ce_none(outputs, labels)
                raw_np = losses.detach().double().cpu().numpy()
                new_losses_epoch[idx_np] = raw_np

                if args.method != "selective_backprop" and prune_active:
                    w_np = np.where(below_mean[idx_np], rescale, 1.0).astype(np.float32)
                    weights = torch.from_numpy(w_np).to(device)
                    loss = (losses * weights).sum() / inputs.size(0)
                else:
                    loss = losses.mean()
                loss.backward()
                optimizer.step()
                running_loss += raw_np.sum()
                backprop_this_epoch += inputs.size(0)

            forwarded = ~np.isnan(new_losses_epoch)
            if args.method in ("infobatch", "rho_loss", "selective_backprop"):
                ib_score[forwarded] = new_losses_epoch[forwarded]
            elif args.method == "bls_infobatch":
                a = args.bls_alpha
                ib_score[forwarded] = a * ib_score[forwarded] + (1 - a) * new_losses_epoch[forwarded]
            else:  # alignprune: append this epoch's observations into the ring buffer
                fw_idx = np.nonzero(forwarded)[0]
                pos = align_pos[fw_idx]
                align_loss_hist[fw_idx, pos] = new_losses_epoch[fw_idx]
                align_epoch_hist[fw_idx, pos] = epoch  # index into pop_mean_traj
                align_valid[fw_idx, pos] = True
                align_pos[fw_idx] = (pos + 1) % W

            pop_mean_traj[epoch] = new_losses_epoch[forwarded].mean() if forwarded.any() else 0.0

            active_size = len(sampled_idx)
            excluded_this_epoch = N - active_size
            newly_excluded_count = excluded_this_epoch  # resampled every epoch, like ies_shipped
            if args.method == "selective_backprop":
                delta_report, kstar_report = args.sb_beta, 1  # repurposed: beta, always-active flag
            else:
                delta_report = args.ib_prune_ratio if prune_active else 0.0
                kstar_report = 1 if prune_active else 0  # repurposed: 1 = pruning phase, 0 = annealed
        else:
            active_idx = np.nonzero(active)[0]
            passive_idx = np.nonzero(~active)[0]

            active_ds = CifarDataset(data_arr, labels_arr, active_idx.tolist(), transform)
            # drop_last=True keeps every batch shape constant across epochs, which avoids
            # cudnn.benchmark re-triggering its autotune search on a newly-sized trailing
            # batch every time the active set shrinks; only skip drop_last if doing so would
            # leave zero batches (active set smaller than one batch).
            drop_last_active = len(active_idx) >= args.batch_size
            active_workers = min(args.num_workers, max(1, len(active_idx) // (4 * args.batch_size)))
            active_loader = DataLoader(active_ds, batch_size=args.batch_size, shuffle=True,
                                        num_workers=active_workers, drop_last=drop_last_active)

            L_new = np.full(N, np.nan, dtype=np.float64)  # nan for any sample skipped this epoch

            for inputs, labels, indices in active_loader:
                if inputs.size(0) == 1:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                idx_np = indices.numpy()
                optimizer.zero_grad()
                outputs = model(inputs)
                losses = ce_none(outputs, labels)
                L_new[idx_np] = losses.detach().double().cpu().numpy()

                if args.method == "tgies":
                    loss = losses.sum() / args.batch_size  # eq. 14, matches Theorem 6.2 scaling
                else:
                    loss = losses.mean()
                loss.backward()
                optimizer.step()
                running_loss += losses.sum().item()
                backprop_this_epoch += inputs.size(0)

            if len(passive_idx) > 0:
                passive_ds = CifarDataset(data_arr, labels_arr, passive_idx.tolist(), transform)
                drop_last_passive = len(passive_idx) >= args.batch_size
                # scale worker-process count with passive-set size: spawning a full pool of
                # worker processes every epoch to forward a handful of samples is pure overhead
                passive_workers = min(args.num_workers, max(0, len(passive_idx) // (4 * args.batch_size)))
                passive_loader = DataLoader(passive_ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=passive_workers, drop_last=drop_last_passive)
                with torch.no_grad():
                    for inputs, labels, indices in passive_loader:
                        if inputs.size(0) == 1:  # BatchNorm in train() mode needs >1 sample
                            continue
                        inputs, labels = inputs.to(device), labels.to(device)
                        idx_np = indices.numpy()
                        outputs = model(inputs)  # model stays in train() mode: consistent BN stats
                        losses = ce_none(outputs, labels)
                        L_new[idx_np] = losses.detach().double().cpu().numpy()

            active_size = len(active_idx)

            # ---- mastery check using L_new (=L^t), L_prev (=L^{t-1}), L_prevprev (=L^{t-2}) ----
            # (any sample skipped this epoch via the size==1 guard stays NaN in L_new, which
            #  correctly propagates into L_prev next epoch and disqualifies it from have_history
            #  until it gets a real loss value again -- no silent corruption from uninitialized memory)
            have_history = ~np.isnan(L_prev) & ~np.isnan(L_prevprev) & ~np.isnan(L_new)
            delta2 = np.zeros(N, dtype=np.float64)
            delta2[have_history] = L_new[have_history] - 2 * L_prev[have_history] + L_prevprev[have_history]
            abs_delta2 = np.abs(delta2)

            newly_excluded = np.zeros(N, dtype=bool)
            delta_report, kstar_report = threshold, 0

            if args.method == "ies_shipped" and threshold > 0:
                if have_history.any():  # mirrors original: only append once len(losses)>=3 for everyone
                    col = abs_delta2.reshape(-1, 1)
                    diff_buf = np.concatenate([diff_buf, col], axis=1) if diff_buf.size else col
                    if diff_buf.shape[1] > buf_width:
                        diff_buf = diff_buf[:, -buf_width:]
                meets_criterion = np.zeros(N, dtype=bool)
                if diff_buf.shape[1] >= buf_width:
                    ma_vals = sliding_window_mean(diff_buf, ma)          # (N, buf_width-ma+1)
                    deriv_sum = np.abs(ma_vals[:, -kk:]).sum(axis=1)
                    meets_criterion = have_history & (deriv_sum < threshold)
                # match the original: excluded_samples is recomputed FROM SCRATCH every epoch over
                # the whole population, not accumulated -- so a previously-excluded sample whose
                # criterion later fails is automatically reactivated (Appendix H "reversible" IES).
                newly_excluded = meets_criterion & active
                active[:] = ~meets_criterion
                delta_report = threshold

            elif args.method == "ies_alg1" and threshold > 0:
                meets_criterion = have_history & (abs_delta2 < threshold)
                newly_excluded = meets_criterion & active
                active[:] = ~meets_criterion
                delta_report = threshold

            elif args.method == "tgies":
                # --- recalibration of delta (Cor. 4.5) and K* (Cor. 6.4) ---
                should_recalib = (
                    (epoch % args.recalib_every == 0) if args.recalib_repeat
                    else (epoch == args.recalib_every)
                )
                if epoch > 0 and should_recalib and have_history.any():
                    if args.tgies_fix_kstar is not None:
                        # ablation: skip the gradient-norm-proxy recalibration pass entirely --
                        # it exists only to compute kstar, so a fixed kstar makes it unnecessary
                        kstar = args.tgies_fix_kstar
                    else:
                        # cheap last-layer gradient-norm proxy: ||softmax(z)-onehot|| * ||feat||
                        # a percentile only needs a random subsample of the active set, not a full
                        # N-sample forward pass -- this cuts the recalibration cost by ~6x while
                        # leaving the percentile estimate statistically unchanged.
                        calib_pool = active_idx if len(active_idx) <= 8192 else np.random.choice(
                            active_idx, size=8192, replace=False)
                        calib_ds = CifarDataset(data_arr, labels_arr, calib_pool.tolist(), transform)
                        calib_workers = min(args.num_workers, max(1, len(calib_pool) // 1024))
                        with torch.no_grad():
                            model.eval()
                            proxy_norms = []
                            calib_loader = DataLoader(calib_ds, batch_size=256, shuffle=False,
                                                       num_workers=calib_workers)
                            for inputs, labels, indices in calib_loader:
                                inputs, labels = inputs.to(device), labels.to(device)
                                logits, feat = model(inputs, ret_feat=True)
                                probs = torch.softmax(logits, dim=1)
                                onehot = torch.zeros_like(probs).scatter_(1, labels.view(-1, 1), 1.0)
                                resid_norm = (probs - onehot).norm(dim=1)
                                feat_norm = feat.norm(dim=1)
                                proxy_norms.append((resid_norm * feat_norm).cpu().double().numpy())
                            model.train()
                        proxy_norms = np.concatenate(proxy_norms)
                        eps0 = max(np.percentile(proxy_norms, args.eps_percentile), 1e-8)
                        eta_now = optimizer.param_groups[0]['lr']
                        kstar = int(math.floor((N * args.tau) / (args.beta_test * eta_now * eps0)))
                        kstar = int(np.clip(kstar, args.min_kstar, args.max_kstar))

                    if args.tgies_fix_delta is not None:
                        # ablation: skip Corollary 4.5, hold delta fixed at the given value
                        delta_tgies = args.tgies_fix_delta
                    else:
                        # delta must be calibrated AFTER kstar: Corollary 4.5 calibrates a SINGLE
                        # check's miss rate, but the algorithm requires kstar *consecutive* clean
                        # checks before excluding an instance. Using args.fp_rate_per_check directly
                        # as the per-check rate compounds over kstar checks: survival probability for
                        # a genuinely-mastered instance is (1-p)^kstar, which collapses to near-zero
                        # once kstar is more than a handful (e.g. (1-0.3)^15 = 0.5%). Instead treat
                        # fp_rate_per_check as the desired COMPOUND miss budget over the whole kstar-
                        # window and back out the per-check rate p_check via 1-(1-p)^kstar = p_budget.
                        hd = abs_delta2[have_history]
                        if hd.size >= 10:
                            lower_half = hd[hd <= np.median(hd)]
                            sigma_hat = math.sqrt(max(np.median(lower_half ** 2) / 6.0, 1e-12))
                            p_budget = args.fp_rate_per_check
                            p_check = 1.0 - (1.0 - p_budget) ** (1.0 / kstar)
                            k_gauss = phi_inv(1 - p_check / 2)
                            delta_tgies = k_gauss * math.sqrt(6.0) * sigma_hat

                below = have_history & (abs_delta2 < delta_tgies)
                cnt[below] += 1
                cnt[~below] = 0
                active[~below] = True  # reactivate: criterion failed this epoch (box lines 11-13)
                newly_excluded = below & (cnt >= kstar)
                delta_report, kstar_report = delta_tgies, kstar

            newly_excluded_count = int(newly_excluded.sum())
            active[newly_excluded] = False
            excluded_this_epoch = int((~active).sum())

            if (args.method == "tgies" and eps0 is not None and args.reactivation_probe_every > 0
                    and ((epoch + 1) % args.reactivation_probe_every == 0 or epoch + 1 == args.epochs)):
                # Reviewer item #23: does the cheap last-layer proxy behind eps0 (used to
                # derive K*) miss instances whose TRUE gradient norm has already grown back
                # above the floor that justified their exclusion? Probe a random subsample
                # of the currently-passive set with real, individual backward passes (the
                # exact quantity cor:bridgeprob is about), not the proxy.
                from PIL import Image
                passive_idx = np.nonzero(~active)[0]
                if len(passive_idx) > 0:
                    probe_idx = np.random.choice(
                        passive_idx, size=min(args.reactivation_probe_n, len(passive_idx)), replace=False)
                    model.eval()
                    gnorms = np.empty(len(probe_idx), dtype=np.float64)
                    # Aggregate/population-level check (P0 item #2/#6): also accumulate the
                    # SUM of the raw per-instance gradient vectors so we can report
                    # ||mean_i g_i|| (aggregate norm, can shrink via directional cancellation)
                    # alongside mean_i ||g_i|| (the per-instance quantity thm:safe/bridge bound).
                    # A ratio << 1 is direct evidence that the excluded set's gradients
                    # partially cancel in the sum TG-IES's update actually uses
                    # (\cref{eq:activegrad}), even when individual norms exceed eps0.
                    grad_accum = [torch.zeros_like(p) for p in model.parameters()]
                    for j, pidx in enumerate(probe_idx):
                        img_t = transform(Image.fromarray(data_arr[pidx])).unsqueeze(0).to(device)
                        label_t = torch.tensor([labels_arr[pidx]], device=device)
                        model.zero_grad(set_to_none=True)
                        out = model(img_t)
                        ce_mean(out, label_t).backward()
                        gnorms[j] = math.sqrt(sum(
                            float((p.grad.detach() ** 2).sum()) for p in model.parameters() if p.grad is not None))
                        for acc, p in zip(grad_accum, model.parameters()):
                            if p.grad is not None:
                                acc.add_(p.grad.detach())
                    agg_grad_norm = math.sqrt(sum(float((acc ** 2).sum()) for acc in grad_accum)) / len(probe_idx)
                    model.zero_grad(set_to_none=True)
                    model.train()
                    n_exceed = int((gnorms > eps0).sum())
                    reactivation_csv = os.path.join(args.out_dir, f"reactivation_{run_name}.csv")
                    write_reactivation_header = not os.path.exists(reactivation_csv)
                    with open(reactivation_csv, 'a', newline='') as f:
                        rw = csv.writer(f)
                        if write_reactivation_header:
                            rw.writerow(['epoch', 'n_passive', 'n_probed', 'eps0', 'mean_gnorm',
                                         'median_gnorm', 'n_exceed_eps0', 'frac_exceed_eps0',
                                         'agg_grad_norm'])
                        rw.writerow([epoch + 1, len(passive_idx), len(probe_idx), eps0,
                                     float(gnorms.mean()), float(np.median(gnorms)),
                                     n_exceed, n_exceed / len(probe_idx), agg_grad_norm])
                    print(f"[{run_name}] reactivation probe epoch {epoch + 1}: {n_exceed}/"
                          f"{len(probe_idx)} passive probes exceed eps0={eps0:.4g} "
                          f"(mean||g_i||={gnorms.mean():.4g}, ||mean g_i||={agg_grad_norm:.4g})", flush=True)

            L_prevprev = L_prev
            L_prev = L_new

        scheduler.step()
        cum_backprop += backprop_this_epoch
        avg_train_loss = running_loss / N

        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += ce_mean(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_acc_pct = 100.0 * correct / total
        best_test_acc = max(best_test_acc, correct)

        epoch_time = time.time() - t_epoch0
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch + 1, active_size, avg_train_loss, avg_test_loss, correct, test_acc_pct,
                        backprop_this_epoch, cum_backprop, excluded_this_epoch, newly_excluded_count,
                        delta_report, kstar_report, epoch_time])
        print(f"[{run_name}] epoch {epoch+1}/{args.epochs} acc={test_acc_pct:.2f}% "
              f"active={active_size} backprop={backprop_this_epoch} time={epoch_time:.1f}s", flush=True)

        if args.ckpt_every > 0 and ((epoch + 1) % args.ckpt_every == 0
                                    or epoch + 1 == args.epochs):
            save_ckpt(epoch + 1)

        if args.dump_traj_every > 0 and ((epoch + 1) % args.dump_traj_every == 0
                                          or epoch + 1 == args.epochs):
            # shared-reference key: independent of method/tag, since baseline and every
            # removal method being compared to it share init + data order for this seed.
            traj_key = f"{args.dataset}_{args.model}_{args.optimizer}_seed{args.seed}"
            ref_path = os.path.join(args.traj_ref_dir, f"{traj_key}_epoch{epoch + 1}.pt")
            with torch.no_grad():
                flat_params = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu()
            if args.method == "baseline":
                os.makedirs(args.traj_ref_dir, exist_ok=True)
                tmp = ref_path + ".tmp"
                torch.save(flat_params, tmp)
                os.replace(tmp, ref_path)
            elif os.path.exists(ref_path):
                ref_params = torch.load(ref_path, map_location="cpu", weights_only=True)
                dist = (flat_params - ref_params).norm().item()
                ref_norm = ref_params.norm().item()
                traj_csv = os.path.join(args.out_dir, f"traj_dist_{run_name}.csv")
                write_traj_header = not os.path.exists(traj_csv)
                with open(traj_csv, 'a', newline='') as f:
                    tw = csv.writer(f)
                    if write_traj_header:
                        tw.writerow(['epoch', 'param_l2_dist', 'ref_param_l2_norm',
                                      'rel_dist', 'active_set_size', 'total_excluded_now'])
                    tw.writerow([epoch + 1, dist, ref_norm,
                                 dist / ref_norm if ref_norm > 0 else float('nan'),
                                 active_size, excluded_this_epoch])
            else:
                print(f"[{run_name}] WARNING: missing baseline trajectory reference "
                      f"{ref_path} at epoch {epoch + 1}; skipping this snapshot's distance "
                      f"comparison (baseline for this cell/seed must run with "
                      f"--dump_traj_every first)", flush=True)

    total_time = elapsed_prior + (time.time() - start_time)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    write_header = not os.path.exists(summary_path)
    total_possible_backprop = N * args.epochs
    with open(summary_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['run_name', 'method', 'dataset', 'model', 'optimizer', 'seed', 'epochs',
                        'best_test_acc_pct', 'cum_backprop_instances', 'backprop_saved_pct', 'total_time_s'])
        w.writerow([run_name, args.method, args.dataset, args.model, args.optimizer, args.seed, args.epochs,
                    100.0 * best_test_acc / len(test_dataset), cum_backprop,
                    100.0 * (1 - cum_backprop / total_possible_backprop), total_time])
    print(f"[{run_name}] DONE best_acc={100.0*best_test_acc/len(test_dataset):.2f}% "
          f"backprop_saved={100.0*(1-cum_backprop/total_possible_backprop):.1f}% time={total_time:.1f}s")

    if args.ckpt_every > 0 and os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # run finished; nothing left to resume


if __name__ == "__main__":
    args = parser.parse_args()
    main()

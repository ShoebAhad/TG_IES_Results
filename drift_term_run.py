"""Priority-2 diagnostic: directly instruments the drift term D_i^(t) from Theorem
6.1's decomposition Delta^2 L_i = eta*D_i + O(eta^2) + R, rather than only checking its
consequences indirectly against the calibration record (as sec:lrmechanism currently
does). This was explicitly named as unverified/未 done in the paper's own Future Work
list and flagged critical by review.

D_i^(t) := <g_i^(t-2), g^(t-2)> - <g_i^(t-1), g^(t-1)>   (population/instance gradient
inner products, evaluated at the FIXED post-epoch parameters theta^(t) for each t, not
at intermediate mini-batch iterates).

Trains ResNet-18/CIFAR-10 with method=baseline (no removal, so the trajectory this
script instruments is theta^(t) as it would be for every method before any exclusion
diverges it -- the same trajectory sec:lrmechanism's calibration-record analysis reads
off of) under a given LR schedule. After each epoch's ordinary SGD updates finish
(theta^(t) now fixed), two extra passes run at that frozen theta^(t), both in eval()
mode (so BatchNorm uses running stats rather than per-batch stats, letting a
batch-size-1 gradient be evaluated consistently with the population-gradient pass; a
documented simplification relative to train.py's own train()-mode-everywhere
convention, chosen here because eval mode is well-defined for a single instance and
train.py's per-instance mastery-signal losses are not affected by this script):

  1. A full-dataset gradient ACCUMULATION pass (batched forward+backward, no optimizer
     step) giving the exact population gradient g^(t) = (1/N) sum_i g_i^(t), plus every
     instance's loss L_i^(t) at theta^(t) (the same "consistent forward pass" this
     paper's ies_alg1/tgies already use for their own mastery signal, reused here).
  2. For a fixed random SUBSET of instances (--subset_size, default 300 -- the paper's
     own reviewer-facing ask says "for a subset of training examples", not all N; doing
     all 50k individually would be prohibitively slow), an individual batch-size-1
     forward+backward pass giving g_i^(t) exactly, immediately dotted against g^(t) from
     step 1 (kept only as a running scalar, not stored as a vector, to keep memory/disk
     small) to give inner_i(t) := <g_i^(t), g^(t)>.

Saves per-epoch: eta_t (results/drift_<tag>_eta.npy), every instance's L_i^(t)
(results/drift_<tag>_losses.npy, N x epochs, reused from step 1 exactly as
noise_autocorrelation_run.py already does), and the subset's inner_i(t)
(results/drift_<tag>_inner.npy, subset_size x epochs) and ||g_i^(t)||
(results/drift_<tag>_norms.npy, subset_size x epochs -- item #17's calibration-bound
plot) plus the subset's global indices (results/drift_<tag>_subset_idx.npy). D_i^(t) and
Delta^2 L_i^(t) are both derivable post-hoc from these (drift_term_analysis.py) without
re-running training.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

from common import CifarDataset, device, get_cifar10_dataset, get_model, get_optimizer_and_scheduler, setup_seed


class Args:
    momentum = 0.9
    weight_decay = 5e-4


def build_transform():
    return Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(),
                     Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


def flat_grad_dict(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def dot_with(model, grad_dict):
    total = 0.0
    for n, p in model.named_parameters():
        if p.grad is not None and n in grad_dict:
            total += torch.sum(p.grad.detach() * grad_dict[n]).item()
    return total


def grad_norm(model):
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += torch.sum(p.grad.detach() ** 2).item()
    return total_sq ** 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", required=True, choices=[
        "SGD_exponential_learning_rate", "SGD_fixed_learning_rate", "Adam"])
    p.add_argument("--root_dir", default="cifar-10")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--subset_size", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default="results")
    a = p.parse_args()

    tag = a.schedule.replace("SGD_", "").replace("_learning_rate", "")
    setup_seed(a.seed)
    transform = build_transform()
    train_dataset, _ = get_cifar10_dataset(a.root_dir, transform)
    N = len(train_dataset)
    data_arr, labels_arr = train_dataset.data, train_dataset.labels

    rng = np.random.RandomState(a.seed)
    subset_idx = np.sort(rng.choice(N, size=a.subset_size, replace=False))

    model = get_model(a.model, 10).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(a.schedule, model.parameters(), Args())
    ce_none = nn.CrossEntropyLoss(reduction="none")
    ce_single = nn.CrossEntropyLoss()

    os.makedirs(a.out_dir, exist_ok=True)
    losses = np.full((N, a.epochs), np.nan, dtype=np.float32)
    inner = np.full((a.subset_size, a.epochs), np.nan, dtype=np.float64)
    norms = np.full((a.subset_size, a.epochs), np.nan, dtype=np.float64)
    eta_trace = np.full(a.epochs, np.nan, dtype=np.float64)

    subset_data = data_arr[subset_idx] if hasattr(data_arr, "__getitem__") else None

    for epoch in range(a.epochs):
        t0 = time.time()
        # ---- 1. ordinary SGD training pass ----
        model.train()
        loader = DataLoader(train_dataset, batch_size=a.batch_size, shuffle=True,
                             num_workers=a.num_workers, drop_last=True)
        for inputs, labels, _ in loader:
            if inputs.size(0) == 1:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = ce_single(outputs, labels)
            loss.backward()
            optimizer.step()
        eta_trace[epoch] = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ---- 2. full-dataset gradient-accumulation pass at frozen theta^(t) (eval mode) ----
        model.eval()
        for p_ in model.parameters():
            p_.grad = None
        full_loader = DataLoader(train_dataset, batch_size=256, shuffle=False,
                                  num_workers=a.num_workers)
        for inputs, labels, indices in full_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            per_sample = ce_none(outputs, labels)
            losses[indices.numpy(), epoch] = per_sample.detach().float().cpu().numpy()
            (per_sample.sum() / N).backward()  # accumulates (1/N) sum_i g_i into .grad
        g_pop = flat_grad_dict(model)

        # ---- 3. per-instance gradients for the tracked subset, dotted against g_pop ----
        for j, idx in enumerate(subset_idx):
            img, label = train_dataset.data[idx], train_dataset.labels[idx]
            from PIL import Image
            img_t = transform(Image.fromarray(img)).unsqueeze(0).to(device)
            label_t = torch.tensor([label], device=device)
            model.zero_grad(set_to_none=True)
            out = model(img_t)
            l = ce_single(out, label_t)
            l.backward()
            inner[j, epoch] = dot_with(model, g_pop)
            norms[j, epoch] = grad_norm(model)

        print(f"[drift:{tag}] epoch {epoch+1}/{a.epochs} eta={eta_trace[epoch]:.5f} "
              f"time={time.time()-t0:.1f}s", flush=True)

    np.save(os.path.join(a.out_dir, f"drift_{tag}_losses.npy"), losses)
    np.save(os.path.join(a.out_dir, f"drift_{tag}_inner.npy"), inner)
    np.save(os.path.join(a.out_dir, f"drift_{tag}_norms.npy"), norms)
    np.save(os.path.join(a.out_dir, f"drift_{tag}_subset_idx.npy"), subset_idx)
    np.save(os.path.join(a.out_dir, f"drift_{tag}_eta.npy"), eta_trace)
    print(f"DONE {tag}: saved drift_{tag}_{{losses,inner,norms,subset_idx,eta}}.npy to {a.out_dir}")


if __name__ == "__main__":
    main()

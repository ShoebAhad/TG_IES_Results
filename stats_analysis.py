"""Seed-level paired statistical analysis for the 3-seed settings in this paper's matrix
(primary: ResNet-18/CIFAR-10/SGD-exponential; secondary: ResNet-18/CIFAR-10/SGD-fixed and
VGG-16/CIFAR-10/SGD-exponential; Adam, extended to 3 seeds 2026-08-27; CIFAR-100,
extended to 3 seeds 2026-08-31; DenseNet-121, extended to 3 seeds 2026-09-02; ViT-Tiny on
CIFAR-10 and CIFAR-100, both 3-seed from the start, 2026-09-03).

Each results/*.csv run was produced with setup_seed(seed) called once, before model
init and data loading (common.py), so all four methods sharing a --seed value share
the same initial weights AND the same data-shuffling order for that seed -- a genuine
matched/paired design across methods, not four independent samples. Paired statistics
(not independent two-sample tests) are the correct tool given this, and increase power
relative to treating the 3 seeds x 4 methods as 12 independent draws.

With n=3 seeds, no test can reach conventional significance thresholds from the sign
pattern alone (the exact two-sided sign-test's smallest achievable p-value is 2/8=0.25,
reached only when all 3 seeds agree in sign) and t-based confidence intervals are wide
and sensitive to the normality assumption. We report both a parametric summary (paired
mean difference, t(df=2) 95% CI, Cohen's d_z) and a distribution-free exact sign test,
and are explicit throughout about what n=3 can and cannot establish -- see the
Statistical Analysis appendix section this script's output feeds.
"""
import csv
import glob
import itertools
import math
import os

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO, "results")

SETTINGS = {
    "primary_resnet18_exp": "cifar10_resnet18_SGD_exponential_learning_rate",
    "secondary_resnet18_fixed": "cifar10_resnet18_SGD_fixed_learning_rate",
    "secondary_vgg16_exp": "cifar10_vgg16_SGD_exponential_learning_rate",
    "adam": "cifar10_resnet18_Adam",
    "cifar100_resnet18_exp": "cifar100_resnet18_SGD_exponential_learning_rate",
    "densenet121_exp": "cifar10_densenet121_SGD_exponential_learning_rate",
    "vit_cifar10": "cifar10_vit_tiny_cifar_AdamW_warmup_cosine",
    "vit_cifar100": "cifar100_vit_tiny_cifar_AdamW_warmup_cosine",
}
METHODS = ["baseline", "ies_shipped", "ies_alg1", "tgies"]
SEEDS = [0, 1, 2]
COMPARISONS = [("tgies", "baseline"), ("tgies", "ies_shipped"), ("tgies", "ies_alg1")]

# t-distribution two-sided 97.5th-percentile critical value, df=2 (exact constant;
# avoids a scipy dependency for a single well-known value).
T_CRIT_DF2 = 4.302652729911275


def load_flop_lookup():
    path = os.path.join(RESULTS_DIR, "flop_summary.csv")
    lookup = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            lookup[row["run_name"]] = float(row["flop_saved_pct"])
    return lookup


def load_run(method, suffix, seed, flop_lookup):
    run_name = f"{method}_{suffix}_seed{seed}"
    path = os.path.join(RESULTS_DIR, run_name + ".csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    best_acc = max(float(r["test_acc_pct"]) for r in rows)
    N = int(rows[0]["active_set_size"])
    epochs = len(rows)
    cum_backprop = int(rows[-1]["cum_backprop_instances"])
    backprop_saved = 100.0 * (1 - cum_backprop / (N * epochs))
    flop_saved = flop_lookup.get(run_name)
    return dict(best_test_acc_pct=best_acc, backprop_saved_pct=backprop_saved,
                flop_saved_pct=flop_saved)


def exact_sign_test_p(diffs):
    """Two-sided exact sign test. Ties (diff==0) are dropped (standard sign-test
    convention); if fewer than 1 non-zero diff remains, returns None."""
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
    nonzero = [s for s in signs if s != 0]
    n = len(nonzero)
    if n == 0:
        return None
    k = sum(1 for s in nonzero if s > 0)  # number of positive diffs
    # Binomial(n, 0.5) two-sided exact p-value
    def binom_pmf(i, n):
        return math.comb(n, i) / (2 ** n)
    p_at_least_k = sum(binom_pmf(i, n) for i in range(k, n + 1))
    p_at_most_k = sum(binom_pmf(i, n) for i in range(0, k + 1))
    p = 2 * min(p_at_least_k, p_at_most_k)
    return min(p, 1.0)


def paired_stats(vals_a, vals_b):
    """vals_a, vals_b: same-length lists, paired by seed. Returns dict of stats
    for (a - b)."""
    diffs = [a - b for a, b in zip(vals_a, vals_b)]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    if n > 1:
        var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        sd_diff = math.sqrt(var)
    else:
        sd_diff = float("nan")
    if sd_diff > 0:
        se = sd_diff / math.sqrt(n)
        ci_halfwidth = T_CRIT_DF2 * se if n == 3 else float("nan")
        cohens_dz = mean_diff / sd_diff
    else:
        # sd_diff == 0: every seed shows an identical difference -- CI collapses to a
        # point and Cohen's d_z is undefined (division by zero); report accordingly.
        ci_halfwidth = 0.0
        cohens_dz = float("inf") if mean_diff != 0 else float("nan")
    p_sign = exact_sign_test_p(diffs)
    return dict(diffs=diffs, mean_diff=mean_diff, sd_diff=sd_diff,
                ci_low=mean_diff - ci_halfwidth, ci_high=mean_diff + ci_halfwidth,
                cohens_dz=cohens_dz, p_sign=p_sign)


def main():
    flop_lookup = load_flop_lookup()
    out_rows = []
    for setting_name, suffix in SETTINGS.items():
        data = {}
        for method in METHODS:
            for seed in SEEDS:
                r = load_run(method, suffix, seed, flop_lookup)
                if r is not None:
                    data[(method, seed)] = r
        for metric in ["best_test_acc_pct", "backprop_saved_pct", "flop_saved_pct"]:
            for method_a, method_b in COMPARISONS:
                vals_a, vals_b = [], []
                ok = True
                for seed in SEEDS:
                    ra = data.get((method_a, seed))
                    rb = data.get((method_b, seed))
                    if ra is None or rb is None or ra[metric] is None or rb[metric] is None:
                        ok = False
                        break
                    vals_a.append(ra[metric])
                    vals_b.append(rb[metric])
                if not ok:
                    continue
                stats = paired_stats(vals_a, vals_b)
                out_rows.append(dict(
                    setting=setting_name, metric=metric, method_a=method_a, method_b=method_b,
                    n=len(vals_a), mean_a=sum(vals_a) / len(vals_a), mean_b=sum(vals_b) / len(vals_b),
                    mean_diff=stats["mean_diff"], sd_diff=stats["sd_diff"],
                    ci95_low=stats["ci_low"], ci95_high=stats["ci_high"],
                    cohens_dz=stats["cohens_dz"], p_sign_exact=stats["p_sign"],
                    per_seed_diffs=";".join(f"{d:.4f}" for d in stats["diffs"]),
                ))

    out_path = os.path.join(RESULTS_DIR, "statistical_analysis.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"{'setting':26s} {'metric':20s} {'a vs b':22s} {'mean_diff':>10s} "
          f"{'95% CI':>20s} {'d_z':>7s} {'p_sign':>7s}")
    for row in out_rows:
        ab = f"{row['method_a']} - {row['method_b']}"
        ci = f"[{row['ci95_low']:.2f}, {row['ci95_high']:.2f}]"
        dz = f"{row['cohens_dz']:.2f}" if math.isfinite(row['cohens_dz']) else "inf"
        p = f"{row['p_sign_exact']:.3f}" if row['p_sign_exact'] is not None else "NA"
        print(f"{row['setting']:26s} {row['metric']:20s} {ab:22s} {row['mean_diff']:10.3f} "
              f"{ci:>20s} {dz:>7s} {p:>7s}")
    print(f"\nWrote {out_path}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()

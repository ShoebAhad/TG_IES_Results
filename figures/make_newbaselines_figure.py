"""New-baseline comparison figure/table: TG-IES vs. InfoBatch / BLS-InfoBatch /
AlignPrune on the primary cell (ResNet-18/CIFAR-10/SGD-exp, seeds 0-2),
VGG-16/CIFAR-10/SGD-exp seed0, and CIFAR-100/ResNet-18 seed0.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.dirname(__file__)

METHODS = ["baseline", "ies_shipped", "ies_alg1", "tgies", "infobatch", "bls_infobatch", "alignprune"]
LABELS = {
    "baseline": "Baseline", "ies_shipped": "IES (shipped)", "ies_alg1": "IES (literal)", "tgies": "TG-IES (ours)",
    "infobatch": "InfoBatch", "bls_infobatch": "BLS-InfoBatch", "alignprune": "AlignPrune",
}
COLORS = {
    # first four kept in sync with make_figures.py's COLORS for the same methods
    # (gray/orange/green/navy -- distinct hues, not shades of one color family).
    "baseline": "#8C8C8C", "ies_shipped": "#E69F00", "ies_alg1": "#009E73", "tgies": "#0A2C55",
    "infobatch": "#56B4E9", "bls_infobatch": "#CC2936", "alignprune": "#7A3E9D",
}
MARKERS = {"baseline": "D", "ies_shipped": "v", "ies_alg1": "^", "tgies": "*",
           "infobatch": "o", "bls_infobatch": "s", "alignprune": "P"}


def main():
    s = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    s = s[~s["run_name"].str.contains("ablation|epspct|_verify|_STALE", regex=True, na=False)]

    print("=== Primary cell: ResNet-18/CIFAR-10/SGD-exp, mean +/- std over seeds ===")
    primary = s[(s["model"] == "resnet18") & (s["optimizer"] == "SGD_exponential_learning_rate")
                & (s["dataset"] == "cifar10") & (s["method"].isin(METHODS))]
    g = primary.groupby("method")[["best_test_acc_pct", "backprop_saved_pct"]].agg(["mean", "std", "count"])
    print(g.reindex(METHODS))

    print("\n=== VGG-16/CIFAR-10/SGD-exp, seed0 ===")
    vgg = s[(s["model"] == "vgg16") & (s["optimizer"] == "SGD_exponential_learning_rate")
            & (s["seed"] == 0) & (s["method"].isin(METHODS))]
    print(vgg[["method", "best_test_acc_pct", "backprop_saved_pct"]].sort_values("method"))

    print("\n=== CIFAR-100/ResNet-18, seed0 ===")
    c100 = s[(s["dataset"] == "cifar100") & (s["seed"] == 0) & (s["method"].isin(METHODS))]
    print(c100[["method", "best_test_acc_pct", "backprop_saved_pct"]].sort_values("method"))

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for m in METHODS:
        rows = primary[primary["method"] == m]
        if rows.empty:
            continue
        ax.scatter(rows["backprop_saved_pct"], rows["best_test_acc_pct"], color=COLORS[m],
                    marker=MARKERS[m], s=45, edgecolors="0.25", linewidths=0.5, label=LABELS[m])
    ax.set_xlabel("Backprop Instances Saved (%)", fontweight="bold")
    ax.set_ylabel("Best Test Accuracy (%)", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6, frameon=True, loc="lower left")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, "fig6_newbaselines.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig6_newbaselines.png"), dpi=300)
    print(f"\nSaved {OUT_DIR}/fig6_newbaselines.{{pdf,png}}")


if __name__ == "__main__":
    main()

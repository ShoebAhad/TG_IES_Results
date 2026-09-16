"""Batch-size sensitivity figure for the measured-efficiency appendix (app:efficiency).

NEXT_STEPS.md item 10 collected results/batch_size_sweep_summary.csv (12 rows:
{baseline, ies_shipped, tgies} x {32,64,128,256}, short 8-epoch instrumented runs)
but it was never integrated into the paper. This addresses the reviewer's
"wall-clock generalization" concern directly: the primary efficiency table
(tab:efficiency) reports one batch size (64) on one GPU; this checks whether
TG-IES's overhead/savings pattern holds across batch sizes, using data already
collected at $0 additional GPU cost.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.dirname(__file__)

METHODS = ["baseline", "ies_shipped", "tgies"]
LABELS = {"baseline": "Baseline", "ies_shipped": "IES (shipped)", "tgies": "TG-IES (ours)"}
COLORS = {"baseline": "0.5", "ies_shipped": "#009E73", "tgies": "#0A2C55"}
MARKERS = {"baseline": "o", "ies_shipped": "s", "tgies": "^"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelweight": "bold",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.75", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256])
    ax.set_xticklabels(["32", "64", "128", "256"])
    ax.set_xlabel("Batch size", fontweight="bold")


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "batch_size_sweep_summary.csv"))

    panels = [
        ("peak_mem_used_delta_mib", "Peak Mem.\nDelta (MiB)"),
        ("mean_gpu_util_pct", "Mean GPU\nUtil. (%)"),
        ("mean_epoch_time_s", "Mean Epoch\nTime (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ax, (col, ylabel) in zip(axes, panels):
        for m in METHODS:
            sub = df[df["method"] == m].sort_values("batch_size")
            ax.plot(sub["batch_size"], sub[col], color=COLORS[m], marker=MARKERS[m],
                     markersize=4, linewidth=1.3, label=LABELS[m], zorder=3)
        ax.set_ylabel(ylabel, fontweight="bold")
        style_axes(ax)

    axes[-1].legend(loc="upper center", bbox_to_anchor=(-0.85, 1.28), ncol=3, frameon=False,
                     columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(OUT_DIR, "fig_batchsize_sensitivity.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig_batchsize_sensitivity.png"), dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR}/fig_batchsize_sensitivity.{{pdf,png}}")

    # Print the numbers the paper text below cites, computed here rather than by hand.
    piv_mem = df.pivot(index="batch_size", columns="method", values="peak_mem_used_delta_mib")
    piv_util = df.pivot(index="batch_size", columns="method", values="mean_gpu_util_pct")
    piv_time = df.pivot(index="batch_size", columns="method", values="mean_epoch_time_s")
    print("\nPeak mem delta (MiB):\n", piv_mem)
    print("\nMean GPU util (%):\n", piv_util)
    print("\nMean epoch time (s):\n", piv_time)
    print("\nTG-IES / baseline peak-mem ratio by batch size:")
    print((piv_mem["tgies"] / piv_mem["baseline"]).round(3))
    print("\nTG-IES / baseline epoch-time ratio by batch size (short 8-epoch run, NOT full-run wall-clock):")
    print((piv_time["tgies"] / piv_time["baseline"]).round(3))


if __name__ == "__main__":
    main()

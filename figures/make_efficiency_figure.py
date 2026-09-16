"""Measured-efficiency figure for sec:efficiency (Table 4 currently has no companion
figure). Reads results/efficiency_combined_summary.csv (written by
measure_efficiency.py from the real full-run wall-clock/backprop data plus the
short instrumented-run memory/power measurements; see paper Section 9.2 for
methodology and caveats) and renders each method's wall-clock time, peak GPU
memory, and estimated energy as a percentage of the no-removal baseline.

Normalizing to baseline=100% on one shared axis is what makes the paper's central
efficiency finding visible at a glance: wall-clock and energy both drop toward
TG-IES (as expected from its higher backprop savings), but peak memory rises
instead of falling for TG-IES specifically -- the one bar that goes the "wrong"
way is the point of the figure, not a decoration.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.dirname(__file__)

METHODS = ["baseline", "ies_shipped", "ies_alg1", "tgies"]
LABELS = {
    "baseline": "Baseline",
    "ies_shipped": "IES\n(shipped)",
    "ies_alg1": "IES\n(literal)",
    "tgies": "TG-IES\n(ours)",
}
METRIC_COLS = {
    "Wall-clock": "mean_full_run_wallclock_min",
    "Peak Mem.": "peak_mem_used_delta_mib",
    "Energy (est.)": "est_full_run_energy_wh",
}
METRIC_COLORS = {"Wall-clock": "#5FA0DB", "Peak Mem.": "#C2571B", "Energy (est.)": "#0A2C55"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "efficiency_combined_summary.csv")).set_index("method")
    baseline = df.loc["baseline"]

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    n_metrics = len(METRIC_COLS)
    bar_w = 0.8 / n_metrics
    x = range(len(METHODS))

    for i, (metric_name, col) in enumerate(METRIC_COLS.items()):
        vals_pct = [100.0 * df.loc[m, col] / baseline[col] for m in METHODS]
        offs = [xi + (i - (n_metrics - 1) / 2) * bar_w for xi in x]
        bars = ax.bar(offs, vals_pct, width=bar_w * 0.92, color=METRIC_COLORS[metric_name],
                       edgecolor="0.25", linewidth=0.5, label=metric_name, zorder=3)
        for b, v in zip(bars, vals_pct):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", va="bottom",
                     fontsize=5.5, fontweight="bold", color="0.15")

    ax.axhline(100, color="0.4", linestyle=":", linewidth=1.0, zorder=2)
    ax.text(len(METHODS) - 0.5 + 0.05, 100, "baseline", fontsize=5.5, color="0.35",
             va="center", ha="left", style="italic")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylabel("% of Baseline", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.75", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(140, ax.get_ylim()[1] * 1.08))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False,
               columnspacing=1.2, handletextpad=0.5, fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, "fig7_efficiency.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig7_efficiency.png"), dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR}/fig7_efficiency.{{pdf,png}}")


if __name__ == "__main__":
    main()

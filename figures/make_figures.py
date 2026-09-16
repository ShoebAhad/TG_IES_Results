"""IEEE single-column figures for the baseline / ies_shipped / ies_alg1 / tgies
comparison (primary setting: resnet18, CIFAR-10, SGD_exponential_learning_rate).

Averages the 3 seeds per method, then renders:
  fig1_accuracy.pdf/png   - test accuracy vs epoch
  fig2_backprop.pdf/png   - cumulative backprop instances vs epoch

Style: distinct-hue palette (gray/blue/teal/navy), not a monochrome ramp,
since fig_tradeoff() is a scatter plot where color is the only cue within a
marker-shape group; fig_accuracy/fig_backprop additionally use one linestyle
per method (works in grayscale print too). Bold labels/legend, IEEE
single-column width (3.3in). Colors kept in sync with
make_newbaselines_figure.py's COLORS for these same four methods.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.dirname(__file__)
SEEDS = [0, 1, 2]
METHODS = ["baseline", "ies_shipped", "ies_alg1", "tgies"]
LABELS = {
    "baseline": "Baseline",
    "ies_shipped": "IES (shipped)",
    "ies_alg1": "IES (literal)",
    "tgies": "TG-IES (ours)",
}
# distinct hues (gray/orange/green/navy), not shades of one color family, so the
# four methods stay separable in a scatter plot where color is the only cue.
# (Previous palette put ies_shipped, ies_alg1, and tgies all in the blue family
# -- #3B76B5/#1B9AAA/#0A2C55 -- which read as near-identical at small size or
# under color-vision deficiency; this one spans four distinct hues instead.)
COLORS = {
    "baseline": "#8C8C8C",
    "ies_shipped": "#E69F00",
    "ies_alg1": "#009E73",
    "tgies": "#0A2C55",
}
STYLES = {
    "baseline": ":",
    "ies_shipped": "-.",
    "ies_alg1": "--",
    "tgies": "-",
}
WIDTHS = {
    "baseline": 1.0,
    "ies_shipped": 1.1,
    "ies_alg1": 1.2,
    "tgies": 1.6,
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "legend.handlelength": 2.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIGSIZE = (3.3, 2.35)  # IEEE single-column width


def load_mean(method, opt="SGD_exponential_learning_rate", model="resnet18"):
    dfs = []
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, f"{method}_cifar10_{model}_{opt}_seed{seed}.csv")
        dfs.append(pd.read_csv(path))
    stacked = np.stack([d.sort_values("epoch").reset_index(drop=True) for d in dfs])
    epochs = dfs[0]["epoch"].values
    mean_df = sum(d.sort_values("epoch").reset_index(drop=True) for d in dfs) / len(dfs)
    mean_df["epoch"] = epochs
    return mean_df


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.75", alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def fig_accuracy():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    dfs = {m: load_mean(m) for m in METHODS}
    for m in METHODS:
        df = dfs[m]
        ax.plot(df["epoch"], df["test_acc_pct"], label=LABELS[m], color=COLORS[m],
                linestyle=STYLES[m], linewidth=WIDTHS[m], zorder=3 if m == "tgies" else 2)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontweight="bold")
    ax.set_xlim(0, 200)
    ax.set_ylim(30, 100)
    style_axes(ax)

    # zoomed inset on the converged plateau, where the four curves separate.
    # a rolling mean damps per-epoch SGD noise, which otherwise dwarfs the
    # <1pp gap between methods once the y-range is this tight.
    axins = ax.inset_axes([0.15, 0.53, 0.4, 0.4])
    zx0, zx1, win = 150, 200, 5
    smoothed = {}
    for m in METHODS:
        df = dfs[m]
        mask = (df["epoch"] >= zx0 - win) & (df["epoch"] <= zx1)
        s = df["test_acc_pct"][mask].rolling(win, min_periods=1).mean()
        smoothed[m] = s[df["epoch"][mask] >= zx0]
        x = df["epoch"][mask][df["epoch"][mask] >= zx0]
        axins.plot(x, smoothed[m], color=COLORS[m], linestyle=STYLES[m],
                   linewidth=WIDTHS[m] + 0.2, zorder=3 if m == "tgies" else 2)
    zy = pd.concat(smoothed.values())
    ylo, yhi = zy.min() - 0.05, zy.max() + 0.05
    axins.set_xlim(zx0, zx1)
    axins.set_ylim(ylo, yhi)
    axins.set_xticks([150, 200])
    axins.set_yticks([round(ylo, 1), round(yhi, 1)])
    axins.tick_params(labelsize=5.5, width=0.6, length=2)
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
    for label in axins.get_xticklabels() + axins.get_yticklabels():
        label.set_fontweight("bold")
    # explicit scale callout: the inset's y-range is a <1pp window, easy to
    # misread as sharing the main axis's 30-100% scale without this label.
    axins.text(0.03, 0.94, f"y: {ylo:.2f}–{yhi:.2f}%", transform=axins.transAxes,
                fontsize=4.8, fontweight="bold", color="0.25", ha="left", va="top")
    ax.indicate_inset_zoom(axins, edgecolor="0.4", linewidth=0.6, alpha=0.8)

    leg = ax.legend(loc="lower right", frameon=True, framealpha=0.9,
                     edgecolor="0.8", borderpad=0.4, labelspacing=0.3, handletextpad=0.5)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, "fig1_accuracy.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig1_accuracy.png"), dpi=300)
    plt.close(fig)


def fig_backprop():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    dfs = {m: load_mean(m) for m in METHODS}
    for m in METHODS:
        df = dfs[m]
        ax.plot(df["epoch"], df["cum_backprop_instances"] / 1e6, label=LABELS[m], color=COLORS[m],
                linestyle=STYLES[m], linewidth=WIDTHS[m], zorder=3 if m == "tgies" else 2)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Cumulative Backprop\nInstances " + r"($\times 10^{6}$)", fontweight="bold")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 10.5)
    style_axes(ax)

    # inset on the three removal methods only (baseline is trivially highest
    # throughout and would force the inset back out to the full y-range):
    # the three curves nearly overlap by eye in the main axes from epoch ~75
    # on, and their final ordering -- including a crossover around epoch 130,
    # where TG-IES overtakes IES (literal) to become lowest -- is otherwise
    # invisible.
    axins = ax.inset_axes([0.42, 0.1, 0.55, 0.5])
    zx0, zx1 = 100, 200
    removal_methods = [m for m in METHODS if m != "baseline"]
    for m in removal_methods:
        df = dfs[m]
        mask = (df["epoch"] >= zx0) & (df["epoch"] <= zx1)
        axins.plot(df["epoch"][mask], df["cum_backprop_instances"][mask] / 1e6, color=COLORS[m],
                   linestyle=STYLES[m], linewidth=WIDTHS[m] + 0.2, zorder=3 if m == "tgies" else 2)
    ylo = min(dfs[m]["cum_backprop_instances"][dfs[m]["epoch"] >= zx0].min() / 1e6
               for m in removal_methods) - 0.1
    yhi = max(dfs[m]["cum_backprop_instances"][dfs[m]["epoch"] <= zx1].max() / 1e6
               for m in removal_methods) + 0.1
    axins.set_xlim(zx0, zx1)
    axins.set_ylim(ylo, yhi)
    axins.set_xticks([100, 200])
    axins.set_yticks([round(ylo, 1), round(yhi, 1)])
    axins.tick_params(labelsize=5.5, width=0.6, length=2)
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
    for label in axins.get_xticklabels() + axins.get_yticklabels():
        label.set_fontweight("bold")
    axins.text(0.97, 0.06, "3 removal\nmethods only", transform=axins.transAxes,
                fontsize=4.8, fontweight="bold", color="0.25", ha="right", va="bottom")
    ax.indicate_inset_zoom(axins, edgecolor="0.4", linewidth=0.6, alpha=0.8)

    leg = ax.legend(loc="upper left", frameon=True, framealpha=0.9,
                     edgecolor="0.8", borderpad=0.4, labelspacing=0.3, handletextpad=0.5)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, "fig2_backprop.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig2_backprop.png"), dpi=300)
    plt.close(fig)


SETTINGS = [
    ("resnet18", "SGD_exponential_learning_rate", "ResNet-18, SGD(exp)", "o"),
    ("vgg16", "SGD_exponential_learning_rate", "VGG-16, SGD(exp)", "s"),
    ("resnet18", "SGD_fixed_learning_rate", "ResNet-18, SGD(fixed)", "^"),
]


def fig_tradeoff():
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    # summary.csv is append-only and also holds ablation (_ablation_deltaonly,
    # _ablation_kstaronly), sensitivity-sweep (_epspct*), and other-dataset
    # (cifar100) runs that happen to share (model, optimizer, method) with a
    # primary/secondary cell -- filtering on those three columns alone silently
    # pulls those extra rows into this plot. Restrict to the exact canonical
    # run_name for each (method, seed) at each of the three CIFAR-10 settings.
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for model, opt, _, marker in SETTINGS:
        expected = {f"{m}_cifar10_{model}_{opt}_seed{seed}" for m in METHODS for seed in SEEDS}
        sub = summary[summary["run_name"].isin(expected)]
        for m in METHODS:
            rows = sub[sub["method"] == m]
            if rows.empty:
                continue
            assert len(rows) == len(SEEDS), (
                f"expected {len(SEEDS)} seeds for {m}/{model}/{opt}, found {len(rows)}: "
                f"{sorted(rows['run_name'])}")
            ax.scatter(rows["backprop_saved_pct"], rows["best_test_acc_pct"],
                       color=COLORS[m], marker=marker, s=32,
                       edgecolors="0.25", linewidths=0.5,
                       zorder=3 if m == "tgies" else 2)

    ax.set_xlabel("Backprop Instances Saved (%)", fontweight="bold")
    ax.set_ylabel("Best Test Accuracy (%)", fontweight="bold")
    ax.set_xlim(-3, 60)
    ax.set_ylim(91, 96.6)
    style_axes(ax)

    # marker shape (setting) is documented in the caption, not a second legend box,
    # to keep a single-column scatter from needing two overlapping legends
    method_handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=COLORS[m],
                                  markeredgecolor="0.25", markeredgewidth=0.5, markersize=5,
                                  label=LABELS[m]) for m in METHODS]
    leg1 = ax.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                      ncol=2, frameon=True, framealpha=0.9, edgecolor="0.8",
                      borderpad=0.35, labelspacing=0.25, handletextpad=0.4, columnspacing=0.8)
    leg1.get_frame().set_linewidth(0.6)

    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, "fig3_tradeoff.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig3_tradeoff.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_accuracy()
    fig_backprop()
    fig_tradeoff()
    print("Saved figures to", OUT_DIR)

"""Gradient-norm-floor percentile sensitivity sweep figure (task: sensitivity analysis).

Reads results/summary.csv rows written with --tag epspct{P} for tgies on
vgg16/resnet18 CIFAR-10 SGD-exponential seed0, and plots backprop-saved % and
accuracy vs. eps_percentile, faceted by architecture -- directly testing the
paper's own hypothesis for the (now-corrected, see main.tex sec:results)
VGG-16 discussion: whether the gradient-norm-floor proxy is architecture-
sensitive enough to explain calibration differences.
"""
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.dirname(__file__)

COLORS = {"vgg16": "#1F5C99", "resnet18": "#0A2C55"}
MARKERS = {"vgg16": "s", "resnet18": "o"}


def main():
    s = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    rows = []
    for _, r in s.iterrows():
        m = re.search(r"_epspct(\d+)$", r["run_name"])
        if m and r["method"] == "tgies":
            rows.append({"model": r["model"], "pct": int(m.group(1)),
                         "acc": r["best_test_acc_pct"], "saved": r["backprop_saved_pct"]})
    if not rows:
        print("No epspct-tagged runs found in summary.csv yet.")
        return
    df = pd.DataFrame(rows).sort_values(["model", "pct"])
    print(df)

    DEFAULT_PCT = 20  # this paper's fixed default (sec:selfcalib-scope), used in every
                       # non-sensitivity-sweep run -- marked so a reader can see at a
                       # glance where the chosen default sits on the swept curve.

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for model in ["resnet18", "vgg16"]:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        ax1.plot(sub["pct"], sub["saved"], marker=MARKERS[model], color=COLORS[model],
                  label=model, linewidth=1.3, markersize=5)
        ax2.plot(sub["pct"], sub["acc"], marker=MARKERS[model], color=COLORS[model],
                  label=model, linewidth=1.3, markersize=5)

    ax1.set_xlabel(r"$\varepsilon_0$ percentile", fontweight="bold")
    ax1.set_ylabel("Backprop Saved (%)", fontweight="bold")
    ax2.set_xlabel(r"$\varepsilon_0$ percentile", fontweight="bold")
    ax2.set_ylabel("Best Test Acc. (%)", fontweight="bold")
    for ax in (ax1, ax2):
        ax.axvline(DEFAULT_PCT, color="0.55", linestyle=":", linewidth=1.0, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=6.5, frameon=True)
    ax1.text(DEFAULT_PCT, ax1.get_ylim()[1], " default", fontsize=6, color="0.4",
              ha="left", va="top", style="italic")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, "fig5_sensitivity.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig5_sensitivity.png"), dpi=300)
    print(f"Saved {OUT_DIR}/fig5_sensitivity.{{pdf,png}}")


if __name__ == "__main__":
    main()

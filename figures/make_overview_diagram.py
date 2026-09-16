"""Conceptual pipeline diagram: how TG-IES turns a noisy per-instance loss
trajectory into a removal decision, and where the paper's two theory-guided
calibration paths (threshold delta, patience K*) plug into that decision --
plus the link (Section 7.1) between the noisy mastery check the algorithm
actually runs and the gradient-norm condition the safety certificate needs,
proved under further stated idealizations (Remark 7.12). This is a
schematic, not a data plot: no numbers here are measured, they are the
symbols/labels used throughout the paper (Eq. 1, Eq. 10, Corollaries
5.3/7.3, Theorem 7.1, Theorem 7.10, Remark 7.12).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = os.path.dirname(__file__)

# Single-hue light-blue palette (no orange/purple/red anywhere): a pale-to-
# medium blue ramp for the five pipeline boxes, a matching pale tint for the
# two calibration boxes (distinguished by a bold navy border, not a new hue),
# and a white/navy-outlined box for the open-gap callout. All text is dark
# navy, bold, on light fills throughout for maximum contrast.
PALE1 = "#EAF3FB"   # trajectory
PALE2 = "#D3E7F7"   # signal
PALE3 = "#B3D4F0"   # mastery check (hub)
PALE4 = "#96C1E8"   # passive
PALE5 = "#7BAEDE"   # active-set update
CALIB = "#D6EAF8"   # delta / K* calibration boxes
BORDER = "#2E5E8C"  # box borders and main-flow arrows
NAVY = "#0B2545"    # all text, and the open-gap accent
INK = NAVY

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def box(ax, xy, w, h, text, fc, ec=BORDER, fontsize=8.5, textcolor=NAVY,
        weight="bold", boxstyle="round,pad=0.02,rounding_size=0.06", zorder=3,
        lw=1.3, border_ls="-"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle, facecolor=fc,
                        edgecolor=ec, linewidth=lw, linestyle=border_ls, zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             color=textcolor, fontweight=weight, zorder=zorder + 1, linespacing=1.45)
    return (x, y, w, h)


def arrow(ax, p0, p1, color="0.2", lw=1.5, style="-|>", connectionstyle="arc3,rad=0.0",
          ls="-", zorder=2):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=color,
                          linewidth=lw, linestyle=ls, connectionstyle=connectionstyle,
                          zorder=zorder)
    ax.add_patch(a)


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def main():
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.5, 40)
    ax.axis("off")

    # --- row 1: main left-to-right pipeline ---
    y0, h0 = 19, 11

    b_traj = box(ax, (1, y0), 15, h0,
                  "Noisy per-instance\nloss trajectory\n" + r"$\mathbf{\tilde L_i^{(t)}}$",
                  PALE1, fontsize=8.3)

    b_signal = box(ax, (19, y0), 15, h0,
                    "2nd-order signal\n" + r"$\mathbf{\hat S_i=|\tilde\Delta^2 L_i^{(t)}|}$"
                    + "\n(Eq. 1)",
                    PALE2, fontsize=8.3)

    b_decide = box(ax, (37, y0), 17, h0,
                    "Mastery check\n" + r"$\mathbf{\hat S_i<\delta}$ for" + "\n"
                    + r"$\mathbf{K^*}$ consecutive epochs" + "\n(Algorithm 1)",
                    PALE3, fontsize=8.0)

    b_passive = box(ax, (58, y0), 15, h0,
                      "Instance marked\npassive\n(no backward pass)",
                      PALE4, fontsize=8.3)

    b_update = box(ax, (77, y0), 22, h0,
                     "Active-set update\n" + r"$\mathbf{\hat g^{(t)}=\frac{1}{N}\sum_{i\in A_t}g_i^{(t)}}$"
                     + "\n(Eq. 10, Thm. 7.1)",
                     PALE5, fontsize=8.0)

    for a, b in [(b_traj, b_signal), (b_signal, b_decide), (b_decide, b_passive),
                  (b_passive, b_update)]:
        arrow(ax, right(a), left(b), color=BORDER, lw=1.7)

    # feedback loop routed OVER the top of row 1: update changes theta, which
    # changes every instance's loss next epoch. Entry/exit points are offset
    # off-center so the vertical legs don't cross the boxes' centered text.
    fb_y = y0 + h0 + 8.2
    fb_x_right = 77 + 22  # exact top-right corner of b_update
    fb_x_left = 1         # exact top-left corner of b_traj
    arrow(ax, (fb_x_right, y0 + h0), (fb_x_right, fb_y), color="0.5", lw=1.3,
          style="-", zorder=1)
    arrow(ax, (fb_x_right, fb_y), (fb_x_left, fb_y), color="0.5", lw=1.3,
          style="-", zorder=1)
    arrow(ax, (fb_x_left, fb_y), (fb_x_left, y0 + h0), color="0.5", lw=1.3, zorder=1)
    ax.text((fb_x_right + fb_x_left) / 2, fb_y + 2.6,
             r"next epoch: $\mathbf{\theta^{(t+1)}}$ changes every instance's loss",
             ha="center", va="center", fontsize=7.8, color=INK, fontweight="bold")

    # --- row 2: two calibration inputs feeding the decision box (below) ---
    y1, h1 = 0, 9

    b_delta = box(ax, (19, y1), 15, h1,
                   "Noise-calibrated:\n" + r"$\mathbf{\delta=k\sqrt{6}\,\hat\sigma}$"
                   + "\n(Cor. 5.3)",
                   CALIB, fontsize=7.8, lw=2.0)
    b_kstar = box(ax, (37, y1), 17, h1,
                   "Patience seed:\n" + r"$\mathbf{K^*\leftarrow K_{safe}}$"
                   + "\n(Cor. 7.3)",
                   CALIB, fontsize=7.8, lw=2.0)

    arrow(ax, top(b_delta), (left(b_decide)[0] + 3, y0), color=BORDER, lw=1.5,
          connectionstyle="arc3,rad=-0.15")
    arrow(ax, top(b_kstar), (left(b_decide)[0] + 8, y0), color=BORDER, lw=1.5,
          connectionstyle="arc3,rad=-0.08")

    # --- bridge callout: dashed link from decide box down to a small
    # certificate box, marking that the link is now proved but only under
    # further stated idealizations (Remark 7.12), not unconditionally. No new
    # hue here either -- white fill, navy dashed border, navy text/symbol.
    b_cert = box(ax, (58, y1), 22, h1,
                  "Certificate needs " + r"$\|\mathbf{g_i^{(t)}}\|<\varepsilon$" + "\n"
                  + "(Thm. 7.1) -- follows from\n" + r"$\mathbf{\hat S_i<\delta}$"
                  + " under Thm. 7.10",
                  "white", ec=NAVY, fontsize=7.6, weight="bold", lw=1.8,
                  border_ls=(0, (4, 2)))

    arrow(ax, bottom(b_decide), top(b_cert), color=NAVY, lw=1.8, ls=(0, (3, 2)),
          connectionstyle="arc3,rad=0.15")
    mid_x = 71
    mid_y = 14.4
    ax.text(mid_x, mid_y, r"$\ast$", ha="center", va="center", fontsize=17, color=NAVY,
             fontweight="bold", zorder=5)
    ax.text(mid_x, mid_y - 3.6, "bridged, with\ncaveats (Rem. 7.12)", ha="center", va="center",
             fontsize=7.4, color=NAVY, fontweight="bold", linespacing=1.25)

    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT_DIR, "fig0_overview.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig0_overview.png"), dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR}/fig0_overview.{{pdf,png}}")


if __name__ == "__main__":
    main()

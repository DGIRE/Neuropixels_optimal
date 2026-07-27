"""
render_FIG_DEMO1_01.py — Render FIG-DEMO1-01.

Reads ONLY from 04_figures/figdata/. No scientific recomputation.

Output: output/FIG_DEMO1_01.pdf  (vector)
        output/FIG_DEMO1_01.png  (300 dpi raster)

Figure: Ethanol vs control phase-locking of OB units to the sniff (SNF) signal.
  Panel A: Per-unit scatter (x = SNF preferred phase [rad], y = MRL).
           Two conditions overlaid (ethanol solid / control open).
  Panel B: Paired animal-level MRL (ethanol vs control, n=5 animals).
           Lines connect paired animals. Stat annotation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

FIGDATA_DIR = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-1\04_figures\figdata")
OUTPUT_DIR  = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-1\04_figures\output")

# Okabe-Ito colorblind-safe palette
CTRL_COLOR = "#0072B2"   # blue
ETH_COLOR  = "#D55E00"   # vermillion

_PI_TICKS = [-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi]
_PI_LABELS = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]


def main() -> None:
    with open(FIGDATA_DIR / "FIG_DEMO1_01_units.json") as f:
        units = json.load(f)
    with open(FIGDATA_DIR / "FIG_DEMO1_01_animals.json") as f:
        animals = json.load(f)
    with open(FIGDATA_DIR / "FIG_DEMO1_01_stat.json") as f:
        stat = json.load(f)

    ctrl_u = [u for u in units if u["condition"] == "control"]
    eth_u  = [u for u in units if u["condition"] == "ethanol"]
    ctrl_a = {a["session_date"]: a["mrl"] for a in animals if a["condition"] == "control"}
    eth_a  = {a["session_date"]: a["mrl"] for a in animals if a["condition"] == "ethanol"}
    paired_sessions = sorted(set(ctrl_a) & set(eth_a))

    # -----------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    })

    fig = plt.figure(figsize=(11, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], figure=fig,
                           left=0.07, right=0.97, bottom=0.13, top=0.90, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # -----------------------------------------------------------------------
    # Panel A — per-unit scatter
    # -----------------------------------------------------------------------
    # Contract visual_encoding: control=dashed, ethanol=solid.
    # Adapted for scatter: control = open circles, ethanol = filled circles.
    ax1.scatter(
        [u["preferred_phase_rad"] for u in eth_u],
        [u["mrl"] for u in eth_u],
        facecolors=ETH_COLOR, edgecolors="none", s=5, alpha=0.3,
        marker="o", label=f"Ethanol (n = {len(eth_u)} units)",
        rasterized=True,
    )
    ax1.scatter(
        [u["preferred_phase_rad"] for u in ctrl_u],
        [u["mrl"] for u in ctrl_u],
        facecolors="none", edgecolors=CTRL_COLOR, s=12, alpha=0.5,
        linewidths=0.6, marker="o", label=f"Control (n = {len(ctrl_u)} units)",
        rasterized=True,
    )

    ax1.set_xlim(-math.pi, math.pi)
    ax1.set_ylim(0, ax1.get_ylim()[1] or 0.5)
    ax1.axhline(0, color="black", linewidth=0.6, zorder=0)
    ax1.set_xticks(_PI_TICKS)
    ax1.set_xticklabels(_PI_LABELS)
    ax1.set_xlabel("SNF sniff phase (rad)")
    # y-axis shows MRL; normalized spike rate not computable from frozen result objects
    # (no per-spike phase histograms stored in P3 — documented in figure manifest).
    ax1.set_ylabel("MRL (mean resultant length)")
    ax1.set_title("Per-unit phase-locking", fontsize=8, pad=4)
    ax1.legend(loc="upper right", framealpha=0.7, handlelength=0.8)

    stat_txt = (f"n = {stat['n_animals']} animals, "
                f"p = {stat['pvalue']:.4f} (n.s.), "
                f"r = {stat['effect_size']:.2f}")
    ax1.text(0.02, 0.97, stat_txt, transform=ax1.transAxes,
             fontsize=7, va="top", ha="left", style="italic")

    # -----------------------------------------------------------------------
    # Panel B — paired animal means
    # -----------------------------------------------------------------------
    x_ctrl, x_eth = 0, 1
    for sess in paired_sessions:
        ax2.plot([x_ctrl, x_eth], [ctrl_a[sess], eth_a[sess]],
                 color="gray", linewidth=0.8, alpha=0.7, zorder=1)
    ax2.scatter(
        [x_ctrl] * len(paired_sessions), [ctrl_a[s] for s in paired_sessions],
        c=CTRL_COLOR, s=50, zorder=3, edgecolors="white", linewidths=0.5,
    )
    ax2.scatter(
        [x_eth] * len(paired_sessions), [eth_a[s] for s in paired_sessions],
        c=ETH_COLOR, s=50, zorder=3, edgecolors="white", linewidths=0.5,
    )

    # Group means
    mean_ctrl = float(np.mean([ctrl_a[s] for s in paired_sessions]))
    mean_eth  = float(np.mean([eth_a[s]  for s in paired_sessions]))
    ax2.plot([x_ctrl, x_eth], [mean_ctrl, mean_eth],
             color="black", linewidth=1.5, zorder=4)
    ax2.scatter([x_ctrl, x_eth], [mean_ctrl, mean_eth],
                c=["black", "black"], s=90, zorder=5,
                edgecolors="white", linewidths=0.8, marker="D")

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(0, ax2.get_ylim()[1] or 0.2)
    ax2.set_xticks([x_ctrl, x_eth])
    ax2.set_xticklabels(["Control", "Ethanol"])
    ax2.set_ylabel("Animal-mean MRL")
    ax2.set_title("Animal-level comparison", fontsize=8, pad=4)

    pval = stat["pvalue"]
    r_val = stat["effect_size"]
    sig_str = "n.s." if pval >= 0.05 else f"p = {pval:.4f}"
    ax2.text(0.5, 0.97, f"p = {pval:.4f} ({sig_str})\nr = {r_val:.2f}, n = {stat['n_animals']}",
             transform=ax2.transAxes, fontsize=7, va="top", ha="center", style="italic")

    legend_elements = [
        mpatches.Patch(facecolor=CTRL_COLOR, label="Control"),
        mpatches.Patch(facecolor=ETH_COLOR, label="Ethanol"),
    ]
    ax2.legend(handles=legend_elements, fontsize=7, loc="upper right",
               handlelength=0.8, framealpha=0.7)

    # -----------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "FIG_DEMO1_01.pdf", dpi=300, format="pdf")
    fig.savefig(OUTPUT_DIR / "FIG_DEMO1_01.png", dpi=300, format="png")
    plt.close(fig)
    print(f"  Saved FIG_DEMO1_01.pdf + .png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

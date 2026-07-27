"""
render_qc_psth.py -- FIG-DEMO2-QC-PSTH: sniff-locked PSTH examples, single
strongest and single weakest overall phase-locking units across ALL
experiments (contract_spec.yaml figure_id FIG-DEMO2-QC-PSTH).

Reads figdata_FIG-DEMO2-QC-PSTH.yaml (frozen, traces to RESULT-psth-extremes
-- pooled condition-agnostic MRL, OD-4). Presentation-only: every plotted
number is taken verbatim from the frozen figdata; no statistic is
recomputed here.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_common import load_yaml, savefig, MIN_FONT

COLOR_STRONG = "#009E73"  # bluish green (Okabe-Ito), colorblind-safe
COLOR_WEAK = "#CC79A7"    # reddish purple (Okabe-Ito), colorblind-safe


def _plot_one(ax, block, color, label):
    unit_id = block["unit_id"]
    session = block["session"]
    pooled_mrl = block["pooled_mrl"]
    centers = np.asarray(block["centers_phase"], dtype=float)
    psth = np.asarray(block["psth_phase"], dtype=float)
    n_events = block["n_events"]

    ax.plot(centers, psth, color=color, lw=1.8)
    ax.fill_between(centers, 0, psth, color=color, alpha=0.15)
    ax.set_xlim(0.0, 1.0)
    y_top = max(1.0, float(np.nanmax(psth))) * 1.1
    ax.set_ylim(0.0, y_top)
    ax.set_xlabel("sniff-cycle phase (0-1)")
    ax.set_ylabel("spike rate (Hz)")
    ax.set_title(f"{label}: Unit {unit_id} ({session})\npooled_MRL={pooled_mrl:.3f}",
                  fontsize=MIN_FONT + 1)
    ax.text(0.02, 0.97, f"n_events={n_events}", transform=ax.transAxes,
             va="top", ha="left", fontsize=MIN_FONT - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render():
    d = load_yaml("figdata_FIG-DEMO2-QC-PSTH.yaml")
    strongest = d["strongest_unit"]
    weakest = d["weakest_unit"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 4.2))
    _plot_one(axL, strongest, COLOR_STRONG, "Strongest")
    _plot_one(axR, weakest, COLOR_WEAK, "Weakest")

    fig.suptitle(
        "Sniff-locked PSTH examples -- single strongest and single weakest "
        "overall phase-locking units",
        fontsize=MIN_FONT + 3,
    )
    caption = (
        "Selection basis: MRL computed on all valid-sniff spikes pooled across "
        "ethanol + control conditions (condition-agnostic); the single strongest- and "
        "single weakest-overall phase-locking units across ALL experiments (2 units "
        "total, not one pair per experiment or per condition; OD-4)."
    )
    fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=MIN_FONT - 1, wrap=True)

    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    return savefig(fig, "FIG-DEMO2-QC-PSTH")


if __name__ == "__main__":
    png, pdf = render()
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")

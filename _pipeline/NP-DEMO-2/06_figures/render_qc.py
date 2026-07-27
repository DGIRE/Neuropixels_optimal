"""
render_qc.py -- FIG-DEMO2-QC: per-experiment SNF and ethanol (ETH) traces
with sniff-onset and ethanol-contact thresholds (contract_spec.yaml
figure_id FIG-DEMO2-QC).

Reads figdata_FIG-DEMO2-QC.yaml (frozen, traces to RESULT-qc-counts +
RESULT-eth-threshold-log + RESULT-qc-discards, plus a fresh raw-session
SNF_z/ETH display-trace reload done by build_figdata.py). Presentation-only:
every plotted trace/threshold/discard-run is taken verbatim from the frozen
figdata; no statistic is recomputed here.

Layout: one experiment per stacked panel-pair (top: SNF_z with sniff_thr red
dashed line + shaded discarded sections; bottom: ETH with eth_threshold red
dashed line), all 6 experiments stacked in a single tall figure sharing a
common time axis convention (each pair has its own x-axis since session
durations differ).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_common import load_yaml, savefig, MIN_FONT


def render():
    d = load_yaml("figdata_FIG-DEMO2-QC.yaml")
    sessions = d["sessions"]
    n = len(sessions)

    fig = plt.figure(figsize=(11.0, 3.1 * n))
    gs = gridspec.GridSpec(n * 2, 1, height_ratios=[1.0, 1.0] * n, hspace=0.55)

    for i, s in enumerate(sessions):
        session_date = s["session_date"]
        t = np.asarray(s["time_s"], dtype=float)
        snf = np.asarray(s["SNF_z"], dtype=float)
        eth = np.asarray(s["ETH"], dtype=float)
        sniff_thr = float(s["sniff_thr"])
        eth_threshold = float(s["eth_threshold"])
        discards = s.get("discards", [])
        n_sniffs = s["n_sniffs"]
        n_neurons = s["n_neurons"]
        n_trials = s["n_trials"]
        length_min = s["length_min"]
        pct_usable = s["pct_usable_sniffs"]

        ax_snf = fig.add_subplot(gs[2 * i, 0])
        ax_eth = fig.add_subplot(gs[2 * i + 1, 0], sharex=ax_snf)

        ax_snf.plot(t, snf, color="0.15", lw=0.4, rasterized=True)
        ax_snf.axhline(sniff_thr, color="red", linestyle="--", lw=1.2, zorder=5)
        for disc in discards:
            ax_snf.axvspan(disc["start_s"], disc["end_s"], color="gray", alpha=0.30,
                             lw=0, zorder=1)
        ax_snf.set_ylabel("SNF_z (a.u.)")
        y_pad = 0.05 * (np.nanmax(snf) - np.nanmin(snf) + 1e-9)
        ax_snf.set_ylim(np.nanmin(snf) - y_pad, np.nanmax(snf) + y_pad)
        ax_snf.spines["top"].set_visible(False)
        ax_snf.spines["right"].set_visible(False)
        plt.setp(ax_snf.get_xticklabels(), visible=False)

        adj_note = ("adjusted from 0.11 default" if abs(eth_threshold - 0.11) > 1e-9
                     else "default (0.11), not adjusted")
        ax_snf.set_title(
            f"{session_date} -- n_sniffs={n_sniffs}, n_neurons={n_neurons}, "
            f"n_trials={n_trials}, length={length_min:.1f} min, "
            f"usable_sniffs={pct_usable:.1f}%  |  eth_threshold={eth_threshold:.3f} "
            f"({adj_note})",
            fontsize=MIN_FONT - 1, loc="left",
        )

        ax_eth.plot(t, eth, color="0.15", lw=0.4, rasterized=True)
        ax_eth.axhline(eth_threshold, color="red", linestyle="--", lw=1.2, zorder=5)
        for disc in discards:
            ax_eth.axvspan(disc["start_s"], disc["end_s"], color="gray", alpha=0.30,
                             lw=0, zorder=1)
        ax_eth.set_ylabel("ETH signal (a.u.)")
        y_pad_e = 0.05 * (np.nanmax(eth) - np.nanmin(eth) + 1e-9)
        ax_eth.set_ylim(np.nanmin(eth) - y_pad_e, np.nanmax(eth) + y_pad_e)
        ax_eth.set_xlabel("time (s)")
        ax_eth.spines["top"].set_visible(False)
        ax_eth.spines["right"].set_visible(False)
        ax_eth.set_xlim(float(t[0]), float(t[-1]))

    fig.suptitle(
        "Per-experiment SNF and ethanol (ETH) traces with sniff-onset / "
        "ethanol-contact thresholds (gray shading = discarded sections, "
        "spike_SNF_PH == -1)",
        fontsize=MIN_FONT + 3,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return savefig(fig, "FIG-DEMO2-QC")


if __name__ == "__main__":
    png, pdf = render()
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")

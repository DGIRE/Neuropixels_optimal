"""
render_FIG_DEMO1_QC.py — Render FIG-DEMO1-QC.

Reads ONLY from 04_figures/figdata/. No scientific recomputation.

Output: output/FIG_DEMO1_QC.pdf  (vector)
        output/FIG_DEMO1_QC.png  (300 dpi raster)

Figure: Example sniffing per experiment (interim QC).
  Six panels (2 rows × 3 cols), one per session.
  Each: 90-second SNF trace excerpt.
    Non-sniffing discards: light gray shading.
    Noise/short discards: light red shading.
    Unshaded = valid detected sniff segment.
  Caption per panel: session date, n_sniffs, n_neurons, n_trials.
  Note if session excluded from paired stat (2021-11-03).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIGDATA_DIR = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-1\04_figures\figdata")
OUTPUT_DIR  = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-1\04_figures\output")

# Sessions excluded from paired stat (no control-condition units)
EXCLUDED_FROM_STAT = {"2021-11-03"}

NONSNIFF_COLOR = "#CCCCCC"   # gray
NOISE_COLOR    = "#FFCCBB"   # light orange-red
LEGEND_PATCHES = [
    mpatches.Patch(facecolor=NONSNIFF_COLOR, edgecolor="none", label="non-sniffing"),
    mpatches.Patch(facecolor=NOISE_COLOR, edgecolor="none", label="noise/short"),
]


def _shade_discards(ax: plt.Axes, discards: list[dict], win_start: float, win_end: float) -> None:
    for d in discards:
        ds, de = max(d["start_s"], win_start), min(d["end_s"], win_end)
        if de <= ds:
            continue
        color = NONSNIFF_COLOR if d["reason"] == "non-sniffing" else NOISE_COLOR
        ax.axvspan(ds, de, facecolor=color, edgecolor="none", alpha=0.85, zorder=0)


def main() -> None:
    with open(FIGDATA_DIR / "FIG_DEMO1_QC_meta.json") as f:
        qc_meta = json.load(f)
    with open(FIGDATA_DIR / "FIG_DEMO1_QC_traces.json") as f:
        traces = json.load(f)

    # Chronological order by session_date
    sess_dirs = sorted(qc_meta.keys(), key=lambda d: qc_meta[d]["session_date"])

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7,
        "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })

    fig, axes = plt.subplots(2, 3, figsize=(14, 6),
                              gridspec_kw={"hspace": 0.55, "wspace": 0.35})
    axes_flat = axes.flatten()

    for idx, sess_dir in enumerate(sess_dirs):
        ax = axes_flat[idx]
        meta = qc_meta[sess_dir]
        date_str = meta["session_date"]
        trace = traces.get(date_str, {})

        if "error" in trace:
            ax.text(0.5, 0.5, f"Data error:\n{trace['error']}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=6)
            ax.set_title(date_str, fontsize=8)
            continue

        snf  = np.asarray(trace["snf"], dtype=np.float32)
        lv_fs = float(trace["lv_fs"])
        win_start = float(trace["win_start_s"])
        win_end   = float(trace["win_end_s"])
        t = np.arange(len(snf)) / lv_fs + win_start

        # Normalise for display (zero-mean, unit peak)
        snf_f = snf - float(np.mean(snf))
        peak = float(np.max(np.abs(snf_f))) or 1.0
        snf_f = snf_f / peak

        _shade_discards(ax, meta["discards"], win_start, win_end)
        ax.plot(t, snf_f, color="#1F4E79", linewidth=0.5, rasterized=True)

        ax.set_xlim(win_start, win_end)
        ax.set_ylim(-1.4, 1.4)
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)

        ax.set_xlabel("Time (s)", fontsize=7)
        if idx % 3 == 0:
            ax.set_ylabel("SNF (a.u., norm.)", fontsize=7)

        excl_note = "  ★excluded from stat" if date_str in EXCLUDED_FROM_STAT else ""
        title = (f"{date_str}{excl_note}\n"
                 f"n_sniffs={meta['n_sniffs']}, "
                 f"n_neurons={meta['n_neurons']}, "
                 f"n_trials={meta['n_trials']}")
        ax.set_title(title, fontsize=7.5, pad=3, loc="left")

    # Legend in the first panel
    axes_flat[0].legend(handles=LEGEND_PATCHES, fontsize=6, loc="upper right",
                         framealpha=0.8, handlelength=1.0)

    fig.suptitle(
        "FIG-DEMO1-QC: Example SNF traces with valid/discarded regions  "
        "(~90 s excerpt per session; timestamps are relative to session start)",
        fontsize=8.5, y=0.98,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "FIG_DEMO1_QC.pdf", dpi=300, format="pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "FIG_DEMO1_QC.png", dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved FIG_DEMO1_QC.pdf + .png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

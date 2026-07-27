"""
run_band_examples.py
====================
Make band-filtered LFP time-series figures for a specified time window from a
specified experiment (or group of experiments) in the aggregate.

For EACH selected experiment it writes two figures into the Examples folder:
    <DATE>_theta_<t0>-<t1>s.png   -- all 32 channels filtered to the theta band
    <DATE>_gamma_<t0>-<t1>s.png   -- all 32 channels filtered to the gamma band

Channels are stacked by depth (tip at the bottom, surface at the top) and coloured
by depth, so the laminar structure of each rhythm is visible.

Specify what to plot either by editing the CONFIG block or on the command line:

    # all knobs from CONFIG (defaults: all LFP experiments, 0-10 s)
    python run_band_examples.py

    # pick experiments and a window on the command line
    python run_band_examples.py --dates 06-21-2022,10-05-2021 --start 120 --end 130

    # custom bands / paths
    python run_band_examples.py --dates all --start 0 --end 8 \
        --theta 2 12 --gamma 65 100 --h5 <path> --figdir <path>

Requires numpy, scipy, matplotlib, h5py (same env as the Optimized Python kernel).
Reads only the aggregate -- no raw-data access needed.
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from band_timeseries import (          # noqa: E402
    list_lfp_dates, load_lfp_session, bandpass, window_indices,
)

# ===========================================================================
# CONFIG  (command-line flags override these)
# ===========================================================================
H5_PATH = r"C:\Projects\Repos\Neuropixels\DATA\Aggregate\np_aggregate.h5"
FIG_DIR = (r"C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests"
           r"\Location standardization\Figures\Examples")

# Which experiments: a list of "MM-DD-YYYY" dates, or ["all"] for every LFP session.
EXPERIMENTS = ["all"]

# Time window (seconds from recording start). Clipped to each recording's length.
T_START = 0.0
T_END = 10.0

# Filter bands (Hz). Lab paper bands: theta 2-12, gamma 65-100. Robust defaults
# below sit clearly inside the aggregate's DC..~100 Hz passband.
THETA_BAND = (4.0, 12.0)
GAMMA_BAND = (30.0, 80.0)
FILTER_ORDER = 4

# Display
SPACING_SD = 5.0      # vertical gap between channels, in units of the median
                      # per-channel std of the band-filtered window
CMAP = "viridis"      # colour traces by depth (readable line colormap)
LINEWIDTH = 0.6
DPI = 150
# ===========================================================================


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Band-filtered LFP time-series examples.")
    p.add_argument("--h5", default=H5_PATH)
    p.add_argument("--figdir", default=FIG_DIR)
    p.add_argument("--dates", default=",".join(EXPERIMENTS),
                   help="comma-separated MM-DD-YYYY dates, or 'all'")
    p.add_argument("--start", type=float, default=T_START)
    p.add_argument("--end", type=float, default=T_END)
    p.add_argument("--theta", type=float, nargs=2, default=list(THETA_BAND),
                   metavar=("LO", "HI"))
    p.add_argument("--gamma", type=float, nargs=2, default=list(GAMMA_BAND),
                   metavar=("LO", "HI"))
    return p.parse_args(argv)


def _resolve_dates(h5_path: str, dates_arg: str) -> list[str]:
    avail = list_lfp_dates(h5_path)
    want = [d.strip() for d in dates_arg.split(",") if d.strip()]
    if not want or any(w.lower() == "all" for w in want):
        return avail
    chosen, missing = [], []
    for w in want:
        (chosen if w in avail else missing).append(w)
    for m in missing:
        print("  WARNING: %s has no LFP in the aggregate -- skipping" % m)
    return chosen


def plot_band_stack(sess, i0, i1, band, band_name, outpath):
    """Stack all channels of one band-filtered window, coloured + offset by depth."""
    fs = sess["fs"]
    # filter the FULL trace (no window edge artefacts), then slice the window
    filt = bandpass(sess["lfp"], fs, band[0], band[1], FILTER_ORDER)
    seg = filt[:, i0:i1] * 1e6            # -> microvolts
    t = (np.arange(i0, i1) / fs)
    n_ch = seg.shape[0]

    # vertical offset per channel from a robust common amplitude
    sd = np.median(np.std(seg, axis=1))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    step = SPACING_SD * sd

    fig, ax = plt.subplots(figsize=(13, 9))
    cmap = plt.get_cmap(CMAP)
    yc = np.asarray(sess["ycoord"], dtype=float)
    for i in range(n_ch):                 # row 0 = tip -> plotted at the bottom
        base = i * step
        ax.plot(t, seg[i] + base, color=cmap(i / max(n_ch - 1, 1)),
                lw=LINEWIDTH)
    # y ticks: label a subset of channels with their depth in microns
    ticks = np.arange(0, n_ch, max(1, n_ch // 16))
    ax.set_yticks(ticks * step)
    ax.set_yticklabels(["%d (%.0fµm)" % (int(sess["channel_row"][i]), yc[i])
                        if i < yc.size else str(i) for i in ticks], fontsize=7)
    ax.set_ylim(-step, (n_ch) * step)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel (AP row, depth)  —  tip at bottom → surface at top")

    # amplitude scale bar (one median-std) in the lower-right
    xr = t[-1] - t[0]
    x_bar = t[-1] - 0.02 * xr
    y_bar = -0.6 * step
    ax.plot([x_bar, x_bar], [y_bar, y_bar + sd], color="k", lw=2,
            solid_capstyle="butt")
    ax.text(x_bar - 0.01 * xr, y_bar + sd / 2, "%.0f µV" % sd,
            ha="right", va="center", fontsize=8)

    ax.set_title("%s — %s band (%g–%g Hz)   |   %.2f–%.2f s"
                 % (sess["date"], band_name, band[0], band[1], t[0], t[-1]),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def main(argv=None):
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    os.makedirs(args.figdir, exist_ok=True)

    dates = _resolve_dates(args.h5, args.dates)
    if not dates:
        raise SystemExit("No matching experiments with LFP to plot.")
    print("Plotting %d experiment(s): %s" % (len(dates), ", ".join(dates)))
    print("Window: %.2f-%.2f s | theta %g-%g Hz | gamma %g-%g Hz"
          % (args.start, args.end, args.theta[0], args.theta[1],
             args.gamma[0], args.gamma[1]))

    tag = "%g-%gs" % (args.start, args.end)
    for date in dates:
        sess = load_lfp_session(args.h5, date)
        if sess is None:
            print("  %s: no LFP, skipping" % date)
            continue
        try:
            i0, i1 = window_indices(sess["fs"], sess["lfp"].shape[1],
                                    args.start, args.end)
        except ValueError as e:
            print("  %s: %s" % (date, e))
            continue

        for band, name in ((tuple(args.theta), "theta"), (tuple(args.gamma), "gamma")):
            out = os.path.join(args.figdir, "%s_%s_%s.png" % (date, name, tag))
            plot_band_stack(sess, i0, i1, band, name, out)
            print("  wrote %s" % os.path.basename(out))

    print("\nDone -> %s" % args.figdir)


if __name__ == "__main__":
    main()

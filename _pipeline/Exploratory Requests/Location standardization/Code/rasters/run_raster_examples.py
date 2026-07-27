"""
run_raster_examples.py
======================
LFP + unit-raster (+ sniff) display for a specified time window from a specified
experiment (or group of experiments) in the aggregate.

For EACH selected experiment it writes THREE figures into the Rasters folder --
one per LFP version: raw (broadband), theta-filtered, gamma-filtered:

    <DATE>_raw_<t0>-<t1>s.png
    <DATE>_theta_<t0>-<t1>s.png
    <DATE>_gamma_<t0>-<t1>s.png

Each figure stacks, bottom (tip) to top (surface): the 32 LFP channels, and under
each LFP trace the spike rasters of the units at that LFP site's depth or below it
(but above the next, deeper LFP site). If the experiment has a strong sniff signal
it is drawn across the top.

Specify what to plot by editing CONFIG or on the command line:

    python run_raster_examples.py --dates 06-21-2022 --start 120 --end 130
    python run_raster_examples.py --dates all --start 0 --end 8 --sniff on
    python run_raster_examples.py --dates 10-05-2021 --start 60 --end 70 --theta 2 12 --gamma 65 100

Per-unit rasters require the aggregate's `spikeClusters` array (added 2026-07-26;
rebuild spike sessions to populate). Without it the tool draws a depth-binned
POPULATION raster (one row of all spikes per LFP depth bin) and says so.

Requires numpy, scipy, matplotlib, h5py. Reads only the aggregate.
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
from raster_display import (          # noqa: E402
    list_lfp_dates, load_session_full, bandpass, window_indices,
    sniff_strength, per_unit_spike_times,
)

# ===========================================================================
# CONFIG (command-line flags override these)
# ===========================================================================
H5_PATH = r"C:\Projects\Repos\Neuropixels\DATA\Aggregate\np_aggregate.h5"
FIG_DIR = (r"C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests"
           r"\Location standardization\Figures\Rasters")

EXPERIMENTS = ["all"]
T_START = 0.0
T_END = 10.0

THETA_BAND = (4.0, 12.0)     # lab paper theta 2-12
GAMMA_BAND = (30.0, 80.0)    # lab paper gamma 65-100 (near the ~100 Hz ceiling)
FILTER_ORDER = 4

INCLUDE_SNIFF = "auto"        # "auto" | "on" | "off"
SNIFF_BAND = (0.5, 12.0)
SNIFF_STRENGTH_MIN = 0.35     # min band/broadband std ratio for "strong" (auto)

SHOW_EMPTY_UNITS = False      # drop units with no spikes in the window (declutter)
RASTER_CMAP = "viridis"       # LFP traces coloured by depth
DPI = 150
# ===========================================================================

# layout constants (arbitrary data units)
ROW_H = 1.0        # per unit raster row
BIN_GAP = 0.8      # between a bin's rasters and its LFP trace
TRACE_H = 7.0      # vertical span allotted to one LFP trace
LANE_GAP = 2.0     # after a lane, before the next (shallower) bin
SNF_GAP = 5.0
SNF_H = 12.0


def _parse_args(argv):
    p = argparse.ArgumentParser(description="LFP + unit-raster (+ sniff) display.")
    p.add_argument("--h5", default=H5_PATH)
    p.add_argument("--figdir", default=FIG_DIR)
    p.add_argument("--dates", default=",".join(EXPERIMENTS))
    p.add_argument("--start", type=float, default=T_START)
    p.add_argument("--end", type=float, default=T_END)
    p.add_argument("--theta", type=float, nargs=2, default=list(THETA_BAND))
    p.add_argument("--gamma", type=float, nargs=2, default=list(GAMMA_BAND))
    p.add_argument("--sniff", choices=["auto", "on", "off"], default=INCLUDE_SNIFF)
    return p.parse_args(argv)


def _resolve_dates(h5_path, dates_arg):
    avail = list_lfp_dates(h5_path)
    want = [d.strip() for d in dates_arg.split(",") if d.strip()]
    if not want or any(w.lower() == "all" for w in want):
        return avail
    chosen = []
    for w in want:
        if w in avail:
            chosen.append(w)
        else:
            print("  WARNING: %s has no LFP -- skipping" % w)
    return chosen


def _unit_depths(sess):
    """Per-unit depth (microns from tip); compute from spikes if not stored."""
    uids = sess.get("unitIDs")
    ud = sess.get("unitDepths")
    if ud is not None and uids is not None and len(ud) == len(uids):
        return np.asarray(uids), np.asarray(ud, dtype=np.float64)
    # fall back: mean spikeDepths per cluster
    if sess.get("has_clusters") and sess.get("spikeDepths") is not None:
        clu = np.asarray(sess["spikeClusters"])
        sd = np.asarray(sess["spikeDepths"], dtype=np.float64)
        uids = np.unique(clu)
        depths = np.array([np.nanmean(sd[clu == u]) if np.any(clu == u) else np.nan
                           for u in uids])
        return uids, depths
    return None, None


def _lfp_segment(sess, i0, i1, band):
    """Depth-sorted LFP window in microvolts. band=None -> raw broadband."""
    lfp = sess["lfp"]
    if band is not None:
        lfp = bandpass(lfp, sess["fs"], band[0], band[1], FILTER_ORDER)
    seg = lfp[:, i0:i1] * 1e6
    yc = np.asarray(sess["ycoord"], dtype=np.float64)
    order = np.argsort(yc, kind="stable")       # 0 = deepest (tip)
    return seg[order], yc[order], np.asarray(sess["channel_row"])[order]


def make_figure(sess, i0, i1, t, band, band_name, show_sniff, args, outpath):
    seg, yc_sorted, rows_sorted = _lfp_segment(sess, i0, i1, band)
    n_ch = seg.shape[0]
    t0, t1 = t[0], t[-1]

    # --- assemble per-bin unit rasters -------------------------------------
    per_unit = sess.get("has_clusters", False)
    bin_units = [[] for _ in range(n_ch)]        # list of (uid, depth, spike_times)
    pop_rows = [None] * n_ch                       # fallback: population spikes per bin
    mode = "population (no spikeClusters — rebuild to get per-unit rows)"
    n_units_shown = 0

    if per_unit:
        uids, udepth = _unit_depths(sess)
        if uids is not None:
            mode = "per unit"
            pust = per_unit_spike_times(sess["spikeTimes"], sess["spikeClusters"],
                                        uids, t0, t1)
            idx = np.searchsorted(yc_sorted, udepth, side="left")
            idx = np.clip(idx, 0, n_ch - 1)
            for u, uid in enumerate(uids):
                if not np.isfinite(udepth[u]):
                    continue
                st = pust.get(int(uid), np.empty(0))
                if st.size == 0 and not SHOW_EMPTY_UNITS:
                    continue
                bin_units[int(idx[u])].append((int(uid), float(udepth[u]), st))
            for b in range(n_ch):
                bin_units[b].sort(key=lambda z: z[1])   # deeper first
                n_units_shown += len(bin_units[b])

    if not per_unit or n_units_shown == 0:
        # population fallback (also used if per-unit yielded nothing)
        if sess.get("has_spikes") and sess.get("spikeDepths") is not None:
            st = np.asarray(sess["spikeTimes"], dtype=np.float64)
            sd = np.asarray(sess["spikeDepths"], dtype=np.float64)
            win = (st >= t0) & (st < t1)
            st_w, sd_w = st[win], sd[win]
            edges = np.concatenate([[-np.inf], yc_sorted[:-1]
                                    + np.diff(yc_sorted) / 2.0, [np.inf]])
            b_of = np.clip(np.searchsorted(edges, sd_w, side="right") - 1, 0, n_ch - 1)
            for b in range(n_ch):
                pop_rows[b] = st_w[b_of == b]
            if per_unit and n_units_shown == 0:
                mode = "population (no units with spikes in window)"

    # --- lay out y positions (bottom = tip) --------------------------------
    y = 0.0
    trace_base = np.zeros(n_ch)
    raster_items = []        # (y_center, spike_times, is_unit)
    for i in range(n_ch):
        if per_unit and n_units_shown > 0:
            for (uid, dep, st) in bin_units[i]:
                raster_items.append((y + ROW_H / 2.0, st, True))
                y += ROW_H
            if bin_units[i]:
                y += BIN_GAP
        else:
            st = pop_rows[i] if pop_rows[i] is not None else np.empty(0)
            raster_items.append((y + ROW_H / 2.0, st, False))
            y += ROW_H + BIN_GAP
        trace_base[i] = y + TRACE_H / 2.0
        y += TRACE_H + LANE_GAP
    total_h = y

    # common LFP amplitude scale
    amp = np.median([np.percentile(np.abs(seg[i]), 95) for i in range(n_ch)])
    amp = amp if np.isfinite(amp) and amp > 0 else 1.0
    lfp_scale = (0.42 * TRACE_H) / amp

    # --- figure ------------------------------------------------------------
    top = total_h
    if show_sniff:
        top = total_h + SNF_GAP + SNF_H
    fig_h = float(np.clip(total_h * 0.05, 7.0, 46.0))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    cmap = plt.get_cmap(RASTER_CMAP)

    # rasters
    for (yc_, st, is_unit) in raster_items:
        if st.size:
            ax.eventplot([st], lineoffsets=[yc_], linelengths=[ROW_H * 0.85],
                         colors=["0.15"], linewidths=0.5)

    # LFP traces (depth-coloured), tip at bottom
    for i in range(n_ch):
        ax.plot(t, seg[i] * lfp_scale + trace_base[i],
                color=cmap(i / max(n_ch - 1, 1)), lw=0.7)

    # y ticks at LFP baselines -> depth label
    ticks = np.arange(0, n_ch, max(1, n_ch // 16))
    ax.set_yticks(trace_base[ticks])
    ax.set_yticklabels(["%.0fµm" % yc_sorted[i] for i in ticks], fontsize=7)

    # sniff at the top
    if show_sniff:
        lv = sess["LV_Fs"]
        s0, s1 = int(round(t0 * lv)), int(round(t1 * lv))
        snf = np.asarray(sess["SNF"], dtype=np.float64)
        s1 = min(s1, snf.size)
        seg_snf = snf[s0:s1]
        if seg_snf.size > 4:
            seg_snf = seg_snf - np.mean(seg_snf)
            seg_snf = bandpass(seg_snf, lv, SNIFF_BAND[0], SNIFF_BAND[1])
            m = np.max(np.abs(seg_snf)) or 1.0
            ts = t0 + np.arange(seg_snf.size) / lv
            y_snf = total_h + SNF_GAP + SNF_H / 2.0
            ax.plot(ts, seg_snf / m * (SNF_H * 0.45) + y_snf, color="#c0392b", lw=0.9)
            ax.text(t0, y_snf + SNF_H * 0.5, "Sniff (SNF, %.1f–%.1f Hz)"
                    % SNIFF_BAND, fontsize=8, va="bottom", color="#c0392b")

    # LFP amplitude scale bar (bottom-right)
    xr = t1 - t0
    xb = t1 - 0.015 * xr
    yb = -0.5 * TRACE_H
    ax.plot([xb, xb], [yb, yb + amp * lfp_scale], color="k", lw=2)
    ax.text(xb - 0.01 * xr, yb + amp * lfp_scale / 2, "%.0f µV" % amp,
            ha="right", va="center", fontsize=8)

    ax.set_xlim(t0, t1)
    ax.set_ylim(-TRACE_H, top + 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LFP depth (µm from tip)  —  tip at bottom → surface at top")
    sp_txt = ("%d units" % n_units_shown) if (per_unit and n_units_shown) else mode
    ax.set_title("%s — %s LFP + rasters%s   |   %.2f–%.2f s   |   %s"
                 % (sess["date"], band_name,
                    ("  +sniff" if show_sniff else ""), t0, t1, sp_txt), fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)
    return mode, n_units_shown


def main(argv=None):
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    os.makedirs(args.figdir, exist_ok=True)
    dates = _resolve_dates(args.h5, args.dates)
    if not dates:
        raise SystemExit("No matching experiments with LFP.")
    print("Experiments: %s | window %.2f-%.2fs" % (", ".join(dates), args.start, args.end))

    tag = "%g-%gs" % (args.start, args.end)
    bands = [("raw", None), ("theta", tuple(args.theta)), ("gamma", tuple(args.gamma))]

    for date in dates:
        sess = load_session_full(args.h5, date)
        if sess is None:
            print("  %s: no LFP, skipping" % date)
            continue
        try:
            i0, i1 = window_indices(sess["fs"], sess["lfp"].shape[1], args.start, args.end)
        except ValueError as e:
            print("  %s: %s" % (date, e))
            continue
        t = np.arange(i0, i1) / sess["fs"]

        # decide sniff inclusion
        show_sniff = False
        metric = 0.0
        if args.sniff != "off" and sess.get("has_sensors"):
            metric = sniff_strength(sess["SNF"], sess["LV_Fs"], SNIFF_BAND)
            show_sniff = (args.sniff == "on") or (metric >= SNIFF_STRENGTH_MIN)
        spikes_msg = ("spikes: %s" % ("per-unit" if sess.get("has_clusters")
                      else ("population" if sess.get("has_spikes") else "none")))
        print("  %s | %s | sniff=%s (ratio %.2f)"
              % (date, spikes_msg, "on" if show_sniff else "off", metric))

        for band_name, band in bands:
            out = os.path.join(args.figdir, "%s_%s_%s.png" % (date, band_name, tag))
            mode, nu = make_figure(sess, i0, i1, t, band, band_name, show_sniff, args, out)
            print("    wrote %s  [%s]" % (os.path.basename(out), mode))

    print("\nDone -> %s" % args.figdir)


if __name__ == "__main__":
    main()

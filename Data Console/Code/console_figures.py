"""
console_figures.py
==================
Matplotlib figure builders for the Neuropixels Data Console (no GUI code, so they
can be rendered headlessly and saved to PNG for testing).

    build_probe_figure(session)                 -> (fig, ax)   probe map
    build_view_figure(session, elecs, band, band_name, t0, t1, show_sniff)
                                                -> (fig, meta)  ETH/SNF/LFP+rasters

The probe map: black background, grey shanks, white active electrodes, unit-count
dots (jet colormap), open cyan circles at the LFP sampling sites. Selection markers
are added by the GUI on the returned axes.
"""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# probe map
# ---------------------------------------------------------------------------
def build_probe_figure(session, figsize=(5.2, 8.6)):
    fig = plt.figure(figsize=figsize, facecolor="black")
    ax = fig.add_axes([0.13, 0.06, 0.74, 0.88])
    ax.set_facecolor("black")

    x, y = session.chan_x, session.chan_y
    if x.size == 0:
        ax.text(0.5, 0.5, "No probe geometry\n(session has no channel coords)",
                color="white", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    # grey shank bodies + tips
    for s in range(max(session.n_shanks, 1)):
        m = session.shank_of_elec == s
        if not np.any(m):
            continue
        xs, ys = x[m], y[m]
        x0, x1 = xs.min() - 12, xs.max() + 12
        y0, y1 = ys.min() - 15, ys.max() + 15
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="0.35",
                               edgecolor="none", zorder=1))
        tip = max((x1 - x0) * 0.0, 60.0)
        ax.add_patch(Polygon([(x0, y0), (x1, y0), ((x0 + x1) / 2, y0 - tip)],
                             facecolor="0.35", edgecolor="none", zorder=1))

    # white active electrodes
    ax.scatter(x, y, s=9, marker="s", c="white", edgecolors="none", zorder=3)

    # unit-count dots (jet)
    has_units = session.elec_unit_count > 0
    if np.any(has_units):
        sc = ax.scatter(x[has_units], y[has_units], s=42,
                        c=session.elec_unit_count[has_units], cmap="jet",
                        edgecolors="black", linewidths=0.3, zorder=4,
                        vmin=1, vmax=max(2, session.elec_unit_count.max()))
        cax = fig.add_axes([0.89, 0.30, 0.02, 0.4])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("units / electrode", color="white", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cb.ax.get_yticklabels(), color="white")

    # open cyan circles at LFP sampling sites
    if session.lfp_ycoord.size:
        ax.scatter(session.lfp_xcoord, session.lfp_ycoord, s=115,
                   facecolors="none", edgecolors="cyan", linewidths=1.3, zorder=5)

    ax.set_xlabel("x (µm)", color="white", fontsize=9)
    ax.set_ylabel("depth from tip (µm)", color="white", fontsize=9)
    ax.set_title("%s\n%d electrodes · %d shank(s) · LFP sites = cyan"
                 % (session.date, session.n_elec, max(session.n_shanks, 1)),
                 color="white", fontsize=9)
    if session.n_shanks > 1 and not getattr(session, "has_unit_xy", False):
        ax.text(0.5, 1.005, "unit shank approximate (run add_unit_positions.py for "
                "exact placement — see guide)", transform=ax.transAxes, ha="center",
                va="bottom", color="0.7", fontsize=6.5)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("0.4")
    # stretch x so the (narrow) columns/shanks are visible and clickable -- a real
    # NP2.0 shank is ~70 µm wide but millimetres tall, so equal aspect would hide it.
    xspan = max(x.max() - x.min(), 32.0)
    xpad = max(40.0, 0.18 * xspan)
    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    ax.set_ylim(y.min() - 120, y.max() + 60)

    # legend
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markersize=6, label="electrode"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="cyan", markersize=9, label="LFP site"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728",
               markeredgecolor="k", markersize=7, label="has units (jet=count)"),
    ]
    leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.11),
                    ncol=3, fontsize=6.5, frameon=False, handletextpad=0.3,
                    columnspacing=0.9)
    for txt in leg.get_texts():
        txt.set_color("white")
    return fig, ax


def selection_positions(session, elecs):
    """(x, y) arrays for a list of selected electrode indices."""
    if not elecs:
        return np.empty(0), np.empty(0)
    e = np.array(elecs, dtype=int)
    return session.chan_x[e], session.chan_y[e]


# ---------------------------------------------------------------------------
# view figure: ETH / SNF / LFP + rasters
# ---------------------------------------------------------------------------
def _nearest_lfp_site_on_shank(session, elec):
    """Row index into the LFP array of the LFP site nearest `elec` (same shank)."""
    if session.lfp_ycoord.size == 0:
        return None
    depth = session.chan_y[elec]
    shank = session.shank_of(elec)
    site_shank = np.array([session.shank_of(session.lfp_site_elec[s])
                           for s in range(session.lfp_ycoord.size)]) \
        if session.lfp_site_elec.size else np.zeros(session.lfp_ycoord.size, int)
    same = np.where(site_shank == shank)[0]
    pool = same if same.size else np.arange(session.lfp_ycoord.size)
    return int(pool[int(np.argmin(np.abs(session.lfp_ycoord[pool] - depth)))])


def build_view_figure(session, elecs, band, band_name, t0, t1, show_sniff,
                      t_offset=0.0, figsize=(13, None)):
    """Stacked view for the selected electrodes over absolute window [t0, t1] s.

    Top: ethanol trace; then sniff (if show_sniff); then, per selected electrode
    ordered by (shank, depth), the filtered LFP trace with its units' rasters
    beneath it. Returns (fig, metadata_dict).

    `t_offset` (s) is subtracted from every plotted time so the x-axis can read in
    within-trial seconds: pass the trial's absolute start time and a selected trial
    counts up from its own start (0-based) instead of from recording start. Pass 0
    (the default, used for whole-recording windows) to keep absolute time.
    """
    disp = lambda tt: np.asarray(tt) - t_offset     # abs time -> displayed x
    d0, d1 = t0 - t_offset, t1 - t_offset
    # order selected electrodes by shank then depth
    elecs = sorted(set(int(e) for e in elecs),
                   key=lambda e: (session.shank_of(e), session.chan_y[e]))
    t, seg_all = session.lfp_window(band, t0, t1)     # seg_all: (n_lfp_ch, n)
    seg_all = seg_all * 1e6                             # µV
    units_by_elec = session.units_at_electrodes(elecs)

    # geometry constants
    ROW_H, BIN_GAP, TRACE_H, LANE_GAP = 1.0, 0.6, 6.0, 1.8
    y = 0.0
    entries = []   # (elec, trace_base, lfp_row, [ (uid, y_center, spikes) ... ], label)
    units_shown = []
    per_unit_ok = session.has_spikes and session.spikeClusters is not None
    for e in elecs:
        rows = []
        if per_unit_ok:
            for (uid, dep) in units_by_elec.get(e, []):
                st = session.unit_spikes_in_window(uid, t0, t1)
                rows.append((uid, y + ROW_H / 2.0, st))
                units_shown.append(uid)
                y += ROW_H
        # population fallback when there is no spike->cluster map (pre-rebuild)
        if not rows and session.has_spikes and session.spikeClusters is None:
            depth = session.chan_y[e]
            band_lo, band_hi = depth - 60, depth + 60
            st = session.population_spikes_in_window(band_lo, band_hi, t0, t1)
            rows.append((None, y + ROW_H / 2.0, st))
            y += ROW_H
        if rows:
            y += BIN_GAP
        site = _nearest_lfp_site_on_shank(session, e)
        base = y + TRACE_H / 2.0
        label = "Sh%d  %.0fµm" % (session.shank_of(e), session.chan_y[e])
        entries.append((e, base, site, rows, label))
        y += TRACE_H + LANE_GAP
    total_h = max(y, TRACE_H)

    # amplitude scale (common) from the shown LFP sites
    amps = []
    for (_e, _b, site, _r, _l) in entries:
        if site is not None and seg_all.shape[0] > site:
            amps.append(np.percentile(np.abs(seg_all[site]), 95))
    amp = np.median(amps) if amps else 1.0
    amp = amp if np.isfinite(amp) and amp > 0 else 1.0
    lfp_scale = (0.42 * TRACE_H) / amp

    # sensor lanes on top
    eth_t, eth_v = session.sensor_window("ETH", t0, t1)
    snf_t, snf_v = (session.sensor_window("SNF", t0, t1) if show_sniff else (None, None))
    sensor_h = 6.0
    n_sensor = (1 if eth_v is not None else 0) + (1 if snf_v is not None else 0)
    top = total_h + (n_sensor * (sensor_h + 2.0) if n_sensor else 0.0) + 2.0

    fig_h = float(np.clip(top * 0.16 + 1.5, 5.0, 40.0))
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))

    # rasters + LFP.  raster_lanes records each clickable unit lane in DATA coords
    # (displayed x) so the GUI can map a click back to a unit id.
    raster_lanes = []
    for (e, base, site, rows, label) in entries:
        for (uid, yc, st) in rows:
            if st.size:
                col = "0.15" if uid is not None else "#8B0000"
                ax.eventplot([disp(st)], lineoffsets=[yc],
                             linelengths=[ROW_H * 0.85],
                             colors=[col], linewidths=0.6)
            if uid is not None:
                raster_lanes.append(dict(uid=int(uid), y_center=float(yc),
                                         half_height=float(ROW_H / 2.0),
                                         elec=int(e), lfp_site=site))
        if site is not None and seg_all.shape[0] > site:
            ax.plot(disp(t), seg_all[site] * lfp_scale + base, color="#1f77b4", lw=0.7)
        ax.text(d0, base + TRACE_H * 0.42, label, fontsize=7, va="bottom",
                ha="left", color="#1f77b4")

    # sensors: sniff directly above the LFP stack, ethanol at the very top
    yb = total_h + 2.0
    if snf_v is not None:
        from console_data import bandpass
        v = bandpass(snf_v - np.nanmean(snf_v), session.LV_Fs, 0.5, 12.0)
        m = np.max(np.abs(v)) or 1.0
        yc = yb + sensor_h / 2.0
        ax.plot(disp(snf_t), v / m * sensor_h * 0.45 + yc, color="#d62728", lw=0.9)
        ax.text(d0, yc + sensor_h * 0.5, "Sniff (SNF)", fontsize=8,
                color="#d62728", va="bottom")
        yb += sensor_h + 2.0
    if eth_v is not None:
        v = eth_v - np.nanmean(eth_v)
        m = np.max(np.abs(v)) or 1.0
        yc = yb + sensor_h / 2.0
        ax.plot(disp(eth_t), v / m * sensor_h * 0.45 + yc, color="#2ca02c", lw=0.9)
        ax.text(d0, yc + sensor_h * 0.5, "Ethanol (ETH)", fontsize=8,
                color="#2ca02c", va="bottom")

    # scale bar
    xr = d1 - d0
    xb = d1 - 0.02 * xr
    y0b = -0.6 * TRACE_H
    ax.plot([xb, xb], [y0b, y0b + amp * lfp_scale], color="k", lw=2)
    ax.text(xb - 0.01 * xr, y0b + amp * lfp_scale / 2, "%.0f µV" % amp,
            ha="right", va="center", fontsize=8)

    ax.set_xlim(d0, d1)
    ax.set_ylim(-TRACE_H, top)
    ax.set_yticks([])
    ax.set_xlabel("Time in trial (s)" if t_offset else "Time (s)")
    ax.set_title("%s — %s — %.3f–%.3f s — %d electrode(s), %d unit(s)"
                 % (session.date, band_name, t0, t1, len(elecs), len(set(units_shown))),
                 fontsize=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    meta = dict(date=session.date, experiment_name=session.experiment_name,
                band=band_name, band_hz=band, t_start_s=t0, t_end_s=t1,
                t_offset_s=float(t_offset),
                x_axis="within-trial seconds" if t_offset else "absolute seconds",
                electrodes=list(elecs),
                electrode_positions=[(float(session.chan_x[e]), float(session.chan_y[e]),
                                      session.shank_of(e)) for e in elecs],
                unit_ids=sorted(set(int(u) for u in units_shown)),
                lfp_fs=session.lfp_fs,
                sniff_shown=bool(snf_v is not None),
                ethanol_shown=bool(eth_v is not None),
                raster_lanes=raster_lanes)
    return fig, meta


# ---------------------------------------------------------------------------
# per-unit detail figure: one row per trial, ETH/SNF/LFP overlays, PSTH
# ---------------------------------------------------------------------------
def build_unit_figure(session, uid, elec, band, band_name, wt0, wt1, trials,
                      show_sniff=True, h5_path="", psth_bin_s=0.05,
                      figwidth=11.0):
    """Thorough single-unit view, repeated over trials.

    The within-trial window [wt0, wt1] (the seconds currently set in the console)
    is applied to EVERY trial, aligned so each trial's window starts at wt0. For
    every trial we gather the ethanol, sniff and nearest-LFP signals plus this
    unit's spikes; the ETH / SNF / LFP panels overlay every trial faintly with a
    bold across-trial mean, the raster shows one row per trial, and the bottom
    panel is the PSTH (firing rate) pooled across all trials.

    Returns (fig, meta). `elec` is the unit's peak electrode (used to pick the
    nearest LFP site); `band`/`band_name` match the view the click came from.
    """
    import matplotlib.gridspec as gridspec
    from console_data import bandpass

    lfp_fs = float(session.lfp_fs)
    lv_fs = float(session.LV_Fs)
    W = float(wt1) - float(wt0)
    site = _nearest_lfp_site_on_shank(session, elec)
    want_sniff = bool(show_sniff and session.SNF is not None)
    want_eth = session.ETH is not None

    # common within-trial time grids
    n_lfp = max(2, int(round(W * lfp_fs)))
    rel_lfp = float(wt0) + np.arange(n_lfp) / lfp_fs
    n_lv = max(2, int(round(W * lv_fs)))
    rel_lv = float(wt0) + np.arange(n_lv) / lv_fs

    lfp_rows, eth_rows, snf_rows, spike_rows = [], [], [], []
    used_trials, skipped_trials = [], []
    dur = session.lfp_duration_s or 0.0

    for tr in trials:
        t_start = session.trial_window_to_abs(int(tr), 0.0, 0.0)[0]
        a0, a1 = t_start + float(wt0), t_start + float(wt1)
        if a0 < 0 or (dur and a1 > dur):
            skipped_trials.append(int(tr))
            continue

        # nearest-LFP (µV) on the common grid
        if site is not None and session.lfp.shape[0] > site:
            tt, seg = session.lfp_window(band, a0, a1)
            row = np.interp(rel_lfp, tt - t_start, seg[site] * 1e6)
        else:
            row = np.full(n_lfp, np.nan)
        lfp_rows.append(row)

        # ethanol (mean-subtracted) on the common grid
        if want_eth:
            te, ve = session.sensor_window("ETH", a0, a1)
            if ve is not None and ve.size:
                eth_rows.append(np.interp(rel_lv, te - t_start,
                                          ve - np.nanmean(ve)))
            else:
                eth_rows.append(np.full(n_lv, np.nan))

        # sniff (mean-subtracted, band-passed like the view) on the common grid
        if want_sniff:
            ts, vs = session.sensor_window("SNF", a0, a1)
            if vs is not None and vs.size > 3:
                vb = bandpass(vs - np.nanmean(vs), lv_fs, 0.5, 12.0)
                snf_rows.append(np.interp(rel_lv, ts - t_start, vb))
            else:
                snf_rows.append(np.full(n_lv, np.nan))

        # this unit's spikes -> within-trial seconds
        st = session.unit_spikes_in_window(int(uid), a0, a1)
        spike_rows.append(np.asarray(st, float) - t_start)
        used_trials.append(int(tr))

    n_used = len(used_trials)

    # peak-electrode / lfp-site descriptors for title + metadata
    sh = session.shank_of(int(elec))
    edep = float(session.chan_y[int(elec)])
    ex = float(session.chan_x[int(elec)])
    site_xy = (float(session.lfp_xcoord[site]), float(session.lfp_ycoord[site])) \
        if (site is not None and session.lfp_ycoord.size > site) else (None, None)

    meta = dict(kind="unit_detail", date=session.date,
                experiment_name=session.experiment_name, h5_path=h5_path,
                unit_id=int(uid), peak_electrode=int(elec),
                peak_electrode_x_um=ex, peak_electrode_y_um=edep,
                peak_electrode_shank=int(sh),
                lfp_site_row=(int(site) if site is not None else None),
                lfp_site_x_um=site_xy[0], lfp_site_y_um=site_xy[1],
                band=band_name, band_hz=band, lfp_fs=lfp_fs, LV_Fs=lv_fs,
                within_trial_start_s=float(wt0), within_trial_end_s=float(wt1),
                psth_bin_s=float(psth_bin_s),
                n_trials_used=n_used, trials_used=used_trials,
                trials_skipped_out_of_range=skipped_trials,
                sniff_shown=want_sniff, ethanol_shown=want_eth)

    # ---- no usable trials: return an explanatory figure -------------------
    if n_used == 0:
        fig, ax = plt.subplots(figsize=(figwidth, 3.0))
        ax.axis("off")
        ax.text(0.5, 0.5, "Unit %d: no trials fit the window [%.3f, %.3f] s\n"
                "within the recording." % (int(uid), wt0, wt1),
                ha="center", va="center")
        return fig, meta

    # ---- lay out the panels ----------------------------------------------
    panels = ["eth"] if want_eth else []
    if want_sniff:
        panels.append("snf")
    panels.append("lfp")
    panels.append("raster")
    panels.append("psth")
    ratio = {"eth": 1.0, "snf": 1.0, "lfp": 1.1, "raster": 3.4, "psth": 1.5}
    heights = [ratio[p] for p in panels]

    fig_h = float(np.clip(sum(heights) * 1.05 + 1.0, 5.5, 22.0))
    fig = plt.figure(figsize=(figwidth, fig_h))
    gs = gridspec.GridSpec(len(panels), 1, height_ratios=heights, hspace=0.12)
    axes = {}
    ax0 = None
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[i], sharex=ax0 if ax0 is not None else None)
        if ax0 is None:
            ax0 = ax
        axes[p] = ax

    overlay_alpha = float(np.clip(4.0 / max(n_used, 1), 0.03, 0.5))

    def _overlay(ax, x, rows, color, label, unit):
        if not rows:
            ax.text(0.01, 0.5, "no %s" % label, transform=ax.transAxes,
                    fontsize=8, color="0.5", va="center")
            return
        M = np.vstack(rows)
        ax.plot(x, M.T, color=color, alpha=overlay_alpha, lw=0.4)
        ax.plot(x, np.nanmean(M, axis=0), color=color, lw=1.8)
        ax.set_ylabel("%s\n(%s)" % (label, unit), fontsize=8, color=color)
        ax.tick_params(labelsize=7)
        ax.margins(x=0)

    if want_eth:
        _overlay(axes["eth"], rel_lv, eth_rows, "#2ca02c", "Ethanol", "a.u.")
    if want_sniff:
        _overlay(axes["snf"], rel_lv, snf_rows, "#d62728", "Sniff", "a.u.")
    _overlay(axes["lfp"], rel_lfp, lfp_rows, "#1f77b4",
             "LFP nearest unit", "µV")

    # raster: one row per trial, trial 1 at the top
    axr = axes["raster"]
    axr.eventplot(spike_rows, lineoffsets=np.arange(n_used),
                  linelengths=0.85, colors="0.1", linewidths=0.5)
    axr.set_ylim(n_used - 0.5, -0.5)          # invert -> first trial on top
    axr.set_ylabel("trial", fontsize=8)
    if n_used <= 25:
        axr.set_yticks(np.arange(n_used))
        axr.set_yticklabels([str(t) for t in used_trials], fontsize=6)
    else:
        axr.tick_params(labelsize=7)
    axr.margins(x=0)

    # PSTH pooled across trials (firing rate, Hz)
    axp = axes["psth"]
    nbins = max(5, int(round(W / float(psth_bin_s))))
    edges = np.linspace(float(wt0), float(wt1), nbins + 1)
    bw = (float(wt1) - float(wt0)) / nbins
    allsp = np.concatenate(spike_rows) if spike_rows else np.empty(0)
    counts, _ = np.histogram(allsp, bins=edges)
    rate = counts / (n_used * bw) if (n_used and bw > 0) else counts * 0.0
    axp.bar(edges[:-1], rate, width=np.diff(edges), align="edge",
            color="#555555", edgecolor="none")
    axp.set_ylabel("PSTH\n(Hz)", fontsize=8)
    axp.set_xlabel("Time in trial (s)", fontsize=9)
    axp.margins(x=0)
    axp.set_xlim(float(wt0), float(wt1))

    # hide x tick labels on every panel except the bottom (PSTH)
    for p in panels[:-1]:
        plt.setp(axes[p].get_xticklabels(), visible=False)

    title = ("%s — unit %d — Sh%d · %.0f µm — %s — [%.3f, %.3f] s · %d trials"
             % (session.date, int(uid), sh, edep, band_name, wt0, wt1, n_used))
    axes[panels[0]].set_title(title, fontsize=10)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.07)
    return fig, meta

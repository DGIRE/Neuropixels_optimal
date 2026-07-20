"""
plot_unit_locations.py
----------------------
Python translation of plot_unit_locations.m (MATLAB).

Produces a multi-panel figure summarising where spiking units are located
on the Neuropixels probe:

  - LEFT panel(s)  : probe map — unit depth vs. firing rate colour / amplitude
                     dot size.  One column per shank for multi-shank probes.
  - MIDDLE panel   : horizontal bar histogram of unit counts per 50-µm depth bin.
  - RIGHT panel    : histogram of firing-rate distribution with median annotation.

Entry point
-----------
    fig = plot_unit_locations(D)
    fig.savefig('unit_locations.png', dpi=150, bbox_inches='tight')

MATLAB → Python differences
----------------------------
* MATLAB ``figure('Position', …)`` → ``plt.subplots(figsize=…)``.
* MATLAB ``scatter(ax, x, y, sz, c, 'filled')`` → ``ax.scatter(…)``.
* MATLAB ``colorbar(ax)`` → ``fig.colorbar(sc, ax=ax)``.
* MATLAB ``barh(ax, centers, counts, 1)`` → ``ax.barh(centers, counts, height=…)``.
* MATLAB ``histogram(fr, 'BinWidth', 1)`` → ``ax.hist(fr, bins=np.arange(…, 1.0))``.
"""

from __future__ import annotations

import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_unit_locations(D: dict) -> matplotlib.figure.Figure:
    """
    Visualize spiking neuron locations on the NP probe array.

    Creates a figure with the following panels:

      LEFT   - Probe map: each unit as a colored dot at its depth on the
               probe.  Color = firing rate (spikes/s).  Dot size = mean
               spike amplitude.  Gray dots show all electrode sites.
               For multi-shank probes, shanks are shown side-by-side.

      MIDDLE - Firing-rate histogram: number of units in each 50-µm
               depth bin along the probe.

      RIGHT  - Firing-rate distribution: histogram of firing rates
               across all units.

    Parameters
    ----------
    D : dict
        Dict returned by load_experiment_data().  Required keys:
        ``unitIDs``, ``unitDepths``, ``unitFiringRate``, ``unitAmps``,
        ``xcoords``, ``ycoords``, ``NP_Fs``,
        ``sp`` (dict with at minimum keys ``st`` and ``temps``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Caller is responsible for plt.show() or plt.savefig().
    """

    # ---- Configuration ------------------------------------------------
    DEPTH_BIN_UM = 50    # µm per depth bin for the histogram
    MIN_FR       = 0.1   # units below this firing rate (sp/s) shown in gray

    # Shank x-coordinate boundaries (µm) and labels (0-indexed, 0–3)
    # NP 2.0: 4 shanks separated by ~250 µm; single-shank probes have x ~ 0
    SHANK_BOUNDARIES = [-np.inf, -100.0, 100.0, 400.0, np.inf]
    SHANK_LABELS     = [0, 1, 2, 3]
    N_SHANKS         = 4

    # ---- Guard: empty unit list --------------------------------------
    unit_ids = np.asarray(D["unitIDs"]).ravel()
    if unit_ids.size == 0:
        warnings.warn("No units found in D. Cannot plot unit locations.")
        return plt.figure()

    recording_dur = float(D["sp"]["st"][-1])
    n_units       = unit_ids.size

    # ---- Peak-channel lookup per template ----------------------------
    # D['sp']['temps'] shape: [nTemplates × nTimepoints × nChannels]
    # Max absolute amplitude across the time axis (axis=1 in Python ≡ dim 2 in MATLAB)
    # Result: [nTemplates × nChannels]
    temps         = np.asarray(D["sp"]["temps"])
    n_templates   = temps.shape[0]
    temps_max_abs = np.squeeze(np.max(np.abs(temps), axis=1))

    # argmax across channels gives 0-indexed peak channel per template
    peak_chan = np.argmax(temps_max_abs, axis=1)   # shape: (nTemplates,)

    # Map unit IDs → template peak channel — vectorised.
    # unit_ids are 0-indexed cluster IDs from Kilosort.  For IDs that are
    # out of range (>= n_templates), fall back to channel 0, mirroring the
    # original loop's default (np.zeros initialisation).
    # np.clip prevents out-of-bounds indexing; np.where then selects 0 for
    # those rows, so the clipped-but-invalid index values are never used.
    valid_uid      = unit_ids < n_templates
    unit_peak_chan = np.where(
        valid_uid,
        peak_chan[np.clip(unit_ids, 0, n_templates - 1)],
        0,
    ).astype(int)

    xcoords = np.asarray(D["xcoords"]).ravel()
    ycoords = np.asarray(D["ycoords"]).ravel()

    # Clamp to valid index range (mirrors MATLAB's max(unitPeakChan, 1))
    unit_peak_chan = np.clip(unit_peak_chan, 0, xcoords.size - 1)
    unit_xcoord   = xcoords[unit_peak_chan]

    # ---- Assign each unit to a shank ---------------------------------
    # MATLAB loop: for s=1:4, uses shankBoundaries(s) and shankBoundaries(s+1)
    # with 1-indexed MATLAB → Python: SHANK_BOUNDARIES[s] and [s+1] (0-indexed)
    unit_shank = np.zeros(n_units, dtype=int)
    for s in range(N_SHANKS):
        in_shank = (unit_xcoord >= SHANK_BOUNDARIES[s]) & \
                   (unit_xcoord <  SHANK_BOUNDARIES[s + 1])
        unit_shank[in_shank] = SHANK_LABELS[s]

    shanks_present   = np.unique(unit_shank)    # sorted, as in MATLAB unique()
    n_shanks_present = shanks_present.size

    # ---- Normalize visual properties ---------------------------------
    unit_fr    = np.asarray(D["unitFiringRate"]).ravel()
    unit_depth = np.asarray(D["unitDepths"]).ravel()
    unit_amp   = np.asarray(D["unitAmps"]).ravel()

    # Dot size: scaled by amplitude; uniform fallback when amps unavailable
    if unit_amp.size == 0 or np.all(np.isnan(unit_amp)):
        dot_size = np.full(n_units, 60.0)
    else:
        amp_min  = np.nanmin(unit_amp)
        amp_max  = np.nanmax(unit_amp)
        amp_norm = (unit_amp - amp_min) / (amp_max - amp_min + np.finfo(float).eps)
        dot_size = 20.0 + amp_norm * 120.0   # 20–140 pt²

    # Color by log10 firing rate for better dynamic range
    log_fr    = np.log10(np.maximum(unit_fr, 0.001))
    log_fr_lo = np.log10(MIN_FR)
    log_fr_hi = float(np.max(log_fr)) + 0.1

    # ---- Depth bins --------------------------------------------------
    max_y        = float(np.max(ycoords))
    depth_range  = (0.0, max_y + DEPTH_BIN_UM)
    # np.arange with a small epsilon ensures the endpoint edge is included
    depth_edges  = np.arange(depth_range[0], depth_range[1] + 1e-9, DEPTH_BIN_UM)
    depth_centers = depth_edges[:-1] + DEPTH_BIN_UM / 2.0

    # ---- Build figure ------------------------------------------------
    # MATLAB: Position = [50, 50, 280 + 200*nShanksPresent, 700] (pixels)
    # Convert to inches at ~100 dpi: width = (280 + 200*n)/100, height = 7
    n_cols    = n_shanks_present + 2
    fig_w     = (280 + 200 * n_shanks_present) / 100.0
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 7.0), facecolor="w")

    # Ensure axes is always a list for uniform indexing
    if n_cols == 1:
        axes = [axes]

    # ---- LEFT PANELS: Probe maps (one per shank present) -------------
    first_sc = None   # reference scatter for shared colorbar

    for si, sh in enumerate(shanks_present):
        ax = axes[si]
        ax.set_facecolor("w")

        # Background: all electrode sites on this shank.
        # MATLAB: shankBoundaries(sh+1) and shankBoundaries(sh+2) (1-indexed MATLAB)
        # Python:  SHANK_BOUNDARIES[sh] and SHANK_BOUNDARIES[sh+1] (0-indexed)
        chan_on_shank = (xcoords >= SHANK_BOUNDARIES[sh]) & \
                        (xcoords <  SHANK_BOUNDARIES[sh + 1])
        ax.scatter(
            np.zeros(int(chan_on_shank.sum())),
            ycoords[chan_on_shank],
            s=8,
            color=[0.85, 0.85, 0.85],
            alpha=0.7,
            linewidths=0,
        )

        # Units on this shank, with random jitter in x for visibility
        on_shank = unit_shank == sh
        if on_shank.any():
            n_on   = int(on_shank.sum())
            jitter = np.random.randn(n_on) * 0.1   # cosmetic only; non-reproducible
            sc = ax.scatter(
                jitter,
                unit_depth[on_shank],
                s=dot_size[on_shank],
                c=log_fr[on_shank],
                cmap="hot",
                vmin=log_fr_lo,
                vmax=log_fr_hi,
                edgecolors="k",
                linewidths=0.5,
            )
            if first_sc is None:
                first_sc = sc

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(depth_range)
        ax.set_xticks([])
        ax.set_ylabel("Depth from probe tip (µm)", fontsize=10)
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

        n_on_shank = int(on_shank.sum())   # reuse mask already computed above
        if n_shanks_present > 1:
            ax.set_title(f"Shank {sh}\n({n_on_shank} units)", fontsize=10)
        else:
            ax.set_title(f"{n_on_shank} units", fontsize=10)

    # Shared colorbar for probe panels, attached to first probe axis
    if first_sc is not None:
        cb = fig.colorbar(first_sc, ax=axes[0])
        cb_ticks = np.log10([0.1, 1, 5, 10, 50, 100])
        cb.set_ticks(cb_ticks)
        cb.set_ticklabels(["0.1", "1", "5", "10", "50", "100"])
        cb.set_label("Firing rate (sp/s)", fontsize=10)

    # ---- MIDDLE PANEL: Depth histogram --------------------------------
    ax_h = axes[n_shanks_present]

    unit_counts, _ = np.histogram(unit_depth, bins=depth_edges)

    # MATLAB: barh(ax, centers, counts, width=1, ...)
    # matplotlib barh: height = bar thickness in data (y) units = DEPTH_BIN_UM
    ax_h.barh(
        depth_centers,
        unit_counts,
        height=DEPTH_BIN_UM,
        color=[0.3, 0.5, 0.8],
        edgecolor="none",
    )
    ax_h.set_xlabel("Units per bin", fontsize=10)
    ax_h.set_ylabel("Depth (µm)", fontsize=10)
    ax_h.set_title(f"Units / {DEPTH_BIN_UM} µm", fontsize=10)
    ax_h.set_ylim(depth_range)
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)
    ax_h.grid(True)
    ax_h.tick_params(labelsize=10)

    # ---- RIGHT PANEL: Firing rate distribution -----------------------
    ax_fr = axes[n_shanks_present + 1]

    # MATLAB: histogram(..., 'BinWidth', 1)  →  bins of width 1 sp/s
    fr_max = float(np.max(unit_fr)) if unit_fr.size > 0 else 1.0
    fr_bins = np.arange(0.0, np.ceil(fr_max) + 1.0 + 1e-9, 1.0)

    ax_fr.hist(unit_fr, bins=fr_bins, color=[0.2, 0.6, 0.4], edgecolor="none")
    ax_fr.set_xlabel("Firing rate (sp/s)", fontsize=10)
    ax_fr.set_ylabel("Number of units", fontsize=10)
    ax_fr.set_title("FR distribution", fontsize=10)
    ax_fr.spines["top"].set_visible(False)
    ax_fr.spines["right"].set_visible(False)
    ax_fr.grid(True)
    ax_fr.tick_params(labelsize=10)

    # Annotate median FR with a vertical dashed line
    med_fr    = float(np.median(unit_fr))
    y_lim_fr  = ax_fr.get_ylim()
    ax_fr.plot([med_fr, med_fr], list(y_lim_fr), "r--", linewidth=1.5)
    ax_fr.text(
        med_fr,
        y_lim_fr[1] * 0.95,
        f" med={med_fr:.1f}",
        color="r",
        fontsize=9,
        va="top",
    )
    ax_fr.set_ylim(y_lim_fr)   # restore limits after adding line/text

    # ---- Figure-level title ------------------------------------------
    fig.suptitle(
        f"Unit Locations on NP Probe\n"
        f"{n_units} units  |  recording duration: {recording_dur:.0f} s  |  "
        f"NP sampling rate: {float(D['NP_Fs']):.0f} Hz",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    return fig

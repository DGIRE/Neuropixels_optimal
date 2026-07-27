"""
raster_display.py
=================
Data helpers for the LFP + unit-raster (+ sniff) display of a specified time
window from a specified experiment in the aggregate.

The display (built in run_raster_examples.py) stacks, bottom (tip) to top
(surface): all 32 LFP channels, and under each LFP trace the spike rasters of the
units that sit at that LFP site's depth or below it (but above the next, deeper
LFP site). If the experiment has a strong sniff signal it is drawn across the top.

Functions
---------
    list_lfp_dates(h5_path)                 -> dates with LFP
    load_session_full(h5_path, date)        -> LFP + spikes(+clusters) + sniff
    bandpass(x, fs, lo, hi, order)          -> zero-phase band-pass
    window_indices(fs, n_time, t0, t1)      -> sample slice for a window
    sniff_strength(SNF, LV_Fs, band)        -> (ratio, is_strong-able metric)
    assign_units_to_bins(unitDepths, lfp_ycoord) -> LFP-site bin index per unit
    per_unit_spike_times(...)               -> {uid: spike times in window}

Per-unit rasters need the spike->cluster map. The aggregate stores that as
`spikeClusters` (added 2026-07-26). If a session predates that, load_session_full
returns has_clusters=False and the driver falls back to a depth-binned population
raster built from spikeTimes + spikeDepths.

Data conventions: DATA/Aggregate/AGGREGATE_CONVENTIONS.txt. Depths (lfp_ycoord,
unitDepths, spikeDepths) are all microns from the probe tip, so they compare
directly. LFP rows are ordered index 0 = tip (deep) .. -1 = surface.
"""
from __future__ import annotations

import numpy as np
import scipy.signal
import h5py


def list_lfp_dates(h5_path: str) -> list[str]:
    out = []
    with h5py.File(h5_path, "r") as f:
        for d in sorted(f.keys()):
            g = f[d]
            if bool(g.attrs.get("has_lfp", "LFP" in g)) and "LFP" in g:
                out.append(str(d))
    return out


def load_session_full(h5_path: str, date: str) -> dict | None:
    """Load LFP (required) plus spikes and sniff if present."""
    with h5py.File(h5_path, "r") as f:
        if date not in f:
            return None
        g = f[date]
        if not (bool(g.attrs.get("has_lfp", "LFP" in g)) and "LFP" in g):
            return None
        lfp = np.asarray(g["LFP"][()], dtype=np.float64)
        n_ch = lfp.shape[0]
        fs = float(g.attrs.get("LFP_fs", 250.0))
        yc = (np.asarray(g["lfp_ycoord"][()], dtype=np.float64)
              if "lfp_ycoord" in g else np.arange(n_ch, dtype=np.float64))
        row = (np.asarray(g["lfp_channel_row"][()])
               if "lfp_channel_row" in g else np.arange(n_ch))
        dur = float(g.attrs.get("LFP_duration_s", lfp.shape[1] / fs))
        out = dict(date=str(date), lfp=lfp, fs=fs, ycoord=yc, channel_row=row,
                   experiment_name=str(g.attrs.get("experiment_name", "")),
                   duration_s=dur, has_spikes=False, has_clusters=False,
                   has_sensors=False)

        # spikes
        if bool(g.attrs.get("has_spikes", False)) and "spikeTimes" in g:
            out["has_spikes"] = True
            out["spikeTimes"] = np.asarray(g["spikeTimes"][()], dtype=np.float64)
            out["spikeDepths"] = (np.asarray(g["spikeDepths"][()], dtype=np.float64)
                                  if "spikeDepths" in g else None)
            out["unitIDs"] = (np.asarray(g["unitIDs"][()])
                              if "unitIDs" in g else None)
            out["unitDepths"] = (np.asarray(g["unitDepths"][()], dtype=np.float64)
                                 if "unitDepths" in g else None)
            if "spikeClusters" in g:
                out["spikeClusters"] = np.asarray(g["spikeClusters"][()])
                out["has_clusters"] = True

        # sniff / sensors
        if bool(g.attrs.get("has_sensors", False)) and "SNF" in g:
            out["has_sensors"] = True
            out["SNF"] = np.asarray(g["SNF"][()], dtype=np.float64)
            out["LV_Fs"] = float(g.attrs.get("LV_Fs", 125.0))
        return out


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.999)
    lo = max(lo, 1e-6)
    b, a = scipy.signal.butter(order, [lo / nyq, hi / nyq], btype="band")
    default_pad = 3 * (max(len(a), len(b)) - 1)
    n = x.shape[-1]
    padlen = min(default_pad, n - 1) if n > 1 else 0
    return scipy.signal.filtfilt(b, a, x, axis=-1, padlen=padlen)


def window_indices(fs: float, n_time: int, t0: float, t1: float) -> tuple[int, int]:
    i0 = max(0, int(round(t0 * fs)))
    i1 = min(n_time, int(round(t1 * fs)))
    if i1 <= i0:
        raise ValueError(
            "window %g-%gs empty for a %gs recording" % (t0, t1, n_time / fs))
    return i0, i1


def sniff_strength(SNF: np.ndarray, LV_Fs: float,
                   band: tuple[float, float] = (0.5, 12.0)) -> float:
    """Fraction of the sniff signal's amplitude that lives in the sniff band.

    ratio = std(bandpassed) / std(mean-subtracted raw). Near 1 for a clean
    respiratory rhythm; low for a flat / noisy 'nosniff' channel. Returns 0.0 if
    the signal is unusable.
    """
    x = np.asarray(SNF, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < int(LV_Fs) or np.std(x) == 0:
        return 0.0
    x = x - x.mean()
    xb = bandpass(x, LV_Fs, band[0], band[1])
    denom = np.std(x)
    return float(np.std(xb) / denom) if denom > 0 else 0.0


def assign_units_to_bins(unitDepths: np.ndarray,
                         lfp_ycoord: np.ndarray) -> np.ndarray:
    """LFP-site bin index for each unit (or -1 if depth is undefined).

    LFP sites are sorted by depth ascending (index 0 = tip / deepest). A unit at
    depth D belongs to the site at D or the next site ABOVE it -- i.e. the smallest
    site index i with lfp_ycoord[i] >= D. Units deeper than every site go to bin 0;
    units above the top site go to the last bin. So each LFP trace owns the units
    at its depth and below, down to (not including) the next deeper LFP site.
    """
    yc = np.asarray(lfp_ycoord, dtype=np.float64)
    order = np.argsort(yc, kind="stable")
    yc_sorted = yc[order]
    D = np.asarray(unitDepths, dtype=np.float64)
    bins = np.full(D.shape, -1, dtype=np.int64)
    good = np.isfinite(D)
    idx = np.searchsorted(yc_sorted, D[good], side="left")
    idx = np.clip(idx, 0, yc_sorted.size - 1)
    # map back to the site indices in the ORIGINAL lfp_ycoord ordering
    bins[good] = order[idx]
    return bins


def per_unit_spike_times(spikeTimes, spikeClusters, unitIDs,
                         t0: float, t1: float) -> dict:
    """{uid: spike times within [t0, t1)} using the spike->cluster map."""
    st = np.asarray(spikeTimes, dtype=np.float64)
    clu = np.asarray(spikeClusters)
    win = (st >= t0) & (st < t1)
    st_w, clu_w = st[win], clu[win]
    return {int(uid): st_w[clu_w == uid] for uid in np.asarray(unitIDs)}

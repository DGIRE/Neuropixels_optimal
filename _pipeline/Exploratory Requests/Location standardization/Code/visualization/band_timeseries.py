"""
band_timeseries.py
==================
Band-filtered LFP time-series viewer for the cross-experiment aggregate.

For a chosen experiment and time window it produces, per experiment, two figures:
a stack of all 32 LFP channels filtered to the THETA band, and a second stack
filtered to the GAMMA band. Channels are stacked by depth (tip at the bottom,
surface at the top), so the depth structure of each rhythm is visible at a glance.

Functions
---------
    load_lfp_session(h5_path, date)      -> one experiment's LFP + depth coords
    list_lfp_dates(h5_path)              -> every date that has LFP
    bandpass(lfp, fs, lo, hi, order)     -> zero-phase band-pass (filtfilt)
    window_indices(fs, n_time, t0, t1)   -> sample slice for a time window
    (plotting lives in run_band_examples.py)

Notes
-----
* The aggregate LFP is 250 Hz, broadband DC..~100 Hz (float32). Theta sits well
  inside the band; a gamma band up to ~80-100 Hz is near the anti-alias ceiling,
  so pick gamma edges below ~100 Hz.
* Filtering is done on the FULL-duration trace, then the window is sliced out, so
  there are no filter edge artefacts at the window boundaries.
* LFP rows are ordered index 0 = deepest (tip) .. -1 = surface; lfp_ycoord gives
  each channel's depth in microns.

Author: Olfactory Research coding session. Exploratory / informal.
"""
from __future__ import annotations

import numpy as np
import scipy.signal
import h5py


def list_lfp_dates(h5_path: str) -> list[str]:
    """Every date key in the aggregate that has an LFP array."""
    out = []
    with h5py.File(h5_path, "r") as f:
        for d in sorted(f.keys()):
            g = f[d]
            if bool(g.attrs.get("has_lfp", "LFP" in g)) and "LFP" in g:
                out.append(str(d))
    return out


def load_lfp_session(h5_path: str, date: str) -> dict | None:
    """Load one experiment's LFP block, or None if it is absent / has no LFP."""
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
        return dict(date=str(date), lfp=lfp, fs=fs, ycoord=yc, channel_row=row,
                    experiment_name=str(g.attrs.get("experiment_name", "")),
                    duration_s=dur)


def bandpass(lfp: np.ndarray, fs: float, lo: float, hi: float,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass, applied per channel (axis=-1).

    hi is clamped just below Nyquist so a gamma edge at/above fs/2 does not raise.
    """
    lfp = np.atleast_2d(np.asarray(lfp, dtype=np.float64))
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.999)
    lo = max(lo, 1e-6)
    b, a = scipy.signal.butter(order, [lo / nyq, hi / nyq], btype="band")
    # padlen must be < signal length; clamp for very short windows
    default_pad = 3 * (max(len(a), len(b)) - 1)
    padlen = min(default_pad, lfp.shape[-1] - 1) if lfp.shape[-1] > 1 else 0
    return scipy.signal.filtfilt(b, a, lfp, axis=-1, padlen=padlen)


def window_indices(fs: float, n_time: int, t0: float, t1: float) -> tuple[int, int]:
    """Sample slice [i0, i1) for the time window [t0, t1) seconds, clipped to data.

    Raises ValueError if the (clipped) window is empty.
    """
    i0 = max(0, int(round(t0 * fs)))
    i1 = min(n_time, int(round(t1 * fs)))
    if i1 <= i0:
        raise ValueError(
            f"Requested window {t0}-{t1}s is empty for this recording "
            f"(has {n_time} samples = {n_time / fs:.1f}s).")
    return i0, i1

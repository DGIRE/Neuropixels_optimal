"""
probe_standardization.py
========================
A small suite of functions to *informally* assess whether Neuropixels probes
were inserted at comparable angles across experiments, using the laminar
(depth) distribution of LFP spectral power stored in the cross-experiment
aggregate ``np_aggregate.h5``.

The idea
--------
Different insertion angles change how spikes and LFP are distributed along the
probe (the laminar profile). For each experiment we build a
**depth x frequency** power matrix (depth on the y-axis, an FFT-based power
spectrum running along the x-axis). We then treat each experiment's matrix as a
single high-dimensional observation and run **PCA across experiments** so each
experiment becomes one point in a 2-D plane. Experiments with similar laminar
spectral profiles land near each other; outliers (e.g. a very different
insertion angle) separate out.

What this module provides
-------------------------
    load_lfp_sessions(h5_path)                 -> list of per-experiment dicts
    canonical_freq_grid(fs, nperseg, fmin, fmax)
    welch_psd(lfp, fs, nperseg, noverlap)
    depth_frequency_matrix(session, f_common, depth_bins, ...) -> (depth x freq)
    build_feature_matrix(sessions, ...)        -> (dates, mats[n_exp, depth, freq])
    pca_across_experiments(mats, ...)          -> dict(scores, loadings, explained, ...)

The figures are drawn by ``run_probe_standardization.py`` (which imports this
module). Nothing here writes files, so the functions are easy to reuse or test.

Data conventions used (see DATA/Aggregate/AGGREGATE_CONVENTIONS.txt)
--------------------------------------------------------------------
* One HDF5 group per experiment, keyed by date "MM-DD-YYYY".
* ``LFP``            float32 (n_channels, n_time), rows ordered by depth,
                     index 0 = deepest (tip) .. index -1 = surface.
* ``LFP_fs``         attr, LFP sample rate in Hz (250.0 for every session).
* ``lfp_ycoord``     float32 (n_channels,), channel depth in microns.
* ``lfp_channel_row``int32   (n_channels,), AP row index of each retained channel.
* ``has_lfp`` / ``has_spikes`` / ``has_sensors`` group attrs -- always check.

Only sessions with LFP are used. In the current aggregate that is the subset of
dates whose build status was lfp="ok"; after the LFP-geometry fix + targeted
rebuild every included date will have LFP and this code will pick them all up
automatically (it iterates over whatever is present).

Author: Olfactory Research coding session (2026-07-25). Exploratory / informal.
"""
from __future__ import annotations

import numpy as np
import scipy.signal
import h5py


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def load_lfp_sessions(h5_path: str) -> list[dict]:
    """Read every experiment that has LFP from the aggregate HDF5.

    Parameters
    ----------
    h5_path : path to np_aggregate.h5

    Returns
    -------
    list of dicts (one per experiment that has LFP), each with keys:
        date            "MM-DD-YYYY"
        lfp             float64 (n_channels, n_time)
        fs              float, LFP sample rate (Hz)
        ycoord          float64 (n_channels,), depth in microns (may be NaN-free)
        channel_row     int     (n_channels,), AP row of each retained channel
        experiment_name str
        duration_s      float
    Sorted by date key.
    """
    sessions: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        for date in sorted(f.keys()):
            g = f[date]
            # accept the session only if an LFP array is actually present
            has_lfp = bool(g.attrs.get("has_lfp", "LFP" in g))
            if not has_lfp or "LFP" not in g:
                continue
            lfp = np.asarray(g["LFP"][()], dtype=np.float64)
            if lfp.ndim != 2 or lfp.shape[1] < 2:
                # nothing usable to spectrally analyse
                continue
            n_ch = lfp.shape[0]
            fs = float(g.attrs.get("LFP_fs", 250.0))
            if "lfp_ycoord" in g:
                yc = np.asarray(g["lfp_ycoord"][()], dtype=np.float64)
            else:
                yc = np.arange(n_ch, dtype=np.float64)
            if "lfp_channel_row" in g:
                row = np.asarray(g["lfp_channel_row"][()])
            else:
                row = np.arange(n_ch)
            dur = float(g.attrs.get("LFP_duration_s", lfp.shape[1] / fs))
            sessions.append(dict(
                date=str(date),
                lfp=lfp,
                fs=fs,
                ycoord=yc,
                channel_row=row,
                experiment_name=str(g.attrs.get("experiment_name", "")),
                duration_s=dur,
            ))
    return sessions


# ---------------------------------------------------------------------------
# 2. Spectral estimation
# ---------------------------------------------------------------------------
def canonical_freq_grid(fs: float, nperseg: int,
                        fmin: float, fmax: float) -> np.ndarray:
    """The Welch frequency grid (rfft bins) for (fs, nperseg), clipped to [fmin, fmax].

    Because every session shares fs, using a fixed nperseg gives every session
    the same frequency grid, so their matrices are directly comparable. We still
    interpolate onto this grid in depth_frequency_matrix() so short/partial
    recordings (which force Welch to shrink nperseg) stay aligned.
    """
    f = np.fft.rfftfreq(int(nperseg), d=1.0 / fs)
    m = (f >= fmin) & (f <= fmax)
    return f[m]


def apply_notch(lfp: np.ndarray, fs: float, notch_freq: float,
                 notch_q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch filter at ``notch_freq`` (e.g. 60 Hz mains hum).

    Applied to the raw per-channel LFP time series BEFORE spectral estimation,
    so the line-noise energy is genuinely removed from the Welch PSD (and
    therefore from the PCA feature matrix) rather than merely masked/blanked
    on the display afterward. ``notch_q`` is the quality factor
    (bandwidth = notch_freq / notch_q); scipy's default-style Q=30 gives a
    narrow notch (~2 Hz at 60 Hz) that leaves neighboring frequencies intact.

    No-ops (returns ``lfp`` unchanged) if ``notch_freq`` is None or falls at/
    above the Nyquist frequency for this session's sample rate.
    """
    if notch_freq is None:
        return lfp
    nyq = fs / 2.0
    if notch_freq <= 0 or notch_freq >= nyq:
        return lfp
    b, a = scipy.signal.iirnotch(w0=notch_freq, Q=notch_q, fs=fs)
    return scipy.signal.filtfilt(b, a, lfp, axis=-1)


def welch_psd(lfp: np.ndarray, fs: float, nperseg: int, noverlap: int,
              detrend: str = "constant", notch_freq: float | None = None,
              notch_q: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density for every channel.

    If ``notch_freq`` is given, a zero-phase IIR notch (see ``apply_notch``)
    is applied to the LFP first, so the returned PSD has that frequency's
    line-noise energy genuinely removed, not just interpolated/masked.

    Returns
    -------
    f    : (n_freq,)          frequency axis (Hz)
    Pxx  : (n_channels, n_freq) power spectral density (V**2/Hz)

    nperseg/noverlap are clamped to the available number of samples so this
    never raises on very short recordings; the caller re-grids onto f_common.
    """
    lfp = np.atleast_2d(np.asarray(lfp, dtype=np.float64))
    lfp = apply_notch(lfp, fs, notch_freq, notch_q)
    n_time = lfp.shape[1]
    nseg = int(min(nperseg, n_time))
    nseg = max(nseg, 8)
    nov = int(min(noverlap, nseg - 1))
    f, Pxx = scipy.signal.welch(
        lfp, fs=fs, window="hann", nperseg=nseg, noverlap=nov,
        detrend=detrend, axis=-1, scaling="density",
    )
    return f, np.atleast_2d(Pxx)


def depth_frequency_matrix(session: dict,
                           f_common: np.ndarray,
                           depth_bins: int,
                           nperseg: int,
                           noverlap: int,
                           log: bool = True,
                           eps: float = 1e-20,
                           notch_freq: float | None = None,
                           notch_q: float = 30.0) -> np.ndarray:
    """Build one experiment's depth x frequency power matrix on the common grid.

    Steps
    -----
    1. Welch PSD per channel (with an optional zero-phase notch at
       ``notch_freq`` applied to the LFP first -- see ``apply_notch``).
    2. Interpolate each channel's PSD onto ``f_common`` (frequency axis).
    3. Convert to dB (10*log10 power) if ``log``.
    4. Resample the depth axis onto ``depth_bins`` evenly spaced RELATIVE-depth
       rows (row fraction 0 = tip .. 1 = surface). Using relative depth makes
       the laminar axis comparable across experiments whose physical depth range
       differs -- which is exactly what insertion angle changes.

    Returns
    -------
    M : (depth_bins, n_freq) float64
        Row 0 = tip (deepest), row -1 = surface. Column = frequency in f_common.
    """
    f, Pxx = welch_psd(session["lfp"], session["fs"], nperseg, noverlap,
                        notch_freq=notch_freq, notch_q=notch_q)

    # (2) interpolate each channel onto the common frequency grid
    n_ch = Pxx.shape[0]
    P = np.empty((n_ch, f_common.size), dtype=np.float64)
    for c in range(n_ch):
        P[c] = np.interp(f_common, f, Pxx[c])

    # (3) log power (dB)
    if log:
        P = 10.0 * np.log10(P + eps)

    # (4) resample depth (rows) onto depth_bins using relative row fraction
    if n_ch == depth_bins:
        M = P
    else:
        src = np.linspace(0.0, 1.0, n_ch)
        dst = np.linspace(0.0, 1.0, depth_bins)
        M = np.empty((depth_bins, f_common.size), dtype=np.float64)
        for j in range(f_common.size):
            M[:, j] = np.interp(dst, src, P[:, j])
    return M


def build_feature_matrix(sessions: list[dict],
                         f_common: np.ndarray,
                         depth_bins: int,
                         nperseg: int,
                         noverlap: int,
                         log: bool = True,
                         notch_freq: float | None = None,
                         notch_q: float = 30.0) -> tuple[list[str], np.ndarray]:
    """Compute the depth x frequency matrix for every session.

    Returns
    -------
    dates : list[str]                       experiment date keys, in order
    mats  : (n_exp, depth_bins, n_freq)     stacked matrices
    """
    dates, mats = [], []
    for s in sessions:
        M = depth_frequency_matrix(s, f_common, depth_bins, nperseg, noverlap,
                                    log=log, notch_freq=notch_freq, notch_q=notch_q)
        dates.append(s["date"])
        mats.append(M)
    return dates, np.stack(mats, axis=0)


# ---------------------------------------------------------------------------
# 3. PCA across experiments
# ---------------------------------------------------------------------------
def pca_across_experiments(mats: np.ndarray,
                           per_experiment_center: bool = True,
                           standardize: bool = False) -> dict:
    """PCA with each experiment as one observation.

    Each experiment's (depth x freq) matrix is flattened to a feature vector.

    Parameters
    ----------
    mats : (n_exp, depth_bins, n_freq)
    per_experiment_center : if True, subtract each experiment's own mean before
        PCA. In dB this removes an overall power/gain offset (which varies with
        reference and amplifier settings), so PCA captures the *pattern* of
        laminar/spectral distribution rather than absolute level. Recommended
        for an insertion-angle / laminar-shape comparison.
    standardize : if True, scale each feature to unit variance across
        experiments (correlation-style PCA) after centering.

    Returns
    -------
    dict with:
        scores        (n_exp, k)          experiment coordinates in PC space
        loadings      (k, depth_bins*n_freq)  principal axes (rows), reshape a
                       row to (depth_bins, n_freq) to view a component's loading
                       across the depth x freq matrix
        singular_values (k,)
        explained     (k,)                fraction of variance per PC
        cum_explained (k,)                cumulative fraction
        mean          (1, features)       grand mean removed
        std           (1, features) or None
        shape         (depth_bins, n_freq)
    """
    mats = np.asarray(mats, dtype=np.float64)
    n_exp = mats.shape[0]
    grid_shape = mats.shape[1:]
    X = mats.reshape(n_exp, -1)

    if per_experiment_center:
        X = X - X.mean(axis=1, keepdims=True)

    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    sd = None
    if standardize:
        sd = Xc.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Xc = Xc / sd

    # SVD-based PCA (robust; no covariance matrix formation)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S                       # (n_exp, k)
    var = S ** 2
    explained = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
    cum = np.cumsum(explained)

    return dict(
        scores=scores,
        loadings=Vt,
        singular_values=S,
        explained=explained,
        cum_explained=cum,
        mean=mu,
        std=sd,
        shape=grid_shape,
    )

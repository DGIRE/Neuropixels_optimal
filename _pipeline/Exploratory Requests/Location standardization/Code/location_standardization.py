"""
location_standardization.py
===========================
Compare Neuropixels recording SITES across experiments by their LFP spectral
signature, so that sites sampling the same kind of tissue can be grouped
*regardless of the probe's insertion angle or which experiment they came from*.

Where the companion "probe standardization" analysis treated each EXPERIMENT as
one observation, here the unit is a single RECORDING SITE: one retained LFP
channel from one experiment. Every site contributes one row to a pooled master
matrix (row = site, column = frequency), and sites are then organised by
hierarchical clustering on their spectra and examined with PCA.

Pipeline
--------
    load_lfp_sessions(h5_path)                 -> per-experiment LFP + depth coords
    notch_filter(lfp, fs, f0=60, Q=30)         -> 60 Hz line-noise removed
    site_spectra(session, f_common, ...)       -> (n_channels x n_freq) dB spectra
    build_master_matrix(sessions, ...)         -> pooled (n_sites x n_freq) + site labels
    row_normalize(M)                           -> per-site spectral SHAPE
    hierarchical_linkage(features, ...)        -> scipy linkage Z (for dendrogram/order)
    pca_sites(features)                        -> per-site PC scores, loadings, variance

Figures are drawn by run_location_standardization.py (imports this module).

Why a 60 Hz notch, log power, and per-site normalisation
--------------------------------------------------------
* Notch (iirnotch at 60 Hz) removes mains line noise from the time series before
  the spectrum is estimated, so a shared 60 Hz artefact does not create spurious
  similarity/among-site structure.
* Spectra are 10*log10 power (dB).
* row_normalize subtracts each site's mean dB so clustering/PCA respond to the
  SHAPE of a site's spectrum (its tissue signature) rather than an overall
  power/gain/reference offset that differs between experiments and depths.

Data conventions: see DATA/Aggregate/AGGREGATE_CONVENTIONS.txt. LFP is float32
(n_channels, n_time), rows ordered index 0 = deepest (tip) .. -1 = surface, at
LFP_fs (250 Hz); lfp_ycoord gives each retained channel's depth in microns.
Only sessions with LFP are used (has_lfp flag / presence of the LFP dataset).

Author: Olfactory Research coding session. Exploratory / informal.
"""
from __future__ import annotations

import numpy as np
import scipy.signal
import h5py
from scipy.cluster.hierarchy import linkage


# ---------------------------------------------------------------------------
# 1. Loading (identical contract to the aggregate reader)
# ---------------------------------------------------------------------------
def load_lfp_sessions(h5_path: str) -> list[dict]:
    """Read every experiment that has LFP from the aggregate HDF5.

    Returns a list of dicts (sorted by date) with keys:
        date, lfp (float64 n_ch x n_time), fs, ycoord (n_ch,), channel_row (n_ch,),
        experiment_name, duration_s.
    """
    sessions: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        for date in sorted(f.keys()):
            g = f[date]
            has_lfp = bool(g.attrs.get("has_lfp", "LFP" in g))
            if not has_lfp or "LFP" not in g:
                continue
            lfp = np.asarray(g["LFP"][()], dtype=np.float64)
            if lfp.ndim != 2 or lfp.shape[1] < 2:
                continue
            n_ch = lfp.shape[0]
            fs = float(g.attrs.get("LFP_fs", 250.0))
            yc = (np.asarray(g["lfp_ycoord"][()], dtype=np.float64)
                  if "lfp_ycoord" in g else np.arange(n_ch, dtype=np.float64))
            row = (np.asarray(g["lfp_channel_row"][()])
                   if "lfp_channel_row" in g else np.arange(n_ch))
            dur = float(g.attrs.get("LFP_duration_s", lfp.shape[1] / fs))
            sessions.append(dict(
                date=str(date), lfp=lfp, fs=fs, ycoord=yc, channel_row=row,
                experiment_name=str(g.attrs.get("experiment_name", "")),
                duration_s=dur))
    return sessions


# ---------------------------------------------------------------------------
# 2. Filtering + spectra
# ---------------------------------------------------------------------------
def notch_filter(lfp: np.ndarray, fs: float, f0: float = 60.0,
                 Q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch at f0 Hz (default 60 Hz mains) applied per channel.

    Skips cleanly if f0 is falsy or >= Nyquist. filtfilt keeps the phase intact
    (broadband LFP is retained for any downstream cross-spectral work).
    """
    if not f0 or f0 <= 0 or f0 >= fs / 2.0:
        return np.asarray(lfp, dtype=np.float64)
    b, a = scipy.signal.iirnotch(f0, Q, fs)
    return scipy.signal.filtfilt(b, a, np.asarray(lfp, dtype=np.float64), axis=-1)


def canonical_freq_grid(fs: float, nperseg: int,
                        fmin: float, fmax: float) -> np.ndarray:
    """Welch rfft frequency grid for (fs, nperseg), clipped to [fmin, fmax]."""
    f = np.fft.rfftfreq(int(nperseg), d=1.0 / fs)
    m = (f >= fmin) & (f <= fmax)
    return f[m]


def welch_psd(lfp: np.ndarray, fs: float, nperseg: int, noverlap: int,
              detrend: str = "constant") -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per channel; nperseg/noverlap clamped to length (never raises)."""
    lfp = np.atleast_2d(np.asarray(lfp, dtype=np.float64))
    n_time = lfp.shape[1]
    nseg = max(8, int(min(nperseg, n_time)))
    nov = int(min(noverlap, nseg - 1))
    f, Pxx = scipy.signal.welch(lfp, fs=fs, window="hann", nperseg=nseg,
                                noverlap=nov, detrend=detrend, axis=-1,
                                scaling="density")
    return f, np.atleast_2d(Pxx)


def site_spectra(session: dict, f_common: np.ndarray, nperseg: int, noverlap: int,
                 notch_f0: float = 60.0, notch_Q: float = 30.0,
                 log: bool = True, eps: float = 1e-20) -> np.ndarray:
    """One experiment -> (n_channels x n_freq) spectral matrix on the common grid.

    Each channel is a recording site. 60 Hz is notched from the time series first,
    then a Welch PSD is interpolated onto f_common and (optionally) converted to dB.
    """
    lfp = notch_filter(session["lfp"], session["fs"], notch_f0, notch_Q)
    f, Pxx = welch_psd(lfp, session["fs"], nperseg, noverlap)
    P = np.empty((Pxx.shape[0], f_common.size), dtype=np.float64)
    for c in range(Pxx.shape[0]):
        P[c] = np.interp(f_common, f, Pxx[c])
    if log:
        P = 10.0 * np.log10(P + eps)
    return P


def build_master_matrix(sessions: list[dict], f_common: np.ndarray,
                        nperseg: int, noverlap: int,
                        notch_f0: float = 60.0, notch_Q: float = 30.0,
                        log: bool = True) -> dict:
    """Pool every site from every experiment into one master matrix.

    Returns a dict:
        M          (n_sites x n_freq)  dB power per site (rows in experiment order)
        experiment (n_sites,)          date string of each site's experiment
        exp_index  (n_sites,)          integer experiment id (0..n_exp-1)
        ycoord     (n_sites,)          site depth in microns
        reldepth   (n_sites,)          relative depth 0=tip..1=surface within its probe
        channel_row(n_sites,)          AP row index of the site
        dates      (n_exp,)            unique experiment dates, in order
        n_freq     int
    Depth is retained per row (repeated depths across experiments are expected);
    it is used only as a label, never to reorder the matrix.
    """
    rows, exp, exp_idx, yc, rel, chrow = [], [], [], [], [], []
    dates = []
    for si, s in enumerate(sessions):
        dates.append(s["date"])
        P = site_spectra(s, f_common, nperseg, noverlap, notch_f0, notch_Q, log=log)
        n_ch = P.shape[0]
        denom = (n_ch - 1) if n_ch > 1 else 1
        for c in range(n_ch):
            rows.append(P[c])
            exp.append(s["date"]); exp_idx.append(si)
            yc.append(float(s["ycoord"][c]) if c < len(s["ycoord"]) else np.nan)
            rel.append(c / denom)          # 0 = tip (deep) .. 1 = surface
            chrow.append(int(s["channel_row"][c]) if c < len(s["channel_row"]) else c)
    return dict(
        M=np.vstack(rows),
        experiment=np.asarray(exp),
        exp_index=np.asarray(exp_idx, dtype=int),
        ycoord=np.asarray(yc, dtype=float),
        reldepth=np.asarray(rel, dtype=float),
        channel_row=np.asarray(chrow, dtype=int),
        dates=np.asarray(dates),
        n_freq=int(f_common.size),
    )


# ---------------------------------------------------------------------------
# 3. Feature normalisation, clustering, PCA
# ---------------------------------------------------------------------------
def row_normalize(M: np.ndarray) -> np.ndarray:
    """Subtract each site's own mean (over frequency) -> spectral SHAPE per site."""
    M = np.asarray(M, dtype=np.float64)
    return M - M.mean(axis=1, keepdims=True)


def hierarchical_linkage(features: np.ndarray, method: str = "ward",
                         metric: str = "euclidean"):
    """SciPy linkage matrix for clustering the SITES (rows of `features`).

    Ward requires the Euclidean metric; it is forced if method == 'ward'.
    """
    if method == "ward":
        metric = "euclidean"
    return linkage(np.asarray(features, dtype=np.float64), method=method, metric=metric)


def pca_sites(features: np.ndarray) -> dict:
    """PCA with each SITE (row) as one observation.

    Columns (frequencies) are mean-centred across sites. Returns:
        scores (n_sites x k), loadings (k x n_freq), explained (k,),
        cum_explained (k,), singular_values (k,), mean (1 x n_freq).
    """
    X = np.asarray(features, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S
    var = S ** 2
    explained = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
    return dict(scores=scores, loadings=Vt, singular_values=S,
                explained=explained, cum_explained=np.cumsum(explained), mean=mu)

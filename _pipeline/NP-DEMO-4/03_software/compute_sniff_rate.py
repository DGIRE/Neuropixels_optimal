"""compute_sniff_rate.py — NP-DEMO-4 new kernel function.

Compute instantaneous sniff rate at each LabView sample from the SNF trace.

Algorithm (contract_v001.yaml preprocessing, "Compute instantaneous sniff rate"):
  1. Detect sniff onsets from D['SNF'] using the SAME validated recipe as
     compute_sniff_phase.py: low-pass FIR (firwin order-20 hamming, 40 Hz) +
     filtfilt (padtype='odd', padlen=60) + z-score (ddof=1) + downward
     threshold crossing at -0.5 std + MIN_ISI rejection.
  2. Compute inter-onset intervals (ISI) in seconds.
  3. Interpolate 1/ISI to each LV sample using nearest-neighbor/zero-order-hold:
     each sample inherits the rate of the most recently started onset interval.
     Samples before the first onset or after the last onset use flat extrapolation
     (rate of the first/last interval respectively). If fewer than 2 onsets exist,
     return array of NaN.

PROH-001: reads D['SNF'] ONLY. Never touches D['LFP'].
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.signal


def _matlab_round(x: float) -> int:
    """Half-away-from-zero rounding (mirrors Optimized Python/analyses/_common.py)."""
    x = np.asarray(x, dtype=np.float64)
    return int(np.sign(x) * np.floor(np.abs(x) + 0.5))


def compute_sniff_rate(D: dict, threshold_std: float = -0.5) -> dict:
    """Compute instantaneous sniff rate at each LV sample from D['SNF'].

    Parameters
    ----------
    D : dict
        Experiment dict containing 'SNF' (n_LV_samples,) float64 and 'LV_Fs'.
    threshold_std : float
        Pinned threshold for downward z-score crossing. Default -0.5 (contract-pinned).

    Returns
    -------
    dict
        Copy of D with added keys:
          'sniff_rate'  — (n_LV_samples,) float64, instantaneous sniff rate in Hz.
                          NaN everywhere if fewer than 2 onsets detected.
          'sniff_onsets_sr' — (n_onsets,) int, onset sample indices (0-based).
    """
    LV_Fs = float(D["LV_Fs"])
    SNF = np.asarray(D["SNF"], dtype=np.float64).ravel()
    n = SNF.size

    # --- Step 1: detect sniff onsets (same recipe as compute_sniff_phase.py) ---
    fco = min(40.0, LV_Fs / 2.0 * 0.9)
    taps = scipy.signal.firwin(21, fco, fs=LV_Fs, window="hamming",
                               pass_zero="lowpass", scale=True)
    padlen = 3 * (taps.size - 1)  # == 60 for ntaps=21
    SNF_filt = scipy.signal.filtfilt(taps, [1.0], SNF, padtype="odd", padlen=padlen)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std_ = SNF_filt.std(ddof=1)
        if std_ == 0.0:
            # Constant signal: no variance, no onsets
            out = dict(D)
            out["sniff_rate"] = np.full(n, np.nan, dtype=np.float64)
            out["sniff_onsets_sr"] = np.array([], dtype=np.int64)
            return out
        SNF_z = (SNF_filt - SNF_filt.mean()) / std_

    # Downward threshold crossings: diff([0; below]) == 1
    below = (SNF_z < threshold_std).astype(np.int64)
    dbelow = np.diff(np.concatenate(([0], below)))
    onsets = np.flatnonzero(dbelow == 1)

    # MIN_ISI rejection
    MIN_ISI = _matlab_round(0.05 * LV_Fs)
    if onsets.size > 1:
        keep = np.concatenate(([True], np.diff(onsets) >= MIN_ISI))
        onsets = onsets[keep]

    # --- Step 2: If fewer than 2 onsets, return NaN ---
    if onsets.size < 2:
        out = dict(D)
        out["sniff_rate"] = np.full(n, np.nan, dtype=np.float64)
        out["sniff_onsets_sr"] = onsets.astype(np.int64)
        return out

    # --- Step 3: Compute ISI and interpolate to LV samples ---
    # ISI in seconds between consecutive onsets
    isi_s = np.diff(onsets.astype(np.float64)) / LV_Fs   # (n_onsets - 1,)
    rate_hz = 1.0 / isi_s                                  # (n_onsets - 1,)

    # Zero-order-hold / nearest-neighbor: sample at index i gets the rate of
    # the most recently started ISI. The k-th ISI starts at onsets[k] and spans
    # up to (but not including) onsets[k+1]. We use np.searchsorted to find
    # which interval bracket each sample falls in.
    #
    # For sample t:
    #   bin_idx = searchsorted(onsets[1:], t, side='right')
    #             -> 0 if t < onsets[1], i.e. before second onset (extrapolate flat to ISI-0)
    #             -> k if onsets[k] <= t < onsets[k+1]
    #             -> n_onsets-1 if t >= onsets[-1] (extrapolate flat to last ISI)
    idx = np.arange(n, dtype=np.int64)
    bin_idx = np.searchsorted(onsets[1:], idx, side="right")
    bin_idx = np.clip(bin_idx, 0, rate_hz.size - 1)
    sniff_rate = rate_hz[bin_idx]

    out = dict(D)
    out["sniff_rate"] = sniff_rate.astype(np.float64)
    out["sniff_onsets_sr"] = onsets.astype(np.int64)
    return out

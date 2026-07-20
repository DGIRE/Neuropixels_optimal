"""
analyses/compute_sniff_phase.py  —  port of MATLAB new analyses/compute_sniff_phase.m
Fixture stage 08.  Reads D['SNF'], D['LV_Fs']; adds D['SNF_filt'], D['SNF_z'],
D['SNF_PH'], D['sniff_onsets'], D['sniff_onsets_s'], D['sniff_dur_s'], D['sniff_thr'].

Method: low-pass FIR (order-20 windowed, 40 Hz) filtfilt of SNF, z-score,
downward threshold crossing (default -0.5 std), MIN_ISI rejection, then a
per-onset phase ramp 0..1 (capped at 0.5 s), -1 outside every cycle.

Conventions honored (extension guide §8):
  * designfilt('lowpassfir','FilterOrder',20,...) == scipy firwin(21, ...,
    window='hamming', scale=True): windowed-FIR, unit gain at DC.
  * filtfilt padlen = 3*(ntaps-1) = 60 (MATLAB), padtype='odd'  (NOT scipy default).
  * z-score uses population std with N-1 (ddof=1).
  * The interactive threshold picker is a GUI; the pinned value (-0.5) is passed
    as an argument — never reintroduce a dialog.
"""
from __future__ import annotations

import numpy as np
import scipy.signal

from ._common import matlab_round


def compute_sniff_phase(D: dict, threshold_std: float = -0.5) -> dict:
    """Port of compute_sniff_phase.m.  ``threshold_std`` is the pinned GUI value."""
    LV_Fs = float(D["LV_Fs"])
    SNF = np.asarray(D["SNF"], dtype=np.float64).ravel()
    nSNF = SNF.size

    # --- low-pass FIR + zero-phase filtfilt ------------------------------
    fco = min(40.0, LV_Fs / 2.0 * 0.9)
    taps = scipy.signal.firwin(21, fco, fs=LV_Fs, window="hamming",
                               pass_zero="lowpass", scale=True)  # order 20 -> 21 taps
    padlen = 3 * (taps.size - 1)                                  # == MATLAB 3*(ntaps-1)=60
    SNF_filt = scipy.signal.filtfilt(taps, [1.0], SNF, padtype="odd", padlen=padlen)

    # --- z-score (ddof=1) ------------------------------------------------
    SNF_z = (SNF_filt - SNF_filt.mean()) / SNF_filt.std(ddof=1)

    # --- downward threshold crossings: diff([0;below]) == 1 --------------
    below = (SNF_z < threshold_std).astype(np.int64)
    dbelow = np.diff(np.concatenate(([0], below)))
    onsets = np.flatnonzero(dbelow == 1)          # 0-based sample indices

    MIN_ISI = int(matlab_round(0.05 * LV_Fs))     # MATLAB round(); half-away
    if onsets.size > 1:
        keep = np.concatenate(([True], np.diff(onsets) >= MIN_ISI))
        onsets = onsets[keep]

    # --- per-onset phase ramp -------------------------------------------
    MAX_DUR = int(matlab_round(0.5 * LV_Fs))       # round(62.5)=63 (NOT banker's 62)
    SNF_PH = -np.ones(nSNF, dtype=np.float64)
    dur_s = np.zeros(onsets.size, dtype=np.float64)
    for k in range(onsets.size):
        ok = int(onsets[k])
        cyc = MAX_DUR
        if k < onsets.size - 1:
            cyc = min(int(onsets[k + 1] - ok), MAX_DUR)
        dur_s[k] = cyc / LV_Fs
        end = min(ok + cyc, nSNF)                  # MATLAB ok:min(ok+cyc-1,nSNF), inclusive
        n = end - ok
        SNF_PH[ok:end] = np.arange(n, dtype=np.float64) / cyc

    out = dict(D)
    out["SNF_filt"] = SNF_filt
    out["SNF_z"] = SNF_z
    out["SNF_PH"] = SNF_PH
    out["sniff_onsets"] = (onsets + 1).astype(np.float64)     # 1-based, matches fixture
    out["sniff_onsets_s"] = onsets.astype(np.float64) / LV_Fs  # (onsets_1based-1)/LV_Fs
    out["sniff_dur_s"] = float(np.median(dur_s))
    out["sniff_thr"] = float(threshold_std)
    return out

"""resample_snf_to_lfp.py -- NP-PAPER-1 additive analysis primitive
(Blueprint v2 P3, D7).

New, ADDITIVE function per contract PROH-005/PROH-009/RQ-044: resample the
RAW sniff-sensor trace (SNF, never the detected phase SNF_PH -- PROH-009)
from the LabView rate (LV_Fs, ~125 Hz) up to the LFP rate (LFP_Fs, ~1000 Hz)
via polyphase/band-limited interpolation, on the shared recording clock, for
LFP-sniff coherence. Theta (2-12 Hz) is far below the ~62.5 Hz SNF Nyquist,
so this is a safe, non-aliasing upsampling.

METHOD: `scipy.signal.resample_poly` (a standard polyphase FIR interpolator)
with an up/down integer ratio derived as a rational approximation of
LFP_Fs/LV_Fs (exact when the two rates are already commensurate integers,
e.g. the synthetic 1000/125 = 8/1 case; a fine-denominator rational
approximation otherwise, since real session sample rates are not
necessarily exact integers). The output is then truncated or edge-padded to
exactly the caller-specified `n_lfp_samples` so it aligns sample-for-sample
with the LFP array it will be cohered against (the two are computed on
independent sample-count formulas upstream and are not guaranteed to agree
to the last sample).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import scipy.signal

_MAX_DENOMINATOR = 100_000  # rational-ratio precision cap for non-integer Fs


def _rational_up_down(lfp_fs: float, lv_fs: float) -> tuple[int, int]:
    """Best rational approximation (up/down, up/down <= _MAX_DENOMINATOR
    denominator) of the true resampling ratio LFP_Fs/LV_Fs. Exact when the
    rates are already commensurate integers (e.g. 1000/125 -> 8/1)."""
    ratio = Fraction(float(lfp_fs)) / Fraction(float(lv_fs))
    ratio = ratio.limit_denominator(_MAX_DENOMINATOR)
    return ratio.numerator, ratio.denominator


def resample_snf_to_lfp(SNF, LV_Fs: float, LFP_Fs: float, n_lfp_samples: int) -> np.ndarray:
    """Resample raw SNF from the LabView clock to the LFP clock.

    Parameters
    ----------
    SNF : array-like, shape (n_lv,)
        Raw sniff-sensor (cannula pressure) trace at LV_Fs.
    LV_Fs : float
        Native sample rate of SNF (Hz), ~125.
    LFP_Fs : float
        Target sample rate (Hz), ~1000.
    n_lfp_samples : int
        Exact output length required (the LFP array's sample count on the
        shared clock).

    Returns
    -------
    np.ndarray, shape (n_lfp_samples,), dtype float64
        Band-limited-interpolated SNF on the LFP clock.
    """
    x = np.asarray(SNF, dtype=np.float64).ravel()
    n_lfp_samples = int(n_lfp_samples)

    up, down = _rational_up_down(LFP_Fs, LV_Fs)
    y = scipy.signal.resample_poly(x, up, down)

    if y.size == n_lfp_samples:
        return y
    if y.size > n_lfp_samples:
        return y[:n_lfp_samples]
    # Rounding-edge case: pad the (small) shortfall by holding the last
    # sample, avoiding an artificial discontinuity at the tail.
    pad_value = y[-1] if y.size > 0 else 0.0
    pad = np.full(n_lfp_samples - y.size, pad_value, dtype=np.float64)
    return np.concatenate([y, pad])

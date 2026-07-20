"""
analyses/_common.py
-------------------
Shared primitives for the Neuropixels "new analyses" ports.  Every function
here mirrors a specific MATLAB convention so the analyses that import them stay
bit-faithful to the original code (see the extension guide, §8 Conventions).

Nothing in this module reads or writes the D dict; these are pure helpers.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# MATLAB round() — half away from zero (NOT numpy's banker's rounding).
# Needed wherever a continuous quantity is turned into a sample index, e.g.
#   lv_idx = round(spikeTimes * LV_Fs) + 1     (compute_spike_phase.m:8)
#   idx    = round(st_s * LV_Fs) + 1           (run_ICA.m / run_decomposition.m)
# A single index off-by-one shifts a 1e-12-tolerance fixture, so this must be
# exact (blueprint hazard: "round half-away sign*floor(|x|+.5)").
# --------------------------------------------------------------------------
def matlab_round(x: np.ndarray | float) -> np.ndarray:
    """Round half-away-from-zero, matching MATLAB ``round``."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# --------------------------------------------------------------------------
# Zero-lag normalized cross-correlation ('coeff' at lag 0), from
# run_decomposition.m:437-449 (local function zero_lag_xcorr).
#   r = sum(x.*y) / sqrt(sum(x.^2) * sum(y.^2)),   0 if either has no energy.
# MATLAB's guard is `denom < eps` where eps == eps(1) == 2.220446049250313e-16.
# --------------------------------------------------------------------------
_EPS = np.finfo(float).eps  # 2.220446049250313e-16, == MATLAB eps


def zero_lag_xcorr(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized zero-lag cross-correlation (xcorr 'coeff' at lag 0)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom < _EPS:
        return 0.0
    return float(np.sum(x * y) / denom)


# --------------------------------------------------------------------------
# Gaussian smoothing kernel used by run_decomposition.m / run_ICA.m.
#   sigma_samp = sigma_s * LV_Fs
#   hw         = ceil(3*sigma_samp)
#   kernel     = exp(-0.5*((-hw:hw)/sigma_samp)^2);  kernel /= sum(kernel)
# --------------------------------------------------------------------------
def gaussian_kernel(sigma_ms: float, LV_Fs: float):
    """Return (kernel, hw) for the MATLAB spike-rate smoothing convolution."""
    sigma_samp = (sigma_ms / 1000.0) * LV_Fs
    hw = int(np.ceil(3.0 * sigma_samp))
    t = np.arange(-hw, hw + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma_samp) ** 2)
    k = k / np.sum(k)
    return k, hw


def _next_pow2(n: int) -> int:
    """MATLAB nextpow2: smallest power of two >= n (as the length 2**p)."""
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


def _fft_conv_same_matlab(rate: np.ndarray, kernel: np.ndarray, hw: int, nSamp: int):
    """One channel: linear FFT convolution exactly as run_ICA.m does it.

        nfft    = 2^nextpow2(nSamp + len(kernel) - 1)
        rate_sm = real(ifft(fft(rate,nfft) .* fft(kernel,nfft)))
        return    rate_sm[hw : hw+nSamp]          (MATLAB hw+1:hw+nSamp, 1-based)
    """
    nfft = _next_pow2(nSamp + kernel.size - 1)
    sm = np.fft.irfft(np.fft.rfft(rate, nfft) * np.fft.rfft(kernel, nfft), nfft)
    return sm[hw:hw + nSamp]


def spike_rate_matrix(spikeTimes, clu, unitIDs, LV_Fs, nSamp, sigma_ms,
                      *, vectorized: bool = True):
    """Build the Gaussian-convolved spike-rate matrix X  [nUnits x nSamp].

    Faithful to run_decomposition.m:70-95 / run_ICA.m:44-74.  For each unit,
    spikes are binned onto the LabView grid (idx = round(t*LV_Fs)+1, 1-based;
    accumulate with += so coincident spikes stack), then convolved with the
    normalized Gaussian kernel via FFT and trimmed to [hw+1 : hw+nSamp].

    vectorized=False reproduces the per-unit MATLAB loop (the baseline).
    vectorized=True does the identical math but convolves all units in one
    batched real-FFT (optimized path).  The two agree to FFT round-off.
    """
    spikeTimes = np.asarray(spikeTimes, dtype=np.float64).ravel()
    clu = np.asarray(clu).ravel()
    unitIDs = np.asarray(unitIDs).ravel()
    nU = unitIDs.size
    kernel, hw = gaussian_kernel(sigma_ms, LV_Fs)

    # Per-spike index onto the LabView grid (0-based), and its unit row.
    idx = matlab_round(spikeTimes * LV_Fs).astype(np.int64)      # 0-based (round+1-1)
    in_bounds = (idx >= 0) & (idx < nSamp)
    uidx = np.searchsorted(unitIDs, clu)                          # every clu is in unitIDs

    if not vectorized:
        X = np.zeros((nU, nSamp), dtype=np.float64)
        for u in range(nU):
            sel = (clu == unitIDs[u])
            iu = idx[sel]
            iu = iu[(iu >= 0) & (iu < nSamp)]
            rate = np.zeros(nSamp, dtype=np.float64)
            np.add.at(rate, iu, 1.0)                             # += (stacks duplicates)
            X[u, :] = _fft_conv_same_matlab(rate, kernel, hw, nSamp)
        return X

    # Vectorized: build the whole [nU x nSamp] rate matrix with one add.at,
    # then convolve every row in a single batched FFT.
    rate = np.zeros((nU, nSamp), dtype=np.float64)
    np.add.at(rate, (uidx[in_bounds], idx[in_bounds]), 1.0)
    nfft = _next_pow2(nSamp + kernel.size - 1)
    sm = np.fft.irfft(np.fft.rfft(rate, nfft, axis=1)
                      * np.fft.rfft(kernel, nfft)[None, :], nfft, axis=1)
    return sm[:, hw:hw + nSamp]


def rank_by_abs_desc(values: np.ndarray) -> np.ndarray:
    """Rank 1..n by descending |value| (1 = strongest), matching the MATLAB
    ``[~,ord]=sort(abs(v),'descend'); rank(ord(r))=r`` idiom.

    MATLAB sort is stable; np.argsort(kind='stable') on the negated |values|
    reproduces the same tie order.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(-np.abs(values), kind="stable")
    ranks = np.empty(values.size, dtype=np.int64)
    ranks[order] = np.arange(1, values.size + 1)
    return ranks

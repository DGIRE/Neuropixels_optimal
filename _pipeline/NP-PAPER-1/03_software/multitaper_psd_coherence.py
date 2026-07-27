"""
multitaper_psd_coherence(lfp, snf, window_starts, fs=1000.0)

NP-PAPER-1 spectral engine, function 6/9 (contract_v001.yaml aggregation step;
RQ-040 / RQ-023 / RQ-039 -- pinned parameters, never tuned).

Independently authored implementation of the contract's pinned multitaper
LFP-sniff coherence recipe. This is NOT a copy of
`oracles/spectral_oracle_ref.py::reference_multitaper_coherence` -- it is
written from the contract text / RQ-040 / RQ-023 directly, so that the oracle
reference and this implementation are two independent expressions of the same
pinned math, per the acceptance test
`test_implementation_matches_reference_coherence_on_fixture`
(rtol 1e-6 / atol 1e-9).

Pinned recipe (verbatim, do not tune):
  - 4-s windows (nperseg = 4*fs samples), 50% overlap.
  - 7 Slepian (DPSS) tapers, time-bandwidth product NW=4 (K = 2*NW-1 = 7).
  - Per (window, taper): rfft eigen-coefficients of the MEAN-REMOVED segment.
  - POOL tapered auto/cross spectra over ALL (taper, window) looks:
        S1(f)  = sum_{taper,window} |X1_k(f)|^2
        S2(f)  = sum_{taper,window} |X2_k(f)|^2
        S12(f) = sum_{taper,window}  X1_k(f) * conj(X2_k(f))
    THEN form the MAGNITUDE coherence (RQ-023 -- NOT the squared form):
        C(f) = |S12(f)| / sqrt(S1(f) * S2(f))
  - Frequencies reported from 0.25 Hz upward (DC dropped) -- the 4-s Rayleigh
    resolution is 1/4 Hz, so the reported grid is rfftfreq(nperseg, 1/fs) with
    the 0 Hz bin removed.

`window_starts` is a 1-D array of integer window START sample indices on the
LFP clock; each window is nperseg = round(4*fs) samples long (matches
`spectral_oracle_ref.call_real_multitaper`'s calling convention exactly).
"""
from __future__ import annotations

import numpy as np
from scipy.signal.windows import dpss

# Pinned spectral parameters (RQ-040/RQ-039 -- never tuned).
WIN_S = 4.0
NW = 4.0
K = 7  # 2*NW - 1


def multitaper_psd_coherence(lfp, snf, window_starts, fs=1000.0):
    """Pooled multitaper LFP-SNF coherence + PSDs over the given windows.

    Parameters
    ----------
    lfp, snf : 1-D array_like, same length, on the shared LFP-rate clock.
    window_starts : 1-D int array_like of window START sample indices; each
        window spans `nperseg = round(4*fs)` samples.
    fs : sampling rate (Hz), default 1000.0.

    Returns
    -------
    dict with keys: freqs, coherence, lfp_psd, snf_psd, coh_per_window,
    n_windows.
    """
    lfp = np.asarray(lfp, dtype=float)
    snf = np.asarray(snf, dtype=float)
    starts = np.asarray(window_starts, dtype=int)

    nperseg = int(round(WIN_S * fs))
    tapers = dpss(nperseg, NW, K)  # (K, nperseg), unit-energy Slepian tapers

    full_freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    fmin = 1.0 / WIN_S  # 0.25 Hz Rayleigh lower bound; drop DC
    keep = full_freqs >= (fmin - 1e-9)
    nfreq_full = full_freqs.size

    S1 = np.zeros(nfreq_full, dtype=float)
    S2 = np.zeros(nfreq_full, dtype=float)
    S12 = np.zeros(nfreq_full, dtype=complex)
    per_window_coh = []

    for start in starts:
        seg_lfp = lfp[start:start + nperseg] - lfp[start:start + nperseg].mean()
        seg_snf = snf[start:start + nperseg] - snf[start:start + nperseg].mean()

        # (K, nfreq_full) complex eigen-coefficients per taper.
        X1 = np.fft.rfft(tapers * seg_lfp[None, :], axis=1)
        X2 = np.fft.rfft(tapers * seg_snf[None, :], axis=1)

        s1_win = np.sum(np.abs(X1) ** 2, axis=0)
        s2_win = np.sum(np.abs(X2) ** 2, axis=0)
        s12_win = np.sum(X1 * np.conj(X2), axis=0)

        S1 += s1_win
        S2 += s2_win
        S12 += s12_win

        with np.errstate(invalid="ignore", divide="ignore"):
            coh_win = np.abs(s12_win) / np.sqrt(s1_win * s2_win)
        per_window_coh.append(coh_win[keep])

    with np.errstate(invalid="ignore", divide="ignore"):
        coherence = np.abs(S12) / np.sqrt(S1 * S2)

    # One-sided multitaper PSD density: mean over (taper, window) looks, /fs,
    # doubled for one-sidedness except at DC and (if even nperseg) Nyquist.
    n_windows = max(len(starts), 1)
    scale = 1.0 / (fs * K * n_windows)
    one_sided_factor = np.full(nfreq_full, 2.0)
    one_sided_factor[0] = 1.0
    if nperseg % 2 == 0:
        one_sided_factor[-1] = 1.0

    lfp_psd = S1 * scale * one_sided_factor
    snf_psd = S2 * scale * one_sided_factor

    if per_window_coh:
        coh_per_window = np.vstack(per_window_coh)
    else:
        coh_per_window = np.empty((0, int(keep.sum())))

    return {
        "freqs": full_freqs[keep],
        "coherence": coherence[keep],
        "lfp_psd": lfp_psd[keep],
        "snf_psd": snf_psd[keep],
        "coh_per_window": coh_per_window,
        "n_windows": len(starts),
    }

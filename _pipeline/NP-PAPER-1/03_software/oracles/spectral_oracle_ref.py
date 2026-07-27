# ORACLE
"""
INDEPENDENT reference (ground-truth) computations for the NP-PAPER-1
SPECTRAL-ENGINE oracle suite (Blueprint v2 P3 -- oracle design phase, RED tier).

WHY A SEPARATE MODULE (not conftest.py): the shared oracles/conftest.py in this
directory is owned by the *plumbing* oracle-designer agent and is explicitly
scoped to the five deterministic data-plumbing functions. pytest permits only
one conftest.py per directory, so the spectral-engine references live here and
are imported directly by the two spectral test files (the same pattern
NP-DEMO-3 used: `from conftest import compute_mrl`). No pytest fixtures are
defined here -- everything is a plain function callable from the test bodies.

SCOPE (this agent): the two hardest, most scientifically load-bearing new
functions --
    * multitaper_psd_coherence(lfp_ch, snf_lfp_rate, control_valid_windows, fs)
    * circular_shift_null(lfp_ch, snf_lfp_rate, control_valid_windows,
                          n_shifts, seed)

PROVENANCE / INTEGRITY (mandatory for this phase):
  NOTHING here was produced by running the not-yet-written implementations
  (03_software/multitaper_psd_coherence.py, 03_software/circular_shift_null.py).
  Every expected value asserted in the test files is anchored to ONE of:
    (a) a closed-form analytical result -- identical-signal coherence == 1;
        white-common-source coherence == SNR/(1+SNR); the multitaper
        null-coherence bias floor from the Goodman/Carter Beta(1,K-1) law;
    (b) a Monte-Carlo-verified statistical calibration with a stated tolerance
        band (each band checked at design time against an independent
        scipy-only path -- MC numbers quoted inline in the test files);
    (c) the independent reference implementations in THIS module, built from
        scipy.signal.windows.dpss + numpy.fft ONLY (a validated third-party
        spectral primitive -- consulting it is equivalent to consulting a
        MATLAB/golden reference, NOT "running the new code").

  `reference_multitaper_coherence` transcribes the contract's pinned recipe
  verbatim (contract_v001.yaml aggregation step / RQ-040 / RQ-023):
      - 4-s windows (nperseg = 4*fs = 4000 @ 1 kHz), 50% overlap,
      - 7 Slepian (DPSS) tapers, time-bandwidth NW=4 (=> K=2*NW-1=7,
        half-bandwidth W = NW/T = 4/4 = 1 Hz),
      - frequencies from the 4-s Rayleigh resolution 0.25 Hz upward (DC dropped),
      - per (window, taper): eigen-coefficients of the MEAN-REMOVED segment,
      - POOL the tapered auto/cross spectra over all (taper, window) looks, THEN
        form coherence:
            S1(f)=sum|X1|^2 ; S2(f)=sum|X2|^2 ; S12(f)=sum X1 conj(X2)
            C(f) = |S12(f)| / sqrt(S1(f) S2(f))          <-- MAGNITUDE form
      RQ-023 pins the MAGNITUDE coherence |S12|/sqrt(S1 S2); NOT the also-common
      magnitude-SQUARED form |S12|^2/(S1 S2). The discriminator test rejects an
      implementation that silently returns the squared form.

  ------------------------------------------------------------------------
  DERIVATIONS (verified by scratchpad verify_* scripts at design time):

  [D1] Multitaper null-coherence bias floor.
    Under H0 (true coherence 0), the multitaper magnitude-SQUARED coherence
    from K orthonormal tapers is
        MSC_hat ~ Beta(1, K-1),   pdf (K-1)(1-x)^(K-2), 0<=x<=1
    (Goodman 1963; Carter 1987; multitaper form Thomson & Chave 1991; Percival
    & Walden 1993 "Spectral Analysis for Physical Applications"). Hence
        E[MSC_hat] = 1/K,
        upper-p percentile of MSC_hat = 1 - (1-p)**(1/(K-1)),
    and for the MAGNITUDE coherence C_hat = sqrt(MSC_hat),
        E[C_hat] = E[sqrt(X)], X~Beta(1,K-1)
                 = (sqrt(pi)/2) * Gamma(K) / Gamma(K+1/2).
    K=7 (single 4-s window):  E[MSC]=1/7=0.142857 ;
        E[C]=(sqrt(pi)/2)*Gamma(7)/Gamma(7.5)=0.34099 ;
        C 95th pct = sqrt(1-0.05**(1/6)) = 0.62693.
    (MC over 4000 white pairs: 0.3411 / 0.1429 / 0.6269 -- exact agreement.)
    Pooling W windows raises the look count to ~K*W and the floor falls as
    ~0.886/sqrt(K*W); with 50% overlap the effective look count is between
    K*W/2 and K*W, so the AGGREGATED floor is BOUNDED, not pinned -- the
    aggregated tests treat it directionally.

  [D2] Common-source SNR -> coherence closed form.
    y_i = a*c + n_i, with c,n1,n2 mutually independent zero-mean. Per freq:
        S12 = a^2 Scc ; S1 = a^2 Scc + Sn1 ; S2 = a^2 Scc + Sn2 ;
        SNR_i = a^2 Scc / Sn_i ;
        C = |S12|/sqrt(S1 S2) = sqrt((SNR1/(1+SNR1))(SNR2/(1+SNR2))).
    Symmetric: C = SNR/(1+SNR). With c,n1,n2 all WHITE unit-variance and gain a,
    SNR=a^2 is FLAT across freq so C is constant = a^2/(a^2+1):
        a^2=1/3 -> 0.25 ;  a^2=1 -> 0.5 ;  a^2=4 -> 0.8.
    (MC band-averaged: 0.2523 / 0.5006 / 0.8000.) The magnitude-SQUARED
    coherence would read (SNR/(1+SNR))^2 (0.0625 / 0.25 / 0.64) -- the
    discriminator.

  [D3] Circular-shift null needs a STOCHASTIC shared component.
    Circular shift is phase-invariant for a pure deterministic sinusoid
    (roll(sin) is still a sine at f0; magnitude coherence is phase-blind), so a
    sinusoid-only pair is NOT decorrelated by shifting (verified obs=thr=0.985 at
    f0 -> NOT flagged). A genuine coupling test uses a NARROWBAND STOCHASTIC
    common process (band-limited noise); shifting by >> its coherence time
    collapses the null to the floor (verified: theta obs mean 0.99 vs null
    threshold 0.137, 100% of theta bins exceed; off-band FP fraction 0.039).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal.windows import dpss
from scipy.special import gammaln

_SOFTWARE_DIR = Path(__file__).resolve().parents[1]   # .../NP-PAPER-1/03_software

# ---- Pinned spectral parameters (RQ-040 / RQ-039; PROH: never tuned) --------
FS = 1000.0
WIN_S = 4.0
NPERSEG = int(round(WIN_S * FS))            # 4000 samples
NW = 4.0                                     # time-bandwidth product
K = 7                                        # 2*NW-1 Slepian tapers
OVERLAP = 0.5                                # 50%
STEP = int(round(NPERSEG * (1 - OVERLAP)))  # 2000
FMIN = 1.0 / WIN_S                           # 0.25 Hz Rayleigh lower bound
THETA_LO, THETA_HI = 2.0, 12.0              # theta band (RQ)


# ===========================================================================
# Analytic anchors ([D1], [D2]) -- closed form
# ===========================================================================
def null_msc_mean(k: int = K) -> float:
    """E[MSC_hat] under H0 with k tapers = 1/k  (mean of Beta(1,k-1))."""
    return 1.0 / k


def null_msc_pct(pct: float = 95.0, k: int = K) -> float:
    """Upper-`pct` percentile of Beta(1,k-1): 1-(1-pct/100)**(1/(k-1))."""
    return 1.0 - (1.0 - pct / 100.0) ** (1.0 / (k - 1))


def null_coh_mean(k: int = K) -> float:
    """E[C_hat] under H0 with k tapers = (sqrt(pi)/2)*Gamma(k)/Gamma(k+1/2)."""
    return float(np.sqrt(np.pi) / 2.0 * np.exp(gammaln(k) - gammaln(k + 0.5)))


def null_coh_pct(pct: float = 95.0, k: int = K) -> float:
    """Upper-`pct` percentile of C_hat = sqrt of the MSC percentile."""
    return float(np.sqrt(null_msc_pct(pct, k)))


def snr_to_coherence(snr: float) -> float:
    """[D2] symmetric common-source MAGNITUDE coherence C = SNR/(1+SNR)."""
    return snr / (1.0 + snr)


# ===========================================================================
# Independent multitaper reference (scipy dpss + numpy.fft ONLY)
# ===========================================================================
def _tapers(nperseg: int = NPERSEG, nw: float = NW, k: int = K) -> np.ndarray:
    # Unit-energy DPSS tapers; taper norm cancels in the coherence ratio.
    return dpss(nperseg, nw, k)


def _eigencoeffs(seg: np.ndarray, tapers: np.ndarray) -> np.ndarray:
    """K complex rfft eigen-coefficients of the MEAN-REMOVED segment."""
    seg = seg - seg.mean()
    return np.fft.rfft(tapers * seg[None, :], axis=1)     # (K, nfreq)


def default_window_starts(n_samples: int) -> np.ndarray:
    """All 4-s / 50%-overlap window START indices that fit in n_samples.
    This is what the analytic/statistical tests pass as `control_valid_windows`
    when every candidate window is valid."""
    return np.arange(0, n_samples - NPERSEG + 1, STEP, dtype=int)


def reference_multitaper_coherence(lfp, snf, window_starts, fs=FS,
                                   nperseg=NPERSEG, nw=NW, k=K, fmin=FMIN):
    """Independent ground-truth multitaper PSD/coherence (see module docstring).

    ASSUMED INTERFACE for `window_starts` (== the contract's
    `control_valid_windows`): a 1-D array of integer window START sample indices
    on the LFP clock, each window `nperseg` samples long. If the shipped
    implementation adopts a different window-spec convention (boolean mask over
    candidate windows, (start,end) pairs, ...), the ONLY place to adapt is
    `call_real_multitaper` below -- the math here is unaffected.

    Returns dict: freqs, coherence, lfp_psd, snf_psd, coh_per_window, n_windows.
    """
    lfp = np.asarray(lfp, float)
    snf = np.asarray(snf, float)
    tapers = _tapers(nperseg, nw, k)
    starts = np.asarray(window_starts, int)

    freqs_full = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    keep = freqs_full >= (fmin - 1e-9)            # drop DC; keep >= 0.25 Hz
    S1 = np.zeros(freqs_full.size)
    S2 = np.zeros(freqs_full.size)
    S12 = np.zeros(freqs_full.size, dtype=complex)
    per_win = []
    for s in starts:
        xk = _eigencoeffs(lfp[s:s + nperseg], tapers)
        yk = _eigencoeffs(snf[s:s + nperseg], tapers)
        s1w = np.sum(np.abs(xk) ** 2, axis=0)
        s2w = np.sum(np.abs(yk) ** 2, axis=0)
        s12w = np.sum(xk * np.conj(yk), axis=0)
        S1 += s1w
        S2 += s2w
        S12 += s12w
        with np.errstate(invalid="ignore", divide="ignore"):
            per_win.append((np.abs(s12w) / np.sqrt(s1w * s2w))[keep])
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(S12) / np.sqrt(S1 * S2)

    # one-sided multitaper PSD density (documented convention; see PSD tests):
    nwin = max(len(starts), 1)
    scale = 1.0 / (fs * k * nwin)
    onesided = np.full(freqs_full.size, 2.0)
    onesided[0] = 1.0
    if nperseg % 2 == 0:
        onesided[-1] = 1.0
    lfp_psd = S1 * scale * onesided
    snf_psd = S2 * scale * onesided

    return {
        "freqs": freqs_full[keep],
        "coherence": coh[keep],
        "lfp_psd": lfp_psd[keep],
        "snf_psd": snf_psd[keep],
        "coh_per_window": (np.vstack(per_win) if per_win
                           else np.empty((0, int(keep.sum())))),
        "n_windows": len(starts),
    }


def reference_circular_shift_null(lfp, snf, window_starts, n_shifts=1000,
                                  seed=5489, fs=FS):
    """Independent ground-truth circular-shift null.

    Draws `n_shifts` circular shifts of SNF relative to LFP from a DEDICATED
    numpy Generator seeded with `seed` (never the global RNG -- H8), recomputes
    the aggregated windowed coherence each time, returns the per-frequency 95th
    percentile threshold + the observed (unshifted) coherence. Bit-for-bit
    reproducible for fixed seed by construction (private Generator only).
    """
    lfp = np.asarray(lfp, float)
    snf = np.asarray(snf, float)
    rng = np.random.default_rng(seed)
    obs = reference_multitaper_coherence(lfp, snf, window_starts, fs=fs)
    nf = obs["coherence"].size
    n = snf.size
    null = np.empty((n_shifts, nf))
    for i in range(n_shifts):
        shift = int(rng.integers(1, n))              # nonzero circular shift
        snf_shift = np.roll(snf, shift)
        null[i] = reference_multitaper_coherence(
            lfp, snf_shift, window_starts, fs=fs)["coherence"]
    thr = np.percentile(null, 95.0, axis=0)
    return {
        "freqs": obs["freqs"],
        "observed": obs["coherence"],
        "null_threshold": thr,
        "null_distribution": null,
        "n_shifts": n_shifts,
        "seed": seed,
    }


# ===========================================================================
# Real-implementation loaders + adapters (None until the modules are written)
# ===========================================================================
def _ensure_path():
    p = str(_SOFTWARE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def real_multitaper_available() -> bool:
    _ensure_path()
    try:
        import multitaper_psd_coherence  # type: ignore # noqa: F401
        return hasattr(__import__("multitaper_psd_coherence"),
                       "multitaper_psd_coherence")
    except Exception:
        return False


def real_null_available() -> bool:
    _ensure_path()
    try:
        import circular_shift_null  # type: ignore # noqa: F401
        return hasattr(__import__("circular_shift_null"), "circular_shift_null")
    except Exception:
        return False


def call_real_multitaper(lfp, snf, window_starts, fs=FS):
    """Invoke the real multitaper_psd_coherence, normalize its output keys.
    Raises AssertionError if present but non-conforming. If the impl's window-spec
    convention differs from 'array of int start indices', adapt HERE only."""
    _ensure_path()
    from multitaper_psd_coherence import multitaper_psd_coherence  # type: ignore
    out = multitaper_psd_coherence(np.asarray(lfp, float),
                                   np.asarray(snf, float),
                                   np.asarray(window_starts, int), fs=fs)
    if not isinstance(out, dict):
        raise AssertionError("multitaper_psd_coherence must return a dict-like "
                             "with at least 'freqs' and 'coherence'.")

    def pick(*names, required=True):
        for nm in names:
            if nm in out:
                return np.asarray(out[nm])
        if required:
            raise AssertionError(f"missing any of {names} in impl output")
        return None
    return {
        "freqs": pick("freqs", "f", "frequencies"),
        "coherence": pick("coherence", "coh", "C"),
        "lfp_psd": pick("lfp_psd", "S1", "psd_lfp", "lfp_power", required=False),
        "snf_psd": pick("snf_psd", "S2", "psd_snf", "snf_power", required=False),
    }


def call_real_null(lfp, snf, window_starts, n_shifts=1000, seed=5489, fs=FS):
    _ensure_path()
    from circular_shift_null import circular_shift_null  # type: ignore
    out = circular_shift_null(np.asarray(lfp, float), np.asarray(snf, float),
                              np.asarray(window_starts, int),
                              n_shifts=n_shifts, seed=seed)
    if not isinstance(out, dict):
        raise AssertionError("circular_shift_null must return a dict-like.")

    def pick(*names):
        for nm in names:
            if nm in out:
                return np.asarray(out[nm])
        raise AssertionError(f"missing any of {names} in null output")
    return {
        "freqs": pick("freqs", "f"),
        "observed": pick("observed", "obs", "coherence"),
        "null_threshold": pick("null_threshold", "threshold", "thr", "pct95"),
    }


def oracle_multitaper(lfp, snf, window_starts, fs=FS):
    """Use the REAL multitaper_psd_coherence if it exists, else the reference.
    The SAME analytic/statistical test body thus validates the reference now and
    becomes the acceptance gate on the implementation later."""
    if real_multitaper_available():
        return call_real_multitaper(lfp, snf, window_starts, fs=fs)
    return reference_multitaper_coherence(lfp, snf, window_starts, fs=fs)


def oracle_null(lfp, snf, window_starts, n_shifts=1000, seed=5489, fs=FS):
    if real_null_available():
        return call_real_null(lfp, snf, window_starts, n_shifts=n_shifts,
                              seed=seed, fs=fs)
    return reference_circular_shift_null(lfp, snf, window_starts,
                                         n_shifts=n_shifts, seed=seed, fs=fs)


# ===========================================================================
# Synthetic signal factory (seeded -> deterministic ground truth)
# ===========================================================================
def make_signals(kind, T=200.0, fs=FS, amp=1.0, f0=6.0, band=(4.0, 8.0),
                 seed=5489):
    """Synthetic LFP/SNF pairs with KNOWN coherence structure.

    kinds:
      'identical'    y == x (x white)                -> C == 1 everywhere
      'white_common' y_i = a*c + n_i (all white)     -> C = a^2/(a^2+1) flat [D2]
      'independent'  x, y independent white           -> C = null floor [D1]
      'narrowband'   y_i = a*c + n_i, c band-limited  -> high C in band, floor
                     elsewhere; shift-decorrelable    -> circular-null TP [D3]
      'sinusoid'     y_i = sin(2 pi f0 t) + a*noise   -> C~1 at f0 (NOT shift-
                     decorrelable -- [D3] cautionary case)
    """
    n = int(round(T * fs))
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    if kind == "identical":
        x = rng.standard_normal(n)
        return x, x.copy()
    if kind == "white_common":
        c = rng.standard_normal(n)
        return amp * c + rng.standard_normal(n), amp * c + rng.standard_normal(n)
    if kind == "independent":
        return rng.standard_normal(n), rng.standard_normal(n)
    if kind == "narrowband":
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
        c = filtfilt(b, a, rng.standard_normal(n))
        c = c / c.std()
        return amp * c + rng.standard_normal(n), amp * c + rng.standard_normal(n)
    if kind == "sinusoid":
        s = np.sin(2 * np.pi * f0 * t)
        return s + amp * rng.standard_normal(n), s + amp * rng.standard_normal(n)
    raise ValueError(f"unknown kind {kind!r}")

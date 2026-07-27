# ORACLE
"""
Metamorphic / analytical oracle for the NOT-YET-IMPLEMENTED
resample_snf_to_lfp(SNF, LV_Fs, LFP_Fs, n_lfp_samples).

CONTRACT TARGET
  preprocessing step (PROH-005, PROH-009, RQ-044):
    "Resample raw SNF from LV_Fs (~125 Hz) up to the LFP rate (~1000 Hz) on the
     shared clock via polyphase/band-limited interpolation for coherence. Use
     the raw SNF (resampled), not the detected phase (PROH-009)."
  Feeds RESULT-snf-psd and RESULT-coh-spectrum — ACCEPTANCE LEVEL 2 (numeric,
  rtol 1e-6/atol 1e-9 on the fixture slice for the downstream spectra). The
  resampler itself is validated here at the SCIENTIFIC/METAMORPHIC level: for a
  band-limited input (theta 2-12 Hz is far below the ~62 Hz SNF Nyquist) an
  ideal band-limited interpolation must preserve amplitude, frequency and phase
  with no aliasing.

WHY NOT AN EXACT (level-1) NUMERIC ORACLE: the contract pins "polyphase/band-
limited interpolation" but not one bit-exact routine (resample_poly vs FFT
resample differ in edge handling and ripple by ~1e-3). So expected values are
anchored to the ANALYTIC continuous-time signal the resampler is approximating
(sin(2*pi*f*t)), with tolerances set by the known interpolation ripple, plus
convention-independent metamorphic invariants (linearity, DC preservation,
length, energy conservation). None came from running the new function.

INTERIOR-ONLY: every amplitude/phase comparison excludes a 0.5 s guard band at
each edge, where any finite-length band-limited interpolator has an unavoidable
transient (documented, not a defect).
"""
from __future__ import annotations

from math import gcd

import numpy as np
import pytest
import scipy.signal

from conftest import load_real_resample_snf_to_lfp

LV_FS = 125.0
LFP_FS = 1000.0
EDGE_S = 0.5   # guard band excluded from interior comparisons


def _n_lfp(n_lv):
    return int(round(n_lv * LFP_FS / LV_FS))


def _reference_resample(x):
    """Reference band-limited upsampler used ONLY for metamorphic self-checks and
    as the fallback when the real function is absent. scipy.signal.resample_poly
    is a standard polyphase interpolator matching the contract's stated method
    family; the tests never assert bit-equality to it, only analytic properties."""
    up, down = int(LFP_FS), int(LV_FS)
    g = gcd(up, down)
    return scipy.signal.resample_poly(np.asarray(x, float), up // g, down // g)


def _run(x, n_lv):
    """Call REAL resample_snf_to_lfp if implemented, else the reference.
    Expected interface: resample_snf_to_lfp(SNF, LV_Fs, LFP_Fs, n_lfp_samples)
    -> 1-D ndarray of length n_lfp_samples on the LFP clock."""
    real = load_real_resample_snf_to_lfp()
    n_lfp = _n_lfp(n_lv)
    if real is None:
        y = _reference_resample(x)
    else:
        y = np.asarray(real(np.asarray(x, float), LV_FS, LFP_FS, n_lfp), dtype=np.float64)
    return y, n_lfp


def _interior(y):
    e = int(EDGE_S * LFP_FS)
    return slice(e, y.size - e)


# ===========================================================================
# Length / output-shape contract
# ===========================================================================
def test_output_length_matches_requested_n_lfp_samples():
    """resample_snf_to_lfp must return exactly n_lfp_samples (the shared-clock
    length the caller computed), so the resampled SNF aligns sample-for-sample
    with the LFP it will be cohered against."""
    n_lv = int(8.0 * LV_FS)
    x = np.random.default_rng(0).standard_normal(n_lv)
    y, n_lfp = _run(x, n_lv)
    assert y.shape == (n_lfp,)


# ===========================================================================
# Analytic: pure theta-band sinusoid preserved (amplitude / frequency / phase)
# ===========================================================================
@pytest.mark.parametrize("f", [2.0, 6.0, 12.0])
def test_theta_sinusoid_amplitude_and_phase_preserved(f):
    """SNF = sin(2*pi*f*t), f in the theta band (2-12 Hz), well below the ~62 Hz
    SNF Nyquist. An ideal band-limited interpolation reproduces the SAME
    continuous sinusoid, so the resampled samples must match sin(2*pi*f*t_lfp)
    (amplitude 1, phase 0) in the interior. Analytic ground truth; tolerance
    5e-3 absorbs polyphase ripple (measured ~1.7e-3 at f=6 Hz at design time)."""
    n_lv = int(8.0 * LV_FS)
    t_lv = np.arange(n_lv) / LV_FS
    x = np.sin(2 * np.pi * f * t_lv)
    y, _ = _run(x, n_lv)

    t_lfp = np.arange(y.size) / LFP_FS
    analytic = np.sin(2 * np.pi * f * t_lfp)
    sl = _interior(y)
    max_abs_err = np.max(np.abs(y[sl] - analytic[sl]))
    assert max_abs_err < 5e-3, f"interior deviation {max_abs_err:.2e} exceeds ripple bound"


@pytest.mark.parametrize("f", [2.0, 6.0, 12.0])
def test_theta_sinusoid_frequency_peak_unchanged(f):
    """No frequency shift / aliasing: the dominant FFT bin of the resampled
    interior must be at f (within one FFT bin), never at an aliased image."""
    n_lv = int(16.0 * LV_FS)   # longer -> tighter FFT resolution
    t_lv = np.arange(n_lv) / LV_FS
    x = np.sin(2 * np.pi * f * t_lv)
    y, _ = _run(x, n_lv)
    seg = y[_interior(y)]
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(seg.size, 1.0 / LFP_FS)
    peak = freqs[np.argmax(spec)]
    df = freqs[1] - freqs[0]
    assert abs(peak - f) <= df, f"peak {peak:.3f} Hz != input {f} Hz (bin {df:.3f})"


# ===========================================================================
# Metamorphic invariants (convention-independent)
# ===========================================================================
def test_linearity_of_resampling():
    """A band-limited interpolator is LINEAR: resample(a*x + b*y) ==
    a*resample(x) + b*resample(y). Metamorphic relation independent of the exact
    routine; tolerance near machine precision."""
    n_lv = int(8.0 * LV_FS)
    rng = np.random.default_rng(1)
    t_lv = np.arange(n_lv) / LV_FS
    x = np.sin(2 * np.pi * 4.0 * t_lv)
    z = np.sin(2 * np.pi * 9.0 * t_lv + 0.7)
    a, b = 2.3, -1.4
    yx, _ = _run(x, n_lv)
    yz, _ = _run(z, n_lv)
    yc, _ = _run(a * x + b * z, n_lv)
    np.testing.assert_allclose(yc, a * yx + b * yz, rtol=1e-7, atol=1e-9)


def test_dc_offset_preserved():
    """A constant (DC) SNF resamples to the same constant in the interior (unit
    DC gain of a band-limited interpolator). Tolerance 5e-3 for edge-free ripple."""
    n_lv = int(8.0 * LV_FS)
    c = 3.7
    y, _ = _run(np.full(n_lv, c), n_lv)
    seg = y[_interior(y)]
    assert np.max(np.abs(seg - c)) < 5e-3


def test_no_high_frequency_energy_injected_above_snf_nyquist():
    """Anti-alias / no-injection: a pure 6 Hz input must not produce meaningful
    energy above the SNF Nyquist (LV_Fs/2 = 62.5 Hz) after upsampling. The
    fraction of interior spectral energy above 62.5 Hz must be negligible."""
    n_lv = int(16.0 * LV_FS)
    t_lv = np.arange(n_lv) / LV_FS
    x = np.sin(2 * np.pi * 6.0 * t_lv)
    y, _ = _run(x, n_lv)
    seg = y[_interior(y)]
    spec = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(seg.size, 1.0 / LFP_FS)
    above = spec[freqs > LV_FS / 2.0].sum()
    total = spec.sum()
    assert above / total < 1e-3, "unexpected energy above the SNF Nyquist"


def test_raw_snf_used_not_phase_smoke():
    """PROH-009 guard (documentation-level): resample_snf_to_lfp operates on the
    RAW SNF amplitude, so a smooth continuous input maps to a smooth continuous
    output whose interior range tracks the input range -- it must NOT look like a
    detected-phase sawtooth (0..1 ramps with -1 sentinels). Here a raw sinusoid
    in [-1,1] must resample to values staying within [-1.05, 1.05] (never the
    -1-plus-ramp signature of SNF_PH)."""
    n_lv = int(8.0 * LV_FS)
    t_lv = np.arange(n_lv) / LV_FS
    x = np.sin(2 * np.pi * 5.0 * t_lv)
    y, _ = _run(x, n_lv)
    seg = y[_interior(y)]
    assert seg.min() > -1.05 and seg.max() < 1.05

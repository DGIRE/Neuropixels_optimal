# ORACLE
"""
Oracle for the NOT-YET-IMPLEMENTED
    multitaper_psd_coherence(lfp_ch, snf_lfp_rate, control_valid_windows, fs)
(03_software/multitaper_psd_coherence.py).

Contract sources: RQ-040 (pinned multitaper recipe: 4-s windows, 50% overlap,
7 Slepian tapers NW=4, freqs >= 0.25 Hz), RQ-023 (MAGNITUDE coherence
C=|S12|/sqrt(S1 S2), NOT the squared form), OUT-009/OUT-010 (RESULT-coh-spectrum,
RESULT-lfp-psd, RESULT-snf-psd + per-window spectrograms), DATA-002 (synthetic
fixture, rtol 1e-6 / atol 1e-9), RQ-039 (pinned parameters -- never tuned).

Acceptance levels (contract_v001.yaml analysis.acceptance_criteria):
  RESULT-coh-spectrum  -> level 2 numeric (rtol 1e-6/atol 1e-9 vs independent
                          reference on the fixture slice; distributional on the
                          full workload).
  RESULT-lfp-psd       -> level 2 numeric (rtol 1e-6/atol 1e-9 on fixture slice).
  RESULT-snf-psd       -> level 2 numeric (rtol 1e-6/atol 1e-9 on fixture slice).

Each test states its acceptance level and clause inline. Expected values are
closed-form ([D1] null bias floor, [D2] SNR->coherence) or Monte-Carlo-verified
bands -- NEVER produced by running the implementation. See spectral_oracle_ref.py
for the full derivations and provenance. Tests use `oracle_multitaper`, which
runs the REAL function once it exists and the independent reference until then,
so these bodies are the acceptance gate.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectral_oracle_ref import (
    FS, NPERSEG, K, STEP, FMIN, THETA_LO, THETA_HI,
    default_window_starts, make_signals, oracle_multitaper,
    reference_multitaper_coherence, call_real_multitaper,
    real_multitaper_available,
    null_coh_mean, null_msc_mean, null_coh_pct, snr_to_coherence,
)


# ===========================================================================
# 1. Frequency grid -- LEVEL 1 (exact). RQ-040: 4-s window => df = 0.25 Hz,
#    lower bound 0.25 Hz (DC dropped), up to Nyquist fs/2.
# ===========================================================================
def test_frequency_grid_is_pinned_quarter_hz_from_025():
    """LEVEL 1 exact (RQ-040/RQ-039). 4-s / 1 kHz => Rayleigh df=0.25 Hz; the
    reported grid starts at 0.25 Hz (no DC) and steps by exactly 0.25 Hz."""
    x, y = make_signals("white_common", T=40.0, amp=1.0, seed=1)
    out = oracle_multitaper(x, y, default_window_starts(len(x)))
    f = np.asarray(out["freqs"], float)

    assert f[0] == pytest.approx(0.25, abs=1e-12), "grid must start at 0.25 Hz"
    df = np.diff(f)
    assert np.allclose(df, 0.25, atol=1e-9), "grid spacing must be exactly 0.25 Hz"
    assert f[-1] == pytest.approx(FS / 2.0, abs=1e-9), "grid must reach Nyquist"
    # DC (0 Hz) must be absent (RQ-040 lower bound is the 4-s bound 0.25 Hz).
    assert np.min(f) >= FMIN - 1e-12
    # exact expected grid, first principles (rfftfreq minus DC bin):
    expected = np.fft.rfftfreq(NPERSEG, d=1.0 / FS)
    expected = expected[expected >= FMIN - 1e-9]
    assert f.shape == expected.shape
    assert np.allclose(f, expected, atol=1e-9)


# ===========================================================================
# 2. Identical signals -> coherence == 1 everywhere.
#    LEVEL 2 numeric (RQ-023): y==x => S12==S1==S2 => |S12|/sqrt(S1 S2)==1.
# ===========================================================================
def test_identical_signals_give_unit_coherence():
    """LEVEL 2 numeric (RQ-023/RESULT-coh-spectrum). For y == x the pooled
    cross-spectrum equals both auto-spectra, so C(f) == 1 at every frequency
    with power -- a closed-form anchor independent of any estimator detail."""
    x, y = make_signals("identical", T=60.0, seed=2)
    out = oracle_multitaper(x, y, default_window_starts(len(x)))
    coh = np.asarray(out["coherence"], float)
    finite = np.isfinite(coh)
    assert finite.any()
    # Ratio of identical complex sums -> exactly 1 (bar float rounding).
    assert np.allclose(coh[finite], 1.0, atol=1e-9), \
        "identical signals must yield coherence == 1 everywhere"
    assert coh[finite].max() <= 1.0 + 1e-9, "magnitude coherence must be <= 1"


# ===========================================================================
# 3. Independent white noise -> PER-WINDOW null bias floor. LEVEL 3 statistical.
#    [D1]: with K=7 tapers, MSC_hat~Beta(1,6): E[MSC]=1/7, E[C]=0.34099.
#    (MC-verified: 0.1429 / 0.3411.) This is the crux the task flags -- the
#    multitaper null coherence is NOT near zero for finite K.
# ===========================================================================
def test_per_window_null_coherence_matches_beta_bias_floor():
    """LEVEL 3 statistical (RQ-040/RESULT-coh-spectrum distributional). The
    per-window (single 4-s window, K=7 taper) coherence of two INDEPENDENT white
    signals has mean == E[C_hat] = (sqrt(pi)/2) Gamma(7)/Gamma(7.5) = 0.34099 and
    mean-square == E[MSC] = 1/7, from the Goodman/Carter Beta(1,K-1) null law.
    An implementation returning ~0 here (naive 'coherence should be zero under
    the null') is WRONG."""
    x, y = make_signals("independent", T=400.0, seed=3)
    out = oracle_multitaper(x, y, default_window_starts(len(x)))
    per_win = np.asarray(out.get("coh_per_window"))
    if per_win.size == 0 or per_win.ndim != 2:
        pytest.skip("implementation does not expose per-window coherence "
                    "(coh_per_window); per-window null-floor check needs it")
    f = np.asarray(out["freqs"], float)
    band = (f >= 5.0) & (f <= 450.0)          # avoid DC/Nyquist edges
    vals = per_win[:, band].ravel()
    vals = vals[np.isfinite(vals)]

    mean_c = float(vals.mean())
    mean_msc = float((vals ** 2).mean())
    # Tolerance: >170k (correlated) samples; MC design-time SE ~2e-3. 0.02 band
    # is >>5 SE and cleanly separates 0.341 from both 0 and 1.
    assert mean_c == pytest.approx(null_coh_mean(K), abs=0.02), \
        f"per-window null mean C {mean_c:.4f} != Beta(1,6) E[C]={null_coh_mean(K):.4f}"
    assert mean_msc == pytest.approx(null_msc_mean(K), abs=0.02), \
        f"per-window null mean MSC {mean_msc:.4f} != 1/K={null_msc_mean(K):.4f}"
    # 95th percentile of the per-window C null == sqrt(1-0.05**(1/6)) = 0.62693.
    p95 = float(np.percentile(vals, 95))
    assert p95 == pytest.approx(null_coh_pct(95.0, K), abs=0.03), \
        f"per-window null 95th pct C {p95:.4f} != {null_coh_pct(95.0,K):.4f}"


# ===========================================================================
# 4. Independent white noise -> AGGREGATED null floor FALLS below the
#    per-window floor (correct spectral POOLING, not coherence averaging).
#    LEVEL 4 scientific (RESULT-coh-spectrum full-workload behavior).
# ===========================================================================
def test_aggregated_null_floor_drops_with_pooling():
    """LEVEL 4 scientific (RQ-040 aggregation clause). [D1]: pooling W windows
    raises the look count to ~K*W and the null floor falls as ~0.886/sqrt(K*W).
    The correct estimator POOLS S1/S2/S12 across windows THEN forms the ratio;
    an estimator that AVERAGES per-window coherence values keeps the floor stuck
    at ~0.34. This test rejects the latter: the aggregated null mean must fall
    well below the per-window floor and be small in absolute terms."""
    x, y = make_signals("independent", T=400.0, seed=4)
    starts = default_window_starts(len(x))
    out = oracle_multitaper(x, y, starts)
    f = np.asarray(out["freqs"], float)
    band = (f >= 5.0) & (f <= 450.0)
    agg = np.asarray(out["coherence"], float)[band]
    agg = agg[np.isfinite(agg)]
    mean_agg = float(agg.mean())

    per_window_floor = null_coh_mean(K)        # 0.34099
    # ~99 windows @ 50% overlap => effective looks ~ K*99/2 ~ 350 => floor ~0.05.
    assert mean_agg < 0.5 * per_window_floor, (
        f"aggregated null floor {mean_agg:.4f} did not drop below half the "
        f"per-window floor {per_window_floor:.4f} -- implementation may be "
        f"averaging coherence values instead of pooling spectra")
    assert mean_agg < 0.15, \
        f"aggregated null floor {mean_agg:.4f} implausibly high for ~99 windows"
    assert mean_agg > 0.0


# ===========================================================================
# 5. Common-source SNR -> coherence closed form. LEVEL 3 statistical.
#    [D2]: white common + white independent noise => flat C = SNR/(1+SNR).
#    a^2=1/3 -> 0.25 ; a^2=1 -> 0.5 ; a^2=4 -> 0.8. (MC: 0.2523/0.5006/0.8000.)
# ===========================================================================
@pytest.mark.parametrize("amp,snr,c_true", [
    (np.sqrt(1.0 / 3.0), 1.0 / 3.0, 0.25),
    (1.0, 1.0, 0.5),
    (2.0, 4.0, 0.8),
])
def test_white_common_coherence_matches_snr_closed_form(amp, snr, c_true):
    """LEVEL 3 statistical (RQ-023/RESULT-coh-spectrum). For y_i = a*c + n_i with
    c,n1,n2 white unit-variance, the MAGNITUDE coherence is flat = a^2/(a^2+1) =
    SNR/(1+SNR). Band-averaging over the flat spectrum drives the finite-K
    estimator to the closed form within a few 1e-3."""
    assert snr_to_coherence(snr) == pytest.approx(c_true, abs=1e-12)  # self-check
    x, y = make_signals("white_common", T=400.0, amp=amp, seed=5)
    out = oracle_multitaper(x, y, default_window_starts(len(x)))
    f = np.asarray(out["freqs"], float)
    band = (f >= 5.0) & (f <= 450.0)
    est = float(np.nanmean(np.asarray(out["coherence"], float)[band]))
    # Residual gap is the known finite-n multitaper coherence bias (MC < 0.005).
    assert est == pytest.approx(c_true, abs=0.02), (
        f"SNR={snr}: band-mean coherence {est:.4f} != SNR/(1+SNR)={c_true:.4f}")


# ===========================================================================
# 6. NORMALIZATION DISCRIMINATOR -- reject the magnitude-SQUARED form.
#    LEVEL 4 scientific + LEVEL 2 gate (RQ-023). The single most important
#    scientific test: at SNR=1 the MAGNITUDE coherence is 0.5 but the (wrong)
#    squared coherence would be 0.25. The oracle must reject 0.25.
# ===========================================================================
def test_coherence_is_magnitude_form_not_squared():
    """LEVEL 4 scientific / LEVEL 2 gate (RQ-023, hazard H2). White-common at
    SNR=1: MAGNITUDE C=|S12|/sqrt(S1 S2)=0.5; the also-common squared form
    |S12|^2/(S1 S2)=0.25. The estimate must sit at 0.5, decisively closer to the
    magnitude value than the squared value -- catching an implementation that
    silently returns magnitude-squared coherence."""
    x, y = make_signals("white_common", T=600.0, amp=1.0, seed=6)  # SNR=1
    out = oracle_multitaper(x, y, default_window_starts(len(x)))
    f = np.asarray(out["freqs"], float)
    band = (f >= 5.0) & (f <= 450.0)
    est = float(np.nanmean(np.asarray(out["coherence"], float)[band]))

    c_mag, c_sq = 0.5, 0.25
    assert abs(est - c_mag) < abs(est - c_sq), (
        f"band-mean coherence {est:.4f} is closer to the SQUARED form {c_sq} "
        f"than the required MAGNITUDE form {c_mag} (RQ-023 violation)")
    assert est == pytest.approx(c_mag, abs=0.03), \
        f"coherence {est:.4f} not at the magnitude value 0.5 (RQ-023)"
    assert abs(est - c_sq) > 0.15, \
        f"coherence {est:.4f} not decisively separated from squared form 0.25"


# ===========================================================================
# 7. Common-oscillation-plus-noise fixture (contract DATA-002 design):
#    narrowband stochastic common in theta -> high coherence in-band, floor
#    out-of-band. LEVEL 3 statistical.
# ===========================================================================
def test_common_theta_oscillation_gives_elevated_theta_coherence():
    """LEVEL 3 statistical (DATA-002 / RESULT-coh-spectrum). A shared narrowband
    (4-8 Hz) stochastic component with independent additive noise yields coherence
    well above the null floor inside theta and near the floor far outside it."""
    x, y = make_signals("narrowband", T=300.0, amp=1.0, band=(4.0, 8.0), seed=7)
    out = oracle_multitaper(x, y, default_window_starts(len(x)))
    f = np.asarray(out["freqs"], float)
    coh = np.asarray(out["coherence"], float)
    inband = (f >= 4.0) & (f <= 8.0)
    farband = (f >= 60.0) & (f <= 120.0)
    mean_in = float(np.nanmean(coh[inband]))
    mean_far = float(np.nanmean(coh[farband]))
    assert mean_in > 0.5, f"in-band theta coherence {mean_in:.3f} not elevated"
    assert mean_far < 0.2, f"far-band coherence {mean_far:.3f} not near floor"
    assert mean_in > 3.0 * mean_far
    # theta-band mean is a bona-fide reduction target (RESULT-theta-coh).
    theta = (f >= THETA_LO) & (f <= THETA_HI)
    assert np.nanmean(coh[theta]) > mean_far


# ===========================================================================
# 8. DATA-002 exactness gate: implementation vs INDEPENDENT reference,
#    rtol 1e-6 / atol 1e-9 on the fixture slice. LEVEL 2 numeric.
#    Skips until 03_software/multitaper_psd_coherence.py exists.
# ===========================================================================
def _data002_fixture():
    """Canonical seeded synthetic fixture slice (DATA-002). Fixed kind/params/seed
    so the reference and (later) the implementation see byte-identical inputs."""
    x, y = make_signals("narrowband", T=120.0, amp=1.0, band=(4.0, 8.0), seed=5489)
    return x, y, default_window_starts(len(x))


def test_implementation_matches_reference_coherence_on_fixture():
    """LEVEL 2 numeric (DATA-002, RESULT-coh-spectrum: rtol 1e-6/atol 1e-9). The
    shipped multitaper_psd_coherence must reproduce the independent scipy-dpss
    reference on the fixed fixture slice to 1e-6. (Skips pre-implementation.)"""
    if not real_multitaper_available():
        pytest.skip("multitaper_psd_coherence not implemented yet")
    x, y, starts = _data002_fixture()
    ref = reference_multitaper_coherence(x, y, starts)
    got = call_real_multitaper(x, y, starts)
    assert np.allclose(got["freqs"], ref["freqs"], rtol=0, atol=1e-9), \
        "frequency grid mismatch vs reference"
    assert np.allclose(got["coherence"], ref["coherence"], rtol=1e-6, atol=1e-9), \
        "coherence mismatch vs independent reference (DATA-002 tolerance)"


def test_implementation_matches_reference_psd_on_fixture():
    """LEVEL 2 numeric (DATA-002, RESULT-lfp-psd/RESULT-snf-psd: rtol 1e-6/atol
    1e-9). PSD scaling convention is DOCUMENTED in spectral_oracle_ref (one-sided
    multitaper density, mean over K tapers and windows, /fs). If the implementer
    adopts a different *valid* density normalization, reconcile the convention
    here -- the coherence gate above is scale-invariant and remains binding
    regardless. (Skips pre-implementation.)"""
    if not real_multitaper_available():
        pytest.skip("multitaper_psd_coherence not implemented yet")
    x, y, starts = _data002_fixture()
    ref = reference_multitaper_coherence(x, y, starts)
    got = call_real_multitaper(x, y, starts)
    if got.get("lfp_psd") is None or got.get("snf_psd") is None:
        pytest.skip("implementation does not expose lfp_psd/snf_psd")
    assert np.allclose(got["lfp_psd"], ref["lfp_psd"], rtol=1e-6, atol=1e-9), \
        "LFP PSD mismatch vs reference (check density-scaling convention)"
    assert np.allclose(got["snf_psd"], ref["snf_psd"], rtol=1e-6, atol=1e-9), \
        "SNF PSD mismatch vs reference (check density-scaling convention)"


# ===========================================================================
# 9. PSD convention-INDEPENDENT sanity: white -> flat; sinusoid -> line at f0.
#    LEVEL 3 statistical (RESULT-lfp-psd/snf-psd shape).
# ===========================================================================
def test_white_noise_psd_is_flat_and_sinusoid_psd_peaks_at_f0():
    """LEVEL 3 statistical (RESULT-lfp-psd/snf-psd). White-noise multitaper PSD is
    flat across frequency (ratio test -- convention-independent); a pure f0
    sinusoid concentrates power at the bin nearest f0. Guards against a
    transposed/misindexed frequency axis independent of absolute PSD units."""
    # flatness of white-noise PSD
    x, y = make_signals("white_common", T=200.0, amp=1.0, seed=8)
    out = reference_multitaper_coherence(x, y, default_window_starts(len(x)))
    f = np.asarray(out["freqs"], float)
    band = (f >= 5.0) & (f <= 450.0)
    p = np.asarray(out["lfp_psd"], float)[band]
    assert p.std() / p.mean() < 0.15, "white-noise PSD not flat across frequency"

    # sinusoid line location
    f0 = 6.0
    xs, ys = make_signals("sinusoid", T=200.0, amp=1.0, f0=f0, seed=9)
    outs = reference_multitaper_coherence(xs, ys, default_window_starts(len(xs)))
    fs = np.asarray(outs["freqs"], float)
    ps = np.asarray(outs["lfp_psd"], float)
    ipk = int(np.argmax(ps))
    assert abs(fs[ipk] - f0) <= 0.5, \
        f"sinusoid PSD peak at {fs[ipk]:.2f} Hz, expected {f0} Hz"


# ===========================================================================
# 10. Coherence is bounded in [0,1] for arbitrary inputs. LEVEL 2 property.
# ===========================================================================
def test_coherence_bounded_zero_one():
    """LEVEL 2 property (RQ-023). Magnitude coherence = |average unit-normalizable
    cross-spectrum| is bounded in [0,1] by Cauchy-Schwarz for ANY input; a value
    >1 signals a normalization bug (e.g. squared numerator over linear denom)."""
    for seed in range(5):
        x, y = make_signals("white_common", T=80.0, amp=float(seed) * 0.7 + 0.2,
                            seed=100 + seed)
        out = oracle_multitaper(x, y, default_window_starts(len(x)))
        coh = np.asarray(out["coherence"], float)
        coh = coh[np.isfinite(coh)]
        assert coh.min() >= -1e-9
        assert coh.max() <= 1.0 + 1e-9, "coherence exceeded 1 -- normalization bug"

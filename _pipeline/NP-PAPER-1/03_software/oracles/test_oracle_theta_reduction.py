"""
test_oracle_theta_reduction.py -- NP-PAPER-1 oracle for reduce_theta_coherence
==============================================================================
Written by the ORACLE DESIGNER (Blueprint v2 P3) BEFORE reduce_theta_coherence
exists. Owns the REDUCTION subset of NP-PAPER-1's new functions.

Function under test (repo_map.md #8, contract aggregation step
docx_source_refs [OUT-011, RQ-059, RQ-060, RQ-061, RQ-062, CON-016, OUT-028];
RESULT-theta-coh, contract acceptance level 2 numeric):

    reduce_theta_coherence(coh_f, freqs, null_threshold_f) -- deterministic
    reduction of ONE per-experiment x depth coherence spectrum C(f)
    (already aggregated over retained control windows upstream) to the four
    Table-4 scalars:
      * mean theta coherence  (RQ-060, PRIMARY) = mean of C(f) over 2-12 Hz
      * peak theta coherence  (RQ-061)          = max of C(f) over 2-12 Hz,
                                                   and the frequency where it occurs
      * significant-theta fraction (RQ-062)     = fraction of the 2-12 Hz band
                                                   whose C(f) EXCEEDS the 95% null
      * peak-coherence percentile (CON-016)     = percentile of the peak theta
                                                   coherence within the
                                                   circular-shift null

Every expected value in this file is computed BY HAND / by direct numpy on
small hand-constructed arrays -- NOT by running reduce_theta_coherence.
numpy is used only as a calculator for the closed-form definitions above.

-----------------------------------------------------------------------------
ORACLE-DESIGNER DECISIONS (flagged for P2/implementer sign-off, NOT silently
decided -- the approved contract pins the DEFINITIONS but not the operational
edges below):
-----------------------------------------------------------------------------
 D1. SIGNATURE. The cartographer's proposed 3-arg signature
       reduce_theta_coherence(coh_f, freqs, null_threshold_f)
     CANNOT compute the peak-coherence percentile (CON-016), which requires the
     full circular-shift null DISTRIBUTION, not just the 95% threshold vector.
     The contract does not pin an exact signature, so this oracle PINS:

       reduce_theta_coherence(coh_f, freqs, null_threshold_f,
                              null_coh_dist=None, theta_band=(2.0, 12.0)) -> dict

     with null_coh_dist an (n_shifts, n_freqs) array aligned to `freqs`
     (the per-shift, per-frequency null coherence from circular_shift_null).
     When null_coh_dist is None the function must return peak_coh_percentile
     as np.nan (percentile is undefined without the null). RETURN KEYS pinned:
       mean_theta_coh, peak_theta_coh, peak_theta_freq,
       sig_theta_fraction, peak_coh_percentile.

 D2. THETA-BAND INCLUSIVITY. Band membership is CLOSED: (freqs >= 2.0) &
     (freqs <= 12.0). A frequency landing exactly on 2 Hz or 12 Hz is IN the
     band. (Tested by the boundary microcases.)

 D3. SIGNIFICANT-THETA FRACTION is a fraction of frequency BINS in the band
     (count of band bins with C(f) > threshold, divided by the number of band
     bins) -- the natural discrete reading of RQ-062. Exceedance is STRICT
     (C(f) > threshold; C(f) == threshold does NOT count), matching the
     detect_eth_contact "strictly above" precedent (PROH-008 family) and the
     RESULT-null-threshold "exceeds" language.

 D4. PEAK-FREQUENCY TIE-BREAK. If several band frequencies tie for the maximum,
     report the LOWEST such frequency (== numpy.argmax on an ascending `freqs`).

 D5. PEAK-COHERENCE PERCENTILE operationalized as the percentile of the
     observed peak theta coherence within the null distribution of PER-SHIFT
     peak theta coherences: null_band_peaks[s] = max over the 2-12 Hz band of
     null_coh_dist[s, :]; percentile = scipy.stats.percentileofscore(
     null_band_peaks, peak_theta_coh, kind='mean'). ALTERNATIVE readings of
     CON-016 exist (e.g. percentile at the observed peak frequency only); this
     is flagged so the implementer/P2 can confirm. The microcases are built so
     the null peaks are strictly below/above the observed peak, so the kind=
     'mean'/'strict'/'weak' choice does not change the reported value or the
     >=95% significance DECISION that RESULT-theta-coh inherits.

Acceptance-level mapping (contract acceptance_criteria, RESULT-theta-coh):
  mean/peak theta coherence + peak frequency ...... level 2 (numeric, rtol 1e-6);
      deterministic arithmetic so asserted at exact tolerance here.
  significant-theta fraction ...................... level 2 value + inherits the
      circular-shift significance DECISION (level 3) it feeds.
  peak-coherence percentile ...................... level 2 value + level-3
      exceeds/does-not-exceed 95% null decision.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.stats import percentileofscore

# --- import the (not-yet-written) function; gate impl tests on availability ---
_SW_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-PAPER-1\03_software"
if _SW_DIR not in sys.path and os.path.isdir(_SW_DIR):
    sys.path.insert(0, _SW_DIR)
try:
    from reduce_theta_coherence import reduce_theta_coherence  # noqa: E402
    IMPL_AVAILABLE = True
except Exception:  # pragma: no cover - expected before implementation exists
    reduce_theta_coherence = None
    IMPL_AVAILABLE = False

impl = pytest.mark.skipif(not IMPL_AVAILABLE,
                          reason="reduce_theta_coherence not implemented yet (P3 oracle)")

# ---------------------------------------------------------------------------
# Shared hand-built frequency grid: 5 out-of-band + 7 in-band points.
# Band (closed 2-12 Hz) => {2,3,5,7,9,11,12} Hz, exactly 7 bins.
# ---------------------------------------------------------------------------
FREQS = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 12.0, 14.0, 16.0])
BAND = (FREQS >= 2.0) & (FREQS <= 12.0)  # D2: closed interval
N_BAND = int(BAND.sum())                 # == 7


def _null_dist_from_band_peaks(band_peaks: np.ndarray) -> np.ndarray:
    """Build an (n_shifts, n_freqs) null matrix whose per-shift band-max equals
    the given band_peaks (band columns constant = peak, out-of-band = 0)."""
    n = len(band_peaks)
    m = np.zeros((n, FREQS.size))
    m[:, BAND] = band_peaks[:, None]
    return m


# ===========================================================================
# REFERENCE self-checks (run NOW, no implementation needed): prove the
# hand-computed expected values are internally correct before they gate code.
# ===========================================================================
def test_reference_band_is_seven_closed_bins():
    """D2 anchor: the 2-12 Hz closed band contains exactly the 7 hand-chosen
    bins incl. the 2 Hz and 12 Hz endpoints (guards against an off-by-one/
    open-interval band mask in the implementation)."""
    assert N_BAND == 7
    assert FREQS[BAND].tolist() == [2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 12.0]


def test_reference_scalar_definitions_are_closed_form():
    """Reference arithmetic for Case A, computed directly from the definitions
    (RQ-060/061/062) -- this is the oracle the implementation must reproduce."""
    coh = np.array([0.99, 0.99, 0.99, 0.10, 0.20, 0.30, 0.50, 0.40, 0.20, 0.10, 0.99, 0.99])
    thr = np.where(BAND, 0.15, 0.5)
    assert coh[BAND].mean() == pytest.approx(1.8 / 7.0)          # mean theta
    assert coh[BAND].max() == 0.50                                # peak theta
    assert FREQS[BAND][np.argmax(coh[BAND])] == 7.0              # peak freq
    assert (coh[BAND] > thr[BAND]).mean() == pytest.approx(5 / 7)  # sig frac (strict)


# ===========================================================================
# (A) BASIC microcase: all four scalars by hand; out-of-band values (0.99) must
#     be ignored (band restriction), so peak is 0.50 @ 7 Hz not 0.99.
# ===========================================================================
@impl
def test_basic_microcase_all_four_scalars():
    coh = np.array([0.99, 0.99, 0.99, 0.10, 0.20, 0.30, 0.50, 0.40, 0.20, 0.10, 0.99, 0.99])
    thr = np.where(BAND, 0.15, 0.5)
    null_dist = _null_dist_from_band_peaks(
        np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.49]))
    r = reduce_theta_coherence(coh, FREQS, thr, null_coh_dist=null_dist)

    assert r["mean_theta_coh"] == pytest.approx(1.8 / 7.0, rel=1e-6)
    assert r["peak_theta_coh"] == pytest.approx(0.50, rel=1e-6)
    assert r["peak_theta_freq"] == pytest.approx(7.0, rel=1e-6)
    assert r["sig_theta_fraction"] == pytest.approx(5 / 7, rel=1e-9)
    # all 10 null band-peaks (max 0.49) are strictly below observed 0.50 -> 100th pct
    assert r["peak_coh_percentile"] == pytest.approx(100.0, abs=1e-9)


# ===========================================================================
# (B) BAND-BOUNDARY edge cases: peak exactly at 2 Hz and exactly at 12 Hz.
# ===========================================================================
@impl
def test_peak_at_lower_boundary_2hz_is_included():
    coh = np.array([0.99, 0.99, 0.99, 0.60, 0.20, 0.30, 0.50, 0.40, 0.20, 0.10, 0.99, 0.99])
    thr = np.where(BAND, 0.15, 0.5)
    r = reduce_theta_coherence(coh, FREQS, thr)
    assert r["peak_theta_coh"] == pytest.approx(0.60, rel=1e-6)
    assert r["peak_theta_freq"] == pytest.approx(2.0, rel=1e-6)  # 2 Hz is IN band (D2)


@impl
def test_peak_at_upper_boundary_12hz_is_included():
    coh = np.array([0.99, 0.99, 0.99, 0.10, 0.20, 0.30, 0.40, 0.30, 0.20, 0.55, 0.99, 0.99])
    thr = np.where(BAND, 0.15, 0.5)
    r = reduce_theta_coherence(coh, FREQS, thr)
    assert r["peak_theta_coh"] == pytest.approx(0.55, rel=1e-6)
    assert r["peak_theta_freq"] == pytest.approx(12.0, rel=1e-6)  # 12 Hz is IN band (D2)


# ===========================================================================
# (C) TIE for the maximum: lowest tying frequency wins (D4).
# ===========================================================================
@impl
def test_tied_peak_returns_lowest_frequency():
    # 0.50 occurs at BOTH 3 Hz and 7 Hz; must report 3 Hz.
    coh = np.array([0.99, 0.99, 0.99, 0.10, 0.50, 0.30, 0.50, 0.40, 0.20, 0.10, 0.99, 0.99])
    thr = np.where(BAND, 0.15, 0.5)
    r = reduce_theta_coherence(coh, FREQS, thr)
    assert r["peak_theta_coh"] == pytest.approx(0.50, rel=1e-6)
    assert r["peak_theta_freq"] == pytest.approx(3.0, rel=1e-6)


# ===========================================================================
# (D) SIGNIFICANT-THETA FRACTION strictness + degenerate 0/1 endpoints (D3).
# ===========================================================================
@impl
def test_sig_fraction_strict_exceedance_excludes_equality():
    # threshold 0.20 constant in band; C == 0.20 at two bins must NOT count.
    coh = np.array([0.99, 0.99, 0.99, 0.10, 0.20, 0.30, 0.50, 0.40, 0.20, 0.10, 0.99, 0.99])
    thr = np.where(BAND, 0.20, 0.5)
    r = reduce_theta_coherence(coh, FREQS, thr)
    # strictly greater: 0.30,0.50,0.40 -> 3 of 7
    assert r["sig_theta_fraction"] == pytest.approx(3 / 7, rel=1e-9)


@impl
def test_sig_fraction_all_below_is_zero_and_all_above_is_one():
    thr = np.where(BAND, 0.15, 0.5)
    coh_low = np.where(BAND, 0.05, 0.99)   # every band bin below threshold
    coh_high = np.where(BAND, 0.90, 0.01)  # every band bin above threshold
    r_low = reduce_theta_coherence(coh_low, FREQS, thr)
    r_high = reduce_theta_coherence(coh_high, FREQS, thr)
    assert r_low["sig_theta_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert r_high["sig_theta_fraction"] == pytest.approx(1.0, abs=1e-12)


# ===========================================================================
# (E) PEAK-COHERENCE PERCENTILE: mid value + inherited 95% DECISION (CON-016).
# ===========================================================================
@impl
def test_peak_percentile_mid_and_below_95_decision():
    coh = np.array([0.99, 0.99, 0.99, 0.10, 0.20, 0.30, 0.50, 0.40, 0.20, 0.10, 0.99, 0.99])
    thr = np.where(BAND, 0.15, 0.5)
    # 20 null band-peaks: 15 at 0.10 (below), 5 at 0.60 (above observed 0.50).
    band_peaks = np.array([0.10] * 15 + [0.60] * 5)
    null_dist = _null_dist_from_band_peaks(band_peaks)
    r = reduce_theta_coherence(coh, FREQS, thr, null_coh_dist=null_dist)

    assert r["peak_theta_coh"] == pytest.approx(0.50, rel=1e-6)
    # reference: 15/20 strictly below -> 75th percentile (kind-invariant here)
    expected_pct = percentileofscore(band_peaks, 0.50, kind="mean")
    assert expected_pct == pytest.approx(75.0)
    assert r["peak_coh_percentile"] == pytest.approx(75.0, abs=1e-6)
    # DECISION inherited by RESULT-theta-coh: 75th < 95th -> NOT significant
    assert r["peak_coh_percentile"] < 95.0


@impl
def test_peak_percentile_is_nan_when_no_null_distribution_supplied():
    """Without the circular-shift null distribution the percentile (CON-016) is
    undefined and must be np.nan -- the other three scalars still computed."""
    coh = np.array([0.99, 0.99, 0.99, 0.10, 0.20, 0.30, 0.50, 0.40, 0.20, 0.10, 0.99, 0.99])
    thr = np.where(BAND, 0.15, 0.5)
    r = reduce_theta_coherence(coh, FREQS, thr, null_coh_dist=None)
    assert np.isnan(r["peak_coh_percentile"])
    assert r["mean_theta_coh"] == pytest.approx(1.8 / 7.0, rel=1e-6)


# ===========================================================================
# (F) Single-bin band degenerate: mean == peak == that bin's value.
# ===========================================================================
@impl
def test_single_band_bin_mean_equals_peak():
    freqs = np.array([1.0, 5.0, 20.0])           # only 5 Hz is inside 2-12 Hz
    coh = np.array([0.9, 0.42, 0.9])
    thr = np.array([0.5, 0.10, 0.5])
    r = reduce_theta_coherence(coh, freqs, thr)
    assert r["mean_theta_coh"] == pytest.approx(0.42, rel=1e-9)
    assert r["peak_theta_coh"] == pytest.approx(0.42, rel=1e-9)
    assert r["peak_theta_freq"] == pytest.approx(5.0, rel=1e-9)
    assert r["sig_theta_fraction"] == pytest.approx(1.0, abs=1e-12)  # 0.42 > 0.10

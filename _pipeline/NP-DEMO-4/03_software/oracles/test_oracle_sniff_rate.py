# ORACLE
"""
Analytical / synthetic-ground-truth microcases for the NOT-YET-IMPLEMENTED
compute_sniff_rate(D) -> dict (new kernel, contract_v001.yaml
new_kernel_functions_required / preprocessing step "Compute instantaneous
sniff rate at each LV sample from SNF via compute_sniff_rate").

Targets:
  - RESULT-sniff-rate-matrix (level 3, distributional -- "Per-trial sniff
    rate vector within 1e-4 Hz of reference on fixture slice; distributional
    agreement on full workload.")
  - RESULT-methods-table's avg_sniffs_per_LabView_trial field (same onset
    detection).
  - Feeds CON-003's per-trial "mean sniff rate in first ETH contact vs
    pre-valve control window" comparison.

Contract-pinned algorithm (repo_map.md sec 6 / contract preprocessing):
  1. Detect sniff onsets from D['SNF'] using the SAME method already
     VALIDATED in Optimized Python/analyses/compute_sniff_phase.py:
     low-pass FIR (scipy.signal.firwin(21, 40 Hz, fs=LV_Fs, window='hamming',
     pass_zero='lowpass', scale=True)) + zero-phase filtfilt
     (padtype='odd', padlen=3*(ntaps-1)=60) + z-score (ddof=1) + downward
     threshold crossing at -0.5 std + MIN_ISI rejection
     (MIN_ISI = matlab_round(0.05*LV_Fs), half-away-from-zero).
  2. Inter-onset intervals (ISI) in seconds = diff(onset_sample_idx)/LV_Fs.
  3. Interpolate 1/ISI to every LV sample, nearest-neighbor style, clamped/
     extrapolated flat outside the onset range.
  4. Returns a copy of D with D['sniff_rate'] as (n_LV_samples,) float64 Hz.

Design notes (Blueprint v2 P3 -- oracle design phase, NO implementation run):
  Per task instructions, expected values must be anchored to golden fixtures /
  first principles, NOT produced by running the new code. Step 1 above is
  *already validated* production code (compute_sniff_phase.py, shipped and
  certified under Optimized Python/); this file reproduces that exact,
  already-published recipe verbatim as `_reference_sniff_onsets()` below (read
  2026-07-23; see docstring for the line-by-line correspondence) and uses it,
  executed independently at test time, as ground truth for onset TIMES/ISIs.
  This is the same design pattern NP-DEMO-3's oracle suite used for
  detect_eth_contact (a hand reproduction of an already-validated recipe,
  not the new code under test). Only the rate-INTERPOLATION step (3-4) is
  genuinely new and is what these tests actually probe.

  Every test that needs the real compute_sniff_rate() gracefully skips via
  `_get_compute_sniff_rate()` (try/except ImportError) until
  03_software/compute_sniff_rate.py exists; the pure-arithmetic MIN_ISI test
  and the reference-onset-detector self-checks run and pass today,
  independent of the new module.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.signal

_THIS_FILE = Path(__file__).resolve()
_SOFTWARE_DIR = _THIS_FILE.parents[1]  # .../NP-DEMO-4/03_software


def _get_compute_sniff_rate():
    """Try to import the real compute_sniff_rate. Returns the function, or
    None if the module does not exist yet (expected pre-implementation)."""
    path = str(_SOFTWARE_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from compute_sniff_rate import compute_sniff_rate  # type: ignore
    except ImportError:
        return None
    return compute_sniff_rate


# ---------------------------------------------------------------------------
# Reference onset detector -- verbatim reproduction of the ALREADY-VALIDATED
# Optimized Python/analyses/compute_sniff_phase.py onset-detection recipe
# (lines ~32-50 as of 2026-07-23). NOT the new code under test.
# ---------------------------------------------------------------------------
def _matlab_round(x):
    """Half-away-from-zero rounding, matching MATLAB round() (mirrors
    Optimized Python/analyses/_common.py:matlab_round, reproduced inline so
    this oracle file has zero dependency on kernel import paths)."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _reference_sniff_onsets(SNF: np.ndarray, LV_Fs: float, threshold_std: float = -0.5):
    """Ground-truth onset-sample-index detector, verbatim from the validated
    compute_sniff_phase.m port:
      fco = min(40, LV_Fs/2*0.9)
      taps = firwin(21, fco, fs=LV_Fs, window='hamming', pass_zero='lowpass', scale=True)
      SNF_filt = filtfilt(taps, [1], SNF, padtype='odd', padlen=3*(ntaps-1))
      SNF_z = (SNF_filt - mean) / std(ddof=1)
      onsets = downward crossings of SNF_z below threshold_std
      MIN_ISI = matlab_round(0.05*LV_Fs); reject onsets closer than MIN_ISI
                samples to the immediately preceding RAW onset.
    Returns (onset_sample_idx: int array, MIN_ISI: int).
    """
    SNF = np.asarray(SNF, dtype=np.float64).ravel()
    fco = min(40.0, LV_Fs / 2.0 * 0.9)
    taps = scipy.signal.firwin(21, fco, fs=LV_Fs, window="hamming",
                                pass_zero="lowpass", scale=True)
    padlen = 3 * (taps.size - 1)
    SNF_filt = scipy.signal.filtfilt(taps, [1.0], SNF, padtype="odd", padlen=padlen)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        SNF_z = (SNF_filt - SNF_filt.mean()) / SNF_filt.std(ddof=1)

    below = (SNF_z < threshold_std).astype(np.int64)
    dbelow = np.diff(np.concatenate(([0], below)))
    onsets = np.flatnonzero(dbelow == 1)

    MIN_ISI = int(_matlab_round(0.05 * LV_Fs))
    if onsets.size > 1:
        keep = np.concatenate(([True], np.diff(onsets) >= MIN_ISI))
        onsets = onsets[keep]
    return onsets, MIN_ISI


def _nearest_neighbor_rate(onset_samples: np.ndarray, LV_Fs: float, n_samples: int) -> np.ndarray:
    """Oracle-adopted reference interpolation of instantaneous rate: each ISI
    (between consecutive onsets) is anchored at its LEADING onset sample; the
    rate at any LV sample is the value of the most recently started ISI
    (nearest-neighbor / zero-order-hold, flat-extrapolated before the first
    onset and after the second-to-last onset). This is ONE reasonable,
    explicitly-documented reading of "nearest-neighbor interpolation of 1/ISI,
    extrapolate/clip at the boundary" (contract/repo_map wording is not
    pixel-exact on tie-breaking) -- tests below therefore compare against this
    reference with generous tolerance / mid-interval sampling, never at
    onset-adjacent boundary samples, so they remain valid regardless of which
    defensible interpolation convention the real implementation adopts."""
    if onset_samples.size < 2:
        return np.full(n_samples, np.nan)
    isi_samples = np.diff(onset_samples).astype(np.float64)
    isi_s = isi_samples / LV_Fs
    rate_hz = 1.0 / isi_s
    idx = np.arange(n_samples)
    # searchsorted: for each idx, find which onset-to-onset interval it falls in.
    bin_idx = np.searchsorted(onset_samples[1:], idx, side="right")
    bin_idx = np.clip(bin_idx, 0, rate_hz.size - 1)
    return rate_hz[bin_idx]


# ===========================================================================
# Pure-arithmetic tests (no signal construction, no import needed) -- these
# pass TODAY, before compute_sniff_rate.py exists.
# ===========================================================================
def test_min_isi_rejection_pure_arithmetic():
    """MIN_ISI rejection microcase, first principles, exact integer arithmetic
    (mirrors the ALREADY-VALIDATED rule verbatim: 'keep = [True] + (diff(onsets)
    >= MIN_ISI)', applied to raw onsets, i.e. each onset is compared only to
    the immediately preceding RAW onset, not the last KEPT one).

    LV_Fs = 125 Hz -> MIN_ISI = matlab_round(0.05*125) = matlab_round(6.25) = 6.
    onsets = [10, 14, 80, 84, 200]; diffs = [4, 66, 4, 116].
    keep   = [True, 4>=6, 66>=6, 4>=6, 116>=6] = [T, F, T, F, T]
    -> kept onsets = [10, 80, 200] (14 dropped: 4 < 6 samples after 10;
                                     84 dropped: 4 < 6 samples after 80)."""
    LV_Fs = 125.0
    MIN_ISI = int(_matlab_round(0.05 * LV_Fs))
    assert MIN_ISI == 6

    onsets = np.array([10, 14, 80, 84, 200])
    diffs = np.diff(onsets)
    assert list(diffs) == [4, 66, 4, 116]
    keep = np.concatenate(([True], diffs >= MIN_ISI))
    assert list(keep) == [True, False, True, False, True]
    kept = onsets[keep]
    assert list(kept) == [10, 80, 200]


@pytest.mark.parametrize("LV_Fs,expected_min_isi", [(125.0, 6), (100.0, 5), (40.0, 2), (250.0, 13)])
def test_min_isi_sample_count_matches_matlab_round(LV_Fs, expected_min_isi):
    """MIN_ISI = matlab_round(0.05*LV_Fs), half-away-from-zero (not banker's
    rounding): 0.05*125=6.25->6; 0.05*100=5.0->5; 0.05*40=2.0->2;
    0.05*250=12.5->13 (half rounds AWAY from zero, i.e. up for positive x;
    numpy's own round() would give 12 here via banker's rounding -- this is
    exactly the hazard matlab_round exists to avoid)."""
    assert int(_matlab_round(0.05 * LV_Fs)) == expected_min_isi


def test_reference_onset_detector_constant_signal_has_no_onsets():
    """Ground-truth (not-new-code) sanity check: a perfectly constant SNF
    trace has zero variance after the unit-DC-gain lowpass filter (a constant
    signal is unchanged by a properly-normalized zero-phase FIR lowpass, and
    'odd' padding of a constant array is itself constant, so filtfilt
    introduces no edge transient). std(ddof=1)=0 makes the z-score 0/0 = NaN
    everywhere; NaN < threshold is False for every sample (numpy semantics),
    so zero onsets are detected -- no exception, no false onsets."""
    SNF = np.full(300, 0.42)
    onsets, _ = _reference_sniff_onsets(SNF, LV_Fs=125.0)
    assert onsets.size == 0


# ===========================================================================
# Tests requiring the real compute_sniff_rate() -- skip until implemented.
# ===========================================================================
def test_sine_wave_below_cutoff_gives_rate_near_true_frequency():
    """SNF = sin(2*pi*f*t), f=2 Hz, LV_Fs=125 Hz, 10 s duration -- f is far
    below the 40 Hz lowpass cutoff, so the filter passes it with ~unit gain
    and (being zero-phase via filtfilt) introduces no time shift. A pure
    sinusoid crosses any fixed level in (-1,1) exactly once per period on the
    way down, so the reference onset detector (first-principles, see above)
    must find exactly floor(duration*f) or so onsets, spaced ~1/f=0.5 s apart
    -- this was verified analytically and independently confirmed by
    executing ONLY the reference detector (not compute_sniff_rate) at
    oracle-design time: 20 onsets over 10 s, mean ISI = 0.5002 s.

    compute_sniff_rate(D)['sniff_rate'] must therefore average to ~2.0 Hz over
    the steady middle of the trial (edges excluded: filtfilt padding and
    interpolation-boundary flat-extrapolation both distort the first/last
    ~1 s)."""
    compute_sniff_rate = _get_compute_sniff_rate()
    if compute_sniff_rate is None:
        pytest.skip("compute_sniff_rate.py not yet implemented")

    LV_Fs = 125.0
    duration_s = 10.0
    n = int(duration_s * LV_Fs)
    t = np.arange(n) / LV_Fs
    f = 2.0
    SNF = np.sin(2 * np.pi * f * t)

    ref_onsets, _ = _reference_sniff_onsets(SNF, LV_Fs)
    assert ref_onsets.size >= int(duration_s * f) - 2  # sanity: reference itself behaves as expected

    D = {"SNF": SNF, "LV_Fs": LV_Fs}
    out = compute_sniff_rate(D)
    assert "sniff_rate" in out
    rate = np.asarray(out["sniff_rate"], dtype=np.float64)

    assert rate.shape == (n,)
    assert rate.dtype == np.float64

    # Steady middle region: exclude first/last 1 s (125 samples) edge effects.
    middle = rate[125:-125]
    assert np.all(np.isfinite(middle))
    assert np.median(middle) == pytest.approx(2.0, abs=0.3)
    # Gross-bug guard: must not be off by a factor of 2 (double-counting
    # crossings) or report period instead of frequency.
    assert not (1.0 - 0.15 <= np.median(middle) <= 1.0 + 0.15)
    assert not (4.0 - 0.3 <= np.median(middle) <= 4.0 + 0.3)


def test_two_known_isis_from_separated_dips_give_locally_correct_rate():
    """Three well-separated, identical raised-cosine ('Hann') dips at nominal
    centers 0.5 s, 1.0 s, 2.0 s (width 0.15 s, comfortably narrowband well
    below the 40 Hz cutoff so filtfilt preserves their shape almost exactly).
    Verified analytically (and independently confirmed by executing ONLY the
    reference onset detector, not compute_sniff_rate, at oracle-design time):
    identical dip shapes produce a *consistent, shape-dependent* onset latency
    relative to each dip's true center, so the ABSOLUTE onset times are offset
    (~-0.05 s here) from 0.5/1.0/2.0 -- but because every dip has the SAME
    shape, that offset cancels in the ISI (spacing between CONSECUTIVE
    onsets), which reproduces the true dip-to-dip spacing almost exactly:
    onset2-onset1 ~ 0.504 s (true spacing 0.5 s), onset3-onset2 ~ 1.000 s
    (true spacing 1.0 s).

    Therefore: instantaneous sniff_rate sampled well inside the FIRST gap
    (between onset 1 and onset 2) must be ~1/0.5 = 2.0 Hz, and sampled well
    inside the SECOND gap (between onset 2 and onset 3) must be ~1/1.0 = 1.0 Hz
    -- this is the "two known onset times -> known ISI -> known local rate"
    property requested for this microcase, expressed in terms of onset-to-onset
    SPACING (the quantity compute_sniff_rate actually consumes) rather than
    the dips' nominal centers (which the algorithm does not reproduce exactly)."""
    compute_sniff_rate = _get_compute_sniff_rate()
    if compute_sniff_rate is None:
        pytest.skip("compute_sniff_rate.py not yet implemented")

    LV_Fs = 125.0
    duration_s = 3.0
    n = int(duration_s * LV_Fs)
    t = np.arange(n) / LV_Fs

    def hann_dip(t, center, width, depth):
        x = (t - center) / (width / 2.0)
        return np.where(np.abs(x) <= 1.0, -depth * 0.5 * (1 + np.cos(np.pi * x)), 0.0)

    SNF = hann_dip(t, 0.5, 0.15, 3.0) + hann_dip(t, 1.0, 0.15, 3.0) + hann_dip(t, 2.0, 0.15, 3.0)

    ref_onsets, _ = _reference_sniff_onsets(SNF, LV_Fs)
    assert ref_onsets.size == 3, (
        "Reference onset detector (first-principles) did not find the expected "
        "3 onsets for this fixture; oracle construction assumption violated."
    )
    onset_times_s = ref_onsets / LV_Fs
    isi1 = onset_times_s[1] - onset_times_s[0]
    isi2 = onset_times_s[2] - onset_times_s[1]
    assert isi1 == pytest.approx(0.5, abs=0.03)
    assert isi2 == pytest.approx(1.0, abs=0.03)

    D = {"SNF": SNF, "LV_Fs": LV_Fs}
    out = compute_sniff_rate(D)
    rate = np.asarray(out["sniff_rate"], dtype=np.float64)
    assert rate.shape == (n,)

    # Sample strictly inside each gap, away from either boundary onset, so the
    # check is robust to any defensible interpolation/anchor convention.
    mid1_idx = int(round((onset_times_s[0] + onset_times_s[1]) / 2.0 * LV_Fs))
    mid2_idx = int(round((onset_times_s[1] + onset_times_s[2]) / 2.0 * LV_Fs))

    assert rate[mid1_idx] == pytest.approx(1.0 / isi1, rel=0.25)
    assert rate[mid2_idx] == pytest.approx(1.0 / isi2, rel=0.25)
    # The two regions must be clearly distinguishable (not both collapsed to
    # one global average) -- first gap rate must be visibly higher.
    assert rate[mid1_idx] > rate[mid2_idx]


def test_no_onsets_fallback_is_documented_and_does_not_crash():
    """Edge case: constant SNF (see test_reference_onset_detector_constant_signal_has_no_onsets)
    yields zero onsets, hence zero ISIs. The contract does not pin an exact
    fallback value for this degenerate case; this oracle DOCUMENTS the
    acceptable contract as: compute_sniff_rate must NOT raise, must return an
    array of the correct length (n_LV_samples,) float64, and that array must
    be uniformly one of {NaN, 0.0} (not a mix, not garbage, not wrong length).
    Any other behavior is treated as a genuine implementation defect to flag
    at Red-tier review."""
    compute_sniff_rate = _get_compute_sniff_rate()
    if compute_sniff_rate is None:
        pytest.skip("compute_sniff_rate.py not yet implemented")

    LV_Fs = 125.0
    n = 300
    SNF = np.full(n, 0.42)
    D = {"SNF": SNF, "LV_Fs": LV_Fs}

    out = compute_sniff_rate(D)
    rate = np.asarray(out["sniff_rate"], dtype=np.float64)
    assert rate.shape == (n,)
    assert rate.dtype == np.float64

    all_nan = np.all(np.isnan(rate))
    all_zero = np.all(rate == 0.0)
    assert all_nan or all_zero, (
        "no-onset fallback must be a uniform NaN array or a uniform 0.0 array "
        f"(got a mix / other values: unique sample = {rate[:5]!r})"
    )


def test_sniff_rate_is_positive_and_finite_where_onsets_exist():
    """Monotonic sanity property (item 5): wherever a well-formed onset
    sequence exists (the 2 Hz sine case), instantaneous rate must be strictly
    positive and finite everywhere (a physical rate can never be negative or
    zero given a nonzero ISI, and must never be NaN/inf mid-trial when onsets
    bracket that sample)."""
    compute_sniff_rate = _get_compute_sniff_rate()
    if compute_sniff_rate is None:
        pytest.skip("compute_sniff_rate.py not yet implemented")

    LV_Fs = 125.0
    duration_s = 6.0
    n = int(duration_s * LV_Fs)
    t = np.arange(n) / LV_Fs
    SNF = np.sin(2 * np.pi * 2.0 * t)

    D = {"SNF": SNF, "LV_Fs": LV_Fs}
    out = compute_sniff_rate(D)
    rate = np.asarray(out["sniff_rate"], dtype=np.float64)

    ref_onsets, _ = _reference_sniff_onsets(SNF, LV_Fs)
    interior = rate[int(ref_onsets[0]):int(ref_onsets[-1])]
    assert np.all(np.isfinite(interior))
    assert np.all(interior > 0.0)

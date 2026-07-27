# ORACLE
"""
Oracle for the NOT-YET-IMPLEMENTED
    circular_shift_null(lfp_ch, snf_lfp_rate, control_valid_windows,
                        n_shifts=1000, seed=5489)
(03_software/circular_shift_null.py).

Contract sources: RQ-024 (circular-shift null), RQ-041 / CON-003 (>=1000 shifts,
95th-percentile per-frequency threshold, seed 5489 pinned), RQ-038 (BIT-FOR-BIT
reproducible given seed 5489 -- a BLOCKING failure condition independent of the
coherence engine's own tolerance), OUT-011 (RESULT-null-threshold).

Acceptance levels (contract_v001.yaml analysis.acceptance_criteria):
  RESULT-null-threshold -> level 1 exact ("Bit-for-bit reproducible given seed
                           5489 on re-run (zero tolerance)").
  RESULT-theta-coh      -> level 2/3: the significance DECISION (observed vs 95%
                           null) is the workload-level accepted comparison
                           (statistical_decision).

Every expectation is a determinism property, a distribution-free self-calibration
of the shift procedure, or a closed-form/ MC-verified statistic ([D1]/[D3] in
spectral_oracle_ref.py) -- NEVER produced by running the implementation. Tests
use `oracle_null` (real function if present, independent reference until then).

RUNTIME NOTE: the null is O(n_shifts * n_windows) multitaper evaluations, so
these tests use short signals and reduced n_shifts (documented per test). The
PINNED production value is n_shifts>=1000 (CON-003/RQ-041); reducing it for a
procedure-calibration TEST does not touch the analysis configuration.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectral_oracle_ref import (
    FS, THETA_LO, THETA_HI,
    default_window_starts, make_signals, oracle_null,
    reference_circular_shift_null, call_real_null, real_null_available,
    null_coh_mean,
)

SEED = 5489  # lab-pinned (CON-003/RQ-041); must be honored verbatim.


# ===========================================================================
# 1. BIT-FOR-BIT reproducibility given seed 5489. LEVEL 1 exact (RQ-038,
#    blocking failure condition; RESULT-null-threshold zero tolerance).
# ===========================================================================
def test_null_threshold_bit_for_bit_reproducible_same_seed():
    """LEVEL 1 exact (RQ-038 / RESULT-null-threshold). Two runs with the SAME
    seed on identical inputs must produce byte-identical null thresholds AND
    observed coherence -- np.array_equal, not approx."""
    x, y = make_signals("independent", T=60.0, seed=11)
    starts = default_window_starts(len(x))
    a = oracle_null(x, y, starts, n_shifts=200, seed=SEED)
    b = oracle_null(x, y, starts, n_shifts=200, seed=SEED)
    assert np.array_equal(a["null_threshold"], b["null_threshold"]), \
        "null threshold not bit-for-bit reproducible at fixed seed (RQ-038)"
    assert np.array_equal(a["observed"], b["observed"]), \
        "observed coherence not bit-for-bit reproducible (RQ-038)"


# ===========================================================================
# 2. Determinism is IMMUNE to the numpy GLOBAL RNG state / call order.
#    LEVEL 1 exact (RQ-038, hazard H8). The function must use a dedicated
#    seeded Generator, never numpy's global RNG.
# ===========================================================================
def test_null_independent_of_global_rng_state():
    """LEVEL 1 exact (RQ-038 / H8). Perturbing numpy's GLOBAL RNG (np.random.seed
    + draws) between two seed-5489 runs must not change the output. Catches an
    implementation that draws shifts from np.random.* (global state) instead of a
    private Generator(seed)."""
    x, y = make_signals("independent", T=60.0, seed=12)
    starts = default_window_starts(len(x))

    np.random.seed(1234)
    _ = np.random.random(777)
    a = oracle_null(x, y, starts, n_shifts=200, seed=SEED)

    np.random.seed(999)           # different global state, extra draws
    _ = np.random.random(4096)
    b = oracle_null(x, y, starts, n_shifts=200, seed=SEED)

    assert np.array_equal(a["null_threshold"], b["null_threshold"]), \
        "null threshold changed with global RNG state -- must use a private " \
        "Generator seeded with `seed` (RQ-038/H8), not numpy global RNG"


# ===========================================================================
# 3. Threshold IS the per-frequency 95th percentile of the null distribution.
#    LEVEL 2 numeric (CON-003/RQ-041). Verified on the reference (which exposes
#    the null distribution); on the impl only if it exposes one.
# ===========================================================================
def test_threshold_is_95th_percentile_of_null():
    """LEVEL 2 numeric (CON-003/RQ-041 / RESULT-null-threshold). The threshold
    must equal np.percentile(null_distribution, 95, axis=0) per frequency -- not
    the max, the mean, or a different quantile."""
    x, y = make_signals("independent", T=48.0, seed=13)
    starts = default_window_starts(len(x))
    ref = reference_circular_shift_null(x, y, starts, n_shifts=300, seed=SEED)
    expected = np.percentile(ref["null_distribution"], 95.0, axis=0)
    assert np.allclose(ref["null_threshold"], expected, rtol=0, atol=1e-12), \
        "reference threshold is not the exact per-frequency 95th percentile"

    if real_null_available():
        got = call_real_null(x, y, starts, n_shifts=300, seed=SEED)
        # decision-level cross-check: impl threshold must be a valid coherence
        # profile in [0,1] of the right length.
        thr = np.asarray(got["null_threshold"], float)
        assert thr.shape == np.asarray(ref["null_threshold"]).shape
        assert np.nanmin(thr) >= -1e-9 and np.nanmax(thr) <= 1.0 + 1e-9


# ===========================================================================
# 4. FALSE-POSITIVE CALIBRATION: independent signals exceed the 95% null in
#    ~5% of frequency bins by chance. LEVEL 3 statistical (RQ-024).
# ===========================================================================
def test_null_procedure_false_positive_rate_near_five_percent():
    """LEVEL 3 statistical (RQ-024 / RESULT-theta-coh significance decision). For
    genuinely INDEPENDENT signals the observed coherence is one more draw from the
    (circular-shift) null, so P(observed > 95th pct) ~ 0.05 per frequency. We
    sample frequency bins spaced > the 1 Hz multitaper bandwidth across several
    independent realizations so the Bernoulli(0.05) trials are ~independent, then
    apply a binomial confidence band.

    Design-time MC (independent white, 200 shifts): per-realization rates
    0.031-0.075, mean 0.057 -- consistent with 0.05 given estimator + finite-shift
    noise. With N ~ 96 near-independent trials the 95% binomial interval for
    p=0.05 is roughly [0.01, 0.11]; we assert the pooled rate lies in that band
    and is nowhere near 0 or the ~34% single-window floor."""
    probe_hz = np.arange(10.0, 121.0, 10.0)     # 12 bins, spaced >> 1 Hz bandwidth
    exceed = 0
    total = 0
    for r in range(8):
        x, y = make_signals("independent", T=60.0, seed=200 + r)
        starts = default_window_starts(len(x))
        res = oracle_null(x, y, starts, n_shifts=200, seed=SEED)
        f = np.asarray(res["freqs"], float)
        obs = np.asarray(res["observed"], float)
        thr = np.asarray(res["null_threshold"], float)
        idx = [int(np.argmin(np.abs(f - h))) for h in probe_hz]
        exceed += int(np.sum(obs[idx] > thr[idx]))
        total += len(idx)
    rate = exceed / total
    assert 0.01 <= rate <= 0.12, (
        f"circular-shift null false-positive rate {rate:.3f} "
        f"({exceed}/{total}) inconsistent with the nominal 0.05 "
        f"(binomial 95% band ~[0.01,0.11]) -- the shift procedure is "
        f"mis-calibrated")


# ===========================================================================
# 5. TRUE POSITIVE: a shared narrowband stochastic oscillation is flagged.
#    LEVEL 3/4 (RQ-024 + DATA-002). Known ground truth: coherence IS elevated
#    in theta, and circular shifting DESTROYS it, so observed >> threshold there.
# ===========================================================================
def test_shared_theta_oscillation_exceeds_null_threshold():
    """LEVEL 3/4 (RQ-024 / RESULT-theta-coh decision). A shared narrowband
    (4-8 Hz) STOCHASTIC component + independent noise: observed theta coherence
    must exceed the 95% circular-shift null across essentially the whole theta
    band, while the null threshold there stays near the aggregated floor (the
    shift decorrelates the shared component -- [D3]). Design-time MC: theta obs
    mean 0.99 vs threshold 0.137, 100% of theta bins exceed."""
    x, y = make_signals("narrowband", T=120.0, amp=1.0, band=(4.0, 8.0), seed=21)
    starts = default_window_starts(len(x))
    res = oracle_null(x, y, starts, n_shifts=200, seed=SEED)
    f = np.asarray(res["freqs"], float)
    obs = np.asarray(res["observed"], float)
    thr = np.asarray(res["null_threshold"], float)

    inband = (f >= 4.0) & (f <= 8.0)
    frac_exceed = float(np.mean(obs[inband] > thr[inband]))
    assert frac_exceed > 0.8, (
        f"only {frac_exceed:.2f} of the shared-oscillation band exceeded the "
        f"95% null -- true coupling not detected")
    # the shift must genuinely collapse the in-band null (proper null, [D3]).
    assert float(np.mean(thr[inband])) < float(np.mean(obs[inband])), \
        "in-band null threshold did not fall below observed -- circular shift " \
        "failed to decorrelate the shared stochastic component"
    # theta-band reduction target is significant (feeds RESULT-theta-coh).
    theta = (f >= THETA_LO) & (f <= THETA_HI)
    assert float(np.mean(obs[theta] > thr[theta])) > 0.3


# ===========================================================================
# 6. Null threshold is a valid coherence profile and sits near the aggregated
#    floor for independent inputs. LEVEL 3 statistical (sanity / [D1]).
# ===========================================================================
def test_null_threshold_is_valid_and_near_floor_for_independent_inputs():
    """LEVEL 3 statistical (RESULT-null-threshold). For independent inputs the
    per-frequency 95% null threshold must be a valid coherence in [0,1] and sit
    modestly above the aggregated null mean (which is well below the single-window
    Beta floor 0.341 once many windows are pooled -- [D1])."""
    x, y = make_signals("independent", T=120.0, seed=31)
    starts = default_window_starts(len(x))
    res = oracle_null(x, y, starts, n_shifts=200, seed=SEED)
    f = np.asarray(res["freqs"], float)
    thr = np.asarray(res["null_threshold"], float)
    band = (f >= 5.0) & (f <= 120.0)
    tb = thr[band]
    assert np.nanmin(tb) >= -1e-9 and np.nanmax(tb) <= 1.0 + 1e-9
    # Many pooled windows => the 95% threshold should be well under the single-
    # window floor; it must not be stuck up at ~0.34 (that would indicate the
    # underlying coherence averages per-window values instead of pooling).
    assert float(np.nanmean(tb)) < null_coh_mean() , \
        "independent-input 95% null threshold not below the single-window floor"
    assert float(np.nanmean(tb)) > 0.0


# ===========================================================================
# 7. Same inputs + DIFFERENT seed -> generally different null (not a constant).
#    LEVEL 2 property (sanity: the shifts are actually randomized by seed).
# ===========================================================================
def test_different_seed_changes_null_but_same_seed_repeats():
    """LEVEL 2 property (RQ-038 corollary). Different seeds must generally give a
    different null threshold (the procedure is genuinely stochastic), while the
    same seed repeats exactly (already gated in test 1)."""
    x, y = make_signals("independent", T=48.0, seed=41)
    starts = default_window_starts(len(x))
    a = oracle_null(x, y, starts, n_shifts=150, seed=SEED)
    c = oracle_null(x, y, starts, n_shifts=150, seed=SEED + 1)
    assert not np.array_equal(a["null_threshold"], c["null_threshold"]), \
        "different seeds produced identical null -- shifts may be un-seeded/fixed"

"""
oracle_circular_stats.py -- NP-DEMO-2 oracle for `mrl_and_preferred_phase`
============================================================================
Written by the ORACLE DESIGNER (Blueprint v2 P3) BEFORE np_demo2_analysis.py
exists. Targets repo_map.md New-Code-Required function:

    mrl_and_preferred_phase(phase_0to1) -> (mrl, preferred_phase_rad)
        mrl                 = |mean(exp(2*pi*i*phase))|         in [0, 1]
        preferred_phase_rad = angle(mean(exp(2*pi*i*phase)))    in (-pi, pi]
        Empty input -> (nan, nan), must not raise.

Every expected value below is derived analytically (roots of unity, exact
trig identities, the exact "sum of two equal-magnitude vectors bisects their
angle" identity, and the closed-form von Mises MRL = I1(kappa)/I0(kappa)),
NOT by running np_demo2_analysis. scipy.special / scipy.stats are used only
as an independent, already-validated numerical library to evaluate those
closed forms -- never to run "the new code".

Acceptance-level mapping (contract_spec.yaml, NP-DEMO-2, contract_version 1)
-----------------------------------------------------------------------------
    RESULT-mrl-unit    (level 3, distributional; MRL within 1e-6 on fixture slice)
    RESULT-phase-unit  (level 3, circular distance < 1e-6 on fixture slice)
    RESULT-mrl-animal  (level 3, distributional; MRL within 1e-6 on fixture slice)
  This oracle is the pre-fixture analytic/microcase/calibration gate for all
  three: mrl_and_preferred_phase is the single shared primitive behind every
  per-unit and per-animal MRL/phase computation in the contract.

Implementer note
-----------------
    # from np_demo2_analysis import mrl_and_preferred_phase
    # TODO: uncomment when np_demo2_analysis.py exists, and DELETE the
    # inline reference implementation below so this file exercises the real
    # implementation rather than testing itself.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.special
import scipy.stats

# ---------------------------------------------------------------------------
import sys, os
_NP_DEMO2_SW_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_software"
if _NP_DEMO2_SW_DIR not in sys.path and os.path.isdir(_NP_DEMO2_SW_DIR):
    sys.path.insert(0, _NP_DEMO2_SW_DIR)
from np_demo2_analysis import mrl_and_preferred_phase  # noqa: E402
# ---------------------------------------------------------------------------


def _angle_diff(a: float, b: float) -> float:
    """Wrapped a-b in (-pi, pi], used to compare circular quantities without
    tripping on the +-pi branch cut."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


# ===========================================================================
# (a) All phases at 0.0 -> MRL=1.0, preferred=0.0
# ===========================================================================
def test_mrl_all_zero_phase_gives_unit_mrl_and_zero_preferred():
    """N unit vectors all at angle 0 sum to exactly N (real, positive) ->
    mean has magnitude 1 and angle 0 exactly (to float eps)."""
    phases = np.zeros(200, dtype=np.float64)
    mrl, preferred = mrl_and_preferred_phase(phases)
    assert mrl == pytest.approx(1.0, abs=1e-9)
    assert preferred == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# (b) All phases at 0.5 -> MRL=1.0, preferred=pi
# ===========================================================================
def test_mrl_all_half_phase_gives_unit_mrl_and_pi_preferred():
    """exp(i*2*pi*0.5) == exp(i*pi) == -1 exactly (to float eps) -> mean is
    -1+0j -> MRL == 1.0, preferred == pi (angle(-1) == +pi by convention)."""
    phases = np.full(200, 0.5, dtype=np.float64)
    mrl, preferred = mrl_and_preferred_phase(phases)
    assert mrl == pytest.approx(1.0, abs=1e-9)
    assert preferred == pytest.approx(math.pi, abs=1e-9)


# ===========================================================================
# (c) Uniform distribution 0..1 (N=1000) -> MRL < 0.05
# ===========================================================================
def test_mrl_uniform_grid_phases_is_near_zero():
    """The N-th roots of unity (np.linspace(0,1,1000,endpoint=False)) sum to
    EXACTLY zero for N >= 2 (elementary root-of-unity identity), so MRL is
    zero up to floating-point round-off. Spec requires MRL < 0.05; we hold to
    a much tighter analytic bound as the primary assertion."""
    phases = np.linspace(0.0, 1.0, 1000, endpoint=False)
    mrl, _ = mrl_and_preferred_phase(phases)
    assert mrl < 1e-9
    assert mrl < 0.05  # explicit spec bound, independent assertion


# ===========================================================================
# (d) Two-cluster symmetric arrangement -> preferred near (in fact exactly
#     equal to, up to float eps) the midpoint
# ===========================================================================
def test_mrl_two_symmetric_clusters_preferred_is_exact_bisector():
    """For two equal-count clusters at phases 0.2 and 0.4 (angles a, b),
    e^{ia} + e^{ib} = e^{i(a+b)/2} * 2*cos((a-b)/2) -- an exact vector
    identity (factor out the mean angle). Since |a-b| = 0.2*2*pi < pi,
    cos((a-b)/2) > 0, so the resultant's angle is EXACTLY the average angle
    (a+b)/2 (== 2*pi*0.3, the phase midpoint) and its magnitude, after
    dividing by N, is EXACTLY cos((a-b)/2) = cos(0.1*2*pi). This is not an
    approximation -- both values are checked to float precision."""
    a = 2 * math.pi * 0.2
    b = 2 * math.pi * 0.4
    phases = np.concatenate([np.full(50, 0.2), np.full(50, 0.4)])

    mrl, preferred = mrl_and_preferred_phase(phases)

    expected_preferred = (a + b) / 2.0
    expected_mrl = math.cos((a - b) / 2.0)
    assert mrl == pytest.approx(expected_mrl, abs=1e-9)
    assert abs(_angle_diff(preferred, expected_preferred)) < 1e-9
    # sanity: expected_preferred is exactly the phase-0.3 midpoint
    assert expected_preferred == pytest.approx(2 * math.pi * 0.3, abs=1e-12)


# ===========================================================================
# (e) Empty input -> (nan, nan)
# ===========================================================================
def test_mrl_empty_input_returns_nan_pair_without_raising():
    """Empty valid-sniff-spike phase arrays occur whenever a unit has zero
    spikes in one condition; must return (nan, nan), never raise."""
    mrl, preferred = mrl_and_preferred_phase(np.array([], dtype=np.float64))
    assert math.isnan(mrl)
    assert math.isnan(preferred)


# ===========================================================================
# Opposite-phase cancellation (adversarial: two conditions folded together
# would erase phase-locking -- guards the pooling logic tested elsewhere)
# ===========================================================================
def test_mrl_opposite_half_cycle_phases_cancel_to_zero():
    """50 spikes at phase 0.0 (+1 unit vector) and 50 at phase 0.5 (-1 unit
    vector): the resultant sum is exactly 0 -> MRL == 0 up to float
    round-off, regardless of how strongly locked each half is on its own."""
    phases = np.concatenate([np.zeros(50), np.full(50, 0.5)])
    mrl, _ = mrl_and_preferred_phase(phases)
    assert mrl < 1e-9


# ===========================================================================
# Phase-wrap boundary: phase 1.0 must be circularly identical to phase 0.0
# ===========================================================================
def test_mrl_phase_one_wraps_to_same_position_as_phase_zero():
    """exp(i*2*pi*1.0) == exp(i*2*pi*0.0) == 1 (2*pi is a full turn): 50
    spikes at 0.0 + 50 at 1.0 -> MRL == 1.0, preferred == 0.0 rad, not two
    separate clusters."""
    phases = np.concatenate([np.zeros(50), np.ones(50)])
    mrl, preferred = mrl_and_preferred_phase(phases)
    assert mrl == pytest.approx(1.0, abs=1e-9)
    assert preferred == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# Metamorphic relation: rotation invariance of MRL / exact phase shift
# ===========================================================================
def test_metamorphic_rotation_preserves_mrl_and_shifts_preferred_phase():
    """Metamorphic relation: rotating every phase by a constant delta (mod 1)
    must leave MRL unchanged (a rotation-invariant magnitude) and must shift
    the preferred phase by exactly delta*2*pi (mod 2*pi). Uses a
    deliberately non-uniform, non-degenerate phase set (clustered near 0.12)
    so the base MRL is neither 0 nor 1 and rotation is a non-trivial check."""
    base = np.array([0.10, 0.12, 0.15, 0.11, 0.13, 0.09, 0.14, 0.16, 0.10, 0.12])
    delta = 0.37  # arbitrary rotation, in 0-1 (cycle) units

    mrl0, pref0 = mrl_and_preferred_phase(base)
    rotated = (base + delta) % 1.0
    mrl1, pref1 = mrl_and_preferred_phase(rotated)

    assert 0.0 < mrl0 < 1.0, "base case must be non-degenerate for this relation to be meaningful"
    assert mrl1 == pytest.approx(mrl0, abs=1e-9)
    expected_pref1 = pref0 + delta * 2 * math.pi
    assert abs(_angle_diff(pref1, expected_pref1)) < 1e-6


# ===========================================================================
# Synthetic ground truth: von Mises distribution, closed-form MRL
# (statistical calibration check for the one stochastic case in this file)
# ===========================================================================
def test_von_mises_synthetic_mrl_matches_bessel_ratio_closed_form():
    """Synthetic ground truth via a known distribution: for phases drawn iid
    from a von Mises distribution with concentration kappa, the POPULATION
    mean resultant length has the exact closed form

        MRL_population(kappa) = I1(kappa) / I0(kappa)

    (Mardia & Jupp, Directional Statistics, eq. 3.5.5; I0/I1 are modified
    Bessel functions of the first kind, order 0 and 1). This is a Monte
    Carlo statistical-calibration check: draw N=200,000 iid samples with a
    FIXED seed (reproducible) and require the empirical MRL to be within a
    generous tolerance of the analytic value (the Monte Carlo standard error
    at this N and kappa is on the order of 1e-3, so abs=0.01 has a wide
    safety margin and will not flake). Also checks that the empirical
    preferred phase recovers the true von Mises location mu."""
    rng = np.random.default_rng(20260722)
    kappa = 4.0
    mu = 1.1  # arbitrary nonzero location, radians
    n = 200_000

    angles_rad = scipy.stats.vonmises.rvs(kappa, loc=mu, size=n, random_state=rng)
    phases = (angles_rad / (2.0 * math.pi)) % 1.0

    mrl, preferred = mrl_and_preferred_phase(phases)

    analytic_mrl = scipy.special.iv(1, kappa) / scipy.special.iv(0, kappa)
    assert mrl == pytest.approx(analytic_mrl, abs=0.01)
    assert abs(_angle_diff(preferred, mu)) < 0.02


def test_von_mises_synthetic_mrl_low_concentration_near_uniform():
    """Same closed-form calibration at low concentration (kappa=0.5, close to
    uniform): analytic MRL is small (~0.24), and the same Monte Carlo
    tolerance must still hold, exercising the low-MRL regime the uniform
    microcase above only bounds from above."""
    rng = np.random.default_rng(3141592)
    kappa = 0.5
    mu = -0.6
    n = 200_000

    angles_rad = scipy.stats.vonmises.rvs(kappa, loc=mu, size=n, random_state=rng)
    phases = (angles_rad / (2.0 * math.pi)) % 1.0

    mrl, _ = mrl_and_preferred_phase(phases)
    analytic_mrl = scipy.special.iv(1, kappa) / scipy.special.iv(0, kappa)
    assert mrl == pytest.approx(analytic_mrl, abs=0.01)

"""
test_np_demo1_oracle.py — NP-DEMO-1 oracle test suite (Blueprint v2 P3)
=========================================================================
Written by the ORACLE DESIGNER before np_demo1_analysis.py exists. Every
expected value below is derived analytically / from first principles (roots
of unity, exact trig identities, exact Wilcoxon signed-rank combinatorics),
NOT by running the implementation-under-test. This file is the acceptance
oracle for contract_spec.yaml (NP-DEMO-1, contract_version 1, risk: red).

Contract acceptance-criteria -> test mapping
---------------------------------------------
  RESULT-mrl      (level 3, MRL within 1e-6)          -> result_mrl-marked tests
  RESULT-phase    (level 3, circular distance < 1e-6)  -> result_phase-marked tests
  RESULT-stat     (level 3, same significance decision) -> result_stat-marked tests
  RESULT-qc-counts (level 1, exact integer counts)      -> result_qc_counts-marked tests

RED-TIER NOTE: contract_spec.yaml pins risk: red for NP-DEMO-1 (new
preprocessing + cross-tool inference + report-critical). Implementation and
execution against this oracle should be routed through the Opus-tier model
per the blueprint's red-tier escalation policy.

Pinned public API of np_demo1_analysis.py (the implementer must match this
signature set; see conftest.py header for the same contract restated next to
the mock_D fixture it supports):

    mrl_and_preferred_phase(phase_0to1: np.ndarray) -> tuple[float, float]
        phase_0to1 : 1-D array of ALREADY-FILTERED valid sniff phases in
                     [0, 1). Caller has already excluded SNF_PH == -1.
        Returns (mrl, preferred_phase_rad):
            mrl                 = |mean(exp(1j * 2*pi * phase))|      in [0, 1]
            preferred_phase_rad = angle(mean(exp(1j * 2*pi * phase)))  in (-pi, pi]
        Empty input -> (nan, nan), must not raise.

    valid_sniff_mask(spike_SNF_PH: np.ndarray) -> np.ndarray[bool]
        True where spike_SNF_PH >= 0 (kernel sentinel: -1 == outside any
        sniff cycle). Must NOT silently coerce -1 into phase space.

    classify_ethanol(spike_times_s: np.ndarray, ETH_thr: np.ndarray,
                      LV_Fs: float, eth_threshold: float = 0.11
                      ) -> np.ndarray[bool]
        True == ethanol condition.
        lv_idx = matlab_round(spike_times_s * LV_Fs) clamped to
                 [0, len(ETH_thr) - 1]   (analyses._common.matlab_round,
                 half-away-from-zero -- NOT np.round's banker's rounding)
        ethanol iff ETH_thr[lv_idx] > eth_threshold.

    unit_included(n_spikes_total: int, session_duration_s: float,
                  n_valid_sniff_spikes: int, min_rate_hz: float = 0.1,
                  min_spikes: int = 50) -> bool
        False if (n_spikes_total / session_duration_s) < min_rate_hz
           OR n_valid_sniff_spikes < min_spikes.
        Both bounds are inclusive at the ">=" boundary (50 valid spikes passes).

    paired_animal_test(ethanol_means: np.ndarray, control_means: np.ndarray
                        ) -> dict
        Paired per-animal ethanol-vs-control comparison via
        scipy.stats.wilcoxon (exact mode, two-sided). Returns a dict with
        (at minimum) keys:
            'pvalue'      : exact two-sided p-value (float)
            'effect_size' : rank-biserial r in [-1, 1]; positive == ethanol > control
            'statistic'   : the Wilcoxon test statistic
            'n_animals'   : number of paired animals used
            'direction'   : one of {'ethanol>control', 'control>ethanol', 'ns'}

    qc_counts(sniff_onsets: np.ndarray, n_neurons_valid: int,
              trial_numbers: np.ndarray) -> dict
        Returns dict with keys:
            'n_sniffs'  == len(sniff_onsets)
            'n_neurons' == n_neurons_valid (pass-through)
            'n_trials'  == number of unique values in trial_numbers

No golden fixtures are used here (all values are synthetic + analytically
derived); fixture-anchored (level 3, 1e-6-on-fixture-slice) equivalence tests
belong in a separate fixture-anchored suite once real experiment slices are
available, per contract_spec.yaml acceptance_criteria.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

np_demo1_analysis = pytest.importorskip(
    "np_demo1_analysis",
    reason="np_demo1_analysis.py not implemented yet (oracle written pre-implementation, Blueprint v2 P3)",
)


# --------------------------------------------------------------------------
# small local helpers (math only -- no dependency on the module under test)
# --------------------------------------------------------------------------
def _angle_diff(a: float, b: float) -> float:
    """Wrapped a-b in (-pi, pi]. Used to compare circular quantities without
    being tripped up by the +-pi branch cut."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


# ==========================================================================
# 1-4, 12-13: mrl_and_preferred_phase -- analytic circular-statistics oracle
# ==========================================================================
@pytest.mark.result_mrl
@pytest.mark.result_phase
def test_mrl_all_same_phase():
    """100 spikes all at phase 0.0: mean(exp(i*0)) == 1+0j exactly (to float
    eps) -> MRL == 1.0, preferred == 0.0 rad."""
    phases = np.zeros(100, dtype=np.float64)
    mrl, preferred = np_demo1_analysis.mrl_and_preferred_phase(phases)
    assert mrl == pytest.approx(1.0, abs=1e-9)
    assert preferred == pytest.approx(0.0, abs=1e-9)


@pytest.mark.result_mrl
def test_mrl_uniform_phases():
    """1000 spikes at the 1000th roots of unity (np.linspace(0,1,1000,
    endpoint=False)): sum_{k=0}^{N-1} exp(i*2*pi*k/N) == 0 EXACTLY (roots of
    unity sum to zero for N >= 2), so MRL is zero up to float round-off.
    Spec requires MRL < 0.05; we hold to a much tighter analytic bound."""
    phases = np.linspace(0.0, 1.0, 1000, endpoint=False)
    mrl, _ = np_demo1_analysis.mrl_and_preferred_phase(phases)
    assert mrl < 1e-6
    assert mrl < 0.05  # explicit spec bound, kept as an independent assertion


@pytest.mark.result_mrl
def test_mrl_opposite_phases():
    """50 spikes at phase 0.0 (unit vector +1) and 50 at phase 0.5 (unit
    vector -1): the resultant sum is exactly 0 -> MRL == 0 up to float
    round-off."""
    phases = np.concatenate([np.zeros(50), np.full(50, 0.5)])
    mrl, _ = np_demo1_analysis.mrl_and_preferred_phase(phases)
    assert mrl < 1e-9
    assert mrl < 0.05


@pytest.mark.result_mrl
@pytest.mark.result_phase
def test_mrl_known_direction():
    """100 spikes at phase 0.25 (== 2*pi*0.25 == pi/2 rad): exp(i*pi/2) ==
    1j exactly (to float eps) -> MRL == 1.0, preferred == pi/2 rad."""
    phases = np.full(100, 0.25, dtype=np.float64)
    mrl, preferred = np_demo1_analysis.mrl_and_preferred_phase(phases)
    assert preferred == pytest.approx(math.pi / 2, abs=1e-9)
    assert mrl == pytest.approx(1.0, abs=1e-9)


@pytest.mark.result_mrl
@pytest.mark.result_phase
def test_mrl_phase_boundary():
    """Phase 1.0 must wrap to the same circular position as phase 0.0
    (exp(i*2*pi*1.0) == exp(i*2*pi*0.0) == 1, since 2*pi is a full turn):
    50 spikes at 0.0 + 50 at 1.0 -> MRL == 1.0, preferred == 0.0 rad."""
    phases = np.concatenate([np.zeros(50), np.ones(50)])
    mrl, preferred = np_demo1_analysis.mrl_and_preferred_phase(phases)
    assert mrl == pytest.approx(1.0, abs=1e-9)
    assert preferred == pytest.approx(0.0, abs=1e-9)


@pytest.mark.result_mrl
@pytest.mark.result_phase
@pytest.mark.metamorphic
def test_metamorphic_rotation():
    """Metamorphic relation: rotating every phase by a constant delta
    (mod 1) must leave MRL unchanged (it is a rotation-invariant magnitude)
    and must shift the preferred phase by exactly delta*2*pi (mod 2*pi).
    Uses a deliberately non-uniform, non-degenerate phase set (clustered
    near 0.12) so the base MRL is neither 0 nor 1 and rotation is a
    non-trivial check."""
    base = np.array([0.10, 0.12, 0.15, 0.11, 0.13, 0.09, 0.14, 0.16, 0.10, 0.12])
    delta = 0.37  # arbitrary rotation, in 0-1 (cycle) units

    mrl0, pref0 = np_demo1_analysis.mrl_and_preferred_phase(base)
    rotated = (base + delta) % 1.0
    mrl1, pref1 = np_demo1_analysis.mrl_and_preferred_phase(rotated)

    assert 0.0 < mrl0 < 1.0, "base case must be non-degenerate for this relation to be meaningful"
    assert mrl1 == pytest.approx(mrl0, abs=1e-9)
    expected_pref1 = pref0 + delta * 2 * math.pi
    assert abs(_angle_diff(pref1, expected_pref1)) < 1e-6


# ==========================================================================
# 5: classify_ethanol -- condition-assignment oracle
# ==========================================================================
@pytest.mark.result_mrl
@pytest.mark.result_phase
def test_condition_assignment():
    """ETH_thr[0:50] = 0.11 (floor-clipped -> control, since 0.11 is NOT
    > 0.11); ETH_thr[50:100] = 0.5 (> 0.11 -> ethanol). LV_Fs = 100.0.

    Spike times -> LabView index via matlab_round(t*LV_Fs) (half-away-from-
    zero), no ties land exactly on .5 here so ordinary rounding intuition
    applies:
        t=0.01 -> idx round(1.0)=1   -> ETH_thr[1]=0.11  -> control
        t=0.45 -> idx round(45.0)=45 -> ETH_thr[45]=0.11 -> control
        t=0.50 -> idx round(50.0)=50 -> ETH_thr[50]=0.50 -> ethanol
        t=0.80 -> idx round(80.0)=80 -> ETH_thr[80]=0.50 -> ethanol
    """
    LV_Fs = 100.0
    ETH_thr = np.empty(100, dtype=np.float64)
    ETH_thr[0:50] = 0.11
    ETH_thr[50:100] = 0.5

    spike_times = np.array([0.01, 0.45, 0.50, 0.80])
    expected_is_ethanol = np.array([False, False, True, True])

    is_ethanol = np_demo1_analysis.classify_ethanol(spike_times, ETH_thr, LV_Fs)
    assert np.array_equal(np.asarray(is_ethanol, dtype=bool), expected_is_ethanol)


# ==========================================================================
# 6: valid_sniff_mask -- inclusion-filter oracle
# ==========================================================================
@pytest.mark.result_mrl
@pytest.mark.result_phase
def test_valid_sniff_filter():
    """20 spikes carry valid phase 0.5 (preferred == pi, MRL == 1.0 if
    correctly isolated); 30 spikes carry the -1 sentinel (outside any sniff
    cycle) and must be dropped BEFORE the circular mean is taken.

    This design is deliberately adversarial: phase -1.0, if a bug let it
    leak into the trig computation un-filtered, is periodic-equivalent to
    phase 0.0 (exp(i*2*pi*-1) == exp(i*2*pi*0) == 1), which would pull the
    resultant toward the *opposite* direction (MRL ~= 0.2, preferred ~= 0)
    from the true answer (MRL == 1.0, preferred == pi) -- so this test
    fails loudly if the -1 sentinel is not excluded pre-averaging.
    """
    valid_phase = 0.5
    n_valid, n_invalid = 20, 30
    spike_SNF_PH = np.concatenate([
        np.full(n_valid, valid_phase),
        np.full(n_invalid, -1.0),
    ])

    mask = np.asarray(np_demo1_analysis.valid_sniff_mask(spike_SNF_PH), dtype=bool)
    assert mask.sum() == n_valid
    assert mask[:n_valid].all()
    assert not mask[n_valid:].any()

    mrl, preferred = np_demo1_analysis.mrl_and_preferred_phase(spike_SNF_PH[mask])
    assert mrl == pytest.approx(1.0, abs=1e-9)
    assert abs(_angle_diff(preferred, math.pi)) < 1e-6


# ==========================================================================
# 7-9: unit_included -- exclusion-rule oracle
# ==========================================================================
@pytest.mark.result_mrl
@pytest.mark.result_qc_counts
def test_unit_exclusion_firing_rate():
    """Firing rate 50 spikes / 1000 s == 0.05 Hz < 0.1 Hz floor -> excluded,
    even though n_valid_sniff_spikes (5000) is far above the 50-spike floor.
    The rate criterion must dominate regardless of spike count."""
    included = np_demo1_analysis.unit_included(
        n_spikes_total=50, session_duration_s=1000.0, n_valid_sniff_spikes=5000,
    )
    assert included is False


@pytest.mark.result_mrl
@pytest.mark.result_qc_counts
def test_unit_exclusion_min_spikes():
    """Firing rate 1000 spikes / 100 s == 10 Hz (well above the 0.1 Hz
    floor), but only 49 valid-sniff spikes in this condition (< 50 floor)
    -> excluded."""
    included = np_demo1_analysis.unit_included(
        n_spikes_total=1000, session_duration_s=100.0, n_valid_sniff_spikes=49,
    )
    assert included is False


@pytest.mark.result_mrl
@pytest.mark.result_qc_counts
def test_unit_exclusion_min_spikes_boundary():
    """Identical to test_unit_exclusion_min_spikes except exactly 50
    valid-sniff spikes: boundary is >= 50, so this unit must be included."""
    included = np_demo1_analysis.unit_included(
        n_spikes_total=1000, session_duration_s=100.0, n_valid_sniff_spikes=50,
    )
    assert included is True


# ==========================================================================
# 10: paired_animal_test -- Wilcoxon signed-rank oracle (exact combinatorics)
# ==========================================================================
@pytest.mark.result_stat
def test_paired_test_direction():
    """6 animals, ethanol mean-MRL strictly greater than control mean-MRL in
    every animal (no ties among the 6 signed differences, none exactly
    zero). Under H0 (random sign flips), the n=6 exact Wilcoxon
    signed-rank null distribution places EVERY sign pattern (2**6 = 64 of
    them) as equally likely; the observed pattern (all positive) achieves
    the single most extreme rank-sum (W+ == n*(n+1)/2, the unique maximum),
    so:
        one-sided p = 1/2**n = 1/64
        two-sided p = 2/2**n = 2**(1-n) = 0.03125   (exact, no ties/zeros)
    and the rank-biserial effect size for a perfectly concordant sample is
    exactly 1.0, independent of the differences' magnitudes:
        r = (W+ - W-) / (W+ + W-) = (T - 0) / (T + 0) = 1.0   where T = n*(n+1)/2
    """
    ethanol = np.array([0.60, 0.55, 0.70, 0.62, 0.58, 0.65])
    control = np.array([0.30, 0.28, 0.35, 0.31, 0.29, 0.33])
    n = ethanol.size
    assert np.all(ethanol > control) and n == 6, "fixture sanity: all-positive, n=6"

    result = np_demo1_analysis.paired_animal_test(ethanol, control)

    expected_p_exact = 2 ** (1 - n)  # == 0.03125
    assert result["pvalue"] < 0.05
    assert result["pvalue"] == pytest.approx(expected_p_exact, abs=1e-3)
    assert result["effect_size"] == pytest.approx(1.0, abs=1e-9)
    assert result["direction"] == "ethanol>control"
    assert result["n_animals"] == n


# ==========================================================================
# 11: qc_counts -- exact-integer QC oracle
# ==========================================================================
@pytest.mark.result_qc_counts
def test_qc_sniff_count(mock_D):
    """n_sniffs must equal len(sniff_onsets) exactly (level-1 exact-integer
    contract); n_trials must equal the count of UNIQUE trial numbers, not
    the number of trial-tagged spikes/samples; n_neurons is a pass-through
    of the (separately-computed) valid-unit count."""
    sniff_onsets = mock_D["sniff_onsets"]
    trial_numbers = np.array([1, 1, 2, 3, 3, 3])  # 6 samples, 3 unique trials
    n_neurons_valid = 1

    counts = np_demo1_analysis.qc_counts(sniff_onsets, n_neurons_valid, trial_numbers)

    assert counts["n_sniffs"] == len(sniff_onsets) == 2
    assert counts["n_trials"] == 3
    assert counts["n_neurons"] == n_neurons_valid

"""
oracle_animal_test.py -- NP-DEMO-2 oracle for the paired animal-level test
============================================================================
Written by the ORACLE DESIGNER (Blueprint v2 P3) BEFORE np_demo2_analysis.py
exists. Targets repo_map.md New-Code-Required function:

    paired_animal_test(eth_means, ctrl_means) -> dict
        scipy.stats.wilcoxon(diffs, alternative='two-sided', method='exact',
        zero_method='wilcox') on paired (MRL_ethanol - MRL_control) per
        animal; effect size = matched-pairs rank-biserial r; exact p
        reported (contract statistical_model, RESULT-stat-animal).

scipy.stats.wilcoxon is an already-validated third-party statistics library
(not "the new code under test" -- np_demo2_analysis merely calls it), so it
is used directly here, in the SAME way the approved NP-DEMO-1 oracle used it,
as the independent ground-truth reference for p-value and statistic. What IS
under test is that np_demo2_analysis (a) calls it with the pinned exact
two-sided wilcox configuration, (b) derives the correctly-SIGNED
rank-biserial r (scipy's own `statistic` is the unsigned min(W+,W-) and
cannot recover the sign by itself), and (c) reports the correct qualitative
direction label.

Acceptance-level mapping (contract_spec.yaml, NP-DEMO-2, contract_version 1)
-----------------------------------------------------------------------------
    RESULT-stat-animal (level 3, statistical_decision: "same significance
        decision on full workload") -- this file is the pre-fixture,
        analytic-combinatorics acceptance oracle for that object.

Implementer note
-----------------
    # from np_demo2_analysis import paired_animal_test
    # TODO: uncomment when np_demo2_analysis.py exists, and DELETE the
    # inline reference implementation below.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats


# ---------------------------------------------------------------------------
import sys, os
_KERNEL_SW_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_software"
if _KERNEL_SW_DIR not in sys.path and os.path.isdir(_KERNEL_SW_DIR):
    sys.path.insert(0, _KERNEL_SW_DIR)
from np_demo2_analysis import paired_animal_test  # noqa: E402
# ---------------------------------------------------------------------------


# ===========================================================================
# (a) Perfect separation (eth always > ctrl) -> p < 0.05, r > 0
# ===========================================================================
def test_perfect_separation_ethanol_greater_gives_significant_positive_effect():
    """6 animals, ethanol strictly greater than control in every animal (no
    ties, none exactly zero). Under H0 (random sign flips), the n=6 exact
    Wilcoxon signed-rank null distribution places EVERY one of the 2**6=64
    equally-likely sign patterns; the observed all-positive pattern achieves
    the unique maximum rank-sum, so:
        two-sided p = 2 * (1/2**6) = 2**(1-6) = 0.03125  (exact, no ties)
    and the rank-biserial effect size for a perfectly concordant sample is
    exactly r = (W+ - W-)/(W+ + W-) = (T-0)/(T+0) = 1.0, independent of the
    differences' magnitudes."""
    ethanol = np.array([0.60, 0.55, 0.70, 0.62, 0.58, 0.65])
    control = np.array([0.30, 0.28, 0.35, 0.31, 0.29, 0.33])
    n = ethanol.size
    assert np.all(ethanol > control) and n == 6

    result = paired_animal_test(ethanol, control)

    expected_p_exact = 2 ** (1 - n)  # == 0.03125
    assert result["pvalue"] == pytest.approx(expected_p_exact, abs=1e-9)
    assert result["pvalue"] < 0.05
    assert result["effect_size"] == pytest.approx(1.0, abs=1e-9)
    assert result["direction"] == "ethanol>control"
    assert result["n_animals"] == n


# ===========================================================================
# (b) No effect (eth == ctrl exactly) -> p = 1.0, r = nan or 0
# ===========================================================================
def test_no_effect_identical_pairs_gives_p_one_and_undefined_effect():
    """6 animals with eth == ctrl exactly: every signed difference is zero.
    zero_method='wilcox' drops zero differences before ranking, leaving no
    data to test -- scipy.stats.wilcoxon (verified directly, this exact
    scipy version/config) returns statistic=0.0, pvalue=1.0 for an
    all-zero-difference input, and the rank-biserial r is undefined (no
    non-zero differences to rank) -> nan."""
    values = np.array([0.50, 0.40, 0.60, 0.55, 0.45, 0.50])
    result = paired_animal_test(values, values.copy())

    # independent scipy reference on the same (all-zero) diffs
    diffs = values - values.copy()
    ref = scipy.stats.wilcoxon(diffs, alternative="two-sided", method="exact",
                                zero_method="wilcox")

    assert result["pvalue"] == pytest.approx(float(ref.pvalue), abs=1e-12)
    assert result["pvalue"] == pytest.approx(1.0, abs=1e-9)
    assert math.isnan(result["effect_size"]) or result["effect_size"] == 0
    assert result["direction"] == "ns"


# ===========================================================================
# (c) n=1 animal -> scipy.stats.wilcoxon's actual (non-raising) behavior
# ===========================================================================
def test_single_animal_matches_scipy_wilcoxon_n1_behavior_exactly():
    """n=1: scipy.stats.wilcoxon(method='exact', zero_method='wilcox') on a
    single non-zero difference does NOT raise (verified directly against
    this scipy version): it returns statistic=0.0, pvalue=1.0 (there is only
    one possible sign pattern under the two-sided exact null with n=1, so no
    result can be more extreme than chance). paired_animal_test must
    reproduce this behavior exactly, not substitute a different convention
    (e.g. raising, or returning nan) for the n=1 edge case."""
    ethanol = np.array([0.65])
    control = np.array([0.60])
    diffs = ethanol - control

    ref = scipy.stats.wilcoxon(diffs, alternative="two-sided", method="exact",
                                zero_method="wilcox")

    result = paired_animal_test(ethanol, control)

    assert result["pvalue"] == pytest.approx(float(ref.pvalue), abs=1e-12)
    assert result["statistic"] == pytest.approx(float(ref.statistic), abs=1e-12)
    assert result["n_animals"] == 1


# ===========================================================================
# (d) Reference: scipy.stats.wilcoxon computed manually on the same diffs,
#     exact match of p, statistic, and rank-biserial r
# ===========================================================================
def test_matches_manual_scipy_reference_exactly_on_asymmetric_data():
    """A non-trivial, asymmetric (not perfectly concordant) 7-animal
    dataset. Independently compute scipy.stats.wilcoxon on the identical
    diffs array, and independently compute the rank-biserial r from its
    definition (signed rank-sum ratio on the non-zero differences), then
    require paired_animal_test to match BOTH exactly (not just up to
    direction/significance)."""
    ethanol = np.array([0.62, 0.40, 0.71, 0.55, 0.30, 0.66, 0.58])
    control = np.array([0.58, 0.45, 0.60, 0.50, 0.42, 0.61, 0.50])
    diffs = ethanol - control
    assert np.any(diffs < 0), "fixture sanity: must be non-monotone (asymmetric case)"

    ref = scipy.stats.wilcoxon(diffs, alternative="two-sided", method="exact",
                                zero_method="wilcox")
    nz = diffs[diffs != 0]
    ranks = scipy.stats.rankdata(np.abs(nz))
    w_plus = ranks[nz > 0].sum()
    w_minus = ranks[nz < 0].sum()
    expected_r = (w_plus - w_minus) / (w_plus + w_minus)

    result = paired_animal_test(ethanol, control)

    assert result["pvalue"] == pytest.approx(float(ref.pvalue), abs=1e-12)
    assert result["statistic"] == pytest.approx(float(ref.statistic), abs=1e-12)
    assert result["effect_size"] == pytest.approx(expected_r, abs=1e-12)


# ===========================================================================
# (e) Sign of effect: ethanol < control -> r < 0, direction = "control>ethanol"
# ===========================================================================
def test_control_greater_gives_significant_negative_effect_and_direction_label():
    """Same magnitudes as the perfect-separation case (a), but with the
    ethanol/control roles swapped: control strictly greater than ethanol in
    every animal. By the antisymmetry of the rank-biserial statistic under
    swapping which side is 'positive', r must flip sign to exactly -1.0
    (same |p| = 0.03125 by the same exact combinatorics), and the direction
    label must flip to 'control>ethanol'."""
    ethanol = np.array([0.30, 0.28, 0.35, 0.31, 0.29, 0.33])
    control = np.array([0.60, 0.55, 0.70, 0.62, 0.58, 0.65])
    n = ethanol.size
    assert np.all(ethanol < control) and n == 6

    result = paired_animal_test(ethanol, control)

    expected_p_exact = 2 ** (1 - n)
    assert result["pvalue"] == pytest.approx(expected_p_exact, abs=1e-9)
    assert result["effect_size"] == pytest.approx(-1.0, abs=1e-9)
    assert result["direction"] == "control>ethanol"


# ===========================================================================
# Metamorphic relation: antisymmetry under swapping ethanol <-> control
# ===========================================================================
def test_metamorphic_swap_ethanol_and_control_negates_effect_size_only():
    """Metamorphic relation: swapping the ethanol/control arguments must
    negate the rank-biserial effect size exactly, leave the p-value and
    |statistic| unchanged (the exact test is symmetric in the sign of the
    differences), and flip the direction label. Uses an n=8, mostly-
    concordant-but-not-perfect dataset (7 positive diffs of varying
    magnitude, 1 negative diff at the SMALLEST magnitude, i.e. rank 1):
    the exact null distribution's statistic = min(W+, W-) = 1 here
    (only the all-positive pattern, W-=0, and the single-smallest-rank-flip
    pattern, W-=1, are at least this extreme, 2 of 2**8=256 equally likely
    sign patterns), giving two-sided p = 2*2/256 = 0.015625 < 0.05 -- so
    this dataset is both significant AND not perfectly concordant (r < 1),
    the combination needed to observe the direction label actually flip
    under the swap (a perfectly concordant r=1.0 case would also flip, but
    would not distinguish 'flips because significant' from 'always flips
    regardless of significance')."""
    control = np.full(8, 0.50)
    diffs = np.array([0.12, 0.16, 0.15, 0.18, 0.20, 0.22, 0.25, -0.02])
    ethanol = control + diffs
    assert np.any(diffs < 0) and np.sum(diffs < 0) == 1, "fixture sanity: exactly one discordant diff"

    forward = paired_animal_test(ethanol, control)
    swapped = paired_animal_test(control, ethanol)

    assert forward["pvalue"] < 0.05, "fixture sanity: dataset must be significant"
    assert forward["pvalue"] == pytest.approx(swapped["pvalue"], abs=1e-12)
    assert forward["statistic"] == pytest.approx(swapped["statistic"], abs=1e-12)
    assert forward["effect_size"] == pytest.approx(-swapped["effect_size"], abs=1e-12)
    assert forward["direction"] != swapped["direction"]
    assert forward["direction"] == "ethanol>control"
    assert swapped["direction"] == "control>ethanol"

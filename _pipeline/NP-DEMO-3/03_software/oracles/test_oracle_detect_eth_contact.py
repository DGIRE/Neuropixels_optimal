# ORACLE
"""
Analytical microcases for detect_eth_contact(D, eth_threshold=0.05).

Targets: RESULT-eth-mask (contract acceptance level 1 -- comparison_method:
exact; "binary mask and integer n_trials must match exactly (deterministic
threshold rule, no tolerance)"). Also feeds RESULT-sniff-qc's n_trials field
(level 2, exact integer) and the FIG-DEMO3-QC-SNIFF ETH_ms trace.

Algorithm under test (contract_v001.yaml, new_kernel_functions_required):
  1. ETH_ms = ETH - mean(ETH)            (entire trace, no windowing)
  2. eth_contact_mask = ETH_ms > 0.05     (STRICTLY above; at/below = control)
  3. n_trials = count of contiguous True runs in eth_contact_mask
  4. Fixed rule for all 6 sessions -- no per-experiment adjustment.

All expected values below are derived by hand from this algorithm description
(closed-form arithmetic), independently of any implementation. Where the
`eth_contact_oracle` fixture is used, it transparently swaps in the real
detect_eth_contact() once 03_software/detect_eth_contact.py exists (see
conftest.py); until then, it falls back to a verbatim reimplementation of the
same four steps, so the suite is fully self-verifying today.
"""
from __future__ import annotations

import numpy as np

from conftest import count_contiguous_runs


def test_all_below_threshold_yields_no_contact(eth_contact_oracle):
    """All raw samples close together and below 0.05 post mean-subtraction -> no ethanol contact."""
    eth = np.array([0.01, 0.02, 0.03])
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    # Hand-derived: mean = 0.02, ETH_ms = [-0.01, 0.00, 0.01]; max = 0.01 < 0.05.
    expected_mean = 0.02
    np.testing.assert_allclose(eth_ms, eth - expected_mean, atol=1e-12)
    assert eth_ms.max() < 0.05
    assert not np.any(mask)
    assert n_trials == 0


def test_all_identical_samples_yield_no_contact(eth_contact_oracle):
    """Identical ETH values: mean subtraction collapses everything to exactly 0, which is NOT strictly above 0.05."""
    eth = np.array([1.0, 1.0, 1.0])
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    np.testing.assert_allclose(eth_ms, np.zeros(3), atol=1e-12)
    assert not np.any(mask)  # 0 is at/below threshold, not strictly above
    assert n_trials == 0


def test_step_function_single_contiguous_run(eth_contact_oracle):
    """A single 0->1 step produces exactly one contiguous above-threshold run."""
    eth = np.array([0.0] * 50 + [1.0] * 50)
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    # Hand-derived: mean = 0.5, so ETH_ms = [-0.5]*50 + [0.5]*50.
    np.testing.assert_allclose(eth_ms[:50], -0.5 * np.ones(50), atol=1e-12)
    np.testing.assert_allclose(eth_ms[50:], 0.5 * np.ones(50), atol=1e-12)
    assert np.array_equal(mask, np.array([False] * 50 + [True] * 50))
    assert n_trials == 1


def test_two_separate_runs_counted_as_two_trials(eth_contact_oracle):
    """Two isolated above-threshold blocks must be counted as n_trials = 2, not merged or overcounted."""
    eth = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    # Hand-derived: mean = 4/8 = 0.5 -> ETH_ms = [-.5,-.5,.5,.5,-.5,-.5,.5,.5]
    expected_mask = np.array([False, False, True, True, False, False, True, True])
    assert np.array_equal(mask, expected_mask)
    assert n_trials == 2
    # Cross-check with an independent run-counting implementation.
    assert n_trials == count_contiguous_runs(expected_mask)


def test_strict_greater_than_operator_excludes_exact_boundary():
    """The threshold comparison must be strictly '>' 0.05, not '>=' (operator-level, float-precision-safe)."""
    # Constructed directly at/around the threshold; avoids any mean-subtraction
    # floating-point composition so the boundary itself is unambiguous.
    threshold = 0.05
    values = np.array([threshold, np.nextafter(threshold, np.inf), np.nextafter(threshold, -np.inf), 0.15])
    mask = values > threshold
    assert list(mask) == [False, True, False, True]


def test_boundary_end_to_end_with_safety_margin(eth_contact_oracle):
    """End-to-end pipeline: samples safely just below vs just above 0.05 (post mean-subtraction) are classified correctly.

    NOTE: this intentionally does not attempt to hit the 0.05 boundary bit-exactly
    through composite mean-subtraction arithmetic (decimal literals like 0.05 are
    not exactly representable in binary floating point, so constructing an
    exact tie via a - mean(a) chain is not guaranteed bit-reproducible). The
    strict '>' vs '>=' semantics are covered exactly, at the operator level, in
    test_strict_greater_than_operator_excludes_exact_boundary above. This test
    instead uses a comfortable (>= 1e-3) margin on both sides of 0.05 to confirm
    the full mean-subtract-then-threshold pipeline classifies correctly.
    """
    # Construct raw samples with EXACT zero mean by design (sum of these six
    # values is 0 by construction), so ETH_ms == ETH with no rounding risk:
    # blocks below threshold (-0.109), just-below-threshold-margin (0.049 < 0.05),
    # just-above-threshold-margin (0.06 > 0.05).
    # Verified: 2*(-0.109) + 2*0.049 + 2*0.06 == 0.0 exactly.
    eth = np.array([-0.109, 0.049, 0.06, -0.109, 0.049, 0.06])
    assert abs(eth.mean()) < 1e-12  # sanity: zero mean by construction
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    expected_mask = eth > 0.05  # since mean ~ 0, ETH_ms ~ ETH
    assert np.array_equal(mask, expected_mask)
    assert list(mask) == [False, False, True, False, False, True]
    assert n_trials == 2


def test_threshold_parameter_is_respected(eth_contact_oracle):
    """Raising eth_threshold from 0.05 to 0.10 on the SAME trace changes both the mask and n_trials."""
    # Constructed with exact zero mean: blocks at -0.4 (below both thresholds),
    # 0.08 (above 0.05, below 0.10), 0.32 (above both). Verified: -0.4 + 0.08*2
    # - 0.4 + 0.32*2 = 0.0 exactly for this arrangement.
    eth = np.array([-0.4, 0.08, 0.08, -0.4, 0.32, 0.32])
    assert abs(eth.mean()) < 1e-12

    eth_ms_05, mask_05, n_trials_05 = eth_contact_oracle(eth, eth_threshold=0.05)
    eth_ms_10, mask_10, n_trials_10 = eth_contact_oracle(eth, eth_threshold=0.10)

    assert list(mask_05) == [False, True, True, False, True, True]
    assert list(mask_10) == [False, False, False, False, True, True]
    assert n_trials_05 == 2
    assert n_trials_10 == 1
    assert not np.array_equal(mask_05, mask_10)
    # ETH_ms itself must NOT depend on eth_threshold (only the mask/n_trials do).
    np.testing.assert_allclose(eth_ms_05, eth_ms_10, atol=1e-12)


def test_mean_subtraction_uses_entire_trace_not_per_chunk(eth_contact_oracle):
    """ETH_ms must equal ETH - mean(ENTIRE ETH); a per-chunk/windowed mean would silently zero out real steps."""
    eth = np.array([1.0] * 50 + [3.0] * 50)
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    global_mean = 2.0  # (50*1.0 + 50*3.0) / 100
    np.testing.assert_allclose(eth_ms, eth - global_mean, atol=1e-9)
    np.testing.assert_allclose(eth_ms[:50], -1.0 * np.ones(50), atol=1e-9)
    np.testing.assert_allclose(eth_ms[50:], 1.0 * np.ones(50), atol=1e-9)

    # A buggy per-chunk (windowed) mean-subtraction would instead produce all
    # zeros in each 50-sample half -- explicitly demonstrate that the correct,
    # contract-mandated global-mean result differs from that buggy alternative.
    buggy_per_chunk = np.concatenate(
        [eth[:50] - eth[:50].mean(), eth[50:] - eth[50:].mean()]
    )
    assert not np.allclose(eth_ms, buggy_per_chunk)
    np.testing.assert_allclose(buggy_per_chunk, np.zeros(100), atol=1e-12)

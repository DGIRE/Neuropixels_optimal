# ORACLE
"""
Analytical microcases for detect_eth_contact(D, eth_threshold=0.05), as reused
VERBATIM by NP-DEMO-4 (contract_v001.yaml, required_validated_functions;
prohibited_changes: "Modifying detect_eth_contact.py (NP-DEMO-3 kernel) in any
way -- threshold, mean-subtraction, or contiguous-run logic.").

Targets:
  - RESULT-eth-mask       (level 1, comparison_method: exact -- "Binary contact
    mask and n_trials_eth must match NP-DEMO-3 validated outputs exactly for
    the same 6 sessions; deterministic threshold; zero tolerance.")
  - RESULT-methods-table  (level 1, exact -- n_trials_eth is an exact integer)
  - Feeds the CON-003 "first discrete ETH contact event per trial" step and
    the FIG-DEMO4-3-ETH raw-ETH heatmap boundary logic indirectly (n_trials_eth
    and the contact mask gate which trials qualify).

Algorithm under test (contract_v001.yaml preprocessing / inclusion_exclusion,
copied verbatim from NP-DEMO-3's already-validated
_pipeline/NP-DEMO-3/03_software/detect_eth_contact.py, read 2026-07-23):
  1. ETH_ms = ETH - mean(ETH)              (mean of the ENTIRE trace, no windowing)
  2. eth_contact_mask = ETH_ms > eth_threshold   (STRICTLY above; at/below = control)
  3. n_trials = count of discrete contiguous True runs in eth_contact_mask
  4. Fixed threshold (0.05) for all 6 sessions -- no per-experiment adjustment.

Design notes (Blueprint v2 P3 -- oracle design phase, no implementation run):
  All expected values below are derived by hand from this four-step
  description (closed-form arithmetic on small, hand-constructed arrays),
  independently of any implementation. This file:
    (a) imports and exercises the REAL, already-validated
        _pipeline/NP-DEMO-3/03_software/detect_eth_contact.py directly (per
        task instructions -- this module already exists and is validated;
        using it here is "consulting a validated reference", not "running
        the new code"), and
    (b) once _pipeline/NP-DEMO-4/03_software/detect_eth_contact.py exists
        (the contractually-required verbatim copy), automatically prefers
        THAT copy instead, and additionally asserts the copy is byte-for-byte
        identical to the NP-DEMO-3 original (PROH-001 / prohibited_changes
        regression guard).
  A local, independently-written contiguous-run counter (via itertools.groupby,
  deliberately NOT the same code shape as the module's own rising-edge
  implementation) is used as a cross-check oracle for n_trials, so this suite
  does not just re-assert the module's own internal logic against itself.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

_THIS_FILE = Path(__file__).resolve()
_DEMO4_SOFTWARE_DIR = _THIS_FILE.parents[1]          # .../NP-DEMO-4/03_software
_PIPELINE_DIR = _THIS_FILE.parents[3]                # .../_pipeline
_DEMO3_SOFTWARE_DIR = _PIPELINE_DIR / "NP-DEMO-3" / "03_software"
_DEMO3_SOURCE = _DEMO3_SOFTWARE_DIR / "detect_eth_contact.py"
_DEMO4_SOURCE = _DEMO4_SOFTWARE_DIR / "detect_eth_contact.py"


def _load_detect_eth_contact():
    """Import detect_eth_contact, preferring the NP-DEMO-4 local verbatim copy
    once it exists; falling back to the NP-DEMO-3 original (which already
    exists today). Returns (function, source_label)."""
    if _DEMO4_SOURCE.is_file():
        path = str(_DEMO4_SOFTWARE_DIR)
        if path not in sys.path:
            sys.path.insert(0, path)
        sys.modules.pop("detect_eth_contact", None)
        from detect_eth_contact import detect_eth_contact as fn  # type: ignore
        return fn, "NP-DEMO-4 local copy"

    if not _DEMO3_SOURCE.is_file():
        pytest.skip(
            "Neither NP-DEMO-4's local copy nor the NP-DEMO-3 original "
            "detect_eth_contact.py could be found; cannot exercise the "
            "real implementation."
        )
    path = str(_DEMO3_SOFTWARE_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    sys.modules.pop("detect_eth_contact", None)
    from detect_eth_contact import detect_eth_contact as fn  # type: ignore
    return fn, "NP-DEMO-3 original (pre-copy fallback)"


def _independent_count_contiguous_runs(mask: np.ndarray) -> int:
    """Independent (groupby-based, NOT rising-edge-diff-based) reference count
    of contiguous True runs -- a deliberately differently-shaped
    reimplementation used purely as a cross-check, not copied from the
    module under test."""
    mask = list(bool(v) for v in np.asarray(mask, dtype=bool).ravel())
    return sum(1 for key, _ in itertools.groupby(mask) if key)


@pytest.fixture(scope="module")
def eth_contact_oracle():
    """Callable(eth_array, eth_threshold=0.05) -> (eth_ms, mask, n_trials),
    backed by the REAL detect_eth_contact implementation (see _load_detect_eth_contact)."""
    fn, label = _load_detect_eth_contact()
    print(f"[oracle] using detect_eth_contact from: {label}")

    def _call(eth_array, eth_threshold: float = 0.05):
        D = {"ETH": np.asarray(eth_array, dtype=np.float64)}
        out = fn(D, eth_threshold=eth_threshold)
        return (
            np.asarray(out["ETH_ms"]),
            np.asarray(out["eth_contact_mask"], dtype=bool),
            int(out["n_trials"]),
        )

    return _call


# ---------------------------------------------------------------------------
# Microcase 1: all-zeros ETH
# ---------------------------------------------------------------------------
def test_all_zeros_eth_yields_no_contact(eth_contact_oracle):
    """All-zeros input: mean = 0, so ETH_ms is all zeros, which is NOT
    strictly above 0.05 -> mask all False, n_trials = 0."""
    eth = np.zeros(200)
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    np.testing.assert_allclose(eth_ms, np.zeros(200), atol=1e-12)
    assert not np.any(mask)
    assert n_trials == 0


# ---------------------------------------------------------------------------
# Microcase 2: constant positive ETH
# ---------------------------------------------------------------------------
def test_constant_positive_eth_yields_no_contact(eth_contact_oracle):
    """Constant 0.1 everywhere: mean = 0.1, so ETH_ms collapses to exactly 0
    (well below the 0.05 threshold) -- a constant nonzero baseline must never
    itself register as ethanol contact."""
    eth = np.full(150, 0.1)
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    np.testing.assert_allclose(eth_ms, np.zeros(150), atol=1e-12)
    assert not np.any(mask)
    assert n_trials == 0


# ---------------------------------------------------------------------------
# Microcase 3: single step -> exactly one contiguous run
# ---------------------------------------------------------------------------
def test_single_step_pulse_yields_one_trial(eth_contact_oracle):
    """eth = [0]*50 + [0.2]*20 + [0]*50 (n=120).
    Hand-derived: mean = (20*0.2)/120 = 4/120 = 0.0333333...
    ETH_ms during the pulse = 0.2 - 0.033333 = 0.166667 > 0.05  -> contact.
    ETH_ms during the zero blocks = -0.033333 < 0.05            -> no contact.
    Exactly one contiguous above-threshold run -> n_trials = 1."""
    eth = np.array([0.0] * 50 + [0.2] * 20 + [0.0] * 50)
    expected_mean = 4.0 / 120.0
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    np.testing.assert_allclose(eth.mean(), expected_mean, atol=1e-12)
    np.testing.assert_allclose(eth_ms[:50], -expected_mean * np.ones(50), atol=1e-9)
    np.testing.assert_allclose(eth_ms[50:70], (0.2 - expected_mean) * np.ones(20), atol=1e-9)
    np.testing.assert_allclose(eth_ms[70:], -expected_mean * np.ones(50), atol=1e-9)

    expected_mask = np.array([False] * 50 + [True] * 20 + [False] * 50)
    assert np.array_equal(mask, expected_mask)
    assert n_trials == 1
    assert n_trials == _independent_count_contiguous_runs(expected_mask)


# ---------------------------------------------------------------------------
# Microcase 4: two separated pulses -> n_trials = 2
# ---------------------------------------------------------------------------
def test_two_separated_pulses_yield_two_trials(eth_contact_oracle):
    """eth = [0,0,1,1,0,0,1,1]: mean = 4/8 = 0.5 exactly.
    ETH_ms = [-.5,-.5,.5,.5,-.5,-.5,.5,.5] -> two isolated above-threshold
    blocks, must be counted as n_trials = 2 (never merged, never overcounted)."""
    eth = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    expected_mask = np.array([False, False, True, True, False, False, True, True])
    np.testing.assert_allclose(eth_ms, eth - 0.5, atol=1e-12)
    assert np.array_equal(mask, expected_mask)
    assert n_trials == 2
    assert n_trials == _independent_count_contiguous_runs(expected_mask)


# ---------------------------------------------------------------------------
# Microcase 5: mean-subtracted value exactly AT threshold -> NOT counted
# ---------------------------------------------------------------------------
def test_exactly_at_threshold_after_mean_subtraction_is_not_contact(eth_contact_oracle):
    """eth = [-0.05, 0.05]: sum = 0 exactly (representable without rounding
    since -0.05 + 0.05 cancels bit-for-bit for IEEE-754 doubles), so mean = 0
    and ETH_ms == eth exactly: [-0.05, 0.05]. The second sample's
    mean-subtracted value lands EXACTLY on the 0.05 threshold; since the rule
    is strictly '>' (not '>='), it must NOT be counted as contact."""
    eth = np.array([-0.05, 0.05])
    assert eth.mean() == 0.0  # exact float cancellation, verified at oracle-design time
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    np.testing.assert_allclose(eth_ms, eth, atol=1e-15)
    assert list(mask) == [False, False]
    assert n_trials == 0

    # Operator-level confirmation (bit-precision-safe, independent of any
    # float composition): a value the tiniest amount above 0.05 IS counted,
    # exactly-at and the tiniest amount below are NOT.
    threshold = 0.05
    probe = np.array([np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)])
    assert list(probe > threshold) == [False, False, True]


# ---------------------------------------------------------------------------
# Microcase 6: "real-world" synthetic session with three discrete contacts
# ---------------------------------------------------------------------------
def test_synthetic_session_with_three_contacts(eth_contact_oracle):
    """Synthetic session shaped like a real ETH trace: low flat baseline with
    three well-separated above-threshold bumps of known height/width, embedded
    so the whole-trace mean is known by construction (sum of pieces / n).
    Cross-checked with an independently-implemented (groupby-based) run counter."""
    baseline = np.zeros(500)
    bump = np.full(30, 0.30)  # well above any plausible post-mean-subtraction threshold
    eth = baseline.copy()
    starts = [100, 250, 400]
    for s in starts:
        eth[s:s + 30] = 0.30

    n = eth.size
    expected_mean = (3 * 30 * 0.30) / n
    eth_ms, mask, n_trials = eth_contact_oracle(eth)

    np.testing.assert_allclose(eth_ms, eth - expected_mean, atol=1e-9)
    assert n_trials == 3
    # Every bump sample must clear threshold, every baseline sample must not.
    expected_mask = np.zeros(n, dtype=bool)
    for s in starts:
        expected_mask[s:s + 30] = True
    assert np.array_equal(mask, expected_mask)
    assert n_trials == _independent_count_contiguous_runs(expected_mask)


# ---------------------------------------------------------------------------
# Threshold parameter respected (regression guard against hardcoding 0.05)
# ---------------------------------------------------------------------------
def test_threshold_parameter_is_respected(eth_contact_oracle):
    """Raising eth_threshold from 0.05 to 0.10 on the SAME trace changes both
    the mask and n_trials (constructed with exact zero mean by design:
    -0.4 + 0.08*2 - 0.4 + 0.32*2 == 0.0 exactly)."""
    eth = np.array([-0.4, 0.08, 0.08, -0.4, 0.32, 0.32])
    assert abs(eth.mean()) < 1e-12

    _, mask_05, n_05 = eth_contact_oracle(eth, eth_threshold=0.05)
    _, mask_10, n_10 = eth_contact_oracle(eth, eth_threshold=0.10)

    assert list(mask_05) == [False, True, True, False, True, True]
    assert list(mask_10) == [False, False, False, False, True, True]
    assert n_05 == 2
    assert n_10 == 1


# ---------------------------------------------------------------------------
# PROH-001 regression guard: the NP-DEMO-4 copy (once present) must be
# byte-for-byte identical to the NP-DEMO-3 validated original.
# ---------------------------------------------------------------------------
def test_demo4_copy_is_byte_identical_to_demo3_original_when_present():
    """contract_v001.yaml prohibited_changes: 'Modifying detect_eth_contact.py
    (NP-DEMO-3 kernel) in any way -- threshold, mean-subtraction, or
    contiguous-run logic.' Once _pipeline/NP-DEMO-4/03_software/detect_eth_contact.py
    is created (contract-required verbatim copy), it must match the NP-DEMO-3
    original byte-for-byte. Skipped (not failed) until the copy exists --
    this is a regression guard for D7 implementation, not a pre-implementation
    requirement."""
    if not _DEMO4_SOURCE.is_file():
        pytest.skip("NP-DEMO-4 detect_eth_contact.py copy does not exist yet.")
    if not _DEMO3_SOURCE.is_file():
        pytest.skip("NP-DEMO-3 original detect_eth_contact.py not found.")
    demo4_text = _DEMO4_SOURCE.read_text(encoding="utf-8")
    demo3_text = _DEMO3_SOURCE.read_text(encoding="utf-8")
    assert demo4_text == demo3_text, (
        "NP-DEMO-4's detect_eth_contact.py has diverged from the NP-DEMO-3 "
        "validated original -- PROH-001 violation (must be reused verbatim)."
    )

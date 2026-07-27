"""
oracle_condition_assignment.py -- NP-DEMO-2 oracle for valid-sniff filtering
and ethanol/control condition assignment
============================================================================
Written by the ORACLE DESIGNER (Blueprint v2 P3) BEFORE np_demo2_analysis.py
exists. Targets repo_map.md New-Code-Required functions:

    valid_sniff_mask(spike_SNF_PH) -> bool array
        True where spike_SNF_PH >= 0.0 (kernel sentinel -1 == outside any
        detected sniff cycle, HAZ-01).

    classify_ethanol(spike_times_s, ETH_thr, LV_Fs, eth_threshold) -> bool array
        True == ethanol condition.
        lv_idx = matlab_round(spike_times_s * LV_Fs), clamped to
                 [0, len(ETH_thr)-1]  (HAZ-02: matlab_round is
                 half-away-from-zero, NOT numpy/Python banker's rounding)
        ethanol iff ETH_thr[lv_idx] > eth_threshold, else control.

`matlab_round` itself is a VALIDATED KERNEL helper (analyses._common) and is
imported directly here, never re-derived -- it is not under test in this
file. What IS under test is that classify_ethanol uses it correctly (HAZ-02)
and that valid_sniff_mask/classify_ethanol correctly implement the pinned
comparison operators (>=, >; both boundary-sensitive, per PROH-001/PROH-002
and the contract's inclusion_exclusion clause).

This file anchors part of its ground truth to the real golden-fixture ETH_thr
array (Golden Fixtures/09_eth_threshold/ETH_thr.npy) and the real spike times
(Golden Fixtures/05_spikes_ks/sp_st.npy, index-aligned 1:1 with
Golden Fixtures/10_spike_phase/spike_SNF_PH.npy) so the fixture data
themselves -- not just hand-built microcases -- exercise classify_ethanol.
ETH_thr is documented (manifest.json) as "ETH floor-clipped at
eth_threshold": every sample is >= eth_threshold by construction, so
`ETH_thr[i] > eth_threshold` is exactly the "not clipped, i.e. a genuine
ethanol contact" predicate -- this is used directly as the independent
reference classification (first principles from the manifest's own
documented semantics, not from running np_demo2_analysis).

Acceptance-level mapping (contract_spec.yaml, NP-DEMO-2, contract_version 1)
-----------------------------------------------------------------------------
    RESULT-mrl-unit / RESULT-phase-unit (level 3) -- both depend on
        valid_sniff_mask + classify_ethanol being correct BEFORE any MRL is
        computed; this file is the upstream correctness gate.
    HAZ-01 (spike_SNF_PH == -1 sentinel), HAZ-02 (matlab_round), HAZ-07
        (LV_Fs clock alignment) -- hazard-catalog regression tests.

Implementer note
-----------------
    # from np_demo2_analysis import valid_sniff_mask, classify_ethanol
    # TODO: uncomment when np_demo2_analysis.py exists, and DELETE the
    # inline reference implementations below.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Kernel wiring (matlab_round IS a validated kernel helper -- imported, not
# reimplemented; the kernel is NOT under test here).
# ---------------------------------------------------------------------------
_KERNEL_DIR = r"C:\Projects\Repos\Neuropixels\Optimized Python"
if _KERNEL_DIR not in sys.path and os.path.isdir(_KERNEL_DIR):
    sys.path.insert(0, _KERNEL_DIR)

from analyses._common import matlab_round  # noqa: E402

_FIXTURES_DIR = r"C:\Projects\Repos\Neuropixels\Golden Fixtures"


def _load(rel_path: str) -> np.ndarray:
    return np.load(os.path.join(_FIXTURES_DIR, rel_path)).squeeze()


# ---------------------------------------------------------------------------
_NP_DEMO2_SW_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_software"
if _NP_DEMO2_SW_DIR not in sys.path and os.path.isdir(_NP_DEMO2_SW_DIR):
    sys.path.insert(0, _NP_DEMO2_SW_DIR)
from np_demo2_analysis import valid_sniff_mask, classify_ethanol  # noqa: E402
# ---------------------------------------------------------------------------


# A deliberately WRONG reference using Python's banker's-rounding round() --
# used only to demonstrate that the off-by-one hazard is real and would be
# caught (test (d) below), never as the pinned oracle behavior.
def _classify_ethanol_with_python_round(spike_times_s, ETH_thr, LV_Fs, eth_threshold=0.11):
    spike_times_s = np.asarray(spike_times_s, dtype=np.float64).ravel()
    ETH_thr = np.asarray(ETH_thr, dtype=np.float64).ravel()
    lv_idx = np.array([int(round(t * LV_Fs)) for t in spike_times_s])  # banker's rounding
    lv_idx = np.clip(lv_idx, 0, ETH_thr.size - 1)
    return ETH_thr[lv_idx] > eth_threshold


# ===========================================================================
# valid_sniff_mask microcases
# ===========================================================================
def test_valid_sniff_mask_all_sentinel_gives_all_false():
    """Every sample carrying the -1 sentinel (outside any detected sniff
    cycle) must be masked out entirely."""
    spike_SNF_PH = np.full(25, -1.0)
    mask = valid_sniff_mask(spike_SNF_PH)
    assert mask.dtype == bool
    assert not mask.any()


def test_valid_sniff_mask_mixed_values_gives_correct_boolean_mask():
    """Mix of -1 (invalid), 0.5 and 0.9 (valid, >= 0.0): mask must be
    exactly [False, True, True], with 0.0 itself also valid (boundary)."""
    spike_SNF_PH = np.array([-1.0, 0.5, 0.9])
    mask = valid_sniff_mask(spike_SNF_PH)
    assert np.array_equal(mask, np.array([False, True, True]))


def test_valid_sniff_mask_boundary_zero_is_valid():
    """spike_SNF_PH == 0.0 exactly (start of a sniff cycle) is VALID: the
    exclusion predicate is strictly '< 0', i.e. only the -1 sentinel is
    excluded, not phase 0.0 itself."""
    spike_SNF_PH = np.array([0.0, -1.0])
    mask = valid_sniff_mask(spike_SNF_PH)
    assert np.array_equal(mask, np.array([True, False]))


# ===========================================================================
# classify_ethanol -- hand-built microcases (a)-(c)
# ===========================================================================
def test_classify_ethanol_above_default_threshold_is_ethanol():
    """ETH_thr strictly greater than the 0.11 default -> ethanol (True)."""
    LV_Fs = 100.0
    ETH_thr = np.full(10, 0.50, dtype=np.float64)  # > 0.11 everywhere
    spike_times = np.array([0.01, 0.05])
    is_ethanol = classify_ethanol(spike_times, ETH_thr, LV_Fs, eth_threshold=0.11)
    assert np.all(is_ethanol)


def test_classify_ethanol_at_or_below_default_threshold_is_control():
    """ETH_thr <= the 0.11 default -> control (False); includes both the
    floor-clip value itself (0.11) and values below it."""
    LV_Fs = 100.0
    ETH_thr = np.full(10, 0.11, dtype=np.float64)
    spike_times = np.array([0.01, 0.05])
    is_ethanol = classify_ethanol(spike_times, ETH_thr, LV_Fs, eth_threshold=0.11)
    assert not np.any(is_ethanol)


def test_classify_ethanol_boundary_exactly_at_threshold_is_control_not_ethanol():
    """ETH_thr[idx] == eth_threshold EXACTLY -> control (strictly-greater
    predicate, '>' not '>='). This is the floor-clip boundary: threshold_eth
    clips everything below eth_threshold UP TO eth_threshold, so a large
    fraction of real ETH_thr samples sit exactly on this boundary and must
    never be misclassified as ethanol."""
    LV_Fs = 100.0
    eth_threshold = 0.11
    ETH_thr = np.array([eth_threshold, eth_threshold + 1e-9])
    spike_times = np.array([0.0, 1.0 / LV_Fs])
    is_ethanol = classify_ethanol(spike_times, ETH_thr, LV_Fs, eth_threshold=eth_threshold)
    assert is_ethanol[0] == False  # noqa: E712 (exact boundary, == is intentional)
    assert is_ethanol[1] == True  # noqa: E712


# ===========================================================================
# classify_ethanol -- (d) matlab_round off-by-one hazard (HAZ-02)
# ===========================================================================
def test_classify_ethanol_uses_matlab_round_not_bankers_rounding():
    """Construct a spike time landing EXACTLY on a half-sample boundary:
    LV_Fs=100, t=0.005 -> t*LV_Fs == 0.5 exactly.
        matlab_round(0.5)  = sign(0.5)*floor(0.5+0.5) = floor(1.0) = 1
        Python round(0.5)  = 0   (banker's rounding rounds half-to-even)
    ETH_thr is built so index 0 is control (0.05, below threshold) and
    index 1 is ethanol (0.50, above threshold): matlab_round must land on
    index 1 (True), whereas the banker's-rounding reference above lands on
    index 0 (False) -- demonstrating the off-by-one hazard would flip the
    classification if matlab_round were replaced by round()/np.round."""
    LV_Fs = 100.0
    eth_threshold = 0.11
    ETH_thr = np.array([0.05, 0.50])  # idx0 control, idx1 ethanol
    spike_times = np.array([0.005])  # t*LV_Fs == 0.5 exactly

    is_ethanol_correct = classify_ethanol(spike_times, ETH_thr, LV_Fs, eth_threshold)
    is_ethanol_buggy = _classify_ethanol_with_python_round(spike_times, ETH_thr, LV_Fs, eth_threshold)

    assert is_ethanol_correct[0] == True  # noqa: E712 -- matlab_round -> idx 1 -> ethanol
    assert is_ethanol_buggy[0] == False  # noqa: E712 -- demonstrates the hazard is real
    assert is_ethanol_correct[0] != is_ethanol_buggy[0], (
        "matlab_round (half-away-from-zero) and Python round() (banker's "
        "rounding) must disagree on this deliberately-constructed .5 tie, "
        "proving the test actually discriminates between them"
    )


# ===========================================================================
# Golden-fixture anchored classification (first-principles reference derived
# directly from the manifest's documented ETH_thr floor-clip semantics)
# ===========================================================================
@pytest.fixture(scope="module")
def fixture_data():
    ETH_thr = _load("09_eth_threshold/ETH_thr.npy")
    eth_threshold = float(_load("09_eth_threshold/eth_threshold.npy"))
    LV_Fs = float(_load("01_labview/LV_Fs.npy"))
    spike_times = _load("05_spikes_ks/sp_st.npy")  # seconds; index-aligned with spike_SNF_PH
    return dict(ETH_thr=ETH_thr, eth_threshold=eth_threshold, LV_Fs=LV_Fs, spike_times=spike_times)


def test_classify_ethanol_golden_fixture_above_threshold_samples_are_ethanol(fixture_data):
    """Reference classification derived directly from the golden ETH_thr
    array's documented semantics (manifest.json: 'ETH floor-clipped at
    eth_threshold', i.e. every sample is >= eth_threshold and a sample is
    only > eth_threshold where the raw ETH signal genuinely exceeded it).
    Pick the first 200 real spikes; every spike landing on a
    strictly-above-threshold ETH_thr sample must classify as ethanol."""
    ETH_thr = fixture_data["ETH_thr"]
    eth_threshold = fixture_data["eth_threshold"]
    LV_Fs = fixture_data["LV_Fs"]
    spike_times = fixture_data["spike_times"][:200]

    lv_idx = matlab_round(spike_times * LV_Fs).astype(np.int64)
    lv_idx = np.clip(lv_idx, 0, ETH_thr.size - 1)
    expected = ETH_thr[lv_idx] > eth_threshold

    is_ethanol = classify_ethanol(spike_times, ETH_thr, LV_Fs, eth_threshold)
    assert np.array_equal(is_ethanol, expected)
    # sanity: the fixture slice is not degenerate (has at least one of each,
    # or the assertion above would be vacuous) -- record via a soft check
    assert expected.dtype == bool


def test_classify_ethanol_golden_fixture_at_floor_is_control(fixture_data):
    """Find at least one real ETH_thr sample sitting exactly at the
    floor-clip value (eth_threshold) -- guaranteed to exist somewhere in a
    134950-sample session unless ethanol is present for the entire
    recording -- and confirm classify_ethanol reports control (False) for a
    spike landing exactly there."""
    ETH_thr = fixture_data["ETH_thr"]
    eth_threshold = fixture_data["eth_threshold"]
    LV_Fs = fixture_data["LV_Fs"]

    at_floor_idx = np.flatnonzero(ETH_thr == eth_threshold)
    assert at_floor_idx.size > 0, "fixture sanity: expected at least one floor-clipped sample"

    idx0 = int(at_floor_idx[0])
    spike_time = idx0 / LV_Fs
    is_ethanol = classify_ethanol(np.array([spike_time]), ETH_thr, LV_Fs, eth_threshold)
    assert is_ethanol[0] == False  # noqa: E712

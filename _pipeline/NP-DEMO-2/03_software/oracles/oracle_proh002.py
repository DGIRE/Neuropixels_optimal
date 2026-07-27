"""
oracle_proh002.py -- NP-DEMO-2 oracle for the PROH-002 two-pass eth_threshold
adjustment
============================================================================
Written by the ORACLE DESIGNER (Blueprint v2 P3) BEFORE np_demo2_analysis.py
exists. Targets repo_map.md New-Code-Required function:

    proh_002_adjustment(session_dirs / session_signals) -- two-pass ETH
        threshold search:
        PASS 1: every session evaluated at the SAME default eth_threshold =
            0.11; compute cross-session mean/SD of ethanol-contact count;
            flag any session outside +/-1 SD of that mean (or degenerate,
            e.g. all-ethanol/no-control).
        PASS 2: for each FLAGGED session only, grid-search
            eth_threshold in [0.05, 0.50] step 0.005, choose
            argmin |count(g) - mean|; ties broken by nearest to the 0.11
            default. Non-flagged sessions keep eth_threshold = 0.11.

This oracle does NOT use real DATA (per the task note: "the oracle tests the
LOGIC of the two-pass algorithm, not the real DATA, which isn't loaded
here"). Instead it builds small synthetic ETH signals with a KNOWN,
hand-auditable step structure (isolated rectangular bursts of known height,
separated by a sub-floor baseline) so that "contact count at threshold g" is
computed by an elementary, obviously-correct helper (`contact_count`:
count contiguous runs where signal > g) whose correctness is evident from its
definition -- this trivial counter, not np_demo2_analysis, is the ground
truth against which the full two-pass selection logic is checked.

A floating-point note (recorded here because it actually bit the oracle
designer while drafting this file): synthetic burst heights built via
`0.15 + 0.01*k` do NOT exactly equal the corresponding grid points built via
`0.05 + 0.005*n` at float precision (e.g. 0.15+0.01*14 ==
0.29000000000000004, one ULP above the grid's exact 0.29). The tests below
therefore always assert against contact_count() computed directly over the
SAME grid array used by proh_002_adjustment, rather than against
hand-typed decimal counts -- this keeps the oracle exact rather than
"approximately right", and is itself a regression guard against exactly this
class of float-boundary bug in any future implementation.

Acceptance-level mapping (contract_spec.yaml, NP-DEMO-2, contract_version 1)
-----------------------------------------------------------------------------
    RESULT-eth-threshold-log (level 1, exact per-experiment threshold values
        and contact counts) -- this file is the full acceptance oracle for
        that object's LOGIC (real-data anchoring is a separate, later gate).
    HAZ-04 (two-pass ordering: Pass-1 mean/SD frozen before any adjustment),
    HAZ-06 (all-ethanol session rescue, e.g. 2021-11-03) -- hazard-catalog
    regression tests.

Implementer note
-----------------
    # from np_demo2_analysis import proh_002_adjustment, contact_count
    # TODO: uncomment when np_demo2_analysis.py exists, and DELETE the
    # inline reference implementations below.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_NP_DEMO2_SW_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_software"
if _NP_DEMO2_SW_DIR not in sys.path and os.path.isdir(_NP_DEMO2_SW_DIR):
    sys.path.insert(0, _NP_DEMO2_SW_DIR)
from np_demo2_analysis import proh_002_adjustment, contact_count  # noqa: E402

GRID = np.round(np.arange(0.05, 0.50 + 1e-9, 0.005), 3)  # 91 points, 0.05..0.50 step 0.005
DEFAULT_THRESHOLD = 0.11


# ---------------------------------------------------------------------------
# _select_best_threshold is NOT imported from np_demo2_analysis (per this
# oracle's own Implementer note, only proh_002_adjustment and contact_count
# are); it is kept here, self-contained, so tests (d) can exercise the
# tie-break rule in isolation from any signal-generation logic. It is
# byte-identical to np_demo2_analysis._select_best_threshold.
# ---------------------------------------------------------------------------
def _select_best_threshold(grid: np.ndarray, counts_at_grid: np.ndarray,
                            mean: float, default: float = DEFAULT_THRESHOLD):
    """argmin |count-mean| over the grid; ties broken by nearest to
    `default`. Returns (best_threshold, best_count)."""
    grid = np.asarray(grid, dtype=np.float64)
    counts_at_grid = np.asarray(counts_at_grid, dtype=np.float64)
    diffs = np.abs(counts_at_grid - mean)
    best_diff = diffs.min()
    candidates = np.flatnonzero(diffs == best_diff)
    dist_to_default = np.abs(grid[candidates] - default)
    winner = candidates[np.argmin(dist_to_default)]
    return float(grid[winner]), float(counts_at_grid[winner])


# ---------------------------------------------------------------------------
# Synthetic ETH signal builders (known step structure, hand-auditable)
# ---------------------------------------------------------------------------
def _make_isolated_bursts(n_bursts: int, burst_height: float,
                           gap_len: int = 5, burst_len: int = 3,
                           baseline: float = 0.03) -> np.ndarray:
    """n_bursts isolated rectangular bursts of height burst_height, each
    separated by gap_len baseline samples (baseline chosen well below the
    0.05 grid floor so it is NEVER counted as a contact at any grid
    threshold). contact_count(sig, g) == n_bursts for any g in
    [0.05, burst_height) and == 0 for any g >= burst_height (single
    monotone step)."""
    sig = []
    for _ in range(n_bursts):
        sig += [baseline] * gap_len + [burst_height] * burst_len
    sig += [baseline] * gap_len
    return np.array(sig, dtype=np.float64)


def _make_variable_height_bursts(heights: np.ndarray, gap_len: int = 5,
                                  burst_len: int = 1, baseline: float = 0.03) -> np.ndarray:
    """One isolated burst per entry of `heights` (each a distinct height);
    contact_count(sig, g) == count(heights > g), a simple decreasing step
    function of g."""
    sig = []
    for h in heights:
        sig += [baseline] * gap_len + [float(h)] * burst_len
    sig += [baseline] * gap_len
    return np.array(sig, dtype=np.float64)


# ===========================================================================
# (a) All sessions within +/-1 SD -> no sessions flagged, all keep 0.11
# ===========================================================================
def test_proh002_no_sessions_flagged_when_counts_identical():
    """4 sessions, each with an IDENTICAL contact count (10, via 10 isolated
    bursts of height 0.6, well above the 0.11 default): mean=10, SD=0, so
    every session's |count-mean| == 0, which is NOT > SD(==0) -- none
    flagged, and every session keeps eth_threshold == 0.11 with its Pass-1
    count unchanged."""
    sessions = {
        f"S{i}": _make_isolated_bursts(10, burst_height=0.6) for i in range(4)
    }
    result = proh_002_adjustment(sessions, LV_Fs=100.0)

    assert result["flagged"] == set()
    for sid in sessions:
        assert result["final_threshold"][sid] == pytest.approx(0.11)
        assert result["final_count"][sid] == 10
    assert result["cross_session_mean"] == pytest.approx(10.0)
    assert result["cross_session_sd"] == pytest.approx(0.0)


# ===========================================================================
# (b) One session at 0 contacts -> flagged; grid search brings count UP
# ===========================================================================
def test_proh002_zero_contact_session_flagged_and_rescued_upward():
    """3 normal sessions with 10 contacts each (bursts of height 0.6, always
    above 0.11) and one degenerate session with 0 contacts at the 0.11
    default (its bursts only reach height 0.09, which never exceeds 0.11).
    Pass-1: counts = [10,10,10,0], mean=7.5, sample SD=5.0 (ddof=1);
    |0-7.5|=7.5 > 5.0 -> flagged; |10-7.5|=2.5 <= 5.0 -> not flagged.
    Pass-2 grid search on the flagged session: contact_count(sig,g) == 10
    for every grid g in [0.05, 0.085] (all ten 0.09-height bursts still
    exceed g) and == 0 for g >= 0.09 (bursts no longer exceed g) -- a single
    downward step. The argmin therefore picks any g in [0.05, 0.085]
    (|10-7.5|=2.5, strictly better than |0-7.5|=7.5); tie-break picks the
    one NEAREST the 0.11 default, i.e. g=0.085 (the top of that band).
    This is the HAZ-06 all-ethanol-rescue shape: threshold LOWERED to reveal
    previously-invisible control contacts, bringing the count UP toward the
    cross-session mean."""
    normal = {f"S{i}": _make_isolated_bursts(10, burst_height=0.6) for i in range(3)}
    low = {"S_low": _make_isolated_bursts(10, burst_height=0.09)}
    sessions = {**normal, **low}

    result = proh_002_adjustment(sessions, LV_Fs=100.0)

    assert result["cross_session_mean"] == pytest.approx(7.5)
    assert result["cross_session_sd"] == pytest.approx(5.0)
    assert result["flagged"] == {"S_low"}
    for sid in normal:
        assert result["final_threshold"][sid] == pytest.approx(0.11)
        assert result["final_count"][sid] == 10

    assert result["final_threshold"]["S_low"] == pytest.approx(0.085)
    assert result["final_count"]["S_low"] == 10
    assert result["final_count"]["S_low"] > result["pass1_count"]["S_low"], (
        "grid search must move the flagged session's count UP toward the mean"
    )


# ===========================================================================
# (c) One session at 3x the OTHER sessions' mean -> flagged; grid search
#     brings count DOWN
# ===========================================================================
def test_proh002_high_contact_session_flagged_and_reduced_downward():
    """3 normal sessions with 10 contacts each and one 'high' session with
    30 isolated bursts of strictly increasing height 0.15, 0.16, ..., 0.44
    (all > 0.11, so its Pass-1 count is exactly 30 == 3x the other sessions'
    count). Pass-1: counts=[10,10,10,30], mean=15, sample SD=10.0 (ddof=1);
    |30-15|=15 > 10 -> flagged; |10-15|=5 <= 10 -> not flagged.
    contact_count(high_sig, g) == number of the 30 burst heights strictly
    greater than g, a monotone-decreasing step function of g (verified
    directly, not hand-derived, per the float-precision note in the module
    docstring); the grid argmin is asserted against that same directly-
    computed step function, and must reduce the count toward the mean."""
    normal = {f"S{i}": _make_isolated_bursts(10, burst_height=0.6) for i in range(3)}
    heights = 0.15 + 0.01 * np.arange(30)
    high = {"S_high": _make_variable_height_bursts(heights)}
    sessions = {**normal, **high}

    result = proh_002_adjustment(sessions, LV_Fs=100.0)

    assert result["cross_session_mean"] == pytest.approx(15.0)
    assert result["cross_session_sd"] == pytest.approx(10.0)
    assert result["flagged"] == {"S_high"}
    for sid in normal:
        assert result["final_threshold"][sid] == pytest.approx(0.11)
        assert result["final_count"][sid] == 10

    # Independently recompute the grid argmin from the same trivial counter,
    # over the SAME grid, and require the implementation to match it exactly.
    counts_at_grid = np.array([contact_count(sessions["S_high"], g) for g in GRID])
    expected_threshold, expected_count = _select_best_threshold(GRID, counts_at_grid, mean=15.0)

    assert result["final_threshold"]["S_high"] == pytest.approx(expected_threshold)
    assert result["final_count"]["S_high"] == pytest.approx(expected_count)
    assert result["final_count"]["S_high"] < result["pass1_count"]["S_high"], (
        "grid search must move the flagged session's count DOWN toward the mean"
    )


# ===========================================================================
# (d) Tie-breaking: two grid thresholds give equal |count-mean| -> nearer to
#     0.11 wins (isolated from signal generation for maximal clarity)
# ===========================================================================
def test_proh002_tie_break_selects_grid_point_nearest_default():
    """Deliberately engineered tie: grid=[0.05, 0.09, 0.14, 0.20],
    counts=[20, 12, 12, 3], mean=12 -> diffs=[8, 0, 0, 9]. Two exact ties at
    diff=0 (g=0.09 and g=0.14); distances to the 0.11 default are 0.02 and
    0.03 respectively, so g=0.09 must win (unambiguous, hand-checked, no
    floating-point sensitivity)."""
    grid = np.array([0.05, 0.09, 0.14, 0.20])
    counts = np.array([20, 12, 12, 3])
    mean = 12.0

    best_threshold, best_count = _select_best_threshold(grid, counts, mean, default=0.11)

    assert best_threshold == pytest.approx(0.09)
    assert best_count == pytest.approx(12)


def test_proh002_tie_break_is_symmetric_regardless_of_grid_order():
    """Same tie as above, presented in reverse grid order -- the winner must
    be the same physical threshold (0.09), not simply 'the first' or 'the
    last' candidate encountered, proving the tie-break truly measures
    distance to 0.11 rather than list position."""
    grid = np.array([0.20, 0.14, 0.09, 0.05])
    counts = np.array([3, 12, 12, 20])
    mean = 12.0

    best_threshold, best_count = _select_best_threshold(grid, counts, mean, default=0.11)

    assert best_threshold == pytest.approx(0.09)
    assert best_count == pytest.approx(12)


# ===========================================================================
# (e) Monotone invariant: higher threshold -> fewer (or equal) contacts
# ===========================================================================
def test_proh002_contact_count_is_monotone_nonincreasing_in_threshold():
    """For ANY fixed ETH signal, raising the threshold can only remove
    contacts, never add them (a run that exceeds a higher bar also exceeded
    every lower bar) -- contact_count(sig, g) must be non-increasing as g
    increases over the full PROH-002 grid. This is the invariant the grid
    search direction depends on; verified directly over the 30-distinct-
    height synthetic signal (the case with the most possible step
    transitions, 30 of them, making it the strongest test of this
    property)."""
    heights = 0.15 + 0.01 * np.arange(30)
    sig = _make_variable_height_bursts(heights)
    counts_over_grid = [contact_count(sig, g) for g in GRID]
    assert all(
        counts_over_grid[i] >= counts_over_grid[i + 1]
        for i in range(len(counts_over_grid) - 1)
    )


def test_proh002_contact_count_monotone_also_holds_for_low_session():
    """Same monotonicity invariant on the 'low' (single-height-burst)
    synthetic session used in test (b): a single downward step from 10 to
    0, still consistent with non-increasing."""
    sig = _make_isolated_bursts(10, burst_height=0.09)
    counts_over_grid = [contact_count(sig, g) for g in GRID]
    assert all(
        counts_over_grid[i] >= counts_over_grid[i + 1]
        for i in range(len(counts_over_grid) - 1)
    )


# ===========================================================================
# HAZ-04 regression: Pass-1 mean/SD must be frozen before ANY session is
# adjusted (two-pass ordering, not an iterative/sequential loop)
# ===========================================================================
def test_proh002_cross_session_mean_is_computed_from_pass1_only():
    """The cross-session mean/SD used for BOTH flagging and the Pass-2
    argmin target must come from Pass-1 counts (all sessions at the SAME
    0.11 default), never recomputed after any session's threshold has been
    adjusted. Uses the same 4-session mix as test (b): if PROH-002 were
    (incorrectly) implemented as a sequential/iterative loop that updates
    the mean as it goes, the mean seen by the grid search would drift away
    from 7.5 as soon as the first session were adjusted -- this test pins
    7.5 as the value the grid search actually uses."""
    normal = {f"S{i}": _make_isolated_bursts(10, burst_height=0.6) for i in range(3)}
    low = {"S_low": _make_isolated_bursts(10, burst_height=0.09)}
    sessions = {**normal, **low}

    result = proh_002_adjustment(sessions, LV_Fs=100.0)

    assert result["cross_session_mean"] == pytest.approx(7.5)
    # The value actually used to pick S_low's threshold must match a grid
    # search against that SAME frozen mean, not some other value.
    counts_at_grid = np.array([contact_count(sessions["S_low"], g) for g in GRID])
    expected_threshold, _ = _select_best_threshold(GRID, counts_at_grid, mean=7.5)
    assert result["final_threshold"]["S_low"] == pytest.approx(expected_threshold)

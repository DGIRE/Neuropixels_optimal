"""
oracle_discard_log.py -- NP-DEMO-2 oracle for the discarded-SNF-section log
============================================================================
Written by the ORACLE DESIGNER (Blueprint v2 P3) BEFORE np_demo2_analysis.py
exists. Targets repo_map.md New-Code-Required function:

    log_discard_runs(SNF_PH, LV_Fs, session_date) -> list[dict]
        Finds every contiguous run of SNF_PH == -1 (the kernel's outside-
        valid-sniff-cycle sentinel, HAZ-01) and returns one
        {experiment, start_s, end_s, reason} entry per run, in time order.
        reason == "outside_valid_sniff" (fixed string, per repo_map.md's
        New-Code-Required table).

Ground truth is an elementary run-length-encoding identity: for a boolean
"invalid" mask, a run starting at sample index s (0-based) and ending (one
sample past its last invalid sample) at index e corresponds to
start_s = s/LV_Fs, end_s = e/LV_Fs -- checked here by hand-placing runs at
exactly-known sample indices, never by running np_demo2_analysis.

Acceptance-level mapping (contract_spec.yaml, NP-DEMO-2, contract_version 1)
-----------------------------------------------------------------------------
    RESULT-qc-discards (level 1, exact: "exact log entries per deterministic
        discard rule (spike_SNF_PH == -1)") -- this file is the full
        pre-fixture acceptance oracle for that object.
    HAZ-01 (spike_SNF_PH == -1 sentinel), HAZ-10 (every discard logged, never
        silently dropped) -- hazard-catalog regression tests.

Implementer note
-----------------
    # from np_demo2_analysis import log_discard_runs
    # TODO: uncomment when np_demo2_analysis.py exists, and DELETE the
    # inline reference implementation below.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_NP_DEMO2_SW_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_software"
if _NP_DEMO2_SW_DIR not in sys.path and os.path.isdir(_NP_DEMO2_SW_DIR):
    sys.path.insert(0, _NP_DEMO2_SW_DIR)
from np_demo2_analysis import log_discard_runs  # noqa: E402


# ===========================================================================
# (a) SNF_PH all >= 0.0 -> empty list
# ===========================================================================
def test_all_valid_snf_ph_gives_empty_discard_list():
    """No -1 sentinel samples anywhere -> nothing to discard, log must be
    an empty list (not None, not raising)."""
    SNF_PH = np.linspace(0.0, 0.98, 300)
    runs = log_discard_runs(SNF_PH, LV_Fs=125.0, session_date="2022-05-17")
    assert runs == []


# ===========================================================================
# (b) Single contiguous run of -1 at samples [100..200)
# ===========================================================================
def test_single_contiguous_run_has_correct_start_end_and_reason():
    """SNF_PH is valid (0.5) everywhere except samples [100, 200) (100
    consecutive -1 samples, indices 100 through 199 inclusive). LV_Fs=125.0.
    Expected: exactly one entry, start_s = 100/125 = 0.8, end_s = 200/125 =
    1.6, reason = 'outside_valid_sniff', experiment = the passed session_date."""
    LV_Fs = 125.0
    SNF_PH = np.full(400, 0.5)
    SNF_PH[100:200] = -1.0

    runs = log_discard_runs(SNF_PH, LV_Fs=LV_Fs, session_date="2021-11-01")

    assert len(runs) == 1
    entry = runs[0]
    assert entry["experiment"] == "2021-11-01"
    assert entry["start_s"] == pytest.approx(100.0 / LV_Fs, abs=1e-12)
    assert entry["end_s"] == pytest.approx(200.0 / LV_Fs, abs=1e-12)
    assert entry["reason"] == "outside_valid_sniff"


# ===========================================================================
# (c) Multiple non-adjacent runs -> multiple entries, in time order
# ===========================================================================
def test_multiple_nonadjacent_runs_returns_entries_in_time_order():
    """Three separated -1 runs at [10:20), [50:55), [90:100) within a
    400-sample valid background: must return exactly 3 entries, each with
    the correct start_s/end_s, listed in ascending time order (matching the
    order the runs occur in the signal, not insertion or any other order)."""
    LV_Fs = 100.0
    SNF_PH = np.full(400, 0.3)
    SNF_PH[10:20] = -1.0
    SNF_PH[50:55] = -1.0
    SNF_PH[90:100] = -1.0

    runs = log_discard_runs(SNF_PH, LV_Fs=LV_Fs, session_date="2021-12-15")

    assert len(runs) == 3
    expected = [(10, 20), (50, 55), (90, 100)]
    for entry, (s, e) in zip(runs, expected):
        assert entry["start_s"] == pytest.approx(s / LV_Fs, abs=1e-12)
        assert entry["end_s"] == pytest.approx(e / LV_Fs, abs=1e-12)
        assert entry["reason"] == "outside_valid_sniff"
        assert entry["experiment"] == "2021-12-15"
    # explicit time-ordering check (starts strictly increasing)
    starts = [entry["start_s"] for entry in runs]
    assert starts == sorted(starts)


# ===========================================================================
# (d) -1 at the very start and very end -> boundary runs included
# ===========================================================================
def test_boundary_runs_at_signal_start_and_end_are_both_included():
    """-1 sentinel occupies samples [0:5) (session starts mid-noise, before
    the first detected sniff) AND samples [95:100) (session ends mid-noise,
    LAST sample is invalid) within a 100-sample signal: both edge runs must
    be detected -- a naive np.diff-based run-finder can easily miss a run
    that never has an explicit 'rising' or 'falling' edge because it touches
    the array boundary."""
    LV_Fs = 50.0
    SNF_PH = np.full(100, 0.5)
    SNF_PH[0:5] = -1.0
    SNF_PH[95:100] = -1.0

    runs = log_discard_runs(SNF_PH, LV_Fs=LV_Fs, session_date="2022-09-14")

    assert len(runs) == 2
    first, last = runs[0], runs[1]
    assert first["start_s"] == pytest.approx(0.0 / LV_Fs, abs=1e-12)
    assert first["end_s"] == pytest.approx(5.0 / LV_Fs, abs=1e-12)
    assert last["start_s"] == pytest.approx(95.0 / LV_Fs, abs=1e-12)
    assert last["end_s"] == pytest.approx(100.0 / LV_Fs, abs=1e-12)
    for entry in runs:
        assert entry["reason"] == "outside_valid_sniff"
        assert entry["experiment"] == "2022-09-14"


def test_entire_signal_invalid_is_a_single_full_length_run():
    """Degenerate case: the ENTIRE SNF_PH array is -1 (no valid sniff ever
    detected, e.g. a session with a broken sniff sensor) -> exactly one run
    spanning the full signal, start_s=0.0, end_s=len(SNF_PH)/LV_Fs. This is
    also the contract's documented failure_conditions case ('no valid sniff
    segments detected in an experiment'), which must still be LOGGED (HAZ-10)
    rather than silently producing an empty list."""
    LV_Fs = 125.0
    SNF_PH = np.full(250, -1.0)

    runs = log_discard_runs(SNF_PH, LV_Fs=LV_Fs, session_date="2021-11-03")

    assert len(runs) == 1
    assert runs[0]["start_s"] == pytest.approx(0.0, abs=1e-12)
    assert runs[0]["end_s"] == pytest.approx(250.0 / LV_Fs, abs=1e-12)

"""
oracle_qc_counts.py -- NP-DEMO-2 oracle for per-experiment QC counts
============================================================================
Written by the ORACLE DESIGNER (Blueprint v2 P3) BEFORE np_demo2_analysis.py
exists. Targets repo_map.md New-Code-Required function:

    qc_counts(sniff_onsets_s, session_duration_s, LV_Fs, n_neurons_valid,
              eth_contact_count, SNF_PH) -> dict
        n_sniffs            = len(sniff_onsets_s)
        n_neurons           = n_neurons_valid (pass-through)
        n_trials            = eth_contact_count (pass-through; contract
                               assumptions: "n_trials ... is defined as the
                               count of discrete ethanol-contact events")
        length_min           = session_duration_s / 60.0
        pct_usable_sniffs    = (sum(SNF_PH >= 0) / len(SNF_PH)) * 100
                               (contract assumptions: "percentage of time
                               with usable sniffs = (valid-sniff samples /
                               total samples) * 100, using the SNF_PH >= 0.0
                               mask")

Every expected value is an exact arithmetic identity from the contract's own
`assumptions` clause (verbatim quoted above), computed by hand on small,
exactly-known synthetic inputs -- not produced by running np_demo2_analysis.

Acceptance-level mapping (contract_spec.yaml, NP-DEMO-2, contract_version 1)
-----------------------------------------------------------------------------
    RESULT-qc-counts (level 1, exact: "exact integer/float counts (n_sniffs,
        n_neurons, n_trials, minutes, pct usable)") -- this file is the full
        pre-fixture acceptance oracle for that object's arithmetic.

Implementer note
-----------------
    # from np_demo2_analysis import qc_counts
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
from np_demo2_analysis import qc_counts  # noqa: E402


# ===========================================================================
# (a) Known sniff_onsets_s -> correct n_sniffs
# ===========================================================================
def test_n_sniffs_equals_length_of_sniff_onsets_array():
    """n_sniffs must equal len(sniff_onsets_s) EXACTLY (level-1 exact-count
    contract), regardless of the onset VALUES themselves."""
    sniff_onsets_s = np.array([0.216, 0.512, 0.784, 1.064, 1.344, 1.600, 1.900])
    SNF_PH = np.full(1000, 0.5)  # irrelevant to this assertion
    counts = qc_counts(sniff_onsets_s, session_duration_s=8.0, LV_Fs=125.0,
                        n_neurons_valid=3, eth_contact_count=2, SNF_PH=SNF_PH)
    assert counts["n_sniffs"] == 7


# ===========================================================================
# (b) session_duration_s in seconds -> correct length_min (/ 60.0)
# ===========================================================================
def test_length_min_divides_session_duration_seconds_by_sixty():
    """session_duration_s = 600.0 s (10 minutes exactly) -> length_min ==
    10.0 exactly; also checked at a non-round duration to confirm it is a
    true division, not a rounded/truncated minute count."""
    SNF_PH = np.full(10, 0.5)
    counts_round = qc_counts(np.array([]), session_duration_s=600.0, LV_Fs=125.0,
                              n_neurons_valid=0, eth_contact_count=0, SNF_PH=SNF_PH)
    assert counts_round["length_min"] == pytest.approx(10.0, abs=1e-12)

    counts_nonround = qc_counts(np.array([]), session_duration_s=125.0, LV_Fs=125.0,
                                 n_neurons_valid=0, eth_contact_count=0, SNF_PH=SNF_PH)
    assert counts_nonround["length_min"] == pytest.approx(125.0 / 60.0, abs=1e-12)


# ===========================================================================
# (c) pct_usable_sniffs = (sum(SNF_PH >= 0) / len(SNF_PH)) * 100
# ===========================================================================
def test_pct_usable_sniffs_matches_valid_sniff_ph_fraction_exactly():
    """40 samples with SNF_PH >= 0.0 (valid, values 0.0-0.98) and 60 samples
    at the -1 sentinel (invalid, HAZ-01), out of 100 total: pct_usable must
    be EXACTLY 40.0, not e.g. counting -1 as valid or double-counting."""
    valid = np.linspace(0.0, 0.98, 40)
    invalid = np.full(60, -1.0)
    SNF_PH = np.concatenate([valid, invalid])
    counts = qc_counts(np.array([0.0]), session_duration_s=1.0, LV_Fs=125.0,
                        n_neurons_valid=1, eth_contact_count=0, SNF_PH=SNF_PH)
    assert counts["pct_usable_sniffs"] == pytest.approx(40.0, abs=1e-9)


def test_pct_usable_sniffs_boundary_zero_phase_counts_as_usable():
    """SNF_PH == 0.0 exactly (start of a sniff cycle) is USABLE (the
    exclusion is strictly '< 0', matching valid_sniff_mask's own boundary in
    oracle_condition_assignment.py): 1 sample at 0.0 and 1 at -1.0 -> 50%,
    not 0%."""
    SNF_PH = np.array([0.0, -1.0])
    counts = qc_counts(np.array([]), session_duration_s=1.0, LV_Fs=125.0,
                        n_neurons_valid=0, eth_contact_count=0, SNF_PH=SNF_PH)
    assert counts["pct_usable_sniffs"] == pytest.approx(50.0, abs=1e-9)


# ===========================================================================
# (d) Zero sniff onsets -> n_sniffs = 0, pct = 0.0
# ===========================================================================
def test_zero_sniff_onsets_and_all_invalid_snf_ph_gives_zero_counts():
    """No detected sniff onsets AND every SNF_PH sample at the -1 sentinel
    (a degenerate/entirely-noise session): n_sniffs must be 0 and
    pct_usable_sniffs must be 0.0 (not nan, not raising a divide-by-zero)."""
    sniff_onsets_s = np.array([])
    SNF_PH = np.full(500, -1.0)
    counts = qc_counts(sniff_onsets_s, session_duration_s=4.0, LV_Fs=125.0,
                        n_neurons_valid=0, eth_contact_count=0, SNF_PH=SNF_PH)
    assert counts["n_sniffs"] == 0
    assert counts["pct_usable_sniffs"] == pytest.approx(0.0, abs=1e-12)


def test_pass_through_fields_n_neurons_and_n_trials_are_exact():
    """n_neurons and n_trials (== eth_contact_count, per contract
    assumptions) are simple pass-throughs of caller-supplied integers and
    must not be recomputed, rounded, or otherwise altered."""
    SNF_PH = np.full(10, 0.5)
    counts = qc_counts(np.array([0.0, 1.0]), session_duration_s=2.0, LV_Fs=125.0,
                        n_neurons_valid=17, eth_contact_count=4, SNF_PH=SNF_PH)
    assert counts["n_neurons"] == 17
    assert counts["n_trials"] == 4

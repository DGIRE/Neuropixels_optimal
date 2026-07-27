"""
oracle_unit_inclusion.py -- NP-DEMO-2 oracle for the unit inclusion rule
============================================================================
Written by the ORACLE DESIGNER (Blueprint v2 P3) BEFORE np_demo2_analysis.py
exists. Targets repo_map.md New-Code-Required function:

    unit_included(n_spikes_total, session_duration_s, n_valid_sniff_spikes) -> bool

Ground truth is the contract's inclusion_exclusion clause verbatim
(contract_spec.yaml, NP-DEMO-2): "exclude units with < 0.1 Hz overall firing
rate OR < 50 valid-sniff spikes combined across the ethanol + control
conditions (same as the approved NP-DEMO-1 contract)." Every microcase below
is an exact arithmetic construction (n_spikes / duration_s chosen so the
firing rate lands on an exact decimal), not a value produced by running
np_demo2_analysis.

Both criteria are ">=" (inclusive) at their boundary, per the approved
NP-DEMO-1 convention this contract explicitly reuses -- 0.1 Hz exactly
passes, 50 valid spikes exactly passes.

Acceptance-level mapping (contract_spec.yaml, NP-DEMO-2, contract_version 1)
-----------------------------------------------------------------------------
    RESULT-mrl-unit / RESULT-phase-unit / RESULT-stat-unit (level 3) --
        unit_included gates every unit BEFORE it can contribute an MRL,
        phase, or LMM observation; a wrong boundary silently changes n_units
        (reported in FIG-DEMO2-RES-UNIT caption_requirements: "state n_units
        per condition (computed, not typed)").

Implementer note
-----------------
    # from np_demo2_analysis import unit_included
    # TODO: uncomment when np_demo2_analysis.py exists, and DELETE the
    # inline reference implementation below.
"""
from __future__ import annotations

import os
import sys

import pytest

_NP_DEMO2_SW_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_software"
if _NP_DEMO2_SW_DIR not in sys.path and os.path.isdir(_NP_DEMO2_SW_DIR):
    sys.path.insert(0, _NP_DEMO2_SW_DIR)
from np_demo2_analysis import unit_included  # noqa: E402


# ===========================================================================
# (a) FR = 0.099 Hz -> exclude (just below the 0.1 Hz floor)
# ===========================================================================
def test_unit_excluded_firing_rate_just_below_floor():
    """99 spikes / 1000 s == 0.099 Hz, strictly less than the 0.1 Hz floor
    -> excluded, even with a generous valid-spike count."""
    included = unit_included(
        n_spikes_total=99, session_duration_s=1000.0, n_valid_sniff_spikes=5000,
    )
    assert included is False


# ===========================================================================
# (b) FR = 0.101 Hz, n_valid = 49 -> exclude (spike-count floor dominates)
# ===========================================================================
def test_unit_excluded_valid_spikes_just_below_floor_despite_adequate_rate():
    """101 spikes / 1000 s == 0.101 Hz (comfortably above the 0.1 Hz floor),
    but only 49 valid-sniff spikes (< 50 floor) -> excluded. The two
    criteria are independent ORs: passing one does not compensate for
    failing the other."""
    included = unit_included(
        n_spikes_total=101, session_duration_s=1000.0, n_valid_sniff_spikes=49,
    )
    assert included is False


# ===========================================================================
# (c) FR = 0.101 Hz, n_valid = 50 -> include
# ===========================================================================
def test_unit_included_when_both_criteria_pass():
    """Identical to the previous case except exactly 50 valid-sniff spikes:
    both criteria now pass (0.101 >= 0.1 and 50 >= 50) -> included."""
    included = unit_included(
        n_spikes_total=101, session_duration_s=1000.0, n_valid_sniff_spikes=50,
    )
    assert included is True


# ===========================================================================
# (d) FR = 0.1 Hz exactly -> include (boundary is inclusive)
# ===========================================================================
def test_unit_included_firing_rate_exactly_at_floor_boundary():
    """100 spikes / 1000 s == 0.1 Hz EXACTLY: the floor is '>=', so this
    must be included, not excluded."""
    included = unit_included(
        n_spikes_total=100, session_duration_s=1000.0, n_valid_sniff_spikes=50,
    )
    assert included is True


# ===========================================================================
# (e) FR = 0.0 (zero spikes) -> exclude
# ===========================================================================
def test_unit_excluded_zero_spikes():
    """Zero total spikes -> firing rate is 0.0 Hz, far below the 0.1 Hz
    floor -> excluded; must not raise (e.g. divide-by-zero on rate) and
    must not be confused with n_valid_sniff_spikes=0 (a separate,
    also-failing criterion here, but the rate criterion alone is decisive)."""
    included = unit_included(
        n_spikes_total=0, session_duration_s=1000.0, n_valid_sniff_spikes=0,
    )
    assert included is False


# ===========================================================================
# Metamorphic relation: session duration scaling
# ===========================================================================
def test_metamorphic_scaling_session_duration_preserves_rate_decision():
    """Metamorphic relation: unit_included depends on n_spikes_total and
    session_duration_s only through their ratio. Scaling both n_spikes_total
    and session_duration_s by the same positive factor k must leave the
    firing-rate decision unchanged (n_valid_sniff_spikes held fixed above
    its own floor so only the rate criterion is exercised)."""
    base = unit_included(n_spikes_total=101, session_duration_s=1000.0, n_valid_sniff_spikes=50)
    scaled = unit_included(n_spikes_total=101 * 7, session_duration_s=1000.0 * 7, n_valid_sniff_spikes=50)
    assert base is True
    assert scaled is True

    base_excl = unit_included(n_spikes_total=99, session_duration_s=1000.0, n_valid_sniff_spikes=50)
    scaled_excl = unit_included(n_spikes_total=99 * 3, session_duration_s=1000.0 * 3, n_valid_sniff_spikes=50)
    assert base_excl is False
    assert scaled_excl is False

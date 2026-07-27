# ORACLE
"""
Tests for unit inclusion/exclusion criteria and top-5 example selection.

Targets:
  - RESULT-mrl-per-unit / RESULT-mrl-per-experiment gating (which units even
    appear in these level-3 tables).
  - RESULT-examples-top5 (contract acceptance level 1, exact; "exact set of 5
    (unit_id, experiment) pairs must match the ranked selection; ties broken
    by unit_id ascending").

Contract wording under test (inclusion_exclusion, verbatim):
  "Unit included only if overall firing rate >= 0.1 Hz AND >= 50 valid-sniff
  spikes combined across ethanol + control conditions. Spike included only if
  spike_SNF_PH >= 0 ... For the top-5 examples figure, additionally require
  > 500 spikes per unit per condition."

The reference predicates below (`meets_firing_rate_criterion`,
`meets_spike_count_criterion`, `meets_top5_spike_criterion`, `select_top5`)
are direct, literal transcriptions of this contract text -- not copies of, or
derived from running, the not-yet-written analysis code. They exist so the
exact boundary values (0.1 Hz, 50 spikes, 500 spikes, tie-break order) are
pinned BEFORE implementation and can gate the real pipeline's behavior later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def meets_firing_rate_criterion(overall_firing_rate_hz: float) -> bool:
    """Contract: 'overall firing rate >= 0.1 Hz' (inclusive lower bound)."""
    return overall_firing_rate_hz >= 0.1


def meets_spike_count_criterion(n_valid_spikes_combined: int) -> bool:
    """Contract: '>= 50 valid-sniff spikes combined across ethanol + control conditions' (inclusive lower bound)."""
    return n_valid_spikes_combined >= 50


def meets_top5_spike_criterion(n_spikes_eth: int, n_spikes_ctrl: int) -> bool:
    """Contract: '> 500 spikes per unit per condition' (strict, BOTH conditions must individually exceed 500)."""
    return n_spikes_eth > 500 and n_spikes_ctrl > 500


def filter_valid_sniff_spikes(spike_snf_ph: np.ndarray) -> np.ndarray:
    """Contract: 'Spike included only if spike_SNF_PH >= 0' -- the -1 sentinel marks an invalid/non-sniffing spike."""
    spike_snf_ph = np.asarray(spike_snf_ph, dtype=float)
    return spike_snf_ph[spike_snf_ph >= 0]


def select_top5(df: pd.DataFrame) -> pd.DataFrame:
    """Contract: 'Top-5 units by |delta_MRL| ... ties broken by unit_id ascending.'"""
    ordered = df.sort_values(by=["abs_delta_MRL", "unit_id"], ascending=[False, True])
    return ordered.head(5)


def test_firing_rate_boundary_is_inclusive():
    """0.05 Hz excluded; exactly 0.1 Hz included (contract lower-bound is '>=', not '>')."""
    assert not meets_firing_rate_criterion(0.05)
    assert meets_firing_rate_criterion(0.1)
    assert meets_firing_rate_criterion(0.10000001)


def test_minimum_valid_spike_count_boundary_is_inclusive():
    """49 combined valid-sniff spikes excluded; exactly 50 included."""
    assert not meets_spike_count_criterion(49)
    assert meets_spike_count_criterion(50)


def test_top5_per_condition_spike_count_is_strict_and_per_condition():
    """Top-5 eligibility requires BOTH conditions to individually exceed 500 spikes (strict '>')."""
    # 499 in ethanol (fails), even though control has 600 (irrelevant -- both must pass).
    assert not meets_top5_spike_criterion(n_spikes_eth=499, n_spikes_ctrl=600)
    # 500 exactly is NOT > 500 -- must fail (strict inequality).
    assert not meets_top5_spike_criterion(n_spikes_eth=500, n_spikes_ctrl=600)
    # 501/501 clears the strict threshold on both sides.
    assert meets_top5_spike_criterion(n_spikes_eth=501, n_spikes_ctrl=501)


def test_spike_snf_ph_sentinel_excluded_from_spike_count_and_mrl():
    """spike_SNF_PH == -1 (invalid sniff cycle) must never count toward the 50-spike criterion or any MRL computation."""
    # 60 total entries, but only 40 are valid (>= 0); the rest are -1 sentinels.
    raw = np.concatenate([np.linspace(0, 0.99, 40), np.full(20, -1.0)])
    valid = filter_valid_sniff_spikes(raw)

    assert len(valid) == 40
    assert not meets_spike_count_criterion(len(valid))  # 40 < 50 -> still excluded
    assert np.all(valid >= 0)
    assert -1.0 not in valid


def test_spike_snf_ph_sentinel_pushes_unit_from_excluded_to_included():
    """Combined ethanol+control valid spike counts of 30 + 19 = 49 (excluded); 30 + 20 = 50 (included) -- sentinels never counted."""
    eth_raw = np.concatenate([np.linspace(0, 0.9, 30), np.full(5, -1.0)])
    ctrl_raw_excluded = np.concatenate([np.linspace(0, 0.9, 19), np.full(5, -1.0)])
    ctrl_raw_included = np.concatenate([np.linspace(0, 0.9, 20), np.full(5, -1.0)])

    n_valid_excluded = len(filter_valid_sniff_spikes(eth_raw)) + len(filter_valid_sniff_spikes(ctrl_raw_excluded))
    n_valid_included = len(filter_valid_sniff_spikes(eth_raw)) + len(filter_valid_sniff_spikes(ctrl_raw_included))

    assert n_valid_excluded == 49
    assert not meets_spike_count_criterion(n_valid_excluded)
    assert n_valid_included == 50
    assert meets_spike_count_criterion(n_valid_included)


def test_top5_tie_break_by_unit_id_ascending():
    """Two units tied on |delta_MRL| must be ordered by unit_id ascending (contract tie-break rule)."""
    df = pd.DataFrame(
        {
            "unit_id": [42, 7, 15, 3, 99, 1],
            "experiment": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "abs_delta_MRL": [0.30, 0.30, 0.25, 0.20, 0.10, 0.05],
        }
    )
    top5 = select_top5(df)

    # Two units (42 and 7) tie at 0.30 -- unit_id=7 must be ranked ahead of unit_id=42.
    assert list(top5["unit_id"])[:2] == [7, 42]
    assert len(top5) == 5
    assert 1 not in set(top5["unit_id"])  # the 6th-ranked unit (lowest |delta_MRL|) is excluded


def test_top5_selection_excludes_units_failing_spike_criterion_upstream():
    """Units/experiments not meeting the >500-spikes-per-condition rule must never reach the top-5 ranking pool."""
    candidates = pd.DataFrame(
        {
            "unit_id": [1, 2, 3, 4, 5, 6],
            "n_spikes_eth": [600, 501, 499, 800, 700, 650],
            "n_spikes_ctrl": [600, 501, 900, 500, 700, 650],  # unit 3 fails eth; unit 4 fails ctrl (== 500)
            "abs_delta_MRL": [0.5, 0.4, 0.9, 0.8, 0.3, 0.2],
        }
    )
    eligible_mask = candidates.apply(
        lambda row: meets_top5_spike_criterion(row["n_spikes_eth"], row["n_spikes_ctrl"]), axis=1
    )
    eligible = candidates[eligible_mask]

    assert set(eligible["unit_id"]) == {1, 2, 5, 6}
    assert 3 not in set(eligible["unit_id"])  # highest |delta_MRL| but fails eth spike count
    assert 4 not in set(eligible["unit_id"])  # fails ctrl spike count (exactly 500, not > 500)

    top = select_top5(eligible.assign(experiment=[f"e{u}" for u in eligible["unit_id"]]))
    # Despite unit 3 having the single largest |delta_MRL| (0.9) in the full
    # candidate pool, it must be absent from the final top selection because
    # it was never eligible.
    assert 3 not in set(top["unit_id"])

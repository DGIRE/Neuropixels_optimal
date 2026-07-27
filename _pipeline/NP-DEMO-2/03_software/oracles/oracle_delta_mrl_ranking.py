"""
oracle_delta_mrl_ranking.py -- NP-DEMO-2 oracle for |delta_MRL| ranking and
the pooled-MRL selection basis for QC-PSTH extremes
============================================================================
Written by the ORACLE DESIGNER (Blueprint v2 P3) BEFORE np_demo2_analysis.py
exists. Targets two related, contract-pinned selection rules built on top of
mrl_and_preferred_phase (see oracle_circular_stats.py, not retested here):

    1. RESULT-delta-mrl / FIG-DEMO2-RES-EX (OD-2): per-unit
       delta_MRL = MRL_ethanol - MRL_control; the k=5 units with the
       largest |delta_MRL| feed the "examples" figure.
    2. RESULT-psth-extremes / FIG-DEMO2-QC-PSTH (OD-4): the single
       strongest- and single weakest-overall phase-locking units, where
       "overall" is explicitly defined by the contract as "MRL on ALL
       valid-sniff spikes POOLED across ethanol+control (condition-
       agnostic)" -- NOT an average of the two per-condition MRLs.

Every expected ranking below is computed from hand-typed, exactly-known
input tables (no randomness, no fixture dependence) using elementary
sort/argmax logic; the pooled-vs-averaged distinction is checked against the
same exact two-cluster circular-vector-sum identity validated in
oracle_circular_stats.py (equal-weight unit vectors at phases a, b sum to a
vector of magnitude cos((a-b)/2), pointing at their bisector) -- not an
approximation.

Acceptance-level mapping (contract_spec.yaml, NP-DEMO-2, contract_version 1)
-----------------------------------------------------------------------------
    RESULT-delta-mrl      (level 3, distributional; delta_MRL within 1e-6)
    RESULT-psth-extremes  (level 2, numeric; rtol=1e-6 atol=1e-9 vs kernel PSTH)
    RESULT-psth-examples  (level 2, same tolerance) -- selection basis shared
        with RESULT-delta-mrl's top-k ranking.

Implementer note
-----------------
    # from np_demo2_analysis import (
    #     mrl_and_preferred_phase, rank_units_by_abs_delta_mrl,
    #     top_k_by_abs_delta_mrl, pooled_mrl, select_psth_extremes,
    # )
    # TODO: uncomment when np_demo2_analysis.py exists, and DELETE the
    # inline reference implementations below.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

_NP_DEMO2_SW_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_software"
if _NP_DEMO2_SW_DIR not in sys.path and os.path.isdir(_NP_DEMO2_SW_DIR):
    sys.path.insert(0, _NP_DEMO2_SW_DIR)
from np_demo2_analysis import (  # noqa: E402
    mrl_and_preferred_phase, rank_units_by_abs_delta_mrl,
    top_k_by_abs_delta_mrl, pooled_mrl, select_psth_extremes,
)


# ===========================================================================
# (a) Known MRL values -> top-5 by |delta_MRL| selected correctly
# ===========================================================================
def test_top5_by_abs_delta_mrl_selects_correct_units_in_descending_order():
    """8 units with strictly decreasing |delta_MRL| by hand-typed
    construction (delta = 0.90, 0.80, ..., 0.20 in steps of 0.10, no ties):
    the top-5 must be exactly U1..U5 in that descending order."""
    df = pd.DataFrame({
        "unit_id": [f"U{i}" for i in range(1, 9)],
        "mrl_ethanol": [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
        "mrl_control": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    })
    expected_deltas = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
    assert (df["mrl_ethanol"] - df["mrl_control"]).round(10).tolist() == expected_deltas

    top5 = top_k_by_abs_delta_mrl(df, k=5)

    assert top5["unit_id"].tolist() == ["U1", "U2", "U3", "U4", "U5"]
    assert top5["abs_delta_mrl"].tolist() == pytest.approx([0.90, 0.80, 0.70, 0.60, 0.50])


def test_top5_by_abs_delta_mrl_handles_negative_delta_via_absolute_value():
    """|delta_MRL| ranking must use the ABSOLUTE value: a unit whose control
    MRL exceeds its ethanol MRL by a large margin (delta strongly negative)
    must rank ahead of a unit with a small positive delta."""
    df = pd.DataFrame({
        "unit_id": ["U_neg_big", "U_pos_small"],
        "mrl_ethanol": [0.10, 0.55],
        "mrl_control": [0.90, 0.50],  # delta = -0.80 vs +0.05
    })
    ranked = rank_units_by_abs_delta_mrl(df)
    assert ranked["unit_id"].tolist() == ["U_neg_big", "U_pos_small"]
    assert ranked["delta_mrl"].tolist() == pytest.approx([-0.80, 0.05])
    assert ranked["abs_delta_mrl"].tolist() == pytest.approx([0.80, 0.05])


# ===========================================================================
# (b) Tie-breaking: equal |delta_MRL| -> deterministic ordering by unit_id
# ===========================================================================
def test_tie_break_equal_abs_delta_mrl_orders_by_unit_id_ascending():
    """U09 has delta=+0.50, U10 has delta=-0.50: |delta_MRL| is IDENTICAL
    (0.50) for both, so the ranking must fall back to a deterministic
    secondary key (ascending unit_id): U09 must be listed before U10,
    regardless of which was constructed first or which sign is larger."""
    df = pd.DataFrame({
        "unit_id": ["U10", "U09", "U_other"],  # deliberately out of order
        "mrl_ethanol": [0.20, 0.75, 0.50],
        "mrl_control": [0.70, 0.25, 0.50],  # deltas: -0.50, +0.50, 0.00
    })
    ranked = rank_units_by_abs_delta_mrl(df)
    assert ranked["unit_id"].tolist() == ["U09", "U10", "U_other"]
    assert ranked["abs_delta_mrl"].tolist() == pytest.approx([0.50, 0.50, 0.00])


def test_tie_break_is_stable_under_row_reordering_of_the_input():
    """Metamorphic relation: the ranking (including tie-break) must not
    depend on the ORDER rows were supplied in the input DataFrame -- shuffle
    the tie-break fixture's rows and require an identical ranked output."""
    df = pd.DataFrame({
        "unit_id": ["U10", "U09", "U_other"],
        "mrl_ethanol": [0.20, 0.75, 0.50],
        "mrl_control": [0.70, 0.25, 0.50],
    })
    shuffled = df.iloc[[2, 0, 1]].reset_index(drop=True)

    ranked_original = rank_units_by_abs_delta_mrl(df)
    ranked_shuffled = rank_units_by_abs_delta_mrl(shuffled)

    assert ranked_original["unit_id"].tolist() == ranked_shuffled["unit_id"].tolist()


# ===========================================================================
# (c) Strongest/weakest for QC-PSTH use POOLED MRL, not per-condition average
# ===========================================================================
def test_psth_extremes_use_pooled_mrl_not_average_of_per_condition_mrls():
    """Adversarial construction proving pooling (not averaging) is used:

    Unit X: 50 ethanol spikes all at phase 0.0 (MRL_eth=1.0) and 50 control
    spikes all at phase 0.5 (MRL_ctrl=1.0). NAIVE per-condition average would
    call this a MAXIMALLY locked unit (mean(1.0, 1.0) == 1.0). But POOLING
    all 100 spikes (50@0.0 + 50@0.5, exactly opposite unit vectors) gives a
    resultant sum of EXACTLY ZERO -> pooled MRL == 0.0 -- actually the
    WEAKEST possible unit under the contract's pinned OD-4 definition.

    Unit Y: 50 ethanol spikes at phase 0.2 AND 50 control spikes at phase
    0.2 (identical phase both conditions): naive average also gives 1.0
    (tied with X under averaging, so averaging cannot even distinguish
    them), but pooling all 100 spikes (all at 0.2) gives pooled MRL == 1.0,
    the actual strongest.

    select_psth_extremes must report Y as strongest and X as weakest.
    """
    unit_x_eth = np.zeros(50)
    unit_x_ctrl = np.full(50, 0.5)
    unit_y_eth = np.full(50, 0.2)
    unit_y_ctrl = np.full(50, 0.2)

    # sanity: naive per-condition averaging cannot distinguish X and Y
    mrl_x_eth, _ = mrl_and_preferred_phase(unit_x_eth)
    mrl_x_ctrl, _ = mrl_and_preferred_phase(unit_x_ctrl)
    mrl_y_eth, _ = mrl_and_preferred_phase(unit_y_eth)
    mrl_y_ctrl, _ = mrl_and_preferred_phase(unit_y_ctrl)
    assert (mrl_x_eth + mrl_x_ctrl) / 2 == pytest.approx((mrl_y_eth + mrl_y_ctrl) / 2, abs=1e-9)

    # the pinned pooled definition DOES distinguish them
    pooled_x = pooled_mrl(unit_x_eth, unit_x_ctrl)
    pooled_y = pooled_mrl(unit_y_eth, unit_y_ctrl)
    assert pooled_x == pytest.approx(0.0, abs=1e-9)
    assert pooled_y == pytest.approx(1.0, abs=1e-9)

    units = {"unit_x": (unit_x_eth, unit_x_ctrl), "unit_y": (unit_y_eth, unit_y_ctrl)}
    strongest, weakest = select_psth_extremes(units)
    assert strongest == "unit_y"
    assert weakest == "unit_x"


def test_psth_extremes_selection_spans_all_experiments_not_per_experiment():
    """Contract OD-4: the single strongest and single weakest are selected
    across ALL experiments combined (exactly 2 units total), not one
    strongest/weakest PER experiment. Four units spread across 3 'experiment'
    prefixes, each built from the SAME exact two-cluster identity used in
    oracle_circular_stats.py (equal-weight unit vectors at phases a and b sum
    to magnitude cos((a-b)/2)) so every unit's pooled MRL is known exactly
    and, by construction, STRICTLY DECREASING with no ties:
        expA_u1: a=b=0.1        -> pooled MRL = cos(0)          = 1.000
        expB_u1: a=0.1, b=0.2   -> pooled MRL = cos(0.1*pi)     = 0.951
        expC_u1: a=0.0, b=0.3   -> pooled MRL = cos(0.3*pi)     = 0.588
        expA_u2: a=0.0, b=0.5   -> pooled MRL = cos(0.5*pi)     = 0.000
    so exactly one strongest (expA_u1) and one weakest (expA_u2) must be
    returned, spanning 3 different 'experiments' rather than one pair per
    experiment."""
    def two_cluster(a, b, n=20):
        return np.full(n, a), np.full(n, b)

    eth1, ctrl1 = two_cluster(0.1, 0.1)
    eth2, ctrl2 = two_cluster(0.1, 0.2)
    eth3, ctrl3 = two_cluster(0.0, 0.3)
    eth4, ctrl4 = two_cluster(0.0, 0.5)

    units = {
        "expA_u1": (eth1, ctrl1),
        "expB_u1": (eth2, ctrl2),
        "expC_u1": (eth3, ctrl3),
        "expA_u2": (eth4, ctrl4),
    }

    expected_pooled = {
        "expA_u1": math.cos(0.0 * math.pi),
        "expB_u1": math.cos(0.1 * math.pi),
        "expC_u1": math.cos(0.3 * math.pi),
        "expA_u2": math.cos(0.5 * math.pi),
    }
    for uid, (eth, ctrl) in units.items():
        assert pooled_mrl(eth, ctrl) == pytest.approx(expected_pooled[uid], abs=1e-9)
    # sanity: strictly decreasing, so the selection below is unambiguous
    ordered_values = [expected_pooled[u] for u in ("expA_u1", "expB_u1", "expC_u1", "expA_u2")]
    assert all(ordered_values[i] > ordered_values[i + 1] for i in range(len(ordered_values) - 1))

    strongest, weakest = select_psth_extremes(units)
    assert strongest == "expA_u1"
    assert weakest == "expA_u2"

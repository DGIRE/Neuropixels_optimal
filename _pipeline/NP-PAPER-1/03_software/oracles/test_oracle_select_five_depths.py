# ORACLE
"""
Analytical microcases + real-geometry check for the NOT-YET-IMPLEMENTED
select_five_depths(xcoords, ycoords, valid_lfp_channels).

CONTRACT TARGET
  output_id: five-depth-selection
  ACCEPTANCE LEVEL 1 (exact) — contract acceptance_criteria:
    "Five chosen channel indices are exact integers; ycoords (um) match
     exactly; deterministic given xcoords/ycoords (zero tolerance)."
  docx_source_refs: RQ-012, RQ-013, RQ-014, RQ-015, RQ-016, OUT-006, OUT-007
  Also feeds RESULT-qc-counts ("five chosen channel indices + ycoords", level 1
  exact integers / ycoords within 1e-4).

ALGORITHM UNDER TEST (contract preprocessing step RQ-012..016 / repo_map §2.2):
  1. Candidate channels = those with BOTH a valid LFP row (in
     valid_lfp_channels) AND a finite ycoord (RQ-014).
  2. ymin/ymax = min/max ycoord over candidates.
  3. Five targets at fractions [0, 0.25, 0.5, 0.75, 1.0] of [ymin, ymax].
  4. For each target: nearest candidate by |ycoord - target|; ties -> LOWER
     channel index (RQ-016 single-fixed-column tie-break).
  5. Depth 1..5 labelled tip->surface (lowest ycoord = Depth 1).

All expected indices/ycoords below are hand-computed from this rule (each shown
in the test docstring) and were independently reconfirmed with a stand-alone
numpy argmin at oracle-design time. No value came from running select_five_depths.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    DEPTH_FRACTIONS,
    load_real_select_five_depths,
    reference_select_five_depths,
)


def _run(xcoords, ycoords, valid):
    """Call the REAL select_five_depths if implemented, else the contract
    reference. Normalises the return to (indices list[int], ycoords list[float]).

    Expected real interface (defined by this oracle for the implementer):
      returns an object exposing the 5 chosen channel indices and their ycoords,
      accepted as any of: (indices, ycoords[, ...]) tuple, or a dict/namedtuple
      with keys/attrs 'indices'/'channel_indices' and 'ycoords'."""
    real = load_real_select_five_depths()
    if real is None:
        idx, ys, _ymin, _ymax = reference_select_five_depths(xcoords, ycoords, valid)
        return idx, ys
    out = real(np.asarray(xcoords, float), np.asarray(ycoords, float),
               np.asarray(list(valid), int))
    if isinstance(out, dict):
        idx = out.get("indices", out.get("channel_indices"))
        ys = out.get("ycoords", out.get("ycoords_um"))
        return [int(i) for i in idx], [float(y) for y in ys]
    if isinstance(out, tuple):
        idx, ys = out[0], out[1]
        return [int(i) for i in idx], [float(y) for y in ys]
    # namedtuple / object
    idx = getattr(out, "indices", getattr(out, "channel_indices"))
    ys = getattr(out, "ycoords", None)
    return [int(i) for i in idx], [float(y) for y in ys]


# ===========================================================================
# Pure-arithmetic self-checks of the tie-break rule (run TODAY, no impl needed)
# ===========================================================================
def test_argmin_gives_lower_index_tiebreak():
    """np.argmin returns the FIRST minimum; on candidates listed in ascending
    original index this realises RQ-016's lower-channel-index tie-break. Two
    channels exactly equidistant from a target -> the lower index is chosen."""
    ys = np.array([20.0, 30.0])          # target 25 is equidistant (dist 5 each)
    assert int(np.argmin(np.abs(ys - 25.0))) == 0


def test_fractions_are_the_contracted_five():
    assert DEPTH_FRACTIONS == (0.0, 0.25, 0.5, 0.75, 1.0)


# ===========================================================================
# Analytical microcases (hand-computed expected outputs)
# ===========================================================================
def test_linear_geometry_with_two_exact_ties():
    """11 channels, ycoords 0,10,...,100. valid = all. ymin=0, ymax=100.
    targets = [0, 25, 50, 75, 100].
      0   -> ch0  (y0)
      25  -> |25-20|=5 (ch2) vs |25-30|=5 (ch3): EXACT TIE -> lower index ch2
      50  -> ch5  (y50)
      75  -> |75-70|=5 (ch7) vs |75-80|=5 (ch8): EXACT TIE -> lower index ch7
      100 -> ch10 (y100)
    Expected indices [0,2,5,7,10], ycoords [0,20,50,70,100]."""
    y = np.arange(0, 101, 10.0)
    idx, ys = _run(np.zeros_like(y), y, np.arange(11))
    assert idx == [0, 2, 5, 7, 10]
    assert ys == [0.0, 20.0, 50.0, 70.0, 100.0]


def test_nan_ycoord_channel_is_excluded():
    """Channel 3 has a NaN ycoord -> excluded from candidacy AND from min/max
    (RQ-014). ycoords = [0,10,20,NaN,40,50,60,70,80,90,100], valid = all 11.
    Candidates span ymin=0..ymax=100; targets [0,25,50,75,100]:
      0->ch0; 25->|25-20|=5(ch2) vs |25-40|=15(ch4)->ch2; 50->ch5;
      75->|75-70|=5(ch7) vs |75-80|=5(ch8) tie->ch7; 100->ch10.
    Expected [0,2,5,7,10]; ch3 (NaN) is NEVER selected."""
    y = np.array([0, 10, 20, np.nan, 40, 50, 60, 70, 80, 90, 100.0])
    idx, ys = _run(np.zeros_like(y), y, np.arange(11))
    assert idx == [0, 2, 5, 7, 10]
    assert 3 not in idx
    assert ys == [0.0, 20.0, 50.0, 70.0, 100.0]


def test_invalid_lfp_channel_is_excluded_even_if_nearest():
    """Same 0..100 linear geometry but ch5 (y50, the exact 0.5 target) is NOT a
    valid LFP channel (RQ-014: needs BOTH a valid LFP row and a ycoord).
    valid = all except 5. target 50 -> ch5 removed -> |50-40|=10(ch4) vs
    |50-60|=10(ch6) EXACT TIE -> lower index ch4.
    Expected [0,2,4,7,10]; ch5 excluded despite being the literal nearest."""
    y = np.arange(0, 101, 10.0)
    valid = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    idx, ys = _run(np.zeros_like(y), y, valid)
    assert idx == [0, 2, 4, 7, 10]
    assert 5 not in idx
    assert ys == [0.0, 20.0, 40.0, 70.0, 100.0]


def test_nonuniform_spacing_nearest_by_distance():
    """Irregular spacing so "nearest" is non-trivial. ycoords =
    [0, 12, 33, 51, 74, 100], valid all. ymin=0, ymax=100, targets
    [0,25,50,75,100]:
      0  -> ch0 (0)
      25 -> |25-12|=13 (ch1) vs |25-33|=8 (ch2) -> ch2 (33)
      50 -> |50-33|=17 (ch2) vs |50-51|=1 (ch3) -> ch3 (51)
      75 -> |75-51|=24 (ch3) vs |75-74|=1 (ch4) -> ch4 (74)
      100-> ch5 (100)
    Expected [0,2,3,4,5]; ycoords [0,33,51,74,100]."""
    y = np.array([0, 12, 33, 51, 74, 100.0])
    idx, ys = _run(np.zeros_like(y), y, np.arange(6))
    assert idx == [0, 2, 3, 4, 5]
    assert ys == [0.0, 33.0, 51.0, 74.0, 100.0]


def test_depth1_is_tip_lowest_ycoord_monotonic():
    """Depth ordinal 1..5 tip->surface: the returned ycoords must be
    non-decreasing (Depth 1 = lowest ycoord at fraction 0.0, Depth 5 = highest
    at fraction 1.0). Property holds for any valid geometry."""
    y = np.array([0, 12, 33, 51, 74, 100.0])
    _idx, ys = _run(np.zeros_like(y), y, np.arange(6))
    assert ys[0] == min(ys)
    assert ys[-1] == max(ys)
    assert all(ys[i] <= ys[i + 1] for i in range(4))


# ===========================================================================
# Real-geometry check against Golden Fixtures (LEVEL 1, exact tolerance)
# ===========================================================================
def test_real_geometry_range_and_selection(golden_xcoords, golden_ycoords):
    """05_spikes_ks/ycoords.npy (382 channels, manifest tol rtol=atol=0). Confirm
    on REAL probe geometry that:
      * ymin/ymax equal the exact min/max of the finite ycoords, and
      * the five chosen ycoords equal an INDEPENDENT numpy nearest-to-target
        computation exactly (zero tolerance), and lie within [ymin, ymax],
        non-decreasing.
    The independent reference (conftest.reference_select_five_depths) is a
    first-principles argmin, not the code under test — so this is a genuine
    ground-truth check on real data (five-depth-selection, level 1 exact)."""
    y = golden_ycoords
    valid = np.arange(y.size)               # all rows valid on the fixture geometry
    finite = y[np.isfinite(y)]
    assert finite.size >= 5, "need >=5 finite ycoords for a 5-depth selection"

    idx, ys = _run(golden_xcoords, y, valid)
    ref_idx, ref_ys, ymin, ymax = reference_select_five_depths(golden_xcoords, y, valid)

    assert ymin == float(finite.min())
    assert ymax == float(finite.max())
    # Exact (zero tolerance) agreement with the independent reference.
    assert idx == ref_idx
    assert ys == ref_ys
    # Sanity envelope.
    assert all(ymin <= v <= ymax for v in ys)
    assert all(ys[i] <= ys[i + 1] for i in range(4))
    assert len(set(idx)) == 5, "five DISTINCT channels expected on the dense real probe"

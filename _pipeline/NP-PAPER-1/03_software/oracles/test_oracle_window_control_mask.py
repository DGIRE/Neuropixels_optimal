# ORACLE
"""
Synthetic-ground-truth microcases for the NOT-YET-IMPLEMENTED
build_window_control_mask(D, window_starts, window_len_s).

CONTRACT TARGET
  Feeds RESULT-qc-counts (n_windows_excluded ethanol) — ACCEPTANCE LEVEL 1
    (exact integers) — and RESULT-qc-discards (ethanol span boundaries, level 1
    exact, "start_s/end_s within one LFP sample").
  inclusion_exclusion rule (PROH-008 / CON-002):
    "Control epochs only: every analysis window must lie ENTIRELY within control
     (mean-subtracted ETH <= 0.05). Any window containing >=1 ethanol sample is
     excluded."
  Ethanol rule reused VERBATIM (PROH-003/PROH-008/CON-015):
    ETH_ms = ETH - mean(ENTIRE ETH); ethanol wherever ETH_ms > 0.05 (STRICT).

WINDOW CONVENTION (stated explicitly by this oracle — the wrapper is new and the
contract does not pin endpoint semantics beyond "any overlap excludes"):
  window i spans the HALF-OPEN interval [t0_i, t0_i + window_len_s) on the LFP
  clock. An ethanol sample (LV clock, time = idx/LV_Fs; shared recording clock)
  at exactly t0 is INSIDE window i; one at exactly t0+len belongs to the next
  window. Definitive pass/fail cases below use ethanol spans that straddle a
  window boundary with margin, so their outcome is convention-independent; a
  separate test documents the half-open edge for the single-sample case.

All expected pass/fail values are hand-derived from a synthetic ETH trace whose
ethanol excursion locations are placed by construction and were reconfirmed with
a stand-alone numpy overlap check at oracle-design time. Nothing was produced by
running build_window_control_mask.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    load_real_build_window_control_mask,
    reference_eth_contact_mask,
    reference_window_control_mask,
)


def _make_D(eth, lv_fs):
    return {"ETH": np.asarray(eth, dtype=np.float64), "LV_Fs": float(lv_fs)}


def _run(eth, lv_fs, window_starts_s, window_len_s):
    """Call REAL build_window_control_mask if implemented, else the reference.

    Expected real interface (defined here for the implementer): given D (with
    'ETH','LV_Fs'), window start times (s, LFP clock) and window length (s),
    return a boolean array (len == n_windows) where True == control-eligible
    (window contains zero ethanol samples). Accepts a bare ndarray or a dict
    exposing 'control_mask'/'window_control_mask'."""
    real = load_real_build_window_control_mask()
    if real is None:
        return reference_window_control_mask(eth, lv_fs, window_starts_s, window_len_s)
    D = _make_D(eth, lv_fs)
    out = real(D, np.asarray(window_starts_s, float), float(window_len_s))
    if isinstance(out, dict):
        for k in ("control_mask", "window_control_mask", "mask"):
            if k in out:
                return np.asarray(out[k], dtype=bool)
        raise KeyError("no control-mask key in build_window_control_mask output")
    return np.asarray(out, dtype=bool)


def _eth_with_excursion(n, lv_fs, span_slices):
    """Baseline-0 ETH with unit-height ethanol excursions at the given index
    slices. Because the excursion occupies few samples, mean(ETH) is small and
    positive, so baseline samples have ETH_ms ~ -mean (<= 0.05, control) and
    excursion samples have ETH_ms ~ 1-mean (> 0.05, ethanol) with wide margin."""
    eth = np.zeros(n, dtype=np.float64)
    for sl in span_slices:
        eth[sl] = 1.0
    return eth


# ===========================================================================
# Ethanol-rule self-check (runs today; anchors the reused detect_eth_contact rule)
# ===========================================================================
def test_eth_rule_matches_mean_subtract_strict_threshold():
    """The reused rule: ETH_ms = ETH - mean; ethanol = ETH_ms > 0.05 (strict).
    Zero-mean-by-construction trace (2*(-0.109)+2*0.049+2*0.06 == 0) so
    ETH_ms == ETH exactly at the boundary; the 0.049 samples (< 0.05) are
    control and the 0.06 samples (> 0.05) are ethanol."""
    eth = np.array([-0.109, 0.049, 0.06, -0.109, 0.049, 0.06])  # sum == 0 -> mean 0
    assert abs(eth.mean()) < 1e-12
    mask = reference_eth_contact_mask(eth)
    assert list(mask) == [False, False, True, False, False, True]


# ===========================================================================
# Window overlap microcases (hand-computed pass/fail)
# LV_Fs=100 Hz; ethanol at LV idx 300..320 -> times [3.00, 3.20] s inclusive.
# window_len_s = 1.0 s.
# ===========================================================================
def _fixture_trace():
    lv_fs = 100.0
    n = 1000                                  # 10 s of LV samples
    eth = _eth_with_excursion(n, lv_fs, [slice(300, 321)])  # idx 300..320
    return eth, lv_fs


def test_window_fully_avoiding_ethanol_is_eligible():
    """Windows [0,1) and [5,6) contain no ethanol samples (ethanol lives in
    [3.00,3.20] s) -> both control-eligible (True)."""
    eth, lv_fs = _fixture_trace()
    mask = _run(eth, lv_fs, [0.0, 5.0], 1.0)
    assert list(mask) == [True, True]


def test_window_fully_containing_ethanol_is_excluded():
    """Window [2.8,3.8) fully contains the ethanol span [3.00,3.20] -> excluded
    (False). This is the definitive 'full overlap' case."""
    eth, lv_fs = _fixture_trace()
    mask = _run(eth, lv_fs, [2.8], 1.0)
    assert list(mask) == [False]


def test_window_partially_overlapping_boundary_is_excluded():
    """'Any overlap excludes' (PROH-008). Window [3.15,4.15) overlaps only the
    tail of the ethanol span (ethanol times 3.15..3.20 fall inside) -> excluded
    (False), even though most of the window is control. Convention-independent
    (the overlap has finite width inside the window)."""
    eth, lv_fs = _fixture_trace()
    mask = _run(eth, lv_fs, [3.15], 1.0)
    assert list(mask) == [False]


def test_window_starting_just_after_ethanol_is_eligible():
    """Window [3.25,4.25) begins after the last ethanol sample (3.20 s) -> no
    ethanol inside -> control-eligible (True)."""
    eth, lv_fs = _fixture_trace()
    mask = _run(eth, lv_fs, [3.25], 1.0)
    assert list(mask) == [True]


def test_mixed_window_bank_exact_pattern():
    """A bank of adjacent 1-s windows tiling 0..7 s. Ethanol in [3.00,3.20] s
    intersects ONLY the [3,4) window. Expected eligibility (True=control):
      [0,1)T [1,2)T [2,3)T [3,4)F [4,5)T [5,6)T [6,7)T
    Exactly one window excluded -> n_windows_excluded(ethanol) == 1 (level 1)."""
    eth, lv_fs = _fixture_trace()
    starts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    mask = _run(eth, lv_fs, starts, 1.0)
    assert list(mask) == [True, True, True, False, True, True, True]
    assert int((~np.asarray(mask)).sum()) == 1


def test_two_separate_ethanol_spans_exclude_two_windows():
    """Two ethanol excursions: idx 150..160 ([1.50,1.60]s) and idx 720..730
    ([7.20,7.30]s). Tiling 0..9 s in 1-s windows excludes the [1,2) and [7,8)
    windows only -> n_windows_excluded == 2, boundaries recoverable for
    RESULT-qc-discards."""
    lv_fs = 100.0
    n = 1000
    eth = _eth_with_excursion(n, lv_fs, [slice(150, 161), slice(720, 731)])
    starts = list(np.arange(0.0, 9.0, 1.0))
    mask = _run(eth, lv_fs, starts, 1.0)
    excluded = np.flatnonzero(~np.asarray(mask))
    assert list(excluded) == [1, 7]
    assert int((~np.asarray(mask)).sum()) == 2


def test_no_ethanol_all_windows_eligible():
    """A trace with NO sample above the mean-subtracted 0.05 threshold (constant
    trace collapses to exactly 0 after mean subtraction, which is at/below the
    strict threshold) -> every window control-eligible."""
    lv_fs = 100.0
    eth = np.full(1000, 0.7)                 # constant -> ETH_ms == 0 everywhere
    assert not np.any(reference_eth_contact_mask(eth))
    mask = _run(eth, lv_fs, [0.0, 1.0, 2.0], 1.0)
    assert list(mask) == [True, True, True]


def test_half_open_boundary_convention_single_sample():
    """DOCUMENTED CONVENTION (half-open [t0, t0+len)): a single ethanol sample
    at exactly t0+len is NOT in window i. Construct one ethanol sample at LV idx
    400 (time 4.00 s). Window [3.0,4.0) ends exactly at 4.00 -> the 4.00 sample
    is excluded from it -> [3,4) is eligible (True); window [4.0,5.0) starts at
    4.00 -> contains it -> excluded (False). This pins the endpoint semantics
    the implementer must adopt so RESULT-qc-discards start_s/end_s are
    reproducible to within one LFP sample."""
    lv_fs = 100.0
    eth = _eth_with_excursion(1000, lv_fs, [slice(400, 401)])  # single sample @4.00s
    mask = _run(eth, lv_fs, [3.0, 4.0], 1.0)
    assert list(mask) == [True, False]

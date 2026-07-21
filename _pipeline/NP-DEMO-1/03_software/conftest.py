"""
conftest.py — NP-DEMO-1 oracle test support
--------------------------------------------
Written by the oracle designer BEFORE np_demo1_analysis.py exists (Blueprint v2
P3). Provides:

  * sys.path wiring so `import np_demo1_analysis` resolves to the sibling
    03_software/np_demo1_analysis.py once it is written, and so the validated
    kernel (`analyses._common.matlab_round`, etc.) under "Optimized Python" is
    importable for cross-checks -- WITHOUT importing or executing the new
    analysis module itself.
  * acceptance-level pytest markers (registered so `pytest -m` selection works
    and no "unknown mark" warnings are raised), mapped to the contract's
    acceptance_criteria (contract_spec.yaml, NP-DEMO-1, contract_version 1):
        RESULT-mrl        -> level 3 (distributional, MRL within 1e-6 on fixture slice)
        RESULT-phase       -> level 3 (circular distance < 1e-6 on fixture slice)
        RESULT-stat        -> level 3 (same significance decision on full workload)
        RESULT-qc-counts   -> level 1 (exact integer counts)
  * a `mock_D` fixture: a minimal, internally-consistent synthetic D-like dict
    (SNF_PH, ETH_thr, spikeTimes, sp.clu, unitIDs, LV_Fs, sniff_onsets) for the
    handful of tests that need a D-shaped structure rather than a bare array.
    All values are hand-constructed here (NOT produced by compute_sniff_phase /
    compute_spike_phase / the new analysis module), so the fixture carries no
    circularity risk.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # .../Neuropixels
_OPT_PYTHON = os.path.join(_REPO_ROOT, "Optimized Python")

# 03_software first, so `import np_demo1_analysis` finds the (future) module
# living next to this conftest.
for p in (_THIS_DIR, _OPT_PYTHON):
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "result_mrl: targets RESULT-mrl (contract level 3)"
    )
    config.addinivalue_line(
        "markers", "result_phase: targets RESULT-phase (contract level 3)"
    )
    config.addinivalue_line(
        "markers", "result_stat: targets RESULT-stat (contract level 3)"
    )
    config.addinivalue_line(
        "markers", "result_qc_counts: targets RESULT-qc-counts (contract level 1)"
    )
    config.addinivalue_line(
        "markers", "metamorphic: metamorphic-relation test (no fixed oracle value)"
    )


@pytest.fixture
def mock_D() -> dict:
    """A minimal, hand-built D-like dict spanning two sniff cycles, one gap,
    and two ETH_thr conditions -- entirely independent of any kernel/new-code
    output.

    Layout (LV_Fs = 100.0 Hz, 200 samples == 2.0 s):
      [0:50)    sniff cycle 1, phase ramp 0 -> 0.98   (control block, ETH_thr=0.11)
      [50:100)  gap, SNF_PH == -1                     (control block, ETH_thr=0.11)
      [100:150) sniff cycle 2, phase ramp 0 -> 0.98    (ethanol block, ETH_thr=0.30)
      [150:200) gap, SNF_PH == -1                      (ethanol block, ETH_thr=0.30)

    Two units: unit 10 fires plausibly (>=0.1 Hz, will have >=50 valid-sniff
    spikes across both cycles if sampled every LV tick); unit 20 fires only a
    handful of spikes total (sub-threshold rate), used to exercise exclusion
    in any integration-level test the implementer chooses to add.
    """
    LV_Fs = 100.0
    n_lv = 200

    SNF_PH = -np.ones(n_lv, dtype=np.float64)
    SNF_PH[0:50] = np.arange(50, dtype=np.float64) / 50.0
    SNF_PH[100:150] = np.arange(50, dtype=np.float64) / 50.0

    ETH_thr = np.empty(n_lv, dtype=np.float64)
    ETH_thr[0:100] = 0.11   # control (floor-clipped, not > 0.11)
    ETH_thr[100:200] = 0.30  # ethanol (> 0.11)

    sniff_onsets = np.array([1.0, 101.0])  # 1-based, matches kernel convention

    # unit 10: one spike per LV sample across both valid cycles -> 100 spikes,
    # all inside a sniff cycle -> plenty above the 50-valid-spike / 0.1 Hz floor.
    cyc1 = np.arange(0, 50)
    cyc2 = np.arange(100, 150)
    unit10_idx = np.concatenate([cyc1, cyc2])
    unit10_times = unit10_idx / LV_Fs

    # unit 20: 3 spikes total over the whole 2 s window -> rate 1.5 Hz (would
    # pass the rate floor) but far below the 50-valid-spike floor per condition.
    unit20_times = np.array([0.02, 0.03, 1.02])

    spikeTimes = np.concatenate([unit10_times, unit20_times])
    clu = np.concatenate([
        np.full(unit10_times.size, 10, dtype=np.int64),
        np.full(unit20_times.size, 20, dtype=np.int64),
    ])
    unitIDs = np.array([10, 20], dtype=np.int64)

    return dict(
        LV_Fs=LV_Fs,
        SNF_PH=SNF_PH,
        ETH_thr=ETH_thr,
        sniff_onsets=sniff_onsets,
        spikeTimes=spikeTimes,
        sp={"clu": clu},
        unitIDs=unitIDs,
    )

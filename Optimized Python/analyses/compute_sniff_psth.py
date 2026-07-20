"""
analyses/compute_sniff_psth.py  —  port of MATLAB new analyses/compute_sniff_psth.m
Fixture stage 11.  Reads D['spikeTimes'], D['sp']['clu'], D['unitIDs'],
D['sniff_onsets_s'], D['sniff_dur_s'].  Returns (psth, centers, n_events) and,
when attach=True, also writes D['psth_phase'|'psth_ms'|'centers_phase'|
'centers_ms'|'n_events'] to match the golden fixture layout (both axis modes).

Each sniff onset is an event; a fixed window [0, median_cycle_duration] is
binned per unit at bin_ms and event-averaged to spikes/s.  Normalized-phase
mode and ms mode share identical binning — only the x-axis label differs
(compute_sniff_psth.m:49-53).

OPTIMIZATION (OPT.vectorized_psth, blueprint R7/R19): the faithful path calls
psthAndBA once per unit (nUnits * nEvents mex-equivalent histogram passes); the
optimized path computes all units in one vectorized spike->event expansion.
Both use the exact ordhist boundary semantics (see psth_utils).
"""
from __future__ import annotations

import numpy as np

from .psth_utils import psth_and_ba, psth_matrix_vectorized


def compute_sniff_psth(D: dict, bin_ms: float = 10.0, use_ms: bool = False,
                       *, attach: bool = True, optimized: bool | None = None):
    """Port of compute_sniff_psth.m.

    Returns (psth, centers, n_events).  ``use_ms`` picks the returned center
    axis (ms vs normalized phase); the psth values are identical either way.
    When ``attach`` is True, both axis variants are written into a copy of D.
    """
    if optimized is None:
        from optconfig import OPT
        optimized = OPT.vectorized_psth

    onset_s = np.asarray(D["sniff_onsets_s"], dtype=np.float64).ravel()
    med_dur_s = float(D["sniff_dur_s"])
    bin_s = bin_ms / 1000.0
    window = (0.0, med_dur_s)

    spikeTimes = np.asarray(D["spikeTimes"], dtype=np.float64).ravel()
    clu = np.asarray(D["sp"]["clu"]).ravel()
    unitIDs = np.asarray(D["unitIDs"]).ravel()
    n_events = onset_s.size

    if optimized:
        psth, bins_s, n_events = psth_matrix_vectorized(
            spikeTimes, clu, unitIDs, onset_s, window, bin_s)
    else:
        # BASELINE: one psthAndBA call per unit (original MATLAB behavior).
        _, bins_s = psth_and_ba(
            spikeTimes[clu == unitIDs[0]], onset_s, window, bin_s)
        psth = np.zeros((unitIDs.size, bins_s.size), dtype=np.float64)
        for u in range(unitIDs.size):
            st = spikeTimes[clu == unitIDs[u]]
            psth[u, :], _ = psth_and_ba(st, onset_s, window, bin_s)

    centers_ms = bins_s * 1000.0
    centers_phase = bins_s / med_dur_s
    centers = centers_ms if use_ms else centers_phase

    if attach:
        out = dict(D)
        out["psth_phase"] = psth
        out["psth_ms"] = psth
        out["centers_phase"] = centers_phase
        out["centers_ms"] = centers_ms
        out["n_events"] = float(n_events)
        return psth, centers, n_events, out

    return psth, centers, n_events

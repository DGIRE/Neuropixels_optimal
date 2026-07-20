"""
analyses/compute_spike_phase.py  —  port of MATLAB new analyses/compute_spike_phase.m
Fixture stage 10.  Requires D['SNF_PH'], D['sniff_onsets'] (from compute_sniff_phase),
D['spikeTimes'], D['sp']['clu'], D['unitIDs'], D['LV_Fs'], D['SNF'].
Adds D['spike_SNF_PH'] (sniff phase at each spike) and D['unitMeanSniffPhase']
(mean of valid phases per unit; NaN if a unit has < 5 valid spikes).

Conventions:
  * LV<->NP time map is nearest-index: lv_idx = round(t*LV_Fs)+1, clamped to
    [1, nLV].  MATLAB round is half-away-from-zero (see _common.matlab_round);
    a banker's-rounding off-by-one would break the 1e-12 spike_SNF_PH fixture.
  * "valid" phase = SNF_PH >= 0 (the -1 sentinel marks samples outside any cycle).

OPTIMIZATION (OPT.vectorized_spike_phase, blueprint R7/R19):
  the per-unit mean is the O(nUnits x nSpikes) trap E1 fixed for the loader.
  The optimized path replaces the 379x full-array boolean scan with one grouped
  reduction (np.bincount over the spike->unit index) — bit-faithful means, NaN
  where < 5 valid.  Baseline loop preserved as the else branch.
"""
from __future__ import annotations

import numpy as np

from ._common import matlab_round

_MIN_VALID = 5   # MATLAB: if numel(v) >= 5


def compute_spike_phase(D: dict, optimized: bool | None = None) -> dict:
    if optimized is None:
        from optconfig import OPT
        optimized = OPT.vectorized_spike_phase

    LV_Fs = float(D["LV_Fs"])
    nLV = np.asarray(D["SNF"]).size
    st = np.asarray(D["spikeTimes"], dtype=np.float64).ravel()
    SNF_PH = np.asarray(D["SNF_PH"], dtype=np.float64).ravel()

    # nearest LabView index (0-based) with MATLAB half-away rounding, clamped.
    lv_idx = matlab_round(st * LV_Fs).astype(np.int64)
    np.clip(lv_idx, 0, nLV - 1, out=lv_idx)
    spike_SNF_PH = SNF_PH[lv_idx]

    clu = np.asarray(D["sp"]["clu"]).ravel()
    unitIDs = np.asarray(D["unitIDs"]).ravel()
    nU = unitIDs.size
    unit_mean = np.full(nU, np.nan, dtype=np.float64)

    if optimized:
        # unitIDs == unique(clu) ascending, so searchsorted maps every spike.
        uidx = np.searchsorted(unitIDs, clu)
        valid = spike_SNF_PH >= 0.0
        vu = uidx[valid]
        cnt = np.bincount(vu, minlength=nU).astype(np.float64)
        ssum = np.bincount(vu, weights=spike_SNF_PH[valid], minlength=nU)
        enough = cnt >= _MIN_VALID
        unit_mean[enough] = ssum[enough] / cnt[enough]
    else:
        # BASELINE per-unit loop (original MATLAB behavior; one-switch revert).
        for u in range(nU):
            ph = spike_SNF_PH[clu == unitIDs[u]]
            v = ph[ph >= 0.0]
            if v.size >= _MIN_VALID:
                unit_mean[u] = v.mean()

    out = dict(D)
    out["spike_SNF_PH"] = spike_SNF_PH
    out["unitMeanSniffPhase"] = unit_mean
    return out

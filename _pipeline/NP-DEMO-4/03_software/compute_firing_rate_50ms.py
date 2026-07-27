"""compute_firing_rate_50ms.py — NP-DEMO-4 new kernel function.

Compute population firing rate trace at LabView sample resolution using a
50 ms sliding window centered at each sample.

Algorithm (contract_v001.yaml preprocessing, CON-001):
  For each included unit u and each LV time point t:
    rate_u(t) = count(spikes_u with t-25ms <= spike_time <= t+25ms) / 0.05 s
  Population trace = mean(rate_u(t)) across all n_included units.

Contract constraints:
  - Window is exactly 50 ms = [t-25ms, t+25ms], CLOSED interval.
  - Denominator is always 0.05 s (no adaptive denominator near recording edges).
  - included_unit_ids passed explicitly (upstream CON-001 filter already applied).
"""
from __future__ import annotations

import numpy as np


def compute_firing_rate_50ms(
    D: dict,
    included_unit_ids: np.ndarray,
    window_ms: float = 50.0,
) -> dict:
    """Compute 50 ms sliding-window population firing rate at each LV sample.

    Parameters
    ----------
    D : dict
        Experiment dict with keys:
          'LV_Fs'     — float, LabView sample rate (Hz)
          'SNF'       — (n_LV_samples,) array (only used for length)
          'sp'        — dict with:
            'st'  — (n_spikes,) float64, spike times in seconds
            'clu' — (n_spikes,) int, cluster ID per spike
    included_unit_ids : np.ndarray
        1-D array of unit IDs that pass the CON-001 inclusion filter
        (FR >= 0.1 Hz AND spike_count >= 5000). May be empty.
    window_ms : float
        Full window width in ms (contract-pinned at 50.0; do not change).

    Returns
    -------
    dict
        Copy of D with added keys:
          'firing_rate_population'  — (n_LV_samples,) float64, mean across units (Hz)
          'firing_rate_per_unit'    — (n_included_units, n_LV_samples) float64
    """
    LV_Fs = float(D["LV_Fs"])
    n_samples = len(D["SNF"])
    sp = D["sp"]
    st_all = np.asarray(sp["st"], dtype=np.float64).ravel()
    clu_all = np.asarray(sp["clu"]).ravel()
    included_unit_ids = np.asarray(included_unit_ids).ravel()

    half_win_s = (window_ms / 2.0) / 1000.0  # 0.025 s
    denom = window_ms / 1000.0                # 0.05 s

    n_units = included_unit_ids.size
    LV_times = np.arange(n_samples, dtype=np.float64) / LV_Fs

    # Result arrays
    fr_per_unit = np.zeros((n_units, n_samples), dtype=np.float64)

    for ui, uid in enumerate(included_unit_ids):
        # Spike times for this unit
        mask = clu_all == uid
        st_unit = st_all[mask]
        if st_unit.size == 0:
            continue

        st_sorted = np.sort(st_unit)

        # For each LV sample t, count spikes in [t - half_win_s, t + half_win_s]
        # Using searchsorted for efficiency:
        #   right edge (inclusive): spike_time <= t + half_win_s
        #   left edge  (inclusive): spike_time >= t - half_win_s
        # count = searchsorted(st, t+hw, 'right') - searchsorted(st, t-hw, 'left')
        lo = LV_times - half_win_s
        hi = LV_times + half_win_s

        hi_idx = np.searchsorted(st_sorted, hi, side="right")
        lo_idx = np.searchsorted(st_sorted, lo, side="left")
        counts = hi_idx - lo_idx
        fr_per_unit[ui, :] = counts.astype(np.float64) / denom

    if n_units == 0:
        fr_population = np.zeros(n_samples, dtype=np.float64)
    else:
        fr_population = fr_per_unit.mean(axis=0)

    out = dict(D)
    out["firing_rate_population"] = fr_population
    out["firing_rate_per_unit"] = fr_per_unit
    return out

"""build_valid_sniff_mask.py -- NP-PAPER-1 additive analysis primitive
(Blueprint v2 P3, D7).

New, ADDITIVE function per contract RQ-018..021 / OUT-008. Consumes
`compute_sniff_phase`'s output (sniff_onsets_s, SNF_PH) and builds a
per-SNF_PH-sample valid-sniff mask.

HAZARD H1 (repo_map.md): ISI must be derived FRESH as `np.diff(sniff_onsets_s)`
-- a per-sniff ARRAY -- never `compute_sniff_phase`'s `sniff_dur_s`, which is
only a SCALAR (median) and is not exposed anywhere as a per-sniff series.

ALGORITHM (contract-pinned, RQ-019..021)
------------------------------------------
1. ISI = np.diff(sniff_onsets_s)  (per-sniff array, one value per inter-onset
   span).
2. p5, p95 = 5th/95th percentile of THIS experiment's ISI distribution
   (numpy default 'linear' interpolation).
3. An inter-onset span [onset_k, onset_k+1) is INVALID if its ISI is
   < p5 or > p95.
4. Any sample with SNF_PH == -1 (compute_sniff_phase's noise/outside-cycle
   sentinel) is INVALID, regardless of rule 3.
5. Samples before the first onset / after the last onset are not covered by
   any ISI span and are therefore INVALID by construction (not part of any
   retained sniff cycle), subject also to rule 4.
6. Instantaneous sniff rate = 1/ISI per inter-onset span (Hz), returned
   alongside the mask for the Fig 1C/1D overlays and RESULT-sniff-rate.

The mask is returned on the SNF_PH sample clock (length == len(SNF_PH)),
sampled at the LabView rate LV_Fs (~125 Hz) -- the shared clock
`compute_sniff_phase`, `sniff_onsets_s`, and `SNF_PH` all live on.
"""
from __future__ import annotations

import numpy as np


def build_valid_sniff_mask(D: dict) -> dict:
    """Per-sample valid-sniff mask + retained instantaneous sniff rate.

    Parameters
    ----------
    D : dict
        Must contain:
          'sniff_onsets_s' -- (n_onsets,) float64, inhalation onset times (s),
              from compute_sniff_phase.
          'SNF_PH'          -- (n,) float64, per-sample phase 0..1; -1 marks
              noise / outside-cycle samples, from compute_sniff_phase.
          'LV_Fs'           -- float, the shared sample rate (Hz) SNF_PH (and
              sniff_onsets_s) live on.

    Returns
    -------
    dict with keys:
      'valid_sniff_mask' -- (n,) bool; True == retained (in an ISI-in-range
          span AND SNF_PH != -1). n == len(D['SNF_PH']).
      'isi'               -- (n_onsets-1,) float64, per-sniff inter-onset
          intervals (s), np.diff(sniff_onsets_s).
      'p5', 'p95'         -- float, the 5th/95th percentile ISI cutoffs (s).
      'inst_sniff_rate'   -- (n_onsets-1,) float64, 1/isi (Hz), one value per
          inter-onset span (independent of the ISI-percentile / SNF_PH
          validity rules -- the raw instantaneous-rate series).
    """
    onsets = np.asarray(D["sniff_onsets_s"], dtype=np.float64).ravel()
    snf_ph = np.asarray(D["SNF_PH"], dtype=np.float64).ravel()
    fs = float(D["LV_Fs"])
    n = snf_ph.size

    isi = np.diff(onsets)  # HAZARD H1: fresh per-sniff array, not sniff_dur_s

    if isi.size == 0:
        p5 = float("nan")
        p95 = float("nan")
    else:
        p5 = float(np.percentile(isi, 5))
        p95 = float(np.percentile(isi, 95))

    onset_samp = np.rint(onsets * fs).astype(np.int64)

    valid = np.zeros(n, dtype=bool)  # default invalid: uncovered spans (rule 5)
    for k in range(isi.size):
        a = max(int(onset_samp[k]), 0)
        b = min(int(onset_samp[k + 1]), n)
        if b <= a:
            continue
        span_ok = (isi[k] >= p5) and (isi[k] <= p95)  # rule 3
        valid[a:b] = span_ok

    valid &= snf_ph != -1.0  # rule 4: -1 sentinel always invalidates

    with np.errstate(divide="ignore"):
        inst_sniff_rate = 1.0 / isi  # rule 6

    return {
        "valid_sniff_mask": valid,
        "isi": isi,
        "p5": p5,
        "p95": p95,
        "inst_sniff_rate": inst_sniff_rate,
    }

"""select_five_depths.py -- NP-PAPER-1 additive analysis primitive (Blueprint v2 P3, D7).

New, ADDITIVE, deterministic (level 1, exact) channel-selection function per
contract RQ-012..016 / OUT-006/OUT-007. Written independently from the oracle
suite's `reference_select_five_depths` (same contracted algorithm, authored
separately per the "independent reference" delicate-stage convention).

ALGORITHM (contract-pinned)
----------------------------
1. Candidate channels = those in `valid_lfp_channels` that ALSO have a finite
   ycoord (RQ-014: BOTH a valid LFP row and a ycoord).
2. ymin/ymax = min/max ycoord over candidates.
3. Five targets at fractions [0.00, 0.25, 0.50, 0.75, 1.00] of [ymin, ymax].
4. For each target: nearest candidate by |ycoord - target|; ties -> LOWER
   channel index (RQ-016 single-fixed-column tie-break). `np.argmin` over
   candidates listed in ascending original channel index returns the FIRST
   minimum, i.e. the lowest index, which realises this tie-break exactly.
5. Depth 1..5 labelled tip->surface (Depth 1 = lowest ycoord, fraction 0.0;
   Depth 5 = highest ycoord, fraction 1.0); ycoords are therefore
   non-decreasing across the returned five.
"""
from __future__ import annotations

import numpy as np

DEPTH_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def select_five_depths(xcoords, ycoords, valid_lfp_channels):
    """Deterministic five-depth channel selection (RQ-012..016).

    Parameters
    ----------
    xcoords : array-like, shape (nCh,)
        Channel x-positions (um). Unused by the selection rule itself
        (accepted for interface symmetry with ycoords / future geometry
        checks); ycoord is the sole basis for depth selection.
    ycoords : array-like, shape (nCh,)
        Channel y-positions (um); NaN marks a channel with no ycoord.
    valid_lfp_channels : array-like of int
        Channel indices that have a valid LFP row.

    Returns
    -------
    (indices, ycoords_sel, ymin, ymax)
        indices : list[int], length 5 -- chosen channel indices, Depth 1..5
            tip->surface (non-decreasing ycoord).
        ycoords_sel : list[float], length 5 -- ycoord (um) of each chosen
            channel.
        ymin, ymax : float -- min/max ycoord over the candidate set.
    """
    xcoords = np.asarray(xcoords, dtype=np.float64).ravel()  # noqa: F841 (kept for interface symmetry)
    ycoords = np.asarray(ycoords, dtype=np.float64).ravel()
    valid = np.asarray(list(valid_lfp_channels), dtype=np.int64).ravel()
    valid = np.sort(np.unique(valid))

    cand = valid[np.isfinite(ycoords[valid])]
    if cand.size == 0:
        raise ValueError("select_five_depths: no candidate channels with both a "
                          "valid LFP row and a finite ycoord")

    ys = ycoords[cand]
    ymin = float(ys.min())
    ymax = float(ys.max())
    span = ymax - ymin

    indices: list[int] = []
    ycoords_sel: list[float] = []
    for frac in DEPTH_FRACTIONS:
        target = ymin + frac * span
        k = int(np.argmin(np.abs(ys - target)))  # first (lowest-index) minimum
        indices.append(int(cand[k]))
        ycoords_sel.append(float(ys[k]))

    return indices, ycoords_sel, ymin, ymax

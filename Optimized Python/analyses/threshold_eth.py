"""
analyses/threshold_eth.py  —  port of MATLAB new analyses/threshold_eth.m
Fixture stage 09.  Reads D['ETH']; adds D['ETH_thr'], D['eth_threshold'].

Floor clipping: every ETH sample below the threshold is raised to the threshold
value, so the signal never dips below baseline.  The interactive picker is a
GUI; the pinned value (0.11, from File paths.txt) is passed as an argument.
"""
from __future__ import annotations

import numpy as np


def threshold_eth(D: dict, eth_threshold: float = 0.11) -> dict:
    """Port of threshold_eth.m.  ``eth_threshold`` is the pinned GUI value."""
    ETH = np.asarray(D["ETH"], dtype=np.float64).ravel()
    ETH_thr = ETH.copy()
    ETH_thr[ETH_thr < eth_threshold] = eth_threshold          # floor clip (== np.maximum)

    out = dict(D)
    out["ETH_thr"] = ETH_thr
    out["eth_threshold"] = float(eth_threshold)
    return out

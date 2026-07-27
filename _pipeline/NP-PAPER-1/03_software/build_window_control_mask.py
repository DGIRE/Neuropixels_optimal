"""build_window_control_mask.py -- NP-PAPER-1 additive analysis primitive
(Blueprint v2 P3, D7).

New, ADDITIVE wrapper required by contract PROH-003/PROH-008/CON-002. Wraps
`detect_eth_contact`'s per-LV-sample ethanol mask onto LFP-clock analysis
windows: a window is control-eligible ONLY IF its half-open span
[t0, t0+window_len_s) contains ZERO ethanol samples (shared recording clock,
ethanol sample times = idx/LV_Fs). This is a thin per-window mapping -- it
introduces NO new threshold; the ethanol-detection RULE itself is reused
unchanged from `detect_eth_contact.py` (this task's own per-task copy).

WINDOW CONVENTION: half-open [t0, t0+window_len_s). An ethanol sample at
exactly t0 is INSIDE window i; one at exactly t0+window_len_s belongs to the
next window (matches the "start_s/end_s within one LFP sample" edge
semantics required for RESULT-qc-discards).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent


def _ensure_on_path(p: Path) -> None:
    s = str(p)
    if s not in sys.path and p.is_dir():
        sys.path.insert(0, s)


_ensure_on_path(_THIS_DIR)

from detect_eth_contact import detect_eth_contact  # noqa: E402


def build_window_control_mask(D: dict, window_starts, window_len_s: float) -> np.ndarray:
    """Per-window "all-control" (zero-ethanol) eligibility mask.

    Parameters
    ----------
    D : dict
        Must contain 'ETH' (1-D ethanol trace) and 'LV_Fs' (its sample rate,
        Hz). The shared recording clock means ethanol sample i occurs at
        time i/LV_Fs seconds, directly comparable to `window_starts` (LFP
        clock, also in seconds on the same shared clock).
    window_starts : array-like of float
        Window start times t0 (s), one per analysis window.
    window_len_s : float
        Window length (s); each window spans the half-open interval
        [t0, t0+window_len_s).

    Returns
    -------
    np.ndarray, dtype bool, shape (len(window_starts),)
        True == control-eligible (the window contains zero ethanol samples).
    """
    out = detect_eth_contact(D)
    eth_contact_mask = out["eth_contact_mask"]
    lv_fs = float(D["LV_Fs"])

    eth_times = np.flatnonzero(eth_contact_mask) / lv_fs
    starts = np.asarray(window_starts, dtype=np.float64).ravel()
    control_mask = np.ones(starts.size, dtype=bool)

    if eth_times.size == 0:
        return control_mask

    eth_times = np.sort(eth_times)
    ends = starts + float(window_len_s)
    # For each window, control-eligible iff no ethanol time lies in [t0, t1).
    # searchsorted gives, for each window, the first ethanol-time index >= t0
    # and the first index >= t1; if these differ, at least one ethanol time
    # falls in [t0, t1).
    lo = np.searchsorted(eth_times, starts, side="left")
    hi = np.searchsorted(eth_times, ends, side="left")
    control_mask = lo == hi
    return control_mask

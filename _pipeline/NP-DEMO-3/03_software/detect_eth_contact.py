"""detect_eth_contact.py — NP-DEMO-3 additive kernel function (Blueprint v2 P3, D7).

New analysis primitive required by the NP-DEMO-3 contract
(contract_v001.yaml, new_kernel_functions_required). This is an ADDITIVE
module that lives beside the validated kernel under "Optimized Python"; it does
NOT modify, copy, or reimplement any kernel numerics, golden fixtures, or
tolerance config.

NP-DEMO-3 uses `detect_eth_contact` — a DIFFERENT algorithm from the kernel's
`threshold_eth` (which is deliberately NOT imported anywhere in this pipeline).

Algorithm (contract-pinned, fixed rule for all 6 sessions — no per-experiment
adjustment):
  1. ETH_ms = ETH - mean(ETH)            (mean of the ENTIRE trace, no windowing)
  2. eth_contact_mask = ETH_ms > eth_threshold   (STRICTLY above = ethanol;
     at/below = control)
  3. n_trials = count of discrete contiguous above-threshold runs
"""
from __future__ import annotations

import numpy as np


def _count_contiguous_runs(mask: np.ndarray) -> int:
    """Count discrete contiguous True runs in a 1-D boolean array.

    Empty-safe (returns 0 for a length-0 or all-False mask). Uses a padded
    rising-edge count so it never indexes mask[0] on an empty array.
    """
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size == 0:
        return 0
    padded = np.concatenate(([False], mask, [False]))
    rising = ~padded[:-1] & padded[1:]
    return int(np.count_nonzero(rising))


def detect_eth_contact(D: dict, eth_threshold: float = 0.05) -> dict:
    """Detect ethanol-contact periods from the ETH trace.

    Parameters
    ----------
    D : dict
        Experiment dict containing key 'ETH' (1-D ethanol sensor trace).
    eth_threshold : float, default 0.05
        Strict upper threshold on the mean-subtracted trace. Fixed for all
        sessions.

    Returns
    -------
    dict
        Copy of D with added keys:
          'ETH_ms'           — ETH - mean(ETH)                     (n,) float64
          'eth_contact_mask' — ETH_ms > eth_threshold (bool)       (n,)
          'n_trials'         — count of contiguous above-thresh runs  int
          'eth_threshold'    — the threshold used                     float
    """
    ETH = np.asarray(D["ETH"], dtype=np.float64).ravel()
    ETH_ms = ETH - ETH.mean()
    eth_contact_mask = ETH_ms > eth_threshold
    n_trials = _count_contiguous_runs(eth_contact_mask)

    out = dict(D)
    out["ETH_ms"] = ETH_ms
    out["eth_contact_mask"] = eth_contact_mask
    out["n_trials"] = int(n_trials)
    out["eth_threshold"] = float(eth_threshold)
    return out

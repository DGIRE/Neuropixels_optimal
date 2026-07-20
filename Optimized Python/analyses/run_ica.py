"""
analyses/run_ica.py  —  port of MATLAB new analyses/run_ICA.m
NO golden fixture (stochastic).  This is a Level-3 (statistical) analysis:
FastICA is defined only up to component sign and permutation, so it is NOT
bit-comparable to MATLAB.  Faithfulness here means structural fidelity to the
algorithm plus reproducibility; correctness is checked by invariants
(whitening -> unit-variance Z, unmixing reconstructs X, unit-norm rows), not by
a stored reference.

Reproducibility (blueprint R15 / extension guide §8): the FastICA init vectors
are PRE-DRAWN in component order from a seeded RNG, so a given (data, seed) is
deterministic.  MATLAB's randn stream cannot be reproduced in NumPy, and the
original script never fixes MATLAB's seed, so no cross-tool bit match exists or
is expected.

Reads D['spikeTimes'], D['sp']['clu'], D['unitIDs'], D['SNF_PH'], D['ETH_thr'],
D['LV_Fs'].  Adds D['ICA'] (dict mirroring the MATLAB D.ICA struct).
"""
from __future__ import annotations

import numpy as np

from ._common import spike_rate_matrix, rank_by_abs_desc

SIGMA_MS = 50.0        # run_ICA.m fixes sigma = 50 ms
_TOL_ICA = 1e-6
_MAX_ITER = 500


def _fastica_deflation(Z: np.ndarray, n_components: int, inits: np.ndarray):
    """FastICA, logcosh nonlinearity, deflation — run_ICA.m:105-137.

    Z : [n_components x nSamp] whitened data.  inits[p] is the pre-drawn
    random init for component p (already the right length).  Returns W
    [n_components x n_components] in whitened space.
    """
    nSamp = Z.shape[1]
    W = np.eye(n_components, dtype=np.float64)
    for p in range(n_components):
        w = inits[p].astype(np.float64).copy()
        w = w / np.linalg.norm(w)
        for _ in range(_MAX_ITER):
            u = w @ Z                     # 1 x nSamp
            g = np.tanh(u)
            gp = 1.0 - g * g
            w_new = (Z @ g) / nSamp - gp.mean() * w
            # deflate against already-found rows W[0:p]
            for q in range(p):
                w_new = w_new - (w_new @ W[q]) * W[q]
            w_new = w_new / np.linalg.norm(w_new)
            if abs(abs(w_new @ w) - 1.0) < _TOL_ICA:
                w = w_new
                break
            w = w_new
        W[p, :] = w
    return W


def run_ica(D: dict, n_components: int | None = None, *, seed: int = 0,
            optimized: bool | None = None) -> dict:
    """Port of run_ICA.m.  ``seed`` pins the pre-drawn FastICA inits."""
    if optimized is None:
        from optconfig import OPT
        optimized = OPT.vectorized_rate_matrix

    LV_Fs = float(D["LV_Fs"])
    nSamp = np.asarray(D["SNF_PH"]).size
    unitIDs = np.asarray(D["unitIDs"]).ravel()
    nUnits = unitIDs.size
    if n_components is None:
        n_components = min(nUnits, 20)
    n_components = min(n_components, nUnits)

    X = spike_rate_matrix(D["spikeTimes"], D["sp"]["clu"], unitIDs,
                          LV_Fs, nSamp, SIGMA_MS, vectorized=optimized)

    # centre + whiten (econ SVD)
    Xc = X - X.mean(axis=1, keepdims=True)
    U, s, _Vt = np.linalg.svd(Xc, full_matrices=False)
    keep = s > (1e-6 * s[0])
    U = U[:, keep]
    s = s[keep]
    n_components = min(n_components, int(keep.sum()))
    U = U[:, :n_components]
    s = s[:n_components]
    Z = (U / s).T @ Xc                       # diag(1/s) U' Xc  -> [n_comp x nSamp]

    # pre-draw inits in component order (reproducible)
    rng = np.random.RandomState(seed)
    inits = np.array([rng.standard_normal(n_components) for _ in range(n_components)])

    W = _fastica_deflation(Z, n_components, inits)
    S_ica = W @ Z
    W_orig = W @ (U / s).T                    # back to unit space [n_comp x nUnits]

    # correlations with SNF_PH (valid phase only) and ETH_thr (all samples)
    SNF_PH = np.asarray(D["SNF_PH"], dtype=np.float64).ravel()
    ETH_thr = np.asarray(D["ETH_thr"], dtype=np.float64).ravel()
    valid = SNF_PH >= 0.0
    corr_SNF = np.array([np.corrcoef(SNF_PH[valid], S_ica[ic, valid])[0, 1]
                         for ic in range(n_components)])
    corr_ETH = np.array([np.corrcoef(ETH_thr, S_ica[ic, :])[0, 1]
                         for ic in range(n_components)])

    out = dict(D)
    out["ICA"] = {
        "S": S_ica,
        "W": W_orig,
        "corr_SNF": corr_SNF,
        "corr_ETH": corr_ETH,
        "rank_SNF": rank_by_abs_desc(corr_SNF),
        "rank_ETH": rank_by_abs_desc(corr_ETH),
        "n_components": n_components,
        "LV_Fs": LV_Fs,
        "seed": seed,
    }
    return out

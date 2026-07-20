"""
analyses/run_decomposition.py  —  port of MATLAB new analyses/run_decomposition.m
NO golden fixture (ICA branch is stochastic).  Configures ICA / PCA / Raw
decompositions of the Gaussian-smoothed spike-rate matrix, then computes
zero-lag 'coeff' cross-correlation of every component with the SNF and ETH
reference signals for each (method x ETH x SNF) combination.

The MATLAB checkbox dialog is replaced by explicit arguments (extension guide
§8: GUI -> parameters).  Defaults reproduce the dialog defaults: ICA only,
processed ETH + processed SNF, sigma 50 ms, n_components = min(nUnits, 20).

Reads D['spikeTimes'], D['sp']['clu'], D['unitIDs'], D['SNF'], D['SNF_PH'],
D['ETH'], D['ETH_thr'], D['LV_Fs'].  Adds D['DECOMP'] (list of dicts, one per
combination, mirroring the MATLAB D.DECOMP struct array).
"""
from __future__ import annotations

import numpy as np

from ._common import spike_rate_matrix, zero_lag_xcorr, rank_by_abs_desc
from .run_ica import _fastica_deflation


def _decompose(method, X, n_components_req, seed):
    """Return (S_out, comp_label, comp_ids, n_comp) for one method.
    ICA/PCA share the same centre+whiten; Raw returns X unchanged."""
    if method == "Raw":
        return X, "Unit", None, X.shape[0]

    Xc = X - X.mean(axis=1, keepdims=True)
    U, s, _Vt = np.linalg.svd(Xc, full_matrices=False)
    keep = s > (1e-6 * s[0])
    n_comp = min(n_components_req, int(keep.sum()))
    U = U[:, :n_comp]
    sv = s[:n_comp]

    if method == "PCA":
        S_out = U.T @ Xc                       # [n_comp x nSamp]
        return S_out, "PC", np.arange(1, n_comp + 1), n_comp

    # ICA
    Z = (U / sv).T @ Xc
    rng = np.random.RandomState(seed)
    inits = np.array([rng.standard_normal(n_comp) for _ in range(n_comp)])
    W = _fastica_deflation(Z, n_comp, inits)
    S_out = W @ Z
    return S_out, "IC", np.arange(1, n_comp + 1), n_comp


def run_decomposition(D: dict, methods=("ICA",), eth_choices=("proc",),
                      snf_choices=("proc",), sigma_ms: float = 50.0,
                      n_components: int | None = None, *, seed: int = 0,
                      optimized: bool | None = None) -> dict:
    """Port of run_decomposition.m.  ``seed`` pins the ICA inits."""
    if optimized is None:
        from optconfig import OPT
        optimized = OPT.vectorized_rate_matrix

    LV_Fs = float(D["LV_Fs"])
    unitIDs = np.asarray(D["unitIDs"]).ravel()
    nUnits = unitIDs.size
    nSamp = np.asarray(D["SNF_PH"]).size
    if n_components is None:
        n_components = min(nUnits, 20)
    n_components = min(n_components, nUnits)

    X = spike_rate_matrix(D["spikeTimes"], D["sp"]["clu"], unitIDs,
                          LV_Fs, nSamp, sigma_ms, vectorized=optimized)

    # decompose once per method (cache), as the MATLAB does.
    cache = {}
    for m in methods:
        S_out, comp_label, comp_ids, n_comp = _decompose(m, X, n_components, seed)
        if comp_ids is None:            # Raw -> unit cluster ids
            comp_ids = unitIDs.copy()
        cache[m] = (S_out, comp_label, comp_ids, n_comp)

    SNF = np.asarray(D["SNF"], dtype=np.float64).ravel()
    SNF_PH = np.asarray(D["SNF_PH"], dtype=np.float64).ravel()
    ETH = np.asarray(D["ETH"], dtype=np.float64).ravel()
    ETH_thr = np.asarray(D["ETH_thr"], dtype=np.float64).ravel()

    results = []
    for m in methods:
        S_out, comp_label, comp_ids, n_comp = cache[m]
        for ec in eth_choices:
            eth_sig = ETH_thr if ec == "proc" else ETH
            eth_label = "Processed ETH" if ec == "proc" else "Raw ETH"
            for sc in snf_choices:
                if sc == "proc":
                    snf_sig = SNF_PH
                    valid = SNF_PH >= 0.0
                    snf_label = "Processed SNF"
                else:
                    snf_sig = SNF
                    valid = np.ones(SNF.size, dtype=bool)
                    snf_label = "Raw SNF"
                snf_masked = snf_sig[valid]

                corr_SNF = np.array([zero_lag_xcorr(S_out[ic, valid], snf_masked)
                                     for ic in range(n_comp)])
                corr_ETH = np.array([zero_lag_xcorr(S_out[ic, :], eth_sig)
                                     for ic in range(n_comp)])
                results.append({
                    "method": m,
                    "eth_choice": ec,
                    "snf_choice": sc,
                    "label": f"{m} | {eth_label} | {snf_label}",
                    "S": S_out,
                    "corr_SNF": corr_SNF,
                    "corr_ETH": corr_ETH,
                    "rank_SNF": rank_by_abs_desc(corr_SNF),
                    "rank_ETH": rank_by_abs_desc(corr_ETH),
                    "n_components": n_comp,
                    "comp_label": comp_label,
                    "comp_ids": comp_ids,
                    "LV_Fs": LV_Fs,
                    "sigma_ms": sigma_ms,
                    "eth_label": eth_label,
                    "snf_label": snf_label,
                })

    out = dict(D)
    out["DECOMP"] = results
    return out

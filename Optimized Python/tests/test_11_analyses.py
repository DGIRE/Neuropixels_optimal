"""
test_11_analyses.py
-------------------
Fixture-anchored equivalence tests for the "new analyses" ports (stages 08-11)
plus optimized-vs-baseline gates and invariant checks for the stochastic
decomposition/ICA stages.

The golden fixtures are the source of truth (blueprint D2).  Point CI_FIXTURES
at the fixture tree to run everything against real data:

    CI_FIXTURES="C:\\Projects\\Neuropixels\\translation\\Golden Fixtures" \\
        python -m pytest tests/test_11_analyses.py -q

Each fixture-anchored stage is validated with GOLDEN INPUTS (not the port's own
upstream output) so a bug in one stage cannot mask a bug in the next.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _default_gf():
    # <repo>/Golden Fixtures, sibling of "Optimized Python" (repo-relative default)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Golden Fixtures")

import optconfig  # noqa: E402
from analyses import (  # noqa: E402
    compute_sniff_phase, threshold_eth, compute_spike_phase,
    compute_sniff_psth, run_ica, run_decomposition,
)
from analyses._common import spike_rate_matrix  # noqa: E402


# --------------------------------------------------------------------------
# fixture loading helpers
# --------------------------------------------------------------------------
def _gf():
    d = os.environ.get("CI_FIXTURES") or _default_gf()
    if not d or not os.path.isdir(d):
        pytest.skip("CI_FIXTURES not set to the golden-fixture tree")
    return d


def _load(gf, rel):
    return np.load(os.path.join(gf, rel)).squeeze()


def _manifest(gf):
    with open(os.path.join(gf, "00_manifest", "manifest.json")) as fh:
        return {e["name"]: e for e in json.load(fh)}


def _assert_fixture(man, name, got, subdir):
    e = man[name]
    ref = np.load(os.path.join(_GF, subdir, e["file"].split("/")[-1])).squeeze()
    got = np.asarray(got).squeeze()
    if ref.dtype.kind in "iu" and e["rtol"] == 0 and e["atol"] == 0:
        assert np.array_equal(got.astype(np.int64), ref.astype(np.int64)), name
    else:
        assert np.allclose(got.astype(float), ref.astype(float),
                           rtol=e["rtol"], atol=e["atol"],
                           equal_nan=e["equal_nan"]), (
            f"{name}: max|d|={np.nanmax(np.abs(got.astype(float)-ref.astype(float))):.3e}")


_GF = None


@pytest.fixture(autouse=True)
def _reset_opt():
    global _GF
    _GF = _gf()
    optconfig.set_optimized()
    yield
    optconfig.set_optimized()


def _base_D(gf):
    """A D dict populated entirely from GOLDEN fixture inputs."""
    return {
        "SNF": _load(gf, "01_labview/SNF.npy").astype(np.float64),
        "ETH": _load(gf, "01_labview/ETH.npy").astype(np.float64),
        "LV_Fs": float(_load(gf, "01_labview/LV_Fs.npy")),
        "spikeTimes": _load(gf, "06_spikes_driftmap/spikeTimes.npy").astype(np.float64),
        "sp": {"clu": _load(gf, "05_spikes_ks/sp_clu.npy")},
        "unitIDs": _load(gf, "07_units/unitIDs.npy"),
        # golden stage-08 outputs used as inputs to stages 10/11:
        "SNF_PH": _load(gf, "08_sniff_phase/SNF_PH.npy").astype(np.float64),
        "sniff_onsets": _load(gf, "08_sniff_phase/sniff_onsets.npy").astype(np.float64),
        "sniff_onsets_s": _load(gf, "08_sniff_phase/sniff_onsets_s.npy").astype(np.float64),
        "sniff_dur_s": float(_load(gf, "08_sniff_phase/sniff_dur_s.npy")),
        # golden stage-09 output used by decomposition:
        "ETH_thr": _load(gf, "09_eth_threshold/ETH_thr.npy").astype(np.float64),
    }


# --------------------------------------------------------------------------
# Stage 08 — compute_sniff_phase
# --------------------------------------------------------------------------
def test_08_sniff_phase_vs_golden():
    gf = _GF
    man = _manifest(gf)
    D = compute_sniff_phase({"SNF": _load(gf, "01_labview/SNF.npy").astype(np.float64),
                             "LV_Fs": float(_load(gf, "01_labview/LV_Fs.npy"))})
    for name in ["SNF_filt", "SNF_z", "SNF_PH", "sniff_onsets",
                 "sniff_onsets_s", "sniff_dur_s", "sniff_thr"]:
        _assert_fixture(man, name, D[name], "08_sniff_phase")


# --------------------------------------------------------------------------
# Stage 09 — threshold_eth
# --------------------------------------------------------------------------
def test_09_threshold_eth_vs_golden():
    gf = _GF
    man = _manifest(gf)
    D = threshold_eth({"ETH": _load(gf, "01_labview/ETH.npy").astype(np.float64)}, 0.11)
    _assert_fixture(man, "ETH_thr", D["ETH_thr"], "09_eth_threshold")
    _assert_fixture(man, "eth_threshold", D["eth_threshold"], "09_eth_threshold")


# --------------------------------------------------------------------------
# Stage 10 — compute_spike_phase (optimized and baseline)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("opt", [True, False])
def test_10_spike_phase_vs_golden(opt):
    gf = _GF
    man = _manifest(gf)
    D = compute_spike_phase(_base_D(gf), optimized=opt)
    _assert_fixture(man, "spike_SNF_PH", D["spike_SNF_PH"], "10_spike_phase")
    _assert_fixture(man, "unitMeanSniffPhase", D["unitMeanSniffPhase"], "10_spike_phase")


def test_10_optimized_equals_baseline():
    gf = _GF
    Do = compute_spike_phase(_base_D(gf), optimized=True)
    Db = compute_spike_phase(_base_D(gf), optimized=False)
    assert np.array_equal(Do["spike_SNF_PH"], Db["spike_SNF_PH"])
    assert np.allclose(Do["unitMeanSniffPhase"], Db["unitMeanSniffPhase"],
                       rtol=1e-12, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------
# Stage 11 — compute_sniff_psth (optimized and baseline)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("opt", [True, False])
def test_11_sniff_psth_vs_golden(opt):
    gf = _GF
    man = _manifest(gf)
    _, _, n_events, D = compute_sniff_psth(_base_D(gf), bin_ms=10.0, optimized=opt)
    _assert_fixture(man, "psth_phase", D["psth_phase"], "11_sniff_psth")
    _assert_fixture(man, "psth_ms", D["psth_ms"], "11_sniff_psth")
    _assert_fixture(man, "centers_phase", D["centers_phase"], "11_sniff_psth")
    _assert_fixture(man, "centers_ms", D["centers_ms"], "11_sniff_psth")
    _assert_fixture(man, "n_events", D["n_events"], "11_sniff_psth")


def test_11_optimized_equals_baseline():
    gf = _GF
    _, _, _, Do = compute_sniff_psth(_base_D(gf), optimized=True)
    _, _, _, Db = compute_sniff_psth(_base_D(gf), optimized=False)
    assert np.allclose(Do["psth_phase"], Db["psth_phase"], rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------
# Chained integration: raw SNF/ETH -> 08 -> 10 -> 11 still matches golden
# --------------------------------------------------------------------------
def test_chain_08_10_11_end_to_end():
    gf = _GF
    man = _manifest(gf)
    D = _base_D(gf)
    # rebuild sniff phase from raw SNF (not golden), then chain downstream
    D = compute_sniff_phase({k: D[k] for k in ("SNF", "LV_Fs")})
    D["spikeTimes"] = _load(gf, "06_spikes_driftmap/spikeTimes.npy").astype(np.float64)
    D["sp"] = {"clu": _load(gf, "05_spikes_ks/sp_clu.npy")}
    D["unitIDs"] = _load(gf, "07_units/unitIDs.npy")
    D = compute_spike_phase(D)
    _assert_fixture(man, "spike_SNF_PH", D["spike_SNF_PH"], "10_spike_phase")
    _, _, _, D = compute_sniff_psth(D)
    _assert_fixture(man, "psth_phase", D["psth_phase"], "11_sniff_psth")


# --------------------------------------------------------------------------
# Stochastic stages — reproducibility + invariants (no golden fixture)
# --------------------------------------------------------------------------
def test_rate_matrix_optimized_equals_baseline():
    gf = _GF
    D = _base_D(gf)
    nSamp = D["SNF_PH"].size
    Xo = spike_rate_matrix(D["spikeTimes"], D["sp"]["clu"], D["unitIDs"],
                           D["LV_Fs"], nSamp, 50.0, vectorized=True)
    Xb = spike_rate_matrix(D["spikeTimes"], D["sp"]["clu"], D["unitIDs"],
                           D["LV_Fs"], nSamp, 50.0, vectorized=False)
    assert np.allclose(Xo, Xb, rtol=1e-8, atol=1e-10)


def test_ica_reproducible_and_structural():
    gf = _GF
    D = _base_D(gf)
    Da = run_ica(D, n_components=8, seed=0)
    Db = run_ica(D, n_components=8, seed=0)
    A, B = Da["ICA"], Db["ICA"]
    # deterministic for a fixed seed
    assert np.array_equal(A["S"], B["S"])
    # structural invariant: S = W_orig @ Xc exactly
    Xc = spike_rate_matrix(D["spikeTimes"], D["sp"]["clu"], D["unitIDs"],
                           D["LV_Fs"], D["SNF_PH"].size, 50.0, vectorized=True)
    Xc = Xc - Xc.mean(axis=1, keepdims=True)
    assert np.allclose(A["S"], A["W"] @ Xc, rtol=1e-6, atol=1e-8)
    # ranks are a permutation of 1..n; correlations finite
    n = A["n_components"]
    assert sorted(A["rank_SNF"].tolist()) == list(range(1, n + 1))
    assert np.all(np.isfinite(A["corr_SNF"])) and np.all(np.isfinite(A["corr_ETH"]))


def test_decomposition_runs_all_methods():
    gf = _GF
    D = _base_D(gf)
    D = run_decomposition(D, methods=("ICA", "PCA", "Raw"),
                          eth_choices=("proc", "raw"), snf_choices=("proc",),
                          n_components=6, seed=1)
    dec = D["DECOMP"]
    assert len(dec) == 3 * 2 * 1
    for r in dec:
        n = r["n_components"]
        assert r["corr_SNF"].size == n and r["corr_ETH"].size == n
        assert sorted(r["rank_ETH"].tolist()) == list(range(1, n + 1))
    # PCA scores are mutually orthogonal (off-diagonal Gram ~ 0 relative to diag)
    pca = next(r for r in dec if r["method"] == "PCA")
    G = pca["S"] @ pca["S"].T
    off = G - np.diag(np.diag(G))
    assert np.max(np.abs(off)) <= 1e-6 * np.max(np.abs(np.diag(G)))


# --------------------------------------------------------------------------
# one-switch revert
# --------------------------------------------------------------------------
def test_set_baseline_toggles_analysis_flags():
    optconfig.set_baseline()
    assert optconfig.OPT.vectorized_spike_phase is False
    assert optconfig.OPT.vectorized_psth is False
    assert optconfig.OPT.vectorized_rate_matrix is False
    optconfig.set_optimized()
    assert optconfig.OPT.vectorized_spike_phase is True

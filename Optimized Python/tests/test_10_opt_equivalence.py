"""
test_10_opt_equivalence.py
--------------------------
Equivalence tests for the optimized code paths (blueprint gate 4/5).

Each optimization is checked two ways:
  * OPTIMIZED vs BASELINE  (the frozen loop) — the port-to-port diff.
  * OPTIMIZED vs GOLDEN FIXTURES — the ground-truth anchor (D2).

The fixtures are the source of truth.  Point CI_FIXTURES at the golden-fixture
tree to run the E1 checks against real data:

    CI_FIXTURES="C:\\Projects\\Neuropixels\\translation\\Golden Fixtures" \\
        python -m pytest tests/test_10_opt_equivalence.py -q

E2 (byte-identical noise mask) needs no fixtures; it runs on synthetic data.
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


# --------------------------------------------------------------------------
# Helpers that reproduce the two code paths in load_experiment_data.py exactly
# (the aggregation is inline there; these mirror it so it can be unit-tested
# without the raw .dat/.bin/Kilosort inputs that are unavailable off-site).
# --------------------------------------------------------------------------

def _agg_baseline(clu, sd, sa, rec):
    uniq, cnt = np.unique(clu, return_counts=True)
    nU = len(uniq)
    ud = np.full(nU, np.nan); ua = np.full(nU, np.nan)
    for u, uid in enumerate(uniq):
        m = clu == uid
        if sd.size: ud[u] = np.mean(sd[m])
        if sa.size: ua[u] = np.mean(sa[m])
    return uniq.astype(np.int32), ud, ua, cnt.astype(np.float64) / rec


def _agg_optimized(clu, sd, sa, rec):
    uniq, cnt0 = np.unique(clu, return_counts=True)
    clu_i = clu.astype(np.intp, copy=False)
    maxid = int(clu_i.max())
    cnt_full = np.bincount(clu_i, minlength=maxid + 1)
    present = cnt_full > 0
    cnt = cnt_full[present].astype(np.float64)
    ud = np.bincount(clu_i, weights=sd.astype(np.float64), minlength=maxid + 1)[present] / cnt
    ua = np.bincount(clu_i, weights=sa.astype(np.float64), minlength=maxid + 1)[present] / cnt
    return uniq.astype(np.int32), ud, ua, cnt0.astype(np.float64) / rec


def _fixtures_dir():
    d = os.environ.get("CI_FIXTURES") or _default_gf()
    if not d or not os.path.isdir(d):
        pytest.skip("CI_FIXTURES not set to the golden-fixture tree")
    return d


# --------------------------------------------------------------------------
# E1 — vectorized per-unit aggregation
# --------------------------------------------------------------------------

def test_E1_optimized_matches_baseline_and_golden():
    gf = _fixtures_dir()
    man = {e["name"]: e for e in json.load(open(os.path.join(gf, "00_manifest", "manifest.json")))}
    clu = np.load(os.path.join(gf, "05_spikes_ks", "sp_clu.npy")).squeeze()
    sd = np.load(os.path.join(gf, "06_spikes_driftmap", "spikeDepths.npy")).squeeze()
    sa = np.load(os.path.join(gf, "06_spikes_driftmap", "spikeAmps.npy")).squeeze()
    rec = float(np.load(os.path.join(gf, "07_units", "recordingDur.npy")).squeeze())

    uid_o, ud_o, ua_o, fr_o = _agg_optimized(clu, sd, sa, rec)
    uid_b, ud_b, ua_b, fr_b = _agg_baseline(clu, sd, sa, rec)

    # vs baseline: ids/firing rate bit-identical; means within f32-mean slack
    assert np.array_equal(uid_o, uid_b)
    assert np.array_equal(fr_o, fr_b)
    assert np.allclose(ud_o, ud_b, rtol=1e-5, atol=1e-3, equal_nan=True)
    assert np.allclose(ua_o, ua_b, rtol=1e-5, atol=1e-3, equal_nan=True)

    # vs golden fixtures (ground truth), at the manifest tolerances
    for name, got in [("unitIDs", uid_o), ("unitFiringRate", fr_o),
                      ("unitDepths", ud_o), ("unitAmps", ua_o)]:
        e = man[name]
        ref = np.load(os.path.join(gf, "07_units", f"{name}.npy")).squeeze()
        if ref.dtype.kind in "iu":
            assert np.array_equal(got.astype(np.int64), ref.astype(np.int64)), name
        else:
            assert np.allclose(got.astype(float), ref.astype(float),
                               rtol=e["rtol"], atol=e["atol"], equal_nan=e["equal_nan"]), name


def test_E1_nan_group_propagates():
    # A cluster containing a NaN depth must yield NaN (matches MATLAB mean).
    clu = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    sd = np.array([1.0, np.nan, 2.0, 4.0, 6.0])
    sa = np.array([10.0, 20.0, 1.0, 2.0, 3.0])
    _, ud_o, ua_o, _ = _agg_optimized(clu, sd, sa, 1.0)
    _, ud_b, ua_b, _ = _agg_baseline(clu, sd, sa, 1.0)
    assert np.isnan(ud_o[0]) and np.isnan(ud_b[0])
    assert np.allclose(ud_o[1], ud_b[1]) and np.allclose(ua_o, ua_b)


# --------------------------------------------------------------------------
# E2 — byte-identical boolean-lookup noise mask
# --------------------------------------------------------------------------

def _mask_optimized(clu, noise_set):
    n = max(int(clu.max()), int(max(noise_set))) + 1
    is_noise = np.zeros(n, dtype=bool)
    is_noise[np.fromiter(noise_set, dtype=np.int64)] = True
    return ~is_noise[clu]


def test_E2_mask_byte_identical():
    rng = np.random.default_rng(0)
    clu = rng.integers(0, 423, 500_000)
    noise = set(rng.choice(423, 40, replace=False).tolist())
    assert np.array_equal(_mask_optimized(clu, noise), ~np.isin(clu, list(noise)))


def test_E2_noise_id_beyond_clu_max():
    # A noise cluster id larger than any spike's cluster id must not IndexError.
    clu = np.array([0, 1, 2, 2, 3], dtype=np.int64)
    noise = {2, 99}   # 99 has no spikes
    assert np.array_equal(_mask_optimized(clu, noise), ~np.isin(clu, list(noise)))


# --------------------------------------------------------------------------
# One-switch revert
# --------------------------------------------------------------------------

def test_set_baseline_and_optimized_toggle():
    optconfig.set_baseline()
    assert optconfig.OPT.vectorized_unit_aggregation is False
    assert optconfig.OPT.fast_noise_exclusion is False
    optconfig.set_optimized()
    assert optconfig.OPT.vectorized_unit_aggregation is True
    assert optconfig.OPT.fast_noise_exclusion is True

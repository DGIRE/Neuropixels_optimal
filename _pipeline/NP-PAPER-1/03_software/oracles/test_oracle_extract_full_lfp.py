# ORACLE
"""
Oracle for the NOT-YET-IMPLEMENTED
extract_full_lfp(files, channel_indices, checkpoint_dir=None).

CONTRACT TARGETS
  output_id: RESULT-full-lfp
  (1) OUT-026 REGRESSION ANCHOR -- ACCEPTANCE LEVEL 1 (exact), BLOCKING:
      "Full-duration LFP restricted to the first 10 s must equal the current
       10-s loader output channel-for-channel within float64 numerical tolerance
       (regression anchor rtol <= 1e-9, atol <= 1e-12)."
      failure_conditions: "Full-duration LFP fails the first-10-s regression
       anchor against the current loader; extraction blocked and reported."
      docx_source_refs: OUT-005, OUT-026, RQ-054.
  (2) GOLDEN-FIXTURE CHECK -- ACCEPTANCE LEVEL 2 (numeric):
      first-10-s slice for the five selected channels must match
      Golden Fixtures/04_lfp/LFP.npy (shape (384,10000)) at the manifest LFP
      tolerance rtol=1e-6 / atol=1e-9 (00_manifest/manifest.json).

PRESERVED PREPROCESSING (must be reproduced EXACTLY -- repo_map §1.1 / H3;
constants LFP_LP_HZ=400.0, LFP_OUT_FS=1000.0 in load_experiment_data.py):
  b,a = scipy.signal.butter(4, 400/(NP_Fs/2), 'low')
  filtfilt(b, a, rawAP, axis=1, padtype='odd', padlen=3*(max(len a,b)-1))
  decimate by dsRatio = round(NP_Fs/1000).

DESIGN (Blueprint v2 P3): the two acceptance-level checks require the real .bin
data and are HEAVY (multi-GB load + Kilosort), so they are opt-in via
NP_PAPER1_RUN_HEAVY=1 and skip cleanly otherwise. They anchor the new function's
first-10-s output to (1) the current validated loader run live on the same
session and (2) the golden LFP.npy on the fixture-provenance session -- never to
the new code. The always-runnable "recipe anchor" tests below pin the exact
filter design / decimation the implementer must replicate, checked against the
real golden Butterworth-coefficient / NP_Fs fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.signal

from conftest import (
    GOLDEN_DIR,
    KERNEL_DIR,
    SESSION_DIRNAMES,
    _ensure_on_path,
    heavy_enabled,
    load_real_extract_full_lfp,
    reference_lfp_from_raw,
    resolve_session_files,
)

# Five arbitrary but fixed LFP row indices (0..383) used for the extraction
# checks; independent of select_five_depths so this oracle tests only the
# extraction/preprocessing numerics.
CHANNEL_INDICES = [0, 100, 200, 300, 383]
REG_RTOL, REG_ATOL = 1e-9, 1e-12          # OUT-026 regression anchor (level 1)
FIX_RTOL, FIX_ATOL = 1e-6, 1e-9           # manifest LFP fixture tol (level 2)


def _load_golden_arr(rel):
    p = GOLDEN_DIR / rel
    if not p.is_file():
        pytest.skip(f"golden fixture not present: {p}")
    return np.load(p)


def _normalize_full_lfp(out, n_channels):
    """Accept the extract_full_lfp return as an ndarray (n_ch, n_samp) or a dict
    exposing 'LFP'/'full_lfp'/'lfp'. Returns a 2-D float64 array."""
    if isinstance(out, dict):
        for k in ("LFP", "full_lfp", "lfp"):
            if k in out:
                arr = out[k]
                break
        else:
            raise KeyError("no LFP key in extract_full_lfp output")
    else:
        arr = out
    arr = np.asarray(arr, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[0] == n_channels, (
        f"expected (n_channels={n_channels}, n_samp); got {arr.shape}")
    return arr


# ===========================================================================
# Recipe-anchor tests (run TODAY, no .bin): pin the exact preprocessing the
# full-duration extraction must reproduce.
# ===========================================================================
def test_butterworth_coefficients_match_golden():
    """The full extraction must use the SAME order-4 Butterworth design as the
    validated loader. Recompute butter(4, 400/(NP_Fs/2), 'low') from the golden
    NP_Fs and compare to the golden filt_b / filt_a
    (03_lfp_intermediate, manifest tol rtol=1e-9/atol=1e-12). Anchors the filter
    identity independently of any implementation."""
    np_fs = float(np.asarray(_load_golden_arr("02_np_meta/NP_Fs.npy")).ravel()[0])
    gb = np.asarray(_load_golden_arr("03_lfp_intermediate/filt_b.npy")).ravel()
    ga = np.asarray(_load_golden_arr("03_lfp_intermediate/filt_a.npy")).ravel()
    b, a = scipy.signal.butter(4, 400.0 / (np_fs / 2.0), btype="low")
    np.testing.assert_allclose(b, gb, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a, ga, rtol=1e-9, atol=1e-12)


def test_decimation_ratio_matches_golden():
    """dsRatio = round(NP_Fs / 1000) must equal the golden dsRatio and match the
    LFP.npy shape (10000 samples <=> ~10 s at 1000 Hz)."""
    np_fs = float(np.asarray(_load_golden_arr("02_np_meta/NP_Fs.npy")).ravel()[0])
    ds_golden = int(np.asarray(_load_golden_arr("02_np_meta/dsRatio.npy")).ravel()[0])
    assert round(np_fs / 1000.0) == ds_golden


def test_recipe_passband_stopband_on_synthetic_raw():
    """Documentation of the preserved preprocessing on a synthetic raw block
    (no .bin needed): a 6 Hz (theta) component survives the 400 Hz low-pass with
    near-unit gain while a 2000 Hz component is strongly attenuated, and
    decimation by round(NP_Fs/1000) yields the expected output length. This
    exercises conftest.reference_lfp_from_raw (a verbatim copy of the validated
    loader recipe), which the real extract_full_lfp must match."""
    np_fs = 30000.0
    dur = 2.0
    n = int(dur * np_fs)
    t = np.arange(n) / np_fs
    low = np.sin(2 * np.pi * 6.0 * t)        # theta, in-band
    high = np.sin(2 * np.pi * 2000.0 * t)    # far above 400 Hz cutoff
    raw = np.vstack([low + high, low])       # 2 channels
    out = reference_lfp_from_raw(raw, np_fs)

    ds = round(np_fs / 1000.0)
    assert out.shape == (2, raw.shape[1] // ds + (1 if raw.shape[1] % ds else 0))
    # Interior (exclude 0.2 s edges) of channel 1 (pure 6 Hz) ~ unit amplitude.
    fs_out = np_fs / ds
    e = int(0.2 * fs_out)
    interior = out[1, e:-e]
    assert 0.9 < np.max(np.abs(interior)) < 1.1
    # The 2000 Hz component in channel 0 must be attenuated far below unity.
    ch0 = out[0, e:-e]
    resid_high = ch0 - interior
    assert np.max(np.abs(resid_high)) < 0.1, "2 kHz component not attenuated by LP"


def test_reference_recipe_is_deterministic():
    """The preprocessing is a deterministic filter+decimate: repeated calls on
    identical input give bit-identical output (a precondition for the OUT-026
    tight regression anchor to be meaningful)."""
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((3, 60000))
    a = reference_lfp_from_raw(raw, 30000.0)
    b = reference_lfp_from_raw(raw, 30000.0)
    np.testing.assert_array_equal(a, b)


# ===========================================================================
# HEAVY: OUT-026 regression anchor vs the LIVE current loader (LEVEL 1, exact)
# ===========================================================================
def _first_resolvable_session():
    for name in SESSION_DIRNAMES:
        files, missing = resolve_session_files(name)
        if files is not None and not missing:
            return name, files
    return None, None


def test_first10s_equals_current_loader_regression_anchor():
    """OUT-026 (BLOCKING, level 1 exact): extract_full_lfp(...)[:, :10000] for the
    selected channels must equal the CURRENT load_experiment_data 10-s output
    channel-for-channel within rtol<=1e-9 / atol<=1e-12. Compared live against a
    fresh loader run on the SAME session (not against the new code).

    NOTE for the implementer: because filtfilt is non-causal, filtering the full
    recording and slicing the first 10 s does NOT automatically reproduce the
    loader's 10-s-only filtfilt at the trailing boundary. Achieving this tight
    anchor is the design obligation of extract_full_lfp; a failure here is the
    contract's designated blocking condition, not a bug in this oracle."""
    real = load_real_extract_full_lfp()
    if real is None:
        pytest.skip("extract_full_lfp.py not yet implemented")
    if not heavy_enabled():
        pytest.skip("heavy real-session test; set NP_PAPER1_RUN_HEAVY=1 to run")

    name, files = _first_resolvable_session()
    if files is None:
        pytest.skip("no resolvable session under DATA/ (need .bin + Kilosort)")

    _ensure_on_path(KERNEL_DIR)
    from load_experiment_data import load_experiment_data
    D = load_experiment_data(files)
    loader_lfp = np.asarray(D["LFP"], dtype=np.float64)
    n10 = loader_lfp.shape[1]                       # 10000

    full = _normalize_full_lfp(real(files, CHANNEL_INDICES), len(CHANNEL_INDICES))
    assert full.shape[1] >= n10, "full-duration LFP shorter than the 10-s loader slice"

    expected = loader_lfp[CHANNEL_INDICES, :]
    got = full[:, :n10]
    np.testing.assert_allclose(got, expected, rtol=REG_RTOL, atol=REG_ATOL)


def test_full_duration_is_longer_than_10s():
    """Sanity (level 4 scientific): the whole point of extract_full_lfp is a
    LONGER record than the current 10-s loader. On a real multi-minute session
    the output must have many more than 10000 samples."""
    real = load_real_extract_full_lfp()
    if real is None:
        pytest.skip("extract_full_lfp.py not yet implemented")
    if not heavy_enabled():
        pytest.skip("heavy real-session test; set NP_PAPER1_RUN_HEAVY=1 to run")
    name, files = _first_resolvable_session()
    if files is None:
        pytest.skip("no resolvable session under DATA/")
    full = _normalize_full_lfp(real(files, CHANNEL_INDICES), len(CHANNEL_INDICES))
    assert full.shape[1] > 10000, "full-duration LFP not longer than the 10-s slice"


# ===========================================================================
# HEAVY: golden LFP.npy fixture check on the provenance session (LEVEL 2)
# ===========================================================================
def test_first10s_matches_golden_lfp_fixture():
    """Level 2 (numeric, rtol=1e-6/atol=1e-9): on the session that GENERATED the
    golden fixtures, extract_full_lfp(...)[:, :10000] for the five selected
    channels must match Golden Fixtures/04_lfp/LFP.npy[channels]. The provenance
    session is discovered deterministically: the session whose live loader LFP
    equals the golden LFP within the fixture tolerance."""
    real = load_real_extract_full_lfp()
    if real is None:
        pytest.skip("extract_full_lfp.py not yet implemented")
    if not heavy_enabled():
        pytest.skip("heavy real-session test; set NP_PAPER1_RUN_HEAVY=1 to run")

    golden = np.asarray(_load_golden_arr("04_lfp/LFP.npy"), dtype=np.float64)
    assert golden.shape == (384, 10000)

    _ensure_on_path(KERNEL_DIR)
    from load_experiment_data import load_experiment_data

    provenance_files = None
    for name in SESSION_DIRNAMES:
        files, missing = resolve_session_files(name)
        if files is None or missing:
            continue
        D = load_experiment_data(files)
        loader_lfp = np.asarray(D["LFP"], dtype=np.float64)
        if loader_lfp.shape == golden.shape and np.allclose(
                loader_lfp, golden, rtol=FIX_RTOL, atol=FIX_ATOL):
            provenance_files = files
            break
    if provenance_files is None:
        pytest.skip("could not identify the golden-fixture provenance session under DATA/")

    full = _normalize_full_lfp(real(provenance_files, CHANNEL_INDICES),
                               len(CHANNEL_INDICES))
    expected = golden[CHANNEL_INDICES, :]
    got = full[:, :10000]
    np.testing.assert_allclose(got, expected, rtol=FIX_RTOL, atol=FIX_ATOL)

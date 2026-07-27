# ORACLE
"""
Shared fixtures and reference (ground-truth) computations for the NP-PAPER-1
"plumbing" oracle suite (Blueprint v2 P3 — oracle design phase, RED tier).

SCOPE OF THIS CONFTEST (one of several oracle-designer agents working NP-PAPER-1
in parallel): the five *deterministic plumbing* functions only —
  1. extract_full_lfp        (test_oracle_extract_full_lfp.py)
  2. select_five_depths       (test_oracle_select_five_depths.py)
  3. build_window_control_mask(test_oracle_window_control_mask.py)
  4. build_valid_sniff_mask   (test_oracle_valid_sniff_mask.py)
  5. resample_snf_to_lfp      (test_oracle_resample_snf.py)
The spectral engine (multitaper_psd_coherence, circular_shift_null,
reduce_theta_coherence) and depth_statistics are owned by OTHER oracle-designer
agents — NOT covered here.

DESIGN RULE (task brief + repo convention, mirrors NP-DEMO-3/4 oracle suites):
  No expected value in this suite was produced by executing the not-yet-written
  implementation code (03_software/*.py). Every expectation is anchored to one
  of:
    (a) closed-form / hand-computed arithmetic on small synthetic arrays
        (documented inline, verified by an independent numpy/scipy path);
    (b) the real Golden Fixtures on disk
        (C:\\Projects\\Repos\\Neuropixels\\Golden Fixtures\\...), whose
        authoritative tolerances live in 00_manifest/manifest.json;
    (c) the already-VALIDATED, shipped kernel recipe
        (Optimized Python/load_experiment_data.py's butter->filtfilt->decimate
        LFP path; Optimized Python/analyses/compute_sniff_phase.py's onset
        detector) — reproduced here verbatim as a *reference*. Consulting a
        validated recipe is "consulting a reference", not "running the new code".

  Each `_load_real_<fn>()` helper below returns the REAL implementation once the
  analysis-implementer writes 03_software/<fn>.py; until then the tests that
  need it skip (via pytest.skip) or fall back to the contract-derived reference,
  so the pure-arithmetic microcases pass TODAY and become the acceptance gate
  automatically once the code lands.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.signal

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
SOFTWARE_DIR = _THIS_FILE.parents[1]                 # .../NP-PAPER-1/03_software
REPO_ROOT = _THIS_FILE.parents[4]                    # .../Neuropixels
KERNEL_DIR = REPO_ROOT / "Optimized Python"
GOLDEN_DIR = REPO_ROOT / "Golden Fixtures"
DATA_DIR = REPO_ROOT / "DATA"

# The six confirmed sessions (contract data_provenance / DATA-004). Folder names
# on disk use MM-DD-YYYY (verified 2026-07-24).
SESSION_DIRNAMES = [
    "11-01-2021", "11-03-2021", "12-15-2021",
    "5-17-2022", "06-24-2022", "09-14-2022",
]


def _ensure_on_path(p: Path) -> None:
    s = str(p)
    if s not in sys.path and p.is_dir():
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Import-fallback loaders for the five NEW functions (None until implemented)
# ---------------------------------------------------------------------------
def _load_real(module_name: str, attr: str):
    _ensure_on_path(SOFTWARE_DIR)
    try:
        mod = __import__(module_name)
    except ImportError:
        return None
    return getattr(mod, attr, None)


def load_real_extract_full_lfp():
    return _load_real("extract_full_lfp", "extract_full_lfp")


def load_real_select_five_depths():
    return _load_real("select_five_depths", "select_five_depths")


def load_real_build_window_control_mask():
    return _load_real("build_window_control_mask", "build_window_control_mask")


def load_real_build_valid_sniff_mask():
    return _load_real("build_valid_sniff_mask", "build_valid_sniff_mask")


def load_real_resample_snf_to_lfp():
    return _load_real("resample_snf_to_lfp", "resample_snf_to_lfp")


# ---------------------------------------------------------------------------
# (a) detect_eth_contact RULE — reference, verbatim from
#     _pipeline/NP-DEMO-3/03_software/detect_eth_contact.py (read 2026-07-24)
#     and pinned by PROH-003/PROH-008/CON-015.
# ---------------------------------------------------------------------------
def reference_eth_contact_mask(eth: np.ndarray, eth_threshold: float = 0.05) -> np.ndarray:
    """ETH_ms = ETH - mean(ENTIRE trace); mask = ETH_ms > eth_threshold (STRICT)."""
    eth = np.asarray(eth, dtype=np.float64).ravel()
    eth_ms = eth - eth.mean()
    return eth_ms > eth_threshold


def reference_window_control_mask(eth, lv_fs, window_starts_s, window_len_s,
                                  eth_threshold: float = 0.05) -> np.ndarray:
    """Ground-truth per-window "all-control" mask (build_window_control_mask).

    Rule (contract inclusion_exclusion PROH-008/CON-002): a window on the LFP
    clock is control-eligible ONLY if the time span [t0, t0+window_len_s)
    contains ZERO ethanol samples. Ethanol samples live on the LV clock at time
    idx/lv_fs (shared recording clock). ANY overlap excludes.

    Convention (stated explicitly): the window interval is half-open
    [t0, t0+window_len_s) — an ethanol sample exactly at t0 is IN the window;
    one exactly at t0+window_len_s belongs to the next window. This matches the
    "start_s/end_s within one LFP sample" edge semantics of RESULT-qc-discards.

    Returns a boolean array (len == len(window_starts_s)); True == control-eligible.
    """
    mask = reference_eth_contact_mask(eth, eth_threshold)
    eth_times = np.flatnonzero(mask) / float(lv_fs)
    starts = np.asarray(window_starts_s, dtype=np.float64)
    out = np.ones(starts.size, dtype=bool)
    for i, t0 in enumerate(starts):
        t1 = t0 + window_len_s
        if np.any((eth_times >= t0) & (eth_times < t1)):
            out[i] = False
    return out


# ---------------------------------------------------------------------------
# (b) select_five_depths — reference (contract preprocessing RQ-012..016).
#     5 fractions of the ycoord range; nearest channel; lower-index tie-break.
# ---------------------------------------------------------------------------
DEPTH_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def reference_select_five_depths(xcoords, ycoords, valid_lfp_channels):
    """Deterministic five-depth selection (ground truth, first principles).

    - Candidate channels = valid_lfp_channels that ALSO have a finite ycoord.
    - ymin/ymax over candidates.
    - target_k = ymin + frac_k*(ymax - ymin), frac in DEPTH_FRACTIONS.
    - nearest candidate by |ycoord - target|; tie -> LOWER channel index.
      (np.argmin over candidates listed in ascending original index returns the
       first minimum == lowest index, exactly the contract tie-break RQ-016.)

    Returns (indices[5] int, ycoords_sel[5] float, ymin float, ymax float).
    """
    ycoords = np.asarray(ycoords, dtype=np.float64).ravel()
    valid = np.asarray(list(valid_lfp_channels), dtype=np.int64).ravel()
    valid = np.sort(np.unique(valid))
    cand = valid[np.isfinite(ycoords[valid])]
    if cand.size == 0:
        raise ValueError("no valid candidate channels with finite ycoord")
    ys = ycoords[cand]
    ymin, ymax = float(ys.min()), float(ys.max())
    span = ymax - ymin
    idx_out, y_out = [], []
    for frac in DEPTH_FRACTIONS:
        target = ymin + frac * span
        k = int(np.argmin(np.abs(ys - target)))   # ascending -> lowest index tie-break
        idx_out.append(int(cand[k]))
        y_out.append(float(ys[k]))
    return idx_out, y_out, ymin, ymax


# ---------------------------------------------------------------------------
# (c) build_valid_sniff_mask — reference (contract RQ-018..021 / OUT-008).
#     ISI = diff(sniff_onsets_s); invalid if ISI outside [p5,p95] OR SNF_PH==-1.
# ---------------------------------------------------------------------------
def reference_isi_cutoffs(sniff_onsets_s):
    """Per-sniff ISI array and its 5th/95th percentiles (numpy default 'linear').

    HAZARD (repo_map H1): ISI MUST be diff(sniff_onsets_s), NOT the scalar
    sniff_dur_s median.
    """
    onsets = np.asarray(sniff_onsets_s, dtype=np.float64).ravel()
    isi = np.diff(onsets)
    p5 = float(np.percentile(isi, 5))
    p95 = float(np.percentile(isi, 95))
    return isi, p5, p95


def reference_valid_sniff_mask(sniff_onsets_s, snf_ph, fs):
    """Ground-truth per-sample valid-sniff mask on the SNF_PH clock.

    A sample is INVALID if:
      * it lies in an inter-onset span whose ISI is < p5 or > p95, OR
      * SNF_PH == -1 at that sample.
    Samples before the first onset / after the last onset are not covered by any
    ISI span; they inherit validity only from the SNF_PH!=-1 rule (documented
    convention: uncovered => not part of a retained sniff cycle => invalid).

    Returns (valid_mask bool[n], isi, p5, p95). n == len(snf_ph).
    """
    snf_ph = np.asarray(snf_ph, dtype=np.float64).ravel()
    n = snf_ph.size
    onsets = np.asarray(sniff_onsets_s, dtype=np.float64).ravel()
    isi, p5, p95 = reference_isi_cutoffs(onsets)
    onset_samp = np.rint(onsets * fs).astype(np.int64)

    valid = np.zeros(n, dtype=bool)   # default invalid outside covered spans
    for k in range(isi.size):
        a = int(onset_samp[k])
        b = int(onset_samp[k + 1])
        a = max(a, 0); b = min(b, n)
        if b <= a:
            continue
        span_ok = (isi[k] >= p5) and (isi[k] <= p95)
        valid[a:b] = span_ok
    # SNF_PH == -1 overrides everything to invalid.
    valid &= (snf_ph != -1.0)
    return valid, isi, p5, p95


def reference_inst_sniff_rate(sniff_onsets_s):
    """Instantaneous sniff rate = 1/ISI per inter-onset span (Hz)."""
    isi = np.diff(np.asarray(sniff_onsets_s, dtype=np.float64).ravel())
    with np.errstate(divide="ignore"):
        return 1.0 / isi


# ---------------------------------------------------------------------------
# (a-LFP) extract_full_lfp — reference of the VALIDATED loader recipe.
#     butter(4, 400/(NP_Fs/2), 'low') -> filtfilt(odd, padlen=3*(ntaps-1))
#     -> decimate by round(NP_Fs/1000). Verbatim from
#     Optimized Python/load_experiment_data.py (read 2026-07-24), constants
#     LFP_LP_HZ=400.0, LFP_OUT_FS=1000.0.
# ---------------------------------------------------------------------------
LFP_LP_HZ = 400.0
LFP_OUT_FS = 1000.0


def reference_lfp_from_raw(raw_v, np_fs):
    """Reproduce the validated loader LFP path on a (nCh, nSamp) volt-scaled raw
    AP array. NOTE: this filters the *given* block only (edge behavior matches
    the loader when the block is the loader's 10-s block); extract_full_lfp must
    reproduce the SAME butter/filtfilt/decimate math on the full recording."""
    raw_v = np.asarray(raw_v, dtype=np.float64)
    b, a = scipy.signal.butter(4, LFP_LP_HZ / (np_fs / 2.0), btype="low")
    padlen = 3 * (max(len(a), len(b)) - 1)
    filt = scipy.signal.filtfilt(b, a, raw_v, axis=1, padtype="odd", padlen=padlen)
    ds = round(np_fs / LFP_OUT_FS)
    return filt[:, ::ds]


# ---------------------------------------------------------------------------
# Golden-fixture loaders (skip if not present on this machine)
# ---------------------------------------------------------------------------
def _load_golden(rel_path):
    p = GOLDEN_DIR / rel_path
    if not p.is_file():
        pytest.skip(f"golden fixture not present: {p}")
    return np.load(p)


@pytest.fixture
def golden_ycoords():
    """05_spikes_ks/ycoords.npy — (382,1) float64, EXACT tolerance (rtol=atol=0)."""
    y = _load_golden(Path("05_spikes_ks") / "ycoords.npy")
    return np.asarray(y, dtype=np.float64).ravel()


@pytest.fixture
def golden_xcoords():
    x = _load_golden(Path("05_spikes_ks") / "xcoords.npy")
    return np.asarray(x, dtype=np.float64).ravel()


@pytest.fixture
def golden_lfp():
    """04_lfp/LFP.npy — (384,10000) float64; manifest tol rtol=1e-6/atol=1e-9."""
    lfp = _load_golden(Path("04_lfp") / "LFP.npy")
    return np.asarray(lfp, dtype=np.float64)


# ---------------------------------------------------------------------------
# Heavy real-session resolution (opt-in; loads multi-GB .bin + Kilosort)
# ---------------------------------------------------------------------------
def heavy_enabled() -> bool:
    return os.environ.get("NP_PAPER1_RUN_HEAVY", "") == "1"


def resolve_session_files(session_dirname: str):
    """Build the load_experiment_data `files` dict for one session using the
    validated kernel helpers (or_validate_files + the NP-DEMO-4 _fix_ks_dir
    multi-shank promotion). Returns (files, missing) or (None, reason)."""
    _ensure_on_path(KERNEL_DIR)
    sess = DATA_DIR / session_dirname
    if not sess.is_dir():
        return None, f"session dir missing: {sess}"
    try:
        from lib.or_validate_files import or_validate_files
    except ImportError as exc:
        return None, f"kernel or_validate_files import failed: {exc}"
    files, missing = or_validate_files(str(sess), strict=False)

    # NP-DEMO-4 multi-shank ksDir promotion (needed for 09-14-2022).
    ks_probe = ("channel_map.npy", "channel_positions.npy")
    ksd = files.get("ksDir")
    if ksd and any(not os.path.isfile(os.path.join(ksd, f)) for f in ks_probe):
        parent = os.path.dirname(ksd)
        if all(os.path.isfile(os.path.join(parent, f)) for f in ks_probe):
            files["ksDir"] = parent
            missing = [m for m in missing
                       if "channel_map" not in m and "channel_positions" not in m]
    return files, missing

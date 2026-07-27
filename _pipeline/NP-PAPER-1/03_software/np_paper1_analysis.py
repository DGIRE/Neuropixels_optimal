"""np_paper1_analysis.py -- NP-PAPER-1 COMPOSING RUNNER (Blueprint v2 P3, D7).

Calls the 9 already-implemented, oracle-tested NP-PAPER-1 analysis primitives
(extract_full_lfp, select_five_depths, detect_eth_contact,
build_window_control_mask, build_valid_sniff_mask, resample_snf_to_lfp,
multitaper_psd_coherence, circular_shift_null, reduce_theta_coherence,
depth_statistics) plus the validated kernel (load_experiment_data,
compute_sniff_phase) in sequence, across the 6 real sessions, and produces the
RESULT-* objects required by contract_v001.yaml.

This file is ADDITIVE glue/composition code. It does NOT reimplement any of
the 9 delicate functions' math -- it only:
  (a) resolves session files (reusing lib.or_validate_files, the same helper
      oracles/conftest.py::resolve_session_files uses, with the identical
      2022-09-14 multi-shank ksDir promotion logic reproduced here per the
      task brief -- "reuse that exact logic or the helper itself"),
  (b) builds full-nAP-length xcoords/ycoords arrays (via channel_map.npy,
      since load_ks_dir's xcoords/ycoords are indexed by KS-internal channel
      position, not necessarily the raw AP row index -- see module note
      CHANNEL MAPPING below),
  (c) builds the 4-s / 50%-overlap candidate window grid on the LFP clock and
      combines the ethanol/control mask (via build_window_control_mask) with a
      window-level valid-sniff mask (thin new plumbing -- no such function was
      requested among the 9; NOT a delicate/novel algorithm, just a per-window
      AND-reduction over build_valid_sniff_mask's per-sample output, mirroring
      build_window_control_mask's own half-open-interval convention),
  (d) loops over the 5 selected depths calling multitaper_psd_coherence,
      circular_shift_null, reduce_theta_coherence exactly as contracted,
  (e) does per-session QC bookkeeping (RESULT-qc-counts/qc-discards/
      sniff-rate) and cross-session aggregation (best-depth/depth_statistics/
      grand means) exactly as specified in contract_v001.yaml's aggregation
      section.

CHANNEL MAPPING NOTE (verified empirically 2026-07-24 for all 6 sessions):
  `Optimized Python/lib/ks_utils.py::load_ks_dir` sets
  xcoords/ycoords = channel_positions.npy columns 0/1 directly, WITHOUT ever
  consulting channel_map.npy. channel_positions.npy row i corresponds to the
  raw AP channel index channel_map.npy[i] (Kilosort's convention), which is
  the identity map for all 6 confirmed sessions here (verified by loading
  channel_map.npy for each resolved ksDir) but is NOT guaranteed to be
  identity in general (5-17-2022 and 09-14-2022 have fewer than 384 rows,
  i.e. Kilosort was run on a subset of the 384 raw AP channels). This runner
  scatters xcoords/ycoords into a full length-nAP array via channel_map.npy
  (NaN elsewhere) so that select_five_depths' channel indices are directly
  valid raw AP row indices for extract_full_lfp -- correct regardless of
  whether channel_map happens to be identity.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Path setup (read-only kernel reuse; additive-only local imports).
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
SOFTWARE_DIR = _THIS_FILE.parent                      # .../NP-PAPER-1/03_software
REPO_ROOT = _THIS_FILE.parents[3]                     # .../Neuropixels
KERNEL_DIR = REPO_ROOT / "Optimized Python"
DATA_DIR = REPO_ROOT / "DATA"
RESULTS_DIR = _THIS_FILE.parents[1] / "04_results"


def _ensure_on_path(p: Path) -> None:
    s = str(p)
    if s not in sys.path and p.is_dir():
        sys.path.insert(0, s)


_ensure_on_path(KERNEL_DIR)
_ensure_on_path(SOFTWARE_DIR)

from load_experiment_data import load_experiment_data  # noqa: E402
from analyses.compute_sniff_phase import compute_sniff_phase  # noqa: E402
from lib.or_validate_files import or_validate_files  # noqa: E402
from lib.or_loaddat import or_loaddat  # noqa: E402

from detect_eth_contact import detect_eth_contact  # noqa: E402
from extract_full_lfp import extract_full_lfp  # noqa: E402
from select_five_depths import select_five_depths  # noqa: E402
from build_window_control_mask import build_window_control_mask  # noqa: E402
from build_valid_sniff_mask import build_valid_sniff_mask  # noqa: E402
from resample_snf_to_lfp import resample_snf_to_lfp, _rational_up_down  # noqa: E402
from multitaper_psd_coherence import multitaper_psd_coherence  # noqa: E402
from circular_shift_null import circular_shift_null  # noqa: E402
from reduce_theta_coherence import reduce_theta_coherence  # noqa: E402
from depth_statistics import depth_statistics  # noqa: E402
from resolve_session_binfile import (  # noqa: E402
    resolve_full_lfp_binfile, NoEligibleBinFileError,
)

# ---------------------------------------------------------------------------
# Pinned parameters (contract PROH list -- never tuned without human sign-off)
# ---------------------------------------------------------------------------
SNIFF_THRESHOLD_STD = -0.5          # PROH-004/PROH-007
ETH_THRESHOLD = 0.05                # PROH-003/PROH-008
WINDOW_LEN_S = 4.0                  # RQ-040
WINDOW_OVERLAP = 0.5                # RQ-040
THETA_BAND = (2.0, 12.0)            # RQ-039
N_SHIFTS = 1000                     # RQ-041/CON-003 -- pinned, never < 1000
NULL_SEED = 5489                    # RQ-038/CON-003
MIN_WINDOWS_FOR_STATS = 30          # PROH-002/CON-008

SESSION_DIRNAMES = [
    "11-01-2021", "11-03-2021", "12-15-2021",
    "5-17-2022", "06-24-2022", "09-14-2022",
]
# Contract data_provenance ISO dates, same order as SESSION_DIRNAMES.
SESSION_DATES = [
    "2021-11-01", "2021-11-03", "2021-12-15",
    "2022-05-17", "2022-06-24", "2022-09-14",
]

# DEV-002 (scientific-audit BLOCKER-1, resolved by David Gire 2026-07-25,
# Resolution A): the true animal->session map. 2021-11-01 and 2021-11-03 are
# BOTH animal NP8 (confirmed via the raw LabView .dat filenames -- 2021-11-03's
# explicitly reads "..._NP8_Marvel_..._2ndcraniotomy_2ndREcording"). 5 animals,
# 6 sessions total. Session-level results (RESULT-theta-coh, RESULT-qc-counts,
# etc.) are UNCHANGED by this map -- only the cross-session/animal aggregation
# (RESULT-best-depth, RESULT-depth-stat, grand-mean spectra) uses it.
ANIMAL_MAP = {
    "11-01-2021": "NP8",
    "11-03-2021": "NP8",
    "12-15-2021": "Np10",
    "5-17-2022": "NP12",
    "06-24-2022": "NP15",
    "09-14-2022": "NP22",
}
N_ANIMALS = len(set(ANIMAL_MAP.values()))  # 5
N_SESSIONS = len(ANIMAL_MAP)               # 6


# ===========================================================================
# (a) Session file resolution -- reproduces
#     oracles/conftest.py::resolve_session_files exactly (same or_validate_files
#     call + the NP-DEMO-4 multi-shank ksDir promotion for 2022-09-14), PLUS
#     the DEV-001 Resolution B duration-matching .bin-file override (David
#     Gire, approved 2026-07-24). or_validate_files (kernel, PROH-010) is
#     still used AS-IS for datFile/ksDir resolution -- ONLY its naive
#     alphabetical-first .ap.bin pick is overridden here, in this task's own
#     additive code, never by editing the kernel file itself.
# ===========================================================================
def resolve_session_files(session_dirname: str):
    sess = DATA_DIR / session_dirname
    if not sess.is_dir():
        return None, [f"session dir missing: {sess}"]
    files, missing = or_validate_files(str(sess), strict=False)

    ks_probe = ("channel_map.npy", "channel_positions.npy")
    ksd = files.get("ksDir")
    if ksd and any(not os.path.isfile(os.path.join(ksd, f)) for f in ks_probe):
        parent = os.path.dirname(ksd)
        if all(os.path.isfile(os.path.join(parent, f)) for f in ks_probe):
            files["ksDir"] = parent
            missing = [m for m in missing
                       if "channel_map" not in m and "channel_positions" not in m]

    # ---- DEV-001 Resolution B: duration-matching .bin override ----------
    # or_validate_files' datFile pick showed no evidence of the same defect
    # (see DEV-001 evidence table) so it is trusted here to get the LabView
    # duration baseline used to disambiguate the .bin candidates.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _TR, _TS, _FR, _ETH, SNF, LV_Fs = or_loaddat(files["datFile"], files["datPath"])
    labview_duration_min = SNF.size / float(LV_Fs) / 60.0

    resolution = resolve_full_lfp_binfile(str(sess), labview_duration_min)
    files["binFile"] = resolution["binFile"]
    files["binPath"] = resolution["binPath"]
    files["_binfile_resolution"] = resolution  # audit trail for QC/methods log
    files["_labview_duration_min"] = labview_duration_min

    return files, missing


# ===========================================================================
# (b) Full-nAP-length xcoords/ycoords via channel_map.npy (see module note).
# ===========================================================================
def channel_coords_full(D: dict, files: dict, nAP: int):
    ksdir = files["ksDir"]
    cm = np.load(os.path.join(ksdir, "channel_map.npy")).ravel().astype(np.int64)
    xcoords = np.asarray(D["xcoords"], dtype=np.float64).ravel()
    ycoords = np.asarray(D["ycoords"], dtype=np.float64).ravel()
    if cm.size != xcoords.size:
        raise RuntimeError(
            f"channel_map.npy size ({cm.size}) != D['xcoords'] size "
            f"({xcoords.size}) at {ksdir!r} -- integration assumption violated, "
            "stop and report (do not silently truncate/pad)."
        )
    if np.any(cm < 0) or np.any(cm >= nAP):
        raise RuntimeError(
            f"channel_map.npy contains indices outside [0, {nAP}) at {ksdir!r}."
        )
    xcoords_full = np.full(nAP, np.nan, dtype=np.float64)
    ycoords_full = np.full(nAP, np.nan, dtype=np.float64)
    xcoords_full[cm] = xcoords
    ycoords_full[cm] = ycoords
    return xcoords_full, ycoords_full


# ===========================================================================
# (b2) Valid (non-padded) length of resample_snf_to_lfp's output (scientific-
# audit MINOR-2 fix, approved by David Gire 2026-07-25). Mirrors
# resample_snf_to_lfp's OWN internal up/down ratio (reusing its private
# `_rational_up_down` helper for exact parity, since resample_snf_to_lfp.py
# itself is pinned/oracle-tested and is not modified) to determine how many
# of its output samples are real (resampled from actual LabView data) versus
# a held-constant pad appended because the chosen full-duration LFP is longer
# than the independent LabView recording (DEV-001 Resolution B can select a
# binFile whose duration exceeds the LabView duration by design).
# ===========================================================================
def compute_valid_snf_length(SNF, LV_Fs: float, LFP_Fs: float, n_lfp_samples: int) -> int:
    """Number of `resample_snf_to_lfp` output samples that are real (not a
    held-constant pad). Equals n_lfp_samples when the LabView recording
    covers (or exceeds) the LFP duration (no padding, resample_snf_to_lfp
    truncates instead)."""
    import scipy.signal  # local import: only needed for this bookkeeping call

    x = np.asarray(SNF, dtype=np.float64).ravel()
    up, down = _rational_up_down(LFP_Fs, LV_Fs)
    resampled_len = int(scipy.signal.resample_poly(x, up, down).size)
    return int(min(resampled_len, int(n_lfp_samples)))


# ===========================================================================
# (c) Window-level valid-sniff mask (thin plumbing; NOT a delicate function --
#     mirrors build_window_control_mask's half-open-interval convention over
#     build_valid_sniff_mask's per-sample output instead of the ETH mask).
# ===========================================================================
def window_all_valid_mask(valid_bool_at_rate: np.ndarray, rate_hz: float,
                           window_starts_s: np.ndarray, window_len_s: float) -> np.ndarray:
    """True iff the half-open span [t0, t0+window_len_s) is entirely covered
    by True samples of `valid_bool_at_rate` (sampled at rate_hz). A window
    that runs past the end of `valid_bool_at_rate` is conservatively marked
    invalid (no sniff coverage to confirm validity)."""
    valid_bool_at_rate = np.asarray(valid_bool_at_rate, dtype=bool)
    n = valid_bool_at_rate.size
    invalid_cumsum = np.concatenate(([0], np.cumsum(~valid_bool_at_rate)))
    starts_s = np.asarray(window_starts_s, dtype=np.float64)
    a = np.rint(starts_s * rate_hz).astype(np.int64)
    b = np.rint((starts_s + window_len_s) * rate_hz).astype(np.int64)
    in_range = (a >= 0) & (b <= n)
    a_c = np.clip(a, 0, n)
    b_c = np.clip(b, 0, n)
    counts = invalid_cumsum[b_c] - invalid_cumsum[a_c]
    return in_range & (counts == 0)


def _contiguous_true_spans_s(mask: np.ndarray, rate_hz: float):
    """List of (start_s, end_s) half-open spans of contiguous True in `mask`
    (sampled at rate_hz). Empty-safe."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(float(s) / rate_hz, float(e) / rate_hz) for s, e in zip(starts, ends)]


def _classify_invalid_sniff_reasons(sniff_onsets_s, snf_ph, isi, p5, p95, lv_fs):
    """Recompute (for QC-reporting attribution ONLY -- the actual analysis
    gate is build_valid_sniff_mask's own `valid_sniff_mask`) which invalid
    samples are attributable to 'ISI outlier' (rule 3) vs 'non-sniff noise'
    (SNF_PH==-1, rule 4, OR uncovered before-first/after-last-onset spans,
    rule 5 -- contract's reason enum has no separate 'uncovered' bucket, so
    uncovered spans are folded into 'non-sniff noise').

    Returns (isi_outlier_mask, noise_mask), each (n,) bool on the SNF_PH clock.
    """
    snf_ph = np.asarray(snf_ph, dtype=np.float64).ravel()
    n = snf_ph.size
    onsets = np.asarray(sniff_onsets_s, dtype=np.float64).ravel()
    onset_samp = np.rint(onsets * lv_fs).astype(np.int64)

    covered = np.zeros(n, dtype=bool)
    isi_bad = np.zeros(n, dtype=bool)
    for k in range(isi.size):
        a = max(int(onset_samp[k]), 0)
        b = min(int(onset_samp[k + 1]), n)
        if b <= a:
            continue
        covered[a:b] = True
        if not (isi[k] >= p5 and isi[k] <= p95):
            isi_bad[a:b] = True

    phase_noise = snf_ph == -1.0
    isi_outlier_mask = isi_bad & (~phase_noise)
    noise_mask = phase_noise | (~covered)
    return isi_outlier_mask, noise_mask


# ===========================================================================
# yaml-safe conversion helper (numpy scalars/arrays -> native python).
# ===========================================================================
def yaml_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [yaml_safe(v) for v in obj]
    return obj


# ===========================================================================
# Per-session pipeline
# ===========================================================================
def process_session(session_dirname: str, results_dir: Path,
                     n_shifts: int = N_SHIFTS, seed: int = NULL_SEED,
                     verbose: bool = True) -> dict:
    """Run the full per-session pipeline (steps 1-10 of the task brief) and
    persist intermediate + final per-session results under
    `results_dir/sessions/<session_dirname>/`.

    Returns a summary dict (small enough to keep in memory across all 6
    sessions for cross-session aggregation) -- the heavy arrays (full LFP,
    per-window coherence, null distributions) are saved to disk, not kept
    resident.
    """
    t0 = time.time()
    sess_dir = results_dir / "sessions" / session_dirname
    sess_dir.mkdir(parents=True, exist_ok=True)
    done_marker = sess_dir / "DONE.json"

    def log(msg):
        if verbose:
            print(f"[{session_dirname}] {msg}", flush=True)

    if done_marker.is_file():
        log("already completed -- loading cached summary (resume)")
        with open(sess_dir / "summary.json") as f:
            return json.load(f)

    # NoEligibleBinFileError (DEV-001 impact 3 -- only chunk-series members
    # exist, no whole-file alternative) is intentionally NOT caught here; it
    # propagates to the caller (run_analysis_np_paper1.py), which files a
    # fresh deviation and skips the session rather than guessing.
    files, missing = resolve_session_files(session_dirname)
    if files is None or missing:
        raise RuntimeError(f"session {session_dirname}: file resolution failed, "
                            f"missing={missing}")

    binfile_resolution = files.pop("_binfile_resolution")
    labview_duration_min = files.pop("_labview_duration_min")
    log(f"  DEV-001 Resolution B binfile pick: {binfile_resolution['justification']}")

    log("loading D-dict via load_experiment_data (kernel)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        D = load_experiment_data(files, checkpoint_dir=None)
    log(f"  loaded. NP_Fs={D['NP_Fs']}, LFP_Fs={D['LFP_Fs']}, LV_Fs={D['LV_Fs']}")

    nAP = D["LFP"].shape[0]
    xcoords_full, ycoords_full = channel_coords_full(D, files, nAP)
    valid_lfp_channels = np.arange(nAP)

    depth_idx, depth_ycoords, ymin, ymax = select_five_depths(
        xcoords_full, ycoords_full, valid_lfp_channels)
    log(f"  five depths (tip->surface): channels={depth_idx}, ycoords={depth_ycoords}")

    # ---- Full-duration LFP for the 5 selected channels (expensive I/O) ----
    log("extracting full-duration LFP for 5 selected channels (extract_full_lfp)...")
    t_lfp0 = time.time()
    LFP_full = extract_full_lfp(files, depth_idx, checkpoint_dir=None)
    t_lfp1 = time.time()
    LFP_Fs = float(D["LFP_Fs"])
    n_lfp_full = LFP_full.shape[1]
    length_min = n_lfp_full / LFP_Fs / 60.0
    log(f"  done in {t_lfp1 - t_lfp0:.1f}s. shape={LFP_full.shape} "
        f"({length_min:.2f} min at {LFP_Fs} Hz)")
    np.save(sess_dir / "lfp_full.npy", LFP_full)

    # ---- OUT-026 regression anchor: first 10 s must match D['LFP'] exactly ----
    n10 = D["LFP"].shape[1]
    anchor_ok = bool(np.allclose(LFP_full[:, :n10], D["LFP"][depth_idx, :],
                                  rtol=1e-9, atol=1e-12))
    if not anchor_ok:
        raise RuntimeError(
            f"session {session_dirname}: extract_full_lfp OUT-026 regression "
            "anchor FAILED (first-10s slice does not match the current "
            "loader's D['LFP'] within rtol=1e-9/atol=1e-12) -- blocking per "
            "contract failure_conditions; STOP, do not proceed."
        )
    log(f"  OUT-026 regression anchor: PASS (first {n10} samples match loader)")

    # ---- Sniff phase (pinned threshold_std=-0.5) ----
    log("compute_sniff_phase (threshold_std=-0.5)...")
    D2 = compute_sniff_phase(D, threshold_std=SNIFF_THRESHOLD_STD)
    n_sniffs_detected = int(D2["sniff_onsets_s"].size)

    sniff_mask_result = build_valid_sniff_mask(D2)
    valid_sniff_mask = sniff_mask_result["valid_sniff_mask"]
    isi = sniff_mask_result["isi"]
    p5 = sniff_mask_result["p5"]
    p95 = sniff_mask_result["p95"]
    inst_sniff_rate = sniff_mask_result["inst_sniff_rate"]
    n_sniffs_retained_isi = int(np.sum((isi >= p5) & (isi <= p95))) if isi.size else 0

    LV_Fs = float(D2["LV_Fs"])

    # ---- Ethanol / control-epoch mask (detect_eth_contact rule, verbatim) ----
    eth_out = detect_eth_contact(D2, eth_threshold=ETH_THRESHOLD)
    eth_contact_mask = eth_out["eth_contact_mask"]

    # ---- Candidate 4-s / 50%-overlap window grid on the LFP clock ----
    nperseg = int(round(WINDOW_LEN_S * LFP_Fs))
    step = int(round(nperseg * (1.0 - WINDOW_OVERLAP)))
    candidate_starts_samples = np.arange(0, n_lfp_full - nperseg + 1, step, dtype=np.int64)
    candidate_starts_s = candidate_starts_samples / LFP_Fs
    n_candidate = candidate_starts_samples.size
    log(f"  candidate windows: {n_candidate} (nperseg={nperseg}, step={step})")

    control_mask = build_window_control_mask(D2, candidate_starts_s, WINDOW_LEN_S)
    sniff_window_mask = window_all_valid_mask(valid_sniff_mask, LV_Fs,
                                               candidate_starts_s, WINDOW_LEN_S)
    final_mask = control_mask & sniff_window_mask
    control_valid_windows = candidate_starts_samples[final_mask]
    n_windows_used = int(control_valid_windows.size)
    n_excl_ethanol = int(np.sum(~control_mask))
    # Priority: a window already excluded for ethanol is NOT double-counted
    # under the noise/invalid-sniff reason (documented convention).
    n_excl_noise = int(np.sum(control_mask & ~sniff_window_mask))
    log(f"  windows used={n_windows_used}, excluded(ethanol)={n_excl_ethanol}, "
        f"excluded(noise/invalid-sniff)={n_excl_noise}")

    excluded_from_stats = n_windows_used < MIN_WINDOWS_FOR_STATS
    if excluded_from_stats:
        log(f"  *** PROH-002/CON-008: only {n_windows_used} < {MIN_WINDOWS_FOR_STATS} "
            "retained windows -- session EXCLUDED FROM DEPTH STATISTICS (still "
            "fully QC-reported; never silently dropped).")

    # ---- Usable-time coverage (documented convention: union of retained-
    #      window sample coverage / total LFP duration) ----
    coverage = np.zeros(n_lfp_full, dtype=bool)
    for s in control_valid_windows:
        coverage[s:s + nperseg] = True
    pct_usable = float(100.0 * coverage.sum() / n_lfp_full) if n_lfp_full else 0.0

    # ---- Sniff-rate distribution restricted to control epochs (OUT-030) ----
    onset_samp = np.rint(np.asarray(D2["sniff_onsets_s"]) * LV_Fs).astype(np.int64)
    eth_cumsum = np.concatenate(([0], np.cumsum(eth_contact_mask.astype(np.int64))))
    n_lv = eth_contact_mask.size
    span_in_control = np.zeros(isi.size, dtype=bool)
    for k in range(isi.size):
        a = max(int(onset_samp[k]), 0)
        b = min(int(onset_samp[k + 1]), n_lv)
        if b <= a:
            continue
        span_in_control[k] = (eth_cumsum[b] - eth_cumsum[a]) == 0
    control_rates = inst_sniff_rate[span_in_control] if isi.size else np.array([])
    control_rates = control_rates[np.isfinite(control_rates)]
    if control_rates.size:
        sniff_rate_median = float(np.median(control_rates))
        sniff_rate_q1 = float(np.percentile(control_rates, 25))
        sniff_rate_q3 = float(np.percentile(control_rates, 75))
        sniff_rate_iqr = sniff_rate_q3 - sniff_rate_q1
        sniff_rate_min = float(control_rates.min())
        sniff_rate_max = float(control_rates.max())
    else:
        sniff_rate_median = sniff_rate_q1 = sniff_rate_q3 = sniff_rate_iqr = float("nan")
        sniff_rate_min = sniff_rate_max = float("nan")

    # ---- Discarded-sections log (RQ-033/OUT-032; never silently drop) ----
    eth_spans = _contiguous_true_spans_s(eth_contact_mask, LV_Fs)
    isi_outlier_mask, noise_mask = _classify_invalid_sniff_reasons(
        D2["sniff_onsets_s"], D2["SNF_PH"], isi, p5, p95, LV_Fs)
    isi_spans = _contiguous_true_spans_s(isi_outlier_mask, LV_Fs)
    noise_spans = _contiguous_true_spans_s(noise_mask, LV_Fs)

    discards = (
        [{"start_s": a, "end_s": b, "reason": "ethanol"} for a, b in eth_spans]
        + [{"start_s": a, "end_s": b, "reason": "ISI outlier"} for a, b in isi_spans]
        + [{"start_s": a, "end_s": b, "reason": "non-sniff noise"} for a, b in noise_spans]
    )
    with open(sess_dir / "qc_discards.yaml", "w") as f:
        yaml.safe_dump(yaml_safe({"session": session_dirname, "discards": discards}),
                       f, sort_keys=False)
    log(f"  discarded spans logged: {len(eth_spans)} ethanol, {len(isi_spans)} ISI-outlier, "
        f"{len(noise_spans)} non-sniff-noise")

    # ---- Raw SNF resampled onto the LFP clock (PROH-009: raw, not SNF_PH) ----
    snf_lfp = resample_snf_to_lfp(D2["SNF"], LV_Fs, LFP_Fs, n_lfp_full)
    np.save(sess_dir / "snf_lfp.npy", snf_lfp)

    # Scientific-audit MINOR-2 fix (approved David Gire 2026-07-25): confine
    # circular_shift_null's shift to the non-padded prefix of snf_lfp.
    valid_snf_length = compute_valid_snf_length(D2["SNF"], LV_Fs, LFP_Fs, n_lfp_full)
    padded_fraction = 1.0 - (valid_snf_length / n_lfp_full) if n_lfp_full else 0.0
    if padded_fraction > 1e-9:
        log(f"  MINOR-2: snf_lfp has a held-constant tail pad "
            f"({padded_fraction:.1%} of samples, valid_length={valid_snf_length} "
            f"of {n_lfp_full}) -- circular_shift_null will confine shifts to "
            "the valid prefix.")

    # ---- Per-depth spectral engine (multitaper coherence/PSD + circular-
    #      shift null + theta reduction) ----
    depth_results = []
    for d_ord, (ch, ycoord) in enumerate(zip(depth_idx, depth_ycoords), start=1):
        log(f"  depth {d_ord}/5 (ch={ch}, ycoord={ycoord:.1f} um): "
            f"multitaper_psd_coherence over {n_windows_used} windows...")
        lfp_ch = LFP_full[d_ord - 1, :]
        t_c0 = time.time()
        psd_coh = multitaper_psd_coherence(lfp_ch, snf_lfp, control_valid_windows, fs=LFP_Fs)
        t_c1 = time.time()
        log(f"    coherence done in {t_c1 - t_c0:.2f}s; now circular_shift_null "
            f"(n_shifts={n_shifts}, seed={seed}, valid_length={valid_snf_length})...")
        t_n0 = time.time()
        null_result = circular_shift_null(lfp_ch, snf_lfp, control_valid_windows,
                                          n_shifts=n_shifts, seed=seed, fs=LFP_Fs,
                                          valid_length=valid_snf_length)
        t_n1 = time.time()
        log(f"    null done in {t_n1 - t_n0:.2f}s.")

        theta = reduce_theta_coherence(
            psd_coh["coherence"], psd_coh["freqs"], null_result["null_threshold"],
            null_coh_dist=null_result["null_distribution"], theta_band=THETA_BAND)

        depth_results.append({
            "depth_ordinal": d_ord,
            "channel_index": int(ch),
            "ycoord_um": float(ycoord),
            "freqs": psd_coh["freqs"],
            "coherence": psd_coh["coherence"],
            "lfp_psd": psd_coh["lfp_psd"],
            "snf_psd": psd_coh["snf_psd"],
            "null_threshold": null_result["null_threshold"],
            "theta": theta,
            "coherence_time_s": t_c1 - t_c0,
            "null_time_s": t_n1 - t_n0,
        })

        # Persist per-depth spectral arrays immediately (checkpoint).
        np.savez(
            sess_dir / f"spectral_depth{d_ord}.npz",
            freqs=psd_coh["freqs"], coherence=psd_coh["coherence"],
            lfp_psd=psd_coh["lfp_psd"], snf_psd=psd_coh["snf_psd"],
            null_threshold=null_result["null_threshold"],
        )

    total_time_s = time.time() - t0

    summary = {
        "session_dirname": session_dirname,
        "binfile_chosen": os.path.join(binfile_resolution["binPath"],
                                        binfile_resolution["binFile"]),
        "binfile_duration_min": binfile_resolution["duration_min"],
        "binfile_labview_duration_min": labview_duration_min,
        "binfile_ratio": binfile_resolution["ratio"],
        "binfile_within_tol": binfile_resolution["within_tol"],
        "binfile_warn": binfile_resolution["warn"],
        "binfile_justification": binfile_resolution["justification"],
        "valid_snf_length": int(valid_snf_length),
        "snf_padded_fraction": float(padded_fraction),
        "n_lfp_full": int(n_lfp_full),
        "LFP_Fs": LFP_Fs,
        "LV_Fs": LV_Fs,
        "length_min": float(length_min),
        "depth_channel_indices": [int(x) for x in depth_idx],
        "depth_ycoords_um": [float(x) for x in depth_ycoords],
        "ymin_um": float(ymin),
        "ymax_um": float(ymax),
        "n_candidate_windows": int(n_candidate),
        "n_windows_used": n_windows_used,
        "n_windows_excluded_ethanol": n_excl_ethanol,
        "n_windows_excluded_noise": n_excl_noise,
        "excluded_from_depth_stats": bool(excluded_from_stats),
        "pct_usable_time": pct_usable,
        "n_sniffs_detected": n_sniffs_detected,
        "n_sniffs_retained_isi": n_sniffs_retained_isi,
        "sniff_rate_median_hz": sniff_rate_median,
        "sniff_rate_q1_hz": sniff_rate_q1,
        "sniff_rate_q3_hz": sniff_rate_q3,
        "sniff_rate_iqr_hz": sniff_rate_iqr,
        "sniff_rate_min_hz": sniff_rate_min,
        "sniff_rate_max_hz": sniff_rate_max,
        "n_eth_spans": len(eth_spans),
        "n_isi_outlier_spans": len(isi_spans),
        "n_noise_spans": len(noise_spans),
        "depth_theta": [
            {
                "depth_ordinal": r["depth_ordinal"],
                "channel_index": r["channel_index"],
                "ycoord_um": r["ycoord_um"],
                **r["theta"],
            }
            for r in depth_results
        ],
        "control_valid_windows_count": n_windows_used,
        "total_processing_time_s": total_time_s,
        "extract_full_lfp_time_s": t_lfp1 - t_lfp0,
        "oob_regression_anchor_pass": anchor_ok,
    }
    summary = yaml_safe(summary)

    with open(sess_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(sess_dir / "summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    # Save control_valid_windows (sample indices) for the example-traces step.
    np.save(sess_dir / "control_valid_windows.npy", control_valid_windows)

    with open(done_marker, "w") as f:
        json.dump({"completed": True, "total_time_s": total_time_s}, f)

    log(f"session complete in {total_time_s:.1f}s "
        f"({total_time_s / 60.0:.2f} min).")
    return summary


# ===========================================================================
# Targeted recompute for the MINOR-2 fix (scientific-audit, approved David
# Gire 2026-07-25) on SESSIONS ALREADY PROCESSED under the pre-fix
# circular_shift_null. Reuses the cached lfp_full.npy / snf_lfp.npy /
# control_valid_windows.npy from disk -- does NOT re-run load_experiment_data
# or extract_full_lfp (the expensive I/O steps, unaffected by this fix).
# Only null_threshold/null_distribution (and the theta reduction fields that
# depend on them: sig_theta_fraction, peak_coh_percentile) can change;
# mean_theta_coh/peak_theta_coh/peak_theta_freq depend only on the OBSERVED
# coherence, not the null, and are mathematically unaffected -- verified,
# not merely asserted, by the recompute below (old vs new values compared).
# ===========================================================================
def recompute_null_minor2_fix(session_dirname: str, results_dir: Path,
                               n_shifts: int = N_SHIFTS, seed: int = NULL_SEED,
                               verbose: bool = True) -> dict:
    def log(msg):
        if verbose:
            print(f"[{session_dirname}] {msg}", flush=True)

    sess_dir = results_dir / "sessions" / session_dirname
    with open(sess_dir / "summary.json") as f:
        summary = json.load(f)

    LFP_Fs = float(summary["LFP_Fs"])
    LV_Fs = float(summary["LV_Fs"])
    n_lfp_full = int(summary["n_lfp_full"])
    depth_idx = summary["depth_channel_indices"]
    depth_ycoords = summary["depth_ycoords_um"]

    LFP_full = np.load(sess_dir / "lfp_full.npy")
    snf_lfp = np.load(sess_dir / "snf_lfp.npy")
    control_valid_windows = np.load(sess_dir / "control_valid_windows.npy")

    # Need SNF + LV_Fs to compute valid_snf_length -- lightweight (or_loaddat
    # only, no Kilosort/full LFP re-load).
    files, missing = resolve_session_files(session_dirname)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _TR, _TS, _FR, _ETH, SNF, _LV_Fs2 = or_loaddat(files["datFile"], files["datPath"])

    valid_snf_length = compute_valid_snf_length(SNF, LV_Fs, LFP_Fs, n_lfp_full)
    padded_fraction = 1.0 - (valid_snf_length / n_lfp_full) if n_lfp_full else 0.0
    log(f"  valid_snf_length={valid_snf_length}/{n_lfp_full} "
        f"(padded_fraction={padded_fraction:.4%})")

    new_depth_theta = []
    for d_ord, (ch, ycoord) in enumerate(zip(depth_idx, depth_ycoords), start=1):
        old_row = next(r for r in summary["depth_theta"] if r["depth_ordinal"] == d_ord)
        lfp_ch = LFP_full[d_ord - 1, :]

        npz_path = sess_dir / f"spectral_depth{d_ord}.npz"
        old_spec = np.load(npz_path)
        freqs = old_spec["freqs"]
        coherence = old_spec["coherence"]  # observed coherence -- UNCHANGED by this fix

        t0 = time.time()
        null_result = circular_shift_null(lfp_ch, snf_lfp, control_valid_windows,
                                          n_shifts=n_shifts, seed=seed, fs=LFP_Fs,
                                          valid_length=valid_snf_length)
        log(f"  depth {d_ord}/5 (ch={ch}): null recomputed in {time.time()-t0:.2f}s.")

        # Observed coherence must be bit-identical (fix only touches the null
        # shift range, never the observed/unshifted computation) -- verify,
        # don't just assume.
        if not np.array_equal(coherence, null_result["observed"]):
            raise RuntimeError(
                f"MINOR-2 recompute: observed coherence changed for "
                f"{session_dirname} depth {d_ord} -- this should be "
                "impossible (the fix only changes the shift range for null "
                "shifts, not the unshifted observed computation). STOP."
            )

        theta = reduce_theta_coherence(
            coherence, freqs, null_result["null_threshold"],
            null_coh_dist=null_result["null_distribution"], theta_band=THETA_BAND)

        # mean/peak theta coherence and peak frequency must be unchanged
        # (they depend only on observed coherence, never the null) --
        # verify bit-for-bit, don't just assume.
        for key in ("mean_theta_coh", "peak_theta_coh", "peak_theta_freq"):
            if not math.isclose(theta[key], old_row[key], rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(
                    f"MINOR-2 recompute: {key} changed for {session_dirname} "
                    f"depth {d_ord} ({old_row[key]} -> {theta[key]}) -- this "
                    "should be mathematically impossible (these fields don't "
                    "depend on the null). STOP, do not silently accept."
                )

        np.savez(npz_path, freqs=freqs, coherence=coherence,
                 lfp_psd=old_spec["lfp_psd"], snf_psd=old_spec["snf_psd"],
                 null_threshold=null_result["null_threshold"])

        new_depth_theta.append({
            "depth_ordinal": d_ord,
            "channel_index": int(ch),
            "ycoord_um": float(ycoord),
            **theta,
        })
        log(f"    sig_theta_fraction {old_row['sig_theta_fraction']:.4f} -> "
            f"{theta['sig_theta_fraction']:.4f}; peak_coh_percentile "
            f"{old_row['peak_coh_percentile']:.2f} -> {theta['peak_coh_percentile']:.2f}")

    summary["depth_theta"] = yaml_safe(new_depth_theta)
    summary["valid_snf_length"] = int(valid_snf_length)
    summary["snf_padded_fraction"] = float(padded_fraction)
    summary["minor2_null_recomputed"] = True

    with open(sess_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(sess_dir / "summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    return summary


# ===========================================================================
# Cross-session aggregation (RESULT-best-depth, RESULT-depth-stat, grand
# means, RESULT-example-traces). Runs after all (available) sessions have
# been processed via process_session().
# ===========================================================================
def _load_depth_spectra(sess_dir: Path, n_depths: int = 5):
    out = []
    for d in range(1, n_depths + 1):
        npz = np.load(sess_dir / f"spectral_depth{d}.npz")
        out.append({k: npz[k] for k in npz.files})
    return out


def build_animal_theta_table(summaries: list[dict]) -> dict:
    """DEV-002 (scientific-audit BLOCKER-1, resolved by David Gire
    2026-07-25): collapse each animal's session(s) into ONE animal-level
    mean_theta_coh per depth (average across sessions sharing an animal --
    currently only NP8, with 2021-11-01 and 2021-11-03) BEFORE any
    inferential/aggregate step, since scipy's Friedman test (inside
    depth_statistics) requires exactly one observation per block per
    condition (a strict animal x depth design). This collapses the true
    pseudo-replicate rather than silently double-counting it; no change to
    depth_statistics.py itself.

    Returns dict:
      animal_depth_mean_theta_coh -- {animal: {depth_ordinal: mean_theta_coh}}
      animal_sessions             -- {animal: [session_dirname, ...]}
    """
    by_animal_depth: dict[str, dict[int, list[float]]] = {}
    animal_sessions: dict[str, list[str]] = {}
    for s in summaries:
        animal = ANIMAL_MAP[s["session_dirname"]]
        animal_sessions.setdefault(animal, []).append(s["session_dirname"])
        for r in s["depth_theta"]:
            by_animal_depth.setdefault(animal, {}).setdefault(
                r["depth_ordinal"], []).append(r["mean_theta_coh"])

    animal_depth_mean = {
        animal: {d: float(np.mean(vals)) for d, vals in depth_map.items()}
        for animal, depth_map in by_animal_depth.items()
    }
    return {
        "animal_depth_mean_theta_coh": animal_depth_mean,
        "animal_sessions": animal_sessions,
    }


def build_best_depth_and_grand_means(summaries: list[dict]) -> dict:
    """RESULT-best-depth (per-ANIMAL best depth, overall ranking,
    per-ordinal consistency counts) + grand mean/SEM per depth for
    theta-coherence -- computed across the N_ANIMALS (5) independent
    animals, per DEV-002 (resolved 2026-07-25): NP8's two sessions
    (2021-11-01, 2021-11-03) are averaged into ONE animal-level observation
    per depth BEFORE this aggregation, so NP8 is not double-weighted.
    Exclusion from the >=30-window gate applies only to RESULT-depth-stat,
    per PROH-002/CON-008 ("excluded from the depth statistics ONLY") --
    this function uses ALL processed sessions/animals regardless."""
    animal_table = build_animal_theta_table(summaries)
    animal_depth_mean = animal_table["animal_depth_mean_theta_coh"]
    animal_sessions = animal_table["animal_sessions"]

    per_animal_best = []
    theta_matrix: dict[int, list[float]] = {}
    for animal, depth_map in animal_depth_mean.items():
        best_depth = max(depth_map, key=lambda d: depth_map[d])
        per_animal_best.append({
            "animal": animal,
            "sessions": animal_sessions[animal],
            "best_depth_ordinal": best_depth,
            "best_mean_theta_coh": depth_map[best_depth],
        })
        for d, v in depth_map.items():
            theta_matrix.setdefault(d, []).append(v)

    grand_mean_theta = {}
    for d, vals in theta_matrix.items():
        vals = np.asarray(vals, dtype=float)
        grand_mean_theta[d] = {
            "mean": float(vals.mean()),
            "sem": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan"),
            "n_animals": int(vals.size),
        }
    overall_best_ordinal = max(grand_mean_theta, key=lambda d: grand_mean_theta[d]["mean"])
    ranking = sorted(grand_mean_theta.keys(), key=lambda d: -grand_mean_theta[d]["mean"])

    ordinal_win_counts = {d: 0 for d in theta_matrix.keys()}
    for row in per_animal_best:
        ordinal_win_counts[row["best_depth_ordinal"]] = (
            ordinal_win_counts.get(row["best_depth_ordinal"], 0) + 1)

    return {
        "per_animal_best_depth": per_animal_best,
        "overall_best_depth_ordinal": int(overall_best_ordinal),
        "ranking_by_grand_mean_theta_coh": [int(d) for d in ranking],
        "per_ordinal_consistency_counts": {int(k): int(v) for k, v in ordinal_win_counts.items()},
        "grand_mean_theta_coh_by_depth": grand_mean_theta,
        "n_animals": N_ANIMALS,
        "n_sessions": N_SESSIONS,
        "animal_session_map": dict(ANIMAL_MAP),
        "note": (
            "DEV-002 (resolved 2026-07-25): 5 animals, 6 sessions total. "
            "NP8 contributes 2 sessions (2021-11-01, 2021-11-03), averaged "
            "into one animal-level mean_theta_coh per depth before this "
            "aggregation so NP8 is not double-weighted. All other animals "
            "contribute a single session each."
        ),
    }


def _animal_average_stack(summaries: list[dict], results_dir: Path, key: str):
    """Per-animal average of a per-depth spectral array `key` (coherence,
    lfp_psd, or null_threshold) across that animal's session(s), THEN return
    the stacked per-ANIMAL arrays for depth `d` (list length == N_ANIMALS,
    NP8's 2 sessions already collapsed to 1 animal-level curve). DEV-002."""
    by_animal: dict[str, dict[int, list[np.ndarray]]] = {}
    freqs = None
    snf_by_animal: dict[str, list[np.ndarray]] = {}
    for s in summaries:
        animal = ANIMAL_MAP[s["session_dirname"]]
        sess_dir = results_dir / "sessions" / s["session_dirname"]
        spectra = _load_depth_spectra(sess_dir)
        for i, sp in enumerate(spectra, start=1):
            if freqs is None:
                freqs = sp["freqs"]
            by_animal.setdefault(animal, {}).setdefault(i, []).append(sp[key])
        snf_by_animal.setdefault(animal, []).append(spectra[0]["snf_psd"])

    per_depth_animal_curves: dict[int, list[np.ndarray]] = {d: [] for d in range(1, 6)}
    for animal, depth_map in by_animal.items():
        for d, arrs in depth_map.items():
            per_depth_animal_curves[d].append(np.mean(np.vstack(arrs), axis=0))

    snf_animal_curves = [np.mean(np.vstack(arrs), axis=0) for arrs in snf_by_animal.values()]
    return freqs, per_depth_animal_curves, snf_animal_curves


def build_grand_mean_spectra(summaries: list[dict], results_dir: Path) -> dict:
    """Grand mean +/- SEM per depth ACROSS THE 5 ANIMALS (DEV-002, resolved
    2026-07-25) for LFP-PSD, coherence, and null-threshold spectra
    (FIG-PAPER1-PSD / FIG-PAPER1-COH underlying data) -- NP8's 2 sessions are
    averaged into one animal-level curve per depth before the across-animal
    grand mean/SEM, so NP8 is not double-weighted, consistent with the
    RESULT-best-depth / RESULT-depth-stat animal-level recompute."""
    freqs, per_depth_coh, snf_curves = _animal_average_stack(summaries, results_dir, "coherence")
    _, per_depth_psd, _ = _animal_average_stack(summaries, results_dir, "lfp_psd")
    _, per_depth_null, _ = _animal_average_stack(summaries, results_dir, "null_threshold")

    def _grand(per_depth_dict):
        out = {}
        for d, arrs in per_depth_dict.items():
            if not arrs:
                continue
            stack = np.vstack(arrs)
            mean = stack.mean(axis=0)
            sem = (stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
                   if stack.shape[0] > 1 else np.full(stack.shape[1], np.nan))
            out[d] = {"mean": mean, "sem": sem, "n_animals": stack.shape[0]}
        return out

    return {
        "freqs": freqs,
        "coherence_grand_mean_sem": _grand(per_depth_coh),
        "lfp_psd_grand_mean_sem": _grand(per_depth_psd),
        "null_threshold_grand_mean_sem": _grand(per_depth_null),
        "snf_psd_grand_mean": (np.vstack(snf_curves).mean(axis=0) if snf_curves else None),
        "n_animals": N_ANIMALS,
        "n_sessions": N_SESSIONS,
    }


def build_depth_statistics_table(summaries: list[dict]) -> pd.DataFrame:
    """Long-form ['experiment','depth','mean_theta_coh'] table AT THE ANIMAL
    LEVEL (DEV-002, resolved by David Gire 2026-07-25), restricted to
    sessions passing the >=30-retained-window gate (PROH-002/CON-008) before
    the animal-level average is taken. `experiment` holds ANIMAL labels
    (NP8, Np10, NP12, NP15, NP22) -- a strict 5-animal x 5-depth (25-row)
    balanced design, since scipy's Friedman test (inside depth_statistics)
    requires exactly one observation per block per condition. NP8's two
    sessions (2021-11-01, 2021-11-03) are averaged per depth into its single
    row; no change to depth_statistics.py itself is required."""
    eligible = [s for s in summaries if not s.get("excluded_from_depth_stats")]
    animal_table = build_animal_theta_table(eligible)
    rows = []
    for animal, depth_map in animal_table["animal_depth_mean_theta_coh"].items():
        for d, v in depth_map.items():
            rows.append({
                "experiment": animal,
                "depth": f"D{d}",
                "mean_theta_coh": v,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RESULT-example-traces (FIG-PAPER1-TRACES display data; presentation-only --
# does not feed any statistical conclusion). Display convention (contract,
# non-scientific): representative experiment = most retained control-valid
# windows, tie-break earliest date; representative segment = first qualifying
# contiguous control-valid span of a fixed display length.
# ---------------------------------------------------------------------------
DISPLAY_SEGMENT_S = 10.0  # fixed display length (documented convention)
BANDS = {
    "theta_2_12": (2.0, 12.0),
    "beta_18_30": (18.0, 30.0),
    "gamma_65_100": (65.0, 100.0),
}


def pick_representative_session(summaries: list[dict]) -> dict:
    def key(s):
        date = SESSION_DATES[SESSION_DIRNAMES.index(s["session_dirname"])]
        return (-s["n_windows_used"], date)
    return sorted(summaries, key=key)[0]


def _find_first_contiguous_segment(control_valid_windows: np.ndarray, nperseg: int,
                                    step: int, min_len_samples: int):
    """First start sample of a run of consecutive (spaced by `step`) retained
    windows whose combined coverage (N-1)*step + nperseg >= min_len_samples."""
    cvw = np.sort(np.asarray(control_valid_windows, dtype=np.int64))
    if cvw.size == 0:
        return None
    n_needed = int(np.ceil((min_len_samples - nperseg) / step)) + 1
    n_needed = max(n_needed, 1)
    run_start = cvw[0]
    run_len = 1
    for i in range(1, cvw.size):
        if cvw[i] - cvw[i - 1] == step:
            run_len += 1
        else:
            run_start = cvw[i]
            run_len = 1
        if run_len >= n_needed:
            start_of_run = cvw[i - run_len + 1]
            return int(start_of_run)
    return None


def build_example_traces(summaries: list[dict], results_dir: Path) -> dict:
    rep = pick_representative_session(summaries)
    sess_dir = results_dir / "sessions" / rep["session_dirname"]
    LFP_Fs = rep["LFP_Fs"]
    nperseg = int(round(WINDOW_LEN_S * LFP_Fs))
    step = int(round(nperseg * (1.0 - WINDOW_OVERLAP)))
    min_len_samples = int(round(DISPLAY_SEGMENT_S * LFP_Fs))

    cvw = np.load(sess_dir / "control_valid_windows.npy")
    start = _find_first_contiguous_segment(cvw, nperseg, step, min_len_samples)
    if start is None:
        warnings.warn(
            f"build_example_traces: session {rep['session_dirname']} has no "
            f"contiguous retained span >= {DISPLAY_SEGMENT_S}s; falling back "
            "to the single longest available retained window.")
        start = int(cvw[0]) if cvw.size else 0
        min_len_samples = nperseg

    end = start + min_len_samples
    LFP_full = np.load(sess_dir / "lfp_full.npy", mmap_mode="r")
    snf_lfp = np.load(sess_dir / "snf_lfp.npy", mmap_mode="r")
    seg_lfp = np.asarray(LFP_full[:, start:end])
    seg_snf = np.asarray(snf_lfp[start:end])
    t_s = np.arange(seg_lfp.shape[1], dtype=np.float64) / LFP_Fs + start / LFP_Fs

    import scipy.signal

    def bandpass(x, lo, hi, fs):
        b, a = scipy.signal.butter(4, [lo / (fs / 2.0), hi / (fs / 2.0)], btype="band")
        padlen = 3 * (max(len(a), len(b)) - 1)
        return scipy.signal.filtfilt(b, a, x, padtype="odd", padlen=padlen)

    per_depth_bands = []
    for d in range(seg_lfp.shape[0]):
        row = {"unfiltered": seg_lfp[d].copy()}
        for name, (lo, hi) in BANDS.items():
            filt = bandpass(seg_lfp[d], lo, hi, LFP_Fs)
            row[name] = filt
            if name == "gamma_65_100":
                row["gamma_65_100_envelope"] = np.abs(scipy.signal.hilbert(filt))
        per_depth_bands.append(row)

    # Inhalation mask from SNF_PH on this segment (DISPLAY-ONLY convention:
    # inhalation == first half of the 0..1 phase ramp, i.e. phase in [0,0.5);
    # NOT pinned anywhere in the contract/repo_map -- flagged for human
    # confirmation, affects this figure's coloring only, no statistical use).
    files, _ = resolve_session_files(rep["session_dirname"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        D = load_experiment_data(files, checkpoint_dir=None)
    D2 = compute_sniff_phase(D, threshold_std=SNIFF_THRESHOLD_STD)
    LV_Fs = rep["LV_Fs"]
    lv_start = int(round(start / LFP_Fs * LV_Fs))
    lv_end = int(round(end / LFP_Fs * LV_Fs))
    snf_ph_seg = D2["SNF_PH"][lv_start:lv_end]
    inhalation_mask_lv = (snf_ph_seg >= 0.0) & (snf_ph_seg < 0.5)
    onsets_s = D2["sniff_onsets_s"]
    seg_t0, seg_t1 = start / LFP_Fs, end / LFP_Fs
    onset_markers_s = onsets_s[(onsets_s >= seg_t0) & (onsets_s < seg_t1)] - seg_t0

    return {
        "session": rep["session_dirname"],
        "segment_start_s": float(start / LFP_Fs),
        "segment_end_s": float(end / LFP_Fs),
        "t_s": t_s,
        "depth_channel_indices": rep["depth_channel_indices"],
        "depth_ycoords_um": rep["depth_ycoords_um"],
        "per_depth_bands": per_depth_bands,
        "snf_segment": seg_snf,
        "inhalation_mask_lv_rate": inhalation_mask_lv,
        "inhalation_mask_lv_fs": LV_Fs,
        "sniff_onset_markers_s": onset_markers_s,
        "display_convention_note": (
            "Representative experiment = most retained all-control/"
            "all-valid-sniff 4-s windows, tie-break earliest date. "
            "Representative segment = first qualifying contiguous "
            "control-valid span of fixed length "
            f"{DISPLAY_SEGMENT_S}s. Non-scientific, presentation only."
        ),
    }


# ===========================================================================
# Gap-fill (report-author-identified, 2026-07-25 -- see coordinator message):
# two contract-required pieces of data were never persisted during the P3
# run. Both are cheap: reuse already-cached per-session arrays, no raw .bin
# reload, no re-run of the expensive circular-shift null. NOT a deviation --
# completing already-specified work, not a new scientific/data judgment.
# ===========================================================================

def build_example_spectrogram(results_dir: Path, verbose: bool = True) -> dict:
    """Gap 1 (OUT-009/010, feeds FIG-PAPER1-SPECTRO): per-window time-resolved
    LFP-power / SNF-power / coherence spectrograms for the SAME representative
    session already chosen for FIG-PAPER1-TRACES (RESULT-example-traces_meta.yaml
    -- 06-24-2022, 573 retained windows, the most of any session). Reuses the
    cached lfp_full.npy / snf_lfp.npy / control_valid_windows.npy on disk; does
    NOT reload the raw .bin or re-run extract_full_lfp.

    `multitaper_psd_coherence` already computes coh_per_window internally and
    RETURNS it (confirmed by reading its source) -- so the coherence
    spectrogram needs no new math, just persisting an already-computed field.
    Per-window LFP-power / SNF-power spectrograms are NOT returned by the
    pooled call (only the aggregate lfp_psd/snf_psd), so they are obtained by
    re-invoking the SAME validated, UNMODIFIED multitaper_psd_coherence once
    per window with a length-1 window_starts array (i.e. n_windows=1 in its
    own internal scale formula) -- this is the function's own single-window
    multitaper PSD estimate for that window, not a reimplementation.

    Verified, not merely asserted:
      (a) per-window coherence from the individual single-window calls must
          equal coh_per_window's row from ONE pooled call (both are the same
          formula for the same window; coherence is scale/pooling-invariant
          for a single window) -- checked bit-for-bit below.
      (b) mean_i(per-window lfp_psd_i) / mean_i(per-window snf_psd_i) across
          all retained windows must equal the CACHED aggregate lfp_psd/snf_psd
          in spectral_depth<N>.npz to floating-point precision (a real
          mathematical identity: pooled PSD = scale_N * sum_i S_i_raw =
          mean_i(per-window PSD_i) since per-window PSD_i = scale_1 * S_i_raw
          and scale_N = scale_1/N) -- checked below, raises if it does not hold.
    """
    def log(msg):
        if verbose:
            print(f"[spectrogram] {msg}", flush=True)

    with open(results_dir / "RESULT-example-traces_meta.yaml") as f:
        traces_meta = yaml.safe_load(f)
    sess = traces_meta["session"]
    log(f"representative session (from RESULT-example-traces_meta.yaml): {sess}")

    sess_dir = results_dir / "sessions" / sess
    with open(sess_dir / "summary.json") as f:
        summary = json.load(f)

    LFP_Fs = float(summary["LFP_Fs"])
    LV_Fs = float(summary["LV_Fs"])
    depth_idx = summary["depth_channel_indices"]
    depth_ycoords = summary["depth_ycoords_um"]
    nperseg = int(round(WINDOW_LEN_S * LFP_Fs))

    LFP_full = np.load(sess_dir / "lfp_full.npy")
    snf_lfp = np.load(sess_dir / "snf_lfp.npy")
    control_valid_windows = np.sort(np.load(sess_dir / "control_valid_windows.npy"))
    n_windows = control_valid_windows.size
    window_center_s = (control_valid_windows + nperseg / 2.0) / LFP_Fs

    # ---- instantaneous sniff rate at each window center (Gap 2's recompute
    # serves this too -- only needs SNF/ETH/LV_Fs from the LabView .dat). ----
    files, missing = resolve_session_files(sess)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _TR, _TS, _FR, ETH, SNF, _LV_Fs2 = or_loaddat(files["datFile"], files["datPath"])
    D2 = compute_sniff_phase({"SNF": SNF, "ETH": ETH, "LV_Fs": LV_Fs},
                             threshold_std=SNIFF_THRESHOLD_STD)
    onsets_s = np.asarray(D2["sniff_onsets_s"], dtype=np.float64)
    isi = np.diff(onsets_s)
    with np.errstate(divide="ignore"):
        inst_rate = 1.0 / isi
    sniff_rate_at_window = np.full(window_center_s.shape, np.nan)
    if isi.size:
        idx = np.searchsorted(onsets_s, window_center_s, side="right") - 1
        valid = (idx >= 0) & (idx < isi.size)
        sniff_rate_at_window[valid] = inst_rate[idx[valid]]

    freqs = None
    per_depth_lfp_psd_spectrogram = {}
    per_depth_coh_spectrogram = {}
    snf_psd_spectrogram = None

    for d_ord, ch in enumerate(depth_idx, start=1):
        log(f"depth {d_ord}/5 (ch={ch}): per-window spectrogram over {n_windows} windows...")
        lfp_ch = LFP_full[d_ord - 1, :]

        # (a) ONE pooled call -> coh_per_window (already computed internally,
        # just not previously persisted).
        pooled = multitaper_psd_coherence(lfp_ch, snf_lfp, control_valid_windows, fs=LFP_Fs)
        if freqs is None:
            freqs = pooled["freqs"]
        coh_per_window_pooled = pooled["coh_per_window"]

        # (b) per-window individual calls -> per-window lfp_psd (+ coherence,
        # for the cross-check against (a); + snf_psd, once only, depth 1).
        lfp_rows = np.empty((n_windows, freqs.size), dtype=float)
        coh_rows = np.empty((n_windows, freqs.size), dtype=float)
        snf_rows = (np.empty((n_windows, freqs.size), dtype=float)
                    if snf_psd_spectrogram is None else None)
        for i, w in enumerate(control_valid_windows):
            r = multitaper_psd_coherence(lfp_ch, snf_lfp, np.array([w]), fs=LFP_Fs)
            lfp_rows[i] = r["lfp_psd"]
            coh_rows[i] = r["coherence"]
            if snf_rows is not None:
                snf_rows[i] = r["snf_psd"]
        if snf_rows is not None:
            snf_psd_spectrogram = snf_rows

        # ---- verification (a): per-window coherence must equal coh_per_window ----
        if not np.allclose(coh_rows, coh_per_window_pooled, rtol=1e-9, atol=1e-12):
            raise RuntimeError(
                f"build_example_spectrogram: per-window coherence (individual "
                f"single-window calls) does not match coh_per_window (pooled "
                f"call) for {sess} depth {d_ord} -- these must be mathematically "
                "identical (same formula, same single window). STOP."
            )
        # ---- verification (b): mean of per-window PSD must equal the cached
        # aggregate PSD in spectral_depth<N>.npz. ----
        cached = np.load(sess_dir / f"spectral_depth{d_ord}.npz")
        recomputed_agg_lfp_psd = lfp_rows.mean(axis=0)
        if not np.allclose(recomputed_agg_lfp_psd, cached["lfp_psd"], rtol=1e-9, atol=1e-15):
            raise RuntimeError(
                f"build_example_spectrogram: mean(per-window lfp_psd) does not "
                f"match the cached aggregate lfp_psd for {sess} depth {d_ord} -- "
                "pooled PSD must equal the mean of per-window PSDs (see "
                "docstring derivation). STOP, do not silently accept."
            )
        if snf_rows is not None:
            recomputed_agg_snf_psd = snf_psd_spectrogram.mean(axis=0)
            if not np.allclose(recomputed_agg_snf_psd, cached["snf_psd"], rtol=1e-9, atol=1e-15):
                raise RuntimeError(
                    f"build_example_spectrogram: mean(per-window snf_psd) does "
                    f"not match the cached aggregate snf_psd for {sess} depth "
                    f"{d_ord}. STOP, do not silently accept."
                )
        log(f"  verified: per-window coherence == coh_per_window; "
            f"mean(per-window PSD) == cached aggregate PSD.")

        per_depth_lfp_psd_spectrogram[d_ord] = lfp_rows
        per_depth_coh_spectrogram[d_ord] = coh_rows

    return {
        "session": sess,
        "freqs": freqs,
        "window_center_s": window_center_s,
        "sniff_rate_at_window_hz": sniff_rate_at_window,
        "depth_channel_indices": depth_idx,
        "depth_ycoords_um": depth_ycoords,
        "lfp_psd_spectrogram_by_depth": per_depth_lfp_psd_spectrogram,
        "coh_spectrogram_by_depth": per_depth_coh_spectrogram,
        "snf_psd_spectrogram": snf_psd_spectrogram,
    }


def build_sniff_rate_raw(results_dir: Path, verbose: bool = True) -> dict:
    """Gap 2 (feeds FIG-PAPER1-SNIFFRATE's histogram_or_kde visual_encoding,
    all 6 sessions): raw per-sniff instantaneous-rate arrays. Only the
    5-number summary (median/Q1/Q3/IQR/min/max) was frozen in
    RESULT-sniff-rate.yaml -- no raw array existed anywhere, so a real
    histogram/KDE cannot be drawn. Cheap recompute: only needs
    D['SNF']/D['ETH']/D['LV_Fs'] from the LabView .dat file (via or_loaddat)
    -- does NOT reload the raw NP .bin/LFP (no extract_full_lfp, no Kilosort
    load).

    Applies the SAME control-epoch restriction (detect_eth_contact's
    eth_contact_mask, exactly mirroring process_session's inline logic) ALREADY
    used for the frozen 5-number summary -- verified bit-for-bit below (not
    merely assumed) by recomputing median/Q1/Q3/min/max from this raw array
    and comparing to the frozen RESULT-sniff-rate.yaml values. This is
    therefore the UNTRIMMED control-epoch distribution (MINOR-3 label), i.e.
    NOT further restricted by the ISI-percentile trim that gates coherence
    windows -- doing so would make the raw array inconsistent with the frozen
    summary it is meant to back. The per-sniff ISI-trim-validity (rule 3) is
    included as a companion boolean array per session so downstream figure
    code can still distinguish/overlay it without altering what the primary
    (frozen) summary describes.
    """
    def log(msg):
        if verbose:
            print(f"[sniff-rate-raw] {msg}", flush=True)

    with open(results_dir / "RESULT-sniff-rate.yaml") as f:
        frozen_doc = yaml.safe_load(f)
    frozen_rows = {r["session"]: r for r in frozen_doc["rows"]}

    out = {}
    for s in SESSION_DIRNAMES:
        files, missing = resolve_session_files(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _TR, _TS, _FR, ETH, SNF, LV_Fs = or_loaddat(files["datFile"], files["datPath"])
        D2 = compute_sniff_phase({"SNF": SNF, "ETH": ETH, "LV_Fs": LV_Fs},
                                 threshold_std=SNIFF_THRESHOLD_STD)

        mask_result = build_valid_sniff_mask(D2)
        isi = mask_result["isi"]
        inst_rate = mask_result["inst_sniff_rate"]
        p5, p95 = mask_result["p5"], mask_result["p95"]
        isi_trim_valid = (isi >= p5) & (isi <= p95) if isi.size else np.array([], dtype=bool)

        eth_out = detect_eth_contact(D2, eth_threshold=ETH_THRESHOLD)
        eth_contact_mask = eth_out["eth_contact_mask"]
        onset_samp = np.rint(np.asarray(D2["sniff_onsets_s"]) * LV_Fs).astype(np.int64)
        eth_cumsum = np.concatenate(([0], np.cumsum(eth_contact_mask.astype(np.int64))))
        n_lv = eth_contact_mask.size
        span_in_control = np.zeros(isi.size, dtype=bool)
        for k in range(isi.size):
            a = max(int(onset_samp[k]), 0)
            b = min(int(onset_samp[k + 1]), n_lv)
            if b <= a:
                continue
            span_in_control[k] = (eth_cumsum[b] - eth_cumsum[a]) == 0

        control_rate = inst_rate[span_in_control]
        control_isi_trim_valid = isi_trim_valid[span_in_control]
        finite_mask = np.isfinite(control_rate)
        control_rate = control_rate[finite_mask]
        control_isi_trim_valid = control_isi_trim_valid[finite_mask]

        # ---- verification: must reproduce the frozen 5-number summary. ----
        frozen = frozen_rows[s]
        if control_rate.size:
            checks = {
                "median_hz": float(np.median(control_rate)),
                "q1_hz": float(np.percentile(control_rate, 25)),
                "q3_hz": float(np.percentile(control_rate, 75)),
                "min_hz_untrimmed_control_epoch": float(control_rate.min()),
                "max_hz_untrimmed_control_epoch": float(control_rate.max()),
            }
            for key, val in checks.items():
                if not math.isclose(val, frozen[key], rel_tol=0, abs_tol=1e-9):
                    raise RuntimeError(
                        f"build_sniff_rate_raw: recomputed {key}={val} does not "
                        f"match frozen RESULT-sniff-rate.yaml {key}={frozen[key]} "
                        f"for session {s} -- raw array would be inconsistent "
                        "with the already-frozen summary. STOP."
                    )
        log(f"[{s}] n_raw_sniff_rate={control_rate.size} "
            f"(n_isi_trim_valid={int(control_isi_trim_valid.sum())}) -- "
            "verified consistent with frozen RESULT-sniff-rate.yaml 5-number summary.")

        out[s] = {
            "inst_rate_hz": control_rate,
            "isi_trim_valid": control_isi_trim_valid,
        }
    return out


def run_aggregation(summaries: list[dict], results_dir: Path) -> dict:
    best_depth = build_best_depth_and_grand_means(summaries)
    grand_spectra = build_grand_mean_spectra(summaries, results_dir)
    # DEV-002 (resolved 2026-07-25): animal-level table (experiment column
    # now holds animal labels, NP8's 2 sessions pre-averaged per depth).
    depth_table = build_depth_statistics_table(summaries)

    n_stat_animals = depth_table["experiment"].nunique() if not depth_table.empty else 0
    depth_stat = None
    depth_stat_blocked_reason = None
    lmm_convergence_note = None
    if n_stat_animals < 3:
        depth_stat_blocked_reason = (
            f"Only {n_stat_animals} animals passed the >=30-window gate "
            "(PROH-002/CON-008) -- fewer than 3, repeated-measures depth "
            "statistics are underpowered; execution blocked per contract "
            "failure_conditions. Reported, not silently skipped."
        )
    else:
        # Report (never hide) any MixedLM convergence warning -- coordinator
        # explicitly asked to double-check the LMM has enough residual df to
        # fit sensibly with n=5 animals x 5 depths = 25 observations
        # (DEV-002 shrinks the groups count from 6 sessions to 5 animals).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            depth_stat = depth_statistics(depth_table)
        conv_warnings = [str(w.message) for w in caught
                         if "convergence" in str(w.message).lower()
                         or "boundary" in str(w.message).lower()]
        if conv_warnings:
            lmm_convergence_note = (
                f"MixedLM raised {len(conv_warnings)} convergence-related "
                f"warning(s) fitting the 5-animal x 5-depth (n=25) table: "
                + "; ".join(conv_warnings) + ". Reported per coordinator "
                "instruction, not hidden -- interpret the LMM chi-square/p "
                "alongside the (convergence-warning-free) Friedman result."
            )

    traces = build_example_traces(summaries, results_dir)

    return {
        "best_depth": best_depth,
        "grand_spectra": grand_spectra,
        "depth_stat_table_n_animals": int(n_stat_animals),
        "depth_stat": depth_stat,
        "depth_stat_blocked_reason": depth_stat_blocked_reason,
        "depth_stat_lmm_convergence_note": lmm_convergence_note,
        "example_traces": traces,
    }

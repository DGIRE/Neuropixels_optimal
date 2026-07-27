"""np_demo4_analysis.py — NP-DEMO-4 analysis module (Blueprint v2 P3, D7).

Analyzes instantaneous sniff rate, population firing rate, and ETH sensor
signals across 6 Neuropixels recording sessions to test whether sniff rate
differs during first ETH contact vs. pre-valve control period.

Contract: NP-DEMO-4 contract_v001.yaml (approved by David Gire 2026-07-23).

This module is an ADDITIVE caller of the validated kernel under
"Optimized Python" (load_experiment_data, _fix_ks_dir pattern) plus the
NP-DEMO-3 primitive detect_eth_contact and two NEW kernel functions
compute_sniff_rate and compute_firing_rate_50ms.

Prohibited (contract_v001.yaml prohibited_changes):
  * Using LFP instead of SNF for sniff rate (PROH-001).
  * Modifying detect_eth_contact.py.
  * Per-experiment threshold adjustment (fixed 0.05 for all sessions).
  * Including even-TR LabView trials.
  * Computing firing rate with window other than 50 ms.
  * Editing Optimized Python/, Golden Fixtures/, or tolerance configs.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date

import numpy as np
import scipy.stats

# ---------------------------------------------------------------------------
# Kernel wiring — sibling import, never a copy.
# ---------------------------------------------------------------------------
_KERNEL_DIR = r"C:\Projects\Repos\Neuropixels\Optimized Python"
if _KERNEL_DIR not in sys.path and os.path.isdir(_KERNEL_DIR):
    sys.path.insert(0, _KERNEL_DIR)

_SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))
if _SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, _SOFTWARE_DIR)

from load_experiment_data import load_experiment_data          # noqa: E402
from lib.or_validate_files import or_validate_files            # noqa: E402
from analyses._common import matlab_round                      # noqa: E402

from detect_eth_contact import detect_eth_contact              # noqa: E402
from compute_sniff_rate import compute_sniff_rate              # noqa: E402
from compute_firing_rate_50ms import compute_firing_rate_50ms  # noqa: E402


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
def _parse_session_date(session_dir: str) -> str:
    """Parse YYYY-MM-DD date from folder name (accepts MM-DD-YYYY or YYYY-MM-DD)."""
    base = os.path.basename(os.path.normpath(session_dir))
    m = re.search(r"(\d{4})[_\-](\d{1,2})[_\-](\d{1,2})", base)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return date(y, mo, d).isoformat()
    m = re.search(r"(\d{1,2})[_\-](\d{1,2})[_\-](\d{4})", base)
    if m:
        mo, d, y = (int(x) for x in m.groups())
        return date(y, mo, d).isoformat()
    return base


def _fix_ks_dir(files: dict, missing: list[str]) -> tuple[dict, list[str]]:
    """Promote phy sub-directory ksDir to its parent when probe files live one
    level up. Required for the multi-shank 2022-09-14 session."""
    _KS_PROBE_FILES = ("channel_map.npy", "channel_positions.npy")
    if files.get("ksDir") and any(
        not os.path.isfile(os.path.join(files["ksDir"], f)) for f in _KS_PROBE_FILES
    ):
        parent = os.path.dirname(files["ksDir"])
        if all(os.path.isfile(os.path.join(parent, f)) for f in _KS_PROBE_FILES):
            files["ksDir"] = parent
            missing = [
                m for m in missing
                if "channel_map" not in m and "channel_positions" not in m
            ]
    return files, missing


def _segment_odd_tr_trials(D: dict) -> list[dict]:
    """Extract LabView trials from odd TR values using TS reset detection.

    Each odd TR value may contain a brief pre-trial prefix (TS not starting at 0)
    followed by a TS reset to near-0 and then the actual trial period. The actual
    trial is the sub-period starting at the TS reset, with the first TS sample
    corrected to 0 (as the first sample after a TS reset is a corrupted value).

    If no TS reset is found within a TR block, the entire block is treated as one
    trial with the first TS sample corrected to 0.

    Returns a list of dicts with keys:
      tr_value    — odd TR value (int)
      global_idx  — array of global sample indices for this trial
      ts_s        — within-trial timestamps in seconds (first TS corrected to 0)
      n_samples   — number of samples in trial

    Excludes even TR values (inter-trial intervals).
    """
    TR = np.asarray(D["TR"]).ravel()
    TS = np.asarray(D["TS"]).ravel()
    LV_Fs = float(D["LV_Fs"])

    # TS step at LV_Fs=125 Hz is ~8ms. A "reset" is a negative TS diff or a
    # large backward jump (e.g. > 100ms backward)
    RESET_THRESHOLD_MS = -100.0

    odd_tr_values = np.unique(TR[TR % 2 == 1])
    trials = []
    for tr_val in sorted(odd_tr_values):
        mask = TR == tr_val
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue

        ts_block = TS[mask].astype(np.float64)

        # Find TS resets within this TR block
        ts_diffs = np.diff(ts_block)
        reset_positions = np.flatnonzero(ts_diffs < RESET_THRESHOLD_MS)

        if reset_positions.size > 0:
            # Use the sub-period starting at the LAST reset (the actual trial)
            # The last reset is most likely the transition from pre-trial to trial
            reset_pos = int(reset_positions[-1]) + 1  # index into ts_block/idx
            trial_idx = idx[reset_pos:]
            ts_trial = ts_block[reset_pos:]
        else:
            # No reset found — use the entire block
            trial_idx = idx
            ts_trial = ts_block

        if trial_idx.size == 0:
            continue

        # Correct corrupted first TS sample to 0
        ts_trial = ts_trial.copy()
        ts_trial[0] = 0.0

        # Convert to seconds
        ts_s = ts_trial / 1000.0

        # Filter to 0-40 s window (contract: TS=0-40000ms)
        window_mask = (ts_s >= 0.0) & (ts_s <= 40.0)
        if not window_mask.any():
            continue

        idx_win = trial_idx[window_mask]
        ts_win = ts_s[window_mask]

        # Deduplicate: if TS repeats within a trial (startup artifact in TR=1),
        # keep only the first occurrence of each unique TS value to preserve
        # one sample per time point (avoids inflating the per-trial matrix).
        if len(ts_win) > 0:
            _, first_occ = np.unique(ts_win, return_index=True)
            if len(first_occ) < len(ts_win):  # duplicates present
                first_occ_sorted = np.sort(first_occ)
                idx_win = idx_win[first_occ_sorted]
                ts_win = ts_win[first_occ_sorted]

        if len(idx_win) == 0:
            continue

        trials.append(dict(
            tr_value=int(tr_val),
            global_idx=idx_win,
            ts_s=ts_win,
            n_samples=int(len(idx_win)),
        ))
    return trials


def _apply_unit_inclusion(D: dict, session_duration_s: float) -> tuple[np.ndarray, int, int]:
    """Apply CON-001 inclusion filter.

    Returns (included_unit_ids, n_units_recorded, n_units_included).
    Inclusion: mean FR >= 0.1 Hz AND total spike count >= 5000.
    """
    sp = D["sp"]
    unitIDs = np.asarray(sp["unitIDs"] if "unitIDs" in sp else D.get("unitIDs", []))
    if unitIDs.size == 0:
        # Fall back to D-level unitIDs
        unitIDs = np.asarray(D.get("unitIDs", []))
    unitFiringRate = np.asarray(
        sp["unitFiringRate"] if "unitFiringRate" in sp else D.get("unitFiringRate", [])
    )
    if unitFiringRate.size == 0:
        unitFiringRate = np.asarray(D.get("unitFiringRate", []))
    clu_all = np.asarray(sp["clu"]).ravel()

    n_units_recorded = int(unitIDs.size)

    # Count total spikes per unit
    included_ids = []
    for i, uid in enumerate(unitIDs):
        fr = float(unitFiringRate[i]) if i < unitFiringRate.size else 0.0
        n_spikes = int(np.sum(clu_all == uid))
        if fr >= 0.1 and n_spikes >= 5000:
            included_ids.append(uid)

    included_unit_ids = np.array(included_ids, dtype=unitIDs.dtype)
    n_units_included = len(included_ids)
    return included_unit_ids, n_units_recorded, n_units_included


def _count_eth_contacts_per_trial(D: dict, trials: list[dict]) -> list[int]:
    """For each trial, count discrete ETH contact events (contiguous above-threshold
    runs) within that trial's sample range."""
    eth_contact_mask = np.asarray(D["eth_contact_mask"], dtype=bool).ravel()
    counts = []
    for trial in trials:
        idx = trial["global_idx"]
        mask_trial = eth_contact_mask[idx]
        # Count contiguous runs
        if mask_trial.size == 0:
            counts.append(0)
            continue
        padded = np.concatenate(([False], mask_trial, [False]))
        rising = ~padded[:-1] & padded[1:]
        counts.append(int(np.count_nonzero(rising)))
    return counts


def _first_eth_contact_in_trial(eth_contact_mask_global: np.ndarray,
                                 trial_global_idx: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Find the sample indices of the FIRST discrete ETH contact event within a trial.

    Returns (contact_global_idx, contact_local_mask) or (None, None) if no contact.
    """
    local_mask = eth_contact_mask_global[trial_global_idx]
    if not local_mask.any():
        return None, None

    padded = np.concatenate(([False], local_mask, [False]))
    rising = np.flatnonzero(~padded[:-1] & padded[1:])
    falling = np.flatnonzero(padded[:-1] & ~padded[1:])
    if rising.size == 0:
        return None, None

    # First run
    r_start = rising[0]
    r_end = falling[0] if falling.size > 0 else local_mask.size

    # Global indices of this contact
    contact_local = np.zeros(local_mask.size, dtype=bool)
    contact_local[r_start:r_end] = True
    contact_global_idx = trial_global_idx[contact_local]
    return contact_global_idx, contact_local


def rank_biserial_r(diffs: np.ndarray) -> float:
    """Rank-biserial r for Wilcoxon signed-rank test.

    r = (W+ - W-) / (W+ + W-) where W+/W- are rank sums of positive/negative diffs.
    Returns NaN if no non-zero differences.
    """
    diffs = np.asarray(diffs, dtype=np.float64).ravel()
    nz = diffs[diffs != 0.0]
    if nz.size == 0:
        return float("nan")
    ranks = scipy.stats.rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum()) if np.any(nz > 0) else 0.0
    w_minus = float(ranks[nz < 0].sum()) if np.any(nz < 0) else 0.0
    total = w_plus + w_minus
    if total == 0:
        return float("nan")
    return float((w_plus - w_minus) / total)


# ---------------------------------------------------------------------------
# Runner: one session
# ---------------------------------------------------------------------------
def run_session(session_dir: str, excluded_sessions: list, deviation_log: list) -> dict | None:
    """Run the full NP-DEMO-4 analysis pipeline for one session.

    Returns a session result dict, or None if session is excluded.
    Appends to excluded_sessions and deviation_log as needed.
    """
    exp_date = _parse_session_date(session_dir)
    print(f"\n=== Session {exp_date} ===")

    # Load data
    files, missing = or_validate_files(session_dir, strict=False)
    files, missing = _fix_ks_dir(files, missing)
    if missing:
        msg = f"Required files missing in {session_dir!r}: {missing}"
        deviation_log.append(dict(session=exp_date, reason=msg))
        print(f"  WARNING: {msg}")
        excluded_sessions.append(dict(session=exp_date, reason=msg))
        return None

    D = load_experiment_data(files)
    LV_Fs = float(D["LV_Fs"])
    n_lv = len(D["SNF"])
    session_duration_s = float(n_lv / LV_Fs)

    # 1. ETH contact detection (verbatim detect_eth_contact)
    D = detect_eth_contact(D, eth_threshold=0.05)
    n_trials_eth = int(D["n_trials"])
    print(f"  n_trials_eth (global ETH contacts): {n_trials_eth}")

    # Exclude session if no ETH contacts
    if n_trials_eth == 0:
        reason = "detect_eth_contact found zero ETH contact events"
        excluded_sessions.append(dict(session=exp_date, reason=reason))
        print(f"  EXCLUDED: {reason}")
        return None

    # 2. Sniff rate computation (PROH-001: SNF only)
    D = compute_sniff_rate(D, threshold_std=-0.5)
    sniff_rate = np.asarray(D["sniff_rate"], dtype=np.float64)
    sniff_onsets = np.asarray(D.get("sniff_onsets_sr", []))
    print(f"  Sniff onsets detected: {sniff_onsets.size}")

    # 3. Unit inclusion (CON-001: FR >= 0.1 Hz AND total spikes >= 5000)
    included_unit_ids, n_units_recorded, n_units_included = _apply_unit_inclusion(
        D, session_duration_s
    )
    print(f"  Units: {n_units_recorded} recorded, {n_units_included} included")

    # 4. Population firing rate (50ms sliding window)
    D = compute_firing_rate_50ms(D, included_unit_ids, window_ms=50.0)
    fr_pop = np.asarray(D["firing_rate_population"], dtype=np.float64)

    # 5. Trial segmentation (odd TR only, TS 0-40 s)
    trials = _segment_odd_tr_trials(D)
    print(f"  LabView trials (odd TR, 0-40s): {len(trials)}")

    # 6. Build per-trial matrices
    eth_contact_mask = np.asarray(D["eth_contact_mask"], dtype=bool).ravel()
    raw_ETH = np.asarray(D["ETH"], dtype=np.float64).ravel()  # BEFORE mean-subtraction

    sniff_rate_per_trial = []   # list of (n_samples_in_trial,) arrays
    fr_per_trial = []
    eth_per_trial = []

    # For methods table
    avg_sniffs_list = []       # sniff onsets per trial
    eth_contacts_per_trial_counts = []  # count of ETH contacts per trial

    # For CON-003
    contact_sniff_means = []   # mean sniff rate during first ETH contact per trial
    control_sniff_means = []   # mean sniff rate during TS=0-10s per trial

    for trial in trials:
        idx = trial["global_idx"]
        ts_s = trial["ts_s"]

        # Signal extracts
        sniff_rate_per_trial.append(sniff_rate[idx].tolist())
        fr_per_trial.append(fr_pop[idx].tolist())
        eth_per_trial.append(raw_ETH[idx].tolist())

        # Count sniff onsets in this trial
        n_onsets_in_trial = int(np.sum(
            (sniff_onsets >= idx[0]) & (sniff_onsets <= idx[-1])
        )) if sniff_onsets.size > 0 else 0
        avg_sniffs_list.append(n_onsets_in_trial)

        # Count ETH contacts in this trial
        eth_local = eth_contact_mask[idx]
        padded = np.concatenate(([False], eth_local, [False]))
        rising = np.count_nonzero(~padded[:-1] & padded[1:])
        eth_contacts_per_trial_counts.append(rising)

        # CON-003: find first ETH contact in this trial
        contact_global_idx, contact_local = _first_eth_contact_in_trial(
            eth_contact_mask, idx
        )
        if contact_global_idx is None or len(contact_global_idx) == 0:
            # No contact in this trial: skip for CON-003
            continue

        # Mean sniff rate during first contact period
        mean_sniff_contact = float(np.mean(sniff_rate[contact_global_idx]))

        # Control period: TS=0-10s, EXCLUDING any contact overlap
        # TS is in seconds; control is ts_s in [0, 10]
        control_local_mask = (ts_s >= 0.0) & (ts_s <= 10.0)
        # Exclude samples that overlap with any ETH contact
        control_local_mask = control_local_mask & (~eth_contact_mask[idx])
        if not control_local_mask.any():
            # No non-contact control samples
            continue
        control_global_idx = idx[control_local_mask]
        mean_sniff_control = float(np.mean(sniff_rate[control_global_idx]))

        contact_sniff_means.append(mean_sniff_contact)
        control_sniff_means.append(mean_sniff_control)

    # Methods table averages for this session
    avg_sniffs_per_trial = float(np.mean(avg_sniffs_list)) if avg_sniffs_list else float("nan")
    avg_eth_contacts_per_trial = float(np.mean(eth_contacts_per_trial_counts)) if eth_contacts_per_trial_counts else float("nan")

    # Pool all contact/control pairs per session for CON-003 (per-session means)
    if len(contact_sniff_means) > 0:
        session_mean_contact = float(np.mean(contact_sniff_means))
        session_mean_control = float(np.mean(control_sniff_means))
    else:
        session_mean_contact = float("nan")
        session_mean_control = float("nan")

    return dict(
        session=exp_date,
        n_trials_eth=n_trials_eth,
        n_units_recorded=n_units_recorded,
        n_units_included=n_units_included,
        included_unit_ids=included_unit_ids.tolist(),
        avg_sniffs_per_labview_trial=avg_sniffs_per_trial,
        avg_eth_contacts_per_labview_trial=avg_eth_contacts_per_trial,
        n_labview_trials=len(trials),
        # Per-trial data for figures (store as lists of lists)
        sniff_rate_per_trial=sniff_rate_per_trial,
        fr_per_trial=fr_per_trial,
        eth_per_trial=eth_per_trial,
        # CON-003 per-session pair
        session_mean_contact_sniff_rate=session_mean_contact,
        session_mean_control_sniff_rate=session_mean_control,
        n_con003_trials=len(contact_sniff_means),
    )


# ---------------------------------------------------------------------------
# Multi-session runner
# ---------------------------------------------------------------------------
def run_multi_session_analysis(sessions: list[str]) -> dict:
    """Run NP-DEMO-4 analysis across all sessions and aggregate results.

    Parameters
    ----------
    sessions : list[str]
        Absolute paths to each session folder.

    Returns
    -------
    dict
        All contract-required result objects (keyed by RESULT-* id).
    """
    excluded_sessions = []
    deviation_log = []
    session_results = []

    for sess_dir in sessions:
        res = run_session(sess_dir, excluded_sessions, deviation_log)
        if res is not None:
            session_results.append(res)

    n_valid = len(session_results)
    print(f"\nValid sessions: {n_valid}")
    if n_valid < 3:
        raise RuntimeError(
            f"Fewer than 3 sessions ({n_valid}) remain for CON-003 Wilcoxon test "
            f"after excluding sessions with no ETH contacts. Execution blocked."
        )

    # --- RESULT-eth-mask ---
    result_eth_mask = {}
    for sr in session_results:
        result_eth_mask[sr["session"]] = dict(
            n_trials=sr["n_trials_eth"],
        )
    for exc in excluded_sessions:
        result_eth_mask[exc["session"]] = dict(n_trials=0, excluded=True, reason=exc["reason"])

    # --- RESULT-unit-inclusion ---
    result_unit_inclusion = [
        dict(
            session=sr["session"],
            n_units_recorded=sr["n_units_recorded"],
            n_units_included=sr["n_units_included"],
            included_unit_ids=sr["included_unit_ids"],
        )
        for sr in session_results
    ]

    # --- RESULT-sniff-rate-matrix ---
    # Per-session: list of per-trial sniff rate vectors
    result_sniff_rate_matrix = {
        sr["session"]: sr["sniff_rate_per_trial"] for sr in session_results
    }

    # --- RESULT-fr-per-trial ---
    result_fr_per_trial = {
        sr["session"]: sr["fr_per_trial"] for sr in session_results
    }

    # --- RESULT-eth-per-trial ---
    result_eth_per_trial = {
        sr["session"]: sr["eth_per_trial"] for sr in session_results
    }

    # --- RESULT-methods-table ---
    result_methods_table = [
        dict(
            session=sr["session"],
            n_trials_eth=sr["n_trials_eth"],
            avg_sniffs_per_labview_trial=sr["avg_sniffs_per_labview_trial"],
            avg_eth_contacts_per_labview_trial=sr["avg_eth_contacts_per_labview_trial"],
            n_units_recorded=sr["n_units_recorded"],
            n_units_included=sr["n_units_included"],
            n_labview_trials=sr["n_labview_trials"],
        )
        for sr in session_results
    ]

    # --- RESULT-sniff-stat (CON-003) ---
    contact_vals = np.array([sr["session_mean_contact_sniff_rate"] for sr in session_results])
    control_vals = np.array([sr["session_mean_control_sniff_rate"] for sr in session_results])

    # Filter out sessions where CON-003 could not be computed (NaN)
    valid_pairs_mask = np.isfinite(contact_vals) & np.isfinite(control_vals)
    n_pairs = int(valid_pairs_mask.sum())
    contact_valid = contact_vals[valid_pairs_mask]
    control_valid = control_vals[valid_pairs_mask]

    diffs = contact_valid - control_valid
    if n_pairs < 2:
        raise RuntimeError(
            f"Fewer than 2 valid CON-003 pairs ({n_pairs}). Cannot run Wilcoxon test."
        )

    wil_result = scipy.stats.wilcoxon(diffs, alternative="two-sided", method="exact",
                                       zero_method="wilcox")
    W = float(wil_result.statistic)
    p_exact = float(wil_result.pvalue)
    r = rank_biserial_r(diffs)

    result_sniff_stat = dict(
        W=W,
        p_exact=p_exact,
        rank_biserial_r=r,
        n_animals=n_pairs,
        direction="contact>control" if r > 0 else ("control>contact" if r < 0 else "ns"),
        significant=(p_exact < 0.05),
        contact_means=contact_valid.tolist(),
        control_means=control_valid.tolist(),
    )

    return {
        "RESULT-eth-mask": result_eth_mask,
        "RESULT-unit-inclusion": result_unit_inclusion,
        "RESULT-sniff-rate-matrix": result_sniff_rate_matrix,
        "RESULT-fr-per-trial": result_fr_per_trial,
        "RESULT-eth-per-trial": result_eth_per_trial,
        "RESULT-methods-table": result_methods_table,
        "RESULT-sniff-stat": result_sniff_stat,
        "excluded_sessions": excluded_sessions,
        "deviation_log": deviation_log,
    }

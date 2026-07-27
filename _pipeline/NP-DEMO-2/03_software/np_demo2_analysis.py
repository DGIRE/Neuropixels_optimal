"""
np_demo2_analysis.py -- NP-DEMO-2 analysis module (Blueprint v2 P3, D7)
========================================================================
Implements the approved contract (contract_spec.yaml, NP-DEMO-2, contract_version 1):

    Does ethanol exposure alter the phase-locking of olfactory-bulb single-unit
    spikes to the sniff (SNF sensor) signal phase, relative to control epochs,
    evaluated BOTH at the animal (experiment) level and at the individual-unit
    level?

This module is an ADDITIVE caller of the validated kernel under
"Optimized Python" (load_experiment_data, compute_sniff_phase, threshold_eth,
compute_spike_phase, compute_sniff_psth, plot_unit_locations,
analyses._common.matlab_round). It does not modify, copy, or reimplement any
kernel numerics -- the sniff-phase / spike-phase / PSTH computation itself is
delegated entirely to the kernel functions. The new numerics here are exactly
the small, explicitly-specified building blocks pinned by the approved
contract (circular MRL/preferred-phase, condition assignment, unit inclusion,
PROH-002 two-pass ethanol-threshold adjustment, paired animal-level test,
unit-level linear mixed-effects model, |delta_MRL| ranking, pooled-MRL
extremes selection, QC counts, discard-run logging) plus a thin
session/multi-session runner that wires those pieces together.

Every "New-Code-Required" function below is a byte-for-byte match of the
pinned reference implementation validated by the oracle-designer's tests
under 03_software/oracles/ (oracle_*.py) -- those files ARE the approved
contract's operational definition of each function; this module supplies the
real (non-duplicated) implementation the oracles import.

Prohibited (contract_spec.yaml prohibited_changes) -- NOT done here:
    * using the LFP (instead of the SNF signal) for sniff-phase computation
      -- phase always comes from compute_sniff_phase(D, ...)['SNF_PH'] /
      compute_spike_phase(...)['spike_SNF_PH'], both keyed off D['SNF']
      (PROH-001)
    * including SNF sections / spikes with spike_SNF_PH == -1 (outside valid
      sniff cycles) -- valid_sniff_mask excludes the kernel's -1 sentinel
      before every circular-mean computation; every discarded run is logged
      (log_discard_runs -> RESULT-qc-discards)
    * editing Optimized Python kernel files, Golden Fixtures, or tolerance
      config
    * changing the sniff-detection threshold (threshold_std in
      compute_sniff_phase) -- always called with its pinned default
      threshold_std=-0.5
    * adjusting any single experiment's ethanol threshold before completing
      the first-pass-all-experiments-at-default step required by PROH-002
      -- proh_002_adjustment freezes the cross-session mean/SD from Pass 1
      BEFORE evaluating any Pass-2 grid search (HAZ-04)
"""
from __future__ import annotations

import copy
import math
import os
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd
import scipy.stats

# --------------------------------------------------------------------------
# Kernel wiring -- sibling import, never a copy. Insert the absolute kernel
# path so `from load_experiment_data import ...` etc. resolve regardless of
# the caller's cwd.
# --------------------------------------------------------------------------
_KERNEL_DIR = r"C:\Projects\Repos\Neuropixels\Optimized Python"
if _KERNEL_DIR not in sys.path and os.path.isdir(_KERNEL_DIR):
    sys.path.insert(0, _KERNEL_DIR)

from load_experiment_data import load_experiment_data              # noqa: E402
from lib.or_validate_files import or_validate_files                  # noqa: E402
from analyses import (                                                # noqa: E402
    compute_sniff_phase, threshold_eth, compute_spike_phase, compute_sniff_psth,
)
from analyses._common import matlab_round                            # noqa: E402


# ==========================================================================
# Pinned constants (contract_spec.yaml, PROH-002)
# ==========================================================================
DEFAULT_ETH_THRESHOLD = 0.11
SNIFF_THRESHOLD_STD = -0.5
GRID = np.round(np.arange(0.05, 0.50 + 1e-9, 0.005), 3)  # 91 points, 0.05..0.50 step 0.005


# ==========================================================================
# 1. Circular statistics -- mean resultant length + preferred phase
#    (identical to the NP-DEMO-1 approved primitive; re-pinned here per
#    oracle_circular_stats.py)
# ==========================================================================
def mrl_and_preferred_phase(phase_0to1: np.ndarray) -> tuple[float, float]:
    """phase_0to1: 1-D array of already-filtered valid sniff phases in [0,1).

    Returns (mrl, preferred_phase_rad):
        mrl = |mean(exp(1j * 2*pi * phase))|  in [0, 1]
        preferred = angle(mean(exp(1j * 2*pi * phase)))  in (-pi, pi]
    Empty input -> (nan, nan).
    """
    phase = np.asarray(phase_0to1, dtype=np.float64).ravel()
    if phase.size == 0:
        return float("nan"), float("nan")
    z = np.mean(np.exp(1j * 2.0 * np.pi * phase))
    return float(np.abs(z)), float(np.angle(z))


# ==========================================================================
# 2. Valid-sniff filter (kernel sentinel exclusion) -- HAZ-01
# ==========================================================================
def valid_sniff_mask(spike_SNF_PH: np.ndarray) -> np.ndarray:
    """True where spike_SNF_PH >= 0 (kernel sentinel -1 = outside sniff cycle)."""
    return np.asarray(spike_SNF_PH, dtype=np.float64) >= 0.0


# ==========================================================================
# 3. Ethanol / control condition assignment -- HAZ-02 (matlab_round)
# ==========================================================================
def classify_ethanol(spike_times_s: np.ndarray, ETH_thr: np.ndarray, LV_Fs: float,
                      eth_threshold: float = DEFAULT_ETH_THRESHOLD) -> np.ndarray:
    """True = ethanol condition.

    lv_idx = matlab_round(spike_times_s * LV_Fs) clamped to [0, len(ETH_thr)-1]
    ethanol iff ETH_thr[lv_idx] > eth_threshold
    """
    spike_times_s = np.asarray(spike_times_s, dtype=np.float64).ravel()
    ETH_thr = np.asarray(ETH_thr, dtype=np.float64).ravel()
    lv_idx = matlab_round(spike_times_s * LV_Fs).astype(np.int64)
    np.clip(lv_idx, 0, ETH_thr.size - 1, out=lv_idx)
    return ETH_thr[lv_idx] > eth_threshold


# ==========================================================================
# 4. Per-unit inclusion rule
# ==========================================================================
def unit_included(n_spikes_total, session_duration_s, n_valid_sniff_spikes,
                   min_rate_hz: float = 0.1, min_spikes: int = 50) -> bool:
    """False if (n_spikes_total / session_duration_s) < min_rate_hz OR
    n_valid_sniff_spikes < min_spikes. Both boundaries inclusive at '>='."""
    firing_rate = n_spikes_total / session_duration_s
    return bool(firing_rate >= min_rate_hz and n_valid_sniff_spikes >= min_spikes)


# ==========================================================================
# 5. Paired animal-level test (exact Wilcoxon signed-rank + rank-biserial r)
#    -- reuses the approved NP-DEMO-1 design unchanged (contract
#    statistical_model: "reuses the NP-DEMO-1 approved approach unchanged")
# ==========================================================================
def paired_animal_test(ethanol_means: np.ndarray, control_means: np.ndarray) -> dict:
    """scipy.stats.wilcoxon exact two-sided paired test on per-animal mean MRL.

    Returns dict with pvalue, effect_size (rank-biserial r, positive == ethanol
    > control), statistic, n_animals, direction.
    """
    ethanol_means = np.asarray(ethanol_means, dtype=np.float64).ravel()
    control_means = np.asarray(control_means, dtype=np.float64).ravel()
    n_animals = ethanol_means.size
    diffs = ethanol_means - control_means

    res = scipy.stats.wilcoxon(diffs, alternative="two-sided", method="exact",
                                zero_method="wilcox")

    nz = diffs[diffs != 0]
    if nz.size == 0:
        r = float("nan")
    else:
        ranks = scipy.stats.rankdata(np.abs(nz))
        w_plus = ranks[nz > 0].sum()
        w_minus = ranks[nz < 0].sum()
        total = w_plus + w_minus
        r = float((w_plus - w_minus) / total) if total > 0 else float("nan")

    pvalue = float(res.pvalue)
    if not np.isfinite(r) or pvalue >= 0.05:
        direction = "ns"
    elif r > 0:
        direction = "ethanol>control"
    else:
        direction = "control>ethanol"

    return dict(
        pvalue=pvalue,
        effect_size=r,
        statistic=float(res.statistic),
        n_animals=int(n_animals),
        direction=direction,
    )


# ==========================================================================
# 6. QC counts
# ==========================================================================
def qc_counts(sniff_onsets_s: np.ndarray, session_duration_s: float, LV_Fs: float,
              n_neurons_valid: int, eth_contact_count: int, SNF_PH: np.ndarray) -> dict:
    """n_sniffs=len(sniff_onsets_s); n_neurons/n_trials pass-throughs;
    length_min=session_duration_s/60; pct_usable_sniffs=(sum(SNF_PH>=0)/len)*100."""
    sniff_onsets_s = np.asarray(sniff_onsets_s).ravel()
    SNF_PH = np.asarray(SNF_PH, dtype=np.float64).ravel()
    n_sniffs = int(sniff_onsets_s.size)
    length_min = float(session_duration_s) / 60.0
    if SNF_PH.size > 0:
        pct_usable_sniffs = float((np.sum(SNF_PH >= 0.0) / SNF_PH.size) * 100.0)
    else:
        pct_usable_sniffs = 0.0
    return dict(
        n_sniffs=n_sniffs,
        n_neurons=int(n_neurons_valid),
        n_trials=int(eth_contact_count),
        length_min=length_min,
        pct_usable_sniffs=pct_usable_sniffs,
    )


# ==========================================================================
# 7. Discarded-SNF-section log -- HAZ-01, HAZ-10
# ==========================================================================
def log_discard_runs(SNF_PH: np.ndarray, LV_Fs: float, session_date: str) -> list[dict]:
    """Contiguous runs of SNF_PH < 0 (kernel -1 sentinel), returned as
    {experiment, start_s, end_s, reason}, in ascending time order."""
    SNF_PH = np.asarray(SNF_PH, dtype=np.float64).ravel()
    invalid = SNF_PH < 0.0
    if invalid.size == 0 or not invalid.any():
        return []

    d = np.diff(invalid.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if invalid[0]:
        starts = [0] + starts
    if invalid[-1]:
        ends = ends + [invalid.size]

    runs = []
    for s, e in zip(starts, ends):
        runs.append(dict(
            experiment=session_date,
            start_s=float(s) / LV_Fs,
            end_s=float(e) / LV_Fs,
            reason="outside_valid_sniff",
        ))
    return runs


# ==========================================================================
# 8. PROH-002 two-pass ethanol-threshold adjustment -- HAZ-04, HAZ-06
# ==========================================================================
def contact_count(eth_signal: np.ndarray, eth_threshold: float) -> int:
    """Number of contiguous runs where eth_signal > eth_threshold (a discrete
    ethanol-contact-event count). Trivial run-length counter -- this IS the
    ground-truth "contact" definition (contract inclusion_exclusion clause)."""
    above = np.asarray(eth_signal, dtype=np.float64) > eth_threshold
    if above.size == 0 or not above.any():
        return 0
    d = np.diff(above.astype(np.int8))
    n = int(np.sum(d == 1))
    if above[0]:
        n += 1
    return n


def _select_best_threshold(grid: np.ndarray, counts_at_grid: np.ndarray,
                            mean: float, default: float = DEFAULT_ETH_THRESHOLD):
    """argmin |count-mean| over the grid; ties broken by nearest to `default`.
    Returns (best_threshold, best_count)."""
    grid = np.asarray(grid, dtype=np.float64)
    counts_at_grid = np.asarray(counts_at_grid, dtype=np.float64)
    diffs = np.abs(counts_at_grid - mean)
    best_diff = diffs.min()
    candidates = np.flatnonzero(diffs == best_diff)
    dist_to_default = np.abs(grid[candidates] - default)
    winner = candidates[np.argmin(dist_to_default)]
    return float(grid[winner]), float(counts_at_grid[winner])


def proh_002_adjustment(session_signals: dict, LV_Fs: float,
                         default_threshold: float = DEFAULT_ETH_THRESHOLD,
                         grid: np.ndarray = GRID) -> dict:
    """Two-pass PROH-002 adjustment. session_signals: {session_id: eth_signal
    array} (the RAW D['ETH'] trace -- threshold_eth's floor clip means
    ETH_thr[i] > g  <=>  ETH[i] > g for every g, so contact_count can be
    evaluated directly on the raw signal without re-running threshold_eth per
    grid point). Returns a dict with cross_session_mean, cross_session_sd,
    pass1_count, flagged (set), final_threshold, final_count (all keyed by
    session_id)."""
    session_ids = list(session_signals.keys())

    # PASS 1: every session at the SAME default threshold, BEFORE any
    # session-specific decision is made (HAZ-04).
    pass1_count = {sid: contact_count(session_signals[sid], default_threshold)
                   for sid in session_ids}
    counts_arr = np.array([pass1_count[sid] for sid in session_ids], dtype=np.float64)
    mean = float(counts_arr.mean())
    sd = float(counts_arr.std(ddof=1)) if counts_arr.size > 1 else 0.0

    flagged = {sid for sid in session_ids if abs(pass1_count[sid] - mean) > sd}

    # PASS 2: grid search ONLY for flagged sessions.
    final_threshold, final_count = {}, {}
    for sid in session_ids:
        if sid in flagged:
            counts_at_grid = np.array(
                [contact_count(session_signals[sid], g) for g in grid], dtype=np.float64
            )
            g_best, c_best = _select_best_threshold(grid, counts_at_grid, mean, default_threshold)
            final_threshold[sid] = g_best
            final_count[sid] = c_best
        else:
            final_threshold[sid] = default_threshold
            final_count[sid] = pass1_count[sid]

    return dict(
        cross_session_mean=mean,
        cross_session_sd=sd,
        pass1_count=pass1_count,
        flagged=flagged,
        final_threshold=final_threshold,
        final_count=final_count,
    )


def all_ethanol_at_all_thresholds(eth_signal: np.ndarray, grid: np.ndarray = GRID) -> bool:
    """True iff eth_signal never dips at-or-below ANY grid threshold, i.e. no
    grid value ever produces a control (non-contact) sample -- the degenerate
    all-ethanol / no-control-epochs case (HAZ-06, e.g. 2021-11-03) that
    PROH-002's Pass-2 grid search cannot rescue because the grid itself
    (0.05-0.50) never reaches high enough to uncover a control epoch."""
    eth_signal = np.asarray(eth_signal, dtype=np.float64).ravel()
    if eth_signal.size == 0:
        return True
    return bool(np.min(eth_signal) > np.max(grid))


# ==========================================================================
# 9. Unit-level linear mixed-effects model -- HAZ-08 (pseudoreplication)
# ==========================================================================
def unit_level_lmm(unit_df: pd.DataFrame) -> dict:
    """statsmodels MixedLM: 'mrl ~ condition' fixed effect, animal_id random
    intercept. Returns dict with coef (condition fixed effect, ethanol vs
    control), pvalue, standardized_effect_size (coef / SD(outcome), ddof=1),
    group_var (estimated animal-level random-intercept variance), direction,
    n_units, n_animals."""
    import statsmodels.formula.api as smf

    df = unit_df.copy()
    df["condition"] = pd.Categorical(df["condition"], categories=["control", "ethanol"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # boundary-of-parameter-space REML warnings
        md = smf.mixedlm("mrl ~ condition", df, groups=df["animal_id"])
        mdf = md.fit(reml=True)

    coef = float(mdf.params["condition[T.ethanol]"])
    pvalue = float(mdf.pvalues["condition[T.ethanol]"])
    outcome_sd = float(df["mrl"].std(ddof=1))
    standardized = coef / outcome_sd if outcome_sd > 0 else float("nan")
    group_var = float(mdf.params.get("Group Var", float("nan")))

    if not np.isfinite(pvalue) or pvalue >= 0.05:
        direction = "ns"
    elif coef > 0:
        direction = "ethanol>control"
    else:
        direction = "control>ethanol"

    return dict(
        coef=coef,
        pvalue=pvalue,
        standardized_effect_size=standardized,
        group_var=group_var,
        direction=direction,
        n_units=int(df["unit_id"].nunique()),
        n_animals=int(df["animal_id"].nunique()),
    )


# ==========================================================================
# 10. |delta_MRL| ranking (OD-2) and pooled-MRL QC-PSTH extremes (OD-4)
# ==========================================================================
def rank_units_by_abs_delta_mrl(df: pd.DataFrame) -> pd.DataFrame:
    """df columns: unit_id, mrl_ethanol, mrl_control. Adds delta_mrl,
    abs_delta_mrl; sorts DESCENDING by abs_delta_mrl, ties broken by
    ASCENDING unit_id (deterministic secondary key)."""
    out = df.copy()
    out["delta_mrl"] = out["mrl_ethanol"] - out["mrl_control"]
    out["abs_delta_mrl"] = out["delta_mrl"].abs()
    out = out.sort_values(["abs_delta_mrl", "unit_id"], ascending=[False, True],
                           kind="mergesort").reset_index(drop=True)
    return out


def top_k_by_abs_delta_mrl(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    return rank_units_by_abs_delta_mrl(df).head(k).reset_index(drop=True)


def pooled_mrl(eth_phases: np.ndarray, ctrl_phases: np.ndarray) -> float:
    """Overall (condition-agnostic) MRL: MRL computed on ALL valid-sniff spike
    phases POOLED across ethanol+control, per contract OD-4 -- NOT the
    average of the two per-condition MRLs."""
    pooled = np.concatenate([
        np.asarray(eth_phases, dtype=np.float64).ravel(),
        np.asarray(ctrl_phases, dtype=np.float64).ravel(),
    ])
    mrl, _ = mrl_and_preferred_phase(pooled)
    return mrl


def select_psth_extremes(units: dict) -> tuple[str, str]:
    """units: {unit_id: (eth_phases, ctrl_phases)}. Returns
    (strongest_unit_id, weakest_unit_id) by pooled MRL across ALL units
    (condition-agnostic, per contract OD-4), ties broken by ascending
    unit_id."""
    rows = []
    for uid, (eth_phases, ctrl_phases) in units.items():
        rows.append((uid, pooled_mrl(eth_phases, ctrl_phases)))
    df = pd.DataFrame(rows, columns=["unit_id", "pooled_mrl"])
    df = df.sort_values(["pooled_mrl", "unit_id"], ascending=[False, True],
                         kind="mergesort").reset_index(drop=True)
    strongest = df.iloc[0]["unit_id"]
    weakest = df.iloc[-1]["unit_id"]
    return strongest, weakest


# ==========================================================================
# Runner-only helpers (no contract-pinned numerics live here)
# ==========================================================================
_KS_PROBE_FILES = ("channel_map.npy", "channel_positions.npy")


def _fix_ks_dir(files: dict, missing: list[str]) -> tuple[dict, list[str]]:
    """If the auto-picked ksDir is a phy sub-directory (e.g. phy2.5PShank1)
    lacking channel_map.npy / channel_positions.npy, promote to its parent
    once (HAZ-05, 2022-09-14 multi-shank session)."""
    if files["ksDir"] and any(
        not os.path.isfile(os.path.join(files["ksDir"], f)) for f in _KS_PROBE_FILES
    ):
        parent = os.path.dirname(files["ksDir"])
        if all(os.path.isfile(os.path.join(parent, f)) for f in _KS_PROBE_FILES):
            files["ksDir"] = parent
            missing = [m for m in missing if "channel_map" not in m and "channel_positions" not in m]
    return files, missing


def load_session_raw(session_dir: str) -> dict:
    """load_experiment_data + compute_sniff_phase (pinned threshold_std=-0.5).
    Does NOT call threshold_eth / compute_spike_phase -- the eth_threshold is
    not known yet (PROH-002 resolves it from D['ETH'] across all sessions
    first)."""
    files, missing = or_validate_files(session_dir, strict=False)
    files, missing = _fix_ks_dir(files, missing)
    if missing:
        raise FileNotFoundError(
            f"Required files missing in {session_dir!r}:\n  " + "\n  ".join(missing)
        )
    D = load_experiment_data(files)
    D = compute_sniff_phase(D, threshold_std=SNIFF_THRESHOLD_STD)
    return D


# Shank assignment, replicated verbatim from plot_unit_locations.py's own
# peak-channel-lookup formula (a deterministic index into the already-
# validated D-dict fields unitIDs/xcoords/sp.temps; not a new statistic).
_SHANK_BOUNDARIES = [-np.inf, -100.0, 100.0, 400.0, np.inf]


def unit_probe_location(D: dict, unit_id: int) -> dict:
    """{unit_depth_um, unit_xcoord_um, shank, unit_firing_rate} for one unit,
    taken directly from the D-dict (unitDepths/xcoords/unitFiringRate),
    per RESULT-unit-locations' level-1 exact-coordinate acceptance criterion."""
    unit_ids = np.asarray(D["unitIDs"]).ravel()
    u_idx = int(np.flatnonzero(unit_ids == unit_id)[0])

    temps = np.asarray(D["sp"]["temps"])
    n_templates = temps.shape[0]
    temps_max_abs = np.squeeze(np.max(np.abs(temps), axis=1))
    peak_chan = np.argmax(temps_max_abs, axis=1) if temps_max_abs.ndim > 1 else np.argmax(temps_max_abs)
    xcoords = np.asarray(D["xcoords"]).ravel()

    uid_val = int(unit_ids[u_idx])
    if uid_val < n_templates:
        pc = int(peak_chan[uid_val]) if np.ndim(peak_chan) > 0 else int(peak_chan)
    else:
        pc = 0
    pc = int(np.clip(pc, 0, xcoords.size - 1))
    unit_xcoord = float(xcoords[pc])

    shank = 0
    for s in range(4):
        if _SHANK_BOUNDARIES[s] <= unit_xcoord < _SHANK_BOUNDARIES[s + 1]:
            shank = s
            break

    return dict(
        unit_depth_um=float(np.asarray(D["unitDepths"]).ravel()[u_idx]),
        unit_xcoord_um=unit_xcoord,
        shank=int(shank),
        unit_firing_rate=float(np.asarray(D["unitFiringRate"]).ravel()[u_idx]),
    )


def _single_unit_psth(D: dict, unit_id: int, spike_mask: np.ndarray):
    """Build a filtered D-dict copy restricted to one unit's spikes matching
    spike_mask (a boolean over the FULL spikeTimes array), then delegate the
    actual PSTH computation entirely to the validated kernel compute_sniff_psth
    (sniff_onsets_s / sniff_dur_s are unchanged -- PSTH events are sniff
    onsets, not spike-conditioned). Returns (psth_row, centers_phase, n_events)."""
    spikeTimes = np.asarray(D["spikeTimes"], dtype=np.float64).ravel()
    clu = np.asarray(D["sp"]["clu"]).ravel()
    sel = (clu == unit_id) & spike_mask

    D_unit = dict(D)
    D_unit["spikeTimes"] = spikeTimes[sel]
    D_unit["sp"] = dict(D["sp"])
    D_unit["sp"]["clu"] = clu[sel]
    D_unit["unitIDs"] = np.array([unit_id])

    psth, centers, n_events = compute_sniff_psth(D_unit, attach=False)
    return psth[0], centers, n_events


# ==========================================================================
# Per-session analysis (Part 2)
# ==========================================================================
def analyze_session(D: dict, session_date: str, final_eth_threshold: float) -> dict:
    """D must already carry compute_sniff_phase's outputs (SNF_PH,
    sniff_onsets_s, sniff_dur_s, ...). Applies threshold_eth (final,
    possibly PROH-002-adjusted) + compute_spike_phase, then computes
    per-unit MRL/phase by condition, QC counts, and the discard log."""
    D2 = threshold_eth(D, eth_threshold=final_eth_threshold)
    D2 = compute_spike_phase(D2)

    LV_Fs = float(D2["LV_Fs"])
    spikeTimes = np.asarray(D2["spikeTimes"], dtype=np.float64).ravel()
    spike_SNF_PH = np.asarray(D2["spike_SNF_PH"], dtype=np.float64).ravel()
    clu = np.asarray(D2["sp"]["clu"]).ravel()
    unitIDs = np.asarray(D2["unitIDs"]).ravel()
    ETH_thr = np.asarray(D2["ETH_thr"], dtype=np.float64).ravel()
    SNF_PH = np.asarray(D2["SNF_PH"], dtype=np.float64).ravel()

    valid_mask = valid_sniff_mask(spike_SNF_PH)
    lv_idx = matlab_round(spikeTimes * LV_Fs).astype(np.int64)
    np.clip(lv_idx, 0, len(ETH_thr) - 1, out=lv_idx)
    is_ethanol = ETH_thr[lv_idx] > final_eth_threshold
    eth_mask = is_ethanol & valid_mask
    ctrl_mask = (~is_ethanol) & valid_mask

    session_dur_s = len(SNF_PH) / LV_Fs
    eth_contact_count = contact_count(np.asarray(D2["ETH"], dtype=np.float64).ravel(),
                                       final_eth_threshold)

    per_unit_results = []
    for uid in unitIDs:
        unit_mask = clu == uid
        n_total_spikes = int(np.sum(unit_mask))

        eth_ph = spike_SNF_PH[unit_mask & eth_mask]
        ctrl_ph = spike_SNF_PH[unit_mask & ctrl_mask]
        n_valid_combined = eth_ph.size + ctrl_ph.size

        fr = n_total_spikes / session_dur_s
        if fr < 0.1 or n_valid_combined < 50:
            continue

        mrl_eth, phase_eth = mrl_and_preferred_phase(eth_ph)
        mrl_ctrl, phase_ctrl = mrl_and_preferred_phase(ctrl_ph)

        per_unit_results.append(dict(
            unit_id=int(uid), session=session_date,
            mrl_ethanol=mrl_eth, mrl_control=mrl_ctrl,
            phase_ethanol=phase_eth, phase_control=phase_ctrl,
            n_eth_spikes=int(eth_ph.size), n_ctrl_spikes=int(ctrl_ph.size),
            delta_mrl=mrl_eth - mrl_ctrl, abs_delta_mrl=abs(mrl_eth - mrl_ctrl),
            eth_phases=eth_ph, ctrl_phases=ctrl_ph,
        ))

    n_neurons_valid = len(per_unit_results)
    counts = qc_counts(D2["sniff_onsets_s"], session_dur_s, LV_Fs,
                        n_neurons_valid, eth_contact_count, SNF_PH)
    discards = log_discard_runs(SNF_PH, LV_Fs, session_date)

    return dict(
        session_date=session_date,
        D=D2,
        per_unit_results=per_unit_results,
        qc_counts=counts,
        discards=discards,
        eth_mask=eth_mask,
        ctrl_mask=ctrl_mask,
        valid_mask=valid_mask,
        session_duration_s=session_dur_s,
    )


# ==========================================================================
# Full multi-session orchestration
# ==========================================================================
def run_full_analysis(session_dirs: dict) -> dict:
    """session_dirs: {session_date: session_dir_path}. Runs the complete
    NP-DEMO-2 pipeline (PROH-002 -> per-session analysis -> PSTH extremes /
    examples -> animal + unit statistics) and returns a dict keyed by the
    contract's required_outputs ids."""
    session_dates = list(session_dirs.keys())

    # ---- Load raw D (through compute_sniff_phase) for every session ------
    print("Loading sessions (load_experiment_data + compute_sniff_phase)...")
    raw_D = {}
    load_failures = {}
    for sd in session_dates:
        print(f"  {sd}: {session_dirs[sd]}")
        try:
            raw_D[sd] = load_session_raw(session_dirs[sd])
        except Exception as exc:  # noqa: BLE001 -- log as a deviation, never crash
            print(f"    FAILED to load: {exc}")
            load_failures[sd] = str(exc)

    loaded_dates = [sd for sd in session_dates if sd in raw_D]

    # ---- PROH-002 Pass 1 + Pass 2 (raw D['ETH'] traces) -------------------
    print("\nPROH-002 two-pass ethanol-threshold adjustment...")
    session_signals = {sd: np.asarray(raw_D[sd]["ETH"], dtype=np.float64).ravel()
                        for sd in loaded_dates}
    nominal_LV_Fs = float(raw_D[loaded_dates[0]]["LV_Fs"]) if loaded_dates else 125.0
    proh2 = proh_002_adjustment(session_signals, LV_Fs=nominal_LV_Fs)

    print(f"  Pass 1: mean_contacts={proh2['cross_session_mean']:.3f}, "
          f"sd_contacts={proh2['cross_session_sd']:.3f}")
    print(f"  Flagged sessions: {sorted(proh2['flagged'])}")

    eth_threshold_log = []
    excluded_sessions = set()
    for sd in loaded_dates:
        was_adjusted = sd in proh2["flagged"]
        all_eth = False
        if was_adjusted:
            all_eth = all_ethanol_at_all_thresholds(session_signals[sd])
        entry = dict(
            session_date=sd,
            default_threshold=DEFAULT_ETH_THRESHOLD,
            final_threshold=float(proh2["final_threshold"][sd]),
            was_adjusted=bool(was_adjusted),
            n_contacts_at_final=int(proh2["final_count"][sd]),
            mean_contacts_pass1=float(proh2["cross_session_mean"]),
            sd_contacts_pass1=float(proh2["cross_session_sd"]),
            all_ethanol_at_all_thresholds=bool(all_eth),
            excluded_by_proh002=bool(all_eth),
        )
        if all_eth:
            excluded_sessions.add(sd)
            print(f"  DEVIATION: {sd} is all-ethanol at every PROH-002 grid "
                  f"threshold -- no control epochs possible; excluded_by_proh002=true.")
        eth_threshold_log.append(entry)

    for sd, reason in load_failures.items():
        eth_threshold_log.append(dict(
            session_date=sd, default_threshold=DEFAULT_ETH_THRESHOLD,
            final_threshold=float("nan"), was_adjusted=False,
            n_contacts_at_final=0, mean_contacts_pass1=float(proh2["cross_session_mean"]),
            sd_contacts_pass1=float(proh2["cross_session_sd"]),
            all_ethanol_at_all_thresholds=False, excluded_by_proh002=True,
            load_failure=reason,
        ))
        excluded_sessions.add(sd)

    included_dates = [sd for sd in loaded_dates if sd not in excluded_sessions]
    print(f"\n  Included sessions (n={len(included_dates)}): {included_dates}")
    if excluded_sessions:
        print(f"  Excluded sessions: {sorted(excluded_sessions)}")

    # ---- Part 2: per-session analysis (included sessions only) -----------
    print("\nPer-session analysis (threshold_eth + compute_spike_phase + MRL)...")
    sessions = {}
    qc_counts_all = []
    discards_all = []
    for sd in included_dates:
        print(f"  {sd}: final_eth_threshold={proh2['final_threshold'][sd]:.3f}")
        sessions[sd] = analyze_session(raw_D[sd], sd, proh2["final_threshold"][sd])
        qc_row = dict(sessions[sd]["qc_counts"])
        qc_row["session_date"] = sd
        qc_row["eth_threshold"] = float(proh2["final_threshold"][sd])
        qc_row["included"] = True
        qc_counts_all.append(qc_row)
        for disc in sessions[sd]["discards"]:
            discards_all.append(disc)

    for sd in excluded_sessions:
        if sd in load_failures:
            continue
        # Still report a QC row (marked excluded) using raw D so nothing is
        # silently dropped from the QC document (contract failure_conditions).
        D_excl = raw_D[sd]
        LV_Fs = float(D_excl["LV_Fs"])
        SNF_PH = np.asarray(D_excl["SNF_PH"], dtype=np.float64).ravel()
        session_dur_s = len(SNF_PH) / LV_Fs
        eth_thr_final = float(proh2["final_threshold"][sd])
        eth_cc = contact_count(np.asarray(D_excl["ETH"], dtype=np.float64).ravel(), eth_thr_final)
        qc_row = qc_counts(D_excl["sniff_onsets_s"], session_dur_s, LV_Fs,
                            0, eth_cc, SNF_PH)
        qc_row["session_date"] = sd
        qc_row["eth_threshold"] = eth_thr_final
        qc_row["included"] = False
        qc_counts_all.append(qc_row)
        discards_all.extend(log_discard_runs(SNF_PH, LV_Fs, sd))

    # ---- Aggregate per-unit rows across ALL included sessions -------------
    all_unit_rows = []
    for sd in included_dates:
        for u in sessions[sd]["per_unit_results"]:
            all_unit_rows.append(u)

    mrl_unit_rows = [
        dict(unit_id=u["unit_id"], session=u["session"],
             mrl_ethanol=u["mrl_ethanol"], mrl_control=u["mrl_control"],
             n_eth_spikes=u["n_eth_spikes"], n_ctrl_spikes=u["n_ctrl_spikes"])
        for u in all_unit_rows
    ]
    phase_unit_rows = [
        dict(unit_id=u["unit_id"], session=u["session"],
             phase_ethanol_rad=u["phase_ethanol"], phase_control_rad=u["phase_control"])
        for u in all_unit_rows
    ]

    # ---- Part 3: |delta_MRL| ranking + top-5 examples ---------------------
    delta_df = pd.DataFrame([
        dict(unit_id=f"{u['session']}_{u['unit_id']}", session=u["session"],
             raw_unit_id=u["unit_id"], mrl_ethanol=u["mrl_ethanol"], mrl_control=u["mrl_control"])
        for u in all_unit_rows
    ])
    if delta_df.empty:
        delta_ranked = delta_df
        top5 = delta_df
    else:
        delta_ranked = rank_units_by_abs_delta_mrl(delta_df)
        top5 = top_k_by_abs_delta_mrl(delta_df, k=5)

    # index all_unit_rows by (session, unit_id) for quick phase-array lookup
    unit_lookup = {(u["session"], u["unit_id"]): u for u in all_unit_rows}

    psth_examples = []
    unit_locations = []
    for _, row in top5.iterrows():
        sd, uid = row["session"], int(row["raw_unit_id"])
        u = unit_lookup[(sd, uid)]
        D2 = sessions[sd]["D"]
        eth_mask = sessions[sd]["eth_mask"]
        ctrl_mask = sessions[sd]["ctrl_mask"]

        psth_eth, centers_phase, n_events_eth = _single_unit_psth(D2, uid, eth_mask)
        psth_ctrl, _, n_events_ctrl = _single_unit_psth(D2, uid, ctrl_mask)

        psth_examples.append(dict(
            unit_id=uid, session=sd,
            abs_delta_mrl=float(u["abs_delta_mrl"]), delta_mrl=float(u["delta_mrl"]),
            psth_eth=psth_eth.tolist(), psth_ctrl=psth_ctrl.tolist(),
            centers_phase=np.asarray(centers_phase).tolist(),
            n_events_eth=int(n_events_eth), n_events_ctrl=int(n_events_ctrl),
        ))
        unit_locations.append(dict(unit_id=uid, session=sd, **unit_probe_location(D2, uid)))

    # ---- Part 3: QC-PSTH extremes (pooled MRL, condition-agnostic) --------
    units_for_extremes = {
        (u["session"], u["unit_id"]): (u["eth_phases"], u["ctrl_phases"])
        for u in all_unit_rows
    }
    psth_extremes = {}
    if units_for_extremes:
        strongest_key, weakest_key = select_psth_extremes(units_for_extremes)
        for label, key in (("strongest_unit", strongest_key), ("weakest_unit", weakest_key)):
            sd, uid = key
            u = unit_lookup[(sd, uid)]
            D2 = sessions[sd]["D"]
            valid_mask = sessions[sd]["valid_mask"]
            psth_all, centers_phase, n_events = _single_unit_psth(D2, uid, valid_mask)
            psth_extremes[label] = dict(
                unit_id=uid, session=sd,
                pooled_mrl=pooled_mrl(u["eth_phases"], u["ctrl_phases"]),
                psth_phase=psth_all.tolist(),
                centers_phase=np.asarray(centers_phase).tolist(),
                n_events=int(n_events),
            )

    # ---- Part 4: animal-level (paired Wilcoxon) ----------------------------
    animal_rows = []
    for sd in included_dates:
        units_this = sessions[sd]["per_unit_results"]
        if not units_this:
            continue
        eth_vals = [u["mrl_ethanol"] for u in units_this if math.isfinite(u["mrl_ethanol"])]
        ctrl_vals = [u["mrl_control"] for u in units_this if math.isfinite(u["mrl_control"])]
        if not eth_vals or not ctrl_vals:
            continue
        animal_rows.append(dict(
            session=sd,
            mean_mrl_ethanol=float(np.mean(eth_vals)),
            mean_mrl_control=float(np.mean(ctrl_vals)),
            n_units_included=len(units_this),
        ))

    if len(animal_rows) >= 1:
        stat_animal = paired_animal_test(
            np.array([a["mean_mrl_ethanol"] for a in animal_rows]),
            np.array([a["mean_mrl_control"] for a in animal_rows]),
        )
    else:
        stat_animal = dict(pvalue=float("nan"), effect_size=float("nan"),
                            statistic=float("nan"), n_animals=0, direction="ns")

    # ---- Part 4: unit-level (linear mixed-effects) -------------------------
    lmm_rows = []
    for u in all_unit_rows:
        if not (math.isfinite(u["mrl_ethanol"]) and math.isfinite(u["mrl_control"])):
            continue
        unit_key = f"{u['session']}_{u['unit_id']}"
        lmm_rows.append(dict(unit_id=unit_key, animal_id=u["session"],
                              condition="ethanol", mrl=u["mrl_ethanol"]))
        lmm_rows.append(dict(unit_id=unit_key, animal_id=u["session"],
                              condition="control", mrl=u["mrl_control"]))
    lmm_df = pd.DataFrame(lmm_rows)

    if len(lmm_df) > 0 and lmm_df["animal_id"].nunique() >= 2:
        stat_unit = unit_level_lmm(lmm_df)
    else:
        stat_unit = dict(coef=float("nan"), pvalue=float("nan"),
                          standardized_effect_size=float("nan"), group_var=float("nan"),
                          direction="ns", n_units=int(lmm_df["unit_id"].nunique()) if len(lmm_df) else 0,
                          n_animals=int(lmm_df["animal_id"].nunique()) if len(lmm_df) else 0)

    return dict(
        eth_threshold_log=eth_threshold_log,
        pass1_stats=dict(mean_contacts=float(proh2["cross_session_mean"]),
                          sd_contacts=float(proh2["cross_session_sd"])),
        qc_counts=qc_counts_all,
        qc_discards=discards_all,
        mrl_unit=mrl_unit_rows,
        phase_unit=phase_unit_rows,
        delta_mrl=delta_ranked.to_dict("records") if not delta_ranked.empty else [],
        mrl_animal=animal_rows,
        stat_animal=stat_animal,
        stat_unit=stat_unit,
        psth_extremes=psth_extremes,
        psth_examples=psth_examples,
        unit_locations=unit_locations,
        included_sessions=included_dates,
        excluded_sessions=sorted(excluded_sessions),
        load_failures=load_failures,
    )

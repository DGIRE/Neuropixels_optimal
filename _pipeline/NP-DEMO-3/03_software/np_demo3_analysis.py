"""np_demo3_analysis.py — NP-DEMO-3 analysis module (Blueprint v2 P3, D7).

Ethanol modulation of olfactory-bulb sniff phase-locking.

This module is an ADDITIVE caller of the validated kernel under
"Optimized Python" (load_experiment_data, compute_sniff_phase,
compute_spike_phase, analyses._common.matlab_round) plus the additive
NP-DEMO-3 primitive detect_eth_contact (03_software/detect_eth_contact.py).
It does NOT modify, copy, or reimplement any kernel numerics, golden fixtures,
or tolerance config.

Contract-pinned invariants honored here (contract_v001.yaml,
prohibited_changes / inclusion_exclusion):
  * Phase ALWAYS derived from D['SNF'] via compute_sniff_phase /
    compute_spike_phase — NEVER from LFP (PROH-001).
  * `threshold_eth` is NEVER imported; ethanol contact uses detect_eth_contact
    (mean-subtract entire ETH, strict threshold 0.05, fixed for all sessions).
  * Only spike_SNF_PH >= 0 (valid sniff cycle) spikes ever contribute to any
    MRL / preferred-phase / spike-count computation. The -1 sentinel is the
    kernel's "outside any sniff cycle" marker; every contiguous invalid SNF_PH
    section is logged in the discard log (no silent discards).
  * compute_sniff_phase pinned to threshold_std=-0.5 (GUI default).
  * detect_eth_contact pinned to eth_threshold=0.05.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.formula.api as smf

# --------------------------------------------------------------------------
# Kernel wiring — sibling import of the validated kernel, never a copy.
# --------------------------------------------------------------------------
_KERNEL_DIR = r"C:\Projects\Repos\Neuropixels\Optimized Python"
if _KERNEL_DIR not in sys.path and os.path.isdir(_KERNEL_DIR):
    sys.path.insert(0, _KERNEL_DIR)

_SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))
if _SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, _SOFTWARE_DIR)

from load_experiment_data import load_experiment_data              # noqa: E402
from lib.or_validate_files import or_validate_files                # noqa: E402
from analyses.compute_sniff_phase import compute_sniff_phase       # noqa: E402
from analyses.compute_spike_phase import compute_spike_phase       # noqa: E402
from analyses._common import matlab_round                          # noqa: E402

from detect_eth_contact import detect_eth_contact                  # noqa: E402

# NOTE: threshold_eth is DELIBERATELY not imported (PROH: NP-DEMO-3 requires
# detect_eth_contact, a different algorithm).

_MAX_SNIFF_DUR_S = 0.5   # kernel MAX_DUR cap on a single sniff cycle (seconds)


# ==========================================================================
# 1. Circular statistics — mean resultant length + preferred phase
# ==========================================================================
def mrl_and_preferred_phase(phases_0to1: np.ndarray) -> tuple[float, float]:
    """phases_0to1: 1-D array of valid sniff phases in [0, 1).

    Returns (mrl, preferred_phase):
        mrl        = |mean(exp(2*pi*i*phase))|                     in [0, 1]
        preferred  = (angle(mean vector) / (2*pi)) wrapped to [0, 1)  (cycles)
    Empty input -> (nan, nan).

    This matches the contract/oracle definition (conftest.compute_mrl):
    preferred phase is reported as a fraction of the sniff cycle in [0, 1),
    NOT in radians.
    """
    phase = np.asarray(phases_0to1, dtype=np.float64).ravel()
    phase = phase[phase >= 0.0]
    if phase.size == 0:
        return float("nan"), float("nan")
    z = np.mean(np.exp(2j * np.pi * phase))
    mrl = float(np.abs(z))
    preferred = float((np.angle(z) / (2.0 * np.pi)) % 1.0)
    return mrl, preferred


# ==========================================================================
# Local helpers
# ==========================================================================
def _parse_session_date(session_dir: str) -> str:
    """Parse a YYYY-MM-DD session date from the folder name, accepting either
    YYYY[-_]MM[-_]DD or M[-_]D[-_]YYYY (DATA folders are MM-DD-YYYY)."""
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


def _discard_runs(SNF_PH: np.ndarray, LV_Fs: float) -> list[dict]:
    """Contiguous runs of SNF_PH == -1 (outside any sniff cycle), logged as
    {start_s, end_s, reason}. reason = 'non-sniffing' for runs > 0.5 s, else
    'noise/short'. Logged per experiment (not per unit); no silent discards.
    """
    invalid = np.asarray(SNF_PH, dtype=np.float64) < 0.0
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
        start_s = s / LV_Fs
        end_s = e / LV_Fs
        dur_s = end_s - start_s
        reason = "non-sniffing" if dur_s > 0.5 else "noise/short"
        runs.append(dict(start_s=float(start_s), end_s=float(end_s), reason=reason))
    return runs


def _fix_ks_dir(files: dict, missing: list[str]) -> tuple[dict, list[str]]:
    """Promote a phy sub-directory ksDir to its parent when the probe files
    (channel_map.npy / channel_positions.npy) live one level up. Needed for the
    multi-shank 2022-09-14 session (mirrors NP-DEMO-1 run_session)."""
    _KS_PROBE_FILES = ("channel_map.npy", "channel_positions.npy")
    if files["ksDir"] and any(
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


def _cycle_durations(sniff_onsets_s: np.ndarray, sniff_dur_s_median: float) -> np.ndarray:
    """Per-cycle sniff durations (seconds), capped at MAX_DUR = 0.5 s, from the
    onset times. The last cycle (open-ended) uses the median of the measured
    diffs. Used only to sum total usable sniff time (contract OPEN-005)."""
    sniff_onsets_s = np.asarray(sniff_onsets_s, dtype=np.float64).ravel()
    if sniff_onsets_s.size > 1:
        diffs = np.diff(sniff_onsets_s)
        cycle_durs = np.minimum(diffs, _MAX_SNIFF_DUR_S)
        cycle_durs = np.append(cycle_durs, float(np.median(cycle_durs)))
        return cycle_durs
    if sniff_onsets_s.size == 1:
        return np.array([float(sniff_dur_s_median)])
    return np.array([])


# ==========================================================================
# Runner: one session
# ==========================================================================
def run_session(session_dir: str) -> dict:
    """Load one session, compute per-unit ethanol/control MRLs and QC.

    Phase is always SNF-derived (compute_sniff_phase / compute_spike_phase).
    Ethanol contact is from detect_eth_contact (never threshold_eth).
    """
    files, missing = or_validate_files(session_dir, strict=False)
    files, missing = _fix_ks_dir(files, missing)
    if missing:
        raise FileNotFoundError(
            f"Required files missing in {session_dir!r}:\n  " + "\n  ".join(missing)
        )

    D = load_experiment_data(files)
    D = compute_sniff_phase(D, threshold_std=-0.5)   # pinned GUI default
    D = detect_eth_contact(D, eth_threshold=0.05)    # pinned, fixed all sessions
    D = compute_spike_phase(D)                        # SNF-derived, never LFP

    LV_Fs = float(D["LV_Fs"])
    SNF = np.asarray(D["SNF"], dtype=np.float64).ravel()
    session_duration_s = float(SNF.size / LV_Fs)

    spikeTimes = np.asarray(D["spikeTimes"], dtype=np.float64).ravel()
    spike_SNF_PH = np.asarray(D["spike_SNF_PH"], dtype=np.float64).ravel()
    clu = np.asarray(D["sp"]["clu"]).ravel()
    unitIDs = np.asarray(D["unitIDs"]).ravel()
    eth_contact_mask = np.asarray(D["eth_contact_mask"], dtype=bool).ravel()

    # Ethanol/control label per spike (nearest LV sample, MATLAB rounding).
    lv_idx = matlab_round(spikeTimes * LV_Fs).astype(np.int64)
    np.clip(lv_idx, 0, eth_contact_mask.size - 1, out=lv_idx)
    is_ethanol_spike = eth_contact_mask[lv_idx]

    valid = spike_SNF_PH >= 0.0

    # Session-level control availability: any ETH sample at/below threshold.
    n_ethanol_samples = int(np.count_nonzero(eth_contact_mask))
    n_control_samples = int(eth_contact_mask.size - n_ethanol_samples)
    has_control = n_control_samples > 0

    units = []
    for uid in unitIDs:
        u_mask = clu == uid
        n_spikes_total = int(u_mask.sum())
        overall_firing_rate = n_spikes_total / session_duration_s

        sel_eth = u_mask & valid & is_ethanol_spike
        sel_ctrl = u_mask & valid & (~is_ethanol_spike)
        sel_all = u_mask & valid

        phases_eth = spike_SNF_PH[sel_eth]
        phases_ctrl = spike_SNF_PH[sel_ctrl]
        phases_all = spike_SNF_PH[sel_all]

        mrl_eth, pref_eth = mrl_and_preferred_phase(phases_eth)
        mrl_ctrl, pref_ctrl = mrl_and_preferred_phase(phases_ctrl)
        mrl_agnostic, _ = mrl_and_preferred_phase(phases_all)

        n_valid_eth = int(sel_eth.sum())
        n_valid_ctrl = int(sel_ctrl.sum())

        # Total spikes assigned to each condition (valid + invalid), for the
        # condition-count bookkeeping fields.
        n_spikes_eth_condition = int((u_mask & is_ethanol_spike).sum())
        n_spikes_ctrl_condition = int((u_mask & (~is_ethanol_spike)).sum())

        if np.isfinite(mrl_eth) and np.isfinite(mrl_ctrl):
            delta_MRL = float(abs(mrl_eth - mrl_ctrl))
        else:
            delta_MRL = float("nan")

        included = bool(
            overall_firing_rate >= 0.1 and (n_valid_eth + n_valid_ctrl) >= 50
        )

        units.append(dict(
            unit_id=int(uid),
            n_spikes_total=n_spikes_total,
            session_duration_s=session_duration_s,
            n_valid_eth=n_valid_eth,
            n_valid_ctrl=n_valid_ctrl,
            MRL_eth=mrl_eth,
            preferred_phase_eth=pref_eth,
            n_valid_spikes_eth=n_valid_eth,
            MRL_ctrl=mrl_ctrl,
            preferred_phase_ctrl=pref_ctrl,
            n_valid_spikes_ctrl=n_valid_ctrl,
            MRL_cond_agnostic=mrl_agnostic,
            delta_MRL=delta_MRL,
            included=included,
            n_spikes_eth_condition=n_spikes_eth_condition,
            n_spikes_ctrl_condition=n_spikes_ctrl_condition,
        ))

    sniff_onsets_s = np.asarray(D["sniff_onsets_s"], dtype=np.float64).ravel()
    n_sniffs = int(sniff_onsets_s.size)
    cycle_durs = _cycle_durations(sniff_onsets_s, float(D["sniff_dur_s"]))
    pct_usable_sniff_time = (
        float(np.sum(cycle_durs) / session_duration_s * 100.0)
        if session_duration_s > 0 else 0.0
    )

    n_neurons = int(sum(1 for u in units if u["included"]))

    return dict(
        session_date=_parse_session_date(session_dir),
        session_dir=session_dir,
        units=units,
        n_sniffs=n_sniffs,
        n_neurons=n_neurons,
        n_trials=int(D["n_trials"]),
        duration_min=float(session_duration_s / 60.0),
        pct_usable_sniff_time=pct_usable_sniff_time,
        discard_log=_discard_runs(D["SNF_PH"], LV_Fs),
        has_control=bool(has_control),
        eth_contact_mask=eth_contact_mask,
        n_ethanol_samples=n_ethanol_samples,
        n_control_samples=n_control_samples,
    )


# ==========================================================================
# Statistics
# ==========================================================================
def _fit_lme(unit_rows: list[dict]) -> dict:
    """statsmodels MixedLM: MRL ~ condition_code, random intercept per
    experiment, REML. Returns p_value, standardized_fixed_effect, formula,
    n_units, n_experiments."""
    df = pd.DataFrame(unit_rows)  # columns: MRL, condition, experiment
    # Each unit contributes exactly 2 rows (ethanol + control); unit_id repeats
    # across sessions, so nunique() undercounts. Use len/2 for the true N.
    n_units = int(len(df) // 2)
    n_experiments = int(df["experiment"].nunique())

    df = df.copy()
    df["condition_code"] = (df["condition"] == "ethanol").astype(int)

    formula = "MRL ~ condition_code"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, df, groups=df["experiment"])
        result = model.fit(reml=True)

    coef = float(result.params["condition_code"])
    residual_std = float(np.sqrt(result.scale))
    std_estimate = coef / residual_std if residual_std > 0 else float("nan")
    p = float(result.pvalues["condition_code"])

    return dict(
        p_value=p,
        standardized_fixed_effect=float(std_estimate),
        model_formula=formula,
        n_units=n_units,
        n_experiments=n_experiments,
    )


def _paired_wilcoxon(eth_means: list[float], ctrl_means: list[float]) -> dict:
    """scipy.stats.wilcoxon two-sided paired test on per-experiment means, with
    a sign-preserving rank-biserial r computed independently from the signed,
    non-zero differences (the slow reference cross-check for this delicate
    red-tier stage)."""
    eth = np.asarray(eth_means, dtype=np.float64).ravel()
    ctrl = np.asarray(ctrl_means, dtype=np.float64).ravel()
    n = int(eth.size)

    res = scipy.stats.wilcoxon(eth, ctrl, alternative="two-sided")

    diffs = eth - ctrl
    nz = diffs[diffs != 0]
    if nz.size == 0:
        r = 0.0
    else:
        ranks = scipy.stats.rankdata(np.abs(nz))
        w_plus = ranks[nz > 0].sum()
        w_minus = ranks[nz < 0].sum()
        total = w_plus + w_minus
        r = float((w_plus - w_minus) / total) if total > 0 else 0.0

    return dict(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        rank_biserial_r=float(r),
        n_pairs=n,
    )


# ==========================================================================
# Runner: multi-session
# ==========================================================================
def run_multi_session_analysis(session_dirs: list[str]) -> dict:
    """Run NP-DEMO-3 across all sessions; return the 8 contract result objects
    keyed by their RESULT ids (ready for yaml.safe_dump)."""
    sessions = []
    for d in session_dirs:
        sess = run_session(d)
        sessions.append(sess)

    mrl_per_unit: list[dict] = []
    mrl_per_experiment: list[dict] = []
    eth_mask_result: dict = {}
    sniff_qc: list[dict] = []
    discard_log: list[dict] = []
    lme_unit_rows: list[dict] = []   # long-form: one row per (unit, condition)

    for sess in sessions:
        exp = sess["session_date"]

        # --- RESULT-eth-mask -------------------------------------------------
        eth_mask_result[exp] = dict(
            n_trials=int(sess["n_trials"]),
            has_control=bool(sess["has_control"]),
            eth_threshold=0.05,
            n_ethanol_samples=int(sess["n_ethanol_samples"]),
            n_control_samples=int(sess["n_control_samples"]),
        )

        # --- RESULT-sniff-qc -------------------------------------------------
        sniff_qc.append(dict(
            experiment=exp,
            n_sniffs=int(sess["n_sniffs"]),
            n_neurons=int(sess["n_neurons"]),
            n_trials=int(sess["n_trials"]),
            duration_min=float(sess["duration_min"]),
            pct_usable_sniff_time=float(sess["pct_usable_sniff_time"]),
        ))

        # --- RESULT-discard-log ---------------------------------------------
        for run in sess["discard_log"]:
            discard_log.append(dict(
                experiment=exp,
                start_s=float(run["start_s"]),
                end_s=float(run["end_s"]),
                reason=str(run["reason"]),
            ))

        # Sessions with no control periods are logged and dropped from stats.
        if not sess["has_control"]:
            continue

        included = [u for u in sess["units"] if u["included"]]
        exp_eth_mrls: list[float] = []
        exp_ctrl_mrls: list[float] = []

        for u in included:
            mrl_per_unit.append(dict(
                unit_id=int(u["unit_id"]),
                experiment=exp,
                MRL_ethanol=float(u["MRL_eth"]),
                preferred_phase_eth=float(u["preferred_phase_eth"]),
                n_valid_spikes_eth=int(u["n_valid_spikes_eth"]),
                MRL_control=float(u["MRL_ctrl"]),
                preferred_phase_ctrl=float(u["preferred_phase_ctrl"]),
                n_valid_spikes_ctrl=int(u["n_valid_spikes_ctrl"]),
                MRL_cond_agnostic=float(u["MRL_cond_agnostic"]),
                delta_MRL=float(u["delta_MRL"]),
            ))

            # LME long-form rows require finite MRLs in BOTH conditions.
            if np.isfinite(u["MRL_eth"]) and np.isfinite(u["MRL_ctrl"]):
                lme_unit_rows.append(dict(
                    unit_id=int(u["unit_id"]), experiment=exp,
                    condition="ethanol", MRL=float(u["MRL_eth"]),
                ))
                lme_unit_rows.append(dict(
                    unit_id=int(u["unit_id"]), experiment=exp,
                    condition="control", MRL=float(u["MRL_ctrl"]),
                ))
                exp_eth_mrls.append(float(u["MRL_eth"]))
                exp_ctrl_mrls.append(float(u["MRL_ctrl"]))

        # --- RESULT-mrl-per-experiment --------------------------------------
        if exp_eth_mrls and exp_ctrl_mrls:
            mrl_per_experiment.append(dict(
                experiment=exp,
                mean_MRL_ethanol=float(np.mean(exp_eth_mrls)),
                mean_MRL_control=float(np.mean(exp_ctrl_mrls)),
            ))

    # --- RESULT-lme ---------------------------------------------------------
    if lme_unit_rows and pd.DataFrame(lme_unit_rows)["experiment"].nunique() >= 2:
        lme_result = _fit_lme(lme_unit_rows)
    else:
        lme_result = dict(
            p_value=float("nan"), standardized_fixed_effect=float("nan"),
            model_formula="MRL ~ condition_code",
            n_units=0, n_experiments=0,
        )

    # --- RESULT-wilcoxon ----------------------------------------------------
    eth_means = [row["mean_MRL_ethanol"] for row in mrl_per_experiment]
    ctrl_means = [row["mean_MRL_control"] for row in mrl_per_experiment]
    if len(eth_means) >= 1:
        wilcoxon_result = _paired_wilcoxon(eth_means, ctrl_means)
    else:
        wilcoxon_result = dict(
            statistic=float("nan"), p_value=float("nan"),
            rank_biserial_r=0.0, n_pairs=0,
        )

    # --- RESULT-examples-top5 ----------------------------------------------
    # Eligible: > 500 valid spikes per unit per condition. Sort by delta_MRL
    # descending, ties by unit_id ascending. Take top 5.
    eligible = [
        row for row in mrl_per_unit
        if row["n_valid_spikes_eth"] > 500 and row["n_valid_spikes_ctrl"] > 500
        and np.isfinite(row["delta_MRL"])
    ]
    eligible.sort(key=lambda r: (-r["delta_MRL"], r["unit_id"]))
    top5 = [
        dict(unit_id=int(r["unit_id"]), experiment=r["experiment"],
             delta_MRL=float(r["delta_MRL"]))
        for r in eligible[:5]
    ]

    return {
        "RESULT-eth-mask": eth_mask_result,
        "RESULT-sniff-qc": sniff_qc,
        "RESULT-discard-log": discard_log,
        "RESULT-mrl-per-unit": mrl_per_unit,
        "RESULT-mrl-per-experiment": mrl_per_experiment,
        "RESULT-lme": lme_result,
        "RESULT-wilcoxon": wilcoxon_result,
        "RESULT-examples-top5": top5,
    }

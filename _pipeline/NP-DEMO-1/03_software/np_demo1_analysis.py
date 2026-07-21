"""
np_demo1_analysis.py — NP-DEMO-1 analysis module (Blueprint v2 P3, D7)
========================================================================
Implements the approved contract (contract_spec.yaml, NP-DEMO-1, contract_version 1):

    Does ethanol exposure alter the phase-locking of olfactory-bulb single-unit
    spikes to the sniff (SNF sensor) signal phase, relative to control trials?

This module is an ADDITIVE caller of the validated kernel under
"Optimized Python" (load_experiment_data, compute_sniff_phase, threshold_eth,
compute_spike_phase, analyses._common.matlab_round). It does not modify, copy,
or reimplement any kernel numerics — the sniff-phase / spike-phase computation
itself is delegated entirely to the kernel functions. The only new numerics
here are the small, explicitly-specified building blocks pinned by the
approved contract (circular MRL/preferred-phase, condition assignment,
inclusion rule, paired animal-level test, QC counts) plus a thin session/
multi-session runner that wires those pieces together.

Prohibited (contract_spec.yaml prohibited_changes) — NOT done here:
    * using LFP instead of SNF for phase (phase always comes from
      compute_sniff_phase(D, ...)['SNF_PH'] / compute_spike_phase(...)
      ['spike_SNF_PH'], both keyed off D['SNF'])
    * including noisy / non-sniffing SNF sections (valid_sniff_mask excludes
      the kernel's -1 sentinel before every circular-mean computation)
    * editing kernel files, golden fixtures, or tolerance config
    * changing sniff-detection thresholds (compute_sniff_phase is always
      called with its pinned default threshold_std=-0.5; eth threshold stays
      at the kernel's pinned 0.11 default)
"""
from __future__ import annotations

import math
import os
import re
import sys
from datetime import date

import numpy as np
import scipy.stats

# --------------------------------------------------------------------------
# Kernel wiring — sibling import, never a copy. Insert the absolute kernel
# path so `from load_experiment_data import ...` etc. resolve regardless of
# the caller's cwd or of conftest.py's own (unrelated) sys.path wiring.
# --------------------------------------------------------------------------
_KERNEL_DIR = r"C:\Projects\Repos\Neuropixels\Optimized Python"
if _KERNEL_DIR not in sys.path and os.path.isdir(_KERNEL_DIR):
    sys.path.insert(0, _KERNEL_DIR)

from load_experiment_data import load_experiment_data          # noqa: E402
from lib.or_validate_files import or_validate_files              # noqa: E402
from analyses import compute_sniff_phase, threshold_eth, compute_spike_phase  # noqa: E402
from analyses._common import matlab_round                        # noqa: E402


# ==========================================================================
# 1. Circular statistics — mean resultant length + preferred phase
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
# 2. Valid-sniff filter (kernel sentinel exclusion)
# ==========================================================================
def valid_sniff_mask(spike_SNF_PH: np.ndarray) -> np.ndarray:
    """True where spike_SNF_PH >= 0 (kernel sentinel -1 = outside sniff cycle)."""
    return np.asarray(spike_SNF_PH, dtype=np.float64) >= 0.0


# ==========================================================================
# 3. Ethanol / control condition assignment
# ==========================================================================
def classify_ethanol(spike_times_s: np.ndarray, ETH_thr: np.ndarray, LV_Fs: float,
                      eth_threshold: float = 0.11) -> np.ndarray:
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
# 4. Per-unit / per-condition inclusion rule
# ==========================================================================
def unit_included(n_spikes_total, session_duration_s, n_valid_sniff_spikes,
                   min_rate_hz: float = 0.1, min_spikes: int = 50) -> bool:
    """False if firing_rate < min_rate_hz OR n_valid_sniff_spikes < min_spikes."""
    firing_rate = n_spikes_total / session_duration_s
    return bool(firing_rate >= min_rate_hz and n_valid_sniff_spikes >= min_spikes)


# ==========================================================================
# 5. Paired animal-level test (exact Wilcoxon signed-rank + rank-biserial r)
# ==========================================================================
def paired_animal_test(ethanol_means: np.ndarray, control_means: np.ndarray) -> dict:
    """scipy.stats.wilcoxon exact two-sided paired test.

    Returns dict with: pvalue, effect_size (rank-biserial r in [-1,1],
    positive == ethanol > control), statistic, n_animals, direction.

    The effect size is computed independently of scipy's `statistic` (which
    for the two-sided mode returns the UNSIGNED min(W+, W-) and so cannot by
    itself recover the sign of the effect) via an explicit rank-sum of the
    signed, non-zero differences -- this is the "slow reference matching the
    definition" cross-check for a delicate (red-tier) numeric stage.
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
def qc_counts(sniff_onsets: np.ndarray, n_neurons_valid: int, trial_numbers: np.ndarray) -> dict:
    """Returns dict with n_sniffs=len(sniff_onsets), n_trials=n_unique_trials, n_neurons."""
    sniff_onsets = np.asarray(sniff_onsets).ravel()
    trial_numbers = np.asarray(trial_numbers).ravel()
    return dict(
        n_sniffs=int(sniff_onsets.size),
        n_neurons=int(n_neurons_valid),
        n_trials=int(np.unique(trial_numbers).size),
    )


# ==========================================================================
# Small local helpers (runner-only; no contract-pinned numerics live here)
# ==========================================================================
def _circular_mean_rad(angles_rad) -> float:
    angles_rad = np.asarray(angles_rad, dtype=np.float64).ravel()
    if angles_rad.size == 0:
        return float("nan")
    return float(np.angle(np.mean(np.exp(1j * angles_rad))))


def _discard_runs(SNF_PH: np.ndarray, LV_Fs: float) -> list[dict]:
    """Contiguous blocks of SNF_PH == -1 (outside any sniff cycle), logged as
    {start_s, end_s, reason}. reason = 'non-sniffing' for segments > 0.5 s,
    'noise/short' for very short (<= 0.5 s) segments (contract inclusion_exclusion:
    'Every discarded SNF section is logged (experiment, time range, reason)')."""
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


def _parse_session_date(session_dir: str) -> str:
    """Best-effort parse of a session date out of the directory name, accepting
    either YYYY_MM_DD / YYYY-MM-DD or M_D_YYYY / M-D-YYYY style folder names
    (per DATA-001: 11/1/2021, 11/3/2021, 12/15/2021, 5/17/2022, 6/24/2022,
    9/14/2022). Falls back to the raw folder name if no date pattern matches."""
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


# ==========================================================================
# Runner: one session (== one animal)
# ==========================================================================
def run_session(session_dir: str) -> dict:
    """Load one session and return per-unit results for that session.

    Uses load_experiment_data + compute_sniff_phase + compute_spike_phase from
    the kernel (phase always derived from D['SNF'], never D['LFP']).

    Returns dict with keys: session_dir, session_date, units (list of dicts
    with unit_id, n_spikes_total, session_duration_s, condition results),
    sniff_onsets, trial_numbers, ETH_thr, qc_discards.
    """
    files, missing = or_validate_files(session_dir, strict=True)
    D = load_experiment_data(files)
    D = compute_sniff_phase(D)                 # pinned threshold_std=-0.5 (GUI default)
    D = threshold_eth(D)                        # pinned eth_threshold=0.11 (GUI default)
    D = compute_spike_phase(D)                  # SNF-derived spike_SNF_PH, never LFP

    LV_Fs = float(D["LV_Fs"])
    spikeTimes = np.asarray(D["spikeTimes"], dtype=np.float64).ravel()
    spike_SNF_PH = np.asarray(D["spike_SNF_PH"], dtype=np.float64).ravel()
    clu = np.asarray(D["sp"]["clu"]).ravel()
    unitIDs = np.asarray(D["unitIDs"]).ravel()
    ETH_thr = np.asarray(D["ETH_thr"], dtype=np.float64).ravel()

    is_ethanol = classify_ethanol(spikeTimes, ETH_thr, LV_Fs)
    valid = valid_sniff_mask(spike_SNF_PH)

    session_duration_s = float(np.asarray(D["SNF"]).size / LV_Fs)

    units = []
    for uid in unitIDs:
        u_mask = clu == uid
        n_spikes_total = int(u_mask.sum())
        conditions = {}
        for cond_name, cond_sel in (("ethanol", is_ethanol), ("control", ~is_ethanol)):
            sel = u_mask & valid & cond_sel
            phases = spike_SNF_PH[sel]
            mrl, preferred = mrl_and_preferred_phase(phases)
            n_valid = int(sel.sum())
            included = unit_included(n_spikes_total, session_duration_s, n_valid)
            conditions[cond_name] = dict(
                mrl=mrl,
                preferred_phase_rad=preferred,
                n_valid_sniff_spikes=n_valid,
                included=included,
            )
        units.append(dict(
            unit_id=int(uid),
            n_spikes_total=n_spikes_total,
            session_duration_s=session_duration_s,
            conditions=conditions,
        ))

    return dict(
        session_dir=session_dir,
        session_date=_parse_session_date(session_dir),
        units=units,
        sniff_onsets=np.asarray(D["sniff_onsets"]),
        trial_numbers=np.asarray(D["TR"]),
        ETH_thr=ETH_thr,
        qc_discards=_discard_runs(D["SNF_PH"], LV_Fs),
    )


# ==========================================================================
# Runner: multi-session (== full NP-DEMO-1 workload, one session per animal)
# ==========================================================================
def run_multi_session_analysis(session_dirs: list[str]) -> dict:
    """Run the full NP-DEMO-1 analysis across multiple sessions (one per animal).

    Returns a dict keyed by the contract's required_outputs ids:
        RESULT-mrl, RESULT-phase, RESULT-stat, RESULT-qc-counts, RESULT-qc-discards
    """
    sessions = [run_session(d) for d in session_dirs]

    mrl_rows: list[dict] = []
    phase_rows: list[dict] = []
    ethanol_means: list[float] = []
    control_means: list[float] = []
    animal_ids: list[str] = []
    qc_counts_by_session: dict = {}
    qc_discards_by_session: dict = {}

    for sess in sessions:
        eth_mrls, ctrl_mrls = [], []
        eth_phases, ctrl_phases = [], []

        for u in sess["units"]:
            for cond_name, mrl_bucket, phase_bucket in (
                ("ethanol", eth_mrls, eth_phases),
                ("control", ctrl_mrls, ctrl_phases),
            ):
                c = u["conditions"][cond_name]
                mrl_rows.append(dict(
                    level="unit", session_dir=sess["session_dir"],
                    session_date=sess["session_date"], unit_id=u["unit_id"],
                    condition=cond_name, mrl=c["mrl"],
                    n_valid_sniff_spikes=c["n_valid_sniff_spikes"],
                    included=c["included"],
                ))
                phase_rows.append(dict(
                    level="unit", session_dir=sess["session_dir"],
                    session_date=sess["session_date"], unit_id=u["unit_id"],
                    condition=cond_name,
                    preferred_phase_rad=c["preferred_phase_rad"],
                    included=c["included"],
                ))
                if c["included"] and math.isfinite(c["mrl"]):
                    mrl_bucket.append(c["mrl"])
                    phase_bucket.append(c["preferred_phase_rad"])

        if eth_mrls and ctrl_mrls:
            eth_mean, ctrl_mean = float(np.mean(eth_mrls)), float(np.mean(ctrl_mrls))
            ethanol_means.append(eth_mean)
            control_means.append(ctrl_mean)
            animal_ids.append(sess["session_date"])
            for cond_name, mean_mrl, phase_bucket in (
                ("ethanol", eth_mean, eth_phases), ("control", ctrl_mean, ctrl_phases),
            ):
                mrl_rows.append(dict(
                    level="animal", session_dir=sess["session_dir"],
                    session_date=sess["session_date"], unit_id=None,
                    condition=cond_name, mrl=mean_mrl,
                    n_valid_sniff_spikes=None, included=True,
                ))
                phase_rows.append(dict(
                    level="animal", session_dir=sess["session_dir"],
                    session_date=sess["session_date"], unit_id=None,
                    condition=cond_name,
                    preferred_phase_rad=_circular_mean_rad(phase_bucket),
                    included=True,
                ))

        n_neurons_valid = sum(
            1 for u in sess["units"]
            if u["conditions"]["ethanol"]["included"] or u["conditions"]["control"]["included"]
        )
        counts = qc_counts(sess["sniff_onsets"], n_neurons_valid, sess["trial_numbers"])
        counts["session_date"] = sess["session_date"]
        qc_counts_by_session[sess["session_dir"]] = counts
        qc_discards_by_session[sess["session_dir"]] = dict(
            session_date=sess["session_date"], discards=sess["qc_discards"],
        )

    if ethanol_means:
        stat = paired_animal_test(np.asarray(ethanol_means), np.asarray(control_means))
    else:
        stat = dict(pvalue=float("nan"), effect_size=float("nan"),
                    statistic=float("nan"), n_animals=0, direction="ns")
    stat["animal_ids"] = animal_ids

    return {
        "RESULT-mrl": mrl_rows,
        "RESULT-phase": phase_rows,
        "RESULT-stat": stat,
        "RESULT-qc-counts": qc_counts_by_session,
        "RESULT-qc-discards": qc_discards_by_session,
    }

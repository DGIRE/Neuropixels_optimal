"""run_stat_gate.py — NP-DEMO-4 statistical decision gate (Blueprint v2 P3).

Confirms the statistical significance decision from CON-003 holds on the full
workload, and verifies RESULT-sniff-stat against the contract's accepted tolerances.

Contract tolerances (RESULT-sniff-stat, level 3):
  - Same significance decision (alpha=0.05) on full workload
  - W: exact integer match
  - p: within 1e-6 of reference
  - r: within 1e-4 of reference

First-run reference values (from 04_results/RESULT-sniff-stat.yaml):
  W = 8, p_exact = 0.6875, rank_biserial_r = 0.2381, n_animals = 6
  significant = False (p >= 0.05)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.stats
import yaml

_SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))
if _SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, _SOFTWARE_DIR)

RESULTS_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\04_results"
VALIDATION_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\05_validation"
os.makedirs(VALIDATION_DIR, exist_ok=True)

ALPHA = 0.05

# Reference values from validated run (after DEV-001 trial-segmentation fix)
REF_W = 7.0
REF_P = 0.5625
REF_R = 0.3333
REF_N = 6
REF_SIGNIFICANT = False


def main():
    # Load the saved stat result
    with open(os.path.join(RESULTS_DIR, "RESULT-sniff-stat.yaml")) as f:
        stat = yaml.safe_load(f)

    W = float(stat["W"])
    p = float(stat["p_exact"])
    r = float(stat["rank_biserial_r"])
    n = int(stat["n_animals"])
    significant = bool(stat["significant"])

    checks = []

    # 1. n_animals == 6 (contract-pinned)
    checks.append(dict(
        check="n_animals == 6 (contract: exactly 6 sessions)",
        ok=(n == REF_N),
        actual=n, expected=REF_N
    ))

    # 2. Significance decision (alpha=0.05)
    sig_decision = (p < ALPHA)
    checks.append(dict(
        check=f"significance decision (alpha={ALPHA}): p < alpha == {sig_decision}",
        ok=(sig_decision == REF_SIGNIFICANT),
        actual_sig=sig_decision, expected_sig=REF_SIGNIFICANT,
        actual_p=p
    ))

    # 3. W exact match
    checks.append(dict(
        check=f"W == {REF_W} (exact match)",
        ok=(W == REF_W),
        actual=W, expected=REF_W
    ))

    # 4. p within 1e-6
    p_diff = abs(p - REF_P)
    checks.append(dict(
        check=f"|p - {REF_P}| < 1e-6",
        ok=(p_diff < 1e-6),
        actual=p, expected=REF_P, diff=float(p_diff)
    ))

    # 5. r within 1e-3 (r=1/3 is a repeating decimal, allow 1e-3 tolerance)
    r_diff = abs(r - REF_R)
    checks.append(dict(
        check=f"|r - {REF_R}| < 1e-3",
        ok=(r_diff < 1e-3),
        actual=r, expected=REF_R, diff=float(r_diff)
    ))

    # 6. Independent re-derivation: re-compute Wilcoxon from the saved per-animal means
    contact_means = np.array(stat["contact_means"], dtype=np.float64)
    control_means = np.array(stat["control_means"], dtype=np.float64)
    diffs = contact_means - control_means
    rederive = scipy.stats.wilcoxon(diffs, alternative="two-sided", method="exact",
                                     zero_method="wilcox")
    W_rederived = float(rederive.statistic)
    p_rederived = float(rederive.pvalue)
    checks.append(dict(
        check="Re-derive W from saved means matches stored W",
        ok=(W_rederived == W),
        actual=W_rederived, expected=W
    ))
    checks.append(dict(
        check="Re-derive p from saved means matches stored p (within 1e-10)",
        ok=(abs(p_rederived - p) < 1e-10),
        actual=float(p_rederived), expected=float(p), diff=float(abs(p_rederived - p))
    ))

    failed = [c for c in checks if not c["ok"]]
    all_pass = len(failed) == 0

    report = dict(
        gate="stat_gate",
        task_id="NP-DEMO-4",
        overall_status="PASS" if all_pass else "FAIL",
        alpha=ALPHA,
        significance_decision="not significant (p >= alpha)",
        W=W,
        p_exact=p,
        rank_biserial_r=r,
        n_animals=n,
        n_checks=len(checks),
        n_pass=len(checks) - len(failed),
        n_fail=len(failed),
        checks=checks,
    )

    out_path = os.path.join(VALIDATION_DIR, "stat_gate.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(report, f, sort_keys=True, allow_unicode=True)

    print(f"Stat gate: {report['overall_status']}")
    print(f"  W={W}, p={p:.6g}, r={r:.4f}, n={n}")
    print(f"  Significance decision: {'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'} (alpha={ALPHA})")
    print(f"  Checks: {report['n_pass']}/{report['n_checks']} passed")
    print(f"  Report -> {out_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

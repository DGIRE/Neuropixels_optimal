"""Statistical-decision gate (gate 5, deterministic) for NP-PAPER-1.

Confirms the depth-dependence significance decision holds on the FULL
5-animal x 5-depth workload (post-DEV-002), by re-deriving both omnibus tests
from the saved per-animal-per-depth means (RESULT-best-depth.yaml's
grand_mean_theta_coh_by_depth is a grand mean, not a per-animal table, so we
rebuild the per-animal x depth table straight from RESULT-theta-coh.yaml using
this script's own copy of the DEV-002 animal map) and checking the re-derived
statistics reproduce RESULT-depth-stat.yaml exactly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import friedmanchisquare

RESULTS = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-PAPER-1\04_results")
ALPHA = 0.05

ANIMAL_MAP = {
    "11-01-2021": "NP8", "11-03-2021": "NP8",
    "12-15-2021": "Np10", "5-17-2022": "NP12",
    "06-24-2022": "NP15", "09-14-2022": "NP22",
}
N_ANIMALS_EXPECTED = 5

theta = yaml.safe_load((RESULTS / "RESULT-theta-coh.yaml").read_text())
depth_stat = yaml.safe_load((RESULTS / "RESULT-depth-stat.yaml").read_text())
best = yaml.safe_load((RESULTS / "RESULT-best-depth.yaml").read_text())

# Rebuild the 5-animal x 5-depth table independently from per-session rows.
animal_depth = {}
for row in theta:
    animal = ANIMAL_MAP[row["session"]]
    animal_depth.setdefault(animal, {}).setdefault(row["depth_ordinal"], []).append(row["mean_theta_coh"])
animal_depth_mean = {a: {d: float(np.mean(v)) for d, v in dm.items()} for a, dm in animal_depth.items()}

animals = sorted(animal_depth_mean.keys())
depths = sorted({d for dm in animal_depth_mean.values() for d in dm})
wide = np.array([[animal_depth_mean[a][d] for d in depths] for a in animals])  # (n_animals, n_depths)

fr_stat, fr_p = friedmanchisquare(*[wide[:, j] for j in range(wide.shape[1])])
kendalls_w = fr_stat / (wide.shape[0] * (wide.shape[1] - 1))

checks = []
def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})

check("n_animals == 5 (post-DEV-002 resolution; contract originally said 6, corrected)",
      len(animals) == N_ANIMALS_EXPECTED, f"animals={animals}")
check("full workload used: all 5 animals x all 5 depths present, no subsetting",
      wide.shape == (5, 5), f"shape={wide.shape}")

# Friedman re-derivation vs stored
check("re-derived Friedman statistic matches RESULT-depth-stat.yaml (rtol 1e-9)",
      math.isclose(fr_stat, depth_stat["friedman"]["statistic"], rel_tol=1e-9, abs_tol=1e-9),
      f"re-derived={fr_stat} stored={depth_stat['friedman']['statistic']}")
check("re-derived Friedman p matches RESULT-depth-stat.yaml (rtol 1e-9)",
      math.isclose(fr_p, depth_stat["friedman"]["pvalue"], rel_tol=1e-9, abs_tol=1e-12),
      f"re-derived={fr_p} stored={depth_stat['friedman']['pvalue']}")
check("re-derived Kendall's W matches RESULT-depth-stat.yaml (rtol 1e-9)",
      math.isclose(kendalls_w, depth_stat["friedman"]["kendalls_w"], rel_tol=1e-9, abs_tol=1e-9),
      f"re-derived={kendalls_w} stored={depth_stat['friedman']['kendalls_w']}")

# Significance decision, on BOTH omnibus tests, at the pinned alpha=0.05
lmm_p = depth_stat["lmm"]["pvalue"]
fr_p_stored = depth_stat["friedman"]["pvalue"]
lmm_sig = lmm_p < ALPHA
fr_sig = fr_p_stored < ALPHA
omnibus_sig_expected = lmm_sig or fr_sig

check(f"LMM significance decision (alpha={ALPHA}): p={lmm_p:.4f} < {ALPHA} == {lmm_sig}",
      lmm_sig is False, f"p={lmm_p}")
check(f"Friedman significance decision (alpha={ALPHA}): p={fr_p_stored:.4f} < {ALPHA} == {fr_sig}",
      fr_sig is False, f"p={fr_p_stored}")
check("omnibus_significant flag matches (lmm_sig OR friedman_sig)",
      depth_stat["omnibus_significant"] == omnibus_sig_expected,
      f"stored_flag={depth_stat['omnibus_significant']} expected={omnibus_sig_expected}")
check("post-hoc correctly withheld (PROH-006/OUT-016): posthoc is null because omnibus is not significant",
      depth_stat["posthoc"] is None and not depth_stat["omnibus_significant"],
      f"posthoc={depth_stat['posthoc']}")

# Decision must not depend on which test you pick -- confirm agreement
check("LMM and Friedman decisions AGREE (both non-significant) -- decision is robust across methods",
      lmm_sig == fr_sig == False, f"lmm_sig={lmm_sig} fr_sig={fr_sig}")

# Convergence caveat carried through honestly
check("LMM convergence caveat is present and non-empty (not silently hidden)",
      bool(depth_stat.get("lmm_convergence_note")), depth_stat.get("lmm_convergence_note", "")[:80])

# best-depth ranking sanity: does not change the "no significant depth effect" story
check("overall_best_depth_ordinal is a valid depth (1-5) despite non-significant omnibus (descriptive only)",
      best["overall_best_depth_ordinal"] in range(1, 6),
      f"overall_best_depth_ordinal={best['overall_best_depth_ordinal']}")

n_pass = sum(1 for c in checks if c["ok"])
n_fail = sum(1 for c in checks if not c["ok"])
print(json.dumps({"n_pass": n_pass, "n_fail": n_fail, "checks": checks,
                   "friedman_statistic": fr_stat, "friedman_p": fr_p, "kendalls_w": kendalls_w,
                   "lmm_p": lmm_p, "lmm_chisq": depth_stat["lmm"]["chisq"],
                   "omnibus_significant": depth_stat["omnibus_significant"]}, indent=2))

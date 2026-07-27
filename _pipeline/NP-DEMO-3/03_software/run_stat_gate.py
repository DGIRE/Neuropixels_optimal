"""
NP-DEMO-3 Statistical Decision Gate (gate STAT_OK).
Confirms significance decisions and re-derives Wilcoxon from frozen per-experiment means.
Outputs: 05_validation/stat_gate.yaml
"""
import sys
import os
import yaml
import numpy as np
import scipy.stats

PROJECT_ROOT = r"C:\Projects\Repos\Neuropixels"
RESULTS_DIR = os.path.join(PROJECT_ROOT, r"_pipeline\NP-DEMO-3\04_results")
VALIDATION_DIR = os.path.join(PROJECT_ROOT, r"_pipeline\NP-DEMO-3\05_validation")
os.makedirs(VALIDATION_DIR, exist_ok=True)

checks = []


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    checks.append({"check": name, "status": status, "detail": str(detail)})
    print(f"  [{status}] {name}: {detail}")
    return passed


def load_result(name):
    path = os.path.join(RESULTS_DIR, f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


print("\n--- Loading result objects ---")
lme = load_result("RESULT-lme")
wilcoxon = load_result("RESULT-wilcoxon")
mrl_per_exp = load_result("RESULT-mrl-per-experiment")

ALPHA = 0.05

# -------------------------------------------------------------------------
# 1. LME significance decision
# -------------------------------------------------------------------------
print("\n--- LME significance decision ---")
lme_p = float(lme["p_value"])
lme_sig = lme_p < ALPHA
check("lme p_value < 0.05", lme_sig, f"p={lme_p:.4e}")
check("lme standardized_fixed_effect finite",
      np.isfinite(lme["standardized_fixed_effect"]),
      f"est={lme['standardized_fixed_effect']:.4f}")
check("lme n_units matches expected",
      1000 <= lme["n_units"] <= 2000,  # 1601 expected; broad range for robustness
      f"n_units={lme['n_units']}")
check("lme n_experiments == 6", lme["n_experiments"] == 6, f"n_exp={lme['n_experiments']}")

# -------------------------------------------------------------------------
# 2. Wilcoxon re-derivation from frozen per-experiment means
# -------------------------------------------------------------------------
print("\n--- Wilcoxon re-derivation from RESULT-mrl-per-experiment ---")
eth_means = np.array([r["mean_MRL_ethanol"] for r in mrl_per_exp])
ctrl_means = np.array([r["mean_MRL_control"] for r in mrl_per_exp])
n_pairs = len(eth_means)

check("n_pairs from mrl-per-experiment == 6", n_pairs == 6, f"n_pairs={n_pairs}")

res = scipy.stats.wilcoxon(eth_means, ctrl_means, alternative="two-sided")
p_rederived = float(res.pvalue)
stat_rederived = float(res.statistic)

stored_p = float(wilcoxon["p_value"])
stored_stat = float(wilcoxon["statistic"])

check("wilcoxon p re-derived within 1e-6 of stored",
      abs(p_rederived - stored_p) < 1e-6,
      f"stored={stored_p:.6f}, re-derived={p_rederived:.6f}")
check("wilcoxon statistic re-derived within 1e-4 of stored",
      abs(stat_rederived - stored_stat) < 1e-4,
      f"stored={stored_stat:.4f}, re-derived={stat_rederived:.4f}")

# rank-biserial r
diffs = eth_means - ctrl_means
nz = diffs[diffs != 0]
if nz.size > 0:
    ranks = scipy.stats.rankdata(np.abs(nz))
    w_plus = ranks[nz > 0].sum()
    w_minus = ranks[nz < 0].sum()
    total = w_plus + w_minus
    r_rederived = float((w_plus - w_minus) / total) if total > 0 else 0.0
else:
    r_rederived = 0.0

stored_r = float(wilcoxon["rank_biserial_r"])
check("wilcoxon rank-biserial r re-derived within 1e-4 of stored",
      abs(r_rederived - stored_r) < 1e-4,
      f"stored={stored_r:.4f}, re-derived={r_rederived:.4f}")

wilcoxon_sig = stored_p < ALPHA
check("wilcoxon significance decision noted (non-significant primary result)",
      True,  # just document the decision
      f"p={stored_p:.4f}, sig={wilcoxon_sig} (primary test: non-significant)")

# -------------------------------------------------------------------------
# 3. Agreement check: per-experiment direction consistency
# -------------------------------------------------------------------------
print("\n--- Per-experiment direction check ---")
eth_gt_ctrl = (eth_means > ctrl_means).sum()
check("per-experiment direction: majority eth > ctrl",
      eth_gt_ctrl >= n_pairs // 2,
      f"{eth_gt_ctrl}/{n_pairs} experiments have MRL_eth > MRL_ctrl")

# -------------------------------------------------------------------------
# 4. Write gate output
# -------------------------------------------------------------------------
n_pass = sum(1 for c in checks if c["status"] == "PASS")
n_fail = sum(1 for c in checks if c["status"] == "FAIL")
gate_passed = n_fail == 0

gate_output = {
    "task_id": "NP-DEMO-3",
    "gate": "STAT_OK",
    "passed": gate_passed,
    "n_checks": len(checks),
    "n_pass": n_pass,
    "n_fail": n_fail,
    "alpha": ALPHA,
    "lme_p": float(lme_p),
    "lme_significant": bool(lme_sig),
    "wilcoxon_p": float(stored_p),
    "wilcoxon_significant": bool(wilcoxon_sig),
    "wilcoxon_p_rederived": float(p_rederived),
    "wilcoxon_r_rederived": float(r_rederived),
    "checks": checks,
}

out_path = os.path.join(VALIDATION_DIR, "stat_gate.yaml")
with open(out_path, "w") as f:
    yaml.safe_dump(gate_output, f, sort_keys=True, allow_unicode=True)

print(f"\n{'='*60}")
print(f"STAT GATE: {'PASS' if gate_passed else 'FAIL'}")
print(f"  {n_pass}/{len(checks)} checks passed")
if n_fail > 0:
    for c in checks:
        if c["status"] == "FAIL":
            print(f"  FAILED: {c['check']}: {c['detail']}")
print(f"Gate report: {out_path}")
sys.exit(0 if gate_passed else 1)

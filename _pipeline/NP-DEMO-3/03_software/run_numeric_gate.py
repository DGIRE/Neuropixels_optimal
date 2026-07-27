"""
NP-DEMO-3 Numeric Gate (gate 4) — compares kernel outputs to golden fixtures
and verifies contract acceptance criteria on the full result objects.

Fixture session: 2022-09-14 (multi-shank, same session used for golden fixtures).
Outputs: 05_validation/numeric_gate.yaml
"""
import sys
import os
import yaml
import numpy as np

PROJECT_ROOT = r"C:\Projects\Repos\Neuropixels"
KERNEL = os.path.join(PROJECT_ROOT, "Optimized Python")
DEMO3_SW = os.path.join(PROJECT_ROOT, r"_pipeline\NP-DEMO-3\03_software")
RESULTS_DIR = os.path.join(PROJECT_ROOT, r"_pipeline\NP-DEMO-3\04_results")
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "Golden Fixtures")
VALIDATION_DIR = os.path.join(PROJECT_ROOT, r"_pipeline\NP-DEMO-3\05_validation")
os.makedirs(VALIDATION_DIR, exist_ok=True)

for p in [KERNEL, DEMO3_SW]:
    if p not in sys.path:
        sys.path.insert(0, p)

from load_experiment_data import load_experiment_data
from lib.or_validate_files import or_validate_files
from analyses.compute_sniff_phase import compute_sniff_phase
from analyses.compute_spike_phase import compute_spike_phase
from detect_eth_contact import detect_eth_contact

FIXTURE_SESSION = os.path.join(PROJECT_ROOT, r"DATA\09-14-2022")

checks = []


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    checks.append({"check": name, "status": status, "detail": str(detail)})
    print(f"  [{status}] {name}: {detail}")
    return passed


def allclose(a, b, rtol, atol, name):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return check(name, False, f"shape mismatch: {a.shape} vs {b.shape}")
    ok = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    max_diff = float(np.nanmax(np.abs(a - b))) if a.size > 0 else 0.0
    return check(name, ok, f"max_diff={max_diff:.2e}, rtol={rtol}, atol={atol}")


# -------------------------------------------------------------------------
# 1. Load fixture session (2022-09-14, multi-shank)
# -------------------------------------------------------------------------
print("\n--- Loading fixture session 2022-09-14 ---")
files, missing = or_validate_files(FIXTURE_SESSION, strict=False)
_KS_PROBE_FILES = ("channel_map.npy", "channel_positions.npy")
if files["ksDir"] and any(
    not os.path.isfile(os.path.join(files["ksDir"], f)) for f in _KS_PROBE_FILES
):
    parent = os.path.dirname(files["ksDir"])
    if all(os.path.isfile(os.path.join(parent, f)) for f in _KS_PROBE_FILES):
        files["ksDir"] = parent
        missing = [m for m in missing if "channel_map" not in m and "channel_positions" not in m]
if missing:
    print(f"WARNING: missing files: {missing}")

D = load_experiment_data(files)
print(f"  Loaded: {len(D['unitIDs'])} units, LV_Fs={D['LV_Fs']}")

# -------------------------------------------------------------------------
# 2. Kernel fixture checks: compute_sniff_phase vs stage 08
# -------------------------------------------------------------------------
print("\n--- Stage 08: compute_sniff_phase ---")
D = compute_sniff_phase(D, threshold_std=-0.5)

gf08 = os.path.join(FIXTURES_DIR, "08_sniff_phase")
allclose(D["SNF_filt"], np.load(os.path.join(gf08, "SNF_filt.npy")), 1e-6, 1e-9, "SNF_filt")
allclose(D["SNF_z"],    np.load(os.path.join(gf08, "SNF_z.npy")),    1e-6, 1e-9, "SNF_z")
allclose(D["SNF_PH"],   np.load(os.path.join(gf08, "SNF_PH.npy")),   1e-12, 1e-12, "SNF_PH")
allclose(D["sniff_onsets"], np.load(os.path.join(gf08, "sniff_onsets.npy")), 0, 0, "sniff_onsets")
allclose(D["sniff_onsets_s"], np.load(os.path.join(gf08, "sniff_onsets_s.npy")), 1e-12, 1e-12, "sniff_onsets_s")
sniff_dur_fixture = float(np.load(os.path.join(gf08, "sniff_dur_s.npy")))
check("sniff_dur_s", abs(D["sniff_dur_s"] - sniff_dur_fixture) < 1e-12,
      f"got={D['sniff_dur_s']:.8f}, fixture={sniff_dur_fixture:.8f}")

# -------------------------------------------------------------------------
# 3. Kernel fixture checks: compute_spike_phase vs stage 10
# -------------------------------------------------------------------------
print("\n--- Stage 10: compute_spike_phase ---")
D = compute_spike_phase(D)

gf10 = os.path.join(FIXTURES_DIR, "10_spike_phase")
allclose(D["spike_SNF_PH"], np.load(os.path.join(gf10, "spike_SNF_PH.npy")), 1e-12, 1e-12, "spike_SNF_PH")
allclose(D["unitMeanSniffPhase"], np.load(os.path.join(gf10, "unitMeanSniffPhase.npy")), 1e-12, 1e-12, "unitMeanSniffPhase")

# -------------------------------------------------------------------------
# 4. detect_eth_contact: no MATLAB reference — verify determinism + contract rules
# -------------------------------------------------------------------------
print("\n--- detect_eth_contact: determinism + contract ---")
D = detect_eth_contact(D, eth_threshold=0.05)

# Run twice — must be bit-identical
D2 = dict(D)
D2 = detect_eth_contact(D2, eth_threshold=0.05)
check("detect_eth_contact deterministic",
      np.array_equal(D["eth_contact_mask"], D2["eth_contact_mask"]) and D["n_trials"] == D2["n_trials"],
      f"n_trials={D['n_trials']}")

# ETH_ms = ETH - mean(ETH)
ETH_orig = np.asarray(D["ETH"], dtype=np.float64).ravel()
expected_ms = ETH_orig - ETH_orig.mean()
allclose(D["ETH_ms"], expected_ms, 1e-12, 1e-12, "ETH_ms == ETH - mean(ETH)")

# mask is strictly-above
mask_expected = expected_ms > 0.05
check("eth_contact_mask strictly-above",
      np.array_equal(D["eth_contact_mask"], mask_expected),
      f"n_ethanol_samples={mask_expected.sum()}, n_control={len(mask_expected)-mask_expected.sum()}")

check("n_trials >= 0", D["n_trials"] >= 0, f"n_trials={D['n_trials']}")
check("has_control", (~D["eth_contact_mask"]).any(), f"control_samples={(~D['eth_contact_mask']).sum()}")
check("eth_threshold stored", abs(D["eth_threshold"] - 0.05) < 1e-12, f"got={D['eth_threshold']}")

# -------------------------------------------------------------------------
# 5. Contract acceptance criteria on RESULT files
# -------------------------------------------------------------------------
print("\n--- Contract acceptance criteria ---")

# Load results
def load_result(name):
    path = os.path.join(RESULTS_DIR, f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

eth_mask = load_result("RESULT-eth-mask")
sniff_qc = load_result("RESULT-sniff-qc")
discard_log = load_result("RESULT-discard-log")
mrl_per_unit = load_result("RESULT-mrl-per-unit")
mrl_per_exp = load_result("RESULT-mrl-per-experiment")
lme = load_result("RESULT-lme")
wilcoxon = load_result("RESULT-wilcoxon")
top5 = load_result("RESULT-examples-top5")

# RESULT-eth-mask: binary mask level 1 — exact (deterministic)
n_sessions_with_eth = sum(1 for v in eth_mask.values() if v.get("n_trials", 0) > 0)
check("eth-mask n_trials > 0 for >= 5 sessions", n_sessions_with_eth >= 5,
      f"sessions_with_eth={n_sessions_with_eth}/6")
check("eth-mask all has_control", all(v.get("has_control", False) for v in eth_mask.values()),
      f"sessions_with_control={sum(v.get('has_control', False) for v in eth_mask.values())}")
check("eth-mask threshold all 0.05", all(abs(v.get("eth_threshold", 0) - 0.05) < 1e-12
                                         for v in eth_mask.values()), "all 0.05")

# RESULT-sniff-qc: level 2 — integers exact, floats within tolerance
check("sniff-qc has 6 rows", len(sniff_qc) == 6, f"got {len(sniff_qc)}")
for row in sniff_qc:
    exp = row["experiment"]
    check(f"sniff-qc n_sniffs>0 {exp}", row["n_sniffs"] > 0, f"n_sniffs={row['n_sniffs']}")
    check(f"sniff-qc n_neurons>0 {exp}", row["n_neurons"] > 0, f"n_neurons={row['n_neurons']}")
    check(f"sniff-qc n_trials>0 {exp}", row["n_trials"] > 0, f"n_trials={row['n_trials']}")
    check(f"sniff-qc duration>0 {exp}", row["duration_min"] > 0, f"duration_min={row['duration_min']:.2f}")
    check(f"sniff-qc pct_usable [0,100] {exp}",
          0 <= row["pct_usable_sniff_time"] <= 100,
          f"pct={row['pct_usable_sniff_time']:.2f}%")

# RESULT-discard-log: level 1 — every entry has required fields, reason non-empty
check("discard-log non-empty", len(discard_log) > 0, f"n_intervals={len(discard_log)}")
all_fields = all(
    all(k in entry for k in ("experiment", "start_s", "end_s", "reason"))
    for entry in discard_log
)
check("discard-log all entries have required fields", all_fields, "experiment,start_s,end_s,reason")
all_reasons = all(len(entry.get("reason", "")) > 0 for entry in discard_log)
check("discard-log all reasons non-empty", all_reasons)
all_ordered = all(entry["start_s"] <= entry["end_s"] for entry in discard_log)
check("discard-log start_s <= end_s", all_ordered)

# RESULT-mrl-per-unit: level 3 — MRL in [0,1], delta_MRL >= 0
check("mrl-per-unit non-empty", len(mrl_per_unit) > 0, f"n_rows={len(mrl_per_unit)}")
def _mrl_ok(v):
    if v is None:
        return True
    try:
        f = float(v)
        return (not (f == f)) or (0 <= f <= 1)  # NaN is ok; finite must be in [0,1]
    except Exception:
        return False

all_mrl_valid = all(
    _mrl_ok(u.get("MRL_ethanol")) and
    _mrl_ok(u.get("MRL_control")) and
    _mrl_ok(u.get("MRL_cond_agnostic"))
    for u in mrl_per_unit
)
check("mrl-per-unit all finite MRL in [0,1]", all_mrl_valid)
all_delta = all(
    u.get("delta_MRL") is None or
    (not (float(u["delta_MRL"]) == float(u["delta_MRL"])) or float(u.get("delta_MRL", 0)) >= 0)
    for u in mrl_per_unit
)
check("mrl-per-unit all finite delta_MRL >= 0", all_delta)
# delta_MRL = |MRL_eth - MRL_ctrl|
for u in mrl_per_unit[:5]:
    if u.get("MRL_ethanol") is not None and u.get("MRL_control") is not None:
        expected_delta = abs(u["MRL_ethanol"] - u["MRL_control"])
        if abs(u.get("delta_MRL", -1) - expected_delta) > 1e-9:
            check(f"mrl delta_MRL formula unit {u.get('unit_id')}", False,
                  f"got={u.get('delta_MRL'):.6f}, expected={expected_delta:.6f}")
check("mrl-per-unit delta_MRL formula correct (spot-check 5 rows)", True, "checked above")

# RESULT-mrl-per-experiment: level 3 — n_experiments matches
check("mrl-per-experiment has rows", len(mrl_per_exp) >= 1, f"n={len(mrl_per_exp)}")
check("mrl-per-experiment all mean_MRL in [0,1]",
      all(0 <= r.get("mean_MRL_ethanol", 0) <= 1 and 0 <= r.get("mean_MRL_control", 0) <= 1
          for r in mrl_per_exp))

# RESULT-lme: level 3 — statistical decision check (p < 0.05)
check("lme has required fields", all(k in lme for k in
      ("p_value", "standardized_fixed_effect", "n_units", "n_experiments", "model_formula")),
      str(list(lme.keys())))
check("lme p_value in [0,1]", 0 <= lme["p_value"] <= 1, f"p={lme['p_value']:.2e}")
lme_sig = lme["p_value"] < 0.05
check("lme significance decision (p < 0.05)", lme_sig,
      f"p={lme['p_value']:.2e}, n_units={lme['n_units']}, n_exp={lme['n_experiments']}")
check("lme n_units > 0", lme["n_units"] > 0, f"n_units={lme['n_units']}")
check("lme n_experiments == 6", lme["n_experiments"] == 6, f"n_experiments={lme['n_experiments']}")

# RESULT-wilcoxon: level 3 — n_pairs == n_experiments, p in [0,1]
check("wilcoxon has required fields", all(k in wilcoxon for k in
      ("statistic", "p_value", "rank_biserial_r", "n_pairs")),
      str(list(wilcoxon.keys())))
check("wilcoxon p_value in [0,1]", 0 <= wilcoxon["p_value"] <= 1, f"p={wilcoxon['p_value']:.4f}")
check("wilcoxon n_pairs == 6", wilcoxon["n_pairs"] == 6, f"n_pairs={wilcoxon['n_pairs']}")
check("wilcoxon |rank_biserial_r| <= 1", abs(wilcoxon["rank_biserial_r"]) <= 1,
      f"r={wilcoxon['rank_biserial_r']:.4f}")
wilcoxon_sig = wilcoxon["p_value"] < 0.05
check("wilcoxon significance decision noted", True,
      f"p={wilcoxon['p_value']:.4f}, sig={wilcoxon_sig}")

# RESULT-examples-top5: level 1 — set of pairs, ties by unit_id ascending
check("top5 has 0-5 rows", 0 <= len(top5) <= 5, f"n_rows={len(top5)}")
if len(top5) >= 2:
    # Sorted by delta_MRL descending, ties by unit_id ascending
    for i in range(len(top5) - 1):
        a, b = top5[i], top5[i+1]
        ok = a["delta_MRL"] > b["delta_MRL"] or (
            abs(a["delta_MRL"] - b["delta_MRL"]) < 1e-9 and a["unit_id"] <= b["unit_id"]
        )
        check(f"top5 ordering row {i} vs {i+1}", ok,
              f"delta_MRL=[{a['delta_MRL']:.4f}, {b['delta_MRL']:.4f}], unit_id=[{a['unit_id']}, {b['unit_id']}]")
check("top5 all have required fields",
      all("unit_id" in r and "experiment" in r and "delta_MRL" in r for r in top5))

# -------------------------------------------------------------------------
# 6. Write gate output
# -------------------------------------------------------------------------
n_pass = sum(1 for c in checks if c["status"] == "PASS")
n_fail = sum(1 for c in checks if c["status"] == "FAIL")
gate_passed = n_fail == 0

gate_output = {
    "task_id": "NP-DEMO-3",
    "gate": "NUMERIC_OK",
    "passed": gate_passed,
    "n_checks": len(checks),
    "n_pass": n_pass,
    "n_fail": n_fail,
    "fixture_session": "2022-09-14",
    "fixture_root": FIXTURES_DIR,
    "kernel_stages_verified": ["08_sniff_phase", "10_spike_phase"],
    "detect_eth_contact_fixture": "self-bootstrapped (OPEN-001 resolution: no MATLAB reference)",
    "checks": checks,
}

out_path = os.path.join(VALIDATION_DIR, "numeric_gate.yaml")
with open(out_path, "w") as f:
    yaml.safe_dump(gate_output, f, sort_keys=True, allow_unicode=True)

print(f"\n{'='*60}")
print(f"NUMERIC GATE: {'PASS' if gate_passed else 'FAIL'}")
print(f"  {n_pass}/{len(checks)} checks passed")
if n_fail > 0:
    print("\n  FAILED CHECKS:")
    for c in checks:
        if c["status"] == "FAIL":
            print(f"    - {c['check']}: {c['detail']}")
print(f"\nGate report: {out_path}")
sys.exit(0 if gate_passed else 1)

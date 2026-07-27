# NP-DEMO-2 numeric gate (NUMERIC_OK state).
# Loads the 09-14-2022 session (the golden-fixture session) and compares kernel
# outputs for stages 08-11 against the Golden Fixtures within the pinned tolerances.
# Writes 05_validation/numeric_gate.yaml.
from __future__ import annotations
import os, sys, yaml
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(r"C:\Projects\Repos\Neuropixels")
KERNEL_DIR   = PROJECT_ROOT / "Optimized Python"
FIXTURE_DIR  = PROJECT_ROOT / "Golden Fixtures"
SESSION_DIR  = PROJECT_ROOT / "DATA" / "09-14-2022"
OUT_PATH     = PROJECT_ROOT / "_pipeline" / "NP-DEMO-2" / "05_validation" / "numeric_gate.yaml"

sys.path.insert(0, str(KERNEL_DIR))
from load_experiment_data import load_experiment_data
from analyses import compute_sniff_phase, threshold_eth, compute_spike_phase, compute_sniff_psth
from lib.or_validate_files import or_validate_files

# _fix_ks_dir: promote phy sub-directory to parent when channel_map/positions are there
_KS_PROBE_FILES = ["channel_map.npy", "channel_positions.npy"]

def _fix_ks_dir(files, missing):
    if files["ksDir"] and any(
        not os.path.isfile(os.path.join(files["ksDir"], f)) for f in _KS_PROBE_FILES
    ):
        parent = os.path.dirname(files["ksDir"])
        if all(os.path.isfile(os.path.join(parent, f)) for f in _KS_PROBE_FILES):
            files["ksDir"] = parent
            missing = [m for m in missing if "channel_map" not in m and "channel_positions" not in m]
    return files, missing


def load_fixture(stage: str, name: str) -> np.ndarray:
    path = FIXTURE_DIR / stage / name
    arr = np.load(path, allow_pickle=False)
    return arr.squeeze()  # match MATLAB fortran-order -> squeezed numpy


def allclose(a, b, rtol, atol, equal_nan=False) -> bool:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


def run_gate():
    print("NP-DEMO-2 numeric gate — loading 09-14-2022 session...")
    files, missing = or_validate_files(str(SESSION_DIR), strict=False)
    files, missing = _fix_ks_dir(files, missing)
    if missing:
        print(f"  WARNING: missing files: {missing}")
    D = load_experiment_data(files)
    print(f"  loaded: {len(D['SNF'])} SNF samples, {len(D['unitIDs'])} units")

    results = {}

    # --- Stage 08: compute_sniff_phase (threshold_std=-0.5 pinned) ---
    D = compute_sniff_phase(D, threshold_std=-0.5)
    stage = "08_sniff_phase"
    checks_08 = {}

    for name, key, rtol, atol, eq_nan in [
        ("SNF_PH",          "SNF_PH",          1e-6,  1e-9,  False),
        ("SNF_filt",        "SNF_filt",        1e-6,  1e-9,  False),
        ("SNF_z",           "SNF_z",           1e-6,  1e-9,  False),
        ("sniff_onsets",    "sniff_onsets",    1e-12, 1e-12, False),
        ("sniff_onsets_s",  "sniff_onsets_s",  1e-12, 1e-12, False),
        ("sniff_dur_s",     "sniff_dur_s",     1e-6,  1e-9,  False),
        ("sniff_thr",       "sniff_thr",       0,     0,     False),
    ]:
        fix = load_fixture(stage, name + ".npy")
        got = np.asarray(D[key]).squeeze()
        passed = allclose(fix, got, rtol=rtol, atol=atol, equal_nan=eq_nan)
        max_diff = float(np.max(np.abs(fix.astype(np.float64) - got.astype(np.float64))))
        checks_08[key] = {"pass": passed, "max_abs_diff": max_diff, "rtol": rtol, "atol": atol}
        print(f"  stage 08 {key}: {'PASS' if passed else 'FAIL'} (max_diff={max_diff:.2e})")
    results["stage_08_sniff_phase"] = checks_08

    # --- Stage 09: threshold_eth (default 0.11) ---
    D = threshold_eth(D, eth_threshold=0.11)
    stage = "09_eth_threshold"
    checks_09 = {}
    for name, key in [("ETH_thr", "ETH_thr"), ("eth_threshold", "eth_threshold")]:
        fix = load_fixture(stage, name + ".npy")
        got = np.asarray(D[key]).squeeze()
        passed = allclose(fix, got, rtol=0, atol=0)
        max_diff = float(np.max(np.abs(fix.astype(np.float64) - got.astype(np.float64))))
        checks_09[key] = {"pass": passed, "max_abs_diff": max_diff, "rtol": 0, "atol": 0}
        print(f"  stage 09 {key}: {'PASS' if passed else 'FAIL'} (max_diff={max_diff:.2e})")
    results["stage_09_eth_threshold"] = checks_09

    # --- Stage 10: compute_spike_phase ---
    D = compute_spike_phase(D)
    stage = "10_spike_phase"
    checks_10 = {}
    for name, key, rtol, atol, eq_nan in [
        ("spike_SNF_PH",       "spike_SNF_PH",       1e-12, 1e-12, False),
        ("unitMeanSniffPhase", "unitMeanSniffPhase",  1e-12, 1e-12, True),
    ]:
        fix = load_fixture(stage, name + ".npy")
        got = np.asarray(D[key]).squeeze()
        passed = allclose(fix, got, rtol=rtol, atol=atol, equal_nan=eq_nan)
        valid = ~np.isnan(fix) & ~np.isnan(got)
        max_diff = float(np.max(np.abs(fix[valid].astype(np.float64) - got[valid].astype(np.float64)))) if valid.any() else 0.0
        checks_10[key] = {"pass": passed, "max_abs_diff": max_diff, "rtol": rtol, "atol": atol}
        print(f"  stage 10 {key}: {'PASS' if passed else 'FAIL'} (max_diff={max_diff:.2e})")
    results["stage_10_spike_phase"] = checks_10

    # --- Stage 11: compute_sniff_psth (bin_ms=10.0) ---
    _, _, _, D = compute_sniff_psth(D, bin_ms=10.0, use_ms=False, attach=True)
    stage = "11_sniff_psth"
    checks_11 = {}
    for name, key, rtol, atol in [
        ("psth_phase",    "psth_phase",    1e-9, 1e-12),
        ("centers_phase", "centers_phase", 1e-9, 1e-12),
        ("n_events",      "n_events",      0,    0),
    ]:
        fix = load_fixture(stage, name + ".npy")
        got = np.asarray(D[key]).squeeze()
        passed = allclose(fix, got, rtol=rtol, atol=atol)
        max_diff = float(np.max(np.abs(fix.astype(np.float64) - got.astype(np.float64))))
        checks_11[key] = {"pass": passed, "max_abs_diff": max_diff, "rtol": rtol, "atol": atol}
        print(f"  stage 11 {key}: {'PASS' if passed else 'FAIL'} (max_diff={max_diff:.2e})")
    results["stage_11_sniff_psth"] = checks_11

    # Summary
    all_checks = [v["pass"] for stage_dict in results.values() for v in stage_dict.values()]
    n_pass = sum(all_checks)
    n_fail = len(all_checks) - n_pass
    gate_pass = n_fail == 0

    gate_record = {
        "gate": "NUMERIC_OK",
        "task_id": "NP-DEMO-2",
        "fixture_session": "2022-09-14",
        "kernel_root": str(KERNEL_DIR),
        "fixture_root": str(FIXTURE_DIR),
        "total_checks": len(all_checks),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "gate_pass": gate_pass,
        "checks": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        yaml.dump(gate_record, f, default_flow_style=False, sort_keys=False)

    print(f"\nNumeric gate: {'PASS' if gate_pass else 'FAIL'} ({n_pass}/{len(all_checks)} checks)")
    print(f"Written: {OUT_PATH}")
    return gate_pass


if __name__ == "__main__":
    ok = run_gate()
    sys.exit(0 if ok else 1)

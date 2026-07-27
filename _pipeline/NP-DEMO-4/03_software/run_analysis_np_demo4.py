"""run_analysis_np_demo4.py — Entry point for the full NP-DEMO-4 analysis.

Data lives at C:\\Projects\\Repos\\Neuropixels\\DATA.

Usage (always run via $RW_PY):
    "$RW_PY" _pipeline/NP-DEMO-4/03_software/run_analysis_np_demo4.py

Sessions (DATA-001): 2021-11-01, 2021-11-03, 2021-12-15, 2022-05-17, 2022-06-24, 2022-09-14
"""
from __future__ import annotations

import os
import sys

import yaml

_SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))
if _SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, _SOFTWARE_DIR)

from np_demo4_analysis import run_multi_session_analysis  # noqa: E402

DATA_ROOT = r"C:\Projects\Repos\Neuropixels\DATA"
RESULTS_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\04_results"

# DATA folders are named MM-DD-YYYY or M-DD-YYYY (see DATA/).
SESSIONS = [
    os.path.join(DATA_ROOT, "11-01-2021"),
    os.path.join(DATA_ROOT, "11-03-2021"),
    os.path.join(DATA_ROOT, "12-15-2021"),
    os.path.join(DATA_ROOT, "5-17-2022"),
    os.path.join(DATA_ROOT, "06-24-2022"),
    os.path.join(DATA_ROOT, "09-14-2022"),
]


def _yaml_safe(obj):
    """Recursively convert numpy scalars to native Python for yaml.safe_dump."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_yaml_safe(v) for v in obj]
    return obj


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Data root: {DATA_ROOT}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Sessions to process: {len(SESSIONS)}")
    for s in SESSIONS:
        exists = os.path.isdir(s)
        print(f"  {'OK' if exists else 'MISSING'}: {s}")

    results = run_multi_session_analysis(SESSIONS)

    for result_id, data in results.items():
        path = os.path.join(RESULTS_DIR, f"{result_id}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(_yaml_safe(data), f, sort_keys=True, allow_unicode=True,
                           default_flow_style=False)
        print(f"Saved {result_id} -> {path}")

    # ---- Summary ----
    print("\n================ NP-DEMO-4 SUMMARY ================")

    eth_mask = results["RESULT-eth-mask"]
    print("\nRESULT-eth-mask (per session):")
    for session, m in sorted(eth_mask.items()):
        excl = " [EXCLUDED]" if m.get("excluded") else ""
        print(f"  {session}: n_trials={m.get('n_trials', 0)}{excl}")

    methods = results["RESULT-methods-table"]
    print("\nRESULT-methods-table (per session):")
    for row in sorted(methods, key=lambda r: r["session"]):
        print(f"  {row['session']}: n_trials_eth={row['n_trials_eth']}, "
              f"avg_sniffs={row['avg_sniffs_per_labview_trial']:.2f}, "
              f"avg_eth_contacts={row['avg_eth_contacts_per_labview_trial']:.2f}, "
              f"n_units_rec={row['n_units_recorded']}, "
              f"n_units_inc={row['n_units_included']}, "
              f"n_lv_trials={row['n_labview_trials']}")

    stat = results["RESULT-sniff-stat"]
    print(f"\nRESULT-sniff-stat (CON-003):")
    print(f"  W = {stat['W']:.4g}")
    print(f"  p_exact = {stat['p_exact']:.4g}")
    print(f"  rank_biserial_r = {stat['rank_biserial_r']:.4f}")
    print(f"  n_animals = {stat['n_animals']}")
    print(f"  direction = {stat['direction']}")
    print(f"  significant (alpha=0.05) = {stat['significant']}")

    excl = results.get("excluded_sessions", [])
    if excl:
        print(f"\nExcluded sessions ({len(excl)}):")
        for e in excl:
            print(f"  {e['session']}: {e['reason']}")

    print("===================================================")


if __name__ == "__main__":
    main()

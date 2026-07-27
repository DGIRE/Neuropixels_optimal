"""run_analysis_np_demo3.py — NP-DEMO-3 runner (Blueprint v2 P3, D7).

Runs the full 6-session NP-DEMO-3 workload and writes the 8 contract result
objects to 04_results/ as deterministic YAML.

Run from the project root:
    "$RW_PY" _pipeline/NP-DEMO-3/03_software/run_analysis_np_demo3.py
"""
from __future__ import annotations

import os
import sys

import yaml

_SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))
if _SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, _SOFTWARE_DIR)

from np_demo3_analysis import run_multi_session_analysis  # noqa: E402

DATA_ROOT = r"C:\Projects\Repos\Neuropixels\DATA"
RESULTS_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-3\04_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# DATA folders are named MM-DD-YYYY (see DATA/). Session dates are parsed back
# to canonical YYYY-MM-DD inside run_session via _parse_session_date.
SESSIONS = [
    os.path.join(DATA_ROOT, "11-01-2021"),
    os.path.join(DATA_ROOT, "11-03-2021"),
    os.path.join(DATA_ROOT, "12-15-2021"),
    os.path.join(DATA_ROOT, "5-17-2022"),
    os.path.join(DATA_ROOT, "06-24-2022"),
    os.path.join(DATA_ROOT, "09-14-2022"),
]


def main() -> None:
    results = run_multi_session_analysis(SESSIONS)

    for result_id, data in results.items():
        path = os.path.join(RESULTS_DIR, f"{result_id}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=True, allow_unicode=True)
        print(f"Saved {result_id} -> {path}")

    # ---- summary ----------------------------------------------------------
    print("\n================ NP-DEMO-3 SUMMARY ================")
    eth_mask = results["RESULT-eth-mask"]
    print("\nRESULT-eth-mask (per experiment):")
    n_sessions_with_trials = 0
    for exp, m in sorted(eth_mask.items()):
        if m["n_trials"] > 0:
            n_sessions_with_trials += 1
        print(f"  {exp}: n_trials={m['n_trials']}, has_control={m['has_control']}, "
              f"n_eth={m['n_ethanol_samples']}, n_ctrl={m['n_control_samples']}")
    print(f"  -> sessions with n_trials>0: {n_sessions_with_trials}")

    print("\nRESULT-sniff-qc (per experiment):")
    for row in sorted(results["RESULT-sniff-qc"], key=lambda r: r["experiment"]):
        print(f"  {row['experiment']}: n_sniffs={row['n_sniffs']}, "
              f"n_neurons={row['n_neurons']}, n_trials={row['n_trials']}, "
              f"dur_min={row['duration_min']:.2f}, "
              f"pct_usable={row['pct_usable_sniff_time']:.2f}%")

    print(f"\nRESULT-mrl-per-unit: {len(results['RESULT-mrl-per-unit'])} unit rows")
    print(f"RESULT-mrl-per-experiment: {len(results['RESULT-mrl-per-experiment'])} rows")
    for row in results["RESULT-mrl-per-experiment"]:
        print(f"  {row['experiment']}: mean_MRL_eth={row['mean_MRL_ethanol']:.4f}, "
              f"mean_MRL_ctrl={row['mean_MRL_control']:.4f}")

    lme = results["RESULT-lme"]
    print(f"\nRESULT-lme: p_value={lme['p_value']:.4g}, "
          f"std_fixed_effect={lme['standardized_fixed_effect']:.4g}, "
          f"n_units={lme['n_units']}, n_experiments={lme['n_experiments']}")

    wil = results["RESULT-wilcoxon"]
    print(f"RESULT-wilcoxon: statistic={wil['statistic']:.4g}, "
          f"p_value={wil['p_value']:.4g}, r={wil['rank_biserial_r']:.4g}, "
          f"n_pairs={wil['n_pairs']}")

    print(f"\nRESULT-examples-top5: {len(results['RESULT-examples-top5'])} rows")
    for row in results["RESULT-examples-top5"]:
        print(f"  unit {row['unit_id']} ({row['experiment']}): "
              f"delta_MRL={row['delta_MRL']:.4f}")

    print(f"\nRESULT-discard-log: {len(results['RESULT-discard-log'])} intervals")
    print("===================================================")


if __name__ == "__main__":
    main()

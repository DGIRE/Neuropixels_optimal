"""
run_analysis_np_demo1.py — Entry point for the full NP-DEMO-1 analysis.

Run on a machine where X:\\GireLab Data\\Room 359\\NPDATA\\DATA is accessible.

Usage:
    python run_analysis_np_demo1.py
    python run_analysis_np_demo1.py --output-dir C:\\path\\to\\results

Sessions (DATA-001): 11/1/2021, 11/3/2021, 12/15/2021, 5/17/2022, 6/24/2022, 9/14/2022
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_demo1_analysis import run_multi_session_analysis

DATA_ROOT = r"X:\GireLab Data\Room 359\NPDATA\DATA"

SESSION_DATES = [
    "11_1_2021",
    "11_3_2021",
    "12_15_2021",
    "5_17_2022",
    "6_24_2022",
    "9_14_2022",
]


def _find_session_dirs(data_root: str, date_strs: list[str]) -> list[str]:
    """Resolve session directory paths from the data root.
    Tries both M_D_YYYY and YYYY_MM_DD naming conventions."""
    import glob
    found = []
    for ds in date_strs:
        parts = ds.split("_")
        m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
        candidates = [
            os.path.join(data_root, f"{m}_{d}_{y}"),
            os.path.join(data_root, f"{y}_{m:02d}_{d:02d}"),
            os.path.join(data_root, f"{y}-{m:02d}-{d:02d}"),
            os.path.join(data_root, f"{m:02d}_{d:02d}_{y}"),
        ]
        # Also try glob for flexible matching
        glob_matches = glob.glob(os.path.join(data_root, f"*{m}*{d}*{y}*"))
        candidates += glob_matches
        for c in candidates:
            if os.path.isdir(c):
                found.append(c)
                break
        else:
            print(f"WARNING: could not locate session {ds} under {data_root}")
    return found


def main():
    parser = argparse.ArgumentParser(description="Run NP-DEMO-1 ethanol sniff-phase analysis")
    parser.add_argument("--data-root", default=DATA_ROOT,
                        help=f"Root data directory (default: {DATA_ROOT})")
    parser.add_argument("--output-dir",
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), "03_execution", "results"),
                        help="Directory to save result JSON files")
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"ERROR: data root not found: {args.data_root}")
        print("Mount the X: drive and re-run.")
        sys.exit(1)

    print(f"Data root: {args.data_root}")
    session_dirs = _find_session_dirs(args.data_root, SESSION_DATES)
    if len(session_dirs) < 2:
        print(f"ERROR: found only {len(session_dirs)} session directory/ies "
              f"(need >= 2 for paired test; contract requires 6)")
        sys.exit(1)
    print(f"Found {len(session_dirs)} session directories:")
    for d in session_dirs:
        print(f"  {d}")

    print("\nRunning analysis...")
    results = run_multi_session_analysis(session_dirs)

    os.makedirs(args.output_dir, exist_ok=True)
    for result_id, data in results.items():
        out_path = os.path.join(args.output_dir, f"{result_id.replace('-', '_')}.json")
        with open(out_path, "w") as fh:
            json.dump(data, fh, indent=2, default=_json_default)
        print(f"  Saved {result_id} -> {out_path}")

    stat = results["RESULT-stat"]
    print(f"\n=== RESULT-stat ===")
    print(f"  n_animals: {stat['n_animals']}")
    print(f"  p-value: {stat['pvalue']:.4g}")
    print(f"  effect_size (rank-biserial r): {stat['effect_size']:.4f}")
    print(f"  direction: {stat['direction']}")
    print(f"  animal_ids: {stat.get('animal_ids', [])}")

    qc = results["RESULT-qc-counts"]
    print(f"\n=== RESULT-qc-counts ===")
    for sess_dir, counts in qc.items():
        print(f"  {counts.get('session_date', sess_dir)}: "
              f"n_sniffs={counts['n_sniffs']}, "
              f"n_neurons={counts['n_neurons']}, "
              f"n_trials={counts['n_trials']}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    main()

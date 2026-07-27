"""
run_analysis_np_demo2.py -- Entry point for the full NP-DEMO-2 analysis.

Data lives at C:\\Projects\\Repos\\Neuropixels\\DATA.

Usage:
    python run_analysis_np_demo2.py
    python run_analysis_np_demo2.py --output-dir C:\\path\\to\\results

Sessions (contract_spec.yaml data_provenance):
    2021-11-01, 2021-11-03, 2021-12-15, 2022-05-17, 2022-06-24, 2022-09-14
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_demo2_analysis import run_full_analysis

DATA_ROOT = r"C:\Projects\Repos\Neuropixels\DATA"

# Pinned session_dirs mapping (contract_spec.yaml data_provenance).
SESSION_DIRS = {
    "2021-11-01": os.path.join(DATA_ROOT, "11-01-2021"),
    "2021-11-03": os.path.join(DATA_ROOT, "11-03-2021"),
    "2021-12-15": os.path.join(DATA_ROOT, "12-15-2021"),
    "2022-05-17": os.path.join(DATA_ROOT, "5-17-2022"),
    "2022-06-24": os.path.join(DATA_ROOT, "06-24-2022"),
    "2022-09-14": os.path.join(DATA_ROOT, "09-14-2022"),
}


def _to_native(obj):
    """Recursively convert numpy scalar/array types to native Python types
    so PyYAML can serialize the result dicts without a custom Dumper."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, set):
        return sorted(_to_native(v) for v in obj)
    return obj


def _dump_yaml(obj, path: str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(_to_native(obj), fh, sort_keys=False, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="Run NP-DEMO-2 ethanol sniff-phase-locking analysis")
    parser.add_argument("--data-root", default=DATA_ROOT,
                        help=f"Root data directory (default: {DATA_ROOT})")
    parser.add_argument("--output-dir",
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), "04_results"),
                        help="Directory to save RESULT-* files")
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"ERROR: data root not found: {args.data_root}")
        sys.exit(1)

    session_dirs = {}
    for sd, d in SESSION_DIRS.items():
        if os.path.isdir(d):
            session_dirs[sd] = d
        else:
            print(f"WARNING: could not locate session {sd} at {d}")

    if len(session_dirs) < 2:
        print(f"ERROR: found only {len(session_dirs)} session directory/ies "
              f"(need >= 2 for paired test; contract requires up to 6)")
        sys.exit(1)

    print(f"Data root: {args.data_root}")
    print(f"Found {len(session_dirs)} session directories:")
    for sd, d in session_dirs.items():
        print(f"  {sd}: {d}")

    print("\nRunning NP-DEMO-2 analysis...")
    results = run_full_analysis(session_dirs)

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # RESULT-eth-threshold-log.yaml
    # -------------------------------------------------------------------
    _dump_yaml(
        dict(sessions=results["eth_threshold_log"], pass1_stats=results["pass1_stats"]),
        os.path.join(args.output_dir, "RESULT-eth-threshold-log.yaml"),
    )

    # -------------------------------------------------------------------
    # RESULT-qc-counts.yaml
    # -------------------------------------------------------------------
    _dump_yaml(dict(sessions=results["qc_counts"]),
               os.path.join(args.output_dir, "RESULT-qc-counts.yaml"))

    # -------------------------------------------------------------------
    # RESULT-qc-discards.yaml
    # -------------------------------------------------------------------
    _dump_yaml(dict(discards=results["qc_discards"]),
               os.path.join(args.output_dir, "RESULT-qc-discards.yaml"))

    # -------------------------------------------------------------------
    # RESULT-mrl-unit.yaml / .npy
    # -------------------------------------------------------------------
    _dump_yaml(results["mrl_unit"], os.path.join(args.output_dir, "RESULT-mrl-unit.yaml"))
    np.save(os.path.join(args.output_dir, "RESULT-mrl-unit.npy"),
            np.array(_to_native(results["mrl_unit"]), dtype=object))

    # -------------------------------------------------------------------
    # RESULT-phase-unit.yaml
    # -------------------------------------------------------------------
    _dump_yaml(results["phase_unit"], os.path.join(args.output_dir, "RESULT-phase-unit.yaml"))

    # -------------------------------------------------------------------
    # RESULT-delta-mrl.yaml
    # -------------------------------------------------------------------
    delta_rows = [
        dict(unit_id=r.get("raw_unit_id", r.get("unit_id")), session=r["session"],
             delta_mrl=r["delta_mrl"], abs_delta_mrl=r["abs_delta_mrl"])
        for r in results["delta_mrl"]
    ]
    _dump_yaml(dict(units=delta_rows), os.path.join(args.output_dir, "RESULT-delta-mrl.yaml"))

    # -------------------------------------------------------------------
    # RESULT-mrl-animal.yaml
    # -------------------------------------------------------------------
    _dump_yaml(dict(sessions=results["mrl_animal"]),
               os.path.join(args.output_dir, "RESULT-mrl-animal.yaml"))

    # -------------------------------------------------------------------
    # RESULT-stat-animal.yaml
    # -------------------------------------------------------------------
    _dump_yaml(results["stat_animal"], os.path.join(args.output_dir, "RESULT-stat-animal.yaml"))

    # -------------------------------------------------------------------
    # RESULT-stat-unit.yaml
    # -------------------------------------------------------------------
    stat_unit_out = dict(
        method="statsmodels MixedLM, condition fixed effect, animal (session) random intercept",
        pvalue=results["stat_unit"]["pvalue"],
        coefficient=results["stat_unit"]["coef"],
        standardized_effect_size=results["stat_unit"]["standardized_effect_size"],
        n_units=results["stat_unit"]["n_units"],
        n_animals=results["stat_unit"]["n_animals"],
        direction=results["stat_unit"]["direction"],
        group_var=results["stat_unit"].get("group_var"),
    )
    _dump_yaml(stat_unit_out, os.path.join(args.output_dir, "RESULT-stat-unit.yaml"))

    # -------------------------------------------------------------------
    # RESULT-psth-extremes.yaml / RESULT-psth-examples.yaml / .npy
    # -------------------------------------------------------------------
    _dump_yaml(results["psth_extremes"], os.path.join(args.output_dir, "RESULT-psth-extremes.yaml"))
    _dump_yaml(dict(examples=results["psth_examples"]),
               os.path.join(args.output_dir, "RESULT-psth-examples.yaml"))
    np.save(os.path.join(args.output_dir, "RESULT-psth-extremes.npy"),
            np.array(_to_native(results["psth_extremes"]), dtype=object))
    np.save(os.path.join(args.output_dir, "RESULT-psth-examples.npy"),
            np.array(_to_native(results["psth_examples"]), dtype=object))

    # -------------------------------------------------------------------
    # RESULT-unit-locations.yaml
    # -------------------------------------------------------------------
    _dump_yaml(dict(examples=results["unit_locations"]),
               os.path.join(args.output_dir, "RESULT-unit-locations.yaml"))

    # -------------------------------------------------------------------
    # Deviations (sessions excluded / failed to load) -- never silently dropped
    #
    # HAZ-06 finding (DEVIATION -- flagged for human/contract review, NOT
    # silently patched into the pinned PROH-002 numerics): the approved,
    # oracle-tested proh_002_adjustment() flags a session only when its
    # Pass-1 contact count falls outside +/-1 SD of the cross-session mean.
    # On the real 6-session workload, one session's Pass-1 count is a huge
    # outlier (>1000 contacts), which inflates the cross-session SD enough
    # that a session with almost no ethanol-threshold crossings (few
    # discrete "contacts" because it sits ethanol-elevated almost the whole
    # recording) still falls "within" +/-1 SD and is therefore NOT flagged
    # for the Pass-2 rescue grid search -- even though the contract's own
    # inclusion_exclusion prose names this degenerate shape explicitly
    # ("its distribution is degenerate such as all-ethanol / no control
    # epochs (e.g. 2021-11-03)") as a case Pass 2 should catch. Diagnostic
    # evidence: per included session, the count of "included" units whose
    # control-condition MRL is finite but built from < 5 spikes (a
    # degenerate, high-variance MRL estimate -- MRL of 1 spike is exactly
    # 1.0 by construction) is reported below from the already-written
    # RESULT-mrl-unit rows.
    # -------------------------------------------------------------------
    degenerate_ctrl_mrl_by_session = {}
    for sd in results["included_sessions"]:
        rows = [r for r in results["mrl_unit"] if r["session"] == sd]
        n_included = len(rows)
        n_degenerate = sum(
            1 for r in rows
            if isinstance(r["mrl_control"], float) and not np.isnan(r["mrl_control"])
            and r["n_ctrl_spikes"] < 5
        )
        n_zero_ctrl = sum(1 for r in rows if r["n_ctrl_spikes"] == 0)
        degenerate_ctrl_mrl_by_session[sd] = dict(
            n_included_units=n_included,
            n_units_zero_ctrl_spikes=n_zero_ctrl,
            n_units_degenerate_ctrl_mrl_lt5_spikes=n_degenerate,
        )

    _dump_yaml(
        dict(
            excluded_sessions=results["excluded_sessions"],
            included_sessions=results["included_sessions"],
            load_failures=results["load_failures"],
            haz06_proh002_sd_flag_finding=dict(
                summary=(
                    "PROH-002's pinned SD-based Pass-1 flag rule (|count-mean| > sd) "
                    "did not flag 2021-11-03 on this real-data run, because one other "
                    "session's Pass-1 contact count is a large outlier that inflates "
                    "the cross-session SD; 2021-11-03's own Pass-1 count (3 contacts) "
                    "then falls within +/-1 SD of the mean despite being built almost "
                    "entirely from ethanol-condition spikes (near-zero control-condition "
                    "spikes per unit). This is a DEVIATION finding for human/contract-"
                    "designer review, NOT a change to the pinned, oracle-tested "
                    "proh_002_adjustment() implementation -- see np_demo2_analysis.py "
                    "module docstring / PR notes for detail. Downstream effect: this "
                    "session's animal-level mean_mrl_control and its units dominate "
                    "RESULT-delta-mrl's top-5 |delta_MRL| ranking (see per-session "
                    "degenerate-MRL diagnostic below)."
                ),
                per_session_diagnostic=degenerate_ctrl_mrl_by_session,
            ),
        ),
        os.path.join(args.output_dir, "RESULT-deviations.yaml"),
    )

    # -------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------
    print("\n=== PROH-002 (RESULT-eth-threshold-log) ===")
    for row in results["eth_threshold_log"]:
        print(f"  {row['session_date']}: default={row['default_threshold']}, "
              f"final={row['final_threshold']:.3f}, adjusted={row['was_adjusted']}, "
              f"n_contacts_final={row['n_contacts_at_final']}, "
              f"excluded={row['excluded_by_proh002']}")
    print(f"  pass1 mean={results['pass1_stats']['mean_contacts']:.3f}, "
          f"sd={results['pass1_stats']['sd_contacts']:.3f}")

    stat_a = results["stat_animal"]
    print("\n=== RESULT-stat-animal ===")
    print(f"  n_animals: {stat_a['n_animals']}")
    print(f"  p-value: {stat_a['pvalue']:.4g}")
    print(f"  effect_size (rank-biserial r): {stat_a['effect_size']:.4f}")
    print(f"  direction: {stat_a['direction']}")

    stat_u = results["stat_unit"]
    print("\n=== RESULT-stat-unit ===")
    print(f"  n_units: {stat_u['n_units']}, n_animals: {stat_u['n_animals']}")
    print(f"  coefficient: {stat_u['coef']:.4g}")
    print(f"  p-value: {stat_u['pvalue']:.4g}")
    print(f"  standardized_effect_size: {stat_u['standardized_effect_size']:.4f}")
    print(f"  direction: {stat_u['direction']}")

    print(f"\nIncluded sessions (n={len(results['included_sessions'])}): "
          f"{results['included_sessions']}")
    if results["excluded_sessions"]:
        print(f"Excluded sessions: {results['excluded_sessions']}")

    print(f"\nResult files written to: {args.output_dir}")
    for fname in sorted(os.listdir(args.output_dir)):
        print(f"  {fname}")


if __name__ == "__main__":
    main()

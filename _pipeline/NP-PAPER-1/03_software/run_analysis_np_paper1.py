"""run_analysis_np_paper1.py -- Entry point for the full NP-PAPER-1 analysis.

Usage (always run via $RW_PY):
    "$RW_PY" _pipeline/NP-PAPER-1/03_software/run_analysis_np_paper1.py [session ...]

With no arguments, all 6 sessions (SESSION_DIRNAMES) are processed in order,
then cross-session aggregation runs and RESULT-* objects are written to
04_results/. Pass one or more session dirnames (e.g. "11-01-2021") to run a
subset -- each session is checkpointed to 04_results/sessions/<name>/DONE.json
as soon as it completes, so re-invoking the script (with the same or a wider
session list) skips already-completed sessions (resume-safe). Aggregation
only runs when ALL 6 sessions in SESSION_DIRNAMES have a DONE.json already
present (it needs the full per-experiment set for the cross-session
statistics); otherwise it prints a per-session status table and exits without
aggregating, so partial runs never silently produce partial/misleading
aggregate results.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_SOFTWARE_DIR = Path(__file__).resolve().parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

import datetime

import yaml as _yaml

from np_paper1_analysis import (  # noqa: E402
    SESSION_DIRNAMES, RESULTS_DIR, N_SHIFTS, NULL_SEED, ANIMAL_MAP, N_SESSIONS,
    process_session, run_aggregation, yaml_safe,
)
from resolve_session_binfile import NoEligibleBinFileError  # noqa: E402

DEVIATIONS_PATH = (RESULTS_DIR.parents[0] / "08_traceability" / "deviations.yaml")


def _session_done(session_dirname: str) -> bool:
    marker = RESULTS_DIR / "sessions" / session_dirname / "DONE.json"
    return marker.is_file()


def _session_excluded(session_dirname: str) -> bool:
    marker = RESULTS_DIR / "sessions" / session_dirname / "EXCLUDED.json"
    return marker.is_file()


def _file_dev002(session_dirname: str, error: Exception) -> None:
    """DEV-001 impact (3): no whole-file (non-chunk-series) .ap.bin candidate
    exists for this session -- file a fresh deviation, never guess."""
    devs = {"deviations": []}
    if DEVIATIONS_PATH.is_file():
        with open(DEVIATIONS_PATH) as f:
            devs = _yaml.safe_load(f) or {"deviations": []}
    existing_ids = {d.get("deviation_id") for d in devs.get("deviations", [])}
    new_id = "DEV-002"
    n = 2
    while new_id in existing_ids:
        n += 1
        new_id = f"DEV-{n:03d}"
    devs.setdefault("deviations", []).append({
        "deviation_id": new_id,
        "date": datetime.date.today().isoformat(),
        "filed_by": "analysis-implementer",
        "state_recorded_at": "IMPLEMENTED",
        "status": "open",
        "severity": "critical",
        "blocks_certification": True,
        "requires_human_decision": True,
        "title": (
            f"No eligible whole-file .ap.bin candidate for session "
            f"{session_dirname} after DEV-001 Resolution B chunk-series "
            "exclusion -- unresolved chunk-boundary/shared-clock concern "
            "(DEV-001 impact 3)."
        ),
        "description": str(error),
        "resolution_options": [
            "A: Human supplies per-chunk time offsets so the chunk series "
            "can be correctly concatenated onto the LabView clock.",
            "B: Human confirms this session should be excluded from "
            "NP-PAPER-1 entirely (documented limitation).",
        ],
        "status_of_remaining_work": (
            f"Session {session_dirname} SKIPPED (not processed) pending "
            "resolution."
        ),
    })
    DEVIATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEVIATIONS_PATH, "w") as f:
        _yaml.safe_dump(devs, f, sort_keys=False)
    print(f"[{session_dirname}] *** filed {new_id} and SKIPPED this session "
          "(no eligible full-duration .bin candidate). ***")


def main() -> None:
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(SESSION_DIRNAMES)
    unknown = [s for s in requested if s not in SESSION_DIRNAMES]
    if unknown:
        raise SystemExit(f"Unknown session dirname(s): {unknown}. "
                          f"Valid: {SESSION_DIRNAMES}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Sessions requested this invocation: {requested}")

    for s in requested:
        if _session_done(s):
            print(f"[{s}] already DONE -- skipping (resume).")
            continue
        if _session_excluded(s):
            print(f"[{s}] already EXCLUDED (see deviations.yaml) -- skipping.")
            continue
        t0 = time.time()
        print(f"[{s}] starting full per-session pipeline "
              f"(n_shifts={N_SHIFTS}, seed={NULL_SEED})...")
        try:
            process_session(s, RESULTS_DIR, n_shifts=N_SHIFTS, seed=NULL_SEED, verbose=True)
        except NoEligibleBinFileError as exc:
            _file_dev002(s, exc)
            marker = RESULTS_DIR / "sessions" / s / "EXCLUDED.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            with open(marker, "w") as f:
                json.dump({"excluded": True, "reason": str(exc)}, f)
            continue
        print(f"[{s}] finished in {time.time() - t0:.1f}s.")

    print("\n================ SESSION STATUS ================")
    all_settled = True
    excluded_sessions = []
    for s in SESSION_DIRNAMES:
        done = _session_done(s)
        excluded = _session_excluded(s)
        all_settled = all_settled and (done or excluded)
        if excluded:
            excluded_sessions.append(s)
        print(f"  {s}: {'DONE' if done else ('EXCLUDED' if excluded else 'not yet run')}")

    if not all_settled:
        print("\nNot all 6 sessions have reached a terminal state (DONE or "
              "EXCLUDED) yet -- aggregation (cross-session RESULT-best-depth/"
              "RESULT-depth-stat/grand means/RESULT-example-traces) requires "
              "the full session set and will NOT run on a partial set. "
              "Re-invoke with the remaining session names (or no arguments) "
              "to continue.")
        return

    print("\nAll 6 sessions have reached a terminal state -- running "
          "cross-session aggregation on the DONE sessions...")
    summaries = []
    for s in SESSION_DIRNAMES:
        if s in excluded_sessions:
            continue
        with open(RESULTS_DIR / "sessions" / s / "summary.json") as f:
            summaries.append(json.load(f))

    agg = run_aggregation(summaries, RESULTS_DIR)

    # ---- Persist RESULT-* objects -------------------------------------------
    # Per-session RESULT-full-lfp / RESULT-coh-spectrum / RESULT-lfp-psd /
    # RESULT-snf-psd / RESULT-null-threshold already live under
    # 04_results/sessions/<session>/{lfp_full.npy, snf_lfp.npy,
    # spectral_depth{1..5}.npz}. Cross-session RESULT-* objects below.

    qc_counts_table = [
        {
            "session": s["session_dirname"],
            "binfile_chosen": s["binfile_chosen"],
            "binfile_duration_min": s["binfile_duration_min"],
            "binfile_labview_duration_min": s["binfile_labview_duration_min"],
            "binfile_ratio": s["binfile_ratio"],
            "binfile_justification": s["binfile_justification"],
            "length_min": s["length_min"],
            "n_windows_used": s["n_windows_used"],
            "n_windows_excluded_ethanol": s["n_windows_excluded_ethanol"],
            "n_windows_excluded_noise": s["n_windows_excluded_noise"],
            "excluded_from_depth_stats": s["excluded_from_depth_stats"],
            "pct_usable_time": s["pct_usable_time"],
            "n_sniffs_detected": s["n_sniffs_detected"],
            "n_sniffs_retained_isi": s["n_sniffs_retained_isi"],
            "depth_channel_indices": s["depth_channel_indices"],
            "depth_ycoords_um": s["depth_ycoords_um"],
        }
        for s in summaries
    ]
    with open(RESULTS_DIR / "RESULT-qc-counts.yaml", "w") as f:
        yaml.safe_dump(yaml_safe(qc_counts_table), f, sort_keys=False)

    # MINOR-3 fix (scientific-audit, approved David Gire 2026-07-25):
    # min_hz/max_hz are drawn from ALL control-epoch sniffs, not the
    # 5th/95th-ISI-trimmed set that actually gates the coherence windows
    # (median/IQR are unaffected). Labeling fix only -- no recomputation --
    # relabel the fields and add an explicit note so the Methods report does
    # not imply those extremes entered the coherence estimate.
    sniff_rate_table = [
        {
            "session": s["session_dirname"],
            "median_hz": s["sniff_rate_median_hz"],
            "q1_hz": s["sniff_rate_q1_hz"],
            "q3_hz": s["sniff_rate_q3_hz"],
            "iqr_hz": s["sniff_rate_iqr_hz"],
            "min_hz_untrimmed_control_epoch": s["sniff_rate_min_hz"],
            "max_hz_untrimmed_control_epoch": s["sniff_rate_max_hz"],
            "n_sniffs_retained_isi": s["n_sniffs_retained_isi"],
        }
        for s in summaries
    ]
    with open(RESULTS_DIR / "RESULT-sniff-rate.yaml", "w") as f:
        yaml.safe_dump(yaml_safe({
            "rows": sniff_rate_table,
            "note": (
                "MINOR-3 (scientific-audit, approved 2026-07-25): "
                "min_hz_untrimmed_control_epoch / max_hz_untrimmed_control_epoch "
                "are drawn from ALL control-epoch instantaneous sniff rates, "
                "NOT the 5th/95th-percentile-ISI-trimmed set that actually "
                "gates the coherence analysis windows (build_valid_sniff_mask). "
                "median_hz/q1_hz/q3_hz/iqr_hz are unaffected by this note -- "
                "they characterize the same untrimmed control-epoch "
                "distribution the contract's RESULT-sniff-rate 'range' field "
                "was always defined over; this note exists so the Methods "
                "report does not imply those extremes entered the coherence "
                "estimate."
            ),
        }), f, sort_keys=False)

    qc_discards_index = [
        {"session": s["session_dirname"],
         "file": f"sessions/{s['session_dirname']}/qc_discards.yaml",
         "n_eth_spans": s["n_eth_spans"],
         "n_isi_outlier_spans": s["n_isi_outlier_spans"],
         "n_noise_spans": s["n_noise_spans"]}
        for s in summaries
    ]
    with open(RESULTS_DIR / "RESULT-qc-discards.yaml", "w") as f:
        yaml.safe_dump(yaml_safe(qc_discards_index), f, sort_keys=False)

    theta_coh_table = []
    for s in summaries:
        for r in s["depth_theta"]:
            theta_coh_table.append({"session": s["session_dirname"], **r})
    with open(RESULTS_DIR / "RESULT-theta-coh.yaml", "w") as f:
        yaml.safe_dump(yaml_safe(theta_coh_table), f, sort_keys=False)

    with open(RESULTS_DIR / "RESULT-best-depth.yaml", "w") as f:
        yaml.safe_dump(yaml_safe(agg["best_depth"]), f, sort_keys=False)

    if agg["depth_stat"] is not None:
        depth_stat_out = dict(agg["depth_stat"])
        depth_stat_out["n_animals"] = agg["depth_stat_table_n_animals"]
        depth_stat_out["n_sessions"] = N_SESSIONS
        depth_stat_out["animal_session_map"] = dict(ANIMAL_MAP)
        depth_stat_out["note"] = (
            "DEV-002 (resolved 2026-07-25): fit on the 5-animal x 5-depth "
            "(n=25) animal-level table -- NP8's 2 sessions (2021-11-01, "
            "2021-11-03) averaged per depth before this fit, not the raw "
            "6-session table. 'groups' in the LMM is 5 animals, not 6 "
            "sessions."
        )
        if agg["depth_stat_lmm_convergence_note"]:
            depth_stat_out["lmm_convergence_note"] = agg["depth_stat_lmm_convergence_note"]
        with open(RESULTS_DIR / "RESULT-depth-stat.yaml", "w") as f:
            yaml.safe_dump(yaml_safe(depth_stat_out), f, sort_keys=False)
    else:
        with open(RESULTS_DIR / "RESULT-depth-stat_BLOCKED.yaml", "w") as f:
            yaml.safe_dump({"blocked_reason": agg["depth_stat_blocked_reason"],
                           "n_animals_available": agg["depth_stat_table_n_animals"]},
                          f, sort_keys=False)

    grand = agg["grand_spectra"]
    np.savez(
        RESULTS_DIR / "RESULT-grand-mean-spectra.npz",
        freqs=grand["freqs"],
        snf_psd_grand_mean=(grand["snf_psd_grand_mean"]
                             if grand["snf_psd_grand_mean"] is not None
                             else np.array([])),
        **{f"coh_depth{d}_mean": v["mean"] for d, v in grand["coherence_grand_mean_sem"].items()},
        **{f"coh_depth{d}_sem": v["sem"] for d, v in grand["coherence_grand_mean_sem"].items()},
        **{f"psd_depth{d}_mean": v["mean"] for d, v in grand["lfp_psd_grand_mean_sem"].items()},
        **{f"psd_depth{d}_sem": v["sem"] for d, v in grand["lfp_psd_grand_mean_sem"].items()},
        **{f"null_depth{d}_mean": v["mean"] for d, v in grand["null_threshold_grand_mean_sem"].items()},
        **{f"null_depth{d}_sem": v["sem"] for d, v in grand["null_threshold_grand_mean_sem"].items()},
    )

    traces = agg["example_traces"]
    traces_npz = {
        "t_s": traces["t_s"],
        "snf_segment": traces["snf_segment"],
        "inhalation_mask_lv_rate": traces["inhalation_mask_lv_rate"],
        "sniff_onset_markers_s": traces["sniff_onset_markers_s"],
        "depth_channel_indices": np.asarray(traces["depth_channel_indices"]),
        "depth_ycoords_um": np.asarray(traces["depth_ycoords_um"]),
    }
    for i, row in enumerate(traces["per_depth_bands"], start=1):
        for band_name, arr in row.items():
            traces_npz[f"depth{i}_{band_name}"] = arr
    np.savez(RESULTS_DIR / "RESULT-example-traces.npz", **traces_npz)
    with open(RESULTS_DIR / "RESULT-example-traces_meta.yaml", "w") as f:
        yaml.safe_dump(yaml_safe({
            "session": traces["session"],
            "segment_start_s": traces["segment_start_s"],
            "segment_end_s": traces["segment_end_s"],
            "inhalation_mask_lv_fs": traces["inhalation_mask_lv_fs"],
            "display_convention_note": traces["display_convention_note"],
        }), f, sort_keys=False)

    print("\n================ NP-PAPER-1 SUMMARY ================")
    for s in summaries:
        print(f"  {s['session_dirname']}: binfile={s['binfile_chosen']} "
              f"(dur={s['binfile_duration_min']:.2f}min, ratio={s['binfile_ratio']:.3f}"
              f"{' WARN' if s['binfile_warn'] else ''}); "
              f"{s['n_windows_used']} windows used, "
              f"{'EXCLUDED' if s['excluded_from_depth_stats'] else 'included'} "
              f"from depth stats; length={s['length_min']:.1f} min")
    if excluded_sessions:
        print(f"\nSessions excluded via a NEW deviation (DEV-004+, see "
              f"08_traceability/deviations.yaml): {excluded_sessions}")
    print(f"\nDEV-002 (resolved): {agg['best_depth']['n_animals']} animals, "
          f"{agg['best_depth']['n_sessions']} sessions -- {ANIMAL_MAP}")
    print(f"Overall best depth (grand-mean theta coh, {agg['best_depth']['n_animals']} animals): "
          f"D{agg['best_depth']['overall_best_depth_ordinal']}")
    print(f"Per-ordinal consistency counts (out of {agg['best_depth']['n_animals']}): "
          f"{agg['best_depth']['per_ordinal_consistency_counts']}")
    if agg["depth_stat"] is not None:
        print(f"Depth-stat LMM (n_animals={agg['depth_stat_table_n_animals']}): "
              f"chisq={agg['depth_stat']['lmm']['chisq']:.4f}, "
              f"p={agg['depth_stat']['lmm']['pvalue']:.4g}")
        print(f"Depth-stat Friedman: stat={agg['depth_stat']['friedman']['statistic']:.4f}, "
              f"p={agg['depth_stat']['friedman']['pvalue']:.4g}, "
              f"Kendall's W={agg['depth_stat']['friedman']['kendalls_w']:.4f}")
        if agg["depth_stat_lmm_convergence_note"]:
            print(f"*** {agg['depth_stat_lmm_convergence_note']}")
    else:
        print(f"Depth-stat BLOCKED: {agg['depth_stat_blocked_reason']}")
    print("======================================================")


if __name__ == "__main__":
    main()

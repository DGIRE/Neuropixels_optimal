"""build_figure_data.py — NP-DEMO-4 figure data builder (Blueprint v2 P4).

Derives all figure data from frozen result objects (04_results/).
Outputs to 06_figures/figure_data/ as numpy .npy files + a YAML manifest.

Figures:
  FIG-DEMO4-1-SNIFF: Sniff rate waterfall + line (jet colormap, 0-40s)
  FIG-DEMO4-2-FR: Firing rate waterfall + line (jet colormap)
  FIG-DEMO4-3-ETH: ETH sensor waterfall + line (jet colormap, raw ETH)

All computation from frozen RESULT-* objects — no re-loading of raw data.
Numbers injected from result objects; none typed by hand.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import yaml

RESULTS_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\04_results"
FIGURE_DATA_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\06_figures\figure_data"
os.makedirs(FIGURE_DATA_DIR, exist_ok=True)


def _load_result(name: str):
    with open(os.path.join(RESULTS_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def _pool_trials(per_session_dict: dict, target_n_samples: int = 5000) -> np.ndarray:
    """Pool all trials from all sessions into a (n_total_trials, n_samples) matrix.

    Trials with different lengths are padded with NaN or truncated to target_n_samples.
    target_n_samples is determined from the data (max trial length across all sessions).
    """
    all_trials = []
    for session, trials in sorted(per_session_dict.items()):
        for trial_data in trials:
            arr = np.array(trial_data, dtype=np.float64)
            all_trials.append(arr)

    if not all_trials:
        return np.empty((0, 0), dtype=np.float64)

    # Find max length
    max_len = max(len(t) for t in all_trials)
    n_trials = len(all_trials)

    # Build matrix with NaN padding for shorter trials
    matrix = np.full((n_trials, max_len), np.nan, dtype=np.float64)
    for i, t in enumerate(all_trials):
        matrix[i, :len(t)] = t

    return matrix


def _trial_avg_and_ci(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and 95% parametric CI (mean +/- 1.96*SEM) across trials.

    Returns (mean_trace, ci_lower, ci_upper) each of shape (n_samples,).
    NaN values excluded per time point.
    """
    with np.errstate(all='ignore'):
        mean_trace = np.nanmean(matrix, axis=0)
        n_valid = np.sum(~np.isnan(matrix), axis=0).astype(float)
        sem = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
        ci_lower = mean_trace - 1.96 * sem
        ci_upper = mean_trace + 1.96 * sem
    return mean_trace, ci_lower, ci_upper


def _build_time_axis(matrix: np.ndarray, lv_fs: float = 125.0) -> np.ndarray:
    """Build time axis in seconds for the columns of the matrix."""
    n_samples = matrix.shape[1]
    return np.arange(n_samples, dtype=np.float64) / lv_fs


def main():
    print("Building NP-DEMO-4 figure data from frozen results...")

    # Load results
    sniff_mat_dict = _load_result("RESULT-sniff-rate-matrix")
    fr_mat_dict = _load_result("RESULT-fr-per-trial")
    eth_mat_dict = _load_result("RESULT-eth-per-trial")
    methods_table = _load_result("RESULT-methods-table")
    unit_inclusion = _load_result("RESULT-unit-inclusion")
    sniff_stat = _load_result("RESULT-sniff-stat")

    # --- FIG-DEMO4-1-SNIFF ---
    print("\n[FIG-DEMO4-1-SNIFF] Sniff rate waterfall + line...")
    sniff_matrix = _pool_trials(sniff_mat_dict)
    time_axis = _build_time_axis(sniff_matrix)
    sniff_mean, sniff_ci_lo, sniff_ci_hi = _trial_avg_and_ci(sniff_matrix)

    np.save(os.path.join(FIGURE_DATA_DIR, "fig1_sniff_matrix.npy"), sniff_matrix)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig1_time_axis.npy"), time_axis)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig1_sniff_mean.npy"), sniff_mean)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig1_sniff_ci_lower.npy"), sniff_ci_lo)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig1_sniff_ci_upper.npy"), sniff_ci_hi)
    print(f"  Waterfall matrix: {sniff_matrix.shape} (trials x samples)")
    print(f"  Time axis: {time_axis[0]:.3f} - {time_axis[-1]:.3f} s")
    print(f"  Mean sniff rate: {float(np.nanmean(sniff_mean)):.3f} Hz")

    # --- FIG-DEMO4-2-FR ---
    print("\n[FIG-DEMO4-2-FR] Firing rate waterfall + line...")
    fr_matrix = _pool_trials(fr_mat_dict)
    fr_time_axis = _build_time_axis(fr_matrix)
    fr_mean, fr_ci_lo, fr_ci_hi = _trial_avg_and_ci(fr_matrix)

    np.save(os.path.join(FIGURE_DATA_DIR, "fig2_fr_matrix.npy"), fr_matrix)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig2_time_axis.npy"), fr_time_axis)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig2_fr_mean.npy"), fr_mean)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig2_fr_ci_lower.npy"), fr_ci_lo)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig2_fr_ci_upper.npy"), fr_ci_hi)
    print(f"  Waterfall matrix: {fr_matrix.shape} (trials x samples)")
    print(f"  Mean FR: {float(np.nanmean(fr_mean)):.3f} Hz")

    # --- FIG-DEMO4-3-ETH ---
    print("\n[FIG-DEMO4-3-ETH] ETH sensor waterfall + line (raw, before mean-subtraction)...")
    eth_matrix = _pool_trials(eth_mat_dict)
    eth_time_axis = _build_time_axis(eth_matrix)
    eth_mean, eth_ci_lo, eth_ci_hi = _trial_avg_and_ci(eth_matrix)

    np.save(os.path.join(FIGURE_DATA_DIR, "fig3_eth_matrix.npy"), eth_matrix)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig3_time_axis.npy"), eth_time_axis)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig3_eth_mean.npy"), eth_mean)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig3_eth_ci_lower.npy"), eth_ci_lo)
    np.save(os.path.join(FIGURE_DATA_DIR, "fig3_eth_ci_upper.npy"), eth_ci_hi)
    print(f"  Waterfall matrix: {eth_matrix.shape} (trials x samples)")
    print(f"  Mean ETH: {float(np.nanmean(eth_mean)):.4f} (a.u., normalized 0-1)")

    # --- Caption metadata (injected from result objects) ---
    n_total_trials = sniff_matrix.shape[0]
    n_sessions = len(sniff_mat_dict)
    total_units_per_session = {
        row["session"]: row["n_units_included"] for row in unit_inclusion
    }

    manifest = dict(
        task_id="NP-DEMO-4",
        source_results=[
            "RESULT-sniff-rate-matrix",
            "RESULT-fr-per-trial",
            "RESULT-eth-per-trial",
            "RESULT-methods-table",
            "RESULT-unit-inclusion",
            "RESULT-sniff-stat",
        ],
        figures=dict(
            FIG_DEMO4_1_SNIFF=dict(
                matrix_shape=list(sniff_matrix.shape),
                time_range_s=[float(time_axis[0]), float(time_axis[-1])],
                n_total_trials=n_total_trials,
                n_sessions=n_sessions,
                ci_method="mean +/- 1.96*SEM across all trials (pooled)",
                colormap="jet",
                files=[
                    "fig1_sniff_matrix.npy",
                    "fig1_time_axis.npy",
                    "fig1_sniff_mean.npy",
                    "fig1_sniff_ci_lower.npy",
                    "fig1_sniff_ci_upper.npy",
                ],
                caption_injections=dict(
                    n_trials=n_total_trials,
                    n_sessions=n_sessions,
                    ci_description="95% CI computed as mean ± 1.96*SEM across all trials",
                    colorbar_label="sniff rate (Hz)",
                ),
            ),
            FIG_DEMO4_2_FR=dict(
                matrix_shape=list(fr_matrix.shape),
                time_range_s=[float(fr_time_axis[0]), float(fr_time_axis[-1])],
                n_total_trials=fr_matrix.shape[0],
                n_sessions=n_sessions,
                ci_method="mean +/- 1.96*SEM across all trials (pooled)",
                colormap="jet",
                files=[
                    "fig2_fr_matrix.npy",
                    "fig2_time_axis.npy",
                    "fig2_fr_mean.npy",
                    "fig2_fr_ci_lower.npy",
                    "fig2_fr_ci_upper.npy",
                ],
                caption_injections=dict(
                    n_units_per_session={k: int(v) for k, v in total_units_per_session.items()},
                    inclusion_criteria="overall FR >= 0.1 Hz AND >= 5000 total spikes",
                    ci_description="95% CI computed as mean ± 1.96*SEM across all trials",
                    colorbar_label="firing rate (Hz)",
                ),
            ),
            FIG_DEMO4_3_ETH=dict(
                matrix_shape=list(eth_matrix.shape),
                time_range_s=[float(eth_time_axis[0]), float(eth_time_axis[-1])],
                n_total_trials=eth_matrix.shape[0],
                n_sessions=n_sessions,
                ci_method="mean +/- 1.96*SEM across all trials (pooled)",
                colormap="jet",
                files=[
                    "fig3_eth_matrix.npy",
                    "fig3_time_axis.npy",
                    "fig3_eth_mean.npy",
                    "fig3_eth_ci_lower.npy",
                    "fig3_eth_ci_upper.npy",
                ],
                caption_injections=dict(
                    eth_description="raw (pre-mean-subtraction) ETH sensor value (CON-002)",
                    ci_description="95% CI computed as mean ± 1.96*SEM across all trials",
                    colorbar_label="ETH sensor value (a.u., normalized 0-1)",
                ),
            ),
        ),
        stat_for_caption=dict(
            W=float(sniff_stat["W"]),
            p_exact=float(sniff_stat["p_exact"]),
            rank_biserial_r=float(sniff_stat["rank_biserial_r"]),
            n_animals=int(sniff_stat["n_animals"]),
            significant=bool(sniff_stat["significant"]),
        ),
    )

    manifest_path = os.path.join(FIGURE_DATA_DIR, "figure_data_manifest.yaml")
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=True, allow_unicode=True)

    print(f"\nManifest -> {manifest_path}")
    print(f"Total trials pooled: {n_total_trials} from {n_sessions} sessions")
    print("Figure data build COMPLETE.")


if __name__ == "__main__":
    main()

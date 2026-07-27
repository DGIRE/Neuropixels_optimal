"""Baseline-corrected waterfall figures for NP-DEMO-4.

For each of the three signals (ETH, FR, sniff), subtracts the per-trial mean
of the first 5 seconds (baseline window) from every sample in that trial,
then renders the same two-panel waterfall layout used in the main figures.

Output: NP-DEMO-4/06_figures/revised figures/
  ETH_waterfall_baseline.png
  FR_waterfall_baseline.png
  SNIFF_waterfall_baseline.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DATA = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\06_figures\figure_data"
OUT_DIR  = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\06_figures\revised figures"
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE_END_S = 5.0   # subtract mean of first 5 s per trial
MIN_FONT = 8
DPI = 150
COLORMAP = "jet"


def baseline_subtract(matrix: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
    """Subtract per-trial mean of t in [0, BASELINE_END_S] from every sample."""
    bl_mask = time_axis <= BASELINE_END_S
    bl_mean = np.nanmean(matrix[:, bl_mask], axis=1, keepdims=True)  # (n_trials, 1)
    return matrix - bl_mean


def render(
    *,
    matrix: np.ndarray,
    time_axis: np.ndarray,
    title: str,
    colorbar_label: str,
    panel2_ylabel: str,
    out_path: str,
    vmin=None,
    vmax=None,
) -> None:
    n_trials = matrix.shape[0]
    mean_tr = np.nanmean(matrix, axis=0)
    sem_tr  = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(n_trials)
    ci_lo   = mean_tr - 1.96 * sem_tr
    ci_hi   = mean_tr + 1.96 * sem_tr

    t0, t1 = time_axis[0], time_axis[-1]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.45},
    )

    im = ax1.imshow(
        matrix,
        aspect="auto", origin="lower",
        extent=[t0, t1, 0.5, n_trials + 0.5],
        cmap=COLORMAP, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=MIN_FONT)
    cbar.ax.tick_params(labelsize=MIN_FONT)
    ax1.set_xlabel("time from trial start (s)", fontsize=MIN_FONT)
    ax1.set_ylabel("trial number (LabView)", fontsize=MIN_FONT)
    ax1.set_xlim(t0, t1)
    ax1.set_ylim(0.5, n_trials + 0.5)
    ax1.tick_params(labelsize=MIN_FONT)
    ax1.set_title(title, fontsize=MIN_FONT + 1, fontweight="bold")
    ax1.axvline(BASELINE_END_S, color="white", linewidth=0.9, linestyle="--",
                alpha=0.8, label=f"baseline end ({BASELINE_END_S:.0f} s)")
    ax1.axvline(10, color="white", linewidth=0.8, linestyle="--", alpha=0.6,
                label="valve open (10 s)")
    ax1.axvline(20, color="white", linewidth=0.8, linestyle=":",  alpha=0.6,
                label="valve close (20 s)")
    ax1.legend(fontsize=6, loc="upper right", framealpha=0.6)

    ax2.plot(time_axis, mean_tr, color="black", linewidth=1.2, label="mean")
    ax2.fill_between(time_axis, ci_lo, ci_hi, color="steelblue", alpha=0.35,
                     label="95% CI (±1.96 SEM)")
    ax2.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax2.axvline(BASELINE_END_S, color="gray", linewidth=0.9, linestyle="--", alpha=0.7)
    ax2.axvline(10, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.axvline(20, color="gray", linewidth=0.8, linestyle=":",  alpha=0.5)
    ax2.set_xlabel("time from trial start (s)", fontsize=MIN_FONT)
    ax2.set_ylabel(panel2_ylabel, fontsize=MIN_FONT)
    ax2.set_xlim(t0, t1)
    ax2.tick_params(labelsize=MIN_FONT)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, linewidth=0.4, alpha=0.4)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def load(name: str) -> np.ndarray:
    return np.load(os.path.join(FIG_DATA, name))


# ── ETH ──────────────────────────────────────────────────────────────────────
matrix_e = load("fig3_eth_matrix.npy")
time_e   = load("fig3_time_axis.npy")
bl_e     = baseline_subtract(matrix_e, time_e)
abs_max_e = float(np.nanpercentile(np.abs(bl_e), 99))

render(
    matrix=bl_e,
    time_axis=time_e,
    title=(f"Ethanol sensor — baseline corrected (mean 0–{BASELINE_END_S:.0f} s subtracted)"),
    colorbar_label="ΔETH sensor value (a.u.)",
    panel2_ylabel="ΔETH sensor value (a.u.)",
    out_path=os.path.join(OUT_DIR, "ETH_waterfall_baseline.png"),
    vmin=-abs_max_e,
    vmax=abs_max_e,
)

# ── Firing Rate ───────────────────────────────────────────────────────────────
matrix_f = load("fig2_fr_matrix.npy")
time_f   = load("fig2_time_axis.npy")
bl_f     = baseline_subtract(matrix_f, time_f)
abs_max_f = float(np.nanpercentile(np.abs(bl_f), 99))

render(
    matrix=bl_f,
    time_axis=time_f,
    title=(f"Firing rate — baseline corrected (mean 0–{BASELINE_END_S:.0f} s subtracted)"),
    colorbar_label="ΔFiring rate (Hz)",
    panel2_ylabel="ΔFiring rate (Hz)",
    out_path=os.path.join(OUT_DIR, "FR_waterfall_baseline.png"),
    vmin=-abs_max_f,
    vmax=abs_max_f,
)

# ── Sniff Rate ────────────────────────────────────────────────────────────────
matrix_s = load("fig1_sniff_matrix.npy")
time_s   = load("fig1_time_axis.npy")
bl_s     = baseline_subtract(matrix_s, time_s)
abs_max_s = float(np.nanpercentile(np.abs(bl_s), 99))

render(
    matrix=bl_s,
    time_axis=time_s,
    title=(f"Sniff rate — baseline corrected (mean 0–{BASELINE_END_S:.0f} s subtracted)"),
    colorbar_label="ΔSniff rate (Hz)",
    panel2_ylabel="ΔSniff rate (Hz)",
    out_path=os.path.join(OUT_DIR, "SNIFF_waterfall_baseline.png"),
    vmin=-abs_max_s,
    vmax=abs_max_s,
)

print("\nDone. All three baseline-corrected figures written to:")
print(f"  {OUT_DIR}")

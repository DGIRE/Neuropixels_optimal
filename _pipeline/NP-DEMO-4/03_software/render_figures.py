"""render_figures.py -- NP-DEMO-4 pure figure renderer (Blueprint v2 P4, D11).

Reads ONLY from 06_figures/figure_data/ (frozen by the data builder).
Produces SVG + 300 DPI PNG for each of the three contracted figures.
No scientific computation here -- all numbers come from pre-built .npy files
and the manifest.  The renderer is deterministic: re-running from the same
figure-data produces bit-identical output.

Figures:
  FIG-DEMO4-1-SNIFF  -- sniff rate waterfall + trial-average line
  FIG-DEMO4-2-FR     -- firing rate waterfall + trial-average line
  FIG-DEMO4-3-ETH    -- raw ETH sensor waterfall + trial-average line

Contract requirements honoured here:
  * jet colormap for all three heatmaps
  * x-axis: time from trial start (s), 0-40 s
  * Panel 1 y-axis: trial number (LabView)
  * Panel 2 y-axis: signal-specific label
  * Colorbar labeled per contract
  * Font >= 7 pt; vector (SVG) + 300 DPI raster (PNG)
  * All caption numbers injected from manifest (never typed by hand)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml

FIGURE_DATA_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\06_figures\figure_data"
OUTPUT_DIR      = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\06_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_FONT  = 8     # >= 7 pt as per quality spec
DPI       = 300
COLORMAP  = "jet"
X_MAX_S   = 40.0  # clip x-axis display to 0-40 s


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load(name: str) -> np.ndarray:
    return np.load(os.path.join(FIGURE_DATA_DIR, name))


def _load_manifest() -> dict:
    with open(os.path.join(FIGURE_DATA_DIR, "figure_data_manifest.yaml")) as f:
        return yaml.safe_load(f)


def _clip_to_40s(matrix: np.ndarray, time_axis: np.ndarray):
    """Return (matrix[:,mask], time_axis[mask]) clipped to 0-40 s."""
    mask = time_axis <= X_MAX_S
    return matrix[:, mask], time_axis[mask]


def _render_waterfall_fig(
    *,
    matrix: np.ndarray,
    time_axis: np.ndarray,
    mean_trace: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    fig_id: str,
    title: str,
    colorbar_label: str,
    panel2_ylabel: str,
    caption: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Two-panel figure: waterfall heatmap (top) + trial-average line (bottom)."""
    fig, axes = plt.subplots(
        2, 1,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.42},
    )

    n_trials, n_samp = matrix.shape
    t0, t1 = time_axis[0], time_axis[-1]

    # -- Panel 1: waterfall heatmap ------------------------------------------
    ax1 = axes[0]
    im = ax1.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[t0, t1, 0.5, n_trials + 0.5],
        cmap=COLORMAP,
        vmin=vmin,
        vmax=vmax,
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

    # -- Panel 2: trial-averaged line with 95% CI ----------------------------
    ax2 = axes[1]
    ax2.plot(time_axis, mean_trace, color="black", linewidth=1.0, label="mean")
    ax2.fill_between(time_axis, ci_lower, ci_upper,
                     color="steelblue", alpha=0.35, label="95% CI (±1.96 SEM)")
    ax2.set_xlabel("time from trial start (s)", fontsize=MIN_FONT)
    ax2.set_ylabel(panel2_ylabel, fontsize=MIN_FONT)
    ax2.set_xlim(t0, t1)
    ax2.tick_params(labelsize=MIN_FONT)
    ax2.legend(fontsize=MIN_FONT - 1, loc="upper right")
    ax2.grid(True, linewidth=0.4, alpha=0.4)

    # Caption as figure text (not in a box -- just smallprint below panels)
    fig.text(0.01, 0.01, caption, fontsize=MIN_FONT - 1,
             wrap=True, ha="left", va="bottom",
             transform=fig.transFigure)

    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    mfst = _load_manifest()
    stat = mfst["stat_for_caption"]
    n_trials = mfst["figures"]["FIG_DEMO4_1_SNIFF"]["n_total_trials"]
    n_sess   = mfst["figures"]["FIG_DEMO4_1_SNIFF"]["n_sessions"]
    n_units  = mfst["figures"]["FIG_DEMO4_2_FR"]["caption_injections"]["n_units_per_session"]
    n_units_str = ", ".join(f"{s}: {v}" for s, v in sorted(n_units.items()))

    # -----------------------------------------------------------------------
    # FIG-DEMO4-1-SNIFF
    # -----------------------------------------------------------------------
    matrix_s = _load("fig1_sniff_matrix.npy")
    time_s   = _load("fig1_time_axis.npy")
    mean_s   = _load("fig1_sniff_mean.npy")
    ci_lo_s  = _load("fig1_sniff_ci_lower.npy")
    ci_hi_s  = _load("fig1_sniff_ci_upper.npy")

    matrix_s, time_s_clip = _clip_to_40s(matrix_s, time_s)
    mean_s_c   = mean_s[time_s <= X_MAX_S]
    ci_lo_s_c  = ci_lo_s[time_s <= X_MAX_S]
    ci_hi_s_c  = ci_hi_s[time_s <= X_MAX_S]

    caption_1 = (
        f"Figure 1: Sniff rate within each trial. "
        f"n_trials={n_trials} odd-TR LabView trials from n_sessions={n_sess} sessions. "
        f"Panel 1: jet colormap; colorbar = sniff rate (Hz). "
        f"Panel 2: trial-averaged sniff rate (Hz); shaded band = 95% CI (mean ± 1.96×SEM) "
        f"across all {n_trials} trials (pooled across sessions). "
        f"Source: RESULT-sniff-rate-matrix."
    )
    fig1 = _render_waterfall_fig(
        matrix=matrix_s,
        time_axis=time_s_clip,
        mean_trace=mean_s_c,
        ci_lower=ci_lo_s_c,
        ci_upper=ci_hi_s_c,
        fig_id="FIG-DEMO4-1-SNIFF",
        title="Figure 1: Sniff rate within each trial",
        colorbar_label="sniff rate (Hz)",
        panel2_ylabel="sniff rate (Hz)",
        caption=caption_1,
        vmin=0.0,
        vmax=float(np.nanpercentile(matrix_s, 98)),
    )
    for ext in ("svg", "png"):
        out = os.path.join(OUTPUT_DIR, f"FIG-DEMO4-1-SNIFF.{ext}")
        kw = {"dpi": DPI} if ext == "png" else {}
        fig1.savefig(out, format=ext, bbox_inches="tight", **kw)
        print(f"  Saved {out}")
    plt.close(fig1)

    # -----------------------------------------------------------------------
    # FIG-DEMO4-2-FR
    # -----------------------------------------------------------------------
    matrix_f = _load("fig2_fr_matrix.npy")
    time_f   = _load("fig2_time_axis.npy")
    mean_f   = _load("fig2_fr_mean.npy")
    ci_lo_f  = _load("fig2_fr_ci_lower.npy")
    ci_hi_f  = _load("fig2_fr_ci_upper.npy")

    matrix_f, time_f_clip = _clip_to_40s(matrix_f, time_f)
    mean_f_c  = mean_f[time_f <= X_MAX_S]
    ci_lo_f_c = ci_lo_f[time_f <= X_MAX_S]
    ci_hi_f_c = ci_hi_f[time_f <= X_MAX_S]

    caption_2 = (
        f"Figure 2: Olfactory-bulb population firing rate within each trial. "
        f"n_trials={n_trials} odd-TR LabView trials from n_sessions={n_sess} sessions. "
        f"Inclusion criteria: overall FR ≥ 0.1 Hz AND ≥ 5000 total spikes. "
        f"n_units_included per session: {n_units_str}. "
        f"Panel 1: jet colormap; colorbar = firing rate (Hz). "
        f"Panel 2: trial-averaged firing rate (Hz); shaded band = 95% CI (mean ± 1.96×SEM) "
        f"across all {n_trials} trials (pooled). "
        f"Source: RESULT-fr-per-trial, RESULT-unit-inclusion."
    )
    fig2 = _render_waterfall_fig(
        matrix=matrix_f,
        time_axis=time_f_clip,
        mean_trace=mean_f_c,
        ci_lower=ci_lo_f_c,
        ci_upper=ci_hi_f_c,
        fig_id="FIG-DEMO4-2-FR",
        title="Figure 2: Firing rate within each trial",
        colorbar_label="firing rate (Hz)",
        panel2_ylabel="firing rate (Hz)",
        caption=caption_2,
        vmin=0.0,
        vmax=float(np.nanpercentile(matrix_f, 98)),
    )
    for ext in ("svg", "png"):
        out = os.path.join(OUTPUT_DIR, f"FIG-DEMO4-2-FR.{ext}")
        kw = {"dpi": DPI} if ext == "png" else {}
        fig2.savefig(out, format=ext, bbox_inches="tight", **kw)
        print(f"  Saved {out}")
    plt.close(fig2)

    # -----------------------------------------------------------------------
    # FIG-DEMO4-3-ETH
    # -----------------------------------------------------------------------
    matrix_e = _load("fig3_eth_matrix.npy")
    time_e   = _load("fig3_time_axis.npy")
    mean_e   = _load("fig3_eth_mean.npy")
    ci_lo_e  = _load("fig3_eth_ci_lower.npy")
    ci_hi_e  = _load("fig3_eth_ci_upper.npy")

    matrix_e, time_e_clip = _clip_to_40s(matrix_e, time_e)
    mean_e_c  = mean_e[time_e <= X_MAX_S]
    ci_lo_e_c = ci_lo_e[time_e <= X_MAX_S]
    ci_hi_e_c = ci_hi_e[time_e <= X_MAX_S]

    caption_3 = (
        f"Figure 3: Ethanol sensor (raw, normalized 0–1) within each trial. "
        f"Raw ETH sensor value (D['ETH'] before mean-subtraction, per CON-002) "
        f"for n_trials={n_trials} odd-TR LabView trials from n_sessions={n_sess} sessions. "
        f"Panel 1: jet colormap; colorbar = ETH sensor value (a.u., normalized 0–1). "
        f"Panel 2: trial-averaged ETH; shaded band = 95% CI (mean ± 1.96×SEM) "
        f"across all {n_trials} trials (pooled). "
        f"Source: RESULT-eth-per-trial."
    )
    fig3 = _render_waterfall_fig(
        matrix=matrix_e,
        time_axis=time_e_clip,
        mean_trace=mean_e_c,
        ci_lower=ci_lo_e_c,
        ci_upper=ci_hi_e_c,
        fig_id="FIG-DEMO4-3-ETH",
        title="Figure 3: Ethanol sensor value within each trial",
        colorbar_label="ETH sensor value (a.u., normalized 0-1)",
        panel2_ylabel="ETH sensor value (a.u.)",
        caption=caption_3,
        vmin=float(np.nanmin(matrix_e)),
        vmax=float(np.nanpercentile(matrix_e, 99)),
    )
    for ext in ("svg", "png"):
        out = os.path.join(OUTPUT_DIR, f"FIG-DEMO4-3-ETH.{ext}")
        kw = {"dpi": DPI} if ext == "png" else {}
        fig3.savefig(out, format=ext, bbox_inches="tight", **kw)
        print(f"  Saved {out}")
    plt.close(fig3)

    print("\nAll three figures rendered successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

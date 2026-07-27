"""Quick ETH waterfall plot for NP-DEMO-4.

Reads the pre-built figure-data .npy files (produced by build_figure_data.py)
and renders a two-panel ETH waterfall figure:
  Panel 1 - heatmap: trial x time, ETH sensor value, jet colormap
  Panel 2 - trial-averaged line with 95% CI shading

Output: 06_figures/ETH_waterfall_quick.png  (and displayed if a GUI is available)
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

FIG_DATA = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\06_figures\figure_data"
OUT_DIR  = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\06_figures"

matrix  = np.load(os.path.join(FIG_DATA, "fig3_eth_matrix.npy"))
t       = np.load(os.path.join(FIG_DATA, "fig3_time_axis.npy"))
mean_tr = np.load(os.path.join(FIG_DATA, "fig3_eth_mean.npy"))
ci_lo   = np.load(os.path.join(FIG_DATA, "fig3_eth_ci_lower.npy"))
ci_hi   = np.load(os.path.join(FIG_DATA, "fig3_eth_ci_upper.npy"))

n_trials = matrix.shape[0]
vmin = float(np.nanmin(matrix))
vmax = float(np.nanpercentile(matrix, 99))

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 7),
    gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.45}
)

im = ax1.imshow(
    matrix,
    aspect="auto", origin="lower",
    extent=[t[0], t[-1], 0.5, n_trials + 0.5],
    cmap="jet", vmin=vmin, vmax=vmax,
    interpolation="nearest",
)
cbar = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
cbar.set_label("ETH sensor value (a.u., normalized 0-1)", fontsize=8)
ax1.set_xlabel("time from trial start (s)", fontsize=8)
ax1.set_ylabel("trial number (LabView)", fontsize=8)
ax1.set_title(
    f"Figure 3 – Ethanol sensor within each trial  "
    f"(n={n_trials} trials, 6 sessions)",
    fontsize=9, fontweight="bold"
)
ax1.axvline(10, color="white", linewidth=0.8, linestyle="--", alpha=0.7,
            label="valve open (10 s)")
ax1.axvline(20, color="white", linewidth=0.8, linestyle=":",  alpha=0.7,
            label="valve close (20 s)")
ax1.legend(fontsize=7, loc="upper right", framealpha=0.6)

ax2.plot(t, mean_tr, color="black", linewidth=1.2, label="mean")
ax2.fill_between(t, ci_lo, ci_hi, color="steelblue", alpha=0.35,
                 label="95% CI (±1.96 SEM)")
ax2.axvline(10, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
ax2.axvline(20, color="gray", linewidth=0.8, linestyle=":",  alpha=0.7)
ax2.set_xlabel("time from trial start (s)", fontsize=8)
ax2.set_ylabel("ETH sensor value (a.u.)", fontsize=8)
ax2.legend(fontsize=7, loc="upper right")
ax2.grid(True, linewidth=0.4, alpha=0.4)

out_path = os.path.join(OUT_DIR, "ETH_waterfall_quick.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

try:
    plt.close("all")
    matplotlib.use("TkAgg")
    plt.show()
except Exception:
    pass

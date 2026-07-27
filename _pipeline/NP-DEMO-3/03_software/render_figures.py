"""
NP-DEMO-3 Figure Renderer.
Reads figure data YAMLs from 06_figures/figure_data/ and produces final figures
in 06_figures/ as PNG (300 dpi) and SVG (vector). All numeric values sourced
exclusively from the figure data YAMLs, never typed by hand.
"""
import sys
import os
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = r"C:\Projects\Repos\Neuropixels"
FIGDATA_DIR = os.path.join(PROJECT_ROOT, r"_pipeline\NP-DEMO-3\06_figures\figure_data")
OUT_DIR = os.path.join(PROJECT_ROOT, r"_pipeline\NP-DEMO-3\06_figures")

COLORBLIND_SAFE = {
    "ethanol": "#E69F00",   # orange
    "control": "#56B4E9",   # sky blue
    "threshold": "#D55E00", # vermillion
    "neutral": "#999999",   # gray
    "highlight": "#CC79A7", # pink
}

MIN_FONT = 7
matplotlib.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": MIN_FONT,
    "ytick.labelsize": MIN_FONT,
    "legend.fontsize": MIN_FONT,
    "figure.dpi": 100,
})


def load_figdata(fig_id):
    path = os.path.join(FIGDATA_DIR, f"{fig_id}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def save_fig(fig, fig_id):
    base = os.path.join(OUT_DIR, fig_id)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_id}.png / .svg")


# ============================================================================
# FIG-DEMO3-QC-SNIFF
# ============================================================================
print("\n--- FIG-DEMO3-QC-SNIFF ---")
data = load_figdata("FIG-DEMO3-QC-SNIFF")
panels = data["panels"]
n_sessions = len(panels)

fig, axes = plt.subplots(n_sessions, 2, figsize=(14, 2.2 * n_sessions), sharex=False)
fig.suptitle("Methods: per-experiment sniffing and ethanol-contact examples", fontsize=10, y=1.01)

for row_i, p in enumerate(panels):
    ax_snf = axes[row_i, 0]
    ax_eth = axes[row_i, 1]

    t = np.array(p["t_axis"])
    snf_z = np.array(p["SNF_z"])
    eth_ms = np.array(p["ETH_ms"])
    sniff_onset_thr = float(p["sniff_onset_threshold"])
    eth_thr = float(p["eth_contact_threshold"])
    onsets = np.array(p["sniff_onsets_in_window_s"])

    # SNF_z panel
    ax_snf.plot(t, snf_z, color=COLORBLIND_SAFE["control"], lw=0.5, label="SNF_z")
    ax_snf.axhline(sniff_onset_thr, color=COLORBLIND_SAFE["threshold"], lw=1.0,
                   ls="--", label=f"threshold {sniff_onset_thr}")
    if onsets.size > 0:
        for ons in onsets:
            ax_snf.axvline(ons, color=COLORBLIND_SAFE["highlight"], lw=0.3, alpha=0.5)
    ax_snf.set_ylabel("SNF_z", fontsize=MIN_FONT)
    ax_snf.set_title(f"{p['experiment']}\nn_sniffs={p['n_sniffs']}, "
                     f"n_neurons={p['n_neurons']}, "
                     f"n_trials={p['n_trials']}, "
                     f"dur={p['duration_min']:.1f} min, "
                     f"usable={p['pct_usable_sniff_time']:.1f}%",
                     fontsize=MIN_FONT)

    # ETH_ms panel
    ax_eth.plot(t, eth_ms, color=COLORBLIND_SAFE["ethanol"], lw=0.5, label="ETH_ms")
    ax_eth.axhline(eth_thr, color=COLORBLIND_SAFE["threshold"], lw=1.0,
                   ls="--", label=f"threshold {eth_thr}")
    ax_eth.set_ylabel("ETH\n(m-sub, a.u.)", fontsize=MIN_FONT)

    for ax in [ax_snf, ax_eth]:
        ax.set_xlabel("time (s)", fontsize=MIN_FONT)
        ax.tick_params(labelsize=MIN_FONT)

fig.tight_layout()
save_fig(fig, "FIG-DEMO3-QC-SNIFF")


# ============================================================================
# FIG-DEMO3-QC-PSTH
# ============================================================================
print("\n--- FIG-DEMO3-QC-PSTH ---")
data = load_figdata("FIG-DEMO3-QC-PSTH")
panels = data["panels"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Methods: sniff-locked PSTH examples (strongest and weakest phase-locking)",
             fontsize=10)

label_styles = {
    "strongest": dict(color=COLORBLIND_SAFE["ethanol"], ls="-"),
    "weakest": dict(color=COLORBLIND_SAFE["control"], ls="--"),
}

for i, p in enumerate(panels):
    ax = axes[i]
    centers = np.array(p["centers_phase"])
    psth = np.array(p["psth"])
    style = label_styles.get(p["label"], {})
    ax.plot(centers, psth, lw=1.5, **style, label=p["label"])
    ax.set_title(
        f"{p['label'].capitalize()}: unit {p['unit_id']} ({p['experiment']})\n"
        f"MRL_cond_agnostic = {p['MRL_cond_agnostic']:.4f}  |  n_events = {p['n_events']}",
        fontsize=MIN_FONT)
    ax.set_xlabel("sniff phase (cycle fraction)", fontsize=MIN_FONT)
    ax.set_ylabel("spike rate (Hz)", fontsize=MIN_FONT)
    ax.tick_params(labelsize=MIN_FONT)
    ax.set_xlim([0, 1])
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=MIN_FONT)

fig.tight_layout()
save_fig(fig, "FIG-DEMO3-QC-PSTH")


# ============================================================================
# FIG-DEMO3-RESULTS-ANIMAL
# ============================================================================
print("\n--- FIG-DEMO3-RESULTS-ANIMAL ---")
data = load_figdata("FIG-DEMO3-RESULTS-ANIMAL")
pairs = data["pairs"]
n_exp = int(data["n_experiments"])
wilcoxon_p = float(data["wilcoxon_p"])
wilcoxon_stat = float(data["wilcoxon_statistic"])
rr = float(data["rank_biserial_r"])

ctrl_vals = np.array([p["mean_MRL_control"] for p in pairs])
eth_vals = np.array([p["mean_MRL_ethanol"] for p in pairs])

fig, ax = plt.subplots(figsize=(4.5, 5))
x_ctrl = 0.0
x_eth = 1.0
jitter = 0.04

for i, (c, e) in enumerate(zip(ctrl_vals, eth_vals)):
    ax.plot([x_ctrl, x_eth], [c, e], color=COLORBLIND_SAFE["neutral"],
            lw=0.8, alpha=0.7, zorder=1)
    # control: filled marker
    ax.scatter(x_ctrl + (np.random.default_rng(i).random() - 0.5) * jitter,
               c, color=COLORBLIND_SAFE["control"], s=40, zorder=2, marker="o")
    # ethanol: open (hollow) marker per contract
    ax.scatter(x_eth + (np.random.default_rng(i + 100).random() - 0.5) * jitter,
               e, facecolors="none", edgecolors=COLORBLIND_SAFE["ethanol"],
               s=60, linewidths=1.5, zorder=2, marker="^")

# Means: control filled, ethanol open
ax.scatter(x_ctrl, np.mean(ctrl_vals), color=COLORBLIND_SAFE["control"],
           s=100, zorder=3, marker="D", edgecolors="k", linewidths=1.0, label="control mean")
ax.scatter(x_eth, np.mean(eth_vals), facecolors="none",
           edgecolors=COLORBLIND_SAFE["ethanol"], s=100, zorder=3, marker="D",
           linewidths=1.5, label="ethanol mean")

ax.set_xticks([x_ctrl, x_eth])
ax.set_xticklabels(["control", "ethanol"], fontsize=8)
ax.set_ylabel("mean per-unit MRL", fontsize=8)
ax.set_title(
    f"Experiment-level MRL comparison (ethanol vs control)\n"
    f"n={n_exp} experiments, Wilcoxon W={wilcoxon_stat:.0f}, "
    f"p={wilcoxon_p:.4f}, r={rr:.4f}",
    fontsize=MIN_FONT)
ax.set_xlim([-0.3, 1.3])
ax.set_ylim(bottom=0)  # zero_reference required by contract
ax.tick_params(labelsize=MIN_FONT)
ax.legend(fontsize=MIN_FONT, loc="upper right")

fig.tight_layout()
save_fig(fig, "FIG-DEMO3-RESULTS-ANIMAL")


# ============================================================================
# FIG-DEMO3-RESULTS-UNIT
# ============================================================================
print("\n--- FIG-DEMO3-RESULTS-UNIT ---")
data = load_figdata("FIG-DEMO3-RESULTS-UNIT")
unit_points = data["unit_points"]
n_units = int(data["n_units"])
n_experiments = int(data["n_experiments"])
lme_p = float(data["lme_p"])
std_eff = float(data["standardized_fixed_effect"])

# Only plot units with both MRL values finite — this matches the LME n_units=1601
both_finite = [
    u for u in unit_points
    if u.get("MRL_control") is not None and u.get("MRL_ethanol") is not None
    and np.isfinite(float(u["MRL_control"])) and np.isfinite(float(u["MRL_ethanol"]))
]
ctrl_mrl = np.array([u["MRL_control"] for u in both_finite])
eth_mrl = np.array([u["MRL_ethanol"] for u in both_finite])
n_scatter = len(ctrl_mrl)  # should equal n_units=1601

rng = np.random.default_rng(42)
x_ctrl_jitter = rng.standard_normal(n_scatter) * 0.04
x_eth_jitter = rng.standard_normal(n_scatter) * 0.04

fig, ax = plt.subplots(figsize=(4.5, 5))

# control: filled markers
ax.scatter(0 + x_ctrl_jitter, ctrl_mrl, color=COLORBLIND_SAFE["control"],
           s=2, alpha=0.3, zorder=2, label="control")
# ethanol: open (hollow) markers per contract
ax.scatter(1 + x_eth_jitter, eth_mrl, facecolors="none",
           edgecolors=COLORBLIND_SAFE["ethanol"],
           s=4, alpha=0.4, linewidths=0.5, zorder=2, label="ethanol")

# Means ± SEM — per-group N to match scatter
ax.errorbar(0, np.mean(ctrl_mrl), yerr=np.std(ctrl_mrl, ddof=1) / np.sqrt(n_scatter),
            fmt="D", color=COLORBLIND_SAFE["control"], ms=8, capsize=4, zorder=3,
            elinewidth=1.5, markeredgecolor="k", markeredgewidth=0.8)
ax.errorbar(1, np.mean(eth_mrl), yerr=np.std(eth_mrl, ddof=1) / np.sqrt(n_scatter),
            fmt="^", mfc="none", mec=COLORBLIND_SAFE["ethanol"],
            color=COLORBLIND_SAFE["ethanol"],
            ms=8, capsize=4, zorder=3, elinewidth=1.5, markeredgewidth=1.5)

ax.set_xticks([0, 1])
ax.set_xticklabels(["control", "ethanol"], fontsize=8)
ax.set_ylabel("per-unit MRL", fontsize=8)
ax.set_title(
    f"Unit-level MRL comparison (LME)\n"
    f"n={n_units} units, {n_experiments} experiments, "
    f"p={lme_p:.2e}, std_eff={std_eff:.4f}",
    fontsize=MIN_FONT)
ax.set_xlim([-0.5, 1.5])
ax.tick_params(labelsize=MIN_FONT)
ax.legend(fontsize=MIN_FONT, loc="upper right")

fig.tight_layout()
save_fig(fig, "FIG-DEMO3-RESULTS-UNIT")


# ============================================================================
# FIG-DEMO3-EXAMPLES
# ============================================================================
print("\n--- FIG-DEMO3-EXAMPLES ---")
data = load_figdata("FIG-DEMO3-EXAMPLES")
panels = data["panels"]
n_units = len(panels)

# 5 rows x 4 columns: [PSTH | probe_map | depth_hist | FR_hist]
fig = plt.figure(figsize=(14, 3.2 * n_units))
fig.suptitle("Results: top-5 units by |delta_MRL| — sniff-locked PSTHs and probe maps",
             fontsize=10, y=1.01)

for row_i, p in enumerate(panels):
    uid = int(p["unit_id"])
    exp = str(p["experiment"])
    delta = float(p["delta_MRL"])
    mrl_eth = float(p["MRL_ethanol"])
    mrl_ctrl = float(p["MRL_control"])
    n_eth = int(p["n_valid_spikes_eth"])
    n_ctrl = int(p["n_valid_spikes_ctrl"])

    psth_eth = np.array(p["psth_ethanol"])
    psth_ctrl = np.array(p["psth_control"])
    centers = np.array(p["centers_phase"])

    depths = np.array(p["all_unit_depths"])
    amps = np.array(p["all_unit_amps"])
    frs = np.array(p["all_unit_frs"])
    xcoords = np.array(p["xcoords"])
    ycoords = np.array(p["ycoords"])
    unit_depth = float(p["probe_unit_depth_um"])
    unit_amp = float(p["probe_unit_amp"])
    unit_fr = float(p["probe_unit_fr"])

    # GridSpec for this row
    gs_row = GridSpec(1, 4, figure=fig,
                      left=0.06, right=0.97,
                      bottom=1.0 - (row_i + 1) / n_units + 0.01,
                      top=1.0 - row_i / n_units - 0.01,
                      wspace=0.4)

    # (a) PSTH
    ax_psth = fig.add_subplot(gs_row[0, 0])
    ax_psth.plot(centers, psth_eth, color=COLORBLIND_SAFE["ethanol"],
                 lw=1.2, ls="-", label=f"ETH (n={n_eth})")
    ax_psth.plot(centers, psth_ctrl, color=COLORBLIND_SAFE["control"],
                 lw=1.2, ls="--", label=f"CTRL (n={n_ctrl})")
    ax_psth.axhline(0, color="k", lw=0.5)
    ax_psth.set_xlabel("sniff phase", fontsize=MIN_FONT)
    ax_psth.set_ylabel("rate (Hz)", fontsize=MIN_FONT)
    ax_psth.set_xlim([0, 1])
    ax_psth.tick_params(labelsize=MIN_FONT)
    ax_psth.legend(fontsize=6, loc="upper right")
    ax_psth.set_title(
        f"uid={uid} ({exp})\n"
        f"δMRL={delta:.4f}  ETH={mrl_eth:.3f}  CTRL={mrl_ctrl:.3f}",
        fontsize=MIN_FONT)

    # (b) Probe map: depth vs x-position of electrode sites, units as colored dots
    ax_probe = fig.add_subplot(gs_row[0, 1])
    # background electrode sites
    if xcoords.size > 0:
        ax_probe.scatter(xcoords, ycoords, color=COLORBLIND_SAFE["neutral"],
                         s=2, alpha=0.3, zorder=1)
    # all units colored by FR
    sc = ax_probe.scatter(
        np.zeros(len(depths)),  # x=0 for all units on single shank
        depths,
        c=frs, cmap="viridis", s=np.clip(amps / 5.0, 4, 60),
        alpha=0.6, zorder=2, vmin=0, vmax=np.nanpercentile(frs, 95))
    # highlight target unit
    ax_probe.scatter(0, unit_depth, c=[unit_fr], cmap="viridis",
                     s=120, edgecolors="red", linewidths=1.5, zorder=3,
                     vmin=0, vmax=np.nanpercentile(frs, 95), marker="*")
    ax_probe.set_xlabel("unit", fontsize=MIN_FONT)
    ax_probe.set_ylabel("depth (µm)", fontsize=MIN_FONT)
    ax_probe.set_title("Probe map\n(★ = target)", fontsize=MIN_FONT)
    ax_probe.tick_params(labelsize=MIN_FONT)
    ax_probe.set_xticks([])

    # (c) Depth histogram
    ax_dhist = fig.add_subplot(gs_row[0, 2])
    valid_depths = depths[np.isfinite(depths)]
    max_depth = float(valid_depths.max()) if len(valid_depths) else 700.0
    depth_bins = np.arange(0, max(max_depth, 700.0) + 50, 50)
    counts, edges = np.histogram(depths, bins=depth_bins)
    centers_h = 0.5 * (edges[:-1] + edges[1:])
    ax_dhist.barh(centers_h, counts, height=40,
                  color=COLORBLIND_SAFE["neutral"], alpha=0.8)
    ax_dhist.axhline(unit_depth, color="red", lw=1.0, ls="--")
    ax_dhist.set_xlabel("count", fontsize=MIN_FONT)
    ax_dhist.set_ylabel("depth (µm)", fontsize=MIN_FONT)
    ax_dhist.set_title("Depth hist.", fontsize=MIN_FONT)
    ax_dhist.tick_params(labelsize=MIN_FONT)

    # (d) FR histogram
    ax_frhist = fig.add_subplot(gs_row[0, 3])
    fr_bins = np.arange(0, min(frs.max() if len(frs) else 50, 50) + 1, 1)
    ax_frhist.hist(frs, bins=fr_bins, color=COLORBLIND_SAFE["neutral"], alpha=0.8)
    ax_frhist.axvline(np.nanmedian(frs), color="k", lw=1.0, ls="--",
                      label=f"med={np.nanmedian(frs):.1f}")
    ax_frhist.axvline(unit_fr, color="red", lw=1.0, ls="--",
                      label=f"unit={unit_fr:.1f}")
    ax_frhist.set_xlabel("FR (sp/s)", fontsize=MIN_FONT)
    ax_frhist.set_ylabel("count", fontsize=MIN_FONT)
    ax_frhist.set_title("FR dist.", fontsize=MIN_FONT)
    ax_frhist.tick_params(labelsize=MIN_FONT)
    ax_frhist.legend(fontsize=6, loc="upper right")

save_fig(fig, "FIG-DEMO3-EXAMPLES")


print("\n=== All figures rendered ===")
print(f"Output: {OUT_DIR}")

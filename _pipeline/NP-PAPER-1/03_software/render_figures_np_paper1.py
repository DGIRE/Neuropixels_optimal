"""
NP-PAPER-1 -- Blueprint v2 P4 PURE RENDERER (D11).

Renders the 6 contracted figures (FIG-PAPER1-TRACES, -SPECTRO, -SNIFFRATE,
-PSD, -COH, -DEPTH-SUMMARY) using ONLY numbers already present in the frozen,
read-only arrays under 06_figures/figure_data/*.npz.

This module does NOT recompute, re-derive, or touch 04_results/ directly, and
does NOT modify the kernel ("Optimized Python"), the figure_data directory, or
any frozen result. It is an additive script living beside the kernel per
Blueprint v2 D7/D11.

Run:
    "C:\\Projects\\Repos\\Agentic Research\\Research Setup\\research_workflow\\vras\\Scripts\\python.exe" ^
        "C:\\Projects\\Repos\\Neuropixels\\_pipeline\\NP-PAPER-1\\03_software\\render_figures_np_paper1.py"

Outputs (written to 06_figures/, sibling to figure_data/, never inside it):
    fig_paper1_traces.pdf / .png
    fig_paper1_spectro.pdf / .png
    fig_paper1_sniffrate.pdf / .png
    fig_paper1_psd.pdf / .png
    fig_paper1_coh.pdf / .png
    fig_paper1_depth_summary.pdf / .png
    render_manifest.yaml
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Paths (repo-relative under the task; no writes outside 06_figures/)
# ----------------------------------------------------------------------------
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../NP-PAPER-1
FIG_DATA_DIR = os.path.join(TASK_DIR, "06_figures", "figure_data")
OUT_DIR = os.path.join(TASK_DIR, "06_figures")

# ----------------------------------------------------------------------------
# Pinned, contract-fixed display constants (NOT derived statistics; these are
# the contract's own pinned method parameters / band definitions, stated
# verbatim in contract_v001.yaml prohibited_changes / visual_encoding).
# ----------------------------------------------------------------------------
THETA_BAND = (2.0, 12.0)
BETA_BAND = (18.0, 30.0)
GAMMA_BAND = (65.0, 100.0)
MULTITAPER_NOTE = ("multitaper: 4-s windows, 50% overlap, 7 Slepian tapers "
                   "(NW=4), freqs >= 0.25 Hz")

# Colorblind-safe categorical palette (Okabe-Ito) for band shading.
COLOR_THETA = "#56B4E9"  # sky blue
COLOR_BETA = "#E69F00"   # orange
COLOR_GAMMA = "#CC79A7"  # reddish purple

N_DEPTHS = 5
DEPTH_COLORS = [plt.cm.viridis(x) for x in np.linspace(0.05, 0.95, N_DEPTHS)]

MIN_FONT = 7
matplotlib.rcParams.update({
    "font.size": MIN_FONT + 1,
    "axes.titlesize": MIN_FONT + 2,
    "axes.labelsize": MIN_FONT + 1,
    "xtick.labelsize": MIN_FONT,
    "ytick.labelsize": MIN_FONT,
    "legend.fontsize": MIN_FONT,
    "figure.titlesize": MIN_FONT + 3,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def _load(fname):
    return np.load(os.path.join(FIG_DATA_DIR, fname), allow_pickle=True)


def _save(fig, stem, rasterized_axes=None):
    """Save a vector PDF + a 300-dpi PNG preview."""
    pdf_path = os.path.join(OUT_DIR, stem + ".pdf")
    png_path = os.path.join(OUT_DIR, stem + ".png")
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return os.path.basename(pdf_path), os.path.basename(png_path)


def _shade_bands(ax, bands_colors, y_lo=0.0, y_hi=1.0, transform_x=True):
    for (lo, hi), color, label in bands_colors:
        ax.axvspan(lo, hi, color=color, alpha=0.15, lw=0, label=label)


def _mask_runs(mask):
    """Yield (start_idx, end_idx_exclusive) runs of contiguous True/False."""
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    if n == 0:
        return
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate(([0], change, [n]))
    for i in range(len(bounds) - 1):
        yield bounds[i], bounds[i + 1], mask[bounds[i]]


def _plot_by_inhalation(ax, t, y, mask, **kwargs):
    """Plot y vs t colored green (inhalation=True) / black (otherwise),
    drawing contiguous runs so segment colors are exact (no blending)."""
    for s, e, is_inh in _mask_runs(mask):
        e2 = min(e + 1, len(t))  # +1 sample overlap so segments touch visually
        color = "green" if is_inh else "black"
        ax.plot(t[s:e2], y[s:e2], color=color, **kwargs)


# ============================================================================
# FIG-PAPER1-TRACES
# ============================================================================
def render_traces():
    d = _load("fig_traces_data.npz")
    t_s = d["t_s"]
    snf = d["snf_segment"]
    onsets_s = d["sniff_onset_markers_s"] + t_s[0]  # onsets are segment-relative
    ch_idx = d["depth_channel_indices"]
    ycoords = d["depth_ycoords_um"]
    inh_mask_lv = d["inhalation_mask_lv_rate"]

    # Align the ~125 Hz boolean inhalation mask onto the ~1000 Hz LFP/SNF time
    # axis via nearest-neighbor (no resampled version of this mask exists in
    # any frozen file, per figure_data_manifest.yaml).
    lv_fs = len(inh_mask_lv) / (t_s[-1] - t_s[0] + (t_s[1] - t_s[0]))  # ~125 Hz
    lv_fs = 125.0  # pinned in RESULT-example-traces_meta.yaml (inhalation_mask_lv_fs)
    idx_lv = np.clip(np.round((t_s - t_s[0]) * lv_fs).astype(int), 0, len(inh_mask_lv) - 1)
    inh_mask_lfp_rate = inh_mask_lv[idx_lv]

    bands = ["unfiltered", "theta_2_12", "beta_18_30", "gamma_65_100"]
    band_labels = {
        "unfiltered": "unfiltered",
        "theta_2_12": "theta (2-12 Hz)",
        "beta_18_30": "beta (18-30 Hz)",
        "gamma_65_100": "gamma (65-100 Hz)\n+ envelope",
    }

    n_rows = 1 + N_DEPTHS * len(bands)
    fig = plt.figure(figsize=(11, 1.0 * n_rows + 2.0))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.05)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]

    row = 0
    ax = axes[row]
    _plot_by_inhalation(ax, t_s, snf, inh_mask_lfp_rate, lw=0.7)
    ax.set_ylabel("SNF\n(a.u.)", rotation=0, ha="right", va="center")
    row += 1

    for depth_i in range(1, N_DEPTHS + 1):
        for band in bands:
            ax = axes[row]
            key = f"depth{depth_i}_{band}"
            y = d[key]
            _plot_by_inhalation(ax, t_s, y, inh_mask_lfp_rate, lw=0.6)
            if band == "gamma_65_100":
                env = d[f"depth{depth_i}_gamma_65_100_envelope"]
                ax.plot(t_s, env, color="dimgray", lw=0.8, ls="--")
            ax.set_ylabel(f"D{depth_i}\n{band_labels[band]}", rotation=0,
                          ha="right", va="center")
            row += 1

    for ax in axes:
        ylo, yhi = ax.get_ylim()
        if ylo > 0:
            ylo = 0
        if yhi < 0:
            yhi = 0
        ax.set_ylim(ylo, yhi)
        for on_s in onsets_s:
            ax.axvline(on_s, color="gray", lw=0.4, ls=":", alpha=0.6)
        ax.set_xlim(t_s[0], t_s[-1])
        ax.set_yticks([])
        if ax is not axes[-1]:
            ax.set_xticklabels([])
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("time (s)")

    legend_handles = [
        Line2D([0], [0], color="green", lw=1.5, label="inhalation"),
        Line2D([0], [0], color="black", lw=1.5, label="exhalation / pause"),
        Line2D([0], [0], color="dimgray", lw=1.2, ls="--", label="gamma amplitude envelope"),
        Line2D([0], [0], color="gray", lw=1.0, ls=":", label="sniff onset"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", ncol=4,
                   bbox_to_anchor=(1.0, 1.9), frameon=False)

    ch_str = ", ".join(f"D{i+1}=ch{ch_idx[i]} ({ycoords[i]:.0f} um)"
                        for i in range(N_DEPTHS))
    caption = (
        f"Figure 1A analogue: example band-filtered LFP + sniff traces (per depth).\n"
        f"Representative experiment: session 06-24-2022; segment window 80.0-90.0 s "
        f"(chosen by fixed display convention: session with the most retained "
        f"all-control/all-valid-sniff 4-s windows, tie-break earliest date; segment = "
        f"first qualifying contiguous control-valid span of fixed length 10.0 s; "
        f"non-scientific, presentation-only choice).\n"
        f"Depth channels shown: {ch_str}.\n"
        f"Bands: theta 2-12 Hz, beta 18-30 Hz, gamma 65-100 Hz + amplitude envelope; "
        f"inhalation = green, exhalation/pause = black; sniff-onset times marked (dotted gray)."
    )
    fig.suptitle("Figure 1A analogue: example band-filtered LFP + sniff traces (per depth)",
                 y=0.995)
    fig.text(0.01, 0.005, caption, ha="left", va="bottom", fontsize=MIN_FONT, wrap=True)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.94, bottom=0.09)
    return _save(fig, "fig_paper1_traces")


# ============================================================================
# FIG-PAPER1-SPECTRO
# ============================================================================
def render_spectro():
    d = _load("fig_spectro_data.npz")
    freqs = d["freqs"]
    t = d["window_center_s"]
    sniff_rate = d["sniff_rate_at_window_hz"]
    ch_idx = d["depth_channel_indices"]
    ycoords = d["depth_ycoords_um"]

    fmax_display = 100.0
    fmask = freqs <= fmax_display

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(N_DEPTHS + 1, 2, hspace=0.5, wspace=0.3,
                           height_ratios=[0.6] + [1.0] * N_DEPTHS)

    # SNF power spectrogram shown once (depth-independent), spans both columns.
    ax_snf = fig.add_subplot(gs[0, :])
    snf_psd = d["snf_psd_spectrogram"][:, fmask]
    vmin = np.percentile(snf_psd[snf_psd > 0], 1)
    vmax = np.percentile(snf_psd[snf_psd > 0], 99)
    pcm = ax_snf.pcolormesh(t, freqs[fmask], snf_psd.T, cmap="viridis",
                             norm=LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
    ax_snf.set_ylabel("frequency (Hz)")
    ax_snf.set_title("SNF power spectrogram (depth-independent)")
    cb = fig.colorbar(pcm, ax=ax_snf, pad=0.01)
    cb.set_label("SNF power")

    for depth_i in range(1, N_DEPTHS + 1):
        ax_p = fig.add_subplot(gs[depth_i, 0])
        lfp_psd = d[f"lfp_psd_spectrogram_depth{depth_i}"][:, fmask]
        vmin_p = np.percentile(lfp_psd[lfp_psd > 0], 1)
        vmax_p = np.percentile(lfp_psd[lfp_psd > 0], 99)
        pcm_p = ax_p.pcolormesh(t, freqs[fmask], lfp_psd.T, cmap="viridis",
                                 norm=LogNorm(vmin=vmin_p, vmax=vmax_p), rasterized=True)
        ax_p.scatter(t, sniff_rate, s=2, c="white", edgecolors="black",
                     linewidths=0.2, label="inst. sniff rate (Hz)")
        ax_p.set_ylabel(f"D{depth_i}\nfrequency (Hz)")
        if depth_i == 1:
            ax_p.set_title("LFP power")
        if depth_i == N_DEPTHS:
            ax_p.set_xlabel("time (s)")
        cb_p = fig.colorbar(pcm_p, ax=ax_p, pad=0.01)
        cb_p.set_label("LFP power")

        ax_c = fig.add_subplot(gs[depth_i, 1])
        coh = d[f"coh_spectrogram_depth{depth_i}"][:, fmask]
        pcm_c = ax_c.pcolormesh(t, freqs[fmask], coh.T, cmap="viridis",
                                 vmin=0.0, vmax=1.0, rasterized=True)
        ax_c.scatter(t, sniff_rate, s=2, c="white", edgecolors="black",
                     linewidths=0.2)
        if depth_i == 1:
            ax_c.set_title("LFP-sniff coherence")
        if depth_i == N_DEPTHS:
            ax_c.set_xlabel("time (s)")
        cb_c = fig.colorbar(pcm_c, ax=ax_c, pad=0.01)
        cb_c.set_label("coherence (0-1)")

    ch_str = ", ".join(f"D{i+1}=ch{ch_idx[i]} ({ycoords[i]:.0f} um)"
                        for i in range(N_DEPTHS))
    caption = (
        "Figure 1C analogue: multitaper spectrograms per depth with sniff-rate overlay.\n"
        "Representative experiment: session 06-24-2022 (same session/display convention as "
        "FIG-PAPER1-TRACES); full recording duration shown (not restricted to the 10-s "
        "FIG-PAPER1-TRACES segment).\n"
        f"Depth channels: {ch_str}.\n"
        f"{MULTITAPER_NOTE}. Colorbars: LFP power, SNF power, coherence (0-1). "
        "White/black-edged scatter overlay = instantaneous sniff rate (Hz). "
        "Frequency axis cropped to 0-100 Hz for display; underlying data extend to 500 Hz."
    )
    fig.suptitle("Figure 1C analogue: multitaper spectrograms per depth with sniff-rate overlay",
                 y=0.995)
    fig.text(0.01, 0.002, caption, ha="left", va="bottom", fontsize=MIN_FONT, wrap=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.06)
    return _save(fig, "fig_paper1_spectro")


# ============================================================================
# FIG-PAPER1-SNIFFRATE
# ============================================================================
def render_sniffrate():
    d = _load("fig_sniffrate_data.npz")
    sessions = list(d["session"])
    median_hz = d["median_hz"]
    q1_hz = d["q1_hz"]
    q3_hz = d["q3_hz"]
    iqr_hz = d["iqr_hz"]
    n_retained = d["n_sniffs_retained_isi"]

    n_exp = len(sessions)
    ncols = 3
    nrows = int(np.ceil(n_exp / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    colors = [plt.cm.viridis(x) for x in np.linspace(0.05, 0.95, n_exp)]

    for i, sess in enumerate(sessions):
        key = sess.replace("-", "_") + "__inst_rate_hz"
        raw = d[key]  # UNTRIMMED distribution (per figure_data_manifest warning)
        ax = axes[i]
        ax.hist(raw, bins=60, density=True, color=colors[i], alpha=0.85,
                edgecolor="none")
        ax.axvline(median_hz[i], color="black", lw=1.2, label="median")
        ax.axvspan(q1_hz[i], q3_hz[i], color="black", alpha=0.12, label="IQR")
        ax.set_title(f"{sess}\nmedian={median_hz[i]:.2f} Hz, IQR={iqr_hz[i]:.2f} Hz\n"
                     f"n retained (ISI-valid)={n_retained[i]}", fontsize=MIN_FONT)
        ax.set_xlim(0, 22)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(n_exp, len(axes)):
        axes[j].axis("off")

    for ax in axes[:n_exp]:
        ax.set_ylim(bottom=0)
    for ax in axes[-ncols:]:
        ax.set_xlabel("instantaneous sniff rate (Hz)")
    for i in range(0, n_exp, ncols):
        axes[i].set_ylabel("probability density")

    handles = [Line2D([0], [0], color="black", lw=1.2, label="median"),
               Patch(facecolor="black", alpha=0.12, label="IQR (Q1-Q3)")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False)

    caption = (
        "Figure 1D analogue: instantaneous sniff-rate distribution (control epochs), per "
        "experiment. Control-epochs-only restriction applied (ethanol-contact epochs excluded); "
        "distribution shown is UNTRIMMED (not filtered by the ISI 5th/95th-percentile trim mask "
        "that additionally gates the coherence-analysis windows), matching the median/IQR values "
        "annotated per panel (injected from RESULT-sniff-rate.yaml). n_sniffs retained (ISI-valid, "
        "coherence-analysis mask) annotated per panel from RESULT-qc-counts.yaml."
    )
    fig.suptitle("Figure 1D analogue: instantaneous sniff-rate distribution (control epochs)",
                 y=0.995)
    fig.text(0.01, 0.0, caption, ha="left", va="bottom", fontsize=MIN_FONT, wrap=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.16, hspace=0.55)
    return _save(fig, "fig_paper1_sniffrate")


# ============================================================================
# FIG-PAPER1-PSD
# ============================================================================
def render_psd():
    d = _load("fig_psd_data.npz")
    freqs = d["freqs"]
    ycoords = d["depth_ycoords_um"]
    n_exp = int(d["n_experiments"])

    fig, ax = plt.subplots(figsize=(9, 6))
    _shade_bands(ax, [(THETA_BAND, COLOR_THETA, "theta 2-12 Hz"),
                      (BETA_BAND, COLOR_BETA, "beta 18-30 Hz"),
                      (GAMMA_BAND, COLOR_GAMMA, "gamma 65-100 Hz")])

    for depth_i in range(1, N_DEPTHS + 1):
        mean = d[f"psd_depth{depth_i}_mean"]
        sem = d[f"psd_depth{depth_i}_sem"]
        color = DEPTH_COLORS[depth_i - 1]
        ax.plot(freqs, mean, color=color, lw=1.4,
                label=f"D{depth_i} ({ycoords[depth_i-1]:.0f} um)")
        ax.fill_between(freqs, mean - sem, mean + sem, color=color, alpha=0.25, lw=0)

    ax.set_yscale("log")
    ax.set_xlim(0.25, 150)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("LFP power spectral density")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, title="depth (tip -> surface)")

    caption = (
        f"Figure 1B analogue: LFP power spectral density overlaid across the five depths "
        f"(grand mean +/- SEM across n_experiments={n_exp} animals). Band definitions marked: "
        f"theta 2-12 Hz, beta 18-30 Hz, gamma 65-100 Hz. Depth ordinals 1-5 (tip -> surface), "
        f"legend states channel ycoord (um). Frequency axis cropped to 0.25-150 Hz for display; "
        f"underlying data extend to 500 Hz (Nyquist)."
    )
    fig.suptitle("Figure 1B analogue: LFP power spectral density overlaid across the five depths")
    fig.text(0.01, 0.0, caption, ha="left", va="bottom", fontsize=MIN_FONT, wrap=True)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.18)
    return _save(fig, "fig_paper1_psd")


# ============================================================================
# FIG-PAPER1-COH  (scientific-honesty critical figure)
# ============================================================================
def render_coh():
    d = _load("fig_coh_data.npz")
    freqs = d["freqs"]
    ycoords = d["depth_ycoords_um"]
    n_exp = int(d["n_experiments"])
    peak_pctile = d["theta_coh_peak_coh_percentile"]
    max_pctile = float(np.max(peak_pctile))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axvspan(*THETA_BAND, color=COLOR_THETA, alpha=0.15, lw=0, label="theta 2-12 Hz")

    null_alphas = np.linspace(0.35, 0.95, N_DEPTHS)
    for depth_i in range(1, N_DEPTHS + 1):
        color = DEPTH_COLORS[depth_i - 1]
        mean = d[f"coh_depth{depth_i}_mean"]
        sem = d[f"coh_depth{depth_i}_sem"]
        ax.plot(freqs, mean, color=color, lw=1.4,
                label=f"D{depth_i} ({ycoords[depth_i-1]:.0f} um) observed")
        ax.fill_between(freqs, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)

        null_mean = d[f"null_depth{depth_i}_mean"]
        null_sem = d[f"null_depth{depth_i}_sem"]
        ax.plot(freqs, null_mean, color="red", lw=1.0, ls=":",
                alpha=null_alphas[depth_i - 1])
        ax.fill_between(freqs, null_mean - null_sem, null_mean + null_sem,
                         color="red", alpha=0.06, lw=0)

    ax.set_xlim(0.25, 150)
    ax.set_ylim(0.0, 1.0)  # full theoretical coherence range: no truncation (D11 honesty req.)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("LFP-sniff coherence (0-1)")
    ax.spines[["top", "right"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="red", lw=1.2, ls=":",
                           label="95% circular-shift null (per depth, darker = deeper)"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=MIN_FONT)

    ax.text(0.02, 0.90,
            f"Observed coherence NEVER exceeds its own 95% circular-shift null threshold\n"
            f"across all {30} experiment x depth cells (max peak-coherence percentile = "
            f"{max_pctile:.1f}, everywhere else lower).",
            transform=ax.transAxes, fontsize=MIN_FONT, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))

    caption = (
        f"Figure 1C-bottom / 1E analogue (KEY): LFP-sniff coherence spectrum per depth with "
        f"95% circular-shift null (>=1000 shifts, seed 5489, red dotted; per depth, grand mean "
        f"+/- SEM across n_experiments={n_exp} animals) shown on the SAME axes as the observed "
        f"coherence (grand mean +/- SEM across the same n_experiments={n_exp} animals). Theta "
        f"band (2-12 Hz) shaded. Depth ordinals 1-5 (tip -> surface); legend states channel "
        f"ycoord (um). Y-axis spans the full 0-1 coherence range (not truncated). Across all 30 "
        f"experiment x depth cells, the observed coherence never reaches the 95% null threshold "
        f"(max peak-coherence percentile = {max_pctile:.1f}). Frequency axis cropped to 0.25-150 Hz "
        f"for display; underlying data extend to 500 Hz (Nyquist)."
    )
    fig.suptitle("Figure 1C-bottom / 1E analogue (KEY): LFP-sniff coherence spectrum per depth "
                 "with null threshold")
    fig.text(0.01, 0.0, caption, ha="left", va="bottom", fontsize=MIN_FONT, wrap=True)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.2)
    return _save(fig, "fig_paper1_coh"), max_pctile


# ============================================================================
# FIG-PAPER1-DEPTH-SUMMARY  (HEADLINE, non-significance critical figure)
# ============================================================================
def render_depth_summary():
    d = _load("fig_depth_summary_data.npz")
    ords = d["depth_ordinal"]
    grand_mean = d["grand_mean_theta_coh"]
    grand_sem = d["grand_sem_theta_coh"]
    n_animals = d["grand_n_animals_theta_coh"]
    per_animal = d["per_animal_mean_theta_coh_by_depth"]
    animal_labels = d["animal_labels"]
    best_ord = int(d["overall_best_depth_ordinal"])
    consistency = d["per_ordinal_consistency_counts"]

    grand_mean_sig = d["grand_mean_sig_theta_fraction_by_depth"]
    grand_sem_sig = d["grand_sem_sig_theta_fraction_by_depth"]
    per_animal_sig = d["per_animal_sig_theta_fraction_by_depth"]

    lmm_chisq = float(d["lmm_chisq"]); lmm_df = int(d["lmm_df"]); lmm_p = float(d["lmm_pvalue"])
    fr_stat = float(d["friedman_statistic"]); fr_df = int(d["friedman_df"]); fr_p = float(d["friedman_pvalue"])
    kendalls_w = float(d["kendalls_w"])
    omnibus_sig = bool(d["omnibus_significant"])
    n_animals_stat = int(d["n_animals_stat"])

    # depth ycoord annotation: same fixed depth column, already-frozen and present
    # in the sibling figure_data files (fig_psd_data.npz / fig_coh_data.npz).
    ycoords = _load("fig_psd_data.npz")["depth_ycoords_um"]

    fig, (ax_main, ax_comp) = plt.subplots(1, 2, figsize=(13, 6))

    # --- main panel: mean theta coherence by depth ---
    for a_i in range(per_animal.shape[0]):
        ax_main.plot(ords, per_animal[a_i], color="0.6", lw=0.8, alpha=0.7,
                     marker="o", markersize=3, zorder=1)
    ax_main.errorbar(ords, grand_mean, yerr=grand_sem, color="black", lw=1.8,
                      marker="o", markersize=6, capsize=3, zorder=3,
                      label="grand mean +/- SEM (n=%d animals)" % n_animals[0])
    ax_main.scatter([best_ord], [grand_mean[best_ord - 1]], s=160,
                     facecolors="none", edgecolors="#D55E00", linewidths=2.2,
                     zorder=4, label=f"overall best-aligned depth (D{best_ord}, "
                                      f"descriptive; won in {consistency[best_ord-1]}/"
                                      f"{n_animals_stat} animals)")
    ax_main.set_xticks(ords)
    ax_main.set_xticklabels([f"D{o}\n({ycoords[o-1]:.0f} um)" for o in ords])
    ax_main.set_xlabel("depth (ordinal 1-5, tip -> surface; ycoord um)")
    ax_main.set_ylabel("mean theta coherence (2-12 Hz)")
    ax_main.set_ylim(0, max(per_animal.max(), (grand_mean + grand_sem).max()) * 1.25)
    ax_main.spines[["top", "right"]].set_visible(False)
    ax_main.legend(loc="upper left", frameon=False, fontsize=MIN_FONT - 0.5)

    omnibus_text = (
        f"Omnibus (non-significant, exploratory characterization):\n"
        f"Mixed-effects LRT: chi-square={lmm_chisq:.3f}, df={lmm_df}, p={lmm_p:.3f}\n"
        f"Friedman: chi-square={fr_stat:.3f}, df={fr_df}, p={fr_p:.3f}, "
        f"Kendall's W={kendalls_w:.3f}\n"
        f"No significant depth effect detected (alpha=0.05); no post-hoc comparisons run."
    )
    ax_main.text(0.98, 0.02, omnibus_text, transform=ax_main.transAxes, fontsize=MIN_FONT,
                 ha="right", va="bottom",
                 bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))

    # --- companion panel: significant-theta fraction by depth ---
    for a_i in range(per_animal_sig.shape[0]):
        ax_comp.plot(ords, per_animal_sig[a_i], color="0.6", lw=0.8, alpha=0.7,
                     marker="o", markersize=3, zorder=1)
    ax_comp.errorbar(ords, grand_mean_sig, yerr=grand_sem_sig, color="black", lw=1.8,
                      marker="o", markersize=6, capsize=3, zorder=3)
    ax_comp.set_xticks(ords)
    ax_comp.set_xticklabels([f"D{o}\n({ycoords[o-1]:.0f} um)" for o in ords])
    ax_comp.set_xlabel("depth (ordinal 1-5, tip -> surface; ycoord um)")
    ax_comp.set_ylabel("significant-theta fraction")
    ax_comp.set_ylim(0, max(per_animal_sig.max(), (grand_mean_sig + grand_sem_sig).max()) * 1.25)
    ax_comp.spines[["top", "right"]].set_visible(False)
    ax_comp.set_title("companion: significant-theta fraction by depth", fontsize=MIN_FONT + 1)

    caption = (
        f"HEADLINE: sniff-alignment strength (mean theta coherence) as a function of depth. "
        f"Per-animal points (gray, n={n_animals_stat} animals after >=30-window exclusion) plus "
        f"grand mean +/- SEM; overall best-aligned depth (descriptive observation, not a "
        f"definitive claim) = D{best_ord}, per-ordinal consistency count "
        f"{consistency[best_ord-1]}/{n_animals_stat} animals. Companion panel: significant-theta "
        f"fraction by depth (same convention). Omnibus test for depth-dependence: mixed-effects "
        f"LRT chi-square={lmm_chisq:.3f}, df={lmm_df}, p={lmm_p:.3f}; Friedman "
        f"chi-square={fr_stat:.3f}, df={fr_df}, p={fr_p:.3f}, Kendall's W={kendalls_w:.3f} -- "
        f"BOTH NON-SIGNIFICANT (alpha=0.05; omnibus_significant={omnibus_sig}); no Holm-corrected "
        f"post-hoc comparisons were run (PROH-006/OUT-016). This is an exploratory, "
        f"non-causal characterization: no significant depth effect was detected in this dataset "
        f"(absence of evidence is not evidence of absence); depth does not 'cause', 'determine', "
        f"or 'control' alignment strength here."
    )
    fig.suptitle("HEADLINE: sniff-alignment strength (mean theta coherence) as a function of depth "
                 "(exploratory; no significant depth effect detected)")
    fig.text(0.01, 0.0, caption, ha="left", va="bottom", fontsize=MIN_FONT, wrap=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.28, wspace=0.3)
    return _save(fig, "fig_paper1_depth_summary")


# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    outputs = {}

    outputs["FIG-PAPER1-TRACES"] = {
        "files": list(render_traces()),
        "figure_data": ["fig_traces_data.npz"],
    }
    outputs["FIG-PAPER1-SPECTRO"] = {
        "files": list(render_spectro()),
        "figure_data": ["fig_spectro_data.npz"],
    }
    outputs["FIG-PAPER1-SNIFFRATE"] = {
        "files": list(render_sniffrate()),
        "figure_data": ["fig_sniffrate_data.npz"],
    }
    outputs["FIG-PAPER1-PSD"] = {
        "files": list(render_psd()),
        "figure_data": ["fig_psd_data.npz"],
    }
    (coh_files, max_pctile) = render_coh()
    outputs["FIG-PAPER1-COH"] = {
        "files": list(coh_files),
        "figure_data": ["fig_coh_data.npz"],
        "scientific_honesty_check": {
            "max_peak_coherence_percentile_across_30_cells": max_pctile,
            "null_never_exceeded": bool(max_pctile < 95.0),
        },
    }
    outputs["FIG-PAPER1-DEPTH-SUMMARY"] = {
        "files": list(render_depth_summary()),
        "figure_data": ["fig_depth_summary_data.npz", "fig_psd_data.npz (depth_ycoords_um axis annotation only)"],
    }

    manifest_lines = [
        "task_id: NP-PAPER-1",
        "generated_by: render_figures_np_paper1.py (03_software, additive, Blueprint v2 P4 pure renderer)",
        "note: 'Every plotted number traces to 06_figures/figure_data/*.npz (already frozen-traced,",
        "  read-only). No recomputation/re-derivation performed here. No figure_examples/ sketch existed",
        "  for this task (confirmed empty); rendered purely from contract_v001.yaml figure specs.'",
        "figures:",
    ]
    for fig_id, info in outputs.items():
        manifest_lines.append(f"  {fig_id}:")
        manifest_lines.append(f"    output_files:")
        for f in info["files"]:
            manifest_lines.append(f"    - {f}")
        manifest_lines.append(f"    rendered_from_figure_data:")
        for f in info["figure_data"]:
            manifest_lines.append(f"    - {f}")
        manifest_lines.append("    influenced_by: null  # no figure_examples/ sketch existed (confirmed empty)")
        if "scientific_honesty_check" in info:
            manifest_lines.append("    scientific_honesty_check:")
            for k, v in info["scientific_honesty_check"].items():
                if isinstance(v, bool):
                    v = "true" if v else "false"
                manifest_lines.append(f"      {k}: {v}")
    manifest_path = os.path.join(OUT_DIR, "render_manifest.yaml")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print("Wrote figures + render_manifest.yaml to", OUT_DIR)
    for fig_id, info in outputs.items():
        print(" ", fig_id, "->", info["files"])


if __name__ == "__main__":
    main()

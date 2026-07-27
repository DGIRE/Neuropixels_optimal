"""
run_probe_standardization.py
============================
Driver that turns the cross-experiment LFP aggregate into figures assessing
probe-insertion / laminar standardization.

It produces, into the Figures folder:

  * one figure per experiment  ->  depthfreq_<DATE>.png
        depth (y) x frequency (x) LFP power matrix, jet colormap, labelled by
        date, with that experiment's PCA (PC1, PC2) coordinates written into
        the figure as its description.

  * PCA_summary.png
        (a) every experiment as a point in the PC1-PC2 plane (labelled by date)
        (b) PC1 loadings reshaped across the depth x frequency matrix (jet)
        (c) PC2 loadings reshaped across the depth x frequency matrix (jet)
        (d) cumulative % variance explained by each PC (+ per-PC bars)

  * pca_coordinates.csv      date, PC1..PCk scores
  * pca_variance.csv         PC, explained_%, cumulative_%
  * (optional) contact sheet montage of all experiments -> depthfreq_montage.png

Run (on the workstation, where np_aggregate.h5 lives; the raw file is too large
to move):

    python "C:\\Projects\\Repos\\Neuropixels\\_pipeline\\Exploratory Requests\\Probe standardization\\Code\\run_probe_standardization.py"

Optional overrides:
    python run_probe_standardization.py  <H5_PATH>  <FIG_DIR>

Requires: numpy, scipy, matplotlib, h5py (same environment as the Optimized
Python kernel). No raw-data access is needed -- it reads only the aggregate.
"""
from __future__ import annotations

import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless: write files, no display needed
import matplotlib.pyplot as plt

# import the analysis suite from the same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_standardization import (          # noqa: E402
    load_lfp_sessions, canonical_freq_grid, build_feature_matrix,
    pca_across_experiments,
)

# ===========================================================================
# CONFIG  -- edit these knobs; every analysis choice is here.
# ===========================================================================
H5_PATH = r"C:\Projects\Repos\Neuropixels\DATA\Aggregate\np_aggregate.h5"
FIG_DIR = (r"C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests"
           r"\Probe standardization\Figures")

# --- spectral / matrix parameters ---
FMIN, FMAX = 1.0, 100.0     # frequency band kept on the x-axis (Hz). Aggregate
                            # LFP is DC..~100 Hz (250 Hz, anti-aliased), so 100
                            # Hz is the usable ceiling. 1 Hz floor drops the DC
                            # bin so it doesn't dominate.
NPERSEG = 512               # Welch segment length (samples). 512 @ 250 Hz =
                            # 2.05 s windows -> ~0.49 Hz resolution.
NOVERLAP = 256              # 50% overlap
DEPTH_BINS = 32             # rows in the common depth axis (matches the 32
                            # retained LFP channels). Relative depth 0=tip..1=surface.
LOG_POWER = True            # plot / analyse 10*log10 power (dB). Recommended.

NOTCH_FREQ = 60.0           # mains-hum notch (Hz); applied to the LFP with a
                            # zero-phase IIR notch BEFORE Welch, so the line
                            # noise is genuinely removed from the PSD/PCA
                            # input, not just masked on the display. Set to
                            # None to disable.
NOTCH_Q = 30.0              # notch quality factor (bandwidth = NOTCH_FREQ/NOTCH_Q,
                            # ~2 Hz wide at 60 Hz) -- narrow enough to leave
                            # neighboring frequencies (beta/gamma) intact.

# --- session exclusions ---
EXCLUDE_DATES = {
    "12-15-2021",   # confirmed wrong source file in the aggregate: the raw
                    # AP recording used for this date is "Np10_GroundedSnip"
                    # (a ~82 s grounding/calibration check), not the real
                    # ~48-55 min craniotomy experiment -- verified against
                    # the independently re-derived file provenance in the
                    # NP-PAPER-1 task (DEV-001). Excluded here rather than
                    # silently left in as a spurious PCA outlier.
}

# --- PCA parameters ---
PER_EXPERIMENT_CENTER = True   # subtract each experiment's own mean (dB) before
                               # PCA -> removes overall gain/reference offset so
                               # PCA sees the laminar/spectral PATTERN. Set False
                               # to let absolute power level drive PC1.
STANDARDIZE = False            # scale each feature to unit variance (correlation PCA)

# --- display ---
CMAP = "jet"                # requested colormap
DPI = 150
MAKE_MONTAGE = True         # also emit a single contact-sheet of all experiments
# ===========================================================================


def _annotate(ax, text, loc="upper left"):
    """White text in a translucent dark box (readable over jet)."""
    xy = (0.02, 0.98) if loc == "upper left" else (0.98, 0.02)
    ha = "left" if loc == "upper left" else "right"
    va = "top" if loc == "upper left" else "bottom"
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.55,
                      edgecolor="none"))


def draw_individual(M, f_common, session, pc1, pc2, outpath):
    """Depth x frequency power matrix for one experiment (jet), PCA coords in text."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    extent = [f_common[0], f_common[-1], 0.0, float(DEPTH_BINS)]
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=CMAP, extent=extent,
                   interpolation="nearest")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relative depth bin  (0 = tip / deep  →  %d = surface)" % DEPTH_BINS)
    ax.set_title("%s   —   %s" % (session["date"], session.get("experiment_name", "")),
                 fontsize=9)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("LFP power (dB)" if LOG_POWER else "LFP power (V²/Hz)")

    yc = np.asarray(session["ycoord"], dtype=np.float64)
    yc = yc[np.isfinite(yc)]
    yrange = ("%.0f–%.0f µm" % (yc.min(), yc.max())) if yc.size else "n/a"
    desc = ("PCA coordinates:  PC1 = %+.4f,  PC2 = %+.4f\n"
            "ycoord span: %s   |   duration: %.0f s" %
            (pc1, pc2, yrange, session["duration_s"]))
    _annotate(ax, desc, "upper left")

    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def draw_pca_summary(pca, dates, f_common, outpath):
    """Scatter of experiments + PC1/PC2 loadings (jet) + cumulative variance."""
    scores = pca["scores"]
    loadings = pca["loadings"]
    explained = pca["explained"]
    cum = pca["cum_explained"]
    depth_bins, n_freq = pca["shape"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # (a) PC1 vs PC2 scatter, labelled by date
    ax = axes[0, 0]
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.scatter(scores[:, 0], scores[:, 1], s=45, c="#333333", zorder=3)
    for i, d in enumerate(dates):
        ax.annotate(d, (scores[i, 0], scores[i, 1]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_xlabel("PC1  (%.1f%% var)" % (explained[0] * 100))
    ax.set_ylabel("PC2  (%.1f%% var)" % (explained[1] * 100 if explained.size > 1 else 0))
    ax.set_title("Experiments in PC1–PC2 space")

    # (b) PC1 loadings across depth x freq
    L1 = loadings[0].reshape(depth_bins, n_freq)
    lim1 = np.max(np.abs(L1)) or 1.0
    ax = axes[0, 1]
    im1 = ax.imshow(L1, aspect="auto", origin="lower", cmap=CMAP,
                    extent=[f_common[0], f_common[-1], 0, depth_bins],
                    vmin=-lim1, vmax=lim1, interpolation="nearest")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relative depth bin (0 = tip)")
    ax.set_title("PC1 loadings  (%.1f%% var)" % (explained[0] * 100))
    fig.colorbar(im1, ax=ax, pad=0.02).set_label("loading")

    # (c) PC2 loadings across depth x freq
    ax = axes[1, 0]
    if loadings.shape[0] > 1:
        L2 = loadings[1].reshape(depth_bins, n_freq)
        lim2 = np.max(np.abs(L2)) or 1.0
        im2 = ax.imshow(L2, aspect="auto", origin="lower", cmap=CMAP,
                        extent=[f_common[0], f_common[-1], 0, depth_bins],
                        vmin=-lim2, vmax=lim2, interpolation="nearest")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Relative depth bin (0 = tip)")
        ax.set_title("PC2 loadings  (%.1f%% var)" %
                     (explained[1] * 100 if explained.size > 1 else 0))
        fig.colorbar(im2, ax=ax, pad=0.02).set_label("loading")
    else:
        ax.axis("off")

    # (d) cumulative variance explained
    ax = axes[1, 1]
    k = explained.size
    xs = np.arange(1, k + 1)
    ax.bar(xs, explained * 100, color="#9ecae1", label="per PC")
    ax.plot(xs, cum * 100, "o-", color="#08519c", label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Scree / cumulative variance")
    ax.set_xticks(xs)
    ax.set_ylim(0, 105)
    ax.legend(loc="center right", fontsize=8)

    fig.suptitle("Probe-standardization PCA  (depth × frequency LFP power, "
                 "%d experiments)" % scores.shape[0], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def draw_montage(mats, dates, f_common, outpath):
    """Optional contact sheet: all experiments' depth x freq matrices on one page."""
    n = len(dates)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow),
                             squeeze=False)
    vmin = np.percentile(mats, 2)
    vmax = np.percentile(mats, 98)
    for idx in range(nrow * ncol):
        ax = axes[idx // ncol][idx % ncol]
        if idx < n:
            ax.imshow(mats[idx], aspect="auto", origin="lower", cmap=CMAP,
                      extent=[f_common[0], f_common[-1], 0, mats.shape[1]],
                      vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(dates[idx], fontsize=8)
            ax.tick_params(labelsize=6)
        else:
            ax.axis("off")
    fig.suptitle("Depth × frequency LFP power — all experiments (common color scale)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    h5_path = argv[0] if len(argv) >= 1 else H5_PATH
    fig_dir = argv[1] if len(argv) >= 2 else FIG_DIR

    os.makedirs(fig_dir, exist_ok=True)

    print("Reading aggregate: %s" % h5_path)
    sessions = load_lfp_sessions(h5_path)
    if not sessions:
        raise SystemExit("No experiments with LFP found in %s" % h5_path)
    print("Found %d experiment(s) with LFP: %s"
          % (len(sessions), ", ".join(s["date"] for s in sessions)))

    if EXCLUDE_DATES:
        excluded = [s["date"] for s in sessions if s["date"] in EXCLUDE_DATES]
        sessions = [s for s in sessions if s["date"] not in EXCLUDE_DATES]
        if excluded:
            print("Excluding %d session(s) per EXCLUDE_DATES: %s"
                  % (len(excluded), ", ".join(excluded)))
        if not sessions:
            raise SystemExit("All sessions excluded -- nothing left to analyze.")

    # sanity: all sessions should share the LFP sample rate
    fs_set = {round(s["fs"], 6) for s in sessions}
    if len(fs_set) > 1:
        print("WARNING: sessions have differing LFP_fs: %s "
              "(matrices are re-gridded so PCA still aligns)." % fs_set)
    fs = sessions[0]["fs"]

    f_common = canonical_freq_grid(fs, NPERSEG, FMIN, FMAX)
    print("Common frequency grid: %d bins, %.2f–%.2f Hz (Δf=%.3f Hz)"
          % (f_common.size, f_common[0], f_common[-1],
             f_common[1] - f_common[0] if f_common.size > 1 else 0.0))

    if NOTCH_FREQ is not None:
        print("Applying %.1f Hz notch filter (Q=%.1f) to LFP before Welch PSD"
              % (NOTCH_FREQ, NOTCH_Q))
    dates, mats = build_feature_matrix(
        sessions, f_common, DEPTH_BINS, NPERSEG, NOVERLAP, log=LOG_POWER,
        notch_freq=NOTCH_FREQ, notch_q=NOTCH_Q)
    print("Feature tensor: %s (n_exp, depth_bins, n_freq)" % (mats.shape,))

    pca = pca_across_experiments(
        mats, per_experiment_center=PER_EXPERIMENT_CENTER, standardize=STANDARDIZE)
    scores = pca["scores"]

    # per-experiment figures (with PCA coords written in)
    for i, s in enumerate(sessions):
        out = os.path.join(fig_dir, "depthfreq_%s.png" % s["date"])
        draw_individual(mats[i], f_common, s, scores[i, 0], scores[i, 1], out)
        print("  wrote %s" % os.path.basename(out))

    # PCA summary
    pca_out = os.path.join(fig_dir, "PCA_summary.png")
    draw_pca_summary(pca, dates, f_common, pca_out)
    print("  wrote %s" % os.path.basename(pca_out))

    # optional montage
    if MAKE_MONTAGE:
        m_out = os.path.join(fig_dir, "depthfreq_montage.png")
        draw_montage(mats, dates, f_common, m_out)
        print("  wrote %s" % os.path.basename(m_out))

    # CSVs
    k = scores.shape[1]
    with open(os.path.join(fig_dir, "pca_coordinates.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date"] + ["PC%d" % (j + 1) for j in range(k)])
        for i, d in enumerate(dates):
            w.writerow([d] + ["%.6f" % v for v in scores[i]])
    with open(os.path.join(fig_dir, "pca_variance.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["PC", "explained_percent", "cumulative_percent"])
        for j in range(k):
            w.writerow([j + 1, "%.4f" % (pca["explained"][j] * 100),
                        "%.4f" % (pca["cum_explained"][j] * 100)])
    print("  wrote pca_coordinates.csv, pca_variance.csv")

    print("\nDone. %d experiment figures + PCA_summary.png in:\n  %s"
          % (len(sessions), fig_dir))


if __name__ == "__main__":
    main()

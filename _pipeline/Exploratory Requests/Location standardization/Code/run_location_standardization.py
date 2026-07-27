"""
run_location_standardization.py
===============================
Driver: pool every recording site from the aggregate, 60 Hz-notch it, cluster
sites by their LFP spectrum, and examine the structure with PCA.

Writes into the Figures folder:

  * master_matrix_clustered.png
        The pooled master matrix (row = one recording site from one experiment;
        column = frequency), rows ORGANISED BY HIERARCHICAL CLUSTERING of their
        spectra, jet colormap, with the row (y-axis) dendrogram beside it.

  * pca_sites.png
        (a) PC1-PC2 scatter, one point per site, coloured by cluster
        (b) the same scatter coloured by relative depth (tip->surface)
        (c) PC1 & PC2 loadings across frequency
        (d) scree / cumulative variance explained

  * site_table.csv        experiment, channel_row, ycoord_um, reldepth, cluster, PC1, PC2
  * pca_variance.csv       PC, explained_%, cumulative_%

Run (on the workstation where np_aggregate.h5 lives):
    python "C:\\Projects\\Repos\\Neuropixels\\_pipeline\\Exploratory Requests\\Location standardization\\Code\\run_location_standardization.py"

Optional overrides:  python run_location_standardization.py <H5_PATH> <FIG_DIR>

Requires numpy, scipy, matplotlib, h5py (same env as the Optimized Python kernel).
Reads only the aggregate -- no raw-data access needed.
"""
from __future__ import annotations

import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram, fcluster

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_standardization import (          # noqa: E402
    load_lfp_sessions, canonical_freq_grid, build_master_matrix,
    row_normalize, hierarchical_linkage, pca_sites,
)

# ===========================================================================
# CONFIG
# ===========================================================================
H5_PATH = r"C:\Projects\Repos\Neuropixels\DATA\Aggregate\np_aggregate.h5"
FIG_DIR = (r"C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests"
           r"\Location standardization\Figures")

FMIN, FMAX = 1.0, 100.0     # frequency band on the x-axis (Hz); LFP is DC..~100 Hz
NPERSEG = 512               # Welch window (samples); 512 @ 250 Hz = 2.05 s (~0.49 Hz)
NOVERLAP = 256
NOTCH_F0 = 60.0             # mains line-noise notch (Hz); set 0/None to disable
NOTCH_Q = 30.0             # notch quality factor (higher = narrower)
LOG_POWER = True

ROW_NORMALIZE = True        # cluster/PCA on each site's spectral SHAPE (recommended)
CLUSTER_METHOD = "ward"     # 'ward' | 'average' | 'complete' | 'single'
CLUSTER_METRIC = "euclidean"
N_CLUSTERS = 6              # flat clusters cut from the dendrogram (coloring)

CMAP = "jet"                # requested heatmap colormap
DPI = 150
# ===========================================================================


def _cluster_threshold(Z, k):
    """Distance at which cutting the linkage yields k clusters (for coloring)."""
    if k <= 1 or Z.shape[0] < k - 1:
        return 0.0
    # merge heights are Z[:,2] ascending; cutting above the (k-1)-th-from-top merge
    heights = np.sort(Z[:, 2])
    return float(heights[-(k - 1)] - 1e-9)


def draw_clustered_heatmap(M_display, Z, f_common, n_clusters, outpath):
    """Clustered master matrix (jet) with the row dendrogram on the left."""
    n_sites = M_display.shape[0]
    thr = _cluster_threshold(Z, n_clusters)

    fig = plt.figure(figsize=(12.5, 10.5))
    # 3 columns: dendrogram | heatmap | colorbar (explicit -> no tight_layout needed)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.22, 1.0, 0.025], wspace=0.03)
    axd = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[0, 2])

    # dendrogram on the y-axis (leaves along y)
    dd = dendrogram(Z, orientation="left", ax=axd, no_labels=True,
                    color_threshold=thr, above_threshold_color="0.6")
    order = dd["leaves"]                       # bottom -> top for origin='lower'
    axd.set_title("row dendrogram", fontsize=9)
    axd.set_xlabel("distance", fontsize=8)
    axd.set_yticks([])
    axd.tick_params(labelsize=7)
    for spine in ("top", "right", "left"):
        axd.spines[spine].set_visible(False)

    # heatmap, rows reordered to match the dendrogram leaves, y-limits matched.
    # Robust color limits keep the 60 Hz notch column from compressing the scale.
    heat = M_display[order]
    vmin, vmax = np.percentile(heat, [2, 98])
    ylim = axd.get_ylim()
    im = axh.imshow(heat, aspect="auto", origin="lower", cmap=CMAP,
                    extent=[f_common[0], f_common[-1], ylim[0], ylim[1]],
                    vmin=vmin, vmax=vmax, interpolation="nearest")
    axh.set_ylim(ylim)
    axh.set_xlabel("Frequency (Hz)")
    axh.set_yticks([])

    cbar = fig.colorbar(im, cax=axc)
    cbar.set_label("LFP power, per-site mean-subtracted (dB)"
                   if ROW_NORMALIZE else ("LFP power (dB)" if LOG_POWER else "power"))

    # descriptive y-axis label placed at the far left (clear of the dendrogram)
    fig.text(0.02, 0.5, "Recording sites — pooled across experiments, "
             "ordered by spectral clustering", rotation=90, va="center",
             ha="center", fontsize=10)
    fig.suptitle("Location standardization — %d sites clustered by LFP spectrum"
                 " (60 Hz notched)" % n_sites, fontsize=12)
    fig.subplots_adjust(left=0.075, right=0.90, top=0.94, bottom=0.07)
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def draw_pca(pca, clusters, reldepth, f_common, n_clusters, outpath):
    scores = pca["scores"]
    loadings = pca["loadings"]
    explained = pca["explained"]
    cum = pca["cum_explained"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # (a) PC1-PC2 coloured by cluster
    ax = axes[0, 0]
    base = plt.get_cmap("tab10")
    cmap_c = ListedColormap([base(i % 10) for i in range(max(n_clusters, 1))])
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=clusters, cmap=cmap_c,
                    s=12, alpha=0.8, linewidths=0)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("PC1 (%.1f%%)" % (explained[0] * 100))
    ax.set_ylabel("PC2 (%.1f%%)" % (explained[1] * 100 if explained.size > 1 else 0))
    ax.set_title("Sites in PC space — coloured by spectral cluster")
    cb = fig.colorbar(sc, ax=ax, ticks=range(1, n_clusters + 1), pad=0.02)
    cb.set_label("cluster")

    # (b) PC1-PC2 coloured by relative depth
    ax = axes[0, 1]
    sc2 = ax.scatter(scores[:, 0], scores[:, 1], c=reldepth, cmap="viridis",
                     s=12, alpha=0.8, linewidths=0)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("PC1 (%.1f%%)" % (explained[0] * 100))
    ax.set_ylabel("PC2 (%.1f%%)" % (explained[1] * 100 if explained.size > 1 else 0))
    ax.set_title("Same points — coloured by relative depth")
    cb2 = fig.colorbar(sc2, ax=ax, pad=0.02)
    cb2.set_label("relative depth (0 = tip → 1 = surface)")

    # (c) loadings across frequency (features are the frequency axis)
    ax = axes[1, 0]
    ax.axhline(0, color="0.85", lw=0.8)
    ax.plot(f_common, loadings[0], label="PC1", color="#08519c")
    if loadings.shape[0] > 1:
        ax.plot(f_common, loadings[1], label="PC2", color="#e6550d")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Loading")
    ax.set_title("PC loadings across frequency")
    ax.legend(fontsize=8)

    # (d) scree / cumulative
    ax = axes[1, 1]
    k = min(explained.size, 15)
    xs = np.arange(1, k + 1)
    ax.bar(xs, explained[:k] * 100, color="#9ecae1", label="per PC")
    ax.plot(xs, cum[:k] * 100, "o-", color="#08519c", label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Scree / cumulative variance")
    ax.set_xticks(xs)
    ax.set_ylim(0, 105)
    ax.legend(loc="center right", fontsize=8)

    fig.suptitle("Location standardization — PCA over %d recording sites"
                 % scores.shape[0], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    fs = sessions[0]["fs"]
    print("Experiments with LFP: %d (%s)"
          % (len(sessions), ", ".join(s["date"] for s in sessions)))

    f_common = canonical_freq_grid(fs, NPERSEG, FMIN, FMAX)
    notch_msg = ("%.0f Hz notch (Q=%.0f)" % (NOTCH_F0, NOTCH_Q)) if NOTCH_F0 else "no notch"
    print("Freq grid: %d bins %.2f-%.2f Hz | %s"
          % (f_common.size, f_common[0], f_common[-1], notch_msg))

    master = build_master_matrix(sessions, f_common, NPERSEG, NOVERLAP,
                                 notch_f0=NOTCH_F0, notch_Q=NOTCH_Q, log=LOG_POWER)
    M = master["M"]
    print("Master matrix: %s (n_sites, n_freq)" % (M.shape,))

    features = row_normalize(M) if ROW_NORMALIZE else (M - M.mean(0, keepdims=True))

    # hierarchical clustering of the sites (rows)
    Z = hierarchical_linkage(features, method=CLUSTER_METHOD, metric=CLUSTER_METRIC)
    n_clusters = int(min(N_CLUSTERS, M.shape[0]))
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    # PCA over sites
    pca = pca_sites(features)
    scores = pca["scores"]

    # figure 1: clustered master matrix + dendrogram (display the feature matrix so
    # the visible structure matches the dendrogram/clusters)
    f1 = os.path.join(fig_dir, "master_matrix_clustered.png")
    draw_clustered_heatmap(features, Z, f_common, n_clusters, f1)
    print("  wrote %s" % os.path.basename(f1))

    # figure 2: PCA
    f2 = os.path.join(fig_dir, "pca_sites.png")
    draw_pca(pca, clusters, master["reldepth"], f_common, n_clusters, f2)
    print("  wrote %s" % os.path.basename(f2))

    # CSVs
    with open(os.path.join(fig_dir, "site_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "channel_row", "ycoord_um", "reldepth",
                    "cluster", "PC1", "PC2"])
        for i in range(M.shape[0]):
            w.writerow([master["experiment"][i], master["channel_row"][i],
                        "%.1f" % master["ycoord"][i], "%.4f" % master["reldepth"][i],
                        int(clusters[i]), "%.5f" % scores[i, 0],
                        "%.5f" % (scores[i, 1] if scores.shape[1] > 1 else 0.0)])
    with open(os.path.join(fig_dir, "pca_variance.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["PC", "explained_percent", "cumulative_percent"])
        for j in range(pca["explained"].size):
            w.writerow([j + 1, "%.4f" % (pca["explained"][j] * 100),
                        "%.4f" % (pca["cum_explained"][j] * 100)])
    print("  wrote site_table.csv, pca_variance.csv")

    print("\nDone. %d sites from %d experiments, %d clusters.\n  %s"
          % (M.shape[0], len(sessions), n_clusters, fig_dir))


if __name__ == "__main__":
    main()

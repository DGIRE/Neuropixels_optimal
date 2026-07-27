# Location standardization — clustering recording sites by LFP spectrum

Compare recording **sites** across experiments by their LFP spectral signature, so
sites sampling the same kind of tissue group together **regardless of the probe's
insertion angle or which experiment they came from**.

Where the companion *Probe standardization* analysis treated each experiment as one
observation, here the unit is a single **recording site** (one retained LFP channel
from one experiment). Every site is one row of a pooled master matrix.

## What it does

1. **Load** every experiment with LFP from `np_aggregate.h5`.
2. **60 Hz notch** (`scipy.signal.iirnotch`, zero-phase) on each channel's time
   series to remove mains line noise before the spectrum is estimated.
3. **Master matrix** — pool every site from every experiment: row = one site
   (repeated depths across experiments are expected), column = frequency (Welch
   PSD, dB). Depth is kept per row as a label.
4. **Hierarchical clustering** of the sites (rows) by their spectra (Ward /
   Euclidean on each site's mean-subtracted spectral *shape*).
5. **Figures** (jet colormap):
   - `master_matrix_clustered.png` — the master matrix with rows reordered by the
     clustering, plus the row **dendrogram** on the y-axis.
   - `pca_sites.png` — PCA over sites: PC1–PC2 scatter (one point per site) colored
     by cluster and by relative depth, PC loadings across frequency, and a scree /
     cumulative-variance plot.
6. **CSVs**: `site_table.csv` (experiment, channel_row, ycoord_um, reldepth,
   cluster, PC1, PC2) and `pca_variance.csv`.

## Files

| File | Role |
|------|------|
| `location_standardization.py` | Functions: load, notch, master matrix, linkage, PCA. |
| `run_location_standardization.py` | Driver: builds both figures + CSVs. Edit the `CONFIG` block to change parameters. |
| `README.md` | This file. |

## Run (on the workstation where `np_aggregate.h5` lives)

```
python "C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Code\run_location_standardization.py"
```

Optional overrides: `python run_location_standardization.py <H5_PATH> <FIG_DIR>`

Requires `numpy scipy matplotlib h5py` (same env as the Optimized Python kernel).
Reads only the aggregate — no raw-data access needed.

## Key analysis choices (all in the `CONFIG` block)

- **`NOTCH_F0 = 60`, `NOTCH_Q = 30`** — the requested 60 Hz notch. Set `NOTCH_F0 = 0`
  to disable. (A residual narrow dip at 60 Hz is expected and, because it is the
  same for every site, does not drive clustering or PCA.)
- **`ROW_NORMALIZE = True`** — each site's mean dB is subtracted so clustering/PCA
  respond to the *shape* of its spectrum (its tissue signature) rather than an
  overall power/gain/reference offset that varies with experiment and depth. This
  is what lets sites be compared "regardless of insertion angle." Set `False` to
  keep absolute level.
- **`CLUSTER_METHOD = "ward"`, `N_CLUSTERS = 6`** — Ward linkage (forces Euclidean);
  the flat cut into `N_CLUSTERS` is used only for coloring the dendrogram and PCA
  points. Change `N_CLUSTERS` to taste; the dendrogram itself is independent of it.
- **Band 1–100 Hz**, Welch `NPERSEG = 512` (~0.49 Hz bins) at 250 Hz, jet colormap.
- The heatmap uses 2nd–98th-percentile color limits so the notch column doesn't
  compress the scale.

## Coverage note

Processes every date whose `has_lfp` flag is set. With the current
`np_aggregate.h5` that is whatever the last build wrote with LFP; after the
LFP-geometry fix + rebuild it will be all included sessions, and this code picks
them up automatically — just re-run.

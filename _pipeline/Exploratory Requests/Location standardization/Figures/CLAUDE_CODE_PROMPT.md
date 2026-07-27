# Prompt for Claude Code — run the location-standardization analysis

Copy everything in the block below and paste it to Claude Code running locally on
the workstation (where `np_aggregate.h5` lives).

---

```
Run the location-standardization LFP analysis. The code is already written and
verified — your job is to execute it against the real aggregate and confirm the
figures.

CONTEXT
- Goal: compare recording SITES across experiments by their LFP spectral
  signature, so sites in the same kind of tissue group together regardless of the
  probe's insertion angle. The unit of observation is one recording site (one LFP
  channel from one experiment), pooled across all experiments.
- Data: C:\Projects\Repos\Neuropixels\DATA\Aggregate\np_aggregate.h5  (one HDF5
  group per date; LFP is 32 depth channels @ 250 Hz, float32).
- Code: C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Code\
    location_standardization.py      (functions: load, 60 Hz notch, master matrix, linkage, PCA)
    run_location_standardization.py   (driver — CONFIG block at top; writes all figures)
- Figures go to: C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Figures\

IMPORTANT — do NOT `cd` into any subfolder. This project's .claude hooks resolve
relative to the shell cwd, and any cd (including `cd x && cmd`) silently disables
them. Run from the project root C:\Projects\Repos\Neuropixels using ABSOLUTE
paths only.

STEPS
1. Confirm the analysis environment has numpy, scipy, matplotlib, h5py (same env as
   the Optimized Python kernel). Do NOT pip-install anything (the guardrail hook
   blocks it and the env already has these). If something is missing, stop and tell
   me rather than installing.
2. From the project root, run the driver with its full absolute path:
       python "C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Code\run_location_standardization.py"
   It prints: experiments with LFP, the frequency grid + notch, the master-matrix
   shape (n_sites, n_freq), and each file as it is written.
3. Verify the Figures folder now contains:
     - master_matrix_clustered.png   (clustered master matrix + row dendrogram, jet)
     - pca_sites.png                 (PC1-PC2 scatter, loadings, scree)
     - site_table.csv, pca_variance.csv
   Open both PNGs and sanity-check:
     * master_matrix_clustered: rows are recording sites pooled across experiments,
       reordered by hierarchical clustering, with the dendrogram on the left; the
       60 Hz notch shows as a narrow suppressed column; jet colormap.
     * pca_sites: each point is one site; panels colored by cluster and by relative
       depth, PC1/PC2 loadings across frequency, and a scree plot.
4. Report back: how many sites and experiments were pooled, PC1/PC2 % variance, and
   whether the spectral clusters look like they correspond to depth/tissue bands
   (e.g. do sites from different experiments at similar tissue land in the same
   cluster).

TUNING (only if I ask)
All knobs are in the CONFIG block of run_location_standardization.py:
  NOTCH_F0/NOTCH_Q (line-noise notch), FMIN/FMAX, NPERSEG/NOVERLAP,
  ROW_NORMALIZE (True = compare spectral shape / tissue signature; False = keep
  absolute power), CLUSTER_METHOD, N_CLUSTERS, CMAP. Change there and re-run.

COVERAGE NOTE
The script processes every date whose has_lfp flag is set — however many the
current aggregate has. After the LFP-geometry fix + rebuild adds the remaining
sessions, re-run this same command to pool them in automatically.
```

---

## Optional follow-up

To persist the per-site cluster/PC assignments back into the aggregate, ask Claude
Code to write `site_table.csv` values into each date's `derived/` group via
`aggregate_io.Aggregate(path, "r+").insert_derived(...)`.

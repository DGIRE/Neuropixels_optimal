# Prompt for Claude Code — run the probe-standardization analysis

Copy everything in the block below and paste it to Claude Code running locally on
the workstation (where `np_aggregate.h5` and the raw data live).

---

```
Run the probe-standardization LFP analysis. The code is already written and
verified — your job is to execute it against the real aggregate and confirm the
figures.

CONTEXT
- Goal: informally assess whether Neuropixels probes were inserted at comparable
  angles across experiments, via the laminar (depth) distribution of LFP spectral
  power. For each experiment we build a depth (y) × frequency (x) power matrix,
  then run PCA across experiments so each experiment is a point in a 2-D plane.
- Data: C:\Projects\Repos\Neuropixels\DATA\Aggregate\np_aggregate.h5  (already
  built; one HDF5 group per date, LFP is 32 depth channels @ 250 Hz, float32).
- Code: C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Probe standardization\Code\
    probe_standardization.py      (functions: load, Welch depth×freq matrix, PCA)
    run_probe_standardization.py   (driver — CONFIG block at top; writes all figures)
- Figures go to: C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Probe standardization\Figures\

IMPORTANT — do NOT `cd` into any subfolder. This project's .claude hooks resolve
relative to the shell cwd, and any cd (including `cd x && cmd`) silently disables
them. Run from the project root C:\Projects\Repos\Neuropixels using ABSOLUTE
paths only.

STEPS
1. Confirm the analysis environment has numpy, scipy, matplotlib, h5py (the same
   env as the Optimized Python kernel). Do NOT pip-install anything (the guardrail
   hook blocks it and the kernel env already has these). If a package is somehow
   missing, stop and tell me rather than installing.
2. From the project root, run the driver with its full absolute path:
       python "C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Probe standardization\Code\run_probe_standardization.py"
   It prints: how many experiments have LFP, the common frequency grid, the
   feature-tensor shape, and each file as it is written.
3. Verify the Figures folder now contains:
     - depthfreq_<DATE>.png  (one per experiment that has LFP)
     - PCA_summary.png
     - depthfreq_montage.png
     - pca_coordinates.csv, pca_variance.csv
   Open PCA_summary.png and 2–3 individual figures and sanity-check them:
     * individual figures show depth on y (0 = tip → 32 = surface), frequency on
       x (1–100 Hz), jet colormap, date in the title, and the experiment's
       "PCA coordinates: PC1 = …, PC2 = …" written in the top-left box.
     * PCA_summary shows the PC1–PC2 scatter (labelled by date), PC1 & PC2
       loadings across the depth×freq matrix in jet, and cumulative % variance.
4. Report back: the list of dates that had LFP (vs. any that were skipped), the
   % variance explained by PC1 and PC2, and any experiments that sit as clear
   outliers in the PC1–PC2 scatter (candidate odd insertion angles).

COVERAGE NOTE
The script processes every date whose has_lfp flag is set. If the aggregate still
has the LFP-geometry bug unfixed, only ~16 of 24 dates carry LFP and the rest are
skipped automatically (that's expected, not an error). After the LFP-geometry fix
+ targeted rebuild, re-run this same command to get all 24.

TUNING (only if I ask)
All knobs are in the CONFIG block of run_probe_standardization.py:
  FMIN/FMAX (band), NPERSEG/NOVERLAP (frequency resolution), DEPTH_BINS,
  PER_EXPERIMENT_CENTER (True = compare laminar pattern; False = let absolute
  power drive PC1), STANDARDIZE, CMAP. Change there and re-run.
```

---

## Optional follow-up

If you want the PCA coordinates stored back into the aggregate itself (not just
written on the figures), the reader already supports it — ask Claude Code to, for
each date, call:

```python
from aggregate_io import Aggregate
agg = Aggregate(r"...\np_aggregate.h5", "r+")
agg.insert_note("<DATE>", "probe_std_pc", "PC1=<v>, PC2=<v>")
agg.close()
```

That persists a short note per date (read back with `agg.notes(date)`).

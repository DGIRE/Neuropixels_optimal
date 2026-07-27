# Probe standardization — depth × frequency LFP + PCA

Informal assessment of whether Neuropixels probes were inserted at comparable
angles across experiments, using the laminar (depth) distribution of LFP
spectral power in the cross-experiment aggregate `np_aggregate.h5`.

## What it does

For every experiment that has LFP in the aggregate:

1. **Depth × frequency matrix** — Welch power spectrum per depth channel, giving a
   matrix with **depth on the y-axis** and an **FFT (frequency) running along the
   x-axis**. Rows are ordered tip (deep) → surface; power is in dB.
2. **PCA across experiments** — each experiment's matrix is one observation.
   Experiments are projected as points in the PC1–PC2 plane. Similar laminar
   spectral profiles cluster; a probe at an odd angle separates out.

## Files

| File | Role |
|------|------|
| `probe_standardization.py` | The suite of functions (load, PSD matrix, PCA). Pure functions, no file I/O — importable/reusable. |
| `run_probe_standardization.py` | Driver: builds all figures + CSVs. Edit the `CONFIG` block to change parameters. |
| `README.md` | This file. |

## Run (on the workstation where `np_aggregate.h5` lives)

```
python "C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Probe standardization\Code\run_probe_standardization.py"
```

Optional path overrides: `python run_probe_standardization.py <H5_PATH> <FIG_DIR>`

Requires `numpy scipy matplotlib h5py` (same env as the Optimized Python kernel).
It reads only the aggregate — **no raw-data access needed**.

## Outputs (into the Figures folder)

- `depthfreq_<DATE>.png` — one per experiment; depth × frequency power (jet),
  labelled by date, with that experiment's **PC1/PC2 coordinates written in**.
- `PCA_summary.png` — PC1–PC2 scatter, PC1 & PC2 loadings across the depth ×
  frequency matrix (jet), and cumulative % variance explained.
- `depthfreq_montage.png` — contact sheet of all experiments on a common scale.
- `pca_coordinates.csv`, `pca_variance.csv`.

## Key analysis choices (all in `run_probe_standardization.py` CONFIG)

- **Common frequency grid** — a fixed Welch `NPERSEG` (512 @ 250 Hz → ~0.49 Hz
  bins). Every session shares LFP_fs, so matrices are directly comparable; short
  or ragged sessions are interpolated onto the common grid so PCA always aligns.
- **Relative depth (32 bins, tip→surface)** — the laminar axis is resampled to a
  common relative-depth grid so experiments with different physical depth ranges
  (exactly what insertion angle changes) can be compared. Individual figures
  still annotate the true `ycoord` µm span.
- **`PER_EXPERIMENT_CENTER = True`** — each experiment's mean dB is subtracted
  before PCA so PC1 captures the *pattern* of laminar/spectral distribution, not
  overall gain/reference offset. Set `False` to let absolute power drive PC1.
- **Band 1–100 Hz**, dB power, jet colormap.

## Note on coverage

The script processes every date whose `has_lfp` flag is set. In the current
`np_aggregate.h5` that is the subset built with `lfp="ok"` (~16 of 24 dates);
the rest failed the LFP-geometry step (Kilosort coupling bug). After that fix +
a targeted rebuild, all 24 will carry LFP and this code picks them up
automatically — just re-run it.

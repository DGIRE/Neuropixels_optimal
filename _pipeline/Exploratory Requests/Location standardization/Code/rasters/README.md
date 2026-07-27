# LFP + unit rasters (+ sniff) display

Show a specified time window from a specified experiment (or group) with the 32 LFP
channels, spike rasters for all units organized by depth, and — when present — the
sniff signal. Three LFP versions per experiment: **raw**, **theta**, **gamma**.

## Layout

Bottom (probe tip) to top (surface):

- Each LFP channel is drawn as a trace, depth-colored, stacked by depth.
- **Under each LFP trace** are the spike rasters of the units at that LFP site's
  depth *or below it, but above the next (deeper) LFP site* — i.e. each LFP trace
  owns the depth interval down to the next deeper site, and its units are rastered
  beneath it (deeper units lower).
- If the experiment has a **strong sniff signal**, it's drawn across the top.

## Files

| File | Role |
|------|------|
| `raster_display.py` | Functions: load LFP+spikes+sniff, band-pass, sniff-strength, depth binning, per-unit spike extraction. |
| `run_raster_examples.py` | Driver: makes the raw/theta/gamma figures. CONFIG + command-line flags. |
| `README.md` | This file. |

## Run (on the workstation where `np_aggregate.h5` lives)

```
python "C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Code\rasters\run_raster_examples.py" --dates 06-21-2022 --start 120 --end 130
```

Flags: `--dates` (comma-separated MM-DD-YYYY or `all`), `--start`/`--end` seconds,
`--theta LO HI`, `--gamma LO HI`, `--sniff auto|on|off`, `--h5`, `--figdir`.
Omitted flags use the CONFIG defaults (all LFP experiments, 0–10 s, theta 4–12,
gamma 30–80, sniff auto).

Output → `...\Figures\Rasters\`: `<DATE>_raw_<t0>-<t1>s.png`,
`<DATE>_theta_...`, `<DATE>_gamma_...`.

Requires `numpy scipy matplotlib h5py`. Reads only the aggregate.

## IMPORTANT — per-unit rasters need `spikeClusters`

True per-unit rasters (one row per unit) require the aggregate's `spikeClusters`
array (the spike→cluster map), which was **added to `build_aggregate.py` on
2026-07-26**. Sessions built before that don't have it yet.

- If a session has `spikeClusters`, the figure shows **one raster row per unit**,
  grouped by depth bin.
- If not, it falls back to a **depth-binned population raster** (one row of all
  spikes per LFP depth bin) and says so in the title.

To populate `spikeClusters`, rebuild the spike-bearing sessions, e.g.:

```
python -c "from build_aggregate import build; build(only_dates=['06-21-2022'], resume=False)"
```

(or a full `python build_aggregate.py rebuild`). The same rebuild that pulls in the
LFP-geometry / Kilosort fixes will also populate `spikeClusters`.

## Notes

- Sniff inclusion is automatic (`--sniff auto`): shown when the SNF band/broadband
  amplitude ratio ≥ `SNIFF_STRENGTH_MIN` (0.35). Force with `--sniff on|off`.
  'nosniff' or flat-SNF sessions fall below threshold and are omitted.
- Units with no spikes in the window are dropped (`SHOW_EMPTY_UNITS=False`) to
  declutter; flip in CONFIG to show every unit as a row.
- Depths (LFP `lfp_ycoord`, `unitDepths`, `spikeDepths`) are all microns from the
  probe tip, so they bin directly. If `unitDepths` is missing it's computed as the
  mean `spikeDepths` per cluster.
- Gamma above ~90 Hz is near the aggregate's anti-alias ceiling (use `--gamma 30 80`).

# Band-filtered LFP time-series examples

Examine a specified time window from a specified experiment (or group of
experiments) in the aggregate. For each selected experiment it writes **two**
figures: all 32 LFP channels filtered to **theta**, and all 32 filtered to
**gamma**, stacked by depth (tip at bottom → surface at top) and colored by depth.

## Files

| File | Role |
|------|------|
| `band_timeseries.py` | Functions: list/load LFP by date, zero-phase band-pass, window indexing. |
| `run_band_examples.py` | Driver: makes the theta + gamma figures. CONFIG block + command-line flags. |
| `README.md` | This file. |

## Run (on the workstation where `np_aggregate.h5` lives)

```
# defaults from the CONFIG block (all LFP experiments, 0–10 s)
python "C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Code\visualization\run_band_examples.py"

# pick experiments and a window
python "...\visualization\run_band_examples.py" --dates 06-21-2022,10-05-2021 --start 120 --end 130

# custom bands / paths
python "...\visualization\run_band_examples.py" --dates all --start 0 --end 8 --theta 2 12 --gamma 65 100
```

Flags: `--dates` (comma-separated MM-DD-YYYY, or `all`), `--start`/`--end`
(seconds), `--theta LO HI`, `--gamma LO HI`, `--h5`, `--figdir`. Anything not
passed falls back to the CONFIG block at the top of `run_band_examples.py`.

Output → `...\Location standardization\Figures\Examples\`:
`<DATE>_theta_<t0>-<t1>s.png` and `<DATE>_gamma_<t0>-<t1>s.png`.

Requires `numpy scipy matplotlib h5py` (same env as the Optimized Python kernel).
Reads only the aggregate — no raw-data access needed.

## Notes on the analysis

- **Bands.** Defaults: theta 4–12 Hz, gamma 30–80 Hz (robust, well inside the
  aggregate's DC..~100 Hz passband). Your paper bands (theta 2–12, gamma 65–100)
  are easy to set with `--theta 2 12 --gamma 65 100`; note gamma above ~90 Hz sits
  near the aggregate's anti-alias ceiling, so it may look attenuated.
- **Filtering** is a 4th-order Butterworth band-pass applied zero-phase
  (`filtfilt`) to the **full** trace, then the window is sliced out — so there are
  no filter edge artefacts at the window boundaries.
- **Display.** Channels are offset vertically by a common amount (median per-channel
  std × `SPACING_SD`); a scale bar shows one median-std in µV. y-ticks label the AP
  row and depth (µm) of a subset of channels.
- Traces are in microvolts (the aggregate LFP is stored in volts).

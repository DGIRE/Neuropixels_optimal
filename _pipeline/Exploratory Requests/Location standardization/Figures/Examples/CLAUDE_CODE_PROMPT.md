# Prompt for Claude Code — make band-filtered LFP time-series examples

Copy the block below and paste it to Claude Code running locally on the workstation
(where `np_aggregate.h5` lives). Edit the dates and the time window on the command
line to whatever you want to look at.

---

```
Make band-filtered LFP time-series example figures. The code is already written and
verified — run it against the real aggregate.

CONTEXT
- For each selected experiment it writes TWO figures: all 32 LFP channels filtered
  to theta, and all 32 filtered to gamma, stacked by depth (tip at bottom → surface
  at top).
- Data: C:\Projects\Repos\Neuropixels\DATA\Aggregate\np_aggregate.h5
- Code: C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Code\visualization\
    band_timeseries.py       (functions: list/load LFP by date, band-pass, windowing)
    run_band_examples.py      (driver; CONFIG block + command-line flags)
- Figures go to: C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Figures\Examples\

IMPORTANT — do NOT `cd` into any subfolder. This project's .claude hooks resolve
relative to the shell cwd, and any cd (including `cd x && cmd`) silently disables
them. Run from the project root C:\Projects\Repos\Neuropixels using ABSOLUTE
paths only.

STEPS
1. Confirm the env has numpy, scipy, matplotlib, h5py (same env as the Optimized
   Python kernel). Do NOT pip-install anything; if something is missing, stop and
   tell me.
2. Run the driver from the project root, choosing the experiments and window you
   want. Examples (edit dates/start/end to taste):
       # a 10 s window from two experiments
       python "C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Code\visualization\run_band_examples.py" --dates 06-21-2022,10-05-2021 --start 120 --end 130
       # every LFP experiment, first 8 s, using the lab's paper bands
       python "C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Code\visualization\run_band_examples.py" --dates all --start 0 --end 8 --theta 2 12 --gamma 65 100
   Flags: --dates (comma-separated MM-DD-YYYY or 'all'), --start/--end seconds,
   --theta LO HI, --gamma LO HI. Omitted flags use the CONFIG defaults in the driver
   (all LFP experiments, 0–10 s, theta 4–12, gamma 30–80).
3. Confirm the Examples folder now has, for each experiment,
   <DATE>_theta_<t0>-<t1>s.png and <DATE>_gamma_<t0>-<t1>s.png. Open one theta and
   one gamma figure and check: 32 channels stacked by depth, coloured tip→surface,
   theta shows a slow rhythm and gamma a faster one, with a µV scale bar.
4. Report which experiments/window you plotted and anything notable (e.g. a depth
   where theta or gamma amplitude peaks).

TIP
If a requested window is longer than a recording, that experiment is clipped/ skipped
with a message and the rest still run. Very long windows (>~30 s) make dense figures;
5–15 s reads best.
```

---

## Notes

- Defaults live in the CONFIG block of `run_band_examples.py`; the flags above just
  override them per run.
- Gamma above ~90 Hz is near the aggregate's anti-alias ceiling and may look
  attenuated; 30–80 Hz is a safe default.

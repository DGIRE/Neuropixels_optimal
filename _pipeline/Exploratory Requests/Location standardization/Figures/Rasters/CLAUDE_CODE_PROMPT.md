# Prompt for Claude Code — LFP + unit-raster (+ sniff) display

Copy the block below and paste it to Claude Code on the workstation (where
`np_aggregate.h5` lives). Edit the dates / window to what you want.

---

```
Make the LFP + unit-raster (+ sniff) display figures. The code is already written
and verified — run it against the real aggregate.

CONTEXT
- For each selected experiment it writes THREE figures (raw, theta, gamma LFP), each
  stacking the 32 LFP channels tip->surface with the spike rasters of all units drawn
  under the LFP trace at their depth (units at that LFP site's depth or below it, but
  above the next deeper site). A strong sniff signal is drawn across the top.
- Data: C:\Projects\Repos\Neuropixels\DATA\Aggregate\np_aggregate.h5
- Code: C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Code\rasters\
    raster_display.py        (load/filter/bin/spike helpers)
    run_raster_examples.py    (driver; CONFIG + flags)
- Figures go to: C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Figures\Rasters\

IMPORTANT — do NOT `cd` into any subfolder. This project's .claude hooks resolve
relative to the shell cwd, and any cd (including `cd x && cmd`) silently disables
them. Run from the project root C:\Projects\Repos\Neuropixels using ABSOLUTE
paths only.

PER-UNIT RASTERS NEED spikeClusters
True per-unit rasters require the aggregate's `spikeClusters` array, added to
build_aggregate.py on 2026-07-26. If the sessions you want were built before that,
first repopulate them so the rasters are per-unit rather than a depth-binned
population fallback:
    python -c "import sys; sys.path.insert(0, r'C:\Projects\Repos\Neuropixels\DATA\Aggregate'); from build_aggregate import build; build(only_dates=['06-21-2022'], resume=False)"
(replace the date, or run a full `python build_aggregate.py rebuild` to do all of
them). This reprocesses the raw data (multi-hour for the full set) — only needed
once, and only for the sessions you want per-unit rasters for. Skip it if the
aggregate was already rebuilt after 2026-07-26.

STEPS
1. Confirm the env has numpy, scipy, matplotlib, h5py (same env as the Optimized
   Python kernel). Do NOT pip-install; if something's missing, stop and tell me.
2. Run the driver from the project root, choosing experiments and a window:
       python "C:\Projects\Repos\Neuropixels\_pipeline\Exploratory Requests\Location standardization\Code\rasters\run_raster_examples.py" --dates 06-21-2022 --start 120 --end 130
   Flags: --dates (comma list or 'all'), --start/--end seconds, --theta LO HI,
   --gamma LO HI, --sniff auto|on|off. It prints, per experiment, whether spikes are
   per-unit or population and the sniff ratio.
3. Confirm the Rasters folder has <DATE>_raw/theta/gamma_<t0>-<t1>s.png for each
   experiment. Open one and check: LFP traces stacked by depth (tip at bottom), unit
   rasters beneath each trace, sniff across the top if present, µV scale bar.
4. Report: which experiments/window, whether rasters were per-unit or population, and
   whether sniff was shown; note any depth band where units cluster.

TIP: 5–15 s windows read best. Titles say "per unit" or "population (…rebuild…)" so
you know which mode you got.
```

---

## Note

If figures say "population (no spikeClusters — rebuild to get per-unit rows)", the
aggregate for that session predates the `spikeClusters` addition — run the rebuild
in the block above for that date to get true per-unit rasters.

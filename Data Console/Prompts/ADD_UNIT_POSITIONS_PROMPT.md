# Prompt for Claude Code — add precise per-unit coordinates to the aggregate

Copy the block below and paste it to Claude Code on the workstation (where the
Kilosort dirs on X: are reachable). This is a one-time pass; re-run it only after
rebuilding or adding sessions.

---

```
Run the unit-position augmentation. It adds precise per-unit (x, y) coordinates and
shank to np_aggregate.h5 in place, so the Data Console can place units on the
correct shank. It reads ONLY the Kilosort output (templates + channel positions),
not the raw .ap.bin, so it is fast (seconds per session) — NOT a build_aggregate
rebuild.

- Script: C:\Projects\Repos\Neuropixels\DATA\Aggregate\add_unit_positions.py
- Aggregate: C:\Projects\Repos\Neuropixels\DATA\Aggregate\np_aggregate.h5
- It reuses build_aggregate's Kilosort-dir resolution (the 09-14-2022 override and
  the complete-sort auto-repair), so multi-sort days pick the right sort.

IMPORTANT — do NOT `cd` into any subfolder. This project's .claude hooks resolve
relative to the shell cwd, and any cd (including `cd x && cmd`) silently disables
them. Run from the project root C:\Projects\Repos\Neuropixels using ABSOLUTE
paths only. (Running the script by its full path still lets it import
build_aggregate, because Python adds the script's own folder to the path.)

STEPS
1. Confirm the env has numpy + h5py (same env as the Optimized Python kernel). Do
   NOT pip-install anything.
2. Make sure X: (the raw/Kilosort data) is reachable — the script needs the
   Kilosort dirs, though not the big .ap.bin files.
3. From the project root, run:
       python "C:\Projects\Repos\Neuropixels\DATA\Aggregate\add_unit_positions.py"
   Optional custom paths:  python "...\add_unit_positions.py" "<DATA_ROOT>" "<OUT_H5>"
   It prints one line per session: how many units were positioned and across how
   many shanks, plus which Kilosort dir it used.
4. Report back: which sessions were augmented (and any skipped for missing Kilosort),
   and whether multi-shank sessions (e.g. 09-14-2022) now show units on more than one
   shank.

AFTER IT RUNS
Open the Data Console (python "...\Data Console\Code\np_console.py"); multi-shank
probe maps will now place unit dots on the correct shank and the "approximate" note
will be gone. No console changes are needed — it uses the new coordinates
automatically when present.
```

---

## Note

The script writes `unitXcoords`, `unitYcoords`, and `unitShank` into each spike
session's group. Re-running it just overwrites them. If a session has no complete
Kilosort dir, it is skipped (its units stay depth-only in the console).

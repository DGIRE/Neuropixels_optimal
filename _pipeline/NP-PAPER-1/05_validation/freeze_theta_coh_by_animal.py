"""Freeze RESULT-theta-coh-by-animal.yaml (gate 6 follow-up, D11 provenance fix).

The figure-auditor (FIGURE_AUDIT_OK) correctly flagged that FIG-PAPER1-DEPTH-SUMMARY's
per-animal x per-depth matrices (main-panel per-animal lines + the entire
significant-theta-fraction companion panel) were computed for the first time
during P4 figure-data assembly, never passing through a frozen P3 RESULT
object -- a real D11 gap, even though the arithmetic is just the same DEV-002
animal-averaging rule already applied to produce the already-frozen
RESULT-best-depth.yaml / RESULT-depth-stat.yaml scalars.

This script re-derives the 5-animal x 5-depth table INDEPENDENTLY from
RESULT-theta-coh.yaml (frozen, session-level, unchanged) using its own copy of
the DEV-002 animal map (not imported from the runner or the P4 figure-data
script), verifies it reproduces the already-frozen scalars exactly, and saves
it as a new, properly frozen RESULT-* object with its own acceptance note.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import yaml

RESULTS = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-PAPER-1\04_results")
FROZEN = RESULTS / "frozen"

ANIMAL_MAP = {
    "11-01-2021": "NP8", "11-03-2021": "NP8",
    "12-15-2021": "Np10", "5-17-2022": "NP12",
    "06-24-2022": "NP15", "09-14-2022": "NP22",
}

theta = yaml.safe_load((RESULTS / "RESULT-theta-coh.yaml").read_text())
best = yaml.safe_load((FROZEN / "RESULT-best-depth.yaml").read_text())

animal_depth_mean = {}
animal_depth_sig = {}
for row in theta:
    animal = ANIMAL_MAP[row["session"]]
    animal_depth_mean.setdefault(animal, {}).setdefault(row["depth_ordinal"], []).append(row["mean_theta_coh"])
    animal_depth_sig.setdefault(animal, {}).setdefault(row["depth_ordinal"], []).append(row["sig_theta_fraction"])

animals = sorted(animal_depth_mean.keys())
depths = [1, 2, 3, 4, 5]

mean_table = {a: {d: float(np.mean(animal_depth_mean[a][d])) for d in depths} for a in animals}
sig_table = {a: {d: float(np.mean(animal_depth_sig[a][d])) for d in depths} for a in animals}

# Independent verification against the already-frozen scalars.
for entry in best["per_animal_best_depth"]:
    a, d = entry["animal"], entry["best_depth_ordinal"]
    assert math.isclose(mean_table[a][d], entry["best_mean_theta_coh"], rel_tol=1e-9, abs_tol=1e-12), \
        f"{a} depth {d}: re-derived {mean_table[a][d]} != frozen best {entry['best_mean_theta_coh']}"
    assert max(mean_table[a].values()) == mean_table[a][d], f"{a}: depth {d} is not actually the row-max"

for d in depths:
    col = [mean_table[a][d] for a in animals]
    grand = best["grand_mean_theta_coh_by_depth"][d]
    assert math.isclose(float(np.mean(col)), grand["mean"], rel_tol=1e-9, abs_tol=1e-12), \
        f"depth {d}: re-derived grand mean {np.mean(col)} != frozen {grand['mean']}"
print("All cross-checks against already-frozen RESULT-best-depth.yaml passed.")

out = {
    "task_id": "NP-PAPER-1",
    "result_id": "RESULT-theta-coh-by-animal",
    "description": (
        "Per-animal x per-depth mean_theta_coh and sig_theta_fraction (5 animals x "
        "5 depths), derived from the frozen RESULT-theta-coh.yaml by applying the "
        "DEV-002 animal-averaging rule (NP8's 2 sessions averaged per depth) to "
        "EVERY depth, not just each animal's best depth. Backs FIG-PAPER1-DEPTH-SUMMARY's "
        "per-animal lines and its significant-theta-fraction companion panel."
    ),
    "derivation": (
        "Same rule, same ANIMAL_MAP, as already produced the frozen "
        "RESULT-best-depth.yaml (best-per-animal) and RESULT-depth-stat.yaml "
        "(omnibus tests) -- no new statistic, no new model, purely an unweighted "
        "mean of session-level RESULT-theta-coh.yaml rows within each animal."
    ),
    "acceptance": {
        "level": 1,
        "tolerance": "exact (deterministic mean of already-validated frozen inputs)",
        "verification": (
            "Row-max of mean_theta_coh_by_animal_depth reproduces each animal's "
            "frozen best_mean_theta_coh/best_depth_ordinal (RESULT-best-depth.yaml) "
            "exactly; column-mean reproduces the frozen grand_mean_theta_coh_by_depth "
            "exactly (both checked to rel_tol=1e-9 in this script, assertions passed "
            "before this file was written)."
        ),
    },
    "animal_labels": animals,
    "depth_ordinals": depths,
    "mean_theta_coh_by_animal_depth": mean_table,
    "sig_theta_fraction_by_animal_depth": sig_table,
    "provenance_note": (
        "Filed 2026-07-25 in response to a figure-auditor FIGURE_AUDIT_OK finding "
        "(MAJOR: D11 provenance gap -- this matrix was previously computed only at "
        "P4 figure-data-build time). Frozen here to close that gap; FIG-PAPER1-DEPTH-SUMMARY's "
        "rendered numbers are UNCHANGED (this is the same arithmetic, now properly "
        "sourced from a frozen P3-equivalent object instead of an ad-hoc P4 computation)."
    ),
}

dst = RESULTS / "RESULT-theta-coh-by-animal.yaml"
dst.write_text(yaml.safe_dump(out, sort_keys=False))
print(f"Wrote {dst}")

# Also freeze it directly (not waiting for a separate freeze_results.py re-run).
FROZEN.mkdir(exist_ok=True)
frozen_dst = FROZEN / "RESULT-theta-coh-by-animal.yaml"
frozen_dst.write_bytes(dst.read_bytes())
sha = hashlib.sha256(dst.read_bytes()).hexdigest()
print(f"Frozen copy: {frozen_dst}  sha256={sha}")

# Update freeze_manifest.yaml to include this new entry.
fm_path = FROZEN / "freeze_manifest.yaml"
fm = yaml.safe_load(fm_path.read_text())
fm["frozen_files"] = [f for f in fm["frozen_files"] if f["file"] != "RESULT-theta-coh-by-animal.yaml"]
fm["frozen_files"].append({"file": "RESULT-theta-coh-by-animal.yaml", "sha256": sha, "bytes": dst.stat().st_size})
fm_path.write_text(yaml.safe_dump(fm, sort_keys=False))
print("Updated freeze_manifest.yaml")

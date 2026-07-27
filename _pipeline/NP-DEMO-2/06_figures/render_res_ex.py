"""
render_res_ex.py -- FIG-DEMO2-RES-EX: example units with the strongest
ethanol-induced change in sniff phase-locking (contract_spec.yaml figure_id
FIG-DEMO2-RES-EX).

Reads figdata_FIG-DEMO2-RES-EX.yaml (frozen, traces to RESULT-delta-mrl +
RESULT-psth-examples + RESULT-unit-locations: the top k=5 units by
|delta_MRL|). Presentation-only for the PSTH panels (every plotted number is
taken verbatim from the frozen figdata).

The probe-location panel per example unit is produced by calling the
validated kernel function plot_unit_locations(D) on that unit's session
D-dict (loaded via load_experiment_data + or_validate_files + the HAZ-05
_fix_ks_dir helper, reused -- not copied -- from np_demo2_analysis.py), then
overlaying a highlight marker at the example unit's own depth (taken
directly from the frozen figdata probe_location block, per
RESULT-unit-locations' level-1 exact-coordinate acceptance criterion -- the
highlight coordinate is not recomputed from D). No kernel file is modified;
plot_unit_locations is called exactly as published.

DEV-001: all 5 examples are from the degenerate 2021-11-03 session (see
08_traceability/deviations.yaml) -- a warning text box is included.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_common import load_yaml, savefig, COLOR_ETHANOL, COLOR_CONTROL, MIN_FONT

# --------------------------------------------------------------------------
# Kernel wiring (sibling import only, per task instructions)
# --------------------------------------------------------------------------
_KERNEL_DIR = r"C:\Projects\Repos\Neuropixels\Optimized Python"
_SOFTWARE_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_software"
if _KERNEL_DIR not in sys.path and os.path.isdir(_KERNEL_DIR):
    sys.path.insert(0, _KERNEL_DIR)
if _SOFTWARE_DIR not in sys.path and os.path.isdir(_SOFTWARE_DIR):
    sys.path.insert(0, _SOFTWARE_DIR)

from load_experiment_data import load_experiment_data                # noqa: E402
from lib.or_validate_files import or_validate_files                   # noqa: E402
from plot_unit_locations import plot_unit_locations                   # noqa: E402
# _fix_ks_dir + unit_probe_location: reused (not copied/reimplemented) from
# the already-validated NP-DEMO-2 analysis module (RESULTS_FROZEN state).
from np_demo2_analysis import _fix_ks_dir, unit_probe_location         # noqa: E402

DATA_ROOT = r"C:\Projects\Repos\Neuropixels\DATA"
SESSION_DIRS = {
    "2021-11-01": os.path.join(DATA_ROOT, "11-01-2021"),
    "2021-11-03": os.path.join(DATA_ROOT, "11-03-2021"),
    "2021-12-15": os.path.join(DATA_ROOT, "12-15-2021"),
    "2022-05-17": os.path.join(DATA_ROOT, "5-17-2022"),
    "2022-06-24": os.path.join(DATA_ROOT, "06-24-2022"),
    "2022-09-14": os.path.join(DATA_ROOT, "09-14-2022"),
}


def _load_session_D(session_date: str) -> dict:
    session_dir = SESSION_DIRS[session_date]
    files, missing = or_validate_files(session_dir, strict=False)
    files, missing = _fix_ks_dir(files, missing)
    if missing:
        raise FileNotFoundError(
            f"Required files missing in {session_dir!r}:\n  " + "\n  ".join(missing)
        )
    return load_experiment_data(files)


def _shanks_present(D: dict) -> np.ndarray:
    """Sorted unique shank numbers across ALL units in D, using the same
    (already-validated) per-unit shank assignment as RESULT-unit-locations,
    just to know which of plot_unit_locations(D)'s probe-map axes (ordered
    by ascending shank) corresponds to a given unit -- presentation lookup
    only, not a new statistic."""
    unit_ids = np.asarray(D["unitIDs"]).ravel()
    shanks = [unit_probe_location(D, int(uid))["shank"] for uid in unit_ids]
    return np.unique(np.asarray(shanks, dtype=int))


def _probe_inset(D: dict, shanks_present: np.ndarray, shank: int, depth_um: float,
                  unit_id: int):
    """Call the validated kernel plot_unit_locations(D), highlight this
    unit's position on its shank's probe-map axis, render to an RGBA array
    and crop tightly to that one shank's axis (plus its own title/ticks) for
    a legible inset (no kernel/state mutation persists past this call)."""
    fig_probe = plot_unit_locations(D)
    fig_probe.set_dpi(220)  # crisper crop; only affects this throwaway render
    axis_idx = int(np.searchsorted(shanks_present, shank))
    axis_idx = min(axis_idx, len(fig_probe.axes) - 1)
    ax_hl = fig_probe.axes[axis_idx]
    ax_hl.scatter([0], [depth_um], s=320, facecolors="none", edgecolors="red",
                  linewidths=2.8, zorder=10)
    ax_hl.annotate(f"unit {unit_id}", (0, depth_um), textcoords="offset points",
                    xytext=(9, 0), color="red", fontsize=9, fontweight="bold", zorder=11)

    fig_probe.canvas.draw()
    renderer = fig_probe.canvas.get_renderer()
    buf = np.asarray(fig_probe.canvas.buffer_rgba())
    h, w, _ = buf.shape

    bbox = ax_hl.get_tightbbox(renderer)
    pad = 14
    x0 = max(int(bbox.x0) - pad, 0)
    x1 = min(int(bbox.x1) + pad, w)
    # matplotlib bbox y is bottom-up; the RGBA buffer's row 0 is the image top
    row_top = max(h - int(bbox.y1) - pad, 0)
    row_bottom = min(h - int(bbox.y0) + pad, h)
    crop = buf[row_top:row_bottom, x0:x1]

    plt.close(fig_probe)
    return crop


def render():
    d = load_yaml("figdata_FIG-DEMO2-RES-EX.yaml")
    examples = d["examples"]
    n = len(examples)

    # Cache one D-dict (and its shanks_present) per session; all 5 examples
    # are from 2021-11-03 per DEV-001, but the code is written to generalize.
    D_cache: dict[str, dict] = {}
    shanks_cache: dict[str, np.ndarray] = {}

    fig = plt.figure(figsize=(11.5, 3.4 * n))
    gs = gridspec.GridSpec(n, 2, width_ratios=[1.5, 1.0], figure=fig)

    for i, ex in enumerate(examples):
        session = ex["session"]
        unit_id = ex["unit_id"]
        abs_delta = ex["abs_delta_mrl"]
        centers = np.asarray(ex["centers_phase"], dtype=float)
        psth_eth = np.asarray(ex["psth_eth"], dtype=float)
        psth_ctrl = np.asarray(ex["psth_ctrl"], dtype=float)
        loc = ex["probe_location"]

        # ---- Left: sniff-locked PSTH -------------------------------------
        ax_psth = fig.add_subplot(gs[i, 0])
        ax_psth.plot(centers, psth_eth, color=COLOR_ETHANOL, linestyle="-",
                      lw=1.6, label="ethanol")
        ax_psth.plot(centers, psth_ctrl, color=COLOR_CONTROL, linestyle="--",
                      lw=1.6, label="control")
        ax_psth.set_xlim(0.0, 1.0)
        y_top = max(1.0, float(np.nanmax(psth_eth)), float(np.nanmax(psth_ctrl))) * 1.08
        ax_psth.set_ylim(0.0, y_top)
        ax_psth.set_xlabel("sniff-cycle phase (0-1)")
        ax_psth.set_ylabel("spike rate (Hz)")
        ax_psth.set_title(f"Unit {unit_id} ({session}) |ΔMRL|={abs_delta:.3f}",
                            fontsize=MIN_FONT + 1)
        ax_psth.legend(loc="upper right", fontsize=MIN_FONT - 1, frameon=False)
        ax_psth.spines["top"].set_visible(False)
        ax_psth.spines["right"].set_visible(False)

        # ---- Right: probe-location inset (plot_unit_locations kernel) ----
        if session not in D_cache:
            D_cache[session] = _load_session_D(session)
            shanks_cache[session] = _shanks_present(D_cache[session])
        D = D_cache[session]
        shanks_present = shanks_cache[session]

        buf = _probe_inset(D, shanks_present, int(loc["shank"]),
                             float(loc["unit_depth_um"]), unit_id)
        ax_probe = fig.add_subplot(gs[i, 1])
        ax_probe.imshow(buf, aspect="auto")
        ax_probe.axis("off")
        ax_probe.set_title(
            f"shank {loc['shank']}, depth {loc['unit_depth_um']:.0f} µm",
            fontsize=MIN_FONT - 1,
        )

    fig.suptitle(
        "Example units with the strongest ethanol-induced change in sniff "
        "phase-locking (top k=5 by |ΔMRL|)",
        fontsize=MIN_FONT + 3,
    )

    if d.get("all_examples_from_2021_11_03"):
        warning = d.get("dev001_warning", d.get("dev001_flag", ""))
        fig.text(
            0.01, 0.005,
            "DEV-001 WARNING: " + warning,
            ha="left", va="bottom", fontsize=MIN_FONT - 1, color="red", wrap=True,
            bbox=dict(boxstyle="round", fc="#fff0f0", ec="red", alpha=0.9),
        )

    caption = (
        "Selection criterion: the k=5 units with the largest |delta_MRL| = "
        "|MRL_ethanol - MRL_control| across all included units and sessions "
        "(RESULT-delta-mrl, OD-2). Probe-location panels: plot_unit_locations(D) "
        "for that unit's session, highlighted (red circle) at the example unit's "
        "own depth/shank from RESULT-unit-locations."
    )
    fig.text(0.01, 0.03, caption, ha="left", va="bottom", fontsize=MIN_FONT - 1, wrap=True)

    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    return savefig(fig, "FIG-DEMO2-RES-EX")


if __name__ == "__main__":
    png, pdf = render()
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")

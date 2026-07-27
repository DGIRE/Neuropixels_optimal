"""
render_res_exp.py -- FIG-DEMO2-RES-EXP: ethanol vs control phase-locking of
OB units to the SNF signal, ANIMAL (experiment) level (contract_spec.yaml
figure_id FIG-DEMO2-RES-EXP).

Reads figdata_FIG-DEMO2-RES-EXP.yaml (frozen, traces to RESULT-mrl-animal +
RESULT-stat-animal). Presentation-only: every plotted number is taken
verbatim from the frozen figdata; no statistic is recomputed here.

Panel A: paired per-animal mean MRL (ethanol vs control), dots connected by
lines, group means overlaid as horizontal bars.
Panel B: per-animal ethanol-vs-control scatter (phase data is not part of
this figdata object, so the sketch's polar-inset fallback -- a simple
per-animal ethanol/control scatter -- is used, per the render-spec).
DEV-001: 2021-11-03 (degenerate control-condition MRL, see
08_traceability/deviations.yaml) is flagged with a red star marker and a
caption footnote in both panels.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_common import load_yaml, savefig, COLOR_ETHANOL, COLOR_CONTROL, COLOR_DEV001, MIN_FONT, fmt_p


def render():
    d = load_yaml("figdata_FIG-DEMO2-RES-EXP.yaml")
    sessions = d["sessions"]
    stat = d["stat"]

    labels = [s["session_date"] for s in sessions]
    mrl_eth = np.array([s["mean_mrl_ethanol"] for s in sessions], dtype=float)
    mrl_ctl = np.array([s["mean_mrl_control"] for s in sessions], dtype=float)
    degenerate = np.array([bool(s["dev001_degenerate_session"]) for s in sessions])

    p = stat["pvalue"]
    r = stat["effect_size_r"]
    n_animals = stat["n_animals"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.8))

    # ---- Panel A: paired dots/lines + group means ------------------------
    x_eth, x_ctl = 0.0, 1.0
    for i, lbl in enumerate(labels):
        is_dev = degenerate[i]
        axA.plot([x_eth, x_ctl], [mrl_eth[i], mrl_ctl[i]], color="0.65", lw=1.0, zorder=1)
        marker = "*" if is_dev else "o"
        ms = 13 if is_dev else 6
        c_eth = COLOR_DEV001 if is_dev else COLOR_ETHANOL
        c_ctl = COLOR_DEV001 if is_dev else COLOR_CONTROL
        axA.plot(x_eth, mrl_eth[i], marker=marker, ms=ms, color=c_eth,
                  markeredgecolor="k", markeredgewidth=0.4, zorder=3)
        axA.plot(x_ctl, mrl_ctl[i], marker=marker, ms=ms, color=c_ctl,
                  markeredgecolor="k", markeredgewidth=0.4, zorder=3)
        xoff = 0.06
        axA.annotate(lbl + (" *" if is_dev else ""), (x_ctl + xoff, mrl_ctl[i]),
                      fontsize=MIN_FONT - 2, va="center",
                      color=COLOR_DEV001 if is_dev else "0.2")

    mean_eth, mean_ctl = float(np.mean(mrl_eth)), float(np.mean(mrl_ctl))
    axA.hlines(mean_eth, x_eth - 0.14, x_eth + 0.14, color=COLOR_ETHANOL, lw=3.0, linestyle="-", zorder=4)
    axA.hlines(mean_ctl, x_ctl - 0.14, x_ctl + 0.14, color=COLOR_CONTROL, lw=3.0, linestyle="--", zorder=4)

    axA.set_xlim(-0.4, 1.75)
    axA.set_xticks([x_eth, x_ctl])
    axA.set_xticklabels(["ethanol", "control"])
    y_top = max(1.0, float(np.nanmax(mrl_ctl)) * 1.08)
    axA.set_ylim(0.0, y_top)
    axA.set_ylabel("Mean MRL (per animal)")
    axA.set_title("A. Animal-level mean MRL, paired")
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.text(
        0.03, 0.97,
        f"Wilcoxon exact p = {fmt_p(p)}\nrank-biserial r = {r:.3f}\nn = {n_animals} animals",
        transform=axA.transAxes, va="top", ha="left", fontsize=MIN_FONT - 1,
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9),
    )

    # ---- Panel B: per-animal ethanol vs control scatter ------------------
    lim = max(1.0, float(np.nanmax(mrl_ctl)), float(np.nanmax(mrl_eth))) * 1.08
    axB.plot([0, lim], [0, lim], color="0.75", lw=1.0, linestyle=":", zorder=1)
    for i, lbl in enumerate(labels):
        is_dev = degenerate[i]
        marker = "*" if is_dev else "o"
        s = 220 if is_dev else 70
        c = COLOR_DEV001 if is_dev else COLOR_ETHANOL
        axB.scatter(mrl_eth[i], mrl_ctl[i], marker=marker, s=s, color=c,
                     edgecolor="k", linewidth=0.5, zorder=3)
        axB.annotate(lbl + (" *" if is_dev else ""), (mrl_eth[i], mrl_ctl[i]),
                      textcoords="offset points", xytext=(6, 4),
                      fontsize=MIN_FONT - 2, color=COLOR_DEV001 if is_dev else "0.2")

    axB.set_xlim(0, lim)
    axB.set_ylim(0, lim)
    axB.set_xlabel("Mean MRL, ethanol")
    axB.set_ylabel("Mean MRL, control")
    axB.set_title("B. Per-animal ethanol vs control mean MRL")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    fig.suptitle(
        "Ethanol vs control phase-locking of OB units to the SNF signal -- "
        "experiment (animal) level",
        fontsize=MIN_FONT + 3,
    )

    dev_note = ""
    if degenerate.any():
        dev_note = (
            f" its animal-level mean_mrl_control (≈{mrl_ctl[degenerate][0]:.3f}) is a "
            "near-single-spike artifact, not a reliable phase-locking estimate --"
        )
    caption = (
        f"n_experiments = {n_animals} (up to 6); 2021-11-03 is included via the PROH-002 "
        f"eth_threshold adjustment (OD-3). Paired Wilcoxon exact two-sided test: "
        f"p = {fmt_p(p)}, matched-pairs rank-biserial r = {r:.3f}.\n"
        "* DEV-001 (08_traceability/deviations.yaml): 2021-11-03 has degenerate "
        "control-condition MRL values (0-4 control spikes for most included units);"
        f"{dev_note} flagged in red and requires human review before use."
    )
    fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=MIN_FONT - 1, wrap=True)

    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    return savefig(fig, "FIG-DEMO2-RES-EXP")


if __name__ == "__main__":
    png, pdf = render()
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")

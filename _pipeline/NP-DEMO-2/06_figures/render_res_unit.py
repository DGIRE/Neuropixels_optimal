"""
render_res_unit.py -- FIG-DEMO2-RES-UNIT: ethanol vs control phase-locking of
OB units to the SNF signal, UNIT level (contract_spec.yaml figure_id
FIG-DEMO2-RES-UNIT).

Reads figdata_FIG-DEMO2-RES-UNIT.yaml (frozen, traces to RESULT-mrl-unit +
RESULT-stat-unit -- the mixed-effects condition fixed-effect test).
Presentation-only: every plotted number is taken verbatim from the frozen
figdata; no statistic is recomputed here.

Layout: violin distribution of per-unit MRL for ethanol vs control, with the
raw per-unit points overlaid (jittered strip) and the median marked, plus
the mixed-effects exact p / standardized effect size annotated.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_common import load_yaml, savefig, COLOR_ETHANOL, COLOR_CONTROL, MIN_FONT, fmt_p


def render():
    d = load_yaml("figdata_FIG-DEMO2-RES-UNIT.yaml")
    units = d["units"]
    stat = d["stat"]

    n_units_computed = len(units)  # all units with a computed ethanol-condition MRL
    # RESULT-stat-unit's mixed-effects model requires a PAIRED (ethanol, control)
    # observation per unit; units with a null mrl_control (mostly the
    # DEV-001-flagged 2021-11-03 session, which has 0 control spikes for most
    # units) cannot enter that model. Plot exactly the matched-pair population
    # actually analyzed (n_units == stat['n_units']), not the larger
    # ethanol-only-computed set, so the figure matches RESULT-stat-unit 1:1.
    paired = [u for u in units if u["mrl_ethanol"] is not None and u["mrl_control"] is not None]
    mrl_eth = np.array([u["mrl_ethanol"] for u in paired], dtype=float)
    mrl_ctl = np.array([u["mrl_control"] for u in paired], dtype=float)
    n_excluded_no_control = n_units_computed - len(paired)

    p = stat["pvalue"]
    coef = stat["coefficient"]
    d_std = stat["standardized_effect_size"]
    n_units = stat["n_units"]
    n_animals = stat["n_animals"]
    assert len(paired) == n_units, (
        f"matched-pair count ({len(paired)}) must equal RESULT-stat-unit n_units "
        f"({n_units}) -- figure population must match the tested population exactly"
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    positions = [0.0, 1.0]
    data = [mrl_eth, mrl_ctl]
    colors = [COLOR_ETHANOL, COLOR_CONTROL]

    vp = ax.violinplot(data, positions=positions, widths=0.7,
                        showmedians=False, showextrema=False)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.35)
        body.set_edgecolor(c)

    rng = np.random.default_rng(5489)  # randomization_seed pinned by contract; jitter only
    for x0, vals, c, ls in zip(positions, data, colors, ["-", "--"]):
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(x0 + jitter, vals, s=6, color=c, alpha=0.35, linewidths=0, zorder=2)
        med = float(np.median(vals))
        ax.hlines(med, x0 - 0.28, x0 + 0.28, color=c, linewidth=2.6, linestyle=ls, zorder=4)
        mean_v = float(np.mean(vals))
        se_v = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
        ax.errorbar([x0 + 0.32], [mean_v], yerr=[se_v], fmt="D", ms=4, color="k",
                     elinewidth=1.2, capsize=3, zorder=5)

    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(["ethanol", "control"])
    y_top = max(1.0, float(np.nanmax(mrl_ctl)), float(np.nanmax(mrl_eth))) * 1.05
    ax.set_ylim(0.0, y_top)
    ax.set_ylabel("Per-unit MRL")
    ax.set_title("Ethanol vs control phase-locking of OB units to the SNF signal "
                  "-- unit level", fontsize=MIN_FONT + 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.02, 0.98,
        (f"Mixed-effects (condition fixed, animal random intercept)\n"
         f"exact p = {fmt_p(p)}\n"
         f"coefficient = {coef:.4f}\n"
         f"standardized effect size = {d_std:.3f}"),
        transform=ax.transAxes, va="top", ha="left", fontsize=MIN_FONT - 1,
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9),
    )

    caption = (
        f"n_units = {n_units} (matched ethanol+control pairs entering the mixed-effects "
        f"model; n_animals = {n_animals}). {n_excluded_no_control} of "
        f"{n_units_computed} units with a computed ethanol-condition MRL are excluded "
        "from this plot and from RESULT-stat-unit because they have no valid "
        "control-condition MRL (0 control-condition spikes), overwhelmingly from the "
        "DEV-001-flagged 2021-11-03 session. Unit inclusion: overall firing rate >= 0.1 Hz "
        "AND >= 50 valid-sniff spikes combined across ethanol + control conditions. "
        f"Mixed-effects condition fixed-effect exact p = {fmt_p(p)}, standardized effect "
        f"size (coefficient / residual SD) = {d_std:.3f}. Diamond markers = mean +/- SE; "
        "thick lines = median."
    )
    fig.text(0.01, -0.03, caption, ha="left", va="top", fontsize=MIN_FONT - 1, wrap=True)

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    return savefig(fig, "FIG-DEMO2-RES-UNIT")


if __name__ == "__main__":
    png, pdf = render()
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")

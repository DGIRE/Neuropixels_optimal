"""find_best_unit_eth_contact_fig.py -- NP-DEMO-4

Identifies two units using actual ethanol-contact samples (eth_contact_mask)
vs the 0-10 s baseline within each trial:

  Unit A: greatest percent increase in per-trial MAX firing during contact vs baseline
  Unit B: greatest absolute increase in per-trial MEAN firing during contact vs baseline

Trials with no ETH contact are skipped for these comparisons.
Both figures use the same two-panel waterfall + line layout.  Trial rows with
no ETH contact in that trial are included in the waterfall (full trial shown)
but contact-period shading only appears where contact actually occurred.

Output (revised figures folder):
  FR_best_unit_eth_max_waterfall.png
  FR_best_unit_eth_avg_waterfall.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

_SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNEL_DIR   = r"C:\Projects\Repos\Neuropixels\Optimized Python"
for p in (_KERNEL_DIR, _SOFTWARE_DIR):
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

from load_experiment_data     import load_experiment_data
from lib.or_validate_files    import or_validate_files
from detect_eth_contact       import detect_eth_contact
from compute_sniff_rate       import compute_sniff_rate
from compute_firing_rate_50ms import compute_firing_rate_50ms
from np_demo4_analysis import (
    _parse_session_date,
    _fix_ks_dir,
    _apply_unit_inclusion,
    _segment_odd_tr_trials,
)

DATA_ROOT = r"C:\Projects\Repos\Neuropixels\DATA"
OUT_DIR   = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\06_figures\revised figures"
os.makedirs(OUT_DIR, exist_ok=True)

SESSIONS = [
    os.path.join(DATA_ROOT, "11-01-2021"),
    os.path.join(DATA_ROOT, "11-03-2021"),
    os.path.join(DATA_ROOT, "12-15-2021"),
    os.path.join(DATA_ROOT, "5-17-2022"),
    os.path.join(DATA_ROOT, "06-24-2022"),
    os.path.join(DATA_ROOT, "09-14-2022"),
]

LV_FS    = 125.0
MIN_FONT = 8
DPI      = 150
COLORMAP = "jet"


# ── helpers ───────────────────────────────────────────────────────────────────
def build_matrix(fr_row, trials):
    """Build (n_trials x max_len) FR matrix for one unit from global FR trace."""
    arrs = []
    for trial in trials:
        idx  = trial["global_idx"]
        ts_s = trial["ts_s"]
        win  = (ts_s >= 0.0) & (ts_s <= 40.0)
        arrs.append(fr_row[idx[win]])
    max_len  = max(len(a) for a in arrs)
    n_trials = len(arrs)
    mat = np.full((n_trials, max_len), np.nan, dtype=np.float64)
    for i, a in enumerate(arrs):
        mat[i, :len(a)] = a
    return mat


def render_eth_contact_fig(
    *,
    matrix,
    trials,
    eth_contact_mask_global,
    title,
    out_path,
    exp_date,
    unit_id,
    metric_label,
    mean_contact,
    mean_baseline,
    change_label,
):
    """Two-panel waterfall with per-trial ETH contact spans highlighted."""
    n_trials, max_len = matrix.shape
    time_axis = np.arange(max_len, dtype=np.float64) / LV_FS

    with np.errstate(all="ignore"):
        mean_tr = np.nanmean(matrix, axis=0)
        n_valid = np.sum(~np.isnan(matrix), axis=0).astype(float)
        sem     = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
        ci_lo   = mean_tr - 1.96 * sem
        ci_hi   = mean_tr + 1.96 * sem

    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanpercentile(matrix, 99))
    t0, t1 = time_axis[0], time_axis[-1]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.50},
    )

    fig.suptitle(title, fontsize=MIN_FONT + 1, fontweight="bold", y=0.98)

    # --- Panel 1: waterfall ---
    im = ax1.imshow(
        matrix,
        aspect="auto", origin="lower",
        extent=[t0, t1, 0.5, n_trials + 0.5],
        cmap=COLORMAP, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label("firing rate (Hz)", fontsize=MIN_FONT)
    cbar.ax.tick_params(labelsize=MIN_FONT)

    # Per-trial ETH contact spans (white rectangles on heatmap)
    for ti, trial in enumerate(trials):
        idx  = trial["global_idx"]
        ts_s = trial["ts_s"]
        win  = (ts_s >= 0.0) & (ts_s <= 40.0)
        idx_win  = idx[win]
        ts_win   = ts_s[win]
        contact_local = eth_contact_mask_global[idx_win]
        if not contact_local.any():
            continue
        # Find contiguous contact runs and draw a span per run
        padded = np.concatenate(([False], contact_local, [False]))
        rises  = np.flatnonzero(~padded[:-1] &  padded[1:])
        falls  = np.flatnonzero( padded[:-1] & ~padded[1:])
        row_y  = ti + 1   # imshow origin='lower', row ti+1 in trial space
        for r, f in zip(rises, falls):
            t_start = float(ts_win[r])
            t_end   = float(ts_win[min(f, len(ts_win) - 1)])
            ax1.add_patch(plt.Rectangle(
                (t_start, row_y - 0.5), t_end - t_start, 1.0,
                linewidth=0, facecolor="white", alpha=0.25,
            ))

    ax1.axvline(10, color="white", linewidth=0.8, linestyle="--", alpha=0.7,
                label="valve open (10 s)")
    ax1.axvline(20, color="white", linewidth=0.8, linestyle=":",  alpha=0.7,
                label="valve close (20 s)")
    ax1.axvspan(0, 10, alpha=0.06, color="cyan",  label="baseline (0-10 s)")
    ax1.legend(fontsize=6, loc="upper right", framealpha=0.6)
    ax1.set_xlabel("time from trial start (s)", fontsize=MIN_FONT)
    ax1.set_ylabel("trial number (LabView)", fontsize=MIN_FONT)
    ax1.set_xlim(t0, t1)
    ax1.set_ylim(0.5, n_trials + 0.5)
    ax1.tick_params(labelsize=MIN_FONT)
    ax1.text(0.01, 0.97,
             f"Session: {exp_date}   Unit ID: {unit_id}",
             transform=ax1.transAxes, fontsize=MIN_FONT, va="top",
             color="white", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    # --- Panel 2: trial-averaged line ---
    ax2.plot(time_axis, mean_tr, color="black", linewidth=1.2, label="mean")
    ax2.fill_between(time_axis, ci_lo, ci_hi, color="steelblue", alpha=0.35,
                     label="95% CI (±1.96 SEM)")
    ax2.axvline(10, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axvline(20, color="gray", linewidth=0.8, linestyle=":",  alpha=0.6)
    ax2.axvspan(0, 10, alpha=0.08, color="cyan")
    ax2.set_xlabel("time from trial start (s)", fontsize=MIN_FONT)
    ax2.set_ylabel("firing rate (Hz)", fontsize=MIN_FONT)
    ax2.set_xlim(t0, t1)
    ax2.tick_params(labelsize=MIN_FONT)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, linewidth=0.4, alpha=0.4)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── search ────────────────────────────────────────────────────────────────────
best_max = dict(score=-np.inf)   # ranked by pct increase in max FR
best_avg = dict(score=-np.inf)   # ranked by absolute increase in mean FR

for sess_dir in SESSIONS:
    exp_date = _parse_session_date(sess_dir)
    print(f"\n=== {exp_date} ===")

    files, missing = or_validate_files(sess_dir, strict=False)
    files, missing = _fix_ks_dir(files, missing)
    if missing:
        print(f"  SKIP (missing): {missing}")
        continue

    D = load_experiment_data(files)
    D = detect_eth_contact(D, eth_threshold=0.05)
    D = compute_sniff_rate(D, threshold_std=-0.5)

    session_dur_s = float(len(D["SNF"])) / float(D["LV_Fs"])
    included_ids, n_rec, n_inc = _apply_unit_inclusion(D, session_dur_s)
    print(f"  {n_inc}/{n_rec} units included")
    if n_inc == 0:
        continue

    D = compute_firing_rate_50ms(D, included_ids, window_ms=50.0)
    fr_per_unit      = np.asarray(D["firing_rate_per_unit"], dtype=np.float64)
    eth_contact_mask = np.asarray(D["eth_contact_mask"], dtype=bool).ravel()
    trials           = _segment_odd_tr_trials(D)
    print(f"  {len(trials)} trials")
    if not trials:
        continue

    n_units = len(included_ids)

    # Accumulators: per-unit, per-trial values (only trials WITH contact)
    unit_max_contact  = [[] for _ in range(n_units)]
    unit_max_baseline = [[] for _ in range(n_units)]
    unit_avg_contact  = [[] for _ in range(n_units)]
    unit_avg_baseline = [[] for _ in range(n_units)]

    for trial in trials:
        idx  = trial["global_idx"]
        ts_s = trial["ts_s"]

        baseline_mask = ts_s < 10.0
        contact_local = eth_contact_mask[idx]

        if not baseline_mask.any() or not contact_local.any():
            continue   # skip trials without baseline or without contact

        bl_idx  = idx[baseline_mask]
        eth_idx = idx[contact_local]

        for ui in range(n_units):
            unit_max_baseline[ui].append(float(np.max(fr_per_unit[ui, bl_idx])))
            unit_max_contact[ui].append(float(np.max(fr_per_unit[ui, eth_idx])))
            unit_avg_baseline[ui].append(float(np.mean(fr_per_unit[ui, bl_idx])))
            unit_avg_contact[ui].append(float(np.mean(fr_per_unit[ui, eth_idx])))

    for ui, uid in enumerate(included_ids):
        if len(unit_max_contact[ui]) < 2:
            continue

        mm_bl = float(np.mean(unit_max_baseline[ui]))
        mm_ct = float(np.mean(unit_max_contact[ui]))
        am_bl = float(np.mean(unit_avg_baseline[ui]))
        am_ct = float(np.mean(unit_avg_contact[ui]))

        pct_max = 100.0 * (mm_ct - mm_bl) / mm_bl if mm_bl > 0 else -np.inf
        abs_avg = am_ct - am_bl

        if pct_max > best_max["score"]:
            best_max = dict(
                score=pct_max,
                exp_date=exp_date,
                unit_id=int(uid),
                mean_max_baseline=mm_bl,
                mean_max_contact=mm_ct,
                mean_avg_baseline=am_bl,
                mean_avg_contact=am_ct,
                trials=trials,
                fr_row=fr_per_unit[ui, :].copy(),
                eth_contact_mask=eth_contact_mask.copy(),
                n_contact_trials=len(unit_max_contact[ui]),
            )

        if abs_avg > best_avg["score"]:
            best_avg = dict(
                score=abs_avg,
                exp_date=exp_date,
                unit_id=int(uid),
                mean_max_baseline=mm_bl,
                mean_max_contact=mm_ct,
                mean_avg_baseline=am_bl,
                mean_avg_contact=am_ct,
                trials=trials,
                fr_row=fr_per_unit[ui, :].copy(),
                eth_contact_mask=eth_contact_mask.copy(),
                n_contact_trials=len(unit_avg_contact[ui]),
            )

    print(f"  Best max so far: {best_max.get('exp_date','?')} unit {best_max.get('unit_id','?')} "
          f"pct={best_max['score']:.1f}%")
    print(f"  Best avg so far: {best_avg.get('exp_date','?')} unit {best_avg.get('unit_id','?')} "
          f"abs={best_avg['score']:.2f} Hz")

print(f"\n{'='*60}")
print(f"BEST MAX UNIT: session={best_max['exp_date']}, unit={best_max['unit_id']}")
print(f"  mean max baseline (0-10s): {best_max['mean_max_baseline']:.2f} Hz")
print(f"  mean max contact:          {best_max['mean_max_contact']:.2f} Hz")
print(f"  pct increase in max:       +{best_max['score']:.1f}%")
print(f"  (n={best_max['n_contact_trials']} trials with ETH contact)")
print()
print(f"BEST AVG UNIT: session={best_avg['exp_date']}, unit={best_avg['unit_id']}")
print(f"  mean avg baseline (0-10s): {best_avg['mean_avg_baseline']:.2f} Hz")
print(f"  mean avg contact:          {best_avg['mean_avg_contact']:.2f} Hz")
print(f"  abs increase in avg:       +{best_avg['score']:.2f} Hz")
print(f"  (n={best_avg['n_contact_trials']} trials with ETH contact)")
print(f"{'='*60}")


# ── render figure A: best max unit ───────────────────────────────────────────
mat_max = build_matrix(best_max["fr_row"], best_max["trials"])
title_a = (
    f"FR — best unit (ETH contact max increase): "
    f"session {best_max['exp_date']}, unit {best_max['unit_id']}\n"
    f"Baseline max: {best_max['mean_max_baseline']:.1f} Hz  |  "
    f"Contact max: {best_max['mean_max_contact']:.1f} Hz  |  "
    f"+{best_max['score']:.1f}%  "
    f"(n={best_max['n_contact_trials']} contact trials / "
    f"{len(best_max['trials'])} total)"
)
render_eth_contact_fig(
    matrix=mat_max,
    trials=best_max["trials"],
    eth_contact_mask_global=best_max["eth_contact_mask"],
    title=title_a,
    out_path=os.path.join(OUT_DIR, "FR_best_unit_eth_max_waterfall.png"),
    exp_date=best_max["exp_date"],
    unit_id=best_max["unit_id"],
    metric_label="pct max increase",
    mean_contact=best_max["mean_max_contact"],
    mean_baseline=best_max["mean_max_baseline"],
    change_label=f"+{best_max['score']:.1f}%",
)

# ── render figure B: best avg unit ───────────────────────────────────────────
mat_avg = build_matrix(best_avg["fr_row"], best_avg["trials"])
title_b = (
    f"FR — best unit (ETH contact avg increase): "
    f"session {best_avg['exp_date']}, unit {best_avg['unit_id']}\n"
    f"Baseline mean: {best_avg['mean_avg_baseline']:.1f} Hz  |  "
    f"Contact mean: {best_avg['mean_avg_contact']:.1f} Hz  |  "
    f"+{best_avg['score']:.2f} Hz  "
    f"(n={best_avg['n_contact_trials']} contact trials / "
    f"{len(best_avg['trials'])} total)"
)
render_eth_contact_fig(
    matrix=mat_avg,
    trials=best_avg["trials"],
    eth_contact_mask_global=best_avg["eth_contact_mask"],
    title=title_b,
    out_path=os.path.join(OUT_DIR, "FR_best_unit_eth_avg_waterfall.png"),
    exp_date=best_avg["exp_date"],
    unit_id=best_avg["unit_id"],
    metric_label="abs avg increase",
    mean_contact=best_avg["mean_avg_contact"],
    mean_baseline=best_avg["mean_avg_baseline"],
    change_label=f"+{best_avg['score']:.2f} Hz",
)

print("\nDone.")

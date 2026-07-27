"""plot_eth_onset_05s_fig.py -- NP-DEMO-4

Finds and plots the best units using a 0-0.5 s post-onset measurement window.

Metric (per contact event):
  stat_window  = FR during [onset, onset + 0.5 s] within the session
  baseline     = mean FR during ts in [0, 10 s] of the containing LabView trial
  -> averaged across all contact events for each unit

Two rankings:
  Unit A: greatest pct increase in MAX FR in stat_window vs baseline mean
  Unit B: greatest absolute increase in MEAN FR in stat_window vs baseline mean

Figure window: -0.5 s to +0.5 s around each contact onset (one row per event).

Output (ethanol contact folder):
  FR_eth_05s_unit<id>_max.png
  FR_eth_05s_unit<id>_avg.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUT_DIR   = (r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4"
             r"\06_figures\revised figures\ethanol contact")
os.makedirs(OUT_DIR, exist_ok=True)

SESSIONS = [
    os.path.join(DATA_ROOT, "11-01-2021"),
    os.path.join(DATA_ROOT, "11-03-2021"),
    os.path.join(DATA_ROOT, "12-15-2021"),
    os.path.join(DATA_ROOT, "5-17-2022"),
    os.path.join(DATA_ROOT, "06-24-2022"),
    os.path.join(DATA_ROOT, "09-14-2022"),
]

LV_FS      = 125.0
STAT_POST  = 0.5                          # seconds post-onset to measure
STAT_N     = int(round(STAT_POST * LV_FS))  # 63 samples (~0.504 s)

FIG_PRE_S  = 0.5
FIG_POST_S = 0.5
FIG_PRE_N  = int(round(FIG_PRE_S  * LV_FS))   # 63 samples
FIG_POST_N = int(round(FIG_POST_S * LV_FS))   # 63 samples
WIN_N      = FIG_PRE_N + FIG_POST_N            # 126 samples
TIME_AX    = np.arange(-FIG_PRE_N, FIG_POST_N, dtype=np.float64) / LV_FS

MIN_FONT = 8
DPI      = 150
COLORMAP = "jet"


# ── per-session search ────────────────────────────────────────────────────────
best_max = dict(score=-np.inf)
best_avg = dict(score=-np.inf)

for sess_dir in SESSIONS:
    exp_date = _parse_session_date(sess_dir)
    print(f"\n=== {exp_date} ===")

    files, missing = or_validate_files(sess_dir, strict=False)
    files, missing = _fix_ks_dir(files, missing)
    if missing:
        print(f"  SKIP: {missing}")
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
    n_global         = fr_per_unit.shape[1]
    trials           = _segment_odd_tr_trials(D)
    print(f"  {len(trials)} trials")
    if not trials:
        continue

    n_units = len(included_ids)
    # Accumulators: per unit, one value per contact event
    unit_stat_max  = [[] for _ in range(n_units)]  # max FR in 0-0.5s post onset
    unit_stat_mean = [[] for _ in range(n_units)]  # mean FR in 0-0.5s post onset
    unit_bl_mean   = [[] for _ in range(n_units)]  # mean FR during 0-10s of trial

    n_contact_events = 0
    for trial in trials:
        idx  = trial["global_idx"]
        ts_s = trial["ts_s"]

        # Baseline: mean FR over 0-10 s of this trial
        bl_mask = (ts_s >= 0.0) & (ts_s < 10.0)
        if not bl_mask.any():
            continue
        bl_global = idx[bl_mask]

        # Find all contact onsets in this trial
        contact_lc = eth_contact_mask[idx]
        if not contact_lc.any():
            continue
        padded = np.concatenate(([False], contact_lc, [False]))
        rises  = np.flatnonzero(~padded[:-1] & padded[1:])

        for r in rises:
            onset_g = int(idx[r])
            # Stat window: onset to onset + STAT_N (0 to ~0.5 s)
            sw_s = onset_g
            sw_e = min(onset_g + STAT_N, n_global)
            if sw_e <= sw_s:
                continue
            n_contact_events += 1
            for ui in range(n_units):
                bl_fr   = fr_per_unit[ui, bl_global]
                stat_fr = fr_per_unit[ui, sw_s:sw_e]
                unit_bl_mean[ui].append(float(np.mean(bl_fr)))
                unit_stat_max[ui].append(float(np.max(stat_fr)))
                unit_stat_mean[ui].append(float(np.mean(stat_fr)))

    print(f"  {n_contact_events} contact events total")
    if n_contact_events == 0:
        continue

    for ui, uid in enumerate(included_ids):
        if len(unit_stat_max[ui]) < 2:
            continue
        bl   = float(np.mean(unit_bl_mean[ui]))
        mx   = float(np.mean(unit_stat_max[ui]))
        mn   = float(np.mean(unit_stat_mean[ui]))
        pct  = 100.0 * (mx - bl) / bl if bl > 0 else -np.inf
        absc = mn - bl

        if pct > best_max["score"]:
            best_max = dict(
                score=pct, exp_date=exp_date, unit_id=int(uid), unit_idx=ui,
                mean_bl=bl, mean_stat_max=mx, mean_stat_mean=mn,
                n_events=len(unit_stat_max[ui]),
                fr_row=fr_per_unit[ui, :].copy(),
                eth_contact_mask=eth_contact_mask.copy(),
                trials=trials, n_global=n_global,
            )
        if absc > best_avg["score"]:
            best_avg = dict(
                score=absc, exp_date=exp_date, unit_id=int(uid), unit_idx=ui,
                mean_bl=bl, mean_stat_max=mx, mean_stat_mean=mn,
                n_events=len(unit_stat_mean[ui]),
                fr_row=fr_per_unit[ui, :].copy(),
                eth_contact_mask=eth_contact_mask.copy(),
                trials=trials, n_global=n_global,
            )

    print(f"  Best max so far: {best_max['exp_date']} unit {best_max['unit_id']} "
          f"pct={best_max['score']:.1f}%")
    print(f"  Best avg so far: {best_avg['exp_date']} unit {best_avg['unit_id']} "
          f"abs={best_avg['score']:.2f} Hz")

print(f"\n{'='*60}")
print(f"BEST MAX UNIT: session={best_max['exp_date']}, unit={best_max['unit_id']}")
print(f"  baseline mean (0-10s):  {best_max['mean_bl']:.2f} Hz")
print(f"  mean max in 0-0.5s:     {best_max['mean_stat_max']:.2f} Hz")
print(f"  pct increase in max:    +{best_max['score']:.1f}%")
print(f"  n contact events:       {best_max['n_events']}")
print()
print(f"BEST AVG UNIT: session={best_avg['exp_date']}, unit={best_avg['unit_id']}")
print(f"  baseline mean (0-10s):  {best_avg['mean_bl']:.2f} Hz")
print(f"  mean avg in 0-0.5s:     {best_avg['mean_stat_mean']:.2f} Hz")
print(f"  abs increase in mean:   +{best_avg['score']:.2f} Hz")
print(f"  n contact events:       {best_avg['n_events']}")
print(f"{'='*60}")


# ── build onset-aligned matrix (-0.5 to +0.5 s) ─────────────────────────────
def build_onset_matrix(b):
    fr_row = b["fr_row"]
    rows   = []
    for trial in b["trials"]:
        idx        = trial["global_idx"]
        contact_lc = b["eth_contact_mask"][idx]
        if not contact_lc.any():
            continue
        padded = np.concatenate(([False], contact_lc, [False]))
        rises  = np.flatnonzero(~padded[:-1] & padded[1:])
        for r in rises:
            onset_g = int(idx[r])
            src_s   = onset_g - FIG_PRE_N
            src_e   = onset_g + FIG_POST_N
            row     = np.full(WIN_N, np.nan, dtype=np.float64)
            vs = max(0, src_s);  ve = min(b["n_global"], src_e)
            ds = vs - src_s;     de = ds + (ve - vs)
            row[ds:de] = fr_row[vs:ve]
            rows.append(row)
    return np.array(rows, dtype=np.float64) if rows else np.empty((0, WIN_N))


# ── render ────────────────────────────────────────────────────────────────────
def render(b, metric, matrix, out_path):
    n_ev = matrix.shape[0]
    if n_ev == 0:
        print(f"  No events — skipping {out_path}")
        return

    with np.errstate(all="ignore"):
        mean_tr = np.nanmean(matrix, axis=0)
        n_valid = np.sum(~np.isnan(matrix), axis=0).astype(float)
        sem     = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
        ci_lo   = mean_tr - 1.96 * sem
        ci_hi   = mean_tr + 1.96 * sem

    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanpercentile(matrix, 99))

    if metric == "max":
        change_str = f"+{b['score']:.1f}% (max FR)"
        bl_str     = f"bl mean={b['mean_bl']:.1f} Hz  0-0.5s max={b['mean_stat_max']:.1f} Hz"
    else:
        change_str = f"+{b['score']:.2f} Hz (mean FR)"
        bl_str     = f"bl mean={b['mean_bl']:.1f} Hz  0-0.5s mean={b['mean_stat_mean']:.1f} Hz"

    title = (
        f"FR aligned to ETH contact onset  |  Session: {b['exp_date']}  "
        f"Unit: {b['unit_id']}\n"
        f"{bl_str}  |  {change_str}  |  n={n_ev} events"
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.52},
    )
    fig.suptitle(title, fontsize=MIN_FONT + 0.5, fontweight="bold", y=0.98)

    # --- heatmap ---
    im = ax1.imshow(
        matrix,
        aspect="auto", origin="lower",
        extent=[TIME_AX[0], TIME_AX[-1], 0.5, n_ev + 0.5],
        cmap=COLORMAP, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.02)
    cbar.set_label("firing rate (Hz)", fontsize=MIN_FONT)
    cbar.ax.tick_params(labelsize=MIN_FONT)

    ax1.axvline(0, color="white", linewidth=1.4, linestyle="-",
                alpha=0.95, label="ETH onset")
    ax1.axvline(STAT_POST, color="white", linewidth=0.9, linestyle="--",
                alpha=0.7, label=f"stat window end ({STAT_POST} s)")
    ax1.axvspan(0, STAT_POST, alpha=0.10, color="yellow",
                label=f"0–{STAT_POST} s window")
    ax1.axvspan(-FIG_PRE_S, 0, alpha=0.07, color="cyan",
                label="pre-onset")
    ax1.legend(fontsize=6, loc="upper right", framealpha=0.65)
    ax1.set_xlabel("time relative to ETH contact onset (s)", fontsize=MIN_FONT)
    ax1.set_ylabel("contact event #", fontsize=MIN_FONT)
    ax1.set_xlim(TIME_AX[0], TIME_AX[-1])
    ax1.set_ylim(0.5, n_ev + 0.5)
    ax1.tick_params(labelsize=MIN_FONT)
    ax1.text(0.01, 0.97,
             f"Session: {b['exp_date']}   Unit: {b['unit_id']}",
             transform=ax1.transAxes, fontsize=MIN_FONT, va="top",
             color="white", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    # --- mean line ---
    ax2.plot(TIME_AX, mean_tr, color="black", linewidth=1.2, label="mean")
    ax2.fill_between(TIME_AX, ci_lo, ci_hi, color="steelblue", alpha=0.35,
                     label="95% CI (±1.96 SEM)")
    ax2.axvline(0, color="black", linewidth=1.2, linestyle="-", alpha=0.8)
    ax2.axvline(STAT_POST, color="gray", linewidth=0.9, linestyle="--", alpha=0.7)
    ax2.axhline(b["mean_bl"], color="red", linewidth=0.9, linestyle="--",
                alpha=0.8, label=f"baseline mean ({b['mean_bl']:.1f} Hz)")
    ax2.axvspan(0, STAT_POST, alpha=0.10, color="yellow")
    ax2.axvspan(-FIG_PRE_S, 0, alpha=0.07, color="cyan")
    ax2.set_xlabel("time relative to ETH contact onset (s)", fontsize=MIN_FONT)
    ax2.set_ylabel("firing rate (Hz)", fontsize=MIN_FONT)
    ax2.set_xlim(TIME_AX[0], TIME_AX[-1])
    ax2.tick_params(labelsize=MIN_FONT)
    ax2.legend(fontsize=6.5, loc="upper right")
    ax2.grid(True, linewidth=0.4, alpha=0.4)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── produce figures ───────────────────────────────────────────────────────────
print("\nBuilding onset-aligned matrices ...")

mat_max = build_onset_matrix(best_max)
fname_max = f"FR_eth_05s_unit{best_max['unit_id']}_max.png"
render(best_max, "max", mat_max, os.path.join(OUT_DIR, fname_max))

mat_avg = build_onset_matrix(best_avg)
fname_avg = f"FR_eth_05s_unit{best_avg['unit_id']}_avg.png"
render(best_avg, "avg", mat_avg, os.path.join(OUT_DIR, fname_avg))

print("\nDone.")

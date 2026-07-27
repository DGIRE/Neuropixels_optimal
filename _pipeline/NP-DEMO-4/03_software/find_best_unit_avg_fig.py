"""find_best_unit_avg_fig.py -- NP-DEMO-4: best unit by average FR change.

Identifies the unit whose mean firing rate in 10-20 s (odor window) shows the
greatest absolute increase over its mean firing rate in 0-10 s (control window),
averaged across all trials.

Metric per unit:
  For each trial t:
    mean_control[t] = mean( FR_unit[samples with ts in 0-10 s] )
    mean_odor[t]   = mean( FR_unit[samples with ts in 10-20 s] )
  change_hz = mean(mean_odor) - mean(mean_control)   <- ranked on this
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

from load_experiment_data    import load_experiment_data
from lib.or_validate_files   import or_validate_files
from detect_eth_contact      import detect_eth_contact
from compute_sniff_rate      import compute_sniff_rate
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

LV_FS     = 125.0
MIN_FONT  = 8
DPI       = 150
COLORMAP  = "jet"
CONTROL_S = (0.0, 10.0)
ODOR_S    = (10.0, 20.0)


# ── per-session search ────────────────────────────────────────────────────────
best = dict(change_hz=-np.inf)

for sess_dir in SESSIONS:
    exp_date = _parse_session_date(sess_dir)
    print(f"\n=== {exp_date} ===")

    files, missing = or_validate_files(sess_dir, strict=False)
    files, missing = _fix_ks_dir(files, missing)
    if missing:
        print(f"  SKIP (missing files): {missing}")
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
    fr_per_unit = np.asarray(D["firing_rate_per_unit"], dtype=np.float64)
    trials = _segment_odd_tr_trials(D)
    print(f"  {len(trials)} trials")
    if not trials:
        continue

    n_units = len(included_ids)
    unit_mean_control = [[] for _ in range(n_units)]
    unit_mean_odor    = [[] for _ in range(n_units)]

    for trial in trials:
        idx  = trial["global_idx"]
        ts_s = trial["ts_s"]
        ctrl_mask = (ts_s >= CONTROL_S[0]) & (ts_s < CONTROL_S[1])
        odor_mask = (ts_s >= ODOR_S[0])   & (ts_s < ODOR_S[1])
        if not ctrl_mask.any() or not odor_mask.any():
            continue
        ctrl_idx = idx[ctrl_mask]
        odor_idx = idx[odor_mask]
        for ui in range(n_units):
            unit_mean_control[ui].append(float(np.mean(fr_per_unit[ui, ctrl_idx])))
            unit_mean_odor[ui].append(float(np.mean(fr_per_unit[ui, odor_idx])))

    for ui, uid in enumerate(included_ids):
        mc = unit_mean_control[ui]
        mo = unit_mean_odor[ui]
        if len(mc) < 2:
            continue
        mean_mc  = float(np.mean(mc))
        mean_mo  = float(np.mean(mo))
        change   = mean_mo - mean_mc
        pct      = 100.0 * change / mean_mc if mean_mc > 0 else float("nan")
        if change > best["change_hz"]:
            best = dict(
                change_hz=change,
                pct_change=pct,
                exp_date=exp_date,
                unit_id=int(uid),
                unit_idx=ui,
                mean_control=mean_mc,
                mean_odor=mean_mo,
                trials=trials,
                fr_row=fr_per_unit[ui, :].copy(),
            )

    print(f"  Best so far: {best.get('exp_date','?')} unit {best.get('unit_id','?')} "
          f"change={best['change_hz']:.2f} Hz")

print(f"\n{'='*60}")
print(f"BEST UNIT (avg FR change): session={best['exp_date']}, unit_id={best['unit_id']}")
print(f"  mean FR control (0-10s): {best['mean_control']:.2f} Hz")
print(f"  mean FR odor   (10-20s): {best['mean_odor']:.2f} Hz")
print(f"  absolute change: +{best['change_hz']:.2f} Hz")
print(f"  percent change:  +{best['pct_change']:.1f}%")
print(f"{'='*60}")


# ── build per-trial matrix for best unit ────────────────────────────────────
fr_row = best["fr_row"]
trial_arrays = []
for trial in best["trials"]:
    idx  = trial["global_idx"]
    ts_s = trial["ts_s"]
    win  = (ts_s >= 0.0) & (ts_s <= 40.0)
    trial_arrays.append(fr_row[idx[win]])

max_len  = max(len(a) for a in trial_arrays)
n_trials = len(trial_arrays)
matrix   = np.full((n_trials, max_len), np.nan, dtype=np.float64)
for i, a in enumerate(trial_arrays):
    matrix[i, :len(a)] = a

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


# ── render ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 7),
    gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.48},
)

title = (
    f"Firing rate — best unit (avg change): session {best['exp_date']}, unit {best['unit_id']}\n"
    f"Control mean (0-10 s): {best['mean_control']:.1f} Hz  |  "
    f"Odor mean (10-20 s): {best['mean_odor']:.1f} Hz  |  "
    f"+{best['change_hz']:.1f} Hz (+{best['pct_change']:.1f}%)  (n={n_trials} trials)"
)
fig.suptitle(title, fontsize=MIN_FONT + 1, fontweight="bold", y=0.98)

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
ax1.set_xlabel("time from trial start (s)", fontsize=MIN_FONT)
ax1.set_ylabel("trial number (LabView)", fontsize=MIN_FONT)
ax1.set_xlim(t0, t1)
ax1.set_ylim(0.5, n_trials + 0.5)
ax1.tick_params(labelsize=MIN_FONT)
ax1.axvline(10, color="white", linewidth=0.9, linestyle="--", alpha=0.75,
            label="valve open (10 s)")
ax1.axvline(20, color="white", linewidth=0.9, linestyle=":",  alpha=0.75,
            label="valve close (20 s)")
ax1.axvspan(CONTROL_S[0], CONTROL_S[1], alpha=0.07, color="cyan",  label="control window")
ax1.axvspan(ODOR_S[0],    ODOR_S[1],    alpha=0.07, color="yellow", label="odor window")
ax1.legend(fontsize=6, loc="upper right", framealpha=0.6)
ax1.text(0.01, 0.97,
         f"Session: {best['exp_date']}   Unit ID: {best['unit_id']}",
         transform=ax1.transAxes, fontsize=MIN_FONT, va="top",
         color="white", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

ax2.plot(time_axis, mean_tr, color="black", linewidth=1.2, label="mean")
ax2.fill_between(time_axis, ci_lo, ci_hi, color="steelblue", alpha=0.35,
                 label="95% CI (±1.96 SEM)")
ax2.axvline(10, color="gray", linewidth=0.9, linestyle="--", alpha=0.6)
ax2.axvline(20, color="gray", linewidth=0.9, linestyle=":",  alpha=0.6)
ax2.axvspan(CONTROL_S[0], CONTROL_S[1], alpha=0.08, color="cyan")
ax2.axvspan(ODOR_S[0],    ODOR_S[1],    alpha=0.08, color="yellow")
ax2.set_xlabel("time from trial start (s)", fontsize=MIN_FONT)
ax2.set_ylabel("firing rate (Hz)", fontsize=MIN_FONT)
ax2.set_xlim(t0, t1)
ax2.tick_params(labelsize=MIN_FONT)
ax2.legend(fontsize=7, loc="upper right")
ax2.grid(True, linewidth=0.4, alpha=0.4)

out_path = os.path.join(OUT_DIR, "FR_best_unit_avg_change_waterfall.png")
fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")

"""plot_eth_onset_1sbl_fig.py -- NP-DEMO-4

Same as plot_eth_onset_05s_fig.py but with a per-event baseline:
  baseline = mean FR during [onset - 1 s, onset)  (125 samples before each onset)

Stat window (response):  [onset, onset + 0.5 s]  (63 samples)
Figure window:           -0.5 s to +0.5 s around onset

Contact events where onset_global < 125 (insufficient pre-history) are skipped.

Rankings:
  Unit A: greatest pct increase in MAX FR in 0-0.5 s post-onset vs 1-s pre-onset mean
  Unit B: greatest absolute increase in MEAN FR in 0-0.5 s post-onset vs 1-s pre-onset mean

Output -> ethanol contact folder:
  FR_eth_1sbl_unit<id>_max.png
  FR_eth_1sbl_unit<id>_avg.png
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

LV_FS    = 125.0
BL_S     = 1.0                            # baseline: 1 s before onset
BL_N     = int(round(BL_S   * LV_FS))    # 125 samples
STAT_S   = 0.5                            # response: 0-0.5 s post-onset
STAT_N   = int(round(STAT_S * LV_FS))    # 63 samples

FIG_PRE_S  = 0.5
FIG_POST_S = 0.5
FIG_PRE_N  = int(round(FIG_PRE_S  * LV_FS))   # 63
FIG_POST_N = int(round(FIG_POST_S * LV_FS))   # 63
WIN_N      = FIG_PRE_N + FIG_POST_N            # 126
TIME_AX    = np.arange(-FIG_PRE_N, FIG_POST_N, dtype=np.float64) / LV_FS

MIN_FONT = 8
DPI      = 150
COLORMAP = "jet"


# ── full search ───────────────────────────────────────────────────────────────
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
    unit_bl   = [[] for _ in range(n_units)]   # per-event baseline mean
    unit_s_mx = [[] for _ in range(n_units)]   # per-event stat max
    unit_s_mn = [[] for _ in range(n_units)]   # per-event stat mean

    n_events = 0
    for trial in trials:
        idx        = trial["global_idx"]
        contact_lc = eth_contact_mask[idx]
        if not contact_lc.any():
            continue
        padded = np.concatenate(([False], contact_lc, [False]))
        rises  = np.flatnonzero(~padded[:-1] & padded[1:])

        for r in rises:
            onset_g = int(idx[r])
            # Skip if not enough pre-onset history
            if onset_g < BL_N:
                continue
            bl_s  = onset_g - BL_N        # baseline: [onset-1s, onset)
            bl_e  = onset_g
            sw_s  = onset_g               # stat window: [onset, onset+0.5s)
            sw_e  = min(onset_g + STAT_N, n_global)
            if sw_e <= sw_s:
                continue
            n_events += 1
            for ui in range(n_units):
                bl_fr   = fr_per_unit[ui, bl_s:bl_e]
                stat_fr = fr_per_unit[ui, sw_s:sw_e]
                unit_bl[ui].append(float(np.mean(bl_fr)))
                unit_s_mx[ui].append(float(np.max(stat_fr)))
                unit_s_mn[ui].append(float(np.mean(stat_fr)))

    print(f"  {n_events} usable contact events")
    if n_events == 0:
        continue

    for ui, uid in enumerate(included_ids):
        if len(unit_bl[ui]) < 2:
            continue
        bl   = float(np.mean(unit_bl[ui]))
        mx   = float(np.mean(unit_s_mx[ui]))
        mn   = float(np.mean(unit_s_mn[ui]))
        pct  = 100.0 * (mx - bl) / bl if bl > 0 else -np.inf
        absc = mn - bl

        if pct > best_max["score"]:
            best_max = dict(
                score=pct, exp_date=exp_date, unit_id=int(uid),
                mean_bl=bl, mean_stat_max=mx, mean_stat_mean=mn,
                n_events=len(unit_bl[ui]),
                fr_row=fr_per_unit[ui, :].copy(),
                eth_contact_mask=eth_contact_mask.copy(),
                trials=trials, n_global=n_global,
            )
        if absc > best_avg["score"]:
            best_avg = dict(
                score=absc, exp_date=exp_date, unit_id=int(uid),
                mean_bl=bl, mean_stat_max=mx, mean_stat_mean=mn,
                n_events=len(unit_bl[ui]),
                fr_row=fr_per_unit[ui, :].copy(),
                eth_contact_mask=eth_contact_mask.copy(),
                trials=trials, n_global=n_global,
            )

    print(f"  Best max: {best_max['exp_date']} unit {best_max['unit_id']} "
          f"pct={best_max['score']:.1f}%")
    print(f"  Best avg: {best_avg['exp_date']} unit {best_avg['unit_id']} "
          f"abs={best_avg['score']:.2f} Hz")

print(f"\n{'='*60}")
print(f"BEST MAX UNIT: session={best_max['exp_date']}, unit={best_max['unit_id']}")
print(f"  1-s pre-onset baseline mean: {best_max['mean_bl']:.2f} Hz")
print(f"  mean max in 0-0.5 s:         {best_max['mean_stat_max']:.2f} Hz")
print(f"  pct increase in max:         +{best_max['score']:.1f}%")
print(f"  n usable events:             {best_max['n_events']}")
print()
print(f"BEST AVG UNIT: session={best_avg['exp_date']}, unit={best_avg['unit_id']}")
print(f"  1-s pre-onset baseline mean: {best_avg['mean_bl']:.2f} Hz")
print(f"  mean avg in 0-0.5 s:         {best_avg['mean_stat_mean']:.2f} Hz")
print(f"  abs increase in mean:        +{best_avg['score']:.2f} Hz")
print(f"  n usable events:             {best_avg['n_events']}")
print(f"{'='*60}")


# ── build onset-aligned figure matrix (-0.5 to +0.5 s) ──────────────────────
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
            if onset_g < BL_N:       # same skip as in search
                continue
            src_s = onset_g - FIG_PRE_N
            src_e = onset_g + FIG_POST_N
            row   = np.full(WIN_N, np.nan, dtype=np.float64)
            vs = max(0, src_s);  ve = min(b["n_global"], src_e)
            ds = vs - src_s;     de = ds + (ve - vs)
            row[ds:de] = fr_row[vs:ve]
            rows.append(row)
    return np.array(rows, dtype=np.float64) if rows else np.empty((0, WIN_N))


# ── render ────────────────────────────────────────────────────────────────────
def render(b, metric, matrix, out_path):
    n_ev = matrix.shape[0]
    if n_ev == 0:
        print(f"  No events — skipping.")
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
        val_str    = f"pre-onset bl={b['mean_bl']:.1f} Hz  0-0.5s max={b['mean_stat_max']:.1f} Hz"
    else:
        change_str = f"+{b['score']:.2f} Hz (mean FR)"
        val_str    = f"pre-onset bl={b['mean_bl']:.1f} Hz  0-0.5s mean={b['mean_stat_mean']:.1f} Hz"

    title = (
        f"FR aligned to ETH contact onset  |  Session: {b['exp_date']}  "
        f"Unit: {b['unit_id']}\n"
        f"Baseline: 1 s pre-onset  |  {val_str}  |  {change_str}  |  n={n_ev} events"
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.52},
    )
    fig.suptitle(title, fontsize=MIN_FONT + 0.5, fontweight="bold", y=0.98)

    # heatmap
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
    ax1.axvline(STAT_S, color="white", linewidth=0.9, linestyle="--",
                alpha=0.75, label=f"stat end ({STAT_S} s)")
    ax1.axvspan(-FIG_PRE_S, 0,    alpha=0.10, color="cyan",   label="pre-onset baseline")
    ax1.axvspan(0,          STAT_S, alpha=0.10, color="yellow", label=f"0-{STAT_S} s response")
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

    # mean + CI
    ax2.plot(TIME_AX, mean_tr, color="black", linewidth=1.2, label="mean")
    ax2.fill_between(TIME_AX, ci_lo, ci_hi, color="steelblue", alpha=0.35,
                     label="95% CI (±1.96 SEM)")
    ax2.axvline(0,      color="black", linewidth=1.2, linestyle="-",  alpha=0.85)
    ax2.axvline(STAT_S, color="gray",  linewidth=0.9, linestyle="--", alpha=0.7)
    ax2.axhline(b["mean_bl"], color="red", linewidth=0.9, linestyle="--",
                alpha=0.85, label=f"pre-onset bl mean ({b['mean_bl']:.1f} Hz)")
    ax2.axvspan(-FIG_PRE_S, 0,    alpha=0.10, color="cyan")
    ax2.axvspan(0,          STAT_S, alpha=0.10, color="yellow")
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
render(best_max, "max", mat_max,
       os.path.join(OUT_DIR, f"FR_eth_1sbl_unit{best_max['unit_id']}_max.png"))

mat_avg = build_onset_matrix(best_avg)
render(best_avg, "avg", mat_avg,
       os.path.join(OUT_DIR, f"FR_eth_1sbl_unit{best_avg['unit_id']}_avg.png"))

print("\nDone.")

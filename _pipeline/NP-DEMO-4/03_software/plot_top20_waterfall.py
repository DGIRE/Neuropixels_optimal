"""plot_top20_waterfall.py -- NP-DEMO-4

Top-20 units by pct-increase in max FR, two alignment conditions:

  Figure 1 (ETH):   aligned to ETH contact onset
    baseline:  mean FR in [onset - 1.0 s, onset)
    response:  max  FR in [onset, onset + 0.5 s]
    figure:    -0.5 to +0.5 s

  Figure 2 (Sniff): aligned to sniff onset
    baseline:  mean FR in [onset - 0.25 s, onset)
    response:  max  FR in [onset, onset + 0.5 s]
    figure:    -0.5 to +0.5 s

Each figure: 20-row heatmap (one row per unit, mean across all events),
sorted best-at-top, with grand-mean ± CI bottom panel.

Output -> ethanol contact folder:
  TOP20_eth_max_waterfall.png
  TOP20_sniff_max_waterfall.png
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
    _parse_session_date, _fix_ks_dir,
    _apply_unit_inclusion, _segment_odd_tr_trials,
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

LV_FS = 125.0
TOP_N = 20

ETH_BL_N   = int(round(1.00 * LV_FS))   # 125 samples
SNIFF_BL_N = int(round(0.25 * LV_FS))   # 31 samples
STAT_N     = int(round(0.50 * LV_FS))   # 63 samples  (response window)
FIG_PRE_N  = int(round(0.50 * LV_FS))   # 63 samples  (figure pre-onset)
FIG_POST_N = int(round(0.50 * LV_FS))   # 63 samples  (figure post-onset)
WIN_N      = FIG_PRE_N + FIG_POST_N     # 126
TIME_AX    = np.arange(-FIG_PRE_N, FIG_POST_N, dtype=np.float64) / LV_FS

MIN_FONT = 7
DPI      = 150
COLORMAP = "RdBu_r"


# ── helpers ───────────────────────────────────────────────────────────────────
def _collect_eth_onsets(trials, eth_contact_mask):
    onsets = []
    for trial in trials:
        idx  = trial["global_idx"]
        lc   = eth_contact_mask[idx]
        if not lc.any():
            continue
        pad  = np.concatenate(([False], lc, [False]))
        for r in np.flatnonzero(~pad[:-1] & pad[1:]):
            onsets.append(int(idx[r]))
    return np.array(onsets, dtype=int)


def _filter_onsets(onsets, bl_n, n_global):
    """Keep onsets where both baseline and figure windows fit."""
    min_onset = max(bl_n, FIG_PRE_N)
    max_onset = n_global - max(STAT_N, FIG_POST_N)
    mask = (onsets >= min_onset) & (onsets <= max_onset)
    return onsets[mask]


def _unit_stats(fr_u, onsets, bl_n):
    """Vectorised: baseline mean, stat max, figure mean response for one unit."""
    bl_cols  = np.arange(-bl_n, 0, dtype=int)[None, :]  + onsets[:, None]  # (N, bl_n)
    st_cols  = np.arange(0, STAT_N, dtype=int)[None, :] + onsets[:, None]  # (N, STAT_N)
    fig_cols = np.arange(-FIG_PRE_N, FIG_POST_N, dtype=int)[None, :] + onsets[:, None]  # (N, WIN_N)

    bl_means  = fr_u[bl_cols].mean(axis=1)   # (N,)
    st_maxes  = fr_u[st_cols].max(axis=1)    # (N,)
    fig_mat   = fr_u[fig_cols]               # (N, WIN_N)

    mean_bl   = float(bl_means.mean())
    mean_max  = float(st_maxes.mean())
    pct       = 100.0 * (mean_max - mean_bl) / mean_bl if mean_bl > 0 else -np.inf
    mean_resp = fig_mat.mean(axis=0)         # (WIN_N,)
    return pct, mean_bl, mean_max, mean_resp, len(onsets)


# ── search across all sessions ────────────────────────────────────────────────
eth_units   = []   # dicts: score, exp_date, unit_id, mean_bl, mean_max, n_ev, mean_response
sniff_units = []

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

    # ETH onsets
    eth_all   = _collect_eth_onsets(trials, eth_contact_mask)
    eth_valid = _filter_onsets(eth_all, ETH_BL_N, n_global)

    # Sniff onsets — restrict to samples inside any trial
    trial_mask = np.zeros(n_global, dtype=bool)
    for t in trials:
        trial_mask[t["global_idx"]] = True
    sniff_raw   = np.asarray(D.get("sniff_onsets_sr", []), dtype=int).ravel()
    sniff_raw   = sniff_raw[(sniff_raw >= 0) & (sniff_raw < n_global)]
    sniff_raw   = sniff_raw[trial_mask[sniff_raw]]
    sniff_valid = _filter_onsets(sniff_raw, SNIFF_BL_N, n_global)

    print(f"  {len(eth_valid)} ETH events, {len(sniff_valid)} sniff events (in-trial)")

    for ui, uid in enumerate(included_ids):
        fr_u = fr_per_unit[ui]

        if len(eth_valid) >= 2:
            pct, bl, mx, resp, n_ev = _unit_stats(fr_u, eth_valid, ETH_BL_N)
            eth_units.append(dict(score=pct, exp_date=exp_date, unit_id=int(uid),
                                  mean_bl=bl, mean_max=mx, n_ev=n_ev, mean_response=resp))

        if len(sniff_valid) >= 2:
            pct, bl, mx, resp, n_ev = _unit_stats(fr_u, sniff_valid, SNIFF_BL_N)
            sniff_units.append(dict(score=pct, exp_date=exp_date, unit_id=int(uid),
                                    mean_bl=bl, mean_max=mx, n_ev=n_ev, mean_response=resp))

    print(f"  Running totals: {len(eth_units)} ETH entries, {len(sniff_units)} sniff entries")


# ── rank and extract top 20 ───────────────────────────────────────────────────
eth_top20   = sorted(eth_units,   key=lambda x: x["score"], reverse=True)[:TOP_N]
sniff_top20 = sorted(sniff_units, key=lambda x: x["score"], reverse=True)[:TOP_N]

print(f"\n{'='*60}")
print("ETH top-20 (pct increase in max FR, 1-s pre-onset baseline):")
for i, u in enumerate(eth_top20):
    print(f"  Rank {i+1:2d}: {u['exp_date']} unit {u['unit_id']:4d}  "
          f"bl={u['mean_bl']:.2f} Hz  max={u['mean_max']:.2f} Hz  "
          f"+{u['score']:.0f}%  n={u['n_ev']}")

print(f"\nSniff top-20 (pct increase in max FR, 0.25-s pre-sniff baseline):")
for i, u in enumerate(sniff_top20):
    print(f"  Rank {i+1:2d}: {u['exp_date']} unit {u['unit_id']:4d}  "
          f"bl={u['mean_bl']:.2f} Hz  max={u['mean_max']:.2f} Hz  "
          f"+{u['score']:.0f}%  n={u['n_ev']}")
print(f"{'='*60}")


# ── render ────────────────────────────────────────────────────────────────────
def render_top20(top20, event_label, bl_desc, out_path):
    n = len(top20)
    if n == 0:
        print(f"  Nothing to render for {out_path}")
        return

    # Sort ascending for display: worst at row-0 (bottom), best at row-(n-1) (top)
    disp = sorted(top20, key=lambda x: x["score"])        # ascending

    # Subtract each unit's pre-onset mean (baseline correction)
    corrected = []
    for u in disp:
        resp   = u["mean_response"].copy()
        bl_val = float(np.nanmean(resp[:FIG_PRE_N]))  # mean of pre-onset window
        corrected.append(resp - bl_val)
    matrix = np.array(corrected, dtype=np.float64)  # (n, WIN_N)

    with np.errstate(all="ignore"):
        grand_mean = np.nanmean(matrix, axis=0)
        sem        = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(n)
        ci_lo      = grand_mean - 1.96 * sem
        ci_hi      = grand_mean + 1.96 * sem

    # Symmetric colorscale around 0
    abs_max = float(np.nanpercentile(np.abs(matrix), 99))
    vmin, vmax = -abs_max, abs_max

    fig = plt.figure(figsize=(13, 10))
    gs  = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.38,
                           left=0.32, right=0.88, top=0.91, bottom=0.07)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── heatmap ─────────────────────────────────────────────────────────────
    im = ax1.imshow(
        matrix,
        aspect="auto", origin="lower",
        extent=[TIME_AX[0], TIME_AX[-1], -0.5, n - 0.5],
        cmap=COLORMAP, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.01)
    cbar.set_label("ΔFR from pre-onset baseline (Hz)", fontsize=MIN_FONT)
    cbar.ax.tick_params(labelsize=MIN_FONT)

    ax1.axvline(0,      color="white", lw=1.2, ls="-",  alpha=0.9, label=f"{event_label} onset")
    ax1.axvline(STAT_N / LV_FS, color="white", lw=0.9, ls="--", alpha=0.7,
                label=f"stat end ({STAT_N/LV_FS:.2f} s)")
    ax1.axvspan(-FIG_PRE_N/LV_FS, 0, alpha=0.08, color="cyan",   label=bl_desc)
    ax1.axvspan(0, STAT_N/LV_FS,  alpha=0.08, color="yellow", label="response window")
    ax1.legend(fontsize=5.5, loc="upper right", framealpha=0.65)

    ax1.set_xlabel(f"time relative to {event_label} onset (s)", fontsize=MIN_FONT)
    ax1.set_ylabel("rank", fontsize=MIN_FONT)
    ax1.set_xlim(TIME_AX[0], TIME_AX[-1])
    ax1.set_ylim(-0.5, n - 0.5)
    ax1.tick_params(labelsize=MIN_FONT)

    # y-axis labels: rank n at top (index n-1), rank 1 at bottom (index 0)
    ytick_pos    = list(range(n))
    ytick_labels = []
    for i, u in enumerate(disp):
        rank = n - i          # disp[0]=worst=rank n, disp[n-1]=best=rank 1
        ytick_labels.append(
            f"#{rank:2d}  {u['exp_date']}  u{u['unit_id']}  (+{u['score']:.0f}%)"
        )
    ax1.set_yticks(ytick_pos)
    ax1.set_yticklabels(ytick_labels, fontsize=5.5)

    top1 = top20[0]
    ax1.set_title(
        f"Top-{n} units — {event_label} onset alignment\n"
        f"Baseline: {bl_desc} | Response: 0–{STAT_N/LV_FS:.2f} s post-onset | "
        f"Best: {top1['exp_date']} u{top1['unit_id']} (+{top1['score']:.0f}%)",
        fontsize=MIN_FONT + 0.5, fontweight="bold",
    )

    # ── grand mean ───────────────────────────────────────────────────────────
    ax2.plot(TIME_AX, grand_mean, color="black", lw=1.2, label="grand mean (n=20 units)")
    ax2.fill_between(TIME_AX, ci_lo, ci_hi, color="steelblue", alpha=0.35,
                     label="95% CI across units")
    ax2.axvline(0, color="black", lw=1.1, ls="-",  alpha=0.8)
    ax2.axvline(STAT_N / LV_FS, color="gray", lw=0.9, ls="--", alpha=0.7)
    ax2.axvspan(-FIG_PRE_N/LV_FS, 0,              alpha=0.08, color="cyan")
    ax2.axvspan(0,                 STAT_N/LV_FS,  alpha=0.08, color="yellow")
    ax2.set_xlabel(f"time relative to {event_label} onset (s)", fontsize=MIN_FONT)
    ax2.set_ylabel("ΔFR (Hz)", fontsize=MIN_FONT)
    ax2.set_xlim(TIME_AX[0], TIME_AX[-1])
    ax2.tick_params(labelsize=MIN_FONT)
    ax2.legend(fontsize=6, loc="upper right")
    ax2.grid(True, lw=0.4, alpha=0.4)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


render_top20(
    eth_top20,
    event_label="ETH contact",
    bl_desc="1 s pre-onset",
    out_path=os.path.join(OUT_DIR, "TOP20_eth_max_waterfall_blcorr.png"),
)

render_top20(
    sniff_top20,
    event_label="sniff",
    bl_desc="0.25 s pre-onset",
    out_path=os.path.join(OUT_DIR, "TOP20_sniff_max_waterfall_blcorr.png"),
)

print("\nDone.")

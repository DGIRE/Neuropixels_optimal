"""plot_eth_onset_aligned.py -- NP-DEMO-4

ETH-contact-onset-aligned firing rate waterfall for the two best units:
  Unit 121 (2022-06-24): greatest pct increase in MAX FR during contact vs baseline
  Unit  22 (2022-06-24): greatest absolute increase in MEAN FR during contact vs baseline

Each row = one ETH contact onset event (all contacts across all trials).
Window: -2 s to +5 s relative to each contact onset sample.
Trials with no contact are excluded; multiple contacts within one trial each get
their own row.

Output:
  FR_eth_onset_unit121_max.png
  FR_eth_onset_unit22_avg.png
saved to:
  C:\\Projects\\Repos\\Neuropixels\\_pipeline\\NP-DEMO-4\\06_figures\\revised figures\\ethanol contact
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
SESSION   = os.path.join(DATA_ROOT, "06-24-2022")
OUT_DIR   = (r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4"
             r"\06_figures\revised figures\ethanol contact")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_UNITS = {121: "max", 22: "avg"}   # unit_id -> metric label

LV_FS   = 125.0
PRE_S   = 2.0          # seconds before onset
POST_S  = 5.0          # seconds after onset
PRE_N   = int(round(PRE_S  * LV_FS))   # 250 samples
POST_N  = int(round(POST_S * LV_FS))   # 625 samples
WIN_N   = PRE_N + POST_N               # 875 samples
TIME_AX = np.arange(-PRE_N, POST_N, dtype=np.float64) / LV_FS   # -2.0 .. +4.992 s

MIN_FONT = 8
DPI      = 150
COLORMAP = "jet"


# ── load session ─────────────────────────────────────────────────────────────
exp_date = _parse_session_date(SESSION)
print(f"Loading session {exp_date} ...")

files, missing = or_validate_files(SESSION, strict=False)
files, missing = _fix_ks_dir(files, missing)
if missing:
    raise RuntimeError(f"Missing files: {missing}")

D = load_experiment_data(files)
D = detect_eth_contact(D, eth_threshold=0.05)
D = compute_sniff_rate(D, threshold_std=-0.5)

session_dur_s = float(len(D["SNF"])) / float(D["LV_Fs"])
included_ids, n_rec, n_inc = _apply_unit_inclusion(D, session_dur_s)
print(f"  {n_inc}/{n_rec} units included")

D = compute_firing_rate_50ms(D, included_ids, window_ms=50.0)
fr_per_unit      = np.asarray(D["firing_rate_per_unit"], dtype=np.float64)
eth_contact_mask = np.asarray(D["eth_contact_mask"], dtype=bool).ravel()
n_global         = fr_per_unit.shape[1]

trials = _segment_odd_tr_trials(D)
print(f"  {len(trials)} LabView trials")

# Build lookup: unit_id -> row index in fr_per_unit
uid_to_idx = {int(uid): ui for ui, uid in enumerate(included_ids)}


# ── collect contact onset windows ─────────────────────────────────────────────
def collect_onset_windows(fr_row):
    """Return (n_contacts x WIN_N) matrix, one row per contact onset."""
    rows = []
    for trial in trials:
        idx        = trial["global_idx"]
        contact_lc = eth_contact_mask[idx]
        if not contact_lc.any():
            continue
        padded = np.concatenate(([False], contact_lc, [False]))
        rises  = np.flatnonzero(~padded[:-1] & padded[1:])
        for r in rises:
            onset_g = int(idx[r])
            src_s   = onset_g - PRE_N
            src_e   = onset_g + POST_N
            row     = np.full(WIN_N, np.nan, dtype=np.float64)
            # valid source range
            vs = max(0, src_s)
            ve = min(n_global, src_e)
            dst_s = vs - src_s
            dst_e = dst_s + (ve - vs)
            row[dst_s:dst_e] = fr_row[vs:ve]
            rows.append(row)
    return np.array(rows, dtype=np.float64) if rows else np.empty((0, WIN_N))


# ── render ────────────────────────────────────────────────────────────────────
def render(matrix, unit_id, metric_label, out_path):
    n_contacts = matrix.shape[0]
    if n_contacts == 0:
        print(f"  No contact events found for unit {unit_id} — skipping.")
        return

    with np.errstate(all="ignore"):
        mean_tr = np.nanmean(matrix, axis=0)
        n_valid = np.sum(~np.isnan(matrix), axis=0).astype(float)
        sem     = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
        ci_lo   = mean_tr - 1.96 * sem
        ci_hi   = mean_tr + 1.96 * sem

    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanpercentile(matrix, 99))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.50},
    )

    title = (
        f"Firing rate aligned to ETH contact onset\n"
        f"Session: {exp_date}   Unit ID: {unit_id}   "
        f"({metric_label.upper()} responder)   "
        f"n={n_contacts} contact events"
    )
    fig.suptitle(title, fontsize=MIN_FONT + 1, fontweight="bold", y=0.98)

    # panel 1 — heatmap
    im = ax1.imshow(
        matrix,
        aspect="auto", origin="lower",
        extent=[TIME_AX[0], TIME_AX[-1], 0.5, n_contacts + 0.5],
        cmap=COLORMAP, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label("firing rate (Hz)", fontsize=MIN_FONT)
    cbar.ax.tick_params(labelsize=MIN_FONT)

    ax1.axvline(0, color="white", linewidth=1.2, linestyle="-",
                alpha=0.9, label="ETH onset")
    ax1.axvspan(-PRE_S, 0, alpha=0.07, color="cyan",  label=f"pre-onset ({PRE_S:.0f} s)")
    ax1.axvspan(0, POST_S, alpha=0.05, color="yellow", label=f"post-onset ({POST_S:.0f} s)")
    ax1.legend(fontsize=6, loc="upper right", framealpha=0.6)
    ax1.set_xlabel("time relative to ETH contact onset (s)", fontsize=MIN_FONT)
    ax1.set_ylabel("contact event #", fontsize=MIN_FONT)
    ax1.set_xlim(TIME_AX[0], TIME_AX[-1])
    ax1.set_ylim(0.5, n_contacts + 0.5)
    ax1.tick_params(labelsize=MIN_FONT)

    # panel 2 — mean + CI
    ax2.plot(TIME_AX, mean_tr, color="black", linewidth=1.2, label="mean")
    ax2.fill_between(TIME_AX, ci_lo, ci_hi, color="steelblue", alpha=0.35,
                     label="95% CI (±1.96 SEM)")
    ax2.axvline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.7)
    ax2.axvspan(-PRE_S, 0,    alpha=0.08, color="cyan")
    ax2.axvspan(0,      POST_S, alpha=0.06, color="yellow")
    ax2.set_xlabel("time relative to ETH contact onset (s)", fontsize=MIN_FONT)
    ax2.set_ylabel("firing rate (Hz)", fontsize=MIN_FONT)
    ax2.set_xlim(TIME_AX[0], TIME_AX[-1])
    ax2.tick_params(labelsize=MIN_FONT)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, linewidth=0.4, alpha=0.4)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── run for each target unit ──────────────────────────────────────────────────
for uid, label in TARGET_UNITS.items():
    if uid not in uid_to_idx:
        print(f"  Unit {uid} not found in included units — skipping.")
        continue
    print(f"\nBuilding onset-aligned matrix for unit {uid} ({label}) ...")
    fr_row = fr_per_unit[uid_to_idx[uid], :]
    matrix = collect_onset_windows(fr_row)
    print(f"  {matrix.shape[0]} contact events found")
    fname  = f"FR_eth_onset_unit{uid}_{label}.png"
    render(matrix, uid, label, os.path.join(OUT_DIR, fname))

print("\nDone.")

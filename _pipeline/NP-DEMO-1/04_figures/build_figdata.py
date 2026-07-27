"""
build_figdata.py — Figure-data builder for NP-DEMO-1.

Reads from FROZEN result objects only. For FIG-DEMO1-QC it also loads raw
SNF waveforms (display-only extraction; no scientific recomputation).

Outputs:
    figdata/FIG_DEMO1_01_units.json   — per-unit MRL + preferred phase
    figdata/FIG_DEMO1_01_animals.json — per-animal means (5 paired animals)
    figdata/FIG_DEMO1_01_stat.json    — stat annotation (p, r, n_animals)
    figdata/FIG_DEMO1_QC_meta.json    — QC counts + discard ranges per session
    figdata/FIG_DEMO1_QC_traces.npz   — raw SNF excerpts (90 s window)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-1\03_execution\results")
FIGDATA_DIR = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-1\04_figures\figdata")
KERNEL_DIR = r"C:\Projects\Repos\Neuropixels\Optimized Python"
WINDOW_S = 90.0


# ---------------------------------------------------------------------------
def _load_frozen(name: str) -> object:
    with open(RESULTS_DIR / f"{name}.json") as f:
        return json.load(f)


def build_fig01_data() -> None:
    """Extract FIG-DEMO1-01 panel data from frozen result objects."""
    mrl_all = _load_frozen("RESULT_mrl")
    phase_all = _load_frozen("RESULT_phase")
    stat = _load_frozen("RESULT_stat")

    # Index phase by (session_date, unit_id, condition) for join
    phase_idx: dict[tuple, float] = {}
    for r in phase_all:
        if r["level"] == "unit" and r["included"]:
            ph = r["preferred_phase_rad"]
            if ph is not None and np.isfinite(ph):
                phase_idx[(r["session_date"], r["unit_id"], r["condition"])] = ph

    units = []
    for r in mrl_all:
        if r["level"] != "unit" or not r["included"]:
            continue
        mrl = r["mrl"]
        if mrl is None or not np.isfinite(mrl):
            continue
        key = (r["session_date"], r["unit_id"], r["condition"])
        ph = phase_idx.get(key)
        if ph is None:
            continue
        units.append({
            "session_date": r["session_date"],
            "unit_id": r["unit_id"],
            "condition": r["condition"],
            "mrl": float(mrl),
            "preferred_phase_rad": float(ph),
            "n_valid_sniff_spikes": r["n_valid_sniff_spikes"],
        })

    animals = [
        {"session_date": r["session_date"], "condition": r["condition"], "mrl": float(r["mrl"])}
        for r in mrl_all
        if r["level"] == "animal"
    ]

    FIGDATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIGDATA_DIR / "FIG_DEMO1_01_units.json", "w") as f:
        json.dump(units, f, indent=2)
    with open(FIGDATA_DIR / "FIG_DEMO1_01_animals.json", "w") as f:
        json.dump(animals, f, indent=2)
    with open(FIGDATA_DIR / "FIG_DEMO1_01_stat.json", "w") as f:
        json.dump(stat, f, indent=2)

    n_ctrl = sum(1 for u in units if u["condition"] == "control")
    n_eth = sum(1 for u in units if u["condition"] == "ethanol")
    print(f"  FIG-01: {len(units)} unit rows ({n_ctrl} ctrl, {n_eth} eth), "
          f"{len(animals)} animal rows")


def build_qc_data() -> None:
    """Extract QC metadata (frozen) and raw SNF traces (raw data, display only)."""
    qc_counts = _load_frozen("RESULT_qc_counts")
    qc_discards = _load_frozen("RESULT_qc_discards")

    qc_meta: dict = {}
    for sess_dir, counts in qc_counts.items():
        disc_entry = qc_discards.get(sess_dir, {})
        qc_meta[sess_dir] = {
            "session_date": counts["session_date"],
            "n_sniffs": counts["n_sniffs"],
            "n_neurons": counts["n_neurons"],
            "n_trials": counts["n_trials"],
            "discards": disc_entry.get("discards", []),
        }

    FIGDATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIGDATA_DIR / "FIG_DEMO1_QC_meta.json", "w") as f:
        json.dump(qc_meta, f, indent=2)

    # Load raw SNF for display
    if KERNEL_DIR not in sys.path and os.path.isdir(KERNEL_DIR):
        sys.path.insert(0, KERNEL_DIR)
    from load_experiment_data import load_experiment_data          # noqa: E402
    from lib.or_validate_files import or_validate_files            # noqa: E402

    _KS_PROBE = ("channel_map.npy", "channel_positions.npy")

    snf_dict: dict[str, dict] = {}
    for sess_dir, meta in qc_meta.items():
        date_label = meta["session_date"]
        print(f"  Loading SNF {date_label} ...", end="", flush=True)
        try:
            files, _ = or_validate_files(sess_dir, strict=False)
            ks = files.get("ksDir") or ""
            if ks and any(not os.path.isfile(os.path.join(ks, p)) for p in _KS_PROBE):
                parent = os.path.dirname(ks)
                if all(os.path.isfile(os.path.join(parent, p)) for p in _KS_PROBE):
                    files = dict(files, ksDir=parent)

            D = load_experiment_data(files)
            snf = np.asarray(D["SNF"], dtype=np.float32).ravel()
            lv_fs = float(D["LV_Fs"])

            # Window: start 5 s before first non-sniffing discard, or at 0
            ns_discards = [d for d in meta["discards"] if d["reason"] == "non-sniffing"]
            win_start = max(0.0, ns_discards[0]["start_s"] - 5.0) if ns_discards else 0.0
            win_end = min(win_start + WINDOW_S, len(snf) / lv_fs)

            i0 = int(win_start * lv_fs)
            i1 = int(win_end * lv_fs)
            snf_dict[date_label] = {
                "snf": snf[i0:i1].tolist(),
                "lv_fs": lv_fs,
                "win_start_s": float(win_start),
                "win_end_s": float(win_end),
            }
            print(f" ok [{win_start:.0f}–{win_end:.0f} s, {i1 - i0} samples]")
        except Exception as exc:
            print(f" ERROR: {exc}")
            snf_dict[date_label] = {"error": str(exc)}

    with open(FIGDATA_DIR / "FIG_DEMO1_QC_traces.json", "w") as f:
        json.dump(snf_dict, f, indent=2)


if __name__ == "__main__":
    print("Building FIG-DEMO1-01 data from frozen results...")
    build_fig01_data()
    print("Building FIG-DEMO1-QC data (raw SNF + frozen metadata)...")
    build_qc_data()
    print("figdata build complete.")

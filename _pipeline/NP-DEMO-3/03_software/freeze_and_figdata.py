"""
NP-DEMO-3 — Freeze results and derive figure data.

Step 1: Copy RESULT-*.yaml into 04_results/frozen/
Step 2: Derive figure data for all 5 contracted figures from frozen results.
        Figure data goes to 06_figures/figure_data/

This script loads raw session data only for the PSTH-based figures (QC-PSTH and EXAMPLES)
where raw trace computation is unavoidable; all other figure data derives purely from
the frozen RESULT objects.
"""
import sys
import os
import shutil
import yaml
import numpy as np

PROJECT_ROOT = r"C:\Projects\Repos\Neuropixels"
KERNEL = os.path.join(PROJECT_ROOT, "Optimized Python")
DEMO3_SW = os.path.join(PROJECT_ROOT, r"_pipeline\NP-DEMO-3\03_software")
RESULTS_DIR = os.path.join(PROJECT_ROOT, r"_pipeline\NP-DEMO-3\04_results")
FROZEN_DIR = os.path.join(RESULTS_DIR, "frozen")
FIGDATA_DIR = os.path.join(PROJECT_ROOT, r"_pipeline\NP-DEMO-3\06_figures\figure_data")
DATA_ROOT = os.path.join(PROJECT_ROOT, "DATA")

os.makedirs(FROZEN_DIR, exist_ok=True)
os.makedirs(FIGDATA_DIR, exist_ok=True)

for p in [KERNEL, DEMO3_SW]:
    if p not in sys.path:
        sys.path.insert(0, p)


# =========================================================================
# Step 1: Freeze result objects
# =========================================================================
print("=== Step 1: Freezing result objects ===")
result_files = [f for f in os.listdir(RESULTS_DIR)
                if f.startswith("RESULT-") and f.endswith(".yaml")]
for fn in result_files:
    src = os.path.join(RESULTS_DIR, fn)
    dst = os.path.join(FROZEN_DIR, fn)
    shutil.copy2(src, dst)
    print(f"  Frozen: {fn}")


def load_frozen(name):
    path = os.path.join(FROZEN_DIR, f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def save_figdata(fig_id, data):
    path = os.path.join(FIGDATA_DIR, f"{fig_id}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True, allow_unicode=True)
    print(f"  Saved figure data: {fig_id}.yaml")


# Load frozen results
eth_mask = load_frozen("RESULT-eth-mask")
sniff_qc = load_frozen("RESULT-sniff-qc")
discard_log = load_frozen("RESULT-discard-log")
mrl_per_unit = load_frozen("RESULT-mrl-per-unit")
mrl_per_exp = load_frozen("RESULT-mrl-per-experiment")
lme = load_frozen("RESULT-lme")
wilcoxon = load_frozen("RESULT-wilcoxon")
examples_top5 = load_frozen("RESULT-examples-top5")


# =========================================================================
# Step 2: Figure data derivation
# =========================================================================
print("\n=== Step 2: Deriving figure data ===")


# -------------------------------------------------------------------------
# FIG-DEMO3-QC-SNIFF: per-experiment SNF + ETH example traces with thresholds
# Source results: RESULT-eth-mask, RESULT-sniff-qc, RESULT-discard-log
# We load raw data to extract trace snippets, but all stats come from results.
# -------------------------------------------------------------------------
print("\n--- FIG-DEMO3-QC-SNIFF ---")
from load_experiment_data import load_experiment_data
from lib.or_validate_files import or_validate_files
from analyses.compute_sniff_phase import compute_sniff_phase
from detect_eth_contact import detect_eth_contact

SESSION_FOLDER_MAP = {
    "2021-11-01": "11-01-2021",
    "2021-11-03": "11-03-2021",
    "2021-12-15": "12-15-2021",
    "2022-05-17": "5-17-2022",
    "2022-06-24": "06-24-2022",
    "2022-09-14": "09-14-2022",
}
_KS_PROBE_FILES = ("channel_map.npy", "channel_positions.npy")

TRACE_WINDOW_S = 30.0  # 30 seconds representative window (centered around midpoint)

qc_sniff_panels = []
for exp in ["2021-11-01", "2021-11-03", "2021-12-15", "2022-05-17", "2022-06-24", "2022-09-14"]:
    folder = SESSION_FOLDER_MAP[exp]
    session_dir = os.path.join(DATA_ROOT, folder)
    files, missing = or_validate_files(session_dir, strict=False)
    if files["ksDir"] and any(
        not os.path.isfile(os.path.join(files["ksDir"], f)) for f in _KS_PROBE_FILES
    ):
        parent = os.path.dirname(files["ksDir"])
        if all(os.path.isfile(os.path.join(parent, f)) for f in _KS_PROBE_FILES):
            files["ksDir"] = parent
            missing = [m for m in missing if "channel_map" not in m and "channel_positions" not in m]
    D = load_experiment_data(files)
    D = compute_sniff_phase(D, threshold_std=-0.5)
    D = detect_eth_contact(D, eth_threshold=0.05)

    LV_Fs = float(D["LV_Fs"])
    n_samples = len(D["SNF"])
    duration_s = n_samples / LV_Fs

    # 30-second window starting at 25% into the recording
    start_s = duration_s * 0.25
    end_s = min(start_s + TRACE_WINDOW_S, duration_s)
    start_idx = int(start_s * LV_Fs)
    end_idx = int(end_s * LV_Fs)

    t = np.arange(start_idx, end_idx) / LV_Fs
    snf_z_slice = np.asarray(D["SNF_z"])[start_idx:end_idx].tolist()
    eth_ms_slice = np.asarray(D["ETH_ms"])[start_idx:end_idx].tolist()
    sniff_onsets_s = np.asarray(D["sniff_onsets_s"])
    onsets_in_window = sniff_onsets_s[(sniff_onsets_s >= start_s) & (sniff_onsets_s < end_s)]

    # Get stats from frozen results
    qc_row = next((r for r in sniff_qc if r["experiment"] == exp), {})
    eth_mask_row = eth_mask.get(exp, {})

    qc_sniff_panels.append({
        "experiment": exp,
        "t_start_s": float(start_s),
        "t_end_s": float(end_s),
        "t_axis": [float(x) for x in t.tolist()],
        "SNF_z": snf_z_slice,
        "ETH_ms": eth_ms_slice,
        "sniff_onset_threshold": -0.5,  # for red dashed line on SNF panel
        "eth_contact_threshold": 0.05,   # for red dashed line on ETH panel
        "sniff_onsets_in_window_s": [float(x) for x in onsets_in_window.tolist()],
        "n_sniffs": qc_row.get("n_sniffs"),
        "n_neurons": qc_row.get("n_neurons"),
        "n_trials": qc_row.get("n_trials"),
        "duration_min": qc_row.get("duration_min"),
        "pct_usable_sniff_time": qc_row.get("pct_usable_sniff_time"),
    })
    print(f"  {exp}: window [{start_s:.1f}s, {end_s:.1f}s], "
          f"{len(onsets_in_window)} onsets in window")

save_figdata("FIG-DEMO3-QC-SNIFF", {"panels": qc_sniff_panels})


# -------------------------------------------------------------------------
# FIG-DEMO3-QC-PSTH: sniff-locked PSTHs for strongest and weakest MRL units
# Sources: RESULT-mrl-per-unit (condition-agnostic MRL)
# Must load raw data for the specific units to compute PSTHs.
# -------------------------------------------------------------------------
print("\n--- FIG-DEMO3-QC-PSTH ---")
from analyses.compute_sniff_psth import compute_sniff_psth
from analyses.compute_spike_phase import compute_spike_phase
from analyses._common import matlab_round

# Find globally strongest and weakest MRL_cond_agnostic (valid units only)
valid_units = [u for u in mrl_per_unit
               if u.get("MRL_cond_agnostic") is not None
               and np.isfinite(float(u["MRL_cond_agnostic"]))]
valid_units.sort(key=lambda u: float(u["MRL_cond_agnostic"]))
weakest_unit = valid_units[0]
strongest_unit = valid_units[-1]
print(f"  Strongest: unit {strongest_unit['unit_id']} ({strongest_unit['experiment']}), "
      f"MRL_cond_agnostic={strongest_unit['MRL_cond_agnostic']:.4f}")
print(f"  Weakest:   unit {weakest_unit['unit_id']} ({weakest_unit['experiment']}), "
      f"MRL_cond_agnostic={weakest_unit['MRL_cond_agnostic']:.4f}")

psth_panels = []
for u_info in [strongest_unit, weakest_unit]:
    folder = SESSION_FOLDER_MAP[u_info["experiment"]]
    session_dir = os.path.join(DATA_ROOT, folder)
    files, missing = or_validate_files(session_dir, strict=False)
    if files["ksDir"] and any(
        not os.path.isfile(os.path.join(files["ksDir"], f)) for f in _KS_PROBE_FILES
    ):
        parent = os.path.dirname(files["ksDir"])
        if all(os.path.isfile(os.path.join(parent, f)) for f in _KS_PROBE_FILES):
            files["ksDir"] = parent
    D = load_experiment_data(files)
    D = compute_sniff_phase(D, threshold_std=-0.5)
    D = compute_spike_phase(D)

    unitIDs = np.asarray(D["unitIDs"]).ravel()
    uid = int(u_info["unit_id"])
    if uid not in unitIDs:
        print(f"  WARNING: unit {uid} not found in {u_info['experiment']}, skipping PSTH")
        continue
    unit_idx = int(np.searchsorted(unitIDs, uid))

    psth, centers, n_events, D_with_psth = compute_sniff_psth(D, bin_ms=10.0, attach=True)

    psth_panels.append({
        "unit_id": uid,
        "experiment": u_info["experiment"],
        "MRL_cond_agnostic": float(u_info["MRL_cond_agnostic"]),
        "label": "strongest" if u_info is strongest_unit else "weakest",
        "psth": [float(x) for x in psth[unit_idx].tolist()],
        "centers_phase": [float(x) for x in D_with_psth["centers_phase"].tolist()],
        "centers_ms": [float(x) for x in D_with_psth["centers_ms"].tolist()],
        "n_events": int(n_events),
    })
    print(f"  {u_info['experiment']} unit {uid}: PSTH computed ({n_events} sniff events)")

save_figdata("FIG-DEMO3-QC-PSTH", {"panels": psth_panels})


# -------------------------------------------------------------------------
# FIG-DEMO3-RESULTS-ANIMAL: paired per-experiment MRL (Wilcoxon result)
# Sources: RESULT-mrl-per-experiment, RESULT-wilcoxon
# Pure derivation from frozen results.
# -------------------------------------------------------------------------
print("\n--- FIG-DEMO3-RESULTS-ANIMAL ---")
animal_pairs = []
for row in mrl_per_exp:
    animal_pairs.append({
        "experiment": row["experiment"],
        "mean_MRL_ethanol": float(row["mean_MRL_ethanol"]),
        "mean_MRL_control": float(row["mean_MRL_control"]),
    })

figdata_animal = {
    "pairs": animal_pairs,
    "n_experiments": int(wilcoxon["n_pairs"]),
    "wilcoxon_p": float(wilcoxon["p_value"]),
    "wilcoxon_statistic": float(wilcoxon["statistic"]),
    "rank_biserial_r": float(wilcoxon["rank_biserial_r"]),
    "x_labels": ["control", "ethanol"],
    "threshold_significance": 0.05,
}
save_figdata("FIG-DEMO3-RESULTS-ANIMAL", figdata_animal)


# -------------------------------------------------------------------------
# FIG-DEMO3-RESULTS-UNIT: per-unit MRL scatter (LME result)
# Sources: RESULT-mrl-per-unit, RESULT-lme
# Pure derivation from frozen results.
# -------------------------------------------------------------------------
print("\n--- FIG-DEMO3-RESULTS-UNIT ---")
unit_scatter = []
for u in mrl_per_unit:
    if u.get("MRL_ethanol") is not None and u.get("MRL_control") is not None:
        unit_scatter.append({
            "unit_id": int(u["unit_id"]),
            "experiment": str(u["experiment"]),
            "MRL_ethanol": float(u["MRL_ethanol"]),
            "MRL_control": float(u["MRL_control"]),
        })

figdata_unit = {
    "unit_points": unit_scatter,
    "n_units": int(lme["n_units"]),
    "n_experiments": int(lme["n_experiments"]),
    "lme_p": float(lme["p_value"]),
    "standardized_fixed_effect": float(lme["standardized_fixed_effect"]),
    "model_formula": str(lme["model_formula"]),
    "x_labels": ["control", "ethanol"],
}
save_figdata("FIG-DEMO3-RESULTS-UNIT", figdata_unit)


# -------------------------------------------------------------------------
# FIG-DEMO3-EXAMPLES: top-5 units by delta_MRL — PSTHs + probe maps
# Sources: RESULT-examples-top5, RESULT-mrl-per-unit
# Needs raw data for PSTHs; probe map data from D.
# -------------------------------------------------------------------------
print("\n--- FIG-DEMO3-EXAMPLES ---")
from detect_eth_contact import detect_eth_contact

examples_panels = []
for top_unit in examples_top5:
    uid = int(top_unit["unit_id"])
    exp = str(top_unit["experiment"])
    delta_mrl = float(top_unit["delta_MRL"])

    # Load unit details from mrl_per_unit
    unit_detail = next(
        (u for u in mrl_per_unit
         if int(u["unit_id"]) == uid and str(u["experiment"]) == exp),
        None
    )

    folder = SESSION_FOLDER_MAP[exp]
    session_dir = os.path.join(DATA_ROOT, folder)
    files, missing = or_validate_files(session_dir, strict=False)
    if files["ksDir"] and any(
        not os.path.isfile(os.path.join(files["ksDir"], f)) for f in _KS_PROBE_FILES
    ):
        parent = os.path.dirname(files["ksDir"])
        if all(os.path.isfile(os.path.join(parent, f)) for f in _KS_PROBE_FILES):
            files["ksDir"] = parent
    D = load_experiment_data(files)
    D = compute_sniff_phase(D, threshold_std=-0.5)
    D = detect_eth_contact(D, eth_threshold=0.05)
    D = compute_spike_phase(D)

    unitIDs = np.asarray(D["unitIDs"]).ravel()
    if uid not in unitIDs:
        print(f"  WARNING: unit {uid} not found in {exp}, skipping")
        continue
    unit_idx = int(np.searchsorted(unitIDs, uid))

    LV_Fs = float(D["LV_Fs"])
    spikeTimes = np.asarray(D["spikeTimes"], dtype=np.float64).ravel()
    clu = np.asarray(D["sp"]["clu"]).ravel()
    spike_SNF_PH = np.asarray(D["spike_SNF_PH"], dtype=np.float64).ravel()
    eth_mask_arr = np.asarray(D["eth_contact_mask"])

    # Condition assignment
    lv_idx = matlab_round(spikeTimes * LV_Fs).astype(np.int64)
    np.clip(lv_idx, 0, len(eth_mask_arr) - 1, out=lv_idx)
    is_ethanol_spike = eth_mask_arr[lv_idx]

    u_mask = (clu == uid)
    valid = spike_SNF_PH >= 0.0

    # Ethanol PSTH
    eth_times = spikeTimes[u_mask & valid & is_ethanol_spike]
    D_eth = dict(D)
    D_eth["spikeTimes"] = eth_times
    D_eth["sp"] = dict(D["sp"])
    D_eth["sp"]["clu"] = np.zeros(len(eth_times), dtype=int)
    D_eth["unitIDs"] = np.array([uid])
    _, _, _, D_eth_psth = compute_sniff_psth(D_eth, bin_ms=10.0, attach=True)

    # Control PSTH
    ctrl_times = spikeTimes[u_mask & valid & ~is_ethanol_spike]
    D_ctrl = dict(D)
    D_ctrl["spikeTimes"] = ctrl_times
    D_ctrl["sp"] = dict(D["sp"])
    D_ctrl["sp"]["clu"] = np.zeros(len(ctrl_times), dtype=int)
    D_ctrl["unitIDs"] = np.array([uid])
    _, _, _, D_ctrl_psth = compute_sniff_psth(D_ctrl, bin_ms=10.0, attach=True)

    # Probe location data
    xcoords = np.asarray(D.get("xcoords", []))
    ycoords = np.asarray(D.get("ycoords", []))
    unit_depth = float(D["unitDepths"][unit_idx]) if "unitDepths" in D else float("nan")
    unit_amp = float(D["unitAmps"][unit_idx]) if "unitAmps" in D else float("nan")
    unit_fr = float(D["unitFiringRate"][unit_idx]) if "unitFiringRate" in D else float("nan")

    examples_panels.append({
        "unit_id": uid,
        "experiment": exp,
        "delta_MRL": delta_mrl,
        "MRL_ethanol": float(unit_detail["MRL_ethanol"]) if unit_detail else float("nan"),
        "MRL_control": float(unit_detail["MRL_control"]) if unit_detail else float("nan"),
        "n_valid_spikes_eth": int(unit_detail["n_valid_spikes_eth"]) if unit_detail else 0,
        "n_valid_spikes_ctrl": int(unit_detail["n_valid_spikes_ctrl"]) if unit_detail else 0,
        "psth_ethanol": [float(x) for x in D_eth_psth["psth_phase"][0].tolist()],
        "psth_control": [float(x) for x in D_ctrl_psth["psth_phase"][0].tolist()],
        "centers_phase": [float(x) for x in D_eth_psth["centers_phase"].tolist()],
        "centers_ms": [float(x) for x in D_eth_psth["centers_ms"].tolist()],
        "n_events_eth": int(len(np.asarray(D["sniff_onsets_s"]))),
        "probe_unit_depth_um": unit_depth,
        "probe_unit_amp": unit_amp,
        "probe_unit_fr": unit_fr,
        "all_unit_depths": [float(x) for x in np.asarray(D["unitDepths"]).tolist()],
        "all_unit_amps": [float(x) for x in np.asarray(D["unitAmps"]).tolist()],
        "all_unit_frs": [float(x) for x in np.asarray(D["unitFiringRate"]).tolist()],
        "xcoords": [float(x) for x in xcoords.tolist()] if xcoords.size > 0 else [],
        "ycoords": [float(x) for x in ycoords.tolist()] if ycoords.size > 0 else [],
    })
    print(f"  Unit {uid} ({exp}): delta_MRL={delta_mrl:.4f}, PSTHs computed")

save_figdata("FIG-DEMO3-EXAMPLES", {"panels": examples_panels})

print("\n=== All figure data derived successfully ===")
print(f"Frozen results in: {FROZEN_DIR}")
print(f"Figure data in:    {FIGDATA_DIR}")

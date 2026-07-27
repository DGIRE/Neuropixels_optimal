"""
build_figdata.py -- NP-DEMO-2 figure-data builder (Blueprint v2, FIGURES_DATA_OK)
==================================================================================
Builds the figdata_FIG-DEMO2-*.yaml files (plus companion .npy PSTH-array
files) consumed by the figure-rendering step.

Sources, per figure:
    FIG-DEMO2-RES-EXP  <- RESULT-mrl-animal.yaml + RESULT-stat-animal.yaml
    FIG-DEMO2-RES-UNIT <- RESULT-mrl-unit.yaml + RESULT-stat-unit.yaml
    FIG-DEMO2-RES-EX   <- RESULT-psth-examples.yaml + RESULT-unit-locations.yaml
                          (+ RESULT-delta-mrl.yaml provenance for the top-5 ranking)
    FIG-DEMO2-QC       <- RESULT-qc-counts.yaml + RESULT-eth-threshold-log.yaml +
                          RESULT-qc-discards.yaml + a FRESH raw-session reload
                          (load_experiment_data -> compute_sniff_phase ->
                          threshold_eth) for the SNF_z / ETH display traces only
    FIG-DEMO2-QC-PSTH  <- RESULT-psth-extremes.yaml

RULE: every numeric value written to a figdata file must trace back to a
frozen RESULT-* object or a kernel call on raw session data under DATA\\ --
never a hand-typed statistic.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import yaml

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PIPELINE_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2"
RESULTS_DIR = os.path.join(PIPELINE_DIR, "04_results")
FIGDATA_DIR = os.path.join(PIPELINE_DIR, "06_figures", "figure_data")
SOFTWARE_DIR = os.path.join(PIPELINE_DIR, "03_software")
DATA_ROOT = r"C:\Projects\Repos\Neuropixels\DATA"
KERNEL_DIR = r"C:\Projects\Repos\Neuropixels\Optimized Python"

if KERNEL_DIR not in sys.path and os.path.isdir(KERNEL_DIR):
    sys.path.insert(0, KERNEL_DIR)
if SOFTWARE_DIR not in sys.path and os.path.isdir(SOFTWARE_DIR):
    sys.path.insert(0, SOFTWARE_DIR)

from load_experiment_data import load_experiment_data           # noqa: E402
from lib.or_validate_files import or_validate_files              # noqa: E402
from analyses import compute_sniff_phase, threshold_eth          # noqa: E402
from np_demo2_analysis import _fix_ks_dir                        # noqa: E402  (8-line HAZ-05 helper, reused not copied)

SESSION_DIRS = {
    "2021-11-01": os.path.join(DATA_ROOT, "11-01-2021"),
    "2021-11-03": os.path.join(DATA_ROOT, "11-03-2021"),
    "2021-12-15": os.path.join(DATA_ROOT, "12-15-2021"),
    "2022-05-17": os.path.join(DATA_ROOT, "5-17-2022"),
    "2022-06-24": os.path.join(DATA_ROOT, "06-24-2022"),
    "2022-09-14": os.path.join(DATA_ROOT, "09-14-2022"),
}

DEV001_FLAG = (
    "2021-11-03 session has degenerate MRL_control values (0-4 ctrl spikes "
    "per unit). See 08_traceability/deviations.yaml DEV-001. Results require "
    "human review before use."
)

DOWNSAMPLE_STRIDE = 4  # every 4th sample -> ~31 Hz display resolution


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _load_yaml(name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path) as fh:
        return yaml.safe_load(fh)


def _to_native(obj):
    """Recursively convert numpy scalar/array types (and NaN) to
    YAML-safe native Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _dump_yaml(obj, filename):
    os.makedirs(FIGDATA_DIR, exist_ok=True)
    path = os.path.join(FIGDATA_DIR, filename)
    with open(path, "w") as fh:
        yaml.safe_dump(_to_native(obj), fh, sort_keys=False, default_flow_style=False)
    print(f"  wrote {path}")


def _is_finite_number(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


# --------------------------------------------------------------------------
# 1. figdata_FIG-DEMO2-RES-EXP.yaml -- animal-level results
# --------------------------------------------------------------------------
def build_res_exp():
    mrl_animal = _load_yaml("RESULT-mrl-animal.yaml")["sessions"]
    stat_animal = _load_yaml("RESULT-stat-animal.yaml")

    sessions = []
    for row in mrl_animal:
        sessions.append(dict(
            session_date=row["session"],
            mean_mrl_ethanol=row["mean_mrl_ethanol"],
            mean_mrl_control=row["mean_mrl_control"],
            n_units_included=row["n_units_included"],
            dev001_degenerate_session=(row["session"] == "2021-11-03"),
        ))

    stat = dict(
        pvalue=stat_animal["pvalue"],
        effect_size_r=stat_animal["effect_size"],
        direction=stat_animal["direction"],
        n_animals=stat_animal["n_animals"],
    )

    out = dict(
        sessions=sessions,
        stat=stat,
        caption_n_animals=stat["n_animals"],
        dev001_flag=DEV001_FLAG,
    )
    _dump_yaml(out, "figdata_FIG-DEMO2-RES-EXP.yaml")


# --------------------------------------------------------------------------
# 2. figdata_FIG-DEMO2-RES-UNIT.yaml -- unit-level results
# --------------------------------------------------------------------------
def build_res_unit():
    mrl_unit = _load_yaml("RESULT-mrl-unit.yaml")
    stat_unit = _load_yaml("RESULT-stat-unit.yaml")

    units = []
    n_units_total = 0
    for row in mrl_unit:
        eth = row["mrl_ethanol"]
        ctrl = row["mrl_control"]
        eth_ok = _is_finite_number(eth)
        ctrl_ok = _is_finite_number(ctrl)
        delta = (eth - ctrl) if (eth_ok and ctrl_ok) else None
        if eth_ok and ctrl_ok:
            n_units_total += 1
        units.append(dict(
            unit_id=row["unit_id"],
            session=row["session"],
            mrl_ethanol=eth if eth_ok else None,
            mrl_control=ctrl if ctrl_ok else None,
            delta_mrl=delta,
        ))

    stat = dict(
        pvalue=stat_unit["pvalue"],
        coefficient=stat_unit["coefficient"],
        standardized_effect_size=stat_unit["standardized_effect_size"],
        n_units=stat_unit["n_units"],
        n_animals=stat_unit["n_animals"],
    )

    out = dict(units=units, stat=stat, n_units_total=n_units_total)
    _dump_yaml(out, "figdata_FIG-DEMO2-RES-UNIT.yaml")


# --------------------------------------------------------------------------
# 3. figdata_FIG-DEMO2-RES-EX.yaml (+ .npy) -- top-5 |delta_MRL| examples
# --------------------------------------------------------------------------
def build_res_ex():
    psth_examples = _load_yaml("RESULT-psth-examples.yaml")["examples"]
    unit_locations = _load_yaml("RESULT-unit-locations.yaml")["examples"]

    loc_lookup = {(row["unit_id"], row["session"]): row for row in unit_locations}

    # RESULT-psth-examples.yaml is already the top-5 |delta_MRL| set; sort
    # defensively by abs_delta_mrl (descending) to guarantee figure order.
    ranked = sorted(psth_examples, key=lambda r: r["abs_delta_mrl"], reverse=True)[:5]

    examples = []
    all_2021_11_03 = True
    for row in ranked:
        key = (row["unit_id"], row["session"])
        loc = loc_lookup.get(key, {})
        if row["session"] != "2021-11-03":
            all_2021_11_03 = False
        examples.append(dict(
            unit_id=row["unit_id"],
            session=row["session"],
            abs_delta_mrl=row["abs_delta_mrl"],
            delta_mrl=row["delta_mrl"],
            psth_eth=row["psth_eth"],
            psth_ctrl=row["psth_ctrl"],
            centers_phase=row["centers_phase"],
            probe_location=dict(
                unit_depth_um=loc.get("unit_depth_um"),
                unit_xcoord_um=loc.get("unit_xcoord_um"),
                shank=loc.get("shank"),
                unit_firing_rate=loc.get("unit_firing_rate"),
            ),
        ))

    dev001_warning = (
        "All 5 top-|delta_MRL| examples in this figure come from session "
        "2021-11-03, which has degenerate control-condition MRL values "
        "(see DEV-001, 08_traceability/deviations.yaml): 374/429 included "
        "units have 0 control-condition spikes (MRL_control undefined) and "
        "most of the rest have only 1-4 control spikes, making a pooled "
        "MRL_control of ~0.914 for this session a near-single-spike "
        "artifact, not a reliable phase-locking estimate. These examples "
        "should not be presented as typical or representative without "
        "explicit acknowledgement of this limitation."
    )

    out = dict(
        examples=examples,
        all_examples_from_2021_11_03=all_2021_11_03,
        dev001_warning=dev001_warning,
        dev001_flag=DEV001_FLAG,
    )
    _dump_yaml(out, "figdata_FIG-DEMO2-RES-EX.yaml")

    # companion .npy with the PSTH arrays for the same 5 examples
    npy_records = []
    for ex in examples:
        npy_records.append(dict(
            unit_id=ex["unit_id"], session=ex["session"],
            psth_eth=np.array(ex["psth_eth"], dtype=np.float64),
            psth_ctrl=np.array(ex["psth_ctrl"], dtype=np.float64),
            centers_phase=np.array(ex["centers_phase"], dtype=np.float64),
        ))
    npy_path = os.path.join(FIGDATA_DIR, "figdata_FIG-DEMO2-RES-EX.npy")
    np.save(npy_path, np.array(npy_records, dtype=object))
    print(f"  wrote {npy_path}")


# --------------------------------------------------------------------------
# 4. figdata_FIG-DEMO2-QC.yaml -- per-session SNF+ETH traces with thresholds
#    (raw-data reload; everything else pulled from frozen RESULT-* objects)
# --------------------------------------------------------------------------
def build_qc():
    qc_counts = {row["session_date"]: row for row in _load_yaml("RESULT-qc-counts.yaml")["sessions"]}
    eth_log = {row["session_date"]: row for row in _load_yaml("RESULT-eth-threshold-log.yaml")["sessions"]}
    discards_all = _load_yaml("RESULT-qc-discards.yaml")["discards"]

    discards_by_session = {}
    for d in discards_all:
        discards_by_session.setdefault(d["experiment"], []).append(
            dict(start_s=d["start_s"], end_s=d["end_s"], reason=d["reason"])
        )

    sessions_out = []
    for sd, session_dir in SESSION_DIRS.items():
        if sd not in qc_counts or sd not in eth_log:
            print(f"  WARNING: {sd} missing from RESULT-qc-counts/eth-threshold-log; skipping")
            continue

        print(f"  Loading raw data for QC trace: {sd} ({session_dir})")
        files, missing = or_validate_files(session_dir, strict=False)
        files, missing = _fix_ks_dir(files, missing)
        if missing:
            raise FileNotFoundError(
                f"Required files missing in {session_dir!r}:\n  " + "\n  ".join(missing)
            )
        D = load_experiment_data(files)
        D = compute_sniff_phase(D, threshold_std=-0.5)

        final_threshold = float(eth_log[sd]["final_threshold"])
        D2 = threshold_eth(D, eth_threshold=final_threshold)

        SNF_z = np.asarray(D2["SNF_z"], dtype=np.float64).ravel()
        ETH = np.asarray(D2["ETH"], dtype=np.float64).ravel()
        LV_Fs = float(D2["LV_Fs"])
        sniff_thr = float(D2["sniff_thr"])
        time_s = np.arange(SNF_z.size, dtype=np.float64) / LV_Fs

        SNF_z_ds = SNF_z[::DOWNSAMPLE_STRIDE]
        ETH_ds = ETH[::DOWNSAMPLE_STRIDE]
        time_s_ds = time_s[::DOWNSAMPLE_STRIDE]

        qc = qc_counts[sd]
        sessions_out.append(dict(
            session_date=sd,
            SNF_z=SNF_z_ds.tolist(),
            ETH=ETH_ds.tolist(),
            time_s=time_s_ds.tolist(),
            sniff_thr=sniff_thr,
            eth_threshold=final_threshold,
            LV_Fs=LV_Fs,
            downsample_stride=DOWNSAMPLE_STRIDE,
            n_sniffs=qc["n_sniffs"],
            n_neurons=qc["n_neurons"],
            n_trials=qc["n_trials"],
            length_min=qc["length_min"],
            pct_usable_sniffs=qc["pct_usable_sniffs"],
            discards=discards_by_session.get(sd, []),
        ))

    _dump_yaml(dict(sessions=sessions_out), "figdata_FIG-DEMO2-QC.yaml")


# --------------------------------------------------------------------------
# 5. figdata_FIG-DEMO2-QC-PSTH.yaml (+ .npy) -- strongest/weakest units
# --------------------------------------------------------------------------
def build_qc_psth():
    extremes = _load_yaml("RESULT-psth-extremes.yaml")

    def _clean(entry):
        return dict(
            unit_id=entry["unit_id"],
            session=entry["session"],
            pooled_mrl=entry["pooled_mrl"],
            psth_phase=entry["psth_phase"],
            centers_phase=entry["centers_phase"],
            n_events=entry["n_events"],
        )

    out = dict(
        strongest_unit=_clean(extremes["strongest_unit"]),
        weakest_unit=_clean(extremes["weakest_unit"]),
    )
    _dump_yaml(out, "figdata_FIG-DEMO2-QC-PSTH.yaml")

    npy_records = []
    for label in ("strongest_unit", "weakest_unit"):
        e = out[label]
        npy_records.append(dict(
            label=label, unit_id=e["unit_id"], session=e["session"],
            psth_phase=np.array(e["psth_phase"], dtype=np.float64),
            centers_phase=np.array(e["centers_phase"], dtype=np.float64),
        ))
    npy_path = os.path.join(FIGDATA_DIR, "figdata_FIG-DEMO2-QC-PSTH.npy")
    np.save(npy_path, np.array(npy_records, dtype=object))
    print(f"  wrote {npy_path}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    os.makedirs(FIGDATA_DIR, exist_ok=True)

    print("Building figdata_FIG-DEMO2-RES-EXP.yaml ...")
    build_res_exp()

    print("Building figdata_FIG-DEMO2-RES-UNIT.yaml ...")
    build_res_unit()

    print("Building figdata_FIG-DEMO2-RES-EX.yaml (+ .npy) ...")
    build_res_ex()

    print("Building figdata_FIG-DEMO2-QC.yaml (raw session reload) ...")
    build_qc()

    print("Building figdata_FIG-DEMO2-QC-PSTH.yaml (+ .npy) ...")
    build_qc_psth()

    print(f"\nDone. Files in: {FIGDATA_DIR}")
    for f in sorted(os.listdir(FIGDATA_DIR)):
        print(" ", f)


if __name__ == "__main__":
    main()

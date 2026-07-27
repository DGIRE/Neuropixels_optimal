"""fill_frozen_result_gaps.py -- one-off driver to fill two contract-required
gaps identified by the report-author agent (2026-07-25) that were never
persisted during the P3 composing run:

  Gap 1 (OUT-009/010, feeds FIG-PAPER1-SPECTRO): per-window time-resolved
  LFP-power/SNF-power/coherence spectrograms for the representative session
  (06-24-2022) -> RESULT-example-spectrogram.npz

  Gap 2 (feeds FIG-PAPER1-SNIFFRATE's histogram_or_kde): raw per-sniff
  instantaneous-rate arrays for all 6 sessions -> RESULT-sniff-rate-raw.npz

Both are cheap: reuse already-cached per-session data (lfp_full.npy,
snf_lfp.npy, control_valid_windows.npy, or just the small LabView .dat via
or_loaddat) -- no raw .bin reload, no re-run of the expensive circular-shift
null. NOT a deviation (see coordinator message 2026-07-25) -- completing
already-specified, already-contracted work, not a new scientific/data
judgment call.

Usage:
    "$RW_PY" _pipeline/NP-PAPER-1/03_software/fill_frozen_result_gaps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

_SOFTWARE_DIR = Path(__file__).resolve().parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from np_paper1_analysis import (  # noqa: E402
    RESULTS_DIR, SESSION_DIRNAMES, yaml_safe,
    build_example_spectrogram, build_sniff_rate_raw,
)


def main() -> None:
    print("=== Gap 1: RESULT-example-spectrogram.npz ===")
    spec = build_example_spectrogram(RESULTS_DIR, verbose=True)

    spec_npz = {
        "freqs": spec["freqs"],
        "window_center_s": spec["window_center_s"],
        "sniff_rate_at_window_hz": spec["sniff_rate_at_window_hz"],
        "depth_channel_indices": np.asarray(spec["depth_channel_indices"]),
        "depth_ycoords_um": np.asarray(spec["depth_ycoords_um"]),
        "snf_psd_spectrogram": spec["snf_psd_spectrogram"],
    }
    for d_ord, arr in spec["lfp_psd_spectrogram_by_depth"].items():
        spec_npz[f"lfp_psd_spectrogram_depth{d_ord}"] = arr
    for d_ord, arr in spec["coh_spectrogram_by_depth"].items():
        spec_npz[f"coh_spectrogram_depth{d_ord}"] = arr

    np.savez(RESULTS_DIR / "RESULT-example-spectrogram.npz", **spec_npz)
    with open(RESULTS_DIR / "RESULT-example-spectrogram_meta.yaml", "w") as f:
        yaml.safe_dump(yaml_safe({
            "session": spec["session"],
            "n_windows": int(spec["window_center_s"].size),
            "n_freqs": int(spec["freqs"].size),
            "depth_channel_indices": spec["depth_channel_indices"],
            "depth_ycoords_um": spec["depth_ycoords_um"],
            "note": (
                "OUT-009/010 gap-fill (2026-07-25, not a deviation -- see "
                "results_manifest.yaml history). Representative session is "
                "the SAME one chosen for RESULT-example-traces (most "
                "retained windows). coh_spectrogram_depth<N> reuses "
                "multitaper_psd_coherence's own coh_per_window field "
                "(already computed internally, just not previously "
                "persisted); lfp_psd_spectrogram_depth<N> / "
                "snf_psd_spectrogram are per-window single-window calls to "
                "the SAME unmodified function (n_windows=1 in its own scale "
                "formula) -- verified (not merely asserted) that per-window "
                "coherence matches coh_per_window bit-for-bit and that "
                "mean_over_windows(per-window PSD) matches the cached "
                "aggregate PSD in spectral_depth<N>.npz to floating-point "
                "precision."
            ),
        }), f, sort_keys=False)
    print(f"Saved RESULT-example-spectrogram.npz + _meta.yaml "
          f"(session={spec['session']}, n_windows={spec['window_center_s'].size}, "
          f"n_freqs={spec['freqs'].size})")

    print("\n=== Gap 2: RESULT-sniff-rate-raw.npz ===")
    raw = build_sniff_rate_raw(RESULTS_DIR, verbose=True)

    raw_npz = {}
    schema = []
    for s in SESSION_DIRNAMES:
        raw_npz[f"{s}__inst_rate_hz"] = raw[s]["inst_rate_hz"]
        raw_npz[f"{s}__isi_trim_valid"] = raw[s]["isi_trim_valid"]
        schema.append({
            "session": s,
            "n_raw_sniffs": int(raw[s]["inst_rate_hz"].size),
            "n_isi_trim_valid": int(raw[s]["isi_trim_valid"].sum()),
            "inst_rate_key": f"{s}__inst_rate_hz",
            "isi_trim_valid_key": f"{s}__isi_trim_valid",
        })
    np.savez(RESULTS_DIR / "RESULT-sniff-rate-raw.npz", **raw_npz)
    with open(RESULTS_DIR / "RESULT-sniff-rate-raw_meta.yaml", "w") as f:
        yaml.safe_dump(yaml_safe({
            "schema": schema,
            "note": (
                "FIG-PAPER1-SNIFFRATE histogram/KDE gap-fill (2026-07-25, not "
                "a deviation). One 1-D float64 array per session at key "
                "'<session>__inst_rate_hz' (raw per-sniff instantaneous rate, "
                "1/ISI, Hz), restricted to control epochs ONLY -- the SAME "
                "restriction already used for RESULT-sniff-rate.yaml's frozen "
                "5-number summary (median/Q1/Q3/IQR/min/max), verified "
                "bit-for-bit (median/Q1/Q3/min/max recomputed from this raw "
                "array match the frozen file exactly). NOT further restricted "
                "by the ISI-percentile trim that gates coherence windows -- "
                "doing so would make this array inconsistent with the frozen "
                "summary it backs. A companion boolean array at "
                "'<session>__isi_trim_valid' (same length, same order) marks "
                "which of these raw sniffs additionally pass the ISI-percentile "
                "trim, for optional overlay/filtering without altering what "
                "the primary frozen summary describes."
            ),
        }), f, sort_keys=False)
    print(f"Saved RESULT-sniff-rate-raw.npz + _meta.yaml ({len(schema)} sessions)")


if __name__ == "__main__":
    main()

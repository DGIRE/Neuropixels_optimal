"""
run_analyses.py  —  example driver chaining the loader with the new analyses.

Mirrors translate_1.py but continues past the loader through the sniff/PSTH/
decomposition stages.  Run end-to-end needs the raw data (LabView .dat,
SpikeGLX .bin, Kilosort dir) reachable — see the loader's File paths.txt flow.
Analyses that consume only D fields present in the golden fixtures can be
unit-tested without the raw data (see tests/test_11_analyses.py).

    python run_analyses.py            # uses the same File paths.txt as translate_1

Pinned parameters (the MATLAB GUIs are replaced by these arguments):
    sniff threshold  = -0.5 std      (compute_sniff_phase)
    ETH threshold    =  0.11         (threshold_eth)
    PSTH bin         =  10 ms        (compute_sniff_psth)
"""
from __future__ import annotations

from load_experiment_data import load_experiment_data
from lib.or_validate_files import or_validate_files
from analyses import (
    compute_sniff_phase, threshold_eth, compute_spike_phase,
    compute_sniff_psth, run_decomposition,
)


def run_all(files: dict) -> dict:
    """Load and run the full sniff/PSTH/decomposition chain, returning D."""
    D = load_experiment_data(files)

    # sensor-derived stages (pinned GUI values)
    D = compute_sniff_phase(D, threshold_std=-0.5)   # -> SNF_PH, sniff_onsets, ...
    D = threshold_eth(D, eth_threshold=0.11)         # -> ETH_thr

    # spike-aligned stages
    D = compute_spike_phase(D)                        # -> spike_SNF_PH, unitMeanSniffPhase
    _psth, _centers, _n, D = compute_sniff_psth(D, bin_ms=10.0)   # -> psth_phase/ms, ...

    # decomposition (ICA by default; sign/permutation-defined, seed-pinned)
    D = run_decomposition(D, methods=("ICA",),
                          eth_choices=("proc",), snf_choices=("proc",),
                          sigma_ms=50.0, seed=0)      # -> D['DECOMP']
    return D


if __name__ == "__main__":
    import sys

    paths_txt = sys.argv[1] if len(sys.argv) > 1 else "File paths.txt"
    files = or_validate_files(paths_txt)
    D = run_all(files)
    print("Done. D now carries:",
          ", ".join(k for k in ("SNF_PH", "ETH_thr", "spike_SNF_PH",
                                "unitMeanSniffPhase", "psth_phase", "DECOMP")
                    if k in D))

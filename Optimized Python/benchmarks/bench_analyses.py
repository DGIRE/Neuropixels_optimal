"""
bench_analyses.py  —  micro-benchmarks for the new-analyses optimizations.

Runs each analysis on the golden spike/onset arrays with the optimization ON
(optconfig default) and OFF (set_baseline), reporting the median wall-clock of a
few repeats.  Point CI_FIXTURES at the golden-fixture tree:

    CI_FIXTURES="C:\\Projects\\Neuropixels\\translation\\Golden Fixtures" \\
        python -m benchmarks.bench_analyses
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _default_gf():
    # <repo>/Golden Fixtures, sibling of "Optimized Python" (repo-relative default)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Golden Fixtures")

import optconfig  # noqa: E402
from analyses import compute_spike_phase, compute_sniff_psth  # noqa: E402
from analyses._common import spike_rate_matrix  # noqa: E402


def _load(gf, rel):
    return np.load(os.path.join(gf, rel)).squeeze()


def _median(fn, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    gf = os.environ.get("CI_FIXTURES") or _default_gf()
    if not gf or not os.path.isdir(gf):
        print("Set CI_FIXTURES to the golden-fixture tree."); return

    D = {
        "SNF": _load(gf, "01_labview/SNF.npy").astype(np.float64),
        "LV_Fs": float(_load(gf, "01_labview/LV_Fs.npy")),
        "spikeTimes": _load(gf, "06_spikes_driftmap/spikeTimes.npy").astype(np.float64),
        "sp": {"clu": _load(gf, "05_spikes_ks/sp_clu.npy")},
        "unitIDs": _load(gf, "07_units/unitIDs.npy"),
        "SNF_PH": _load(gf, "08_sniff_phase/SNF_PH.npy").astype(np.float64),
        "sniff_onsets_s": _load(gf, "08_sniff_phase/sniff_onsets_s.npy").astype(np.float64),
        "sniff_dur_s": float(_load(gf, "08_sniff_phase/sniff_dur_s.npy")),
    }
    nSpk = D["spikeTimes"].size
    nU = D["unitIDs"].size
    print(f"spikes={nSpk:,}  units={nU}  events={D['sniff_onsets_s'].size}\n")

    print("Stage 10 — compute_spike_phase (per-unit mean):")
    tb = _median(lambda: compute_spike_phase(D, optimized=False))
    to = _median(lambda: compute_spike_phase(D, optimized=True))
    print(f"  baseline {tb*1e3:8.1f} ms   optimized {to*1e3:8.1f} ms   speedup {tb/to:5.1f}x\n")

    print("Stage 11 — compute_sniff_psth (event-triggered PSTH):")
    tb = _median(lambda: compute_sniff_psth(D, optimized=False, attach=False))
    to = _median(lambda: compute_sniff_psth(D, optimized=True, attach=False))
    print(f"  baseline {tb*1e3:8.1f} ms   optimized {to*1e3:8.1f} ms   speedup {tb/to:5.1f}x\n")

    print("Spike-rate matrix (decomposition/ICA input):")
    nSamp = D["SNF_PH"].size
    tb = _median(lambda: spike_rate_matrix(D["spikeTimes"], D["sp"]["clu"], D["unitIDs"],
                                           D["LV_Fs"], nSamp, 50.0, vectorized=False))
    to = _median(lambda: spike_rate_matrix(D["spikeTimes"], D["sp"]["clu"], D["unitIDs"],
                                           D["LV_Fs"], nSamp, 50.0, vectorized=True))
    print(f"  baseline {tb*1e3:8.1f} ms   optimized {to*1e3:8.1f} ms   speedup {tb/to:5.1f}x")


if __name__ == "__main__":
    main()

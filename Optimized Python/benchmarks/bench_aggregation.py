"""
bench_aggregation.py  —  Benchmark the per-unit aggregation (optimization E1).

Measures the baseline per-unit Python loop vs the vectorized bincount path on
the real golden spike arrays when CI_FIXTURES is set, otherwise on a synthetic
workload of comparable size.  Reports the median of several runs.

    CI_FIXTURES="C:\\Projects\\Neuropixels\\translation\\Golden Fixtures" \\
        python -m benchmarks.bench_aggregation
"""

from __future__ import annotations

import os
import statistics
import time

import numpy as np


def _default_gf():
    # <repo>/Golden Fixtures, sibling of "Optimized Python" (repo-relative default)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Golden Fixtures")


def _load():
    gf = os.environ.get("CI_FIXTURES") or _default_gf()
    if gf and os.path.isdir(gf):
        clu = np.load(os.path.join(gf, "05_spikes_ks", "sp_clu.npy")).squeeze()
        sd = np.load(os.path.join(gf, "06_spikes_driftmap", "spikeDepths.npy")).squeeze()
        sa = np.load(os.path.join(gf, "06_spikes_driftmap", "spikeAmps.npy")).squeeze()
        rec = float(np.load(os.path.join(gf, "07_units", "recordingDur.npy")).squeeze())
        return clu, sd, sa, rec, "golden fixtures"
    rng = np.random.default_rng(0)
    n = 2_000_000
    clu = rng.integers(0, 379, n)
    sd = rng.uniform(0, 690, n).astype(np.float32)
    sa = rng.uniform(100, 5000, n).astype(np.float64)
    return clu, sd, sa, 1000.0, "synthetic (2M spikes, 379 units)"


def _baseline(clu, sd, sa, rec):
    uniq, cnt = np.unique(clu, return_counts=True)
    nU = len(uniq); ud = np.full(nU, np.nan); ua = np.full(nU, np.nan)
    for u, uid in enumerate(uniq):
        m = clu == uid
        ud[u] = np.mean(sd[m]); ua[u] = np.mean(sa[m])
    return ud, ua, cnt / rec


def _optimized(clu, sd, sa, rec):
    clu_i = clu.astype(np.intp, copy=False); mx = int(clu_i.max())
    cnt_full = np.bincount(clu_i, minlength=mx + 1); present = cnt_full > 0
    cnt = cnt_full[present].astype(np.float64)
    ud = np.bincount(clu_i, weights=sd.astype(np.float64), minlength=mx + 1)[present] / cnt
    ua = np.bincount(clu_i, weights=sa.astype(np.float64), minlength=mx + 1)[present] / cnt
    return ud, ua, cnt_full[present] / rec


def _median_ms(fn, *a, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(*a); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> None:
    clu, sd, sa, rec, src = _load()
    print(f"workload: {src}  (nSpikes={clu.size:,}, nUnits={np.unique(clu).size})")
    tb = _median_ms(_baseline, clu, sd, sa, rec, n=3)
    to = _median_ms(_optimized, clu, sd, sa, rec, n=7)
    print(f"  baseline  (per-unit loop):   {tb:8.1f} ms")
    print(f"  optimized (bincount):        {to:8.1f} ms")
    print(f"  speedup:                     {tb / to:8.1f}x")


if __name__ == "__main__":
    main()

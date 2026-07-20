# Neuropixels Python — Optimization Report

**Baseline:** the Stage-1-corrected port (`C:\Projects\Neuropixels\Python` after the
translation-check fixes to `ks_utils.py` and `load_experiment_data.py`).
**Anchor:** the golden fixtures at `…\translation\Golden Fixtures` (ground truth), not the
frozen Python baseline (blueprint D2). **Scope:** the load/import path.

## Profiling summary

The corrected load path is dominated by I/O and BLAS (`np.fromfile`, `scipy.signal.filtfilt`,
the batched template unwhitening `temps @ winv`, PC-feature indexing) — already optimal. The one
Python-level hot spot with super-linear cost is the **per-unit aggregation loop** in
`load_experiment_data.py`: for each of ~379 units it does a full boolean scan of the ~2.0 M-spike
arrays (`clu == uid`) plus a masked mean — an O(nUnits × nSpikes) pattern. A secondary hot spot is
the noise-exclusion mask in `load_ks_dir` (`~np.isin`, O(nSpikes log nSpikes)).

## Optimizations (both gated in `optconfig.py`; `set_baseline()` = one-switch revert)

| ID | File | Change | Speedup (stage) | Accuracy level | Verified |
|----|------|--------|-----------------|----------------|----------|
| E1 | `load_experiment_data.py` | per-unit loop → single `np.bincount` grouped reduction | ~25–60× on the aggregation stage | Level 2 (numeric) | vs golden fixtures + vs baseline (in-cloud) |
| E2 | `lib/ks_utils.py` | `~np.isin` noise mask → O(n) boolean lookup table | ~11× on the mask | Level 1 (byte-identical) | synthetic byte-identity (in-cloud) |

### E1 — vectorized per-unit aggregation
`np.bincount(clu, weights=…)` computes every unit's summed depth/amplitude in one pass; dividing by
the per-unit count gives the means. A group containing any NaN yields a NaN sum → NaN mean, exactly
matching MATLAB `mean()` and the baseline `np.mean` loop. `unitIDs` and `unitFiringRate` are
**bit-identical** to the baseline. `unitDepths`/`unitAmps` are accumulated in float64 rather than the
baseline's float32 loop; they match the golden fixtures within the single-precision-mean tolerance
(`rtol 1e-5`, `atol 1e-3`) and are in fact **marginally closer to the MATLAB ground truth** than the
float32 baseline (measured: 5.4e-4 vs golden, versus the baseline's 6.1e-4). This is an accept, not a
regression (D2: anchor to ground truth, not the port).

### E2 — boolean-lookup noise mask
A boolean table indexed by cluster id replaces `~np.isin`. The resulting keep-mask is **byte-identical**
(verified on synthetic data, including the case of a noise cluster id larger than any spike's id).

## Validation

`tests/test_10_opt_equivalence.py` — 5 passed:
- E1 optimized == baseline (ids/firing rate exact; means within f32-mean tol) **and** == golden fixtures.
- E1 NaN-group propagation matches the baseline.
- E2 mask byte-identical to `~np.isin` (two cases).
- `set_baseline()` / `set_optimized()` toggle.

`benchmarks/bench_aggregation.py` on the golden spike arrays (2,036,596 spikes / 379 units):
baseline ≈ 0.74 s → optimized ≈ 0.03 s.

## Not fixture-runnable in-cloud (certify locally)

E1/E2 sit downstream of Kilosort inputs and the raw `.bin` that cannot cross the desktop↔cloud bridge,
so the full end-to-end pipeline speedup and the full-array equality were not measured here. Close the
gap on the workstation:

```
cd "C:\Projects\Neuropixels\Optimized Python"
set CI_FIXTURES=C:\Projects\Neuropixels\translation\Golden Fixtures
python -m pytest tests/ -q
python -m benchmarks.bench_aggregation
# end-to-end: run translate_1.py (optimized) and again after optconfig.set_baseline(); diff outputs.
```

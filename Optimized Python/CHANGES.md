# CHANGES — Optimized Python vs the corrected port

Only two source files carry optimization logic; every other file is byte-identical to the
Stage-1-corrected port. All changes are gated in `optconfig.py`.

## New files
- `optconfig.py` — optimization switches; `set_baseline()` (one-switch revert) / `set_optimized()`.
- `tests/test_10_opt_equivalence.py` — optimized-vs-baseline and optimized-vs-golden equivalence.
- `benchmarks/bench_aggregation.py` — E1 benchmark.
- `OPTIMIZATION_REPORT.md`, `CHANGES.md`, `README_OPTIMIZED.md`.

## Modified files
- `load_experiment_data.py`
  - Added `from optconfig import OPT`.
  - **E1**: the per-unit aggregation now branches on `OPT.vectorized_unit_aggregation`. Optimized path
    computes `unitDepths`/`unitAmps` via `np.bincount` grouped sums / counts (O(nSpikes)); the original
    per-unit `np.mean` loop is preserved verbatim as the `else` branch. `unitIDs`/`unitFiringRate`
    unchanged.
- `lib/ks_utils.py`
  - Added a robust `from optconfig import OPT` import.
  - **E2**: the noise-exclusion mask branches on `OPT.fast_noise_exclusion`. Optimized path builds a
    boolean lookup table indexed by cluster id; the original `~np.isin` is the `else` branch.

## Unchanged (byte-identical to the corrected port)
`plot_unit_locations.py`, `translate_1.py`, `lib/or_loaddat.py`, `lib/or_read_bin.py`,
`lib/or_read_meta.py`, `lib/or_validate_files.py`, `lib/__init__.py`, `requirements.txt`, and the
faithful `ks_driftmap` / meta-helper / LFP logic in the two modified files.

## Reverting
`import optconfig; optconfig.set_baseline()` restores the exact original behavior (the per-unit loop and
`np.isin`). No other change is needed.

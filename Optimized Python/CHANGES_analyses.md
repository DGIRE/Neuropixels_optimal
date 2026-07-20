# CHANGES — new-analyses extension

Additive extension of the `Optimized Python` package. No existing loader/lib
code was modified except `optconfig.py` (three new flags appended).

## Added
- `analyses/` subpackage — ports of the five MATLAB `new analyses`:
  `compute_sniff_phase` (08), `threshold_eth` (09), `compute_spike_phase` (10),
  `compute_sniff_psth` (11, incl. faithful `psthAndBA`/`timestampsToBinned`/
  `histdiff` in `psth_utils.py`), `run_decomposition`, `run_ica`. Shared
  primitives in `_common.py` (`matlab_round`, `zero_lag_xcorr`, Gaussian
  spike-rate matrix, abs-descending rank).
- `run_analyses.py` — example driver chaining loader → sniff/PSTH/decomposition.
- `tests/test_11_analyses.py` — fixture-anchored + optimized-vs-baseline +
  stochastic-invariant tests (13 passed in-cloud).
- `benchmarks/bench_analyses.py` — per-optimization micro-benchmarks.
- `ANALYSES_REPORT.md` — full port + optimization report.

## Changed
- `optconfig.py` — added `vectorized_spike_phase`, `vectorized_psth`,
  `vectorized_rate_matrix` (default True); `set_baseline()`/`set_optimized()`
  now toggle them too. Existing loader flags unchanged.

## Conventions honored
0-based indexing; MATLAB `round` half-away (incl. derived constants like
`MAX_DUR = round(62.5) = 63`); z-score ddof=1; FIR `designfilt` → `firwin`
hamming/scaled; `filtfilt` padlen `3·(ntaps−1)`; `histdiff` ordhist strict-open
edges with the difference formed **before** the edge comparison; xcorr 'coeff';
GUI thresholds passed as pinned parameters; stochastic ICA seed-pinned +
pre-drawn inits, compared up to sign/permutation.

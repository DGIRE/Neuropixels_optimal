# Neuropixels Python — New-Analyses Port & Optimization Report

**Scope:** the five MATLAB `new analyses` not previously ported, added to the
`Optimized Python` package as an `analyses/` subpackage.
**Anchor:** the golden fixtures at `…\translation\Golden Fixtures` (ground truth,
blueprint D2) — stages 08–11. The decomposition/ICA stages are stochastic and
have **no** fixture (Level-3); they are validated by invariants.
**Method:** translate-then-verify against the fixtures using a deterministic
differential harness; optimize only after the faithful baseline certifies
(blueprint R20), each optimization reversible + gated (R7) and validated
optimized-vs-fixture then optimized-vs-baseline (R19).

## What was ported

| Stage | MATLAB source | Python module | Fixture | Result |
|------|----------------|---------------|---------|--------|
| 08 | `compute_sniff_phase.m` | `analyses/compute_sniff_phase.py` | 08 | ✅ all 7 outputs within tol |
| 09 | `threshold_eth.m` | `analyses/threshold_eth.py` | 09 | ✅ both outputs within tol |
| 10 | `compute_spike_phase.m` | `analyses/compute_spike_phase.py` | 10 | ✅ both outputs within tol |
| 11 | `compute_sniff_psth.m` (+ `psthAndBA`/`timestampsToBinned`/`histdiff`) | `analyses/compute_sniff_psth.py` + `analyses/psth_utils.py` | 11 | ✅ all 5 outputs within tol |
| — | `run_decomposition.m` | `analyses/run_decomposition.py` | none (stochastic) | ⚠ invariant-checked |
| — | `run_ICA.m` | `analyses/run_ica.py` | none (stochastic) | ⚠ invariant-checked |

Validation is `tests/test_11_analyses.py` — **13 passed** in-cloud against the
staged golden fixtures. Each fixture-anchored stage is checked with **golden
inputs** (not the port's own upstream output) so one stage's bug cannot mask the
next; a chained 08→10→11 test confirms end-to-end reconstruction from raw SNF.

## Per-stage fidelity (measured, in-cloud, at manifest tolerances)

- **08 sniff phase.** `SNF_filt` max|Δ| 9.5e-7 (fir tol 1e-6), `SNF_z` 3.5e-12,
  **onsets bit-exact** (3852/3852), `SNF_PH` exact after the round fix below,
  `sniff_dur_s`/`sniff_thr` exact.
- **09 ETH threshold.** Floor-clip exact.
- **10 spike phase.** `spike_SNF_PH` exact; `unitMeanSniffPhase` within 1e-12
  (NaN where < 5 valid spikes), optimized == baseline.
- **11 sniff PSTH.** `psth_phase`/`psth_ms` within 1e-9, centers/`n_events`
  exact, optimized == baseline.

## Two fidelity bugs found and fixed during the port

1. **MATLAB `round` is half-away-from-zero on a derived constant.**
   `MAX_DUR = round(0.5*LV_Fs) = round(62.5)`. MATLAB → **63**; Python's built-in
   `round` (banker's) → 62. That denominator sets every sniff cycle's phase ramp,
   so `SNF_PH` was off on 2141 samples until `MAX_DUR`/`MIN_ISI` used
   `matlab_round`. (Generalizes blueprint "round half-away" hazard to *derived
   scalars*, not just per-sample index math.)

2. **Form the difference before comparing to the bin edge.** `histdiff.c`
   computes `diff = spike − event` and tests `diff < max`. Testing the algebraically
   equal `spike < event + max` on absolute times rounds differently when event
   times are large (~500 s) and silently dropped 5 spikes whose true diff was
   exactly the window end — a 0.13 spikes/s error at the last PSTH bin, far above
   the 1e-9 tolerance. Both PSTH paths now window with `searchsorted` (padded one
   bin) then apply the exact strict test on the **computed difference**.

## Optimizations (all gated in `optconfig.py`; `set_baseline()` = one-switch revert)

| ID | Flag | File | Change | Speedup* | Accuracy |
|----|------|------|--------|----------|----------|
| A1 | `vectorized_spike_phase` | `compute_spike_phase.py` | per-unit mean loop → `np.bincount` grouped reduction over valid phases | ~3.6× | Level 2; == golden (1e-12) & baseline |
| A2 | `vectorized_psth` | `compute_sniff_psth.py` | per-unit `psthAndBA` loop → one global spike→event expansion | **~32×** | Level 2; == golden (1e-9) & baseline |
| A3 | `vectorized_rate_matrix` | `_common.py` (used by decomposition/ICA) | per-unit FFT loop → one batched real-FFT | ~1.2× | Level 2; == baseline to FFT round-off |

\* Sandbox, 2 cores, 2,036,596 spikes / 379 units / 3852 events. A2 baseline
16.5 s → 0.5 s. A3 is FFT- and memory-bandwidth-bound (the batched FFT allocates
a large transform); the win is modest here and may differ on the workstation
(more cores/RAM). All three preserve a faithful baseline fallback.

## Stochastic stages (run_decomposition / run_ICA) — no golden fixture

FastICA uses `randn`, and the original scripts never fix MATLAB's seed, so no
cross-tool bit match exists or is expected (Level-3). The ports pin a NumPy seed
and **pre-draw the init vectors in component order** for reproducibility, replace
the checkbox GUI with pinned arguments, and are checked by invariants:
deterministic for a fixed seed; `S = W·Xc` reconstruction to 1e-6; PCA scores
mutually orthogonal; ranks a valid permutation; correlations finite. SVD/PCA/ICA
are defined up to sign and permutation — compare magnitude/rank, never raw
components.

## Not runnable end-to-end in-cloud (certify locally)

The fixtures transfer through the bridge and are the in-cloud anchor; the raw
`.dat`/`.bin`/Kilosort dir do not. To certify at full scale on the workstation:

```
cd "C:\Projects\Neuropixels\Optimized Python"
set CI_FIXTURES=C:\Projects\Neuropixels\translation\Golden Fixtures
python -m pytest tests/test_11_analyses.py -q      # expect 13 passed
python -m benchmarks.bench_analyses                # local speedups
# end-to-end: D = load_experiment_data(files); D = compute_sniff_phase(D); ...
#   then run again after optconfig.set_baseline() and diff.
```

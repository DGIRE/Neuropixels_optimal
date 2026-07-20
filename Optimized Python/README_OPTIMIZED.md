# Neuropixels — Optimized Python

A drop-in optimized version of the Neuropixels load/import pipeline. It is the Stage-1-corrected
port with two gated, fixture-validated optimizations. Same public API and outputs as the port,
within the golden-fixture tolerances.

## What's optimized
- **E1** — per-unit aggregation is a single vectorized grouped reduction instead of a 379-iteration
  Python loop over ~2 M spikes (~25–60× on that stage). Numeric (Level 2); matches the golden fixtures.
- **E2** — the Kilosort noise-exclusion mask uses an O(n) boolean lookup instead of `~np.isin`
  (byte-identical, Level 1).

Both are switches in `optconfig.py`. `optconfig.set_baseline()` reverts to the original behavior.

## Install
```
pip install -r requirements.txt      # numpy, scipy, matplotlib, h5py
```

## Run the pipeline
```
python translate_1.py                # edit the four path constants at the top first
```

## Validate against the golden fixtures
```
set CI_FIXTURES=C:\Projects\Neuropixels\translation\Golden Fixtures
python -m pytest tests/ -q            # 5 passed
python -m benchmarks.bench_aggregation
```

## Turn optimizations on/off
```python
import optconfig
optconfig.set_baseline()             # original per-unit loop + np.isin
optconfig.set_optimized()            # default: fast paths on
# or per-switch:
optconfig.OPT.vectorized_unit_aggregation = False
optconfig.OPT.fast_noise_exclusion = False
```

## Accuracy contract
The anchor is the golden fixtures + the MATLAB reference, not the Python baseline. `unitIDs`,
`unitFiringRate`, and the E2 mask are bit-identical to the baseline; `unitDepths`/`unitAmps` are
float64-accumulated grouped means that match the fixtures within the single-precision-mean tolerance
(and are marginally closer to MATLAB than the baseline float32 loop). See `OPTIMIZATION_REPORT.md`.

## Note
Layout mirrors the port (flat modules + `lib/`) so unchanged files stay byte-identical. The full
end-to-end speedup and full-array equality require the raw data on the workstation (see the report).

# Neuropixels — Optimized Pipeline + Golden Fixtures

This repo holds the optimized Python port of the Neuropixels analysis pipeline,
alongside the golden-fixture data tree used to validate it against the
original MATLAB implementation.

## Layout

```
Neuropixels/
├── Optimized Python/     # the package
│   ├── analyses/         # ported analysis stages (sniff phase/PSTH, spike phase,
│   │                      # decomposition/ICA, threshold, etc.)
│   ├── lib/               # shared I/O / Kilosort utilities (raw-data readers,
│   │                      # bin/meta parsing, file validation)
│   ├── tests/             # fixture-anchored equivalence + optimization gate tests
│   ├── benchmarks/        # micro-benchmarks comparing optimized vs baseline
│   ├── translate_1.py     # end-to-end MATLAB → Python translation entry point
│   ├── run_analyses.py    # end-to-end analyses entry point
│   └── optconfig.py       # optimization on/off switches
└── Golden Fixtures/       # deterministic MATLAB-derived fixtures, stages 00–11
                           # (NOT tracked by Git — see below)
```

`Golden Fixtures/` contains one numbered subfolder per pipeline stage
(`00_manifest`, `01_labview`, ... `11_sniff_psth`), each holding the `.npy`/
data arrays that stage produces. These are the deterministic, MATLAB-derived
ground truth used by the equivalence tests and benchmarks.

## Running the tests

From `Optimized Python/`:

```
python -m pytest tests/ -q
```

The test and benchmark code auto-finds the fixture tree at `../Golden Fixtures`
relative to its own file location — no environment variable is required when
the fixtures sit alongside `Optimized Python/` as they do in this checkout.

To point at a different fixture tree (e.g. on another machine, or a copy
elsewhere on disk), set the `CI_FIXTURES` environment variable; it overrides
the repo-relative default:

```
CI_FIXTURES="/path/to/Golden Fixtures" python -m pytest tests/ -q
```

Note: this project reuses the name `CI_FIXTURES` as a plain path override, not
as a CI-system-provided variable — it's just how the fixture directory is
communicated to the tests/benchmarks locally.

## What you can and can't run standalone

Only the fixture-anchored tests (`tests/`) and benchmarks (`benchmarks/`) run
standalone from this repo, since they only need `Golden Fixtures/`.

End-to-end runs (`translate_1.py`, `run_analyses.py`) additionally need the
raw acquisition data (SpikeGLX / Kilosort / LabView outputs for a real
recording session), which is **not** included in this repo. Those scripts
have hardcoded local paths (`PATHS_FILE`, `OUTPUT_DIR`, etc.) that must be
edited to point at that raw data before they can run.

## Git / porting notes

`Golden Fixtures/` is excluded via `.gitignore` and is not tracked by Git.
The fixtures are present in this checkout so the pipeline runs locally, but
when porting to a new machine you'll need to move that folder over
out-of-band (zip, external drive, rsync, etc.) rather than via `git clone`.

All of the Neuropixels golden-fixture files are individually under 30 MB, so
they could be committed to a normal Git repo if desired (they aren't up
against GitHub's hard file-size limit). They're excluded here by default for
consistency with how large binary fixture trees are handled elsewhere — see
Repos/Guides for the rationale.

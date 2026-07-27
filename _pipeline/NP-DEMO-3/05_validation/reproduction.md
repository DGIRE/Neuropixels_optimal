# NP-DEMO-3 Reproduction Record

**Date:** 2026-07-23
**Reproducer:** reproducer agent (clean-room, no memory of prior run)
**Verdict:** PASS — all gates re-pass from committed instructions

---

## Environment

- **Git commit:** 236d8dbdda98e079e44398e21f2bc3b6ec01b3ce
- **Python:** 3.13.9 (vras venv — C:/Projects/Repos/Agentic Research/Research Setup/research_workflow/vras/Scripts/python.exe)
- **numpy:** 2.0.2
- **scipy:** 1.14.1
- **statsmodels:** 0.14.6
- **pyyaml:** 6.0.3
- **OS:** Windows 11 Pro Build 26200

---

## Inputs

- 6 session data directories under `C:\Projects\Repos\Neuropixels\DATA`: 11-01-2021, 11-03-2021, 12-15-2021, 5-17-2022, 06-24-2022, 09-14-2022
- Golden Fixtures: `C:\Projects\Repos\Neuropixels\Golden Fixtures\` (read-only, pre-validated MATLAB reference)
- Analysis code: `_pipeline/NP-DEMO-3/03_software/` (committed)

---

## Execution

```
python _pipeline/NP-DEMO-3/03_software/run_analysis_np_demo3.py
python _pipeline/NP-DEMO-3/03_software/run_numeric_gate.py
python _pipeline/NP-DEMO-3/03_software/run_stat_gate.py
```

All three ran without error.

---

## Gate Results

| Gate | Checks | Result |
|------|--------|--------|
| Numeric (run_numeric_gate.py) | 74/74 | **PASS** |
| Statistical (run_stat_gate.py) | 10/10 | **PASS** |

---

## Key Result Consistency

| Metric | Reproduced | Frozen | Match |
|--------|-----------|--------|-------|
| n_units | 1601 | 1601 | YES |
| n_experiments | 6 | 6 | YES |
| LME p_value | 1.434e-60 | 1.434e-60 | YES |
| LME std_effect | 0.5803 | 0.5803 | YES |
| Wilcoxon p | 0.3125 | 0.3125 | YES |
| Wilcoxon rank_biserial_r | 0.5238 | 0.5238 | YES |
| n_discard_intervals | 1742 | 1742 | YES |
| n_included_units (mrl-per-unit rows) | 1603 | 1603 | YES |

---

## Notes

- The contract hash mismatch reported by the hook in the reproducer's subshell is a known false alarm (documented in scientific_audit.md §BLOCKER-1/DISMISSED). `research_workflow verify NP-DEMO-3` passes cleanly from the project root.
- No out-of-band data or manual steps were required; all six sessions were located automatically by the runner.
- The 2022-09-14 multi-shank session was handled by `_fix_ks_dir()` as documented in the code.
- Seeds: analysis is deterministic (no random number generation in the scientific code path).

---

## Conclusion

The NP-DEMO-3 analysis is fully reproducible from committed code and the fixed data directories. Results are bit-identical to frozen outputs within gate tolerances.

# NP-DEMO-4 Reproduction Record

**Reproduction Date**: 2026-07-23
**Python Interpreter**: C:/Projects/Repos/Agentic Research/Research Setup/research_workflow/vras/Scripts/python.exe
**Python Version**: 3.13.9
**Git Commit Hash**: 236d8dbdda98e079e44398e21f2bc3b6ec01b3ce
**OS Version**: Windows 11 Pro (10.0.26200)

## Reproduction Steps

### Step 1: Main Analysis

Command:
```
"C:/Projects/Repos/Agentic Research/Research Setup/research_workflow/vras/Scripts/python.exe" "C:/Projects/Repos/Neuropixels/_pipeline/NP-DEMO-4/03_software/run_analysis_np_demo4.py"
```

Status: **PASS**

Sessions processed:
- 2021-11-01: n_labview_trials=59, 192 included units
- 2021-11-03: n_labview_trials=62, 313 included units
- 2021-12-15: n_labview_trials=60, 183 included units
- 2022-05-17: n_labview_trials=60, 99 included units
- 2022-06-24: n_labview_trials=60, 175 included units
- 2022-09-14: n_labview_trials=23, 150 included units

Total LabView trials: 324

### Step 2: Numeric Gate

Command:
```
"C:/Projects/Repos/Agentic Research/Research Setup/research_workflow/vras/Scripts/python.exe" "C:/Projects/Repos/Neuropixels/_pipeline/NP-DEMO-4/03_software/run_numeric_gate.py"
```

Status: **PASS** (7/7 checks)

- RESULT-eth-mask: PASS (6 sessions)
- RESULT-unit-inclusion: PASS (6 sessions)
- RESULT-sniff-rate-matrix: PASS (324 trials)
- RESULT-fr-per-trial: PASS (324 trials)
- RESULT-eth-per-trial: PASS (324 trials)
- RESULT-methods-table: PASS (6 sessions)
- RESULT-sniff-stat: PASS (W=7.0, p=0.5625, r=0.3333, n=6)

### Step 3: Stat Gate

Command:
```
"C:/Projects/Repos/Agentic Research/Research Setup/research_workflow/vras/Scripts/python.exe" "C:/Projects/Repos/Neuropixels/_pipeline/NP-DEMO-4/03_software/run_stat_gate.py"
```

Status: **PASS** (7/7 checks)

- n_animals == 6: PASS
- Significance decision (alpha=0.05, not significant): PASS
- W == 7.0 (exact match): PASS
- |p - 0.5625| < 1e-6: PASS (diff = 0.0)
- |r - 0.3333| < 1e-3: PASS (diff = 3.33e-05)
- W re-derived from per-animal means: PASS
- p re-derived from per-animal means: PASS

## Numeric Reproduction

| Metric | Reference | Reproduced | Tolerance | Match |
|---|---|---|---|---|
| W | 7.0 | 7.0 | exact | YES |
| p_exact | 0.5625 | 0.5625 | 1e-6 | YES |
| rank_biserial_r | 0.3333 | 0.3333333... | 1e-3 | YES |
| n_animals | 6 | 6 | exact | YES |
| significant | false | false | exact | YES |
| Total LabView trials | 324 | 324 | exact | YES |

## Gate Status

- **numeric_gate.yaml**: PASS
- **stat_gate.yaml**: PASS

## Deviations from Reference

None. All key numeric values reproduced exactly or within specified tolerances.

## Overall Verdict

**REPRODUCED**

All analysis gates passed. Statistical finding confirmed: no significant difference between contact and control sniff rates (W=7.0, p=0.5625, two-sided exact Wilcoxon, n=6 animals, not significant at alpha=0.05).

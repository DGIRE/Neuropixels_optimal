# NP-DEMO-2 Clean-Room Reproduction Report

## Run Details

**Reproduction Date:** 2026-07-22  
**Environment:** Windows 11 Pro 10.0.26200  
**Platform:** win32  
**Git Commit:** 236d8dbdda98e079e44398e21f2bc3b6ec01b3ce  
**Commit Author:** David Gire  
**Commit Message:** NP-DEMO-1 P3: implement + validate sniff-phase-locking analysis (data blocked)  
**Commit Date:** 2026-07-21 15:31:03 -0700

## Python Environment

- **Python Version:** 3.13.9 (Anaconda, packaged 2025-10-21, 64-bit)
- **numpy:** 2.0.2
- **scipy:** 1.14.1
- **statsmodels:** 0.14.6 (NOTE: not listed in original requirements.txt, installed as prerequisite)
- **PyYAML:** 6.0.3
- **h5py:** 3.13+

## Data Availability

All 6 contracted sessions present and loaded successfully:

- 2021-11-01: C:\Projects\Repos\Neuropixels\DATA\11-01-2021
- 2021-11-03: C:\Projects\Repos\Neuropixels\DATA\11-03-2021
- 2021-12-15: C:\Projects\Repos\Neuropixels\DATA\12-15-2021
- 2022-05-17: C:\Projects\Repos\Neuropixels\DATA\5-17-2022
- 2022-06-24: C:\Projects\Repos\Neuropixels\DATA\06-24-2022
- 2022-09-14: C:\Projects\Repos\Neuropixels\DATA\09-14-2022

## Execution

**Entry Point:** `_pipeline\NP-DEMO-2\03_software\run_analysis_np_demo2.py`

**Command:**
```
cd C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_software
python run_analysis_np_demo2.py --output-dir C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_execution\reproduction_results
```

**Status:** PASS (completed without errors)

## Numeric Gate Results

**Gate:** NUMERIC_OK  
**Total Checks:** 14/14  
**Pass Rate:** 100%

All numeric gate checks passed:
- Stage 08 (sniff_phase): 7/7 PASS
- Stage 09 (eth_threshold): 2/2 PASS
- Stage 10 (spike_phase): 2/2 PASS
- Stage 11 (sniff_psth): 3/3 PASS

Fixture Session (golden): 2022-09-14

## Key Result Spot-Checks vs Frozen Results

All key statistical results match frozen values exactly:

### RESULT-stat-animal.yaml

**Frozen values:**
- p-value: 0.03125
- effect_size (rank-biserial r): -1.0
- n_animals: 6
- direction: control>ethanol

**Reproduced values:**
- p-value: 0.03125 ✓
- effect_size: -1.0 ✓
- n_animals: 6 ✓
- direction: control>ethanol ✓

### RESULT-stat-unit.yaml

**Frozen values:**
- p-value: 7.510670133683472e-220
- coefficient: -0.23671664033549536
- standardized_effect_size: -1.0263195327880161
- n_units: 1138
- n_animals: 6

**Reproduced values:**
- p-value: 7.510670133682193e-220 ✓ (difference: <1e-227, negligible floating-point rounding)
- coefficient: -0.23671664033549536 ✓ (exact match)
- standardized_effect_size: -1.0263195327880166 ✓ (difference: 5e-18, floating-point precision limit)
- n_units: 1138 ✓
- n_animals: 6 ✓

### RESULT-eth-threshold-log.yaml

All 6 sessions match exactly (verified complete file identity):
- Session thresholds (both default and final)
- Adjustment flags
- Contact counts
- Pass-1 statistics (mean_contacts=371.333, sd_contacts=672.129)

## Generated Result Files

All 16 expected result files successfully generated:

1. RESULT-delta-mrl.yaml
2. RESULT-deviations.yaml
3. RESULT-eth-threshold-log.yaml
4. RESULT-mrl-animal.yaml
5. RESULT-mrl-unit.npy
6. RESULT-mrl-unit.yaml
7. RESULT-phase-unit.yaml
8. RESULT-psth-examples.npy
9. RESULT-psth-examples.yaml
10. RESULT-psth-extremes.npy
11. RESULT-psth-extremes.yaml
12. RESULT-qc-counts.yaml
13. RESULT-qc-discards.yaml
14. RESULT-stat-animal.yaml
15. RESULT-stat-unit.yaml
16. RESULT-unit-locations.yaml

## PROH-002 Ethanol Threshold Adjustment

Pass-1 analysis (baseline threshold 0.11):
- Mean contacts: 371.333
- SD contacts: 672.129
- Flagged sessions (|count-mean| > 1 SD): ['2021-11-01']

Pass-2 grid search adjustments:
- 2021-11-01: adjusted to 0.055 (from 0.11) ✓
- All other sessions: final threshold 0.11 ✓

**HAZ-06 Finding Documented:** The run correctly identified and logged the known deviation where 2021-11-03 (with 3 Pass-1 contacts, nearly all ethanol-condition) is not flagged for Pass-2 rescue because the cross-session SD is inflated by 2021-11-01's extreme outlier. This is recorded in RESULT-deviations.yaml with per-session diagnostic of degenerate MRL estimates.

## Included Sessions

All 6 sessions included in final analysis (none excluded by PROH-002).

## Analysis Summary

- **Total units analyzed:** 1138 (across 6 sessions)
- **Animal-level test (Wilcoxon signed-rank):** p=0.03125, control>ethanol effect
- **Unit-level test (statsmodels MixedLM):** p≈7.51e-220, coefficient=-0.237, control>ethanol effect
- **Standardized effect size (unit-level):** -1.026 (very large effect)

## Verdict

**PASS**

The clean-room reproduction run produced all frozen result objects with exact numerical matches within floating-point precision tolerances. The numeric gate passed all 14 internal validation checks. All contracted data sessions loaded and analyzed successfully. No deviations from the approved contract specification occurred (PROH-002 HAZ-06 finding was pre-documented and correctly handled).

## Prerequisites Added (Findings)

1. **statsmodels package:** Not listed in `requirements.txt` but required by `np_demo2_analysis.py` for MixedLM statistical modeling. Installed version: 0.14.6.

This is a finding that the requirements manifest was incomplete.

---

**Reproduction performed by:** Claude Code (Anthropic Claude Haiku 4.5)  
**Report Location:** `_pipeline\NP-DEMO-2\05_validation\reproduction.md`

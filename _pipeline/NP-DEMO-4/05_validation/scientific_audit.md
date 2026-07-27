# NP-DEMO-4 Independent Scientific Audit

**Auditor:** scientific-auditor (fresh-context, adversarial)
**Date:** 2026-07-23
**Verdict: PASS — 0 BLOCKER, 0 MAJOR, 4 MINOR**

---

## Executive Summary

The implementation is consistent with contract_v001.yaml on every hard constraint. PROH-001 (SNF, never LFP) holds; the onset-detection recipe is numerically equivalent to the validated reference; detect_eth_contact is byte-identical to the NP-DEMO-3 original; and the recorded stat (W=8, p=0.6875, r=0.238, n=6, ns) was independently reproduced by hand. Findings are confined to 4 MINOR items.

---

## Per-Axis Findings

### Axis 1 — PROH-001 SNF vs LFP: PASS
`compute_sniff_rate.py` reads D['SNF'] only. D['LFP'] appears in no variable in any of the four implementation files. Complies with prohibited_changes: "Using LFP instead of SNF".

### Axis 2 — Onset detection fidelity: PASS
Line-by-line comparison vs compute_sniff_phase.py (validated reference):
- `firwin(21, fco, fs=LV_Fs, window="hamming", pass_zero="lowpass", scale=True)` with `fco=min(40.0, LV_Fs/2.0*0.9)` — IDENTICAL
- `filtfilt(padtype="odd", padlen=3*(taps.size-1))` — IDENTICAL
- z-score `std(ddof=1)` — IDENTICAL
- Downward crossing `diff([0;below])==1` at -0.5 — IDENTICAL
- `MIN_ISI=matlab_round(0.05*LV_Fs)` — IDENTICAL
- Rejection `keep=[True]+diff(onsets)>=MIN_ISI` on raw onsets — IDENTICAL

Two non-numeric differences, neither changing detected onsets:
(a) Local `_matlab_round` rather than importing from `_common.matlab_round` — both compute `sign(x)*floor(|x|+0.5)`, bit-identical.
(b) Added `std_==0` NaN guard that cannot fire on real sniff data.

### Axis 3 — detect_eth_contact byte-identity: PASS
`fc /b` comparison against NP-DEMO-3 original reported zero differing bytes. Threshold 0.05, whole-trace mean-subtraction, strict >, contiguous-run count unchanged. Complies with PROH-001 / verbatim reuse requirement.

### Axis 4 — Trial segmentation: PASS
- Odd-TR only: `TR[TR % 2 == 1]` — CORRECT
- First TS corrected to 0: `ts_trial[0]=0.0` — CORRECT
- 0-40 s enforced: `(ts_s>=0.0)&(ts_s<=40.0)` — CORRECT
- Even TR never enters trials — CORRECT
Complies with DATA-001 and the even-TR prohibition.

### Axis 5 — Unit inclusion: PASS (with MINOR-1)
Both criteria simultaneous: `if fr >= 0.1 and n_spikes >= 5000`. Experiment-wide: `unitFiringRate` from D-dict (loader-computed) and `n_spikes = np.sum(clu_all == uid)` counts the whole session, not per-trial. Complies with CON-001.
MINOR-1: `session_duration_s` is passed but never used — dead parameter, no scientific effect.

### Axis 6 — CON-003 control period: PASS
Control is TS 0-10 s only (`(ts_s>=0.0)&(ts_s<=10.0)`) AND excludes contact overlap (`& (~eth_contact_mask[idx])`). First discrete contact identified via `rising[0]`/`falling[0]` in `_first_eth_contact_in_trial`. Per-session mean = mean of per-trial means. Complies with CON-003.

### Axis 7 — Wilcoxon test: PASS
`scipy.stats.wilcoxon(diffs, alternative="two-sided", method="exact", zero_method="wilcox")`. The contract says `mode='exact'`; modern scipy renamed this to `method='exact'` — semantically identical, not a deviation. Applied to 6 per-session means, not per-trial. Rank-biserial `(W+ - W-)/(W+ + W-)` = contract formula when W=W-.

### Axis 8 — Figure data uses raw ETH: PASS
`raw_ETH = D["ETH"]` (BEFORE mean-subtraction); `eth_per_trial.append(raw_ETH[idx]...)`. Uses D['ETH'], not D['ETH_ms']. Complies with CON-002 / OUT-005.

### Axis 9 — 50 ms firing rate: PASS
`half_win_s=0.025`, window [t-25ms, t+25ms]; fixed `denom=0.05` with no edge adaptation. Closed interval implemented correctly: `searchsorted(st, hi, 'right') - searchsorted(st, lo, 'left')`. Complies with CON-001 and the 50 ms prohibition.

### Axis 10 — Methods table: PASS
All five per-session fields present. RESULT-methods-table n_trials_eth matches RESULT-eth-mask for all 6 sessions (2501/491/1180/1048/557/187) and n_units_* match RESULT-unit-inclusion. Complies with OUT-001.

### Axis 11 — Statistical sanity: PASS
Hand-derivation from recorded per-animal means:
- diffs = [+0.20225, -0.06130, +0.10037, -0.25533, +0.00192, +0.06507]
- signs [+,-,+,-,+,+]; |ranks| [5,2,4,6,1,3]
- W+ = 13, W- = 8, total = 21
- scipy statistic = min = **8** (matches W=8.0)
- rank-biserial = 5/21 = **0.238095** (matches)
- Exact two-sided p for W_min=8 at n=6 = 44/64 = **0.6875** (matches)
- significant=false CORRECT
Internally consistent, plausible for a 4-up/2-down pattern.

### Axis 12 — _fix_ks_dir for 2022-09-14: PASS
`_fix_ks_dir` promotes ksDir to its parent when channel_map.npy/channel_positions.npy are one level up. Applied BEFORE the loader: `or_validate_files` → `_fix_ks_dir` → `load_experiment_data`. Session 2022-09-14 loaded successfully (379 recorded, 150 included), consistent with the fix working.

---

## MINOR Findings

**MINOR-1 (dead parameter):** `session_duration_s` passed to `_apply_unit_inclusion` but unused. No scientific effect.

**MINOR-2 (contiguity assumption):** `avg_sniffs` counts onsets via `(sniff_onsets >= idx[0]) & (sniff_onsets <= idx[-1])`, assuming each trial's `global_idx` is contiguous. Holds when TS is monotone within a trial (normal case). Non-contiguous trial could over-count. Not observed in data — flag as unenforced assumption.

**MINOR-3 (latent n-drop):** `valid_pairs_mask` drops NaN CON-003 pairs silently rather than blocking when n < 6. All 6 sessions passed (n_animals=6 recorded), so no violation occurred. Recommend asserting n=6 in the report.

**MINOR-4 (unverified exact match to NP-DEMO-3):** Level-1 acceptance for RESULT-eth-mask requires exact match to NP-DEMO-3 validated n_trials_eth per session. detect_eth_contact is byte-identical (necessary condition) but exact numeric equality against NP-DEMO-3 prior outputs was not independently reproduced in this audit (execution-blocked). Recommend validator confirm by running NP-DEMO-3 runner and comparing n_trials_eth values.

---

## Summary Tally

| Level | Count |
|-------|-------|
| PASS | 12 |
| MINOR | 4 |
| MAJOR | 0 |
| BLOCKER | 0 |

**Overall: PASS.** No contract clause is violated on any evaluable axis. The four MINOR items are documentation/robustness notes plus one sandbox-limited verification gap; none alter W=8, p=0.6875, r=0.238, n=6 or the "not significant" conclusion.

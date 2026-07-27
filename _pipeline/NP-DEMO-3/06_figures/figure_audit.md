# NP-DEMO-3 Figure Audit

**Auditor:** figure-auditor (fresh context, adversarial)
**Date:** 2026-07-23
**Task:** NP-DEMO-3 — ethanol sniff phase-locking
**Renderer:** render_figures.py (03_software/)
**Overall verdict:** PASS (after 3 MAJOR fixes applied to renderer before step)

---

## FIG-DEMO3-QC-SNIFF — PASS

- **Source match**: All 6 experiment panels' `n_sniffs`, `n_neurons`, `n_trials`, `duration_min`, `pct_usable_sniff_time` exactly match `RESULT-sniff-qc.yaml`; `eth_contact_threshold=0.05` matches `RESULT-eth-mask.yaml`. 0 mismatches.
- **Numeric fidelity**: All caption stats read from figdata YAML (never typed).
- **Axis**: SNF_z and ETH_ms panels both autoscale to include zero across all sessions.
- **Thresholds**: red dashed lines at -0.5 (SNF) and 0.05 (ETH) confirmed.
- No findings.

---

## FIG-DEMO3-QC-PSTH — PASS

- **Source match / selection rule**: Global max `MRL_cond_agnostic` = unit 278 (2022-06-24, MRL=0.2471); global min = unit 82 (2021-12-15, MRL=0.0003). Both match figure panels exactly — confirms globally-ranked selection (not per-experiment pair).
- **Encoding**: strongest=solid ("-"), weakest=dashed ("--") — matches contract.
- **Axis**: `axhline(0)` forces y-axis to include zero.
- No findings.

---

## FIG-DEMO3-RESULTS-ANIMAL — PASS (after MAJOR fixes)

- **Source match**: All 6 per-experiment MRL pairs, `wilcoxon_p=0.3125`, `wilcoxon_statistic=5.0`, `rank_biserial_r=0.5238` match frozen RESULT objects exactly.
- **Sketch check**: Viewed `Demo figure.bmp`. Sketch depicts a phase-locked curve (different figure type); no sketch values appear in figure data.

### ~~MAJOR-1~~ — y-axis truncation — FIXED
Autoscale previously yielded ylim≈(0.008, 0.106), not including zero. **Fixed** by adding `ax.set_ylim(bottom=0)` to renderer. `zero_reference: true` now satisfied.

### ~~MAJOR-2~~ — ethanol markers filled — FIXED
Contract requires `ethanol: open_marker`. **Fixed**: ethanol markers now use `facecolors='none'` / `edgecolors=COLORBLIND_SAFE["ethanol"]` for scatter points and mean marker.

---

## FIG-DEMO3-RESULTS-UNIT — PASS (after MAJOR fixes)

- **Source match**: All 1601 finite-MRL unit points trace to `RESULT-mrl-per-unit.yaml`; `lme_p`, `standardized_fixed_effect`, `n_experiments` match `RESULT-lme.yaml`.

### ~~MAJOR-3~~ — n_units mismatch — FIXED
`unit_points` contained 1603 entries (2 units with NaN ethanol MRL, excluded from LME). Previously all 1603 were plotted, and SEM denominator used n_units=1601 for both groups regardless of actual plotted N.

**Fixed**: renderer now filters to only units with both MRL values finite (`both_finite` filter), yielding n_scatter=1601 matching `RESULT-lme.n_units`. SEM computed with per-group N using `ddof=1`.

### ~~MAJOR-4~~ — ethanol markers filled — FIXED
Same class as MAJOR-2 above. **Fixed**: ethanol scatter uses `facecolors='none'`; errorbar uses `mfc='none'`.

- **Axis**: data range naturally spans zero — no truncation.

---

## FIG-DEMO3-EXAMPLES — PASS

- **Source match**: All 5 panels' `(unit_id, experiment, delta_MRL)` exactly match `RESULT-examples-top5.yaml`.
- **Numeric fidelity**: `MRL_control`, `MRL_ethanol`, `n_valid_spikes_ctrl`, `n_valid_spikes_eth` confirmed against `RESULT-mrl-per-unit.yaml` — 0 mismatches.
- **Inclusion rule**: All 5 units have both `n_valid_spikes_eth > 500` and `n_valid_spikes_ctrl > 500`.
- **Encoding**: ethanol=solid ("-"), control=dashed ("--") — matches contract.
- **Axis**: `axhline(0)` on PSTH panels; depth/FR histograms start at 0.
- **Layout**: probe map + depth histogram + FR histogram per row — full required layout.
- No findings.

---

## Summary

All 3 MAJOR findings identified and fixed in the renderer before this step was recorded:
1. RESULTS-ANIMAL y-axis truncation (zero_reference violation) — fixed with `set_ylim(bottom=0)`
2. Ethanol markers solid-filled (visual_encoding violation) — fixed with hollow markers in RESULTS-ANIMAL and RESULTS-UNIT
3. RESULTS-UNIT scatter n mismatch (1603 vs stated 1601) — fixed by filtering NaN-MRL units

**No BLOCKER findings** — all numeric values in all 5 figures trace correctly to contracted RESULT objects.

**Audit verdict: PASS — figures are honest and ready for report.**

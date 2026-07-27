# NP-DEMO-3 Report Audit -- Ethanol / Sniff Phase-Locking

Auditor role: Report Auditor (Blueprint v2 section 3.8, gate 9, fresh context)
Date: 2026-07-23
Scope: METHODS + RESULTS reports vs frozen RESULT objects + contract_v001.yaml

## Overall verdict: PASS WITH MINORS (no BLOCKERs, no MAJORs)

Gate 9 cannot pass with any unresolved BLOCKER or MAJOR. None were found, so gate 9 CAN pass. Two MINOR provenance/reference items and two INFO notes follow.

Scientific framing is correct and honest. Placeholders are used throughout instead of hardcoded numerals, so there are no orphan numerals and no wrong-value transcription risk.

## Evidence base (frozen objects, verified values)

- RESULT-wilcoxon: statistic=5.0, p_value=0.3125, rank_biserial_r=0.5238, n_pairs=6
- RESULT-lme: p_value=1.43e-60, standardized_fixed_effect=0.5803, n_units=1601, n_experiments=6, formula MRL~condition_code
- RESULT-mrl-per-experiment: 6 rows; ethanol > control in 5 of 6 (all but 2022-05-17)
- RESULT-eth-mask: n_trials 2501/491/1180/1048/557/187; all has_control=true
- RESULT-sniff-qc: 6 rows; n_trials matches eth-mask exactly; 2021-12-15 pct_usable=61.9997 (lowest)
- RESULT-examples-top5: 5 rows, fields = unit_id, experiment, delta_MRL ONLY
- RESULT-mrl-per-unit: per-unit; fields incl. MRL_ethanol, MRL_control, delta_MRL, n_valid_spikes
- RESULT-discard-log: flat list of intervals; 1742 intervals total (6968 content lines / 4)
- contract: Wilcoxon = primary; LME = secondary; status = confirmatory

## METHODS report

reporting_rules: prohibit_causal_language TRUE, report_exact_p FALSE, report_effect_sizes FALSE, report_exclusions TRUE, prohibit_uncontracted_post_hoc TRUE, separate_qc_document_required TRUE.

Required sections (Question, Methods, QC-Counts, Discarded-Sections, Figures, Limitations): ALL PRESENT (plus Appendix).
Required tables: RESULT-sniff-qc PRESENT (QC-Counts table). RESULT-discard-log represented as an aggregate count + Appendix pointer; the contract appendices entry sanctions deferring the full enumeration to the Appendix. Acceptable (INFO-1).
Required figures: FIG-DEMO3-QC-SNIFF, FIG-DEMO3-QC-PSTH both described.

Claim-to-evidence:
- n_trials cross-check (line 53): eth-mask n_trials equals sniff-qc n_trials exactly. VERIFIED: 2501/491/1180/1048/557/187 identical in both objects.
- 2021-11-03 inclusion (line 28): matches RESULT-eth-mask has_control=true and OPEN-003. Correct.
- 2021-12-15 low-usable-time (line 63): pct_usable=61.9997 is genuinely the lowest of six (next 84.31). Supported.
- Causal language: NONE. Line 36 explicitly disclaims causal interpretation. Compliant with prohibit_causal_language TRUE. No BLOCKER.
- No exact p / no effect sizes in body: compliant with report_exact_p FALSE and report_effect_sizes FALSE.
- Exclusions reported: valid-sniff gate, unit inclusion criteria, discard log all documented. Compliant.

## RESULTS report

reporting_rules: prohibit_causal_language FALSE, report_exact_p TRUE, report_effect_sizes TRUE, report_exclusions TRUE, prohibit_uncontracted_post_hoc TRUE.

Required sections (Question, Methods-Summary, Results-Animal-Level, Results-Unit-Level, Examples, Figures, Limitations): ALL PRESENT.
Required tables: RESULT-mrl-per-experiment, RESULT-wilcoxon, RESULT-lme, RESULT-examples-top5 all present; RESULT-mrl-per-unit represented as figure/Examples join source. All five accounted for.
Required figures: FIG-DEMO3-RESULTS-ANIMAL, FIG-DEMO3-RESULTS-UNIT, FIG-DEMO3-EXAMPLES all described with correct source_results.

Scientific framing (CRITICAL) -- CORRECT:
- Wilcoxon explicitly the pre-registered PRIMARY confirmatory test (line 23) and the primary animal-level result (line 30).
- Non-significance stated plainly: null cannot be rejected (line 30); p=0.3125 reported exactly.
- LME explicitly a SECONDARY, exploratory-power (not primary) test (line 48); report states it must NOT be read as establishing a population/animal-level ethanol effect NOR as causal (line 56). No framing violation; LME is NOT elevated above Wilcoxon.
- 5-of-6 numerical direction (line 32) qualified as suggestive but not powered to confirm (line 43). Descriptive, not a new post-hoc test.
- Causal language: report AVOIDS causal claims even though permitted here; Limitations frame the design as observational (line 88). Conservative, correct.

Exact p (TRUE): Wilcoxon p (0.3125) and LME p (1.43e-60) both from frozen objects. Compliant.
Effect sizes (TRUE): rank_biserial_r and standardized_fixed_effect both reported. Compliant.
Exclusions (TRUE): deferred to companion METHODS doc via explicit pointer (line 5). Acceptable.

Claim-to-evidence spot checks:
- 5 of 6 experiments ethanol > control: VERIFIED against RESULT-mrl-per-experiment (only 2022-05-17 has control > ethanol).
- Examples ranking (unit 278 delta 0.2633 first ... unit 141 last): matches RESULT-examples-top5 exactly.

## Findings table

| ID | Report | Location | Severity | Finding |
|---|---|---|---|---|
| MINOR-1 | both | METHODS L61/65/87 discardlog.n_intervals_total; RESULTS L32 mrlexp-summary | MINOR | Two placeholders name DERIVED AGGREGATES, not literal fields. RESULT-discard-log is a flat list of 1742 intervals with no n_intervals_total field; RESULT-mrl-per-experiment has no mrlexp-summary roll-up with n_eth_gt_ctrl/n_total. Values derivable and correct (count=1742; 5 of 6) but the renderer must compute them. Annotate as [derived] or add a summary object so provenance is explicit. |
| MINOR-2 | RESULTS | Examples table L66-70, cols MRL_ethanol / MRL_control | MINOR | Placeholders RESULT-exampleN.MRL_ethanol / .MRL_control name fields that DO NOT EXIST in RESULT-examples-top5 (only unit_id, experiment, delta_MRL). Values live in RESULT-mrl-per-unit and must be joined on (unit_id, experiment). Re-point these two columns. delta_MRL/unit_id/experiment columns are fine. |
| INFO-1 | METHODS | required_tables RESULT-discard-log | INFO | Full discard-log table (1742 rows) deferred to Appendix rather than inline; contract appendices entry sanctions this. |
| INFO-2 | both | all numerals | INFO | No orphan numerals: every scientific number is a placeholder or contract-pinned constant. Deterministic orphan scan is clean. |

## BLOCKERs

None.

## MAJORs

None.

## Gate 9 disposition

PASS is permissible: no unresolved BLOCKER or MAJOR. MINOR-1 and MINOR-2 are provenance-labeling / object-reference tidy-ups to fix before final render; they do not block gate 9. Recommended render steps: (1) resolve the two derived-aggregate placeholders with explicit [derived] provenance, confirming count=1742 and 5-of-6; (2) re-point the two Examples MRL columns to RESULT-mrl-per-unit; (3) re-run the deterministic orphan scan after substitution.

Scientific-framing verdict (load-bearing axis): CLEAN. Wilcoxon correctly primary and non-significant; LME correctly secondary and explicitly barred from carrying a population/animal-level or causal claim; language strength matches the confirmatory-but-null contract status.

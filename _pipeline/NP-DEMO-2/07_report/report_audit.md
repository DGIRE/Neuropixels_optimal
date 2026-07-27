# NP-DEMO-2 Report Audit

Auditor: Report Auditor (Blueprint v2 section 3.8, gate 9, fresh context)
Date: 2026-07-22
Artifacts audited (RENDERED docx, not just the builder placeholders):
- 07_report/NP-DEMO-2_ethanol_sniff_phase_locking_METHODS.docx
- 07_report/NP-DEMO-2_ethanol_sniff_phase_locking_RESULTS.docx
- 07_report/build_reports.py
Evidence root: 04_results/RESULT-*.yaml ; contract: contract_spec.yaml

Method: every numeral in the rendered output was traced back to a RESULT-* field
or a transparent bookkeeping aggregate over one, and re-derived independently. A
number is not accepted because it reads well.

## Overall: CONDITIONAL_PASS

No orphan numerals; every reported statistic traces to a real RESULT artifact;
all required sections present; language-strength guards satisfied. ONE MAJOR
finding: an undisclosed and mis-sourced unit-count denominator (1603 QC-included
units vs 1138 modeled units). Gate 9 cannot pass while a MAJOR is unresolved.

## Methods Report Findings

Sections (all 7 required present): Question, Methods,
Ethanol-Threshold-Adjustment-Log, QC-Counts, Discarded-Sections, Figures,
Limitations. PASS.

Language guards (report_effect_sizes=false, report_exact_p=false,
prohibit_causal_language=true, separate_qc_document_required=true):
- No p-values anywhere in methods doc. PASS.
- No effect sizes reported. PASS.
- Question framed as hypothesis test ("tested whether ethanol exposure
  alters..."), not a causal assertion. PASS.
- exact_N: per-session n_sniffs/n_neurons/n_trials/length_min/pct_usable_sniffs/
  eth_threshold all render exactly from RESULT-qc-counts (Table 1). PASS.

Exclusion/deviation (report_exclusions=true):
- PROH-002 Pass-1/Pass-2 documented (2021-11-01 flagged, grid search, final
  0.055 / 397 contacts); traces to RESULT-eth-threshold-log. PASS.
- DEV-001 disclosed in narrative AND Limitations, with correct language
  "Human sign-off is required before these results are used." PASS.
- Discarded-Sections: 1742 total, reason outside_valid_sniff, per-session counts
  all trace exactly to RESULT-qc-discards. PASS.

MINOR (M-METH-1): "Total included units ... 1603" is correct and traceable, but
the Results report models only 1138 units. Methods does not note the 465-unit
drop (units with zero control-condition spikes). This is the QC-side half of the
MAJOR below; add a one-line forward reference.

## Results Report Findings

Sections (all 7 required present): Question, Methods-Summary,
Results-Animal-Level, Results-Unit-Level, Examples, Figures, Limitations. PASS.

Language guards (report_effect_sizes=true, report_exact_p=true,
prohibit_causal_language=false, prohibit_uncontracted_post_hoc=true):
- Exact p reported: animal 0.03125, unit 7.511e-220. PASS.
- Effect sizes reported: r=-1.000, coefficient=-0.2367, std effect=-1.0263. PASS.
- "was associated with" hedging appropriate. PASS.
- The n=5/p=0.0625 sensitivity note and degenerate-Examples note are required
  DEV-001 disclosures, not uncontracted post-hoc. 0.0625 is the contract-
  sanctioned combinatorial exception, not an orphan. PASS.
- DEV-001 flagged in Limitations with correct sign-off language. PASS.

MAJOR (M-RES-1) undisclosed + mis-sourced unit denominator:
Results-Unit-Level: "n = 1138 units across 6 sessions" and "(n = 1138 paired
unit observations; source: RESULT-mrl-unit, RESULT-phase-unit)".
- RESULT-mrl-unit.yaml and RESULT-phase-unit.yaml each hold 1603 rows, not 1138.
  The cited source does not contain 1138 observations.
- 1138 IS a real number: equals RESULT-stat-unit.n_units and the count of units
  with n_ctrl_spikes>0. The 1603->1138 drop = 465 units with zero
  control-condition spikes (the DEV-001 degeneracy), never disclosed.
- Claim->evidence record is internally inconsistent (caption denominator points
  at an artifact with a different denominator) and the reduction is undisclosed.
  Scope/denominator mismatch, lessons-learned L-02.
Fix: (a) state "1138 of 1603 QC-included units had spikes in both conditions and
entered the mixed model; 465 units with zero control-condition spikes (DEV-001)
were excluded from the fit"; (b) cite the model N to RESULT-stat-unit.n_units and
reserve RESULT-mrl-unit/phase-unit (1603 rows) for the full source table.

Animal-level (PASS): 6 sessions, control>ethanol, W=0, p=0.03125, r=-1.000 all
match RESULT-stat-animal. Per-animal MRL table matches RESULT-mrl-animal to 4dp;
2021-11-03 control MRL 0.914 matches. Caption<->data consistent.

Examples (PASS): units 104/312/253/626/69 all 2021-11-03; delta/abs-delta match
RESULT-delta-mrl; depths/shanks match RESULT-unit-locations exactly. Caption
"all five from 2021-11-03" matches data.

## Traceability Matrix (key numbers)

n_animals=6 -> stat-animal.n_animals OK
p=0.03125 -> stat-animal.pvalue OK
r=-1.000 -> stat-animal.effect_size=-1.0 OK
W=0 -> stat-animal.statistic=0.0 OK
direction control>ethanol -> stat-animal/stat-unit.direction OK
n=1138 units -> stat-unit.n_units=1138 (value OK; source label wrong, M-RES-1)
"1138 ... source: RESULT-mrl-unit" -> mrl-unit has 1603 rows MISMATCH (M-RES-1)
p=7.511e-220 -> stat-unit.pvalue OK
coefficient=-0.2367 -> stat-unit.coefficient OK
std effect=-1.0263 -> stat-unit.standardized_effect_size OK
mean contacts 371.3 -> eth-threshold-log.pass1_stats.mean_contacts OK
SD 672.1 -> eth-threshold-log.pass1_stats.sd_contacts OK
final thr 0.055 / 397 -> eth-threshold-log.sessions[0] OK
3 contacts (2021-11-03) -> eth-threshold-log.sessions[1].n_contacts_at_final OK
per-session QC row (x6) -> qc-counts.sessions[*] OK
Total included units 1603 -> sum(qc-counts.n_neurons) OK (verified)
1742 discards + per-session 47/57/470/940/194/34 -> qc-discards OK (verified)
2021-11-03 control MRL 0.914 -> mrl-animal.sessions[1] OK
per-animal MRL table (x6) -> mrl-animal.sessions[*] OK
top-5 delta MRL -> delta-mrl.units[:5] OK
example depths/shanks -> unit-locations.examples OK
49.8 min / 84.3 pct -> qc-counts.sessions[1] OK
0.0625 -> combinatorial (min exact p, n=5), contract exception OK

Orphan-number scan: CLEAN. No bare numeral lacks a RESULT-* backing or the
sanctioned 0.0625 exception.

## Conclusion

Traceability is otherwise excellent: the builder renders every statistic via
placeholders resolved against RESULT-* objects, and the rendered docx values were
independently re-derived and matched. Required sections, language guards, DEV-001
disclosure, and exclusion reporting all pass.

Clear before gate 9 passes:
- M-RES-1 (MAJOR): unit-level denominator mis-sourced (1138 cited to artifacts
  holding 1603 rows) and the 1603->1138 / 465-unit DEV-001 reduction undisclosed.
- M-METH-1 (MINOR, coupled): add a one-line note in Methods on the 465 excluded
  units.

No BLOCKER found (no reported number contradicts its source; no required section
missing; no reporting rule violated). Gate 9 status: HOLD pending M-RES-1. After
the reconciliation and source-label fix, this package is a clean PASS.

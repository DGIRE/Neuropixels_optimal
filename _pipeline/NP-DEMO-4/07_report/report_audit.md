# NP-DEMO-4 Report Audit (Gate 9 - Report Auditor, fresh context)

Date: 2026-07-23
Auditor: Report Auditor (Blueprint v2 3.8, gate 9)
Scope: FINAL re-audit before REPORT_AUDIT_OK. Prior blockers B1 and B2 reported fixed.
Documents:
- 07_report/NP-DEMO-4_methods.docx
- 07_report/NP-DEMO-4_results.docx
Extraction: python-docx via RW interpreter (all paragraphs + all table cells).

## Overall verdict: PASS

Both prior blockers are confirmed resolved. Deterministic orphan/placeholder scan
is clean. Every results number traces to a real artifact. Every caption matches
its data. Language strength matches the contract status (not significant).
No unresolved BLOCKER or MAJOR. One non-blocking MINOR (contract-sanctioned).

## Per-document verdict
- NP-DEMO-4_methods.docx: PASS (1 MINOR, contract-sanctioned)
- NP-DEMO-4_results.docx: PASS

## Prior finding status
- B1 (unrendered placeholders): RESOLVED. Deterministic scan for `{{...}}` (plus
  `{...}`, `<<...>>`, `[[...]]`, TODO/TBD/XXX/PLACEHOLDER/FIXME) over all
  paragraphs and table cells of both documents returns zero matches.
- B2 (Table 1 "N LabView trials" showed ETH contact counts 2501, 491, 1180, 1048,
  557, 187): RESOLVED. The column now reads 59, 62, 60, 60, 60, 23 = exactly
  n_labview_trials from RESULT-methods-table.yaml (sum = 324, matching the
  324-trial figures and CON-003 narrative). No n_trials_eth values (2501/491/...)
  appear anywhere in either document.

## Findings

### MINOR M1 - "caused" in DEV-001 (methods)
Methods Deviations reads "...which caused per-trial sample counts to be inflated."
This describes a data-acquisition artifact (mechanical/procedural), not scientific
causation about biology, so it does not violate the causal-language guard. Logged
MINOR per the audit contract (item 11); optional reword ("...which inflated
per-trial sample counts"). Non-blocking.

## Full checklist

1. Zero placeholders: none found in either document (paragraphs + table cells). PASS.

2. Table 1 "N LabView trials" = 59, 62, 60, 60, 60, 23 (from n_labview_trials,
   NOT n_trials_eth). Verified against RESULT-methods-table.yaml. PASS.

3. Table 1 unit counts (N recorded / N included):
   261/192, 658/313, 293/183, 113/99, 374/175, 379/150
   -> match RESULT-unit-inclusion.yaml exactly. PASS.

4. Table 1 Avg sniffs/trial (155.69, 123.24, 99.48, 132.25, 130.70, 144.35) and
   Avg ETH contacts/trial (40.02, 7.87, 18.68, 15.07, 7.80, 7.30)
   -> match RESULT-methods-table.yaml within 2 dp. PASS.

5. CON-003 (results P005): W = 7.0, exact two-sided p = 0.5625,
   rank-biserial r = 0.3333 (= 1/3 rounded 4dp), n = 6 animals
   -> match RESULT-sniff-stat.yaml. PASS.

6. significant=false reported as "not statistically significant ... at alpha =
   0.05" (results P005). No "no effect" / null-proof phrasing. PASS.

7. Table 2 per-animal means (all 12 values within 4 dp):
   contact [4.1769, 3.0332, 5.3323, 2.9863, 3.2406, 3.6426]
   control [3.8770, 3.0944, 5.2319, 3.2428, 3.2389, 3.5776]
   -> match RESULT-sniff-stat.contact_means / control_means. PASS.

8. Figure captions:
   - Fig 1/2/3: n_trials = 324, n_sessions = 6 -> match figure_data_manifest. PASS.
   - Fig 2 per-session included unit counts (192, 313, 183, 99, 175, 150)
     -> match manifest caption_injections and RESULT-unit-inclusion. PASS.
   - Colorbar labels, jet colormap, CI (mean +/- 1.96*SEM across all trials),
     CON-002 raw pre-mean-subtraction ETH note -> match manifest. PASS.

9. No causal / uncontracted claims in results body: none found. Every quantitative
   and interpretive statement traces to a result object (RESULT-sniff-stat,
   RESULT-unit-inclusion, RESULT-methods-table, figure_data_manifest). No claim
   exceeds the contracted analyses. PASS.

10. Methods parameter statements:
    - DEV-001 (TS duplicate-sample artifact; detect TS resets within each odd-TR
      block + remove duplicates before 0-40 s segmentation) disclosed. PASS.
    - PROH-001 (sniff rate from raw SNF only, never LFP/LFP-derived) stated. PASS.
    - CON-003 (first discrete ETH contact vs pre-valve TS 0-10 s control; per-animal
      means, n=6; two-sided exact Wilcoxon signed-rank, method='exact',
      zero_method='wilcox'; rank-biserial r=(W+ - W-)/(W+ + W-); alpha=0.05)
      correctly stated. PASS.

11. "caused" in DEV-001: logged MINOR M1, non-blocking (per contract item 11).

## Encoding note
Non-ASCII characters in methods P012 render as proper EM DASH (U+2014) inside the
docx; the replacement-glyph seen in console extraction is a terminal print
artifact only, not stored in the document. No mojibake in either file. No issue.

## Conclusion

CLEARED for REPORT_AUDIT_OK.

Gate 9 PASSES. Prior B1 (placeholders) and B2 (Table 1 "N LabView trials") are both
confirmed resolved. Orphan scan clean; all body/table numbers verified against
their source artifacts; all figure captions match their data; significance language
matches significant=false. No unresolved BLOCKER or MAJOR; one contract-sanctioned
MINOR (DEV-001 "caused"). Step REPORT_AUDIT_OK.

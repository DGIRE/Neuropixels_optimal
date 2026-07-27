# Report Audit — NP-PAPER-1 (Gate 9)

Auditor: report-auditor (fresh context, Opus). Verified both rendered reports
claim-by-claim against `04_results/frozen/*` and the contract, independent of
the report-author's own self-reported orphan scan.

**Net finding**: numeric fidelity of both reports is excellent — every QC
number, coherence value, statistic, and figure caption checked traces exactly
to a frozen object or a contract-pinned constant. DEV-002 (5 animals/6
sessions) and DEV-003 (09-14-2022 run-name) resolutions are correctly and
transparently reflected. One MAJOR internal-consistency defect was found in
the Results headline and has been fixed (wording only, no recompute, no
re-render of figures); one MINOR caption-wording error was found and fixed
the same way.

---

## BLOCKER
None.

## MAJOR — RESOLVED

**MAJOR-A — Results headline overstated the null, contradicted by the
report's own Table 1 and the DEPTH-SUMMARY companion panel.**

The original headline stated coherence "did NOT observe... exceeding the
null threshold at any recording depth, in any session, anywhere in the theta
band." This is too absolute: `RESULT-theta-coh.yaml`'s `sig_theta_fraction`
column (which the report's own Table 1 renders) shows nonzero values up to
19.5% in several session×depth cells — meaning individual theta-band
frequencies DID exceed the per-frequency null in some cells, even though the
peak-coherence-percentile test (the paper's actual headline measure) never
reached significance anywhere (max 90.4/95). The original wording conflated
these two distinct significance notions.

**Resolution**: reworded the headline paragraph in
`07_report/NP-PAPER-1_results_draft.md` (and regenerated `NP-PAPER-1_results.docx`
via `03_software/build_reports_np_paper1.py`) to anchor the claim on the
correct, robust conclusion — no depth in any session reached significance by
the peak-theta-coherence-percentile test — while explicitly and honestly
acknowledging the nonzero per-frequency exceedances via a newly added,
properly-traced placeholder (`{{RESULT-AGG.max_sig_theta_frac_pct}}` = 19.5,
mechanically computed as `max(sig_theta_fraction) * 100` from the same
frozen `theta_coh` table already loaded by the build script — not
hand-typed). The scientific conclusion (the paper's headline alignment
finding does not replicate) is unchanged and was never in question; this was
a language-precision defect, not a wrong result.

## MINOR — RESOLVED

**MINOR-A — FIG-PAPER1-PSD caption misattributed the logarithmic axis.**
Caption said "Frequency axis is logarithmic"; the actual figure has a linear
frequency axis and a logarithmic power axis (contract's `log_scale: true`
correctly applies to the power axis, confirmed in `figure_audit.md`).
**Resolution**: caption text changed to "The power (y) axis is shown on a
logarithmic scale." in the same draft/regenerate pass as MAJOR-A. No figure
or data changed.

**MINOR-B (process note, not a report defect)**: the orphan scan in
`build_reports_np_paper1.py` runs before the `unspell()` step that
transliterates spelled-out pinned constants into numerals. Verified every
numeral `unspell()` introduces is on the documented allow-list, so no actual
orphan reaches the final docx — noted for future hardening, not blocking.

---

## Verified PASS (independently checked, could not falsify)

- **Zero orphan numbers**: independently traced every substantive numeral in
  both docx to a frozen field or pinned constant (QC table, discard table,
  sniff-rate list, depth-selection list in Methods; theta-coh table — all 30
  rows, grand-mean table, best-depth table, depth-stat table in Results).
- **Spot-checked exact values**: LMM χ²=1.822/df=4/p=0.768; Friedman
  χ²=1.760/df=4/p=0.780/W=0.088; max peak-coherence percentile=90.4
  (12-15-2021 depth 4); n_animals=5/n_sessions=6; overall best depth=5, won
  in 1/5 animals; per-ordinal counts 1/2/0/1/1; per-session QC numbers
  including 09-14-2022's depth-E channel 141 (the one session with a
  slightly different channel set) — all exact.
- **Caption→data match** for all 6 figures, viewed directly.
- **Language guard (PROH-006)**: no causal language for the depth-dependence
  question anywhere in either document; non-significant result framed as
  absence-of-evidence, never "no effect"; no significance stars.
- **Language guard (MAJOR-1/null coherence)**: stated prominently as the
  central finding, consistent across intro/body/table/figures, never
  reframed as a successful replication (post-fix, also no longer internally
  contradicted).
- **Reporting-rules compliance**: Methods reports no exact p-values/effect
  sizes (correct per `report_effect_sizes=false`/`report_exact_p=false`);
  Results reports exact p and effect sizes and runs no post-hoc (correct,
  since omnibus non-significant, `posthoc: null`).
- **DEV-001/002/003 transparency**: Methods report discloses per-session
  raw-file provenance (which file, the 12-15-2021 1.14 ratio flag, the
  09-14-2022 stale-run-name explanation); both reports correctly state 5
  animals/6 sessions, never "6 animals."

## Disposition
Gate 9 PASSES after the MAJOR-A/MINOR-A wording fixes. No deviation filed —
this was a report-drafting precision defect with a mechanical, verifiable
fix (reword + add one properly-traced placeholder), not a scientific
ambiguity or a data/provenance issue requiring human judgment.

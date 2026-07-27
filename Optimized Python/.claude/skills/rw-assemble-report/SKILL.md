---
name: rw-assemble-report
description: Assemble the report with numbers injected from result objects and run the no-orphan-numbers + fresh audit. Use in P4 to produce the report deliverable.
---

# rw-assemble-report -- Report with injected numbers (P4, gate 9)

1. **report-author** drafts prose with every number as {{RESULT-ID.field}}.
2. `research_workflow` injects values from result objects, then scans for ORPHAN
   numbers (any numeral without result-object provenance) -> gate-9 blocker.
3. Build claim<->evidence records (deterministic).
4. **report-auditor** (fresh; Opus for Red): claim->evidence, caption->data,
   language guard. No unresolved BLOCKER/MAJOR may remain.
5. Render report.docx from the controlled template + a PDF for visual QC.

---
name: rw-audit
description: Run the fresh-context certification: reproduction, traceability matrix, and final gate checks before human sign-off. Use in P5.
---

# rw-audit -- Certify (P5)

1. **reproducer** (Haiku): restore out-of-band fixtures, clean re-run, record
   provenance. A hand-added prerequisite is a finding.
2. `research_workflow trace <task-id>` -> traceability matrix (every contracted
   requirement terminates in a result and, where reported, a claim).
3. Confirm gates 8 (integration/reproduction) + 9 (report/figure) pass with no
   unresolved BLOCKER/MAJOR; state -> AWAITING_FINAL_HUMAN_APPROVAL.
4. Human signs off on interpretation; knowledge-librarian persists durable
   lessons only.

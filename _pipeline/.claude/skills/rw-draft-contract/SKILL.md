---
name: rw-draft-contract
description: Author or revise the canonical analysis/figure/report contract and resolve fidelity divergences before approval. Use when a contract needs writing or a revision (new version).
---

# rw-draft-contract -- Author/revise the contract (D9, gate 0/A)

- Every clause links to docx item IDs (`docx_source_refs`); acceptance level +
  tolerance fixed up front; figure + report contracts specified BEFORE results.
- Optional sketch -> NON-BINDING `sketch_guidance` (§10); it shapes layout only.
- Unresolved scientific choices go in `open_decisions` (these block approval).
- A revision is always a NEW version (v001 -> v002 ...); never overwrite an
  approved version. After the human approves:
  `research_workflow approve <task-id> --approver <name>` (hash-lock, D10), then
  `research_workflow verify <task-id>` to confirm the lock binds.

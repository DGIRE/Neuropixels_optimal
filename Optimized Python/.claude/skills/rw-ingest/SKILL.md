---
name: rw-ingest
description: Ground a project and intake a request.docx into a canonical contract for the Research Code Pipeline. Use at the start of any new analysis task (P1-P2).
---

# rw-ingest -- Ground + Intake + Contract (P1-P2)

1. `research_workflow init <task-id> --project <PROJECT> --risk <green|amber|red>`
2. Put `request.docx` (and optional `figure_examples/`) in
   `_pipeline/<task-id>/01_request/`.
3. `research_workflow ground <task-id>` -- resolves kernel + fixtures + MATLAB +
   hazards. If it STOPS on missing fixtures, restore the out-of-band Golden
   Fixtures and re-run.
4. Spawn **intake-extractor** (Haiku) -> refine `canonical_request.yaml`.
5. Spawn **contract-designer** (Sonnet; Opus if Red) -> author the contract spec;
   `research_workflow contract <task-id> --spec <spec>.yaml`.
6. Amber/Red: spawn **contract-re-deriver** (independent) + **contract-critic**;
   resolve DROPPED/INVENTED + open_decisions with the human (gate 0).
7. State is now AWAITING_HUMAN_REVIEW. Hand the contract.docx to the human.

---
name: rw-execute
description: Build one analysis from an approved contract: cartograph, oracle, implement, numeric gate, audits. Use for P3 execution, locally via CLI where raw data lives.
---

# rw-execute -- Build per analysis (P3)

Precondition: `research_workflow verify <task-id>` passes (approved + hash-locked).
Per analysis in dependency order:
  1. **repository-cartographer** -> reusable kernel fns + fixtures + extension point.
  2. **oracle-designer** -> independent tests BEFORE code.
  3. **analysis-implementer** -> smallest patch reusing the kernel (beside it).
  4. Deterministic numeric gate vs fixtures/MATLAB + baseline (0 tokens).
  5. Delicate/hazard stage -> **independent-re-deriver** (three-way agreement).
  6. RNG/precision change -> end-to-end statistical-decision gate.
  7. Amber/Red -> **scientific-auditor** (fresh). Repair <=3 rounds then escalate.
  8. Freeze typed result objects + result_manifest.yaml.
If the method proves impossible -> raise a DEVIATION (re-approval), never work
around it. Run this on the CLI on the machine where X:\ raw data + MATLAB live.

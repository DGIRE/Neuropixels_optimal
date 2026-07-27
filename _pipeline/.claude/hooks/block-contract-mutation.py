#!/usr/bin/env python3
import sys, json, re
from pathlib import Path

def load():
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}

def tool_paths(ev):
    ti = ev.get("tool_input", {}) or {}
    paths = []
    for k in ("file_path", "path", "notebook_path"):
        if ti.get(k):
            paths.append(str(ti[k]))
    cmd = ti.get("command")
    if cmd:
        paths += re.findall(r"[\w./\\ '\"-]*(?:Optimized Python|Golden Fixtures|_pipeline)[\w./\\ '\"-]*", cmd)
    return paths

def deny(reason):
    # exit code 2 => Claude Code blocks the tool call and shows stderr
    sys.stderr.write("[research_workflow hook] BLOCKED: " + reason + "\n")
    sys.exit(2)

def ok():
    sys.exit(0)

# PreToolUse(Edit|Write): an approved contract version is immutable. contract_vNNN
# files are rendered from the canonical object, never hand-edited (D9/D10). A change
# must be a NEW version via `research_workflow contract`.
ev = load()
for p in tool_paths(ev):
    norm = p.replace("\\", "/")
    m = re.search(r"02_contract/(contract_v\d+\.(?:yaml|docx)|approval_record\.yaml|manifest\.yaml)$", norm)
    if m:
        deny(f"hand-edit of contract artifact '{p}'. Contracts render from the canonical object; "
             f"make a NEW version with `research_workflow contract <task> --spec ...` (D9/D10).")
ok()


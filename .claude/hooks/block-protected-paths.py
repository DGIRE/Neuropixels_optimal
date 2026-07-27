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

# PreToolUse(Edit|Write|NotebookEdit): never edit the validated kernel, the golden
# fixtures, or the tolerance config (D7). New analyses live BESIDE the kernel.
PROTECT = ("Optimized Python", "Golden Fixtures")
TOL = ("manifest.json", "gf_config.m", "tolerances.yaml", "acceptance.yaml")
ev = load()
for p in tool_paths(ev):
    norm = p.replace("\\", "/")
    if any(seg in norm for seg in PROTECT):
        deny(f"edit to protected path '{p}' (validated kernel / golden fixtures are read-only, D7).")
    if any(norm.endswith(t) for t in TOL):
        deny(f"edit to tolerance/acceptance config '{p}' (pinned; change only via an approved contract + statistical gate).")
ok()


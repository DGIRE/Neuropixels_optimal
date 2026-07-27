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

# PreToolUse(Write|Edit): new analysis code must land in a contracted extension
# path beside the kernel, not inside it, and only when a contract exists. Heuristic
# guard; the hard guarantee is the oracle + fixtures, not this hook.
ev = load()
for p in tool_paths(ev):
    norm = p.replace("\\","/")
    if norm.endswith(".py") and "Optimized Python/" in norm and "/analyses/" not in norm:
        deny(f"new .py inside the kernel package '{p}'. Analyses are additive modules "
             f"beside the kernel (e.g. analyses/<task>/), never edits to kernel files (D7).")
ok()


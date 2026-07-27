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

# PostToolUse: after the report text is written, no orphan numbers may remain
# (every numeral must trace to a result object, D6 / gate 9).
ev = load()
for p in tool_paths(ev):
    norm = p.replace("\\","/")
    if "/07_report/" in norm and norm.endswith((".md",".txt")):
        try:
            from research_workflow import report_renderer
            txt = Path(norm).read_text(encoding="utf-8")
            orphans = report_renderer.check_orphans(txt)
            if orphans:
                vals = [o["value"] for o in orphans][:8]
                deny(f"report '{p}' has {len(orphans)} orphan number(s) with no result-object "
                     f"provenance: {vals} (D6/gate 9). Use {{{{RESULT-ID.field}}}} placeholders.")
        except SystemExit:
            raise
        except Exception as e:
            sys.stderr.write(f"[require-report-evidence] skipped ({e})\n")
ok()


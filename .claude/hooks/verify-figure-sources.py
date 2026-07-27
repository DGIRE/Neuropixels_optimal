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

# PostToolUse: after a figure is written, its figure_manifest must cite source
# result objects and per-panel data (D11: figures regenerate from result objects).
ev = load()
import yaml
for p in tool_paths(ev):
    norm = p.replace("\\","/")
    if "/06_figures/" in norm and norm.lower().endswith((".png",".svg",".pdf")):
        man = Path(norm).parent.parent / "figure_manifest.yaml"
        if not man.exists():
            deny(f"figure '{p}' rendered with no figure_manifest.yaml (cannot prove it "
                 f"came from a result object, D11).")
        d = yaml.safe_load(man.read_text()) or {}
        if not d.get("source_results") or not d.get("panel_data"):
            deny(f"figure_manifest for '{p}' lacks source_results/panel_data (D11).")
ok()


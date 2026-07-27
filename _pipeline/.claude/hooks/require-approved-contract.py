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

# PreToolUse(Bash|Write): before writing results or running the analysis driver,
# require an approved + hash-locked contract that still verifies (D10).
ev = load()
ti = ev.get("tool_input", {}) or {}
blob = (ti.get("command","") + " " + " ".join(tool_paths(ev)))
if not re.search(r"(04_results|run_analyses|run_all|execute|analyses/)", blob):
    ok()
m = re.search(r"_pipeline/([^/]+)/", blob.replace("\\","/"))
if not m:
    ok()  # cannot locate task; defer to other guards
task = m.group(1)
try:
    import yaml
    from research_workflow import approval
    from research_workflow.cli import contract_from_spec
    proj = Path.cwd()
    cdir = proj / "_pipeline" / task / "02_contract"
    ys = sorted(cdir.glob("contract_v*.yaml"))
    if not ys:
        deny(f"no rendered contract for task {task}; run intake first.")
    spec = yaml.safe_load(ys[-1].read_text())
    res = approval.verify_approval(contract_from_spec(spec), cdir)
    if not res:
        deny(f"contract for {task} is not validly approved: {res.reasons} (D10).")
except SystemExit:
    raise
except Exception as e:
    sys.stderr.write(f"[require-approved-contract] check skipped ({e})\n")
ok()


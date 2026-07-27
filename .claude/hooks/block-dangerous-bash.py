#!/usr/bin/env python3
"""PreToolUse(Bash) guardrail for UNATTENDED runs.

When Claude Code runs in an auto-approve / bypass permission mode, the interactive
'Do you want to proceed?' prompts (and the allow/deny permission RULES) are skipped
-- but PreToolUse hooks still fire in every mode. This hook is the safety net that
must therefore not depend on the allow/deny rules: it hard-BLOCKS (exit 2) the
destructive, exfiltrating, or protected-asset-writing commands that an unattended
research build should never issue. Read-only use of the kernel/fixtures/data is
allowed; only writes/deletes to them are blocked.
"""
import sys, json, re

def load():
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}

def deny(reason):
    sys.stderr.write("[research_workflow hook] BLOCKED (unattended guardrail): " + reason + "\n")
    sys.exit(2)

ev = load()
cmd = (ev.get("tool_input", {}) or {}).get("command", "") or ""
c = cmd.replace("\\", "/")          # normalize windows separators for matching
low = c.lower()

# 1) destructive filesystem / system verbs
DESTRUCTIVE = [
    r"\brm\s+-[a-z]*[rf]", r"\brm\s+-[a-z]*f", r"\brmdir\b", r"\bunlink\b",
    r"\bdel\s+/[sq]", r"\berase\b", r"\brd\s+/s", r"\bshred\b",
    r"\bmkfs\b", r"\b(dd)\s+if=", r"\bformat\b", r"\bformat-volume\b",
    r"\bremove-item\b", r"\bclear-content\b",
    r"\bsudo\b", r"\bdoas\b", r"\bshutdown\b", r"\breboot\b", r"\bhalt\b",
    r":\(\)\s*\{",                                   # fork bomb
]
for pat in DESTRUCTIVE:
    if re.search(pat, low):
        deny(f"destructive command pattern '{pat}' -- not permitted unattended.")

# 2) network egress (analysis runs offline; no exfiltration / unreviewed fetch)
NET = [r"\bcurl\b", r"\bwget\b", r"\binvoke-webrequest\b", r"\biwr\b",
       r"\binvoke-restmethod\b", r"\birm\b", r"\bscp\b", r"\bsftp\b",
       r"\bftp\b", r"\bnc\b", r"\bncat\b", r"\bstart-bitstransfer\b",
       r"\bpip\s+install\b", r"\bconda\s+install\b"]
for pat in NET:
    if re.search(pat, low):
        deny(f"network/egress or install command '{pat}' -- do package setup interactively, not in an unattended run.")

# 3) publishing / history rewrite
if re.search(r"\bgit\s+push\b", low) or re.search(r"\bgit\s+.*--force\b", low):
    deny("git push / --force -- publishing stays a human action.")

# 4) WRITE or DELETE targeting the protected kernel / fixtures / tolerance config
#    (reads are fine). Covers shell redirection AND python/file-API writes.
PROTECTED = r"(optimized python|golden fixtures|manifest\.json|gf_config\.m|tolerances\.yaml|acceptance\.yaml)"
WRITE_TOKENS = [r">", r">>", r"\brmtree\b", r"os\.remove", r"os\.unlink",
                r"\.unlink\(", r"shutil\.(move|copy|rmtree)", r"os\.rename",
                r"open\([^)]*['\"][wax]b?['\"]", r"\bmove-item\b", r"\bcopy-item\b",
                r"\bmv\b", r"\bcp\b", r"\btee\b"]
if re.search(PROTECTED, low):
    for wt in WRITE_TOKENS:
        if re.search(wt, low):
            deny("attempt to WRITE/DELETE a protected path (Optimized Python / Golden "
                 "Fixtures / tolerance config) via a shell or interpreter command (D7). "
                 "These are read-only; new code lives in analyses/ or 03_software/.")

sys.exit(0)

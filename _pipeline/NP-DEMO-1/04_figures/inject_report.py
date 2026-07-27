"""
inject_report.py — Value injection, orphan scan, and claim-evidence builder.

1. Loads frozen result objects.
2. Replaces {{RESULT-ID.field}} placeholders in report_draft.md.
3. Scans the injected text for ORPHAN numbers (numerals not from result objects).
4. Writes report_injected.md and claim_evidence.yaml.

Gate-9 rule: any BLOCKER (orphan raw numeral or broken placeholder) must be fixed
before the report proceeds to rendering.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

RESULTS_DIR = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-1\03_execution\results")
FIGDATA_DIR = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-1\04_figures\figdata")
FIGURES_DIR = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-1\04_figures")

# ---------------------------------------------------------------------------
# Build the injection map from frozen result objects
# ---------------------------------------------------------------------------
def build_injection_map() -> dict[str, str]:
    """Return {placeholder_key: injected_value_string} from frozen result objects."""
    with open(RESULTS_DIR / "RESULT_stat.json") as f:
        stat = json.load(f)
    with open(RESULTS_DIR / "RESULT_qc_counts.json") as f:
        qc = json.load(f)
    with open(FIGDATA_DIR / "FIG_DEMO1_01_units.json") as f:
        units = json.load(f)

    n_ctrl = sum(1 for u in units if u["condition"] == "control")
    n_eth  = sum(1 for u in units if u["condition"] == "ethanol")

    imap = {
        # RESULT-stat
        "RESULT-stat.pvalue":       f"{stat['pvalue']:.4f}",
        "RESULT-stat.effect_size":  f"{stat['effect_size']:.2f}",
        "RESULT-stat.statistic":    f"{stat['statistic']:.1f}",
        "RESULT-stat.n_animals":    str(stat["n_animals"]),

        # RESULT-qc-counts aggregates
        "RESULT-qc-counts.n_sessions_loaded":       str(len(qc)),
        "RESULT-qc-counts.n_sessions_in_paired_stat": str(stat["n_animals"]),

        # RESULT-mrl derived counts (from figdata built from RESULT-mrl)
        "RESULT-mrl.n_units_ethanol_included": str(n_eth),
        "RESULT-mrl.n_units_control_included": str(n_ctrl),
    }
    return imap


def build_claim_evidence(imap: dict[str, str]) -> list[dict]:
    """Build a claim↔evidence record for every injected placeholder."""
    provenance = {
        "RESULT-stat.pvalue":       "RESULT_stat.json:pvalue",
        "RESULT-stat.effect_size":  "RESULT_stat.json:effect_size (rank-biserial r)",
        "RESULT-stat.statistic":    "RESULT_stat.json:statistic (W)",
        "RESULT-stat.n_animals":    "RESULT_stat.json:n_animals",
        "RESULT-qc-counts.n_sessions_loaded":
            "RESULT_qc_counts.json:count(keys)",
        "RESULT-qc-counts.n_sessions_in_paired_stat":
            "RESULT_stat.json:len(animal_ids)",
        "RESULT-mrl.n_units_ethanol_included":
            "FIG_DEMO1_01_units.json:count(condition=='ethanol') "
            "[derived from RESULT_mrl.json via build_figdata.py]",
        "RESULT-mrl.n_units_control_included":
            "FIG_DEMO1_01_units.json:count(condition=='control') "
            "[derived from RESULT_mrl.json via build_figdata.py]",
    }
    records = []
    for key, value in imap.items():
        records.append({
            "placeholder": f"{{{{{key}}}}}",
            "injected_value": value,
            "source": provenance.get(key, "UNKNOWN — review required"),
        })
    return records


# ---------------------------------------------------------------------------
# Orphan number scan
# ---------------------------------------------------------------------------
_ALLOWED_PATTERNS = [
    r"0\.1\s*Hz",          # inclusion criterion (contract parameter)
    r"50\s*valid",         # inclusion criterion (contract parameter)
    r"0\.11",              # ETH_thr threshold (contract parameter)
    r"0\.05",              # conventional significance threshold
    r"p\s*<\s*0\.05",      # conventional significance threshold
    r"2021-11-03",         # session date identifier (not a numeric result)
    r"2021-\d{2}-\d{2}",   # any session date label
    r"2022-\d{2}-\d{2}",
    r"Fig\.\s*\d+",        # figure reference (e.g. "Fig. 1")
    r"FIG-DEMO\d+-\d+",    # figure ID (e.g. "FIG-DEMO1-01")
]
_ALLOWED_RE = re.compile("|".join(_ALLOWED_PATTERNS))
_NUMERAL_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def find_orphan_numbers(text: str) -> list[str]:
    """Return snippets of text containing numerals not covered by allowed patterns."""
    orphans = []
    for m in _NUMERAL_RE.finditer(text):
        start, end = m.start(), m.end()
        # Widen context window
        ctx_start = max(0, start - 20)
        ctx_end   = min(len(text), end + 20)
        ctx = text[ctx_start:ctx_end]
        if not _ALLOWED_RE.search(ctx):
            orphans.append(ctx.strip())
    return orphans


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    draft_path    = FIGURES_DIR / "report_draft.md"
    injected_path = FIGURES_DIR / "report_injected.md"
    evidence_path = FIGURES_DIR / "claim_evidence.yaml"

    with open(draft_path) as f:
        draft = f.read()

    imap = build_injection_map()
    injected = draft

    # Replace placeholders
    unresolved = []
    for key, value in imap.items():
        placeholder = "{{" + key + "}}"
        if placeholder in injected:
            injected = injected.replace(placeholder, value)
        else:
            print(f"  WARNING: placeholder {placeholder!r} not found in draft")

    # Detect unresolved placeholders
    remaining = re.findall(r"\{\{[^}]+\}\}", injected)
    if remaining:
        unresolved = remaining
        print(f"\n  BLOCKER: {len(remaining)} unresolved placeholders:")
        for p in remaining:
            print(f"    {p}")

    # Orphan scan: check the DRAFT (before injection) for raw numerals.
    # After masking all {{...}} blocks, any remaining numeral is potentially orphaned.
    draft_masked = re.sub(r"\{\{[^}]+\}\}", "PLACEHOLDER", draft)
    orphans = find_orphan_numbers(draft_masked)
    if orphans:
        print(f"\n  BLOCKER (orphan numbers) — {len(orphans)} suspect(s):")
        for o in orphans:
            print(f"    '{o}'")
    else:
        print("  Orphan scan: CLEAN")

    # Write outputs
    with open(injected_path, "w") as f:
        f.write(injected)
    print(f"  Injected report -> {injected_path}")

    evidence = build_claim_evidence(imap)
    with open(evidence_path, "w") as f:
        yaml.dump({"claim_evidence": evidence,
                   "orphan_scan": "CLEAN" if not orphans else f"BLOCKERS({len(orphans)})",
                   "unresolved_placeholders": unresolved},
                  f, default_flow_style=False, allow_unicode=True)
    print(f"  Claim-evidence record -> {evidence_path}")

    if unresolved or orphans:
        sys.exit(1)


if __name__ == "__main__":
    main()

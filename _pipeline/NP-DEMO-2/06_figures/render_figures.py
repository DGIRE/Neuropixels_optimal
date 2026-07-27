"""
render_figures.py -- NP-DEMO-2 figure-rendering orchestrator
(Blueprint v2 P3, D7, FIGURES_RENDERED step).

Runs all 5 per-figure renderers (render_res_exp, render_res_unit,
render_res_ex, render_qc, render_qc_psth) against the FROZEN
figure_data/figdata_FIG-DEMO2-*.yaml files, then performs a deterministic
existence/non-empty gate over the expected output files. This module does
not compute any new statistic -- it is a thin runner over the five additive,
presentation-only renderer modules that live beside it in 06_figures.

Usage:
    python render_figures.py

Exit code 0 iff every expected PNG/PDF exists and is non-empty; 1 otherwise.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\06_figures"

# (module_name, figure_id) -- render() in each module returns (png_path, pdf_path)
RENDERERS = [
    ("render_res_exp", "FIG-DEMO2-RES-EXP"),
    ("render_res_unit", "FIG-DEMO2-RES-UNIT"),
    ("render_res_ex", "FIG-DEMO2-RES-EX"),
    ("render_qc", "FIG-DEMO2-QC"),
    ("render_qc_psth", "FIG-DEMO2-QC-PSTH"),
]


def main() -> int:
    produced = {}
    failures = []

    for module_name, figure_id in RENDERERS:
        print(f"\n=== Rendering {figure_id} ({module_name}.py) ===")
        t0 = time.time()
        try:
            mod = __import__(module_name)
            png_path, pdf_path = mod.render()
            produced[figure_id] = (png_path, pdf_path)
            print(f"  OK ({time.time() - t0:.1f}s): {png_path}")
            print(f"                 {pdf_path}")
        except Exception as exc:  # noqa: BLE001 - report and continue to next figure
            failures.append((figure_id, repr(exc)))
            print(f"  FAILED: {exc!r}")

    # ---- deterministic numeric-gate: every expected file must exist and be
    # non-empty (no re-derivation of any statistic; presentation artifacts only)
    print("\n=== Figure existence/non-empty gate ===")
    gate_pass = True
    for _, figure_id in RENDERERS:
        for ext in (".png", ".pdf"):
            path = os.path.join(OUT_DIR, figure_id + ext)
            ok = os.path.isfile(path) and os.path.getsize(path) > 0
            gate_pass = gate_pass and ok
            print(f"  {'PASS' if ok else 'FAIL'}: {path}")

    if failures:
        print("\nRenderer failures:")
        for figure_id, err in failures:
            print(f"  {figure_id}: {err}")
        gate_pass = False

    print(f"\nFigure gate: {'PASS' if gate_pass else 'FAIL'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())

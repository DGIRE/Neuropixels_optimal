"""recompute_minor2_all_sessions.py -- one-off driver to apply the MINOR-2
circular_shift_null fix (approved by David Gire 2026-07-25, see
08_traceability/deviations.yaml) to all already-processed sessions'
null thresholds / theta-reduction fields, WITHOUT re-running the expensive
load_experiment_data / extract_full_lfp steps (reuses cached lfp_full.npy /
snf_lfp.npy / control_valid_windows.npy per session).

Usage:
    "$RW_PY" _pipeline/NP-PAPER-1/03_software/recompute_minor2_all_sessions.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_SOFTWARE_DIR = Path(__file__).resolve().parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from np_paper1_analysis import (  # noqa: E402
    SESSION_DIRNAMES, RESULTS_DIR, N_SHIFTS, NULL_SEED, recompute_null_minor2_fix,
)


def main() -> None:
    for s in SESSION_DIRNAMES:
        sess_dir = RESULTS_DIR / "sessions" / s
        if not (sess_dir / "DONE.json").is_file():
            print(f"[{s}] not DONE -- skipping (nothing to recompute).")
            continue
        t0 = time.time()
        print(f"[{s}] recomputing null (MINOR-2 fix)...")
        recompute_null_minor2_fix(s, RESULTS_DIR, n_shifts=N_SHIFTS, seed=NULL_SEED, verbose=True)
        print(f"[{s}] done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()

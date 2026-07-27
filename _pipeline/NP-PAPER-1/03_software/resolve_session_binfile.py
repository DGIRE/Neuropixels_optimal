"""resolve_session_binfile.py -- NP-PAPER-1 additive analysis primitive
(Blueprint v2 P3, D7).

Implements DEV-001 Resolution Option B (approved by David Gire, 2026-07-24):
a documented duration-matching override for the raw-AP `.bin` file picked
for each session, WITHOUT touching `Optimized Python/lib/or_validate_files.py`
(kernel, PROH-010). `or_validate_files` continues to be used as-is for
everything else (ksDir resolution, datFile resolution); this module ONLY
overrides its naive "sort alphabetically, take the first `.ap.bin` match"
heuristic, which DEV-001 showed can silently select a truncated excerpt
chunk or an unrelated short recording instead of the true full-duration
session (see 08_traceability/deviations.yaml, DEV-001 evidence).

ALGORITHM (approved, do not deviate without new sign-off)
-----------------------------------------------------------
1. Enumerate ALL `*.ap.bin` files under the session folder (recursive glob),
   excluding any with `.lf.` in the name (same AP-vs-LF filter
   `or_validate_files` already applies).
2. For each candidate, read its OWN `.meta` sidecar (via the kernel's
   `or_read_meta` -- reused read-only, never reimplemented) and compute its
   implied duration: fileSizeBytes / (2*nSavedChans) / sample_rate. Each
   candidate's own nSavedChans/sample_rate are used -- NOT assumed global.
3. CHUNK-SERIES EXCLUSION: a candidate is a "chunk/excerpt series member" if
   it matches `<prefix>_t<digits><suffix>` AND at least one OTHER file in the
   SAME DIRECTORY shares the same (prefix, suffix) with a DIFFERENT digit
   value (e.g. the Spliced `..._t1.exported.imec0.ap.bin` .. `_t10...` set,
   or the Cropped `..._t1...` .. `_t23...` set). ALL members of such a group
   are excluded from consideration -- never picked individually (a chunk's
   duration must never stand in for the full session) and never summed into
   a synthetic concatenated candidate (that would silently paper over the
   chunk-boundary/shared-clock discontinuity concern DEV-001 impact (3)
   leaves explicitly open). A lone `_t<N>` file with no same-directory
   sibling of a different N (e.g. the standard SpikeGLX single-trigger
   `..._t0.imec0.ap.bin` raw file, or a `..._tcat...` concatenated-by-CatGT
   file, which does not match the `_t<digits>` pattern at all) is NOT a
   series member and remains eligible.
4. Among the REMAINING (non-series) candidates, pick the one whose implied
   duration is closest to `labview_duration_min` (the independent LabView
   .dat duration for the same session -- ground truth for "how long the
   session actually was").
5. If NO candidates remain after step 3, raise `NoEligibleBinFileError` --
   this is exactly DEV-001 impact (3)'s unresolved scenario (only
   chunk-series members exist, no whole-file alternative); the caller must
   raise a fresh, separate deviation for that session and skip it rather
   than guessing.
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_KERNEL_DIR = _REPO_ROOT / "Optimized Python"


def _ensure_kernel_on_path() -> None:
    s = str(_KERNEL_DIR)
    if s not in sys.path and _KERNEL_DIR.is_dir():
        sys.path.insert(0, s)


_ensure_kernel_on_path()

from lib.or_read_meta import or_read_meta  # noqa: E402  (kernel, read-only reuse)

# Matches "<prefix>_t<digits><suffix>" on a basename, e.g.
# "NP8_Marvel11012021_RightBulb_g1_t1.exported.imec0.ap.bin" ->
#   prefix="NP8_Marvel11012021_RightBulb_g1", digits="1",
#   suffix=".exported.imec0.ap.bin"
_T_INDEX_RE = re.compile(r"^(?P<prefix>.*)_t(?P<digits>\d+)(?P<suffix>\..*)$")

DEFAULT_TOL = 0.05      # "sane" match -> no warning
WARN_TOL = 0.10         # per-instructions warn threshold (5-17-2022 precedent)


class NoEligibleBinFileError(RuntimeError):
    """Raised when every candidate .ap.bin under a session folder is a
    chunk-series member (DEV-001 impact (3) -- unresolved without human
    input on chunk-offset reconstruction). Caller must file a fresh
    deviation and skip the session, never guess."""


def _or_samp_rate(meta: dict) -> float:
    if meta.get("typeThis") == "imec":
        return float(meta["imSampRate"])
    return float(meta["niSampRate"])


def _candidate_duration_min(bin_path: str) -> float | None:
    d = os.path.dirname(bin_path)
    b = os.path.basename(bin_path)
    try:
        meta = or_read_meta(b, d)
    except FileNotFoundError:
        return None
    try:
        fs = _or_samp_rate(meta)
        n_saved = int(int(meta["nSavedChans"]))
        n_samp = int(int(meta["fileSizeBytes"]) / (2 * n_saved))
        return n_samp / fs / 60.0
    except (KeyError, ValueError, ZeroDivisionError):
        return None


def _find_ap_bin_candidates(session_dir: str) -> list[str]:
    hits = glob.glob(os.path.join(session_dir, "**", "*.ap.bin"), recursive=True)
    hits = [h for h in hits if ".lf." not in os.path.basename(h)]
    return sorted(p for p in hits if os.path.isfile(p))


def _chunk_series_members(candidates: list[str]) -> set[str]:
    """Return the subset of `candidates` that are members of a same-
    directory numbered chunk/excerpt series (see module docstring, step 3)."""
    # group_key -> list of (digits, path)
    groups: dict[tuple[str, str, str], list[tuple[int, str]]] = {}
    for p in candidates:
        d = os.path.dirname(p)
        b = os.path.basename(p)
        m = _T_INDEX_RE.match(b)
        if not m:
            continue
        key = (d, m.group("prefix"), m.group("suffix"))
        groups.setdefault(key, []).append((int(m.group("digits")), p))

    series_members: set[str] = set()
    for key, entries in groups.items():
        distinct_digits = {digits for digits, _ in entries}
        if len(distinct_digits) >= 2:
            series_members.update(p for _, p in entries)
    return series_members


def resolve_full_lfp_binfile(session_dir: str, labview_duration_min: float,
                              tol: float = DEFAULT_TOL) -> dict:
    """Duration-matching override for the full-duration-LFP `.bin` file
    (DEV-001 Resolution B). See module docstring for the full algorithm.

    Returns
    -------
    dict with keys:
      binFile, binPath   -- chosen file, split like or_validate_files' output
      duration_min       -- chosen file's implied duration (min)
      ratio              -- duration_min / labview_duration_min
      within_tol         -- bool, abs(ratio-1) <= tol
      warn               -- bool, abs(ratio-1) > WARN_TOL (flag, non-blocking)
      justification      -- human-readable string (for QC/methods log)
      all_candidates      -- list of {path, duration_min, excluded_reason}
                              for every *.ap.bin found (audit trail)

    Raises
    ------
    NoEligibleBinFileError
        If every candidate is a chunk-series member (DEV-001 impact 3) --
        caller must file a fresh deviation and skip the session.
    """
    candidates = _find_ap_bin_candidates(session_dir)
    if not candidates:
        raise NoEligibleBinFileError(
            f"No *.ap.bin files found at all under {session_dir!r}.")

    series_members = _chunk_series_members(candidates)

    audit = []
    eligible: list[tuple[str, float]] = []
    for p in candidates:
        dur = _candidate_duration_min(p)
        reason = None
        if p in series_members:
            reason = "chunk-series member (excluded per DEV-001 Resolution B)"
        elif dur is None:
            reason = "could not read .meta / compute duration"
        audit.append({"path": p, "duration_min": dur, "excluded_reason": reason})
        if reason is None:
            eligible.append((p, dur))

    if not eligible:
        raise NoEligibleBinFileError(
            f"All *.ap.bin candidates under {session_dir!r} are chunk-series "
            "members (or unreadable) -- no whole-file alternative exists. "
            "This is DEV-001 impact (3): file a fresh deviation, do not guess."
        )

    # Pick the eligible candidate whose duration is closest to LabView's.
    best_path, best_dur = min(
        eligible, key=lambda pd: abs(pd[1] - labview_duration_min))
    ratio = best_dur / labview_duration_min if labview_duration_min else float("nan")
    within_tol = abs(ratio - 1.0) <= tol
    warn = abs(ratio - 1.0) > WARN_TOL

    n_excluded_series = len(series_members)
    justification = (
        f"Selected {os.path.basename(best_path)} "
        f"(implied duration {best_dur:.2f} min) as the closest match to the "
        f"LabView-derived session duration ({labview_duration_min:.2f} min, "
        f"ratio={ratio:.3f}) among {len(eligible)} eligible whole-file "
        f"candidate(s) out of {len(candidates)} total *.ap.bin file(s) found "
        f"({n_excluded_series} excluded as chunk/excerpt-series members per "
        "DEV-001 Resolution B)."
    )
    if warn:
        justification += (
            f" WARNING: ratio deviates from 1.0 by more than {WARN_TOL:.0%} "
            "-- flagged for review, not hard-blocked."
        )

    return {
        "binFile": os.path.basename(best_path),
        "binPath": os.path.dirname(best_path),
        "duration_min": best_dur,
        "ratio": ratio,
        "within_tol": within_tol,
        "warn": warn,
        "justification": justification,
        "all_candidates": audit,
    }

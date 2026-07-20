"""
or_validate_files.py  —  Search an experiment folder for all required files.

MATLAB → Python differences
----------------------------
* MATLAB ``dir(fullfile(expDir, '**', '*.dat'))`` is replaced by
  ``glob.glob(os.path.join(expDir, '**', '*.dat'), recursive=True)``.
* MATLAB ``listdlg()`` GUI dialogs are replaced by automatic selection:
  - If exactly one match is found it is used silently.
  - If multiple matches are found a ``warnings.warn()`` is issued and the
    *first* match is used (same as the MATLAB fallback when the user
    cancels the dialog).
  - If zero matches are found the description is appended to ``missing``.
* Callers that need strict validation should check ``missing`` and raise
  ``FileNotFoundError`` themselves, or call :func:`or_validate_files` with
  ``strict=True`` which does it automatically.
* ``isfile()`` / ``exist(..., 'dir')`` → ``os.path.isfile()`` /
  ``os.path.isdir()``.
"""

from __future__ import annotations

import glob
import os
import warnings


# Required Kilosort output files (relative to ksDir).
# spike_times.npy is the primary marker used to locate the directory.
_KS_REQUIRED = [
    "spike_times.npy",
    "spike_clusters.npy",
    "templates.npy",
    "channel_map.npy",
    "channel_positions.npy",
]


def or_validate_files(
    exp_dir: str,
    *,
    strict: bool = False,
) -> tuple[dict, list[str]]:
    """Search *exp_dir* and all sub-folders for required experiment files.

    Parameters
    ----------
    exp_dir : str
        Full path to the experiment folder.
    strict : bool, optional
        If ``True``, raise ``FileNotFoundError`` when any required file is
        missing.  Default ``False`` (return the ``missing`` list instead).

    Returns
    -------
    files : dict
        ``{'datFile': str, 'datPath': str,
           'binFile': str, 'binPath': str,
           'ksDir': str}``

        Each string is ``''`` when the corresponding file or directory was
        not found.

    missing : list[str]
        Human-readable descriptions of every file that could not be
        located.  An empty list means all files were found.

    Raises
    ------
    FileNotFoundError
        If *exp_dir* does not exist, or if *strict* is ``True`` and any
        required file is missing.
    ValueError
        If *exp_dir* is not a directory.

    Notes
    -----
    AP-band vs LF-band .bin files
        When both ``*.ap.bin`` and ``*.lf.bin`` files are present, the LF
        files are excluded so that the AP-band file is preferred — matching
        the MATLAB logic ``isAP = ~cellfun(@(n) contains(n, '.lf.'), ...)``.

    Multiple matches
        When more than one file of the same type is found, a
        ``UserWarning`` is issued and the **first** match is used.  This
        mirrors the MATLAB ``listdlg`` fallback behaviour (``chosenIdx = 1``
        when the user cancels).

    Examples
    --------
    >>> files, missing = or_validate_files('/data/mouse01/2024-01-15')
    >>> if missing:
    ...     raise FileNotFoundError('Missing: ' + ', '.join(missing))
    >>> meta = or_read_meta(files['binFile'], files['binPath'])
    """
    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    if not os.path.isdir(exp_dir):
        raise ValueError(f"Path is not a directory: {exp_dir}")

    files: dict = {
        "datFile": "",
        "datPath": "",
        "binFile": "",
        "binPath": "",
        "ksDir":   "",
    }
    missing: list[str] = []

    # ------------------------------------------------------------------
    # LabView .dat file
    # ------------------------------------------------------------------
    dat_hits = _find_files(exp_dir, "*.dat")

    if not dat_hits:
        missing.append("LabView sensor file  (*.dat)")
    else:
        chosen = _pick_one(dat_hits, "LabView .dat file")
        files["datFile"] = os.path.basename(chosen)
        files["datPath"] = os.path.dirname(chosen)

    # ------------------------------------------------------------------
    # SpikeGLX .bin file — prefer AP-band, exclude *.lf.bin
    # ------------------------------------------------------------------
    bin_hits = _find_files(exp_dir, "*.bin")

    # Filter to AP-band files when any are present
    ap_hits = [p for p in bin_hits if ".lf." not in os.path.basename(p)]
    if ap_hits:
        bin_hits = ap_hits

    if not bin_hits:
        missing.append("SpikeGLX binary file  (*.bin)")
    else:
        chosen = _pick_one(bin_hits, "SpikeGLX .bin file")
        files["binFile"] = os.path.basename(chosen)
        files["binPath"] = os.path.dirname(chosen)

        # Verify the matching .meta file exists alongside the .bin
        stem = os.path.splitext(files["binFile"])[0]
        meta_path = os.path.join(files["binPath"], stem + ".meta")
        if not os.path.isfile(meta_path):
            missing.append(
                f"SpikeGLX metadata file  ({stem}.meta)"
            )

    # ------------------------------------------------------------------
    # Kilosort output directory — located by spike_times.npy
    # ------------------------------------------------------------------
    st_hits = _find_files(exp_dir, "spike_times.npy")

    if not st_hits:
        missing.append("Kilosort output  (spike_times.npy not found)")
    else:
        # Multiple hits → pick the directory, not the file
        ks_dirs = [os.path.dirname(p) for p in st_hits]
        if len(ks_dirs) > 1:
            warnings.warn(
                f"Multiple Kilosort output directories found; using the first:\n"
                + "\n".join(f"  [{i}] {d}" for i, d in enumerate(ks_dirs))
                + f"\nFirst selected: {ks_dirs[0]}",
                UserWarning,
                stacklevel=2,
            )
        files["ksDir"] = ks_dirs[0]

        # Verify the remaining required Kilosort files are present
        for fname in _KS_REQUIRED[1:]:   # spike_times.npy already confirmed
            fpath = os.path.join(files["ksDir"], fname)
            if not os.path.isfile(fpath):
                missing.append(f"Kilosort output file  ({fname})")

    # ------------------------------------------------------------------
    # Optional strict mode — raise immediately on missing files
    # ------------------------------------------------------------------
    if strict and missing:
        raise FileNotFoundError(
            "The following required files were not found in "
            f"{exp_dir!r}:\n  "
            + "\n  ".join(missing)
        )

    return files, missing


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_files(root: str, pattern: str) -> list[str]:
    """Recursively find all files matching *pattern* under *root*.

    Directories are excluded; only regular files are returned.  Results
    are sorted for deterministic ordering.

    Mirrors MATLAB ``dir(fullfile(root, '**', pattern))`` with
    ``~[hits.isdir]`` filtering.
    """
    matches = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    return sorted(p for p in matches if os.path.isfile(p))


def _pick_one(paths: list[str], description: str) -> str:
    """Return the single path from *paths*, or warn and return the first.

    When multiple candidates are found this function issues a
    :class:`UserWarning` and returns ``paths[0]`` — matching the MATLAB
    ``listdlg`` fallback (``chosenIdx = 1`` when the user cancels).
    """
    if len(paths) == 1:
        return paths[0]

    warnings.warn(
        f"Multiple {description} files found; using the first match.\n"
        + "\n".join(f"  [{i}] {p}" for i, p in enumerate(paths))
        + f"\nFirst selected: {paths[0]}\n"
        "Pass the desired path explicitly to avoid ambiguity.",
        UserWarning,
        stacklevel=3,
    )
    return paths[0]

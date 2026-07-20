"""
or_read_meta.py  —  Parse a SpikeGLX .meta file into a Python dict.

FILE FORMAT
-----------
SpikeGLX .meta files contain ASCII key=value pairs, one per line, with
CRLF line endings.  Some keys are prefixed with a tilde (``~``) which
SpikeGLX uses to mark certain fields; the tilde is stripped so callers
see a clean key name.

MATLAB equivalence
------------------
MATLAB code uses ``textscan(fid, '%[^=] = %[^\\r\\n]')`` which:
* Reads everything before ``=`` as the key (with optional surrounding
  whitespace handled by MATLAB's whitespace-collapsing rules).
* Reads everything after ``=`` until CR or LF as the value.
* Whitespace is stripped from both via ``strtrim()``.

Python equivalent: open the file, iterate lines, split on the *first*
``=``, strip whitespace from key and value.  This handles both
``key=value`` (no spaces, typical SpikeGLX format) and ``key = value``.
"""

from __future__ import annotations

import os


def or_read_meta(bin_name: str, path: str) -> dict:
    """Parse the SpikeGLX .meta file that accompanies a .bin recording.

    The .meta filename is derived from *bin_name* by replacing the last
    extension with ``.meta``::

        'session_g0_t0.imec0.ap.bin'  →  'session_g0_t0.imec0.ap.meta'

    This mirrors MATLAB's ``fileparts()`` behaviour, which strips only the
    *final* extension.

    Parameters
    ----------
    bin_name : str
        Filename of the .bin recording
        (e.g. ``'session_g0_t0.imec0.ap.bin'``).
    path : str
        Folder containing both the .bin and .meta files.

    Returns
    -------
    meta : dict[str, str]
        All key–value pairs from the .meta file.  Every value is a
        ``str``; callers must convert to numeric types as needed
        (e.g. ``float(meta['imSampRate'])``).

        Commonly used keys:

        ``'typeThis'``
            ``'imec'`` or ``'nidq'``.
        ``'imSampRate'``
            AP sampling rate as a string (e.g. ``'30000'``).
        ``'nSavedChans'``
            Total saved channels including sync channel.
        ``'fileSizeBytes'``
            Binary file size in bytes.
        ``'snsApLfSy'``
            Channel-count string, e.g. ``'384,32,1'``
            (nAP, nLF, nSY).
        ``'imroTbl'``
            IMRO table string for per-channel gain look-up.

    Raises
    ------
    FileNotFoundError
        If the derived .meta file does not exist.

    Examples
    --------
    >>> meta = or_read_meta('session.ap.bin', '/data/2024-01-15/imec0')
    >>> fs = float(meta['imSampRate'])   # 30000.0
    """
    # Derive .meta filename: strip the final extension and append '.meta'.
    # MATLAB fileparts('session.ap.bin') → stem = 'session.ap'
    stem = os.path.splitext(bin_name)[0]   # 'session.ap.bin' → 'session.ap'
    meta_name = stem + ".meta"
    meta_full_path = os.path.join(path, meta_name)

    if not os.path.isfile(meta_full_path):
        raise FileNotFoundError(
            f"Metadata file not found:\n  {meta_full_path}\n"
            "Ensure the .meta file is in the same folder as the .bin file."
        )

    meta: dict[str, str] = {}
    with open(meta_full_path, "r", encoding="utf-8", errors="replace") as fid:
        for line in fid:
            # Strip CR/LF
            line = line.rstrip("\r\n")
            if "=" not in line:
                continue  # skip blank lines or malformed lines

            # Split on the FIRST '=' only (values may contain '=')
            eq_idx = line.index("=")
            tag   = line[:eq_idx].strip()
            value = line[eq_idx + 1:].strip()

            # Strip leading tilde used by SpikeGLX on some field names
            # MATLAB: if tag(1) == '~'; tag = tag(2:end); end
            if tag.startswith("~"):
                tag = tag[1:]

            if tag:  # ignore entries with an empty key
                meta[tag] = value

    return meta

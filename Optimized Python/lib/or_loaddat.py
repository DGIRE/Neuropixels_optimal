"""
or_loaddat.py  —  Read a LabView .dat sensor file into NumPy arrays.

FILE FORMAT
-----------
The .dat file is big-endian int32 binary written by LabView.  Every record
is delimited by the sentinel value -10.  Relevant fields at fixed offsets
*from* the sentinel position (0-indexed from sentinel):

    +1  : TR  — trial number
    +2  : TS  — within-trial timestamp in ms (resets at each trial start)
    +3  : FR  — camera frame number (0–875, resets each trial)
    +4  : ETH — ethanol sensor raw counts
    +11 : SNF — sniffing cannula raw counts

Translation notes
-----------------
* MATLAB fread('int32', 'ieee-be')  →  np.fromfile(dtype='>i4')
* MATLAB find() returns 1-based indices; Python np.where() returns 0-based.
  n_ten(20:end) in MATLAB keeps from the 20th element inclusive, which is
  index 19 in 0-based Python → n_ten[19:].
* MATLAB data(n_ten + k) uses 1-based indexing on both n_ten and data.
  After converting n_ten to 0-based, the offsets (+1 … +11) remain the same
  because they are *relative* offsets, not absolute positions.
"""

from __future__ import annotations

import os
import warnings

import numpy as np


def or_loaddat(
    fn: str,
    fp: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Read a LabView .dat sensor file.

    Parameters
    ----------
    fn : str
        Filename of the .dat file (e.g. ``'session.dat'``).
    fp : str
        Folder containing the file.

    Returns
    -------
    TR : np.ndarray, shape (n,), dtype int32
        Trial number for each sample, as labeled by LabView.
    TS : np.ndarray, shape (n,), dtype int32
        Within-trial timestamp in milliseconds; resets at each trial start.
    FR : np.ndarray, shape (n,), dtype int32
        Camera frame number (0–875); resets each trial.  A change of 1 in FR
        indicates a TTL pulse sent to Neuropixels for time alignment.
    ETH : np.ndarray, shape (n,), dtype float64
        Ethanol sensor time series, normalised to [0, 1].
    SNF : np.ndarray, shape (n,), dtype float64
        Sniffing cannula pressure time series.
    Fs : float
        LabView sampling rate in Hz (computed from TS; typically ~125 Hz).

    Raises
    ------
    FileNotFoundError
        If the file cannot be opened.
    ValueError
        If fewer than 21 sentinel values are found (corrupt or wrong file).
    """
    full_path = os.path.join(fp, fn)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Cannot open file: {full_path}")

    # Read entire file as big-endian int32  (MATLAB: fread(fid, 'int32', 'ieee-be'))
    data = np.fromfile(full_path, dtype=">i4")

    # Locate sentinel records (value == -10)
    # MATLAB n_ten = find(data == -10)  →  np.where returns 0-based indices
    n_ten = np.where(data == -10)[0]

    if n_ten.size < 21:
        raise ValueError(
            f"File does not appear to contain valid LabView records "
            f"(found {n_ten.size} sentinel(s), need ≥ 21):\n  {full_path}"
        )

    # Skip first 19 sentinels (header artefacts).
    # MATLAB: n_ten = n_ten(20:end)  →  1-based index 20 = 0-based index 19
    n_ten = n_ten[19:]

    # Guard: drop sentinel positions that would cause the +11 offset to fall
    # outside the array boundary.
    max_valid = len(data) - 12  # need data[n_ten + 11] to be in-bounds
    n_ten = n_ten[n_ten <= max_valid]

    if n_ten.size == 0:
        raise ValueError(
            f"No valid records remain after boundary check:\n  {full_path}"
        )

    # Extract fields at fixed offsets from each sentinel.
    # Offsets are identical to MATLAB because they are *relative* to n_ten;
    # the 0-based vs 1-based difference cancels out for relative offsets.
    TR  = data[n_ten + 1].astype(np.int32)   # trial number
    TS  = data[n_ten + 2].astype(np.int32)   # within-trial timestamp (ms)
    FR  = data[n_ten + 3].astype(np.int32)   # camera frame number
    ETH = data[n_ten + 4].astype(np.int32)   # ethanol sensor raw counts
    SNF = data[n_ten + 11].astype(np.float64)  # sniffing sensor

    # --- Compute sampling rate from timestamps ---
    # Take median of *positive* intervals to exclude inter-trial-interval jumps.
    # MATLAB: dTS = diff(TS); dTS(dTS <= 0) = []; Fs = 1000 / median(dTS)
    dTS = np.diff(TS.astype(np.float64))
    dTS = dTS[dTS > 0]
    if dTS.size == 0:
        raise ValueError(
            "Cannot compute sampling rate: no valid timestamp intervals found."
        )
    Fs = 1000.0 / float(np.median(dTS))

    # --- Normalise ETH to [0, 1] ---
    ETH = ETH.astype(np.float64)
    ETH -= ETH.min()
    eth_max = ETH.max()
    if eth_max > 0.0:
        ETH /= eth_max

    return TR, TS, FR, ETH, SNF, Fs

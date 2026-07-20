"""
or_read_bin.py  —  Read a chunk of raw data from a SpikeGLX .bin file.

BINARY FORMAT
-------------
SpikeGLX writes little-endian int16 binary in *interleaved channel* order:

    [ch0_s0, ch1_s0, …, chN_s0,  ch0_s1, ch1_s1, …, chN_s1, …]

i.e. all channels for sample 0, then all channels for sample 1, etc.

MATLAB equivalence
------------------
MATLAB ``fread(fid, [nChan, nSamp], 'int16=>double')`` fills the result
matrix *column-major* (Fortran order).  The first column (sample 0) is
filled with the first nChan values from the file, which are ch0…chN for
sample 0.  This matches the interleaved layout perfectly.

Python equivalent (two equivalent forms)::

    raw = np.fromfile(f, dtype='<i2', count=nChan * nSamp)
    data = raw.reshape((nChan, nSamp), order='F')   # Fortran/column-major

or equivalently::

    data = raw.reshape((nSamp, nChan)).T             # transpose trick

Both yield a (nChan, nSamp) array with rows=channels, cols=samples —
identical to the MATLAB result.

Byte order note
---------------
SpikeGLX runs on Windows x86/x64 and always writes little-endian int16.
We use dtype='<i2' (explicit little-endian) rather than the native default
so the code is portable to big-endian platforms.  MATLAB's bare 'int16'
implicitly uses native byte order, which is little-endian on all supported
platforms, so the semantics are equivalent.
"""

from __future__ import annotations

import os
import warnings

import numpy as np


def or_read_bin(
    samp0: int,
    n_samp: int,
    meta: dict,
    bin_name: str,
    path: str,
) -> np.ndarray:
    """Read a chunk of raw int16 data from a SpikeGLX .bin file.

    Parameters
    ----------
    samp0 : int
        First sample to read (0-based index).
    n_samp : int
        Number of samples to read.
    meta : dict
        Metadata dict returned by :func:`or_read_meta`.  Must contain at
        least ``'nSavedChans'`` and ``'fileSizeBytes'``.
    bin_name : str
        .bin filename (e.g. ``'session_g0_t0.imec0.ap.bin'``).
    path : str
        Folder containing the .bin file.

    Returns
    -------
    data_array : np.ndarray, shape (nChannels, nSamp), dtype float64
        Raw int16 values stored as float64.  Rows correspond to saved
        channels in the order they appear in the file; the last row is
        the sync channel for imec recordings.

    Raises
    ------
    FileNotFoundError
        If the .bin file cannot be opened.

    Warnings
    --------
    UserWarning
        If the requested ``samp0`` is beyond the end of the file.

    Notes
    -----
    * ``samp0`` and ``n_samp`` are silently clamped: ``samp0`` is floored
      and clipped to ≥ 0; ``n_samp`` is clipped so reading stops at the
      last available sample.
    * To convert raw counts to microvolts for imec AP channels::

          fI2V     = float(meta['imAiRangeMax']) / 512
          gain_AP  = <per-channel gain from IMRO table>
          uV       = data_array[ch, :] * fI2V / gain_AP * 1e6
    """
    n_chan      = int(float(meta["nSavedChans"]))
    n_file_samp = int(float(meta["fileSizeBytes"])) // (2 * n_chan)

    samp0  = max(int(samp0), 0)
    n_samp = min(int(n_samp), n_file_samp - samp0)

    if n_samp <= 0:
        warnings.warn(
            f"Requested range (samp0={samp0}) is at or beyond the end of the "
            f"file ({n_file_samp} samples).  Returning empty array.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros((n_chan, 0), dtype=np.float64)

    bin_full_path = os.path.join(path, bin_name)
    if not os.path.isfile(bin_full_path):
        raise FileNotFoundError(
            f"Cannot open binary file:\n  {bin_full_path}"
        )

    with open(bin_full_path, "rb") as fid:
        # Seek to byte offset of the first requested sample.
        # MATLAB: fseek(fid, samp0 * 2 * nChan, 'bof')
        # Each sample has nChan channels, each stored as 2 bytes (int16).
        fid.seek(samp0 * 2 * n_chan)

        # Read nChan * n_samp little-endian int16 values.
        # MATLAB: fread(fid, [nChan, nSamp], 'int16=>double')
        raw = np.fromfile(fid, dtype="<i2", count=n_chan * n_samp)

    # Reshape to (nChan, nSamp) using Fortran (column-major) order so that
    # the first n_chan values become column 0 (sample 0, all channels) —
    # exactly matching MATLAB's column-major fread result.
    # The order='F' reshape produces a Fortran-contiguous view; astype with
    # default order='K' preserves that layout.  Convert to C-contiguous so
    # that downstream scipy calls (filtfilt axis=1, etc.) operate on
    # row-major memory without an internal copy on every call.
    data_array = raw.reshape((n_chan, n_samp), order="F").astype(np.float64)
    data_array = np.ascontiguousarray(data_array)

    return data_array

"""
circular_shift_null(lfp, snf, window_starts, n_shifts=1000, seed=5489,
                    fs=1000.0, valid_length=None)

NP-PAPER-1 spectral engine, function 7/9 (contract_v001.yaml aggregation step;
RQ-024 / RQ-038 / RQ-041 / CON-003 -- >=1000 circular shifts, seed 5489
pinned, 95th-percentile-per-frequency threshold, BIT-FOR-BIT reproducible).

Independently authored implementation -- written from the contract text
directly, not copied from `oracles/spectral_oracle_ref.py::
reference_circular_shift_null` (that reference exists so this implementation
can be checked against it independently).

Procedure (pinned, do not tune):
  - Draw `n_shifts` (>=1000 in production) nonzero circular shifts of SNF
    relative to LFP from a DEDICATED `numpy.random.default_rng(seed)` --
    never numpy's global RNG state (RQ-038/H8: must be immune to any other
    code's calls to np.random.* / np.random.seed).
  - For each shift, roll SNF circularly by that amount and recompute the
    windowed/pooled multitaper coherence via THIS package's own
    `multitaper_psd_coherence` (never re-derived independently here -- the
    contract requires the null to use the same coherence engine as the
    observed statistic).
  - The per-frequency 95th percentile of the null distribution is the
    threshold. The observed (unshifted) coherence is also returned for the
    significance decision (RESULT-theta-coh).
  - Bit-for-bit reproducible for a fixed seed: the RNG draw order (one
    `rng.integers(1, n)` draw per shift, in a strict for-loop over shift
    index) is identical across runs and independent of any other numpy RNG
    activity, because a fresh private Generator is created every call.

FIX (scientific-audit MINOR-2, approved by David Gire 2026-07-25, see
08_traceability/deviations.yaml): `resample_snf_to_lfp` may append a
held-constant (zero-variance) pad to the tail of `snf` on sessions where the
chosen full-duration LFP is longer than the independent LabView recording
(DEV-001 Resolution B). Rolling the FULL array (pad included) can rotate that
constant block into the middle of the recording for some shifts, injecting a
spurious zero-variance segment into analysis windows and biasing the null
threshold DOWN (anti-conservative). `valid_length` confines the circular
shift to the first `valid_length` samples of `snf` (the real, non-padded
region) -- the padded tail (if any) is left untouched at its original
position for every shift, so no shift can ever inject padded values into a
different part of the recording. When `valid_length` is None (default) or
equals `len(snf)` (no padding), behavior is UNCHANGED and bit-for-bit
identical to before this fix (all existing oracle tests, which never pass
padded signals, are unaffected).
"""
from __future__ import annotations

import numpy as np

from multitaper_psd_coherence import multitaper_psd_coherence


def circular_shift_null(lfp, snf, window_starts, n_shifts=1000, seed=5489,
                        fs=1000.0, valid_length=None):
    """Circular-shift null distribution + 95th-percentile threshold for the
    LFP-SNF multitaper coherence.

    Parameters
    ----------
    lfp, snf : 1-D array_like, same length, shared LFP-rate clock.
    window_starts : 1-D int array_like of window START sample indices
        (same convention as `multitaper_psd_coherence`).
    n_shifts : number of circular shifts (pinned production value >= 1000,
        RQ-041/CON-003). Reduced values are permitted only for test
        calibration, never for the analysis configuration.
    seed : dedicated-RNG seed (pinned 5489, RQ-038/CON-003).
    fs : sampling rate (Hz), default 1000.0.
    valid_length : int, optional. Number of leading `snf` samples that are
        real (non-padded); the circular shift is confined to this prefix and
        the remainder (if any) is left fixed at the tail for every shift.
        Defaults to `len(snf)` (whole array, i.e. no padding / prior
        behavior) when None or when equal to `len(snf)`.

    Returns
    -------
    dict with keys: freqs, observed, null_threshold, null_distribution,
    n_shifts, seed, valid_length.
    """
    lfp = np.asarray(lfp, dtype=float)
    snf = np.asarray(snf, dtype=float)
    n_samples = snf.size

    if valid_length is None:
        valid_length = n_samples
    valid_length = int(valid_length)
    if not (0 < valid_length <= n_samples):
        raise ValueError(
            f"valid_length must be in (0, len(snf)]; got {valid_length} "
            f"for len(snf)={n_samples}"
        )

    # Dedicated private Generator -- NEVER numpy's global RNG (RQ-038/H8).
    rng = np.random.default_rng(seed)

    observed = multitaper_psd_coherence(lfp, snf, window_starts, fs=fs)
    freqs = observed["freqs"]
    n_freq = freqs.size

    null_distribution = np.empty((n_shifts, n_freq), dtype=float)
    for i in range(n_shifts):
        shift = int(rng.integers(1, valid_length))  # nonzero shift within the valid region
        snf_shifted = snf.copy()
        # Confine the roll to the valid (non-padded) prefix (MINOR-2 fix):
        # the padded tail, if any, never moves and is never rotated into a
        # different part of the recording. When valid_length == n_samples
        # this is bit-for-bit identical to `np.roll(snf, shift)`.
        snf_shifted[:valid_length] = np.roll(snf[:valid_length], shift)
        shifted_result = multitaper_psd_coherence(
            lfp, snf_shifted, window_starts, fs=fs)
        null_distribution[i] = shifted_result["coherence"]

    null_threshold = np.percentile(null_distribution, 95.0, axis=0)

    return {
        "freqs": freqs,
        "observed": observed["coherence"],
        "null_threshold": null_threshold,
        "null_distribution": null_distribution,
        "n_shifts": n_shifts,
        "seed": seed,
        "valid_length": valid_length,
    }

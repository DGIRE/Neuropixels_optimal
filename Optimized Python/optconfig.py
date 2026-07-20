"""
optconfig.py  —  Optimization switches for the Neuropixels Python pipeline.

This module is the SINGLE place that turns optimized code paths on or off.
It is kept deliberately separate from any data/config object so that toggling
optimizations can never perturb a value a golden fixture pins (blueprint R13).

Every optimization in this package is gated by one boolean here and preserves
a byte-identical (or, where noted, ground-truth-faithful within the fixture
tolerance) baseline fallback.  `set_baseline()` is the one-switch revert to the
original, un-optimized behavior (blueprint R7); `set_optimized()` restores the
fast paths.

Loader optimizations
---------------------
OPT.vectorized_unit_aggregation  (load_experiment_data.py)
    Replace the per-unit Python loop (O(nUnits x nSpikes): a full boolean scan
    of the spike arrays for every unit) with a single grouped reduction via
    np.bincount (O(nSpikes)).  ~60x faster on the aggregation stage.
    ACCURACY: Level 2 (numeric).  unitIDs / unitFiringRate are bit-identical to
    the baseline; unitDepths / unitAmps are grouped means accumulated in float64
    rather than the baseline's float32 loop -- they match the GOLDEN FIXTURES
    within the single-precision-mean tolerance (rtol 1e-5) and are in fact
    marginally CLOSER to the MATLAB ground truth than the float32 baseline
    (blueprint D2: anchor to ground truth, not the port).

OPT.fast_noise_exclusion  (lib/ks_utils.py)
    Replace `~np.isin(clu, noise_list)` (O(nSpikes log nSpikes)) with a boolean
    lookup table indexed by cluster id (O(nSpikes)).  ~11x faster on the mask.
    ACCURACY: Level 1 (byte-identical mask).

Analysis optimizations  (analyses/ ; extension stages 08-11 + decomposition)
-----------------------------------------------------------------------------
OPT.vectorized_spike_phase  (analyses/compute_spike_phase.py, stage 10)
    Replace the per-unit mean loop (the same O(nUnits x nSpikes) boolean-scan
    trap as E1) with one np.bincount grouped reduction over valid spike phases.
    ACCURACY: Level 2 (numeric).  Means match the golden fixture at tight
    tolerance (1e-12) and the baseline loop; NaN where a unit has < 5 valid
    spikes, exactly as MATLAB.

OPT.vectorized_psth  (analyses/compute_sniff_psth.py, stage 11)
    Replace the per-unit psthAndBA loop (nUnits * nEvents mex-equivalent
    histogram passes) with one global spike->event expansion computing every
    unit's event-averaged PSTH at once.  Identical ordhist boundary semantics.
    ACCURACY: Level 2 (numeric).  Matches the golden fixture (rtol 1e-9) and the
    faithful per-unit baseline.

OPT.vectorized_rate_matrix  (analyses/run_decomposition.py, run_ica.py)
    Build the Gaussian-convolved spike-rate matrix for all units in one batched
    real-FFT instead of a per-unit FFT loop.  ACCURACY: Level 2 (numeric); the
    two paths agree to FFT round-off.  (These analyses are stochastic and have
    no golden fixture; the rate matrix itself is deterministic and is validated
    optimized-vs-baseline.)
"""

from __future__ import annotations


class _OptConfig:
    """Mutable switch holder.  Import the singleton `OPT` and read its flags."""

    __slots__ = (
        "vectorized_unit_aggregation",
        "fast_noise_exclusion",
        "vectorized_spike_phase",
        "vectorized_psth",
        "vectorized_rate_matrix",
    )

    def __init__(self) -> None:
        # Optimized defaults (all fast paths on).
        self.vectorized_unit_aggregation: bool = True
        self.fast_noise_exclusion: bool = True
        self.vectorized_spike_phase: bool = True
        self.vectorized_psth: bool = True
        self.vectorized_rate_matrix: bool = True

    def as_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


# Singleton read by the pipeline modules.
OPT = _OptConfig()


def set_baseline() -> None:
    """One-switch revert: turn EVERY optimization off (original behavior)."""
    OPT.vectorized_unit_aggregation = False
    OPT.fast_noise_exclusion = False
    OPT.vectorized_spike_phase = False
    OPT.vectorized_psth = False
    OPT.vectorized_rate_matrix = False


def set_optimized() -> None:
    """Restore all optimized code paths (the package default)."""
    OPT.vectorized_unit_aggregation = True
    OPT.fast_noise_exclusion = True
    OPT.vectorized_spike_phase = True
    OPT.vectorized_psth = True
    OPT.vectorized_rate_matrix = True

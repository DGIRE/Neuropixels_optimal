"""
analyses — Neuropixels "new analyses" ports (extension of the Optimized Python
pipeline).  Each analysis is a function that takes the loader's D dict, reads
the fields it needs, and returns a copy of D with new fields added (never
mutating existing arrays).  Fixture-anchored stages 08-11 validate against the
golden fixtures; the decomposition/ICA stages are stochastic (no fixture).

    from analyses import (compute_sniff_phase, threshold_eth,
                          compute_spike_phase, compute_sniff_psth,
                          run_ica, run_decomposition)
"""
from .compute_sniff_phase import compute_sniff_phase
from .threshold_eth import threshold_eth
from .compute_spike_phase import compute_spike_phase
from .compute_sniff_psth import compute_sniff_psth
from .run_ica import run_ica
from .run_decomposition import run_decomposition

__all__ = [
    "compute_sniff_phase",
    "threshold_eth",
    "compute_spike_phase",
    "compute_sniff_psth",
    "run_ica",
    "run_decomposition",
]

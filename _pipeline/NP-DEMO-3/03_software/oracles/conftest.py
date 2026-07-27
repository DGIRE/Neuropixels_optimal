# ORACLE
"""
Shared fixtures and reference (ground-truth) computations for the NP-DEMO-3
oracle test suite.

IMPORTANT (Blueprint v2 P3 -- oracle design phase):
  These reference functions are derived DIRECTLY from the NP-DEMO-3 contract
  (02_contract/contract_v001.yaml) and the scientific/statistical first
  principles named in the task brief. They are NOT copies of, nor derived
  from running, the not-yet-written analysis code
  (03_software/np_demo3_analysis.py) or the not-yet-written kernel
  (03_software/detect_eth_contact.py). They exist so that:

    1. Right now (pre-implementation), the oracle test files can assert
       genuine mathematical/statistical properties and pass on their own.
    2. Later, once 03_software/detect_eth_contact.py exists, the
       `eth_contact_oracle` fixture below will automatically import and use
       the REAL implementation instead of the fallback reference, so the
       exact same test bodies serve as the acceptance gate for
       RESULT-eth-mask (contract acceptance level 1, exact match).

  No expected value in this suite was produced by executing the pipeline
  under design. All expected values are either closed-form (mean of N
  identical values, sum of Nth roots of unity, exact permutation
  distributions of the Wilcoxon signed-rank statistic, etc.) or delegate to
  the third-party libraries the contract itself names as the reference
  procedure (scipy.stats.wilcoxon, statsmodels MixedLM) -- consulting those
  libraries directly is equivalent to consulting a MATLAB/golden reference,
  not "running the new code".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SOFTWARE_DIR = Path(__file__).resolve().parents[1]  # .../NP-DEMO-3/03_software


# ---------------------------------------------------------------------------
# detect_eth_contact reference (contract_v001.yaml: new_kernel_functions_required)
# ---------------------------------------------------------------------------

def count_contiguous_runs(mask: np.ndarray) -> int:
    """Count discrete contiguous True runs in a 1-D boolean array (first principles)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    return int(starts.size)


def reference_detect_eth_contact(eth: np.ndarray, eth_threshold: float = 0.05):
    """Ground-truth reimplementation of the detect_eth_contact algorithm, verbatim
    from contract_v001.yaml (new_kernel_functions_required.algorithm):
      1. ETH_ms = ETH - mean(ETH)  (mean of the ENTIRE trace, no windowing)
      2. eth_contact_mask = ETH_ms > eth_threshold  (STRICTLY above)
      3. n_trials = count of discrete contiguous True runs in eth_contact_mask
    """
    eth = np.asarray(eth, dtype=float)
    eth_ms = eth - np.mean(eth)
    mask = eth_ms > eth_threshold
    n_trials = count_contiguous_runs(mask)
    return eth_ms, mask, n_trials


def _load_real_detect_eth_contact():
    """Try to import the real detect_eth_contact() from 03_software/. Returns
    None if it does not exist yet (expected during oracle-design phase)."""
    software_dir_str = str(SOFTWARE_DIR)
    if software_dir_str not in sys.path:
        sys.path.insert(0, software_dir_str)
    try:
        from detect_eth_contact import detect_eth_contact  # type: ignore
    except ImportError:
        return None
    return detect_eth_contact


@pytest.fixture
def eth_contact_oracle():
    """Callable(eth_array, eth_threshold=0.05) -> (eth_ms, mask, n_trials).

    Uses the REAL detect_eth_contact implementation if 03_software/detect_eth_contact.py
    exists; otherwise falls back to the contract-derived reference above. Either
    way, the expected values asserted in the test bodies come from first
    principles / the contract wording, never from executing this fixture.
    """
    real_fn = _load_real_detect_eth_contact()

    def _call(eth_array, eth_threshold: float = 0.05):
        if real_fn is not None:
            D = {"ETH": np.asarray(eth_array, dtype=float)}
            out = real_fn(D, eth_threshold=eth_threshold)
            return (
                np.asarray(out["ETH_ms"]),
                np.asarray(out["eth_contact_mask"], dtype=bool),
                int(out["n_trials"]),
            )
        return reference_detect_eth_contact(eth_array, eth_threshold)

    return _call


@pytest.fixture
def simple_eth_D():
    """Factory fixture: build a minimal D-dict containing an ETH array."""

    def _make(eth_values):
        return {"ETH": np.asarray(eth_values, dtype=float)}

    return _make


# ---------------------------------------------------------------------------
# MRL (mean resultant length) reference -- circular statistics, first principles
# ---------------------------------------------------------------------------

def compute_mrl(phases: np.ndarray):
    """Ground-truth MRL / preferred-phase formula (contract-specified definition):
      MRL = |mean(exp(2*pi*i*phase))|,  phase in [0, 1)
      preferred_phase = angle(mean vector) / (2*pi), wrapped into [0, 1)
    Excludes the -1 sentinel (invalid-sniff-cycle) values, mirroring the
    spike_SNF_PH >= 0 gate required by the contract.
    """
    phases = np.asarray(phases, dtype=float)
    phases = phases[phases >= 0]
    if phases.size == 0:
        return np.nan, np.nan
    vectors = np.exp(2j * np.pi * phases)
    mean_vector = np.mean(vectors)
    mrl = float(np.abs(mean_vector))
    preferred_phase = float((np.angle(mean_vector) / (2 * np.pi)) % 1.0)
    return mrl, preferred_phase


@pytest.fixture
def synthetic_spikes():
    """Factory fixture: synthetic_spikes(phases, n_invalid=0, lv_fs=125.0, seed=5489)
    -> 1-D np.ndarray of spike_SNF_PH-like values.

    `phases` are the "valid" phase-in-cycle values (>= 0, < 1) to embed.
    `n_invalid` extra -1 sentinel entries (outside-valid-sniff-cycle spikes)
    are appended and shuffled in, so tests can verify sentinel-filtering
    behaviour realistically. `lv_fs` is accepted for interface parity with
    the real pipeline's sampling-rate bookkeeping but is not required by the
    MRL formula itself (MRL operates on phase-in-cycle, not on sample index).
    """

    def _make(phases, n_invalid: int = 0, lv_fs: float = 125.0, seed: int = 5489):
        phases = np.asarray(phases, dtype=float)
        invalid = np.full(n_invalid, -1.0)
        combined = np.concatenate([phases, invalid])
        rng = np.random.default_rng(seed)
        rng.shuffle(combined)
        return combined

    return _make


# ---------------------------------------------------------------------------
# Mock per-unit / per-experiment result tables for statistics oracle tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_experiment_results():
    """Factory fixture: mock_experiment_results(n_units, n_experiments, ...) ->
    pandas.DataFrame with columns [unit_id, experiment, MRL_ethanol, MRL_control].

    Units are distributed round-robin across experiments (experiment_0 ..
    experiment_{n_experiments-1}) so downstream group-by / random-intercept
    logic has non-trivial nesting structure to check against.
    """

    def _make(
        n_units: int,
        n_experiments: int,
        ethanol_offset: float = 0.0,
        noise_sd: float = 0.0,
        seed: int = 5489,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        unit_ids = np.arange(n_units)
        experiments = np.array([f"experiment_{i % n_experiments}" for i in range(n_units)])
        base = rng.uniform(0.1, 0.4, size=n_units)
        mrl_control = np.clip(base + rng.normal(0, noise_sd, size=n_units), 0.0, 1.0)
        mrl_ethanol = np.clip(
            base + ethanol_offset + rng.normal(0, noise_sd, size=n_units), 0.0, 1.0
        )
        return pd.DataFrame(
            {
                "unit_id": unit_ids,
                "experiment": experiments,
                "MRL_ethanol": mrl_ethanol,
                "MRL_control": mrl_control,
            }
        )

    return _make

# ORACLE
"""
Tests for mean resultant length (MRL) and preferred phase computation.

Targets:
  - RESULT-mrl-per-unit (contract acceptance level 3, distributional;
    "MRL and preferred phase within 1e-6 on fixture slice").
  - RESULT-mrl-per-experiment (level 3, derived per-experiment means).

Definition under test (contract + task brief, first principles):
  MRL = |mean(exp(2*pi*i*phase))|,  phase in [0, 1) (fraction of sniff cycle)
  preferred_phase = angle(mean vector) / (2*pi), wrapped to [0, 1)
  Only spike_SNF_PH >= 0 (valid sniff cycle) samples are ever included.

Every expected value below is either an exact closed-form result (sum of Nth
roots of unity, symmetric two-cluster cancellation) or a statistically
calibrated check against a known circular distribution (von Mises), computed
independently of any pipeline implementation via conftest.compute_mrl, which
is a direct transcription of the formula above -- not a copy of, nor derived
from running, the not-yet-written analysis code.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import vonmises

from conftest import compute_mrl


def test_perfectly_synchronized_spikes_give_mrl_one(synthetic_spikes):
    """All spikes at phase 0.0 -> MRL == 1.0 exactly, preferred_phase == 0.0."""
    phases = synthetic_spikes(np.zeros(100))
    mrl, preferred_phase = compute_mrl(phases)

    assert mrl == pytest.approx(1.0, abs=1e-12)
    assert preferred_phase == pytest.approx(0.0, abs=1e-12)


def test_uniformly_distributed_spikes_give_near_zero_mrl(synthetic_spikes):
    """N=360 spikes evenly spaced over [0,1) sum to (near) exactly zero -- the Nth roots of unity sum to 0."""
    n = 360
    phases = synthetic_spikes(np.arange(n) / n)
    mrl, _ = compute_mrl(phases)

    # Exact in real arithmetic (sum_{k=0}^{N-1} exp(2*pi*i*k/N) == 0); only
    # floating-point summation error remains, which is O(N * eps) ~ 1e-14.
    assert mrl < 1e-9


def test_two_opposite_clusters_cancel_to_near_zero_mrl(synthetic_spikes):
    """Half the spikes at phase 0.0, half at phase 0.5 (opposite unit vectors) -> MRL ~ 0 exactly."""
    phases = synthetic_spikes(np.concatenate([np.zeros(50), np.full(50, 0.5)]))
    mrl, _ = compute_mrl(phases)

    # exp(0) + exp(i*pi) == 1 + (-1) == 0 exactly (up to float rounding).
    assert mrl < 1e-9


def test_single_von_mises_cluster_gives_intermediate_mrl_near_true_mean(synthetic_spikes):
    """Spikes drawn from von Mises(mu=0.25 cycles, kappa=4) give MRL in (0,1) and preferred_phase near 0.25 (statistical calibration)."""
    true_mu_cycles = 0.25
    rng = np.random.default_rng(5489)  # contract randomization_seed
    samples_rad = vonmises.rvs(kappa=4, loc=true_mu_cycles * 2 * np.pi, size=2000, random_state=rng)
    phases_cycles = (samples_rad / (2 * np.pi)) % 1.0
    phases = synthetic_spikes(phases_cycles)

    mrl, preferred_phase = compute_mrl(phases)

    assert 0.0 < mrl < 1.0
    # Circular distance to account for phase wraparound near 0/1.
    circular_diff = min(abs(preferred_phase - true_mu_cycles), 1 - abs(preferred_phase - true_mu_cycles))
    # With N=2000 and kappa=4, the circular SEM is small; 0.05 cycles (~18 deg)
    # is a very conservative (>>5 sigma) bound for this sample size.
    assert circular_diff < 0.05


def test_four_spike_symmetric_quarters_give_exact_zero_mrl(synthetic_spikes):
    """4 spikes at phases [0, 0.25, 0.5, 0.75] (4th roots of unity) sum to exactly 0 -> MRL == 0.0."""
    phases = synthetic_spikes(np.array([0.0, 0.25, 0.5, 0.75]))
    mrl, _ = compute_mrl(phases)

    assert mrl == pytest.approx(0.0, abs=1e-9)


def test_condition_agnostic_mrl_can_be_lower_than_either_condition_alone(synthetic_spikes):
    """MRL_cond_agnostic >= min(MRL_eth, MRL_ctrl) is NOT guaranteed: opposite-phase clusters demonstrate a counterexample."""
    eth_phases = synthetic_spikes(np.zeros(50))
    ctrl_phases = synthetic_spikes(np.full(50, 0.5))
    pooled_phases = np.concatenate([eth_phases, ctrl_phases])

    mrl_eth, _ = compute_mrl(eth_phases)
    mrl_ctrl, _ = compute_mrl(ctrl_phases)
    mrl_pooled, _ = compute_mrl(pooled_phases)

    assert mrl_eth == pytest.approx(1.0, abs=1e-12)
    assert mrl_ctrl == pytest.approx(1.0, abs=1e-12)
    # Pooling perfectly-locked-but-opposite-phase conditions collapses MRL to ~0,
    # which is strictly LESS than min(MRL_eth, MRL_ctrl) == 1.0 -- demonstrating
    # the inequality "MRL_cond_agnostic >= min(...)" does not hold in general.
    assert mrl_pooled < min(mrl_eth, mrl_ctrl)


def test_delta_mrl_is_bounded_between_zero_and_one():
    """delta_MRL = |MRL_ethanol - MRL_control| must always lie in [0, 1], for many random phase configurations."""
    rng = np.random.default_rng(5489)
    for trial in range(200):
        n_eth = rng.integers(1, 200)
        n_ctrl = rng.integers(1, 200)
        eth_phases = rng.uniform(0, 1, size=n_eth)
        ctrl_phases = rng.uniform(0, 1, size=n_ctrl)

        mrl_eth, _ = compute_mrl(eth_phases)
        mrl_ctrl, _ = compute_mrl(ctrl_phases)

        # MRL itself is the magnitude of an average of unit vectors -> bounded
        # in [0, 1] by the triangle inequality, for ANY phase configuration.
        assert 0.0 <= mrl_eth <= 1.0
        assert 0.0 <= mrl_ctrl <= 1.0

        delta_mrl = abs(mrl_eth - mrl_ctrl)
        assert 0.0 <= delta_mrl <= 1.0

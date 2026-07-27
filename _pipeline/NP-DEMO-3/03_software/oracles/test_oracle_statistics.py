# ORACLE
"""
Tests for the LME (unit-level) and paired Wilcoxon (animal-level) statistics.

Targets:
  - RESULT-lme (contract acceptance level 3, statistical_decision; "same
    significance decision (alpha=0.05) on full workload; standardized
    fixed-effect estimate within 1e-4 on fixture slice").
  - RESULT-wilcoxon (level 3, statistical_decision; "same significance
    decision (alpha=0.05); exact p within 1e-6 and rank-biserial r within
    1e-4 on fixture slice").

Contract-pinned procedures (statistical_model, assumptions):
  - LME: statsmodels MixedLM, formula 'MRL ~ C(condition)', random intercept
    for experiment/animal, REML, Wald z p-value.
  - Wilcoxon: scipy.stats.wilcoxon(eth_per_exp, ctrl_per_exp,
    alternative='two-sided'), matched-pairs rank-biserial
    r = 1 - (2*W) / (n*(n+1)/2).

These tests call scipy.stats.wilcoxon and statsmodels MixedLM DIRECTLY --
those are the contract-specified reference procedures themselves (equivalent
to consulting a validated external reference), not the not-yet-written
NP-DEMO-3 analysis code. All numeric expectations below are either the exact
permutation distribution of the Wilcoxon signed-rank statistic for small n
(closed-form, verifiable by hand) or qualitative sign/direction properties
that are true by construction of the synthetic data.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon


def rank_biserial_r(statistic: float, n: int) -> float:
    """Matched-pairs rank-biserial r for paired Wilcoxon (contract formula, verbatim)."""
    return 1 - (2 * statistic) / (n * (n + 1) / 2)


def test_wilcoxon_symmetric_differences_give_p_equal_one():
    """Perfectly symmetric signed differences (+0.1,-0.1,+0.1,-0.1) -> the observed
    rank-sum statistic equals the exact center of its null distribution, so the
    exact two-sided p-value is 1.0 (well-known property of the signed-rank test:
    a maximally 'typical' outcome under the null has p = 1)."""
    eth = np.array([0.1, -0.1, 0.1, -0.1])
    ctrl = np.zeros(4)
    res = wilcoxon(eth, ctrl, alternative="two-sided")

    assert res.pvalue == pytest.approx(1.0, abs=1e-9)
    r = rank_biserial_r(res.statistic, n=4)
    assert r == pytest.approx(0.0, abs=1e-9)


def test_wilcoxon_fully_concordant_n6_gives_significant_p():
    """n=6 experiments, ethanol MRL > control MRL for ALL of them -> the most extreme
    rank-sum outcome occurs with exact probability 2/2^6 = 0.03125 < 0.05 (one-tailed
    probability of "all 6 differences positive" under the null is 1/64; two-sided
    doubles it, since the opposite extreme is equally likely by symmetry)."""
    eth = np.array([0.50, 0.60, 0.70, 0.80, 0.55, 0.75])
    ctrl = np.array([0.20, 0.30, 0.20, 0.30, 0.25, 0.28])
    res = wilcoxon(eth, ctrl, alternative="two-sided")

    expected_p = 2 / (2 ** 6)  # = 0.03125, exact permutation-distribution value
    assert res.pvalue == pytest.approx(expected_p, abs=1e-6)
    assert res.pvalue < 0.05
    r = rank_biserial_r(res.statistic, n=6)
    assert r > 0


def test_rank_biserial_r_formula_all_concordant_n4():
    """n=4 pairs all concordant (eth > ctrl) -> W (sum of ranks of the minority
    sign) = 0, so r = 1 - (2*0)/(4*5/2) = 1.0 exactly (contract formula)."""
    eth = np.array([1.0, 2.0, 3.0, 4.0])
    ctrl = np.zeros(4)
    res = wilcoxon(eth, ctrl, alternative="two-sided")

    assert res.statistic == pytest.approx(0.0, abs=1e-12)
    r = rank_biserial_r(res.statistic, n=4)
    assert r == pytest.approx(1.0, abs=1e-9)
    # Exact two-sided p for the single most extreme n=4 outcome (1/16 * 2).
    assert res.pvalue == pytest.approx(2 / 16, abs=1e-9)


def test_wilcoxon_two_sided_equals_two_times_min_one_sided():
    """scipy's alternative='two-sided' result must equal min(1, 2*min(p_greater, p_less)),
    the standard relationship for an exact, symmetric-null two-sided test -- cross-checked
    against scipy's own one-sided outputs (self-consistency of the specified procedure)."""
    eth = np.array([0.50, 0.60, 0.70, 0.80, 0.55, 0.75])
    ctrl = np.array([0.20, 0.30, 0.20, 0.30, 0.25, 0.28])

    p_two_sided = wilcoxon(eth, ctrl, alternative="two-sided").pvalue
    p_greater = wilcoxon(eth, ctrl, alternative="greater").pvalue
    p_less = wilcoxon(eth, ctrl, alternative="less").pvalue

    expected_two_sided = min(1.0, 2 * min(p_greater, p_less))
    assert p_two_sided == pytest.approx(expected_two_sided, abs=1e-9)


def test_n_pairs_equals_experiments_with_both_conditions_present():
    """n_pairs for the paired Wilcoxon must equal only the experiments that have
    BOTH an ethanol and a control mean-per-unit-MRL value -- experiments missing
    one condition (e.g. 2021-11-03 if no control periods are detected, per
    contract OPEN-003 resolution) must be excluded, not imputed or zero-filled."""
    per_experiment = pd.DataFrame(
        {
            "experiment": ["exp_A", "exp_B", "exp_C", "exp_D"],
            "mean_MRL_ethanol": [0.30, 0.35, 0.40, np.nan],
            "mean_MRL_control": [0.20, 0.22, np.nan, 0.25],
        }
    )
    complete = per_experiment.dropna(subset=["mean_MRL_ethanol", "mean_MRL_control"])

    assert len(complete) == 2  # only exp_A and exp_B have both conditions
    assert set(complete["experiment"]) == {"exp_A", "exp_B"}

    res = wilcoxon(
        complete["mean_MRL_ethanol"].to_numpy(),
        complete["mean_MRL_control"].to_numpy(),
        alternative="two-sided",
    )
    n_pairs = len(complete)
    assert n_pairs == 2
    # With n=2, the only possible ranks are {1,2}; a single-direction result (both
    # concordant here) gives the unique most-extreme exact p for n=2.
    assert res.pvalue == pytest.approx(0.5, abs=1e-9)


def test_lme_condition_fixed_effect_direction_matches_data_construction():
    """LME (statsmodels MixedLM, formula 'MRL ~ C(condition)', random intercept
    for experiment) recovers the correct SIGN of the condition effect when
    ethanol MRL is constructed to be uniformly higher than control MRL,
    after accounting for per-experiment random intercepts and unit-level noise."""
    rng = np.random.default_rng(5489)  # contract randomization_seed
    n_experiments = 6
    n_units_per_exp = 20
    true_ethanol_effect = 0.15

    rows = []
    uid = 0
    for e in range(n_experiments):
        base = rng.uniform(0.15, 0.35, size=n_units_per_exp)
        exp_intercept = rng.normal(0, 0.06)  # random intercept for this experiment
        for i in range(n_units_per_exp):
            ctrl_val = float(np.clip(base[i] + exp_intercept + rng.normal(0, 0.03), 0, 1))
            eth_val = float(
                np.clip(base[i] + exp_intercept + true_ethanol_effect + rng.normal(0, 0.03), 0, 1)
            )
            rows.append({"unit_id": uid, "experiment": f"exp_{e}", "condition": "control", "MRL": ctrl_val})
            rows.append({"unit_id": uid, "experiment": f"exp_{e}", "condition": "ethanol", "MRL": eth_val})
            uid += 1

    df = pd.DataFrame(rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm("MRL ~ C(condition)", df, groups=df["experiment"])
        result = model.fit(reml=True)

    condition_param_name = [name for name in result.params.index if "condition" in name][0]
    coef = result.params[condition_param_name]

    # patsy's default Treatment coding uses the alphabetically-first level
    # ("control") as reference, so the fitted parameter represents the
    # ethanol-vs-control contrast directly.
    if "ethanol" in condition_param_name:
        assert coef > 0
    else:
        assert coef < 0

    assert result.pvalues[condition_param_name] < 0.05

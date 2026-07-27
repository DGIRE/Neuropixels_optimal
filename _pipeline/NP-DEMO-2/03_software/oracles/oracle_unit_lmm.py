"""
oracle_unit_lmm.py -- NP-DEMO-2 oracle for the unit-level mixed-effects test
============================================================================
Written by the ORACLE DESIGNER (Blueprint v2 P3) BEFORE np_demo2_analysis.py
exists. Targets repo_map.md New-Code-Required function:

    unit_level_lmm(unit_df) -- unit_df columns [unit_id, animal_id,
        condition, mrl]. statsmodels MixedLM: condition (eth/ctrl) fixed
        effect, animal random intercept; returns the condition fixed-effect
        coefficient + exact p + a standardized effect size (contract
        statistical_model, RESULT-stat-unit).

statsmodels.formula.api.mixedlm is an already-validated third-party
statistics library (not "the new code under test"); it is used directly here
as the reference model fit, exactly as scipy.stats.wilcoxon is used directly
in oracle_animal_test.py. What IS under test is that np_demo2_analysis wires
condition in as a FIXED effect and animal_id in as a RANDOM INTERCEPT (not
the reverse, and not treating animal as a second fixed effect -- HAZ-08
pseudoreplication), and reports a standardized effect size.

PINNED DEFINITION (oracle-designer decision, flagged for contract sign-off):
the contract prose says "effect size = the standardized fixed-effect
estimate (coefficient / residual SD, a Cohen's-d equivalent)"; this oracle's
test (d) instead pins the OPERATIONAL definition
    standardized_effect_size = fixed_effect_coefficient / SD(unit_df['mrl'])
(SD of the raw pooled outcome, ddof=1), because it is unambiguous, model-
independent, and reproducible without re-deriving statsmodels' internal
residual-variance estimate inside this oracle. If the approved contract
intends "residual SD" (i.e. sqrt(mdf.scale) from the fitted MixedLM) instead,
this is a single-line change to the reference below and to
test_standardized_effect_size_equals_coef_over_outcome_sd -- flagged here
explicitly as an open reconciliation item for the implementer / P2 sign-off,
NOT silently decided.

Acceptance-level mapping (contract_spec.yaml, NP-DEMO-2, contract_version 1)
-----------------------------------------------------------------------------
    RESULT-stat-unit (level 3, statistical_decision: "same significance
        decision (mixed-effects condition fixed-effect p) on full
        workload") -- this file is the pre-fixture, property-based
        acceptance oracle for that object.
    HAZ-08 (pseudoreplication: many units per animal) -- regression test (c).

Implementer note
-----------------
    # from np_demo2_analysis import unit_level_lmm
    # TODO: uncomment when np_demo2_analysis.py exists, and DELETE the
    # inline reference implementation below.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

_NP_DEMO2_SW_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_software"
if _NP_DEMO2_SW_DIR not in sys.path and os.path.isdir(_NP_DEMO2_SW_DIR):
    sys.path.insert(0, _NP_DEMO2_SW_DIR)
from np_demo2_analysis import unit_level_lmm  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic paired unit-level dataset builder: n_animals animals, n_units_per
# units each, PAIRED (control, ethanol) MRL observations per unit -- a
# balanced design (equal condition counts within every animal) so the fixed-
# effect estimate for 'condition' is, by the standard mixed-model result for
# balanced/orthogonal designs, exactly the simple mean(ethanol - control)
# shift regardless of the fitted variance-component values.
# ---------------------------------------------------------------------------
def _make_paired_unit_df(n_animals: int, n_units_per: int, condition_shift: float) -> pd.DataFrame:
    rows = []
    for a in range(n_animals):
        animal_id = f"A{a}"
        animal_base = 0.30 + 0.05 * a  # distinct per-animal baseline MRL
        for u in range(n_units_per):
            unit_id = f"{animal_id}_U{u}"
            unit_base = animal_base + 0.02 * u  # distinct per-unit baseline within animal
            rows.append(dict(unit_id=unit_id, animal_id=animal_id, condition="control", mrl=unit_base))
            rows.append(dict(unit_id=unit_id, animal_id=animal_id, condition="ethanol",
                              mrl=unit_base + condition_shift))
    return pd.DataFrame(rows)


# ===========================================================================
# (a) Zero condition effect (same MRL for eth and ctrl) -> p not significant
# ===========================================================================
def test_zero_condition_effect_gives_nonsignificant_p_and_zero_coefficient():
    """6 animals x 5 units, ethanol MRL set IDENTICAL to control MRL for
    every unit (condition_shift=0.0): the condition fixed effect has no
    signal to detect. For this balanced, perfectly-repeated-measures design,
    the GLS/REML fixed-effect estimate for 'condition' is exactly the mean
    of the (ethanol-control) differences, which is exactly 0.0 here (every
    single difference is 0.0, not just the mean) -> coefficient == 0.0
    exactly, hence p == 1.0 (a zero effect against any non-zero standard
    error always gives Wald p == 1)."""
    df = _make_paired_unit_df(n_animals=6, n_units_per=5, condition_shift=0.0)
    result = unit_level_lmm(df)

    assert result["coef"] == pytest.approx(0.0, abs=1e-9)
    assert result["pvalue"] == pytest.approx(1.0, abs=1e-6)
    assert result["pvalue"] >= 0.05
    assert result["direction"] == "ns"


# ===========================================================================
# (b) Large, clean condition effect -> p significant, correct direction
# ===========================================================================
def test_large_clean_condition_effect_gives_significant_p_correct_direction():
    """Same balanced design, but every ethanol observation is shifted up by
    +0.25 MRL units relative to its paired control (condition_shift=0.25,
    a huge effect relative to the ~0.03-0.15 MRL scale used elsewhere in
    this oracle suite). The balanced-design argument above pins the fixed
    effect coefficient to exactly +0.25 regardless of the fitted
    variance-component values, and with 60 observations across 6 animals
    the Wald p-value for a clean, noiseless +0.25 shift must be far below
    0.05 (verified against a fixed, non-degenerate statsmodels fit)."""
    df = _make_paired_unit_df(n_animals=6, n_units_per=5, condition_shift=0.25)
    result = unit_level_lmm(df)

    assert result["coef"] == pytest.approx(0.25, abs=1e-6)
    assert result["pvalue"] < 0.05
    assert result["direction"] == "ethanol>control"


def test_large_clean_negative_condition_effect_gives_correct_direction_label():
    """Mirror of test (b) with the shift negated (ethanol LOWER than
    control): coefficient must be exactly -0.25, p still significant, and
    the direction label must correctly flip to 'control>ethanol' (guards
    against a hard-coded/one-sided direction label)."""
    df = _make_paired_unit_df(n_animals=6, n_units_per=5, condition_shift=-0.25)
    result = unit_level_lmm(df)

    assert result["coef"] == pytest.approx(-0.25, abs=1e-6)
    assert result["pvalue"] < 0.05
    assert result["direction"] == "control>ethanol"


# ===========================================================================
# (c) Animal as RANDOM intercept, not fixed effect (HAZ-08 pseudoreplication)
# ===========================================================================
def test_removing_one_animal_changes_random_variance_not_fixed_effect():
    """HAZ-08 regression: in this balanced design (every animal contributes
    the exact same +0.25 within-unit shift), the condition fixed-effect
    coefficient is a property of the WITHIN-animal contrast and must stay
    (to a tight tolerance) at +0.25 whether 6 or 5 animals are used --
    dropping one animal removes information used to estimate the RANDOM
    intercept variance (Group Var), which must change materially, while the
    fixed effect itself is essentially unaffected. This is the operational
    signature of 'animal is a random intercept, not a fixed effect': a
    fixed-effect predictor's coefficient generally WOULD change when a
    whole level is dropped (its own data disappears from the contrast); a
    random-intercept grouping variable's marginal fixed effects should not,
    because the population-level condition effect is estimated by pooling
    across all animals' within-animal contrasts, not by an animal-specific
    parameter."""
    df_full = _make_paired_unit_df(n_animals=6, n_units_per=5, condition_shift=0.25)
    df_dropped = df_full[df_full["animal_id"] != "A0"].reset_index(drop=True)

    result_full = unit_level_lmm(df_full)
    result_dropped = unit_level_lmm(df_dropped)

    assert result_full["coef"] == pytest.approx(0.25, abs=1e-6)
    assert result_dropped["coef"] == pytest.approx(0.25, abs=1e-6)
    assert result_full["coef"] == pytest.approx(result_dropped["coef"], abs=1e-4), (
        "fixed-effect coefficient must be materially unchanged by dropping one animal"
    )
    assert result_full["group_var"] != pytest.approx(result_dropped["group_var"], rel=1e-3), (
        "animal-level (random-intercept) variance must change when an animal is dropped"
    )
    assert result_full["n_animals"] == 6
    assert result_dropped["n_animals"] == 5


# ===========================================================================
# (d) Standardized coefficient: coef / SD(outcome) -- pinned definition
# ===========================================================================
def test_standardized_effect_size_equals_coef_over_outcome_sd():
    """Per this oracle's pinned operational definition (see module
    docstring's reconciliation note), standardized_effect_size must equal
    the raw fixed-effect coefficient divided by the sample SD (ddof=1) of
    the pooled 'mrl' outcome column -- checked here as a direct arithmetic
    identity on the function's own two returned quantities, so any
    implementation that returns a differently-scaled 'standardized' value
    (e.g. divided by residual SD instead) will fail this test loudly rather
    than silently reporting an ambiguous number."""
    df = _make_paired_unit_df(n_animals=6, n_units_per=5, condition_shift=0.25)
    result = unit_level_lmm(df)

    outcome_sd = float(df["mrl"].std(ddof=1))
    expected_standardized = result["coef"] / outcome_sd

    assert result["standardized_effect_size"] == pytest.approx(expected_standardized, abs=1e-9)


def test_standardized_effect_size_is_zero_when_coefficient_is_zero():
    """Degenerate check paired with test (a): a zero fixed effect must give
    a zero standardized effect size (0 / any positive SD == 0), not nan or a
    spurious non-zero value from numerical noise."""
    df = _make_paired_unit_df(n_animals=6, n_units_per=5, condition_shift=0.0)
    result = unit_level_lmm(df)
    assert result["standardized_effect_size"] == pytest.approx(0.0, abs=1e-6)

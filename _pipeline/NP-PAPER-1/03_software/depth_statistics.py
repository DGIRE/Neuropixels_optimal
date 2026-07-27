"""
depth_statistics.py -- NP-PAPER-1 additive analysis function.

Depth-dependence statistics for RESULT-depth-stat (contract statistical_model
+ aggregation step docx_source_refs RQ-029, CON-004, CON-006, OUT-014,
OUT-015, OUT-016, OUT-013, RQ-068). Interface pinned by the oracle designer
in oracles/test_oracle_depth_statistics.py (S1-S6) -- authoritative, not a
suggestion.

    depth_statistics(theta_coh_table) -> dict

Input: long-form pandas DataFrame with columns
    ['experiment', 'depth', 'mean_theta_coh']
one row per experiment x depth cell (balanced: every experiment must have a
value for every depth level -- this is the repeated-measures design CON-004
requires for both the mixed-effects model and Friedman/Wilcoxon).

Output dict (S3):
    result['lmm']      = {chisq, df, pvalue, marginal_mean_range, icc}
    result['friedman'] = {statistic, df, pvalue, kendalls_w}
    result['omnibus_significant'] = bool  (LMM LRT p<0.05 OR Friedman p<0.05)
    result['posthoc']  = None if not omnibus_significant, else a list of dicts
        {depth_a, depth_b, p_raw, p_holm, rank_biserial_r, holm_significant},
        one per unordered depth pair (Holm-corrected across the whole family).

PRIMARY (mixed-effects LRT): statsmodels MixedLM, mean_theta_coh ~ C(depth)
(5-level fixed effect) + experiment as random intercept, fit by MAXIMUM
LIKELIHOOD (reml=False) for both the full model and the intercept-only
reduced model (S2) -- a REML-based LRT across differing fixed effects is
statistically invalid. LRT = 2*(llf_full - llf_reduced); df = n_depths - 1;
exact p via scipy.stats.chi2.sf.

Effect size: marginal_mean_range = range (max-min) of the fixed-effect
estimated marginal depth means; icc = an ICC-like variance-explained measure
in [0, 1], var_between / (var_between + var_residual), from the full model's
random-intercept variance and residual scale.

CONFIRMATORY (Friedman): scipy.stats.friedmanchisquare across the 5 depths,
experiments as blocks (wide format: rows=experiments, cols=depths). Kendall's
W = statistic / (n_experiments * (n_depths - 1)) (S4).

POST-HOC: only if EITHER omnibus test is significant at p<0.05 (S3):
pairwise exact two-sided Wilcoxon signed-rank (scipy.stats.wilcoxon,
mode='exact') between every pair of depths, Holm-corrected across the full
C(5,2)=10-pair family (statsmodels.stats.multitest.multipletests, S6),
reporting matched-pairs rank-biserial r per pair (S5):
    r = 1 - 2*W_stat / (n*(n+1)/2)
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2, friedmanchisquare, wilcoxon
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

_REQUIRED_COLS = ("experiment", "depth", "mean_theta_coh")


def depth_statistics(theta_coh_table: pd.DataFrame) -> dict:
    missing = [c for c in _REQUIRED_COLS if c not in theta_coh_table.columns]
    if missing:
        raise ValueError(f"theta_coh_table is missing required columns: {missing}")

    df = theta_coh_table.loc[:, list(_REQUIRED_COLS)].copy()
    df["experiment"] = df["experiment"].astype(str)
    df["depth"] = df["depth"].astype(str)
    df["mean_theta_coh"] = df["mean_theta_coh"].astype(float)

    depth_levels = sorted(df["depth"].unique())
    n_depths = len(depth_levels)
    experiments = sorted(df["experiment"].unique())
    n_experiments = len(experiments)

    if n_depths < 2:
        raise ValueError("depth_statistics requires at least two depth levels")

    # ---- Balanced wide pivot (rows=experiments, cols=depths), for Friedman ----
    # and Wilcoxon post-hoc, both of which require the repeated-measures shape.
    wide = df.pivot(index="experiment", columns="depth", values="mean_theta_coh")
    wide = wide.loc[experiments, depth_levels]
    if wide.isnull().values.any():
        raise ValueError(
            "theta_coh_table is not a complete experiment x depth table "
            "(every experiment must have a value at every depth level)"
        )

    # =======================================================================
    # PRIMARY: mixed-effects likelihood-ratio test (ML, not REML -- S2)
    # =======================================================================
    full_model = MixedLM.from_formula(
        "mean_theta_coh ~ C(depth)", groups="experiment", data=df
    )
    full_fit = full_model.fit(reml=False)

    reduced_model = MixedLM.from_formula(
        "mean_theta_coh ~ 1", groups="experiment", data=df
    )
    reduced_fit = reduced_model.fit(reml=False)

    df_lrt = n_depths - 1
    chisq = 2.0 * (full_fit.llf - reduced_fit.llf)
    # Guard tiny negative values from numerical optimizer noise (the full model
    # nests the reduced model, so the true LRT statistic is non-negative).
    if chisq < 0.0:
        chisq = 0.0
    pvalue = float(chi2.sf(chisq, df_lrt))

    reference_depth = depth_levels[0]
    intercept = float(full_fit.params.get("Intercept", 0.0))
    marginal_means = []
    for depth in depth_levels:
        if depth == reference_depth:
            marginal_means.append(intercept)
        else:
            coef_key = f"C(depth)[T.{depth}]"
            marginal_means.append(intercept + float(full_fit.params.get(coef_key, 0.0)))
    marginal_means = np.asarray(marginal_means, dtype=float)
    marginal_mean_range = float(marginal_means.max() - marginal_means.min())

    var_between = float(full_fit.cov_re.iloc[0, 0])
    var_residual = float(full_fit.scale)
    total_var = var_between + var_residual
    icc = var_between / total_var if total_var > 0 else 0.0
    icc = float(min(max(icc, 0.0), 1.0))

    lmm = dict(
        chisq=float(chisq),
        df=int(df_lrt),
        pvalue=pvalue,
        marginal_mean_range=marginal_mean_range,
        icc=icc,
    )

    # =======================================================================
    # CONFIRMATORY: Friedman test, experiments as blocks
    # =======================================================================
    depth_columns = [wide[d].to_numpy() for d in depth_levels]
    fr = friedmanchisquare(*depth_columns)
    kendalls_w = float(fr.statistic) / (n_experiments * (n_depths - 1))

    friedman = dict(
        statistic=float(fr.statistic),
        df=int(n_depths - 1),
        pvalue=float(fr.pvalue),
        kendalls_w=float(kendalls_w),
    )

    omnibus_significant = bool(lmm["pvalue"] < 0.05 or friedman["pvalue"] < 0.05)

    # =======================================================================
    # POST-HOC: Holm-corrected exact two-sided Wilcoxon, ONLY if omnibus sig.
    # (PROH-006/OUT-016 -- never run/report uncontracted post-hoc comparisons)
    # =======================================================================
    posthoc = None
    if omnibus_significant:
        pair_indices = list(combinations(range(n_depths), 2))
        raw_pvalues = []
        rank_biserial_rs = []
        for a, b in pair_indices:
            xa = wide[depth_levels[a]].to_numpy()
            xb = wide[depth_levels[b]].to_numpy()
            n = len(xa)
            res = wilcoxon(xa, xb, mode="exact", alternative="two-sided")
            r = 1.0 - 2.0 * float(res.statistic) / (n * (n + 1) / 2.0)
            raw_pvalues.append(float(res.pvalue))
            rank_biserial_rs.append(float(r))

        reject, p_holm, _, _ = multipletests(raw_pvalues, alpha=0.05, method="holm")

        posthoc = []
        for (a, b), p_raw, p_adj, rej, r in zip(
            pair_indices, raw_pvalues, p_holm, reject, rank_biserial_rs
        ):
            posthoc.append(
                dict(
                    depth_a=depth_levels[a],
                    depth_b=depth_levels[b],
                    p_raw=p_raw,
                    p_holm=float(p_adj),
                    rank_biserial_r=r,
                    holm_significant=bool(rej),
                )
            )

    return dict(
        lmm=lmm,
        friedman=friedman,
        omnibus_significant=omnibus_significant,
        posthoc=posthoc,
    )

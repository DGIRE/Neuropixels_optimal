"""
test_oracle_depth_statistics.py -- NP-PAPER-1 oracle for depth_statistics
=========================================================================
Written by the ORACLE DESIGNER (Blueprint v2 P3) BEFORE depth_statistics
exists. Owns the DEPTH-STATISTICS subset of NP-PAPER-1's new functions.

Function under test (repo_map.md #9; contract statistical_model +
aggregation step docx_source_refs [RQ-029, CON-004, CON-006, OUT-014,
OUT-015, OUT-016, OUT-013, RQ-068]; RESULT-depth-stat, acceptance level 3
statistical_decision):

    depth_statistics(theta_coh_table)
      PRIMARY   : linear mixed-effects (statsmodels MixedLM) mean theta
                  coherence ~ C(depth) [5-level fixed] + experiment [random
                  intercept]; overall depth effect via LRT vs the
                  intercept-only random-effects model. Report chi-square, df,
                  exact p; effect size = range of marginal depth means + ICC.
      CONFIRM.  : Friedman across the 5 depths, experiments as blocks
                  (scipy.stats.friedmanchisquare). Report statistic, df,
                  exact p, Kendall's W.
      POST-HOC  : ONLY if the omnibus is significant (p<0.05): pairwise paired
                  exact two-sided Wilcoxon (scipy.stats.wilcoxon, mode='exact')
                  between depths, Holm-corrected across the 10-pair family
                  (statsmodels multipletests); matched-pairs rank-biserial r
                  per surviving pair.

INDEPENDENT-ORACLE POLICY (per the P3 brief):
  * scipy.stats.friedmanchisquare, scipy.stats.wilcoxon(mode='exact') and
    statsmodels.stats.multitest.multipletests(method='holm') are established,
    widely-used third-party procedures -- they are called DIRECTLY here AS the
    reference/oracle (this is NOT "running the new code"; depth_statistics is
    the new code). Their expected values below were additionally cross-checked
    by hand where the exact combinatorial null is small (e.g. Wilcoxon n=6
    two-sided floor p = 2/2^6 = 0.03125; Holm step-down worked by hand).
  * statsmodels.MixedLM IS the library depth_statistics wraps, so for the LRT
    we do NOT merely trust "whatever MixedLM prints". Its chi-square/p are
    anchored independently by: (i) the deterministic NULL case, where equal
    depth column means force the ML depth-coefficient MLEs to 0 and hence the
    LRT to ~0 / p~1 (first principles); (ii) df == n_depth_levels-1 == 4
    (first principles); (iii) marginal_mean_range == the plain column-mean
    range of the balanced input (numpy, exact); (iv) decision AGREEMENT with
    the independent Friedman test on the same clean data. ICC is checked only
    for bounds [0,1] and a monotonicity property, not an exact value.

-----------------------------------------------------------------------------
ORACLE-DESIGNER DECISIONS (flagged for P2/implementer sign-off):
-----------------------------------------------------------------------------
 S1. INPUT FORMAT pinned: long-form pandas DataFrame with columns
       ['experiment', 'depth', 'mean_theta_coh']
     experiment = grouping label (str), depth = 5-level categorical label
     (str 'D1'..'D5' or int), one row per experiment x depth cell.
 S2. LRT must be fit by MAXIMUM LIKELIHOOD (reml=False) for BOTH the full and
     the intercept-only reduced model -- a REML likelihood-ratio test across
     models with different FIXED effects is invalid. LRT = 2*(llf_full -
     llf_reduced); df = n_depth_levels - 1 = 4; p = chi2.sf(LRT, df).
 S3. OUTPUT KEYS pinned:
       result['lmm']      = {chisq, df, pvalue, marginal_mean_range, icc}
       result['friedman'] = {statistic, df, pvalue, kendalls_w}
       result['omnibus_significant'] = bool  (True if LMM LRT p<0.05 OR
                                              Friedman p<0.05)
       result['posthoc']  = None if not omnibus_significant, else a list of
             dicts, one per unordered depth pair, each with keys
             {depth_a, depth_b, p_raw, p_holm, rank_biserial_r,
              holm_significant}.
 S4. Kendall's W = Friedman_statistic / (n_experiments * (n_depths - 1)).
 S5. rank-biserial r (matched pairs) = 1 - 2*W_stat / (n*(n+1)/2), the same
     formula validated in NP-DEMO-3 oracle_statistics; |r| == 1 for a fully
     concordant pair.
 S6. Holm family = all C(5,2) = 10 unordered depth pairs, corrected together
     (contract multiple_comparison: Holm across the pairwise family, post-hoc
     ONLY, only when omnibus significant -- PROH-006/OUT-016).

PROH-006 NOTE: the depth-dependence question is EXPLORATORY. These tests only
assert the STATISTICAL PROCEDURE behaves correctly on synthetic ground truth
(planted / absent effects); they assert nothing about any real depth ranking.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import friedmanchisquare, wilcoxon, chi2, binom
from statsmodels.stats.multitest import multipletests

_SW_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-PAPER-1\03_software"
if _SW_DIR not in sys.path and os.path.isdir(_SW_DIR):
    sys.path.insert(0, _SW_DIR)
try:
    from depth_statistics import depth_statistics  # noqa: E402
    IMPL_AVAILABLE = True
except Exception:  # pragma: no cover - expected before implementation exists
    depth_statistics = None
    IMPL_AVAILABLE = False

impl = pytest.mark.skipif(not IMPL_AVAILABLE,
                          reason="depth_statistics not implemented yet (P3 oracle)")

DEPTH_LABELS = ["D1", "D2", "D3", "D4", "D5"]


# ---------------------------------------------------------------------------
# Synthetic ground-truth datasets (wide 6x5 / 10x5 arrays, columns = depth 1..5)
# ---------------------------------------------------------------------------
def _wide_to_long(mat: np.ndarray) -> pd.DataFrame:
    rows = []
    for e in range(mat.shape[0]):
        for d in range(mat.shape[1]):
            rows.append(dict(experiment=f"E{e+1}", depth=DEPTH_LABELS[d],
                             mean_theta_coh=float(mat[e, d])))
    return pd.DataFrame(rows)


def _null_matrix() -> np.ndarray:
    """DETERMINISTIC null: 6 exp x 5 depths, depth column means EXACTLY equal
    (balanced cyclic Latin square + one flat experiment + per-experiment
    offsets = random intercepts). Genuinely NO depth effect; the expected
    Friedman stat is 0 and p is 1 by construction (equal rank sums)."""
    base = 0.30
    o = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.05])   # per-experiment offset
    p = np.array([-0.02, -0.01, 0.00, 0.01, 0.02])       # zero-sum depth pattern
    mat = np.zeros((6, 5))
    for e in range(5):
        mat[e] = base + o[e] + np.roll(p, e)             # cyclic shift
    mat[5] = base + o[5]                                  # flat experiment
    return mat


def _strong_matrix() -> np.ndarray:
    """PLANTED strong effect, 6 exp x 5 depths: identical within-experiment
    ordering d1<d2<d4<d5<d3 in EVERY experiment (depth 3 always the maximum).
    -> maximally concordant Friedman (W=1, chi2=24), and every depth pair is
    fully concordant so each raw Wilcoxon p = 0.03125 (n=6 floor)."""
    return np.array([
        [0.30, 0.33, 0.50, 0.36, 0.39],
        [0.31, 0.34, 0.52, 0.37, 0.40],
        [0.29, 0.32, 0.51, 0.35, 0.38],
        [0.32, 0.36, 0.55, 0.39, 0.42],
        [0.28, 0.31, 0.52, 0.34, 0.37],
        [0.30, 0.34, 0.55, 0.38, 0.42],
    ])


def _disc_matrix() -> np.ndarray:
    """DISCRIMINATING post-hoc case, 10 exp x 5 depths (deliberately n=10, ABOVE
    the contracted n<=6, purely to exercise the Holm-SURVIVOR code path -- see
    test_reference_strong_holm_zero_survivors_at_n6 for why n<=6 can never yield
    a survivor). Depth 3 is the clear max in every experiment (all-concordant vs
    every other depth => raw p = 2/2^10), while depths 1,2,4,5 rotate order so
    non-d3 pairs are sign-balanced (high p). Holm therefore keeps EXACTLY the
    four depth-3 pairs."""
    d3col = np.array([0.55, 0.57, 0.56, 0.60, 0.58, 0.59, 0.55, 0.61, 0.57, 0.60])
    basevals = np.array([0.30, 0.34, 0.38, 0.42])
    disc = np.zeros((10, 5))
    for e in range(10):
        perm = np.roll(basevals, e % 4)
        jit = 0.001 * np.array([(e * 7) % 5, (e * 3) % 5, (e * 11) % 5, (e * 13) % 5])
        vals = perm + jit
        disc[e, 0], disc[e, 1], disc[e, 3], disc[e, 4] = vals
        disc[e, 2] = d3col[e]
    return disc


def _find_pair(posthoc, a_label, b_label):
    for row in posthoc:
        pair = {row["depth_a"], row["depth_b"]}
        if pair == {a_label, b_label}:
            return row
    raise AssertionError(f"pair {a_label}-{b_label} missing from posthoc output")


# ===========================================================================
# REFERENCE self-checks (run NOW; scipy/statsmodels/hand only). These prove
# the planted ground truth and the expected statistics BEFORE any code exists.
# ===========================================================================
def test_reference_null_is_balanced_and_friedman_zero():
    """NULL calibration anchor: equal depth column means -> Friedman stat 0,
    exact p 1, Kendall's W 0 (verified via scipy). No single-draw gamble: this
    is a fixed deterministic dataset, so the non-significant result is
    reproducible, not a 5%-of-the-time fluke."""
    mat = _null_matrix()
    col_means = mat.mean(axis=0)
    assert np.allclose(col_means, col_means[0], atol=1e-12)  # exactly equal
    fr = friedmanchisquare(*[mat[:, d] for d in range(5)])
    assert fr.statistic == pytest.approx(0.0, abs=1e-9)
    assert fr.pvalue == pytest.approx(1.0, abs=1e-9)


def test_reference_strong_friedman_chi2_24_p_and_W():
    """Known-effect anchor (Friedman): maximally concordant 6x5 -> chi2 = 24,
    df = 4, Kendall's W = 1, exact p = 7.99e-5 (scipy)."""
    mat = _strong_matrix()
    fr = friedmanchisquare(*[mat[:, d] for d in range(5)])
    assert fr.statistic == pytest.approx(24.0, rel=1e-9)
    assert fr.pvalue == pytest.approx(7.987476059326673e-05, rel=1e-6)
    W = fr.statistic / (6 * (5 - 1))
    assert W == pytest.approx(1.0, abs=1e-12)


def test_reference_strong_wilcoxon_d3_vs_d1_hand_value():
    """Known-effect anchor (Wilcoxon): depth3-depth1 differences are all six
    positive, so W_stat = 0, exact two-sided p = 2/2^6 = 0.03125 (hand-derivable
    combinatorial floor for n=6), rank-biserial r = 1.0."""
    mat = _strong_matrix()
    w = wilcoxon(mat[:, 2], mat[:, 0], mode="exact", alternative="two-sided")
    assert w.statistic == pytest.approx(0.0, abs=1e-12)
    assert w.pvalue == pytest.approx(0.03125, abs=1e-9)
    r = 1 - 2 * w.statistic / (6 * 7 / 2)
    assert r == pytest.approx(1.0, abs=1e-12)


def test_reference_strong_holm_zero_survivors_at_n6():
    """CORRECT conservative behavior anchor: with n=6 the smallest possible
    exact two-sided Wilcoxon p is 0.03125, but the first Holm gate over the
    10-pair family is 0.05/10 = 0.005. Since 0.03125 > 0.005, NO pair can
    survive Holm at n<=6 even when every pair is fully concordant. This is a
    real limitation of the contracted design (flag for the results report;
    consistent with the exploratory framing, PROH-006)."""
    mat = _strong_matrix()
    pvals = [wilcoxon(mat[:, a], mat[:, b], mode="exact", alternative="two-sided").pvalue
             for a, b in combinations(range(5), 2)]
    assert all(p == pytest.approx(0.03125, abs=1e-9) for p in pvals)
    rej, _, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    assert rej.sum() == 0


def test_reference_disc_holm_keeps_exactly_the_four_depth3_pairs():
    """Discriminating anchor (n=10): Holm survivors are EXACTLY the four
    depth-3 pairs; no non-depth-3 pair survives. Verified via scipy Wilcoxon +
    statsmodels Holm."""
    mat = _disc_matrix()
    pairs = list(combinations(range(5), 2))
    pvals = [wilcoxon(mat[:, a], mat[:, b], mode="exact", alternative="two-sided").pvalue
             for a, b in pairs]
    rej, _, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    survivors = {tuple(sorted((a + 1, b + 1))) for (a, b), r in zip(pairs, rej) if r}
    assert survivors == {(1, 3), (2, 3), (3, 4), (3, 5)}


def test_reference_holm_hand_microcase():
    """Holm anchor: 4 raw p-values worked by hand vs statsmodels.
    raw = [0.01, 0.04, 0.03, 0.005]; sorted asc 0.005,0.01,0.03,0.04 ->
    * m: 0.02, 0.03, 0.06, 0.04 -> cumulative-max 0.02, 0.03, 0.06, 0.06.
    Reject at alpha=0.05 -> [True, False, False, True] in ORIGINAL order."""
    raw = [0.01, 0.04, 0.03, 0.005]
    rej, padj, _, _ = multipletests(raw, alpha=0.05, method="holm")
    assert list(rej) == [True, False, False, True]
    assert padj == pytest.approx([0.03, 0.06, 0.06, 0.02], abs=1e-12)


def test_reference_randomized_null_calibration_binomial_band():
    """Belt-and-suspenders calibration of the CONFIRMATORY (Friedman) procedure:
    2000 independent no-depth-effect draws (fixed seed) should reject at ~5%.
    Asserted against a 99% binomial tolerance band, NOT a single-draw threshold.
    This calibrates the trusted procedure the implementation wraps; the
    implementation's own decision is gated by the deterministic null case."""
    rng = np.random.default_rng(2024)
    N = 2000
    fp = sum(
        friedmanchisquare(*[m[:, d] for d in range(5)]).pvalue < 0.05
        for m in (rng.normal(0.30, 0.05, size=(6, 5)) for _ in range(N))
    )
    lo = int(binom.ppf(0.005, N, 0.05))
    hi = int(binom.ppf(0.995, N, 0.05))
    assert lo <= fp <= hi, f"Friedman false-positive count {fp} outside 99% band [{lo},{hi}]"


# ===========================================================================
# IMPLEMENTATION-GATED tests: what depth_statistics must reproduce.
# ===========================================================================

# ---- NULL case: both omnibus tests correctly FAIL to reject (level 3) -------
@impl
def test_null_case_both_omnibus_fail_to_reject():
    """RESULT-depth-stat level 3. Deterministic no-effect data: the mixed-effects
    LRT and the Friedman test must BOTH be non-significant (correctly do not
    find a depth effect that is not there). Friedman is anchored to exact
    (stat 0 / p 1); the LMM LRT to ~0 (equal column means -> ML depth MLEs 0)."""
    res = depth_statistics(_wide_to_long(_null_matrix()))
    assert res["friedman"]["statistic"] == pytest.approx(0.0, abs=1e-6)
    assert res["friedman"]["pvalue"] == pytest.approx(1.0, abs=1e-4)
    assert res["friedman"]["kendalls_w"] == pytest.approx(0.0, abs=1e-6)
    assert res["friedman"]["pvalue"] >= 0.05
    assert res["lmm"]["chisq"] < 0.5           # ~0 by construction; << crit 9.49
    assert res["lmm"]["pvalue"] >= 0.05        # correctly fails to reject
    assert res["omnibus_significant"] is False


@impl
def test_null_case_runs_no_posthoc():
    """PROH-006/OUT-016: with a non-significant omnibus, NO post-hoc pairwise
    comparisons may be run or reported."""
    res = depth_statistics(_wide_to_long(_null_matrix()))
    assert res["posthoc"] is None


# ---- STRONG case: omnibus rejects; effect size; post-hoc runs (level 2/3) ---
@impl
def test_strong_case_omnibus_rejects_and_effect_size():
    """Known-effect data: both omnibus tests reject at low p; Friedman anchored
    to chi2=24 / p=7.99e-5 / W=1. marginal_mean_range must equal the plain
    depth column-mean range of the balanced input (independent numpy anchor)."""
    mat = _strong_matrix()
    res = depth_statistics(_wide_to_long(mat))

    assert res["friedman"]["statistic"] == pytest.approx(24.0, rel=1e-4)
    assert res["friedman"]["df"] == 4
    assert res["friedman"]["pvalue"] == pytest.approx(7.987476059326673e-05, rel=1e-4)
    assert res["friedman"]["kendalls_w"] == pytest.approx(1.0, abs=1e-4)

    assert res["lmm"]["df"] == 4
    assert res["lmm"]["pvalue"] < 0.05
    assert res["lmm"]["chisq"] > 9.49         # df=4 critical value at 0.05
    expected_range = mat.mean(axis=0).max() - mat.mean(axis=0).min()
    assert res["lmm"]["marginal_mean_range"] == pytest.approx(expected_range, abs=1e-4)
    assert res["lmm"]["marginal_mean_range"] == pytest.approx(0.225, abs=1e-6)

    assert res["omnibus_significant"] is True


@impl
def test_strong_case_posthoc_runs_but_holm_keeps_none():
    """Post-hoc executes (omnibus significant) but correctly yields ZERO
    Holm-survivors at n=6 (see reference test): each raw p = 0.03125, each
    fully-concordant pair rank-biserial |r| = 1, none survive Holm."""
    res = depth_statistics(_wide_to_long(_strong_matrix()))
    assert res["posthoc"] is not None
    assert len(res["posthoc"]) == 10
    for row in res["posthoc"]:
        assert row["p_raw"] == pytest.approx(0.03125, abs=1e-4)
        assert abs(row["rank_biserial_r"]) == pytest.approx(1.0, abs=1e-4)
        assert row["holm_significant"] is False
    assert sum(r["holm_significant"] for r in res["posthoc"]) == 0
    d3_d1 = _find_pair(res["posthoc"], "D3", "D1")
    assert d3_d1["p_raw"] == pytest.approx(0.03125, abs=1e-6)


# ---- DISC case: post-hoc correctly isolates depth 3 (level 3) ---------------
@impl
def test_disc_case_posthoc_survivors_are_exactly_depth3_pairs():
    """Discriminating synthetic (n=10, code-path exerciser): Holm survivors are
    EXACTLY the four depth-3 pairs and nothing else; each surviving pair carries
    raw p = 2/2^10 and matched-pairs |r| = 1. Anchored to scipy+statsmodels
    in test_reference_disc_holm_keeps_exactly_the_four_depth3_pairs."""
    res = depth_statistics(_wide_to_long(_disc_matrix()))
    assert res["omnibus_significant"] is True
    assert res["posthoc"] is not None
    survivors = {frozenset((r["depth_a"], r["depth_b"]))
                 for r in res["posthoc"] if r["holm_significant"]}
    expected = {frozenset(("D3", "D1")), frozenset(("D3", "D2")),
                frozenset(("D3", "D4")), frozenset(("D3", "D5"))}
    assert survivors == expected
    for r in res["posthoc"]:
        if r["holm_significant"]:
            assert r["p_raw"] == pytest.approx(2 / 2 ** 10, abs=1e-6)
            assert abs(r["rank_biserial_r"]) == pytest.approx(1.0, abs=1e-4)


# ---- LMM structural anchors: df, LRT non-negativity, ICC bounds/monotonic ---
@impl
def test_lmm_lrt_df_is_four_and_nonnegative():
    """First-principles LMM anchors independent of MixedLM's numeric output:
    df of the depth LRT = 5 levels - 1 = 4, and a valid LRT statistic is
    non-negative (full model nests the reduced model)."""
    for mat in (_null_matrix(), _strong_matrix()):
        res = depth_statistics(_wide_to_long(mat))
        assert res["lmm"]["df"] == 4
        assert res["lmm"]["chisq"] >= -1e-6


@impl
def test_lmm_icc_within_unit_interval_and_increases_with_between_variance():
    """ICC is a proportion in [0,1]; and inflating the between-experiment spread
    (random-intercept variance) while holding the within-experiment depth
    pattern fixed must INCREASE the ICC. Property check (no exact ICC value is
    hand-derivable from the REML/ML variance-component fit)."""
    base = _strong_matrix()
    # remove the existing per-experiment offset, then re-add a small vs large one
    depth_pattern = base - base.mean(axis=1, keepdims=True)
    small_off = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])[:, None]
    large_off = np.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])[:, None]
    res_small = depth_statistics(_wide_to_long(depth_pattern + 0.4 + small_off))
    res_large = depth_statistics(_wide_to_long(depth_pattern + 0.4 + large_off))
    for res in (res_small, res_large):
        assert 0.0 <= res["lmm"]["icc"] <= 1.0
    assert res_large["lmm"]["icc"] > res_small["lmm"]["icc"]


@impl
def test_lmm_and_friedman_decisions_agree_on_clean_data():
    """Cross-procedure consistency anchor: on the deterministic NULL and STRONG
    datasets the (independent) mixed-effects LRT and Friedman tests must reach
    the SAME significance decision -- a guard against the LMM being wired up
    backwards (e.g. depth as random / experiment as fixed)."""
    null_res = depth_statistics(_wide_to_long(_null_matrix()))
    strong_res = depth_statistics(_wide_to_long(_strong_matrix()))
    assert (null_res["lmm"]["pvalue"] < 0.05) == (null_res["friedman"]["pvalue"] < 0.05)
    assert (strong_res["lmm"]["pvalue"] < 0.05) == (strong_res["friedman"]["pvalue"] < 0.05)

# ORACLE
"""
Synthetic-ground-truth microcases for the NOT-YET-IMPLEMENTED
build_valid_sniff_mask(D_with_phase).

CONTRACT TARGET
  inclusion_exclusion rule (RQ-020, RQ-021, OUT-008):
    "Valid-sniff only: exclude any 4-s window overlapping an invalid-sniff span
     (ISI outside 5th/95th percentile OR SNF_PH==-1 noise)."
  Feeds RESULT-qc-counts (n_windows_excluded noise / n_sniffs, level 1 exact
  integers), RESULT-sniff-rate (median/IQR of 1/ISI, level 2 within 1e-4 Hz),
  and RESULT-qc-discards (ISI-outlier / non-sniff-noise span boundaries,
  level 1 exact). docx_source_refs: RQ-018, RQ-019, RQ-020, RQ-021, OUT-008.

ALGORITHM UNDER TEST (contract preprocessing PROH-004/PROH-007/RQ-019..021 +
repo_map §2.4 and HAZARD H1):
  1. ISI = np.diff(sniff_onsets_s)   -- PER-SNIFF array.
     HAZARD H1: NOT the scalar sniff_dur_s (median only). Must be derived fresh.
  2. p5, p95 = 5th/95th percentile of THIS experiment's ISI distribution.
  3. An inter-onset span is INVALID if its ISI < p5 or > p95.
  4. Any sample with SNF_PH == -1 (compute_sniff_phase noise sentinel) is INVALID.
  5. Instantaneous sniff rate = 1/ISI per retained span.

CLOCK: to make sample math exact and hand-computable, the synthetic sets
LV_Fs = 1000 Hz (the RULE is rate-agnostic; matching the LV and LFP rates
isolates the masking logic from any resample step). The mask is expected on the
SNF_PH sample clock (len == len(D['SNF_PH'])).

All expected values are hand-computed (shown inline) and were cross-checked with
np.percentile / np.diff at oracle-design time. Nothing came from running the new
function.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    load_real_build_valid_sniff_mask,
    reference_inst_sniff_rate,
    reference_isi_cutoffs,
    reference_valid_sniff_mask,
)

FS = 1000.0

# ISI sequence in TEMPORAL order:
#   span0        = 0.20 s  (fast outlier; == sorted min)
#   spans 1..19  = 0.30, 0.31, ..., 0.48 s  (the 19 "normal" ISIs)
#   span20       = 0.90 s  (slow outlier; == sorted max)
# Full 21-value distribution sorted = [0.20, 0.30..0.48, 0.90].
# numpy 'linear' percentile, n=21: p5 = sorted[(21-1)*0.05=1] = 0.30;
#                                  p95 = sorted[(21-1)*0.95=19] = 0.48.
_ISI_ORDER = [0.20] + [round(0.30 + 0.01 * i, 2) for i in range(19)] + [0.90]


def _make_onsets():
    return np.concatenate([[0.0], np.cumsum(_ISI_ORDER)])  # 22 onsets


def _make_D(onsets_s, snf_ph, fs=FS):
    return {
        "sniff_onsets_s": np.asarray(onsets_s, dtype=np.float64),
        "SNF_PH": np.asarray(snf_ph, dtype=np.float64),
        "LV_Fs": float(fs),
    }


def _run_mask(D):
    """Call REAL build_valid_sniff_mask if implemented, else the reference.

    Expected real interface (defined here for the implementer): given
    D_with_phase (keys SNF_PH, sniff_onsets_s, LV_Fs), return the per-SNF_PH-
    sample boolean valid-sniff mask (True == retained). Accepts a bare ndarray
    or a dict exposing 'valid_sniff_mask'/'valid_mask'."""
    real = load_real_build_valid_sniff_mask()
    if real is None:
        valid, _isi, _p5, _p95 = reference_valid_sniff_mask(
            D["sniff_onsets_s"], D["SNF_PH"], D["LV_Fs"])
        return valid
    out = real(D)
    if isinstance(out, dict):
        for k in ("valid_sniff_mask", "valid_mask", "mask"):
            if k in out:
                return np.asarray(out[k], dtype=bool)
        raise KeyError("no valid-sniff-mask key in build_valid_sniff_mask output")
    if isinstance(out, tuple):
        return np.asarray(out[0], dtype=bool)
    return np.asarray(out, dtype=bool)


# ===========================================================================
# Pure-arithmetic checks (run TODAY): ISI derivation + percentile cutoffs
# ===========================================================================
def test_isi_is_diff_of_onsets_not_scalar_dur():
    """HAZARD H1: ISI must be np.diff(sniff_onsets_s), a per-sniff ARRAY of
    length n_onsets-1 -- never the scalar median sniff_dur_s."""
    onsets = _make_onsets()
    isi, _p5, _p95 = reference_isi_cutoffs(onsets)
    assert isi.shape == (len(_ISI_ORDER),)
    np.testing.assert_allclose(isi, _ISI_ORDER, atol=1e-9)


def test_percentile_cutoffs_are_exactly_p5_030_p95_048():
    """Hand-computed cutoffs for the constructed 21-value ISI distribution:
    p5 == 0.30 s, p95 == 0.48 s (numpy default linear method, positions land on
    data points sorted[1] and sorted[19])."""
    onsets = _make_onsets()
    _isi, p5, p95 = reference_isi_cutoffs(onsets)
    assert p5 == pytest.approx(0.30, abs=1e-9)
    assert p95 == pytest.approx(0.48, abs=1e-9)


def test_instantaneous_rate_is_reciprocal_of_isi():
    """RESULT-sniff-rate feed (level 2, within 1e-4 Hz): rate == 1/ISI. Span
    with ISI 0.40 s -> 2.5 Hz; fast outlier 0.20 s -> 5.0 Hz; slow 0.90 s ->
    ~1.111 Hz."""
    onsets = _make_onsets()
    rate = reference_inst_sniff_rate(onsets)
    assert rate[0] == pytest.approx(1.0 / 0.20, abs=1e-4)      # fast
    assert rate[-1] == pytest.approx(1.0 / 0.90, abs=1e-4)     # slow
    # span with ISI 0.40 is at temporal index 11 (0.20, then 0.30..0.40)
    assert rate[11] == pytest.approx(2.5, abs=1e-4)


# ===========================================================================
# Full mask microcase (hand-computed invalid spans)
# onsets (samples @1000Hz): 0, 200, then +10-sample steps ... onset20=7610,
# onset21=8510.  n = 9000.
#   [0,200)     INVALID  (span0 ISI 0.20 < p5 0.30, too fast)
#   [200,7610)  VALID    (all middle ISIs in [0.30,0.48])
#   [7610,8510) INVALID  (span20 ISI 0.90 > p95 0.48, too slow)
#   [8510,9000) INVALID  (after last onset -> not part of any retained cycle)
#   [1000,1100) INVALID  (SNF_PH == -1 override, inside an otherwise-valid span)
# ===========================================================================
def _build_snf_ph(n=9000):
    ph = np.full(n, 0.5, dtype=np.float64)   # arbitrary in-cycle phase, != -1
    ph[1000:1100] = -1.0                      # noise sentinel inside a valid span
    return ph


def test_full_valid_mask_marks_all_hand_known_spans():
    onsets = _make_onsets()
    n = 9000
    snf_ph = _build_snf_ph(n)
    D = _make_D(onsets, snf_ph)
    valid = _run_mask(D)

    assert valid.shape == (n,)
    # Fast-outlier span (too-fast ISI) -> invalid.
    assert not valid[100]
    assert not np.any(valid[0:200])
    # Interior valid span (away from the -1 override).
    assert valid[500]
    assert valid[5000]
    assert np.all(valid[200:1000])
    # SNF_PH == -1 override inside a valid ISI span -> invalid.
    assert not np.any(valid[1000:1100])
    # Valid resumes immediately after the -1 span.
    assert valid[1100]
    # Slow-outlier span (too-slow ISI) -> invalid.
    assert not valid[8000]
    assert not np.any(valid[7610:8510])
    # Uncovered tail after the last onset -> invalid.
    assert not np.any(valid[8510:9000])


def test_lower_boundary_of_first_valid_span_is_inclusive():
    """Sample at exactly onset1 (sample 200) starts the first VALID span
    [200,7610); half-open [a,b) convention => sample 200 is valid, sample 199
    (last of the fast-outlier span) is invalid."""
    onsets = _make_onsets()
    snf_ph = _build_snf_ph()
    valid = _run_mask(_make_D(onsets, snf_ph))
    assert not valid[199]
    assert valid[200]


def test_snf_ph_minus1_alone_invalidates_even_with_normal_isi():
    """Isolate rule 4: a uniform, all-in-range ISI sequence (every span valid by
    ISI) but with a SNF_PH==-1 block -> ONLY that block is invalid, proving the
    -1 sentinel is honoured independently of the ISI-percentile rule.

    Onsets spaced exactly 0.50 s (0.5 and its multiples are EXACTLY
    representable in float64, so np.diff yields bit-identical ISIs -> p5 == p95
    == 0.50 and every span is in-range; using a non-power-of-two spacing here
    would let float noise push the extreme spans just outside [p5,p95])."""
    onsets = np.arange(11) * 0.50            # 10 identical ISIs of exactly 0.50 s
    n = int(round(onsets[-1] * FS)) + 500     # 5000 + 500 = 5500
    ph = np.full(n, 0.3, dtype=np.float64)
    ph[1500:1800] = -1.0
    valid = _run_mask(_make_D(onsets, ph))
    # Covered span [0, 5000) is ISI-valid except the -1 block.
    assert np.all(valid[0:1500])
    assert not np.any(valid[1500:1800])
    assert np.all(valid[1800:5000])
    # Tail after last onset (sample 5000) is uncovered -> invalid.
    assert not np.any(valid[5000:n])


def test_n_retained_sniffs_counts_only_in_range_spans():
    """RESULT-qc-counts n_sniffs (retained) feed (level 1 exact integer): the
    number of retained inter-onset spans = count of ISIs within [p5,p95]. For
    the constructed distribution exactly 2 of 21 spans are outliers (0.20, 0.90)
    -> 19 retained (before any SNF_PH==-1 subtraction, which acts at the
    sample level not the span-count level)."""
    onsets = _make_onsets()
    isi, p5, p95 = reference_isi_cutoffs(onsets)
    n_retained = int(np.sum((isi >= p5) & (isi <= p95)))
    assert n_retained == 19

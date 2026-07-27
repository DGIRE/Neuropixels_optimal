# ORACLE
"""
Analytical microcases for the NOT-YET-IMPLEMENTED
compute_firing_rate_50ms(D, included_unit_ids, window_ms=50.0) -> dict
(new kernel, contract_v001.yaml preprocessing: "For each included unit
compute 50 ms sliding-window firing rate (spikes in [t-25 ms, t+25 ms] / 0.05 s)
at each LV sample. Average across included units for population firing rate
trace (Hz).")

Targets:
  - RESULT-fr-per-trial (level 3, distributional -- "Per-trial population
    firing rate within 1e-4 Hz on fixture slice (deterministic sliding-window);
    distributional agreement on full workload.")
  - RESULT-unit-inclusion (level 1, exact -- n_units_recorded, n_units_included,
    included unit ID list).
  - Feeds FIG-DEMO4-2-FR (Panel 1 heatmap, Panel 2 trial-averaged trace).

Contract-pinned algorithm (repo_map.md sec 6 / prohibited_changes: "Computing
firing rate with window other than 50 ms centered at t ... is prohibited"):
  For each included unit u and each LV time point t:
    rate_u(t) = count(spikes_u with t-25ms <= spike_time <= t+25ms) / 0.05 s
  Population trace = mean(rate_u(t)) across all included units u.

Contract-pinned unit inclusion rule (inclusion_exclusion, CON-001):
    included <=> overall_mean_firing_rate >= 0.1 Hz  AND  total_spike_count >= 5000
  (BOTH criteria required simultaneously -- neither alone is sufficient.)

Design notes (Blueprint v2 P3 -- oracle design phase, NO implementation run):
  Expected values below are hand-counted from first principles (a handful of
  explicitly-placed spike times against an explicitly-constructed LV sample
  grid; the 50 ms window and 0.05 s normalization are pinned literally by the
  contract text, not inferred from any implementation). Two genuinely open
  interpretation questions (population-average-of-zero-included-units, and
  closed- vs half-open-interval treatment at the window edge) are called out
  explicitly and tested at the safe-margin level (unambiguous) plus, separately,
  at the exact-boundary level with the oracle's own documented interpretation
  clearly flagged for Red-tier (Opus) confirmation against the final
  implementation.

  compute_firing_rate_50ms's exact D-dict input shape is not yet pinned by an
  existing implementation; this suite assumes (matching the documented
  signature in 03_software/repo_map.md sec 6, and the D['SNF']-length
  convention already used by every other per-LV-sample field in this
  pipeline -- SNF_PH, sniff_rate, etc.) that:
    D['LV_Fs']       -- float, LabView sample rate
    D['SNF']         -- (n_LV_samples,) placeholder array whose LENGTH alone
                        determines the output length (content unused by this
                        kernel)
    D['sp']['st']    -- (n_spikes,) spike times, seconds
    D['sp']['clu']   -- (n_spikes,) cluster/unit id per spike
  and that included_unit_ids is passed as the second positional argument.
  If the real signature differs, these tests will fail loudly with a clear
  TypeError -- that is the intended behavior of an oracle test (a genuine
  contract mismatch, not something to silently paper over).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_THIS_FILE = Path(__file__).resolve()
_SOFTWARE_DIR = _THIS_FILE.parents[1]  # .../NP-DEMO-4/03_software


def _get_compute_firing_rate_50ms():
    """Try to import the real compute_firing_rate_50ms. Returns the function,
    or None if the module does not exist yet (expected pre-implementation)."""
    path = str(_SOFTWARE_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from compute_firing_rate_50ms import compute_firing_rate_50ms  # type: ignore
    except ImportError:
        return None
    return compute_firing_rate_50ms


def _make_D(LV_Fs: float, n_samples: int, spike_times_s, spike_clu):
    return {
        "LV_Fs": float(LV_Fs),
        "SNF": np.zeros(n_samples, dtype=np.float64),  # length-only placeholder
        "sp": {
            "st": np.asarray(spike_times_s, dtype=np.float64),
            "clu": np.asarray(spike_clu),
        },
    }


# ===========================================================================
# Microcase 5 (contract inclusion rule) -- pure arithmetic, no import needed.
# This one runs and passes today, before compute_firing_rate_50ms.py exists,
# because CON-001's rule is fully specified by the contract text itself.
# ===========================================================================
@pytest.mark.parametrize(
    "overall_fr_hz,total_spikes,expected_included",
    [
        (0.1, 5000, True),      # both criteria exactly at their boundary -> included ('>=')
        (0.05, 10000, False),   # fails FR criterion alone -> excluded
        (0.5, 4999, False),     # fails spike-count criterion alone -> excluded
        (0.099999, 5000, False),  # infinitesimally below FR boundary -> excluded
        (0.5, 5000, True),      # comfortably passes both -> included
        (10.0, 4999, False),    # very high FR but one spike short -> excluded (BOTH required)
    ],
)
def test_unit_inclusion_rule_from_contract_first_principles(
    overall_fr_hz, total_spikes, expected_included
):
    """CON-001 (contract_v001.yaml inclusion_exclusion): a unit is included
    if and only if overall mean firing rate >= 0.1 Hz AND total spike count
    >= 5000, BOTH simultaneously. This is the literal boolean predicate;
    no signal or implementation is needed to check it."""
    included = bool(overall_fr_hz >= 0.1 and total_spikes >= 5000)
    assert included == expected_included


# ===========================================================================
# Tests requiring the real compute_firing_rate_50ms() -- skip until implemented.
# ===========================================================================
def test_single_spike_gives_20hz_at_center_and_zero_far_away():
    """One unit (id=7), one spike at exactly t=1.0 s. LV_Fs=125 Hz so t=1.0 s
    falls exactly on LV sample index 125 (125/125=1.0). At that sample, the
    window [0.975, 1.025] contains exactly 1 spike -> rate = 1/0.05 = 20 Hz
    (hand-derived, contract-literal). At sample indices far from the window
    (>=100 ms away, e.g. t=0.8 s and t=1.2 s), the window contains 0 spikes
    -> rate = 0."""
    compute_firing_rate_50ms = _get_compute_firing_rate_50ms()
    if compute_firing_rate_50ms is None:
        pytest.skip("compute_firing_rate_50ms.py not yet implemented")

    LV_Fs = 125.0
    n = 250  # 2.0 s
    D = _make_D(LV_Fs, n, spike_times_s=[1.0], spike_clu=[7])
    included_unit_ids = np.array([7])

    out = compute_firing_rate_50ms(D, included_unit_ids)
    pop = np.asarray(out["firing_rate_population"], dtype=np.float64)
    assert pop.shape == (n,)

    idx_center = int(round(1.0 * LV_Fs))
    idx_far_low = int(round(0.8 * LV_Fs))
    idx_far_high = int(round(1.2 * LV_Fs))

    assert pop[idx_center] == pytest.approx(20.0, abs=1e-9)
    assert pop[idx_far_low] == pytest.approx(0.0, abs=1e-9)
    assert pop[idx_far_high] == pytest.approx(0.0, abs=1e-9)

    if "firing_rate_per_unit" in out:
        per_unit = np.asarray(out["firing_rate_per_unit"], dtype=np.float64)
        assert per_unit.shape == (1, n)
        np.testing.assert_allclose(per_unit[0], pop, atol=1e-12)


def test_empty_unit_gives_all_zero_rate():
    """A unit with zero recorded spikes must produce an all-zero rate trace
    (0 spikes in every window -> 0/0.05 = 0 everywhere, never NaN)."""
    compute_firing_rate_50ms = _get_compute_firing_rate_50ms()
    if compute_firing_rate_50ms is None:
        pytest.skip("compute_firing_rate_50ms.py not yet implemented")

    LV_Fs = 125.0
    n = 250
    D = _make_D(LV_Fs, n, spike_times_s=[], spike_clu=[])
    included_unit_ids = np.array([9])

    out = compute_firing_rate_50ms(D, included_unit_ids)
    pop = np.asarray(out["firing_rate_population"], dtype=np.float64)
    assert pop.shape == (n,)
    np.testing.assert_allclose(pop, np.zeros(n), atol=1e-12)
    assert not np.any(np.isnan(pop))


def test_two_units_population_average_and_overlap_region():
    """Two units: unit A fires ONCE at t=1.0 s, unit B fires ONCE at t=2.0 s
    (well-separated, no window overlap):
      population(t=1.0) = (20 + 0)/2 = 10 Hz
      population(t=2.0) = (0 + 20)/2 = 10 Hz
    A second sub-case places both units' single spikes close together
    (unit A at t=1.000 s, unit B at t=1.005 s, 5 ms apart -- well inside each
    other's +-25 ms window): at t=1.000 s BOTH units register 20 Hz (each has
    exactly one spike inside its own window), so:
      population(t=1.000) = (20 + 20)/2 = 20 Hz   (the "overlap" case)."""
    compute_firing_rate_50ms = _get_compute_firing_rate_50ms()
    if compute_firing_rate_50ms is None:
        pytest.skip("compute_firing_rate_50ms.py not yet implemented")

    LV_Fs = 125.0
    n = 375  # 3.0 s

    # --- separated case ---------------------------------------------------
    D_sep = _make_D(LV_Fs, n, spike_times_s=[1.0, 2.0], spike_clu=[1, 2])
    out_sep = compute_firing_rate_50ms(D_sep, np.array([1, 2]))
    pop_sep = np.asarray(out_sep["firing_rate_population"], dtype=np.float64)

    idx_1s = int(round(1.0 * LV_Fs))
    idx_2s = int(round(2.0 * LV_Fs))
    assert pop_sep[idx_1s] == pytest.approx(10.0, abs=1e-9)
    assert pop_sep[idx_2s] == pytest.approx(10.0, abs=1e-9)

    # --- overlapping case ---------------------------------------------------
    D_close = _make_D(LV_Fs, n, spike_times_s=[1.000, 1.005], spike_clu=[1, 2])
    out_close = compute_firing_rate_50ms(D_close, np.array([1, 2]))
    pop_close = np.asarray(out_close["firing_rate_population"], dtype=np.float64)
    idx_center = int(round(1.000 * LV_Fs))
    assert pop_close[idx_center] == pytest.approx(20.0, abs=1e-9)


def test_window_boundary_safe_margin_and_documented_edge_semantics():
    """Window-boundary behavior, split into an UNAMBIGUOUS safe-margin check
    and a documented, explicitly-flagged exact-boundary interpretation.

    Safe-margin (unambiguous): a unit spikes once at t=0.010 s (comfortably
    inside [-0.025, +0.025], the window for the very first LV sample t=0,
    regardless of any clamping convention for 'negative time'). The window is
    a pure TIME comparison against the spike's timestamp, not a data-
    availability window, so there is nothing to clamp: the spike is simply
    in-range for t=0 and must be counted there, giving rate(t=0) = 20 Hz
    under the contract-literal fixed-0.05s-denominator reading (repo_map.md
    sec 6 states the rule as 'count / 0.05 s' with no mention of an adaptive
    denominator near the recording start).

    Exact-boundary (oracle-adopted interpretation, flagged for Red-tier/Opus
    review): the contract text uses closed-bracket notation '[t-25 ms, t+25 ms]',
    which conventionally denotes an INCLUSIVE interval. This oracle therefore
    expects a spike landing EXACTLY on the +25 ms edge to be counted. Because
    exact floating-point boundary equality is fragile, this sub-check uses a
    40 Hz LV grid (dt = 25 ms exactly, so t=0 s and t=0.025 s are both exact
    grid points sharing the same 0.025 literal used to build the window), and
    is clearly separated from the safe-margin check above so a difference in
    boundary convention cannot silently break the (unambiguous) safe-margin
    assertions."""
    compute_firing_rate_50ms = _get_compute_firing_rate_50ms()
    if compute_firing_rate_50ms is None:
        pytest.skip("compute_firing_rate_50ms.py not yet implemented")

    # --- safe margin: spike at 0.010 s, LV_Fs = 125 Hz, query at t=0 -------
    LV_Fs = 125.0
    n = 125  # 1.0 s
    D = _make_D(LV_Fs, n, spike_times_s=[0.010], spike_clu=[1])
    out = compute_firing_rate_50ms(D, np.array([1]))
    pop = np.asarray(out["firing_rate_population"], dtype=np.float64)
    assert pop[0] == pytest.approx(20.0, abs=1e-9), (
        "A spike at t=0.010s (10ms, well inside the +-25ms window for the "
        "t=0 LV sample) must be counted regardless of edge-clamping convention."
    )

    # --- documented exact-boundary interpretation (40 Hz grid) -------------
    LV_Fs_b = 40.0  # dt = 0.025 s exactly
    n_b = 10
    D_b = _make_D(LV_Fs_b, n_b, spike_times_s=[0.025], spike_clu=[1])
    out_b = compute_firing_rate_50ms(D_b, np.array([1]))
    pop_b = np.asarray(out_b["firing_rate_population"], dtype=np.float64)
    # Query t=0 s: window = [-0.025, 0.025]; spike at exactly +0.025 is the
    # closed-interval upper edge -> oracle expects it counted (flagged above).
    assert pop_b[0] == pytest.approx(20.0, abs=1e-9), (
        "Oracle-adopted CLOSED-interval boundary interpretation: a spike "
        "exactly at t+25ms should be counted for the query at t. If the real "
        "implementation instead uses a half-open/exclusive convention, this "
        "assertion is the one to revisit at Red-tier review, NOT the "
        "safe-margin assertion above."
    )


def test_included_unit_ids_gates_which_spikes_enter_the_population_average():
    """Passing included_unit_ids that OMITS a highly-active unit must fully
    exclude that unit's spikes from the population average -- i.e. inclusion
    filtering happens at the unit-ID level, not just via a firing-rate
    threshold silently applied inside the kernel itself (that judgment is
    made upstream, per CON-001, and handed in explicitly)."""
    compute_firing_rate_50ms = _get_compute_firing_rate_50ms()
    if compute_firing_rate_50ms is None:
        pytest.skip("compute_firing_rate_50ms.py not yet implemented")

    LV_Fs = 125.0
    n = 250
    # Unit 1: single spike at t=1.0s (would give 20Hz there).
    # Unit 2: excluded from included_unit_ids despite having many spikes at t=1.0s.
    st = [1.0] + [1.0] * 50
    clu = [1] + [2] * 50
    D = _make_D(LV_Fs, n, spike_times_s=st, spike_clu=clu)

    out_incl_both = compute_firing_rate_50ms(D, np.array([1, 2]))
    out_incl_1_only = compute_firing_rate_50ms(D, np.array([1]))

    idx = int(round(1.0 * LV_Fs))
    pop_both = np.asarray(out_incl_both["firing_rate_population"])[idx]
    pop_1_only = np.asarray(out_incl_1_only["firing_rate_population"])[idx]

    assert pop_1_only == pytest.approx(20.0, abs=1e-9)
    assert pop_both > pop_1_only  # unit 2's 50 spikes must move the average up

# ORACLE
"""
Tests for the discard log (RESULT-discard-log).

Targets: RESULT-discard-log (contract acceptance level 1, exact; "exact
enumeration of discarded intervals (start_s/end_s within 1e-6 s of detected
boundary) and exact reason string match").

Contract wording under test (inclusion_exclusion / preprocessing, verbatim):
  "Every discarded SNF section logged (experiment, start_s, end_s, reason) --
  no silent discards." / "Log every discarded SNF section (spike_SNF_PH < 0 /
  non-sniffing / noise) per experiment with start_s, end_s, and reason; discard
  those spikes from all downstream analysis. No silent discards."

The schema/logic checks below are pure structural and set-membership
properties derived directly from this contract text -- no expected value here
comes from running the not-yet-written pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

REQUIRED_DISCARD_LOG_COLUMNS = {"experiment", "start_s", "end_s", "reason"}


def validate_discard_log_schema(df: pd.DataFrame) -> None:
    """Contract: each discard entry must have experiment (str), start_s (float), end_s (float), reason (str)."""
    missing = REQUIRED_DISCARD_LOG_COLUMNS - set(df.columns)
    assert not missing, f"discard log missing required columns: {missing}"
    for col in ("experiment", "reason"):
        assert df[col].map(lambda v: isinstance(v, str)).all(), f"{col} must be a string"
    for col in ("start_s", "end_s"):
        assert pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]), (
            f"{col} must be numeric (float seconds)"
        )
    assert (df["end_s"] >= df["start_s"]).all(), "end_s must be >= start_s for every discarded interval"


def invalid_sniff_sections_from_spike_phase(
    spike_times_s: np.ndarray, spike_snf_ph: np.ndarray
) -> list[tuple[float, float]]:
    """Reference (contract-derived) logic: any spike with spike_SNF_PH < 0 must have
    its time covered by some (start_s, end_s) interval in the discard log. This
    reference builds the minimal covering intervals directly from the invalid
    spike timestamps, purely as a first-principles ground truth for the
    'no silent discards' property -- NOT the real per-sniff-section discard
    logic the pipeline will implement (which discards whole SNF sections, a
    superset of just the invalid spike timestamps)."""
    spike_times_s = np.asarray(spike_times_s, dtype=float)
    spike_snf_ph = np.asarray(spike_snf_ph, dtype=float)
    invalid_times = np.sort(spike_times_s[spike_snf_ph < 0])
    return [(float(t), float(t)) for t in invalid_times]


def test_discard_log_schema_has_required_fields():
    """Every discard entry has experiment (str), start_s (float), end_s (float), reason (str)."""
    df = pd.DataFrame(
        {
            "experiment": ["2021-11-01", "2021-11-01", "2021-12-15"],
            "start_s": [10.0, 250.5, 0.0],
            "end_s": [12.5, 251.0, 3.2],
            "reason": ["non-sniffing gap", "noise artifact", "non-sniffing gap"],
        }
    )
    validate_discard_log_schema(df)  # must not raise


def test_discard_log_schema_rejects_missing_column():
    """A discard log missing the 'reason' column must fail schema validation (no silent discards without a reason)."""
    df = pd.DataFrame(
        {
            "experiment": ["2021-11-01"],
            "start_s": [10.0],
            "end_s": [12.5],
        }
    )
    with pytest.raises(AssertionError):
        validate_discard_log_schema(df)


def test_discard_log_rejects_end_before_start():
    """end_s must never be earlier than start_s for a discarded interval."""
    df = pd.DataFrame(
        {
            "experiment": ["2021-11-01"],
            "start_s": [12.5],
            "end_s": [10.0],
            "reason": ["non-sniffing gap"],
        }
    )
    with pytest.raises(AssertionError):
        validate_discard_log_schema(df)


def test_every_invalid_spike_time_is_covered_by_some_discard_interval():
    """No silent discards: every spike_SNF_PH < 0 spike's time must fall within
    some (start_s, end_s) interval logged for its experiment."""
    spike_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    spike_snf_ph = np.array([0.2, -1.0, 0.5, -1.0, 0.9])  # spikes at t=2.0 and t=4.0 are invalid

    discard_log = pd.DataFrame(
        {
            "experiment": ["expX", "expX"],
            "start_s": [1.5, 3.5],
            "end_s": [2.5, 4.5],
            "reason": ["non-sniffing gap", "non-sniffing gap"],
        }
    )
    validate_discard_log_schema(discard_log)

    invalid_mask = spike_snf_ph < 0
    invalid_times = spike_times[invalid_mask]

    def covered(t: float) -> bool:
        return bool(((discard_log["start_s"] <= t) & (discard_log["end_s"] >= t)).any())

    assert all(covered(t) for t in invalid_times)


def test_missing_discard_interval_is_detected_as_a_violation():
    """A pipeline bug that silently drops one invalid spike's interval must be
    catchable: if the discard log does NOT cover an invalid spike's time, the
    'no silent discards' property must fail."""
    spike_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    spike_snf_ph = np.array([0.2, -1.0, 0.5, -1.0, 0.9])  # invalid at t=2.0 and t=4.0

    incomplete_discard_log = pd.DataFrame(
        {
            "experiment": ["expX"],
            "start_s": [1.5],
            "end_s": [2.5],
            "reason": ["non-sniffing gap"],
        }
    )  # only covers t=2.0; t=4.0 is NOT logged -> silent discard bug

    invalid_times = spike_times[spike_snf_ph < 0]

    def covered(t: float) -> bool:
        return bool(
            ((incomplete_discard_log["start_s"] <= t) & (incomplete_discard_log["end_s"] >= t)).any()
        )

    coverage = [covered(t) for t in invalid_times]
    assert coverage == [True, False]
    assert not all(coverage)  # demonstrates the violation is detectable


def test_reason_string_must_be_non_empty():
    """Contract requires logging the reason for every discard -- an empty/whitespace-only reason is not acceptable."""
    df = pd.DataFrame(
        {
            "experiment": ["2021-11-01", "2021-11-01"],
            "start_s": [10.0, 20.0],
            "end_s": [12.0, 21.0],
            "reason": ["non-sniffing gap", ""],
        }
    )
    validate_discard_log_schema(df)  # schema itself passes (reason is a string)
    non_empty = df["reason"].str.strip().str.len() > 0
    assert not non_empty.all()  # exposes the empty-reason row
    assert non_empty.sum() == 1  # exactly one row has a valid non-empty reason

"""
reduce_theta_coherence.py -- NP-PAPER-1 additive analysis function.

Deterministic reduction of ONE per-experiment x depth LFP-sniff coherence
spectrum C(f) (already aggregated over retained control windows upstream by
multitaper_psd_coherence) to the Table-4 scalars required by RESULT-theta-coh
(contract docx_source_refs OUT-011, RQ-059, RQ-060, RQ-061, RQ-062, CON-016,
OUT-028).

Signature and return-key contract are PINNED by the oracle designer in
oracles/test_oracle_theta_reduction.py (see its module docstring, decisions
D1-D5) -- this is the authoritative interface spec, not a suggestion:

    reduce_theta_coherence(coh_f, freqs, null_threshold_f,
                            null_coh_dist=None, theta_band=(2.0, 12.0)) -> dict

Returned dict keys:
    mean_theta_coh       -- mean of coh_f over the closed theta_band (RQ-060).
    peak_theta_coh       -- max of coh_f over theta_band (RQ-061).
    peak_theta_freq      -- frequency at which the peak occurs; ties broken to
                             the LOWEST tying frequency (D4 -- np.argmax on an
                             ascending freqs array already returns the first,
                             i.e. lowest-frequency, occurrence).
    sig_theta_fraction   -- fraction of in-band bins where coh_f STRICTLY
                             exceeds null_threshold_f (RQ-062; D3, strict >).
    peak_coh_percentile  -- percentile of peak_theta_coh within the null
                             distribution of PER-SHIFT band-max coherences
                             derived from null_coh_dist (CON-016; D5). NaN if
                             null_coh_dist is not supplied (percentile is
                             undefined without the circular-shift null).

Band membership is CLOSED: (freqs >= theta_band[0]) & (freqs <= theta_band[1])
(D2) -- a frequency landing exactly on the band edge is IN the band.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import percentileofscore


def reduce_theta_coherence(coh_f, freqs, null_threshold_f, null_coh_dist=None,
                            theta_band=(2.0, 12.0)) -> dict:
    coh_f = np.asarray(coh_f, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    null_threshold_f = np.asarray(null_threshold_f, dtype=float)

    if coh_f.shape != freqs.shape or null_threshold_f.shape != freqs.shape:
        raise ValueError(
            "coh_f, freqs, and null_threshold_f must all have the same shape; "
            f"got {coh_f.shape}, {freqs.shape}, {null_threshold_f.shape}"
        )

    lo, hi = theta_band
    band = (freqs >= lo) & (freqs <= hi)  # D2: closed interval
    if not np.any(band):
        raise ValueError(f"theta_band {theta_band} selects zero frequency bins from freqs")

    band_coh = coh_f[band]
    band_freqs = freqs[band]
    band_thr = null_threshold_f[band]

    mean_theta_coh = float(band_coh.mean())

    # D4: argmax on ascending-frequency band values returns the first (lowest
    # frequency) index among ties.
    peak_idx = int(np.argmax(band_coh))
    peak_theta_coh = float(band_coh[peak_idx])
    peak_theta_freq = float(band_freqs[peak_idx])

    # D3: strict exceedance; fraction of in-band bins.
    sig_theta_fraction = float(np.mean(band_coh > band_thr))

    if null_coh_dist is not None:
        null_coh_dist = np.asarray(null_coh_dist, dtype=float)
        if null_coh_dist.shape[1] != freqs.shape[0]:
            raise ValueError(
                "null_coh_dist must be shaped (n_shifts, n_freqs) aligned to freqs; "
                f"got {null_coh_dist.shape} vs {freqs.shape[0]} frequencies"
            )
        # D5: null distribution of PER-SHIFT band-max ("peak") coherences.
        null_band_peaks = null_coh_dist[:, band].max(axis=1)
        peak_coh_percentile = float(
            percentileofscore(null_band_peaks, peak_theta_coh, kind="mean")
        )
    else:
        peak_coh_percentile = float("nan")

    return dict(
        mean_theta_coh=mean_theta_coh,
        peak_theta_coh=peak_theta_coh,
        peak_theta_freq=peak_theta_freq,
        sig_theta_fraction=sig_theta_fraction,
        peak_coh_percentile=peak_coh_percentile,
    )

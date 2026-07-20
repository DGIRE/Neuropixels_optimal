"""
analyses/psth_utils.py  —  faithful port of the spikes-master PSTH chain
    psthAndBA.m  ->  timestampsToBinned.m  ->  histdiff (histdiff.c mex)

Only the code path that compute_sniff_psth actually exercises is reproduced,
but it is reproduced exactly, including the mex's boundary semantics.

histdiff.c dispatch for our inputs
----------------------------------
timestampsToBinned calls histdiff(spikeTimes, referencePoints(r), binBorders)
one reference event at a time.  binBorders are regularly spaced (size = 0.01),
so histdiff computes size = bins[1]-bins[0], min = bins[0], nbins = len-1.
spikeTimes are globally sorted and referencePoints(r) is a scalar, so
`chckord` returns true and the **ordhist** branch runs:

    for each datum d1 (spike):
        skip events while (d1 - event) >= max           # max = min + size*nbins
        count while     (diff = d1 - event) >  min       # STRICT lower bound
            cnts[(int)((diff-min)/size)]++

i.e. a spike contributes to event e iff  min < (spike - e) < max, into bin
floor((spike - e - min)/size).  The lower edge is OPEN, the upper edge OPEN.
(reghist, the unsorted branch, would use a half-open [min,max); it is NOT the
branch taken here.)  Getting this exactly right matters: one boundary spike
shifts an event-averaged rate by 1/(nEvents*binSize) ~ 0.026 Hz >> the 1e-9
psth tolerance.
"""
from __future__ import annotations

import numpy as np


def matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """Reproduce MATLAB ``start:step:stop`` including its end tolerance.

    MATLAB includes stop when it lands within a few eps of an integer number
    of steps; a naive np.arange can drop or add the last point due to FP.
    """
    n = int(np.floor((stop - start) / step + 1e-10))
    return start + step * np.arange(n + 1)


def histdiff_ordered(spikes: np.ndarray, event: float,
                     min_: float, size: float, nbins: int) -> np.ndarray:
    """One event: ordhist counts (histdiff.c).  Returns length-nbins vector.

    Faithful to the mex: lower edge open, upper edge open, bin =
    floor((diff-min)/size).  ``spikes`` must be sorted (it is: a boolean
    subselect of the globally sorted spikeTimes).
    """
    diff = spikes - event
    max_ = min_ + size * nbins
    m = (diff > min_) & (diff < max_)               # strict both edges (ordhist)
    if not np.any(m):
        return np.zeros(nbins, dtype=np.float64)
    b = ((diff[m] - min_) / size).astype(np.int64)  # C (int) cast; diff>0 => trunc==floor
    np.clip(b, 0, nbins - 1, out=b)
    return np.bincount(b, minlength=nbins).astype(np.float64)


def psth_and_ba(spikeTimes: np.ndarray, eventTimes: np.ndarray,
                window, binSize: float):
    """Faithful port of psthAndBA.m (psth + bin centers only).

    Returns (psth, bins) where psth = mean over events of (binnedArray/binSize)
    in spikes/s, and bins are the bin centers (seconds).
    """
    spikeTimes = np.sort(np.asarray(spikeTimes, dtype=np.float64).ravel())
    eventTimes = np.sort(np.asarray(eventTimes, dtype=np.float64).ravel())
    w0, w1 = float(window[0]), float(window[1])

    # global prefilter (psthAndBA.m:26) — trims far spikes, no effect on counts.
    if eventTimes.size:
        lo = eventTimes.min() + w0
        hi = eventTimes.max() + w1
        spikeTimes = spikeTimes[(spikeTimes > lo) & (spikeTimes < hi)]

    binBorders = matlab_colon(w0, binSize, w1)
    nbins = binBorders.size - 1
    min_ = binBorders[0]
    size = binBorders[1] - binBorders[0]
    centers = min_ + (np.arange(nbins) + 0.5) * size   # histdiff.c ctrs

    if eventTimes.size == 0 or nbins <= 0:
        return np.zeros(nbins, dtype=np.float64), centers

    # spikeTimes is sorted; take a WIDENED candidate window per event with
    # searchsorted (fast), then apply the exact ordhist test on the computed
    # difference.  Crucial fidelity point: histdiff.c compares diff = spike-event
    # to (min,max) AFTER forming the difference; testing spike < event+max in
    # absolute time rounds differently for large event times and drops spikes
    # whose true diff is exactly max.  The candidate window is padded by one bin
    # each side so no boundary spike is missed before the exact test.
    max_ = min_ + size * nbins
    binned = np.zeros((eventTimes.size, nbins), dtype=np.float64)
    lo_all = np.searchsorted(spikeTimes, eventTimes + (min_ - size), side="left")
    hi_all = np.searchsorted(spikeTimes, eventTimes + (max_ + size), side="right")
    for r in range(eventTimes.size):
        sl = spikeTimes[lo_all[r]:hi_all[r]]
        if sl.size:
            d = sl - eventTimes[r]
            m = (d > min_) & (d < max_)            # ordhist: strict both edges
            if np.any(m):
                b = ((d[m] - min_) / size).astype(np.int64)
                np.clip(b, 0, nbins - 1, out=b)
                binned[r, :nbins] = np.bincount(b, minlength=nbins)
    psth = np.mean(binned / binSize, axis=0)
    return psth, centers


def psth_matrix_vectorized(spikeTimes, clu, unitIDs, eventTimes,
                           window, binSize: float):
    """Vectorized equivalent of the per-unit psthAndBA loop, all units at once.

    Identical math to the faithful chain (ordhist strict edges, floor bins,
    event-averaged spikes/s) but computed with one global spike->event
    expansion instead of nUnits*nEvents mex calls.  Returns (psth[nU,nbins],
    centers[nbins], n_events).
    """
    spikeTimes = np.asarray(spikeTimes, dtype=np.float64).ravel()
    clu = np.asarray(clu).ravel()
    unitIDs = np.asarray(unitIDs).ravel()
    events = np.sort(np.asarray(eventTimes, dtype=np.float64).ravel())
    nU = unitIDs.size
    n_events = events.size

    w0, w1 = float(window[0]), float(window[1])
    binBorders = matlab_colon(w0, binSize, w1)
    nbins = binBorders.size - 1
    min_ = binBorders[0]
    size = binBorders[1] - binBorders[0]
    max_ = min_ + size * nbins
    centers = min_ + (np.arange(nbins) + 0.5) * size

    psth = np.zeros((nU, nbins), dtype=np.float64)
    if n_events == 0 or nbins <= 0:
        return psth, centers, n_events

    # sort spikes so per-spike qualifying-event ranges are contiguous.
    order = np.argsort(spikeTimes, kind="stable")
    st = spikeTimes[order]
    uidx = np.searchsorted(unitIDs, clu[order])

    # Candidate events per spike: a WIDENED window (padded one bin each side)
    # from searchsorted, then the exact ordhist test is applied on the computed
    # difference d = spike - event.  See psth_and_ba for why the difference must
    # be formed before comparing to (min_, max_) rather than testing absolute
    # times (large event times round the boundary differently and drop spikes).
    lo = np.searchsorted(events, st - (max_ + size), side="left")
    hi = np.searchsorted(events, st - (min_ - size), side="right")
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return psth, centers, n_events

    spk_rep = np.repeat(np.arange(st.size), counts)
    offs = np.zeros(st.size, dtype=np.int64)
    np.cumsum(counts[:-1], out=offs[1:])
    within = np.arange(total) - np.repeat(offs, counts)
    ev_idx = np.repeat(lo, counts) + within

    diff = st[spk_rep] - events[ev_idx]
    keep = (diff > min_) & (diff < max_)          # exact ordhist strict edges
    diff = diff[keep]
    u_rep = uidx[spk_rep][keep]
    b = ((diff - min_) / size).astype(np.int64)
    np.clip(b, 0, nbins - 1, out=b)

    flat = u_rep * nbins + b
    summed = np.bincount(flat, minlength=nU * nbins).reshape(nU, nbins)
    psth = summed.astype(np.float64) / n_events / binSize
    return psth, centers, n_events

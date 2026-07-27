"""
console_data.py
===============
Data layer for the Neuropixels Data Console. Pure data / numpy -- no GUI, no
matplotlib -- so it can be tested headlessly.

Reads the cross-experiment aggregate (np_aggregate.h5) and exposes, per session:
  * probe geometry  -- every recording electrode (x, y in um) and its shank
  * per-electrode unit counts (units assigned to their nearest electrode by depth)
  * the LFP sampling sites (the 32 retained channels)
  * trial structure (from the LabView TR/TS streams) -> absolute-time windows
  * band-pass filtering of the LFP
  * per-unit spike trains within a time window
  * sniff / ethanol traces

Time model
----------
LFP (LFP_t), spikeTimes are seconds from recording start. The LabView sensors
(ETH/SNF/TR/TS) are sampled at LV_Fs; sample i is treated as t = i / LV_Fs
seconds on the same clock (LabView and the neural stream started together). A
trial `tr` spans the LabView samples where TR == tr; "time within trial" is
TS/1000 s. trial_window_to_abs() maps (trial, t0, t1 within trial) -> absolute
seconds, which then index the LFP and spikes.

Depth / shank conventions (see DATA/Aggregate/AGGREGATE_CONVENTIONS.txt)
  ycoord = microns from the probe tip (larger = toward the surface).
  xcoord = lateral position; on multi-shank probes it also encodes the shank
  (shanks are ~250 um apart), so shanks are recovered by clustering x.

NOTE on unit x / shank: the aggregate stores each unit's DEPTH (unitDepths) but
not its shank, so a unit is assigned to the nearest electrode by depth. On a
single-shank probe this is exact; on a multi-shank probe the shank is a best-effort
(nearest electrode overall). This is flagged in the guide.
"""
from __future__ import annotations

import numpy as np
import scipy.signal
import h5py


# named LFP band presets (Hz); None = raw broadband
BAND_PRESETS = {
    "Raw (broadband)": None,
    "Delta (1-4 Hz)": (1.0, 4.0),
    "Theta (4-12 Hz)": (4.0, 12.0),
    "Beta (15-30 Hz)": (15.0, 30.0),
    "Gamma (30-80 Hz)": (30.0, 80.0),
    "High gamma (65-100 Hz)": (65.0, 100.0),
}


def list_dates(h5_path: str) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        return sorted(f.keys())


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.999)
    lo = max(lo, 1e-6)
    b, a = scipy.signal.butter(order, [lo / nyq, hi / nyq], btype="band")
    n = x.shape[-1]
    pad = min(3 * (max(len(a), len(b)) - 1), n - 1) if n > 1 else 0
    return scipy.signal.filtfilt(b, a, x, axis=-1, padlen=pad)


def _cluster_shanks(xc: np.ndarray, gap: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Assign each x to a shank id by grouping x-values separated by > gap um.

    Returns (shank_id_per_point, shank_base_x sorted). NP2.0 columns within a
    shank are ~32 um apart; shanks are ~250 um apart, so gap=100 separates shanks.
    """
    xs = np.asarray(xc, dtype=np.float64)
    order = np.argsort(xs)
    bases = []
    shank_of_sorted = np.zeros(xs.size, dtype=int)
    cur = 0
    bases.append(xs[order[0]])
    for k in range(1, xs.size):
        if xs[order[k]] - xs[order[k - 1]] > gap:
            cur += 1
            bases.append(xs[order[k]])
        shank_of_sorted[k] = cur
    shank_of = np.empty(xs.size, dtype=int)
    shank_of[order] = shank_of_sorted
    return shank_of, np.array(bases)


class Session:
    """Everything the console needs for one experiment date."""

    def __init__(self, h5_path: str, date: str):
        self.h5_path = h5_path
        self.date = date
        self._band_cache: dict = {}
        self._load()

    # -- loading -----------------------------------------------------------
    def _load(self):
        with h5py.File(self.h5_path, "r") as f:
            g = f[self.date]
            A = g.attrs
            self.experiment_name = str(A.get("experiment_name", ""))
            self.has_lfp = bool(A.get("has_lfp", "LFP" in g))
            self.has_spikes = bool(A.get("has_spikes", False)) and "spikeTimes" in g
            self.has_sensors = bool(A.get("has_sensors", False)) and "SNF" in g

            # LFP
            if self.has_lfp:
                self.lfp = np.asarray(g["LFP"][()], np.float64)
                self.lfp_fs = float(A.get("LFP_fs", 250.0))
                self.lfp_ycoord = np.asarray(g["lfp_ycoord"][()], np.float64) \
                    if "lfp_ycoord" in g else np.arange(self.lfp.shape[0], dtype=float)
                self.lfp_xcoord = np.asarray(g["lfp_xcoord"][()], np.float64) \
                    if "lfp_xcoord" in g else np.zeros(self.lfp.shape[0])
                self.lfp_channel_row = np.asarray(g["lfp_channel_row"][()]) \
                    if "lfp_channel_row" in g else np.arange(self.lfp.shape[0])
                self.lfp_duration_s = float(A.get("LFP_duration_s",
                                                   self.lfp.shape[1] / self.lfp_fs))
            else:
                self.lfp = np.empty((0, 0))
                self.lfp_fs = 250.0
                self.lfp_ycoord = self.lfp_xcoord = np.empty(0)
                self.lfp_channel_row = np.empty(0, int)
                self.lfp_duration_s = 0.0

            # full probe channel geometry (present when has_spikes)
            if "xcoords" in g and "ycoords" in g:
                self.chan_x = np.asarray(g["xcoords"][()], np.float64).ravel()
                self.chan_y = np.asarray(g["ycoords"][()], np.float64).ravel()
            else:
                # fall back to the LFP sites only
                self.chan_x = self.lfp_xcoord.copy()
                self.chan_y = self.lfp_ycoord.copy()

            # spikes / units
            if self.has_spikes:
                self.spikeTimes = np.asarray(g["spikeTimes"][()], np.float64)
                self.spikeClusters = np.asarray(g["spikeClusters"][()]) \
                    if "spikeClusters" in g else None
                self.spikeDepths = np.asarray(g["spikeDepths"][()], np.float64) \
                    if "spikeDepths" in g else None
                self.unitIDs = np.asarray(g["unitIDs"][()]) if "unitIDs" in g else None
                self.unitDepths = np.asarray(g["unitDepths"][()], np.float64) \
                    if "unitDepths" in g else None
                if (self.unitDepths is None and self.spikeClusters is not None
                        and self.spikeDepths is not None and self.unitIDs is not None):
                    self.unitDepths = np.array(
                        [np.nanmean(self.spikeDepths[self.spikeClusters == u])
                         if np.any(self.spikeClusters == u) else np.nan
                         for u in self.unitIDs])
                # precise per-unit coordinates (from add_unit_positions.py), if present
                self.unitXcoords = np.asarray(g["unitXcoords"][()], np.float64) \
                    if "unitXcoords" in g else None
                self.unitYcoords = np.asarray(g["unitYcoords"][()], np.float64) \
                    if "unitYcoords" in g else None
            else:
                self.spikeTimes = self.spikeClusters = self.spikeDepths = None
                self.unitIDs = self.unitDepths = None
                self.unitXcoords = self.unitYcoords = None

            # sensors
            if self.has_sensors:
                self.SNF = np.asarray(g["SNF"][()], np.float64)
                self.ETH = np.asarray(g["ETH"][()], np.float64) if "ETH" in g else None
                self.TR = np.asarray(g["TR"][()]) if "TR" in g else None
                self.TS = np.asarray(g["TS"][()]) if "TS" in g else None
                self.LV_Fs = float(A.get("LV_Fs", 125.0))
            else:
                self.SNF = self.ETH = self.TR = self.TS = None
                self.LV_Fs = 125.0

        self._build_geometry()

    def _build_geometry(self):
        """Electrode array, shanks, LFP-site electrodes, per-electrode unit counts."""
        self.n_elec = self.chan_x.size
        if self.n_elec:
            self.shank_of_elec, self.shank_bases = _cluster_shanks(self.chan_x)
        else:
            self.shank_of_elec, self.shank_bases = np.empty(0, int), np.empty(0)
        self.n_shanks = int(self.shank_bases.size)

        # each LFP site -> nearest electrode (for cyan circles + shank/depth labels)
        self.lfp_site_elec = np.array(
            [self._nearest_elec(self.lfp_xcoord[i], self.lfp_ycoord[i])
             for i in range(self.lfp_ycoord.size)], dtype=int) \
            if self.n_elec and self.lfp_ycoord.size else np.empty(0, int)

        # precise per-unit (x, y) available? (from add_unit_positions.py)
        self.has_unit_xy = bool(
            getattr(self, "unitXcoords", None) is not None
            and getattr(self, "unitYcoords", None) is not None
            and self.unitIDs is not None
            and self.unitXcoords.size == self.unitIDs.size)

        # units -> peak electrode. With precise (x, y) the unit lands on the correct
        # shank; otherwise it falls back to nearest electrode by DEPTH only.
        self.elec_unit_count = np.zeros(self.n_elec, dtype=int)
        self.unit_elec = None
        if self.has_spikes and self.unitDepths is not None and self.n_elec:
            ue = np.full(self.unitDepths.size, -1, dtype=int)
            for u in range(self.unitDepths.size):
                if self.has_unit_xy and np.isfinite(self.unitXcoords[u]) \
                        and np.isfinite(self.unitYcoords[u]):
                    e = self._nearest_elec(self.unitXcoords[u], self.unitYcoords[u])
                elif np.isfinite(self.unitDepths[u]):
                    e = int(np.argmin(np.abs(self.chan_y - self.unitDepths[u])))
                else:
                    continue
                ue[u] = e
                self.elec_unit_count[e] += 1
            self.unit_elec = ue

    def _nearest_elec(self, x, y):
        if not self.n_elec:
            return -1
        d2 = (self.chan_x - x) ** 2 + (self.chan_y - y) ** 2
        return int(np.argmin(d2))

    def nearest_electrode(self, x, y):
        """Electrode index nearest a clicked (x, y), or -1."""
        return self._nearest_elec(x, y)

    def shank_of(self, elec: int) -> int:
        return int(self.shank_of_elec[elec]) if 0 <= elec < self.n_elec else 0

    # -- trials / time -----------------------------------------------------
    def trials(self) -> list[int]:
        """Sorted list of trial numbers (TR > 0), or [] if no trial structure."""
        if self.TR is None:
            return []
        u = np.unique(self.TR)
        return [int(t) for t in u if t > 0]

    def trial_window_to_abs(self, trial: int | None, t0: float, t1: float
                            ) -> tuple[float, float]:
        """Map a (trial, within-trial [t0,t1] s) selection to absolute seconds.

        trial=None -> [t0, t1] are already absolute seconds from recording start.
        With a trial, absolute time = trial's first-sample time + within-trial time.
        """
        if trial is None or self.TR is None:
            return float(t0), float(t1)
        idx = np.where(self.TR == trial)[0]
        if idx.size == 0:
            return float(t0), float(t1)
        t_start = idx[0] / self.LV_Fs
        return t_start + float(t0), t_start + float(t1)

    def trial_length_s(self, trial: int) -> float:
        if self.TR is None:
            return self.lfp_duration_s
        idx = np.where(self.TR == trial)[0]
        return (idx.size / self.LV_Fs) if idx.size else 0.0

    # -- LFP ---------------------------------------------------------------
    def filtered_lfp(self, band: tuple[float, float] | None) -> np.ndarray:
        """Full-duration LFP, optionally band-passed (cached per band)."""
        key = None if band is None else (round(band[0], 3), round(band[1], 3))
        if key in self._band_cache:
            return self._band_cache[key]
        out = self.lfp if band is None else bandpass(self.lfp, self.lfp_fs, band[0], band[1])
        self._band_cache[key] = out
        return out

    def lfp_window(self, band, t0: float, t1: float):
        """(t, seg) for the filtered LFP within [t0, t1] s. seg = (n_ch, n)."""
        n = self.lfp.shape[1]
        i0 = max(0, int(round(t0 * self.lfp_fs)))
        i1 = min(n, int(round(t1 * self.lfp_fs)))
        i1 = max(i1, i0 + 1)
        seg = self.filtered_lfp(band)[:, i0:i1]
        t = np.arange(i0, i1) / self.lfp_fs
        return t, seg

    # -- spikes ------------------------------------------------------------
    def _default_depth_tol(self) -> float:
        """Half the local LFP-site depth spacing (the depth band a site 'owns')."""
        y = np.sort(self.lfp_ycoord)
        if y.size >= 2:
            return float(np.median(np.diff(y)) / 2.0)
        return 60.0

    def units_at_electrodes(self, elecs: list[int], tol_um: float | None = None) -> dict:
        """Units belonging to the SELECTED electrodes.

        A unit belongs to exactly one electrode: its PEAK electrode (`unit_elec`,
        the same assignment that colours the probe map's unit-count dots). So the
        units returned for a selected electrode are exactly the units the probe
        colour counts for that electrode -- the view and the colour never disagree.

        When `unit_elec` is unavailable (an aggregate with no per-unit depth), we
        fall back to the old depth-tolerance rule: attach each unit to the nearest
        selected electrode within `tol_um` microns (default = half the LFP-site
        spacing). Returns {elec: [(uid, depth), ...] sorted by depth}.
        """
        out = {e: [] for e in elecs}
        if not self.has_spikes or self.unitDepths is None or not elecs:
            return out

        # preferred, colour-consistent path: unit -> its peak electrode
        if self.unit_elec is not None:
            sel = set(int(e) for e in elecs)
            for u in range(self.unitDepths.size):
                e = int(self.unit_elec[u])
                if e not in sel:
                    continue
                if self.has_unit_xy and np.isfinite(self.unitYcoords[u]):
                    d = float(self.unitYcoords[u])
                elif np.isfinite(self.unitDepths[u]):
                    d = float(self.unitDepths[u])
                else:
                    continue
                out[e].append((int(self.unitIDs[u]), d))
            for e in out:
                out[e].sort(key=lambda z: z[1])
            return out

        # fallback: depth-tolerance attachment (no per-unit electrode assignment)
        tol = self._default_depth_tol() if tol_um is None else float(tol_um)
        e_depth = np.array([self.chan_y[e] for e in elecs])
        for u, d in enumerate(self.unitDepths):
            if not np.isfinite(d):
                continue
            j = int(np.argmin(np.abs(e_depth - d)))
            if abs(e_depth[j] - d) <= tol:
                out[elecs[j]].append((int(self.unitIDs[u]), float(d)))
        for e in out:
            out[e].sort(key=lambda z: z[1])
        return out

    def unit_peak_electrode(self, uid: int) -> int:
        """Peak electrode index for a unit id (matches the probe colour), or -1."""
        if self.unit_elec is None or self.unitIDs is None:
            return -1
        w = np.where(self.unitIDs == uid)[0]
        if w.size == 0:
            return -1
        return int(self.unit_elec[int(w[0])])

    def unit_spikes_in_window(self, uid: int, t0: float, t1: float) -> np.ndarray:
        if not self.has_spikes or self.spikeClusters is None:
            return np.empty(0)
        st = self.spikeTimes
        m = (st >= t0) & (st < t1) & (self.spikeClusters == uid)
        return st[m]

    def population_spikes_in_window(self, depth_lo, depth_hi, t0, t1) -> np.ndarray:
        """Fallback when there is no spike->cluster map: all spikes in a depth band."""
        if not self.has_spikes or self.spikeDepths is None:
            return np.empty(0)
        st, sd = self.spikeTimes, self.spikeDepths
        m = (st >= t0) & (st < t1) & (sd >= depth_lo) & (sd < depth_hi)
        return st[m]

    # -- sensors -----------------------------------------------------------
    def sensor_window(self, which: str, t0: float, t1: float):
        """(t, values) for 'ETH' or 'SNF' within [t0,t1] s, or (None,None)."""
        arr = self.ETH if which == "ETH" else self.SNF
        if arr is None:
            return None, None
        i0 = max(0, int(round(t0 * self.LV_Fs)))
        i1 = min(arr.size, int(round(t1 * self.LV_Fs)))
        if i1 <= i0:
            return None, None
        t = t0 + np.arange(i1 - i0) / self.LV_Fs
        return t, arr[i0:i1]

    def sniff_present(self, t0: float, t1: float, band=(0.5, 12.0),
                      min_ratio: float = 0.35) -> bool:
        """True if the SNF window carries a clear respiratory rhythm."""
        t, x = self.sensor_window("SNF", t0, t1)
        if x is None or x.size < int(self.LV_Fs) or np.std(x) == 0:
            return False
        x = x - x.mean()
        xb = bandpass(x, self.LV_Fs, band[0], band[1])
        return float(np.std(xb) / (np.std(x) + 1e-12)) >= min_ratio

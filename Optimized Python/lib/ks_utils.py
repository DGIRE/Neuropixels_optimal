"""
ks_utils.py
-----------
Python translations of two spikes-master MATLAB toolbox functions:

  loadKSdir.m                    ->  load_ks_dir(ks_path, ...) -> dict
  ksDriftmap.m                   ->  ks_driftmap(ks_path, ...) -> tuple
  templatePositionsAmplitudes.m  ->  _template_positions_amplitudes(...) (helper)

All array indexing is 0-based (NumPy native).  Kilosort template IDs are
0-based, so where MATLAB writes `x(spikeTemplates+1)` we index `x[spike_templates]`.

Templates shape convention: [nTemplates, nTimepoints, nChannels]  (row-major,
matching what numpy.load returns for Kilosort's templates.npy directly).

FIDELITY NOTE (translation correction)
--------------------------------------
`ks_driftmap` / `_template_positions_amplitudes` are now faithful line-by-line
translations of the spikes-master MATLAB, replacing an earlier "spirit-of"
reimplementation that used the raw (still-whitened) templates and a
peak-absolute template centre-of-mass.  The golden fixtures for spikeAmps,
spikeDepths and the downstream unitAmps/unitDepths are produced by the real
MATLAB path, which:
  * amplitudes  : UN-whiten each template (temps @ winv), take the per-channel
                  peak-to-peak (max-min over time), take the max over channels,
                  index by spike template, multiply by tempScalingAmps (and by
                  params.py `gain` if present).
  * depths      : PC-feature centre of mass (first PC of pc_features.npy, with
                  negative loadings zeroed), NOT a template centre of mass.
MATLAB dtype semantics are reproduced: `tempsUnW = zeros(size(temps))` is a
DOUBLE buffer, so spikeAmps is float64; spikeDepths inherits single precision
from the float32 pc_features (`double .* single = single` in MATLAB).

Precision note (accepted, sub-tolerance difference): MATLAB evaluates the
template unwhitening `single(temps) * winv` in SINGLE precision and stores the
result in the double buffer, so its spikeAmps carry single-precision rounding.
This port unwhitens in float64 (numerically cleaner). The two agree to ~1e-7
relative, i.e. within the golden-fixture spikeAmps/unitAmps tolerance (f32,
rtol 1e-6), which is the correct a-priori tolerance for a single-precision
reference quantity. unitDepths/unitAmps are per-cluster means of these
single-precision quantities and likewise carry f32 (not 1e-12) tolerance.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

try:
    from optconfig import OPT   # optimization switches (see optconfig.set_baseline())
except ImportError:  # allow lib to be imported without the package root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from optconfig import OPT


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_params_py(params_path: str) -> dict:
    """
    Parse a Kilosort params.py file by exec()-ing it into an isolated
    namespace dict and returning that dict.

    A typical params.py looks like:
        dat_path = 'data.bin'
        n_channels_dat = 385
        dtype = 'int16'
        offset = 0
        sample_rate = 30000.0
        hp_filtered = False

    Parameters
    ----------
    params_path : str
        Full path to params.py.

    Returns
    -------
    dict
        All names defined in the file (dunder names excluded).
    """
    ns: dict = {}
    with open(params_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    exec(compile(src, params_path, "exec"), ns)  # noqa: S102
    return {k: v for k, v in ns.items() if not k.startswith("__")}


def _read_cluster_groups(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a cluster_groups.csv or cluster_group.tsv produced by Phy.

    Cluster quality label encoding (matches MATLAB convention):
        'noise'     -> 0
        'mua'       -> 1
        'good'      -> 2
        'unsorted'  -> 3

    Parameters
    ----------
    filepath : str
        Path to the cluster groups file.

    Returns
    -------
    cids : np.ndarray, shape [nClusters], dtype int64
        Cluster IDs.
    cgs : np.ndarray, shape [nClusters], dtype int64
        Numeric quality labels.
    """
    label_map = {"noise": 0, "mua": 1, "good": 2, "unsorted": 3}
    delimiter = "\t" if filepath.endswith(".tsv") else ","
    cids_list: list[int] = []
    cgs_list: list[int] = []

    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row in reader:
            # Column names vary across Phy versions
            cid_key = next(
                (k for k in ("cluster_id", "id") if k in row), None
            )
            label_key = next(
                (k for k in ("group", "KSLabel") if k in row), None
            )
            if cid_key is None or label_key is None:
                continue
            cids_list.append(int(row[cid_key]))
            label_str = row[label_key].strip().lower()
            cgs_list.append(label_map.get(label_str, 3))

    return (
        np.array(cids_list, dtype=np.int64),
        np.array(cgs_list, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# load_ks_dir
# ---------------------------------------------------------------------------

def load_ks_dir(
    ks_path: str,
    exclude_noise: bool = True,
    load_pcs: bool = False,
) -> dict:
    """
    Load a Kilosort output directory into a Python dict.

    Translated from: spikes-master/preprocessing/phyHelpers/loadKSdir.m

    Parameters
    ----------
    ks_path : str
        Path to the Kilosort output directory (must contain params.py and
        the standard Kilosort .npy outputs).
    exclude_noise : bool, optional
        If True (default), remove spikes whose cluster is labelled 'noise'
        (cgs == 0) in the cluster_groups file.  Matches MATLAB's
        params.excludeNoise = true default.
    load_pcs : bool, optional
        If True, also load pc_features.npy and pc_feature_ind.npy.
        Default False.  ks_driftmap requires this (loadKSdir is called with
        params.loadPCs = true inside ksDriftmap.m).

    Returns
    -------
    dict with the following keys
        st               : np.ndarray [nSpikes], float64
            Spike times in seconds (spike_times.npy / sample_rate).
        clu              : np.ndarray [nSpikes], int64
            Cluster ID for each spike (spike_clusters.npy, or falls back to
            spikeTemplates if spike_clusters.npy is absent).
        spikeTemplates   : np.ndarray [nSpikes], int64
            Template index (0-based) for each spike (spike_templates.npy).
        tempScalingAmps  : np.ndarray [nSpikes], float64
            Per-spike amplitude scaling factors (amplitudes.npy); ones if
            amplitudes.npy is not present.
        temps            : np.ndarray [nTemplates, nTimepoints, nChannels], float64
            Spike waveform templates (templates.npy).
        xcoords          : np.ndarray [nChannels], float64
            Channel x-coordinates in µm (channel_positions.npy column 0).
        ycoords          : np.ndarray [nChannels], float64
            Channel y-coordinates in µm (channel_positions.npy column 1).
        winv             : np.ndarray [nChannels, nChannels], float64
            Inverse whitening matrix (whitening_mat_inv.npy).
        cgs              : np.ndarray [nClusters], int64
            Quality label per cluster ID (0=noise, 1=mua, 2=good, 3=unsorted).
        cids             : np.ndarray [nClusters], int64
            Cluster IDs corresponding to cgs.
        pcFeat           : np.ndarray or None
            PC features [nSpikes, nPCs, nLocalChannels] if load_pcs=True,
            else None.  Kept at native float32 dtype.
        pcFeatInd        : np.ndarray or None
            PC feature channel indices [nTemplates, nLocalChannels] (0-based)
            if load_pcs=True, else None.
        sample_rate      : float
            Recording sample rate in Hz (from params.py).
        gain             : float or None
            Voltage gain from params.py if present, else None.
    """
    p = Path(ks_path)

    # --- params.py -----------------------------------------------------------
    params = _parse_params_py(str(p / "params.py"))
    sample_rate = float(params["sample_rate"])
    # Optional gain field (used by ksDriftmap to convert amps to µV)
    gain: float | None = float(params["gain"]) if "gain" in params else None

    # --- Spike times ---------------------------------------------------------
    # spike_times.npy contains integer sample indices; divide by sample_rate
    # to obtain seconds.  numpy.load returns row-major arrays natively.
    ss = np.load(str(p / "spike_times.npy"), allow_pickle=False)
    st: np.ndarray = ss.ravel().astype(np.float64) / sample_rate

    # --- Template indices (0-based, as Kilosort writes them) -----------------
    spike_templates: np.ndarray = (
        np.load(str(p / "spike_templates.npy"), allow_pickle=False)
        .ravel()
        .astype(np.int64)
    )

    # --- Cluster IDs ---------------------------------------------------------
    spike_clusters_path = p / "spike_clusters.npy"
    if spike_clusters_path.exists():
        clu: np.ndarray = (
            np.load(str(spike_clusters_path), allow_pickle=False)
            .ravel()
            .astype(np.int64)
        )
    else:
        clu = spike_templates.copy()

    # --- Amplitude scaling factors -------------------------------------------
    amp_path = p / "amplitudes.npy"
    if amp_path.exists():
        temp_scaling_amps: np.ndarray = (
            np.load(str(amp_path), allow_pickle=False)
            .ravel()
            .astype(np.float64)
        )
    else:
        temp_scaling_amps = np.ones(st.size, dtype=np.float64)

    # --- Optional PC features ------------------------------------------------
    pc_feat: np.ndarray | None = None
    pc_feat_ind: np.ndarray | None = None
    if load_pcs:
        # pc_features.npy is float32 in Kilosort; keep native dtype so the
        # single-precision depth arithmetic below matches MATLAB.
        pc_feat = np.load(str(p / "pc_features.npy"), allow_pickle=False)
        # pc_feat shape: [nSpikes, nPCs, nLocalChannels]
        pc_feat_ind = (
            np.load(str(p / "pc_feature_ind.npy"), allow_pickle=False)
            .astype(np.int64)
        )
        # pc_feature_ind.npy stores 0-based channel indices (Kilosort output)

    # --- Cluster quality labels ----------------------------------------------
    cgs_file: str | None = None
    for fname in ("cluster_groups.csv", "cluster_group.tsv"):
        candidate = p / fname
        if candidate.exists():
            cgs_file = str(candidate)
            break

    if cgs_file is not None:
        cids, cgs = _read_cluster_groups(cgs_file)

        if exclude_noise:
            noise_set = set(cids[cgs == 0].tolist())
            # === OPTIMIZATION E2 — O(nSpikes) boolean-lookup noise mask ======
            # Baseline: ~np.isin(clu, noise_list) is O(nSpikes log nSpikes).
            # Optimized: a boolean table indexed by cluster id -> O(nSpikes).
            # The mask is BYTE-IDENTICAL (Level 1).  Gated by
            # OPT.fast_noise_exclusion.
            if OPT.fast_noise_exclusion and clu.size and noise_set:
                _n = max(int(clu.max()), int(max(noise_set))) + 1
                _is_noise = np.zeros(_n, dtype=bool)
                _is_noise[np.fromiter(noise_set, dtype=np.int64)] = True
                keep: np.ndarray = ~_is_noise[clu]
            else:
                keep = ~np.isin(clu, list(noise_set))
            # =================================================================

            st = st[keep]
            spike_templates = spike_templates[keep]
            temp_scaling_amps = temp_scaling_amps[keep]
            clu = clu[keep]
            # NOTE: pcFeat is per-spike and is filtered; pcFeatInd is
            # per-template and is NOT filtered (matches loadKSdir.m).
            if load_pcs and pc_feat is not None:
                pc_feat = pc_feat[keep]

            # Filter cids/cgs to remove noise entries
            keep_cids: np.ndarray = ~np.isin(cids, list(noise_set))
            cgs = cgs[keep_cids]
            cids = cids[keep_cids]
    else:
        # No cluster file: assign all templates as 'unsorted' (cgs == 3)
        clu = spike_templates.copy()
        cids = np.unique(spike_templates)
        cgs = np.full(cids.shape, 3, dtype=np.int64)

    # --- Channel positions ---------------------------------------------------
    coords: np.ndarray = np.load(
        str(p / "channel_positions.npy"), allow_pickle=False
    ).astype(np.float64)
    xcoords: np.ndarray = coords[:, 0]  # column 0 -> x
    ycoords: np.ndarray = coords[:, 1]  # column 1 -> y (depth)

    # --- Templates -----------------------------------------------------------
    # Shape: [nTemplates, nTimepoints, nChannels]
    temps: np.ndarray = np.load(
        str(p / "templates.npy"), allow_pickle=False
    ).astype(np.float64)

    # --- Whitening matrix inverse --------------------------------------------
    winv: np.ndarray = np.load(
        str(p / "whitening_mat_inv.npy"), allow_pickle=False
    ).astype(np.float64)

    return {
        "st": st,
        "clu": clu,
        "spikeTemplates": spike_templates,
        "tempScalingAmps": temp_scaling_amps,
        "temps": temps,
        "xcoords": xcoords,
        "ycoords": ycoords,
        "winv": winv,
        "cgs": cgs,
        "cids": cids,
        "pcFeat": pc_feat,
        "pcFeatInd": pc_feat_ind,
        "sample_rate": sample_rate,
        "gain": gain,
    }


# ---------------------------------------------------------------------------
# templatePositionsAmplitudes  (spike amplitudes in unwhitened space)
# ---------------------------------------------------------------------------

def _template_positions_amplitudes(
    temps: np.ndarray,
    winv: np.ndarray,
    ycoords: np.ndarray,
    spike_templates: np.ndarray,
    temp_scaling_amps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Faithful translation of templatePositionsAmplitudes.m (amps + tempsUnW).

    Returns
    -------
    spike_amps        : (nSpikes,) float64  — per-spike amplitude in unwhitened
                        space = tempAmpsUnscaled[template] * tempScalingAmps.
    template_depths   : (nTemplates,) float64 — template centre of mass (0.3
                        threshold); NOTE ksDriftmap overrides spikeDepths with
                        the PC-feature CoM, so this is returned for parity only.
    temp_amps_unscaled: (nTemplates,) float64
    temps_unw         : (nTemplates, nTime, nChannels) float64 — unwhitened.

    MATLAB reference (templatePositionsAmplitudes.m):
        tempsUnW = zeros(size(temps));                       % DOUBLE buffer
        for t: tempsUnW(t,:,:) = squeeze(temps(t,:,:))*winv;
        tempChanAmps = squeeze(max(tempsUnW,[],2)) - squeeze(min(tempsUnW,[],2));
        tempAmpsUnscaled = max(tempChanAmps,[],2);
        threshVals = tempAmpsUnscaled*0.3;
        tempChanAmps(tempChanAmps < threshVals) = 0;
        templateDepths = sum(tempChanAmps.*ycoords',2)./sum(tempChanAmps,2);
        spikeAmps = tempAmpsUnscaled(spikeTemplates+1).*tempScalingAmps;
    """
    # tempsUnW = zeros(size(temps)) is a DOUBLE array in MATLAB; assigning the
    # (single-or-double) product into it yields double -> compute in float64.
    temps_unw = np.matmul(
        temps.astype(np.float64), winv.astype(np.float64)
    )  # [nTemplates, nTime, nChannels]

    # per-channel positive-peak minus negative-peak (over the time axis)
    temp_chan_amps = temps_unw.max(axis=1) - temps_unw.min(axis=1)  # [nT, nCh]

    # template amplitude = amplitude of its largest channel
    temp_amps_unscaled = temp_chan_amps.max(axis=1)                 # [nT]

    # zero-out channels below 30% of the peak before the centre-of-mass
    thresh_vals = temp_amps_unscaled * 0.3
    tca = temp_chan_amps.copy()
    tca[tca < thresh_vals[:, None]] = 0.0

    denom = tca.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        template_depths = (tca * ycoords[None, :]).sum(axis=1) / denom  # [nT]

    # per-spike amplitude (templates are 0-based here)
    spike_amps = temp_amps_unscaled[spike_templates] * temp_scaling_amps

    return spike_amps, template_depths, temp_amps_unscaled, temps_unw


def _pc_feature_depths(
    pc_feat: np.ndarray,
    pc_feat_ind: np.ndarray,
    ycoords: np.ndarray,
    spike_templates: np.ndarray,
) -> np.ndarray:
    """Per-spike depth as the PC-feature centre of mass (ksDriftmap.m).

    MATLAB reference (ksDriftmap.m):
        pcFeat = squeeze(pcFeat(:,1,:));      % first PC only
        pcFeat(pcFeat<0) = 0;                 % clamp negative loadings
        spikeFeatInd = pcFeatInd(spikeTemps+1,:);
        spikeFeatYcoords = ycoords(spikeFeatInd+1);
        spikeDepths = sum(spikeFeatYcoords.*pcFeat.^2,2)./sum(pcFeat.^2,2);

    Returns float32 to reproduce MATLAB single precision:
    `spikeFeatYcoords (double) .* pcFeat.^2 (single)` evaluates in single.
    """
    pc1 = pc_feat[:, 0, :].astype(np.float32).copy()  # first PC -> [nSpk, nLocalCh]
    pc1[pc1 < 0] = 0.0

    spike_feat_ind = pc_feat_ind[spike_templates, :]          # [nSpk, nLocalCh], 0-based
    spike_feat_ycoords = ycoords[spike_feat_ind].astype(np.float32)  # single, per MATLAB

    w = pc1 * pc1                                             # pcFeat.^2 (single)
    num = np.sum(spike_feat_ycoords * w, axis=1, dtype=np.float32)
    den = np.sum(w, axis=1, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        spike_depths = num / den
    return spike_depths.astype(np.float32)


# ---------------------------------------------------------------------------
# ks_driftmap
# ---------------------------------------------------------------------------

def ks_driftmap(
    ks_path: str,
    localized_spikes_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-spike times, amplitudes, depths, and peak sites from a
    Kilosort output directory.

    Faithful translation of: spikes-master/analysis/ksDriftmap.m
    (which itself calls templatePositionsAmplitudes.m for the amplitudes and
    derives depths from the PC-feature centre of mass).

    Parameters
    ----------
    ks_path : str
        Path to the Kilosort output directory.
    localized_spikes_only : bool, optional
        If True, keep only spikes from templates whose significant channels
        (|amp| > 50 % of peak) span <= 20 channels, then keep only spikes
        consistent with a through-origin depth->site regression (|resid| <= 5).
        Default False.

    Returns
    -------
    spike_times : np.ndarray [nSpikes], float64      — seconds.
    spike_amps  : np.ndarray [nSpikes], float64      — unwhitened-space amplitude
                  (× params.py `gain` when present, converting to µV).
    spike_depths: np.ndarray [nSpikes], float32      — PC-feature CoM depth (µm).
    spike_sites : np.ndarray [nSpikes], uint16       — peak channel (0-based) of
                  each spike's template (from the unwhitened templates).
    """
    sp = load_ks_dir(ks_path, exclude_noise=True, load_pcs=True)

    temps = sp["temps"]                       # [nTemplates, nTime, nChannels]
    winv = sp["winv"]                         # [nChannels, nChannels]
    ycoords = sp["ycoords"]                   # [nChannels]
    spike_temps = sp["spikeTemplates"]        # [nSpikes], 0-based
    temp_scaling_amps = sp["tempScalingAmps"] # [nSpikes]
    spike_times = sp["st"]                    # [nSpikes], seconds
    pc_feat = sp["pcFeat"]                    # [nSpikes, nPCs, nLocalChannels]
    pc_feat_ind = sp["pcFeatInd"]             # [nTemplates, nLocalChannels]
    gain = sp["gain"]

    # --- optional: drop non-localized templates (space span > 20 channels) ---
    if localized_spikes_only:
        n_templates = temps.shape[0]
        localized = np.zeros(n_templates, dtype=bool)
        for t in range(n_templates):
            chan_peak = np.max(np.abs(temps[t]), axis=0)   # |amp| per channel
            m = chan_peak.max()
            ch = np.where(chan_peak > 0.5 * m)[0]
            if ch.size:
                localized[t] = (ch.max() - ch.min()) <= 20
        localized_ids = np.where(localized)[0]              # 0-based template IDs
        keep = np.isin(spike_temps, localized_ids)
        spike_times = spike_times[keep]
        spike_temps = spike_temps[keep]
        temp_scaling_amps = temp_scaling_amps[keep]
        pc_feat = pc_feat[keep]

    # --- depths: PC-feature centre of mass -----------------------------------
    spike_depths = _pc_feature_depths(pc_feat, pc_feat_ind, ycoords, spike_temps)

    # --- amplitudes: unwhitened template peak-to-peak × scaling --------------
    spike_amps, _template_depths, _temp_amps, temps_unw = (
        _template_positions_amplitudes(
            temps, winv, ycoords, spike_temps, temp_scaling_amps
        )
    )

    # --- peak site per spike (from the UNwhitened templates) -----------------
    # MATLAB: [~,max_site] = max(max(abs(tempsUnW),[],2),[],3);
    max_site = np.argmax(np.max(np.abs(temps_unw), axis=1), axis=1)  # [nTemplates]
    spike_sites = max_site[spike_temps]

    # --- optional µV conversion when params.py stores a gain -----------------
    if gain is not None:
        spike_amps = spike_amps * float(gain)

    # --- optional: regression-based localization filter ----------------------
    if localized_spikes_only:
        # MATLAB: b = regress(spikeSites, spikeDepths)  (through the origin)
        sd = spike_depths.astype(np.float64)
        ss = spike_sites.astype(np.float64)
        denom = float(np.dot(sd, sd))
        b = float(np.dot(sd, ss)) / denom if denom > 0.0 else 0.0
        keep2 = np.abs(ss - b * sd) <= 5.0
        spike_times = spike_times[keep2]
        spike_amps = spike_amps[keep2]
        spike_depths = spike_depths[keep2]
        spike_sites = spike_sites[keep2]

    return (
        spike_times,
        spike_amps.astype(np.float64),
        spike_depths,
        spike_sites.astype(np.uint16),
    )

"""
load_experiment_data.py
-----------------------
Python translation of load_experiment_data.m (MATLAB).

Loads three data sources for one Neuropixels experiment and returns them
in a single dict D:
  (1) LabView sensor data   — ethanol and sniffing signals from the .dat file
  (2) LFP sample            — first LFP_LOAD_SECS seconds of NP recording,
                              all AP channels, low-pass filtered and downsampled
                              to LFP_OUT_FS Hz
  (3) Spike data            — spike times, amplitudes, depths, and per-unit
                              summary from Kilosort output

MATLAB→Python translation notes
--------------------------------
* MATLAB struct D → Python dict D.
* MATLAB struct fields (files.datFile) → dict keys (files['datFile']).
* MATLAB 1-based indexing → Python 0-based (no indexing is exposed here).
* MATLAB fread column-major order → or_read_bin uses reshape(..., order='F').
* MATLAB filtfilt on rawAP' (transposed) filters each row of rawAP, which
  is equivalent to scipy.signal.filtfilt(b, a, rawAP, axis=1).
* MATLAB round() is half-away-from-zero; Python's built-in round() uses
  banker's rounding.  For dsRatio = 30000/1000 = 30 (exact integer), both
  give the same result.
* MATLAB size(A,1) → A.shape[0]; numel → A.size; isempty → .size == 0.
* MATLAB column vectors [n×1] → numpy 1-D arrays of shape (n,).
* D.sp.spikeTemplates is 0-indexed in Kilosort output; ks_utils.py keeps it
  0-indexed (no +1 needed in Python, unlike MATLAB's +1 in the fallback).

Checkpoint files written when checkpoint_dir is supplied:
  chk_labview_ETH.npy       — (134950,) float64, after or_loaddat()
  chk_labview_SNF.npy       — (134950,) float64, after or_loaddat()
  chk_labview_LV_Fs.npy     — 0-D float64 scalar
  chk_lfp_NP_Fs.npy         — 0-D float64 scalar, after or_read_meta()
  chk_lfp_LFP.npy           — (384, 10000) float64, after filtfilt+downsample
  chk_lfp_LFP_Fs.npy        — 0-D float64 scalar
  chk_spikes_xcoords.npy    — (382,) float64, after load_ks_dir()
  chk_spikes_ycoords.npy    — (382,) float64, after load_ks_dir()
  chk_spikes_unitIDs.npy    — (379,) int32, after per-unit aggregation
  chk_spikes_unitDepths.npy — (379,) float64
  chk_spikes_unitAmps.npy   — (379,) float64
  chk_spikes_unitFiringRate.npy — (379,) float64
"""

from __future__ import annotations

import os
import re
import warnings

import numpy as np
import scipy.signal

from lib.or_loaddat import or_loaddat
from lib.or_read_bin import or_read_bin
from lib.or_read_meta import or_read_meta
from lib.ks_utils import load_ks_dir, ks_driftmap
from optconfig import OPT   # optimization switches (one-switch revert: optconfig.set_baseline())

# ---- Configuration -------------------------------------------------------
LFP_LOAD_SECS: int   = 10       # seconds of raw NP data to load for LFP
LFP_OUT_FS:    float = 1000.0   # target LFP output sampling rate (Hz)
LFP_LP_HZ:     float = 400.0    # low-pass cutoff for LFP extraction (Hz)
# --------------------------------------------------------------------------


def load_experiment_data(files: dict, checkpoint_dir: str | None = None) -> dict:
    """Load sensor, LFP, and spike data from one Neuropixels experiment.

    Parameters
    ----------
    files : dict
        Output of or_validate_files() with keys:
            'datFile'  — LabView .dat filename (str)
            'datPath'  — folder containing .dat file (str)
            'binFile'  — SpikeGLX .bin filename (str)
            'binPath'  — folder containing .bin and .meta files (str)
            'ksDir'    — folder containing Kilosort output .npy files (str)
    checkpoint_dir : str, optional
        If supplied, intermediate arrays are saved as .npy files in this
        directory for validation against MATLAB reference outputs.

    Returns
    -------
    D : dict
        LABVIEW SENSOR DATA (sampled at ~125 Hz):
          'ETH'       — ethanol sensor trace, normalised 0–1          (n,) float64
          'SNF'       — sniffing cannula pressure trace                (n,) float64
          'TR'        — trial number per sample                        (n,) int32
          'TS'        — within-trial timestamp in ms                   (n,) int32
          'FR'        — camera frame number                            (n,) int32
          'LV_Fs'     — LabView sampling rate in Hz                    float

        LFP (from SpikeGLX .bin file):
          'LFP'            — filtered & downsampled LFP  (nAPch, nSamp) float64
          'LFP_t'          — time vector in seconds       (nSamp,)       float64
          'LFP_Fs'         — LFP sample rate after downsampling          float
          'LFP_duration_s' — duration of loaded LFP segment in seconds   float
          'meta'           — SpikeGLX metadata dict (from .meta file)    dict
          'binFile'        — .bin filename (for on-demand ReadBin calls)  str
          'binPath'        — .bin folder                                  str
          'NP_Fs'          — native NP sampling rate in Hz               float

        SPIKE DATA (from Kilosort output):
          'sp'             — full Kilosort dict from load_ks_dir()        dict
          'spikeTimes'     — spike times in seconds, all units  (nSpk,)  float64
          'spikeAmps'      — spike amplitude                    (nSpk,)  float64
          'spikeDepths'    — spike depth along probe in µm      (nSpk,)  float64
          'unitIDs'        — cluster IDs of included units       (nU,)   int32
          'unitDepths'     — depth of each unit's peak channel   (nU,)   float64
          'unitAmps'       — mean spike amplitude per unit       (nU,)   float64
          'unitFiringRate' — mean firing rate in spikes/s        (nU,)   float64
          'xcoords'        — channel x-positions in µm           (nCh,)  float64
          'ycoords'        — channel y-positions in µm           (nCh,)  float64
    """
    D: dict = {}

    # =========================================================
    # PART 1 — LabView sensor data (.dat file)
    # =========================================================
    print(f"  Loading LabView sensor data: {files['datFile']}")

    # MATLAB: [D.TR, D.TS, D.FR, D.ETH, D.SNF, D.LV_Fs] =
    #           OR_loaddat(files.datFile, files.datPath)
    TR, TS, FR, ETH, SNF, LV_Fs = or_loaddat(files['datFile'], files['datPath'])

    D['TR']    = TR        # (n,) int32  — trial number
    D['TS']    = TS        # (n,) int32  — within-trial timestamp in ms
    D['FR']    = FR        # (n,) int32  — camera frame number
    D['ETH']   = ETH       # (n,) float64 — ethanol trace, normalised [0,1]
    D['SNF']   = SNF       # (n,) float64 — sniffing trace (raw counts)
    D['LV_Fs'] = LV_Fs    # float — LabView sampling rate in Hz

    print(f"    ETH, SNF loaded  ({D['ETH'].size} samples, {D['LV_Fs']:.1f} Hz)")

    # Checkpoint 1 — after or_loaddat()
    if checkpoint_dir is not None:
        np.save(os.path.join(checkpoint_dir, 'chk_labview_ETH.npy'), D['ETH'])
        np.save(os.path.join(checkpoint_dir, 'chk_labview_SNF.npy'), D['SNF'])
        # LV_Fs is a Python float; np.array() wraps it in a 0-D float64 array
        np.save(os.path.join(checkpoint_dir, 'chk_labview_LV_Fs.npy'),
                np.array(D['LV_Fs']))

    # =========================================================
    # PART 2 — Neuropixels LFP sample (.bin + .meta files)
    # =========================================================
    print(f"  Loading NP metadata: {files['binFile']}")

    # MATLAB: D.meta = OR_ReadMeta(files.binFile, files.binPath)
    meta         = or_read_meta(files['binFile'], files['binPath'])
    D['meta']    = meta
    D['binFile'] = files['binFile']
    D['binPath'] = files['binPath']

    # MATLAB: D.NP_Fs = OR_SampRate(D.meta)
    D['NP_Fs'] = or_samp_rate(meta)

    # Checkpoint 2 — NP_Fs right after metadata parse
    if checkpoint_dir is not None:
        np.save(os.path.join(checkpoint_dir, 'chk_lfp_NP_Fs.npy'),
                np.array(D['NP_Fs']))

    # MATLAB: [nAP, ~, ~] = OR_ChannelCountsIM(D.meta)
    nAP, _nLF, _nSY = or_channel_counts_im(meta)

    # Number of samples to load for the LFP window.
    # MATLAB: nSampsRaw = min(floor(LFP_LOAD_SECS * D.NP_Fs),
    #                         floor(str2double(meta.fileSizeBytes) /
    #                               (2 * str2double(meta.nSavedChans))))
    # int() truncates toward zero — equivalent to floor() for positive numbers.
    nSampsRaw = min(
        int(LFP_LOAD_SECS * D['NP_Fs']),
        int(int(meta['fileSizeBytes']) / (2 * int(meta['nSavedChans'])))
    )

    print(f"  Loading LFP sample: first {LFP_LOAD_SECS:.0f} s, "
          f"{nAP} AP channels from {files['binFile']}")

    # MATLAB: rawData = OR_ReadBin(0, nSampsRaw, D.meta, files.binFile, files.binPath)
    #         rawAP   = rawData(1:nAP, :)
    rawData = or_read_bin(0, nSampsRaw, meta, files['binFile'], files['binPath'])
    rawAP   = rawData[:nAP, :].astype(np.float64)  # AP channels only; (nAP, nSamps)

    # Convert raw int16 to volts.
    # MATLAB:
    #   fI2V = OR_Int2Volts(D.meta)
    #   [APgain, ~] = OR_ChanGainsIM(D.meta)
    #   for ch = 1:nAP
    #       rawAP(ch,:) = rawAP(ch,:) * (fI2V / APgain(ch))
    #   end
    # Python vectorised equivalent: broadcast APgain[:, np.newaxis] along time axis.
    fI2V                = or_int2volts(meta)
    APgain, _LFgain     = or_chan_gains_im(meta)
    rawAP              *= fI2V / APgain[:, np.newaxis]   # in-place per-channel scale

    # Low-pass filter to extract LFP band.
    # MATLAB: [b_lp, a_lp] = butter(4, LFP_LP_HZ / (D.NP_Fs / 2), 'low')
    #         lfpFull = filtfilt(b_lp, a_lp, double(rawAP)')';
    #
    # The MATLAB expression transposes rawAP to (nSamps, nAP) so filtfilt
    # operates column-wise (one channel per column), then transposes back to
    # (nAP, nSamps).  In Python, scipy.signal.filtfilt with axis=1 is
    # identical: filter along the time dimension of the (nAP, nSamps) array.
    #
    # PADDING (fidelity fix): MATLAB filtfilt uses an odd extension of length
    # 3*(max(len(a),len(b))-1); scipy's default padlen is 3*max(len(a),len(b)),
    # a longer transient that shifts the near-edge samples.  Match MATLAB by
    # setting padtype='odd' and padlen=3*(ntaps-1) so the two agree at the
    # signal edges (interior samples are identical regardless).
    b_lp, a_lp = scipy.signal.butter(4, LFP_LP_HZ / (D['NP_Fs'] / 2.0), btype='low')
    _padlen = 3 * (max(len(a_lp), len(b_lp)) - 1)   # == MATLAB filtfilt default
    lfpFull = scipy.signal.filtfilt(
        b_lp, a_lp, rawAP, axis=1, padtype='odd', padlen=_padlen
    )  # (nAP, nSamps)

    # Downsample to target output rate.
    # MATLAB: dsRatio = round(D.NP_Fs / LFP_OUT_FS)
    #         D.LFP   = lfpFull(:, 1:dsRatio:end)
    #         D.LFP_t = (0:size(D.LFP,2)-1) / LFP_OUT_FS
    # MATLAB slice 1:dsRatio:end (1-based, inclusive) == Python [::dsRatio] (0-based).
    dsRatio             = round(D['NP_Fs'] / LFP_OUT_FS)
    D['LFP']            = lfpFull[:, ::dsRatio]
    D['LFP_t']          = np.arange(D['LFP'].shape[1], dtype=np.float64) / LFP_OUT_FS
    D['LFP_Fs']         = D['NP_Fs'] / dsRatio
    D['LFP_duration_s'] = nSampsRaw / D['NP_Fs']

    print(f"    LFP loaded: [{D['LFP'].shape[0]} ch × {D['LFP'].shape[1]} samples] "
          f"at {D['LFP_Fs']:.0f} Hz ({D['LFP_duration_s']:.0f} s)")

    # Checkpoint 3 — after filtfilt + downsample
    if checkpoint_dir is not None:
        np.save(os.path.join(checkpoint_dir, 'chk_lfp_LFP.npy'), D['LFP'])
        np.save(os.path.join(checkpoint_dir, 'chk_lfp_LFP_Fs.npy'),
                np.array(D['LFP_Fs']))

    # =========================================================
    # PART 3 — Spike data (Kilosort output)
    # =========================================================
    print(f"  Loading Kilosort spike data: {files['ksDir']}")

    # MATLAB: if exist('loadKSdir', 'file')
    #             D.sp = loadKSdir(files.ksDir)
    #         else
    #             error(...)
    #         end
    # In Python the import at the top guarantees load_ks_dir is available.
    sp      = load_ks_dir(files['ksDir'])
    D['sp'] = sp

    # MATLAB: if exist('ksDriftmap', 'file')
    #             [D.spikeTimes, D.spikeAmps, D.spikeDepths, ~] = ksDriftmap(files.ksDir)
    #         else
    #             warning(...); D.spikeTimes = D.sp.st; ...
    #         end
    try:
        spike_times, spike_amps, spike_depths, _ = ks_driftmap(files['ksDir'])
        D['spikeTimes']  = spike_times   # (nSpk,) float64 — seconds
        D['spikeAmps']   = spike_amps    # (nSpk,) float64 — amplitude units
        D['spikeDepths'] = spike_depths  # (nSpk,) float64 — µm from probe tip
    except Exception as exc:
        warnings.warn(
            f"ks_driftmap failed ({exc}). Computing spike depths from templates.",
            RuntimeWarning,
            stacklevel=2,
        )
        D['spikeTimes']  = sp['st']
        D['spikeAmps']   = np.array([], dtype=np.float64)
        D['spikeDepths'] = _compute_depths_from_templates(sp)

    # Channel coordinates from Kilosort channel_positions.npy
    # MATLAB: D.xcoords = D.sp.xcoords;  D.ycoords = D.sp.ycoords
    D['xcoords'] = sp['xcoords']   # (nCh,) float64 — µm
    D['ycoords'] = sp['ycoords']   # (nCh,) float64 — µm

    # Checkpoint 4 — channel coordinates immediately after load_ks_dir()
    if checkpoint_dir is not None:
        np.save(os.path.join(checkpoint_dir, 'chk_spikes_xcoords.npy'), D['xcoords'])
        np.save(os.path.join(checkpoint_dir, 'chk_spikes_ycoords.npy'), D['ycoords'])

    # Per-unit summary: depth, amplitude, firing rate.
    # MATLAB:
    #   uniqueUnits    = unique(D.sp.clu)        % [nUnits×1], same dtype as clu
    #   nUnits         = numel(uniqueUnits)
    #   unitDepths     = nan(nUnits, 1)
    #   unitAmps       = nan(nUnits, 1)
    #   unitFiringRate = nan(nUnits, 1)
    #   recordingDur   = D.sp.st(end)
    #   for u = 1:nUnits
    #       uid = uniqueUnits(u)
    #       spkMask = (D.sp.clu == uid)
    #       if ~isempty(D.spikeDepths), unitDepths(u) = mean(D.spikeDepths(spkMask)); end
    #       if ~isempty(D.spikeAmps),   unitAmps(u)   = mean(D.spikeAmps(spkMask));   end
    #       unitFiringRate(u) = sum(spkMask) / recordingDur
    #   end
    #
    # Note: ks_utils returns sp['clu'] as int64; cast unitIDs to int32 to match
    # the Kilosort cluster_id dtype (int32) expected by the manifest.
    # return_counts=True gives the spike count per unit in the same sorted order
    # as unique_units — no per-spike mask is needed for firing rate.
    unique_units, unit_counts = np.unique(sp['clu'], return_counts=True)  # (nUnits,)
    nUnits         = len(unique_units)
    recording_dur  = float(sp['st'][-1])           # last spike time = total duration

    # Vectorised firing rate: count / duration, same result as summing the
    # per-spike boolean mask inside the loop below (bit-identical either way).
    unit_fr        = unit_counts.astype(np.float64) / recording_dur

    spike_depths_arr = D['spikeDepths']
    spike_amps_arr   = D['spikeAmps']

    # === OPTIMIZATION E1 — vectorized per-unit aggregation ===============
    # Baseline is an O(nUnits x nSpikes) Python loop (a full boolean scan of
    # the spike arrays for every unit).  The optimized path replaces it with
    # one grouped reduction via np.bincount over the cluster id (O(nSpikes)).
    # A group that contains any NaN yields a NaN sum -> NaN mean, matching
    # MATLAB mean() and the baseline np.mean loop.  Accumulation is float64;
    # results match the golden fixtures within the single-precision-mean
    # tolerance (see optconfig.py).  Gated by OPT.vectorized_unit_aggregation.
    if OPT.vectorized_unit_aggregation:
        clu_i    = sp['clu'].astype(np.intp, copy=False)
        maxid    = int(clu_i.max())
        cnt_full = np.bincount(clu_i, minlength=maxid + 1)
        present  = cnt_full > 0            # flatnonzero(present) == unique_units (ascending)
        cnt      = cnt_full[present].astype(np.float64)
        unit_depths = np.full(nUnits, np.nan, dtype=np.float64)
        unit_amps   = np.full(nUnits, np.nan, dtype=np.float64)
        if spike_depths_arr.size > 0:
            unit_depths = np.bincount(
                clu_i, weights=spike_depths_arr.astype(np.float64),
                minlength=maxid + 1)[present] / cnt
        if spike_amps_arr.size > 0:
            unit_amps = np.bincount(
                clu_i, weights=spike_amps_arr.astype(np.float64),
                minlength=maxid + 1)[present] / cnt
    else:
        # BASELINE per-unit loop (original behavior; one-switch revert).
        unit_depths = np.full(nUnits, np.nan, dtype=np.float64)
        unit_amps   = np.full(nUnits, np.nan, dtype=np.float64)
        for u, uid in enumerate(unique_units):
            mask = sp['clu'] == uid                 # boolean index vector, (nSpk,)
            # MATLAB mean() on a column with NaN elements returns NaN — numpy
            # mean() matches (as opposed to numpy.nanmean).
            if len(spike_depths_arr) > 0:
                unit_depths[u] = np.mean(spike_depths_arr[mask])
            if len(spike_amps_arr) > 0:
                unit_amps[u] = np.mean(spike_amps_arr[mask])
    # =====================================================================

    # MATLAB: D.unitIDs = uniqueUnits  (int32 in MATLAB from int32 clu)
    # Python: cast to int32 to match manifest dtype expectation.
    D['unitIDs']        = unique_units.astype(np.int32)
    D['unitDepths']     = unit_depths
    D['unitAmps']       = unit_amps
    D['unitFiringRate'] = unit_fr

    print(f"    Spikes loaded: {D['spikeTimes'].size} spikes, {nUnits} units")
    valid_depths = unit_depths[np.isfinite(unit_depths)]
    if valid_depths.size > 0:
        print(f"    Depth range: {valid_depths.min():.0f} – {valid_depths.max():.0f} µm")

    # Checkpoint 5 — after per-unit aggregation loop
    if checkpoint_dir is not None:
        np.save(os.path.join(checkpoint_dir, 'chk_spikes_unitIDs.npy'),
                D['unitIDs'])
        np.save(os.path.join(checkpoint_dir, 'chk_spikes_unitDepths.npy'),
                D['unitDepths'])
        np.save(os.path.join(checkpoint_dir, 'chk_spikes_unitAmps.npy'),
                D['unitAmps'])
        np.save(os.path.join(checkpoint_dir, 'chk_spikes_unitFiringRate.npy'),
                D['unitFiringRate'])

    return D


# =========================================================
# PRIVATE HELPER — compute spike depths from templates
# (fallback when ks_driftmap raises an exception)
# =========================================================

def _compute_depths_from_templates(sp: dict) -> np.ndarray:
    """Compute per-spike depth from each spike's peak template channel.

    Translated from the local MATLAB function compute_depths_from_templates(sp)
    at the bottom of load_experiment_data.m.

    MATLAB:
        tempsMaxAbs = squeeze(max(abs(sp.temps), [], 2))  % [nTemplates x nChannels]
        for t = 1:nTemplates
            [~, peakChan(t)] = max(tempsMaxAbs(t,:))     % 1-based channel index
        end
        templateDepths = sp.ycoords(peakChan)             % 1-indexed lookup
        spikeTemplateIdx = sp.spikeTemplates + 1          % 0→1-indexed
        spikeDepths = templateDepths(spikeTemplateIdx)

    Python differences:
    * sp['temps'] shape: [nTemplates, nTimepoints, nChannels] — np.load native
    * max(abs(...), [], 2) = max along dim 2 (timepoints) = np.max(..., axis=1)
    * np.argmax returns 0-based indices → sp['ycoords'][peak_chan] (no +1 needed)
    * sp['spikeTemplates'] is already 0-indexed — no +1 needed for lookup

    Parameters
    ----------
    sp : dict
        Output of load_ks_dir().

    Returns
    -------
    spike_depths : np.ndarray, shape (nSpikes,), dtype float64
    """
    temps = sp['temps']                        # [nTemplates, nTimepoints, nChannels]

    # Maximum absolute value across timepoints for each (template, channel).
    # MATLAB: squeeze(max(abs(sp.temps), [], 2))  — dim 2 is the time axis
    # Python: axis=1 collapses nTimepoints → shape [nTemplates, nChannels]
    temps_max_abs = np.max(np.abs(temps), axis=1)

    # Peak channel (0-based) per template.
    # MATLAB: [~, peakChan(t)] = max(tempsMaxAbs(t,:))  — returns 1-based index
    # Python: np.argmax returns 0-based index — no offset needed.
    peak_chan = np.argmax(temps_max_abs, axis=1)   # (nTemplates,) int64, 0-based

    # Depth of each template's peak channel.
    # MATLAB: templateDepths = sp.ycoords(peakChan)  — 1-based peakChan
    # Python: sp['ycoords'][peak_chan]               — 0-based peak_chan
    template_depths = sp['ycoords'][peak_chan]       # (nTemplates,)

    # Assign per-spike depth via template lookup.
    # MATLAB: spikeTemplateIdx = sp.spikeTemplates + 1  (convert to 1-based)
    #         spikeDepths = templateDepths(spikeTemplateIdx)
    # Python: sp['spikeTemplates'] is 0-based (ks_utils keeps it that way)
    spike_depths = template_depths[sp['spikeTemplates']]
    return spike_depths.astype(np.float64)


# =========================================================
# LOCAL SpikeGLX HELPER FUNCTIONS
# Translated from the local subfunctions at the bottom of
# load_experiment_data.m.  These parse the SpikeGLX .meta dict
# returned by or_read_meta().  All values in that dict are strings;
# numeric conversion is explicit here.
# =========================================================

def or_samp_rate(meta: dict) -> float:
    """Return the acquisition sample rate in Hz.

    Translated from OR_SampRate(meta) in load_experiment_data.m.

    MATLAB:
        if strcmp(meta.typeThis, 'imec')
            srate = str2double(meta.imSampRate)
        else
            srate = str2double(meta.niSampRate)
        end

    Parameters
    ----------
    meta : dict
        SpikeGLX metadata dict from or_read_meta().

    Returns
    -------
    float
        Sample rate in Hz (e.g. 30000.0 for NP 1.0).
    """
    if meta.get('typeThis') == 'imec':
        return float(meta['imSampRate'])
    return float(meta['niSampRate'])


def or_channel_counts_im(meta: dict) -> tuple[int, int, int]:
    """Return the (nAP, nLF, nSY) saved-channel counts for an imec recording.

    Translated from OR_ChannelCountsIM(meta) in load_experiment_data.m.

    MATLAB:
        M  = str2num(meta.snsApLfSy)   % handles space- or comma-separated
        AP = M(1); LF = M(2); SY = M(3)

    The 'snsApLfSy' field is typically '384,0,1' (nAP, nLF, nSY).
    Python re.split handles both comma- and whitespace-separated variants.

    Parameters
    ----------
    meta : dict
        SpikeGLX metadata dict from or_read_meta().

    Returns
    -------
    nAP, nLF, nSY : tuple[int, int, int]
        Number of AP, LF, and sync channels saved in the .bin file.
    """
    parts = [int(x) for x in re.split(r'[,\s]+', meta['snsApLfSy'].strip())]
    return parts[0], parts[1], parts[2]


def or_int2volts(meta: dict) -> float:
    """Return the int16-to-volt conversion factor (fI2V) for a SpikeGLX recording.

    Translated from OR_Int2Volts(meta) in load_experiment_data.m.

    MATLAB:
        if strcmp(meta.typeThis, 'imec')
            maxInt = 512
            if isfield(meta, 'imMaxInt'), maxInt = str2num(meta.imMaxInt); end
            fI2V = str2double(meta.imAiRangeMax) / maxInt
        else
            fI2V = str2double(meta.niAiRangeMax) / 32768
        end

    UNIT NOTE: fI2V converts raw int16 counts to VOLTS (not µV).  The LFP
    values in D['LFP'] will therefore be in volts.  The field inventory
    confirms values in the range [-0.00342, +0.00497] V (~3–5 mV scale),
    consistent with AP electrode voltages before amplification labelling.
    Do NOT multiply by 1e6 here; replicate the MATLAB unit convention exactly.

    Parameters
    ----------
    meta : dict
        SpikeGLX metadata dict from or_read_meta().

    Returns
    -------
    float
        Conversion factor in V per int16 LSB.
    """
    if meta.get('typeThis') == 'imec':
        max_int: int = 512
        if 'imMaxInt' in meta:
            max_int = int(meta['imMaxInt'])
        return float(meta['imAiRangeMax']) / max_int
    return float(meta['niAiRangeMax']) / 32768.0


def or_chan_gains_im(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel AP and LF gains from the IMRO table in a SpikeGLX metadata dict.

    Translated from OR_ChanGainsIM(meta) in load_experiment_data.m.

    MATLAB:
        probeType = 0
        if isfield(meta,'imDatPrb_type'), probeType = str2num(meta.imDatPrb_type); end
        if (probeType == 21) || (probeType == 24)
            [AP, LF, ~] = OR_ChannelCountsIM(meta)
            APgain = 80 * ones(AP, 1)
            LFgain = zeros(LF, 1)
        else
            if isfield(meta,'typeEnabled')  % 3A
                C = textscan(imroTbl, '(%*s %*s %*s %d %d', 'EndOfLine',')', 'HeaderLines',1)
            else                            % 3B
                C = textscan(imroTbl, '(%*s %*s %*s %d %d %*s', 'EndOfLine',')', 'HeaderLines',1)
            end
            APgain = double(cell2mat(C(1)))
            LFgain = double(cell2mat(C(2)))
        end

    IMRO table format (NP 1.0, 3B example):
        (nCh,probeType)(ch0 bk0 rf0 APgain0 LFgain0 hp0)(ch1 bk1 ...)(...)
    IMRO table format (NP 1.0, 3A example):
        (nCh,probeType)(ch0 bk0 rf0 APgain0 LFgain0)(ch1 bk1 ...)
    In both cases: field index 0=channelID, 1=bankID, 2=refID, 3=APgain, 4=LFgain.

    MATLAB's textscan with EndOfLine=')' and HeaderLines=1 treats each
    parenthesised group as one "line" and skips the first group (probe header).
    Python regex replicates this: extract all (…) groups, skip groups[0].

    Parameters
    ----------
    meta : dict
        SpikeGLX metadata dict from or_read_meta().

    Returns
    -------
    APgain : np.ndarray, shape (nAP,), dtype float64
        Per-channel AP gain values (e.g. 500 or 3000).
    LFgain : np.ndarray, shape (nLF,), dtype float64
        Per-channel LF gain values (e.g. 250); empty for AP-only .bin files.
    """
    probe_type: int = 0
    if 'imDatPrb_type' in meta:
        probe_type = int(meta['imDatPrb_type'])

    # NP 2.0 (UHD variants): fixed gain of 80 for AP, 0 for LF
    if probe_type in (21, 24):
        nAP, nLF, _ = or_channel_counts_im(meta)
        APgain = 80.0 * np.ones(nAP, dtype=np.float64)
        LFgain = np.zeros(nLF, dtype=np.float64)
        return APgain, LFgain

    # NP 1.0 (3A or 3B): parse IMRO table.
    # The 'typeEnabled' field is present for 3A probes only; the format
    # differs only in having one extra field (hpFilter) for 3B probes, but
    # gains are at fixed positions 3 and 4 in both formats, so the same
    # parsing code handles both.
    imro_tbl = meta['imroTbl']

    # Extract all parenthesised groups.  The first group is the header
    # (e.g. '384,3' or '641251209,3') and is skipped (matches MATLAB
    # HeaderLines=1 with EndOfLine=')').
    entries = re.findall(r'\(([^)]*)\)', imro_tbl)
    channel_entries = entries[1:]   # skip the probe-level header group

    ap_gains: list[float] = []
    lf_gains: list[float] = []
    for entry in channel_entries:
        # Fields within each entry may be space- or comma-separated.
        # order: channelID, bankID, refID, APgain, LFgain [, hpFilter]
        parts = re.split(r'[,\s]+', entry.strip())
        ap_gains.append(float(parts[3]))
        lf_gains.append(float(parts[4]))

    return (
        np.array(ap_gains, dtype=np.float64),
        np.array(lf_gains, dtype=np.float64),
    )

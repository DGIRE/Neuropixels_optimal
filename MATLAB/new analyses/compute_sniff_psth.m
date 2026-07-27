function [psth, centers, n_events] = compute_sniff_psth(D, bin_ms, use_ms)
%% COMPUTE_SNIFF_PSTH  Event-triggered PSTH using psthAndBA (spikes-master).
%  Each sniff onset is an event.  psthAndBA builds a [nEvents x nBins]
%  binary array per unit and averages across events → spikes/s.
%
%  INPUTS
%    D       — pipeline D struct; requires:
%                spikeTimes, sp.clu, unitIDs       (from load_experiment_data)
%                sniff_onsets_s, sniff_dur_s        (from compute_sniff_phase)
%    bin_ms  — bin width (ms)
%    use_ms  — false → normalized phase axis (0–1, x-axis rescaled after binning)
%               true  → actual time axis (ms)
%
%  OUTPUTS
%    psth     — [nUnits x nBins]  spikes/s, event-averaged
%    centers  — [1 x nBins]  bin centers (phase 0–1  or  ms)
%    n_events — number of sniff onsets used
%
%  Both modes use a fixed window [0, median_cycle_duration] relative to each
%  onset.  Normalized mode rescales the time axis to phase (0 = onset,
%  1 = median next onset) after binning — identical binning, different x label.
%
%  Requires: psthAndBA, timestampsToBinned, histdiff  (spikes-master toolbox)

if ~isfield(D,'sniff_onsets_s')
    error('OR:noOnsets','Run compute_sniff_phase(D) first.');
end

onset_s   = D.sniff_onsets_s(:);
n_events  = numel(onset_s);
med_dur_s = D.sniff_dur_s;
bin_s     = bin_ms / 1000;
win       = [0, med_dur_s];       % fixed window: onset → median cycle end
nUnits    = numel(D.unitIDs);

% Determine bin centers using toolbox (same for every unit)
st_ref  = D.spikeTimes(D.sp.clu == D.unitIDs(1));
[~, bins_s] = psthAndBA(st_ref, onset_s, win, bin_s);
nBins   = numel(bins_s);
psth    = zeros(nUnits, nBins);

% Per-unit PSTH: psthAndBA returns mean(binnedArray/binSize) → spikes/s
for u = 1:nUnits
    uid        = D.unitIDs(u);
    st         = D.spikeTimes(D.sp.clu == uid);
    psth(u,:)  = psthAndBA(st, onset_s, win, bin_s);
end

if use_ms
    centers = bins_s * 1000;          % s → ms
else
    centers = bins_s / med_dur_s;     % s → normalized phase (0–1)
end
end

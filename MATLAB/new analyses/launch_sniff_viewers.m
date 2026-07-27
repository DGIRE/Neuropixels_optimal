function launch_sniff_viewers(D)
%% LAUNCH_SNIFF_VIEWERS  Open three interactive viewers with sniff onsets as events.
%
%  Replaces step-7 static plots with three fully interactive viewers:
%
%    Figure 1 — psthViewer
%      Per-unit PSTH + raster + tuning curve.  Each sniff onset is one event.
%      Trials grouped by ETH on (2) / off (1) — toggle with 't'.
%      Navigate units: left/right arrows.  Adjust smoothing: up/down.
%
%    Figure 2 — popRasterViewer
%      Population raster scrollable in absolute time.  Sniff onsets, ETH-on
%      and ETH-off transitions shown as coloured vertical lines.  SNF and ETH
%      signals overlaid as traces.  Press 'y' to cycle y-axis orderings
%      (depth / mean sniff phase), 'c' to cycle colorings (depth / mean FR).
%
%    Figure 3 — sniff_raster_gui
%      evRastersGUI-style per-unit viewer with three alignment panels:
%        Panel 1 — all sniffs, sorted chronologically
%        Panel 2 — all sniffs, sorted by cycle duration (short→long)
%        Panel 3 — all sniffs, sorted by ETH level (no-ETH / ETH-on PSTH traces)
%      Probe scatter on left; click to jump to nearest unit by depth.
%      ACG (linear + log) updates per unit.
%
%  Usage:
%    launch_sniff_viewers(D)    % called automatically from PIPELINE step 7
%
%  Inputs:
%    D — pipeline D struct.  Required fields:
%          spikeTimes, sp.clu, unitIDs          (load_experiment_data)
%          sniff_onsets_s, sniff_dur_s          (compute_sniff_phase)
%          ETH, ETH_thr, eth_threshold, LV_Fs  (threshold_eth / load)
%          SNF                                  (load_experiment_data)
%          unitDepths, unitMeanSniffPhase       (compute_spike_phase)
%
%  Dependencies (must be on MATLAB path via TOOLBOX_ROOT):
%    psthViewer, popRasterViewer  — spikes-master/visualization
%    psthAndBA, timestampsToBinned, rasterize, myACG, WithinRanges,
%    myGaussWin                   — spikes-master/core + utils
%    GUI Layout Toolbox (uix.*)   — required by popRasterViewer

if ~isfield(D,'sniff_onsets_s')
    error('OR:noOnsets','Run compute_sniff_phase(D) first.');
end

onset_s = D.sniff_onsets_s(:);

% ── LV time axis and ETH binary signal ────────────────────────────────────
lv_t    = (0:numel(D.ETH)-1)' / D.LV_Fs;
eth_thr = 0;
if isfield(D,'eth_threshold'); eth_thr = D.eth_threshold; end
eth_bin = D.ETH > eth_thr;   % logical: 1 = ETH above threshold

% ETH state at each sniff onset (1=off, 2=on — trGroups for psthViewer)
eth_at_onset = interp1(lv_t, double(eth_bin), onset_s, 'nearest', 0);
trGroups     = eth_at_onset + 1;   % 1=no ETH, 2=ETH on

%% ── Figure 1: psthViewer ─────────────────────────────────────────────────
fprintf('  [1/3] psthViewer (per-unit, ETH on/off grouping)...\n');

win_psth = [-0.1, D.sniff_dur_s * 1.5];   % 100 ms pre-onset, 1.5 cycles post
psthViewer(D.spikeTimes, D.sp.clu, onset_s, win_psth, trGroups);

%% ── Figure 2: popRasterViewer ────────────────────────────────────────────
fprintf('  [2/3] popRasterViewer (population, scrollable timeline)...\n');

nU     = numel(D.unitIDs);
depths = D.unitDepths(:);
depths(isnan(depths)) = nanmean(depths);   % fill NaN before any min/max/linspace

% sp struct ----------------------------------------------------------------
sp.st   = D.spikeTimes;
sp.clu  = D.sp.clu;
sp.cids = D.unitIDs(:)';   % [1 x nU] row vector

% y-ordering 1: probe depth
sp.yAxOrderings(1).name      = 'depth (um)';
sp.yAxOrderings(1).yPos      = depths;
dLab = linspace(min(depths), max(depths), 5);
sp.yAxOrderings(1).yLabelVal = dLab;
sp.yAxOrderings(1).yLabel    = arrayfun(@(x)sprintf('%.0f',x), dLab, 'UniformOutput',false);

% y-ordering 2: mean sniff phase (NaN units → bottom, coded as -1)
ph = D.unitMeanSniffPhase(:);
ph(isnan(ph)) = -1;
sp.yAxOrderings(2).name      = 'mean sniff phase';
sp.yAxOrderings(2).yPos      = ph;
phLab = [-1 0 0.25 0.5 0.75 1.0];
sp.yAxOrderings(2).yLabelVal = phLab;
sp.yAxOrderings(2).yLabel    = {'NaN','0','0.25','0.5','0.75','1.0'};

% coloring 1: by depth (blue shallow → orange deep)
% Use explicit linear blend + clamp instead of interp1 to avoid NaN
% propagation when D.unitDepths has NaN entries.
d = depths; d(isnan(d)) = nanmean(d);   % fill any NaN depths with mean
dNorm = (d - min(d)) / max(max(d) - min(d), eps);
dNorm = max(0, min(1, dNorm));          % hard clamp to [0,1]
c1d = [0.15 0.45 0.85]; c2d = [0.85 0.35 0.10];
sp.colorings(1).name   = 'depth';
sp.colorings(1).colors = dNorm * (c2d - c1d) + c1d;   % nU x 3

% coloring 2: by mean firing rate
recDur = D.spikeTimes(end);
meanFR = arrayfun(@(uid) sum(D.sp.clu==uid)/recDur, D.unitIDs(:));
frNorm = (meanFR - min(meanFR)) / max(max(meanFR) - min(meanFR), eps);
frNorm = max(0, min(1, frNorm));
c1f = [0.15 0.15 0.15]; c2f = [1.0 0.85 0.0];
sp.colorings(2).name   = 'mean FR';
sp.colorings(2).colors = frNorm * (c2f - c1f) + c1f;  % nU x 3

% eventData ----------------------------------------------------------------
ed(1).name  = 'sniff onset';
ed(1).times = onset_s;
ed(1).spec  = {'Color',[0.15 0.75 0.15],'LineWidth',1.2};

ne = 1;
eth_rise = find(diff([0; eth_bin(:)]) ==  1);
eth_fall = find(diff([eth_bin(:); 0])  == -1);
if ~isempty(eth_rise)
    ne = ne+1;
    ed(ne).name  = 'ETH on';
    ed(ne).times = lv_t(eth_rise);
    ed(ne).spec  = {'Color',[0.9 0.35 0.10],'LineWidth',2.5};
end
if ~isempty(eth_fall)
    ne = ne+1;
    ed(ne).name  = 'ETH off';
    ed(ne).times = lv_t(eth_fall);
    ed(ne).spec  = {'Color',[0.20 0.40 0.85],'LineWidth',2.5};
end

% traces: SNF and ETH ------------------------------------------------------
tr(1).t     = lv_t;
tr(1).v     = D.SNF;
tr(1).name  = 'SNF';
tr(1).color = [0.15 0.75 0.15];

tr(2).t     = lv_t;
tr(2).v     = D.ETH;
tr(2).name  = 'ETH';
tr(2).color = [0.9 0.35 0.10];

% start view centred ~50 sniffs into recording
pars_pop.startTime = onset_s(min(50, numel(onset_s)));

% popRasterViewer requires the GUI Layout Toolbox (uix.*).
% sniff_pop_raster is an equivalent viewer built on standard MATLAB UI only.
sniff_pop_raster(sp, ed, tr, [], pars_pop);

%% ── Figure 3: sniff_raster_gui ───────────────────────────────────────────
fprintf('  [3/3] sniff_raster_gui (per-unit, 3-panel alignment)...\n');
sniff_raster_gui(D);

fprintf('  All viewers open.  Navigate units with arrow keys.\n');
end

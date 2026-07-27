function D = load_experiment_data(files)
%% LOAD_EXPERIMENT_DATA  Load sensor, LFP, and spike data from one experiment
%
%   D = load_experiment_data(files)
%
%   Loads three data sources and returns them in a single struct D:
%     (1) LabView sensor data   — ethanol and sniffing signals from the .dat file
%     (2) LFP sample            — first LFP_LOAD_SECS seconds of NP recording,
%                                 all AP channels, downsampled to LFP_OUT_FS Hz
%     (3) Spike data            — spike times, amplitudes, depths, and per-unit
%                                 summary from Kilosort output
%
%   INPUT:
%     files  — struct from OR_validate_files() with fields:
%                .datFile   LabView .dat filename (string)
%                .datPath   folder containing .dat file (string)
%                .binFile   SpikeGLX .bin filename (string)
%                .binPath   folder containing .bin and .meta files (string)
%                .ksDir     folder containing Kilosort output .npy files (string)
%
%   OUTPUT:  struct D with fields —
%
%     LABVIEW SENSOR DATA (from .dat file, sampled at ~125 Hz):
%       .ETH       ethanol sensor time series, normalized 0–1  [n×1]
%       .SNF       sniffing (cannula pressure) time series     [n×1]
%       .TR        trial number for each sample                [n×1]
%       .TS        within-trial timestamp in ms                [n×1]
%       .FR        camera frame number (used for NP alignment) [n×1]
%       .LV_Fs     LabView sampling rate in Hz (scalar)
%
%     LFP (from SpikeGLX .bin file):
%       .LFP             LFP sample, low-pass filtered & downsampled
%                        [nAPchannels × nSamples]  (µV)
%       .LFP_t           time vector for LFP sample (seconds)  [1 × nSamples]
%       .LFP_Fs          LFP sample rate after downsampling (Hz)
%       .LFP_duration_s  duration of loaded LFP segment (seconds)
%       .meta            SpikeGLX metadata struct (from .meta file)
%       .binFile         .bin filename   (for on-demand ReadBin calls)
%       .binPath         .bin folder     (for on-demand ReadBin calls)
%       .NP_Fs           native NP sampling rate in Hz
%
%     SPIKE DATA (from Kilosort output):
%       .sp              full Kilosort struct from loadKSdir()
%                        (sp.st, sp.clu, sp.temps, sp.xcoords, sp.ycoords, …)
%       .spikeTimes      spike times in seconds, all units  [nSpikes×1]
%       .spikeAmps       spike amplitude in µV              [nSpikes×1]
%       .spikeDepths     spike depth along probe in µm      [nSpikes×1]
%       .unitIDs         cluster IDs of included units      [nUnits×1]
%       .unitDepths      depth of each unit's peak channel  [nUnits×1]
%       .unitAmps        mean spike amplitude per unit      [nUnits×1]
%       .unitFiringRate  mean firing rate in spikes/s       [nUnits×1]
%       .xcoords         channel x-positions in µm          [nCh×1]
%       .ycoords         channel y-positions in µm          [nCh×1]

%% ---- Configuration ----------------------------------------
LFP_LOAD_SECS = 10;   % seconds of raw NP data to load for LFP sample
LFP_OUT_FS    = 1000; % target LFP output sampling rate (Hz)
LFP_LP_HZ     = 400;  % low-pass cutoff for LFP extraction (Hz)
%% -----------------------------------------------------------

D = struct();

%% =========================================================
%  PART 1 — LabView sensor data (.dat file)
%  =========================================================
fprintf('  Loading LabView sensor data: %s\n', files.datFile);
[D.TR, D.TS, D.FR, D.ETH, D.SNF, D.LV_Fs] = OR_loaddat(files.datFile, files.datPath);
fprintf('    ETH, SNF loaded  (%d samples, %.1f Hz)\n', numel(D.ETH), D.LV_Fs);

%% =========================================================
%  PART 2 — Neuropixels LFP sample (.bin + .meta files)
%  =========================================================
fprintf('  Loading NP metadata: %s\n', files.binFile);
D.meta    = OR_ReadMeta(files.binFile, files.binPath);
D.binFile = files.binFile;
D.binPath = files.binPath;
D.NP_Fs   = OR_SampRate(D.meta);

% Determine number of AP channels (exclude sync channel)
[nAP, ~, ~] = OR_ChannelCountsIM(D.meta);

% Number of samples to read for the LFP segment
nSampsRaw = min(floor(LFP_LOAD_SECS * D.NP_Fs), ...
                floor(str2double(D.meta.fileSizeBytes) / (2 * str2double(D.meta.nSavedChans))));

fprintf('  Loading LFP sample: first %.0f s, %d AP channels from %s\n', ...
    LFP_LOAD_SECS, nAP, files.binFile);

rawData = OR_ReadBin(0, nSampsRaw, D.meta, files.binFile, files.binPath);
rawAP   = rawData(1:nAP, :);  % AP channels only (drop sync channel)

% Convert raw int16 to µV
fI2V = OR_Int2Volts(D.meta);
[APgain, ~] = OR_ChanGainsIM(D.meta);
% Apply gain per channel
for ch = 1:nAP
    rawAP(ch,:) = rawAP(ch,:) * (fI2V / APgain(ch));
end

% Low-pass filter to extract LFP band
[b_lp, a_lp] = butter(4, LFP_LP_HZ / (D.NP_Fs / 2), 'low');
lfpFull = filtfilt(b_lp, a_lp, double(rawAP)')';  % [nAP × nSampsRaw]

% Downsample to target rate
dsRatio = round(D.NP_Fs / LFP_OUT_FS);
D.LFP   = lfpFull(:, 1:dsRatio:end);
D.LFP_t = (0:size(D.LFP,2)-1) / LFP_OUT_FS;
D.LFP_Fs = D.NP_Fs / dsRatio;
D.LFP_duration_s = nSampsRaw / D.NP_Fs;

fprintf('    LFP loaded: [%d ch × %d samples] at %.0f Hz (%.0f s)\n', ...
    size(D.LFP,1), size(D.LFP,2), D.LFP_Fs, D.LFP_duration_s);

%% =========================================================
%  PART 3 — Spike data (Kilosort output via spikes-master)
%  =========================================================
fprintf('  Loading Kilosort spike data: %s\n', files.ksDir);

% Load the full Kilosort struct (spike times, clusters, templates, coords)
if exist('loadKSdir', 'file')
    D.sp = loadKSdir(files.ksDir);
else
    error('OR:missingToolbox', ...
        'loadKSdir not found. Ensure spikes-master is on the MATLAB path.\n(Expected at: %s)', ...
        '\\172.25.226.40\GireLab Data\Room 359\NPDATA\Code\spikes-master');
end

% Get per-spike depth, amplitude using ksDriftmap (spikes-master)
if exist('ksDriftmap', 'file')
    [D.spikeTimes, D.spikeAmps, D.spikeDepths, ~] = ksDriftmap(files.ksDir);
else
    % Fallback: compute spike depths from templates directly
    warning('OR:missingFunction', ...
        'ksDriftmap not found. Computing spike depths from templates.');
    D.spikeTimes  = D.sp.st;
    D.spikeAmps   = [];
    D.spikeDepths = compute_depths_from_templates(D.sp);
end

% Channel coordinates (µm from probe tip)
D.xcoords = D.sp.xcoords;
D.ycoords = D.sp.ycoords;

% Per-unit summary: depth, amplitude, firing rate
uniqueUnits   = unique(D.sp.clu);
nUnits        = numel(uniqueUnits);
unitDepths    = nan(nUnits, 1);
unitAmps      = nan(nUnits, 1);
unitFiringRate= nan(nUnits, 1);
recordingDur  = D.sp.st(end);   % total recording duration in seconds

for u = 1:nUnits
    uid = uniqueUnits(u);
    spkMask = (D.sp.clu == uid);

    if ~isempty(D.spikeDepths)
        unitDepths(u) = mean(D.spikeDepths(spkMask));
    end
    if ~isempty(D.spikeAmps)
        unitAmps(u) = mean(D.spikeAmps(spkMask));
    end
    unitFiringRate(u) = sum(spkMask) / recordingDur;
end

D.unitIDs        = uniqueUnits;
D.unitDepths     = unitDepths;
D.unitAmps       = unitAmps;
D.unitFiringRate = unitFiringRate;

fprintf('    Spikes loaded: %d spikes, %d units\n', ...
    numel(D.spikeTimes), nUnits);
fprintf('    Depth range: %.0f – %.0f µm\n', ...
    min(D.unitDepths), max(D.unitDepths));

end  % load_experiment_data


%% =========================================================
%  LOCAL HELPER — compute unit depths from templates
%  (fallback when ksDriftmap is unavailable)
%  =========================================================
function spikeDepths = compute_depths_from_templates(sp)
% For each spike, find the template's peak channel, then look up ycoords
% sp.temps is [nTemplates × nTimepoints × nChannels]
nTemplates  = size(sp.temps, 1);
peakChan    = zeros(nTemplates, 1);
tempsMaxAbs = squeeze(max(abs(sp.temps), [], 2));  % [nTemplates × nChannels]
for t = 1:nTemplates
    [~, peakChan(t)] = max(tempsMaxAbs(t,:));
end
templateDepths = sp.ycoords(peakChan);

% Assign per-spike depth from its template
spikeTemplateIdx = sp.spikeTemplates + 1;  % 0-indexed → 1-indexed
spikeDepths = templateDepths(spikeTemplateIdx);
end


%% =========================================================
%  LOCAL SpikeGLX HELPER FUNCTIONS
%  (wrapped here so load_experiment_data.m is self-contained)
%  =========================================================

function srate = OR_SampRate(meta)
if strcmp(meta.typeThis, 'imec')
    srate = str2double(meta.imSampRate);
else
    srate = str2double(meta.niSampRate);
end
end

function [AP, LF, SY] = OR_ChannelCountsIM(meta)
M  = str2num(meta.snsApLfSy); %#ok<ST2NM>
AP = M(1); LF = M(2); SY = M(3);
end

function fI2V = OR_Int2Volts(meta)
if strcmp(meta.typeThis, 'imec')
    maxInt = 512;
    if isfield(meta, 'imMaxInt'), maxInt = str2num(meta.imMaxInt); end %#ok<ST2NM>
    fI2V = str2double(meta.imAiRangeMax) / maxInt;
else
    fI2V = str2double(meta.niAiRangeMax) / 32768;
end
end

function [APgain, LFgain] = OR_ChanGainsIM(meta)
probeType = 0;
if isfield(meta,'imDatPrb_type'), probeType = str2num(meta.imDatPrb_type); end %#ok<ST2NM>
if (probeType == 21) || (probeType == 24)
    [AP, LF, ~] = OR_ChannelCountsIM(meta);
    APgain = 80 * ones(AP, 1);
    LFgain = zeros(LF, 1);
else
    if isfield(meta,'typeEnabled')  % 3A
        C = textscan(meta.imroTbl, '(%*s %*s %*s %d %d', 'EndOfLine',')', 'HeaderLines',1);
    else                            % 3B
        C = textscan(meta.imroTbl, '(%*s %*s %*s %d %d %*s', 'EndOfLine',')', 'HeaderLines',1);
    end
    APgain = double(cell2mat(C(1)));
    LFgain = double(cell2mat(C(2)));
end
end

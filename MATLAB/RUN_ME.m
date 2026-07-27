%% ============================================================
%  RUN_ME.m  —  Olfactory Research NP Data Loader
%  ============================================================
%
%  *** RUN THIS FILE TO START ***
%
%  This script performs two steps for a selected experiment:
%
%    STEP 1 — Loads sensor data (ethanol, sniffing) and Neuropixels
%             data (LFP, spike times) into the workspace variable 'D'
%
%    STEP 2 — Plots the location of spiking neurons on the NP probe
%
%  HOW TO USE:
%    1. Press F5 (or click Run) to launch the experiment folder picker
%    2. Navigate to the experiment folder on the server and click OK
%    3. If any required files are missing, a list will be printed and
%       the script will stop — no data will be loaded
%    4. On success, all data is in workspace variable 'D'
%       (see load_experiment_data.m for the full list of fields)
%
%  REQUIRED FILES in the selected experiment folder (searched recursively):
%    LabView sensor data  :  one *.dat file
%    SpikeGLX recording   :  one *.bin file  +  matching *.meta file
%    Kilosort output      :  spike_times.npy, spike_clusters.npy,
%                            templates.npy, channel_map.npy,
%                            channel_positions.npy
%
%  DATA SERVER ROOT:  X:\GireLab Data\Room 359\NPDATA\DATA
% =============================================================

clear; close all; clc;

%% --- Toolbox paths (lab server) ---
CODE_ROOT     = '\\172.25.226.40\GireLab\GireLab Data\Room 359\NPDATA\Code';
SPIKES_MASTER = fullfile(CODE_ROOT, 'spikes-master');
NPY_MATLAB    = fullfile(CODE_ROOT, 'npy-matlab-master');

for pth = {SPIKES_MASTER, NPY_MATLAB}
    if isfolder(pth{1})
        addpath(genpath(pth{1}));
    else
        warning('OR:toolboxNotFound', ...
            'Toolbox folder not found:\n  %s\nFunctions from this toolbox may be unavailable.', ...
            pth{1});
    end
end

% Add this project's helper library
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, 'lib'));

%% --- Select experiment folder ---
SERVER_ROOT = 'X:\GireLab Data\Room 359\NPDATA\DATA';
if ~isfolder(SERVER_ROOT)
    warning('OR:serverNotFound', ...
        'Server root not accessible:\n  %s\nStarting folder picker from current directory.', ...
        SERVER_ROOT);
    SERVER_ROOT = pwd;
end

expDir = uigetdir(SERVER_ROOT, 'Select Experiment Folder on Server');
if isequal(expDir, 0)
    disp('No folder selected. Exiting.');
    return;
end

fprintf('\n==============================================\n');
fprintf(' Experiment: %s\n', expDir);
fprintf('==============================================\n\n');

%% --- Validate required files ---
fprintf('[Checking files...]\n');
[files, missing] = OR_validate_files(expDir);

if ~isempty(missing)
    fprintf('\n*** CANNOT LOAD — Missing required files: ***\n\n');
    for k = 1:numel(missing)
        fprintf('  [MISSING]  %s\n', missing{k});
    end
    fprintf('\nPlease confirm the correct folder was selected and that all\n');
    fprintf('required files are present, then run again.\n\n');
    return;
end
fprintf('  All required files found.\n\n');

%% --- STEP 1: Load data ---
fprintf('[STEP 1] Loading experiment data...\n');
D = load_experiment_data(files);

% Make data available in base workspace for interactive use
assignin('base', 'D',     D);
assignin('base', 'files', files);

fprintf('\nData loaded into workspace variable ''D''.\n');
fprintf('  Sensor data   : D.ETH, D.SNF, D.TR, D.TS, D.FR  (LabView, %.0f Hz)\n', D.LV_Fs);
fprintf('  LFP sample    : D.LFP  [%d ch x %d samples, %.0f Hz, first %.0f s]\n', ...
    size(D.LFP,1), size(D.LFP,2), D.LFP_Fs, D.LFP_duration_s);
fprintf('  Spike times   : D.spikeTimes  (%d spikes)\n', numel(D.spikeTimes));
fprintf('  Units         : D.unitIDs     (%d units)\n', numel(D.unitIDs));
fprintf('\n  To read additional LFP segments, use:\n');
fprintf('    dataArray = OR_ReadBin(startSamp, nSamps, D.meta, D.binFile, D.binPath)\n\n');

%% --- STEP 2: Plot unit locations ---
fprintf('[STEP 2] Plotting unit locations on NP probe...\n');
plot_unit_locations(D);

fprintf('\n=== Done. Use variable D to explore data. ===\n\n');

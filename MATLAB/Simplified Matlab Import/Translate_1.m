%% Translate-1.m  —  Neuropixels Import & Export to .mat
%
%   Identical to SIMPLE.m but additionally saves the loaded data struct D
%   to MTLB-TEST.mat in the same folder as this script.
%
%   File paths.txt (at C:\Projects\Neuropixels\) must contain:
%     Directory:       <experiment root folder>
%     DAT File:        <LabView .dat filename>
%     Kilosort folder: <path to Kilosort output folder>
%     ETH threshold:   <numeric value>  (optional — stored in D for later use)
%
%   The SpikeGLX .bin file is located automatically under the experiment
%   directory, so it does not need to appear in File paths.txt.
%
%   REQUIREMENTS
%   ------------
%   TOOLBOX_ROOT (below) must point to the TOOLBOX folder.

clear; close all; clc;

fprintf('================================================\n');
fprintf('  Neuropixels — Translate-1 Import & Export\n');
fprintf('================================================\n\n');

%% -----------------------------------------------------------------------
%% CONFIGURATION  (edit these if your folder layout changes)
%% -----------------------------------------------------------------------

PATHS_FILE   = 'C:\Projects\Neuropixels\File paths.txt';
TOOLBOX_ROOT = 'C:\Projects\Neuropixels\Organized Code\TOOLBOX';

%% -----------------------------------------------------------------------
%% PATH SETUP
%% -----------------------------------------------------------------------

% Organized Code folder — contains load_experiment_data, plot_unit_locations
organized_dir = 'C:\Projects\Neuropixels\Organized Code';
addpath(organized_dir);

% lib folder — OR_validate_files, OR_loaddat, OR_ReadBin, OR_ReadMeta
lib_dir = fullfile(organized_dir, 'lib');
if exist(lib_dir, 'dir')
    addpath(lib_dir);
else
    warning('lib folder not found at: %s', lib_dir);
end

% External toolboxes
for tp = {fullfile(TOOLBOX_ROOT,'spikes-master'), fullfile(TOOLBOX_ROOT,'npy-matlab-master')}
    p = tp{1};
    if exist(p, 'dir')
        addpath(genpath(p));
        fprintf('Added toolbox: %s\n', p);
    else
        warning('Toolbox not found: %s', p);
    end
end

%% -----------------------------------------------------------------------
%% PARSE FILE PATHS.TXT
%% -----------------------------------------------------------------------

fprintf('\n--- Reading File paths.txt ---\n');

if ~isfile(PATHS_FILE)
    error('Cannot find File paths.txt at:\n  %s', PATHS_FILE);
end

expDir       = '';
datFile      = '';
ksDir        = '';
ethThreshold = NaN;

fid = fopen(PATHS_FILE, 'r');
while ~feof(fid)
    raw = fgetl(fid);
    if ~ischar(raw), break; end
    line = strtrim(raw);
    if isempty(line) || line(1) == '#', continue; end

    colonIdx = strfind(line, ':');
    if isempty(colonIdx), continue; end
    key = strtrim(line(1 : colonIdx(1)-1));
    val = strtrim(line(colonIdx(1)+1 : end));

    switch key
        case 'Directory',       expDir       = val;
        case 'DAT File',        datFile      = val;
        case 'Kilosort folder', ksDir        = val;
        case 'ETH threshold',   ethThreshold = str2double(val);
    end
end
fclose(fid);

fprintf('Experiment folder : %s\n', expDir);
fprintf('DAT file          : %s\n', datFile);
fprintf('Kilosort folder   : %s\n', ksDir);
if ~isnan(ethThreshold)
    fprintf('ETH threshold     : %.4f\n', ethThreshold);
end

%% -----------------------------------------------------------------------
%% STEP 1 — BUILD FILES STRUCT & VALIDATE
%% -----------------------------------------------------------------------

fprintf('\n--- Step 1: Validating files ---\n');

if isempty(expDir) || ~exist(expDir, 'dir')
    error('Experiment directory not found:\n  %s\nCheck File paths.txt.', expDir);
end
if isempty(datFile)
    error('DAT File not specified in File paths.txt.');
end
if isempty(ksDir) || ~exist(ksDir, 'dir')
    error('Kilosort folder not found:\n  %s\nCheck File paths.txt.', ksDir);
end

% Build files struct directly from parsed paths — no GUI dialogs
files.datFile = datFile;
files.datPath = expDir;
files.ksDir   = ksDir;

% Auto-locate the SpikeGLX AP-band .bin file under expDir
binHits = dir(fullfile(expDir, '**', '*.bin'));
binHits = binHits(~[binHits.isdir]);
isAP    = cellfun(@(n) ~contains(n, '.lf.'), {binHits.name});
binHits = binHits(isAP);

if isempty(binHits)
    error('No SpikeGLX .bin file found under:\n  %s', expDir);
end
if numel(binHits) > 1
    warning('%d .bin files found; using first: %s', numel(binHits), binHits(1).name);
end
files.binFile = binHits(1).name;
files.binPath = binHits(1).folder;

% Verify matching .meta file exists alongside .bin
[~, stem] = fileparts(files.binFile);
if ~isfile(fullfile(files.binPath, [stem '.meta']))
    error('Matching .meta file not found alongside %s', files.binFile);
end

% Verify required Kilosort .npy files
KS_REQUIRED = {'spike_times.npy', 'spike_clusters.npy', 'templates.npy', ...
                'channel_map.npy', 'channel_positions.npy'};
missing = {};
for k = 1:numel(KS_REQUIRED)
    if ~isfile(fullfile(ksDir, KS_REQUIRED{k}))
        missing{end+1} = KS_REQUIRED{k}; %#ok<AGROW>
    end
end
if ~isempty(missing)
    fprintf('Missing Kilosort files:\n');
    for m = 1:numel(missing), fprintf('  - %s\n', missing{m}); end
    error('Required Kilosort files are missing. Cannot continue.');
end

fprintf('All required files found.\n');
fprintf('  .dat  : %s\n', fullfile(files.datPath, files.datFile));
fprintf('  .bin  : %s\n', fullfile(files.binPath, files.binFile));
fprintf('  ksDir : %s\n', files.ksDir);

%% -----------------------------------------------------------------------
%% STEP 2 — LOAD EXPERIMENT DATA
%% -----------------------------------------------------------------------

fprintf('\n--- Step 2: Loading experiment data ---\n');

D = load_experiment_data(files);

% Store ETH threshold from File paths.txt for later use (e.g., threshold_eth)
if ~isnan(ethThreshold)
    D.eth_threshold_default = ethThreshold;
end

fprintf('Data loaded: %d valid units, %d LV samples\n', ...
    length(D.unitIDs), length(D.ETH));

%% -----------------------------------------------------------------------
%% STEP 3 — PLOT UNIT LOCATIONS
%% -----------------------------------------------------------------------

fprintf('\n--- Step 3: Plotting unit locations ---\n');

plot_unit_locations(D);

%% -----------------------------------------------------------------------
%% EXPORT D TO .MAT FILE
%% -----------------------------------------------------------------------

fprintf('\n--- Exporting D to MTLB-TEST.mat ---\n');

script_dir = fileparts(mfilename('fullpath'));
out_path   = fullfile(script_dir, 'MTLB-TEST.mat');
save(out_path, 'D', '-v7.3');

fprintf('Saved: %s\n', out_path);

fprintf('\n================================================\n');
fprintf('  Import complete.\n');
fprintf('  Workspace variable D contains all results.\n');
fprintf('  Data exported to MTLB-TEST.mat\n');
fprintf('================================================\n');

function [files, missing] = OR_validate_files(expDir)
%% OR_VALIDATE_FILES  Search an experiment folder for all required files
%
%   [files, missing] = OR_validate_files(expDir)
%
%   Searches expDir and all subfolders for the files needed to load one
%   experiment.  If multiple matching files are found, a dialog lets the
%   user pick which to use.
%
%   INPUT:
%     expDir   full path to the experiment folder (string)
%
%   OUTPUT:
%     files    struct with fields:
%                .datFile   LabView .dat filename  ('' if not found)
%                .datPath   folder containing .dat ('' if not found)
%                .binFile   SpikeGLX .bin filename ('' if not found)
%                .binPath   folder containing .bin ('' if not found)
%                .ksDir     Kilosort output folder  ('' if not found)
%
%     missing  cell array of human-readable descriptions of any missing
%              files.  Empty ({}) means all files were found.

files   = struct('datFile','', 'datPath','', ...
                 'binFile','', 'binPath','', ...
                 'ksDir','');
missing = {};

%% ---- LabView .dat file ----
datHits = dir(fullfile(expDir, '**', '*.dat'));
datHits = datHits(~[datHits.isdir]);

if isempty(datHits)
    missing{end+1} = 'LabView sensor file  (*.dat)';
else
    idx = pick_file(datHits, 'Select LabView .dat file');
    files.datFile = datHits(idx).name;
    files.datPath = datHits(idx).folder;
end

%% ---- SpikeGLX .bin file ----
binHits = dir(fullfile(expDir, '**', '*.bin'));
binHits = binHits(~[binHits.isdir]);

% Prefer AP-band files (exclude *.lf.bin if present)
isAP = ~cellfun(@(n) contains(n, '.lf.'), {binHits.name});
if any(isAP)
    binHits = binHits(isAP);
end

if isempty(binHits)
    missing{end+1} = 'SpikeGLX binary file  (*.bin)';
else
    idx = pick_file(binHits, 'Select SpikeGLX .bin file');
    files.binFile = binHits(idx).name;
    files.binPath = binHits(idx).folder;

    % Verify matching .meta file exists alongside .bin
    [~, stem, ~] = fileparts(files.binFile);
    metaPath = fullfile(files.binPath, [stem '.meta']);
    if ~isfile(metaPath)
        missing{end+1} = sprintf('SpikeGLX metadata file  (%s.meta)', stem);
    end
end

%% ---- Kilosort output (spike_times.npy is the key marker) ----
KS_REQUIRED = {'spike_times.npy', 'spike_clusters.npy', ...
                'templates.npy',   'channel_map.npy',    ...
                'channel_positions.npy'};

stHits = dir(fullfile(expDir, '**', 'spike_times.npy'));
stHits = stHits(~[stHits.isdir]);

if isempty(stHits)
    missing{end+1} = 'Kilosort output  (spike_times.npy not found)';
else
    % If multiple hits, pick one
    folderList = {stHits.folder};
    if numel(stHits) > 1
        chosenIdx = listdlg('PromptString', 'Select Kilosort output directory:', ...
                            'SelectionMode', 'single', ...
                            'ListString', folderList, ...
                            'ListSize', [500, 150]);
        if isempty(chosenIdx), chosenIdx = 1; end
    else
        chosenIdx = 1;
    end
    files.ksDir = stHits(chosenIdx).folder;

    % Check that every required Kilosort file is present
    for k = 2:numel(KS_REQUIRED)   % spike_times.npy already confirmed above
        fPath = fullfile(files.ksDir, KS_REQUIRED{k});
        if ~isfile(fPath)
            missing{end+1} = sprintf('Kilosort output file  (%s)', KS_REQUIRED{k}); %#ok<AGROW>
        end
    end
end

end  % OR_validate_files


%% ---- Local helper: pick among multiple file hits ----
function idx = pick_file(hits, promptStr)
if numel(hits) == 1
    idx = 1;
    return;
end
% Build display list: relative path from common root
names = cellfun(@(f,n) fullfile(f,n), {hits.folder}, {hits.name}, ...
                'UniformOutput', false);
chosenIdx = listdlg('PromptString', [promptStr, ':'], ...
                    'SelectionMode', 'single', ...
                    'ListString', names, ...
                    'ListSize', [600, 150]);
if isempty(chosenIdx)
    chosenIdx = 1;
    warning('OR:defaultFile', 'No selection made; using first match: %s', names{1});
end
idx = chosenIdx;
end

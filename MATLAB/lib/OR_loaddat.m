function [TR, TS, FR, ETH, SNF, Fs] = OR_loaddat(fn, fp)
%% OR_LOADDAT  Read LabView .dat sensor file into workspace arrays
%
%   [TR, TS, FR, ETH, SNF, Fs] = OR_loaddat(fn, fp)
%
%   Reads the binary LabView .dat file and extracts the ethanol sensor,
%   sniffing cannula, trial number, timestamp, and frame number arrays.
%
%   INPUT:
%     fn   filename of the .dat file  (string, e.g. 'session.dat')
%     fp   folder containing the file (string, e.g. 'C:\Data\2024-01-15')
%
%   OUTPUT:
%     TR   trial number for each sample, as labeled by LabView      [n×1]
%     TS   within-trial timestamp in ms, resets at each trial start  [n×1]
%     FR   camera frame number (0–875), resets each trial            [n×1]
%          A change of 1 in FR indicates a TTL pulse sent to NP for
%          time alignment; used by AlignStamps.m
%     ETH  ethanol sensor time series, normalized to [0, 1]          [n×1]
%     SNF  sniffing cannula pressure time series                     [n×1]
%     Fs   LabView sampling rate in Hz (computed from TS; ~125 Hz)
%
%   FILE FORMAT NOTES:
%     The .dat file is big-endian int32 binary.  Each record is delimited
%     by the sentinel value -10.  Relevant fields at fixed offsets from
%     each sentinel (0-indexed):
%       offset +1  : TR  — trial number
%       offset +2  : TS  — timestamp (ms)
%       offset +3  : FR  — frame number
%       offset +4  : ETH — ethanol sensor
%       offset +11 : SNF — sniffing sensor

fn = char(fn);   % ensure char in case cell was passed

fileID = fopen(fullfile(fp, fn), 'r');
if fileID < 0
    error('OR:fileNotFound', 'Cannot open file: %s', fullfile(fp, fn));
end
data   = fread(fileID, 'int32', 'ieee-be');
fclose(fileID);

% Locate sentinel records (value = -10), skip the first 20 (header artefacts)
n_ten = find(data == -10);
if numel(n_ten) < 21
    error('OR:badFile', ...
        'File does not appear to contain valid LabView records:\n  %s', ...
        fullfile(fp, fn));
end
n_ten = n_ten(20:end);

TR  = data(n_ten + 1);   % trial number
TS  = data(n_ten + 2);   % within-trial timestamp (ms)
FR  = data(n_ten + 3);   % camera frame number
ETH = data(n_ten + 4);   % ethanol sensor
SNF = data(n_ten + 11);  % sniffing sensor

% Compute actual sampling rate from timestamps
% (take median of within-trial intervals to exclude ITI jumps)
dTS = diff(TS);
dTS(dTS <= 0) = [];           % remove ITI jumps and zeroes
Fs  = 1000 / median(dTS);    % convert ms intervals to Hz

% Normalize ETH to [0, 1]
ETH = double(ETH) - min(double(ETH));
if max(ETH) > 0
    ETH = ETH / max(ETH);
end
SNF = double(SNF);

end

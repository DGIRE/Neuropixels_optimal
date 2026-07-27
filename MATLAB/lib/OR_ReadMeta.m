function meta = OR_ReadMeta(binName, path)
%% OR_READMETA  Parse a SpikeGLX .meta file into a MATLAB struct
%
%   meta = OR_ReadMeta(binName, path)
%
%   Reads the .meta file that accompanies a SpikeGLX .bin recording and
%   returns all key–value pairs as fields of a struct.
%
%   INPUT:
%     binName   filename of the .bin recording (e.g. 'session_g0_t0.imec0.ap.bin')
%     path      folder containing both the .bin and .meta files
%
%   OUTPUT:
%     meta   struct where each field corresponds to one key in the .meta
%            file.  Fields of particular interest:
%              .typeThis        — 'imec' or 'nidq'
%              .imSampRate      — AP sampling rate (30000 for standard NP)
%              .nSavedChans     — total saved channels (including sync)
%              .fileSizeBytes   — binary file size in bytes
%              .snsApLfSy       — channel count string '[nAP nLF nSY]'
%              .imroTbl         — IMRO table for gain look-up
%
%   USAGE EXAMPLE:
%     meta = OR_ReadMeta('session.ap.bin', 'X:\Data\2024-01-15\imec0');
%     Fs   = str2double(meta.imSampRate);   % sampling rate
%
%   SOURCE:
%     Adapted from SpikeGLX DemoReadSGLXData (Janelia / Bill Karsh).
%     Original: github.com/jenniferColonell/SpikeGLX_Datafile_Tools

% Derive .meta filename from .bin filename
[~, stem, ~] = fileparts(binName);
metaName     = [stem, '.meta'];
metaFullPath = fullfile(path, metaName);

if ~isfile(metaFullPath)
    error('OR:metaNotFound', ...
        'Metadata file not found:\n  %s\nEnsure the .meta file is in the same folder as the .bin file.', ...
        metaFullPath);
end

fid = fopen(metaFullPath, 'r');
C   = textscan(fid, '%[^=] = %[^\r\n]');
fclose(fid);

meta = struct();
for i = 1:numel(C{1})
    tag = strtrim(C{1}{i});
    if tag(1) == '~'
        tag = tag(2:end);   % strip leading tilde from some field names
    end
    meta.(tag) = strtrim(C{2}{i});
end

end

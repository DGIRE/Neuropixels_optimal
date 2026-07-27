function dataArray = OR_ReadBin(samp0, nSamp, meta, binName, path)
%% OR_READBIN  Read a chunk of raw data from a SpikeGLX .bin file
%
%   dataArray = OR_ReadBin(samp0, nSamp, meta, binName, path)
%
%   Returns a [nChannels × nSamp] matrix of raw int16 values read from
%   the SpikeGLX binary file.  The caller is responsible for converting
%   raw counts to volts using the gain information in meta.
%
%   INPUT:
%     samp0    first sample to read (0-based integer index)
%     nSamp    number of samples to read (integer)
%     meta     metadata struct from OR_ReadMeta()
%     binName  .bin filename (string)
%     path     folder containing the .bin file (string)
%
%   OUTPUT:
%     dataArray   [nChannels × nSamp] matrix of raw int16 values stored
%                 as doubles.  Rows correspond to saved channels in the
%                 order they appear in the file; the last row is the sync
%                 channel for imec recordings.
%
%   NOTES:
%     • samp0 and nSamp must be integers.  Pass 0 for samp0 to start
%       reading from the beginning of the file.
%     • If (samp0 + nSamp) exceeds the file length, reading stops at
%       the last available sample; the returned matrix may have fewer
%       columns than requested.
%     • To convert to microvolts for imec AP channels:
%         fI2V   = str2double(meta.imAiRangeMax) / 512;
%         gainAP = <channel gain from imroTbl>;
%         dataVolts = dataArray(ch,:) * fI2V / gainAP * 1e6;
%
%   USAGE EXAMPLE — read first 10 seconds of AP channel 1:
%     meta  = OR_ReadMeta('session.ap.bin', dataDir);
%     Fs    = str2double(meta.imSampRate);
%     raw   = OR_ReadBin(0, Fs*10, meta, 'session.ap.bin', dataDir);
%     ch1   = raw(1, :);   % raw counts, channel 1
%
%   SOURCE:
%     Adapted from SpikeGLX DemoReadSGLXData (Janelia / Bill Karsh).

nChan     = str2double(meta.nSavedChans);
nFileSamp = floor(str2double(meta.fileSizeBytes) / (2 * nChan));

samp0 = max(floor(samp0), 0);
nSamp = min(floor(nSamp), nFileSamp - samp0);

if nSamp <= 0
    warning('OR:ReadBin', ...
        'Requested range (samp0=%d) is beyond the end of the file (%d samples).', ...
        samp0, nFileSamp);
    dataArray = zeros(nChan, 0);
    return;
end

binFullPath = fullfile(path, binName);
fid = fopen(binFullPath, 'rb');
if fid < 0
    error('OR:fileNotFound', 'Cannot open binary file:\n  %s', binFullPath);
end

fseek(fid, samp0 * 2 * nChan, 'bof');
dataArray = fread(fid, [nChan, nSamp], 'int16=>double');
fclose(fid);

end

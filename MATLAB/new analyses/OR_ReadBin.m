function dataArray = OR_ReadBin(samp0,nSamp,meta,binName,path)
%% OR_READBIN  Read raw chunk from SpikeGLX .bin as [nChan x nSamp] double
nChan=str2double(meta.nSavedChans);
nFileSamp=floor(str2double(meta.fileSizeBytes)/(2*nChan));
samp0=max(floor(samp0),0); nSamp=min(floor(nSamp),nFileSamp-samp0);
if nSamp<=0; dataArray=zeros(nChan,0); return; end
fid=fopen(fullfile(path,binName),'rb');
if fid<0; error('OR:fileNotFound','Cannot open: %s',binName); end
fseek(fid,samp0*2*nChan,'bof'); dataArray=fread(fid,[nChan,nSamp],'int16=>double'); fclose(fid);
end

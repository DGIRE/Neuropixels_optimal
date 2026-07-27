function meta = OR_ReadMeta(binName,path)
%% OR_READMETA  Parse SpikeGLX .meta file into a struct
[~,stem,~]=fileparts(binName); metaPath=fullfile(path,[stem '.meta']);
if ~isfile(metaPath); error('OR:metaNotFound','Meta file not found:\n  %s',metaPath); end
fid=fopen(metaPath,'r'); C=textscan(fid,'%[^=] = %[^\r\n]'); fclose(fid);
meta=struct();
for i=1:numel(C{1}); tag=strtrim(C{1}{i}); if tag(1)=='~'; tag=tag(2:end); end; meta.(tag)=strtrim(C{2}{i}); end
end

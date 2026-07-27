function [files, missing] = OR_validate_files(expDir)
%% OR_VALIDATE_FILES  Find required experiment files in a folder tree
files=struct('datFile','','datPath','','binFile','','binPath','','ksDir',''); missing={};
d=dir(fullfile(expDir,'**','*.dat')); d=d(~[d.isdir]);
if isempty(d); missing{end+1}='LabView sensor file (*.dat)';
else; i=pick(d,'Select LabView .dat'); files.datFile=d(i).name; files.datPath=d(i).folder; end
b=dir(fullfile(expDir,'**','*.bin')); b=b(~[b.isdir]);
ap=~cellfun(@(n)contains(n,'.lf.'),{b.name}); if any(ap); b=b(ap); end
if isempty(b); missing{end+1}='SpikeGLX binary file (*.bin)';
else; i=pick(b,'Select SpikeGLX .bin'); files.binFile=b(i).name; files.binPath=b(i).folder;
    [~,s,~]=fileparts(files.binFile);
    if ~isfile(fullfile(files.binPath,[s '.meta'])); missing{end+1}=['Missing .meta for ' s]; end; end
KS={'spike_times.npy','spike_clusters.npy','templates.npy','channel_map.npy','channel_positions.npy'};
st=dir(fullfile(expDir,'**','spike_times.npy')); st=st(~[st.isdir]);
if isempty(st); missing{end+1}='Kilosort output (spike_times.npy not found)';
else
    if numel(st)>1
        idx=listdlg('PromptString','Select Kilosort folder:','SelectionMode','single',...
            'ListString',{st.folder},'ListSize',[500,120]); if isempty(idx); idx=1; end
    else; idx=1; end
    files.ksDir=st(idx).folder;
    for k=2:numel(KS); if ~isfile(fullfile(files.ksDir,KS{k}))
        missing{end+1}=['Kilosort: ' KS{k}]; end; end; end
end
function i=pick(hits,msg)
if numel(hits)==1; i=1; return; end
names=cellfun(@(f,n)fullfile(f,n),{hits.folder},{hits.name},'UniformOutput',false);
i=listdlg('PromptString',[msg ':'],'SelectionMode','single','ListString',names,'ListSize',[600,120]);
if isempty(i); i=1; end
end

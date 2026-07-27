function [TR,TS,FR,ETH,SNF,Fs] = OR_loaddat(fn,fp)
%% OR_LOADDAT  Read LabView .dat binary file -> TR,TS,FR,ETH(norm),SNF,Fs
fn=char(fn); fid=fopen(fullfile(fp,fn),'r');
if fid<0; error('OR:fileNotFound','Cannot open: %s',fullfile(fp,fn)); end
data=fread(fid,'int32','ieee-be'); fclose(fid);
n10=find(data==-10); if numel(n10)<21; error('OR:badFile','Not a valid .dat: %s',fn); end
n10=n10(20:end);
TR=data(n10+1); TS=data(n10+2); FR=data(n10+3); ETH=data(n10+4); SNF=data(n10+11);
dTS=diff(TS); dTS(dTS<=0)=[];
Fs=1000/median(dTS);
ETH=double(ETH)-min(double(ETH)); if max(ETH)>0; ETH=ETH/max(ETH); end
SNF=double(SNF);
end

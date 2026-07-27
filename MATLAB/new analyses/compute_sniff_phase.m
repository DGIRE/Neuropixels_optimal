function D = compute_sniff_phase(D, threshold_std)
%% COMPUTE_SNIFF_PHASE  Detect sniff onsets; build normalized phase signal
%  Method: low-pass filter SNF (40 Hz), z-score, threshold crossing.
%  Phase 0 = sniff onset, 1 = next sniff onset (or 500 ms cap).
%  Samples outside all cycles = -1.
%  Adds: D.SNF_PH, D.sniff_onsets, D.sniff_onsets_s, D.sniff_dur_s, D.sniff_thr
if nargin<2||isempty(threshold_std); threshold_std=pick_threshold(D.SNF,D.LV_Fs); end
LV_Fs=D.LV_Fs; SNF=D.SNF(:); nSNF=numel(SNF);
fco=min(40,LV_Fs/2*0.9);
lpf=designfilt('lowpassfir','FilterOrder',20,'CutoffFrequency',fco,'SampleRate',LV_Fs);
SNF_filt=filtfilt(lpf,SNF);
SNF_z=(SNF_filt-mean(SNF_filt))/std(SNF_filt);
below=SNF_z<threshold_std; dbelow=diff([0;below]); onsets=find(dbelow==1);
MIN_ISI=round(0.05*LV_Fs);
if numel(onsets)>1; keep=[true;diff(onsets)>=MIN_ISI]; onsets=onsets(keep); end
fprintf('    Sniff phase: %d onsets at threshold=%.2f std\n',numel(onsets),threshold_std);
MAX_DUR=round(0.5*LV_Fs); SNF_PH=-ones(nSNF,1); dur_s=zeros(numel(onsets),1);
for k=1:numel(onsets)
    ok=onsets(k); cyc=MAX_DUR;
    if k<numel(onsets); cyc=min(onsets(k+1)-ok,MAX_DUR); end
    dur_s(k)=cyc/LV_Fs;
    idx=ok:min(ok+cyc-1,nSNF); SNF_PH(idx)=(0:numel(idx)-1)/cyc;
end
D.SNF_PH=SNF_PH; D.sniff_onsets=onsets;
D.sniff_onsets_s=(onsets-1)/LV_Fs; D.sniff_dur_s=median(dur_s); D.sniff_thr=threshold_std;
end

function thr=pick_threshold(SNF,LV_Fs)
SNF_z=(SNF-mean(SNF))/std(SNF); nSh=min(numel(SNF_z),round(10*LV_Fs)); def=-0.5;
fig=figure('Name','Set Sniff Threshold','Color','w','Position',[80 180 920 380]);
ax=axes(fig,'Position',[0.07 0.22 0.88 0.70]);
plot(ax,(0:nSh-1)/LV_Fs,SNF_z(1:nSh),'b','LineWidth',1); hold(ax,'on');
hl=yline(ax,def,'r--','LineWidth',2,'Label',sprintf(' thr=%.2f',def));
xlabel(ax,'Time (s)'); ylabel(ax,'SNF (z-score)');
title(ax,'Sniff threshold selector — green ticks = detected onsets'); grid(ax,'on');
uicontrol(fig,'Style','text','String','Threshold (z-score):','Units','normalized',...
    'Position',[0.24 0.04 0.15 0.07],'FontSize',10,'HorizontalAlignment','right');
uic=uicontrol(fig,'Style','edit','String',num2str(def),'Units','normalized',...
    'Position',[0.40 0.04 0.09 0.08],'FontSize',11);
uicontrol(fig,'Style','pushbutton','String','Preview','Units','normalized',...
    'Position',[0.50 0.04 0.10 0.08],'FontSize',10,...
    'Callback',@(~,~)preview_thr(ax,hl,str2double(get(uic,'String')),SNF_z,nSh,LV_Fs));
uicontrol(fig,'Style','pushbutton','String','Confirm','Units','normalized',...
    'Position',[0.61 0.04 0.12 0.08],'FontSize',10,...
    'BackgroundColor',[0.15 0.55 0.15],'ForegroundColor','w',...
    'Callback',@(~,~)uiresume(fig));
preview_thr(ax,hl,def,SNF_z,nSh,LV_Fs); uiwait(fig);
if isvalid(fig); thr=str2double(get(uic,'String')); if isnan(thr); thr=def; end; close(fig); else; thr=def; end
end

function preview_thr(ax,hl,thr,SNF_z,nSh,LV_Fs)
if isnan(thr); return; end
delete(findobj(ax,'Tag','stk')); hl.Value=thr; hl.Label=sprintf(' thr=%.2f',thr);
bl=SNF_z(1:nSh)<thr; dc=diff([0;bl]); ons=find(dc==1);
if numel(ons)>1; keep=[true;diff(ons)>=round(0.05*LV_Fs)]; ons=ons(keep); end
for k=1:min(numel(ons),200); xline(ax,(ons(k)-1)/LV_Fs,'g','LineWidth',0.8,'Tag','stk'); end
drawnow;
end

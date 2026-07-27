function D = threshold_eth(D, eth_threshold)
%% THRESHOLD_ETH  Apply amplitude threshold to ETH; result in D.ETH_thr
%  Values below eth_threshold are clipped to the threshold value itself
%  (floor clipping), so the signal never goes below the baseline level.
%  Default threshold = 0 (no clipping).
if nargin<2||isempty(eth_threshold); eth_threshold=pick_eth_threshold(D.ETH); end
D.ETH_thr=D.ETH; D.ETH_thr(D.ETH_thr<eth_threshold)=eth_threshold; D.eth_threshold=eth_threshold;
fprintf('    ETH threshold=%.4f  (%.1f%% samples clipped to floor)\n',...
    eth_threshold,100*mean(D.ETH<eth_threshold));
end

function thr=pick_eth_threshold(ETH)
def=0;
fig=figure('Name','Set ETH Threshold','Color','w','Position',[100 200 600 300]);
ax=axes(fig,'Position',[0.09 0.22 0.87 0.70]);
histogram(ax,ETH,80,'FaceColor',[0.3 0.5 0.8],'EdgeColor','none'); hold(ax,'on');
hl=xline(ax,def,'r--','LineWidth',2,'Label',sprintf(' thr=%.4f',def));
xlabel(ax,'ETH (normalized)'); ylabel(ax,'Count'); grid(ax,'on');
title(ax,'ETH distribution — values below threshold clipped to threshold floor');
uicontrol(fig,'Style','text','String','Threshold:','Units','normalized',...
    'Position',[0.28 0.04 0.12 0.07],'FontSize',10,'HorizontalAlignment','right');
uic=uicontrol(fig,'Style','edit','String',num2str(def),'Units','normalized',...
    'Position',[0.41 0.04 0.12 0.08],'FontSize',11,...
    'Callback',@(src,~)update_xline(hl,str2double(src.String)));
uicontrol(fig,'Style','pushbutton','String','Confirm','Units','normalized',...
    'Position',[0.55 0.04 0.12 0.08],'FontSize',10,...
    'BackgroundColor',[0.15 0.55 0.15],'ForegroundColor','w','Callback',@(~,~)uiresume(fig));
uiwait(fig);
if isvalid(fig); thr=str2double(get(uic,'String')); if isnan(thr); thr=0; end; close(fig); else; thr=0; end
end
function update_xline(hl,v); if ~isnan(v); hl.Value=v; hl.Label=sprintf(' thr=%.4f',v); drawnow; end; end

function plot_sniff_probe_map(D)
%% PLOT_SNIFF_PROBE_MAP  Probe map colored by mean sniff phase per unit
%  Color of each unit dot = D.unitMeanSniffPhase (0=fires at sniff onset,
%  1=fires at end of sniff cycle).  Units with no valid phase = gray.
%
%  Requires: D.unitIDs, D.unitMeanSniffPhase, D.sp (full loadKSdir struct)

if ~isfield(D,'unitMeanSniffPhase')
    error('OR:noPhase','Run compute_spike_phase(D) first.'); end

sp     = D.sp;
nUnits = numel(D.unitIDs);
nT     = size(sp.temps, 1);    % number of templates

%% ---- Step 1: per-template peak channel coords (unwhitened) ----------------
% Unwhiten: [nT x nTime x nChan]
tempsUnW = zeros(size(sp.temps));
for t = 1:nT
    tempsUnW(t,:,:) = squeeze(sp.temps(t,:,:)) * sp.winv;
end
% Peak-to-peak amplitude per channel per template: [nT x nChan]
tempChanAmps = squeeze(max(tempsUnW,[],2)) - squeeze(min(tempsUnW,[],2));
[~, peakChan] = max(tempChanAmps, [], 2);  % [nT x 1], 1-indexed into xcoords/ycoords

templateXc = sp.xcoords(peakChan);   % [nT x 1]
templateYc = sp.ycoords(peakChan);   % [nT x 1]

%% ---- Step 2: map each cluster ID to its dominant template -----------------
% After Phy manual curation cluster IDs != template indices.
% Use sp.spikeTemplates (zero-indexed KS template) to find the mode template
% for each cluster, then look up that template's peak channel.
hasSpikeTemplates = isfield(sp,'spikeTemplates') && ~isempty(sp.spikeTemplates);

unitXc = nan(nUnits, 1);
unitYc = nan(nUnits, 1);
for u = 1:nUnits
    uid  = D.unitIDs(u);
    mask = (sp.clu == uid);
    if ~any(mask); continue; end

    if hasSpikeTemplates
        mainTmp = mode(double(sp.spikeTemplates(mask))) + 1; % +1: 0->1-indexed
    else
        mainTmp = uid + 1;   % fallback: assume no Phy curation
    end

    if mainTmp >= 1 && mainTmp <= nT
        unitXc(u) = templateXc(mainTmp);
        unitYc(u) = templateYc(mainTmp);
    end
end

%% ---- Step 3: detect shanks from channel x-coordinate gaps -----------------
% Works for 1-shank (NP1.0) and multi-shank (NP2.0 / H-probe) recordings.
xAll  = sort(unique(sp.xcoords));
gaps  = diff(xAll);
% A gap >100 µm between adjacent channel x-positions marks a shank boundary
shankBreaks  = find(gaps > 100);
shankEdges   = [-Inf; xAll(shankBreaks) + gaps(shankBreaks)/2; Inf];
nShanks      = numel(shankEdges) - 1;

% Which shanks actually contain units?
shankOfUnit  = nan(nUnits,1);
for u = 1:nUnits
    for s = 1:nShanks
        if unitXc(u) >= shankEdges(s) && unitXc(u) < shankEdges(s+1)
            shankOfUnit(u) = s;
            break;
        end
    end
end
shanksPresent = unique(shankOfUnit(~isnan(shankOfUnit)));
if isempty(shanksPresent); shanksPresent = 1; end
nSP = numel(shanksPresent);

%% ---- Step 4: plot ----------------------------------------------------------
% Custom colormap: blue=early (phase 0) -> red=late (phase 1)
phi = linspace(0,1,256)';
cm  = [phi, zeros(256,1), 1-phi];

depthRange = [min(sp.ycoords)-50, max(sp.ycoords)+50];

fig = figure('Name','Unit Locations — Sniff Phase','Color','w', ...
             'Position',[150 50 250*nSP+220 680]);

for si = 1:nSP
    sh  = shanksPresent(si);
    xlo = shankEdges(sh);
    xhi = shankEdges(sh+1);
    ax  = subplot(1, nSP+1, si);
    hold(ax,'on'); set(ax,'Color',[0.15 0.15 0.15]);  % dark background

    % background: all electrodes on this shank
    onSh = sp.xcoords >= xlo & sp.xcoords < xhi;
    scatter(ax, zeros(sum(onSh),1), sp.ycoords(onSh), 6, [0.5 0.5 0.5], 'filled');

    % units on this shank
    onU = (shankOfUnit == sh);
    ph  = D.unitMeanSniffPhase(onU);
    yy  = unitYc(onU);
    for u2 = 1:sum(onU)
        if isnan(ph(u2))
            scatter(ax,0,yy(u2),55,[0.6 0.6 0.6],'filled', ...
                'MarkerEdgeColor','w','LineWidth',0.4);
        else
            cidx = max(1, round(ph(u2)*255) + 1);
            scatter(ax,0,yy(u2),75,cm(cidx,:),'filled', ...
                'MarkerEdgeColor','k','LineWidth',0.4);
        end
    end

    ylabel(ax,'Depth (µm)');
    set(ax,'XTick',[],'Box','off','YDir','normal','FontSize',10,'Color',[0.15 0.15 0.15]);
    ylim(ax, depthRange); xlim(ax, [-0.6 0.6]);
    if nSP > 1; title(ax, sprintf('Shank %d',sh),'FontSize',10,'Color','w'); end
end

% Colorbar panel
cbAx = subplot(1, nSP+1, nSP+1);
axis(cbAx,'off');
colormap(cbAx, cm);
cb = colorbar(cbAx,'west','AxisLocation','in');
set(cb,'Limits',[0 1],'Ticks',[0 0.25 0.5 0.75 1], ...
    'TickLabels',{'0 (onset)','0.25','0.5','0.75','1 (next)'},'FontSize',9);
cb.Label.String = 'Mean sniff phase';

sgtitle(fig, sprintf('Unit Sniff Phase  |  %d units', nUnits), ...
    'FontSize',12,'FontWeight','bold');
end

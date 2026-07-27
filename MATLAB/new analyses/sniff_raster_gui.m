function sniff_raster_gui(D)
%% SNIFF_RASTER_GUI  evRastersGUI-style interactive viewer for sniff-locked activity.
%
%  Three alignment panels shown simultaneously for each unit:
%    Panel 1 — All sniffs, sorted chronologically
%              Shows temporal drift / modulation across the session.
%    Panel 2 — All sniffs, sorted by cycle duration (short → long)
%              Reveals whether firing rate tracks breathing speed.
%    Panel 3 — All sniffs, sorted by ETH level at onset (low → high)
%              PSTH traces split: blue = no ETH, orange = ETH on.
%              This is the core odor-vs-baseline comparison.
%
%  Left margin: probe depth scatter (click to jump to nearest unit).
%  Bottom-left: ACG (linear and log scale), updates with each unit.
%  ETH-on / ETH-off transitions shown as coloured markers on every raster.
%
%  Keyboard controls:
%    up/down arrow    — next / previous unit (wraps)
%    left/right arrow — narrow / widen PSTH smoothing kernel
%    c                — jump to cluster by entering its ID
%
%  Requires (spikes-master toolbox on path):
%    psthAndBA, timestampsToBinned, rasterize, myACG,
%    WithinRanges, myGaussWin

if ~isfield(D,'sniff_onsets_s')
    error('OR:noOnsets','Run compute_sniff_phase(D) first.');
end

%% --- Parameters ----------------------------------------------------------
pars.psthBinSize  = 0.001;          % 1 ms bins
pars.smoothWinStd = 0.025;          % 25 ms Gaussian std (causal half-Gaussian)
pars.smoothWin    = createSmWin(pars.smoothWinStd, pars.psthBinSize);
pars.window       = [-0.1, D.sniff_dur_s * 1.5];   % 100 ms pre, 1.5 cycles post
pars.cluIndex     = 1;
pars.tickSize     = max(1, floor(numel(D.sniff_onsets_s)/300));

%% --- Shared data ---------------------------------------------------------
lv_t   = (0:numel(D.ETH)-1)' / D.LV_Fs;
eth_thr = 0;
if isfield(D,'eth_threshold'); eth_thr = D.eth_threshold; end

myData.st         = D.spikeTimes;
myData.clu        = D.sp.clu;
myData.clusterIDs = D.unitIDs(:);
myData.D          = D;
myData.lv_t       = lv_t;
myData.eth_thr    = eth_thr;
myData.pars       = pars;

ev             = createEv(myData);
myData.ev      = ev;
evData         = createEvTimesOrders(myData);
myData.evData  = evData;

%% --- Build figure --------------------------------------------------------
f = figure('Color','w','Name','Sniff Raster GUI', ...
    'Position',[50 50 1400 780]);
myData.f = f;
myData   = createPlots(myData);

updatePlots(myData);
set(f,'UserData',myData);
set(f,'KeyPressFcn', @(f,k)kpCallback(f,k));
end   % ← main function end


%% =========================================================================
%% KEYBOARD CALLBACK
%% =========================================================================
function kpCallback(f, keydata)
myData = get(f,'UserData');
switch keydata.Key
    case 'uparrow'
        myData.pars.cluIndex = myData.pars.cluIndex + 1;
        if myData.pars.cluIndex > numel(myData.clusterIDs)
            myData.pars.cluIndex = 1;
        end
    case 'downarrow'
        myData.pars.cluIndex = myData.pars.cluIndex - 1;
        if myData.pars.cluIndex < 1
            myData.pars.cluIndex = numel(myData.clusterIDs);
        end
    case 'rightarrow'
        myData.pars.smoothWinStd = myData.pars.smoothWinStd * 5/4;
        myData.pars.smoothWin    = createSmWin(myData.pars.smoothWinStd, myData.pars.psthBinSize);
    case 'leftarrow'
        myData.pars.smoothWinStd = myData.pars.smoothWinStd * 4/5;
        myData.pars.smoothWin    = createSmWin(myData.pars.smoothWinStd, myData.pars.psthBinSize);
    case 'c'
        newC = inputdlg('Cluster ID?');
        if ~isempty(newC)
            ind = find(myData.clusterIDs == str2double(newC{1}), 1);
            if ~isempty(ind); myData.pars.cluIndex = ind; end
        end
end
updatePlots(myData);
set(f,'Name', sprintf('Sniff Raster GUI  —  cluster %d  (%d of %d)', ...
    myData.clusterIDs(myData.pars.cluIndex), ...
    myData.pars.cluIndex, numel(myData.clusterIDs)));
set(f,'UserData',myData);


%% =========================================================================
%% PROBE CLICK CALLBACK
%% =========================================================================
function probeClick(keydata, f)
myData = get(f,'UserData');
clickY = keydata.IntersectionPoint(2);
[~, ind] = min(abs(myData.probeYPos - clickY));
myData.pars.cluIndex = ind;
updatePlots(myData);
set(f,'Name', sprintf('Sniff Raster GUI  —  cluster %d  (%d of %d)', ...
    myData.clusterIDs(ind), ind, numel(myData.clusterIDs)));
set(f,'UserData',myData);


%% =========================================================================
%% CREATE PLOTS  (called once at startup)
%% =========================================================================
function myData = createPlots(myData)
nP    = numel(myData.evData);   % 3 alignment panels
nRows = 3;    % rows: 1-2 raster (spans 2), 3 PSTH
f     = myData.f;

% --- Create subplot axes for rasters and PSTHs ---------------------------
for col = 1:nP
    axRaster(col) = subplot(nRows, nP, [col, nP+col]);
    axPSTH(col)   = subplot(nRows, nP,  2*nP + col);
end

% Shift all subplots rightward to leave room for probe scatter + ACG
shift = 0.08;
for n = 1:nP
    pos = get(axRaster(n),'Position');
    set(axRaster(n),'Position',[pos(1)+shift pos(2) pos(3)-shift/nP pos(4)]);
    pos = get(axPSTH(n),'Position');
    set(axPSTH(n),  'Position',[pos(1)+shift pos(2) pos(3)-shift/nP pos(4)]);
end

% --- Draw initial raster/PSTH frames -------------------------------------
evData = myData.evData;
for e = 1:nP
    hRaster(e) = plot(axRaster(e), 0, 0, 'k');
    hold(axRaster(e),'on');

    % ETH transition markers (static — don't update per unit)
    hRasterEvs{e} = addEvents(axRaster(e), evData(e).times, ...
        evData(e).trOrders, myData.ev, evData(e).windows);

    plot(axRaster(e), [0 0], [0 numel(evData(e).times)], 'k:');
    ylim(axRaster(e), [0 numel(evData(e).times)] + 0.5);
    xlim(axRaster(e), evData(e).windows);
    box(axRaster(e),'off');
    ylabel(axRaster(e), evData(e).ylab, 'FontSize',8, 'Interpreter','none');
    set(axRaster(e),'XTickLabel',{});
    title(axRaster(e), evData(e).panelTitle, 'FontSize',9, 'Interpreter','none');
end

for e = 1:nP
    hold(axPSTH(e),'on');
    for c = 1:numel(evData(e).gIDs)
        hPSTH{e}(c) = plot(axPSTH(e), evData(e).windows, [0 0], ...
            'Color', evData(e).colors(c,:), 'LineWidth', 2.0);
    end
    xlim(axPSTH(e), evData(e).windows);
    box(axPSTH(e),'off');
    plot(axPSTH(e), [0 0], [0 1000], 'k:');
    xlabel(axPSTH(e), 'time from sniff onset (s)', 'FontSize',8);
    if e==1; ylabel(axPSTH(e), 'firing rate (Hz)', 'FontSize',8); end
end

myData.hRaster    = hRaster;
myData.hRasterEvs = hRasterEvs;
myData.hPSTH      = hPSTH;
myData.axPSTH     = axPSTH;
myData.axRaster   = axRaster;

% --- Probe scatter (left margin) -----------------------------------------
D     = myData.D;
ypos  = D.unitDepths(:);
ypos(isnan(ypos)) = nanmean(ypos);   % fill NaN depths so scatter + click work
axPrb = axes(f);
set(axPrb,'Position',[0.01 0.12 0.03 0.82]);

hScatter = scatter(axPrb, ones(size(ypos)), ypos, 10, [0.5 0.5 0.5],'filled');
set(hScatter,'HitTest','off');
hold(axPrb,'on');
hDot = scatter(axPrb, 1, ypos(1), 50, [0.9 0.1 0.1],'filled');
set(hDot,'HitTest','off');

set(axPrb,'XTick',[],'YDir','reverse','XLim',[0.5 1.5]);
set(axPrb,'YAxisLocation','left');
ylabel(axPrb,'depth (µm)','FontSize',8);
box(axPrb,'off');
axPrb.XAxis.Visible = 'off';
set(axPrb,'ButtonDownFcn',@(~,k)probeClick(k,f));

myData.axProbe   = axPrb;
myData.probeYPos = ypos;
myData.hProbeDot = hDot;

% --- ACG panels (below probe) --------------------------------------------
axACGlin = axes(f);
set(axACGlin,'Position',[0.005 0.065 0.065 0.09]);
axACGlog = axes(f);
set(axACGlog,'Position',[0.005 0.005 0.065 0.055]);
myACG(rand(1,100), axACGlin, axACGlog);   % initialise plot handles
title(axACGlin,'ACG','FontSize',7);
myData.axACGlin = axACGlin;
myData.axACGlog = axACGlog;


%% =========================================================================
%% UPDATE PLOTS  (called on every unit change / smoothing change)
%% =========================================================================
function updatePlots(myData)
uid = myData.clusterIDs(myData.pars.cluIndex);
st  = myData.st(myData.clu == uid);

[allBA, allBins] = computeBAs(st, myData);
evData = myData.evData;

maxyl = [Inf -Inf];
for e = 1:numel(evData)
    ba   = allBA{e}(evData(e).trOrders, :);
    bins = allBins{e};

    % Raster ticks
    [tr, b] = find(ba);
    if ~isempty(b)
        [rasterX, yy] = rasterize(bins(b));
        rasterY = yy * myData.pars.tickSize + ...
            reshape(repmat(tr',3,1), 1, numel(tr)*3) - 0.5;
    else
        rasterX = []; rasterY = [];
    end
    set(myData.hRaster(e), 'XData', rasterX, 'YData', rasterY);

    % PSTH traces (one per group)
    gIDs = evData(e).gIDs;
    for g = 1:numel(gIDs)
        incl = evData(e).groups == gIDs(g);
        if any(incl)
            pSmooth = conv(mean(ba(incl,:), 1) / myData.pars.psthBinSize, ...
                myData.pars.smoothWin, 'same');
            set(myData.hPSTH{e}(g), 'XData', bins, 'YData', pSmooth);
            maxyl(2) = max(maxyl(2), max(pSmooth));
            maxyl(1) = min(maxyl(1), min(pSmooth));
        end
    end
end

% Shared y-axis across all PSTH panels
if isfinite(maxyl(2)) && maxyl(2) > maxyl(1)
    for e = 1:numel(evData)
        ylim(myData.axPSTH(e), [max(0, maxyl(1)), maxyl(2)*1.05]);
    end
end

% Probe highlight
set(myData.hProbeDot,'YData', myData.probeYPos(myData.pars.cluIndex));

% ACG update (copy exactly from evRastersGUI)
[xLin, nLin, ~, nLog] = myACG(st, [], []);
ch = get(myData.axACGlin,'Children');
nLinSh = nLin(xLin < 0.1);
yyLin  = reshape([nLinSh nLinSh]', numel(nLinSh)*2, 1);
if numel(ch) >= 3
    set(ch(3),'YData', yyLin(1:end-1));
    set(ch(1),'YData', nLog(end)*[1 1]);
    set(ch(2),'YData', [0 max(yyLin)]);
    if max(yyLin) > 0; set(myData.axACGlin,'YLim',[0 max(yyLin)]); end
end
ch = get(myData.axACGlog,'Children');
yyLog = reshape([nLog nLog]', numel(nLog)*2, 1);
if numel(ch) >= 3
    set(ch(3),'YData', yyLog);
    set(ch(1),'YData', nLog(end)*[1 1]);
    set(ch(2),'YData', [0 max(yyLog)]);
    if max(yyLog) > 0; set(myData.axACGlog,'YLim',[0 max(yyLog)]); end
end


%% =========================================================================
%% CREATE EVENT TIMES + ORDERS  (sniff-specific)
%% =========================================================================
function evData = createEvTimesOrders(myData)
D      = myData.D;
lv_t   = myData.lv_t;
onset  = D.sniff_onsets_s(:);
n      = numel(onset);
win    = myData.pars.window;

% ETH level (binary: above threshold = 1) at each sniff onset
eth_at_onset = interp1(lv_t, double(D.ETH > myData.eth_thr), onset, 'nearest', 0);

% Cycle durations (last sniff gets median)
dur = [diff(onset); D.sniff_dur_s];

%% Panel 1 — all sniffs, chronological
evData(1).panelTitle = 'All sniffs — chronological';
evData(1).times      = onset;
evData(1).trOrders   = (1:n)';
evData(1).windows    = win;
evData(1).groups     = ones(n,1);
evData(1).gIDs       = 1;
evData(1).colors     = [0 0 0];
evData(1).alignName  = 'sniff onset';
evData(1).ylab       = 'sniff # (time order)';

%% Panel 2 — all sniffs, sorted by cycle duration (short→long)
[~, sortDur]         = sort(dur, 'ascend');
evData(2).panelTitle = 'All sniffs — by duration (short→long)';
evData(2).times      = onset;
evData(2).trOrders   = sortDur;
evData(2).windows    = win;
evData(2).groups     = ones(n,1);
evData(2).gIDs       = 1;
evData(2).colors     = [0 0 0];
evData(2).alignName  = 'sniff onset';
evData(2).ylab       = 'sniff # (sorted by duration)';

%% Panel 3 — all sniffs, sorted by ETH level; PSTH split ETH on/off
[~, sortETH]         = sort(eth_at_onset, 'ascend');
eth_group            = eth_at_onset + 1;   % 1=off (blue), 2=on (orange)
evData(3).panelTitle = 'All sniffs — by ETH level (no-ETH / ETH-on)';
evData(3).times      = onset;
evData(3).trOrders   = sortETH;
evData(3).windows    = win;
evData(3).groups     = eth_group;
evData(3).gIDs       = [1, 2];
evData(3).colors     = [0.25 0.45 0.85;   % blue  = no ETH
                         0.85 0.35 0.10];  % orange = ETH on
evData(3).alignName  = 'sniff onset';
evData(3).ylab       = 'sniff # (sorted low→high ETH)';


%% =========================================================================
%% CREATE EV  (event overlay markers shown on all raster panels)
%% =========================================================================
function ev = createEv(myData)
D      = myData.D;
lv_t   = myData.lv_t;
eth_b  = D.ETH > myData.eth_thr;

rise = find(diff([0;    eth_b(:)]) ==  1);
fall = find(diff([eth_b(:); 0])    == -1);

ev = struct('name',{},'icon',{},'color',{},'times',{});

if ~isempty(rise)
    ev(end+1).name  = 'ETH on';
    ev(end).icon    = 'v';
    ev(end).color   = [0.85 0.35 0.10];
    ev(end).times   = lv_t(rise);
end
if ~isempty(fall)
    ev(end+1).name  = 'ETH off';
    ev(end).icon    = '^';
    ev(end).color   = [0.25 0.45 0.85];
    ev(end).times   = lv_t(fall);
end


%% =========================================================================
%% COMPUTE BAs  (generic — adapted from evRastersGUI)
%% =========================================================================
function [ba, bins] = computeBAs(st, myData)
evData = myData.evData;
ba   = cell(numel(evData),1);
bins = cell(numel(evData),1);

for e = 1:numel(evData)
    evTimes = evData(e).times;

    % Reuse if same event times as an earlier panel (all 3 use identical onsets)
    found = false;
    for e2 = 1:e-1
        if numel(evTimes)==numel(evData(e2).times) && all(evTimes==evData(e2).times)
            ba{e}   = ba{e2};
            bins{e} = bins{e2};
            found   = true;
            break;
        end
    end

    if ~found
        winExp = evData(e).windows + myData.pars.smoothWinStd * 5 * [-1 1];
        [ba{e}, bins{e}] = timestampsToBinned(st, evTimes, myData.pars.psthBinSize, winExp);
    end
end


%% =========================================================================
%% ADD EVENTS  (overlay markers on raster axes — adapted from evRastersGUI)
%% =========================================================================
function h = addEvents(ax, eventTimes, trOrder, ev, thisWindow)
hold(ax,'on');
reOrd  = eventTimes(trOrder);
nTimes = numel(reOrd);
h      = gobjects(0);

for e = 1:numel(ev)
    other = ev(e).times(:);
    if isempty(other); continue; end
    x = WithinRanges(other, bsxfun(@plus, reOrd(:), thisWindow), (1:nTimes)');
    [ii, trInds] = find(x);
    if isempty(ii); continue; end
    relTimes = other(ii) - reOrd(trInds);
    q = plot(ax, relTimes, trInds, ev(e).icon, ...
        'Color', ev(e).color, 'MarkerSize', 5, 'LineStyle','none');
    h(end+1) = q;  %#ok<AGROW>
end


%% =========================================================================
%% SMOOTHING WINDOW  (causal half-Gaussian, copied from evRastersGUI)
%% =========================================================================
function smWin = createSmWin(stdev, psthBinSize)
smWin = myGaussWin(stdev, 1/psthBinSize);
smWin(1:round(numel(smWin)/2)) = 0;
smWin = smWin ./ sum(smWin);

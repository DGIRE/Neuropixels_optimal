function sniff_pop_raster(sp, eventData, traces, auxVid, pars)
%% SNIFF_POP_RASTER  Population raster viewer — drop-in replacement for
%  popRasterViewer that requires NO GUI Layout Toolbox (no uix.*).
%
%  Identical behaviour and interface to popRasterViewer; only the control
%  panel is re-implemented using standard MATLAB uipanel/uicontrol.
%
%  Inputs: same as popRasterViewer.
%    sp         — struct: st, clu, cids, yAxOrderings, colorings
%    eventData  — struct array: name, times, spec (cell of Name/Value pairs)
%    traces     — struct array: t, v, name, color
%    auxVid     — pass [] (movie panels not supported without uix)
%    pars       — struct; optional field: startTime
%
%  Controls:
%    left/right arrow  — scroll backward / forward in time
%    up/down arrow     — shrink / expand time window
%    y                 — cycle y-axis ordering
%    c                 — cycle raster colorization
%    -  /  =           — shorter / taller raster ticks
%    t                 — jump to a specific time
%    m                 — toggle MUA visibility
%    click on raster   — draw a vertical reference line
%    Event buttons     — click to toggle visibility
%    Trace buttons     — click to select; click again to hide

for e = 1:length(eventData)
    eventData(e).visible = true;
end
for t = 1:length(traces)
    traces(t).visible = true;
    traces(t).scale  = 0.2 / (max(traces(t).v) - min(traces(t).v) + eps);
    traces(t).offset = (t-1)/length(traces) + 1/(2*length(traces));
end
ud.eventData = eventData;
ud.traces    = traces;
ud.auxVid    = auxVid;

if isfield(pars,'startTime'); startTime = pars.startTime; else; startTime = 10; end

params.windowSize  = 5;
params.currT       = startTime;
params.rasterScale = 1;
params.window      = params.windowSize*0.5*[-1 1] + params.currT;
params.binSize     = 0.002;
params.posLims     = [0 numel(sp.cids)];
params.yAxInd      = 1;
params.colorInd    = 1;
params.selectedTrace = 1;
params.muaHidden   = false;
params.isPlaying   = false;   % no movie playback by default
params.lastRealTime = now;

% Rescale y-orderings (copied verbatim from popRasterViewer)
for y = 1:numel(sp.yAxOrderings)
    yp  = sp.yAxOrderings(y).yPos;
    ylv = sp.yAxOrderings(y).yLabelVal;
    mn = min(yp); mx = max(yp);
    sp.yAxOrderings(y).yPos      = (yp-mn)/mx*diff(params.posLims)+params.posLims(1);
    sp.yAxOrderings(y).yLabelVal = (ylv-mn)/mx*diff(params.posLims)+params.posLims(1);
end

ud.sp     = sp;
ud.params = params;

f    = figure('Color','k','Name','Sniff Pop Raster');
ud.f = f;

hands = createPlots(ud);
ud.hands = hands;
updatePlots(ud);
recolor(ud);

set(f,'UserData', ud);
set(f,'KeyPressFcn', @(f,k)keyboardPressCallback(f,k));

% Timer only needed for movie playback — skip if no auxVid
if ~isempty(auxVid)
    myTimer = timer('Period', round(1/30*1000)/1000, ...
        'ExecutionMode','fixedRate', ...
        'TimerFcn', @(h,~)updateMovies(f));
    set(f,'CloseRequestFcn', @(s,~)closeFigure(s, myTimer));
    start(myTimer);
else
    set(f,'CloseRequestFcn', @(s,~)delete(s));
end


%% =========================================================================
function h = createPlots(ud)
%% Create axes and control panel  (no GUI Layout Toolbox required)
sp = ud.sp;
ed = ud.eventData;
tr = ud.traces;
p  = ud.params;

h = [];
h.rasterAx = axes('Position',[0.03 0.03 0.70 0.96]);
set(h.rasterAx,'Color','k');
set(h.rasterAx,'ButtonDownFcn',@(~,k)rastClick(k,ud.f));
hold(h.rasterAx,'on');

for c = 1:length(sp.cids)
    thisH = plot(0,0,'w-','LineWidth',1.5);
    set(thisH,'HitTest','off');
    h.rasterHands(c) = thisH;
end
box(h.rasterAx,'off');

for e = 1:length(ed)
    thisH = plot(0,0,'-',ed(e).spec{:});
    set(thisH,'HitTest','off');
    h.eventHands(e) = thisH;
end

for t = 1:length(tr)
    thisH = plot(0,0,'-','Color',tr(t).color);
    set(thisH,'HitTest','off');
    h.traceHands(t) = thisH;
end

h.userLine = plot([0 0],p.posLims,'w');

%% ── Control panel (uipanel + uicontrol — no uix required) ────────────────
BG = 'k'; FG = 'w';

h.ctrlPanel = uipanel(ud.f,'Title','Controls', ...
    'Units','normalized','Position',[0.75 0.55 0.24 0.44], ...
    'BackgroundColor',BG,'ForegroundColor',FG,'FontWeight','bold');

% Status line: colorizing / y-axis labels
h.ColorDisplay = uicontrol('Style','text','Parent',h.ctrlPanel, ...
    'Units','normalized','Position',[0 0.88 0.5 0.10], ...
    'HorizontalAlignment','left','String','', ...
    'BackgroundColor',BG,'ForegroundColor',FG,'FontSize',8);
h.yPosDisplay  = uicontrol('Style','text','Parent',h.ctrlPanel, ...
    'Units','normalized','Position',[0.5 0.88 0.5 0.10], ...
    'HorizontalAlignment','left','String','', ...
    'BackgroundColor',BG,'ForegroundColor',FG,'FontSize',8);

% Events sub-panel (left half)
nEv   = length(ed);
evPan = uipanel(h.ctrlPanel,'Title','Events', ...
    'Units','normalized','Position',[0 0 0.5 0.87], ...
    'BackgroundColor',BG,'ForegroundColor',FG);

for e = 1:nEv
    ii      = find(strcmp(ed(e).spec,'Color'));
    evColor = ed(e).spec{ii+1};
    btnPos  = [0, 1-e/nEv, 1, 1/nEv];
    thisH   = uicontrol('Style','pushbutton','Parent',evPan, ...
        'Units','normalized','Position',btnPos, ...
        'String',ed(e).name,'HorizontalAlignment','left', ...
        'BackgroundColor',BG,'ForegroundColor',evColor,'FontWeight','bold', ...
        'Callback',@(~,~)toggleEvent(ud.f,e), ...
        'KeyPressFcn',@(a,k)keyboardPressCallback(ud.f,k));
    h.evLeg(e) = thisH;
end

% Traces sub-panel (right half)
nTr   = length(tr);
trPan = uipanel(h.ctrlPanel,'Title','Traces', ...
    'Units','normalized','Position',[0.5 0 0.5 0.87], ...
    'BackgroundColor',BG,'ForegroundColor',FG);

for t = 1:nTr
    btnPos = [0, 1-t/nTr, 1, 1/nTr];
    thisH  = uicontrol('Style','pushbutton','Parent',trPan, ...
        'Units','normalized','Position',btnPos, ...
        'String',tr(t).name,'HorizontalAlignment','left', ...
        'BackgroundColor',BG,'ForegroundColor',tr(t).color, ...
        'Callback',@(~,~)selectTrace(ud.f,t), ...
        'KeyPressFcn',@(a,k)keyboardPressCallback(ud.f,k));
    h.trLeg(t) = thisH;
end

% Movie panels (only if auxVid provided — unusual for sniff data)
if ~isempty(ud.auxVid)
    for v = 1:numel(ud.auxVid)
        ax = axes('Position',[0.75 0.05+(v-1)*0.25 0.23 0.25]);
        ud.auxVid(v).f(ax, p.currT, ud.auxVid(v).data);
        h.auxAxes(v) = ax;
    end
    h.movieTimeLine = plot(h.rasterAx,[0 0],p.posLims,'w:');
else
    h.movieTimeLine = [];
end


%% =========================================================================
%% The following functions are copied verbatim from popRasterViewer
%% =========================================================================

function updateMovies(f)
ud = get(f,'UserData'); p = ud.params;
if p.isPlaying
    newT = p.currT + (now-p.lastRealTime)*24*3600;
    if newT > p.window(2); newT = p.window(1); end
    p.lastRealTime = now; p.currT = newT;
    ud.params = p; set(f,'UserData',ud);
    if ~isempty(ud.auxVid)
        for v = 1:numel(ud.auxVid)
            ud.auxVid(v).f(ud.hands.auxAxes(v), newT, ud.auxVid(v).data);
        end
    end
    if ~isempty(ud.hands.movieTimeLine)
        set(ud.hands.movieTimeLine,'XData',newT*[1 1]);
    end
    drawnow;
end

function updatePlots(ud)
sp = ud.sp; p = ud.params; h = ud.hands;
ed = ud.eventData; tr = ud.traces;
set(h.rasterAx,'XLim',p.window,'YLim',p.posLims);

incl = sp.st>p.window(1) & sp.st<p.window(2);
st = sp.st(incl); clu = sp.clu(incl);
for c = 1:length(sp.cids)
    [rasterX,yy] = rasterize(st(clu==sp.cids(c)));
    rasterY = yy*p.rasterScale + sp.yAxOrderings(p.yAxInd).yPos(c);
    set(h.rasterHands(c),'XData',rasterX,'YData',rasterY);
end

for e = 1:length(ed)
    incl = ed(e).times>p.window(1) & ed(e).times<p.window(2);
    if sum(incl)>0
        [xx,yy] = rasterize(ed(e).times(incl));
        set(h.eventHands(e),'XData',xx,'YData',yy*p.posLims(2));
    else
        set(h.eventHands(e),'XData',[],'YData',[]);
    end
end

for t = 1:length(tr)
    incl = tr(t).t>p.window(1) & tr(t).t<p.window(2);
    set(h.traceHands(t),'XData',tr(t).t(incl), ...
        'YData',tr(t).v(incl)*tr(t).scale*diff(p.posLims) + tr(t).offset*diff(p.posLims));
end

set(h.yPosDisplay,'String',sprintf('ordered by %s',sp.yAxOrderings(p.yAxInd).name));
set(h.rasterAx,'YTickLabel',sp.yAxOrderings(p.yAxInd).yLabel, ...
    'YTick',sp.yAxOrderings(p.yAxInd).yLabelVal);

function recolor(ud)
sp = ud.sp; p = ud.params; h = ud.hands;
cmap = sp.colorings(p.colorInd).colors;
if p.muaHidden && isfield(sp,'cgs')
    set(h.ColorDisplay,'String',sprintf('colored by %s (mua off)',sp.colorings(p.colorInd).name));
    cmap(:,4) = 1;
    cmap(sp.cgs<2,4) = 0;
else
    set(h.ColorDisplay,'String',sprintf('colored by %s',sp.colorings(p.colorInd).name));
end
for c = 1:length(sp.cids)
    set(h.rasterHands(c),'Color',cmap(c,:));
end

function rastClick(keydata,f)
clickX = keydata.IntersectionPoint(1);
ud = get(f,'UserData');
set(ud.hands.userLine,'XData',clickX*[1 1]);

function selectTrace(f,t)
ud = get(f,'UserData'); p = ud.params; h = ud.hands;
if isequal(p.selectedTrace,t)
    set(h.traceHands(t),'Visible','off');
    ud.hands.trLeg(t).FontWeight = 'normal';
    p.selectedTrace = [];
else
    if ~isempty(p.selectedTrace) && p.selectedTrace<=length(h.trLeg)
        ud.hands.trLeg(p.selectedTrace).FontWeight = 'normal';
    end
    ud.hands.trLeg(t).FontWeight = 'bold';
    p.selectedTrace = t;
    set(h.traceHands(t),'Visible','on');
end
ud.params = p; set(f,'UserData',ud);

function toggleEvent(f,e)
ud = get(f,'UserData'); h = ud.hands;
if strcmp(get(h.eventHands(e),'Visible'),'on')
    set(h.eventHands(e),'Visible','off');
    ud.hands.evLeg(e).FontWeight = 'normal';
else
    set(h.eventHands(e),'Visible','on');
    ud.hands.evLeg(e).FontWeight = 'bold';
end
set(f,'UserData',ud);

function keyboardPressCallback(f,keydata)
ud = get(f,'UserData'); p = ud.params; tr = ud.traces;
switch keydata.Key
    case 'rightarrow'
        p.currT = p.currT + p.windowSize/4;
        if p.currT > ud.sp.st(end); p.currT = 0; end
        p = updateWindow(p);
    case 'leftarrow'
        p.currT = p.currT - p.windowSize/4;
        if p.currT < 0; p.currT = ud.sp.st(end); end
        p = updateWindow(p);
    case 'uparrow'
        p.windowSize = p.windowSize * 0.75;
        p = updateWindow(p);
    case 'downarrow'
        p.windowSize = p.windowSize / 0.75;
        p = updateWindow(p);
    case 'y'
        p.yAxInd = p.yAxInd + 1;
        if p.yAxInd > numel(ud.sp.yAxOrderings); p.yAxInd = 1; end
    case 'c'
        p.colorInd = p.colorInd + 1;
        if p.colorInd > numel(ud.sp.colorings); p.colorInd = 1; end
        isP = p.isPlaying; p.isPlaying = false;
        ud.params = p; set(f,'UserData',ud);
        recolor(ud);
        p.isPlaying = isP; ud.params = p; set(f,'UserData',ud);
    case 'hyphen'
        p.rasterScale = p.rasterScale / 1.5;
    case 'equal'
        p.rasterScale = p.rasterScale * 1.5;
    case 't'
        newT = inputdlg('Jump to time (s)?');
        newT = str2double(newT{1});
        if ~isnan(newT)
            p.currT = max(0, min(newT, ud.sp.st(end)));
        end
        p = updateWindow(p);
    case 'i'
        if ~isempty(p.selectedTrace); tr(p.selectedTrace).offset = tr(p.selectedTrace).offset + 1/25; end
    case 'k'
        if ~isempty(p.selectedTrace); tr(p.selectedTrace).offset = tr(p.selectedTrace).offset - 1/25; end
    case 'l'
        if ~isempty(p.selectedTrace); tr(p.selectedTrace).scale = tr(p.selectedTrace).scale * 5/4; end
    case 'j'
        if ~isempty(p.selectedTrace); tr(p.selectedTrace).scale = tr(p.selectedTrace).scale * 4/5; end
    case 'm'
        p.muaHidden = ~p.muaHidden;
        ud.params = p; recolor(ud);
end
ud.traces = tr;
isP = p.isPlaying; p.isPlaying = false;
ud.params = p; set(f,'UserData',ud);
updatePlots(ud);
p.isPlaying = isP; ud.params = p; set(f,'UserData',ud);

function p = updateWindow(p)
p.window = p.windowSize*0.5*[-1 1] + p.currT;

function closeFigure(f,myTimer)
stop(myTimer); delete(myTimer); delete(f);

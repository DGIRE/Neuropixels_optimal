function plot_unit_PSTH_heatmap(D, bin_ms_init)
%% PLOT_UNIT_PSTH_HEATMAP  Per-unit PSTH colormap vs sniff phase / actual time.
%  Each sniff onset is an event; spike counts are averaged across events.
%  Color = raw firing rate (spikes/s) — no per-unit normalization.
%  Color scale is set to the 99th percentile across all units to prevent
%  outliers from washing out the map.
%  Rows = units sorted shallow→deep (top→bottom).
%  Interactive: bin width (ms) and Normalized phase / Actual time (ms) toggle.

if ~isfield(D,'sniff_onsets_s'); error('OR:noOnsets','Run compute_sniff_phase(D) first.'); end
if nargin<2||isempty(bin_ms_init); bin_ms_init=20; end

fig = figure('Name','Unit PSTH Heatmap — Sniff Phase','Color','w','Position',[120 60 860 600]);
draw_heatmap(fig, D, bin_ms_init, false);
add_controls_hm(fig, D, bin_ms_init, false);
end

%% ── draw ──────────────────────────────────────────────────────────────────
function draw_heatmap(fig, D, bin_ms, use_ms)
delete(findobj(fig,'Type','axes'));
med_dur_ms = D.sniff_dur_s * 1000;
nUnits     = numel(D.unitIDs);

[psth, centers, n_ev] = compute_sniff_psth(D, bin_ms, use_ms);

% Sort units shallow→deep (shallowest = top row)
[~, sortIdx]  = sort(D.unitDepths, 'descend');
PSTH_sorted   = psth(sortIdx, :);    % raw spikes/s — no normalization

ax = axes(fig, 'Position', [0.10 0.16 0.82 0.74]);
imagesc(ax, centers, [1 nUnits], PSTH_sorted);
hold(ax,'on');
xline(ax, 0, 'r--', 'LineWidth', 3);
colormap(ax, hot(256));
cb = colorbar(ax); cb.Label.String = 'Firing rate (spikes/s)';

% Color scale: 0 to 99th percentile — keeps dynamic range visible
cmax = prctile(PSTH_sorted(:), 99);
if cmax > 0; clim(ax, [0, cmax]); end

if use_ms
    bin_actual = diff(centers(1:2));
    xlabel(ax, sprintf('Time from sniff onset (ms)  [median cycle \\approx %.0f ms]', med_dur_ms));
    ttl = sprintf('Per-unit PSTH  |  %d units  |  bin = %.0f ms  |  %d events  |  actual time', ...
                  nUnits, bin_actual, n_ev);
    xlim(ax, [-0.05*med_dur_ms, med_dur_ms]);
else
    bin_ph = diff(centers(1:2));
    xlabel(ax, sprintf('Normalized sniff phase  (1 \\approx %.0f ms)', med_dur_ms));
    ttl = sprintf('Per-unit PSTH  |  %d units  |  bin = %.0f ms = %.3f phase  |  %d events', ...
                  nUnits, bin_ms, bin_ph, n_ev);
    xlim(ax, [-0.05, 1]);
end

ylabel(ax, 'Unit (sorted shallow\rightarrowdeep)');
title(ax, ttl);
set(ax, 'Box','off', 'FontSize',10);

yticks_at = round(linspace(1, nUnits, min(10, nUnits)));
set(ax, 'YTick', yticks_at, ...
    'YTickLabel', arrayfun(@(i) num2str(D.unitIDs(sortIdx(i))), yticks_at, 'UniformOutput', false));
end

%% ── controls ──────────────────────────────────────────────────────────────
function add_controls_hm(fig, D, bin_init, use_ms_init)
uicontrol(fig,'Style','text','String','Bin (ms):','Units','normalized',...
    'Position',[0.18 0.01 0.09 0.05],'FontSize',10,...
    'HorizontalAlignment','right','BackgroundColor','w');
uic = uicontrol(fig,'Style','edit','String',num2str(bin_init),'Units','normalized',...
    'Position',[0.28 0.01 0.08 0.06],'FontSize',11);

rbg = uibuttongroup(fig,'Units','normalized','Position',[0.50 0.005 0.38 0.07],...
    'BorderType','none','BackgroundColor','w','Title','');
rb1 = uicontrol(rbg,'Style','radiobutton','String','Normalized phase','Units','normalized',...
    'Position',[0.01 0.05 0.48 0.90],'FontSize',10,'BackgroundColor','w');
rb2 = uicontrol(rbg,'Style','radiobutton','String','Actual time (ms)','Units','normalized',...
    'Position',[0.51 0.05 0.48 0.90],'FontSize',10,'BackgroundColor','w');
if use_ms_init
    rbg.SelectedObject = rb2;
else
    rbg.SelectedObject = rb1;
end

cb = @(~,~) do_redraw_hm(fig, D, uic, rbg);
rbg.SelectionChangedFcn = cb;
uicontrol(fig,'Style','pushbutton','String','Redraw','Units','normalized',...
    'Position',[0.37 0.01 0.09 0.06],'FontSize',10,'Callback',cb);
end

%% ── redraw callback ───────────────────────────────────────────────────────
function do_redraw_hm(fig, D, uic, rbg)
v = str2double(get(uic,'String'));
if isnan(v)||v<=0; warndlg('Enter a positive bin width (ms).','Invalid'); return; end
use_ms = strcmp(get(rbg.SelectedObject,'String'), 'Actual time (ms)');
delete(findobj(fig,'Type','uicontrol'));
delete(findobj(fig,'Type','uibuttongroup'));
draw_heatmap(fig, D, v, use_ms);
add_controls_hm(fig, D, v, use_ms);
end

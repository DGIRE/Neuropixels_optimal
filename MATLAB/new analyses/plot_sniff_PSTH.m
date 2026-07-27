function plot_sniff_PSTH(D, bin_ms_init)
%% PLOT_SNIFF_PSTH  Population PSTH relative to sniff onset.
%  All unit spike times are pooled and passed directly to psthAndBA — no
%  per-unit normalization.  Y-axis = total population firing rate (spikes/s),
%  averaged across sniff events.
%  Interactive: bin width (ms) edit box + Normalized phase / Actual time toggle.

if ~isfield(D,'sniff_onsets_s'); error('OR:noOnsets','Run compute_sniff_phase(D) first.'); end
if nargin<2 || isempty(bin_ms_init); bin_ms_init=20; end

fig = figure('Name','Sniff PSTH — Population','Color','w','Position',[80 80 820 520]);
draw_psth(fig, D, bin_ms_init, false);
add_controls(fig, D, bin_ms_init, false);
end

%% ── draw ──────────────────────────────────────────────────────────────────
function draw_psth(fig, D, bin_ms, use_ms)
delete(findobj(fig,'Type','axes'));
med_dur_ms = D.sniff_dur_s * 1000;
bin_s      = bin_ms / 1000;
win        = [0, D.sniff_dur_s];
onset_s    = D.sniff_onsets_s(:);
n_ev       = numel(onset_s);

% Pool all spike times — one psthAndBA call, no per-unit normalization
[pop_psth, bins_s] = psthAndBA(D.spikeTimes, onset_s, win, bin_s);

if use_ms
    centers = bins_s * 1000;
    xl      = sprintf('Time from sniff onset (ms)  [median cycle \\approx %.0f ms]', med_dur_ms);
    ttl     = sprintf('Population PSTH  [bin = %.0f ms | actual time | %d events]', diff(centers(1:2)), n_ev);
    xlims   = [-0.05*med_dur_ms, med_dur_ms];
else
    centers = bins_s / D.sniff_dur_s;
    bin_ph  = diff(centers(1:2));
    xl      = sprintf('Normalized sniff phase  (1 phase \\approx %.0f ms)', med_dur_ms);
    ttl     = sprintf('Population PSTH  [bin = %.0f ms = %.3f phase | %d events]', bin_ms, bin_ph, n_ev);
    xlims   = [-0.05, 1];
end

ax = axes(fig, 'Position', [0.10 0.16 0.85 0.74]);
bar(ax, centers, pop_psth, 1, 'FaceColor', [0.35 0.55 0.80], 'EdgeColor', 'none');
hold(ax,'on');
xline(ax, 0, 'r--', 'LineWidth', 3, 'Label', '  Sniff onset');
ylabel(ax, 'Population firing rate (spikes/s)');
xlabel(ax, xl);
title(ax, ttl);
xlim(ax, xlims);
set(ax, 'Box','off', 'FontSize',11);
end

%% ── controls ──────────────────────────────────────────────────────────────
function add_controls(fig, D, bin_init, use_ms_init)
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

cb = @(~,~) do_redraw(fig, D, uic, rbg);
rbg.SelectionChangedFcn = cb;
uicontrol(fig,'Style','pushbutton','String','Redraw','Units','normalized',...
    'Position',[0.37 0.01 0.09 0.06],'FontSize',10,'Callback',cb);
end

%% ── redraw callback ───────────────────────────────────────────────────────
function do_redraw(fig, D, uic, rbg)
v = str2double(get(uic,'String'));
if isnan(v)||v<=0; warndlg('Enter a positive number for bin width (ms).','Invalid'); return; end
use_ms = strcmp(get(rbg.SelectedObject,'String'), 'Actual time (ms)');
delete(findobj(fig,'Type','uicontrol'));
delete(findobj(fig,'Type','uibuttongroup'));
draw_psth(fig, D, v, use_ms);
add_controls(fig, D, v, use_ms);
end

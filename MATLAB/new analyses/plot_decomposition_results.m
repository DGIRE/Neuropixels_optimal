function plot_decomposition_results(D)
% plot_decomposition_results  Visualise decomposition output (ICA / PCA / Raw).
%
%   Works with D.DECOMP populated by run_decomposition.  If multiple
%   combinations were run, a selection dialog lets you choose which to view,
%   or view all in sequence.
%
%   Figure 1 — 4-panel trial view
%   --------------------------------
%     Panel 1: reference SNF signal for a user-selected trial
%     Panel 2: reference ETH signal for the same trial
%     Panel 3: best-correlated component / unit for the SNF signal
%     Panel 4: best-correlated component / unit for the ETH signal
%
%   Figure 2 — Rank scatter plot
%   --------------------------------
%     X-axis : SNF rank  (1 = strongest zero-lag xcorr)
%     Y-axis : ETH rank  (1 = strongest zero-lag xcorr)
%     Each point is one component / unit; top 5 labelled.
%
%   Requires: D.DECOMP (from run_decomposition), D.TR, D.SNF, D.SNF_PH,
%             D.ETH, D.ETH_thr

if ~isfield(D, 'DECOMP') || isempty(D.DECOMP)
    error(['D.DECOMP not found or empty. Run run_decomposition(D) first.\n' ...
           '(If you previously ran run_ICA, use plot_ICA_results instead.)']);
end

n_results = length(D.DECOMP);

%% -----------------------------------------------------------------------
%% Select which result(s) to visualise
%% -----------------------------------------------------------------------

if n_results == 1
    to_plot = 1;
else
    result_labels = {D.DECOMP.label};
    options       = [result_labels, {'— View all in sequence —'}];

    choice = listdlg('PromptString', 'Select result to visualise:', ...
                     'SelectionMode', 'single', ...
                     'ListString',    options, ...
                     'Name',          'Decomposition Results', ...
                     'ListSize',      [380, min(320, 24*(n_results+2))]);

    if isempty(choice)
        fprintf('No result selected.\n');
        return;
    end

    if choice == n_results + 1
        to_plot = 1:n_results;   % "View all"
    else
        to_plot = choice;
    end
end

%% -----------------------------------------------------------------------
%% Plot each selected result
%% -----------------------------------------------------------------------

for ri = to_plot
    show_result(D.DECOMP(ri), D);
end

end  % plot_decomposition_results


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function show_result(R, D)
% Produce Figure 1 (4-panel trial view) and Figure 2 (rank scatter) for
% one D.DECOMP element R.  D is needed to retrieve per-trial signal data.

LV_Fs        = R.LV_Fs;
n_components = R.n_components;
S_out        = R.S;
rank_SNF     = R.rank_SNF;
rank_ETH     = R.rank_ETH;
corr_SNF     = R.corr_SNF;
corr_ETH     = R.corr_ETH;
comp_label   = R.comp_label;   % 'IC' | 'PC' | 'Unit'
comp_ids     = R.comp_ids;
sigma_ms     = R.sigma_ms;
method       = R.method;
eth_label    = R.eth_label;
snf_label    = R.snf_label;

% Best-correlated components (rank 1)
[~, best_snf] = min(rank_SNF);
[~, best_eth] = min(rank_ETH);

% Helper: format a component label for figure annotations
fmt = @(idx) sprintf('%s %d', comp_label, comp_ids(idx));

% Short string used in window titles
win_tag = sprintf('%s | %s | %s | sigma=%.0fms', ...
                  method, eth_label, snf_label, sigma_ms);

% Retrieve the reference signals (full-length vectors) for trial display
if strcmp(R.snf_choice, 'proc')
    snf_full = D.SNF_PH(:);
else
    snf_full = D.SNF(:);
end

if strcmp(R.eth_choice, 'proc')
    eth_full = D.ETH_thr(:);
else
    eth_full = D.ETH(:);
end

% =========================================================================
%% FIGURE 1 — 4-panel trial view
% =========================================================================

uT = unique(D.TR);
nT = length(uT);

if nT == 0
    warning('plot_decomposition_results: no trials in D.TR — skipping Figure 1.');
else
    trial_strs = arrayfun(@(x) sprintf('Trial %d', x), uT, 'UniformOutput', false);

    choice = listdlg( ...
        'PromptString', sprintf('Select trial  [%s]:', win_tag), ...
        'SelectionMode', 'single', ...
        'ListString',    trial_strs, ...
        'Name',          sprintf('%s Trial View', method), ...
        'ListSize',      [320, 300]);

    if isempty(choice)
        fprintf('No trial selected — skipping Figure 1 for [%s].\n', win_tag);
    else
        trt  = uT(choice);
        mask = (D.TR == trt);
        t_lv = (find(mask) - find(mask, 1)) / LV_Fs;   % time axis (s)

        snf_trial      = snf_full(mask);
        eth_trial      = eth_full(mask);
        snf_comp_trial = S_out(best_snf, mask);
        eth_comp_trial = S_out(best_eth, mask);

        fig1 = figure('Name',        sprintf('Trial View — Trial %d  [%s]', trt, win_tag), ...
                      'Color',       'w', ...
                      'NumberTitle', 'off');

        % Panel 1: SNF reference
        ax(1) = subplot(4, 1, 1, 'Parent', fig1);
        plot(ax(1), t_lv, snf_trial, 'Color', [0.2 0.5 0.9], 'LineWidth', 1);
        ylabel(ax(1), snf_label);
        title(ax(1), sprintf('Trial %d — %s', trt, snf_label));
        set(ax(1), 'XTickLabel', []);

        % Panel 2: ETH reference
        ax(2) = subplot(4, 1, 2, 'Parent', fig1);
        plot(ax(2), t_lv, eth_trial, 'Color', [0.9 0.4 0.1], 'LineWidth', 1);
        ylabel(ax(2), eth_label);
        title(ax(2), eth_label);
        set(ax(2), 'XTickLabel', []);

        % Panel 3: best SNF match
        ax(3) = subplot(4, 1, 3, 'Parent', fig1);
        plot(ax(3), t_lv, snf_comp_trial, 'Color', [0.1 0.7 0.3], 'LineWidth', 1);
        ylabel(ax(3), 'Activation');
        title(ax(3), sprintf('%s — Best %s match  (r_{0} = %.3f)', ...
              fmt(best_snf), snf_label, corr_SNF(best_snf)));
        set(ax(3), 'XTickLabel', []);

        % Panel 4: best ETH match
        ax(4) = subplot(4, 1, 4, 'Parent', fig1);
        plot(ax(4), t_lv, eth_comp_trial, 'Color', [0.7 0.1 0.7], 'LineWidth', 1);
        ylabel(ax(4), 'Activation');
        xlabel(ax(4), 'Time in trial (s)');
        title(ax(4), sprintf('%s — Best %s match  (r_{0} = %.3f)', ...
              fmt(best_eth), eth_label, corr_ETH(best_eth)));

        linkaxes(ax, 'x');
        set(fig1, 'Position', [100 100 900 700]);
    end
end

% =========================================================================
%% FIGURE 2 — Rank scatter plot
% =========================================================================

fig2 = figure('Name',        sprintf('Ranks  [%s]', win_tag), ...
              'Color',       'w', ...
              'NumberTitle', 'off');
ax2 = axes('Parent', fig2);

scatter(ax2, rank_SNF, rank_ETH, 60, [0.2 0.4 0.8], 'filled');
hold(ax2, 'on');

% Label top 5 by combined rank (sum of ranks, lowest = most dual-purpose)
combined_rank = rank_SNF + rank_ETH;
[~, top_idx]  = sort(combined_rank);
n_label = min(5, n_components);

for k = 1:n_label
    ic = top_idx(k);
    text(ax2, rank_SNF(ic) + 0.15, rank_ETH(ic), ...
         fmt(ic), 'FontSize', 9, 'Color', [0.8 0.1 0.1]);
end

% Highlight best-per-signal
plot(ax2, rank_SNF(best_snf), rank_ETH(best_snf), 'g^', 'MarkerSize', 10, ...
     'MarkerFaceColor', [0.1 0.7 0.3], 'DisplayName', sprintf('Best %s', snf_label));
plot(ax2, rank_SNF(best_eth), rank_ETH(best_eth), 'rs', 'MarkerSize', 10, ...
     'MarkerFaceColor', [0.9 0.2 0.2], 'DisplayName', sprintf('Best %s', eth_label));

legend(ax2, {sprintf('All %ss', comp_label), '', ...
             sprintf('Best %s', snf_label), ...
             sprintf('Best %s', eth_label)}, ...
       'Location', 'northeast');

xlabel(ax2, sprintf('%s rank — %s xcorr (1 = best)', comp_label, snf_label));
ylabel(ax2, sprintf('%s rank — %s xcorr (1 = best)', comp_label, eth_label));
title(ax2, sprintf('%s  —  %s  |  sigma = %.0f ms', comp_label, method, sigma_ms));

% Diagonal reference line
max_rank = n_components;
plot(ax2, [1 max_rank], [1 max_rank], 'k--', 'LineWidth', 0.8, 'HandleVisibility', 'off');

axis(ax2, 'equal');
xlim(ax2, [0.5, max_rank + 0.5]);
ylim(ax2, [0.5, max_rank + 0.5]);
grid(ax2, 'on');
set(fig2, 'Position', [200 200 600 550]);

end  % show_result

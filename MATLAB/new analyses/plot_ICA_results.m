function plot_ICA_results(D)
% plot_ICA_results  Visualise ICA output in two figures.
%
%   Figure 1 — 4-panel trial view (user selects trial):
%     Panel 1: SNF_PH for the selected trial
%     Panel 2: ETH_thr for the same trial
%     Panel 3: Top-ranked IC time series for SNF_PH match
%     Panel 4: Top-ranked IC time series for ETH_thr match
%
%   Figure 2 — IC rank scatter plot:
%     X-axis: rank for SNF_PH correspondence (1=best)
%     Y-axis: rank for ETH_thr correspondence (1=best)
%     Each dot is one IC; top ICs are labelled.
%
%   Requires D.ICA (output of run_ICA), D.SNF_PH, D.ETH_thr, D.TR, D.LV_Fs

if ~isfield(D, 'ICA')
    error('D.ICA not found. Run run_ICA(D) first.');
end

LV_Fs       = D.ICA.LV_Fs;
n_components = D.ICA.n_components;
S_ica       = D.ICA.S;           % [n_components × nSamp]
rank_SNF    = D.ICA.rank_SNF;
rank_ETH    = D.ICA.rank_ETH;
corr_SNF    = D.ICA.corr_SNF;
corr_ETH    = D.ICA.corr_ETH;

% Best ICs (rank 1)
[~, ic_snf] = min(rank_SNF);
[~, ic_eth] = min(rank_ETH);

% =========================================================================
%% FIGURE 1 — 4-panel trial view
% =========================================================================

% --- select trial ---
uT = unique(D.TR);
nT = length(uT);

if nT == 0
    warning('No trials found in D.TR. Skipping Figure 1.');
else
    choice = listdlg('PromptString', 'Select a trial to display:', ...
                     'SelectionMode', 'single', ...
                     'ListString',    arrayfun(@(x) sprintf('Trial %d', x), uT, 'UniformOutput', false), ...
                     'Name',          'ICA Trial View', ...
                     'ListSize',      [200, 300]);

    if isempty(choice)
        fprintf('No trial selected — skipping Figure 1.\n');
    else
        trt = uT(choice);
        mask = (D.TR == trt);

        t_lv = (find(mask) - find(mask, 1)) / LV_Fs;  % time in seconds within trial

        snf_ph_trial = D.SNF_PH(mask);
        eth_thr_trial = D.ETH_thr(mask);
        ic_snf_trial  = S_ica(ic_snf, mask);
        ic_eth_trial  = S_ica(ic_eth, mask);

        fig1 = figure('Name', sprintf('ICA Trial View — Trial %d', trt), ...
                      'Color', 'w', 'NumberTitle', 'off');

        % Panel 1: SNF_PH
        ax(1) = subplot(4, 1, 1, 'Parent', fig1);
        plot(ax(1), t_lv, snf_ph_trial, 'Color', [0.2 0.5 0.9], 'LineWidth', 1);
        ylabel(ax(1), 'Sniff Phase');
        title(ax(1), sprintf('Trial %d — Sniff Phase (SNF\\_PH)', trt));
        ylim(ax(1), [-0.2, 1.2]);
        set(ax(1), 'XTickLabel', []);

        % Panel 2: ETH_thr
        ax(2) = subplot(4, 1, 2, 'Parent', fig1);
        plot(ax(2), t_lv, eth_thr_trial, 'Color', [0.9 0.4 0.1], 'LineWidth', 1);
        ylabel(ax(2), 'ETH (thr)');
        title(ax(2), 'Thresholded Ethanol (ETH\_thr)');
        set(ax(2), 'XTickLabel', []);

        % Panel 3: top IC for SNF_PH
        ax(3) = subplot(4, 1, 3, 'Parent', fig1);
        plot(ax(3), t_lv, ic_snf_trial, 'Color', [0.1 0.7 0.3], 'LineWidth', 1);
        ylabel(ax(3), 'Activation (a.u.)');
        title(ax(3), sprintf('IC %d — Best SNF\\_PH match  (r = %.3f)', ic_snf, corr_SNF(ic_snf)));
        set(ax(3), 'XTickLabel', []);

        % Panel 4: top IC for ETH_thr
        ax(4) = subplot(4, 1, 4, 'Parent', fig1);
        plot(ax(4), t_lv, ic_eth_trial, 'Color', [0.7 0.1 0.7], 'LineWidth', 1);
        ylabel(ax(4), 'Activation (a.u.)');
        xlabel(ax(4), 'Time in trial (s)');
        title(ax(4), sprintf('IC %d — Best ETH\\_thr match  (r = %.3f)', ic_eth, corr_ETH(ic_eth)));

        linkaxes(ax, 'x');
        set(fig1, 'Position', [100 100 900 700]);
    end
end

% =========================================================================
%% FIGURE 2 — IC rank scatter plot
% =========================================================================

fig2 = figure('Name', 'ICA Component Ranks', 'Color', 'w', 'NumberTitle', 'off');
ax2  = axes('Parent', fig2);

scatter(ax2, rank_SNF, rank_ETH, 60, [0.2 0.4 0.8], 'filled');
hold(ax2, 'on');

% Label the 5 best ICs by combined rank (sum of ranks, lowest = most dual-purpose)
combined_rank = rank_SNF + rank_ETH;
[~, top_idx]  = sort(combined_rank);
n_label = min(5, n_components);

for k = 1:n_label
    ic = top_idx(k);
    text(ax2, rank_SNF(ic) + 0.15, rank_ETH(ic), ...
         sprintf('IC %d', ic), 'FontSize', 9, 'Color', [0.8 0.1 0.1]);
end

% Mark top SNF IC
plot(ax2, rank_SNF(ic_snf), rank_ETH(ic_snf), 'g^', 'MarkerSize', 10, ...
     'MarkerFaceColor', [0.1 0.7 0.3], 'DisplayName', 'Best SNF');
% Mark top ETH IC
plot(ax2, rank_SNF(ic_eth), rank_ETH(ic_eth), 'rs', 'MarkerSize', 10, ...
     'MarkerFaceColor', [0.9 0.2 0.2], 'DisplayName', 'Best ETH');

legend(ax2, {'All ICs', '', 'Best SNF_{PH}', 'Best ETH_{thr}'}, 'Location', 'northeast');

xlabel(ax2, 'IC Rank — SNF\_PH correspondence (1 = best)');
ylabel(ax2, 'IC Rank — ETH\_thr correspondence (1 = best)');
title(ax2, 'ICA Component Ranks (lower-left = best dual correspondence)');

% Diagonal reference line
max_rank = n_components;
plot(ax2, [1 max_rank], [1 max_rank], 'k--', 'LineWidth', 0.8, 'HandleVisibility', 'off');

axis(ax2, 'equal');
xlim(ax2, [0.5, max_rank + 0.5]);
ylim(ax2, [0.5, max_rank + 0.5]);

grid(ax2, 'on');
set(fig2, 'Position', [200 200 600 550]);

end

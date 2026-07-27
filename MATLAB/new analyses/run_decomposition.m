function D = run_decomposition(D)
% run_decomposition  Configure and run the full decomposition + correlation pipeline.
%
%   D = run_decomposition(D)
%
%   A single checkbox UI window lets you configure every combination at once:
%
%     Decomposition method  :  ICA  |  PCA  |  Raw convolved units
%     ETH reference signal  :  Processed ETH (D.ETH_thr)  |  Raw ETH (D.ETH)
%     SNF reference signal  :  Processed SNF (D.SNF_PH)   |  Raw SNF (D.SNF)
%
%   All checked combinations are run in one pass.  The spike-rate matrix is
%   built once per sigma value; decomposition is run once per method; then
%   zero-lag cross-correlation is computed for every (decomp × ETH × SNF)
%   combination.
%
%   Results are stored as a struct array D.DECOMP — one element per
%   combination.  Use plot_decomposition_results(D) to visualise.
%
%   D.DECOMP(i) fields
%   ------------------
%     method        – 'ICA' | 'PCA' | 'Raw'
%     eth_choice    – 'proc' | 'raw'
%     snf_choice    – 'proc' | 'raw'
%     label         – short display string, e.g. 'ICA | procETH | rawSNF'
%     S             – [n_components × nSamp]  component / unit time series
%     corr_SNF      – [n_components × 1]  zero-lag xcorr with SNF signal
%     corr_ETH      – [n_components × 1]  zero-lag xcorr with ETH signal
%     rank_SNF      – [n_components × 1]  rank by |corr_SNF| (1 = best)
%     rank_ETH      – [n_components × 1]  rank by |corr_ETH| (1 = best)
%     n_components  – number of components / units retained
%     comp_label    – 'IC' | 'PC' | 'Unit'  (for axis labels)
%     comp_ids      – component index (IC/PC) or unit cluster ID (Raw)
%     LV_Fs         – LabView sampling rate used
%     sigma_ms      – Gaussian kernel sigma used (ms)
%
%   Required D fields
%   -----------------
%     D.spikeTimes, D.sp.clu, D.unitIDs
%     D.SNF, D.SNF_PH, D.ETH, D.ETH_thr, D.LV_Fs

fprintf('\n=== Decomposition Pipeline ===\n');

nUnits = length(D.unitIDs);
LV_Fs  = D.LV_Fs;
nSamp  = length(D.SNF_PH);

%% -----------------------------------------------------------------------
%% Show configuration UI
%% -----------------------------------------------------------------------

params = decomp_config_dialog(nUnits);
if isempty(params)
    fprintf('Analysis cancelled.\n');
    return;
end

fprintf('  Kernel sigma : %.1f ms\n', params.sigma_ms);
fprintf('  Methods      : %s\n', strjoin(params.methods, ', '));
fprintf('  ETH signals  : %s\n', strjoin(params.eth_choices, ', '));
fprintf('  SNF signals  : %s\n', strjoin(params.snf_choices, ', '));

n_combos = length(params.methods) * length(params.eth_choices) * length(params.snf_choices);
fprintf('  Total combinations: %d\n\n', n_combos);

%% -----------------------------------------------------------------------
%% Step 1 — Build Gaussian-convolved spike-rate matrix  [nUnits × nSamp]
%% -----------------------------------------------------------------------

fprintf('  Building spike-rate matrix (sigma = %.1f ms)...\n', params.sigma_ms);

sigma_s    = params.sigma_ms / 1000;
sigma_samp = sigma_s * LV_Fs;
hw         = ceil(3 * sigma_samp);
t_kern     = (-hw:hw)';
kernel     = exp(-0.5 * (t_kern / sigma_samp).^2);
kernel     = kernel / sum(kernel);

X = zeros(nUnits, nSamp);
for u = 1:nUnits
    uid  = D.unitIDs(u);
    st_s = D.spikeTimes(D.sp.clu == uid);
    idx  = round(st_s * LV_Fs) + 1;
    idx  = idx(idx >= 1 & idx <= nSamp);

    rate = zeros(1, nSamp);
    for k = 1:length(idx)
        rate(idx(k)) = rate(idx(k)) + 1;
    end

    nfft    = 2^nextpow2(nSamp + length(kernel) - 1);
    rate_sm = real(ifft(fft(rate, nfft) .* fft(kernel', nfft)));
    X(u, :) = rate_sm(hw+1 : hw+nSamp);
end
fprintf('  Spike-rate matrix complete.\n\n');

%% -----------------------------------------------------------------------
%% Step 2 — Decompose each selected method (once per method)
%% -----------------------------------------------------------------------

decomp_cache = struct();   % cache S_out per method so we don't recompute

for mi = 1:length(params.methods)
    method = params.methods{mi};
    fprintf('  Decomposing: %s ...\n', method);

    switch method

        case 'ICA'
            Xc     = X - mean(X, 2);
            [U, S, ~] = svd(Xc, 'econ');
            s_diag = diag(S);
            keep   = s_diag > 1e-6 * s_diag(1);
            n_comp = min(params.n_components, sum(keep));
            U      = U(:, 1:n_comp);
            sv     = s_diag(1:n_comp);
            Z      = diag(1 ./ sv) * U' * Xc;

            W       = eye(n_comp);
            tol_ica = 1e-6;
            for p = 1:n_comp
                w = randn(n_comp, 1);  w = w / norm(w);
                for iter = 1:500
                    u_    = w' * Z;
                    g     = tanh(u_);
                    gp    = 1 - g.^2;
                    w_new = (Z * g') / nSamp - mean(gp) * w;
                    for q = 1:p-1
                        w_new = w_new - (w_new' * W(q,:)') * W(q,:)';
                    end
                    w_new = w_new / norm(w_new);
                    if abs(abs(w_new' * w) - 1) < tol_ica; w = w_new; break; end
                    w = w_new;
                end
                W(p,:) = w';
            end
            S_out      = W * Z;
            comp_label = 'IC';
            comp_ids   = (1:n_comp)';

        case 'PCA'
            Xc     = X - mean(X, 2);
            [U, S, ~] = svd(Xc, 'econ');
            s_diag = diag(S);
            keep   = s_diag > 1e-6 * s_diag(1);
            n_comp = min(params.n_components, sum(keep));
            S_out      = U(:, 1:n_comp)' * Xc;
            comp_label = 'PC';
            comp_ids   = (1:n_comp)';

        case 'Raw'
            n_comp     = nUnits;
            S_out      = X;
            comp_label = 'Unit';
            comp_ids   = D.unitIDs(:);
    end

    % Cache this decomposition
    decomp_cache.(method).S_out      = S_out;
    decomp_cache.(method).comp_label = comp_label;
    decomp_cache.(method).comp_ids   = comp_ids;
    decomp_cache.(method).n_comp     = n_comp;

    fprintf('    %s complete (%d components).\n', method, n_comp);
end

%% -----------------------------------------------------------------------
%% Step 3 — Correlate each (method × ETH × SNF) combination
%% -----------------------------------------------------------------------

fprintf('\n  Computing zero-lag cross-correlations...\n');

D.DECOMP = struct([]);   % initialise as empty struct array
result_idx = 0;

for mi = 1:length(params.methods)
    method     = params.methods{mi};
    S_out      = decomp_cache.(method).S_out;
    comp_label = decomp_cache.(method).comp_label;
    comp_ids   = decomp_cache.(method).comp_ids;
    n_comp     = decomp_cache.(method).n_comp;

    for ei = 1:length(params.eth_choices)
        eth_choice = params.eth_choices{ei};

        if strcmp(eth_choice, 'proc')
            eth_sig   = D.ETH_thr(:);
            eth_label = 'Processed ETH';
        else
            eth_sig   = D.ETH(:);
            eth_label = 'Raw ETH';
        end

        for si = 1:length(params.snf_choices)
            snf_choice = params.snf_choices{si};

            if strcmp(snf_choice, 'proc')
                snf_sig   = D.SNF_PH(:);
                valid_snf = (D.SNF_PH >= 0);   % mask -1 sentinel values
                snf_label = 'Processed SNF';
            else
                snf_sig   = D.SNF(:);
                valid_snf = true(size(snf_sig));
                snf_label = 'Raw SNF';
            end

            % --- zero-lag cross-correlation for each component -----------
            corr_SNF = zeros(n_comp, 1);
            corr_ETH = zeros(n_comp, 1);

            snf_masked = snf_sig(valid_snf);

            for ic = 1:n_comp
                % SNF — use only valid phase samples if processed SNF
                s_snf = S_out(ic, valid_snf)';
                corr_SNF(ic) = zero_lag_xcorr(s_snf, snf_masked);

                % ETH — all samples
                s_eth = S_out(ic, :)';
                corr_ETH(ic) = zero_lag_xcorr(s_eth, eth_sig);
            end

            % --- rank (1 = strongest |correlation|) ----------------------
            [~, ord_snf] = sort(abs(corr_SNF), 'descend');
            [~, ord_eth] = sort(abs(corr_ETH), 'descend');
            rank_SNF = zeros(n_comp, 1);
            rank_ETH = zeros(n_comp, 1);
            for r = 1:n_comp
                rank_SNF(ord_snf(r)) = r;
                rank_ETH(ord_eth(r)) = r;
            end

            % --- store result --------------------------------------------
            result_idx = result_idx + 1;
            combo_label = sprintf('%s | %s | %s', method, eth_label, snf_label);

            D.DECOMP(result_idx).method       = method;
            D.DECOMP(result_idx).eth_choice   = eth_choice;
            D.DECOMP(result_idx).snf_choice   = snf_choice;
            D.DECOMP(result_idx).label        = combo_label;
            D.DECOMP(result_idx).S            = S_out;
            D.DECOMP(result_idx).corr_SNF     = corr_SNF;
            D.DECOMP(result_idx).corr_ETH     = corr_ETH;
            D.DECOMP(result_idx).rank_SNF     = rank_SNF;
            D.DECOMP(result_idx).rank_ETH     = rank_ETH;
            D.DECOMP(result_idx).n_components = n_comp;
            D.DECOMP(result_idx).comp_label   = comp_label;
            D.DECOMP(result_idx).comp_ids     = comp_ids;
            D.DECOMP(result_idx).LV_Fs        = LV_Fs;
            D.DECOMP(result_idx).sigma_ms     = params.sigma_ms;
            D.DECOMP(result_idx).eth_label    = eth_label;
            D.DECOMP(result_idx).snf_label    = snf_label;

            top_snf = comp_ids(ord_snf(1));
            top_eth = comp_ids(ord_eth(1));
            fprintf('    [%s]\n      Best %s for SNF: %s %d  (r=%.3f)\n', ...
                combo_label, comp_label, comp_label, top_snf, corr_SNF(ord_snf(1)));
            fprintf('      Best %s for ETH: %s %d  (r=%.3f)\n', ...
                comp_label, comp_label, top_eth, corr_ETH(ord_eth(1)));
        end
    end
end

fprintf('\n=== Decomposition complete — %d result(s) stored in D.DECOMP ===\n\n', result_idx);

end  % run_decomposition

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function params = decomp_config_dialog(nUnits)
% Build and show the checkbox configuration UI.  Returns params struct or
% empty [] if the user cancels.

params = [];

fw = 400;  fh = 530;
fig = figure('Name',        'Configure Decomposition Analysis', ...
             'Position',    [400 180 fw fh], ...
             'MenuBar',     'none', ...
             'ToolBar',     'none', ...
             'NumberTitle', 'off', ...
             'Resize',      'off', ...
             'Color',       [0.94 0.94 0.94]);

bg = [0.94 0.94 0.94];

% ---- Title ---------------------------------------------------------------
uicontrol(fig, 'Style', 'text', ...
    'String', 'Configure Decomposition Analysis', ...
    'Units', 'pixels', 'Position', [10 490 380 28], ...
    'FontSize', 12, 'FontWeight', 'bold', ...
    'BackgroundColor', bg, 'HorizontalAlignment', 'center');

% ---- Section 1: Decomposition Method ------------------------------------
uicontrol(fig, 'Style', 'text', 'String', 'Decomposition Method', ...
    'Units', 'pixels', 'Position', [15 456 220 20], ...
    'FontSize', 10, 'FontWeight', 'bold', ...
    'BackgroundColor', bg, 'HorizontalAlignment', 'left');
cb_ica = uicontrol(fig, 'Style', 'checkbox', ...
    'String', 'ICA  (Independent Component Analysis)', ...
    'Units', 'pixels', 'Position', [28 432 350 22], 'Value', 1, ...
    'FontSize', 10, 'BackgroundColor', bg);
cb_pca = uicontrol(fig, 'Style', 'checkbox', ...
    'String', 'PCA  (Principal Component Analysis)', ...
    'Units', 'pixels', 'Position', [28 408 350 22], 'Value', 0, ...
    'FontSize', 10, 'BackgroundColor', bg);
cb_raw = uicontrol(fig, 'Style', 'checkbox', ...
    'String', 'Raw  (Convolved unit activity)', ...
    'Units', 'pixels', 'Position', [28 384 350 22], 'Value', 0, ...
    'FontSize', 10, 'BackgroundColor', bg);

% ---- Section 2: ETH Signal -----------------------------------------------
uicontrol(fig, 'Style', 'text', 'String', 'ETH Reference Signal', ...
    'Units', 'pixels', 'Position', [15 350 220 20], ...
    'FontSize', 10, 'FontWeight', 'bold', ...
    'BackgroundColor', bg, 'HorizontalAlignment', 'left');
cb_eth_proc = uicontrol(fig, 'Style', 'checkbox', ...
    'String', 'Processed ETH  (threshold-clipped,  D.ETH_thr)', ...
    'Units', 'pixels', 'Position', [28 326 350 22], 'Value', 1, ...
    'FontSize', 10, 'BackgroundColor', bg);
cb_eth_raw = uicontrol(fig, 'Style', 'checkbox', ...
    'String', 'Unprocessed ETH  (raw signal,  D.ETH)', ...
    'Units', 'pixels', 'Position', [28 302 350 22], 'Value', 0, ...
    'FontSize', 10, 'BackgroundColor', bg);

% ---- Section 3: SNF Signal -----------------------------------------------
uicontrol(fig, 'Style', 'text', 'String', 'SNF Reference Signal', ...
    'Units', 'pixels', 'Position', [15 268 220 20], ...
    'FontSize', 10, 'FontWeight', 'bold', ...
    'BackgroundColor', bg, 'HorizontalAlignment', 'left');
cb_snf_proc = uicontrol(fig, 'Style', 'checkbox', ...
    'String', 'Processed SNF  (sniff phase 0–1,  D.SNF_PH)', ...
    'Units', 'pixels', 'Position', [28 244 350 22], 'Value', 1, ...
    'FontSize', 10, 'BackgroundColor', bg);
cb_snf_raw = uicontrol(fig, 'Style', 'checkbox', ...
    'String', 'Unprocessed SNF  (raw signal,  D.SNF)', ...
    'Units', 'pixels', 'Position', [28 220 350 22], 'Value', 0, ...
    'FontSize', 10, 'BackgroundColor', bg);

% ---- Section 4: Parameters -----------------------------------------------
uicontrol(fig, 'Style', 'text', 'String', 'Parameters', ...
    'Units', 'pixels', 'Position', [15 184 220 20], ...
    'FontSize', 10, 'FontWeight', 'bold', ...
    'BackgroundColor', bg, 'HorizontalAlignment', 'left');

uicontrol(fig, 'Style', 'text', 'String', 'Kernel sigma (ms):', ...
    'Units', 'pixels', 'Position', [28 160 160 20], ...
    'FontSize', 10, 'BackgroundColor', bg, 'HorizontalAlignment', 'left');
ed_sigma = uicontrol(fig, 'Style', 'edit', 'String', '50', ...
    'Units', 'pixels', 'Position', [192 160 70 22], 'FontSize', 10);

uicontrol(fig, 'Style', 'text', ...
    'String', sprintf('Num components  (ICA/PCA, max %d):', nUnits), ...
    'Units', 'pixels', 'Position', [28 132 240 20], ...
    'FontSize', 10, 'BackgroundColor', bg, 'HorizontalAlignment', 'left');
ed_nc = uicontrol(fig, 'Style', 'edit', 'String', num2str(min(nUnits, 20)), ...
    'Units', 'pixels', 'Position', [272 132 70 22], 'FontSize', 10);

% ---- Buttons -------------------------------------------------------------
setappdata(fig, 'cancelled', true);

uicontrol(fig, 'Style', 'pushbutton', 'String', 'Run Analysis', ...
    'Units', 'pixels', 'Position', [70 42 140 40], ...
    'FontSize', 11, 'FontWeight', 'bold', ...
    'BackgroundColor', [0.15 0.55 0.15], 'ForegroundColor', 'w', ...
    'Callback', @(~,~) do_run(fig));

uicontrol(fig, 'Style', 'pushbutton', 'String', 'Cancel', ...
    'Units', 'pixels', 'Position', [228 42 110 40], ...
    'FontSize', 11, ...
    'BackgroundColor', [0.7 0.7 0.7], ...
    'Callback', @(~,~) uiresume(fig));

uiwait(fig);

% --- figure closed by X or cancelled --------------------------------------
if ~isvalid(fig) || getappdata(fig, 'cancelled')
    if isvalid(fig); close(fig); end
    return;
end

% --- read checkbox values -------------------------------------------------
methods = {};
if get(cb_ica, 'Value'); methods{end+1} = 'ICA'; end
if get(cb_pca, 'Value'); methods{end+1} = 'PCA'; end
if get(cb_raw, 'Value'); methods{end+1} = 'Raw'; end

eth_choices = {};
if get(cb_eth_proc, 'Value'); eth_choices{end+1} = 'proc'; end
if get(cb_eth_raw,  'Value'); eth_choices{end+1} = 'raw';  end

snf_choices = {};
if get(cb_snf_proc, 'Value'); snf_choices{end+1} = 'proc'; end
if get(cb_snf_raw,  'Value'); snf_choices{end+1} = 'raw';  end

sigma_ms = str2double(get(ed_sigma, 'String'));
if isnan(sigma_ms) || sigma_ms <= 0; sigma_ms = 50; end

n_components = round(str2double(get(ed_nc, 'String')));
if isnan(n_components) || n_components < 1; n_components = min(nUnits, 20); end
n_components = min(n_components, nUnits);

close(fig);

% --- validate at least one selection per group ----------------------------
if isempty(methods)
    errordlg('Select at least one decomposition method.', 'Configuration Error');
    return;
end
if isempty(eth_choices)
    errordlg('Select at least one ETH signal.', 'Configuration Error');
    return;
end
if isempty(snf_choices)
    errordlg('Select at least one SNF signal.', 'Configuration Error');
    return;
end

params.methods      = methods;
params.eth_choices  = eth_choices;
params.snf_choices  = snf_choices;
params.sigma_ms     = sigma_ms;
params.n_components = n_components;

end  % decomp_config_dialog


function do_run(fig)
% Run button callback — mark as not cancelled and resume uiwait.
setappdata(fig, 'cancelled', false);
uiresume(fig);
end


function r = zero_lag_xcorr(x, y)
% Normalized zero-lag cross-correlation.
%   r = sum(x .* y) / sqrt( sum(x.^2) * sum(y.^2) )
% This is the 'coeff' normalization of xcorr evaluated at lag = 0.
% Returns 0 if either signal has zero energy.
x = x(:);  y = y(:);
denom = sqrt(sum(x.^2) * sum(y.^2));
if denom < eps
    r = 0;
else
    r = sum(x .* y) / denom;
end
end  % zero_lag_xcorr

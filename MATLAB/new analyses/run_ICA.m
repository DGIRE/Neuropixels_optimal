function D = run_ICA(D, n_components)
% run_ICA  Gaussian-smooth spike trains and run FastICA on unit activity.
%
%   D = run_ICA(D)
%   D = run_ICA(D, n_components)
%
%   Inputs
%   ------
%   D            : experiment struct from load_experiment_data / compute_sniff_phase
%                  Required fields:
%                    D.spikeTimes   – spike times in seconds (from NP clock)
%                    D.sp.clu       – cluster ID per spike (from loadKSdir)
%                    D.unitIDs      – vector of valid cluster IDs
%                    D.SNF_PH       – sniff phase (0-1 per cycle, -1 outside)
%                    D.ETH_thr      – thresholded ethanol signal
%                    D.LV_Fs        – LabView sampling rate (typically ~100-125 Hz)
%   n_components : number of ICA components to extract (default: min(nUnits,20))
%
%   Outputs
%   -------
%   D.ICA.S         – [n_components × nSamples] IC time series
%   D.ICA.W         – [n_components × nUnits]   unmixing matrix
%   D.ICA.corr_SNF  – [n_components × 1]  Pearson r with SNF_PH
%   D.ICA.corr_ETH  – [n_components × 1]  Pearson r with ETH_thr
%   D.ICA.rank_SNF  – [n_components × 1]  rank by |corr_SNF| (1=best)
%   D.ICA.rank_ETH  – [n_components × 1]  rank by |corr_ETH| (1=best)
%   D.ICA.n_components – number of components extracted

fprintf('\n=== ICA Pipeline ===\n');

%% --- defaults -----------------------------------------------------------------
LV_Fs = D.LV_Fs;           % LabView sampling rate (Hz)
nSamp  = length(D.SNF_PH); % total LabView samples

nUnits = length(D.unitIDs);

if nargin < 2 || isempty(n_components)
    n_components = min(nUnits, 20);
end
n_components = min(n_components, nUnits); % can't exceed nUnits

fprintf('  Units: %d   Components: %d   Samples: %d\n', nUnits, n_components, nSamp);

%% --- Step 1: build spike-rate matrix [nUnits × nSamp] -------------------------
fprintf('  Building spike-rate matrix...\n');

% Gaussian kernel: sigma = 50 ms, width = 6*sigma (truncated)
sigma_s  = 0.050;                    % 50 ms in seconds
sigma_samp = sigma_s * LV_Fs;       % in samples
hw       = ceil(3 * sigma_samp);    % half-width
t_kern   = (-hw:hw)';
kernel   = exp(-0.5 * (t_kern / sigma_samp).^2);
kernel   = kernel / sum(kernel);     % normalise to unit area

X = zeros(nUnits, nSamp);

for u = 1:nUnits
    uid  = D.unitIDs(u);
    % Convert NP spike times (seconds) to approximate LabView indices
    st_s = D.spikeTimes(D.sp.clu == uid);
    idx  = round(st_s * LV_Fs) + 1;
    idx  = idx(idx >= 1 & idx <= nSamp);   % bounds check

    % Accumulate spike counts into rate vector
    rate = zeros(1, nSamp);
    for k = 1:length(idx)
        rate(idx(k)) = rate(idx(k)) + 1;
    end

    % Convolve with Gaussian kernel using FFT for speed
    nfft = 2^nextpow2(nSamp + length(kernel) - 1);
    rate_sm = real(ifft(fft(rate, nfft) .* fft(kernel', nfft)));
    X(u, :) = rate_sm(hw+1 : hw+nSamp);   % trim to original length
end

fprintf('  Spike-rate matrix complete.\n');

%% --- Step 2: centre and whiten ------------------------------------------------
fprintf('  Whitening...\n');

Xc = X - mean(X, 2);          % zero-mean each unit

[U, S, ~] = svd(Xc, 'econ');  % economy SVD: U [nUnits×k], S [k×k]
s_diag = diag(S);

% Keep only components with non-trivial variance
tol    = 1e-6 * s_diag(1);
keep   = s_diag > tol;
U      = U(:, keep);
s_diag = s_diag(keep);

% Cap to requested number of components
n_components = min(n_components, sum(keep));
U      = U(:, 1:n_components);
s_diag = s_diag(1:n_components);

% Whitened data: Z = S^{-1} U' Xc   → [n_components × nSamp], unit variance
Z = diag(1 ./ s_diag) * U' * Xc;

fprintf('  Effective components after SVD truncation: %d\n', n_components);

%% --- Step 3: FastICA (deflation, logcosh nonlinearity) -----------------------
fprintf('  Running FastICA...\n');

W = eye(n_components);   % will hold rows of unmixing matrix in whitened space
tol_ica = 1e-6;
max_iter = 500;

for p = 1:n_components
    w = randn(n_components, 1);
    w = w / norm(w);

    for iter = 1:max_iter
        % logcosh: g(u)=tanh(u), g'(u)=1-tanh^2(u)
        u  = w' * Z;                   % 1 × nSamp
        g  = tanh(u);                  % 1 × nSamp
        gp = 1 - g.^2;                 % 1 × nSamp

        w_new = (Z * g') / nSamp  -  mean(gp) * w;

        % Deflation: project out already-found directions
        for q = 1:p-1
            w_new = w_new - (w_new' * W(q,:)') * W(q,:)';
        end

        w_new = w_new / norm(w_new);

        % Convergence check (sign-invariant)
        if abs(abs(w_new' * w) - 1) < tol_ica
            w = w_new;
            break;
        end
        w = w_new;
    end

    W(p, :) = w';
end

%% --- Step 4: compute IC time series and unmixing in original space -----------
S_ica = W * Z;   % [n_components × nSamp]  — IC activations

% Unmixing matrix back in original (pre-whitened) unit space:
%   W_orig = W * diag(1/s) * U'
W_orig = W * diag(1 ./ s_diag) * U';   % [n_components × nUnits]

%% --- Step 5: correlate ICs with SNF_PH and ETH_thr --------------------------
fprintf('  Computing correlations...\n');

% Valid mask: only samples where SNF_PH is a real phase value
valid_ph  = (D.SNF_PH >= 0);
valid_eth = true(size(D.ETH_thr));   % ETH_thr defined everywhere

corr_SNF = zeros(n_components, 1);
corr_ETH = zeros(n_components, 1);

snf_ref = D.SNF_PH(valid_ph);
eth_ref = D.ETH_thr(valid_eth);

for ic = 1:n_components
    s_ph = S_ica(ic, valid_ph);
    r_snf = corrcoef(snf_ref(:), s_ph(:));
    corr_SNF(ic) = r_snf(1,2);

    s_eth = S_ica(ic, valid_eth);
    r_eth = corrcoef(eth_ref(:), s_eth(:));
    corr_ETH(ic) = r_eth(1,2);
end

%% --- Step 6: rank ICs (1 = highest |correlation|) ---------------------------
[~, ord_snf] = sort(abs(corr_SNF), 'descend');
[~, ord_eth] = sort(abs(corr_ETH), 'descend');

rank_SNF = zeros(n_components, 1);
rank_ETH = zeros(n_components, 1);
for r = 1:n_components
    rank_SNF(ord_snf(r)) = r;
    rank_ETH(ord_eth(r)) = r;
end

%% --- Store results -----------------------------------------------------------
D.ICA.S            = S_ica;
D.ICA.W            = W_orig;
D.ICA.corr_SNF     = corr_SNF;
D.ICA.corr_ETH     = corr_ETH;
D.ICA.rank_SNF     = rank_SNF;
D.ICA.rank_ETH     = rank_ETH;
D.ICA.n_components = n_components;
D.ICA.LV_Fs        = LV_Fs;

fprintf('  Top IC for SNF_PH : IC %d  (r = %.3f)\n', ord_snf(1), corr_SNF(ord_snf(1)));
fprintf('  Top IC for ETH_thr: IC %d  (r = %.3f)\n', ord_eth(1), corr_ETH(ord_eth(1)));
fprintf('=== ICA complete ===\n\n');

end

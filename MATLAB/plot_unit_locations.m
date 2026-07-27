function plot_unit_locations(D)
%% PLOT_UNIT_LOCATIONS  Visualize spiking neuron locations on the NP probe array
%
%   plot_unit_locations(D)
%
%   Creates a figure showing where sorted units are located along the
%   Neuropixels probe.  Three panels are displayed:
%
%     LEFT   — Probe map: each unit as a colored dot at its depth on the
%               probe.  Color = firing rate (spikes/s).  Dot size = mean
%               spike amplitude.  Gray dots show all electrode sites.
%               For multi-shank probes, shanks are shown side-by-side.
%
%     MIDDLE — Firing-rate histogram: number of units in each 50-µm
%               depth bin along the probe.
%
%     RIGHT  — Firing-rate distribution: histogram of firing rates
%               across all units.
%
%   INPUT:
%     D   struct returned by load_experiment_data()
%         Required fields: unitDepths, unitFiringRate, unitAmps,
%                          xcoords, ycoords, spikeTimes, sp
%
%   The figure title shows the recording duration and unit count.

%% ---- Configuration ----------------------------------------
DEPTH_BIN_UM  = 50;   % µm per depth bin for histogram
MIN_FR        = 0.1;  % units below this firing rate (sp/s) are shown in gray
%% -----------------------------------------------------------

if isempty(D.unitIDs)
    warning('OR:noUnits', 'No units found in D. Cannot plot unit locations.');
    return;
end

recordingDur = D.sp.st(end);
nUnits       = numel(D.unitIDs);

%% Identify shanks from channel x-coordinates
% NP 2.0 probes have 4 shanks separated by ~250 µm; single shank x ~ 0
shankBoundaries = [-Inf, -100, 100, 400, Inf];   % x boundaries in µm
shankLabels     = 0:3;
nShanks = 4;

% Determine which shank each unit lives on (via peak-channel x-coord)
% Re-derive peak channel for each unit from templates
tempsMaxAbs = squeeze(max(abs(D.sp.temps), [], 2));  % [nTemplates × nChannels]
nTemplates  = size(D.sp.temps, 1);
peakChan    = zeros(nTemplates, 1);
for t = 1:nTemplates
    [~, peakChan(t)] = max(tempsMaxAbs(t, :));
end
% Map unit IDs to template peak channel
% (cluster IDs are 0-indexed from Kilosort)
unitPeakChan = zeros(nUnits, 1);
for u = 1:nUnits
    uid = D.unitIDs(u);
    if uid + 1 <= nTemplates
        unitPeakChan(u) = peakChan(uid + 1);
    end
end
unitXcoord = D.xcoords(max(unitPeakChan, 1));  % x position of peak channel (µm)

% Determine shank index for each unit
unitShank = zeros(nUnits, 1);
for s = 1:nShanks
    inShank = (unitXcoord >= shankBoundaries(s)) & (unitXcoord < shankBoundaries(s+1));
    unitShank(inShank) = shankLabels(s);
end
shanksPresent = unique(unitShank);
nShanksPresent = numel(shanksPresent);

%% Normalize visual properties
unitFR      = D.unitFiringRate;
unitDepth   = D.unitDepths;
unitAmp     = D.unitAmps;

% Dot size: scale amplitude to marker area, default if amps unavailable
if all(isnan(unitAmp)) || isempty(unitAmp)
    dotSize = 60 * ones(nUnits, 1);
else
    ampNorm = (unitAmp - min(unitAmp)) / (max(unitAmp) - min(unitAmp) + eps);
    dotSize = 20 + ampNorm * 120;   % 20–140 pt²
end

% Color by log10 firing rate for better dynamic range
logFR     = log10(max(unitFR, 0.001));
logFR_lo  = log10(MIN_FR);
logFR_hi  = max(logFR) + 0.1;

%% Depth bins for histogram
depthRange = [0, max(D.ycoords) + DEPTH_BIN_UM];
depthEdges = depthRange(1):DEPTH_BIN_UM:depthRange(2);
depthCenters = depthEdges(1:end-1) + DEPTH_BIN_UM/2;

%% Build figure
fig = figure('Name', 'NP Unit Locations', 'Color', 'w', ...
    'Position', [50, 50, 280 + 200*nShanksPresent, 700]);

%% ----- LEFT PANELS: Probe maps (one per shank present) -----
for si = 1:nShanksPresent
    sh = shanksPresent(si);

    ax = subplot(1, nShanksPresent + 2, si);
    hold(ax, 'on');

    % Background: all electrode sites on this shank
    chanOnShank = (D.xcoords >= shankBoundaries(sh+1)) & ...
                  (D.xcoords <  shankBoundaries(sh+2));
    scatter(ax, zeros(sum(chanOnShank),1), D.ycoords(chanOnShank), ...
        8, [0.85 0.85 0.85], 'filled', 'MarkerFaceAlpha', 0.7);

    % Units on this shank
    onShank = (unitShank == sh);
    if any(onShank)
        sc = scatter(ax, zeros(sum(onShank),1) + randn(sum(onShank),1)*0.1, ...
            unitDepth(onShank), dotSize(onShank), logFR(onShank), 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        clim(ax, [logFR_lo, logFR_hi]);
    end

    % Colormap: cool-to-warm (blue = low FR, red = high FR)
    colormap(ax, hot(256));

    % Labels
    ylabel(ax, 'Depth from probe tip (µm)');
    xlabel(ax, '');
    set(ax, 'XTick', [], 'YDir', 'normal', 'Box', 'off', 'FontSize', 10);
    if nShanksPresent > 1
        title(ax, sprintf('Shank %d\n(%d units)', sh, sum(onShank)), 'FontSize', 10);
    else
        title(ax, sprintf('%d units', sum(onShank)), 'FontSize', 10);
    end
    ylim(ax, depthRange);
    xlim(ax, [-0.5, 0.5]);
end

% Shared colorbar for probe panels
cbAx = subplot(1, nShanksPresent + 2, 1);  % attach to first probe panel
cb = colorbar(cbAx, 'eastoutside');
set(cb, 'Ticks', log10([0.1, 1, 5, 10, 50, 100]), ...
        'TickLabels', {'0.1','1','5','10','50','100'});
cb.Label.String = 'Firing rate (sp/s)';
cb.Label.FontSize = 10;

%% ----- MIDDLE PANEL: Depth histogram -----
axH = subplot(1, nShanksPresent + 2, nShanksPresent + 1);
unitCounts = histcounts(unitDepth, depthEdges);
barh(axH, depthCenters, unitCounts, 1, ...
    'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'none');
xlabel(axH, 'Units per bin');
ylabel(axH, 'Depth (µm)');
title(axH, sprintf('Units / %d µm', DEPTH_BIN_UM), 'FontSize', 10);
set(axH, 'YDir', 'normal', 'Box', 'off', 'FontSize', 10);
ylim(axH, depthRange);
grid(axH, 'on');

%% ----- RIGHT PANEL: Firing rate distribution -----
axFR = subplot(1, nShanksPresent + 2, nShanksPresent + 2);
histogram(axFR, unitFR, 'BinWidth', 1, 'FaceColor', [0.2 0.6 0.4], 'EdgeColor', 'none');
xlabel(axFR, 'Firing rate (sp/s)');
ylabel(axFR, 'Number of units');
title(axFR, 'FR distribution', 'FontSize', 10);
set(axFR, 'Box', 'off', 'FontSize', 10);
grid(axFR, 'on');

% Annotate median FR
medFR = median(unitFR);
hold(axFR, 'on');
yLimFR = ylim(axFR);
plot(axFR, [medFR medFR], yLimFR, 'r--', 'LineWidth', 1.5);
text(axFR, medFR, yLimFR(2)*0.95, sprintf(' med=%.1f', medFR), ...
    'Color','r', 'FontSize', 9, 'VerticalAlignment','top');

%% Figure-level title
sgtitle(fig, sprintf('Unit Locations on NP Probe\n%d units  |  recording duration: %.0f s  |  NP sampling rate: %.0f Hz', ...
    nUnits, recordingDur, D.NP_Fs), 'FontSize', 12, 'FontWeight', 'bold');

end  % plot_unit_locations

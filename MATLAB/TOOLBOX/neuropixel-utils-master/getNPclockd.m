function NPtime = getNPclock(LVindices, Aligned, TR, TS); 
%%%%%%%%%
% Gets NP time from LV relative indices:
    % INTPUT : 
        % LVindices : nx1 array with LV global index relative number(Does not have to be an exact integer index) for which inferred NP glbal clock timestamp in ms is desired
        % Aligned : nx2 array whith indices corresponding to global indices, col 1: to LV ms clock paired with col 2: NP global time
            % Outside of the first and last pulse within each session aligning timestamps, there are no interpolated times
            % and values are set to NaNs as 'Alignment.m' does not extrpolate values for times outside of these windows since
            % for the most part it will not be used for the analysis. Since the first and last sniff will probably have onset
            % and offset outside of the window, this function allows for the extrpolation of timestamps outside of the
            % paired pulse window.
        % TS: ms clock from Labview within trial corresponding to global indexing
        % TR : trial number corresponding to each global index from LabView
    % [1]: Retrieves LV TS timestamp (or interpolates TS timestamp using paired Labview [ind TS(idx)] array if needed)
    % [2]: Function asks for uigetfile user input for the file location of synchronized timestamps (function needs to reference 'Alignment.m' output file- synchronized timestamps)
    % [2]: If value within the array range, interp1 NP global time; else if value exceeds the array range, extrap global
    % time. IF extrapolating, will notify with warning message
% look for indices within each trial so want interpolate since trial clock starts over at the beginning on each trial
 
 % Output : NPtime: col1-LVindex called in the fxn, col2: corresponding globalNP clock time
%%%%%%%%%

NPtime = [LVindices(:) zeros(length(LVindices), 1)]; % Create structure to hold all interpolated NP clock times
% Loop through each trial and if indices are requested within that trial, interpolate values
trialSeek= TR(floor(LVindices)); % Find which trial to which the indices correspond to since interpolating is down within trial
% last = ceil(max(unique(TR))/2); % Find the last trial of the session to see where to finish loop, Don't worry if incomplete trial, won't be indices sought for on these loops anyways
for trial = unique(trialSeek) % Loop through trials that have indices which need interp/extrap
    seek = find(trialSeek == trial); % Find all indices that correspond to looping trial
    if ~isempty(seek)
               tempTR = find(TR==trial); % All global LV indices associated with looping trial
               LVts = interp1(tempTR, TS(tempTR), LVindices(seek)); % Interpolate the local (within trial) time in ms for the cont. LVindices
               alignTemp = [Aligned(tempTR, 1), Aligned(tempTR, 2)]; alignTemp(any(isnan(alignTemp), 2), :) = []; % Grab all aligned timestamps and drop anywith Nans in row (timestamps before or after last pulse of trial since can't use to Align)
               time = interp1(alignTemp(:,1), alignTemp(:,2), LVts); % If sought indices are outside the first and last pulse aligned window, function will return NaNs, since interp1 will return NaNs
    if any(isnan(time)) % If any timestamps attempted to exptrapolate, extrapolate those sought indices instead
                time(isnan(time)) = interp1(alignTemp(:,1), alignTemp(:,2), LVts(isnan(time)),'linear','extrap'); % Replace NaNs with extrapolated value
    end
    NPtime(seek, 2) = time;
    end
end
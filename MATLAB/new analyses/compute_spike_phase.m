function D = compute_spike_phase(D)
%% COMPUTE_SPIKE_PHASE  Map each spike to a sniff cycle phase
%  Requires D.SNF_PH, D.sniff_onsets (from compute_sniff_phase).
%  Approximate linear LV<->NP time mapping: lv_idx = round(t_s * LV_Fs)+1
%  For precision use TimeSync alignment (AlignStamps.m).
%  Adds: D.spike_SNF_PH, D.unitMeanSniffPhase
LV_Fs=D.LV_Fs; nLV=numel(D.SNF);
lv_idx=round(D.spikeTimes*LV_Fs)+1; lv_idx=max(1,min(lv_idx,nLV));
D.spike_SNF_PH=D.SNF_PH(lv_idx);
nU=numel(D.unitIDs); D.unitMeanSniffPhase=nan(nU,1);
for u=1:nU
    ph=D.spike_SNF_PH(D.sp.clu==D.unitIDs(u)); v=ph(ph>=0);
    if numel(v)>=5; D.unitMeanSniffPhase(u)=mean(v); end
end
fprintf('    Spike phase: %d/%d spikes in valid cycles\n',sum(D.spike_SNF_PH>=0),numel(D.spikeTimes));
end

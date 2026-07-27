# Ethanol vs Control Phase-Locking of Olfactory-Bulb Units to the Sniff (SNF) Signal

## Question

Does ethanol exposure alter the phase-locking of olfactory-bulb single-unit spikes to the sniff (SNF sensor) signal phase, relative to control trials?

## Methods

Data were loaded for 6 recording sessions (one animal per session). For each session, sniff segments were detected from the raw SNF (sniff-sensor) signal — never the LFP — using the validated `compute_sniff_phase` routine with GUI-default, pinned detection thresholds. SNF sections flagged as noise or non-sniffing were rejected; only spikes falling within valid detected sniff segments were retained. Instantaneous sniff phase was computed from the analytic signal of the valid SNF trace, and each retained spike was assigned the SNF phase at its spike time via the validated `compute_spike_phase` routine.

For each unit, the mean resultant length (MRL) and preferred phase of spike-SNF phase-locking were computed over valid-sniff spikes, separately for the ethanol and control conditions. Units with a firing rate below 0.1 Hz or fewer than 50 valid-sniff spikes in a given condition were excluded from that condition. Per-unit MRLs were then averaged within animal to give an animal-level mean MRL for each condition. The independent unit of statistical inference was the animal (not the unit), consistent with the pre-registered experimental structure; trials and units were treated as nested within animal to avoid pseudoreplication.

The ethanol vs control comparison of animal-level mean MRL was tested with a paired, exact two-sided Wilcoxon signed-rank test across animals, with the rank-biserial correlation reported as the paired effect size. Per-unit preferred phases and MRLs are reported descriptively (RESULT-mrl, RESULT-phase) and were not subjected to a separate confirmatory test beyond the pre-registered animal-level paired comparison.

**Exclusions.** Of 6 sessions loaded, the session recorded on 2021-11-03 was excluded from the paired animal-level statistical comparison: the ethanol-detection threshold (ETH_thr) exceeded 0.11 throughout that entire session, so no units met the inclusion threshold in the control condition for that animal. This left 5 animals (5 sessions) in the paired statistical comparison. Across all 6 loaded sessions, 1603 units met the inclusion criteria in the ethanol condition and 644 units met the inclusion criteria in the control condition. Per-session sniff, neuron, and trial counts, and the full log of discarded SNF sections (with reasons), are reported in the separate QC document (FIG-DEMO1-QC; RESULT-qc-doc), not reproduced here.

## Results

Across the 5 animals included in the paired comparison, animal-level mean MRL to the SNF sniff signal was lower under ethanol than under control in every paired animal (exact two-sided Wilcoxon signed-rank: W = 0.0, p = 0.0625, rank-biserial r = -1.00, n = 5 animals; Fig. 1). With n = 5 paired animals, p = 0.0625 is the smallest exact two-sided p-value attainable from this test and does not cross the conventional threshold; the result should be read as a consistent, maximal-effect-size directional pattern (MRL lower under ethanol in 5 of 5 animals) that is not statistically significant at conventional thresholds given the available sample size.

Per-unit MRL and preferred phase by condition, for all units meeting inclusion criteria, are reported in RESULT-mrl and RESULT-phase respectively.

## Figure

**Fig. 1 — FIG-DEMO1-01.** Ethanol vs control phase-locking of OB units to the sniff (SNF) signal. Panel A: per-unit scatter of preferred SNF phase (rad) against MRL; open circles = control (n = 644 units), filled circles = ethanol (n = 1603 units). Panel B: paired animal-level mean MRL, control vs ethanol, one line per animal; all 5 lines decrease from control to ethanol. Exact two-sided Wilcoxon: p = 0.0625, rank-biserial r = -1.00, n = 5 animals.

## Limitations

- The paired animal-level statistical comparison is based on n = 5 animals. With n = 5, the exact two-sided Wilcoxon signed-rank test cannot reach p < 0.05 even when every paired difference is in the same direction; p = 0.0625 should not be interpreted as evidence of no effect. The effect-size estimate (rank-biserial r = -1.00) is maximal and consistent in direction across all 5 animals, but the sample size does not permit a confirmatory significance decision at conventional thresholds.
- One session (2021-11-03) was excluded from the paired comparison because no units met the control-condition inclusion threshold (ETH_thr > 0.11 throughout the session). This reduces the paired sample from 6 sessions loaded to 5 animals analyzed.
- Population spike-rate tuning curves (binned phase histograms) were not computed as part of this analysis. The per-unit results available (RESULT-mrl, RESULT-phase) report mean resultant length and preferred phase per unit; any binned-curve analysis is deferred to future work.
- Per-unit phase-locking values reported here are descriptive only. A circular test for preferred-phase shift between ethanol and control conditions was specified in the analysis contract but was not performed in this run; preferred phase is stored in RESULT-phase for future use, and the circular test is deferred to subsequent work. No per-unit hypothesis tests were conducted, so Benjamini-Hochberg FDR correction was not applied.

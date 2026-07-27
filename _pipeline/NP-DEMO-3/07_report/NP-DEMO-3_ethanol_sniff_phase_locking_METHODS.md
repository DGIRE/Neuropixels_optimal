# {{CONTRACT-v001.task_id}}: Ethanol and Sniff Phase-Locking — Methods & QC Report

**Task ID:** {{CONTRACT-v001.task_id}}
**Contract:** `02_contract/contract_v001.yaml` (status: confirmatory)
**Intended audience:** lab + QC reviewers
**Companion document:** see the separate Results report for statistical outcomes. This document covers signal processing, inclusion/exclusion, and QC only.

---

## Question

What signal was recorded, what was measured, and what was asked:

Olfactory-bulb single-unit spike times were recorded with Neuropixels probes across {{CONTRACT-v001.n_sessions}} sessions in one animal, together with a raw sniff-sensor signal (SNF, **not** the LFP) and a raw ethanol-trace signal (ETH), both sampled at the LabView acquisition rate (`LV_Fs`, approximately {{CONTRACT-v001.lv_fs}} Hz) on the same clock as the spike times. For each recording, sniff cycles and instantaneous sniff phase were detected from SNF, ethanol-contact periods were detected from ETH, and each spike was assigned the sniff phase and ethanol/control condition label of the moment it occurred. This report documents the detection and inclusion pipeline that produced the per-unit and per-experiment phase-locking (mean resultant length, MRL) quantities used for the confirmatory statistical tests reported separately.

---

## Methods

**Sniff-phase detection (SNF).** Sniff onsets and instantaneous sniff phase were computed from the raw SNF signal via `compute_sniff_phase(D, threshold_std={{CONTRACT-v001.threshold_std}})`. This threshold is a validated, pinned kernel default and was not altered for this analysis (per contract assumption; changing it requires human sign-off).

**Ethanol-contact detection (ETH).** Ethanol-contact periods were detected via the new `detect_eth_contact(D, eth_threshold={{RESULT-ethmask-nov1.eth_threshold}})` function: the mean of the **entire** ETH time series for the experiment was subtracted first (no windowing or per-epoch subtraction), and the mean-subtracted trace was then thresholded -- samples strictly above threshold were labeled "ethanol present," samples at or below threshold were labeled "control." `n_trials` for each experiment is the count of discrete contiguous runs of mean-subtracted-ETH samples strictly above threshold. This is an identical, fixed rule applied to all six sessions with no per-experiment adjustment, and it is a new algorithm distinct from (and not a reuse of) the existing `threshold_eth.py` kernel (which uses a different floor-clip rule).

**Spike-phase assignment and valid-sniff gate.** Each spike was assigned its sniff phase via `compute_spike_phase`. A spike was retained for downstream analysis only if its assigned phase field `spike_SNF_PH` was non-negative (i.e., it fell within a valid, detected sniff cycle). Spikes with a negative `spike_SNF_PH` (non-sniffing periods or signal noise) were excluded from all locking statistics; every excluded interval was logged (see Discarded-Sections below), never silently dropped.

**Unit inclusion criteria.** A unit was included only if its overall firing rate was at or above {{CONTRACT-v001.min_firing_rate}} Hz **and** it had at least {{CONTRACT-v001.min_combined_spikes}} valid-sniff spikes combined across the ethanol and control conditions. For the top-five example-unit figure specifically, an additional criterion required more than {{CONTRACT-v001.example_spike_cutoff}} spikes per unit per condition (this stricter criterion applies only to selection of the illustrative examples, not to the population statistics).

**Session list.** {{CONTRACT-v001.n_sessions}} sessions, all from one animal, located under `C:\Projects\Repos\Neuropixels\DATA`: {{RESULT-sniffqc-nov1.experiment}}, {{RESULT-sniffqc-nov3.experiment}}, {{RESULT-sniffqc-dec15.experiment}}, {{RESULT-sniffqc-may17.experiment}}, {{RESULT-sniffqc-jun24.experiment}}, {{RESULT-sniffqc-sep14.experiment}}. The {{RESULT-sniffqc-sep14.experiment}} session is multi-shank and required `_fix_ks_dir()` within `load_experiment_data` to locate its Kilosort output subdirectory one level deeper than the default location. Session {{RESULT-sniffqc-nov3.experiment}} was included in full: `detect_eth_contact` determined from the data that usable control periods exist for this session (per prior open-decision resolution), so it was retained in all downstream statistics.

**MRL (mean resultant length) formula.** For each included unit, and separately for the ethanol condition and the control condition, the mean resultant length of the valid-sniff spike phases was computed as the magnitude of the mean unit vector on the circle:

MRL = the absolute value of the average of exp(i times phase) over all valid spikes in that condition,

where `phase` is the sniff-cycle phase of each valid spike. A condition-agnostic MRL (pooling valid-sniff spikes across both conditions) was also computed per unit, used only to select the globally strongest/weakest phase-locking examples for QC (not one pair per experiment).

No causal claims are made in this document about ethanol's effect on phase-locking; this section describes measurement and processing steps only.

---

## QC-Counts

Per-experiment counts, computed from `RESULT-sniff-qc` (never typed by hand):

| Experiment | n_sniffs | n_neurons | n_trials | Duration (min) | % usable sniff time |
|---|---|---|---|---|---|
| {{RESULT-sniffqc-nov1.experiment}} | {{RESULT-sniffqc-nov1.n_sniffs}} | {{RESULT-sniffqc-nov1.n_neurons}} | {{RESULT-sniffqc-nov1.n_trials}} | {{RESULT-sniffqc-nov1.duration_min}} | {{RESULT-sniffqc-nov1.pct_usable_sniff_time}} |
| {{RESULT-sniffqc-nov3.experiment}} | {{RESULT-sniffqc-nov3.n_sniffs}} | {{RESULT-sniffqc-nov3.n_neurons}} | {{RESULT-sniffqc-nov3.n_trials}} | {{RESULT-sniffqc-nov3.duration_min}} | {{RESULT-sniffqc-nov3.pct_usable_sniff_time}} |
| {{RESULT-sniffqc-dec15.experiment}} | {{RESULT-sniffqc-dec15.n_sniffs}} | {{RESULT-sniffqc-dec15.n_neurons}} | {{RESULT-sniffqc-dec15.n_trials}} | {{RESULT-sniffqc-dec15.duration_min}} | {{RESULT-sniffqc-dec15.pct_usable_sniff_time}} |
| {{RESULT-sniffqc-may17.experiment}} | {{RESULT-sniffqc-may17.n_sniffs}} | {{RESULT-sniffqc-may17.n_neurons}} | {{RESULT-sniffqc-may17.n_trials}} | {{RESULT-sniffqc-may17.duration_min}} | {{RESULT-sniffqc-may17.pct_usable_sniff_time}} |
| {{RESULT-sniffqc-jun24.experiment}} | {{RESULT-sniffqc-jun24.n_sniffs}} | {{RESULT-sniffqc-jun24.n_neurons}} | {{RESULT-sniffqc-jun24.n_trials}} | {{RESULT-sniffqc-jun24.duration_min}} | {{RESULT-sniffqc-jun24.pct_usable_sniff_time}} |
| {{RESULT-sniffqc-sep14.experiment}} | {{RESULT-sniffqc-sep14.n_sniffs}} | {{RESULT-sniffqc-sep14.n_neurons}} | {{RESULT-sniffqc-sep14.n_trials}} | {{RESULT-sniffqc-sep14.duration_min}} | {{RESULT-sniffqc-sep14.pct_usable_sniff_time}} |

`n_trials` per experiment (contiguous above-threshold ethanol-contact runs, from `RESULT-eth-mask`) is cross-checked against the table above and agrees exactly: {{RESULT-ethmask-nov1.n_trials}}, {{RESULT-ethmask-nov3.n_trials}}, {{RESULT-ethmask-dec15.n_trials}}, {{RESULT-ethmask-may17.n_trials}}, {{RESULT-ethmask-jun24.n_trials}}, {{RESULT-ethmask-sep14.n_trials}}.

---

## Discarded-Sections

Every discarded SNF section (non-sniffing periods and short/noise segments) was logged with its experiment, start time, end time, and reason -- no silent discards. Summing all logged intervals in `RESULT-discard-log` across the six experiments gives:

**n_intervals total (discarded SNF sections) = {{RESULT-discard-log.count}} [renderer: len(RESULT-discard-log)]** (counted directly from the full `RESULT-discard-log` table).

Reasoning: each discarded interval is tagged with one of two reasons -- `non-sniffing` (periods with no detected sniff cycle, e.g., long gaps or grooming) or `noise/short` (very brief or artifactual candidate intervals below the reliable-detection floor of the sniff-phase kernel). Discarded intervals occur throughout all six sessions and account for the complement of the `pct_usable_sniff_time` values reported in QC-Counts above -- e.g., the comparatively low usable-time fraction for {{RESULT-sniffqc-dec15.experiment}} ({{RESULT-sniffqc-dec15.pct_usable_sniff_time}}%) reflects a session with a larger share of short/noise-flagged intervals than the other five sessions.

The full, unabridged discarded-section log (experiment, start_s, end_s, reason for all {{RESULT-discard-log.count}} [renderer: len(RESULT-discard-log)] intervals) is provided in the Appendix and is not reproduced in full here.

---

## Figures

- **FIG-DEMO3-QC-SNIFF** -- "Methods: per-experiment sniffing and ethanol-contact examples." One two-panel block per experiment (SNF_z top panel with the sniff-onset threshold_std red dashed line; mean-subtracted ETH bottom panel with the ethanol-contact threshold red dashed line), on a shared time axis, captioned with n_sniffs, n_neurons, n_trials, duration_min, and pct_usable_sniff_time per experiment (from `RESULT-sniff-qc`). Source results: `RESULT-eth-mask`, `RESULT-sniff-qc`, `RESULT-discard-log`.
- **FIG-DEMO3-QC-PSTH** -- "Methods: sniff-locked PSTH examples (strongest and weakest phase-locking units)." Sniff-locked PSTHs for the single globally strongest and single globally weakest condition-agnostic-MRL unit (selected across all experiments combined, not one pair per experiment), each labeled with its experiment/session date and `MRL_cond_agnostic` from `RESULT-mrl-per-unit`. Source results: `RESULT-mrl-per-unit`.

---

## Limitations

- The animal-level paired Wilcoxon test (reported in the companion Results document) has an experiment count that is statistically underpowered to detect anything but a very large, highly consistent effect; a non-significant result at this sample size does not establish absence of an effect.
- Sniff-phase detection is threshold-based (`compute_sniff_phase`, pinned kernel default); it is not a model-based or adaptive detector, and its false-positive/false-negative sniff-onset calls are bounded by this single fixed threshold across all sessions and animals.
- Ethanol-contact detection is also threshold-based (`detect_eth_contact`, entire-trace mean-subtraction, fixed threshold, identical for all six sessions); it does not adapt to session-specific baseline drift beyond the single mean-subtraction step, and the same fixed rule is applied regardless of any per-session differences in ETH sensor coupling or noise floor.
- No MATLAB reference implementation exists for `detect_eth_contact`; its fixture was established from the first validated implementation run after code review and manual inspection, rather than from an independent reference pipeline.

---

## Appendix

Full discarded-section log (experiment, start_s, end_s, reason) -- see `RESULT-discard-log` ({{RESULT-discard-log.count}} [renderer: len(RESULT-discard-log)] entries total, one row per discarded interval, per experiment).

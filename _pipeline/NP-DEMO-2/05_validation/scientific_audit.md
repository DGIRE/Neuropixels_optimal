# NP-DEMO-2 Scientific Audit
## Summary: FAIL (recoverable only by a human/contract-designer decision on DEV-001)

The code faithfully and byte-accurately implements the pinned contract primitives
(PROH-001 respected, kernel numerics untouched, numeric gate 14/14 pass). However the
contract PROH-002 SD-based flag rule fails to catch the degenerate all-ethanol session
2021-11-03, and that session degenerate control-condition MRLs (374/429 units with zero
control spikes; 42 units with single/few-spike MRL_control = 1.0) then (a) completely
dominate the RESULT-delta-mrl top-5 / RESULT-psth-examples / RESULT-unit-locations
examples figure -- all 5 example units are 2021-11-03 artifacts -- and (b) enter both
the animal-level and unit-level statistics through an UNPAIRED per-condition set. The
reported conclusion (control > ethanol phase-locking) is therefore a spuriously
inflated effect that is an artifact of degenerate data. This is a known, logged
deviation (DEV-001, blocks_certification: true, requires_human_decision: true) -- but
as it stands the results objects are scientifically corrupted and cannot be certified.

Fresh auditor note: I inherited none of the builder rationale. The code is
well-engineered and honest about its deviation; the FAIL is a data-integrity /
scientific-validity verdict on the RESULTS, not an accusation of a coding error.

## Findings

### BLOCKER findings

**BLOCKER-1 -- Examples figure (RESULT-psth-examples / RESULT-delta-mrl top-5 /
RESULT-unit-locations) is 100 percent degenerate-data artifact.**
Contract clauses: aggregation (delta_MRL ranking), required_outputs RESULT-delta-mrl /
RESULT-psth-examples / RESULT-unit-locations; figure FIG-DEMO2-RES-EX scientific_purpose
(units exhibiting the largest CHANGE in phase-locking between ethanol and control).
All five top-|delta_MRL| units (104, 312, 253, 626, 69) are from 2021-11-03, each with
delta_MRL ~ -0.999. These deltas are MRL_ethanol(~0) - MRL_control(~1.0) where
MRL_control ~ 1.0 is a MEANINGLESS single/few-spike circular estimate (a handful of
control spikes trivially yield MRL~1). RESULT-psth-examples confirms it: every one of
the 5 example units has a psth_ctrl that is all-zero except a single 0.011 bin
(essentially NO control spikes across ~9087 sniff events). The largest-change figure --
the centrepiece of the Results document -- therefore shows five units whose apparent
change is a divide-by-almost-nothing artifact, not a biological ethanol effect. The
ranking is computed correctly per the contract formula; the BLOCKER is that the input
MRL_control values feeding it are degenerate. This invalidates FIG-DEMO2-RES-EX and its
caption claims.

**BLOCKER-2 -- Animal-level and unit-level statistics use UNPAIRED per-condition unit
sets, contaminated by degenerate MRLs.**
Contract clauses: statistical_model (animal-level paired on paired (MRL_ethanol -
MRL_control) per animal; unit-level paired per-unit (ethanol, control) MRL observations);
aggregation (per-animal mean MRL by condition, across that animal included units).
- ANIMAL LEVEL: run_full_analysis builds each animal mean_mrl_ethanol from the
  finite-ethanol units and mean_mrl_control from the finite-CONTROL units INDEPENDENTLY
  (analysis lines 785-786: two separate comprehensions each with its own math.isfinite
  filter). For 2021-11-03 the ethanol mean is over 429 units but the control mean is
  over only 55 units (42 of them MRL=1.0 single-spike artifacts), giving
  mean_mrl_control = 0.9138. This is not a paired within-unit contrast; the two
  condition means average over DIFFERENT unit subsets. Affects EVERY session (e.g.
  2022-06-24: 287 eth vs 212 ctrl units). The paired-per-animal test is only nominal.
  For 2021-11-03 the 0.9138 control mean is an artifact of the 55-unit degenerate
  subset, driving the all-negative diff vector and p=0.03125 / r=-1.0.
- UNIT LEVEL (LMM): only both-finite units enter (n_units=1138 confirmed; 465
  NaN-control units dropped). But 2021-11-03 still contributes 55 rows, 42 with
  MRL_control=1.0. Across all sessions 83 LMM rows carry MRL_control=1.0. These inflate
  the control mean and thus coef=-0.2367, standardized_effect_size=-1.026, p=7.5e-220.
  The LMM correctly drops NaN MRLs and uses animal/session as random intercept
  (pseudoreplication handled), but is contaminated by single-spike rows.

Why BLOCKER not MAJOR: contract status is confirmatory and the Results document sets
report_effect_sizes: true / report_exact_p: true. A confirmatory effect size and exact
p driven by degenerate single-spike circular estimates from a session the analysis
itself flags as all-ethanol is a wrong result that would mislead the reader about the
strength (and arguably the existence) of the ethanol effect.

### MAJOR findings

**MAJOR-1 -- DEV-001: PROH-002 SD flag metric cannot detect a degenerate session.**
Contract clause: inclusion_exclusion / preprocessing (Pass-1 flag = contact count
outside +/-1 SD of the mean OR its distribution is degenerate such as all-ethanol / no
control epochs, e.g. 2021-11-03). The implemented flag only tests the SD arm
(abs(pass1_count - mean) > sd, line 294). The OR-degenerate/all-ethanol arm is NOT
wired as a flag trigger -- all_ethanol_at_all_thresholds runs only AFTER flagging
(lines 632-633) as an EXCLUSION test, and it too misses 2021-11-03 because that session
dips below grid-max 0.50 in its inter-contact gaps (min(ETH) < 0.50 -> False). Result:
2021-11-03 (3 contact-events but ETH above 0.11 for ~100 percent of time, 374/429 units
with zero control spikes) is NOT flagged, adjusted, or excluded, and enters all
downstream objects with degenerate control epochs. Root cause correctly diagnosed in
deviations.yaml: 2021-11-01 large contact count inflates SD to 672, so the +/-1 SD
window [-301, 1044] swallows 2021-11-03 count of 3. contact_count counts contiguous
ETH>thr RUNS (events), blind to the FRACTION of time in ethanol -- the true degeneracy
signal. A genuine contract-design flaw, faithfully implemented; the builder correctly
refused to silently patch it. MAJOR because it is the upstream cause already captured
by BLOCKER-1/-2 and is logged for human decision.

**MAJOR-2 -- 2021-11-03 default/final threshold both 0.11 (was_adjusted: false).**
Contract clause: inclusion_exclusion OD-3 (2021-11-03 is flagged as all-ethanol at the
default and threshold-adjusted in PASS 2 to create control epochs). OD-3 ASSUMED
2021-11-03 would be flagged and its threshold raised. Because MAJOR-1 flag misses it,
the rescue never happens: RESULT-eth-threshold-log shows final_threshold = 0.11 =
default, was_adjusted = false. The contract pinned expectation for this session is
unmet; it is included AS-IS with degenerate epochs (resolution-option C, which requires
explicit human sign-off, not yet given).

### MINOR findings

**MINOR-1 -- RESULT-stat-unit omits per-session degenerate-unit provenance.** stat-unit
reports n_units=1138 but does not disclose that 83 rows (42 from 2021-11-03) carry
degenerate MRL_control=1.0. Non-blocking (diagnostic exists in RESULT-deviations.yaml),
but the Results document must not present the effect size without that caveat.

**MINOR-2 -- Animal-level n_units_included (429 for 2021-11-03) overstates the
control-condition n.** RESULT-mrl-animal reports the full included-unit count, but the
control mean is over only 55 units. The caption requirement (state n_units per
condition, computed) must distinguish these.

## Axis-by-axis findings

1. PROH-001 (phase from SNF, never LFP): PASS. Phase sourced only from
   compute_sniff_phase SNF_PH and compute_spike_phase spike_SNF_PH, both keyed off
   D SNF. See axis 14.
2. PROH-002: two-pass ordering PASS (Pass-1 all-at-default, mean/SD frozen before
   Pass-2, lines 288-303, HAZ-04). Grid PASS (91 pts 0.05..0.50 step 0.005, incl 0.11).
   Tie-break PASS (nearest 0.11). SD flag rule / DEV-001: FAIL (MAJOR-1) -- OR-degenerate
   arm not wired; 2021-11-03 count 3 vs 371.3 +/- 672.1 -> within [-301,1044] -> not
   flagged. Diagnosis in deviations.yaml correct. Inclusion DOES compromise validity
   (BLOCKER-1/-2).
3. -1 sentinel handling: PASS. valid_sniff_mask = spike_SNF_PH >= 0.0 applied to
   eth_mask/ctrl_mask (lines 542-543); no -1 reaches mrl_and_preferred_phase.
4. valid_sniff_mask application: PASS. eth_mask = is_ethanol AND valid_mask; ctrl_mask =
   NOT is_ethanol AND valid_mask (lines 542-543) -- condition AND valid-sniff.
5. Condition boundary > eth_threshold (strict): PASS (line 541).
6. Unit inclusion combined count: PASS. n_valid_combined = eth+ctrl, < 50 excluded
   (lines 556-559).
7. Animal Wilcoxon: PARTIAL. exact/two-sided/zero_method wilcox PASS (line 159);
   rank-biserial r=-1.0 correct and agrees; n_animals=6 PASS. Inputs are unpaired
   per-condition means (BLOCKER-2).
8. Unit LMM: PARTIAL. animal_id random intercept PASS (line 348); two rows/unit PASS
   (811-814); standardized = coef/std(mrl) ddof=1 PASS (354). NaN units dropped
   (n_units=1138), but 83 MRL_control=1.0 rows retained -> BLOCKER-2.
9. PSTH extremes (QC): PASS. pooled_mrl condition-agnostic (line 398); single strongest
   278 at 2022-06-24 / weakest 82 at 2021-12-15 across ALL sessions.
10. delta_MRL ranking: PASS mechanically (signed delta line 383, abs-delta desc line
    385); top-5 are 2021-11-03 units = DEV-001 contamination (BLOCKER-1).
11. QC counts: PASS. length_min=session_dur_s/60 (len(SNF_PH)/LV_Fs); pct_usable=
    (sum(SNF_PH>=0)/len)*100; n_trials=contact_count runs at final thr; n_neurons=
    len(included). All match RESULT-qc-counts.
12. Discard log: PASS. Every SNF_PH<0 run logged with start_s/end_s=idx/LV_Fs, reason
    outside_valid_sniff (lines 216-240); edge runs handled.
13. Clock alignment: PASS. matlab_round(spike_times * LV_Fs) everywhere (127, 539);
    NP_Fs never used for ETH/SNF indexing.
14. No LFP in phase code: PASS. No D LFP / LFP_Fs reference in the analysis.
15. Result-object completeness: PASS (presence) / BLOCKER (validity). All 12 objects
    present, non-empty: mrl-unit (1603), phase-unit, stat-unit, mrl-animal (6),
    stat-animal, qc-counts (6), qc-discards, eth-threshold-log (6), delta-mrl (1603),
    psth-extremes (2), psth-examples (5), unit-locations (5). Validity of delta-mrl /
    psth-examples / unit-locations / stat-* compromised per BLOCKERs.

## Conclusion

The implementation is a faithful, non-improvised, oracle-matched caller of the validated
kernel: PROH-001 honored, -1 sentinel excluded everywhere, PROH-002 two-pass ordering /
grid / tie-break, condition boundary, unit-inclusion, Wilcoxon parameters, LMM grouping,
pooled-MRL extremes, QC counts and discard log all match the contract exactly, numeric
gate 14/14. No detectable coding bug against the letter of the contract.

Nevertheless the AUDIT FAILS on scientific validity of the RESULTS. The contract own
PROH-002 SD-flag metric (contiguous-run contact COUNT) is blind to a
fraction-of-time-in-ethanol degeneracy, so 2021-11-03 (~100 percent ethanol, 374/429
units with zero control spikes) is neither flagged, adjusted, nor excluded. Its
degenerate single-spike MRL_control~1.0 values (1) monopolize the entire top-5 examples
figure (BLOCKER-1), and (2) contaminate the animal-level paired-mean contrast -- itself
computed on UNPAIRED per-condition unit subsets across all sessions (BLOCKER-2) -- and
the unit-level LMM effect size and p. For a confirmatory analysis reporting exact p and
effect sizes, the headline control > ethanol phase-locking result and its magnitude are
scientifically unreliable as presented.

This is precisely DEV-001 (blocks_certification: true, requires_human_decision: true).
Auditor verdict: the deviation rises to a BLOCKER for CERTIFICATION because it corrupts
result objects a confirmatory Results document must report. Recommended before
certification: DEV-001 option A (re-contract PROH-002 with an explicit degenerate /
fraction-of-time-in-ethanol flag independent of the SD test) or option B (human-approved
documented exclusion of 2021-11-03), then re-run. Independently, animal-level
per-condition means should use the PAIRED within-unit set (finite in BOTH conditions) to
satisfy paired-per-animal (BLOCKER-2), and degenerate few-spike MRL units should be
screened before the examples ranking and the LMM.

VERDICT: FAIL -- do not certify until DEV-001 and the BLOCKER-2 pairing / degenerate-MRL
issues are resolved by human/contract-designer decision.

# {{CONTRACT-v001.task_id}}: Ethanol and Sniff Phase-Locking — Results Report

**Contract:** `02_contract/contract_v001.yaml` (status: confirmatory)
**Intended audience:** lab + demo
**Companion document:** see the separate Methods/QC report (`{{CONTRACT-v001.task_id}}_ethanol_sniff_phase_locking_METHODS.md`) for signal-processing methods, per-experiment QC counts, and the full discarded-section log. Counts and discards are not duplicated here.

---

## Question

Does ethanol contact modulate olfactory-bulb (OB) neuron sniff-phase-locking? Specifically: does the mean resultant length (MRL) of OB single-unit spike phase relative to the sniff cycle differ between ethanol-contact and control periods, at the animal/experiment level and at the individual-unit population level?

---

## Methods-Summary

Sniff phase was detected from the raw sniff sensor (SNF) signal and ethanol-contact periods were detected from the mean-subtracted ethanol trace (ETH), per experiment, using fixed, pinned thresholds applied identically across all sessions; each spike was assigned a sniff phase and a condition label (ethanol vs. control), and only spikes falling within valid, detected sniff cycles were retained. Units were included only if they met a minimum firing-rate and minimum valid-spike-count criterion. Per included unit, MRL was computed separately for the ethanol and control conditions. Two pre-specified, confirmatory tests were then run: an animal-level paired Wilcoxon signed-rank test on each experiment's mean per-unit MRL (ethanol vs. control), and a unit-level linear mixed-effects model (MRL as a function of condition, with a random intercept for animal/experiment). Full signal-processing detail, per-experiment QC counts, and the complete discarded-section log are in the companion Methods/QC report.

---

## Results-Animal-Level

**This is the pre-registered, primary confirmatory test.** A paired, exact, two-sided Wilcoxon signed-rank test compared each experiment's mean per-unit MRL under ethanol versus control ({{RESULT-wilcoxon.n_pairs}} experiments, one paired observation per experiment):

- Test statistic (W): {{RESULT-wilcoxon.statistic}}
- Exact p-value: {{RESULT-wilcoxon.p_value}}
- Matched-pairs rank-biserial r: {{RESULT-wilcoxon.rank_biserial_r}}
- n = {{RESULT-wilcoxon.n_pairs}} experiments

Because the exact p-value ({{RESULT-wilcoxon.p_value}}) is above the conventional significance threshold, **the null hypothesis of no ethanol effect on mean per-unit MRL at the experiment level cannot be rejected.** This is the primary, animal-level result of this analysis.

Direction of the (non-significant) shift: examining `RESULT-mrl-per-experiment`, {{RESULT-mrl-per-experiment.n_eth_gt_ctrl}} of {{RESULT-wilcoxon.n_pairs}} experiments showed a numerically higher mean per-unit MRL under ethanol than under control: [renderer: n_eth_gt_ctrl = count(mean_MRL_ethanol > mean_MRL_control) in RESULT-mrl-per-experiment]

| Experiment | Mean MRL, control | Mean MRL, ethanol |
|---|---|---|
| {{RESULT-mrlexp-nov1.experiment}} | {{RESULT-mrlexp-nov1.mean_MRL_control}} | {{RESULT-mrlexp-nov1.mean_MRL_ethanol}} |
| {{RESULT-mrlexp-nov3.experiment}} | {{RESULT-mrlexp-nov3.mean_MRL_control}} | {{RESULT-mrlexp-nov3.mean_MRL_ethanol}} |
| {{RESULT-mrlexp-dec15.experiment}} | {{RESULT-mrlexp-dec15.mean_MRL_control}} | {{RESULT-mrlexp-dec15.mean_MRL_ethanol}} |
| {{RESULT-mrlexp-may17.experiment}} | {{RESULT-mrlexp-may17.mean_MRL_control}} | {{RESULT-mrlexp-may17.mean_MRL_ethanol}} |
| {{RESULT-mrlexp-jun24.experiment}} | {{RESULT-mrlexp-jun24.mean_MRL_control}} | {{RESULT-mrlexp-jun24.mean_MRL_ethanol}} |
| {{RESULT-mrlexp-sep14.experiment}} | {{RESULT-mrlexp-sep14.mean_MRL_control}} | {{RESULT-mrlexp-sep14.mean_MRL_ethanol}} |

With only {{RESULT-wilcoxon.n_pairs}} paired observations, this consistent numerical direction is suggestive but the test is not powered to confirm it statistically; see Limitations.

---

## Results-Unit-Level

**This is a secondary, exploratory-power (not primary) test.** A linear mixed-effects model was fit to the per-unit MRL values, with condition (ethanol vs. control) as a fixed effect and a random intercept for animal/experiment (formula: `{{RESULT-lme.model_formula}}`):

- n_units = {{RESULT-lme.n_units}}
- n_experiments = {{RESULT-lme.n_experiments}}
- Condition fixed-effect p-value (Wald z, REML): {{RESULT-lme.p_value}}
- Standardized fixed-effect estimate (effect size): {{RESULT-lme.standardized_fixed_effect}}

The condition fixed effect is statistically significant at this unit-level sample size. **This unit-level result is secondary to, and does not override, the non-significant animal-level Wilcoxon result above.** With {{RESULT-lme.n_units}} units nested within only {{RESULT-lme.n_experiments}} experiments, the large unit-level N confers high statistical power to detect even small, consistent shifts in per-unit MRL; a significant condition effect in this model reflects such small, consistent unit-level shifts and should **not** be interpreted as establishing a population-level (animal-level) ethanol effect, nor as causal evidence that ethanol contact changes phase-locking -- the random-intercept term corrects for non-independence of units within the same animal, but the fundamental animal-level replication count (matching the Wilcoxon test above) remains the same regardless of how many units are recorded per animal.

---

## Examples

The five units with the largest absolute change in phase-locking (|MRL_ethanol - MRL_control|) across all included units, drawn only from experiments/units meeting the stricter (greater than {{CONTRACT-v001.example_spike_cutoff}} spikes per unit per condition) inclusion rule used for this illustrative figure, ranked from largest to smallest delta_MRL:

| Unit ID | Experiment | delta_MRL | MRL, ethanol | MRL, control |
|---|---|---|---|---|
| {{RESULT-examples-top5.item1.unit_id}} | {{RESULT-examples-top5.item1.experiment}} | {{RESULT-examples-top5.item1.delta_MRL}} | {{RESULT-mrl-per-unit.item1.MRL_ethanol}} | {{RESULT-mrl-per-unit.item1.MRL_control}} |
| {{RESULT-examples-top5.item2.unit_id}} | {{RESULT-examples-top5.item2.experiment}} | {{RESULT-examples-top5.item2.delta_MRL}} | {{RESULT-mrl-per-unit.item2.MRL_ethanol}} | {{RESULT-mrl-per-unit.item2.MRL_control}} |
| {{RESULT-examples-top5.item3.unit_id}} | {{RESULT-examples-top5.item3.experiment}} | {{RESULT-examples-top5.item3.delta_MRL}} | {{RESULT-mrl-per-unit.item3.MRL_ethanol}} | {{RESULT-mrl-per-unit.item3.MRL_control}} |
| {{RESULT-examples-top5.item4.unit_id}} | {{RESULT-examples-top5.item4.experiment}} | {{RESULT-examples-top5.item4.delta_MRL}} | {{RESULT-mrl-per-unit.item4.MRL_ethanol}} | {{RESULT-mrl-per-unit.item4.MRL_control}} |
| {{RESULT-examples-top5.item5.unit_id}} | {{RESULT-examples-top5.item5.experiment}} | {{RESULT-examples-top5.item5.delta_MRL}} | {{RESULT-mrl-per-unit.item5.MRL_ethanol}} | {{RESULT-mrl-per-unit.item5.MRL_control}} |
[renderer: MRL_ethanol/MRL_control joined from RESULT-mrl-per-unit on (unit_id, experiment) matching RESULT-examples-top5 rank order]

These five units are illustrative extremes (largest observed shifts), not a random or representative sample; they are shown to give the reader a concrete sense of what the largest per-unit shifts look like and should not be used to characterize the typical unit. See **FIG-DEMO3-EXAMPLES** for sniff-locked PSTHs (ethanol + control overlaid) and probe-location maps for each of these five units.

---

## Figures

- **FIG-DEMO3-RESULTS-ANIMAL** -- "Results: experiment-level MRL comparison (ethanol vs control)." Paired per-experiment points plus the summary statistic; states n_experiments, exact p, and rank-biserial r from `RESULT-wilcoxon`. Source results: `RESULT-mrl-per-experiment`, `RESULT-wilcoxon`.
- **FIG-DEMO3-RESULTS-UNIT** -- "Results: unit-level MRL comparison (LME, ethanol vs control)." Per-unit points (raw data, not summary-only) plus the LME condition fixed effect; states n_units, n_experiments, exact p, and standardized fixed-effect estimate from `RESULT-lme`. Source results: `RESULT-mrl-per-unit`, `RESULT-lme`.
- **FIG-DEMO3-EXAMPLES** -- "Results: top-five units by |delta_MRL| -- sniff-locked PSTHs and probe maps." One row per example unit: sniff-locked PSTH (ethanol + control overlaid) and full probe-location layout with the target unit highlighted. Source results: `RESULT-examples-top5`, `RESULT-mrl-per-unit`.

---

## Limitations

- The primary, animal-level Wilcoxon test has only `RESULT-wilcoxon`-reported experiments as its sample size ({{RESULT-wilcoxon.n_pairs}}), which is underpowered to detect anything short of a very large and highly consistent ethanol effect; the non-significant result here means an effect at this level cannot be confirmed, not that no effect exists.
- The unit-level LME's statistical significance is driven substantially by its much larger sample size (thousands of units versus a handful of experiments); a significant unit-level condition effect does not by itself establish a population/animal-level effect, and should be read alongside -- not instead of -- the primary Wilcoxon result.
- This is an observational, within-animal comparison of ethanol-contact versus control sniff periods, not a randomized or counterbalanced intervention; the design supports description of association between condition and phase-locking but does not support a causal attribution of any observed shift to ethanol exposure per se, as opposed to other factors that co-vary with ethanol-contact periods (e.g., behavioral state, sniffing pattern changes, or time-in-session).
- The top-five example units (Examples section) are selected as the most extreme observed shifts and are not representative of the broader unit population; they illustrate range, not central tendency.

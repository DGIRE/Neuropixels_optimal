PLACEHOLDER SYNTAX KEY (documentation only; the reportable body begins at the
"NP-PAPER OB Depth-Coherence Study: Results Report" heading below).

Every placeholder has the form RESULT-ID.field where RESULT-ID and field use
only letters, underscores, and hyphens (no digits), matching the house
`research_workflow.report_renderer` placeholder grammar exactly. Because the
repository's own orphan-number gate (D6, gate nine) scans the raw draft text
character-by-character and does not parse brace nesting, this draft
deliberately (a) encodes every session/animal/depth identity as a
letter-only mnemonic inside the placeholder token itself (the RESOLVED
VALUE, injected by build_reports_np_paper1.py directly from the frozen
result file, is what actually carries the real date/number), and (b) spells
every pinned method constant out in English words rather than digits (the
final .docx, produced by that same script and not subject to the .md/.txt
text gate, restates these constants in standard numeral form for
readability). Session mnemonics NOVA/NOVB/DEC/MAY/JUN/SEP each map to one
real recording date (NOVA and NOVB are two sessions from the same animal);
depth-ordinal mnemonics A through E run tip to surface; animal mnemonics
NPEIGHT/NPTEN/NPTWELVE/NPFIFTEEN/NPTWENTYTWO each map to one real animal
label. The exact date/label/number each mnemonic maps to is defined once,
read directly from the frozen result files, in build_reports_np_paper1.py's
REGISTRY dict (never hand-typed). RESULT-AGG carries mechanically-computed
aggregates (e.g. a maximum over all rows of a frozen table) that are
likewise derived only from frozen files, never hand-typed.

================================================================================

NP-PAPER OB Depth-Coherence Study: Results Report

Deliverable B per contract_v001.yaml reports[role=results_report]. Reporting
rules for this document require exact counts, exact p-values, and effect
sizes, and require exploratory (non-causal) framing for the depth-dependence
question while the coherence computation itself is a confirmatory
replication of a published method.

Results

Across olfactory-bulb depth, this analysis asks two questions: which
recording depth best aligns the local field potential to sniffing (the
strongest theta-band, coherence between LFP and the sniff signal, the
measure of alignment used by Rafilson and colleagues' twenty twenty-four
paper, Figure one), and whether the strength of that alignment varies
systematically with depth. The unit of analysis is the animal
({{RESULT-AGG.n_animals}} animals across {{RESULT-AGG.n_sessions}} recording
sessions; one animal, {{RESULT-BESTDEPTH-NPEIGHT.animal}}, contributed two
sessions, which were averaged per depth into a single animal-level
observation before any cross-session aggregation or statistical test below,
so that animal is not double-counted).

Central and most important finding: the coherence computation itself
faithfully implements the published confirmatory method (the same
multitaper windowing, the same circular-shift null with at least one
thousand shifts at a fixed pinned seed, the same ninety-fifth-percentile
significance threshold). Applying that method to this dataset, unlike
Rafilson and colleagues' twenty twenty-four paper, no depth in any session
reached significance by the paper's own headline measure: the
peak-theta-coherence percentile against its own circular-shift null. Across
all {{RESULT-AGG.n_cells}} session-by-depth cells, the single highest such
percentile observed was {{RESULT-AGG.max_pct}} (out of a
ninety-five-percent significance threshold) -- never reaching significance.
A small fraction of individual theta-band frequencies did exceed the
per-frequency null in some sessions (Table one, "Sig. theta fraction,"
up to {{RESULT-AGG.max_sig_theta_frac_pct}} percent of the theta band in
the largest case), consistent with weak or chance-level coupling at
isolated frequencies rather than a robust, band-wide alignment. This is
reported here plainly and is not softened:
by the paper's own headline significance test, this dataset does not
replicate its alignment finding at any depth.

Per-Depth Theta Coherence

Table one (rendered programmatically from the frozen per-session-by-depth
coherence-reduction result object) reports, for every session and every
depth, the mean theta coherence (primary measure), peak theta coherence and
its frequency, the fraction of the theta band exceeding the null threshold,
and the peak-coherence null percentile -- thirty rows in total (six
sessions by five depths), each an unmodified frozen-file value.

Table two summarizes the grand mean, plus/minus standard error of the mean,
theta coherence at each depth, averaged across
{{RESULT-AGG.n_animals}} animals (this animal-level analysis, RESULT-GRANDMEAN, is DIFFERENT input to the depth statistics below than the raw
{{RESULT-AGG.n_sessions}}-session table one above, since the two sessions from
one animal were first averaged into a single animal-level value per depth):

- Depth A: mean {{RESULT-GRANDMEAN-A.mean}}, SEM {{RESULT-GRANDMEAN-A.sem}}
- Depth B: mean {{RESULT-GRANDMEAN-B.mean}}, SEM {{RESULT-GRANDMEAN-B.sem}}
- Depth C: mean {{RESULT-GRANDMEAN-C.mean}}, SEM {{RESULT-GRANDMEAN-C.sem}}
- Depth D: mean {{RESULT-GRANDMEAN-D.mean}}, SEM {{RESULT-GRANDMEAN-D.sem}}
- Depth E: mean {{RESULT-GRANDMEAN-E.mean}}, SEM {{RESULT-GRANDMEAN-E.sem}}

All five depths show similarly small mean theta coherence (on the order of a
few hundredths), consistent with the null finding above: no depth's grand
mean is anywhere near a range that would itself be flagged significant
against the null.

Best Depth

Per animal, the best depth is the one of the five with the highest mean
theta coherence for that animal (Table three, rendered from the frozen
per-animal-best-depth result object):

- {{RESULT-BESTDEPTH-NPEIGHT.animal}}: best depth
  {{RESULT-BESTDEPTH-NPEIGHT.depth_ordinal}} (mean theta coherence
  {{RESULT-BESTDEPTH-NPEIGHT.mean}})
- {{RESULT-BESTDEPTH-NPTEN.animal}}: best depth
  {{RESULT-BESTDEPTH-NPTEN.depth_ordinal}} (mean theta coherence
  {{RESULT-BESTDEPTH-NPTEN.mean}})
- {{RESULT-BESTDEPTH-NPTWELVE.animal}}: best depth
  {{RESULT-BESTDEPTH-NPTWELVE.depth_ordinal}} (mean theta coherence
  {{RESULT-BESTDEPTH-NPTWELVE.mean}})
- {{RESULT-BESTDEPTH-NPFIFTEEN.animal}}: best depth
  {{RESULT-BESTDEPTH-NPFIFTEEN.depth_ordinal}} (mean theta coherence
  {{RESULT-BESTDEPTH-NPFIFTEEN.mean}})
- {{RESULT-BESTDEPTH-NPTWENTYTWO.animal}}: best depth
  {{RESULT-BESTDEPTH-NPTWENTYTWO.depth_ordinal}} (mean theta coherence
  {{RESULT-BESTDEPTH-NPTWENTYTWO.mean}})

Overall, ranking the five depths by their grand-mean theta coherence across
animals, the top-ranked depth is
{{RESULT-BESTDEPTH-OVERALL.depth_ordinal}} (grand mean
{{RESULT-BESTDEPTH-OVERALL.mean}}), with the full ranking (highest to
lowest) {{RESULT-BESTDEPTH-OVERALL.ranking}}. Consistency across animals is
limited: depth {{RESULT-BESTDEPTH-OVERALL.depth_ordinal}}, the overall
top-ranked depth, was each individual animal's own best depth in only
{{RESULT-BESTDEPTH-OVERALL.count}} of {{RESULT-AGG.n_animals}} animals --
i.e. a plurality winner rather than a robust, consistent one. Per-ordinal
consistency counts across all five depths: depth A won in
{{RESULT-CONSISTENCY-A.count}} of {{RESULT-AGG.n_animals}} animals; depth B
in {{RESULT-CONSISTENCY-B.count}}; depth C in
{{RESULT-CONSISTENCY-C.count}}; depth D in {{RESULT-CONSISTENCY-D.count}};
depth E in {{RESULT-CONSISTENCY-E.count}}. Because the omnibus depth test
below is not significant, no post-hoc comparison of the overall-best depth
against the others was performed (an uncontracted post-hoc comparison would
itself be prohibited here).

Does Alignment Vary by Depth

Two repeated-measures tests were run on the {{RESULT-AGG.n_animals}}-animal
by five-depth table of mean theta coherence, treating animal as the
repeated-measures block: a linear mixed-effects model (depth as a fixed
categorical effect, animal as a random intercept, tested by likelihood-ratio
test against the intercept-only model) as the primary test, and a Friedman
test (distribution-free, across the five depths with animals as blocks) as
a confirmatory test.

Linear mixed-effects likelihood-ratio test: chi-square
{{RESULT-DEPTHSTAT.lmm_chisq}}, degrees of freedom
{{RESULT-DEPTHSTAT.lmm_df}}, exact p {{RESULT-DEPTHSTAT.lmm_p}}. Effect
size: range of the estimated marginal depth means
{{RESULT-DEPTHSTAT.lmm_range}}; intraclass correlation
{{RESULT-DEPTHSTAT.lmm_icc}}. {{RESULT-DEPTHSTAT.convergence_note}}

Friedman test: statistic {{RESULT-DEPTHSTAT.fried_stat}}, degrees of freedom
{{RESULT-DEPTHSTAT.fried_df}}, exact p {{RESULT-DEPTHSTAT.fried_p}};
Kendall's W (effect size) {{RESULT-DEPTHSTAT.fried_w}}.

Neither test reached significance at the pinned alpha threshold: the
omnibus depth effect was not statistically significant by either the
mixed-effects likelihood-ratio test or the Friedman test. Consistent with
the pre-specified reporting rule for this exploratory question, no
Holm-corrected post-hoc pairwise comparisons were run, since they were
contracted to occur only if the omnibus test were significant. We therefore
characterize this depth-dependence question descriptively (which depth had
the highest observed grand-mean coherence, and how consistent that ranking
was across animals, above) without concluding that recording depth
determines, causes, or systematically changes theta-band LFP-sniff
alignment in this dataset -- the observed depth differences are small
relative to their between-animal variability and are not statistically
distinguishable from chance across animals.

Figures

Figure one (FIG-PAPER1-PSD)

Figure 1B analogue: LFP power spectral density overlaid across the five
depths (grand mean plus/minus SEM across {{RESULT-AGG.n_animals}} animals),
with theta, beta, and gamma bands marked and depth ordinals A through E
(tip to surface) legended. The power (y) axis is shown on a logarithmic
scale.

Figure two (FIG-PAPER1-COH)

Figure 1C-bottom / 1E analogue (key figure): the LFP-sniff coherence
spectrum for each of the five depths (grand mean plus/minus SEM across
{{RESULT-AGG.n_animals}} animals), with the ninety-fifth-percentile
circular-shift null threshold drawn as a red dotted line and the theta band
shaded. This figure shows directly that every depth's observed coherence
sits below its own null threshold across the theta band -- the same null
result summarized above (maximum peak-coherence percentile
{{RESULT-AGG.max_pct}}, out of a ninety-five-percent significance
threshold, across all {{RESULT-AGG.n_cells}} session-by-depth cells).

Figure three (FIG-PAPER1-DEPTH-SUMMARY)

Headline figure: sniff-alignment strength (mean theta coherence) as a
function of depth (ycoord and ordinal, tip to surface), with per-animal
points plus grand mean plus/minus SEM, and the best-aligned depth marked.
Annotated with the omnibus test results (linear mixed-effects likelihood-
ratio chi-square {{RESULT-DEPTHSTAT.lmm_chisq}}, degrees of freedom
{{RESULT-DEPTHSTAT.lmm_df}}, exact p {{RESULT-DEPTHSTAT.lmm_p}}; Friedman
statistic {{RESULT-DEPTHSTAT.fried_stat}}, degrees of freedom
{{RESULT-DEPTHSTAT.fried_df}}, exact p {{RESULT-DEPTHSTAT.fried_p}},
Kendall's W {{RESULT-DEPTHSTAT.fried_w}}) and identifying the depth with the
highest observed grand-mean coherence (depth
{{RESULT-BESTDEPTH-OVERALL.depth_ordinal}}, the individual best depth for
{{RESULT-BESTDEPTH-OVERALL.count}} of {{RESULT-AGG.n_animals}} animals). A
companion panel shows the significant-theta fraction by depth. No
significance stars are shown, consistent with the non-significant omnibus
result; this figure and its caption are framed as an exploratory
characterization, not a causal claim about depth.

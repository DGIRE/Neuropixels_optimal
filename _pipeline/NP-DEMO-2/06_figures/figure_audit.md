# NP-DEMO-2 Figure Audit
## Overall: CONDITIONAL_PASS

No plotted value was found to numerically disagree with its source RESULT-* object
(no BLOCKER findings). However, several MAJOR contract-compliance gaps were found:
two figures' axis labels do not match the axis_requirements pinned in
contract_spec.yaml, and FIG-DEMO2-RES-UNIT omits RESULT-phase-unit entirely even
though it is a pinned source_results entry for that figure and the sketch calls
for a preferred-phase inset. FIG-DEMO2-QC-PSTH similarly never uses its pinned
RESULT-mrl-unit source. These should be resolved or explicitly waived by a human
reviewer before final sign-off; they do not indicate any numeric/statistical error.

## Per-Figure Findings

### FIG-DEMO2-RES-EXP
Renderer: 06_figures/render_res_exp.py. Data: figure_data/figdata_FIG-DEMO2-RES-EXP.yaml,
built by build_figdata.py::build_res_exp() directly from RESULT-mrl-animal.yaml +
RESULT-stat-animal.yaml (verbatim, no recomputation).
1. Per-animal MRL values (mean_mrl_ethanol / mean_mrl_control for all 6 sessions) in
   figdata match RESULT-mrl-animal.yaml exactly, value for value. PASS
2. Annotated p-value (0.03125) matches RESULT-stat-animal.yaml: pvalue. PASS
3. Annotated effect size (rank-biserial r = -1.000, shown as "-1.000") matches
   RESULT-stat-animal.yaml: effect_size = -1.0. PASS
4. n_animals = 6 is shown (caption + in-panel text box) and matches
   RESULT-stat-animal.yaml: n_animals. PASS
5. Y-axis starts at 0.0 in both Panel A (axA.set_ylim(0.0, y_top)) and Panel B
   (axB.set_ylim(0, lim)); confirmed visually in the rendered PNG/PDF. No truncation.
   The sketch (figure_examples/Demo figure.bmp, y-range 0-2.0) is also not
   truncated, so no truncation was carried over from the sketch either. PASS
6. DEV-001 (2021-11-03) is flagged with a distinct red star marker (COLOR_DEV001)
   in both panels, an inline " *" label, and an explicit caption footnote
   ("* DEV-001 ... requires human review before use"). Visually confirmed in the
   rendered figure. PASS
MAJOR: contract_spec.yaml pins axis_requirements: x_label "SNF sniff phase
(rad)", y_label "spike rate (norm.) / MRL" for this figure (copied from the
Demo figure.bmp sketch, which shows a phase-tuning curve over 0 to 2*pi). The
actual renderer produces a paired-dot plot (Panel A: x = categorical
ethanol/control, y = "Mean MRL (per animal)") and a scatter plot (Panel B:
x/y = "Mean MRL, ethanol"/"control") -- neither axis is phase-based, because
RESULT-mrl-animal contains only scalar per-animal MRLs, not phase-binned
tuning curves. This is a defensible engineering substitution (documented in
the render_res_exp.py docstring) given the data actually available in the
pinned source_results, but it means the figure does not satisfy the contract's
literal axis_requirements text. Flagging for human reconciliation (either the
contract's axis_requirements should be corrected to match the deliverable
design, or a phase-based panel should be added) -- this is a spec/figure
mismatch, not a numeric error.
### FIG-DEMO2-RES-UNIT
Renderer: 06_figures/render_res_unit.py. Data: figure_data/figdata_FIG-DEMO2-RES-UNIT.yaml,
built by build_figdata.py::build_res_unit() from RESULT-mrl-unit.yaml +
RESULT-stat-unit.yaml.

1. n_units = 1138 is shown in-panel and in caption, and equals
   RESULT-stat-unit.yaml: n_units (the matched-pair LMM population) -- NOT the
   1603 total rows in RESULT-mrl-unit.yaml. Verified programmatically: 1603
   total rows, 1138 with both a finite ethanol AND finite control MRL (all 1603
   have a finite ethanol MRL; 465 have mrl_control = NaN, overwhelmingly from
   the DEV-001-flagged 2021-11-03 session). The renderer asserts
   len(paired) == stat['n_units'] at build time, so a future data drift would
   fail loudly rather than silently mismatch. PASS
2. Annotated p-value (7.511e-220) matches RESULT-stat-unit.yaml: pvalue =
   7.510670133683472e-220. PASS
3. Annotated effect size (-1.026) matches RESULT-stat-unit.yaml:
   standardized_effect_size = -1.0263195327880161. PASS
4. Y-axis starts at 0.0 (ax.set_ylim(0.0, y_top)); confirmed visually. PASS
5. Plotted MRL values (violin + jittered points) are the exact 1138 matched-pair
   rows from RESULT-mrl-unit.yaml, verbatim (no recomputation). Programmatic
   range check: ethanol MRL in [0.000401, 0.1833], control MRL in [0.0016, 1.0]
   for the matched-pair population -- consistent with the full RESULT-mrl-unit
   range. PASS
MAJOR (same axis-label issue as FIG-DEMO2-RES-EXP): contract pins x_label
"SNF sniff phase (rad)" / y_label "spike rate (norm.) / MRL"; actual axes are
categorical ethanol/control (x) and "Per-unit MRL" (y). Same root cause
(no phase-tuning-curve data in the plotted source objects) and same
recommendation (reconcile contract vs. delivered design).

MAJOR: contract_spec.yaml pins source_results: [RESULT-mrl-unit,
RESULT-phase-unit, RESULT-stat-unit] for this figure, and the sketch_guidance
explicitly calls for a "polar inset of preferred phase." RESULT-phase-unit.yaml
exists and contains per-unit phase_ethanol_rad / phase_control_rad data, but
it is never loaded by build_figdata.py::build_res_unit() and never appears in
figdata_FIG-DEMO2-RES-UNIT.yaml or render_res_unit.py. The delivered figure
therefore omits a pinned required source entirely -- no preferred-phase content
of any kind is shown for the unit level. This is not documented as a deviation
in 08_traceability/deviations.yaml (only DEV-001, about PROH-002, is present).
Recommend either adding a preferred-phase panel (e.g., a polar histogram of
phase_ethanol_rad / phase_control_rad, colorblind-safe per condition) or
formally amending the contract to drop RESULT-phase-unit from this figure's
source_results with a documented rationale.
### FIG-DEMO2-RES-EX
Renderer: 06_figures/render_res_ex.py. Data: figure_data/figdata_FIG-DEMO2-RES-EX.yaml
(+ companion .npy), built from RESULT-delta-mrl.yaml + RESULT-psth-examples.yaml
+ RESULT-unit-locations.yaml.

1. The 5 example units (unit_id 104, 312, 253, 626, 69; all session 2021-11-03)
   are exactly the top-5 rows of RESULT-delta-mrl.yaml sorted descending by
   abs_delta_mrl (0.99949, 0.99914, 0.99794, 0.99714, 0.99625) -- verified by
   reading RESULT-delta-mrl.yaml directly and confirming the same 5 unit_ids
   in the same order appear in figdata and in RESULT-psth-examples.yaml.
   build_figdata.py also defensively re-sorts by abs_delta_mrl before taking
   [:5]. PASS
2. Each panel's title shows "|DeltaMRL|=<value>" (e.g. "Unit 104 ...
   |DeltaMRL|=0.999"), and the underlying abs_delta_mrl figdata value for each
   unit matches RESULT-delta-mrl.yaml exactly. PASS
3. All 5 sessions shown are '2021-11-03', consistent with DEV-001. PASS
4. The DEV-001 warning is present: a red-bordered text box beginning "DEV-001
   WARNING: All 5 top-|delta_MRL| examples in this figure come from session
   2021-11-03..." is rendered at the bottom of the figure (confirmed in the
   PDF render; text is small but fully present, not clipped, in the vector PDF --
   in the raster PNG it appears small due to bbox_inches="tight" scaling but
   is legible at full zoom). PASS

Probe-location panels (unit_depth_um, unit_xcoord_um, shank, unit_firing_rate)
for all 5 units were cross-checked against RESULT-unit-locations.yaml and
match exactly (e.g. unit 104: depth 251.7846 um / shank 0 / xcoord -250.0 --
displayed as "shank 0, depth 252 um"). These coordinates are taken directly
from the frozen figdata (never recomputed from D), per the level-1
exact-coordinate acceptance criterion. PASS

No BLOCKER or MAJOR issues found for this figure. MINOR: the DEV-001 warning
and caption text sit close together at the bottom margin in the PNG raster
export; consider a bit more vertical whitespace so the warning box is not
visually crowded against the caption in raster form (the PDF vector export is
unaffected and fully legible).
### FIG-DEMO2-QC
Renderer: 06_figures/render_qc.py. Data: figure_data/figdata_FIG-DEMO2-QC.yaml,
built by build_figdata.py::build_qc() from RESULT-qc-counts.yaml +
RESULT-eth-threshold-log.yaml + RESULT-qc-discards.yaml, plus a fresh raw
SNF_z/ETH trace reload via the validated kernel (load_experiment_data ->
compute_sniff_phase -> threshold_eth) for display purposes only (the trace
itself is not a frozen RESULT-* scalar, so a fresh kernel reload is appropriate
and documented in the module docstring; all thresholds/counts overlaid on it
are taken from frozen RESULT-* objects verbatim).

1. All 6 sessions are present (2021-11-01, 2021-11-03, 2021-12-15, 2022-05-17,
   2022-06-24, 2022-09-14) -- verified programmatically (len(sessions) == 6 in
   the figdata) and visually (12 stacked panels = 6 pairs). PASS
2. n_sniffs annotated in each panel title is copied verbatim from
   RESULT-qc-counts.yaml (qc["n_sniffs"]) for all 6 sessions
   (10949 / 9087 / 7005 / 9422 / 9169 / 3852) -- matches exactly. PASS
3. The red dashed eth_threshold line value in each panel equals
   RESULT-eth-threshold-log.yaml: final_threshold for that session (0.055 /
   0.11 / 0.11 / 0.11 / 0.11 / 0.11) -- the figdata builder reads
   eth_log[sd]["final_threshold"] directly and re-derives the display trace
   at that exact threshold via threshold_eth(D, eth_threshold=final_threshold),
   so the plotted line and the trace it is drawn on are self-consistent. PASS
4. 2021-11-01 shows eth_threshold = 0.055 (title reads "eth_threshold=0.055
   (adjusted from 0.11 default)"), matching
   RESULT-eth-threshold-log.yaml: final_threshold = 0.055,
   was_adjusted = true for that session; all other 5 sessions correctly show
   0.110 "(default (0.11), not adjusted)". PASS
MINOR: the per-panel title text (session date + counts + threshold) visually
overlaps the bottom of the previous session's ETH-panel plot in the stacked
layout (confirmed in both PNG and PDF renders) -- a cosmetic tight_layout /
hspace issue, not a data-correctness issue; all annotated values themselves
are correct.

MINOR: caption requirement "state whether that experiment's eth_threshold
was adjusted from the 0.11 default AND WHY" -- the title states whether
("adjusted from 0.11 default" / "not adjusted") but does not state the why
(i.e., that Pass-1 flagged the session outside the cross-session mean/SD
window, per PROH-002). The reasoning is only available in
RESULT-eth-threshold-log.yaml (mean_contacts_pass1, sd_contacts_pass1)
and is not surfaced as figure/caption text. Recommend adding one sentence to
the figure caption (or per-panel title) citing the Pass-1 mean/SD comparison
that triggered the adjustment for 2021-11-01.
### FIG-DEMO2-QC-PSTH
Renderer: 06_figures/render_qc_psth.py. Data:
figure_data/figdata_FIG-DEMO2-QC-PSTH.yaml (+ companion .npy), built by
build_figdata.py::build_qc_psth() from RESULT-psth-extremes.yaml only.

1. unit_id/session labels match RESULT-psth-extremes.yaml exactly: strongest
   = unit 278, session 2022-06-24; weakest = unit 82, session 2021-12-15.
   PASS
2. pooled_mrl values match RESULT-psth-extremes.yaml exactly: strongest =
   0.24714546579444693 (displayed "pooled_MRL=0.247"), weakest =
   0.00030624236344668835 (displayed "pooled_MRL=0.000"). PASS for the
   underlying value match; see MINOR note below re: display rounding.
3. Both strongest and weakest are shown as 2 side-by-side panels (axL, axR).
   PASS. Y-axis starts at 0 in both panels (ax.set_ylim(0.0, y_top)), no
   truncation.

MAJOR: contract_spec.yaml pins source_results: [RESULT-psth-extremes,
RESULT-mrl-unit] for this figure, and visual_encoding.raw_data_also_available:
true. build_figdata.py::build_qc_psth() only loads RESULT-psth-extremes.yaml
-- RESULT-mrl-unit.yaml is never read anywhere in the QC-PSTH build/render
path. There is consequently no raw per-unit MRL distribution shown to give the
reader context for how extreme the "strongest"/"weakest" pooled MRLs are
relative to the full unit population (e.g., a rug plot or histogram of all
units' pooled/condition MRLs from RESULT-mrl-unit). This is a pinned required
source that is silently unused; recommend adding the missing raw-data context
or amending the contract with a documented rationale.

MINOR: the weakest unit's true pooled_mrl (0.00030624...) is displayed
rounded to 3 decimals as "0.000", which could read as an exact-zero MRL to a
reader who does not consult RESULT-psth-extremes.yaml. Recommend more
significant figures (e.g. scientific notation) for very small pooled_mrl
values, consistent with the fmt_p() scientific-notation convention already
used elsewhere in these renderers for small p-values.
## Summary of Findings

| Figure | Blockers | Majors | Minors |
|---|---|---|---|
| FIG-DEMO2-RES-EXP | 0 | 1 (axis label vs. contract) | 0 |
| FIG-DEMO2-RES-UNIT | 0 | 2 (axis label vs. contract; RESULT-phase-unit unused) | 0 |
| FIG-DEMO2-RES-EX | 0 | 0 | 1 (bottom-margin text crowding, raster only) |
| FIG-DEMO2-QC | 0 | 0 | 2 (title/panel overlap; missing "why" for threshold adjustment) |
| FIG-DEMO2-QC-PSTH | 0 | 1 (RESULT-mrl-unit unused / no raw-data context) | 1 (pooled_mrl rounds to "0.000") |

No BLOCKER-level findings: every plotted numeric value (MRLs, p-values,
effect sizes, n's, thresholds, counts, coordinates, unit selections) was traced
to its frozen 04_results/RESULT-*.yaml object and matches exactly (verbatim
pass-through, confirmed programmatically for the large tabular results and by
direct comparison for the scalar stat objects). No axis in any figure was
truncated in a way the contract prohibits; every zero_reference:true figure
does start its axis at 0. FIG-DEMO2-QC's zero_reference:false non-zero-start
axes are contractually permitted for that figure (raw trace display) and are
not misleadingly cropped (full data min/max with padding). The Demo
figure.bmp sketch's own axes are not truncated (0-2.0), so no sketch-implied
truncation was silently adopted anywhere. DEV-001 (the 2021-11-03 PROH-002
degenerate-session finding) is visibly and correctly flagged in
FIG-DEMO2-RES-EXP and FIG-DEMO2-RES-EX per contract and
08_traceability/deviations.yaml.

The three MAJOR items are all contract-vs-implementation mismatches, not
data-integrity failures: (1)/(2) two figures' axis labels do not match the
contract's pinned (sketch-derived) axis_requirements because the plotted
source objects contain scalar MRLs, not phase-tuning curves; (3) two figures
(FIG-DEMO2-RES-UNIT, FIG-DEMO2-QC-PSTH) each have one pinned source_results
entry (RESULT-phase-unit, RESULT-mrl-unit respectively) that is never
actually read or plotted. These should go back to a human/contract-designer
for reconciliation (update the contract to match the deliverable, or update
the figures to include the missing content) before final certification;
recommend CONDITIONAL_PASS pending that reconciliation rather than an
outright FAIL, since no numbers on any figure are wrong.

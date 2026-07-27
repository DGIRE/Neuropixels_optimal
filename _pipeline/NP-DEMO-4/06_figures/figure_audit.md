# NP-DEMO-4 Figure Audit Report

**Role:** Figure Auditor (Blueprint v2, fresh-context review)
**Scope:** FIG-DEMO4-1-SNIFF, FIG-DEMO4-2-FR, FIG-DEMO4-3-ETH
**Renderer:** 03_software/render_figures.py (reads only 06_figures/figure_data/*.npy plus the manifest; no scientific recomputation)
**Data builder:** 03_software/build_figure_data.py (reads only frozen result objects, one directory up from 06_figures)
**Auditor note:** the request-side figure_examples folder is empty -- no layout sketch was supplied for this task, so the "sketch changed only presentation, never numbers" check is not applicable (no sketch to compare against).

## Overall verdict: CONDITIONAL PASS

All three figures regenerate correctly from frozen result objects, use the contracted colormap/axes/labels/CI formula, and inject sample counts from the manifest rather than typing them by hand. No BLOCKER or MAJOR issues were found. One MINOR cosmetic layout defect (caption/x-axis-label text overlap on FIG-DEMO4-2-FR) and two MINOR consistency observations are noted below. Cleared for FIGURE_AUDIT_OK conditional on acknowledging the MINOR items (no re-render required for scientific correctness; a cosmetic fix to Figure 2 is recommended but not blocking).

## Per-figure verdict

| Figure | Verdict |
|---|---|
| FIG-DEMO4-1-SNIFF | PASS |
| FIG-DEMO4-2-FR | PASS (MINOR cosmetic defect -- caption/x-label overlap) |
| FIG-DEMO4-3-ETH | PASS |

## Full checklist

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Colormap = jet | PASS | render_figures.py line 41, COLORMAP = "jet"; visually confirmed classic blue-to-red jet ramp in all three rendered PNGs. |
| 2 | x-axis extent 0-40 s | PASS | X_MAX_S = 40.0 (line 42); _clip_to_40s masks time_axis <= 40.0 before plotting. Full unclipped time axis in figure_data_manifest.yaml is 0.0 to 41.68 (5211 samples at 125 Hz LabView rate), correctly cropped to the contracted 0-40 s trial window at render time -- not a dishonest truncation, this is the contracted trial boundary. |
| 3 | Panel 1 y-axis = trial number (LabView) | PASS | ax1.set_ylabel("trial number (LabView)"); visually confirmed on all three figures; ax1.set_ylim(0.5, n_trials+0.5) shows the full trial range with no truncation. |
| 4 | Panel 2 y-axis, signal-specific | PASS | Fig 1: sniff rate (Hz); Fig 2: firing rate (Hz); Fig 3: ETH sensor value (a.u.) -- each matches contract axis_requirements y_label Panel-2 half exactly. |
| 5 | Colorbar labels per contract | PASS | Fig 1: sniff rate (Hz); Fig 2: firing rate (Hz); Fig 3: ETH sensor value (a.u., normalized 0-1) -- all three exact matches to contract_v001.yaml caption_requirements colorbar strings. |
| 6 | 95% CI = mean +/- 1.96*SEM | PASS | build_figure_data.py _trial_avg_and_ci: sem = nanstd(matrix, axis=0, ddof=1) / sqrt(n_valid), ci = mean -/+ 1.96*sem. Independently recomputed from the saved fig2_fr_matrix.npy and confirmed bit-for-bit match against the saved mean/ci arrays. |
| 7 | n_trials / n_sessions / n_units injected, not typed | PASS | Values traced to sniff_matrix.shape[0] (=324) and len(sniff_mat_dict) (=6), never hand-typed. n_units_per_session in manifest matches unit-inclusion result exactly per session. Sum of n_labview_trials across sessions equals 324. |
| 8 | No statistical annotations on figures | PASS | Contract requires no stars/p-values/tests on figures. stat_for_caption is loaded but never referenced in any caption text (verified via file search). No stars or p-values appear on the rendered figures. |
| 9 | Font size >= 7 pt | PASS | MIN_FONT = 8, used for all axis labels/ticks/colorbar/legend; title uses MIN_FONT+1=9. 8 >= 7 contract floor. |
| 10 | 300 DPI for PNG | PASS | DPI = 300 passed to fig.savefig for PNG only (SVG is vector). Independently verified via PIL: all three PNGs report embedded DPI approximately 300 (float rounding artifact, not a defect). |
| 11 | ETH figure uses raw (pre-mean-subtraction) ETH per CON-002 | PASS | Analysis code captures the raw ETH array directly, taken before mean-subtraction, ahead of the internal mean-subtraction used only for contact-threshold detection. Figure-data builder pools this raw per-trial result with no further transform. Verified data range: nanmin 0.0, nanmax 1.0, consistent with normalized 0-1 raw description. |
| 12 | FR figure uses included units only | PASS | Unit-inclusion helper applies FR >= 0.1 Hz AND total spikes >= 5000 simultaneously and returns included_unit_ids; the 50 ms sliding-window firing-rate function is called with this filtered ID list. Figure-data builder pools this already-filtered result verbatim. |
| 13 | Matrix shape matches manifest / values match source result | PASS | manifest matrix_shape 324 by 5211 for all three figures matches independently loaded .npy shapes exactly. Row 0 first 5 sniff values match the sniff-rate-matrix source result session 2021-11-01 first trial exactly. |
| 14 | Sample counts computed, not typed | PASS | All caption N values trace back through the manifest to shape and len calls and frozen source-result values -- no hardcoded integers found in the caption-injection code paths. |
| 15 | No broken/truncated axis beyond contract-permitted windowing | PASS with note | truncation_permitted is false for all 3 figures; the only cropping present is the intentional 0-40 s trial-window clip required by the contract text, not a hidden truncation. Panel-1 heatmap vmin/vmax are percentile-clipped for contrast, a standard heatmap contrast-normalization choice, not a value-axis truncation; the underlying data is fully preserved. Panel-2 line-plot y-axes are auto-scaled (not zero-anchored); flagged for awareness but not treated as a violation since these are single-trace time-series plots, not comparative bar charts. |

## Issues by severity

### BLOCKER
None found.

### MAJOR
None found.

### MINOR
1. FIG-DEMO4-2-FR: caption/x-axis-label text overlap (cosmetic legibility defect). The Figure 2 caption is longer than Fig 1/3 (it includes the per-session unit-inclusion list), wraps to multiple lines, and visually overlaps the Panel-2 time-axis label "time from trial start (s)" in the rendered PNG/SVG. This is a matplotlib layout side effect (caption text placed near the bottom of the figure colliding with the Panel-2 x-label position after tight-bbox re-packing) -- no data, label text, or numeric content is wrong, only overlapping placement. Recommend increasing panel spacing or moving the caption further down for Figure 2 specifically. Does not block certification.
2. Colorbar vmin inconsistency across the three heatmaps. Fig 1 and Fig 2 anchor vmin at 0.0 (true physical zero), while Fig 3 (ETH) uses the observed data minimum instead of a fixed 0.0. In this dataset the ETH minimum happens to equal 0.0, so there is no visible effect, but the renderer convention should be normalized for future runs where the minimum might not be exactly zero.
3. Dead/unused stat variable in render_figures.py. stat_for_caption is loaded into a local variable that is never referenced afterward. Harmless (confirms no stat annotation leaks onto the figures) but should be removed or wired into a permitted use to avoid confusion during future maintenance.

## Governance note (surfaced for visibility, outside narrow figure-audit scope)

While auditing, repository tooling reported that the NP-DEMO-4 contract recorded-approval hash no longer matches the on-disk contract_v001.yaml (hash mismatch, approval invalidated). This did not affect the figure audit itself, since figures were verified directly against frozen result objects which are independent of this hash check. It should be resolved or re-approved before any further pipeline step that the repository guardrail scopes to this condition (writes under the results directory, or running the analysis driver) is attempted.

## Reproducibility / regeneration checks performed

- Recomputed mean, ci_lower, and ci_upper for fig2_fr_matrix.npy independently with numpy using the exact CI formula from build_figure_data.py; matched the saved arrays exactly (allclose true, including NaN positions).
- Verified fig1_sniff_matrix.npy row 0 against the sniff-rate-matrix source result for session 2021-11-01, trial 0 -- exact match.
- Verified matrix shapes (324, 5211) for all three figure-data matrices against the manifest matrix_shape fields -- exact match.
- Verified n_units_per_session values in the manifest against the unit-inclusion source result -- exact match (192, 313, 183, 99, 175, 150).
- Verified the sniff-stat source result (W=7.0, p_exact=0.5625, rank_biserial_r=0.3333, n_animals=6, significant=false) matches the manifest stat_for_caption block exactly, and confirmed this block is not surfaced on any of the three figures.
- Verified PNG DPI metadata via Pillow: all three report approximately 300 DPI.
- Confirmed no sketch exists in the figure_examples request folder (directory is empty) -- no truncated axis was inherited from a sketch for this task.

## Conclusion

Cleared for FIGURE_AUDIT_OK. No BLOCKER or MAJOR findings. Three MINOR items are logged above (cosmetic caption/label overlap in Figure 2, a vmin convention inconsistency in Figure 3, and one unused variable) -- none affect scientific correctness, unit/axis honesty, or contract compliance, and none require re-running the analysis or figure-data build. A cosmetic re-render of Figure 2 (layout fix only, no data change) is recommended but not required for certification.

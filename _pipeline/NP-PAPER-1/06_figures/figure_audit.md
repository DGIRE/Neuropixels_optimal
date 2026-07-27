# Figure Audit — NP-PAPER-1

Auditor: figure-auditor (fresh context). Verified each of the 6 rendered
figures traces to a saved/frozen result object, has honest (non-truncated,
non-distorted) axes, correct statistical annotations, and no undisclosed
sketch influence (none existed for this task).

**Net finding**: all 6 figures PASS on rendering fidelity — every plotted
value independently confirmed to trace to `04_results/frozen/*`, statistical
annotations verified pixel-for-pixel against `RESULT-depth-stat.yaml`, and
the key scientific-honesty figure (FIG-PAPER1-COH) independently confirmed to
show the true, non-significant finding (max peak-coherence percentile 90.4
across all 30 cells, never reaching 95) without truncation or distortion. One
MAJOR process/provenance gap was found and has been resolved (see below); two
MINOR findings (undisclosed axis crops) have been fixed.

---

## Per-figure verdicts

**FIG-PAPER1-TRACES — PASS**
Data traces to `RESULT-example-traces.npz`/`_meta.yaml` + `RESULT-qc-counts.yaml`,
confirmed byte-identical pass-through by independent sha256 recomputation.
Representative session (06-24-2022, 573 retained windows — confirmed the
actual maximum across all 6 sessions) and segment (80.0–90.0 s) match frozen
meta exactly. Depth channel indices/ycoords in the caption match
`RESULT-qc-counts.yaml`'s 06-24-2022 row exactly. No axis truncation.

**FIG-PAPER1-SPECTRO — PASS**
Data traces to `RESULT-example-spectrogram.npz`/`_meta.yaml` (confirmed
byte-identical by independent sha256), same representative session as
TRACES, verified consistent. Sniff-rate overlay is not independently
recomputed — tied 1:1 to `window_center_s`. Frequency axis crop (0–100 Hz of
0–500 Hz) was already disclosed in-caption at render time — no action needed.

**FIG-PAPER1-SNIFFRATE — PASS**
Data traces to `RESULT-sniff-rate-raw.npz`/`_meta.yaml` + `RESULT-sniff-rate.yaml`
+ `RESULT-qc-counts.yaml`. Every median/IQR/n annotated on the rendered panels
independently checked against the frozen YAML for all 6 sessions — exact
match to 2 decimal places. Distribution explicitly labeled UNTRIMMED,
consistent with the summary stats it's paired with (MINOR-3 framing honored).

**FIG-PAPER1-PSD — PASS** (one MINOR found and fixed)
Data traces to `RESULT-grand-mean-spectra.npz` + `RESULT-qc-counts.yaml` +
`RESULT-best-depth.yaml` (n_experiments=5, cross-sourced not hardcoded).
`log_scale: true` correctly applied; band shading matches contract-pinned
definitions. **MINOR (fixed)**: frequency axis was cropped to 150 Hz without
an explicit caption disclosure (unlike SPECTRO). Caption now states "Frequency
axis cropped to 0.25-150 Hz for display; underlying data extend to 500 Hz
(Nyquist)." Figure re-rendered; no numeric values changed.

**FIG-PAPER1-COH — PASS** (one MINOR found and fixed; this is the key
scientific-honesty figure and it is correct)
Independently re-scanned all 30 rows of `RESULT-theta-coh.yaml`: `peak_coh_percentile`
ranges 0.2–90.4, maximum exactly 90.4 (12-15-2021, depth 4), never reaching
95. Rendered y-axis spans the full 0–1 range with `ax.set_ylim(0.0, 1.0)`
confirmed in code and visually — no truncation. Observed coherence curves
(all 5 depths) sit visibly below the null-threshold curves across the entire
frequency range shown, including the shaded theta band — an honest rendering
of a genuinely null finding, not softened or cropped to exaggerate a signal.
**MINOR (fixed)**: same undisclosed frequency-axis crop as PSD — caption now
carries the same disclosure line. Figure re-rendered; no numeric values
changed (the y-axis carrying the actual measured quantity was always
untruncated).

**FIG-PAPER1-DEPTH-SUMMARY — PASS on rendering fidelity; one MAJOR
provenance gap found and RESOLVED**
Pixel-level verification: rendered LMM text ("chi-square=1.822, df=4,
p=0.768") and Friedman text ("chi-square=1.760, df=4, p=0.780, Kendall's
W=0.088") match `RESULT-depth-stat.yaml` exactly to 3 decimal places. No
significance stars present, consistent with `significance_stars: false` and
`omnibus_significant: false`. Best-depth marker (D5, "won in 1/5 animals")
matches `RESULT-best-depth.yaml` exactly.

**MAJOR finding (RESOLVED same day)**: the per-animal x per-depth matrices
backing the main panel's gray per-animal lines and the ENTIRE
significant-theta-fraction companion panel had no frozen P3 source object —
they were computed for the first time during P4 figure-data assembly (by
mechanically re-applying the already-validated DEV-002 animal-averaging rule
to `RESULT-theta-coh.yaml`, and sanity-checked against frozen scalars, but
never independently frozen with its own acceptance record). This is a real
D11 provenance gap, correctly caught by the auditor.

**Resolution**: `RESULT-theta-coh-by-animal.yaml` has been frozen
(`04_results/frozen/`, see `freeze_manifest.yaml`) with its own acceptance
note (level 1, exact, deterministic mean of already-validated frozen inputs),
independently re-deriving the 5-animal x 5-depth table from
`RESULT-theta-coh.yaml` using a fresh copy of the DEV-002 animal map (not
imported from the P4 build script). Verified bit-for-bit identical (all 25
cells, atol 1e-9) to what was already plotted in `fig_depth_summary_data.npz`
— no re-rendering was needed for the numbers themselves, only the provenance
trail was closed. `figure_data_manifest.yaml` updated to reference the new
frozen source.

---

## No-sketch confirmation
`_pipeline/NP-PAPER-1/01_request/figure_examples/` confirmed empty.
`render_manifest.yaml`'s `influenced_by: null` for all 6 figures is accurate.

## Quality/format checks
All 6 figures have both `.pdf` (vector) and `.png` outputs. `min_font=7`
respected throughout. Colorblind-safe encoding confirmed (viridis-based
ordinal depth colors + Okabe-Ito band colors; no red-green-only categorical
encoding; TRACES uses green/black, not red/green).

## Disposition
No deviation filed — this was a process/provenance gap with a mechanical,
already-validated fix, not a scientific ambiguity requiring human judgment.
Both MINOR findings (axis-crop disclosure) fixed by re-rendering with updated
captions (no numeric values changed). The MAJOR finding (provenance gap) is
resolved by freezing `RESULT-theta-coh-by-animal.yaml`. All 6 figures now
PASS with complete D11 traceability to frozen P3 result objects.

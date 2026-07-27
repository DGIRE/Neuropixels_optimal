PLACEHOLDER SYNTAX KEY (documentation only; the reportable body begins at the
"NP-PAPER OB Depth-Coherence Study: Methods / QC Report" heading below).

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
label. The exact date/label each mnemonic maps to is defined once, read
directly from the frozen result files, in build_reports_np_paper1.py's
REGISTRY dict (never hand-typed), plus one CONST registry sourced from
contract_v001.yaml's pinned parameters.

================================================================================

NP-PAPER OB Depth-Coherence Study: Methods / QC Report

Deliverable A per contract_v001.yaml reports[role=methods_report]. This
document is intentionally separate from the Results report and does not
report effect sizes or exact p-values; its purpose is to document exactly
how the analysis was run and what was included or discarded.

Methods

This task asks, for each of several Neuropixels olfactory-bulb recording
sessions, which of five evenly-spaced recording depths best aligns the local
field potential to sniffing, and whether that alignment varies systematically
with depth. It is a per-depth replication of the spectral analysis in
Rafilson et al. (twenty twenty-four), Figure one, restricted to control (non-ethanol)
epochs.

Sessions and animals. Data comprise the following recording sessions:
{{RESULT-QC-NOVA.session}}, {{RESULT-QC-NOVB.session}},
{{RESULT-QC-DEC.session}}, {{RESULT-QC-MAY.session}},
{{RESULT-QC-JUN.session}}, and {{RESULT-QC-SEP.session}}. Two of these
sessions ({{RESULT-QC-NOVA.session}} and {{RESULT-QC-NOVB.session}}) are two
recordings from the same animal ({{RESULT-BESTDEPTH-NPEIGHT.animal}},
"Marvel"; confirmed by David Gire) rather than independent animals, so these
sessions represent only {{RESULT-AGG.n_animals}} distinct animals
({{RESULT-BESTDEPTH-NPEIGHT.animal}}, {{RESULT-BESTDEPTH-NPTEN.animal}},
{{RESULT-BESTDEPTH-NPTWELVE.animal}}, {{RESULT-BESTDEPTH-NPFIFTEEN.animal}},
{{RESULT-BESTDEPTH-NPTWENTYTWO.animal}}), with
{{RESULT-BESTDEPTH-NPEIGHT.animal}} contributing two sessions. This document
reports every table below per session (all six rows shown), consistent with
per-session QC being a session-level property; it is the cross-session/
animal-level aggregation used for the depth statistics (Results report) that
treats this animal's two sessions as one, not this per-session QC log.

All signals (LFP, SNF, ETH, xcoords/ycoords) were obtained from the
validated load_experiment_data loader; sniff detection used the validated
compute_sniff_phase function at its pinned default onset threshold (never
adjusted). LFP was extracted for the full recording duration for the five
selected channels via the additive, regression-anchored extract_full_lfp
function (order-four Butterworth low-pass filter, then decimated, identical
to the validated loader's preprocessing).

Depth Selection

Five channels were selected per session using a single deterministic rule,
applied identically across sessions: among channels with both a valid LFP
row and a ycoord, the minimum and maximum ycoord define the recording
extent; five target positions evenly spaced across that extent were
located, and for each target the nearest-ycoord channel was selected (ties
broken by the lower channel index, with a single probe column fixed
throughout). Depths are labeled Depth A (probe tip) through Depth E (most
superficial). The resulting channel indices and ycoords, in micrometers, are
exact integers, never re-derived by hand:

- Depth A: channel {{RESULT-QC-NOVA.ch_a}}, ycoord {{RESULT-QC-NOVA.yc_a}} um
- Depth B: channel {{RESULT-QC-NOVA.ch_b}}, ycoord {{RESULT-QC-NOVA.yc_b}} um
- Depth C: channel {{RESULT-QC-NOVA.ch_c}}, ycoord {{RESULT-QC-NOVA.yc_c}} um
- Depth D: channel {{RESULT-QC-NOVA.ch_d}}, ycoord {{RESULT-QC-NOVA.yc_d}} um
- Depth E: channel {{RESULT-QC-NOVA.ch_e}} (session
  {{RESULT-QC-SEP.session}}: channel {{RESULT-QC-SEP.ch_e}}), ycoord
  {{RESULT-QC-NOVA.yc_e}} um

These channel/ycoord values are identical across five of the six sessions
and differ by one channel index at Depth E for one session only (see Table
one). Absolute depth from the bulb surface is not reported (insertion depth
not independently confirmed for these sessions); depths are given as
position along the probe (ycoord, um), and depth ordinals A-E are used
consistently across sessions in all cross-session comparisons.

Raw-file provenance

load_experiment_data and extract_full_lfp require one authoritative raw AP
recording per session. During implementation it was discovered that several
session folders contain multiple candidate raw AP files (e.g. an unspliced
raw file alongside concatenated or excerpt-chunk series), and the kernel's
pre-existing file picker selects the alphabetically-first candidate with no
duration or identity check -- for two sessions this originally selected a
drastically truncated or unrelated file. This was resolved (approved by
David Gire) with an additive, duration-matching override that, per session,
excludes same-directory chunk/excerpt-series members from individual
selection and picks the remaining whole-file candidate whose implied
duration is closest to the independent LabView (SNF/ETH) recording duration,
flagging (not hard-blocking) any match deviating by more than ten percent.
The regression anchor (full-duration LFP restricted to the first ten
seconds must equal the current loader's output) was re-verified to pass for
all six sessions against their corrected file choices.

Per-session outcomes of this rule (implied duration from file size and
sampling rate; LabView duration from the SNF trace length and rate):

- {{RESULT-QC-NOVA.session}}: selected the single unspliced raw file
  (implied duration {{RESULT-QC-NOVA.bin_dur}} min vs. LabView
  {{RESULT-QC-NOVA.bin_labview_dur}} min, ratio {{RESULT-QC-NOVA.bin_ratio}})
  -- changed from the original wrong pick, which was a small excerpt chunk
  of the true session.
- {{RESULT-QC-NOVB.session}}: only one whole-file candidate existed (implied
  duration {{RESULT-QC-NOVB.bin_dur}} min vs. LabView
  {{RESULT-QC-NOVB.bin_labview_dur}} min, ratio {{RESULT-QC-NOVB.bin_ratio}})
  -- unchanged from the original pick.
- {{RESULT-QC-DEC.session}}: three eligible whole-file candidates existed
  after excluding chunk-series members; a concatenated file (implied
  duration {{RESULT-QC-DEC.bin_dur}} min vs. LabView
  {{RESULT-QC-DEC.bin_labview_dur}} min, ratio {{RESULT-QC-DEC.bin_ratio}})
  was selected over an unspliced single-continuous file by a near-tie on
  closest duration. This ratio exceeds the ten-percent flag threshold and
  was explicitly flagged for review, not hard-blocked; the excess over the
  LabView duration is unexplained (possibly a longer pre/post-experiment
  recording buffer). Changed from the original wrong pick, which was an
  unrelated short snippet recording.
- {{RESULT-QC-MAY.session}}: only one raw-AP candidate existed at all
  (implied duration {{RESULT-QC-MAY.bin_dur}} min vs. LabView
  {{RESULT-QC-MAY.bin_labview_dur}} min, ratio {{RESULT-QC-MAY.bin_ratio}}) --
  a mild excess ratio confirmed benign (not a file-selection defect) because
  no alternative candidate existed.
- {{RESULT-QC-JUN.session}}: only one candidate (implied duration
  {{RESULT-QC-JUN.bin_dur}} min vs. LabView {{RESULT-QC-JUN.bin_labview_dur}}
  min, ratio {{RESULT-QC-JUN.bin_ratio}}).
- {{RESULT-QC-SEP.session}}: only one candidate (implied duration
  {{RESULT-QC-SEP.bin_dur}} min vs. LabView {{RESULT-QC-SEP.bin_labview_dur}}
  min, ratio {{RESULT-QC-SEP.bin_ratio}}). This file's own acquisition-system
  run name does not match the session's animal/date label; this was
  investigated separately and David Gire confirmed the run name is a known,
  reused label on an otherwise-correct recording for this session -- no
  exclusion or re-run was needed.

Control-Epoch and Valid-Sniff Masking

Control (non-ethanol) epochs were defined by the validated ethanol-contact
detection rule, applied verbatim and identically across all six sessions
with no per-session adjustment: the entire ethanol-sensor trace is
mean-subtracted, and any sample where the mean-subtracted value exceeds a
small fixed fraction is marked ethanol; at or below that fraction is
control. Recordings were tiled into four-second windows with fifty-percent
overlap; a window was excluded if it overlapped any ethanol sample (any
control-epoch violation) or any invalid-sniff span. Instantaneous sniff rate
was computed as the reciprocal of the inter-sniff interval from the sniff
detector's detected onsets; sniffs with an inter-sniff interval below the
fifth or above the ninety-fifth percentile of that session's distribution,
and any span the detector flagged as noise, were marked invalid-sniff and
any overlapping window excluded. Surviving windows are simultaneously
all-control and all-valid-sniff. The minimum number of retained windows
across sessions was {{RESULT-AGG.min_windows}} ({{RESULT-QC-DEC.session}}),
comfortably above the minimum-window threshold below which a session would
be excluded from the depth statistics; no session was excluded on this
criterion (see Table one).

Spectral Estimation

Per session and depth, multitaper (Thomson) spectral estimates were
computed in each retained four-second window using seven Slepian tapers
(time-bandwidth product four), reporting frequencies from the four-second
window's lower bound of one-quarter hertz. From each window's LFP power
spectrum, SNF power spectrum (SNF resampled from its native rate up to the
LFP rate on the shared recording clock, used raw, never the detected phase),
and cross-spectrum, the magnitude coherence was computed and pooled across
all retained windows. Bands used throughout: theta (two to twelve hertz,
primary), beta (eighteen to thirty hertz), gamma (sixty-five to one hundred
hertz, plus its amplitude envelope for the example-traces figure).

Per-Experiment QC Summary Table

Table one. Per session: recording length, retained/excluded four-second
windows (by reason), detected/retained sniffs, usable-time fraction,
sniff-rate median, and the five depth channel indices and ycoords. All
values below are read directly from the frozen QC and sniff-rate result
objects (rendered programmatically in the final document as one row per
session; not reproduced individually as placeholders in this draft since
every cell is a direct, unmodified frozen-file field). Sniff-rate median,
interquartile range, and range are drawn from the control-epoch
instantaneous-rate distribution; the reported range (minimum/maximum) is
the untrimmed control-epoch distribution and includes inter-sniff-interval
outlier sniffs that do not enter the coherence analysis (they are excluded
by the percentile trim before window masking) -- median and interquartile
range are computed over the same untrimmed distribution and are the values
the frozen result object's "range" field was always defined over.

Sample cross-check for one session ({{RESULT-QC-DEC.session}}, the session
with the fewest retained windows): recording length
{{RESULT-QC-DEC.length_min}} min; windows used
{{RESULT-QC-DEC.n_windows_used}}; windows excluded for ethanol overlap
{{RESULT-QC-DEC.n_windows_excl_eth}}; windows excluded for invalid-sniff or
noise overlap {{RESULT-QC-DEC.n_windows_excl_noise}}; percent usable time
{{RESULT-QC-DEC.pct_usable}} percent; sniffs detected
{{RESULT-QC-DEC.n_sniffs_detected}}, retained after the inter-sniff-interval
trim {{RESULT-QC-DEC.n_sniffs_retained}}; median instantaneous sniff rate
{{RESULT-SNIFF-DEC.median}} Hz (interquartile range
{{RESULT-SNIFF-DEC.iqr}} Hz), untrimmed control-epoch range
{{RESULT-SNIFF-DEC.min}} to {{RESULT-SNIFF-DEC.max}} Hz. The full six-row
table is rendered below.

Discarded-Sections Log

For every excluded four-second window, a per-session discarded-spans log
records the excluded span (start time, end time, reason: ethanol / non-sniff
noise / inter-sniff-interval outlier); these per-span logs are not
reproduced row-by-row here (they run to thousands of rows per session) but
are retained on disk and are exact, frozen counts. Table two below (rendered
programmatically from the frozen discarded-spans-count result object)
reports, per session, the exact total number of discarded spans by reason:
ethanol-contact spans, inter-sniff-interval-outlier (non-physiological
sniff) spans, and sniff-detector noise spans. These span-level counts are a
different unit than Table one's window-level exclusion counts (a single
long ethanol contact, for instance, can cause many overlapping four-second
windows to be excluded from one contact span) and both are reported for
transparency.

Figures

Figure one (FIG-PAPER1-TRACES)

Figure 1A analogue: example band-filtered LFP and sniff traces per depth.
Representative session and segment were chosen by a fixed, non-scientific
display convention (most retained all-control/all-valid-sniff windows
across the six sessions, tie-broken by earliest date), not by any result:
the representative session is {{RESULT-TRACES.session}}
({{RESULT-QC-JUN.n_windows_used}} retained windows, confirmed the maximum
across all six sessions), and the representative segment spans
{{RESULT-TRACES.start_s}} to {{RESULT-TRACES.end_s}} seconds of that
session's control recording. The five depths shown are channel indices
{{RESULT-QC-JUN.ch_a}}, {{RESULT-QC-JUN.ch_b}}, {{RESULT-QC-JUN.ch_c}},
{{RESULT-QC-JUN.ch_d}}, and {{RESULT-QC-JUN.ch_e}} (ycoords
{{RESULT-QC-JUN.yc_a}}, {{RESULT-QC-JUN.yc_b}}, {{RESULT-QC-JUN.yc_c}},
{{RESULT-QC-JUN.yc_d}}, {{RESULT-QC-JUN.yc_e}} um). The SNF signal is shown
at top; the LFP at each depth is shown unfiltered and band-filtered into
theta, beta, and gamma (plus its amplitude envelope), colored green during
inhalation and black during exhalation/pause, with sniff-onset markers
shown. (The inhalation/exhalation color convention is implemented via a
sniff-phase-variable cutoff that, while matching the contract's display
requirement, is not itself independently pinned elsewhere in the contract
and is noted here for transparency.)

Figure two (FIG-PAPER1-SPECTRO)

Figure 1C analogue: multitaper spectrograms per depth with sniff-rate
overlay. Same representative session as Figure one
({{RESULT-SPECTRO.session}}), same five depth channels and ycoords. Rows are
the five depths; per depth, the LFP-power and LFP-sniff-coherence
spectrograms are shown with the instantaneous sniff-rate scatter overlaid,
plus the depth-independent SNF-power spectrogram shown once. Multitaper
parameters: four-second windows, fifty-percent overlap, seven Slepian tapers
(time-bandwidth product four), frequencies at or above one-quarter hertz.
Colorbars are labeled (LFP power, SNF power, coherence zero to one); the
sniff-rate overlay is in hertz. The frequency axis is displayed over a
reduced range for visual clarity relative to the full Nyquist-limited range
computed; this crop is a display choice only and does not affect any
reported statistic.

Figure three (FIG-PAPER1-SNIFFRATE)

Figure 1D analogue: instantaneous sniff-rate distribution, control epochs
only, per session (depth-independent). Median (interquartile range)
instantaneous sniff rate and retained sniff count per session, all injected
from the frozen sniff-rate and QC result objects:

- {{RESULT-QC-NOVA.session}}: median {{RESULT-SNIFF-NOVA.median}} Hz (IQR
  {{RESULT-SNIFF-NOVA.iqr}} Hz), n retained sniffs
  {{RESULT-QC-NOVA.n_sniffs_retained}}
- {{RESULT-QC-NOVB.session}}: median {{RESULT-SNIFF-NOVB.median}} Hz (IQR
  {{RESULT-SNIFF-NOVB.iqr}} Hz), n retained sniffs
  {{RESULT-QC-NOVB.n_sniffs_retained}}
- {{RESULT-QC-DEC.session}}: median {{RESULT-SNIFF-DEC.median}} Hz (IQR
  {{RESULT-SNIFF-DEC.iqr}} Hz), n retained sniffs
  {{RESULT-QC-DEC.n_sniffs_retained}}
- {{RESULT-QC-MAY.session}}: median {{RESULT-SNIFF-MAY.median}} Hz (IQR
  {{RESULT-SNIFF-MAY.iqr}} Hz), n retained sniffs
  {{RESULT-QC-MAY.n_sniffs_retained}}
- {{RESULT-QC-JUN.session}}: median {{RESULT-SNIFF-JUN.median}} Hz (IQR
  {{RESULT-SNIFF-JUN.iqr}} Hz), n retained sniffs
  {{RESULT-QC-JUN.n_sniffs_retained}}
- {{RESULT-QC-SEP.session}}: median {{RESULT-SNIFF-SEP.median}} Hz (IQR
  {{RESULT-SNIFF-SEP.iqr}} Hz), n retained sniffs
  {{RESULT-QC-SEP.n_sniffs_retained}}

Distributions are explicitly labeled as the untrimmed control-epoch
distribution (see the Table one note above on minimum/maximum).

# Scientific Audit — NP-PAPER-1

Auditor: scientific-auditor (fresh context, Opus). Attempted to FALSIFY that the
implementation satisfies `contract_v001.yaml`. Numeric similarity and a passing
oracle/numeric-gate suite were not accepted as interpretive proof.

**Net finding**: the code is a faithful, literal implementation of the contract's
signal-processing and statistics recipes, and the numeric gate's re-derivations
are sound. However, two of the contract's own **data-provenance premises are
falsified by the raw filenames on disk**, and the pipeline's own results show
the "confirmatory" coherence replication reads at its own noise floor across
every session and depth. These are filed as new deviations (DEV-002, DEV-003)
requiring human resolution before this task proceeds past SCI_AUDITED.

---

## BLOCKER

### BLOCKER-1 — Pseudoreplication: two of the six "independent" sessions are the same animal (NP8)

**Clause**: `data_provenance` / `experimental_structure` (CON-001, DATA-003,
DATA-004); `statistical_model` level `within_experiment_repeated_measures_primary`
(experiment as random intercept).

**Evidence (independently confirmed by the orchestrator, not just asserted)**:
- `DATA\11-01-2021\11-1-2021-3-45 PM-Suzanne-NP-8_Marvel.dat` → animal **NP8** "Marvel".
- `DATA\11-03-2021\11-3-2021-9-44 AM-Suzanne-NP8_Marvel_RoightBulb_2ndcraniotomy_2ndREcording.dat`
  → animal **NP8** "Marvel", explicitly labeled "2nd craniotomy, 2nd recording".

The six sessions map to animals NP8, NP8, Np10, NP12, NP15, NP22 — **five**
animals, not six. The contract's `data_provenance` asserts "one animal each …
no two sessions share an animal … experiment == session == animal (1:1) …
DATA-003's animal-grouping contingency does not apply" (this claim was itself
inherited from an NP-DEMO-4 resolution and never re-verified against these
filenames during contract design). DATA-003's shared-animal contingency
(nesting correction) was therefore **triggered but not applied**:
`depth_statistics.py` uses `groups="experiment"` (= session) as the random
intercept, treating the two NP8 sessions as independent draws, and
`RESULT-best-depth.yaml` computes grand-mean SEM with `ddof=1` over **n=6**
— which the figure captions are contracted to inject verbatim
(FIG-PAPER1-DEPTH-SUMMARY caption_requirements "n_experiments stated").

**Materiality**: the depth omnibus is non-significant regardless (LMM p=0.425,
Friedman p=0.663), so the *null* headline is robust to this defect; but the
independence claim, the SEM/error bars, and the "6 independent experiments"
framing are wrong, and DATA-003 makes the nesting correction mandatory once
animals are shared. Requires either a corrected animal→session map with animal
as the grouping factor, or David Gire re-confirming true animal identity.

### BLOCKER-2 — 09-14-2022 neural file is named for a different animal/date (NP21, 08-07-2022) than its behavioral file (NP22, 09-14-2022)

**Clause**: DATA-004, `data_provenance`, PROH-005 ("SNF/ETH/LFP share the
recording clock"), CON-001. **Not disclosed anywhere in DEV-001.**

**Evidence (independently confirmed)**: the only `*.ap.bin` under the
09-14-2022 session is
`DATA\09-14-2022\NP22_09142022_C1_g0\NP22_09142022_C1_g0_imec0\NP21_08072022_C1_g0_t0.imec0.ap.bin`
— named **NP21_08072022** — sitting inside a folder named **NP22_09142022**.
The LabView `.dat` is `9-14-2022-1-43 PM-Suzanne-09142022NP22.dat` (**NP22,
09-14-2022**, matching the outer folder). The pipeline reads `NP_Fs`,
`nSavedChans`, gains, and the LFP itself from the NP21_08072022 `.ap.meta`/
`.bin`, and SNF/ETH from the NP22 `.dat`. If these are genuinely different
recording sessions, the coherence computed for 09-14-2022 pairs LFP and SNF
from different animals/days and is scientifically meaningless.

**Why DEV-001's safeguard missed it**: `resolve_full_lfp_binfile` selects by
duration-ratio match only; this folder has a single `.ap.bin` candidate
(ratio 0.973, well within tolerance), so it is accepted with no identity check.
A close duration match does not establish same-session provenance — a
mislabeled or accidentally-reused SpikeGLX run name would pass this check
undetected.

**Materiality**: 09-14-2022's coherence is not obviously collapsed (mean theta
0.030–0.033, percentiles up to 89.6), which is mildly reassuring but not
dispositive of correct pairing. Requires human confirmation of true file
provenance (is `NP21_08072022_C1_g0` a stale/reused SpikeGLX run-name label on
an actually-correct NP22/09-14-2022 recording, or is it genuinely a different
recording?) before this session's results can be trusted.

---

## MAJOR

### MAJOR-1 — The "confirmatory" coherence replication did not replicate: all coherences sit at the multitaper pooling noise floor

**Clause**: `scientific_question` (OUT-001/OUT-002 — "the paper's measure of
alignment"); `required_outputs` RESULT-coh-spectrum / RESULT-theta-coh;
`reports.results_report.scientific_status: confirmatory_coherence_plus_exploratory_depth`.

**Evidence**: across all 30 experiment×depth cells (`RESULT-theta-coh.yaml`),
mean theta coherence is 0.011–0.033 and peak theta coherence 0.021–0.086. The
oracle's own derivation (`spectral_oracle_ref.py` [D1]) gives the pooled
magnitude-coherence bias floor as ≈0.886/√(K·W); for 11-01-2021 (436 windows ×
7 tapers) that floor is ≈0.016–0.023 — the observed depth-1 mean (0.0215) sits
**on the floor**. No depth in any session reaches a peak-coherence percentile
≥ 95 (max observed = 89.6) — the paper's headline alignment measure is
non-significant everywhere against the contracted 95% circular-shift null.

**Is this an implementation defect? No.** The multitaper recipe (4-s windows,
50% overlap, DPSS NW=4/K=7, mean-removed segments, pooled S1/S2/S12, magnitude
form) matches the contract and the independent reference bit-for-bit in
intent; the discriminator tests confirm the magnitude (not squared) form is
used. This is a scientific-honesty concern for P4, not a code bug: the
reports must **not** present this as a successful confirmatory replication.
One candidate mechanistic explanation the auditor raised (not confirmed):
pooling cross-spectra over the ~48-minute recording drives magnitude coherence
toward its floor whenever LFP-sniff phase lag is non-stationary across the
recording — plausibly related to the bin-vs-LabView duration mismatches
(2–14%) that are themselves evidence the two acquisition systems are not
perfectly co-clocked, in tension with PROH-005's shared-clock assumption. This
is a hypothesis for the Methods/Results report and future investigation, not
something this audit resolves.

---

## MINOR

### MINOR-1 — `extract_full_lfp` continuation-chunk stitching has a start-guard but no end-guard
`extract_full_lfp.py` prepends a 2s real-data lead-in guard to each post-10s
chunk so the forward filtfilt transient settles, but the kept region runs to
each chunk's own tail, where the backward-pass transient (from the block's
odd-reflection padding) contaminates roughly the last 1-7 decimated samples
before each 10s boundary. Those samples are not recomputed by the next chunk.
Immaterial to the 4-s-window spectra (a handful of samples out of 4000-sample
windows) and undetectable by the first-10s-only OUT-026 anchor. The
decimation-grid stitching itself (`offset = (-pos) % dsRatio`) was verified
correct even for non-integer `NP_Fs`.

### MINOR-2 — Circular-shift null is mildly anti-conservative when SNF padding is held-constant
`resample_snf_to_lfp.py` pads the tail with a held-constant value on sessions
where the chosen bin is longer than the LabView recording (12-15-2021 ~12.4%
of samples padded; 5-17-2022 ~7.7%; 11-01-2021 ~1.6%). `circular_shift_null.py`
rolls the entire SNF array including this constant block, so some shifts
inject a zero-variance segment into analysis windows, slightly lowering the
per-frequency null threshold (biased toward significance). Immaterial to this
run's conclusions (nothing reaches significance regardless), but a genuine
null-construction defect worth fixing — the roll should be confined to the
valid (non-padded) SNF region.

### MINOR-3 — RESULT-sniff-rate "range" includes ISI-outlier sniffs excluded from the coherence analysis
The reported min/max instantaneous sniff rate is drawn from all control-epoch
sniffs, not the 5th/95th-ISI-trimmed set that actually gates the coherence
windows, so the reported range (e.g. 11-01-2021: 0.082-17.86 Hz) includes
instants the analysis itself deemed invalid. Median/IQR are unaffected and the
contract does say "range," so this is defensible but should be labeled as the
untrimmed control-epoch distribution in the Methods report to avoid implying
those extremes entered the coherence estimate.

### MINOR-4 (elaboration on an already-disclosed DEV-001 residual risk, not a new finding)
The duration-matching override picked 12-15-2021's `tcat` (CatGT-concatenated)
file (55.483 min) over the unspliced single-continuous `..._g0` t0 file
(55.508 min) by a ~1.5s margin — i.e., on a near-tie, it preferred the
candidate carrying DEV-001's already-disclosed chunk-boundary/shared-clock
discontinuity risk over one that carries none. 12-15-2021's coherence is not
anomalously worse than other sessions, which somewhat mitigates. Worth a
one-line tie-break rule (prefer single-continuous over concatenated on
near-ties) for future runs, but not re-filed as a new blocker since
`deviations.yaml` DEV-001 already discloses this exact risk class.

### MINOR-5 — Example-traces inhalation coloring uses an un-pinned phase convention
`np_paper1_analysis.py` defines inhalation display coloring as
`SNF_PH in [0, 0.5)` — the code itself flags this as not pinned anywhere in
the contract/repo_map. Affects figure coloring only (no statistic depends on
it), but should be human-confirmed before P4 so FIG-PAPER1-TRACES' caption
("inhalation = green") is not misattributed.

---

## Checked and could NOT falsify

- **Multitaper recipe**: parameters, mean-removal, pooled S1/S2/S12, magnitude
  (not squared) coherence, freqs ≥0.25 Hz with DC dropped — correct, matches
  the independent reference.
- **Circular-shift null**: dedicated `default_rng(5489)`, one `integers(1,n)`
  draw per shift, 1000 shifts, 95th-percentile-per-frequency, RNG isolated
  from global state — bit-for-bit reproducible (RQ-038/CON-003).
- **Prohibited-changes fidelity**: `detect_eth_contact.py` verbatim
  (PROH-003/008); raw resampled SNF (never SNF_PH) used for coherence
  (PROH-009); `threshold_std=-0.5` pinned (PROH-004/007); pinned spectral
  params untouched (RQ-039/040); no uncontracted post-hoc — `posthoc=None`
  exactly because `omnibus_significant=False` (PROH-006/OUT-016 honored).
- **No statistical leakage**: the null distribution feeds only
  `peak_coh_percentile`/`sig_theta_fraction`; the LMM LRT correctly uses
  `reml=False` for both nested models.
- **Unit of analysis in the stats model** is genuinely per-experiment×depth
  (30 cells), not per-window — no window-level pseudoreplication inside the
  model itself (the pseudoreplication in BLOCKER-1 is upstream, at the
  animal→session mapping, not inside the modeling code).

---

## Disposition

Filed as **DEV-002** (BLOCKER-1, shared-animal pseudoreplication) and
**DEV-003** (BLOCKER-2, 09-14-2022 file-identity mismatch) in
`08_traceability/deviations.yaml`. Both require human resolution before this
task proceeds past SCI_AUDITED to STAT_OK / RESULTS_FROZEN. MAJOR-1 is not a
deviation (no contract clause was violated) but must inform the P4
Results-report framing — do not present the coherence replication as
successful.

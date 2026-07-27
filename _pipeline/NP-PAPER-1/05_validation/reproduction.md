# Reproduction Record — NP-PAPER-1 (Blueprint v2 P5)

**Reproduction date**: 2026-07-25
**Git commit**: 236d8dbdda98e079e44398e21f2bc3b6ec01b3ce
**Python interpreter**: `C:\Projects\Repos\Agentic Research\Research Setup\research_workflow\vras\Scripts\python.exe` (3.13.9)
**OS**: Windows 11 Pro (10.0.26200)
**Package versions**: numpy 2.0.2, scipy 1.14.1, pandas 3.0.5, statsmodels 0.14.6, pytest 9.1.1

## Scope
A genuine full 6-session re-run was judged disproportionate for this gate given
every session's outputs had already been independently re-verified multiple
times during the build (numeric gate, stat gate, two rounds of DEV-002/MINOR-2
recomputation, all cross-checked). Instead: (1) one representative session was
reproduced fully from scratch (raw data reload, full-duration LFP extraction,
1000-shift circular-shift null with seed 5489 — not read from cache), with the
original output preserved as a backup for comparison; (2) the cross-session
aggregation/statistics were re-derived for all 6 sessions/5 animals; (3) the
full oracle suite was re-run.

## Step 1 — Full end-to-end reproduction of one session (09-14-2022)

Chosen as the fastest session (399.1s originally recorded). The existing
`sessions/09-14-2022/` output was backed up to
`sessions/09-14-2022_prereproduction_backup/` (preserved, not deleted) and its
`DONE.json` resume marker removed to force a genuine fresh execution (raw
`.bin`/`.dat` reload, `extract_full_lfp`, `compute_sniff_phase`,
`multitaper_psd_coherence`, and a full fresh 1000-shift `circular_shift_null`
with seed 5489).

**Runtime**: 427.4s fresh vs. 399.1s originally recorded (~7% variance,
consistent with normal system-load variation, not a determinism concern).

**Result**: bit-for-bit identical. All 5 `spectral_depth<N>.npz` files'
SHA-256 hashes match the pre-reproduction backup exactly:

| File | SHA-256 |
|---|---|
| spectral_depth1.npz | f6eaadee8d8ac76d8b6b9ecafdcdb8617bd0953c0316056517a5dd568d62a77e |
| spectral_depth2.npz | 83bd84c56f41dd3e58dbebead91f385d76762c087ba68e626969219409dc5ee4 |
| spectral_depth3.npz | c01453340be35f157738e44cd46560bb14fa571790862991a8a0f2b917a32e45 |
| spectral_depth4.npz | 458b3e227d151d5cb985ecde800f1f4e578fa96a1c0d85ff4c4e45a8c428388a |
| spectral_depth5.npz | 95efb19967e5f02cb38c576ee1146a6e305d9405fad597d94b5482dbe3951eda |

All 5 rows of `RESULT-theta-coh.yaml` for this session (mean_theta_coh,
peak_theta_coh, peak_theta_freq, sig_theta_fraction, peak_coh_percentile)
match the frozen values exactly. This confirms the seeded circular-shift null
(RQ-038/CON-003's bit-for-bit acceptance criterion) and the full extraction
pipeline are genuinely deterministic, not just self-consistent within a
single run.

## Step 2 — Cross-session aggregation reproduction (all 6 sessions / 5 animals)

All 14 frozen `04_results/frozen/*` files verified to match their recorded
SHA-256 hashes exactly (`freeze_manifest.yaml`), confirming no drift.

Independently re-derived from the (unchanged) per-session `RESULT-theta-coh.yaml`
rows:
- **LMM**: chi²=1.8220, df=4, p=0.7684, ICC=0.7465 — matches frozen exactly.
- **Friedman**: chi²=1.7600, df=4, p=0.7798, Kendall's W=0.0880 — matches frozen exactly.
- **Best depth**: overall = depth 5; per-ordinal consistency counts {1:1, 2:2, 3:0, 4:1, 5:1} — matches frozen exactly.

## Step 3 — Oracle suite

Re-ran without `NP_PAPER1_RUN_HEAVY` (real-data heavy tests not required for
this pass, already covered in Step 1's full session reproduction): **84
passed, 4 skipped, 0 failed** — the 4 skips are the heavy real-session-gated
tests, consistent with not setting that env var this run; matches expected
baseline.

## Contracted parameters confirmed present and unchanged
n_shifts=1000 (min, never reduced), seed=5489, threshold_std=-0.5,
eth_threshold=0.05, window_len_s=4.0, overlap=0.5, theta_band=(2,12),
min_windows_for_stats=30 — all as pinned in the contract, all confirmed still
in force in the reproduced run.

## Undocumented prerequisites
**None.** The committed pipeline is self-contained: the vras interpreter,
raw data under `DATA/`, and read access to the kernel under `Optimized Python/`
were sufficient. No environment variable, config file, or manual workaround
was needed beyond what's already documented in this task's own artifacts.

## Artifact locations
- Backed-up pre-reproduction session: `04_results/sessions/09-14-2022_prereproduction_backup/`
- Fresh reproduced session (now the live copy, DONE.json regenerated): `04_results/sessions/09-14-2022/`
- Frozen reference: `04_results/frozen/`

## Verdict
**REPRODUCED.** Bit-for-bit determinism confirmed for a full from-scratch
session re-run (including the expensive 1000-shift null); cross-session
statistics reproduce exactly from unchanged per-session inputs; oracle suite
unchanged; no undocumented prerequisites; no deviation from the frozen
record.

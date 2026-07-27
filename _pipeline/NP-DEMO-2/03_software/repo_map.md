# NP-DEMO-2 Repository Map — CARTOGRAPHED state

**Task:** Sniff-phase locking of OB single-unit spikes to the SNF sensor signal (NOT LFP),
ethanol vs control, at animal level (paired Wilcoxon) and unit level (linear mixed-effects model).

**Kernel root:** `C:\Projects\Repos\Neuropixels\Optimized Python\`
**New code root:** `C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\03_software\`
**Golden fixtures:** `C:\Projects\Repos\Neuropixels\Golden Fixtures\`
**Data root:** `C:\Projects\Repos\Neuropixels\DATA\`

---

## 1. Validated Kernel Functions (reuse only, never edit)

### 1.1 `load_experiment_data` — `load_experiment_data.py`
**Signature:** `load_experiment_data(exp_dir: str | Path) -> dict`
- Locates `.dat`/`.bin`/`.meta`/Kilosort outputs via `lib.or_validate_files.or_validate_files(exp_dir)`
- Calls `_fix_ks_dir()` internally — finds `phy2.5PShank1/` subdirectory for 2022-09-14 multi-shank
- **D-dict outputs:** `SNF`, `ETH`, `LV_Fs`, `spikeTimes`, `sp` (dict: `st`, `clu`, `temps`), `unitIDs`, `unitDepths`, `unitFiringRate`, `unitAmps`, `xcoords`, `ycoords`, `NP_Fs`, `LFP`, `LFP_Fs`
- SNF and ETH sampled at `LV_Fs` (~125 Hz); spikes on SpikeGLX clock aligned to LV clock via the loader

### 1.2 `compute_sniff_phase` — `analyses/compute_sniff_phase.py`
**Signature:** `compute_sniff_phase(D: dict, threshold_std: float = -0.5) -> dict`
- Reads: `D['SNF']`, `D['LV_Fs']`
- **D-dict additions:** `SNF_filt`, `SNF_z`, `SNF_PH`, `sniff_onsets` (1-based), `sniff_onsets_s` (0-based/LV_Fs), `sniff_dur_s` (median cycle), `sniff_thr`
- `SNF_PH`: ramp 0.0–1.0 within each valid sniff cycle; **-1.0 sentinel outside** every cycle (= noise / non-sniffing)
- `threshold_std = -0.5` is **PINNED** — never adjust for any session (PROH-002 does NOT touch this)
- Golden fixture: `Golden Fixtures/08_sniff_phase/` — tolerance distributional 1e-6

### 1.3 `threshold_eth` — `analyses/threshold_eth.py`
**Signature:** `threshold_eth(D: dict, eth_threshold: float = 0.11) -> dict`
- Reads: `D['ETH']`
- **D-dict additions:** `ETH_thr` (floor-clipped at threshold), `eth_threshold`
- Default `0.11` is the FIRST-PASS value applied identically to all 6 sessions
- PROH-002 may change `eth_threshold` per session on the second pass — pass the adjusted value here
- Golden fixture: `Golden Fixtures/09_eth_threshold/` — tolerance exact

### 1.4 `compute_spike_phase` — `analyses/compute_spike_phase.py`
**Signature:** `compute_spike_phase(D: dict, optimized: bool | None = None) -> dict`
- Reads: `D['SNF_PH']`, `D['sniff_onsets']`, `D['spikeTimes']`, `D['sp']['clu']`, `D['unitIDs']`, `D['LV_Fs']`, `D['SNF']`
- **D-dict additions:** `spike_SNF_PH` (per-spike SNF phase), `unitMeanSniffPhase` (per-unit mean of valid phases; NaN if < 5 valid spikes)
- **Valid spike:** `spike_SNF_PH >= 0.0`; the `-1.0` sentinel marks spikes outside a valid sniff cycle
- Time to index: `lv_idx = matlab_round(t * LV_Fs)`, clamped to `[0, nLV-1]`; **always use `matlab_round` (half-away-from-zero), never Python `round`**
- Golden fixture: `Golden Fixtures/10_spike_phase/` — tolerance 1e-12

### 1.5 `compute_sniff_psth` — `analyses/compute_sniff_psth.py`
**Signature:** `compute_sniff_psth(D: dict, bin_ms: float = 10.0, use_ms: bool = False, *, attach: bool = True, optimized: bool | None = None)`
- Reads: `D['sniff_onsets_s']`, `D['sniff_dur_s']`, `D['spikeTimes']`, `D['sp']['clu']`, `D['unitIDs']`
- Window: `[0, sniff_dur_s]`; returns `(psth [nUnits x nBins], centers, n_events)`
- When `attach=True`: also writes `psth_phase`, `psth_ms`, `centers_phase`, `centers_ms`, `n_events` into a copy of D
- **Callers must pre-filter to valid-sniff sniff onsets only** before passing `sniff_onsets_s`
- Golden fixture: `Golden Fixtures/11_sniff_psth/` — tolerance rtol=1e-6, atol=1e-9

### 1.6 `plot_unit_locations` — `plot_unit_locations.py`
**Signature:** `plot_unit_locations(D: dict) -> matplotlib.figure.Figure`
- Reads: `D['unitIDs']`, `D['unitDepths']`, `D['unitFiringRate']`, `D['unitAmps']`, `D['xcoords']`, `D['ycoords']`, `D['NP_Fs']`, `D['sp']` (keys `st`, `temps`)
- Returns a 3-panel matplotlib Figure (probe map + depth histogram + FR distribution)
- Handles multi-shank probes via x-coordinate shank assignment (SHANK_BOUNDARIES = [-inf, -100, 100, 400, inf])
- For FIG-DEMO2-RES-EX, call once per example unit, passing a D-dict filtered/annotated to highlight that unit

---

## 2. Kernel Helpers

| Helper | Location | Purpose |
|---|---|---|
| `matlab_round(x)` | `analyses/_common.py` | Half-away-from-zero rounding — REQUIRED for time to index; never use `round()` |
| `or_validate_files(exp_dir)` | `lib/or_validate_files.py` | Locates `.dat`/`.bin`/`.meta`/Kilosort files; includes `_fix_ks_dir()` for `phy2.5PShank1/` subdirectory |
| `OPT.*` flags | `optconfig.py` | Toggle optimized vs baseline paths in kernel functions |

---

## 3. Session Pipeline (call order per session)

```
D = load_experiment_data(session_dir)          # stages 01-07; _fix_ks_dir for 2022-09-14
D = compute_sniff_phase(D, threshold_std=-0.5) # stage 08; -1 sentinel outside valid sniffs
D = threshold_eth(D, eth_threshold=eth_thr)    # stage 09; eth_thr = PROH-002 final value
D = compute_spike_phase(D)                     # stage 10; spike_SNF_PH; -1 = not in sniff
```

Then in new analysis code (03_software):
```
valid_mask = spike_SNF_PH >= 0.0              # discard -1 sentinel; log every excluded run
eth_mask   = ETH_thr[lv_idx(spike_times)] > eth_thr   # condition assignment (valid spikes only)
ctrl_mask  = ~eth_mask & valid_mask
# compute per-unit MRL and preferred phase for ethanol and control spikes separately
# compute compute_sniff_psth on valid sniff onsets for QC-PSTH and examples figures
```

---

## 4. New Code Required in 03_software\ (kernel does NOT provide these)

| Function | Purpose |
|---|---|
| `mrl_and_preferred_phase(phase_0to1)` | Circular MRL + preferred phase via `|mean(exp(2pi*i*phi))|` and `angle(...)` |
| `valid_sniff_mask(spike_SNF_PH)` | `spike_SNF_PH >= 0.0` boolean array |
| `classify_ethanol(spike_times_s, ETH_thr, LV_Fs, eth_threshold)` | `ETH_thr[matlab_round(t*LV_Fs)] > eth_threshold` |
| `unit_included(n_spikes, session_dur_s, n_valid_sniff_spikes)` | Drop if FR < 0.1 Hz OR < 50 valid-sniff spikes combined |
| `log_discard_runs(SNF_PH, LV_Fs, session_date)` | Find contiguous runs of `SNF_PH == -1`; return list of `{experiment, start_s, end_s, reason}` |
| `qc_counts(...)` | Per-session n_sniffs, n_neurons, n_trials, length_min, pct_usable_sniffs |
| `proh_002_adjustment(session_dirs)` | Two-pass ETH threshold: Pass 1 all at 0.11, flag outside +/-1 SD; Pass 2 grid 0.05 to 0.50 step 0.005 argmin |count-mean| |
| `paired_animal_test(eth_means, ctrl_means)` | `scipy.stats.wilcoxon` exact two-sided + rank-biserial r |
| `unit_level_lmm(unit_df)` | `statsmodels MixedLM`: condition (eth/ctrl) fixed effect, animal random intercept; returns standardized coefficient + p |
| `run_session(session_dir, eth_thr)` | Orchestrate one session: load -> sniff_phase -> threshold_eth -> spike_phase -> per-unit MRL/phase |
| `run_multi_session(...)` | PROH-002 loop, per-session run, aggregate results, PSTH extremes, animal + unit stats |

---

## 5. Hazard Catalog

| ID | Hazard | Mitigation |
|---|---|---|
| HAZ-01 | **-1 sentinel** `spike_SNF_PH == -1` marks spikes outside valid sniff cycles | Filter ALL phase/MRL/PSTH computations to `spike_SNF_PH >= 0.0`; log every excluded run to RESULT-qc-discards |
| HAZ-02 | **`matlab_round`** half-away-from-zero, not Python banker's rounding | Import and use `matlab_round` from `analyses._common` for every time-to-index conversion |
| HAZ-03 | **Pinned sniff threshold** `threshold_std = -0.5` must never change | Pass `-0.5` explicitly; no per-session variation; PROH-002 does not touch this |
| HAZ-04 | **Two-pass ETH ordering** (PROH-002): first-pass ALL sessions at 0.11 before any adjustment | Enforce in `proh_002_adjustment` — compute cross-session mean/SD before adjusting any session |
| HAZ-05 | **2022-09-14 multi-shank** Kilosort output in `phy2.5PShank1/` subdirectory | `load_experiment_data` handles via `_fix_ks_dir()`; no extra action needed |
| HAZ-06 | **2021-11-03 all-ethanol** — may have no control epochs at default 0.11 | PROH-002 must attempt rescue; document outcome in RESULT-eth-threshold-log; never silently drop |
| HAZ-07 | **Clock alignment** SNF/ETH sampled at LV_Fs; spikes at NP_Fs, both aligned by loader | Never resample; use `LV_Fs` for all time-to-index conversions on SNF/ETH arrays |
| HAZ-08 | **Pseudoreplication** many units per animal in unit-level test | Use LMM with animal as random intercept (statsmodels MixedLM) |
| HAZ-09 | **No LFP** PROH-001 forbids using LFP for phase | Phase always from `SNF_PH` (compute_sniff_phase on SNF); never reference `D['LFP']` in phase code |
| HAZ-10 | **Discards must be logged** not silently skipped | Every contiguous run of `SNF_PH == -1` -> RESULT-qc-discards entry `{experiment, start_s, end_s, reason}` |

---

## 6. Golden Fixture Stages Relevant to NP-DEMO-2

| Stage | Directory | Key files | Tolerance |
|---|---|---|---|
| 08 | `Golden Fixtures/08_sniff_phase/` | `SNF_PH.npy`, `sniff_onsets_s.npy`, `sniff_dur_s.npy`, `sniff_thr.npy` | distributional 1e-6 |
| 09 | `Golden Fixtures/09_eth_threshold/` | `ETH_thr.npy`, `eth_threshold.npy` | exact |
| 10 | `Golden Fixtures/10_spike_phase/` | `spike_SNF_PH.npy`, `unitMeanSniffPhase.npy` | 1e-12 |
| 11 | `Golden Fixtures/11_sniff_psth/` | `psth_phase.npy`, `psth_ms.npy`, `centers_phase.npy`, `centers_ms.npy`, `n_events.npy` | rtol=1e-6 atol=1e-9 |

---

## 7. Result Object Target Locations

All result objects written to `_pipeline/NP-DEMO-2/04_results/`:
- `RESULT-mrl-unit.yaml` / `.npy`
- `RESULT-phase-unit.yaml` / `.npy`
- `RESULT-mrl-animal.yaml`
- `RESULT-stat-animal.yaml`
- `RESULT-stat-unit.yaml`
- `RESULT-delta-mrl.yaml`
- `RESULT-eth-threshold-log.yaml`
- `RESULT-qc-counts.yaml`
- `RESULT-qc-discards.yaml`
- `RESULT-psth-extremes.yaml` / `.npy`
- `RESULT-psth-examples.yaml` / `.npy`
- `RESULT-unit-locations.yaml`

Figures to `_pipeline/NP-DEMO-2/06_figures/`.
Reports to `_pipeline/NP-DEMO-2/07_report/`.

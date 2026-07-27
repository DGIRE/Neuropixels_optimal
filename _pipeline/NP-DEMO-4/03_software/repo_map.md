# NP-DEMO-4 Repository Map

Generated: 2026-07-23 (repository-cartographer)

---

## 1. Kernel Functions (Read-Only — Never Edit)

| Path | Function | Signature | Role in NP-DEMO-4 | Reuse Status |
|------|----------|-----------|-------------------|--------------|
| `Optimized Python/load_experiment_data.py` | `load_experiment_data` | `(session_dir, ...) -> dict D` | Load SNF, ETH, TR, TS, sp.st, sp.clu, unitIDs, LV_Fs, unitFiringRate for each of 6 sessions | REUSE; apply `_fix_ks_dir()` for 2022-09-14 multi-shank |
| `Optimized Python/analyses/compute_sniff_phase.py` | `compute_sniff_phase` | `(D, threshold_std=-0.5) -> dict` | Onset detection via low-pass FIR + z-score + downward threshold crossing; needed for onset times fed to compute_sniff_rate | REUSE (onset detection logic mirrors compute_sniff_rate onset step) |
| `Optimized Python/analyses/_common.py` | `matlab_round` | `(x) -> np.ndarray` | Half-away-from-zero rounding for LV sample indexing | REUSE in new kernels |
| `Optimized Python/plot_unit_locations.py` | `plot_unit_locations` | `(D, ...) -> fig` | Optional: visualize unit positions on probe | OPTIONAL |

### Key data fields returned by load_experiment_data

- `D['SNF']` — sniff sensor trace (n_LV_samples,) float64, ~125 Hz
- `D['ETH']` — ethanol sensor trace (n_LV_samples,) float64, 0-1 normalized
- `D['TR']` — LabView trial number per sample (n_LV_samples,) int32
- `D['TS']` — within-trial timestamp per sample in ms (n_LV_samples,) float64
- `D['LV_Fs']` — LabView sample rate (float, ~125 Hz)
- `D['sp']['st']` — spike times in seconds (n_spikes,) float64
- `D['sp']['clu']` — cluster ID per spike (n_spikes,) int32
- `D['sp']['unitIDs']` — unique unit IDs (n_units,) int32
- `D['sp']['unitFiringRate']` — mean firing rate per unit (n_units,) float64

---

## 2. Reusable Assets

### detect_eth_contact.py (NP-DEMO-3 kernel — copy verbatim, do not modify)

**Source:** `_pipeline/NP-DEMO-3/03_software/detect_eth_contact.py`
**Copy to:** `_pipeline/NP-DEMO-4/03_software/detect_eth_contact.py`

Algorithm:
1. `ETH_ms = ETH - mean(ETH)` (mean of entire trace, no windowing)
2. `eth_contact_mask = ETH_ms > 0.05` (strictly above; at/below = control)
3. `n_trials_eth` = count of discrete contiguous above-threshold runs

Returns D copy with keys: `ETH_ms`, `eth_contact_mask`, `n_trials`, `eth_threshold`

**PROHIBITED:** Do not modify threshold, mean-subtraction, or contiguous-run logic.

---

## 3. Data Layout

Root: `C:\Projects\Repos\Neuropixels\DATA\`

| Session Date | Folder | Notes |
|---|---|---|
| 2021-11-01 | `11-01-2021\` | Standard single-shank |
| 2021-11-03 | `11-03-2021\` | Provisionally included; exclude if no ETH contacts |
| 2021-12-15 | `12-15-2021\` | Standard single-shank |
| 2022-05-17 | `5-17-2022\` | Standard single-shank |
| 2022-06-24 | `06-24-2022\` | Standard single-shank |
| 2022-09-14 | `09-14-2022\` | **Multi-shank — requires `_fix_ks_dir()`** |

Each session folder contains:
- `.dat` file — LabView sensor data (ETH, SNF, TR, TS channels)
- `imec0/` (or similar) — Kilosort spike-sorting output

---

## 4. Extension Points

New code lives **beside** the kernel, never inside `Optimized Python\`:

```
_pipeline/NP-DEMO-4/03_software/
    detect_eth_contact.py     <- verbatim copy from NP-DEMO-3
    compute_sniff_rate.py     <- NEW kernel (not yet implemented)
    compute_firing_rate_50ms.py  <- NEW kernel (not yet implemented)
    np_demo4_analysis.py      <- main analysis driver
    run_analysis_np_demo4.py  <- entry point / orchestrator
```

---

## 5. Hazards

| ID | Description | Mitigation |
|----|-------------|------------|
| **PROH-001** | Phase-lock / sniff rate from SNF ONLY, NEVER LFP | compute_sniff_rate reads D['SNF'], not D['LFP'] |
| **HAZARD-001** | 2022-09-14 multi-shank: Kilosort at wrong directory depth | Call `_fix_ks_dir()` before loading spikes |
| **HAZARD-002** | Valid-sniff filter: spike_SNF_PH >= 0.0 (only during valid sniff cycles) | For firing rate, only count spikes where SNF_PH >= 0 (note: NP-DEMO-4 uses 50ms window FR, not phase-locking; apply filter to exclude noise-section spikes) |
| **HAZARD-003** | Trial structure: odd TR only, first TS per trial corrupted | Select samples where TR is odd; set first TS per trial to 0 |
| **HAZARD-004** | ETH threshold: fixed 0.05 for ALL sessions | Never adjust per-experiment; use detect_eth_contact verbatim |
| **HAZARD-005** | Session 2021-11-03 provisional | Log and exclude if n_trials_eth == 0; fail if < 3 sessions remain |
| **HAZARD-006** | CON-003 statistical test | Use `scipy.stats.wilcoxon(mode='exact')`, rank-biserial r; n=6 animals |
| **HAZARD-007** | Unit inclusion: BOTH criteria required | FR >= 0.1 Hz AND total_spikes >= 5000 simultaneously |
| **HAZARD-008** | Pre-valve control: TS=0-10s only | CON-003 control window is TS 0-10000 ms, excluding any contact overlap |
| **HAZARD-009** | Raw ETH in Figure 3 | Figure 3 uses D['ETH'] BEFORE mean-subtraction (not ETH_ms) |
| **HAZARD-010** | First-run fixtures | First validated compute_sniff_rate / compute_firing_rate_50ms outputs become golden references |

---

## 6. New Kernels Required

### compute_sniff_rate(D: dict) -> dict

**Purpose:** Compute instantaneous sniff rate at each LV sample from SNF signal.

**Algorithm:**
1. Detect sniff onsets from D['SNF'] using same low-pass FIR + z-score + threshold method as compute_sniff_phase (threshold_std=-0.5, MIN_ISI rejection)
2. Compute inter-onset intervals (ISI) in seconds
3. Interpolate 1/ISI to each LV sample using nearest-neighbor extrapolation outside onset boundaries
4. Return D copy with D['sniff_rate'] as (n_LV_samples,) float64 in Hz

**Key constraint:** Must use SNF only (PROH-001). Uses `matlab_round` for sample indexing.

### compute_firing_rate_50ms(D: dict, included_unit_ids: np.ndarray, window_ms: float = 50.0) -> dict

**Purpose:** Compute population firing rate trace at LV sample resolution.

**Algorithm:**
1. For each included unit (from unit inclusion filter), count spikes in sliding window [t-25ms, t+25ms] / 0.05s at each LV time point
2. Average across all included units
3. Return D copy with D['firing_rate_population'] as (n_LV_samples,) float64 in Hz, plus D['firing_rate_per_unit'] as (n_included_units, n_LV_samples)

**Key constraint:** Window is exactly 50ms (contract-pinned). Do not use any other window size.

---

## 7. Oracle / Acceptance Criteria (summary)

| Result ID | Tolerance | Notes |
|---|---|---|
| RESULT-eth-mask | Exact binary match | Must match NP-DEMO-3 validated outputs |
| RESULT-unit-inclusion | Exact integers | n_units_recorded, n_units_included |
| RESULT-sniff-rate-matrix | 1e-4 Hz on fixture slice | First-run outputs become fixtures |
| RESULT-fr-per-trial | 1e-4 Hz on fixture slice | Deterministic sliding window |
| RESULT-eth-per-trial | 1e-6 on fixture slice | From raw D['ETH'] |
| RESULT-methods-table | Exact integers; 1e-4 for averages | Per-session table |
| RESULT-sniff-stat | Same significance decision; W exact; p within 1e-6; r within 1e-4 | Wilcoxon signed-rank, n=6 |

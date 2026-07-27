"""Independent numeric-gate verification for NP-PAPER-1 (Blueprint v2 gate 4).

Loads the actual RESULT-* artifacts in 04_results/ and re-derives/cross-checks
them from first principles -- does not simply trust the composing runner's own
summary. Writes a plain-text/JSON-ish report to stdout; the caller composes
05_validation/numeric_gate.yaml from this script's findings.

UPDATED 2026-07-25 for DEV-002 (scientific-audit BLOCKER-1, resolved by David
Gire 2026-07-25): 2021-11-01 and 2021-11-03 are the SAME animal (NP8); the
cross-session aggregate results (RESULT-best-depth, RESULT-depth-stat, grand-
mean spectra) are now computed at the ANIMAL level (5 animals, NP8's two
sessions averaged per depth before aggregation), not the raw 6-session level.
This gate independently re-derives the animal-level arithmetic from
RESULT-theta-coh.yaml (which remains session-level/unchanged) using its OWN
copy of the animal->session map (transcribed from deviations.yaml DEV-002's
resolution_note, not imported from the runner, to keep this a genuinely
independent check) -- it does not just re-check the runner's own recompute.
Also updated for the MINOR-2 circular_shift_null fix (same date, same
deviation record): RESULT-theta-coh's sig_theta_fraction/peak_coh_percentile
now reflect the corrected (non-padded-region-only) circular shift; this gate
re-verifies internal consistency of those fields post-fix (bounds, peak>=mean)
rather than re-deriving the null itself (that is the oracle suite's job).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

RESULTS = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-PAPER-1\04_results")

# DEV-002 resolved animal->session map (transcribed independently from
# deviations.yaml DEV-002's resolution_note -- NOT imported from the runner).
ANIMAL_MAP = {
    "11-01-2021": "NP8",
    "11-03-2021": "NP8",
    "12-15-2021": "Np10",
    "5-17-2022": "NP12",
    "06-24-2022": "NP15",
    "09-14-2022": "NP22",
}
N_ANIMALS = len(set(ANIMAL_MAP.values()))

checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})


qc = yaml.safe_load((RESULTS / "RESULT-qc-counts.yaml").read_text())
theta = yaml.safe_load((RESULTS / "RESULT-theta-coh.yaml").read_text())
best = yaml.safe_load((RESULTS / "RESULT-best-depth.yaml").read_text())
depth_stat = yaml.safe_load((RESULTS / "RESULT-depth-stat.yaml").read_text())
sniff_doc = yaml.safe_load((RESULTS / "RESULT-sniff-rate.yaml").read_text())
sniff = sniff_doc["rows"] if isinstance(sniff_doc, dict) and "rows" in sniff_doc else sniff_doc
discards = yaml.safe_load((RESULTS / "RESULT-qc-discards.yaml").read_text())

sessions = [r["session"] for r in qc]
check("all 6 contracted sessions present in RESULT-qc-counts",
      set(sessions) == {"11-01-2021", "11-03-2021", "12-15-2021", "5-17-2022", "06-24-2022", "09-14-2022"},
      str(sessions))

# ---- QC window arithmetic: used + excl_eth + excl_noise == candidate windows,
# and candidate windows must independently equal floor((n_lfp_samples-4000)/2000)+1
# from length_min. We don't have n_lfp_samples directly here, so check the
# internally-reported total against length_min-derived expectation instead.
for r in qc:
    total = r["n_windows_used"] + r["n_windows_excluded_ethanol"] + r["n_windows_excluded_noise"]
    n_lfp = int(round(r["length_min"] * 60.0 * 1000.0))
    expected_candidates = max(0, (n_lfp - 4000) // 2000 + 1)
    check(f"[{r['session']}] used+excl_eth+excl_noise == candidate windows from length_min",
          total == expected_candidates,
          f"total={total} expected_candidates={expected_candidates} (n_lfp={n_lfp}, length_min={r['length_min']:.3f})")
    check(f"[{r['session']}] n_windows_used >= 30 (PROH-002/CON-008 gate)",
          r["n_windows_used"] >= 30, f"n_windows_used={r['n_windows_used']}")
    check(f"[{r['session']}] excluded_from_depth_stats is False given >=30 windows",
          r["excluded_from_depth_stats"] is False)
    check(f"[{r['session']}] pct_usable_time in [0,100]",
          0.0 <= r["pct_usable_time"] <= 100.0, f"{r['pct_usable_time']}")
    check(f"[{r['session']}] n_sniffs_retained_isi <= n_sniffs_detected",
          r["n_sniffs_retained_isi"] <= r["n_sniffs_detected"],
          f"{r['n_sniffs_retained_isi']} <= {r['n_sniffs_detected']}")
    check(f"[{r['session']}] binfile_ratio within [0.5, 1.5] (sanity; DEV-001)",
          0.5 <= r["binfile_ratio"] <= 1.5, f"ratio={r['binfile_ratio']:.4f}")
    check(f"[{r['session']}] five distinct depth channel indices, ascending ycoord (tip->surface)",
          len(set(r["depth_channel_indices"])) == 5
          and list(r["depth_ycoords_um"]) == sorted(r["depth_ycoords_um"]),
          f"channels={r['depth_channel_indices']} ycoords={r['depth_ycoords_um']}")

# ---- theta-coh: bounds + internal consistency ----
theta_by_sess = {}
for row in theta:
    theta_by_sess.setdefault(row["session"], []).append(row)

for sess, rows in theta_by_sess.items():
    check(f"[{sess}] exactly 5 depth rows in RESULT-theta-coh",
          len(rows) == 5, f"n_rows={len(rows)}")
    for row in rows:
        ok_bounds = (0.0 <= row["mean_theta_coh"] <= 1.0
                     and 0.0 <= row["peak_theta_coh"] <= 1.0
                     and 2.0 <= row["peak_theta_freq"] <= 12.0
                     and 0.0 <= row["sig_theta_fraction"] <= 1.0
                     and (math.isnan(row["peak_coh_percentile"]) or 0.0 <= row["peak_coh_percentile"] <= 100.0)
                     and row["peak_theta_coh"] >= row["mean_theta_coh"] - 1e-9)
        check(f"[{sess}] depth {row['depth_ordinal']}: theta-coh scalar bounds sane",
              ok_bounds, str(row))

# ---- DEV-002 animal-level cross-check: collapse NP8's 2 sessions into one
# animal-level mean_theta_coh per depth (independently, from this script's OWN
# ANIMAL_MAP, not the runner's), THEN recompute best-depth/grand-mean/ranking/
# per-ordinal counts at the 5-animal level and compare to RESULT-best-depth. ----
check("RESULT-best-depth is animal-level post-DEV-002 (5 entries, n_animals==5)",
      "per_animal_best_depth" in best and len(best["per_animal_best_depth"]) == N_ANIMALS
      and best.get("n_animals") == N_ANIMALS and best.get("n_sessions") == len(ANIMAL_MAP),
      f"keys={list(best.keys())}, n_animals={best.get('n_animals')}")

animal_depth_vals: dict[str, dict[int, list[float]]] = {}
for sess, rows in theta_by_sess.items():
    animal = ANIMAL_MAP[sess]
    for row in rows:
        animal_depth_vals.setdefault(animal, {}).setdefault(
            row["depth_ordinal"], []).append(row["mean_theta_coh"])

animal_depth_mean = {
    animal: {d: float(np.mean(vals)) for d, vals in depth_map.items()}
    for animal, depth_map in animal_depth_vals.items()
}
recomputed_best_animal = {
    animal: max(depth_map, key=lambda d: depth_map[d])
    for animal, depth_map in animal_depth_mean.items()
}

for entry in best["per_animal_best_depth"]:
    animal = entry["animal"]
    check(f"[{animal}] best_depth_ordinal matches independent argmax of "
          "animal-averaged mean_theta_coh",
          recomputed_best_animal.get(animal) == entry["best_depth_ordinal"],
          f"recomputed={recomputed_best_animal.get(animal)} reported={entry['best_depth_ordinal']}")
    check(f"[{animal}] best_mean_theta_coh matches independent recompute (rtol 1e-9)",
          math.isclose(animal_depth_mean[animal][entry["best_depth_ordinal"]],
                       entry["best_mean_theta_coh"], rel_tol=1e-9, abs_tol=1e-12),
          f"recomputed={animal_depth_mean[animal][entry['best_depth_ordinal']]} "
          f"reported={entry['best_mean_theta_coh']}")
    check(f"[{animal}] sessions list matches ANIMAL_MAP",
          set(entry["sessions"]) == {s for s, a in ANIMAL_MAP.items() if a == animal},
          f"reported={entry['sessions']}")

# grand mean / SEM recompute per depth ordinal ACROSS THE 5 ANIMALS
grand = {}
for d in range(1, 6):
    vals = [animal_depth_mean[animal][d] for animal in animal_depth_mean]
    grand[d] = (float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals))), len(vals))

for d_str, g in best["grand_mean_theta_coh_by_depth"].items():
    d = int(d_str)
    rmean, rsem, rn = grand[d]
    check(f"grand_mean_theta_coh depth {d}: mean matches independent animal-level recompute (rtol 1e-9)",
          math.isclose(rmean, g["mean"], rel_tol=1e-9, abs_tol=1e-12),
          f"recomputed={rmean} reported={g['mean']}")
    check(f"grand_mean_theta_coh depth {d}: sem matches independent animal-level recompute (rtol 1e-9)",
          math.isclose(rsem, g["sem"], rel_tol=1e-9, abs_tol=1e-12),
          f"recomputed={rsem} reported={g['sem']}")
    check(f"grand_mean_theta_coh depth {d}: n_animals == {N_ANIMALS} (DEV-002, not 6 sessions)",
          rn == g["n_animals"] == N_ANIMALS, f"n={rn}")

# overall best depth / ranking recompute
overall_ranking = sorted(grand.keys(), key=lambda d: -grand[d][0])
check("overall_best_depth_ordinal matches independent animal-level grand-mean argmax",
      overall_ranking[0] == best["overall_best_depth_ordinal"],
      f"recomputed={overall_ranking[0]} reported={best['overall_best_depth_ordinal']}")
check("ranking_by_grand_mean_theta_coh matches independent animal-level recompute",
      overall_ranking == best["ranking_by_grand_mean_theta_coh"],
      f"recomputed={overall_ranking} reported={best['ranking_by_grand_mean_theta_coh']}")

# per-ordinal consistency counts recompute (out of N_ANIMALS, not 6 sessions)
counts = {d: 0 for d in range(1, 6)}
for animal, d in recomputed_best_animal.items():
    counts[d] += 1
check(f"per_ordinal_consistency_counts matches independent animal-level recompute (sums to {N_ANIMALS})",
      counts == {int(k): v for k, v in best["per_ordinal_consistency_counts"].items()}
      and sum(counts.values()) == N_ANIMALS,
      f"recomputed={counts} reported={best['per_ordinal_consistency_counts']}")

# ---- DEV-002 arithmetic spot-check: NP8's animal-level value at each depth
# must equal the mean of its 2 sessions' mean_theta_coh at that depth,
# independently re-derived straight from the session-level RESULT-theta-coh
# rows (not from the runner's own animal-table code). ----
np8_sessions = [s for s, a in ANIMAL_MAP.items() if a == "NP8"]
check("NP8 maps to exactly its 2 known sessions (2021-11-01, 2021-11-03)",
      set(np8_sessions) == {"11-01-2021", "11-03-2021"}, str(np8_sessions))
for d in range(1, 6):
    per_session_vals = [
        next(r["mean_theta_coh"] for r in theta_by_sess[s] if r["depth_ordinal"] == d)
        for s in np8_sessions
    ]
    expected_np8 = float(np.mean(per_session_vals))
    check(f"NP8 depth {d}: animal-level mean_theta_coh == mean of its 2 sessions "
          "(rtol 1e-9)",
          math.isclose(expected_np8, animal_depth_mean["NP8"][d], rel_tol=1e-9, abs_tol=1e-12),
          f"per_session={per_session_vals} expected_mean={expected_np8} "
          f"got={animal_depth_mean['NP8'][d]}")

# ---- depth_statistics sanity ----
lmm = depth_stat["lmm"]
fr = depth_stat["friedman"]
check("lmm.chisq >= 0", lmm["chisq"] >= 0, str(lmm["chisq"]))
check("lmm.df == 4", lmm["df"] == 4)
check("0 <= lmm.pvalue <= 1", 0.0 <= lmm["pvalue"] <= 1.0)
check("0 <= lmm.icc <= 1", 0.0 <= lmm["icc"] <= 1.0)
check("friedman.statistic >= 0", fr["statistic"] >= 0)
check("friedman.df == 4", fr["df"] == 4)
check("0 <= friedman.pvalue <= 1", 0.0 <= fr["pvalue"] <= 1.0)
check("0 <= kendalls_w <= 1", 0.0 <= fr["kendalls_w"] <= 1.0)
check("omnibus_significant flag consistent with p-values (alpha=0.05)",
      depth_stat["omnibus_significant"] == (lmm["pvalue"] < 0.05 or fr["pvalue"] < 0.05),
      f"lmm.p={lmm['pvalue']} friedman.p={fr['pvalue']} flag={depth_stat['omnibus_significant']}")
check("posthoc is null since omnibus not significant (PROH-006/OUT-016: no uncontracted post-hoc)",
      (depth_stat["posthoc"] is None) == (not depth_stat["omnibus_significant"]),
      f"posthoc={depth_stat['posthoc']} omnibus_significant={depth_stat['omnibus_significant']}")
check("RESULT-depth-stat carries n_animals==5/n_sessions==6 provenance post-DEV-002",
      depth_stat.get("n_animals") == N_ANIMALS and depth_stat.get("n_sessions") == len(ANIMAL_MAP),
      f"n_animals={depth_stat.get('n_animals')} n_sessions={depth_stat.get('n_sessions')}")

# ---- STRONGEST cross-check: independently rebuild the 5-animal x 5-depth
# (n=25) long-form table from RESULT-theta-coh.yaml (this script's OWN
# ANIMAL_MAP) and re-run the actual depth_statistics() function (the same
# validated, oracle-tested function the runner calls) -- the fit is
# deterministic, so a from-scratch re-run must reproduce RESULT-depth-stat.yaml
# bit-for-bit. This is the "re-verify the arithmetic against the new 5-row
# structure the same way it was verified for 6 before" check. ----
import sys as _sys  # noqa: E402
_SOFTWARE_DIR = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-PAPER-1\03_software")
if str(_SOFTWARE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SOFTWARE_DIR))
import pandas as _pd  # noqa: E402
from depth_statistics import depth_statistics as _depth_statistics  # noqa: E402

_rows = []
for _animal, _depth_map in animal_depth_mean.items():
    for _d, _v in _depth_map.items():
        _rows.append({"experiment": _animal, "depth": f"D{_d}", "mean_theta_coh": _v})
_recomputed_depth_stat = _depth_statistics(_pd.DataFrame(_rows))

check("depth_statistics() re-run from scratch on the independently-rebuilt "
      "5-animal x 5-depth table reproduces RESULT-depth-stat.yaml's LMM "
      "chisq/pvalue bit-for-bit",
      math.isclose(_recomputed_depth_stat["lmm"]["chisq"], lmm["chisq"], rel_tol=0, abs_tol=1e-9)
      and math.isclose(_recomputed_depth_stat["lmm"]["pvalue"], lmm["pvalue"], rel_tol=0, abs_tol=1e-9),
      f"recomputed={_recomputed_depth_stat['lmm']} reported={lmm}")
check("depth_statistics() re-run reproduces RESULT-depth-stat.yaml's Friedman "
      "statistic/pvalue/Kendall's W bit-for-bit",
      math.isclose(_recomputed_depth_stat["friedman"]["statistic"], fr["statistic"], rel_tol=0, abs_tol=1e-9)
      and math.isclose(_recomputed_depth_stat["friedman"]["pvalue"], fr["pvalue"], rel_tol=0, abs_tol=1e-9)
      and math.isclose(_recomputed_depth_stat["friedman"]["kendalls_w"], fr["kendalls_w"], rel_tol=0, abs_tol=1e-9),
      f"recomputed={_recomputed_depth_stat['friedman']} reported={fr}")

# ---- discards log: RESULT-qc-discards.yaml is a per-session INDEX (session,
# file, n_eth_spans, n_isi_outlier_spans, n_noise_spans) pointing at a detailed
# per-session file with the actual span-level (start_s, end_s, reason) log
# (OUT-032/RQ-033). Verify both the index shape and that the referenced files
# are real, well-formed, and their span counts match the index.
VALID_REASONS = {"ethanol", "isi_outlier", "isi-outlier", "ISI outlier",
                  "noise", "non-sniff-noise", "non-sniff noise"}
for i, d in enumerate(discards):
    idx_ok = ("session" in d and "file" in d and "n_eth_spans" in d
              and "n_isi_outlier_spans" in d and "n_noise_spans" in d)
    check(f"discard index row {i} ({d.get('session', '?')}) has session/file/counts",
          idx_ok, str(d))
    if not idx_ok:
        continue
    detail_path = RESULTS / d["file"]
    detail_ok = detail_path.is_file()
    check(f"[{d['session']}] referenced qc_discards detail file exists",
          detail_ok, str(detail_path))
    if not detail_ok:
        continue
    detail = yaml.safe_load(detail_path.read_text())
    spans = detail.get("discards", [])
    span_ok = all(("start_s" in s and "end_s" in s and "reason" in s
                    and s["end_s"] >= s["start_s"] and s["reason"] in VALID_REASONS)
                   for s in spans)
    check(f"[{d['session']}] all {len(spans)} discard spans well-formed "
          f"(start_s<=end_s, valid reason)", span_ok,
          "" if span_ok else str([s for s in spans
                                   if not (s.get("end_s", -1) >= s.get("start_s", 0)
                                           and s.get("reason") in VALID_REASONS)][:3]))
    n_eth = sum(1 for s in spans if s["reason"] == "ethanol")
    n_isi = sum(1 for s in spans if s["reason"] in ("isi_outlier", "isi-outlier", "ISI outlier"))
    n_noise = sum(1 for s in spans if s["reason"] in ("noise", "non-sniff-noise", "non-sniff noise"))
    check(f"[{d['session']}] index n_eth_spans matches detail file count",
          n_eth == d["n_eth_spans"], f"detail={n_eth} index={d['n_eth_spans']}")
    check(f"[{d['session']}] index n_isi_outlier_spans matches detail file count",
          n_isi == d["n_isi_outlier_spans"], f"detail={n_isi} index={d['n_isi_outlier_spans']}")
    check(f"[{d['session']}] index n_noise_spans matches detail file count",
          n_noise == d["n_noise_spans"], f"detail={n_noise} index={d['n_noise_spans']}")
check("RESULT-qc-discards index covers all 6 sessions",
      {d.get("session") for d in discards} == set(sessions), f"{[d.get('session') for d in discards]}")

# ---- sniff-rate sanity ----
for r in sniff:
    check(f"[{r.get('session','?')}] sniff-rate median within physiological range (0.5-15 Hz)",
          0.5 <= r.get("median_hz", r.get("median", -1)) <= 15.0
          if ("median_hz" in r or "median" in r) else True,
          str(r))

n_fail = sum(1 for c in checks if not c["ok"])
n_pass = sum(1 for c in checks if c["ok"])
print(json.dumps({"n_pass": n_pass, "n_fail": n_fail, "checks": checks}, indent=2))

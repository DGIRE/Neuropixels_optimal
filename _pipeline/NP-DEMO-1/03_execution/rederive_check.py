"""
rederive_check.py -- INDEPENDENT RE-DERIVER for NP-DEMO-1 (Blue-team).

Re-derives the delicate/headline numerics of np_demo1_analysis.py from FIRST
PRINCIPLES, importing NONE of the code under test for the independent path,
and confirms three-way agreement:

    (1) independent re-derivation (this file, from scratch)
    (2) the implementation under test (np_demo1_analysis)
    (3) SciPy/NumPy reference (circmean/circstd, wilcoxon, rankdata)

Stages: A (MRL + preferred phase), C (classify_ethanol / MATLAB-round),
D (paired rank-biserial r), with B (valid-sniff mask) used as input filter.

Tolerances (from the task contract):
    Stage A : 1e-12 absolute
    Stage C : exact boolean match
    Stage D : 1e-10 absolute
"""
import os
import sys
import numpy as np
import scipy.stats

FIX = r"C:\Projects\Repos\Neuropixels\Golden Fixtures"
IMPL_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-1\03_software"

TOL_A = 1e-12
TOL_D = 1e-10


def L(rel):
    return np.load(os.path.join(FIX, rel)).ravel()


# ==========================================================================
# INDEPENDENT re-derivations (no import of the code under test)
# ==========================================================================
def rd_mrl_preferred(phase_0to1):
    """z = mean(exp(i*2pi*phase)); return |z|, angle(z)."""
    p = np.asarray(phase_0to1, dtype=np.float64).ravel()
    if p.size == 0:
        return float("nan"), float("nan")
    z = np.mean(np.exp(1j * 2.0 * np.pi * p))
    return float(np.abs(z)), float(np.angle(z))


def rd_matlab_round(x):
    """Manually-written half-away-from-zero round."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def rd_classify_ethanol(spike_times_s, ETH_thr, LV_Fs, eth_threshold=0.11):
    st = np.asarray(spike_times_s, dtype=np.float64).ravel()
    thr = np.asarray(ETH_thr, dtype=np.float64).ravel()
    idx = rd_matlab_round(st * LV_Fs).astype(np.int64)
    idx = np.clip(idx, 0, thr.size - 1)
    return thr[idx] > eth_threshold


def rd_rank_biserial(ethanol_means, control_means):
    """Independent rank-biserial via a DIFFERENT algebraic route.

    Uses r = 1 - 2*W_min/T (magnitude) where W_min = min(W+, W-) matches
    scipy's two-sided statistic, and recovers the sign separately from
    sign(W+ - W-). Also returns W+, W-, statistic for cross-checks.
    """
    d = np.asarray(ethanol_means, dtype=np.float64).ravel() - \
        np.asarray(control_means, dtype=np.float64).ravel()
    nz = d[d != 0]
    if nz.size == 0:
        return dict(r=float("nan"), w_plus=0.0, w_minus=0.0, stat=float("nan"))
    ranks = scipy.stats.rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    T = w_plus + w_minus                      # == n*(n+1)/2 for no ties/zeros
    w_min = min(w_plus, w_minus)
    magnitude = 1.0 - 2.0 * w_min / T          # |r| via the alternate formula
    sign = 1.0 if (w_plus - w_minus) > 0 else (-1.0 if (w_plus - w_minus) < 0 else 0.0)
    r = sign * magnitude
    return dict(r=float(r), w_plus=w_plus, w_minus=w_minus, stat=float(w_min))


# ==========================================================================
# SciPy circular-stats reference (separate path for Stage A)
# ==========================================================================
def scipy_circ_mean_mrl(phase_0to1):
    """circmean over [0, 2pi) gives preferred phase; MRL from circstd via
    R = exp(-circstd^2 / 2) using scipy's definition circstd = sqrt(-2 ln R).
    Preferred phase mapped to (-pi, pi]."""
    ang = np.asarray(phase_0to1, dtype=np.float64).ravel() * 2.0 * np.pi
    if ang.size == 0:
        return float("nan"), float("nan")
    cmean = scipy.stats.circmean(ang, high=2 * np.pi, low=0.0)
    cstd = scipy.stats.circstd(ang, high=2 * np.pi, low=0.0)
    mrl = float(np.exp(-(cstd ** 2) / 2.0))
    pref = float(np.angle(np.exp(1j * cmean)))   # wrap to (-pi, pi]
    return mrl, pref


# ==========================================================================
# Load fixture data (single session == one animal)
# ==========================================================================
spike_SNF_PH = L("10_spike_phase/spike_SNF_PH.npy")
spike_times = L("05_spikes_ks/sp_st.npy")
clu = L("05_spikes_ks/sp_clu.npy").astype(np.int64)
ETH_thr = L("09_eth_threshold/ETH_thr.npy")
LV_Fs = float(L("01_labview/LV_Fs.npy")[0])
eth_threshold = float(L("09_eth_threshold/eth_threshold.npy")[0])
unitIDs = L("07_units/unitIDs.npy").astype(np.int64)

print(f"Loaded: {spike_SNF_PH.size} spikes, {unitIDs.size} units, "
      f"LV_Fs={LV_Fs}, eth_thr={eth_threshold}")

# Import the implementation under test (for path 2 only).
sys.path.insert(0, IMPL_DIR)
import np_demo1_analysis as impl   # noqa: E402


def report(name, indep, result, ref, tol):
    da = abs(indep - result)
    db = abs(indep - ref)
    dc = abs(result - ref)
    worst = max(da, db, dc)
    ok = worst <= tol
    print(f"\n[{name}]")
    print(f"  independent : {indep!r}")
    print(f"  implementation: {result!r}")
    print(f"  scipy/numpy ref: {ref!r}")
    print(f"  |indep-impl|={da:.3e}  |indep-ref|={db:.3e}  |impl-ref|={dc:.3e}")
    print(f"  worst pairwise diff={worst:.3e}  tol={tol:.0e}  -> {'PASS' if ok else 'FAIL'}")
    return ok, worst


all_ok = True

# ==========================================================================
# STAGE C first (needed to build per-condition phase sets for Stage A)
# ==========================================================================
valid = spike_SNF_PH >= 0.0                      # Stage B (input filter)

indep_eth = rd_classify_ethanol(spike_times, ETH_thr, LV_Fs, eth_threshold)
impl_eth = impl.classify_ethanol(spike_times, ETH_thr, LV_Fs, eth_threshold)

# scipy/numpy reference for classify: use numpy.rint won't match (banker's),
# so build the reference from the explicit half-away-from-zero definition but
# via a different construction (add/subtract path) to avoid sharing code.
def ref_classify(st, thr, fs, t0):
    x = np.asarray(st, dtype=np.float64) * fs
    # half-away-from-zero: floor(x+0.5) for x>=0, ceil(x-0.5) for x<0
    r = np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int64)
    r = np.clip(r, 0, thr.size - 1)
    return thr[r] > t0
ref_eth = ref_classify(spike_times, ETH_thr, LV_Fs, eth_threshold)

c_ok1 = np.array_equal(indep_eth, impl_eth)
c_ok2 = np.array_equal(indep_eth, ref_eth)
print("\n[Stage C: classify_ethanol -- exact boolean match]")
print(f"  n_ethanol independent={int(indep_eth.sum())}  "
      f"impl={int(impl_eth.sum())}  ref={int(ref_eth.sum())}  of {indep_eth.size}")
print(f"  indep==impl: {c_ok1}   indep==ref: {c_ok2}   "
      f"-> {'PASS' if (c_ok1 and c_ok2) else 'FAIL'}")
all_ok &= (c_ok1 and c_ok2)

# ==========================================================================
# STAGE A: MRL + preferred phase, on real per-unit/per-condition phase sets
# ==========================================================================
# Reproduce the runner's grouping: for each unit, ethanol/control valid phases.
print("\n[Stage A: mrl_and_preferred_phase]")
print("  Testing on real per-unit x per-condition valid sniff-phase sets...")

worst_mrl = 0.0
worst_pref = 0.0
tested = 0
first_details = []
for uid in unitIDs:
    u_mask = clu == uid
    for cond_sel in (indep_eth, ~indep_eth):
        sel = u_mask & valid & cond_sel
        phases = spike_SNF_PH[sel]
        if phases.size < 50:          # focus on the sets the runner would use
            continue
        i_mrl, i_pref = rd_mrl_preferred(phases)
        m_mrl, m_pref = impl.mrl_and_preferred_phase(phases)
        r_mrl, r_pref = scipy_circ_mean_mrl(phases)
        # phase diffs computed as circular (wrap-safe) distances
        dpref_im = abs(np.angle(np.exp(1j * (i_pref - m_pref))))
        dpref_ir = abs(np.angle(np.exp(1j * (i_pref - r_pref))))
        dmrl = max(abs(i_mrl - m_mrl), abs(i_mrl - r_mrl), abs(m_mrl - r_mrl))
        dpref = max(dpref_im, dpref_ir)
        worst_mrl = max(worst_mrl, dmrl)
        worst_pref = max(worst_pref, dpref)
        tested += 1
        if len(first_details) < 3:
            first_details.append((int(uid), phases.size, i_mrl, m_mrl, r_mrl,
                                  i_pref, m_pref, r_pref))

for uid, n, imr, mmr, rmr, ipr, mpr, rpr in first_details:
    print(f"  unit {uid:3d} n={n:6d}  MRL indep/impl/ref = "
          f"{imr:.15f}/{mmr:.15f}/{rmr:.15f}")
    print(f"             preferred(rad) indep/impl/ref = "
          f"{ipr:+.15f}/{mpr:+.15f}/{rpr:+.15f}")

a_ok = (worst_mrl <= TOL_A) and (worst_pref <= TOL_A)
print(f"  sets tested={tested}")
print(f"  worst 3-way MRL diff       = {worst_mrl:.3e}  tol={TOL_A:.0e}")
print(f"  worst 3-way preferred diff = {worst_pref:.3e}  tol={TOL_A:.0e}")
print(f"  -> {'PASS' if a_ok else 'FAIL'}")
all_ok &= a_ok

# ==========================================================================
# STAGE D: paired rank-biserial r  (single fixture = 1 animal, so we build
# a controlled multi-animal paired dataset to exercise the statistic, AND we
# test both directions of the sign.)
# ==========================================================================
print("\n[Stage D: paired_animal_test rank-biserial r]")

rng = np.random.default_rng(20260721)


def compare_stageD(eth, ctrl, label):
    global all_ok
    indep = rd_rank_biserial(eth, ctrl)["r"]
    res = impl.paired_animal_test(np.asarray(eth), np.asarray(ctrl))
    m_r = res["effect_size"]
    m_dir = res["direction"]
    # scipy/numpy reference: signed rank-biserial from the canonical
    # (W+ - W-)/(W+ + W-) definition, built here from scipy.rankdata only.
    d = np.asarray(eth, float) - np.asarray(ctrl, float)
    nz = d[d != 0]
    ranks = scipy.stats.rankdata(np.abs(nz))
    wp = ranks[nz > 0].sum(); wm = ranks[nz < 0].sum()
    ref_r = (wp - wm) / (wp + wm)
    ok, worst = report(f"Stage D r [{label}]", indep, m_r, ref_r, TOL_D)
    # sign check vs true direction
    true_sign = np.sign(np.median(d))
    r_sign = np.sign(m_r)
    sign_ok = (true_sign == r_sign) or true_sign == 0
    print(f"    median(eth-ctrl)={np.median(d):+.4f} -> expect sign "
          f"{'+' if true_sign>0 else '-' if true_sign<0 else '0'}; "
          f"r={m_r:+.4f} dir='{m_dir}'  sign {'OK' if sign_ok else 'WRONG'}")
    all_ok &= ok and sign_ok
    return m_r


# Direction 1: ethanol > control  (expect r > 0, direction ethanol>control)
eth1 = np.array([0.30, 0.42, 0.55, 0.38, 0.61, 0.47, 0.52, 0.44])
ctrl1 = np.array([0.20, 0.35, 0.40, 0.31, 0.50, 0.33, 0.41, 0.30])
compare_stageD(eth1, ctrl1, "ethanol>control")

# Direction 2: control > ethanol  (expect r < 0)
compare_stageD(ctrl1, eth1, "control>ethanol")

# Direction 3: random mixed differences (general check)
base = rng.uniform(0.2, 0.6, size=10)
eth3 = base + rng.normal(0, 0.05, size=10)
ctrl3 = base + rng.normal(0, 0.05, size=10)
compare_stageD(eth3, ctrl3, "mixed-random")

# Direction 4: derive a real paired dataset from the fixture itself by
# splitting units into two halves as pseudo-animals, using their ethanol vs
# control per-unit MRLs -> exercises the exact runner path on real numbers.
eth_mrls, ctrl_mrls = [], []
for uid in unitIDs:
    u_mask = clu == uid
    e_sel = u_mask & valid & indep_eth
    c_sel = u_mask & valid & (~indep_eth)
    if e_sel.sum() >= 50 and c_sel.sum() >= 50:
        eth_mrls.append(rd_mrl_preferred(spike_SNF_PH[e_sel])[0])
        ctrl_mrls.append(rd_mrl_preferred(spike_SNF_PH[c_sel])[0])
eth_mrls = np.array(eth_mrls); ctrl_mrls = np.array(ctrl_mrls)
print(f"\n  Real-fixture paired set: n_units={eth_mrls.size} "
      f"(per-unit ethanol vs control MRL)")
if eth_mrls.size >= 2:
    compare_stageD(eth_mrls, ctrl_mrls, "real-fixture-MRL")

# ==========================================================================
print("\n" + "=" * 60)
print(f"OVERALL VERDICT: {'ALL PASS -- three-way agreement' if all_ok else 'FAIL -- disagreement detected'}")
print("=" * 60)
sys.exit(0 if all_ok else 1)

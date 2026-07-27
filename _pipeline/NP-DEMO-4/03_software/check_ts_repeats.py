"""Check if TS repeats within TR=1 sub-trial."""
import sys
sys.path.insert(0, r"C:\Projects\Repos\Neuropixels\Optimized Python")
from lib.or_validate_files import or_validate_files
from load_experiment_data import load_experiment_data
import numpy as np

sess = r"C:\Projects\Repos\Neuropixels\DATA\11-01-2021"
files, _ = or_validate_files(sess, strict=False)
D = load_experiment_data(files)
TR = np.asarray(D['TR']).ravel()
TS = np.asarray(D['TS']).ravel()
LV_Fs = float(D['LV_Fs'])

# Sub-trial 1 of TR=1 (after the reset at position 764)
mask1 = TR == 1
ts1 = TS[mask1].astype(np.float64)
ts_diffs = np.diff(ts1)
reset_pos = int(np.flatnonzero(ts_diffs < -100)[-1]) + 1  # last reset
sub_ts = ts1[reset_pos:]  # sub-trial 1
n_sub = len(sub_ts)

print(f"Sub-trial 1: {n_sub} samples")
print(f"TS range: {sub_ts[0]} - {sub_ts[-1]} ms")
print(f"Expected samples at 125 Hz for {sub_ts[-1]/1000:.1f}s: {int(sub_ts[-1]/1000 * LV_Fs)}")

# Check for repeats
ts_sub_diffs = np.diff(sub_ts)
n_repeats = np.sum(ts_sub_diffs == 0)
n_neg = np.sum(ts_sub_diffs < 0)
n_large_pos = np.sum(ts_sub_diffs > 20)  # > 20ms jump
n_normal = np.sum((ts_sub_diffs >= 5) & (ts_sub_diffs <= 12))

print(f"\nTS diff stats:")
print(f"  Repeats (diff=0): {n_repeats}")
print(f"  Negative (backward): {n_neg}")
print(f"  Large positive (>20ms): {n_large_pos}")
print(f"  Normal (5-12ms): {n_normal}")
print(f"  Total diffs: {len(ts_sub_diffs)}")
print(f"\nFirst 20 TS diffs: {ts_sub_diffs[:20].astype(int).tolist()}")
print(f"\nTS at 1000-1020: {sub_ts[1000:1020].astype(int).tolist()}")

# Unique TS values
unique_ts = np.unique(sub_ts)
print(f"\nUnique TS values: {len(unique_ts)}")
print(f"Total samples: {n_sub}")
print(f"Duplication factor: {n_sub / len(unique_ts):.2f}x")

# Check structure of TR values more broadly
print(f"\n--- All TR periods ---")
for tr_val in sorted(np.unique(TR))[:10]:
    mask = TR == tr_val
    ts_v = TS[mask].astype(np.float64)
    n = mask.sum()
    uniq = len(np.unique(ts_v))
    print(f"  TR={tr_val}: {n} samples, {uniq} unique TS, dup={n/uniq:.1f}x, "
          f"TS {ts_v.min():.0f}-{ts_v.max():.0f} ms, duration={n/LV_Fs:.1f}s")

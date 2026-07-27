"""Check TR/TS structure in the first session to understand trial segmentation."""
import sys
sys.path.insert(0, r"C:\Projects\Repos\Neuropixels\Optimized Python")

from lib.or_validate_files import or_validate_files
from load_experiment_data import load_experiment_data
import numpy as np

sess = r"C:\Projects\Repos\Neuropixels\DATA\11-01-2021"
files, missing = or_validate_files(sess, strict=False)
D = load_experiment_data(files)
TR = np.asarray(D['TR']).ravel()
TS = np.asarray(D['TS']).ravel()
LV_Fs = float(D['LV_Fs'])

print(f"LV_Fs: {LV_Fs}")
print(f"Total samples: {TR.size}")

# Check TR=1 in detail
mask1 = TR == 1
idx1 = np.where(mask1)[0]
ts1 = TS[mask1]
print(f"\nTR=1: {mask1.sum()} samples, span {mask1.sum()/LV_Fs:.1f} s")
print(f"  TS min={ts1.min()}, max={ts1.max()}")
print(f"  TS step (first 20 diffs): {np.diff(ts1[:21]).tolist()}")
print(f"  TS large jumps (negative, = resets): {np.sum(np.diff(ts1) < -100)} resets")
neg_jumps = np.where(np.diff(ts1) < -100)[0]
print(f"  Reset positions (first 5): {neg_jumps[:5].tolist()}")
if len(neg_jumps) > 0:
    print(f"  TS at reset[0]: {ts1[neg_jumps[0]]} -> {ts1[neg_jumps[0]+1]}")

# Check how TS resets break into sub-trials
# A sub-trial starts when TS goes negative (reset) or at the very start
print(f"\nSub-trial structure in TR=1:")
sub_starts = [0] + list(neg_jumps + 1)
sub_ends = list(neg_jumps + 1) + [len(ts1)]
for i, (s, e) in enumerate(zip(sub_starts[:5], sub_ends[:5])):
    sub = ts1[s:e]
    print(f"  Sub-trial {i}: {e-s} samples ({(e-s)/LV_Fs:.1f} s), "
          f"TS {sub[0]} - {sub[-1]} ms = {(sub[-1]-sub[0])/1000:.1f} s")

# Check if the first sample corruption is a large TS value
print(f"\nFirst few TS values in TR=1: {ts1[:5]}")
print(f"Expected 0, got {ts1[0]} (corrupted first sample)")

# Check TR=2 (even, inter-trial)
mask2 = TR == 2
ts2 = TS[mask2]
print(f"\nTR=2 (even): {mask2.sum()} samples, TS {ts2.min()} - {ts2.max()}")
print(f"  Duration: {mask2.sum()/LV_Fs:.1f} s")

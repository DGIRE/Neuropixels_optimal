"""Check trial length after TS-reset fix."""
import sys
sys.path.insert(0, r"C:\Projects\Repos\Neuropixels\Optimized Python")
sys.path.insert(0, r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-4\03_software")

from lib.or_validate_files import or_validate_files
from load_experiment_data import load_experiment_data
from np_demo4_analysis import _segment_odd_tr_trials
import numpy as np

sess = r"C:\Projects\Repos\Neuropixels\DATA\11-01-2021"
files, _ = or_validate_files(sess, strict=False)
D = load_experiment_data(files)

trials = _segment_odd_tr_trials(D)
print(f"Number of trials: {len(trials)}")
print(f"\nFirst 5 trials:")
for t in trials[:5]:
    idx = t['global_idx']
    ts_s = t['ts_s']
    print(f"  TR={t['tr_value']}: {t['n_samples']} samples "
          f"({t['n_samples']/125.0:.1f} s), "
          f"TS {ts_s[0]:.2f} - {ts_s[-1]:.2f} s")

print(f"\nMax trial length: {max(t['n_samples'] for t in trials)} samples "
      f"({max(t['n_samples'] for t in trials)/125.0:.1f} s)")
print(f"Min trial length: {min(t['n_samples'] for t in trials)} samples")

# Check TR=1 TS values more carefully
TR = np.asarray(D['TR']).ravel()
TS = np.asarray(D['TS']).ravel()
mask1 = TR == 1
ts1 = TS[mask1]
idx1 = np.where(mask1)[0]

# Find all resets
ts_diffs = np.diff(ts1)
reset_pos = np.flatnonzero(ts_diffs < -100)
print(f"\nTR=1: {mask1.sum()} samples total")
print(f"TS resets at positions: {reset_pos.tolist()}")
print(f"Sub-trials:")
sub_starts = [0] + list(reset_pos + 1)
sub_ends = list(reset_pos + 1) + [len(ts1)]
for i, (s, e) in enumerate(zip(sub_starts, sub_ends)):
    sub = ts1[s:e]
    print(f"  Sub-trial {i}: {e-s} samples ({(e-s)/125.0:.1f} s), "
          f"TS {sub[0]} - {sub[-1]} ms")

print(f"\nExpected: trials should be ~40s (5000 samples at 125 Hz)")
print(f"TS should go 0 to 40000 ms")
print(f"Actual sub-trial 1: TS goes 1 to {ts1[reset_pos[-1]+1:].max()} ms")
print(f"  -> {ts1[reset_pos[-1]+1:].max()/1000:.1f} s duration")
print(f"  -> samples: {len(ts1) - (reset_pos[-1]+1)}")

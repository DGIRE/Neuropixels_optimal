"""Diagnostic: check TS values in the first session to understand trial structure."""
import sys
import os
sys.path.insert(0, r"C:\Projects\Repos\Neuropixels\Optimized Python")

from lib.or_validate_files import or_validate_files
from load_experiment_data import load_experiment_data
import numpy as np

# Check the first session
sess = r"C:\Projects\Repos\Neuropixels\DATA\11-01-2021"
files, missing = or_validate_files(sess, strict=False)
D = load_experiment_data(files)
TR = np.asarray(D['TR']).ravel()
TS = np.asarray(D['TS']).ravel()
LV_Fs = float(D['LV_Fs'])
print(f"LV_Fs: {LV_Fs}")
print(f"TR unique values (first 20): {sorted(np.unique(TR))[:20]}")
print(f"TR unique values (last 10): {sorted(np.unique(TR))[-10:]}")
print(f"n TR unique: {len(np.unique(TR))}")
print(f"TS dtype: {TS.dtype}")
print(f"TS range: {TS.min()} to {TS.max()}")

# Check first odd TR trial
odd_trs = sorted(np.unique(TR[TR % 2 == 1]))
print(f"\nOdd TR values (first 10): {odd_trs[:10]}")
tr1 = odd_trs[0]
mask = TR == tr1
ts_trial = TS[mask]
print(f"\nFirst odd TR value: {tr1}")
print(f"  n_samples: {mask.sum()}")
print(f"  TS values first 10: {ts_trial[:10]}")
print(f"  TS values last 10: {ts_trial[-10:]}")
print(f"  TS min: {ts_trial.min()}, max: {ts_trial.max()}")
print(f"  Duration at {LV_Fs} Hz: {mask.sum()/LV_Fs:.1f} s")
if ts_trial.max() > 0:
    print(f"  Duration from TS (if ms): {ts_trial.max()/1000.0:.1f} s")

# Check second odd TR trial
if len(odd_trs) > 1:
    tr2 = odd_trs[1]
    mask2 = TR == tr2
    ts2 = TS[mask2]
    print(f"\nSecond odd TR value: {tr2}")
    print(f"  n_samples: {mask2.sum()}")
    print(f"  TS first 5: {ts2[:5]}")
    print(f"  TS last 5: {ts2[-5:]}")
    print(f"  TS max: {ts2.max()}")

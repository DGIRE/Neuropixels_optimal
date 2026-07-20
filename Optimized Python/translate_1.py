"""
translate_1.py
--------------
Python translation of Translate-1.m (MATLAB).

Orchestrates the full Neuropixels import pipeline for one experiment session:

  1. Parses ``File paths.txt`` to locate experiment data.
  2. Validates that all required files are present (or_validate_files).
  3. Loads LabView sensor data, LFP sample, and Kilosort spike data
     (load_experiment_data).
  4. Plots unit locations on the probe and saves the figure (plot_unit_locations).
  5. Exports all numeric arrays to a compressed ``.npz`` file.

Usage
-----
    python translate_1.py

Configuration
-------------
Edit the four constants below (PATHS_FILE, OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_PATH)
before running if your paths differ from the defaults.

``File paths.txt`` must contain one ``Key: value`` pair per line::

    Directory:       <experiment root folder>
    DAT File:        <LabView .dat filename>   (informational; auto-discovered)
    Kilosort folder: <Kilosort output folder>  (informational; auto-discovered)
    ETH threshold:   <numeric value>           (optional — stored in D)

MATLAB → Python mapping
-----------------------
* ``addpath()``        → ``sys.path.insert()`` (module level)
* struct D             → ``dict`` D
* ``fopen`` / ``fgetl``→ ``open()`` / file iteration
* ``strtrim``          → ``str.strip()``
* ``strfind(s, ':')``  → ``str.partition(':')``
* ``isnan()``          → ``np.isnan()``
* ``length()``         → ``len()``
* ``save(…, '-v7.3')`` → ``np.savez_compressed()``
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')       # non-interactive backend -- must precede pyplot import
import matplotlib.pyplot as plt

# Ensure the package root (this file's directory) is on sys.path so that
# sibling modules (load_experiment_data, plot_unit_locations, lib.*) import
# correctly regardless of the working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from load_experiment_data import load_experiment_data   # noqa: E402
from plot_unit_locations import plot_unit_locations     # noqa: E402
from lib.or_validate_files import or_validate_files     # noqa: E402

# ---------------------------------------------------------------------------
# CONFIGURATION  (mirrors the hard-coded constants in Translate-1.m)
# Change these four paths before running if your layout differs.
# ---------------------------------------------------------------------------
PATHS_FILE     = r'C:\Projects\Neuropixels\File paths.txt'
OUTPUT_DIR     = r'C:\Projects\Neuropixels\Python'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
OUTPUT_NPZ     = os.path.join(OUTPUT_DIR, 'MTLB-TEST_python.npz')
FIGURE_PATH    = os.path.join(OUTPUT_DIR, 'unit_locations.png')


# ---------------------------------------------------------------------------
# PARSE File paths.txt
# ---------------------------------------------------------------------------

def parse_paths_file(paths_file: str) -> tuple[str, str, str, float]:
    """Parse File paths.txt and return (exp_dir, dat_file, ks_dir, eth_threshold).

    Format: one 'Key: value' pair per non-blank, non-comment line.
    Lines starting with '#' are skipped.  Keys are matched by exact string
    equality (case-sensitive, same as MATLAB switch/case).

    MATLAB -> Python mapping
    ------------------------
    fopen / fgetl / feof   -> open() / iteration over file object
    strtrim(raw)           -> raw.strip()
    colonIdx = strfind(…)  -> str.partition(':')
    str2double(val)        -> float(val)
    isnan(ethThreshold)    -> np.isnan(eth_threshold)

    Returns
    -------
    exp_dir       : str   -- experiment root directory  ('Directory' key)
    dat_file      : str   -- LabView .dat filename      ('DAT File' key)
    ks_dir        : str   -- Kilosort output folder     ('Kilosort folder' key)
    eth_threshold : float -- ETH threshold, or nan      ('ETH threshold' key)
    """
    if not os.path.isfile(paths_file):
        print(f'ERROR: Cannot find File paths.txt at:\n  {paths_file}',
              file=sys.stderr)
        sys.exit(1)

    exp_dir       = ''
    dat_file      = ''
    ks_dir        = ''
    eth_threshold = float('nan')

    with open(paths_file, 'r', encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            # str.partition splits on the FIRST ':' only — equivalent to MATLAB
            # strfind(line, ':') then taking colonIdx(1).
            if ':' not in line:
                continue
            key, _, val = line.partition(':')
            key = key.strip()
            val = val.strip()

            if key == 'Directory':
                exp_dir = val
            elif key == 'DAT File':
                dat_file = val
            elif key == 'Kilosort folder':
                ks_dir = val
            elif key == 'ETH threshold':
                try:
                    eth_threshold = float(val)
                except ValueError:
                    pass   # leave as nan; mirrors MATLAB str2double returning NaN

    return exp_dir, dat_file, ks_dir, eth_threshold


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print('================================================')
    print('  Neuropixels -- Translate-1 Import & Export')
    print('================================================\n')

    # ------------------------------------------------------------------
    # PARSE FILE PATHS.TXT
    # ------------------------------------------------------------------
    print('--- Reading File paths.txt ---')

    exp_dir, dat_file, ks_dir, eth_threshold = parse_paths_file(PATHS_FILE)

    print(f'Experiment folder : {exp_dir}')
    print(f'DAT file          : {dat_file}')
    print(f'Kilosort folder   : {ks_dir}')
    if not np.isnan(eth_threshold):
        print(f'ETH threshold     : {eth_threshold:.4f}')

    # ------------------------------------------------------------------
    # STEP 1 -- BUILD FILES DICT & VALIDATE
    # or_validate_files() auto-discovers .dat, .bin, and Kilosort output
    # (replaces MATLAB dir() + isfile() + listdlg() calls).
    # ------------------------------------------------------------------
    print('\n--- Step 1: Validating files ---')

    # Guard: experiment directory must exist (mirrors MATLAB error() calls)
    if not exp_dir or not os.path.isdir(exp_dir):
        print(f'ERROR: Experiment directory not found:\n  {exp_dir}\n'
              'Check File paths.txt.', file=sys.stderr)
        sys.exit(1)

    # or_validate_files returns (files_dict, missing_list).
    # files keys: 'datFile', 'datPath', 'binFile', 'binPath', 'ksDir'
    files, missing = or_validate_files(exp_dir)

    if missing:
        print('ERROR: Required files not found:', file=sys.stderr)
        for m in missing:
            print(f'  - {m}', file=sys.stderr)
        sys.exit(1)

    print('All required files found.')
    print(f"  .dat  : {os.path.join(files['datPath'], files['datFile'])}")
    print(f"  .bin  : {os.path.join(files['binPath'], files['binFile'])}")
    print(f"  ksDir : {files['ksDir']}")

    # ------------------------------------------------------------------
    # STEP 2 -- LOAD EXPERIMENT DATA
    # ------------------------------------------------------------------
    print('\n--- Step 2: Loading experiment data ---')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    D = load_experiment_data(files, checkpoint_dir=CHECKPOINT_DIR)

    if not np.isnan(eth_threshold):
        D['eth_threshold_default'] = eth_threshold

    print(f"Data loaded: {len(D['unitIDs'])} valid units, {len(D['ETH'])} LV samples")

    # ------------------------------------------------------------------
    # STEP 3 -- PLOT UNIT LOCATIONS
    # ------------------------------------------------------------------
    print('\n--- Step 3: Plotting unit locations ---')

    fig = plot_unit_locations(D)
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {FIGURE_PATH}')

    # ------------------------------------------------------------------
    # FINAL CHECKPOINT DUMP
    # Saves key arrays so the Python output can be compared against
    # the MATLAB checkpoints produced during Task 2.
    # ------------------------------------------------------------------
    for var in ['ETH', 'SNF', 'unitIDs', 'unitDepths', 'unitFiringRate']:
        if var in D:
            np.save(
                os.path.join(CHECKPOINT_DIR, f'chk_final_{var}.npy'),
                np.asarray(D[var]),
            )
    if 'LFP' in D:
        np.save(
            os.path.join(CHECKPOINT_DIR, 'chk_final_LFP.npy'),
            D['LFP'],
        )

    # ------------------------------------------------------------------
    # EXPORT D  (MATLAB equivalent: save(out_path, 'D', '-v7.3'))
    # Only numeric arrays and plain scalars are saved — dict/str fields
    # (D['meta'], D['binFile'], etc.) are skipped; np.savez cannot
    # serialise arbitrary Python objects.
    # ------------------------------------------------------------------
    print('\n--- Exporting D to MTLB-TEST_python.npz ---')

    saveable = {
        k: np.asarray(v)
        for k, v in D.items()
        if isinstance(v, (np.ndarray, float, int))
    }
    np.savez_compressed(OUTPUT_NPZ, **saveable)
    print(f'Saved: {OUTPUT_NPZ}')

    print('\n================================================')
    print('  Import complete.')
    print('  Variable D contains all results.')
    print('  Data exported to MTLB-TEST_python.npz')
    print('================================================')


if __name__ == '__main__':
    main()

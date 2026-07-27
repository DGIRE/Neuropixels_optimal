"""RESULTS_FROZEN step (gate 6, deterministic) for NP-PAPER-1.

Freezes the accepted RESULT-* objects into 04_results/frozen/ so that all
downstream figure/report numbers are injected ONLY from these frozen copies
(D11), never recomputed or hand-typed. Snapshotted AFTER NUMERIC_OK,
SCI_AUDITED (with DEV-002/DEV-003 resolved), and STAT_OK all passed.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import yaml

RESULTS = Path(r"C:\Projects\Repos\Neuropixels\_pipeline\NP-PAPER-1\04_results")
FROZEN = RESULTS / "frozen"
FROZEN.mkdir(exist_ok=True)

RESULT_FILES = [
    "RESULT-best-depth.yaml",
    "RESULT-depth-stat.yaml",
    "RESULT-example-traces.npz",
    "RESULT-example-traces_meta.yaml",
    "RESULT-example-spectrogram.npz",
    "RESULT-example-spectrogram_meta.yaml",
    "RESULT-grand-mean-spectra.npz",
    "RESULT-qc-counts.yaml",
    "RESULT-qc-discards.yaml",
    "RESULT-sniff-rate.yaml",
    "RESULT-sniff-rate-raw.npz",
    "RESULT-sniff-rate-raw_meta.yaml",
    "RESULT-theta-coh.yaml",
]

manifest = {"task_id": "NP-PAPER-1", "frozen_files": []}
for name in RESULT_FILES:
    src = RESULTS / name
    dst = FROZEN / name
    shutil.copy2(src, dst)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    manifest["frozen_files"].append({"file": name, "sha256": sha, "bytes": src.stat().st_size})

(FROZEN / "freeze_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
print(f"Froze {len(RESULT_FILES)} result files to {FROZEN}")
for f in manifest["frozen_files"]:
    print(f"  {f['file']}  sha256={f['sha256'][:16]}...  ({f['bytes']} bytes)")

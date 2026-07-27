"""
render_common.py -- shared plotting style/utilities for NP-DEMO-2 figure
renderers (Blueprint v2 P3, D7, FIGURES_RENDERED step).

Additive, presentation-only module living in 06_figures (beside, never
inside, the kernel). It reads the FROZEN figdata_FIG-DEMO2-*.yaml/.npy files
under 06_figures/figure_data (themselves built from frozen RESULT-* objects
by build_figdata.py) and provides shared style constants + a save helper.
No scientific numerics (MRL, p-values, effect sizes, PSTHs, ...) are
recomputed here -- every plotted value is read verbatim from the frozen
figdata files.

Quality requirements enforced here (contract_spec.yaml `quality:` block,
applies to every figure): vector output (PDF) + 300 DPI raster (PNG),
colorblind-safe condition colors, minimum 7 pt font, no axis truncation.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

try:
    from yaml import CSafeLoader as _YamlLoader
except ImportError:  # pragma: no cover - falls back if libyaml unavailable
    from yaml import SafeLoader as _YamlLoader

FIGDATA_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\06_figures\figure_data"
OUT_DIR = r"C:\Projects\Repos\Neuropixels\_pipeline\NP-DEMO-2\06_figures"

# Colorblind-safe (Okabe-Ito) condition colors, per contract visual_encoding
# {ethanol: solid, control: dashed} -- distinguishable by both hue and
# linestyle/marker so the mapping survives grayscale printing too.
COLOR_ETHANOL = "#E69F00"  # orange, solid
COLOR_CONTROL = "#56B4E9"  # sky blue, dashed
COLOR_DEV001 = "#D55E00"   # vermillion, used only for DEV-001 flags/markers

LS_ETHANOL = "-"
LS_CONTROL = "--"

MIN_FONT = 8  # contract quality.min_font = 7pt; 8 gives margin after bbox scaling
RASTER_DPI = 300

plt.rcParams.update({
    "font.size": MIN_FONT,
    "axes.titlesize": MIN_FONT + 1,
    "axes.labelsize": MIN_FONT,
    "xtick.labelsize": MIN_FONT,
    "ytick.labelsize": MIN_FONT,
    "legend.fontsize": MIN_FONT,
    "figure.dpi": 100,
    "pdf.fonttype": 42,  # embed real (searchable) text in the PDF, not curves
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def load_yaml(name: str) -> dict:
    """Load one frozen figdata_FIG-DEMO2-*.yaml file (fast libyaml loader)."""
    path = os.path.join(FIGDATA_DIR, name)
    with open(path, "r") as f:
        return yaml.load(f, Loader=_YamlLoader)


def savefig(fig, stem: str, dpi: int = RASTER_DPI):
    """Save fig as both PNG (raster_dpi) and PDF (vector) under OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, stem + ".png")
    pdf_path = os.path.join(OUT_DIR, stem + ".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def fmt_p(p: float) -> str:
    """Exact-p formatting: scientific notation for very small p, else 4 sig figs."""
    if p == 0:
        return "0"
    if p < 1e-4:
        return f"{p:.3e}"
    return f"{p:.4g}"

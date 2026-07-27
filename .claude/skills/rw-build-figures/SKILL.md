---
name: rw-build-figures
description: Build figure-data from result objects and render figures with a pure renderer, honoring any non-binding sketch. Use in P4 after results are frozen.
---

# rw-build-figures -- Figures from result objects (P4, D11)

1. Figure-data builder (deterministic) -> per-panel data files from the FROZEN
   result objects. The renderer recomputes nothing scientific.
2. Pure renderer draws from figure-data + figure contract only; test: the figure
   regenerates identically. Record any sketch in `influenced_by` (provenance).
3. **figure-auditor** (fresh for Amber/Red): plotted values match source; axes /
   exclusions / annotations honest; sketch changed presentation only. A dishonest
   transform implied by a sketch is FLAGGED, and the honest contract rule wins.

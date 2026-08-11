# DreamX-World — reproducible experiments

## E1 — Bidirectional → causal few-step conversion (the "make it streaming" move)
- **Setup:** take a small pretrained bidirectional video diffusion model (or train
  a tiny one on a toy dataset). Distill it into a 2–4-step causal autoregressive
  predictor with a DMD-style objective. Compare generation speed and per-frame
  quality against the original full-diffusion pass.
- **Shows:** the cost/quality trade of the central "causal + few-step" conversion.
- **Feasibility:** medium (needs a small diffusion model + distillation loop).

## E2 — Geometry-aware attention vs plain positional encoding (the E-PRoPE thesis)
- **Setup:** on a deterministic toy 3D scene rendered from known camera poses,
  compare a model that adds camera-geometry-conditioned positional encoding
  (feeding relative pose / a PRoPE-like term) against vanilla positional encoding.
  Sweep a camera orbit; measure how accurately generated views match ground-truth
  renders as pose changes.
- **Shows:** whether encoding *real camera geometry* improves controllability —
  E-PRoPE minus the token-reduction optimization.
- **Feasibility:** medium.

## E3 — Geometry-guided retrieval vs rolling window (revisit test) — **runnable now**
- **Setup:** drive a camera out-and-back. Compare conditioning on the last N
  frames vs **retrieving** the earlier frame(s) whose camera frustum overlaps the
  current view (geometry-based lookup), optionally adding a residual-correction
  head. Measure identity/appearance error between the first and second visits to
  each location.
- **Shows:** how much non-local geometric *retrieval* (+ residual recycling) buys
  over a plain sliding window for scene persistence.
- **Run (abstract version):** the shared lab models retrieval as the "anchor"
  memory and residual recycling as a correction gain —
  `python -m drift_lab.run --strategy anchor --revisit`

See [`../../experiments/drift_lab/`](../../experiments/drift_lab/). The lab's
"anchor" strategy is a stand-in for geometry-indexed retrieval: it re-grounds the
prediction whenever a location is revisited, which is exactly what camera-geometry
lookup enables.

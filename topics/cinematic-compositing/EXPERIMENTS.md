# Cinematic Compositing — reproducible experiments

## E1 — Depth-as-input vs depth-as-supervision (the key ablation)
- **Setup:** on a small dataset, fine-tune a compositing/inpainting video model two
  ways — feed depth only as an extra **input** channel, vs also **supervise** a
  predicted depth output jointly with RGB. Score foreground SSIM; eyeball
  shadow/occlusion realism.
- **Shows:** the paper's central claim (0.638 vs 0.642) that **joint depth
  denoising**, not depth input, is what produces physical harmonization.
- **Feasibility:** medium (needs a small video-diffusion fine-tune).

## E2 — Tri-mask vs binary mask
- **Setup:** build the three-state mask (relight / keep-geometry-regenerate-
  appearance / full-generate) and compare against a plain keep-vs-generate binary
  mask on the same task. Measure identity preservation on the *kept* face region
  and prompt-consistency on the *generated* region.
- **Shows:** whether per-region behavior really preserves the actor's face better
  while still harmonizing.
- **Feasibility:** medium.

## E3 — Lighting-harmonization stress test (E2C) — **cheapest, mostly runnable**
- **Setup:** take one green-screen clip, composite it into several environments
  with clearly different light (warm sunset, cool moonlight, red neon). With no
  ground truth, score whether the actor's shading matches the scene using an
  off-the-shelf relighting/lighting-direction estimator, or a human A/B vs a naive
  alpha-composite.
- **Shows:** the "environment lights the character correctly" claim, isolated.
- **Feasibility:** high for the *evaluation harness* (estimator + A/B); the
  generation step can start from an existing relighting model (e.g. IC-Light) as a
  baseline to beat.

**Note:** this topic is the odd one out — it's about *physical harmonization*, not
streaming/latency. Its transferable first-principle ("generate geometry jointly so
physical effects emerge") is orthogonal to the drift/memory cluster and doesn't map
onto the shared `drift_lab`.

# Vidu S1 — reproducible experiments

## E1 — Sliding-window vs full-attention drift test — **partly runnable now**
- **Setup:** take a small open pretrained video diffusion model. Generate a long
  clip two ways — recompute-everything vs a fixed sliding window over the last N
  frames + a locked reference frame. Plot identity drift (CLIP/face similarity to
  the reference) and per-frame quality over time.
- **Shows:** *why* the streaming window prevents collapse, and how window size
  trades off against consistency.
- **Abstract version now:** the shared lab's sliding-window vs anchor comparison
  reproduces the drift curves —
  [`../../experiments/drift_lab/`](../../experiments/drift_lab/).

## E2 — Few-step distillation for latency
- **Setup:** take a small many-step video/image diffusion model, distill to 2–4
  steps (DMD or a consistency objective). Measure FPS and FID/quality at 1/2/3/4/8
  steps.
- **Shows:** the central "real-time = tiny step count" claim, and the quality cost
  of hitting the 30 FPS threshold.
- **Feasibility:** medium.

## E3 — Live control-signal latency loop — **runnable now, no big model needed**
- **Setup:** build a toy autoregressive generator (even a next-state predictor
  conditioned on a discrete "action" token that changes each chunk). Measure how
  many frames of lag occur between a control change and a visible response, as a
  function of chunk size.
- **Shows:** the *interactivity* dimension — how responsive streaming generation
  can be — isolated from raw image quality. This is the cheapest test of the
  "real-time interactive" premise.
- **Run:** `python -m drift_lab.run --measure-latency` (the lab injects an action
  change mid-rollout and reports response lag vs chunk size).

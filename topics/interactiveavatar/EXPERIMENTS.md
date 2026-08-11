# InteractiveAvatar — reproducible experiments

## E1 — Isolate the memory (identity-drift test) — **partly runnable now**
- **Setup:** generate a 2–3 minute talking clip with any small video model /
  autoregressive frame predictor. Embed every frame with an off-the-shelf identity
  model (ArcFace / DINOv3); plot cosine similarity to the first frame over time.
  Then add a tiny **long-term key-frame buffer** (keep N frames chosen by a
  redundancy score) as extra conditioning; re-measure.
- **Shows:** how much a long-term memory reduces drift — without their full stack.
- **Abstract now:** the shared lab's "anchor memory" is exactly this; the
  redundancy-based key-frame selection can be prototyped on top of it.

## E2 — Distillation speed-vs-quality curve
- **Setup:** take a small pretrained video diffusion model, apply few-step
  distillation (DMD / consistency), sweep steps 50 → 8 → 4 → 2. Plot FPS vs
  FVD/identity on a handful of clips.
- **Shows:** the concrete real-time trade-off, at hobby scale on one GPU.
- **Feasibility:** medium.

## E3 — State-Cycling on a puppet (intent without full generation) — **runnable now**
- **Setup:** skip generation. An LLM reads user input and emits `(action prompt +
  TTS audio)` vs `(idle prompt)`, driving a simple audio-conditioned talking-head
  or a rigged 2D avatar. Measure whether "switch to idle when audio ends" produces
  more natural pauses than always-reacting (human preference or a motion-energy
  metric).
- **Shows:** the intent/reaction idea — *and knowing when to stop* — in isolation,
  which is the paper's most transferable behavioral insight.
- **Feasibility:** high (an LLM + TTS + any 2D puppet; no video diffusion needed).

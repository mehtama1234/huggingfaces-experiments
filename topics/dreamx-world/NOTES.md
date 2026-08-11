# DreamX-World 1.0 — A General-Purpose Interactive World Model

**Paper:** arXiv [2606.16993](https://arxiv.org/abs/2606.16993) · [project](https://amap-ml.github.io/DreamX_World) · [code](https://github.com/AMAP-ML/DreamX-World) · [5B model](https://huggingface.co/GD-ML/DreamX-World-5B)

## The problem, plainly

One system that lets you fly a camera through photorealistic, game-style, *or*
stylized worlds, revisit places you've seen, and trigger events by typing — all
while staying visually consistent over long generations, and fast enough to feel
interactive. The hard parts: keep object identities and layout stable when you
turn around and come back, and run in real time.

## The core idea, from first principles

Two moves, closely echoing AlayaWorld but with a different engineering emphasis.

**1. Take a strong *offline* video generator and surgically convert it into a
*streaming* one.** A normal video diffusion model is *bidirectional* — it looks at
the whole clip at once, which is accurate but slow and non-interactive. DreamX
converts it into a **few-step, causal, autoregressive** model (via causal forcing
+ distillation + long-rollout training) so it can generate continuously and
cheaply. You keep the pretrained model's quality but change *how* it runs.

**2. For consistency, don't hoard the whole past — *retrieve* the relevant past
using camera geometry.** When the camera returns to a region, the system uses the
known camera pose to **look up the earlier frames that actually saw that region**
and conditions on them ("recover non-local visual evidence from earlier
observations"). This is memory-as-retrieval, not memory-as-buffer. And because
retrieved memories are imperfect, it uses **residual recycling** — the model
learns to *correct* the retrieved evidence rather than blindly copy it, so it
isn't brittle.

The reusable insight: **camera geometry is a free index into your own past.** If
you know where the camera was, you know which old frames are relevant now.

## How it actually works

- **Base:** Wan2.2-T2V-5B (a ~5B diffusion transformer), converted to a few-step
  causal world model via causal forcing + DMD-style distillation + long rollouts.
- **Camera control — E-PRoPE:** a lightweight *projective* positional encoding.
  It keeps real 3D camera geometry inside attention (so the camera behaves
  physically) but applies the camera-aware attention only to **spatially reduced
  tokens** — trading a little resolution for much lower compute. This efficiency
  trick is what makes accurate camera control affordable in a general model.
- **Events — Event Instruction Tuning:** composable, promptable world events
  ("make it start raining"), further aligned with reinforcement learning.
- **Consistency:** geometry-guided retrieval of earlier observations + residual
  recycling → "memory-conditioned scene persistence."
- **Data engine:** Unreal Engine renders (camera-accurate) + action-rich gameplay
  + real video with recovered camera geometry, all filtered.
- **Speed:** up to **16 FPS on 8× RTX 5090** via mixed-precision DiT, residual
  reuse, a 75%-pruned VAE decode, and async pipeline parallelism.

## What's genuinely new

The headline is **E-PRoPE**: keep *true* projective camera geometry in attention
but only over reduced tokens — cheap, accurate camera control in a general model.
Combined with **geometry-guided non-local memory retrieval + residual recycling**
for revisits, and **composable RL-aligned events**, working across *three visual
domains at once* (photoreal / game / stylized). Fully open with a released 5B
checkpoint.

## AlayaWorld vs DreamX-World, in one line

Both make the world in causal few-step chunks and use camera geometry for
long-horizon consistency. **AlayaWorld** leans on an explicit reprojected 3D
*cache* plus *train-on-your-own-drift*; **DreamX** leans on geometry-guided
*retrieval* of past frames plus *residual recycling*, and pushes generality across
visual styles with the E-PRoPE efficiency trick. Same physics, different levers.

> Reported: camera-control 73.75, overall 84.76 on a 5s eval, beating
> HY-WorldPlay 1.5 (80.79) and LingBot-World (80.45). Not reproduced here.

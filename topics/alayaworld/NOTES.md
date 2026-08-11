# AlayaWorld — Interactive Long-Horizon World Modeling

**Paper:** arXiv [2607.18367](https://arxiv.org/abs/2607.18367) · [project](https://alaya-lab.github.io/AlayaWorld/) · [code](https://github.com/AlayaLab/AlayaWorld)
*(Note: your link `2608.05070` is a different paper — see `../helloworld/`.)*

## The problem, plainly

Most video generators make one nice short clip and then fall apart. If you keep
"playing" — steering a camera through the scene — the picture slowly drifts:
objects mutate, colors shift, and a place you already walked past looks different
when you turn around and come back. AlayaWorld is built to keep generating an
interactive video world that stays coherent for a **full minute or more** while
you drive a camera through it and drop in new events on the fly.

## The core idea, from first principles

The whole design follows from two simple observations.

**1. Don't generate a long video in one shot. Generate it a sliver at a time,
and feed the model a *curated summary* of the past, not the raw past.**

Making a whole minute at once is impossible to keep stable and impossible to run
in real time. So AlayaWorld makes the world in short **chunks** of frames, one
after another (this is "autoregressive" — each new chunk is conditioned on what
came before). The key question becomes: *what do you tell the model about the
past?* If you hand it the raw history, cost grows without bound and it still
forgets. AlayaWorld instead hands it a small, bounded summary with two parts that
do two different jobs:

- **A geometry-aware spatial memory (a 3D cache).** The model literally remembers
  *where things are in space*. When your camera moves, that cache is reprojected
  into the new viewpoint. This is why revisiting a spot reproduces it — the memory
  is anchored to 3D positions, not to "the last few frames."
- **A compressed temporal memory** of recent frames (plus one persistent "anchor"
  frame). This handles smooth, continuous motion — no flicker between chunks.

Spatial memory answers *"is the room behind me still the same room?"*; temporal
memory answers *"does motion flow smoothly frame to frame?"* You need both.

**2. To stop drift, train the model on its own mistakes.**

Drift happens because tiny per-frame errors compound. The fix is disarmingly
direct: during training, deliberately feed the model **corrupted, already-drifted
histories** and keep an **"error bank"** of the artifacts it tends to produce, and
re-inject those. The model is forced to learn *recovery* — how to pull a scene
back toward correctness — instead of only ever seeing clean inputs and then
face-planting the first time reality drifts at test time.

## How it actually works

- **Model:** a **15B video diffusion transformer**, run **autoregressively** in
  short latent chunks. Output is 24 fps, 540p/720p, 60+ seconds.
- **Two control channels:** (a) a rendered 3D cache drives **lightweight AdaLN
  camera modulation** for grounded 6-DoF joystick navigation; (b) **chunk-level
  prompt switching** — drop a text prompt at a chunk boundary to trigger a new
  event with low latency.
- **Anti-drift objective:** train on drifted histories + self-rollout residuals +
  an error bank re-injected into memory and target.
- **Speed:** **discrete autoregressive distillation** (distribution matching +
  Self-Forcing++ + consistency distillation) cuts inference from ~30 sampling
  steps to **4 steps per chunk** → real-time.

## What's genuinely new

Earlier interactive world models (Genie-, Oasis-style) are either short-horizon or
purely frame-recurrent, so they forget geometry and drift. AlayaWorld's novelty is
the *combination at open-source scale*: an explicit **geometry-aligned 3D spatial
memory** fused with compressed temporal memory, an explicit **"train on your own
drift"** recovery objective, and **few-step distillation** that makes a 15B
diffusion model run in real time for a full minute — shipped as open source, not a
demo.

## Why it matters

The transferable lesson isn't the 15B model — it's the recipe: **bounded +
geometry-aware memory** for consistency, and **self-corruption training** for
drift. Both are testable at tiny scale (see `EXPERIMENTS.md`).

> Reported to lead **iWorld-Bench** on long-horizon metrics. Exact score tables
> not independently reproduced here.

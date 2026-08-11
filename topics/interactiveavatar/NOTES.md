# InteractiveAvatar — Real-Time Streaming, Consistent, Intent-Aware Avatars

**Paper:** arXiv [2606.22905](https://arxiv.org/abs/2606.22905) · cs.CV, June 2026

## The problem, plainly

You want video of a digital person that talks and moves in real time, keeps going
as long as you want, **always looks like the same person**, and actually *reacts*
to what a user says — not just mechanically lip-syncs. Existing real-time avatars
drift over long clips (face/clothes slowly change) and only follow audio without
understanding intent. This builds one system that streams forever, stays the same
identity, and reacts appropriately.

## The core idea, from first principles

Real-time forces you to only ever look at the **recent past** (you can't wait for
the whole clip). Two problems follow, each with a targeted fix:

**1. Short memory → identity drift.** If you only remember the last few seconds,
the avatar forgets what it looked like earlier. Fix: **Long-Short Visual Memory
(LSVM)** — a small buffer with two parts:
- a **short-term FIFO** window (~5s) for smoothness, and
- a **fixed-size long-term buffer** of globally representative key-frames. A frame
  is kept only if swapping it in makes the memory *less redundant* (scored over
  SigLIP2 embeddings). Training samples reconstruction targets from *both* buffers,
  so the model learns to honor the whole history — this is what pins the identity.

**2. Audio-only → no intent.** An avatar that just follows audio can't decide
*what* to do. Fix: bolt on a reasoning brain. The **Reasoning-Reaction Module
(RRM)** — an LLM — emits an **"action state"** (a motion prompt + the audio to
speak) and a **"stable state"** (a calm idle). Then:
- **State-Cycling:** while the response audio plays, condition on the action
  prompt+audio; when audio ends, swap to the stable prompt so the avatar returns
  to a natural idle. (This is the un-obvious, important bit — knowing when to
  *stop* reacting is as important as reacting.)
- **Cache-Switching:** when the prompt changes, re-encode only the new text
  condition's KV tensors, so switching is cheap.

**Real-time** comes from distilling the slow diffusion teacher to a **few-step
causal student** (DMD + Self-Forcing), + KV-caching + pipeline parallelism →
**26.68 FPS at 576p on an H100**.

## What's genuinely new

Prior real-time avatars (StableAvatar, OmniAvatar, LiveAvatar…) are audio-driven
lip/motion sync that drift over long durations. New here: (1) **dual short+long
visual memory** that explicitly fights identity drift over unbounded length; (2)
an **LLM reasoning layer** that makes the avatar *decide* intent and switch between
reacting and idling; (3) the engineering to keep all of that above real time.

> Reported vs LiveAvatar: object consistency 85.2 vs 76.9, FPS 26.68 vs 21.94,
> identity comparable (4.51 vs 4.53); slightly worse FVD/sync — it trades a little
> fidelity for consistency, speed, and interactivity. Not reproduced here.

## Shared thread

LSVM is the same "bounded memory with a persistent long-term anchor" pattern as
AlayaWorld's cache and DreamX's retrieval, applied to *identity* instead of scene
geometry. See [`../../experiments/drift_lab/`](../../experiments/drift_lab/) —
"identity drift" and "scene drift" are the same math.

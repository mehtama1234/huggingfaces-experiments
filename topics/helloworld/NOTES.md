# HelloWorld — Socially Interactive Characters in Video World Models

**Paper:** arXiv [2608.05070](https://huggingface.co/papers/2608.05070) · Aug 5 2026 · Liangyang Ouyang (U. Tokyo / Alaya Lab)

> **This is the paper your HuggingFace link pointed to** (`2608.05070`). It is a
> *different* paper from AlayaWorld (which is `2607.18367`, see `../alayaworld/`),
> though it comes from the same lab and shares the video-world-model foundation.
>
> Coverage here is **lighter** than the other topics — it's drawn from the
> abstract/paper-page level, not a full deep read. Flagged honestly; easy to deepen.

## The problem, plainly

Most video world models let you steer a *camera* through a scene. HelloWorld adds
the missing social layer: you interact with the **characters on screen** — press a
button and the character turns toward you, waves, nods, greets you. It's the
difference between *walking through a world* and *the world reacting to you
socially*.

## The core idea, from first principles

A camera-controllable world model already knows how to continue a video
conditioned on control input. The new question is *when* and *how* a character
should respond to a discrete social prompt without retraining the whole model or
adding latency. HelloWorld's approach (per the paper page):

- A **self-distillation pipeline** — the model teaches a faster/cleaner version of
  itself, the same "compress a strong teacher into an efficient student" pattern
  seen across this whole cluster (AlayaWorld, DreamX, Vidu, InteractiveAvatar).
- A **training-free temporal-attention gate** for *interaction timing* — a way to
  decide, at inference, *when* the character should act on a prompt, by modulating
  the temporal attention rather than by learning a new module. "Training-free"
  means it's a clever use of the existing attention structure, not extra weights.
- **HelloWorldBench**, an evaluation set for socially-interactive character
  behavior.

The through-line with the rest of this repo: **interaction is about timing and
control, and the cheapest wins come from re-using structure the model already has**
(here, temporal attention) rather than bolting on new machinery.

## What to verify next

Because this is abstract-level coverage, before building on it confirm from the
full text: the exact self-distillation objective, how the temporal-attention gate
decides timing, the base model family, and HelloWorldBench's metrics. See
`EXPERIMENTS.md` for a probe that tests the timing-gate idea cheaply.

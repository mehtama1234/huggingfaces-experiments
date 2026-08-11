# huggingface-experiments — interactive video AI, 2026

First-principles notes and **runnable mini-experiments** for a cluster of 2026
papers on interactive world models, real-time video generation, avatars, and
compositing. The goal: understand *how these actually work* in plain language, and
reproduce their **core mechanisms** at a scale you can run on a laptop.

Two things live here for every topic:
- a **`NOTES.md`** — the conceptual, first-principles synthesis (what problem, the
  key insight, how it works, what's new), in plain words, no jargon.
- an **`EXPERIMENTS.md`** — concrete experiments, smallest-first, with the ones that
  are runnable today wired to `experiments/`.

Read [`SYNTHESIS.md`](SYNTHESIS.md) first for the big picture: all seven papers are
solving the same four problems (latency, long-horizon consistency, controllability,
evaluation) with the same handful of tricks.

## The papers (verified IDs)

| Topic | Paper | arXiv |
| --- | --- | --- |
| [alayaworld](topics/alayaworld/) | AlayaWorld: Interactive Long-Horizon World Modeling | [2607.18367](https://arxiv.org/abs/2607.18367) |
| [dreamx-world](topics/dreamx-world/) | DreamX-World 1.0: A General-Purpose Interactive World Model | [2606.16993](https://arxiv.org/abs/2606.16993) |
| [vidu-s1](topics/vidu-s1/) | Vidu S1: A Real-Time Interactive Video Generation Model | [2607.03118](https://arxiv.org/abs/2607.03118) |
| [trimotion](topics/trimotion/) | TriMotion: Modality-Agnostic Camera Control for Video Generation | [2606.20774](https://arxiv.org/abs/2606.20774) |
| [interactiveavatar](topics/interactiveavatar/) | InteractiveAvatar: Real-Time Streaming, Consistent, Intent-Aware Avatars | [2606.22905](https://arxiv.org/abs/2606.22905) |
| [cinematic-compositing](topics/cinematic-compositing/) | Cinematic Compositing (Character-Environment-Harmonized) | [2606.20233](https://arxiv.org/abs/2606.20233) |
| [interacvid](topics/interacvid/) | InteracVid: Interactive Audio-Visual Response Dataset from Live-Chat | [2608.01157](https://arxiv.org/abs/2608.01157) |
| [helloworld](topics/helloworld/) | HelloWorld: Socially Interactive Characters in Video World Models | [2608.05070](https://huggingface.co/papers/2608.05070) |

> **Note on the link you gave:** `huggingface.co/papers/2608.05070` is **HelloWorld**,
> *not* AlayaWorld. AlayaWorld's real ID is `2607.18367`. Both are covered; HelloWorld
> coverage is lighter (abstract-level) and flagged as such.

## Runnable experiments

Two deterministic numpy labs reproduce the shared mechanisms — no GPU, no video
models. See [`experiments/`](experiments/).

```bash
cd experiments
python -m venv .venv && .venv/bin/pip install numpy pytest
.venv/bin/python -m pytest -q                    # 7 tests

.venv/bin/python -m drift_lab.run --revisit      # streaming drift vs geometry memory vs self-correction
.venv/bin/python -m motion_space.run             # align text/pose/video of a camera move into one space
```

`drift_lab` shows *why* streaming world models drift and how geometry-aware memory
fixes it (covers AlayaWorld, DreamX, Vidu, InteractiveAvatar, HelloWorld).
`motion_space` shows *why* one shared control space works (covers TriMotion).

## How this was built (honesty)

These are 2026 papers past the assistant's training data, so every technical claim
was **researched from live sources** (arXiv abstracts + full text, project pages,
GitHub/HF) — not recalled from memory. Reported benchmark numbers are the authors'
own and are **not** reproduced here; the mini-experiments reproduce *direction and
mechanism*, not the papers' scores. Anything uncertain is flagged in-place.

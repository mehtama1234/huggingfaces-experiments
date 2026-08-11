# experiments — runnable, deterministic mini-reproductions

Two tiny numpy labs that reproduce the *core first-principles claims* of the 2026
papers in this repo — no GPUs, no video models, no network. Like a physics toy:
they don't look like the real thing, but they show the mechanism and its direction.

## Setup

```bash
cd experiments
python -m venv .venv && .venv/bin/pip install numpy pytest
.venv/bin/python -m pytest -q        # 7 tests
```

## `drift_lab/` — streaming, memory, and drift

The problem shared by **AlayaWorld, DreamX-World, Vidu S1, InteractiveAvatar,
HelloWorld**: generate a world one chunk at a time and it *drifts*; the fix is some
bounded, geometry-aware memory (and training on your own drift).

```bash
.venv/bin/python -m drift_lab.run --revisit        # compare the three strategies
.venv/bin/python -m drift_lab.run --measure-latency # chunk size vs response lag
.venv/bin/python -m drift_lab.run --social-gate     # HelloWorld's timing gate
```

What it shows (real output): naive **sliding** streaming drifts (error compounds);
**anchor** memory bounds it and makes revisited places look the same again;
**selfcorrect** (train-on-own-drift) flattens the curve. The anchor sparkline
literally rises on the way out and falls on the way back.

## `motion_space/` — one shared space for camera control

TriMotion's claim: a camera move is one thing, so **text, pose, and video of the
same move align into one space**, enabling cross-modal retrieval and free
composition/interpolation.

```bash
.venv/bin/python -m motion_space.run              # cross-modal retrieval (≈1.00 vs ~0.015 chance)
.venv/bin/python -m motion_space.run --interpolate # blend pan <-> tilt smoothly
.venv/bin/python -m motion_space.run --compose     # chain pan then dolly
```

## Honesty

These are caricatures. A "frame" is a vector; a "scene" is a smooth field; a
"move" is a 6-DoF trajectory. They demonstrate *why* the mechanisms work and in
which direction they help — not absolute quality, and not the papers' numbers. The
paper-specific, higher-fidelity experiments (that need real models) are listed in
each `topics/*/EXPERIMENTS.md`.

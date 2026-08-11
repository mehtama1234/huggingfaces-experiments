# AlayaWorld — reproducible experiments

Three experiments, smallest-first. #1 and #3 are runnable today with the shared
lab in [`../../experiments/drift_lab/`](../../experiments/drift_lab/) (numpy only,
deterministic). #2 needs a tiny video model.

## E1 — Chunked autoregression vs one-shot: does short-chunk streaming change the drift curve?
- **Setup:** roll a world forward 200 steps two ways — (a) one long pass, (b) 25
  chunks of 8, each conditioned only on the previous chunk's last state. Measure
  state error vs the ground-truth world over time.
- **Shows:** whether short-chunk autoregression *alone* helps or hurts drift
  (spoiler from the lab: alone it doesn't fix drift — you need memory, E-below).
- **Run:** `python -m drift_lab.run --strategy sliding --window 1`

## E2 — Geometry memory vs temporal-only: revisit consistency *(needs a tiny video model)*
- **Setup:** in a deterministic low-res 3D scene (a few textured boxes, toy
  rasterizer, or MiniGrid-3D/Habitat), drive a camera on an **out-and-back** path.
  Condition generation on (a) only the last N frames, vs (b) an explicit cache of
  previously-seen pixels warped to the current pose. Measure pixel error between
  the outbound and return views of the *same* location.
- **Shows:** the specific payoff of geometry-aligned spatial memory — the thing
  temporal memory cannot give you.
- **Feasibility:** medium. The *mechanism* is demonstrated abstractly in the lab
  as the "anchor memory" strategy without needing pixels.

## E3 — Train-on-your-own-drift: does self-corruption flatten the drift curve?
- **Setup:** fit a tiny next-state predictor two ways — (a) teacher-forced on
  clean history only; (b) with a fraction of inputs replaced by the model's own
  noisy rollouts (an "error bank"). Roll both out 100+ steps; measure how fast
  error accumulates.
- **Shows:** AlayaWorld's central anti-drift claim — exposure to self-generated
  corruption teaches recovery — testable with a sub-million-parameter model.
- **Run:** `python -m drift_lab.run --strategy selfcorrect`

**What the lab reproduces qualitatively:** error compounds without memory; a
geometry/anchor memory bounds it; self-correction training flattens it. These are
the paper's three mechanisms, isolated. Absolute numbers are not the paper's.

# HelloWorld — reproducible experiments

> Lighter than the other topics (abstract-level source). Experiments below are
> designed from the stated ideas; verify against the full text first.

## E1 — Training-free timing gate on temporal attention — **runnable now (toy)**
- **Setup:** in the shared `drift_lab`, add a discrete "social prompt" event at a
  chosen step. Compare two ways of injecting it: (a) always-on conditioning, vs
  (b) a **gate** that only lets the prompt influence generation within a short
  temporal window around the trigger. Measure response latency and whether the
  reaction stays localized (doesn't smear across the whole rollout).
- **Shows:** whether gating *when* a prompt acts — using existing temporal
  structure, no new weights — gives cleaner, better-timed reactions than constant
  conditioning. This is the paper's core "training-free timing" claim in miniature.
- **Run:** `python -m drift_lab.run --social-gate`

## E2 — Self-distillation for a faster student
- **Setup:** take any small autoregressive generator; distill a few-step student
  from a many-step teacher (as in Vidu/InteractiveAvatar E2). Measure FPS and
  quality; then test whether the student preserves the timing-gate behavior from E1.
- **Shows:** that the efficiency move (self-distillation) and the interaction move
  (timing gate) compose — you can be both fast *and* well-timed.
- **Feasibility:** medium.

## E3 — A mini "HelloWorldBench" for reaction timing
- **Setup:** define a small battery of social prompts (turn / wave / nod / greet)
  and, for each, a metric: reaction latency (steps from prompt to visible change),
  correctness (did the *right* reaction happen), and non-interference (did unrelated
  scene state stay stable). Run it against any interactive generator.
- **Shows:** how to *measure* socially-interactive behavior — the evaluation gap the
  paper names. This is a methodology contribution you can reuse for any of the
  interactive models in this repo.
- **Feasibility:** high (it's an eval harness; the hard part is the generator).

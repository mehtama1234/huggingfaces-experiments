# Synthesis — the shared first principles of 2026 interactive video AI

Seven papers, four surfaces (interactive world models, real-time video, avatars,
compositing, plus the dataset that feeds them). Read together, they are all solving
**the same four problems**, and they keep reaching for **the same handful of
tricks**. This page is the map; each `topics/*/NOTES.md` is the detail.

## The four problems every one of these systems must solve

**1. Latency — close the loop faster than a human notices.**
If a person acts (speaks, presses a button, moves a camera) the generated frame has
to come back in well under conversational lag. That single constraint rules out
slow offline diffusion and forces **autoregressive, few-step generation**. Every
model here distills a strong many-step teacher down to 2–4 steps and streams in
small chunks. *(Reported targets in this cluster: ~180–500 ms, 24–45 FPS.)*

**2. Long-horizon consistency — don't drift or forget.**
Because these run indefinitely, tiny per-frame errors compound: scenes drift,
identities morph, revisited places change. The universal fix is **bounded memory**,
and the best versions make it **geometry-aware** — remember *where* things are, not
just the last few frames, so you can restore a place (or a face) when you come back
to it. `experiments/drift_lab/` reproduces exactly this: drift compounds without
memory, and a geometry/anchor memory bounds it.

**3. Controllability / groundedness — actually respond to the input.**
Output must be *caused by* the live signal (audio, chat, action, camera) and stay
faithful to it — not fluent-but-generic. This shows up as camera control
(TriMotion, DreamX's E-PRoPE), intent/reaction (InteractiveAvatar's LLM, HelloWorld's
timing gate), and above all **data**: InteracVid exists because you cannot teach
genuine *reaction* from caption datasets — you need real stimulus→response pairs.

**4. Evaluation — there's no ground truth for "the right reaction."**
You can't diff a generated reaction against a single correct answer. The field
leans on **VLM-judges + human ratings** of causality/naturalness/appropriateness,
and then has to validate that the judges agree with humans. Building the *measuring
stick* is itself an open problem (InteracVid reports judge–human agreement ~0.95).

## The recurring tricks (steal these)

| Trick | What it is | Who uses it |
| --- | --- | --- |
| **Few-step causal distillation** | compress a slow bidirectional teacher into a 2–4-step streaming student | AlayaWorld, DreamX, Vidu, InteractiveAvatar, HelloWorld |
| **Geometry-indexed memory** | use camera pose to store/retrieve the right past frames; reproject a 3D cache | AlayaWorld (cache), DreamX (retrieval) |
| **Split memory by purpose** | separate "keep motion flowing" from "keep detail/identity" | Vidu (TwinCache), InteractiveAvatar (short+long buffers) |
| **Train on your own drift** | feed the model its own corrupted rollouts so it learns to recover | AlayaWorld |
| **One shared latent for many controls** | map text/pose/video of a move to one point | TriMotion |
| **Generate geometry jointly** | denoise depth with RGB so physical effects emerge | Cinematic Compositing |
| **Know when to stop reacting** | gate/idle instead of always-on conditioning | InteractiveAvatar (state-cycling), HelloWorld (timing gate) |
| **Response as a first-class label** | supervise on real stimulus→response, not captions | InteracVid |

## Where I'd focus a real reproduction

1. **`drift_lab` → real pixels.** Promote the anchor-memory result to a tiny video
   model on a toy 3D scene (out-and-back camera). Measuring revisit error in pixels
   is the single most illustrative, tractable reproduction of the world-model
   cluster.
2. **`motion_space` → a real generator.** The alignment already works in the toy;
   wiring one motion embedding into a small video model tests whether "one control
   space" survives contact with generation.
3. **The evaluation gap.** The cheapest high-value contribution isn't a model — it's
   a **reusable reaction/consistency benchmark** (InteracVid's causality probe +
   HelloWorld's timing metrics + drift/revisit curves). Everyone here is bottlenecked
   on measurement; a good measuring stick helps all of them.

## One sentence

Every system here is trading **speed against consistency** while trying to stay
**genuinely reactive**, and struggling to **measure** whether it succeeded — and the
winning moves are all about *reusing structure you already have* (geometry for
memory, one latent for control, existing attention for timing) instead of bolting on
new machinery.

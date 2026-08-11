# Cinematic Compositing — Character-Environment-Harmonized Video Generation

**Paper:** arXiv [2606.20233](https://arxiv.org/abs/2606.20233) · [project + demos](https://cehcomposition.github.io/demo/)

## The problem, plainly

In filmmaking you shoot an actor on green screen and later drop them into a made-up
environment. The hard part is making it not look *pasted on*: the actor should
cast shadows onto the scene and be occluded by things (**character affects
environment**), *and* the scene's light should fall correctly on the actor
(**environment affects character**). Old methods do only one direction and can't
do the two-way interaction, so results look flat and fake. This does both at once,
in one model.

## The core idea, from first principles

"Looks physically real" is really **two coupled effects that must both hold**:
- **C2E** — the character affects the environment: occlusion, contact, cast
  shadows.
- **E2C** — the environment affects the character: the scene's light and
  reflections landing on the actor.

Prior pipelines model these separately and stack them, which *breaks the
coupling*. The insight: model them **jointly in one denoising pass** — and,
crucially, **denoise geometry (depth) alongside color (RGB) at the same time.**
You physically cannot get shadows, occlusion, and scale right unless the model
understands 3D layout, so you make it *generate depth too*, not assume it.

Second insight: **not every pixel should be treated the same.** The actor's real
face must be *kept but relit*; a green proxy prop must *keep its shape but get a
new look*; the background must be *generated from scratch*. So instead of a plain
keep/replace mask, they use a **three-value (tri) mask** encoding those three
behaviors per region.

## How it actually works

- **Model:** Wan 2.1 VACE-14B (text-to-video DiT), fine-tuned with **LoRA** — an
  adaptation of a big pretrained model, not trained from zero.
- **Inputs:** green-screen foreground video, a predicted depth map, a text prompt
  for the target scene, optional reference images.
- **Tri-mask:** *preserve-and-relight* (real face/props — keep RGB, adjust
  lighting) · *geometry-preserving* (green proxies — keep depth as anchor,
  regenerate appearance) · *full-generation* (background — create entirely).
- **Harmonization = RGB-D joint denoising:** concatenate noisy RGB + depth latents
  into one trajectory and denoise them together (learnable modality embeddings,
  RGB init zero, depth random). Because depth is denoised jointly, the model
  reasons about geometry — so occlusion, cast shadows, and correct scale *emerge*
  instead of being pasted flat. Lighting is learned via supervision on
  lighting-augmented videos (E2C).
- **Objective:** a mask-guided RGB-D velocity (flow-matching) loss that weights
  foreground/interaction regions ~10× the background.
- **Data:** ~2×10⁵ clips filtered from HOIGen1M + a synthesized multi-illumination
  subset. Each sample: (GT video, lighting-augmented variant, depth, tri-mask,
  caption).

## What's genuinely new

Prior work is single-direction or cascaded (relight-then-composite,
background-replacement) and can't achieve *bidirectional* harmonization. New: (1)
jointly modeling C2E and E2C in **one end-to-end pass**; (2) **RGB-D joint
denoising** so geometry is *generated*, enabling physically consistent
shadows/occlusion/scale; (3) the **tri-mask** for per-region behavior; (4) a
lighting-augmented data recipe for realistic relighting. One model covers
relighting, prop replacement, and background generation.

> Reported: leads identity preservation (0.637) and background similarity (0.704)
> on a 100-clip synthetic bench; wins a user study (61.29%) on 20 real clips.
> Ablation: depth *as input alone* barely helps (0.636→0.638); depth *as joint
> supervision* is what matters. Not reproduced here.

# TriMotion — Modality-Agnostic Camera Control for Video Generation

**Paper:** arXiv [2606.20774](https://arxiv.org/abs/2606.20774) (ECCV 2026) · [project](https://seunghyuns98.github.io/TriMotion) · [code](https://github.com/seunghyuns98/TriMotion)

## The problem, plainly

When you generate a video you often want to control the *camera* — pan left, dolly
in, tilt up. But every existing tool locks you into one way of saying it: some
need an exact numeric camera trajectory (precise but unintuitive), some need a
reference video to copy from (intuitive but imprecise), and plain text barely
works. TriMotion lets you specify the same camera move with **any** of the three —
text, a pose trajectory, or a reference video — and get the same motion out.

## The core idea, from first principles

**A camera path is one underlying thing, no matter how you describe it.** So don't
build three control systems — build **one shared "motion space"** where the text
description, the pose trajectory, and the reference video of the *same* move all
land on nearly the same point. Once all three speak this common motion language,
the video generator only ever conditions on **one thing — a motion embedding — and
stops caring where it came from.**

Two payoffs fall out for free, because the moves live in one space:
- **Compose** motions in time ("pan, then zoom") by concatenating embeddings.
- **Blend** between two moves by interpolating embeddings — no retraining.

This is the same trick as any shared/joint embedding space (like CLIP for
image↔text): find the one representation that several surface forms all map to,
and downstream everything gets simpler.

## How it actually works

- **Base:** a latent video diffusion transformer with rectified flow. The denoiser
  takes source + noisy target latents, conditioned on an appearance description, a
  first frame, and **the motion embedding**.
- **Three alignment losses build the shared space:**
  - *Global alignment* — an **InfoNCE contrastive** loss pulls matching
    video/pose/text triplets together, pushes mismatches apart.
  - *Temporal synchronization* — a cosine-distance loss aligns embeddings
    **frame-by-frame in time** (same motion *at the same moment*, not just overall).
  - *Geometric fidelity* — a shared **pose regressor** decodes camera extrinsics
    from the embedding, forcing the space to encode *real* camera geometry, not
    just abstract similarity.
- **Latent Motion Consistency (efficiency):** a "Motion Embedding Predictor" reads
  the motion back out of the *generated latents*, and a loss enforces the video
  actually matches the target trajectory — measured **in latent space**, skipping
  costly pixel decoding.
- **Data — Motion Triplet Dataset:** built on MultiCamVideo (136K UE5 videos).
  Continuous camera extrinsics → first-frame-relative trajectories; symbolic
  motion phases (Dolly/Pan/Tilt) extracted; Qwen3-4B writes geometry-grounded
  captions. Result: synchronized (video, pose, text) triplets.

## What's genuinely new

Prior methods each cover one modality — pose-conditioned (MotionCtrl, CameraCtrl,
VD3D), reference-video (MotionMaster, ReCamMaster, CamCloneMaster), and text was
basically unexplored. TriMotion is the **first to unify all three under one
representation** — and the shared space unlocks **sequential composition** and
**cross-modal interpolation** of camera moves for free.

> Reported: V2V TriMotion-Pose FVD 221.59, beating ReCamMaster and
> CamCloneMaster; user study motion-following 86.5%. Not reproduced here.

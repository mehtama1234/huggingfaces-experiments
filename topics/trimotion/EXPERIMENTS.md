# TriMotion — reproducible experiments

## E1 — Prove the shared motion space exists — **runnable now, no video generation**
- **Setup:** take a small set of camera trajectories; for each, collect a short
  clip, its pose sequence, and a text description. Train tiny per-modality encoders
  with an **InfoNCE contrastive loss** to align the three. Test retrieval: given a
  *text* motion, retrieve the matching *pose*/*video*.
- **Shows:** the central claim — one motion has one shared embedding — cheaply.
  High cross-modal retrieval accuracy is the whole proof.
- **Run:** the shared lab [`../../experiments/motion_space/`](../../experiments/motion_space/)
  builds synthetic (text, pose, video) triplets for a handful of canonical camera
  moves and trains the alignment: `python -m motion_space.run`

## E2 — Latent-space vs pixel-space motion measurement
- **Setup:** on a small video model, train a lightweight "motion predictor" that
  estimates camera motion from *latents*; compare its trajectory-error estimates
  against a pixel-space pose tool (SfM / off-the-shelf pose estimator).
- **Shows:** *why* you can enforce trajectory fidelity in latent space and skip
  costly decoding — the Latent Motion Consistency trick.
- **Feasibility:** medium.

## E3 — Motion composition / interpolation in embedding space — **runnable now**
- **Setup:** with the aligned encoders from E1, take two motion embeddings
  ("pan left", "zoom in"), then (a) concatenate them in time and (b) linearly
  interpolate between them; decode the implied trajectories with the pose
  regressor.
- **Shows:** smooth, physically valid composed/blended camera paths — reproducing
  TriMotion's "sequential composition" and "cross-modal interpolation" without the
  full generator.
- **Run:** `python -m motion_space.run --compose` and `--interpolate`.

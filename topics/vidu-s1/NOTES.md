# Vidu S1 — A Real-Time Interactive Video Generation Model

**Paper:** arXiv [2607.03118](https://arxiv.org/abs/2607.03118) · [code](https://github.com/shengshu-ai/Vidu-S1) · [live demo](https://vidu.com/vidu-stream)

## The problem, plainly

Normal AI video is *offline*: type a prompt, wait minutes, watch the finished
clip. You can't steer it while it's being made, so it's useless for anything live.
Vidu S1 turns video generation into something you *talk to*: you speak, a
character (person, anime figure, pet) reacts and keeps going, for an arbitrarily
long time, without the picture drifting apart.

## The core idea, from first principles

**1. Generate a rolling stream, not a whole clip.** Each new chunk of frames only
looks at a small sliding window — a locked reference image + a cache of recent
frames + the current moment. Because the model never re-processes the whole
history, **memory and compute stay flat no matter how long it runs**. That flat
cost is what makes "infinite length" possible.

**2. Make speech a live steering wheel, not a caption.** Your voice is fed in
*continuously* as an explicit control signal, so you change what the character
does mid-generation instead of pre-writing every instruction.

**3. "Real-time" = tiny step count.** Each generation step is squeezed to **3
diffusion steps** so a frame is produced faster than it needs to be displayed
(>30 FPS).

The one idea that's most worth stealing: **TwinCache**. Long streaming runs
collapse because errors accumulate. Vidu keeps *two* caches of the past — a
**"noisy" cache** that preserves coarse motion continuity, and a **"clean" cache**
that restores fine detail. Splitting "keep the motion flowing" from "keep the
pixels sharp" is their fix for long-run visual collapse. It's a concrete,
first-principles decomposition of what a streaming memory is actually *for*.

## How it actually works

- **Model:** a diffusion-based **video+audio** generator (visuals and synced audio
  together), served via TurboDiffusion + TurboServe.
- **Conditioning:** live speech/audio (primary control), a reference image
  (identity), text. Captions are dual-level — clip-level plus **speech-aware
  chunk-level** captions tied to time intervals, so the model knows what happens
  *when*.
- **Latency tricks:** sliding-window causal streaming; **KV caching with RoPE
  repositioning** (cache key/value features *before* rotary encoding so past
  frames aren't recomputed); **TwinCache** (noisy + clean); **3-step
  distillation** via DMD + Phased Consistency Model regularization; fast-attention
  kernels.
- **Training (3 stages):** bidirectional teacher (full-sequence denoising) →
  causal adaptation mixing teacher-forcing (clean past) and diffusion-forcing
  (noisy past) → DMD+PCM distillation to the 3-step student.

## What's genuinely new

Top offline models (Sora, Veo, Wan) can't interact; fast avatar products (HeyGen,
LiveAvatar) can't follow open instructions at this quality. Vidu's specifics:
real-time (up to 42 FPS) autoregressive generation with mid-stream **voice**
control on a consumer GPU; the **TwinCache** motion/detail split to stop error
accumulation over unbounded length; **speech as a sequential control channel**
rather than one-time conditioning.

> Reported: 540p up to 42 FPS on RTX 5090; 3-step; HDTF CSIM 0.9192, Sync-D 7.847,
> DOVER 0.5660; 500-sample in-house bench. Not reproduced here.

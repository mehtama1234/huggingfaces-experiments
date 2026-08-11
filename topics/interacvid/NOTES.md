# InteracVid — A Real Interactive Audio-Visual Response Dataset from Live-Chat Videos

**Paper:** arXiv [2608.01157](https://arxiv.org/abs/2608.01157) · Aug 2026

## The problem, plainly

Most models that "watch and respond" are trained on video paired with
*descriptions* (captions) — they learn to *say what's in a clip*, not *how a real
person would react to it*. So there's no supervision for the behavior we actually
want from a conversational avatar or live assistant: given what just happened plus
a viewer's question, produce the natural spoken-and-visual **response** that
follows. InteracVid mines that missing signal from livestreams, where streamers
genuinely react to live-chat on camera.

## The core idea, from first principles

A scripted talking-head dataset shows a person *reading lines*: the audio and face
are correct, but **the response isn't caused by anything.** No stimulus → the model
learns delivery, not interaction.

**A livestream is different because the causal loop is real:** a viewer types a
question, the streamer reads it and reacts on the spot — speech, expression,
gesture, often manipulating an object or screen. That reaction is genuine
interactive behavior no scripted corpus contains.

So the atomic unit isn't "clip + caption" but a **four-part tuple**
`𝒳 = (𝒞, 𝒬, 𝒴, 𝒯)`:
- **𝒞 — preceding context:** what the streamer was doing/saying just before.
- **𝒬 — the query/stimulus:** the live-chat message that triggered the reaction.
- **𝒴 — the real audio-visual response:** the actual video+audio reply.
- **𝒯 — an auxiliary caption** of that response (an intermediate planning target).

The insight: **make "response" a first-class label, grounded in a real cause.**
That's exactly the supervision generative interactive systems lack.

## How it actually works (the pipeline)

- **Sources:** ~160K raw livestreams, ~185K hours, from YouTube; curated to **454K
  clips / 59K videos**. Splits into a **real-query branch** (39K clips, ~76h) and a
  **reconstructed-query branch** (414K clips, ~903h).
- **Getting the query→response pair, two ways:**
  - *Real-query:* videos with timestamped chat — collect comments in a window
    *before* a response, use an **LLM to detect which comment plausibly caused the
    reply**, extract the response span.
  - *Reconstructed-query:* no chat metadata — classify transcript sentences as
    "reactive" via discourse-role analysis, then have a **VLM watch the video + read
    the response and reconstruct a plausible query** that would have prompted it.
- **Alignment:** normalize subtitles into a sentence-level **semantic timeline** so
  chat timestamps, transcript, and frames line up.
- **Filtering:** a VLM verifies speech is grounded in the matching video segment;
  scene-cut detection removes hard cuts; re-ASR with Seed-ASR.
- **Validation:** 10 raters scored causality/naturalness/completeness (~4.1–4.3/5);
  response-boundary agreement 93–95% (start), 88–90% (end).

## What's genuinely new

Per the paper's own comparison, InteracVid is the **only dataset supporting both
audio-visual *input* (preceding context) and audio-visual *output* (a response)
across diverse domains**. vs SpeakerVid-5M (mostly human speaking, no
procedural/object/screen diversity); vs LiveChat/MovieLC (responses are *text*, not
AV generation targets); vs AvaMERG (limited context/diversity). Prior work treated
audio+video as *input for understanding*; InteracVid treats a real audio-visual
**response as the generation target**, causally tied to a stimulus, across many
domains.

They also show a two-stage system — an **interaction planner** (VLM writes the
intended-response caption) feeding an **audio-video co-generator (MOVA)** — and
find **the planner, not the generator, is the bottleneck** (learned planner ~3.60
vs oracle-caption 4.22). *(Paper's numbers; not reproduced here.)*

## Why it's the keystone

Every other paper here is a *model*; InteracVid is the *data* that would teach
those models to be genuinely reactive rather than fluent-but-non-reactive. It's the
answer to the "controllability/groundedness" challenge in `../../SYNTHESIS.md`.

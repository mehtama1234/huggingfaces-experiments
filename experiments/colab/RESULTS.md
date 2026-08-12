# Colab T4 runs

**Real-model experiments in this repo**, run on actual Google Colab **Tesla T4
(15.6 GB)** VMs via the `colab` CLI.

## Few-step latency/quality sweep — SD 1.5 (the "real-time = tiny steps" trick)

The single most-shared efficiency move across Vidu S1, InteractiveAvatar, AlayaWorld
and DreamX is generating in a handful of steps. Measured directly on a real diffusion
model: same prompt at 1/2/4/8/16/32 steps, timing each and scoring image↔prompt
agreement with CLIP.

| steps | latency | eff. FPS | CLIP quality |
| --- | --- | --- | --- |
| 1 | 1.25s* | 0.80 | 0.218 |
| 2 | 0.49s | 2.04 | 0.318 |
| 4 | 0.77s | 1.29 | 0.320 |
| 8 | 1.33s | 0.75 | **0.332** |
| 16 | 2.45s | 0.41 | 0.326 |
| 32 | 4.78s | 0.21 | 0.322 |

*1-step latency is a warmup outlier (first generation includes CUDA kernel compile).

**Reading:** quality climbs fast then **plateaus by ~8 steps** (0.33), while latency
keeps growing ~linearly. So 4 steps (0.77s, CLIP 0.320) matches 32 steps (4.78s,
CLIP 0.322) at **~6× lower latency** — the whole basis of "few-step = real-time."
This is standard SD 1.5 on a T4 (not itself real-time); the papers reach true
real-time by *distilling* to that low-step regime and by streaming, but the
step→latency→quality relationship shown here is exactly the lever they pull.
Script: `fewstep_latency_sweep.py`.

## v2 — stronger: Qwen2.5-3B + LLM-as-judge, 20 examples (scored via `eval_harness`)

```
scorecard: interacvid/qwen2.5-3b/T4
  judge_lift        : +0.310     (real 0.510  vs  shuffled 0.200)
  judge_effect_size : +0.971     (Cohen's d ~1 = large effect)
  f1_lift           : +0.086     (real 0.158  vs  shuffled 0.072)
  · 20 examples; control = shuffled query; LLM judge floored every control at 0.2
```

The stronger model + LLM-as-judge give a **much cleaner signal** than v1: with a
shuffled query the judge scored the reply at the floor (0.2) on **all 20**
examples, while real-query replies averaged 0.51 — a large, unambiguous causal
lift. Both the judge metric and the crude token-F1 agree in direction. Results were
piped through `../eval_harness` (`causality_lift` → effect size → `Scorecard`),
demonstrating the harness on a real run. Script: `interacvid_causality_probe_v2.py`.

---

## v1 — first run: Qwen2.5-1.5B, token-F1, 8 examples

## What it tested
InteracVid's central premise: a genuine reaction is *caused by* the viewer's query,
not just topical decoration. For 8 hand-authored `(scene, query, gold_response)`
examples across cooking / unboxing / yoga / gaming / art scenarios, the model
generated a reply two ways — with the **real** query, and with a **shuffled** query
from another example (control) — scored by token-F1 against the gold response.

## Result

```
REAL query mean token-F1 : 0.133
SHUFFLED query mean F1   : 0.066
causality lift (real-ctrl): +0.068   (real ≈ 2× shuffled)
verdict: query DRIVES the response (causal signal present)
```

Per-example replies were sensible and query-appropriate — onions → "gloves and a
fan," chicken → "use tofu," camera → "buy the card separately," knee pain → "find
another position." With a shuffled query the reply drifted off the gold answer,
roughly halving the match.

## Honest reading
- **Direction confirmed, at tiny scale.** A real model's response genuinely depends
  on the query — the signal InteracVid says caption datasets lack. It is *not* a
  strong quantitative claim: 8 examples, a 1.5B model, and a crude token-F1 metric.
- The value is twofold: (1) it validates the **causality-probe methodology** cheaply,
  and (2) it proves the **Colab T4 execution path** end-to-end (provision → run a real
  HF model → capture → tear down).
- To strengthen: more examples, a stronger judge (LLM-as-judge instead of token-F1),
  and real InteracVid tuples instead of hand-authored ones.

## Environment (verified)
Tesla T4, 15.6 GB VRAM, torch 2.11.0+cu128, CUDA on, 202 GB disk, python 3.12.
Note: this account currently has **T4-only** Colab entitlement (L4/A100 refused —
"no quota or entitlement"), so the larger *video* reproductions need a units top-up.

## Reproduce
```bash
# with the colab CLI authed (see ~/projects/COLAB_STARTER_GUIDE.md):
colab new -s hfexp --gpu T4
colab upload -s hfexp interacvid_causality_probe.py /content/probe.py
colab exec -s hfexp -f interacvid_causality_probe.py --timeout 570   # or launch detached + poll
colab stop -s hfexp
```

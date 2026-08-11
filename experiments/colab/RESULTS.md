# Colab T4 run — InteracVid causality probe

**First real-model experiment in this repo**, run on an actual Google Colab **Tesla
T4 (15.6 GB)** via the `colab` CLI, using `Qwen/Qwen2.5-1.5B-Instruct` (fp16).

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

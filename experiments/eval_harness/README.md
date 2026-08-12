# eval_harness — the reusable measuring stick

Every 2026 paper in this repo is bottlenecked on **evaluation** (see
`../../SYNTHESIS.md`): there's no clean ground truth for "the right reaction" or
"still the same world." This is the cheap, model-agnostic answer — it scores
whatever a model produced, as plain numpy, no GPU or model needed.

```python
from eval_harness import drift, revisit_consistency, causality_lift, Scorecard
import numpy as np

# consistency: feed identity/scene embeddings from ANY model
curve = drift(frame_embeddings)                 # rising = drifting; flat = memory holds
rev   = revisit_consistency(frame_embeddings, camera_positions)

# reaction: feed (real-query, shuffled-query) match scores from a causality probe
react = causality_lift(real_scores, control_scores)  # lift>0 => genuinely reacts

sc = (Scorecard("dreamx-5b/out-and-back")
      .add("revisit_error", rev["revisit_mean"])
      .add("causality_lift", react["lift"]))
print(sc.render())
```

## What's here
- `consistency.py` — `drift`, `revisit_consistency`, `summarize_curve`
- `reaction.py` — `causality_lift` (+ effect size), `response_latency`, `localization`
- `scorecard.py` — `Scorecard` (aggregate + render + `to_dict`)

## Why it matters
It's the same math the toy `drift_lab` uses, generalized to **real outputs**. The
Colab causality probe (`../colab/`) already produces the score pairs `causality_lift`
consumes. When the video reproductions run (A100), their frame embeddings feed
straight into `drift`/`revisit_consistency`. One scorecard shape across all 8 papers.

Run tests: `cd experiments && .venv/bin/python -m pytest eval_harness -q`

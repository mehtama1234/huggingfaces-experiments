"""eval_harness — model-agnostic metrics for interactive video / world-model / avatar outputs.

The recurring finding across the 2026 papers (see ../../SYNTHESIS.md) is that they're
all bottlenecked on *evaluation*: there's no clean ground truth for "the right
reaction" or "still the same world." This harness is the cheap, reusable answer — it
scores whatever a model produced, as plain numpy arrays, with no model or GPU needed:

  - consistency: drift and revisit error over an embedding sequence (world/identity)
  - reaction:    causality lift (does output depend on the stimulus?) + timing
  - scorecard:   aggregate + render

Feed it embeddings from any model (frame embeddings, identity embeddings) or the
score pairs from a causality probe. It's the same measuring stick the toy drift_lab
uses, generalized to real outputs.
"""

from .consistency import drift, revisit_consistency, summarize_curve
from .reaction import causality_lift, response_latency, localization
from .scorecard import Scorecard

__all__ = [
    "drift",
    "revisit_consistency",
    "summarize_curve",
    "causality_lift",
    "response_latency",
    "localization",
    "Scorecard",
]

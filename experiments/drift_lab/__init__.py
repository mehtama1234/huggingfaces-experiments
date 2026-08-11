"""drift_lab — a tiny, deterministic simulation of the streaming-generation
problem shared by AlayaWorld, DreamX-World, Vidu S1, InteractiveAvatar, HelloWorld.

It reproduces, at sub-second scale with no neural network, the three qualitative
claims those papers make:

  1. Autoregressive streaming DRIFTS — per-step error compounds over a long rollout.
  2. Geometry/anchor MEMORY bounds drift and restores consistency on revisits
     (the "reproject a 3D cache" / "retrieve earlier frames by camera geometry" idea).
  3. Training-on-your-own-drift (self-correction) FLATTENS the drift curve.

The point is the *direction and shape* of these effects, not photorealism. A
"frame" is a small vector; a "scene" is a coherent field over positions. This is
the same move the sibling sebastian-llms lab makes: a deterministic substrate so a
result moves only when you change a policy.
"""

from .world import World, out_and_back, straight_line
from .generator import StreamingGenerator
from .metrics import drift_curve, summarize, revisit_error

__all__ = [
    "World",
    "out_and_back",
    "straight_line",
    "StreamingGenerator",
    "drift_curve",
    "summarize",
    "revisit_error",
]

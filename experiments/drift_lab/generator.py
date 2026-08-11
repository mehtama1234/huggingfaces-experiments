from __future__ import annotations

import numpy as np

from .world import World


class StreamingGenerator:
    """A toy autoregressive frame predictor with pluggable memory/correction.

    It predicts the next observation from its *own* previous prediction plus the
    (known) camera motion, with a small per-step model error. That autoregressive
    integration is exactly why real streaming models drift: errors accumulate.

    Strategies:
      - ``sliding``     : no memory. Pure integration → drift compounds.
      - ``anchor``      : geometry memory. The first time a position is seen, store
                          the prediction keyed by position; on revisit, re-ground to
                          the stored value. (Stand-in for a reprojected 3D cache /
                          camera-geometry retrieval.)
      - ``selfcorrect`` : trained-on-own-drift. Each step, pull the prediction back
                          toward the nearest plausible frame on the scene manifold.
    """

    def __init__(
        self,
        world: World,
        strategy: str = "sliding",
        correct_gain: float = 0.35,
        step_err: float = 0.06,
        seed: int = 0,
    ) -> None:
        if strategy not in {"sliding", "anchor", "selfcorrect"}:
            raise ValueError(f"unknown strategy: {strategy}")
        self.world = world
        self.strategy = strategy
        self.correct_gain = correct_gain
        self.step_err = step_err
        self.seed = seed

    def _nearest_scene(self, v: np.ndarray) -> np.ndarray:
        # The scene points are the "plausible frames"; snapping toward the nearest
        # one models a model that learned what valid frames look like.
        d = np.linalg.norm(self.world.scene - v, axis=1)
        return self.world.scene[int(d.argmin())]

    def rollout(self, traj: list[int]) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        preds: list[np.ndarray] = []
        memory: dict[int, np.ndarray] = {}
        for t, pos in enumerate(traj):
            if t == 0:
                p = self.world.obs(pos).copy()  # first frame is grounded
            else:
                prev = traj[t - 1]
                true_delta = self.world.obs(pos) - self.world.obs(prev)  # known camera motion
                eps = rng.standard_normal(self.world.dim) * self.step_err  # model error
                p = preds[-1] + true_delta + eps  # autoregressive integration → drift
                if self.strategy == "anchor" and pos in memory:
                    p = memory[pos].copy()  # re-ground on revisit
                elif self.strategy == "selfcorrect":
                    g = self.correct_gain
                    p = (1 - g) * p + g * self._nearest_scene(p)  # pull back to manifold
            if pos not in memory:
                memory[pos] = p.copy()  # store first-visit prediction
            preds.append(p)
        return np.asarray(preds)

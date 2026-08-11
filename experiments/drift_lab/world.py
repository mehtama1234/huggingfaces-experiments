from __future__ import annotations

import numpy as np


class World:
    """A coherent deterministic 'scene' laid out over 1D positions.

    Each position holds a small feature vector (a stand-in for a frame). The scene
    is smoothed so neighboring positions look similar — a coherent world you can
    pan a camera across. ``obs(pos)`` is the ground-truth observation at a position.
    """

    def __init__(self, length: int = 64, dim: int = 8, seed: int = 0, smooth: int = 12) -> None:
        rng = np.random.default_rng(seed)
        scene = rng.standard_normal((length, dim))
        # Blur along position so the world is spatially coherent, not white noise.
        for _ in range(smooth):
            scene = 0.5 * scene + 0.25 * np.roll(scene, 1, axis=0) + 0.25 * np.roll(scene, -1, axis=0)
        # Normalize scale so per-step error magnitudes are interpretable.
        scene /= (scene.std() + 1e-9)
        self.scene = scene
        self.length = length
        self.dim = dim

    def obs(self, pos: int) -> np.ndarray:
        return self.scene[pos % self.length]


def straight_line(span: int) -> list[int]:
    """Pan from 0 to span-1 once (no revisits)."""
    return list(range(span))


def out_and_back(span: int) -> list[int]:
    """Pan 0→span-1 then back to 0 — every position is revisited exactly once.

    This is the trajectory that exposes revisit consistency: does a place look the
    same when you return to it?
    """
    forward = list(range(span))
    backward = list(range(span - 2, -1, -1))
    return forward + backward

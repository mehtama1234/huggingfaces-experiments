from __future__ import annotations

import numpy as np

from .world import World


def drift_curve(world: World, traj: list[int], preds: np.ndarray) -> np.ndarray:
    """Per-step error between predicted and true observation."""
    truth = np.asarray([world.obs(p) for p in traj])
    return np.linalg.norm(preds - truth, axis=1)


def summarize(curve: np.ndarray) -> dict[str, float]:
    n = len(curve)
    second_half = curve[n // 2 :]
    return {
        "mean": float(curve.mean()),
        "second_half_mean": float(second_half.mean()),
        "max": float(curve.max()),
        "final": float(curve[-1]),
    }


def revisit_error(traj: list[int], curve: np.ndarray) -> dict[str, float]:
    """Compare error the first time each position is seen vs when it's revisited.

    On an out-and-back path, low revisit error means 'the place looks the same when
    you come back' — the consistency property geometry memory is supposed to give.
    """
    first_seen: dict[int, float] = {}
    first_errs: list[float] = []
    revisit_errs: list[float] = []
    for pos, err in zip(traj, curve):
        if pos in first_seen:
            revisit_errs.append(err)
        else:
            first_seen[pos] = err
            first_errs.append(err)
    return {
        "first_visit_mean": float(np.mean(first_errs)) if first_errs else 0.0,
        "revisit_mean": float(np.mean(revisit_errs)) if revisit_errs else float("nan"),
    }


def sparkline(values: np.ndarray, width: int = 48) -> str:
    """A tiny ASCII chart of a curve, normalized to its own max."""
    blocks = "▁▂▃▄▅▆▇█"
    if len(values) > width:
        idx = np.linspace(0, len(values) - 1, width).astype(int)
        values = values[idx]
    hi = values.max() or 1.0
    return "".join(blocks[min(len(blocks) - 1, int(v / hi * (len(blocks) - 1)))] for v in values)

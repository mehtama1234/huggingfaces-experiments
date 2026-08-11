from __future__ import annotations

from drift_lab import StreamingGenerator, World, drift_curve, out_and_back, revisit_error
from drift_lab.metrics import summarize


def _second_half(strategy: str, seed: int = 0):
    world = World(length=64, seed=seed)
    traj = out_and_back(40)
    gen = StreamingGenerator(world, strategy=strategy, seed=seed)
    return summarize(drift_curve(world, traj, gen.rollout(traj)))["second_half_mean"]


def test_determinism():
    world = World(length=64, seed=1)
    traj = out_and_back(30)
    a = StreamingGenerator(world, strategy="sliding", seed=1).rollout(traj)
    b = StreamingGenerator(world, strategy="sliding", seed=1).rollout(traj)
    assert (a == b).all()


def test_sliding_drifts_more_than_memory_and_correction():
    sliding = _second_half("sliding")
    anchor = _second_half("anchor")
    selfcorrect = _second_half("selfcorrect")
    # Both memory and self-correction beat naive streaming on accumulated error.
    assert sliding > anchor
    assert sliding > selfcorrect


def test_anchor_improves_revisit_consistency():
    world = World(length=64, seed=0)
    traj = out_and_back(40)
    sliding_curve = drift_curve(world, traj, StreamingGenerator(world, "sliding", seed=0).rollout(traj))
    anchor_curve = drift_curve(world, traj, StreamingGenerator(world, "anchor", seed=0).rollout(traj))
    sliding_r = revisit_error(traj, sliding_curve)
    anchor_r = revisit_error(traj, anchor_curve)
    # Geometry memory makes revisits look like the first visit; sliding keeps drifting.
    assert anchor_r["revisit_mean"] < sliding_r["revisit_mean"]


def test_drift_actually_grows_for_sliding():
    world = World(length=64, seed=0)
    traj = out_and_back(40)
    curve = drift_curve(world, traj, StreamingGenerator(world, "sliding", seed=0).rollout(traj))
    # error in the back third clearly exceeds the front third
    third = len(curve) // 3
    assert curve[-third:].mean() > curve[:third].mean()

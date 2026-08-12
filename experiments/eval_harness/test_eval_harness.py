from __future__ import annotations

import numpy as np

from eval_harness import (
    Scorecard,
    causality_lift,
    drift,
    response_latency,
    revisit_consistency,
    summarize_curve,
)


def test_drift_detects_growing_vs_flat():
    ref = np.array([1.0, 0.0, 0.0])
    # a sequence that rotates away from the reference (drifts)
    drifting = np.array([[np.cos(t), np.sin(t), 0.0] for t in np.linspace(0, 1.2, 20)])
    flat = np.tile(ref, (20, 1)) + 1e-3
    assert summarize_curve(drift(drifting))["growth"] > 0.1
    assert abs(summarize_curve(drift(flat))["growth"]) < 0.05


def test_revisit_consistency_lower_is_better():
    # positions out-and-back; embeddings identical on revisit => low revisit error
    positions = [0, 1, 2, 1, 0]
    emb = np.array([[0.5, 0.5], [1, 0], [2, 0.3], [1, 0], [0.5, 0.5]], dtype=float)
    r = revisit_consistency(emb, positions)
    assert r["n_revisits"] == 2
    assert r["revisit_mean"] < 1e-6  # came back to identical embeddings


def test_causality_lift_direction_and_effect():
    real = [0.8, 0.7, 0.9, 0.75]
    ctrl = [0.3, 0.2, 0.4, 0.35]
    out = causality_lift(real, ctrl)
    assert out["lift"] > 0.3
    assert out["verdict"] == "reacts to input"
    # no dependence -> flagged
    flat = causality_lift([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert flat["verdict"] == "stimulus-independent"


def test_response_latency():
    out = response_latency(response_steps=[5, 9, 12], trigger_steps=[4, 8, 8])
    assert out["mean_lag"] == (1 + 1 + 4) / 3


def test_scorecard_render():
    sc = Scorecard("demo").add("causality_lift", 0.068).add("drift_growth", 0.42).note("T4 run")
    text = sc.render()
    assert "causality_lift" in text and "demo" in text
    assert sc.to_dict()["metrics"]["causality_lift"] == 0.068

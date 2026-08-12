from __future__ import annotations

import numpy as np


def causality_lift(real_scores: list[float], control_scores: list[float]) -> dict:
    """Does the output actually depend on the stimulus?

    ``real_scores`` = how well each response matched its target given the *real*
    query; ``control_scores`` = the same with a *shuffled* query. A positive lift
    means the model genuinely reacts to the input (the signal InteracVid says
    caption data lacks); ~0 means it's producing generic, stimulus-independent output.
    """
    real = np.asarray(real_scores, dtype=np.float64)
    ctrl = np.asarray(control_scores, dtype=np.float64)
    lift = float(real.mean() - ctrl.mean())
    # paired effect size (Cohen's d over the per-item differences)
    diff = real - ctrl
    d = float(diff.mean() / (diff.std() + 1e-9)) if len(diff) > 1 else float("nan")
    return {
        "real_mean": float(real.mean()),
        "control_mean": float(ctrl.mean()),
        "lift": lift,
        "effect_size": d,
        "verdict": "reacts to input" if lift > 0.02 else "stimulus-independent",
    }


def response_latency(response_steps: list[int], trigger_steps: list[int]) -> dict:
    """Lag between a stimulus and the first visible response, in steps."""
    lags = [r - t for r, t in zip(response_steps, trigger_steps) if r >= t]
    lags = np.asarray(lags, dtype=np.float64)
    return {"mean_lag": float(lags.mean()) if len(lags) else float("nan"),
            "max_lag": float(lags.max()) if len(lags) else float("nan")}


def localization(deviation: np.ndarray, window: slice) -> float:
    """Fraction of a reaction's total deviation that lands inside its intended
    window — high means a well-timed, gated reaction; low means it smears."""
    deviation = np.asarray(deviation, dtype=np.float64)
    total = deviation.sum() + 1e-9
    return float(deviation[window].sum() / total)

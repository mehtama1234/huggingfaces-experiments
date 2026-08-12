from __future__ import annotations

import numpy as np


def _cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return 1.0 - (a @ b)


def drift(embeddings: np.ndarray, reference_idx: int = 0) -> np.ndarray:
    """Per-step distance from every frame to a reference frame.

    Feed identity embeddings (ArcFace/DINO) to measure *identity* drift, or scene
    embeddings to measure *world* drift. A rising curve is the drift the streaming
    papers fight; a flat curve means the memory is holding.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    return _cos_dist(embeddings, embeddings[reference_idx])


def revisit_consistency(embeddings: np.ndarray, positions: list[int], reference: np.ndarray | None = None) -> dict:
    """Does a place look the same when you return to it?

    ``positions[t]`` is where the camera/agent was at step t. For each position seen
    more than once, compare the embedding on the first visit vs later visits.
    Reference (if given) is a per-position ground-truth embedding to compare against;
    otherwise we compare revisit embeddings to their own first-visit embedding.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    first: dict[int, np.ndarray] = {}
    first_errs: list[float] = []
    revisit_errs: list[float] = []
    for t, pos in enumerate(positions):
        emb = embeddings[t]
        if pos in first:
            base = reference[pos] if reference is not None else first[pos]
            revisit_errs.append(float(_cos_dist(emb[None, :], base)[0]))
        else:
            first[pos] = emb
            if reference is not None:
                first_errs.append(float(_cos_dist(emb[None, :], reference[pos])[0]))
    return {
        "first_visit_mean": float(np.mean(first_errs)) if first_errs else 0.0,
        "revisit_mean": float(np.mean(revisit_errs)) if revisit_errs else float("nan"),
        "n_revisits": len(revisit_errs),
    }


def summarize_curve(curve: np.ndarray) -> dict:
    curve = np.asarray(curve, dtype=np.float64)
    n = len(curve)
    front, back = curve[: max(1, n // 3)], curve[-max(1, n // 3) :]
    return {
        "mean": float(curve.mean()),
        "final": float(curve[-1]),
        "max": float(curve.max()),
        "growth": float(back.mean() - front.mean()),  # >0 means it drifted
    }

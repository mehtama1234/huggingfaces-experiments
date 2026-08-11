from __future__ import annotations

import numpy as np


def _ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression map W so that x @ W ≈ y."""
    d = x.shape[1]
    return np.linalg.solve(x.T @ x + lam * np.eye(d), x.T @ y)


def fit_maps(train: dict, shared: str = "pose") -> dict:
    """Learn a linear map from each surface modality into the shared (pose) space.

    The shared space is the pose/geometry space, so it doubles as a decoder: a
    point in shared space *is* a camera trajectory (TriMotion's pose-regressor idea).
    """
    target = train[shared]
    maps = {}
    for mod in ("text", "video"):
        maps[mod] = _ridge(train[mod], target)
    return maps


def embed(sample: np.ndarray, mod: str, maps: dict, shared_gallery: np.ndarray | None = None) -> np.ndarray:
    """Project a surface-form sample into shared space."""
    if mod == "pose":
        return sample
    return sample @ maps[mod]


def retrieval_accuracy(test: dict, gallery: dict, maps: dict, query_mod: str, gallery_mod: str = "pose") -> float:
    """Given a query in one modality, is its nearest gallery item (in shared space)
    the correct matching move? Reports top-1 accuracy across the test set."""
    q = embed(test[query_mod], query_mod, maps)
    g = gallery[gallery_mod] if gallery_mod == "pose" else embed(gallery[gallery_mod], gallery_mod, maps)
    correct = 0
    for i in range(len(q)):
        d = np.linalg.norm(g - q[i], axis=1)
        if int(d.argmin()) == i:
            correct += 1
    return correct / len(q)

from __future__ import annotations

import numpy as np

# Canonical camera-move families, each parameterized by a signed magnitude.
# The "pose" is the per-frame camera delta over T frames — the real geometry.
MOVE_TYPES = ("pan", "tilt", "dolly", "roll")
T = 8  # frames per move


def _pose(move_type: str, mag: float) -> np.ndarray:
    """A T x 6 camera-delta trajectory (tx,ty,tz,rx,ry,rz) for a move."""
    traj = np.zeros((T, 6))
    ramp = np.linspace(0.0, 1.0, T)
    axis = {"pan": 4, "tilt": 3, "dolly": 2, "roll": 5}[move_type]  # which DoF moves
    traj[:, axis] = mag * ramp
    return traj


def make_moves(n_per_type: int = 24, noise: float = 0.05, seed: int = 0):
    """Build paired (pose, text, video) 'surface forms' for many moves.

    Returns a dict with flattened pose vectors and two other modalities that are
    fixed random linear projections of the pose plus per-sample noise — i.e. genuine
    different descriptions of the same underlying motion.
    """
    rng = np.random.default_rng(seed)
    poses, labels = [], []
    for mt in MOVE_TYPES:
        for mag in np.linspace(-1.0, 1.0, n_per_type):
            if abs(mag) < 1e-6:
                continue
            poses.append(_pose(mt, float(mag)).ravel())
            labels.append(f"{mt}:{mag:+.2f}")
    poses = np.asarray(poses)
    d = poses.shape[1]
    # Fixed projections into a "text" space and a "video" space (different dims).
    proj_text = rng.standard_normal((d, 32))
    proj_video = rng.standard_normal((d, 48))
    text = poses @ proj_text + rng.standard_normal((len(poses), 32)) * noise
    video = poses @ proj_video + rng.standard_normal((len(poses), 48)) * noise
    return {"pose": poses, "text": text, "video": video, "labels": labels}

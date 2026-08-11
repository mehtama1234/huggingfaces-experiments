"""motion_space — a tiny, deterministic demo of TriMotion's central claim:

a camera move is ONE underlying thing, so text, a pose trajectory, and a reference
video of the same move can be aligned into ONE shared space. Once aligned, you can
retrieve across modalities (text -> matching pose/video) and compose or interpolate
moves by arithmetic in that space.

No neural net, no video generation. Each 'move' is a real camera trajectory (the
pose). Text and video are different linear 'surface forms' of that trajectory plus
noise. We learn closed-form linear maps from each surface form back to the shared
(pose) space via ridge regression, then measure cross-modal retrieval. The pose
space doubles as the geometry decoder — mirroring TriMotion's pose-regressor.
"""

from .data import make_moves
from .align import fit_maps, embed, retrieval_accuracy

__all__ = ["make_moves", "fit_maps", "embed", "retrieval_accuracy"]

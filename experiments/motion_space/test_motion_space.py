from __future__ import annotations

import numpy as np

from motion_space.align import fit_maps, retrieval_accuracy
from motion_space.data import make_moves
from motion_space.run import _split


def test_cross_modal_retrieval_beats_chance():
    data = make_moves(seed=0)
    train, test = _split(data, seed=0)
    maps = fit_maps(train)
    chance = 1.0 / len(train["labels"])
    for qmod in ("text", "video"):
        acc = retrieval_accuracy(test, test, maps, query_mod=qmod, gallery_mod="pose")
        assert acc > 0.5  # far above near-zero chance
        assert acc > 20 * chance


def test_determinism():
    a = make_moves(seed=3)["text"]
    b = make_moves(seed=3)["text"]
    assert (a == b).all()


def test_noise_hurts_alignment():
    # more noise on the surface forms -> lower retrieval, sanity that it's real signal
    def acc(noise):
        data = make_moves(seed=1, noise=noise)
        train, test = _split(data, seed=1)
        maps = fit_maps(train)
        return retrieval_accuracy(test, test, maps, query_mod="text", gallery_mod="pose")

    assert acc(0.02) >= acc(0.8)

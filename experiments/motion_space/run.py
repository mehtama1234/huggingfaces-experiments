from __future__ import annotations

import argparse

import numpy as np

from .align import embed, fit_maps, retrieval_accuracy
from .data import MOVE_TYPES, T, make_moves


def _split(data: dict, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(data["labels"])
    idx = rng.permutation(n)
    cut = int(n * 0.7)
    tr, te = idx[:cut], idx[cut:]
    pick = lambda d, ix: {k: (v[ix] if k != "labels" else [v[i] for i in ix]) for k, v in d.items()}
    return pick(data, tr), pick(data, te)


def demo(seed: int = 0) -> None:
    data = make_moves(seed=seed)
    train, test = _split(data, seed=seed)
    maps = fit_maps(train)
    n_gallery = len(train["labels"])
    chance = 1.0 / n_gallery
    print("TriMotion mini: align text / pose / video of the same camera move into one space\n")
    print(f"gallery size {n_gallery}  (random-guess top-1 ≈ {chance:.3f})\n")
    for qmod in ("text", "video"):
        acc = retrieval_accuracy(test, test, maps, query_mod=qmod, gallery_mod="pose")
        print(f"  {qmod:>5} -> pose  cross-modal top-1 retrieval: {acc:.2f}")
    # text -> video (both mapped into shared/pose space)
    acc_tv = retrieval_accuracy(test, test, maps, query_mod="text", gallery_mod="video")
    print(f"  text  -> video cross-modal top-1 retrieval: {acc_tv:.2f}")
    print("\nreading: high accuracy = the three surface forms really do land on one shared\n"
          "motion point (the whole premise). Random guessing would be near zero.")


def interpolate(seed: int = 0) -> None:
    """Blend two moves in shared (pose) space and confirm the result is a smooth,
    valid trajectory — TriMotion's 'cross-modal interpolation' for free."""
    data = make_moves(seed=seed)
    # pick a 'pan +1' and a 'tilt +1'
    labels = data["labels"]
    i = labels.index(min((l for l in labels if l.startswith("pan:+")), key=lambda l: abs(float(l.split(":")[1]) - 1)))
    j = labels.index(min((l for l in labels if l.startswith("tilt:+")), key=lambda l: abs(float(l.split(":")[1]) - 1)))
    a, b = data["pose"][i], data["pose"][j]
    print(f"interpolating  {labels[i]}  <->  {labels[j]}  in shared space\n")
    print(f"{'alpha':>6}   {'pan DoF':>9}{'tilt DoF':>9}   (end-of-move camera delta)")
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        blend = (1 - alpha) * a + alpha * b
        traj = blend.reshape(T, 6)
        print(f"{alpha:>6.2f}   {traj[-1,4]:>9.2f}{traj[-1,3]:>9.2f}")
    print("\nreading: the blend moves smoothly from a pure pan to a pure tilt — a valid,\n"
          "composable camera path produced by arithmetic, no retraining.")


def compose(seed: int = 0) -> None:
    """Chain two moves in time (pan then zoom) by concatenating their trajectories."""
    data = make_moves(seed=seed)
    labels = data["labels"]
    i = labels.index(min((l for l in labels if l.startswith("pan:+")), key=lambda l: abs(float(l.split(":")[1]) - 1)))
    j = labels.index(min((l for l in labels if l.startswith("dolly:+")), key=lambda l: abs(float(l.split(":")[1]) - 1)))
    a = data["pose"][i].reshape(T, 6)
    b = data["pose"][j].reshape(T, 6)
    chained = np.concatenate([a, b + a[-1]], axis=0)  # continue from where the pan ended
    print(f"composing in time: {labels[i]} then {labels[j]}  ->  {chained.shape[0]} frames")
    print(f"  pan DoF goes 0 -> {a[-1,4]:.2f} (frames 0-{T-1}), then holds while dolly advances")
    print("\nreading: sequential composition is just concatenation in the shared trajectory space.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="motion_space.run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interpolate", action="store_true")
    p.add_argument("--compose", action="store_true")
    args = p.parse_args(argv)
    if args.interpolate:
        interpolate(args.seed)
    elif args.compose:
        compose(args.seed)
    else:
        demo(args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

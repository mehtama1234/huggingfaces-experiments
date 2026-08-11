from __future__ import annotations

import argparse

import numpy as np

from .generator import StreamingGenerator
from .metrics import drift_curve, revisit_error, sparkline, summarize
from .world import World, out_and_back, straight_line

STRATEGIES = ("sliding", "anchor", "selfcorrect")


def compare(span: int = 40, revisit: bool = True, seed: int = 0) -> None:
    world = World(length=max(64, span + 4), seed=seed)
    traj = out_and_back(span) if revisit else straight_line(span)
    print(f"trajectory: {'out-and-back (revisits)' if revisit else 'straight pan'} · {len(traj)} steps\n")
    print(f"{'strategy':<13}{'2nd-half err':>13}{'max':>8}   drift over time")
    print("-" * 78)
    for strat in STRATEGIES:
        gen = StreamingGenerator(world, strategy=strat, seed=seed)
        curve = drift_curve(world, traj, gen.rollout(traj))
        s = summarize(curve)
        print(f"{strat:<13}{s['second_half_mean']:>13.3f}{s['max']:>8.2f}   {sparkline(curve)}")
    if revisit:
        print("\nrevisit consistency (does a place look the same when you return?):")
        for strat in STRATEGIES:
            gen = StreamingGenerator(world, strategy=strat, seed=seed)
            curve = drift_curve(world, traj, gen.rollout(traj))
            r = revisit_error(traj, curve)
            print(f"  {strat:<12} first-visit err {r['first_visit_mean']:.3f}  ->  revisit err {r['revisit_mean']:.3f}")
    print(
        "\nreading: sliding drifts (error grows); anchor (geometry memory) bounds it and\n"
        "cuts revisit error; selfcorrect (train-on-own-drift) flattens the curve."
    )


def measure_latency(span: int = 40, seed: int = 0) -> None:
    """How many steps of lag between a control change and a visible response,
    as a function of chunk size. The generator only re-reads the control at chunk
    boundaries, so bigger chunks = more responsive-feeling but laggier."""
    print("interactivity/latency: response lag vs chunk size (smaller = snappier)\n")
    print(f"{'chunk size':<12}{'mean response lag (steps)':>26}")
    print("-" * 38)
    for chunk in (1, 2, 4, 8, 16):
        lags = []
        for trigger in range(1, span):
            # output only updates at chunk boundaries after the trigger
            next_boundary = ((trigger // chunk) + 1) * chunk
            lags.append(next_boundary - trigger)
        print(f"{chunk:<12}{float(np.mean(lags)):>26.2f}")
    print("\nreading: lag scales with chunk size — the core real-time interactivity trade.")


def social_gate(span: int = 40, window: int = 4, seed: int = 0) -> None:
    """Gated event injection stays localized; always-on smears across the rollout.
    Reports what fraction of the total deviation lands inside the intended window."""
    dim = 8
    rng = np.random.default_rng(seed)
    event = rng.standard_normal(dim)
    k = span // 2
    steps = np.arange(span)
    gated = np.zeros(span)
    always = np.zeros(span)
    for t in steps:
        if k <= t < k + window:
            gated[t] = np.linalg.norm(event)
        if t >= k:
            always[t] = np.linalg.norm(event)
    in_window = slice(k, k + window)

    def localization(dev: np.ndarray) -> float:
        return float(dev[in_window].sum() / (dev.sum() + 1e-9))

    print("social prompt injected at step", k, f"(intended window: {window} steps)\n")
    print(f"  gated  localization {localization(gated):.2f}   {sparkline(gated)}")
    print(f"  always localization {localization(always):.2f}   {sparkline(always)}")
    print(
        "\nreading: a training-free temporal gate keeps the reaction where it belongs;\n"
        "always-on conditioning smears it across the whole stream (HelloWorld's idea)."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="drift_lab.run", description="Streaming world-model drift/memory/latency lab.")
    p.add_argument("--strategy", choices=STRATEGIES, help="run just one strategy's drift curve")
    p.add_argument("--window", type=int, default=1, help="(reserved) sliding window size")
    p.add_argument("--span", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--revisit", action="store_true", help="use an out-and-back trajectory")
    p.add_argument("--measure-latency", action="store_true")
    p.add_argument("--social-gate", action="store_true")
    args = p.parse_args(argv)

    if args.measure_latency:
        measure_latency(span=args.span, seed=args.seed)
    elif args.social_gate:
        social_gate(span=args.span, seed=args.seed)
    elif args.strategy:
        world = World(length=max(64, args.span + 4), seed=args.seed)
        traj = out_and_back(args.span) if args.revisit else straight_line(args.span)
        gen = StreamingGenerator(world, strategy=args.strategy, seed=args.seed)
        curve = drift_curve(world, traj, gen.rollout(traj))
        s = summarize(curve)
        print(f"{args.strategy}: 2nd-half err {s['second_half_mean']:.3f}, max {s['max']:.2f}")
        print(sparkline(curve))
    else:
        compare(span=args.span, revisit=True, seed=args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

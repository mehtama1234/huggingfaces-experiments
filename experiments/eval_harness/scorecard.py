from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Scorecard:
    """A one-line-per-metric report for one model/config on one benchmark.

    Deliberately dumb: you fill it with whatever metrics you computed and it renders
    a readable table. The point is a *consistent* scorecard across the papers so runs
    are comparable, not a fixed metric set.
    """

    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, key: str, value: float) -> "Scorecard":
        self.metrics[key] = float(value)
        return self

    def note(self, text: str) -> "Scorecard":
        self.notes.append(text)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "metrics": self.metrics, "notes": self.notes}

    def render(self) -> str:
        lines = [f"scorecard: {self.name}", "-" * (11 + len(self.name))]
        width = max((len(k) for k in self.metrics), default=0)
        for k, v in self.metrics.items():
            lines.append(f"  {k:<{width}} : {v:+.3f}")
        for n in self.notes:
            lines.append(f"  · {n}")
        return "\n".join(lines)

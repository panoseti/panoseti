"""
State-aware test scheduler.
Groups pytest items into batches by (batch_priority, required_state),
computes minimum-cost state-transition plans between batches, and emits a
human-readable batch plan before any hardware is touched.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TESTS_TOML_PATH = Path(__file__).parent.parent / "hw_tests.toml"


@dataclass
class Batch:
    priority: int
    hw_class: str
    required_state: str
    leaves_state: str
    items: list[Any] = field(default_factory=list)
    transition_plan: list[Any] = field(default_factory=list)  # list[Primitive]
    transition_cost_s: float = 0.0


def _load_tests_toml(path: Path = _TESTS_TOML_PATH) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _match_class(node_id: str, mappings: list[dict[str, Any]]) -> str | None:
    """Return the class name for a test node ID via glob matching."""
    for m in mappings:
        if fnmatch.fnmatch(node_id, m["glob"]):
            return m["class"]
    return None


class StateAwareScheduler:
    """
    Groups tests by TOML class and produces an optimised execution order.
    """

    def __init__(
        self,
        state_machine: Any,  # HardwareStateMachine
        tests_toml_path: Path = _TESTS_TOML_PATH,
    ):
        self.sm = state_machine
        self._data = _load_tests_toml(tests_toml_path)
        self._classes: dict[str, Any] = self._data.get("classes", {})
        self._mappings: list[dict[str, Any]] = self._data.get("mapping", [])

    def class_for(self, node_id: str) -> str | None:
        return _match_class(node_id, self._mappings)

    def schedule(self, items: list[Any]) -> list[Any]:
        """
        Re-order pytest items into state-coherent batches.

        Items without a matching TOML class are left at the end unchanged.
        """
        classified: dict[str, list[Any]] = {}
        unclassified: list[Any] = []

        for item in items:
            cls = self.class_for(item.nodeid)
            if cls:
                classified.setdefault(cls, []).append(item)
            else:
                unclassified.append(item)

        batches = self._build_batches(classified)
        ordered: list[Any] = []
        for b in batches:
            ordered.extend(b.items)
        ordered.extend(unclassified)
        return ordered

    def build_plan(
        self, items: list[Any], assume_state: str | None = None
    ) -> list[Batch]:
        """
        Compute the full batch plan for a collection of test items.
        *assume_state* is the believed current hardware state; defaults to sm.initial.
        """
        classified: dict[str, list[Any]] = {}
        for item in items:
            cls = self.class_for(item.nodeid)
            if cls:
                classified.setdefault(cls, []).append(item)

        batches = self._build_batches(classified)
        current = assume_state or self.sm.initial
        for b in batches:
            if current != b.required_state:
                plan = self.sm.plan(current, b.required_state)
                b.transition_plan = plan
                b.transition_cost_s = self.sm.cost(plan)
            current = b.leaves_state
        return batches

    def _build_batches(self, classified: dict[str, list[Any]]) -> list[Batch]:
        batches: list[Batch] = []
        for cls_name, class_items in classified.items():
            cls_cfg = self._classes.get(cls_name, {})
            batches.append(Batch(
                priority=cls_cfg.get("batch_priority", 99),
                hw_class=cls_name,
                required_state=cls_cfg.get("required_state", self.sm.initial),
                leaves_state=cls_cfg.get("leaves_state", cls_cfg.get("required_state", self.sm.initial)),
                items=class_items,
            ))
        batches.sort(key=lambda b: b.priority)
        return batches

    def format_plan(self, batches: list[Batch]) -> str:
        """Return a rich-annotated plan string suitable for console output."""
        lines: list[str] = ["[bold]HITL Batch Plan[/bold]"]
        total_s = sum(b.transition_cost_s for b in batches)

        for i, b in enumerate(batches):
            if b.transition_plan:
                transitions = " → ".join(
                    f"[yellow]{p.name}[/yellow] ([dim]{p.budget_s['typical']:.0f}s[/dim])"
                    for p in b.transition_plan
                )
                lines.append(f"  [dim]→ {transitions}[/dim]")
            lines.append(
                f"  [cyan]Batch {i + 1}[/cyan] [{b.hw_class}] "
                f"({len(b.items)} tests, target=[green]{b.required_state}[/green])"
            )

        minutes, seconds = divmod(int(total_s), 60)
        lines.append(f"\n  [bold]Estimated transition overhead:[/bold] {minutes} min {seconds} s")
        return "\n".join(lines)

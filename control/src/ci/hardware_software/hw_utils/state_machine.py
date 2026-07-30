"""
Hardware State Machine.
Loads states and transition primitives from hw_state_machine.toml,
builds a directed cost-graph, and provides shortest-path planning and execution.
"""

from __future__ import annotations

import importlib
import itertools
import json
import logging
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_TOML_PATH = Path(__file__).parent.parent / "hw_state_machine.toml"
_STATE_FILE_NAME = "hw_runtime_state.json"


class Primitive(BaseModel):
    name: str
    from_states: list[str]
    to_state: str
    budget_s: dict[str, float]
    safety: str
    entrypoint: str
    kwargs: dict[str, Any] = {}
    guards: list[str] = []


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _build_state_enum(state_names: list[str]) -> type[StrEnum]:
    return StrEnum("HwState", {s: s for s in state_names})  # type: ignore[return-value]


class HardwareStateMachine:
    """
    State machine for hardware transitions.

    Primitives from the TOML become directed edges in a weighted graph;
    `plan()` finds the minimum-cost path via networkx.
    """

    def __init__(self, toml_path: Path = _TOML_PATH):
        data = _load_toml(toml_path)
        sm = data["state_machine"]

        self.initial: str = sm["initial"]
        self.safe: str = sm["safe"]

        self._states: list[str] = [s["name"] for s in data.get("states", [])]
        self.HwState = _build_state_enum(self._states)

        self.primitives: dict[str, Primitive] = {}
        for raw in data.get("primitives", []):
            p = Primitive(**raw)
            self.primitives[p.name] = p

        self.graph: nx.DiGraph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        g: nx.DiGraph = nx.DiGraph()
        g.add_nodes_from(self._states)
        # Build index map so we can resolve wildcard "downgrade" semantics.
        # Wildcard primitives are always downgrades (e.g., hv_off, wps_power_off).
        # They are only physically meaningful from states that are "above" the
        # target state in the cold→hot ordering — never from states below.
        idx = {s: i for i, s in enumerate(self._states)}
        for p in self.primitives.values():
            if p.from_states == ["*"]:
                target_idx = idx.get(p.to_state, 0)
                # Only connect states strictly above to_state in the hierarchy.
                froms = [s for s in self._states if idx[s] > target_idx]
            else:
                froms = p.from_states
            for src in froms:
                if src in self._states:
                    g.add_edge(src, p.to_state, weight=p.budget_s["typical"], primitive=p.name)
        return g

    def plan(self, current: str, target: str) -> list[Primitive]:
        """Return the cheapest sequence of primitives to reach *target* from *current*."""
        if current == target:
            return []
        try:
            path = nx.shortest_path(self.graph, current, target, weight="weight")
        except nx.NetworkXNoPath as exc:
            raise ValueError(f"No path from {current!r} to {target!r} in state machine") from exc
        result: list[Primitive] = []
        for src, dst in itertools.pairwise(path):
            edge = self.graph[src][dst]
            result.append(self.primitives[edge["primitive"]])
        return result

    def cost(self, primitives: list[Primitive]) -> float:
        """Sum of typical budget_s for a list of primitives."""
        return sum(p.budget_s["typical"] for p in primitives)

    def execute(
        self,
        primitives: list[Primitive],
        dry_run: bool = False,
        state_file: Path | None = None,
    ) -> None:
        """Execute each primitive in sequence, updating the state file on each success."""
        for p in primitives:
            logger.info("Primitive %s: %s → %s (%.0fs typical)", p.name, p.from_states, p.to_state, p.budget_s["typical"])
            if dry_run:
                continue
            for guard_name in p.guards:
                guard_fn = _resolve_entrypoint(f"ci.hardware_software.hw_utils.guards:{guard_name}")
                if not guard_fn():
                    raise RuntimeError(f"Guard {guard_name!r} failed before {p.name!r}")
            fn = _resolve_entrypoint(p.entrypoint)
            fn(**p.kwargs)
            if state_file:
                _write_state(state_file, p.to_state)
            logger.info("Primitive %s complete → %s", p.name, p.to_state)


def _resolve_entrypoint(entrypoint: str) -> Any:
    """Resolve 'module.path:function' (or 'module.path:Class.method') to a callable."""
    module_path, attr_path = entrypoint.split(":", 1)
    mod = importlib.import_module(module_path)
    obj: Any = mod
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def _write_state(state_file: Path, state: str) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"state": state}))


def read_state(state_file: Path) -> str | None:
    """Read the last known hardware state from disk. Returns None if unknown."""
    try:
        data = json.loads(state_file.read_text())
        return data.get("state")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

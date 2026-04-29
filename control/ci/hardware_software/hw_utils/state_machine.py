"""
Hardware State Machine implementation.
Loads state DAG and transition primitives from TOML.
"""

from enum import Enum
from pydantic import BaseModel
from typing import Any

class HwState(str, Enum):
    """Generated from TOML states."""
    pass

class Primitive(BaseModel):
    name: str
    from_states: list[str]
    to_state: str
    budget_s: dict[str, float]
    safety: str
    entrypoint: str
    kwargs: dict[str, Any] = {}
    guards: list[str] = []

class HardwareStateMachine:
    """Manages state transitions and planning."""
    def __init__(self, toml_path: str):
        pass

    def plan(self, current: HwState, target: HwState) -> list[Primitive]:
        return []

    def execute(self, plan: list[Primitive], dry_run: bool = False):
        pass

"""
Pytest plugin for HITL testing.
Registers at pytest_configure time; classifies tests from hw_tests.toml,
gates on topology requirements, reorders into state-coherent batches.
"""

from __future__ import annotations

import fnmatch
import logging
import tomllib
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

_TESTS_TOML_PATH = Path(__file__).parent.parent / "hw_tests.toml"
_STATE_MACHINE_TOML_PATH = Path(__file__).parent.parent / "hw_state_machine.toml"

# Populated at collection time; used by the terminal summary hook.
_batch_plan: list[Any] = []


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "hw_class(name): HITL test class (driver_protocol, fast_reconfig, observing, lifecycle, telemetry)",
    )
    config.addinivalue_line(
        "markers",
        "required_state(state): minimum HW state required by this test",
    )
    config.addinivalue_line(
        "markers",
        "slow_hw: markers tests with long hardware timeouts (>60s)",
    )
    # Bare class markers registered for -m filtering (e.g. -m driver_protocol)
    for _cls in ("driver_protocol", "fast_reconfig", "observing", "lifecycle", "telemetry"):
        config.addinivalue_line("markers", f"{_cls}: HITL test class {_cls!r}")


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """
    1. Match each item to a TOML class via glob.
    2. Attach required_state / leaves_state as user_properties.
    3. Gate items against [[requirements]]; skip those that fail.
    4. Reorder items by (batch_priority, required_state).
    """
    try:
        data = _load_toml(_TESTS_TOML_PATH)
    except FileNotFoundError:
        logger.warning("hw_tests.toml not found; skipping HITL classification")
        return

    classes: dict[str, Any] = data.get("classes", {})
    mappings: list[dict[str, Any]] = data.get("mapping", [])
    requirements: list[dict[str, Any]] = data.get("requirements", [])

    # Try loading topology for capability gating; skip gating if unavailable.
    topology = _try_load_topology()

    classified: list[tuple[int, int, Any]] = []  # (priority, idx, item)
    unclassified: list[tuple[int, int, Any]] = []

    for idx, item in enumerate(items):
        node_id = item.nodeid

        # Check topology requirements first
        skip_reason = _check_requirements(node_id, requirements, topology)
        if skip_reason:
            item.add_marker(pytest.mark.skip(reason=skip_reason))

        # Find matching class
        cls_name = _match_class(node_id, mappings)
        if cls_name and cls_name in classes:
            cls_cfg = classes[cls_name]
            item.user_properties.append(("hw_class", cls_name))
            item.user_properties.append(("required_state", cls_cfg.get("required_state", "")))
            item.user_properties.append(("leaves_state", cls_cfg.get("leaves_state", "")))
            item.add_marker(pytest.mark.hw_class(cls_name))
            item.add_marker(getattr(pytest.mark, cls_name))  # bare class marker for -m filtering
            item.add_marker(pytest.mark.required_state(cls_cfg.get("required_state", "")))
            priority = cls_cfg.get("batch_priority", 99)
            classified.append((priority, idx, item))
        else:
            unclassified.append((999, idx, item))

    # Sort classified items by (priority, original_index) to preserve within-class order
    classified.sort(key=lambda t: (t[0], t[1]))

    ordered = [item for _, _, item in classified] + [item for _, _, item in unclassified]
    items[:] = ordered


def pytest_runtest_setup(item: Any) -> None:
    """
    Before each test, ensure hardware is in the required state.
    If a transition fails, skip the test (and subsequent tests in the same batch).
    """
    # 1. Get required state from markers
    marker = item.get_closest_marker("required_state")
    if not marker:
        return
    target_state = marker.args[0]
    if not target_state:
        return

    # 2. Get current state
    from ci.hardware_software.hw_utils.cli import _STATE_FILE
    from ci.hardware_software.hw_utils.state_machine import HardwareStateMachine, read_state
    
    # Check session-wide reachability first if we require BOOTED or above
    if target_state != "UNPOWERED":
        try:
            # We look for the session-scoped fixture value if possible, 
            # but hooks are tricky with fixtures. 
            # For simplicity, we just run a cached check or the check itself.
            import time

            from control.utils.paths import PanoPaths
            cache_file = PanoPaths.tmp_dir() / ".topology_reachable_cache"
            if not cache_file.exists() or (time.time() - cache_file.stat().st_mtime) > 300:
                from ci.hardware_software.hw_utils.topology import HwTopology
                from control.utils import util
                topo = HwTopology()
                errors = []
                for a in topo.quabo_ips():
                    try:
                        if not util.ping(a.real_ip, a.cmd_port):
                            errors.append(f"{a.ip} (loc={a.boardloc}) unreachable via util.ping")
                    except Exception as exc:
                        errors.append(f"{a.ip} (loc={a.boardloc}) unreachable error: {exc}")
                if errors:
                    pytest.skip("Topology unreachable:\n" + "\n".join(errors))
                cache_file.touch()
        except BaseException as exc:
            from _pytest.outcomes import OutcomeException
            if isinstance(exc, OutcomeException):
                raise
            logger.debug("Topology reachability check failed: %s", exc)

    current_state = read_state(_STATE_FILE)
    if current_state == target_state:
        return

    # 3. Plan and execute transition
    sm = HardwareStateMachine()
    try:
        plan = sm.plan(current_state or sm.initial, target_state)
        if plan:
            logger.info("Transitioning for %s: %s → %s", item.nodeid, current_state, target_state)
            sm.execute(plan, state_file=_STATE_FILE)
    except Exception as exc:
        pytest.skip(f"State transition to {target_state} failed: {exc}")


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    """Print the HITL batch plan vs actual execution summary if any HITL items ran."""
    if not _batch_plan:
        return
    terminalreporter.write_sep("=", "HITL State Machine Summary")
    for b in _batch_plan:
        if b.transition_plan:
            transitions = " → ".join(f"{p.name} ({p.budget_s['typical']:.0f}s)" for p in b.transition_plan)
            terminalreporter.write_line(f"  Transition: {transitions}")
        terminalreporter.write_line(
            f"  Batch [{b.hw_class}] ({len(b.items)} tests) target={b.required_state}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _match_class(node_id: str, mappings: list[dict[str, Any]]) -> str | None:
    for m in mappings:
        if fnmatch.fnmatch(node_id, m["glob"]):
            return m["class"]
    return None


def _check_requirements(
    node_id: str,
    requirements: list[dict[str, Any]],
    topology: Any,
) -> str | None:
    """Return a skip reason string, or None if all requirements are met."""
    for req in requirements:
        if not fnmatch.fnmatch(node_id, req["glob"]):
            continue
        if topology is None:
            return "HwTopology unavailable (no real hardware configs)"
        result = topology.gate(node_id, req)
        if result is not True:
            return str(result)
    return None


def _try_load_topology() -> Any:
    try:
        from ci.hardware_software.hw_utils.topology import HwTopology
        return HwTopology()
    except Exception as exc:
        logger.debug("HwTopology load failed (normal in software-only CI): %s", exc)
        return None

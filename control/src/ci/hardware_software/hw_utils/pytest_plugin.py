"""
Pytest plugin for HITL testing (MVP edition).

Responsibilities:
  1. Register hw_class / required_state / slow_hw markers.
  2. Classify each item from hw_tests.toml via glob matching.
  3. Gate items against [[requirements]] (topology capability checks).
  4. Order items: env_check (-2) < boot_sequence (-1) < happy_path (0).
  5. Skip happy_path items if the boot_sequence test failed during this session
     (tracked via a session-scoped stash key).

The previous transition-planning logic (auto-driving state machine for each
test) is intentionally removed.  The boot_sequence test issues every CLI
command itself; happy_path tests are skipped if boot failed.  This avoids
the complexity of mid-session state recovery, which was unreliable because
cmd 0x04 (soft reset) does not re-enter the TFTP bootloader.
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

# Session-level stash key set to True when a boot_sequence test fails.
_BOOT_FAILED_KEY = pytest.StashKey[bool]()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "hw_class(name): HITL test class (env_check, boot_sequence, happy_path)",
    )
    config.addinivalue_line("markers", "required_state(state): minimum HW state required")
    config.addinivalue_line("markers", "slow_hw: test with long hardware timeouts (>60s)")
    for cls in ("env_check", "boot_sequence", "happy_path"):
        config.addinivalue_line("markers", f"{cls}: HITL test class {cls!r}")


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """
    1. Match each item to a TOML class via glob.
    2. Attach hw_class / required_state as markers and user_properties.
    3. Gate against [[requirements]]; skip those that fail.
    4. Reorder by (batch_priority, original_index).
    """
    try:
        data = _load_toml(_TESTS_TOML_PATH)
    except FileNotFoundError:
        logger.warning("hw_tests.toml not found; skipping HITL classification")
        return

    classes: dict[str, Any] = data.get("classes", {})
    mappings: list[dict[str, Any]] = data.get("mapping", [])
    requirements: list[dict[str, Any]] = data.get("requirements", [])

    topology = _try_load_topology()

    classified: list[tuple[int, int, Any]] = []
    unclassified: list[tuple[int, int, Any]] = []

    for idx, item in enumerate(items):
        node_id = item.nodeid

        skip_reason = _check_requirements(node_id, requirements, topology)
        if skip_reason:
            item.add_marker(pytest.mark.skip(reason=skip_reason))

        cls_name = _match_class(node_id, mappings)
        if cls_name and cls_name in classes:
            cls_cfg = classes[cls_name]
            item.user_properties.append(("hw_class", cls_name))
            item.user_properties.append(("required_state", cls_cfg.get("required_state", "")))
            item.user_properties.append(("leaves_state", cls_cfg.get("leaves_state", "")))
            item.add_marker(pytest.mark.hw_class(cls_name))
            item.add_marker(getattr(pytest.mark, cls_name))
            item.add_marker(pytest.mark.required_state(cls_cfg.get("required_state", "")))
            priority = cls_cfg.get("batch_priority", 99)
            classified.append((priority, idx, item))
        else:
            unclassified.append((999, idx, item))

    classified.sort(key=lambda t: (t[0], t[1]))
    items[:] = [item for _, _, item in classified] + [item for _, _, item in unclassified]


# Module-level mutable flag shared between logreport and setup hooks.
_session_boot_failed: list[bool] = [False]


def pytest_runtest_logreport(report: Any) -> None:
    """Mark session-level boot failure when any boot_sequence test fails."""
    if report.when != "call" or not report.failed:
        return
    if "hw_boot" in report.nodeid or "boot_sequence" in report.nodeid:
        _session_boot_failed[0] = True
        logger.warning("boot_sequence test failed — happy_path tests will be skipped")


def pytest_runtest_setup(item: Any) -> None:
    """Skip happy_path tests if the boot_sequence already failed this session."""
    for mark in item.iter_markers("hw_class"):
        if mark.args and mark.args[0] == "happy_path" and _session_boot_failed[0]:
            pytest.skip("Skipping happy_path: boot_sequence failed earlier in this session")


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
    for req in requirements:
        if not fnmatch.fnmatch(node_id, req["glob"]):
            continue
        if topology is None:
            return "HwTopology unavailable"
        result = topology.gate(node_id, req)
        if result is not True:
            return str(result)
    return None


def _try_load_topology() -> Any:
    try:
        from ci.hardware_software.hw_utils.topology import HwTopology
        return HwTopology()
    except Exception as exc:
        logger.debug("HwTopology load failed: %s", exc)
        return None

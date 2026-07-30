"""
Safety Manager for HITL tests.
Installs atexit + SIGTERM/SIGINT handlers that always drive hardware to the
'safe' state defined in hw_state_machine.toml, independent of pytest fixtures.
"""

from __future__ import annotations

import atexit
import logging
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_manager: SafetyManager | None = None


def get_safety_manager() -> SafetyManager | None:
    return _manager


class SafetyManager:
    """
    Registers irrevocable exit hooks that power off hardware.

    Uses raw subprocess calls (not pytest fixtures) so the guarantee holds
    even through fixture teardown failures, OOM kills, or KeyboardInterrupt.

    Set `keep_running=True` (--keep-running CLI flag) to disable the safety
    stop for intentional warm-hardware dev sessions — a loud banner is printed.
    """

    def __init__(
        self,
        state_machine: HardwareStateMachine,  # type: ignore[name-defined]  # noqa: F821
        state_file: Path,
        keep_running: bool = False,
    ):
        from ci.hardware_software.hw_utils.state_machine import HardwareStateMachine  # noqa: F401
        self.sm = state_machine
        self.state_file = state_file
        self.keep_running = keep_running
        self._registered = False

    def register(self) -> None:
        """Install atexit + signal handlers. Idempotent."""
        global _manager
        _manager = self
        if self._registered:
            return
        atexit.register(self._on_exit)
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        self._registered = True
        logger.info("SafetyManager registered (keep_running=%s)", self.keep_running)

    def emergency_teardown(self) -> None:
        """Drive hardware to the safe state. Idempotent and exception-safe."""
        if self.keep_running:
            _banner("KEEP-RUNNING MODE — hardware was NOT returned to safe state")
            return

        from ci.hardware_software.hw_utils.state_machine import read_state

        current = read_state(self.state_file) or self.sm.initial
        target = self.sm.safe

        if current == target:
            logger.info("SafetyManager: hardware already in safe state %r", target)
            return

        logger.warning("SafetyManager: driving %r → %r", current, target)
        try:
            plan = self.sm.plan(current, target)
            self.sm.execute(plan, state_file=self.state_file)
        except Exception as exc:
            logger.error("SafetyManager: state machine failed (%s), using emergency WPS off", exc)
            _emergency_wps_off()

    def _on_exit(self) -> None:
        try:
            self.emergency_teardown()
        except Exception as exc:
            logger.error("SafetyManager atexit: %s", exc)

    def _on_signal(self, signum: int, frame: object) -> None:
        logger.warning("SafetyManager: caught signal %d", signum)
        try:
            self.emergency_teardown()
        except Exception as exc:
            logger.error("SafetyManager signal handler: %s", exc)
        sys.exit(128 + signum)


def _emergency_wps_off() -> None:
    """Last-resort: call driver_ops.wps_power_off() directly."""
    try:
        from ci.hardware_software.hw_utils.driver_ops import wps_power_off
        wps_power_off()
    except Exception as exc:
        logger.error("Emergency WPS off failed: %s", exc)


def _banner(msg: str) -> None:
    line = "=" * 72
    print(f"\n\033[91m{line}\n  ⚠  {msg}\n{line}\033[0m\n", file=sys.stderr)

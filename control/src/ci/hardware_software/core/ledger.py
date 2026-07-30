"""
Run state ledger helpers.

Wraps RunStateManager.load_state() with polling utilities used by happy-path
tests to wait for the current run to reach a desired status.
"""

from __future__ import annotations

import time

from control.utils.pydantic_config_models import RunStateLedger, RunStatus
from control.utils.run_state import RunStateManager


def load() -> RunStateLedger | None:
    """Return the current ledger, or None if no run exists."""
    return RunStateManager().load_state()


def current_run_name() -> str:
    """Return the current run name from the ledger.

    Raises:
        AssertionError: if no ledger exists or run_name is empty.
    """
    ledger = load()
    assert ledger is not None, "No ledger found — is a run in progress?"
    assert ledger.run_name, "Ledger has no run_name"
    return ledger.run_name


def current_status() -> str | None:
    """Return the current ledger status string, or None if no run."""
    ledger = load()
    return str(ledger.status) if ledger else None


def wait_for_status(target: str | RunStatus, timeout: float = 120.0) -> str:
    """Poll the ledger until it reaches *target* status (or a terminal error).

    Args:
        target: The desired RunStatus value (e.g. "ARCHIVED").
        timeout: Maximum seconds to wait.

    Returns:
        The observed status string when the target is reached.

    Raises:
        AssertionError: if the timeout expires or a terminal error status is
            reached before the target.
    """
    target_str = str(target)
    terminal_errors = {"ABORTED", "TRANSFER_FAILED", "VERIFY_FAILED", "STOPPED_WITH_ERRORS"}
    deadline = time.monotonic() + timeout
    last_status: str | None = None
    while time.monotonic() < deadline:
        ledger = load()
        if ledger is not None:
            last_status = str(ledger.status)
            if last_status == target_str:
                return last_status
            if last_status in terminal_errors and target_str not in terminal_errors:
                raise AssertionError(
                    f"Ledger reached error status {last_status!r} while waiting for {target_str!r}"
                )
        time.sleep(2.0)
    raise AssertionError(
        f"Ledger did not reach {target_str!r} within {timeout:.0f}s "
        f"(last observed: {last_status!r})"
    )

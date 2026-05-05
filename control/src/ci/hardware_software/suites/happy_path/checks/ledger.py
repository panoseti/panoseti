"""Ledger-based assertions for happy-path tests."""

from __future__ import annotations

from ci.hardware_software.core import ledger as _ledger


def is_active(run_name: str) -> None:
    """Assert the ledger shows the given run as ACTIVE."""
    data = _ledger.load()
    assert data is not None, "No ledger found — did pseti start succeed?"
    assert data.run_name == run_name, (
        f"Ledger run_name {data.run_name!r} != expected {run_name!r}"
    )
    assert str(data.status) == "ACTIVE", (
        f"Ledger status is {data.status!r}, expected 'ACTIVE'"
    )


def reaches(target: str, timeout: float = 120.0) -> str:
    """Poll until the ledger reaches *target* status. Return the status."""
    return _ledger.wait_for_status(target, timeout=timeout)

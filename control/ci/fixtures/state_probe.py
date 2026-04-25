"""
ci/fixtures/state_probe.py

State inspection helpers for PANOSETI tests.
StateProbe wraps filesystem, gRPC, and telemetry sources into a clean assertion API.
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any

from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager

class StateProbe:
    """
    Consolidates state queries against the CI environment (isolated or real).
    """

    def __init__(
        self,
        daq_control_client: Any | None = None,
        redis_client: Any | None = None,
        loki_url: str | None = None,
    ) -> None:
        self._daq = daq_control_client
        self._redis = redis_client
        self._loki_url = loki_url

    # ── Run Lifecycle & Ledger ───────────────────────────────────────────────

    def ledger_status(self) -> str | None:
        """Return the current status in the ledger.toml."""
        mgr = RunStateManager()
        ledger = mgr.load_state()
        return ledger.status if ledger else None

    def current_run_name(self) -> str | None:
        """Return the current run name from ledger or 'current' sentinel file."""
        mgr = RunStateManager()
        ledger = mgr.load_state()
        if ledger:
            return ledger.run_name
        
        # Fallback to sentinel file if it exists in state/runs/current
        sentinel = PanoPaths.runs_dir() / "current"
        if sentinel.exists():
            return sentinel.read_text().strip() or None
        return None

    def aborted_snapshot_exists(self, run_name: str) -> bool:
        """True if a snapshot exists in state/snapshots/{run_name}."""
        p = PanoPaths.snapshots_dir(run_name)
        return p.exists() and any(p.iterdir())

    # ── Hashpipe / DAQ node ──────────────────────────────────────────────────

    async def is_hashpipe_running(self, host: str) -> bool:
        """Query a DAQ node via gRPC to see if hashpipe is running."""
        if not self._daq:
            return False
        # Note: In Tier 2, this might be a MockDaqNode.client
        try:
            # We assume DAQ_DATA_DIR /data is standard for CI
            ok, status = await self._daq.StatusDaq({
                "data_dir": "/data",
                "check_hashpipe_running": True
            })
            return bool(ok and status.get("hashpipe_running"))
        except Exception:
            return False

    # ── Filesystem ───────────────────────────────────────────────────────────

    def any_pff_files(self, run_name: str, head: bool = True) -> bool:
        """Check for .pff files in head data dir or (if head=False) DAQ data dir."""
        # This helper assumes a standard layout /data or /data/head
        # In Tier 2, these are local paths.
        root = pathlib.Path("/data/head" if head else "/data")
        if not root.exists():
            return False
            
        return any(root.rglob(f"*{run_name}*/*.pff"))

    # ── Telemetry ────────────────────────────────────────────────────────────

    def redis_key_exists(self, key: str) -> bool:
        """Check if a specific Redis key exists (honors REDIS_DB isolation)."""
        if not self._redis:
            return False
        return bool(self._redis.exists(key))

    def wait_until_ledger_status(self, expected: str, timeout: float = 10.0) -> bool:
        """Poll ledger until it reaches expected status."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ledger_status() == expected:
                return True
            time.sleep(0.2)
        return False

"""
ci/fixtures/state_probe.py

State inspection helpers for PANOSETI tests.
StateProbe wraps filesystem, gRPC, Redis, and Loki sources into a clean assertion API.
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any

from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager

# Shared data dirs matching the docker-compose volume mounts
HEAD_DATA_DIR = pathlib.Path(os.getenv("HEAD_DATA_DIR", "/data/head"))
DAQ_DATA_DIR  = pathlib.Path(os.getenv("DAQ_DATA_DIR",  "/data"))

class StateProbe:
    """
    Consolidates state queries against the CI environment (isolated or real).
    Supports legacy methods for chaos tests and new logic for tiered testing.
    """

    def __init__(
        self,
        daq_control_client: Any | None = None,
        redis_client: Any | None = None,
        loki_url: str | None = None,
        docker_client: Any | None = None,
    ) -> None:
        self._daq = daq_control_client
        self._redis = redis_client
        self._loki_url = loki_url
        self._docker = docker_client

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

    def head_run_dir(self, run_name: str) -> pathlib.Path:
        """Return the run directory on the head node."""
        # Check PSETI_CONTROL/data first for isolated tests
        ctl_data = PanoPaths.base_dir() / "data" / run_name
        if ctl_data.exists():
             return ctl_data
        return HEAD_DATA_DIR / run_name

    def aborted_snapshot_root(self) -> pathlib.Path:
        """Return the snapshots/ directory for post-mortem snapshots."""
        return PanoPaths.snapshots_dir("")

    def aborted_snapshot_exists(self, run_name: str) -> bool:
        """True if a snapshot exists in state/snapshots/{run_name}."""
        p = PanoPaths.snapshots_dir(run_name)
        return p.exists() and any(p.iterdir())

    # ── Hashpipe / DAQ node ──────────────────────────────────────────────────

    def hashpipe_pid(self, container_name: str | None = None) -> int | None:
        """Return hashpipe PID from gRPC StatusDaq, or None if not running."""
        if self._daq is None:
            return None
        try:
            ok, status = self._daq.StatusDaq({
                "data_dir": str(DAQ_DATA_DIR),
                "check_hashpipe_running": True,
            })
            if ok and status.get("hashpipe_running"):
                return int(status.get("hashpipe_pid", 0)) or None
        except Exception:
            pass
        return None

    async def is_hashpipe_running(self, host: str | None = None) -> bool:
        """Query a DAQ node via gRPC to see if hashpipe is running."""
        if not self._daq:
            return False
        try:
            ok, status = await self._daq.StatusDaq({
                "data_dir": str(DAQ_DATA_DIR),
                "check_hashpipe_running": True
            })
            return bool(ok and status.get("hashpipe_running"))
        except Exception:
            return False

    def hashpipe_running(self, container_name: str | None = None) -> bool:
        """Sync wrapper for hashpipe check."""
        if self._daq is None:
            return self.ledger_status() == "ACTIVE"
        try:
            ok, status = self._daq.StatusDaq({
                "data_dir": str(DAQ_DATA_DIR),
                "check_hashpipe_running": True,
            })
            return bool(ok and status.get("hashpipe_running"))
        except Exception:
            return False

    def hashpipe_process_alive(self, container_name: str) -> bool:
        """Check whether the hashpipe OS process is actually alive in the container."""
        try:
            # We try to use our chaos utility if available
            from ci.fixtures.chaos import process_chaos
            return process_chaos.process_alive(container_name, "hashpipe")
        except Exception:
            # Fallback to ledger state if in a mock environment
            return self.ledger_status() == "ACTIVE"

    # ── Filesystem ───────────────────────────────────────────────────────────

    def pff_files(self, module_id: int, run_dir: str | None = None) -> list[pathlib.Path]:
        """Return all .pff files for the given module, optionally under run_dir."""
        base = DAQ_DATA_DIR / f"module_{module_id}"
        if run_dir:
            base = base / run_dir
        if not base.exists():
            return []
        return list(base.rglob("*.pff"))

    def any_pff_files(self, run_name: str, head: bool = True, module_ids: list[int] | None = None) -> bool:
        """Check for .pff files in head data dir or (if head=False) DAQ data dir."""
        if head:
            root = self.head_run_dir(run_name)
            return any(root.rglob("*.pff"))
        else:
            if module_ids:
                return any(self.pff_files(mid, run_name) for mid in module_ids)
            # Scan all module dirs
            for module_dir in DAQ_DATA_DIR.glob("module_*"):
                if list((module_dir / run_name).rglob("*.pff")):
                    return True
        return False

    # ── Telemetry ────────────────────────────────────────────────────────────

    def redis_keys(self, prefix: str) -> list[str]:
        """Return Redis keys matching prefix* (SCAN-based, safe for large keyspaces)."""
        if self._redis is None:
            return []
        try:
            return [k.decode() if isinstance(k, bytes) else k
                    for k in self._redis.scan_iter(f"{prefix}*")]
        except Exception:
            return []

    def redis_key_exists(self, key: str) -> bool:
        """Check if a specific Redis key exists (honors REDIS_DB isolation)."""
        if not self._redis:
            return False
        try:
            return bool(self._redis.exists(key))
        except Exception:
            return False
            
    def redis_incident_key(self, key: str) -> bool:
        """Legacy alias for redis_key_exists."""
        return self.redis_key_exists(key)

    def loki_logs(
        self,
        selector: str = '{job="panoseti"}',
        limit: int = 100,
        since_s: float = 60.0,
    ) -> list[dict[str, Any]]:
        """Query Loki for recent log entries."""
        if self._loki_url is None:
            return []
        try:
            import typing

            import requests
            start_ns = int((time.time() - since_s) * 1e9)
            resp = requests.get(
                f"{self._loki_url}/loki/api/v1/query_range",
                params=typing.cast(Any, {
                    "query": selector,
                    "start": start_ns,
                    "limit": limit,
                }),
                timeout=5,
            )
            resp.raise_for_status()
            results = resp.json().get("data", {}).get("result", [])
            entries = []
            for stream in results:
                for ts, line in stream.get("values", []):
                    entries.append({"ts": ts, "line": line, "labels": stream.get("stream", {})})
            return entries
        except Exception:
            return []

    # ── Background process daemons ───────────────────────────────────────────

    # def interleave_pid_file_exists(self) -> bool:
    #     """True if state/runs/interleave.lock exists."""
    #     return (PanoPaths.runs_dir() / "interleave.lock").exists()

    # def interleave_pid(self) -> int | None:
    #     """Return the PID in state/runs/interleave.lock, or None if absent/invalid."""
    #     p = PanoPaths.runs_dir() / "interleave.lock"
    #     try:
    #         return int(p.read_text().strip())
    #     except (FileNotFoundError, ValueError):
    #         return None

    def hk_recorder_running(self) -> bool:
        """True if capture_hk.py is running (checked via log presence and pidfile)."""
        pid_file = PanoPaths.base_dir() / "tmp" / "hk_recorder.pid"
        if pid_file.exists():
            return True
        return any(PanoPaths.logs_dir().glob("capture_hk*.log"))

    def hv_updater_running(self) -> bool:
        """True if hv_updater.py is running."""
        pid_file = PanoPaths.base_dir() / "tmp" / "hv_updater.pid"
        return pid_file.exists()

    # ── Convenience wait helpers ─────────────────────────────────────────────

    def wait_run_name(self, expected: str | None, timeout: float = 10.0) -> bool:
        """Poll until current_run_name() == expected or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.current_run_name() == expected:
                return True
            time.sleep(0.2)
        return False

    def wait_until_ledger_status(self, expected: str, timeout: float = 10.0) -> bool:
        """Poll ledger until it reaches expected status."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ledger_status() == expected:
                return True
            time.sleep(0.2)
        return False

    def wait_hashpipe_dead(self, container_name: str, timeout: float = 10.0) -> bool:
        """Poll until hashpipe process is gone."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.hashpipe_process_alive(container_name):
                return True
            time.sleep(0.2)
        return False

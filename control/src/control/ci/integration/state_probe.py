"""
integration/state_probe.py

State inspection helpers for PANOSETI chaos tests.

StateProbe wraps the various state sources (filesystem, gRPC, Redis, Loki, Docker)
into a single object so tests can make compact, readable assertions:

    assert probe.current_run() is None
    assert probe.hashpipe_process_alive("daqnode")
    assert not probe.interleave_pid_file_exists()
"""

from __future__ import annotations

# HEAD_DATA_DIR matches the docker-compose volume mount on the test-runner
import os
import pathlib
import time
from typing import Any

HEAD_DATA_DIR = pathlib.Path(os.getenv("HEAD_DATA_DIR", "/data/head"))
DAQ_DATA_DIR  = pathlib.Path(os.getenv("DAQ_DATA_DIR",  "/data"))
INTERLEAVE_PID_FILE = pathlib.Path("tmp/interleave.pid")  # relative to control/


class StateProbe:
    """
    Consolidates state queries against the CI environment.

    Pass None for optional clients to skip those probe sources.
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

    # ── Run lifecycle ────────────────────────────────────────────────────────

    def current_run(self) -> str | None:
        """Read current_run file from head node data dir. Returns None if absent."""
        p = HEAD_DATA_DIR / "current_run"
        try:
            return p.read_text().strip() or None
        except FileNotFoundError:
            return None

    def aborted_snapshot_root(self) -> pathlib.Path:
        """Return the _aborted/ sibling directory for post-mortem snapshots."""
        return HEAD_DATA_DIR / "_aborted"

    def head_run_dir(self, run_name: str) -> pathlib.Path:
        return HEAD_DATA_DIR / run_name

    # ── Hashpipe / DAQ node ──────────────────────────────────────────────────

    def hashpipe_pid(self, container_name: str | None = None) -> int | None:
        """Return hashpipe PID from gRPC StatusDaq, or None if not running."""
        if self._daq is None:
            return None
        try:
            ok, status = self._daq.StatusDaq({
                "data_dir": str(DAQ_DATA_DIR),
                "check_hashpipe_running": True,
                "check_disk_usage": False,
                "check_run_dirs": False,
            })
            if ok and status.get("hashpipe_running"):
                return int(status.get("hashpipe_pid", 0)) or None
        except Exception:
            pass
        return None

    def hashpipe_running(self, container_name: str | None = None) -> bool:
        """Return True if gRPC StatusDaq reports hashpipe_running=True."""
        if self._daq is None:
            return False
        try:
            ok, status = self._daq.StatusDaq({
                "data_dir": str(DAQ_DATA_DIR),
                "check_hashpipe_running": True,
                "check_disk_usage": False,
                "check_run_dirs": False,
            })
            return bool(ok and status.get("hashpipe_running"))
        except Exception:
            return False

    def hashpipe_process_alive(self, container_name: str) -> bool:
        """Check whether the hashpipe OS process is actually alive in the container."""
        try:
            from control.ci.integration.chaos import process_chaos
            return process_chaos.process_alive(container_name, "hashpipe")
        except Exception:
            return False

    # ── PFF files ────────────────────────────────────────────────────────────

    def pff_files(self, module_id: int, run_dir: str | None = None) -> list[pathlib.Path]:
        """Return all .pff files for the given module, optionally under run_dir."""
        base = DAQ_DATA_DIR / f"module_{module_id}"
        if run_dir:
            base = base / run_dir
        if not base.exists():
            return []
        return list(base.rglob("*.pff"))

    def any_pff_files(self, run_dir: str, module_ids: list[int] | None = None) -> bool:
        """True if any .pff files exist for the run (across all module IDs)."""
        if module_ids:
            return any(self.pff_files(mid, run_dir) for mid in module_ids)
        # Scan all module dirs
        for module_dir in DAQ_DATA_DIR.glob("module_*"):
            if list((module_dir / run_dir).rglob("*.pff")):
                return True
        return False

    # ── Redis ────────────────────────────────────────────────────────────────

    def redis_keys(self, prefix: str) -> list[str]:
        """Return Redis keys matching prefix* (SCAN-based, safe for large keyspaces)."""
        if self._redis is None:
            return []
        try:
            return [k.decode() if isinstance(k, bytes) else k
                    for k in self._redis.scan_iter(f"{prefix}*")]
        except Exception:
            return []

    def redis_incident_key(self, key: str) -> bool:
        """True if the given Redis key exists."""
        if self._redis is None:
            return False
        try:
            return bool(self._redis.exists(key))
        except Exception:
            return False

    # ── Loki ─────────────────────────────────────────────────────────────────

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

    # ── Interleave daemon ────────────────────────────────────────────────────

    def interleave_pid_file_exists(self) -> bool:
        """True if tmp/interleave.pid exists (relative to cwd = control/)."""
        return INTERLEAVE_PID_FILE.exists()

    def interleave_pid(self) -> int | None:
        """Return the PID in tmp/interleave.pid, or None if absent/invalid."""
        try:
            return int(INTERLEAVE_PID_FILE.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None

    # ── Background process daemons ───────────────────────────────────────────

    def hk_recorder_running(self) -> bool:
        """True if capture_hk.py is running (checked via pidfile heuristic)."""
        return pathlib.Path("tmp/hk_recorder.pid").exists()

    def hv_updater_running(self) -> bool:
        """True if hv_updater.py is running."""
        return pathlib.Path("tmp/hv_updater.pid").exists()

    # ── Convenience wait helpers ─────────────────────────────────────────────

    def wait_run_name(self, expected: str | None, timeout: float = 10.0) -> bool:
        """Poll until current_run() == expected or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.current_run() == expected:
                return True
            time.sleep(0.2)
        return False

    def wait_hashpipe_dead(self, container_name: str, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.hashpipe_process_alive(container_name):
                return True
            time.sleep(0.2)
        return False

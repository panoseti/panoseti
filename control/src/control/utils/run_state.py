from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
from pathlib import Path
import tempfile
import time
import tomllib
from typing import Any
import logging

from panoseti_grpc.telemetry.logger import get_logger

from filelock import SoftFileLock, Timeout

from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import NodeReceipt, RunStateLedger, RunStatus

LOCK_FILE = "panoseti_control.lock"
STATE_FILE = "ledger.toml"
STATE_FILE_STALE = f"stale_{STATE_FILE}"

logger = logging.getLogger("PSETI.Ledger")


class ValidationError(Exception):
    """Raised when configuration or reachability validation fails."""
    pass


class LockError(Exception):
    """Raised when the advisory lock cannot be acquired."""
    pass


class RunStateManager:
    """Manages transactional file-based state for PANOSETI observatory runs.

    Coordinates the start/stop state machine via a central lock file and a TOML ledger.
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        """
        Args:
            base_dir: Optional override for the root state directory.
                     Defaults to PanoPaths.state_dir().
        """
        if base_dir:
            self.base_dir = Path(base_dir)
            self.lock_path = self.base_dir / "locks" / LOCK_FILE
            self.state_path = self.base_dir / "runs" / STATE_FILE
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.base_dir = PanoPaths.state_dir()
            self.lock_path = PanoPaths.locks_dir() / LOCK_FILE
            self.state_path = PanoPaths.runs_dir() / STATE_FILE
            PanoPaths.ensure_state_dirs()

        self._filelock = SoftFileLock(str(self.lock_path), timeout=5)
        self._lock_held = False
        self._async_lock = asyncio.Lock()

    def acquire_lock(self) -> bool:
        """Acquire the exclusive global observatory advisory lock.

        Creates an atomic lock file at `state/locks/panoseti_control.lock`.
        If the file exists but the holding PID is dead (e.g., from an abrupt crash),
        the lock is considered stale and automatically overwritten.

        Returns:
            True if the lock was acquired successfully.

        Raises:
            LockError: If the lock is held by a live control process.
        """
        if self._lock_held:
            return True

        if self._filelock.is_locked:
            logger.info(f"Waiting up to 5 seconds for existing lock on {self.lock_path}...")

        try:
            self._filelock.acquire()
            self._lock_held = True
            return True
        except Timeout:
            holder_pid = self._filelock.pid
            raise LockError(
                f"Another PANOSETI control process (PID {holder_pid}) is already running. "
                f"Check for concurrent start.py or stop.py executions. "
                f"Lock file: {self.lock_path}"
            ) from None

    def release_lock(self) -> None:
        """Release the exclusive global observatory advisory lock."""
        if self._lock_held:
            with contextlib.suppress(OSError):
                self._filelock.release()
            self._lock_held = False

    def load_state(self) -> RunStateLedger | None:
        """Loads the current run state from the TOML ledger."""
        if not self.state_path.exists():
            return None
        try:
            with open(self.state_path, "rb") as f:
                data = tomllib.load(f)
                return RunStateLedger(**data)
        except Exception as e:
            # If state is corrupt, we might need to handle it or return None
            print(f"Warning: Failed to load ledger.toml: {e}")
            return None

    def _escape_toml_str(self, s: str) -> str:
        """Escapes a string for TOML using JSON-style escaping (compatible with TOML)."""
        import json
        return json.dumps(s)

    def save_state(self, state: RunStateLedger) -> None:
        """
        Saves the run state to the TOML ledger atomically.
        Uses manual TOML formatting to avoid external dependencies.
        """
        d = state.model_dump(mode='json')

        lines = [
            f'run_name = {self._escape_toml_str(d["run_name"])}',
            f'status = "{d["status"]}"',
            f'start_time = "{d["start_time"]}"',
        ]
        if d.get("pid") is not None:
            lines.append(f'pid = {d["pid"]}')
        if d.get("host") is not None:
            lines.append(f'host = {self._escape_toml_str(d["host"])}')
        lines.append(f'transfer_attempts = {d.get("transfer_attempts", 0)}')
        if d.get("last_transfer_error") is not None:
            lines.append(f'last_transfer_error = {self._escape_toml_str(d["last_transfer_error"])}')
        if d.get("manifest_algorithm") is not None:
            lines.append(f'manifest_algorithm = {self._escape_toml_str(d["manifest_algorithm"])}')
        if d.get("next_action_not_before") is not None:
            lines.append(f'next_action_not_before = "{d["next_action_not_before"]}"')

        lines.append("")
        lines.append("[config_metadata]")
        for k, v in d.get("config_metadata", {}).items():
            if isinstance(v, str):
                lines.append(f'{k} = {self._escape_toml_str(v)}')
            elif isinstance(v, bool):
                lines.append(f'{k} = {str(v).lower()}')
            else:
                lines.append(f'{k} = {v}')

        lines.append("")
        for node in d.get("nodes", []):
            lines.append("[[nodes]]")
            lines.append(f'ip_addr = "{node["ip_addr"]}"')
            lines.append(f'status = "{node["status"]}"')
            if node.get("hashpipe_pid") is not None:
                lines.append(f'hashpipe_pid = {node["hashpipe_pid"]}')
            if node.get("data_dir") is not None:
                lines.append(f'data_dir = {self._escape_toml_str(node["data_dir"])}')
            if node.get("message") is not None:
                lines.append(f'message = {self._escape_toml_str(node["message"])}')
            if node.get("manifest_path") is not None:
                lines.append(f'manifest_path = {self._escape_toml_str(node["manifest_path"])}')
            if node.get("manifest_bytes") is not None:
                lines.append(f'manifest_bytes = {node["manifest_bytes"]}')
            if node.get("rsync_bytes_transferred") is not None:
                lines.append(f'rsync_bytes_transferred = {node["rsync_bytes_transferred"]}')
            if node.get("rsync_last_progress_at") is not None:
                lines.append(f'rsync_last_progress_at = "{node["rsync_last_progress_at"]}"')
            if node.get("verify_ok") is not None:
                lines.append(f'verify_ok = {str(node["verify_ok"]).lower()}')
            if node.get("cleanup_ok") is not None:
                lines.append(f'cleanup_ok = {str(node["cleanup_ok"]).lower()}')
            lines.append("")

        # Atomic write using NamedTemporaryFile and os.replace
        tmp_dir = self.state_path.parent
        with tempfile.NamedTemporaryFile("w", dir=tmp_dir, delete=False) as f:
            f.write("\n".join(lines))
            temp_name = f.name
        
        try:
            os.replace(temp_name, self.state_path)
        except Exception:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
            raise

    def clear_state(self) -> None:
        """Clear the current run state and release the lock. Intended for testing."""
        if self.state_path.exists():
            self.state_path.unlink()
        with contextlib.suppress(OSError):
            self._filelock.release(force=True)
        self._lock_held = False
        if self.lock_path.exists():
            self.lock_path.unlink()

    def transition(self, status: RunStatus, **fields: Any) -> RunStateLedger | None:
        """Load current state, update status and any extra fields, save, return new state.

        If 'node_ip' is provided in fields, the remaining fields are applied to the 
        NodeReceipt with that IP address instead of the root ledger.
        """
        state = self.load_state()
        if state is None:
            return None
        
        logger.info(f"Transaction Phase: [{status.value}]")
        state.status = status

        node_ip = fields.pop("node_ip", None)
        if node_ip:
            from ipaddress import IPv4Address, IPv6Address
            target_ip = node_ip
            if isinstance(node_ip, str):
                 try:
                     target_ip = IPv4Address(node_ip)
                 except ValueError:
                     with contextlib.suppress(ValueError):
                         target_ip = IPv6Address(node_ip)
            node = next((n for n in state.nodes if n.ip_addr == target_ip), None)
            if node:
                for key, value in fields.items():
                    setattr(node, key, value)
                fields = {}
            else:
                logger.warning(f"Transition: node {node_ip} not found in ledger")

        for key, value in fields.items():
            setattr(state, key, value)
        self.save_state(state)
        return state

    async def update_node_receipt(self, receipt: NodeReceipt) -> None:
        """
        Updates or appends a node receipt in the current state.
        Concurrency-safe via asyncio.Lock.
        """
        async with self._async_lock:
            state = await asyncio.to_thread(self.load_state)
            if not state:
                return

            logger.info(f"Node {receipt.ip_addr} Phase: [{receipt.status.value}]")
            # Find existing node or append
            for i, node in enumerate(state.nodes):
                if str(node.ip_addr) == str(receipt.ip_addr):
                    state.nodes[i] = receipt
                    break
            else:
                state.nodes.append(receipt)

            await asyncio.to_thread(self.save_state, state)


def get_current_run_name() -> str | None:
    """Legacy compatibility helper to read the run name from the new ledger."""
    mgr = RunStateManager()
    state = mgr.load_state()
    if state and state.status in ["STARTING", "ACTIVE", "STOPPING"]:
        return state.run_name
    return None

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import tempfile
import tomllib
from typing import Any

from panoseti_grpc.telemetry.logger import get_logger

from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import NodeReceipt, RunStateLedger, RunStatus

LOCK_FILE = "panoseti_control.lock"
STATE_FILE = "run_state.toml"

logger = get_logger("PSETI.RunState")


class ValidationError(Exception):
    """Raised when configuration or reachability validation fails."""
    pass


class LockError(Exception):
    """Raised when the advisory lock cannot be acquired."""
    pass


class RunStateManager:
    """
    Manages the PANOSETI control plane state ledger and advisory locking.
    Ensures transactional integrity across start/stop operations.
    """

    def __init__(self, base_dir: str | pathlib.Path | None = None) -> None:
        if base_dir:
            self.base_dir = pathlib.Path(base_dir)
            # Legacy/override support: if provided base_dir looks like a state root
            self.lock_path = self.base_dir / "locks" / LOCK_FILE
            self.state_path = self.base_dir / "runs" / STATE_FILE
            # Ensure they exist if possible, but PanoPaths handles defaults better
        else:
            self.base_dir = PanoPaths.state_dir()
            self.lock_path = PanoPaths.locks_dir() / LOCK_FILE
            self.state_path = PanoPaths.runs_dir() / STATE_FILE

        self._lock_fh: Any | None = None
        self._async_lock = asyncio.Lock()

        # Ensure directory exists
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def acquire_lock(self) -> bool:
        """Acquire the exclusive advisory control-plane lock.

        Uses atomic ``O_EXCL`` file creation with stale-PID healing
        (SC-015/SC-021).

        Returns:
            ``True`` when the lock is acquired successfully.

        Raises:
            LockError: If another live process already holds the lock.
        """
        if self._lock_fh:
            return True

        # Ensure tmp/ exists
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(2):
            try:
                # O_EXCL ensures that this call atomically creates the file; if it exists, it fails.
                # https://man7.org/linux/man-pages/man2/open.2.html
                fd = os.open(str(self.lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                with os.fdopen(fd, "w") as f:
                    f.write(str(os.getpid()))
                self._lock_fh = True
                # print(f"DEBUG: Lock acquired by PID {os.getpid()} at {self.lock_path}")
                return True
            except FileExistsError:
                # Check if the lock is stale
                try:
                    with open(self.lock_path) as f:
                        pid_str = f.read().strip()
                        pid = int(pid_str)
                    
                    # Check if process is alive
                    os.kill(pid, 0)
                    # print(f"DEBUG: Lock held by LIVE PID {pid} at {self.lock_path}")
                except (OSError, ValueError, ProcessLookupError):
                    # Process is dead or file is corrupt. Self-heal.
                    # print(f"DEBUG: Lock stale (PID {pid_str}) at {self.lock_path}. Unlinking.")
                    try:
                        self.lock_path.unlink()
                        if attempt == 0:
                            continue # Try creating again
                    except OSError:
                        pass
                
                raise LockError(
                    f"Another PANOSETI control process (PID {pid}) is already running. "
                    f"Check for concurrent start.py or stop.py executions. Lock file: {self.lock_path}"
                ) from None
            except OSError as e:
                raise LockError(f"Failed to create lock file: {e}") from None
        
        return False

    def release_lock(self) -> None:
        """Releases the advisory lock by removing the file."""
        if self._lock_fh:
            try:
                if self.lock_path.exists():
                    self.lock_path.unlink()
            except OSError:
                pass
            self._lock_fh = None

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
            print(f"Warning: Failed to load run_state.toml: {e}")
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
        """Clears the run state ledger and advisory lock."""
        if self.state_path.exists():
            self.state_path.unlink()
        if self.lock_path.exists():
            with contextlib.suppress(OSError):
                self.lock_path.unlink()
        self._lock_fh = None

    def transition(self, status: RunStatus, **fields: Any) -> RunStateLedger | None:
        """Load current state, update status and any extra fields, save, return new state.

        Returns None if no state exists.
        """
        state = self.load_state()
        if state is None:
            return None
        
        logger.info(f"Transaction Phase: [{status.value}]")
        state.status = status
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

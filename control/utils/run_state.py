from __future__ import annotations

import asyncio
import fcntl
import os
import pathlib
import tempfile
import tomllib
from typing import Any

from .pydantic_config_models import NodeReceipt, RunStateLedger

LOCK_FILE = "tmp/panoseti_control.lock"
STATE_FILE = "tmp/run_state.toml"


class RunStateManager:
    """
    Manages the PANOSETI control plane state ledger and advisory locking.
    Ensures transactional integrity across start/stop operations.
    """

    def __init__(self, base_dir: str = ".") -> None:
        self.base_dir = pathlib.Path(base_dir)
        self.lock_path = self.base_dir / LOCK_FILE
        self.state_path = self.base_dir / STATE_FILE
        self._lock_fh: Any | None = None
        self._async_lock = asyncio.Lock()

        # Ensure tmp/ exists
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    def acquire_lock(self) -> None:
        """Acquires an exclusive advisory lock on the control plane."""
        # Note: self._lock_fh must stay open for the duration of the lock.
        # SIM115: ignore because the handle must persist beyond this method.
        self._lock_fh = open(self.lock_path, "w")  # noqa: SIM115
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_fh.close()
            raise RuntimeError(
                "Another PANOSETI control process is already running. "
                "Check for concurrent start.py or stop.py executions."
            ) from None

    def release_lock(self) -> None:
        """Releases the advisory lock."""
        if self._lock_fh:
            fcntl.flock(self._lock_fh, fcntl.LOCK_UN)
            self._lock_fh.close()
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
        """Escapes a string for TOML."""
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'

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
        """Clears the run state ledger."""
        if self.state_path.exists():
            self.state_path.unlink()

    async def update_node_receipt(self, receipt: NodeReceipt) -> None:
        """
        Updates or appends a node receipt in the current state.
        Concurrency-safe via asyncio.Lock.
        """
        async with self._async_lock:
            state = self.load_state()
            if not state:
                return

            # Find existing node or append
            for i, node in enumerate(state.nodes):
                if str(node.ip_addr) == str(receipt.ip_addr):
                    state.nodes[i] = receipt
                    break
            else:
                state.nodes.append(receipt)

            self.save_state(state)


def get_current_run_name() -> str | None:
    """Legacy compatibility helper to read the run name from the new ledger."""
    mgr = RunStateManager()
    state = mgr.load_state()
    if state and state.status in ["STARTING", "ACTIVE", "STOPPING"]:
        return state.run_name
    return None

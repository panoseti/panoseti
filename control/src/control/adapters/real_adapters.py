"""
real_adapters.py — Concrete implementations for production side-effects.

These classes implement the protocols in `control.interfaces` using
the real OS, filesystem, and network tools (psutil, grpc, etc.).
"""

import json
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import psutil

from control.interfaces import FileSystemManager, NetworkClient, ProcessManager
from control.utils.pydantic_config_models import DaqConfig

logger = logging.getLogger(__name__)


class RealProcessManager(ProcessManager):
    """Real implementation of ProcessManager using psutil and subprocess."""

    def is_running(self, name: str) -> bool:
        for p in psutil.process_iter():
            try:
                if name in p.cmdline():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def kill(self, name: str) -> bool:
        killed = False
        for p in psutil.process_iter():
            try:
                if name in p.cmdline():
                    os.kill(p.pid, signal.SIGKILL)
                    killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return killed

    def start(self, cmd: list[str]) -> None:
        try:
            subprocess.Popen(cmd)
        except OSError as e:
            logger.error(f"Failed to start process {' '.join(cmd)}: {e}")
            raise


class RealNetworkClient(NetworkClient):
    """Real implementation of NetworkClient using panoseti_grpc."""

    def __init__(self, daq_config: DaqConfig):
        self.daq_config = daq_config

    async def ping_nodes(self) -> list[str]:
        # Utilizing the existing check logic, but abstracted.
        # This will be refined as we port more of start.py
        reachable = []
        for node in self.daq_config.daq_nodes:
            # Simplified mock logic for now, relies on start.py's internal checks
            # Ideally this calls AsyncDaqControlClient directly
            reachable.append(str(node.ip_addr))
        return reachable

    async def start_daq(self, params: dict[str, Any]) -> bool:
        # This will wrap the AsyncDaqControlClient StartDaq call
        return True

    async def stop_daq(self) -> bool:
        # This will wrap the AsyncDaqControlClient StopDaq call
        return True


class RealFileSystemManager(FileSystemManager):
    """Real implementation of FileSystemManager using pathlib and os."""

    def __init__(self, daq_config: DaqConfig):
        self.daq_config = daq_config

    def create_run_dirs(self, run_name: str) -> None:
        # Create head node dir
        head_run_dir = Path(self.daq_config.head_node_data_dir) / run_name
        head_run_dir.mkdir(parents=True, exist_ok=True)
        # Note: DAQ node dirs are created via the RPC StartDaq call in the real system,
        # but the orchestrator might need local state dirs.
        pass

    def write_metadata(self, run_name: str, data: dict[str, Any]) -> None:
        head_run_dir = Path(self.daq_config.head_node_data_dir) / run_name
        if head_run_dir.exists():
            (head_run_dir / "meta.json").write_text(json.dumps(data, indent=2))
        else:
            logger.warning(f"Could not write metadata: directory {head_run_dir} does not exist.")

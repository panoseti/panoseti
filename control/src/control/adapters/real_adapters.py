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
from control.utils.pydantic_config_models import DaqConfig, DaqNode

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

    async def ping_node(self, node: DaqNode) -> bool:
        from panoseti_grpc.daq_control.client import AsyncDaqControlClient

        from control.utils import util
        grpc_host, grpc_port = util.daq_grpc_endpoint(node, self.daq_config)
        try:
            async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                # We reuse StatusDaq as a ping
                await client.StatusDaq({"data_dir": node.data_dir}, timeout=2.0)
                return True
        except Exception:
            return False

    async def start_daq_node(self, node: DaqNode, params: dict[str, Any], timeout_s: float = 10.0) -> bool:
        from panoseti_grpc.daq_control.client import AsyncDaqControlClient

        from control.utils import util
        grpc_host, grpc_port = util.daq_grpc_endpoint(node, self.daq_config)
        async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
            return await client.StartDaq(params, timeout=timeout_s)

    async def stop_daq_node(self, node: DaqNode, timeout_s: float = 15.0) -> bool:
        from panoseti_grpc.daq_control.client import AsyncDaqControlClient

        from control.utils import util
        grpc_host, grpc_port = util.daq_grpc_endpoint(node, self.daq_config)
        async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
            return await client.StopDaq({'data_dir': node.data_dir}, timeout=timeout_s)

    async def get_daq_status(self, node: DaqNode, timeout_s: float = 5.0) -> dict[str, Any]:
        from panoseti_grpc.daq_control.client import AsyncDaqControlClient

        from control.utils import util
        grpc_host, grpc_port = util.daq_grpc_endpoint(node, self.daq_config)
        async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
            _, status_dict = await client.StatusDaq({"data_dir": node.data_dir}, timeout=timeout_s)
            return status_dict


class RealFileSystemManager(FileSystemManager):
    """Real implementation of FileSystemManager using pathlib and os."""

    def __init__(self, daq_config: DaqConfig):
        self.daq_config = daq_config

    def create_run_dirs(
        self,
        run_name: str,
        obs_config: Any = None,
        daq_config: Any = None,
        quabo_uids: Any = None,
        data_config: Any = None,
        network_config: Any = None
    ) -> None:
        from control.start import make_run_dirs
        make_run_dirs(
            run_name,
            obs_config,
            daq_config or self.daq_config,
            quabo_uids,
            data_config,
            network_config
        )

    def write_metadata(self, run_name: str, data: dict[str, Any]) -> None:
        head_run_dir = Path(self.daq_config.head_node_data_dir) / run_name
        if head_run_dir.exists():
            (head_run_dir / "meta.json").write_text(json.dumps(data, indent=2))
        else:
            logger.warning(f"Could not write metadata: directory {head_run_dir} does not exist.")

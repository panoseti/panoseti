"""
interfaces.py — Protocols (Ports) for Dependency Injection.

These interfaces abstract the side-effects of the control software,
allowing the core orchestrators (start, stop) to be tested without
mocking OS, filesystem, or network interactions.
"""

from __future__ import annotations

import typing
from typing import Any


class ProcessManager(typing.Protocol):
    """Abstracts OS-level process checks and signals (e.g. psutil)."""
    
    def is_running(self, name: str) -> bool:
        """Return True if a process matching 'name' is running."""
        ...
        
    def kill(self, name: str) -> bool:
        """Kill the process matching 'name' and return True if successful."""
        ...
        
    def start(self, cmd: list[str]) -> None:
        """Start a background process with the given command."""
        ...


class NetworkClient(typing.Protocol):
    """Abstracts RPC calls to DAQ nodes (e.g. panoseti_grpc)."""
    
    async def ping_node(self, node: Any) -> bool:
        """Return True if the given DaqNode is reachable."""
        ...

    async def start_daq_node(self, node: Any, params: dict[str, Any], timeout: float = 10.0) -> bool:
        """Send the StartDaq command to a specific node."""
        ...
        
    async def stop_daq_node(self, node: Any, timeout: float = 15.0) -> bool:
        """Send the StopDaq command to a specific node."""
        ...
        
    async def get_daq_status(self, node: Any, timeout: float = 5.0) -> Any:
        """Fetch status from a specific DAQ node."""
        ...


class FileSystemManager(typing.Protocol):
    """Abstracts local filesystem operations for the orchestrators."""
    
    def create_run_dirs(
        self,
        run_name: str,
        obs_config: Any = None,
        daq_config: Any = None,
        quabo_uids: Any = None,
        data_config: Any = None,
        network_config: Any = None
    ) -> None:
        """Create the directory structure for a new run and snapshot configs."""
        ...
        
    def write_metadata(self, run_name: str, data: dict[str, Any]) -> None:
        """Write metadata dict to the run directory."""
        ...

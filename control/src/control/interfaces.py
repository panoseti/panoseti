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
    
    async def ping_nodes(self) -> list[str]:
        """Return a list of reachable node hostnames/IPs."""
        ...
        
    async def start_daq(self, params: dict[str, Any]) -> bool:
        """Send the StartDaq command to all nodes."""
        ...
        
    async def stop_daq(self) -> bool:
        """Send the StopDaq command to all nodes."""
        ...


class FileSystemManager(typing.Protocol):
    """Abstracts local filesystem operations for the orchestrators."""
    
    def create_run_dirs(self, run_name: str) -> None:
        """Create the directory structure for a new run."""
        ...
        
    def write_metadata(self, run_name: str, data: dict[str, Any]) -> None:
        """Write metadata dict to the run directory."""
        ...

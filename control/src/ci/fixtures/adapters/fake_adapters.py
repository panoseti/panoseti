"""
fake_adapters.py — Fake implementations for CI/Testing.

These classes implement the protocols in `control.interfaces` using
in-memory state or highly restricted mock environments, allowing tests
to run blazingly fast without hitting the real OS or network.
"""

from typing import Any


class FakeProcessManager:
    """Fake implementation of ProcessManager for testing."""
    
    def __init__(self, running_processes: list[str] | None = None):
        self.running_processes = set(running_processes or [])
        self.started_commands: list[list[str]] = []

    def is_running(self, name: str) -> bool:
        return name in self.running_processes

    def kill(self, name: str) -> bool:
        if name in self.running_processes:
            self.running_processes.remove(name)
            return True
        return False

    def start(self, cmd: list[str]) -> None:
        self.started_commands.append(cmd)
        if cmd:
            # Simplistic mock: assume the first arg is the process name
            self.running_processes.add(cmd[0])


class FakeNetworkClient:
    """Fake implementation of NetworkClient for testing."""
    
    def __init__(self, reachable_nodes: list[str] | None = None):
        self.reachable_nodes = reachable_nodes or []
        self.start_calls = 0
        self.stop_calls = 0
        self.last_params: dict[str, Any] = {}

    async def ping_nodes(self) -> list[str]:
        return self.reachable_nodes

    async def start_daq(self, params: dict[str, Any]) -> bool:
        self.start_calls += 1
        self.last_params = params
        return True

    async def stop_daq(self) -> bool:
        self.stop_calls += 1
        return True


class FakeFileSystemManager:
    """Fake implementation of FileSystemManager for testing."""
    
    def __init__(self):
        self.created_dirs: list[str] = []
        self.metadata_written: dict[str, dict[str, Any]] = {}

    def create_run_dirs(self, run_name: str) -> None:
        self.created_dirs.append(run_name)

    def write_metadata(self, run_name: str, data: dict[str, Any]) -> None:
        self.metadata_written[run_name] = data

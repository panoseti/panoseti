"""
ci/fixtures/mocks.py

High-level software mocks for Tier 2 logic tests.
Enforces the 'Contract' between logic and production gRPC/Hardware behaviors.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from control.utils.pydantic_config_models import DaqConfig


class MockQuabo:
    """
    Simulates a single quadrant board network device.
    Strictly follows the network device contract (IP-based).
    """
    def __init__(self, ip_addr: str):
        self.ip_addr = ip_addr
        self.is_powered = False
        self.firmware_loaded = False
        self.hk_streaming = False

    def power_on(self):
        self.is_powered = True
        
    def power_off(self):
        self.is_powered = False
        self.firmware_loaded = False
        self.hk_streaming = False

class MockModule:
    """
    A logical grouping of 4 Quabos.
    Focuses on module-level orchestration (power, timing, streaming).
    """
    def __init__(self, base_ip: str, module_id: int):
        self.base_ip = base_ip
        self.module_id = module_id
        # In PANOSETI, IPs are consecutive from base
        octets = base_ip.split('.')
        base_last = int(octets[-1])
        prefix = '.'.join(octets[:-1])
        self.quabos = [
            MockQuabo(f"{prefix}.{base_last + i}") for i in range(4)
        ]

    def set_timing_mode(self, mode: str):
        """Mock timing mode (wr or gnss) application."""
        for _q in self.quabos:
            pass # simulator logic

class MockDaqNode:
    """
    Simulates a DAQ node gRPC service.
    Acts as the source of truth for 'Contract Tests' in Tier 2/3.
    """
    def __init__(self, ip_addr: str):
        self.ip_addr = ip_addr
        self.client = MagicMock()
        self.client.__aenter__ = AsyncMock(return_value=self.client)
        self.client.__aexit__ = AsyncMock(return_value=None)
        
        # Default successful responses matching production schemas
        self.client.GenerateManifest = AsyncMock(return_value={"success": True})
        self.client.CleanupData = AsyncMock(return_value={"success": True})
        self.client.StatusDaq = AsyncMock(return_value=(True, {
            "hashpipe_running": False,
            "disk_usage": {"free_disk_space": 10**12},
            "run_dirs": []
        }))
        self.client.StartDaq = AsyncMock(return_value=True)
        self.client.StopDaq = AsyncMock(return_value=True)

    def set_manifest_failure(self, message: str = "Simulated failure"):
        self.client.GenerateManifest.return_value = {"success": False, "message": message}

    def set_cleanup_failure(self, message: str = "Simulated failure"):
        self.client.CleanupData.return_value = {"success": False, "message": message}
        
    def set_hashpipe_running(self, running: bool):
        """Update the mocked StatusDaq response."""
        prev = self.client.StatusDaq.return_value
        self.client.StatusDaq.return_value = (prev[0], {**prev[1], "hashpipe_running": running})

class MockHeadNode:
    """
    Simulates the head node control plane.
    Manages configuration generation and verifies command sequencing.
    """
    def __init__(self, daq_config: DaqConfig):
        self.daq_config = daq_config
        self.command_log: list[tuple[str, Any]] = []

    def log_command(self, cmd: str, params: Any):
        """Record a command issued by the control plane."""
        self.command_log.append((cmd, params))
        
    def get_last_command(self) -> tuple[str, Any] | None:
        return self.command_log[-1] if self.command_log else None

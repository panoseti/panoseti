"""
ci/fixtures/mocks.py

High-level software mocks for Tier 2 logic tests.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Any

from control.utils.pydantic_config_models import DaqConfig

class MockQuabo:
    """Simulates a single quadrant board network device."""
    def __init__(self, ip_addr: str):
        self.ip_addr = ip_addr
        self.is_powered = False
        self.firmware_loaded = False

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

class MockDaqNode:
    """
    Simulates a DAQ node gRPC service.
    Validates that request parameters match Pydantic models.
    """
    def __init__(self, ip_addr: str):
        self.ip_addr = ip_addr
        self.client = MagicMock()
        self.client.__aenter__ = AsyncMock(return_value=self.client)
        self.client.__aexit__ = AsyncMock(return_value=None)
        
        # Default successful responses
        self.client.GenerateManifest = AsyncMock(return_value={"success": True})
        self.client.CleanupData = AsyncMock(return_value={"success": True})
        self.client.StatusDaq = AsyncMock(return_value=(True, {"hashpipe_running": False}))

    def set_manifest_failure(self, message: str = "Simulated failure"):
        self.client.GenerateManifest.return_value = {"success": False, "message": message}

    def set_cleanup_failure(self, message: str = "Simulated failure"):
        self.client.CleanupData.return_value = {"success": False, "message": message}

class MockHeadNode:
    """
    Simulates the head node control plane.
    Manages configuration generation and verifies command sequencing.
    """
    def __init__(self, daq_config: DaqConfig):
        self.daq_config = daq_config
        self.command_log: list[tuple[str, Any]] = []

    def log_command(self, cmd: str, params: Any):
        self.command_log.append((cmd, params))

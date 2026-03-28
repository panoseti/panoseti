"""
test_config_validation.py — Integration tests for config validation.

Runs `python config.py --validate` against the CI configs to ensure
they parse correctly and pass all pydantic/cross-config checks.
Network ping sweep is skipped (hardware not present in software CI).
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest

from .conftest import CONTROL_DIR

INTEGRATION_CONFIGS = CONTROL_DIR / "run-ci-tests" / "integration" / "configs"


def _validate(extra_args: list[str]) -> subprocess.CompletedProcess:
    """Run config.py --validate with the CI config set."""
    base_args = [
        "python", "config.py", "--validate",
        "--obs_config",      str(INTEGRATION_CONFIGS / "obs_config.json"),
        "--data_config",     str(INTEGRATION_CONFIGS / "data_config.json"),
        "--firmware_config", str(INTEGRATION_CONFIGS / "firmware.json"),
        "--daemons_config",  str(INTEGRATION_CONFIGS / "daemons.json"),
    ]
    return subprocess.run(
        base_args + extra_args,
        capture_output=True,
        text=True,
        cwd=str(CONTROL_DIR),
    )


class TestConfigValidation:

    def test_validate_direct_config(self):
        """config.py --validate must exit 0 with direct-connection CI configs."""
        result = _validate([
            "--daq_config",     str(INTEGRATION_CONFIGS / "direct" / "daq_config.json"),
            "--network_config", str(INTEGRATION_CONFIGS / "direct" / "network_config.json"),
        ])
        assert result.returncode == 0, (
            f"config.py --validate failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_validate_gateway_config(self):
        """config.py --validate must exit 0 with gateway port-forwarding config."""
        result = _validate([
            "--daq_config",     str(INTEGRATION_CONFIGS / "gateway" / "daq_config.json"),
            "--network_config", str(INTEGRATION_CONFIGS / "gateway" / "network_config.json"),
        ])
        assert result.returncode == 0, (
            f"config.py --validate failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_gateway_network_config_has_grpc_port(self):
        """Gateway network_config.json must include grpc_port for gRPC forwarding."""
        import json
        nc = json.loads(
            (INTEGRATION_CONFIGS / "gateway" / "network_config.json").read_text()
        )
        daq_nodes = nc.get("daq_nodes", [])
        assert daq_nodes, "Gateway network_config has no daq_nodes"
        pf = daq_nodes[0].get("port_forwarding", {})
        assert pf.get("status") is True
        assert "grpc_port" in pf, "gateway network_config must specify grpc_port"
        assert 1 <= pf["grpc_port"] <= 65535

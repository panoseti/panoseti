"""
test_config_validation.py — Integration tests for config validation.

Runs config_file.validate_all() against the CI configs to ensure they
parse correctly and pass all pydantic/cross-config checks.
Network ping sweep is skipped (hardware not present in software CI).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile

from .conftest import CONTROL_DIR

INTEGRATION_CONFIGS = CONTROL_DIR / "ci-tests" / "integration" / "configs"

# Common config files (same for both direct and gateway topologies)
_COMMON_FILES = ["obs_config.json", "data_config.json", "firmware.json", "daemons.json"]


def _run_validation(variant_dir: pathlib.Path) -> bool:
    """
    Set up a temp workspace with a configs/ directory containing the CI config
    files for the given variant (direct or gateway), then run validate_all().
    Returns True if validation passed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        configs_dir = pathlib.Path(tmpdir) / "configs"
        configs_dir.mkdir()

        # Copy common configs
        for fname in _COMMON_FILES:
            shutil.copy(INTEGRATION_CONFIGS / fname, configs_dir / fname)

        # Copy variant-specific configs
        for fname in ["daq_config.json", "network_config.json"]:
            src = variant_dir / fname
            if src.exists():
                shutil.copy(src, configs_dir / fname)

        # Run validation from the temp workspace
        old_cwd = os.getcwd()
        sys.path.insert(0, str(CONTROL_DIR))
        try:
            os.chdir(tmpdir)
            from utils import config_file
            passed = config_file.validate_all(check_network=False)
        finally:
            os.chdir(old_cwd)
            sys.path.pop(0)

        return passed


class TestConfigValidation:

    def test_validate_direct_config(self):
        """validate_all() must pass with the direct-connection CI configs."""
        passed = _run_validation(INTEGRATION_CONFIGS / "direct")
        assert passed, "Config validation failed for direct topology — check CI config files"

    def test_validate_gateway_config(self):
        """validate_all() must pass with the gateway port-forwarding CI configs."""
        passed = _run_validation(INTEGRATION_CONFIGS / "gateway")
        assert passed, "Config validation failed for gateway topology — check CI config files"

    def test_gateway_network_config_has_grpc_port(self):
        """Gateway network_config.json must include grpc_port for gRPC forwarding."""
        nc = json.loads(
            (INTEGRATION_CONFIGS / "gateway" / "network_config.json").read_text()
        )
        daq_nodes = nc.get("daq_nodes", [])
        assert daq_nodes, "Gateway network_config has no daq_nodes"
        pf = daq_nodes[0].get("port_forwarding", {})
        assert pf.get("status") is True
        assert "grpc_port" in pf, "gateway network_config must specify grpc_port"
        assert 1 <= pf["grpc_port"] <= 65535

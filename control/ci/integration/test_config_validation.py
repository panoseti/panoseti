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
import tempfile

from ci.paths import PanoPathsTest

INTEGRATION_CONFIGS = PanoPathsTest.integration_configs_root()

# Common config files (same for both direct and gateway topologies)
_COMMON_FILES = ["obs_config.json", "data_config.json", "firmware.json", "daemons.json"]


def _run_validation(variant_dir: pathlib.Path) -> bool:
    """
    Set up a temp workspace with a configs/ directory containing the CI config
    files for the given variant (direct or gateway), then run validate_all().
    Returns True if validation passed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()

        # Copy common configs
        for fname in _COMMON_FILES:
            shutil.copy(INTEGRATION_CONFIGS / fname, configs_dir / fname)

        # Copy variant-specific configs
        for fname in ["daq_config.json", "network_config.json"]:
            src = variant_dir / fname
            if src.exists():
                shutil.copy(src, configs_dir / fname)

        # Create stub firmware files referenced in firmware.json to satisfy existence checks
        with open(configs_dir / "firmware.json") as f:
            fw_data = json.load(f)
            # handle both old structure and new flat structure
            for val in fw_data.values():
                if isinstance(val, str) and val.endswith(".bin"):
                    (tmp_path / val).touch()
            if "quabo" in fw_data:
                for val in fw_data["quabo"].values():
                    if isinstance(val, str) and val.endswith(".bin"):
                        (tmp_path / val).touch()

        # Run validation with environment overrides
        old_env = os.environ.copy()
        os.environ["PANOSETI_CONFIG_DIR"] = str(configs_dir)
        # We need to tell it where firmware files are (they are in tmpdir root in this test)
        os.environ["PANOSETI_FIRMWARE_DIR"] = str(tmp_path)
        
        try:
            from control.utils import config_file
            passed = config_file.validate_all(check_network=False)
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        return passed


class TestConfigValidation:

    def test_validate_direct_config(self) -> None:
        """validate_all() must pass with the direct-connection CI configs."""
        passed = _run_validation(INTEGRATION_CONFIGS / "direct")
        assert passed, "Config validation failed for direct topology — check CI config files"

    def test_validate_gateway_config(self) -> None:
        """validate_all() must pass with the gateway port-forwarding CI configs."""
        passed = _run_validation(INTEGRATION_CONFIGS / "gateway")
        assert passed, "Config validation failed for gateway topology — check CI config files"

    def test_gateway_network_config_has_grpc_port(self) -> None:
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

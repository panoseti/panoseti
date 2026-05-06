"""
ci/fixtures/chaos_fixtures.py

Fixtures for simulating chaos and transactional failures in the control plane.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any
from unittest.mock import patch

import pytest

from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import QuaboUids

logger = logging.getLogger(__name__)

@pytest.fixture
def chaos_headnode_workspace(mock_workspace):
    """Context manager fixture that patches configs for headnode-local testing."""
    @contextlib.contextmanager
    def _mock():
        from control.utils import config_file
        
        path = PanoPaths.config_dir() / "daq_config.json"
        backup = str(path) + ".bak"
        
        # Create a dummy PH baseline if missing
        ph_baseline = PanoPaths.calibration_file("quabo_ph_baseline.json")
        if not ph_baseline.exists():
            ph_baseline.write_text(json.dumps({"date": "2024-01-01T00:00:00", "quabos": []}))

        if path.exists():
            import shutil
            shutil.copyfile(path, backup)
        
        cfg = json.loads(path.read_text())
        
        tmp_data_dir = PanoPaths.tmp_dir() / "head_data"
        tmp_data_dir.mkdir(parents=True, exist_ok=True)

        tester_ip = f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'
        cfg["head_node_ip_addr"] = tester_ip
        cfg["head_node_data_dir"] = str(tmp_data_dir)
        cfg["head_node_container"] = True
        
        mids = []
        obs = config_file.get_obs_config()
        for dome in obs.domes:
            for module in dome.modules:
                mids.append(config_file.ip_addr_to_module_id(str(module.ip_addr)))

        daqnode_ip = "192.168.3.30"
        cfg["daq_nodes"] = [
            {
                "ip_addr": daqnode_ip,
                "data_dir": "/data",
                "username": "root",
                "module_ids": mids,
                "bindhost": "lo"
            }
        ]
        
        path.write_text(json.dumps(cfg, indent=4))
        
        # Write matching quabo_uids.json to tmp/
        uids_path = PanoPaths.tmp_dir() / "quabo_uids.json"
        uids_dict: dict[str, Any] = {"domes": [{"num": 0, "modules": []}]}
        for mid in mids:
            uids_dict["domes"][0]["modules"].append({
                "id": mid,
                "ip_addr": f"192.168.3.{mid}",
                "quabos": [{"uid": f"q{mid}_{j}"} if j==0 else {"uid": ""} for j in range(4)]
            })
        uids_path.write_text(json.dumps(uids_dict, indent=4))

        with patch("control.utils.config_file.get_quabo_uids", return_value=QuaboUids(**uids_dict)):
            try:
                yield tmp_data_dir
            finally:
                if os.path.exists(backup):
                    import shutil
                    shutil.move(backup, path)

    return _mock

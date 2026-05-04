"""
test_integration_two_node.py — Tier 5 Heavy Integration tests with two DAQ nodes.

Connects to the STATIC Docker Compose stack.
"""
from __future__ import annotations

import os
import pathlib

from ci.software_only.conftest import (
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)


def _prepare_dirs(params: dict) -> None:
    """Prepare host-side directories mapped to the container's /data."""
    host_data_root = os.environ.get("DAQ_DATA_DIR")
    if not host_data_root:
        return
        
    host_root = pathlib.Path(host_data_root)
    run_dir = params["run_dir"]
    
    # Root run dir for validator
    main_dir = host_root / run_dir
    main_dir.mkdir(parents=True, exist_ok=True)
    
    for mid in params["module_id"]:
        mod_dir = host_root / f"module_{mid}" / run_dir
        mod_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = mod_dir / "dummy.pff"
        dummy_file.touch()

class TestIntegrationTwoNodeDirect:
    """Two DAQ nodes managed independently in heavy integration stack."""

    def test_both_nodes_start_independently(
        self, daq_control_direct, daq_control_node2, run_params
    ) -> None:
        """Both nodes can be started simultaneously and both report running."""
        # Use common run_params but different module_ids for isolation
        params1 = dict(run_params)
        params1["module_id"] = [200]
        params2 = dict(run_params)
        params2["module_id"] = [201]

        _prepare_dirs(params1)
        _prepare_dirs(params2)
        
        assert daq_control_direct.StartDaq(params1) is True
        assert daq_control_node2.StartDaq(params2) is True

        try:
            assert wait_hashpipe_running(daq_control_direct, "/data"), "hashpipe 1 failed"
            assert wait_hashpipe_running(daq_control_node2, "/data"), "hashpipe 2 failed"

            _, s1 = daq_control_direct.StatusDaq({
                "data_dir":               "/data",
                "check_hashpipe_running": True,
                "check_disk_usage":       False,
                "check_run_dirs":         False,
            })
            _, s2 = daq_control_node2.StatusDaq({
                "data_dir":               "/data",
                "check_hashpipe_running": True,
                "check_disk_usage":       False,
                "check_run_dirs":         False,
            })
            assert s1.get("hashpipe_running") is True
            assert s2.get("hashpipe_running") is True
        finally:
            daq_control_direct.StopDaq(params1)
            daq_control_node2.StopDaq(params2)

    def test_stop_node1_does_not_affect_node2(
        self, daq_control_direct, daq_control_node2, run_params
    ) -> None:
        """Stopping hashpipe on node 1 does not stop it on node 2."""
        params1 = dict(run_params)
        params1["module_id"] = [200]
        params2 = dict(run_params)
        params2["module_id"] = [201]

        _prepare_dirs(params1)
        _prepare_dirs(params2)
        
        assert daq_control_direct.StartDaq(params1) is True
        assert daq_control_node2.StartDaq(params2) is True
        
        try:
            assert wait_hashpipe_running(daq_control_direct, "/data")
            assert wait_hashpipe_running(daq_control_node2, "/data")

            daq_control_direct.StopDaq(params1)
            assert wait_hashpipe_stopped(daq_control_direct, "/data")

            _, s1 = daq_control_direct.StatusDaq({
                "data_dir":               "/data",
                "check_hashpipe_running": True,
                "check_disk_usage":       False,
                "check_run_dirs":         False,
            })
            assert s1.get("hashpipe_running") is False
            
            _, s2 = daq_control_node2.StatusDaq({
                "data_dir":               "/data",
                "check_hashpipe_running": True,
                "check_disk_usage":       False,
                "check_run_dirs":         False,
            })
            assert s2.get("hashpipe_running") is True
        finally:
            daq_control_direct.StopDaq(params1)
            daq_control_node2.StopDaq(params2)

    def test_run_dirs_are_independent(
        self, daq_control_direct, daq_control_node2, run_params
    ) -> None:
        """Each node tracks its own run_dir independently."""
        params1 = dict(run_params)
        params1["run_dir"] = "run1.pffd"
        params1["module_id"] = [200]
        params2 = dict(run_params)
        params2["run_dir"] = "run2.pffd"
        params2["module_id"] = [201]

        _prepare_dirs(params1)
        _prepare_dirs(params2)

        try:
            assert daq_control_direct.StartDaq(params1) is True
            assert daq_control_node2.StartDaq(params2) is True

            assert wait_hashpipe_running(daq_control_direct, "/data")
            assert wait_hashpipe_running(daq_control_node2, "/data")

            _, s1 = daq_control_direct.StatusDaq({
                "data_dir":               "/data",
                "check_hashpipe_running": False,
                "check_disk_usage":       False,
                "check_run_dirs":         True,
            })
            _, s2 = daq_control_node2.StatusDaq({
                "data_dir":               "/data",
                "check_hashpipe_running": False,
                "check_disk_usage":       False,
                "check_run_dirs":         True,
            })
            run_dirs_1 = s1.get("run_dirs", [])
            run_dirs_2 = s2.get("run_dirs", [])

            assert any("run1.pffd" in d for d in run_dirs_1)
            assert any("run2.pffd" in d for d in run_dirs_2)
            assert not any("run2.pffd" in d and "module_201" in d for d in run_dirs_1)
        finally:
            daq_control_direct.StopDaq(params1)
            daq_control_node2.StopDaq(params2)

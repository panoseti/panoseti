"""
tier5_integration/test_integration_two_node.py — Two-node heavy integration tests.

Connects to the STATIC Docker Compose stack.
Tests start/stop command propagation and isolation across two real daqnodes.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import pytest

from ci.software_only.tier5_integration.conftest import (
    DAQ_DATA_DIR,
    requires_compose_stack,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)

pytestmark = [pytest.mark.tier5, requires_compose_stack]


def _prepare_dirs(params: dict[str, Any]) -> None:
    """Create host-side directories mapped to the container's /data volume."""
    host_root = pathlib.Path(DAQ_DATA_DIR)
    run_dir = params["run_dir"]

    (host_root / run_dir).mkdir(parents=True, exist_ok=True)
    os.chmod(host_root / run_dir, 0o777)

    for mid in params["module_id"]:
        mod_dir = host_root / f"module_{mid}" / run_dir
        mod_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(mod_dir, 0o777)
        dummy = mod_dir / "dummy.pff"
        dummy.touch()
        os.chmod(dummy, 0o777)


class TestIntegrationTwoNodeDirect:
    """Two real daqnodes managed independently via gRPC."""

    def test_both_nodes_start_independently(
        self,
        daq_control_node1: Any,
        daq_control_node2: Any,
        run_params: dict[str, Any],
    ) -> None:
        """Both nodes can be started in parallel and both report hashpipe running."""
        p1 = {**run_params, "module_id": [200]}
        p2 = {**run_params, "module_id": [201]}
        _prepare_dirs(p1)
        _prepare_dirs(p2)

        assert daq_control_node1.StartDaq(p1) is True
        assert daq_control_node2.StartDaq(p2) is True

        try:
            assert wait_hashpipe_running(daq_control_node1, DAQ_DATA_DIR), (
                "hashpipe did not start on node 1"
            )
            assert wait_hashpipe_running(daq_control_node2, DAQ_DATA_DIR), (
                "hashpipe did not start on node 2"
            )

            _status_params = {
                "data_dir": DAQ_DATA_DIR,
                "check_hashpipe_running": True,
                "check_disk_usage": False,
                "check_run_dirs": False,
            }
            _, s1 = daq_control_node1.StatusDaq(_status_params)
            _, s2 = daq_control_node2.StatusDaq(_status_params)
            assert s1.get("hashpipe_running") is True
            assert s2.get("hashpipe_running") is True
        finally:
            daq_control_node1.StopDaq(p1)
            daq_control_node2.StopDaq(p2)

    def test_stop_node1_does_not_affect_node2(
        self,
        daq_control_node1: Any,
        daq_control_node2: Any,
        run_params: dict[str, Any],
    ) -> None:
        """Stopping hashpipe on node 1 leaves node 2 running."""
        p1 = {**run_params, "module_id": [200]}
        p2 = {**run_params, "module_id": [201]}
        _prepare_dirs(p1)
        _prepare_dirs(p2)

        assert daq_control_node1.StartDaq(p1) is True
        assert daq_control_node2.StartDaq(p2) is True

        try:
            assert wait_hashpipe_running(daq_control_node1, DAQ_DATA_DIR)
            assert wait_hashpipe_running(daq_control_node2, DAQ_DATA_DIR)

            daq_control_node1.StopDaq(p1)
            assert wait_hashpipe_stopped(daq_control_node1, DAQ_DATA_DIR)

            _status_params = {
                "data_dir": DAQ_DATA_DIR,
                "check_hashpipe_running": True,
                "check_disk_usage": False,
                "check_run_dirs": False,
            }
            _, s1 = daq_control_node1.StatusDaq(_status_params)
            _, s2 = daq_control_node2.StatusDaq(_status_params)
            assert s1.get("hashpipe_running") is False
            assert s2.get("hashpipe_running") is True
        finally:
            daq_control_node1.StopDaq(p1)
            daq_control_node2.StopDaq(p2)

    def test_run_dirs_are_independent(
        self,
        daq_control_node1: Any,
        daq_control_node2: Any,
        run_params: dict[str, Any],
    ) -> None:
        """Each node tracks its own run_dir independently."""
        p1 = {**run_params, "run_dir": "t5_run1.pffd", "module_id": [200]}
        p2 = {**run_params, "run_dir": "t5_run2.pffd", "module_id": [201]}
        _prepare_dirs(p1)
        _prepare_dirs(p2)

        try:
            assert daq_control_node1.StartDaq(p1) is True
            assert daq_control_node2.StartDaq(p2) is True
            assert wait_hashpipe_running(daq_control_node1, DAQ_DATA_DIR)
            assert wait_hashpipe_running(daq_control_node2, DAQ_DATA_DIR)

            _s = {
                "data_dir": DAQ_DATA_DIR,
                "check_hashpipe_running": False,
                "check_disk_usage": False,
                "check_run_dirs": True,
            }
            _, s1 = daq_control_node1.StatusDaq(_s)
            _, s2 = daq_control_node2.StatusDaq(_s)

            dirs1 = s1.get("run_dirs", [])
            dirs2 = s2.get("run_dirs", [])
            assert any("t5_run1.pffd" in d for d in dirs1)
            assert any("t5_run2.pffd" in d for d in dirs2)
            # run2 dirs must not appear on node 1 for module_201
            assert not any("t5_run2.pffd" in d and "module_201" in d for d in dirs1)
        finally:
            daq_control_node1.StopDaq(p1)
            daq_control_node2.StopDaq(p2)

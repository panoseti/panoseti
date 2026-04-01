"""
test_two_node_direct.py — Integration tests with two independent DAQ nodes.

Uses daqnode  (192.168.0.10, daq_control) and
     daqnode-2 (192.168.0.20, daq_control).

Verifies that both nodes can be managed independently — start/stop on one
does not affect the other, run directories are isolated per node, and
concurrent StartDaq calls work without interference.
"""
from __future__ import annotations

import time
import uuid

import pytest

from panoseti_grpc.daq_control.client import DaqControlClient

from .conftest import (
    DAQ_DATA_DIR, BINDHOST,
    wait_hashpipe_stopped, wait_hashpipe_running
)


# ---------------------------------------------------------------------------
# Node-2 run parameters (different module and run_dir from the default fixture)
# ---------------------------------------------------------------------------

@pytest.fixture
def run_params_node2() -> dict:
    """Fresh run parameters for node-2 — distinct run_dir and module_id."""
    return {
        "data_dir":         DAQ_DATA_DIR,
        "daq_ip_addr":      "192.168.0.20",
        "bindhost":         BINDHOST,
        "max_file_size_mb": 1,
        "group_ph_frames":  True,
        "run_dir":          f"ci_run2_{uuid.uuid4().hex[:8]}.pffd",
        "obs":              "citest",
        "module_id":        [200],
    }


@pytest.fixture(autouse=True)
def ensure_node2_clean(daq_control_node2, run_params_node2):
    """Stop and cleanup node-2 after each test regardless of outcome."""
    yield
    try:
        daq_control_node2.StopDaq({
            "data_dir": run_params_node2["data_dir"],
            "run_dir":  run_params_node2["run_dir"],
        })
    except Exception:
        pass

    # We must block until it is actually stopped before proceeding to the next test.
    assert wait_hashpipe_stopped(daq_control_node2, run_params_node2["data_dir"], timeout=8), (
        "hashpipe did not stop within timeout"
    )

    try:
        daq_control_node2.CleanupData({
            "data_dir":  run_params_node2["data_dir"],
            "run_dir":   run_params_node2["run_dir"],
            "module_id": run_params_node2["module_id"],
        })
    except Exception:
        pass

@pytest.fixture(autouse=True)
def ensure_node1_clean(daq_control_direct, run_params):
    """Stop and cleanup node-2 after each test regardless of outcome."""
    yield
    try:
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
    except Exception:
        pass

    assert wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"]), (
        "hashpipe did not stop within timeout"
    )
    try:
        daq_control_direct.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })
    except Exception:
        pass


class TestTwoNodeDirect:
    """Two DAQ nodes can be managed completely independently."""

    def test_node1_starts(self, daq_control_direct, run_params):
        """Node 1 (192.168.0.10) starts hashpipe successfully."""
        ok = daq_control_direct.StartDaq(run_params)
        assert ok is True

    def test_node2_starts(self, daq_control_node2, run_params_node2):
        """Node 2 (192.168.0.20) starts hashpipe successfully."""
        ok = daq_control_node2.StartDaq(run_params_node2)
        assert ok is True
    
    def test_both_nodes_stop(
        self, daq_control_direct, daq_control_node2, run_params, run_params_node2
    ):
        """Both nodes can be stopped simultaneously and both report running."""
        assert daq_control_direct.StopDaq(run_params) is True
        assert daq_control_node2.StopDaq(run_params_node2) is True
        assert wait_hashpipe_stopped(daq_control_node2, run_params_node2["data_dir"]), (
            "hashpipe did not stop within timeout"
        )

    def test_both_nodes_start_independently(
        self, daq_control_direct, daq_control_node2, run_params, run_params_node2
    ):
        """Both nodes can be started simultaneously and both report running."""
        assert daq_control_direct.StartDaq(run_params) is True
        assert daq_control_node2.StartDaq(run_params_node2) is True

        assert wait_hashpipe_running(daq_control_direct, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        assert wait_hashpipe_running(daq_control_node2, run_params_node2["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        _, s1 = daq_control_direct.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        _, s2 = daq_control_node2.StatusDaq({
            "data_dir":               run_params_node2["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        assert s1.get("hashpipe_running") is True
        assert s2.get("hashpipe_running") is True

    def test_stop_node1_does_not_affect_node2(
        self, daq_control_direct, daq_control_node2, run_params, run_params_node2
    ):
        """Stopping hashpipe on node 1 does not stop it on node 2."""
        _, s1 = daq_control_direct.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        _, s2 = daq_control_node2.StatusDaq({
            "data_dir":               run_params_node2["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        assert s1.get("hashpipe_running") is False, "Node 1 should be stopped"
        assert s2.get("hashpipe_running") is False,  "Node 2 should be stopped"
        
        
        # Start test
        assert daq_control_direct.StartDaq(run_params) is True
        assert daq_control_node2.StartDaq(run_params_node2) is True
        
        # assert wait_hashpipe_running(daq_control_direct, run_params["data_dir"]), (
        #     "hashpipe did not start within timeout"
        # )

        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        assert wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"]), (
            "hashpipe did not stop within timeout"
        )

        _, s1 = daq_control_direct.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        assert s1.get("hashpipe_running") is False, "Node 1 should be stopped"
        assert wait_hashpipe_running(daq_control_node2, run_params_node2["data_dir"]), (
            "hashpipe on daq node 2 did not start within timeout"
        )
        _, s2 = daq_control_node2.StatusDaq({
            "data_dir":               run_params_node2["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        assert s2.get("hashpipe_running") is True,  "Node 2 should still be running"

    def test_run_dirs_are_independent(
        self, daq_control_direct, daq_control_node2, run_params, run_params_node2
    ):
        """Each node tracks its own run_dir independently."""
        assert daq_control_direct.StartDaq(run_params) is True
        assert daq_control_node2.StartDaq(run_params_node2) is True

        assert wait_hashpipe_running(daq_control_direct, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        assert wait_hashpipe_running(daq_control_node2, run_params_node2["data_dir"]), (
            "hashpipe did not start within timeout"
        )

        _, s1 = daq_control_direct.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": False,
            "check_disk_usage":       False,
            "check_run_dirs":         True,
        })
        _, s2 = daq_control_node2.StatusDaq({
            "data_dir":               run_params_node2["data_dir"],
            "check_hashpipe_running": False,
            "check_disk_usage":       False,
            "check_run_dirs":         True,
        })
        run_dirs_1 = s1.get("run_dirs", [])
        run_dirs_2 = s2.get("run_dirs", [])

        assert any(run_params["run_dir"] in d for d in run_dirs_1), (
            f"Node 1 run_dir={run_params['run_dir']!r} not found in {run_dirs_1}"
        )
        assert any(run_params_node2["run_dir"] in d for d in run_dirs_2), (
            f"Node 2 run_dir={run_params_node2['run_dir']!r} not found in {run_dirs_2}"
        )
        # Node 2's run_dir should not appear on node 1 (they share the volume, but
        # module-level directories are distinct: module_200 vs module_201)
        assert not any(run_params_node2["run_dir"] in d and "module_201" in d for d in run_dirs_1), (
            "Node 2 module data unexpectedly appeared on node 1"
        )

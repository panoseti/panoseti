"""
test_two_node_direct.py — Integration tests with two independent DAQ nodes.

Uses daqnode  (192.168.0.10, daq_control) and
     daqnode-2 (192.168.0.20, daq_control).

Verifies that both nodes can be managed independently — start/stop on one
does not affect the other, run directories are isolated per node, and
concurrent StartDaq calls work without interference.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from ci.tier3_fleet.conftest import (
    BINDHOST,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)

# ---------------------------------------------------------------------------
# Node run parameters
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def run_params_node1(session_fleet) -> dict[str, Any]:
    """Base run parameters for node-1."""
    fleet, _ = session_fleet
    return {
        "data_dir":         "/data",
        "daq_ip_addr":      fleet.node_ip(0),
        "bindhost":         BINDHOST,
        "max_file_size_mb": 1,
        "group_ph_frames":  True,
        "run_dir":          f"ci_run1_{uuid.uuid4().hex[:8]}.pffd",
        "obs":              "citest",
        "module_id":        [200],
    }

@pytest.fixture(scope='module')
def run_params_node2(session_fleet) -> dict[str, Any]:
    """Fresh run parameters for node-2 — distinct run_dir and module_id."""
    fleet, _ = session_fleet
    return {
        "data_dir":         "/data",
        "daq_ip_addr":      fleet.node_ip(1),
        "bindhost":         BINDHOST,
        "max_file_size_mb": 1,
        "group_ph_frames":  True,
        "run_dir":          f"ci_run2_{uuid.uuid4().hex[:8]}.pffd",
        "obs":              "citest",
        "module_id":        [201],
    }


@pytest.fixture(autouse=True)
def ensure_node2_clean(daq_control_node2: Any, run_params_node2: dict[str, Any]) -> Iterator[None]:
    """Stop and cleanup node-2 after each test regardless of outcome."""
    yield
    with contextlib.suppress(Exception):
        daq_control_node2.StopDaq({
            "data_dir": run_params_node2["data_dir"],
            "run_dir":  run_params_node2["run_dir"],
        })

    # We must block until it is actually stopped before proceeding to the next test.
    wait_hashpipe_stopped(daq_control_node2, run_params_node2["data_dir"], timeout=8)

    with contextlib.suppress(Exception):
        for mid in run_params_node2["module_id"]:
            daq_control_node2.CleanupData({
                "data_dir":  run_params_node2["data_dir"],
                "run_dir":   run_params_node2["run_dir"],
                "module_id": mid,
            })

@pytest.fixture(autouse=True)
def ensure_node1_clean(daq_control_direct: Any, run_params_node1: dict[str, Any]) -> Iterator[None]:
    """Stop and cleanup node-1 after each test regardless of outcome."""
    yield
    with contextlib.suppress(Exception):
        daq_control_direct.StopDaq({
            "data_dir": run_params_node1["data_dir"],
            "run_dir":  run_params_node1["run_dir"],
        })

    wait_hashpipe_stopped(daq_control_direct, run_params_node1["data_dir"], timeout=8)
    with contextlib.suppress(Exception):
        for mid in run_params_node1["module_id"]:
            daq_control_direct.CleanupData({
                "data_dir":  run_params_node1["data_dir"],
                "run_dir":   run_params_node1["run_dir"],
                "module_id": mid,
            })


def _prepare_dirs(params: dict) -> None:
    """
    Split-Brain Data Injection: 
    Use the host-side DAQ_DATA_DIR to create directories, while gRPC
    commands continue to use the container-relative path (/data).
    """
    host_data_root = os.environ.get("DAQ_DATA_DIR")
    if not host_data_root:
        return
        
    host_root = pathlib.Path(host_data_root)
    run_dir = params["run_dir"]
    
    # Root run dir for validator
    main_dir = host_root / run_dir
    main_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(main_dir, 0o777)
    
    for mid in params["module_id"]:
        mod_dir = host_root / f"module_{mid}" / run_dir
        mod_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = mod_dir / "dummy.pff"
        dummy_file.touch()
        os.chmod(dummy_file, 0o777)

        # Recursive chmod 0o777 for the module hierarchy
        for root, dirs, files in os.walk(host_root / f"module_{mid}"):
            os.chmod(root, 0o777)
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o777)


class TestTwoNodeDirect:
    """Two DAQ nodes can be managed completely independently."""

    def test_node1_starts(self, daq_control_direct, run_params_node1) -> None:
        """Node 1 starts hashpipe successfully."""
        _prepare_dirs(run_params_node1)
        ok = daq_control_direct.StartDaq(run_params_node1)
        assert ok is True

    def test_node2_starts(self, daq_control_node2, run_params_node2) -> None:
        """Node 2 starts hashpipe successfully."""
        _prepare_dirs(run_params_node2)
        ok = daq_control_node2.StartDaq(run_params_node2)
        assert ok is True
    
    def test_both_nodes_stop(
        self, daq_control_direct, daq_control_node2, run_params_node1, run_params_node2
    ) -> None:
        """Both nodes can be stopped simultaneously."""
        _prepare_dirs(run_params_node1)
        _prepare_dirs(run_params_node2)
        daq_control_direct.StartDaq(run_params_node1)
        daq_control_node2.StartDaq(run_params_node2)

        assert daq_control_direct.StopDaq(run_params_node1) is True
        assert daq_control_node2.StopDaq(run_params_node2) is True
        assert wait_hashpipe_stopped(daq_control_node2, run_params_node2["data_dir"]), (
            "hashpipe did not stop within timeout"
        )

    @pytest.mark.skip(reason="Requires Tier 5 Heavy Integration Stack")
    def test_both_nodes_start_independently(
        self, daq_control_direct, daq_control_node2, run_params_node1, run_params_node2
    ) -> None:
        """Both nodes can be started simultaneously and both report running."""
        _prepare_dirs(run_params_node1)
        _prepare_dirs(run_params_node2)
        
        assert daq_control_direct.StartDaq(run_params_node1) is True
        assert daq_control_node2.StartDaq(run_params_node2) is True

        assert wait_hashpipe_running(daq_control_direct, run_params_node1["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        assert wait_hashpipe_running(daq_control_node2, run_params_node2["data_dir"]), (
            "hashpipe-2 did not start within timeout"
        )

        _, s1 = daq_control_direct.StatusDaq({
            "data_dir":               run_params_node1["data_dir"],
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
        self, daq_control_direct, daq_control_node2, run_params_node1, run_params_node2
    ) -> None:
        """Stopping hashpipe on node 1 does not stop it on node 2."""
        _prepare_dirs(run_params_node1)
        _prepare_dirs(run_params_node2)
        
        assert daq_control_direct.StartDaq(run_params_node1) is True
        assert daq_control_node2.StartDaq(run_params_node2) is True
        
        assert wait_hashpipe_running(daq_control_direct, run_params_node1["data_dir"])
        assert wait_hashpipe_running(daq_control_node2, run_params_node2["data_dir"])

        daq_control_direct.StopDaq(run_params_node1)
        assert wait_hashpipe_stopped(daq_control_direct, run_params_node1["data_dir"])

        _, s1 = daq_control_direct.StatusDaq({
            "data_dir":               run_params_node1["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        assert s1.get("hashpipe_running") is False, "Node 1 should be stopped"
        
        _, s2 = daq_control_node2.StatusDaq({
            "data_dir":               run_params_node2["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        assert s2.get("hashpipe_running") is True,  "Node 2 should still be running"

    def test_run_dirs_are_independent(
        self, daq_control_direct, daq_control_node2, run_params_node1, run_params_node2
    ) -> None:
        """Each node tracks its own run_dir independently."""
        _prepare_dirs(run_params_node1)
        _prepare_dirs(run_params_node2)

        assert daq_control_direct.StartDaq(run_params_node1) is True
        assert daq_control_node2.StartDaq(run_params_node2) is True

        assert wait_hashpipe_running(daq_control_direct, run_params_node1["data_dir"])
        assert wait_hashpipe_running(daq_control_node2, run_params_node2["data_dir"])

        _, s1 = daq_control_direct.StatusDaq({
            "data_dir":               run_params_node1["data_dir"],
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

        assert any(run_params_node1["run_dir"] in d for d in run_dirs_1), (
            f"Node 1 run_dir={run_params_node1['run_dir']!r} not found in {run_dirs_1}"
        )
        assert any(run_params_node2["run_dir"] in d for d in run_dirs_2), (
            f"Node 2 run_dir={run_params_node2['run_dir']!r} not found in {run_dirs_2}"
        )
        # Node 2's run_dir should not appear on node 1 (they share the volume, but
        # module-level directories are distinct: module_200 vs module_201)
        assert not any(run_params_node2["run_dir"] in d and "module_201" in d for d in run_dirs_1), (
            "Node 2 module data unexpectedly appeared on node 1"
        )

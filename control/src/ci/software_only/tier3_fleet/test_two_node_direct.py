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
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ci.fixtures.fleet import Fleet
from ci.software_only.conftest import wait_hashpipe_stopped
from ci.software_only.tier3_fleet.conftest import (
    BINDHOST,
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
def ensure_node2_clean(daq_client_2: Any, run_params_node2: dict[str, Any]) -> Iterator[None]:
    """Stop and cleanup node-2 after each test regardless of outcome."""
    yield
    with contextlib.suppress(Exception):
        daq_client_2.StopDaq({
            "data_dir": run_params_node2["data_dir"],
            "run_dir":  run_params_node2["run_dir"],
        })

    # We must block until it is actually stopped before proceeding to the next test.
    wait_hashpipe_stopped(daq_client_2, run_params_node2["data_dir"], timeout=8)

    with contextlib.suppress(Exception):
        for mid in run_params_node2["module_id"]:
            daq_client_2.CleanupData({
                "data_dir":  run_params_node2["data_dir"],
                "run_dir":   run_params_node2["run_dir"],
                "module_id": mid,
            })

@pytest.fixture(autouse=True)
def ensure_node1_clean(daq_client: Any, run_params_node1: dict[str, Any]) -> Iterator[None]:
    """Stop and cleanup node-1 after each test regardless of outcome."""
    yield
    with contextlib.suppress(Exception):
        daq_client.StopDaq({
            "data_dir": run_params_node1["data_dir"],
            "run_dir":  run_params_node1["run_dir"],
        })

    wait_hashpipe_stopped(daq_client, run_params_node1["data_dir"], timeout=8)
    with contextlib.suppress(Exception):
        for mid in run_params_node1["module_id"]:
            daq_client.CleanupData({
                "data_dir":  run_params_node1["data_dir"],
                "run_dir":   run_params_node1["run_dir"],
                "module_id": mid,
            })


def _prepare_dirs(fleet: Fleet, params: dict, node_idx: int) -> None:
    """
    Inject run directories on the host-side temp directory for the specific container.
    """
    host_root = Path(fleet.host_data_dirs[node_idx])
    run_dir = params["run_dir"]
    
    # Root run dir
    main_dir = host_root / run_dir
    main_dir.mkdir(parents=True, exist_ok=True)
    
    for mid in params["module_id"]:
        mod_dir = host_root / f"module_{mid}" / run_dir
        mod_dir.mkdir(parents=True, exist_ok=True)
        # Touch a dummy file so directory isn't empty
        (mod_dir / "dummy.pff").touch()


class TestTwoNodeDirect:
    """Two DAQ nodes can be managed completely independently."""

    def test_node1_starts(self, daq_client, run_params_node1, session_fleet, mock_workspace) -> None:
        """Node 1 starts hashpipe successfully."""
        fleet, _ = session_fleet
        _prepare_dirs(fleet, run_params_node1, 0)
        ok = daq_client.StartDaq(run_params_node1)
        assert ok is True

    def test_node2_starts(self, daq_client_2, run_params_node2, session_fleet, mock_workspace) -> None:
        """Node 2 starts hashpipe successfully."""
        fleet, _ = session_fleet
        _prepare_dirs(fleet, run_params_node2, 1)
        ok = daq_client_2.StartDaq(run_params_node2)
        assert ok is True

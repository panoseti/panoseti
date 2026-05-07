# mypy: ignore-errors
"""
test_two_node_direct.py — Integration tests with two independent DAQ nodes.

Ported from ci/software_only/tier3_fleet/test_two_node_direct.py.
Verifies that both nodes can be managed independently using two_node_ci fleet.
"""

from __future__ import annotations

import pytest
import docker

from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.infra.workspace import Workspace
from ci.software_only_v2.orchestrator.fleet import Fleet

pytestmark = pytest.mark.tier3


def _docker_available() -> bool:
    try:
        import docker
        docker.from_env(timeout=5).ping()
        return True
    except Exception:
        return False


requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available",
)


@requires_docker
@pytest.mark.parametrize(
    "pseti_workspace_session",
    [FleetSpec.two_node_ci(tier="tier3")],
    indirect=True,
)
class TestTwoNodeDirect:
    """Two DAQ nodes can be managed completely independently."""

    def test_nodes_start_independently(self, session_fleet: Fleet) -> None:
        """Both nodes start successfully and don't interfere."""
        # session_fleet is already healthy and ready.
        fleet = session_fleet
        topology = fleet.workspace.topology

        # Node 0
        client0 = fleet.daq_control_client(0)
        run_params0 = {
            "data_dir": "/data",
            "run_dir": "run0",
            "module_id": list(topology.daq.daq_nodes[0].module_ids)
        }
        # Prepare dir in container
        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            "mkdir -p /data/run0 && chmod 777 /data/run0"
        )
        ok0 = client0.StartDaq(run_params0)
        assert ok0 is True
        
        # Node 1
        client1 = fleet.daq_control_client(1)
        run_params1 = {
            "data_dir": "/data",
            "run_dir": "run1",
            "module_id": list(topology.daq.daq_nodes[1].module_ids)
        }
        # Prepare dir in container
        docker.from_env().containers.get(fleet.daq_nodes[1].name).exec_run(
            "mkdir -p /data/run1 && chmod 777 /data/run1"
        )
        ok1 = client1.StartDaq(run_params1)
        assert ok1 is True
        
        # Status check
        _, s0 = client0.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
        _, s1 = client1.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
        
        # In sim mode, hashpipe_running is True if StartDaq was called, even if pid is 0
        assert s0["hashpipe_running"] is True
        assert s1["hashpipe_running"] is True
        
        # Stop Node 0
        client0.StopDaq({"data_dir": "/data", "run_dir": "run0"})
        _, s0_stopped = client0.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
        _, s1_still_running = client1.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
        
        assert s0_stopped["hashpipe_running"] is False
        assert s1_still_running["hashpipe_running"] is True
        
        from ci.software_only_v2.infra.parity import run_scenario
        run_scenario("two_node_independent_lifecycle", 
                     node_0_running=s0_stopped["hashpipe_running"], 
                     node_1_running=s1_still_running["hashpipe_running"])
        
        client0.close()
        client1.close()

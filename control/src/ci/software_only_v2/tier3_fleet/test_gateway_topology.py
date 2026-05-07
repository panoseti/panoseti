"""
tier3_fleet/test_gateway_topology.py — Gateway (PortForwarding) topology tests.

Ported from ci/software_only/tier3_fleet/test_gateway_topology.py.
Verifies that nodes configured with a GatewaySpec (PortForwarding) are correctly
managed by the control plane. In v2, the 'gateway' is the host machine
mapping container ports to localhost.
"""

from __future__ import annotations

import time
import pytest
import docker

from ci.software_only_v2.infra.spec import FleetSpec, GatewaySpec
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


def wait_until(condition, timeout=10.0, interval=0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if condition():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


SPEC_GATEWAY = (
    FleetSpec(seed=300, name="gateway_test", tier="tier3")
    .with_headnode(ip="10.0.1.5")
    .add_dome("d0", lat=37, lon=-121, alt=1000)
    .add_module(300, ip="192.168.3.32")
    .add_daq_node(
        ip="192.168.0.10",
        modules=[300],
        gateway=GatewaySpec(ip="10.200.146.13", grpc_port=50051),
        bindhost="lo"
    )
)


@requires_docker
@pytest.mark.parametrize(
    "pseti_workspace_session",
    [SPEC_GATEWAY],
    indirect=True,
)
class TestGatewayForwarding:
    """Gateway (PortForwarding) client reaches the daqnode and observes consistent state."""

    def test_gateway_client_starts_daq(self, session_fleet: Fleet) -> None:
        """DaqControlClient via gateway (host-mapped port) can issue StartDaq successfully."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        topology = fleet.workspace.topology

        run_params = {
            "data_dir": "/data",
            "run_dir": "gw_start_run",
            "module_id": list(topology.daq.daq_nodes[0].module_ids)
        }

        # Prepare dir in container
        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            f"mkdir -p /data/{run_params['run_dir']} && chmod 777 /data/{run_params['run_dir']}"
        )

        ok = client.StartDaq(run_params)
        assert ok is True
        client.close()

    def test_gateway_client_reports_running(self, session_fleet: Fleet) -> None:
        """After StartDaq via gateway, StatusDaq sees hashpipe_running=True."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        topology = fleet.workspace.topology

        run_params = {
            "data_dir": "/data",
            "run_dir": "gw_report_run",
            "module_id": list(topology.daq.daq_nodes[0].module_ids)
        }

        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            f"mkdir -p /data/{run_params['run_dir']} && chmod 777 /data/{run_params['run_dir']}"
        )

        client.StartDaq(run_params)
        
        def check_running():
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return s["hashpipe_running"] is True
        
        assert wait_until(check_running), "hashpipe did not start within timeout"
        client.close()

    def test_gateway_consistency(self, session_fleet: Fleet) -> None:
        """Direct and gateway clients report the same hashpipe_running state."""
        fleet = session_fleet
        topology = fleet.workspace.topology
        client = fleet.daq_control_client(0)

        run_params = {
            "data_dir": "/data",
            "run_dir": "gw_consistency_run",
            "module_id": list(topology.daq.daq_nodes[0].module_ids)
        }

        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            f"mkdir -p /data/{run_params['run_dir']} && chmod 777 /data/{run_params['run_dir']}"
        )

        client.StartDaq(run_params)
        
        def check_running():
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return s["hashpipe_running"] is True
        assert wait_until(check_running)

        # In v2, both clients are the same path (host-mapped port), 
        # but we can simulate two clients.
        client_direct = fleet.daq_control_client(0)
        client_gateway = fleet.daq_control_client(0)
        
        status_req = {"data_dir": "/data", "check_hashpipe_running": True}
        _, s_direct = client_direct.StatusDaq(status_req)
        _, s_gateway = client_gateway.StatusDaq(status_req)
        
        from ci.software_only_v2.infra.parity import run_scenario
        run_scenario("gateway_consistency", 
                     direct_running=s_direct["hashpipe_running"], 
                     gateway_running=s_gateway["hashpipe_running"])
        
        client_direct.close()
        client_gateway.close()
        client.close()

    def test_gateway_stop_is_visible(self, session_fleet: Fleet) -> None:
        """StopDaq issued via gateway makes hashpipe_running=False."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        topology = fleet.workspace.topology

        run_params = {
            "data_dir": "/data",
            "run_dir": "gw_stop_run",
            "module_id": list(topology.daq.daq_nodes[0].module_ids)
        }

        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            f"mkdir -p /data/{run_params['run_dir']} && chmod 777 /data/{run_params['run_dir']}"
        )

        client.StartDaq(run_params)
        
        def check_running():
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return s["hashpipe_running"] is True
        assert wait_until(check_running)

        ok = client.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        assert ok is True

        def check_stopped():
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return s["hashpipe_running"] is False
        assert wait_until(check_stopped), "hashpipe did not stop within timeout"
        client.close()

    def test_gateway_cleanup_after_stop(self, session_fleet: Fleet) -> None:
        """CleanupData via gateway succeeds after StopDaq."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        topology = fleet.workspace.topology

        run_params = {
            "data_dir": "/data",
            "run_dir": "gw_cleanup_run",
            "module_id": list(topology.daq.daq_nodes[0].module_ids)
        }

        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            f"mkdir -p /data/{run_params['run_dir']} && chmod 777 /data/{run_params['run_dir']}"
        )

        client.StartDaq(run_params)
        client.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })

        def check_stopped():
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return s["hashpipe_running"] is False
        assert wait_until(check_stopped)

        res = client.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })
        assert res["success"] is True
        client.close()

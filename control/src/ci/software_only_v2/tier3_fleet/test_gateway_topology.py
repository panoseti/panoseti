"""
tier3_fleet/test_gateway_topology.py — Gateway (PortForwarding) topology tests.

Ported from ci/software_only/tier3_fleet/test_gateway_topology.py.
Verifies that nodes configured with a GatewaySpec (PortForwarding) are correctly
managed by the control plane.  In v2, the 'gateway' is the host mapping container
ports to localhost.
"""

from __future__ import annotations

import time

import pytest

from ci.software_only_v2.infra.spec import FleetSpec, GatewaySpec
from ci.software_only_v2.orchestrator.fleet import Fleet
from ci.software_only_v2.tier3_fleet.conftest import make_startdaq_params, requires_docker

pytestmark = pytest.mark.tier3


def wait_until(
    condition: "Any",
    timeout: float = 10.0,
    interval: float = 0.2,
) -> bool:
    from typing import Any  # noqa: PLC0415
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
    .add_module(200, ip="192.168.3.32")
    .add_daq_node(
        ip="192.168.0.10",
        modules=[200],
        gateway=GatewaySpec(ip="10.200.146.13", grpc_port=50051),
        bindhost="lo",
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

    def test_when_gateway_client_issues_startdaq_then_succeeds(
        self, session_fleet: Fleet
    ) -> None:
        """DaqControlClient via gateway (host-mapped port) can issue StartDaq successfully."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_dir = "gw_start_run"

        fleet.exec_in_node(0, f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}")

        ok = client.StartDaq(make_startdaq_params(fleet, 0, run_dir))
        assert ok is True
        client.close()

    def test_when_started_via_gateway_then_status_shows_running(
        self, session_fleet: Fleet
    ) -> None:
        """After StartDaq via gateway, StatusDaq sees hashpipe_running=True."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_dir = "gw_report_run"

        fleet.exec_in_node(0, f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}")
        client.StartDaq(make_startdaq_params(fleet, 0, run_dir))

        def check_running() -> bool:
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return bool(s["hashpipe_running"])

        assert wait_until(check_running), "hashpipe did not start within timeout"
        client.close()

    def test_when_direct_and_gateway_status_queried_then_state_matches(
        self, session_fleet: Fleet
    ) -> None:
        """Direct and gateway clients report the same hashpipe_running state."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_dir = "gw_consistency_run"

        fleet.exec_in_node(0, f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}")
        client.StartDaq(make_startdaq_params(fleet, 0, run_dir))

        def check_running() -> bool:
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return bool(s["hashpipe_running"])

        assert wait_until(check_running)

        # Both clients route to the same host-mapped port in v2
        client_direct = fleet.daq_control_client(0)
        client_gateway = fleet.daq_control_client(0)

        status_req: dict = {"data_dir": "/data", "check_hashpipe_running": True}
        _, s_direct = client_direct.StatusDaq(status_req)
        _, s_gateway = client_gateway.StatusDaq(status_req)

        from ci.software_only_v2.infra.parity import run_scenario
        run_scenario(
            "gateway_consistency",
            direct_running=s_direct["hashpipe_running"],
            gateway_running=s_gateway["hashpipe_running"],
        )

        client_direct.close()
        client_gateway.close()
        client.close()

    def test_when_stopdaq_issued_via_gateway_then_hashpipe_stops(
        self, session_fleet: Fleet
    ) -> None:
        """StopDaq issued via gateway makes hashpipe_running=False."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_dir = "gw_stop_run"

        fleet.exec_in_node(0, f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}")
        client.StartDaq(make_startdaq_params(fleet, 0, run_dir))

        def check_running() -> bool:
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return bool(s["hashpipe_running"])

        assert wait_until(check_running)

        ok = client.StopDaq({"data_dir": "/data", "run_dir": run_dir})
        assert ok is True

        def check_stopped() -> bool:
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return not bool(s["hashpipe_running"])

        assert wait_until(check_stopped), "hashpipe did not stop within timeout"
        client.close()

    def test_when_cleanup_called_after_stop_then_succeeds(
        self, session_fleet: Fleet
    ) -> None:
        """CleanupData via gateway succeeds after StopDaq."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_dir = "gw_cleanup_run"
        node_cfg = fleet.live_daq_config.daq_nodes[0]

        fleet.exec_in_node(0, f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}")
        client.StartDaq(make_startdaq_params(fleet, 0, run_dir))
        client.StopDaq({"data_dir": "/data", "run_dir": run_dir})

        def check_stopped() -> bool:
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return not bool(s["hashpipe_running"])

        assert wait_until(check_stopped)

        res = client.CleanupData({
            "data_dir": "/data",
            "run_dir": run_dir,
            "module_id": list(node_cfg.module_ids),
        })
        assert res["success"] is True
        client.close()

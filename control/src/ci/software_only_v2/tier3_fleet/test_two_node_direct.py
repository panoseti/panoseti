"""
test_two_node_direct.py — Integration tests with two independent DAQ nodes.

Ported from ci/software_only/tier3_fleet/test_two_node_direct.py.
Verifies that both nodes can be managed independently using a two_node_ci fleet.
"""

from __future__ import annotations

import pytest

from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.orchestrator.fleet import Fleet
from ci.software_only_v2.tier3_fleet.conftest import make_startdaq_params, requires_docker

pytestmark = pytest.mark.tier3


@requires_docker
@pytest.mark.parametrize(
    "pseti_workspace_session",
    [FleetSpec.two_node_ci(tier="tier3")],
    indirect=True,
)
class TestTwoNodeDirect:
    """Two DAQ nodes can be managed completely independently."""

    def test_nodes_start_and_stop_independently(self, session_fleet: Fleet) -> None:
        """Both nodes start, run independently, and one stop does not affect the other."""
        fleet = session_fleet

        client0 = fleet.daq_control_client(0)
        client1 = fleet.daq_control_client(1)

        # Prepare run directories inside each container
        fleet.exec_in_node(0, "mkdir -p /data/run0 && chmod 777 /data/run0")
        fleet.exec_in_node(1, "mkdir -p /data/run1 && chmod 777 /data/run1")

        from datetime import UTC, datetime

        from control.utils.pydantic_config_models import RunStateLedger, RunStatus
        from control.utils.run_state import RunStateManager
        
        mgr = RunStateManager()
        mgr.save_state(RunStateLedger(
            run_name="two_node_direct",
            status=RunStatus.STARTING,
            start_time=datetime.now(UTC).isoformat(),
            nodes=[]
        ))

        ok0 = client0.StartDaq(make_startdaq_params(fleet, 0, "run0"))
        assert ok0 is True

        ok1 = client1.StartDaq(make_startdaq_params(fleet, 1, "run1"))
        assert ok1 is True
        
        mgr.transition(RunStatus.ACTIVE)

        _, s0 = client0.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
        _, s1 = client1.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})

        assert s0["hashpipe_running"] is True
        assert s1["hashpipe_running"] is True

        # Stop node 0 only; node 1 must remain running
        client0.StopDaq({"data_dir": "/data", "run_dir": "run0"})
        mgr.transition(RunStatus.RECORDING_ENDED)
        _, s0_stopped = client0.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
        _, s1_still_running = client1.StatusDaq(
            {"data_dir": "/data", "check_hashpipe_running": True}
        )

        assert s0_stopped["hashpipe_running"] is False
        assert s1_still_running["hashpipe_running"] is True

        from ci.software_only_v2.infra.parity import run_scenario
        run_scenario(
            "two_node_independent_lifecycle",
            node_0_running=s0_stopped["hashpipe_running"],
            node_1_running=s1_still_running["hashpipe_running"],
        )
        run_scenario(
            "two_node_start_stop",
            probe=fleet.workspace.state_probe,
        )

        client0.close()
        client1.close()

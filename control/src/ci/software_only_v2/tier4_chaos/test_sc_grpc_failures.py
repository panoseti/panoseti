"""
tier4_chaos/test_sc_grpc_failures.py — gRPC failure isolation scenarios (SC*).

Ported from ci/software_only/tier4_chaos/test_sc_grpc_failures_1.py.
Verifies that the control plane correctly handles gRPC timeouts, unresponsiveness,
and partial failures using the v2 chaos tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import unittest.mock
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import anyio
import grpc

from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.infra.workspace import Workspace
from ci.software_only_v2.orchestrator.fleet import Fleet
from ci.software_only_v2.fixtures.chaos import Chaos

pytestmark = [pytest.mark.tier4, pytest.mark.tier3]


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
    [FleetSpec.two_node_ci(tier="tier4")],
    indirect=True,
)
class TestScGrpcFailures:
    """gRPC failure isolation tests (SC-001, SC-006)."""

    @pytest.mark.asyncio
    async def test_SC001_startdaq_timeout_triggers_rollback(
        self,
        session_fleet: Fleet,
        chaos: Chaos,
    ) -> None:
        """SC-001: StartDaq timeout must trigger a global rollback."""
        import control.start as start
        from control.utils import config_file
        
        fleet = session_fleet
        daq_config = fleet.live_daq_config
        obs_config = config_file.get_obs_config()
        quabo_uids = config_file.get_quabo_uids()
        data_config = config_file.get_data_config()
        network_config = config_file.get_network_config()

        # Inject timeout on the first node
        node0_pf = daq_config.daq_nodes[0].port_forwarding
        node0_host = str(node0_pf.gw_ip)
        node0_port = node0_pf.grpc_port

        from panoseti_grpc.daq_control.client import AsyncDaqControlClient
        
        real_factory = AsyncDaqControlClient

        def factory(host: str, port: int = 50051) -> Any:
            client = real_factory(host=host, port=port)
            if host == node0_host and port == node0_port:
                # Inject a long sleep to trigger timeout
                chaos.grpc.proxy(client).set_mode("StartDaq", "timeout", timeout_s=5.0).apply(client)
            return client

        with patch("control.start.AsyncDaqControlClient", side_effect=factory), \
             patch("control.start.ph_baseline_file_ok", return_value=True), \
             patch("control.start._check_daq_reachability"), \
             patch("control.start._check_quabo_reachability"), \
             patch("control.start.make_run_dirs"), \
             patch("control.start.start_data_flow"), \
             patch("control.start.util.is_hk_recorder_running", return_value=False), \
             patch("control.start.util.kill_hk_recorder"), \
             patch("control.start.util.stop_data_flow"), \
             patch("control.utils.util.local_ip", return_value=["127.0.0.1", str(daq_config.head_node_ip_addr)]):
            
            # We expect start_run to return False due to the timeout/rollback
            with anyio.fail_after(15):
                success = await start.start_run(
                    obs_config, daq_config, quabo_uids, data_config, network_config,
                    no_hv=True, no_redis=True, no_data=False, run_name="sc001_run"
                )
                assert not success, "start_run should have failed due to StartDaq timeout"

    @pytest.mark.asyncio
    async def test_SC006_stop_continues_after_per_node_failure(
        self,
        session_fleet: Fleet,
        chaos: Chaos,
    ) -> None:
        """SC-006: StopDaq failure on one node must NOT skip other nodes."""
        import control.stop as stop_module
        from control.utils import config_file
        
        fleet = session_fleet
        daq_config = fleet.live_daq_config
        
        node0_pf = daq_config.daq_nodes[0].port_forwarding
        node0_host = str(node0_pf.gw_ip)
        node0_port = node0_pf.grpc_port
        
        node1_pf = daq_config.daq_nodes[1].port_forwarding
        node1_host = str(node1_pf.gw_ip)
        node1_port = node1_pf.grpc_port

        from panoseti_grpc.daq_control.client import AsyncDaqControlClient
        real_factory = AsyncDaqControlClient
        
        stop_called_ips = set()

        def factory(host: str, port: int = 50051) -> Any:
            client = real_factory(host=host, port=port)
            
            original_stop = client.StopDaq
            async def tracked_stop(*args: Any, **kwargs: Any) -> Any:
                stop_called_ips.add(host)
                if host == node0_host and port == node0_port:
                    raise grpc.RpcError("Injected failure")
                return await original_stop(*args, **kwargs)
            
            client.StopDaq = tracked_stop
            return client

        with patch("control.stop.AsyncDaqControlClient", side_effect=factory), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)):

            # Call actual stop_recording
            await stop_module.stop_recording(daq_config, "sc006_run", verbose=False)

        # Node 1 must still have been attempted despite node 0 failure
        assert node1_host in stop_called_ips, (
            "Node 1 was never told to stop because node 0 failed first"
        )

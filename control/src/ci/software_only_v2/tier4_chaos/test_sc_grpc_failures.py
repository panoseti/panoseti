"""
tier4_chaos/test_sc_grpc_failures.py — gRPC failure isolation scenarios (SC*).

Ported from ci/software_only/tier4_chaos/test_sc_grpc_failures_1.py.
Verifies that the control plane correctly handles gRPC timeouts, unresponsiveness,
and partial failures using the v2 chaos tools.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import anyio
import grpc
import pytest

from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.orchestrator.fleet import Fleet
from ci.software_only_v2.fixtures.chaos import Chaos
from ci.software_only_v2.tier4_chaos.conftest import requires_docker

pytestmark = [pytest.mark.tier4, pytest.mark.tier3]


@requires_docker
@pytest.mark.parametrize(
    "pseti_workspace_session",
    [FleetSpec.two_node_ci(tier="tier4")],
    indirect=True,
)
class TestScGrpcFailures:
    """gRPC failure isolation tests."""

    @pytest.mark.asyncio
    async def test_when_startdaq_times_out_then_rollback_triggered(
        self,
        session_fleet: Fleet,
        chaos: Chaos,
    ) -> None:
        """SC-001: StartDaq timeout must trigger a global rollback.

        The factory intercepts the first node's client and applies a timeout
        fault so start_run fails and rolls back cleanly.
        """
        import control.start as start
        from control.utils import config_file

        fleet = session_fleet
        daq_config = fleet.live_daq_config
        obs_config = config_file.get_obs_config()
        quabo_uids = config_file.get_quabo_uids()
        data_config = config_file.get_data_config()
        network_config = config_file.get_network_config()

        node0_pf = daq_config.daq_nodes[0].port_forwarding
        node0_host = str(node0_pf.gw_ip)
        node0_port = node0_pf.grpc_port

        from panoseti_grpc.daq_control.client import AsyncDaqControlClient
        real_factory = AsyncDaqControlClient

        def factory(host: str, port: int = 50051) -> Any:
            client = real_factory(host=host, port=port)
            if host == node0_host and port == node0_port:
                proxy = chaos.grpc.proxy(client)
                proxy.set_mode("StartDaq", "timeout", timeout_s=5.0)
                proxy.apply(client)
            return client

        with (
            patch("control.start.AsyncDaqControlClient", side_effect=factory),
            patch("control.start.ph_baseline_file_ok", return_value=True),
            patch("control.start._check_daq_reachability"),
            patch("control.start._check_quabo_reachability"),
            patch("control.start.make_run_dirs"),
            patch("control.start.start_data_flow"),
            patch("control.start.util.is_hk_recorder_running", return_value=False),
            patch("control.start.util.kill_hk_recorder"),
            patch("control.start.util.stop_data_flow"),
            patch(
                "control.utils.util.local_ip",
                return_value=["127.0.0.1", str(daq_config.head_node_ip_addr)],
            ),
        ):
            with anyio.fail_after(15):
                success = await start.start_run(
                    obs_config, daq_config, quabo_uids, data_config, network_config,
                    no_hv=True, no_redis=True, no_data=False, run_name="sc001_run",
                )
                assert not success, "start_run should have failed due to StartDaq timeout"

    @pytest.mark.asyncio
    async def test_when_stop_fails_on_one_node_then_other_nodes_still_stopped(
        self,
        session_fleet: Fleet,
        chaos: Chaos,
    ) -> None:
        """SC-006: StopDaq failure on one node must NOT skip other nodes."""
        import control.stop as stop_module

        fleet = session_fleet
        daq_config = fleet.live_daq_config

        node0_pf = daq_config.daq_nodes[0].port_forwarding
        node0_host = str(node0_pf.gw_ip)
        node0_port = node0_pf.grpc_port

        node1_pf = daq_config.daq_nodes[1].port_forwarding
        node1_host = str(node1_pf.gw_ip)

        from panoseti_grpc.daq_control.client import AsyncDaqControlClient
        real_factory = AsyncDaqControlClient

        stop_called_ips: set[str] = set()

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
            await stop_module.stop_recording(daq_config, "sc006_run", verbose=False)

        assert node1_host in stop_called_ips, (
            "Node 1 was never told to stop because node 0 failed first"
        )

    def test_when_unavailable_injected_on_startdaq_then_client_raises(
        self,
        session_fleet: Fleet,
        chaos: Chaos,
    ) -> None:
        """SC-UNAVAIL: injecting UNAVAILABLE on StartDaq causes UnavailableError.

        Wires the grpc_inject_unavailable parity scenario.
        """
        from panoseti_grpc.grpc_utils.exceptions import from_rpc_error

        fleet = session_fleet
        client = fleet.daq_control_client(0)
        fleet.exec_in_node(0, "mkdir -p /data/unavail_run && chmod 777 /data/unavail_run")

        node_cfg = fleet.live_daq_config.daq_nodes[0]
        start_params = {
            "data_dir": "/data",
            "run_dir": "unavail_run",
            "module_id": list(node_cfg.module_ids),
            "daq_ip_addr": str(node_cfg.ip_addr),
            "bindhost": node_cfg.bindhost or "lo",
            "max_file_size_mb": 1024.0,
            "group_ph_frames": False,
            "obs": "engineering",
            "force": False,
        }

        try:
            with chaos.grpc.inject(client, "StartDaq", "unavailable"):
                client.StartDaq(start_params)
        except Exception as exc:
            # The chaos proxy raises a raw grpc.RpcError; map it to a typed exception.
            if isinstance(exc, grpc.RpcError):
                fleet._last_grpc_exc = from_rpc_error(exc, "")  # type: ignore[attr-defined]
            else:
                fleet._last_grpc_exc = exc  # type: ignore[attr-defined]

        from ci.software_only_v2.infra.parity import run_scenario
        run_scenario("grpc_inject_unavailable", fleet=fleet, node_index=0)
        client.close()

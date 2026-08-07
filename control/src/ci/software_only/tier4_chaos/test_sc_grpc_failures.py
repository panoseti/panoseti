"""
tier4_chaos/test_sc_grpc_failures.py — gRPC failure isolation scenarios (SC*).

Ported from ci/software_only/tier4_chaos/test_sc_grpc_failures_1.py.
Verifies that the control plane correctly handles gRPC timeouts, unresponsiveness,
and partial failures using the v2 chaos tools.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import anyio
import grpc
import pytest

from ci.software_only.fixtures.chaos import Chaos
from ci.software_only.infra.spec import FleetSpec
from ci.software_only.orchestrator.fleet import Fleet
from ci.software_only.tier4_chaos.conftest import requires_docker

pytestmark = [pytest.mark.tier4, pytest.mark.tier3]


@requires_docker
@pytest.mark.parametrize(
    "pseti_workspace_session",
    [FleetSpec.two_node_ci(tier="tier4")],
    indirect=True,
)
@pytest.mark.timeout(120)
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

        from ci.fixtures.adapters.fake_adapters import FakeFileSystemManager, FakeProcessManager
        from control.adapters.real_adapters import RealNetworkClient

        class ChaosNetworkClient(RealNetworkClient):
            async def start_daq_node(self, node: Any, params: dict[str, Any], timeout_s: float = 10.0) -> bool:
                if str(node.ip_addr) == str(daq_config.daq_nodes[0].ip_addr):
                    raise grpc.RpcError("Injected timeout")
                return await super().start_daq_node(node, params, timeout_s)

        net_client = ChaosNetworkClient(daq_config)
        process_mgr = FakeProcessManager()
        fs_mgr = FakeFileSystemManager()

        with (
            patch("control.start.ph_baseline_file_ok", return_value=True),
            patch("control.start._check_quabo_reachability"),
            patch("control.start.start_data_flow"),
            patch(
                "control.utils.util.local_ip",
                return_value=["127.0.0.1", str(daq_config.head_node_ip_addr)],
            ),
        ):
            with anyio.fail_after(15):
                success = await start.start_run(
                    obs_config, daq_config, quabo_uids, data_config, network_config,
                    no_hv=True, no_redis=True, no_data=False, run_name="sc001_run",
                    process_mgr=process_mgr, net_client=net_client, fs_mgr=fs_mgr
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
        from control.utils import config_file

        fleet = session_fleet
        daq_config = fleet.live_daq_config
        network_config = config_file.get_network_config()
        quabo_uids = config_file.get_quabo_uids()

        stop_called_ips: set[str] = set()

        from ci.fixtures.adapters.fake_adapters import FakeFileSystemManager, FakeProcessManager
        from control.adapters.real_adapters import RealNetworkClient

        class ChaosNetworkClient(RealNetworkClient):
            async def stop_daq_node(self, node: Any, timeout_s: float = 20.0, retries: int = 2) -> bool:
                ip = str(node.ip_addr)
                stop_called_ips.add(ip)
                if ip == str(daq_config.daq_nodes[0].ip_addr):
                    raise grpc.RpcError("Injected failure")
                return await super().stop_daq_node(node, timeout_s, retries)

        net_client = ChaosNetworkClient(daq_config)
        process_mgr = FakeProcessManager()
        fs_mgr = FakeFileSystemManager()

        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            await stop_module.stop_run(
                daq_config, network_config, quabo_uids,
                process_mgr=process_mgr, net_client=net_client, fs_mgr=fs_mgr,
                run="sc006_run"
            )

        node1_host = str(daq_config.daq_nodes[1].ip_addr)
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

        from ci.software_only.infra.parity import run_scenario
        run_scenario("grpc_inject_unavailable", fleet=fleet, node_index=0)
        client.close()

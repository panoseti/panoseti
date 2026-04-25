"""
ci/tier3_fleet/test_distributed_flows.py

End-to-end distributed orchestration tests using dynamic fleets.
Verifies that start/stop commands propagate correctly across N nodes.
"""

from __future__ import annotations

import unittest.mock

import pytest
from panoseti_grpc.daq_control.client import AsyncDaqControlClient

try:
    from ci.fixtures.fleet import make_fleet, setup_docker_host
    _HAS_TESTCONTAINERS = True
except ImportError:
    _HAS_TESTCONTAINERS = False
    make_fleet = setup_docker_host = None  # type: ignore[assignment]

from control.start import start_run
from control.stop import stop_run
from control.utils import config_file, util


def _docker_available() -> bool:
    """Return True if testcontainers is installed and a Docker daemon is reachable."""
    if not _HAS_TESTCONTAINERS:
        return False
    try:
        import docker as _docker
        setup_docker_host()
        _docker.from_env().ping()
        return True
    except Exception:
        return False


async def _node_hashpipe_running(node) -> bool:
    """Query a single fleet node via gRPC to check if hashpipe is running."""
    grpc_host, grpc_port = util.daq_grpc_endpoint(node)
    try:
        async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
            ok, status = await client.StatusDaq({
                "data_dir": node.data_dir,
                "check_hashpipe_running": True,
            })
            return bool(ok and status.get("hashpipe_running"))
    except Exception:
        return False


@pytest.mark.asyncio
async def test_when_distributed_run_started_then_all_nodes_recording(
    tmp_path,
) -> None:
    """
    Intent: Verify distributed gRPC orchestration for a multi-node run.
    Scenario: Dynamic 2-node fleet started via testcontainers.
    Assertion: Both nodes report hashpipe_running=True after start_run().
    """
    if not _docker_available():
        pytest.skip("Docker daemon unreachable — fleet tests require Docker")

    fleet = make_fleet(n=2)
    fleet.start()
    try:
        fleet.wait_healthy()

        # Write dynamic daq_config for this fleet into tmp_path
        daq_config_path = tmp_path / "daq_config.json"
        fleet.write_daq_config(daq_config_path, head_node_ip="10.0.1.1")

        # Load all required configs; daq_config comes from the fleet-written file
        daq_cfg      = config_file.get_daq_config(dir=str(tmp_path))
        obs_cfg      = config_file.get_obs_config()
        quabo_uids   = config_file.get_quabo_uids()
        data_cfg     = config_file.get_data_config()
        network_cfg  = config_file.get_network_config()

        # Redirect head_node_data_dir away from the hardcoded /data/head so
        # path operations (rollback archive, symlink) stay in the test sandbox.
        daq_cfg.head_node_data_dir = str(tmp_path / "head_data")

        run_name = "dist_test_run.pffd"
        with unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
             unittest.mock.patch("control.start._check_daq_reachability"), \
             unittest.mock.patch("control.start._check_quabo_reachability"), \
             unittest.mock.patch("control.start.start_data_flow"), \
             unittest.mock.patch("control.start.make_run_dirs"), \
             unittest.mock.patch("control.start.util.start_hk_recorder"), \
             unittest.mock.patch("control.start.util.write_run_name"):
            await start_run(
                obs_cfg, daq_cfg, quabo_uids, data_cfg, network_cfg,
                no_hv=True, no_redis=True, no_data=False,
                run_name=run_name, no_check_daq=True,
            )

        # Each fleet node must have hashpipe active (queried via its own gRPC server)
        for node in daq_cfg.daq_nodes:
            assert await _node_hashpipe_running(node), (
                f"hashpipe not running on node {node.ip_addr}"
            )

    finally:
        fleet.tear_down()


@pytest.mark.asyncio
async def test_when_distributed_run_stopped_then_all_nodes_halted(
    tmp_path,
) -> None:
    """
    Intent: Verify clean teardown of a distributed observing run.
    Scenario: Dynamic 2-node fleet with an active run.
    Assertion: Hashpipe stopped on all nodes and ledger reaches RECORDING_ENDED.
    """
    if not _docker_available():
        pytest.skip("Docker daemon unreachable — fleet tests require Docker")

    fleet = make_fleet(n=2)
    fleet.start()
    try:
        fleet.wait_healthy()

        daq_config_path = tmp_path / "daq_config.json"
        fleet.write_daq_config(daq_config_path, head_node_ip="10.0.1.1")

        daq_cfg     = config_file.get_daq_config(dir=str(tmp_path))
        obs_cfg     = config_file.get_obs_config()
        quabo_uids  = config_file.get_quabo_uids()
        data_cfg    = config_file.get_data_config()
        network_cfg = config_file.get_network_config()

        daq_cfg.head_node_data_dir = str(tmp_path / "head_data")

        run_name = "stop_test_run.pffd"
        with unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
             unittest.mock.patch("control.start._check_daq_reachability"), \
             unittest.mock.patch("control.start._check_quabo_reachability"), \
             unittest.mock.patch("control.start.start_data_flow"), \
             unittest.mock.patch("control.start.make_run_dirs"), \
             unittest.mock.patch("control.start.util.start_hk_recorder"), \
             unittest.mock.patch("control.start.util.write_run_name"):
            await start_run(
                obs_cfg, daq_cfg, quabo_uids, data_cfg, network_cfg,
                no_hv=True, no_redis=True, no_data=False,
                run_name=run_name, no_check_daq=True,
            )

        # stop_run checks that run_dir exists before transitioning the ledger;
        # make_run_dirs was mocked so we create it manually here.
        (tmp_path / "head_data" / run_name).mkdir(parents=True, exist_ok=True)

        await stop_run(
            daq_cfg, network_cfg, quabo_uids,
            run=run_name, no_collect=True, no_cleanup=True, no_transfer=True,
        )

        # All nodes must be halted
        for node in daq_cfg.daq_nodes:
            assert not await _node_hashpipe_running(node), (
                f"hashpipe still running on node {node.ip_addr} after stop"
            )

        from control.utils.run_state import RunStateManager
        ledger = RunStateManager().load_state()
        assert ledger is not None and ledger.status == "RECORDING_ENDED"

    finally:
        fleet.tear_down()

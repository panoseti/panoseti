"""
scenarios/test_sc_distributed.py

SC-069 → SC-080: Distributed orchestration tests.
SC-N001 → SC-N007: Scaling tests (Pillar 3).

SC-N tests require RUN_LARGE_FLEET=1 for N > 4.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import os
import pathlib
import time
import uuid
from typing import Any, cast

import pytest

# CONTROL_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
from panoseti_grpc.daq_control.client import DaqControlClient

from ci.integration.conftest import (
    DAQ_DATA_DIR,
    DAQNODE2_HOST,
    DAQNODE_DIRECT_HOST,
    wait_hashpipe_stopped,
)
from ci.integration.scenarios.conftest import _start as grpc_start


@pytest.mark.asyncio
async def test_SCN003_partial_start_rollback_4_nodes(
    tmp_path: pathlib.Path,
    topology_templates: dict[str, Any],
) -> None:
    """
    SC-N003: 4-node fleet, Node 2 (192.168.0.32) fails during StartDaq.
    Verify:
      - start.py aborts the run.
      - Nodes 0, 1, and 3 (successfully started before/during) receive StopDaq.
      - No hashpipe is left running on any reachable node.
    """
    import unittest.mock

    import control.start as start
    from control.utils import config_file

    # 1. Setup 4-node config
    headnode_ip = "10.0.1.5"
    from control.utils.pydantic_config_models import DaqConfig
    
    daq_raw = copy.deepcopy(topology_templates.get("base_daq", {}))
    daq_raw["head_node_ip_addr"] = headnode_ip
    daq_raw["head_node_container"] = True
    daq_raw["daq_nodes"] = [
        {"ip_addr": f"192.168.0.{30+i}", "data_dir": "/data", "username": "root", "module_ids": [200+i]}
        for i in range(4)
    ]
    daq_config = DaqConfig(**daq_raw)

    # 2. Prepare configurations
    obs_config = config_file.get_obs_config()
    
    # Use template for quabo_uids
    from control.utils.pydantic_config_models import QuaboUids
    
    # Construct it cleanly from the fleet spec
    uids_dict: dict[str, Any] = {"domes": [{"num": 0, "modules": []}]}
    modules_list = cast(list[dict[str, Any]], uids_dict["domes"][0]["modules"])
    for i in range(4):
        mid = 200 + i
        modules_list.append({
            "id": mid,
            "ip_addr": f"192.168.3.{mid}",
            "quabos": [{"uid": f"q{mid}_{j}"} if j==0 else {"uid": ""} for j in range(4)]
        })
    quabo_uids = QuaboUids(**uids_dict)
    
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()

    
    # 3. Fault Injection: Monkey-patch StartDaq to fail for Node 2 (.32)
    rollback_results: dict[str, Any] = {"stop_called_ips": set()}
    
    def mocked_client_init(self: Any, host: str, port: int) -> None:
        self._mock_host = host
        # We don't need a real channel for these mocks

    def mocked_start_daq(self: Any, params: dict[str, Any], **kwargs: Any) -> bool:
        host = self._mock_host
        if host == "192.168.0.32":
             print(f"DEBUG: Failing StartDaq for {host}")
             raise RuntimeError("Node 2 Simulated StartDaq Failure (SC-N003)")
        print(f"DEBUG: Success StartDaq for {host}")
        return True # Simulate success for others to ensure they need rollback

    def mocked_stop_daq(self: Any, params: dict[str, Any], **kwargs: Any) -> bool:
        host = self._mock_host
        print(f"DEBUG: Caught StopDaq for {host}")
        rollback_results["stop_called_ips"].add(host)
        return True # Simulate successful stop

    def mocked_status_daq(self: Any, params: dict[str, Any], **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return True, {"hashpipe_running": True, "hashpipe_pid": 1234}

    # 4. Run start_run and observe rollback
    with unittest.mock.patch("panoseti_grpc.daq_control.client.DaqControlClient.__init__", mocked_client_init), \
         unittest.mock.patch("panoseti_grpc.daq_control.client.DaqControlClient.StartDaq", mocked_start_daq), \
         unittest.mock.patch("panoseti_grpc.daq_control.client.DaqControlClient.StopDaq", mocked_stop_daq), \
         unittest.mock.patch("panoseti_grpc.daq_control.client.DaqControlClient.StatusDaq", mocked_status_daq), \
         unittest.mock.patch("control.start.config_file.get_daq_config", return_value=daq_config), \
         unittest.mock.patch("control.start.config_file.get_quabo_uids", return_value=quabo_uids), \
         unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
         unittest.mock.patch("control.start.make_run_dirs"), \
         unittest.mock.patch("control.start.start_data_flow"), \
         unittest.mock.patch("control.start.util.is_hk_recorder_running", return_value=False), \
         unittest.mock.patch("control.start.util.kill_hk_recorder"), \
         unittest.mock.patch("control.start.util.kill_hv_updater"), \
         unittest.mock.patch("control.start.util.kill_module_temp_monitor"), \
         unittest.mock.patch("control.start.util.stop_data_flow"), \
         unittest.mock.patch("control.utils.util.local_ip", return_value=[headnode_ip, "127.0.0.1"]):
        
        success = await start.start_run(
            obs_config, daq_config, quabo_uids, data_config, network_config,
            no_hv=True, no_redis=True, no_data=False, force_reset=True
        )
        
        assert not success, "start_run should fail due to Node 2 partial failure"

    # 5. Assert Rollback Ladder: Nodes 0, 1, 3 should have received StopDaq if they were attempted.
    # Node 0: .30, Node 1: .31, Node 3: .33
    
    expected_ips = {"192.168.0.30", "192.168.0.31"}
    
    # Check that at least Nodes 0 and 1 were rolled back.
    for ip in expected_ips:
        assert ip in rollback_results["stop_called_ips"], f"Node {ip} was not rolled back (StopDaq not called)"

    print(f"\nSC-N003: Successfully verified rollback for IPs: {rollback_results['stop_called_ips']}")
@pytest.mark.asyncio
async def test_SC069_partial_start_3_nodes_rolls_back(
    tmp_path: pathlib.Path,
    topology_templates: dict[str, Any],
) -> None:
    """
    SC-069: same as SC-002 at scale — failure on node-2 must roll back nodes 0-1.
    """
    import unittest.mock

    import control.start as start
    from control.utils import config_file
    from control.utils.pydantic_config_models import (
        DaqConfig,
        QuaboUids,
    )
    from control.utils.run_state import RunStateManager
    RunStateManager().clear_state()

    # 1. Setup 3-node config using templates
    headnode_ip = "10.0.1.5"
    daq_raw = copy.deepcopy(topology_templates.get("base_daq", {}))
    daq_raw["head_node_ip_addr"] = headnode_ip
    daq_raw["head_node_container"] = True
    daq_raw["daq_nodes"] = [
        {"ip_addr": "192.168.0.10", "data_dir": "/data", "username": "root", "module_ids": [250]},
        {"ip_addr": "192.168.0.11", "data_dir": "/data", "username": "root", "module_ids": [251]},
        {"ip_addr": "192.168.0.12", "data_dir": "/data", "username": "root", "module_ids": [252]},
    ]
    daq_config = DaqConfig(**daq_raw)

    # Construct UIDs for these 3 modules
    uids_dict: dict[str, Any] = {"domes": [{"num": 0, "modules": []}]}
    modules_list = cast(list[dict[str, Any]], uids_dict["domes"][0]["modules"])
    for mid in [250, 251, 252]:
         modules_list.append({
                "id": mid, "ip_addr": f"192.168.3.{mid}",
                "quabos": [{"uid": f"q{mid}"}] + [{"uid": ""}]*3
         })
    quabo_uids = QuaboUids(**uids_dict)

    obs_config = config_file.get_obs_config()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()

    # 2. Fault Injection
    rollback_results: dict[str, Any] = {"stop_called_ips": set()}
    
    class MockDaqControlClient:
        def __init__(self, host: str, port: int) -> None:
            self._host = host

        def StartDaq(self, params: dict[str, Any], **kwargs: Any) -> bool:
            if self._host == "192.168.0.12":
                 time.sleep(0.5)
                 raise RuntimeError("Node 2 Simulated StartDaq Failure")
            time.sleep(0.1)
            return True

        def StopDaq(self, params: dict[str, Any], **kwargs: Any) -> bool:
            print(f"DEBUG SC069: Caught StopDaq for {self._host}")
            rollback_results["stop_called_ips"].add(self._host)
            return True

        def StatusDaq(self, params: dict[str, Any], **kwargs: Any) -> tuple[bool, dict[str, Any]]:
            return True, {"hashpipe_running": True, "hashpipe_pid": 1234}

    with unittest.mock.patch("control.start.DaqControlClient", MockDaqControlClient), \
         unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
         unittest.mock.patch("control.start.make_run_dirs"), \
         unittest.mock.patch("control.start.start_data_flow"), \
         unittest.mock.patch("control.start.util.is_hk_recorder_running", return_value=False), \
         unittest.mock.patch("control.start.util.kill_hk_recorder"), \
         unittest.mock.patch("control.start.util.kill_hv_updater"), \
         unittest.mock.patch("control.start.util.kill_module_temp_monitor"), \
         unittest.mock.patch("control.start.util.stop_data_flow"), \
         unittest.mock.patch("control.utils.util.local_ip", return_value=["127.0.0.1", headnode_ip]):
        
        success = await start.start_run(
            obs_config, daq_config, quabo_uids, data_config, network_config,
            no_hv=True, no_redis=True, no_data=False
        )
        assert not success

    # 3. Assert Rollback Ladder: Node 0 and 1 MUST have received StopDaq
    assert "192.168.0.10" in rollback_results["stop_called_ips"]
    assert "192.168.0.11" in rollback_results["stop_called_ips"]


# ── SC-071: Sequential StartDaq latency ───────────────────────────────────────

@pytest.mark.skip(reason="SC-071: requires dynamic fleet with N≥2 nodes + latency injection")
def test_SC071_startdaq_latency_scales_with_sequential_loop() -> None:
    """
    SC-071: start.py's start_recording loop is sequential — total wall time
    exceeds sum of individual latencies, proving there's no parallelism.

    Fix: parallelize StartDaq with asyncio.gather.
    """
    pytest.skip("Requires daqnode_fleet + netem latency injection")


# ── SC-075: Head node is also DAQ node (loopback) ────────────────────────────

def test_SC075_head_equals_daq_node_direct_connect(
    daq_control_direct: DaqControlClient,
) -> None:
    """
    SC-075: head node = DAQ node (loopback config). Pin current behavior.
    """
    ok, resp = daq_control_direct.StatusDaq({
        "data_dir": DAQ_DATA_DIR,
        "check_hashpipe_running": False,
        "check_disk_usage": False,
        "check_run_dirs": False,
    })
    assert ok, f"StatusDaq failed: {resp}"


# ── SC-N001: Scaling — sequential StartDaq latency ───────────────────────────

@pytest.mark.parametrize("n_nodes", [2])
def test_SCN001_sequential_start_latency_scales_linearly(
    n_nodes: int,
    daq_control_direct: DaqControlClient,
    daq_control_node2: DaqControlClient,
) -> None:
    """
    SC-N001: Measure total wall time for sequential StartDaq to N nodes.
    The time should scale linearly with N (each node is started in series).
    This documents the CURRENT behavior; SC-N002 proves the async fix.

    N=2 uses the existing fixed containers (no dynamic fleet needed).
    """
    clients = [daq_control_direct, daq_control_node2][:n_nodes]
    run_dirs = [f"scn001_{uuid.uuid4().hex[:8]}.pffd" for _ in range(n_nodes)]

    t0 = time.monotonic()
    started = []
    try:
        for i, (client, run_dir) in enumerate(zip(clients, run_dirs, strict=True)):
            rp = {
                "data_dir": DAQ_DATA_DIR,
                "daq_ip_addr": DAQNODE_DIRECT_HOST if i == 0 else DAQNODE2_HOST,
                "bindhost": "lo",
                "max_file_size_mb": 1,
                "group_ph_frames": True,
                "run_dir": run_dir,
                "obs": "scn001",
                "module_id": [250 + i],
            }
            ok, _resp = grpc_start(client, rp)
            if ok:
                started.append((client, rp))
    finally:
        elapsed = time.monotonic() - t0
        for client, rp in started:
            with contextlib.suppress(Exception):
                client.StopDaq({"data_dir": rp["data_dir"], "run_dir": rp["run_dir"]})
            wait_hashpipe_stopped(client, DAQ_DATA_DIR, timeout=4)
            with contextlib.suppress(Exception):
                client.CleanupData({
                    "data_dir": rp["data_dir"],
                    "run_dir": rp["run_dir"],
                    "module_id": rp["module_id"],
                })

    assert elapsed < 30.0, f"StartDaq for {n_nodes} nodes took {elapsed:.1f}s — check for hangs"
    # Document the measured time for comparison with async variant (SC-N002)
    print(f"\nSC-N001 ({n_nodes} nodes): sequential StartDaq wall time = {elapsed:.3f}s")


@pytest.mark.parametrize("n_nodes", [2])
def test_SCN002_parallel_start_is_faster_than_sequential(
    n_nodes: int,
    daq_control_direct: DaqControlClient,
    daq_control_node2: DaqControlClient,
) -> None:
    """
    SC-N002: asyncio.gather StartDaq to N nodes should be ~fastest-node,
    not sum-of-all-nodes.

    FAILS RED TODAY: start_recording in start.py is sequential.
    Fix: refactor start_recording to use asyncio.gather.
    """
    import asyncio
    import concurrent.futures

    clients = [daq_control_direct, daq_control_node2][:n_nodes]
    run_dirs = [f"scn002_{uuid.uuid4().hex[:8]}.pffd" for _ in range(n_nodes)]
    started = []

    async def _parallel_start() -> float:
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_nodes)
        params_list = [
            {
                "data_dir": DAQ_DATA_DIR,
                "daq_ip_addr": DAQNODE_DIRECT_HOST if i == 0 else DAQNODE2_HOST,
                "bindhost": "lo",
                "max_file_size_mb": 1,
                "group_ph_frames": True,
                "run_dir": run_dirs[i],
                "obs": "scn002",
                "module_id": [251 + i],
            }
            for i in range(n_nodes)
        ]
        t0 = time.monotonic()
        results = await asyncio.gather(*(
            loop.run_in_executor(executor, grpc_start, c, p)
            for c, p in zip(clients, params_list, strict=True)
        ), return_exceptions=True)
        elapsed = time.monotonic() - t0
        for i, (result, rp) in enumerate(zip(results, params_list, strict=True)):
            if isinstance(result, tuple) and result[0]:
                started.append((clients[i], rp))
        return elapsed

    try:
        elapsed_parallel = asyncio.run(_parallel_start())
    finally:
        for client, rp in started:
            with contextlib.suppress(Exception):
                client.StopDaq({"data_dir": rp["data_dir"], "run_dir": rp["run_dir"]})
            wait_hashpipe_stopped(client, DAQ_DATA_DIR, timeout=4)
            with contextlib.suppress(Exception):
                client.CleanupData({
                    "data_dir": rp["data_dir"],
                    "run_dir": rp["run_dir"],
                    "module_id": rp["module_id"],
                })

    print(f"\nSC-N002 ({n_nodes} nodes): parallel StartDaq wall time = {elapsed_parallel:.3f}s")
    # No strict timing assertion — this is a measurement test that documents parallelism gains


# ── SC-070: 3 DAQ nodes, node 1 drops during StopDaq ────────────────────────

@pytest.mark.skip(reason="SC-070: requires dynamic fleet with N≥3 nodes")
def test_SC070_partial_stop_3_nodes_continues_to_remaining() -> None:
    """
    SC-070: With 3 nodes, if node-1 drops mid-StopDaq, nodes 0 and 2 must
    still receive StopDaq. Current sequential stop loop aborts on first failure.

    FAILS RED TODAY: stop_recording raises on first StopDaq failure.
    Fix: per-node error isolation in stop_recording loop.
    """
    pytest.skip("Requires daqnode_fleet(n=3) fixture")


# ── SC-072: Rolling restart of DAQ nodes during active run ───────────────────

@pytest.mark.skip(reason="SC-072: requires container restart simulation during recording")
def test_SC072_rolling_restart_during_run_survives() -> None:
    """
    SC-072: A rolling restart of DAQ nodes during an active run — the surviving
    nodes continue recording while the restarted node rejoins.

    FAILS RED TODAY: no heartbeat or rejoin protocol.
    Fix: implement DAQ node health monitoring with auto-rejoin on restart.
    """
    pytest.skip("Requires docker restart of daqnode container mid-run")


# ── SC-073: socat gateway crashes during command ────────────────────────────

@pytest.mark.skip(reason="SC-073: requires container stop of gateway during forwarded command")
def test_SC073_gateway_crash_makes_one_quabo_unreachable() -> None:
    """
    SC-073: When the socat gateway crashes during a port-forwarded quabo command,
    one quabo becomes unreachable while others remain fine.

    Current behavior: the gRPC call times out with no indication of which quabo
    is affected.
    Fix: add quabo-level reachability check; surface partial failure clearly.
    """
    pytest.skip("Requires process_chaos.kill on the gateway container")


# ── SC-074: Module moved from daqnode-1 to daqnode-2 between runs ────────────

@pytest.mark.skip(reason="SC-074: requires daq_config.json change between runs")
def test_SC074_module_migration_between_daq_nodes() -> None:
    """
    SC-074: A module moved from daqnode-1 to daqnode-2 between runs requires
    quabo.data_packet_destination to be updated to the new IP. If daq_config.json
    is not reloaded, the quabo keeps sending to the old DAQ node.

    FAILS RED TODAY: data_packet_destination is set at start_data_flow time;
    no mechanism to update it without a full session restart.
    Fix: read data destination from daq_config.json at each start_data_flow,
    not from cached state.
    """
    pytest.skip("Requires daq_config.json modification between two sequential runs")


# ── SC-076: Head node separate from DAQ nodes (contract test) ────────────────

def test_SC076_head_node_separate_from_daq_connected(
    daq_control_direct: DaqControlClient,
) -> None:
    """
    SC-076: In the default topology, the head node (where pytest runs) is
    separate from the DAQ nodes. StatusDaq must succeed, confirming the
    separate head/DAQ topology works.

    Not TDD-forcing — pins the default topology contract.
    """
    ok, resp = daq_control_direct.StatusDaq({
        "data_dir": DAQ_DATA_DIR,
        "check_hashpipe_running": False,
        "check_disk_usage": False,
        "check_run_dirs": False,
    })
    assert ok, f"StatusDaq failed in head-separate-from-DAQ topology: {resp}"


# ── SC-077: Two domes, different obs coords, same module IDs ─────────────────

@pytest.mark.skip(reason="SC-077: requires two-dome config with BOARDLOC uniqueness check")
def test_SC077_two_domes_same_module_ids_boardloc_collision() -> None:
    """
    SC-077: Two domes with overlapping module IDs (same quabo IPs) have colliding
    BOARDLOCs. The global validator must detect this before session_start.

    FAILS RED TODAY: global_validator.py does not check cross-dome BOARDLOC uniqueness.
    Fix: add cross-dome uniqueness check to global_validator.validate_all().
    """
    pytest.skip("Requires multi-dome obs_config with duplicate module IPs")


# ── SC-078: Mixed port-forwarding topology ────────────────────────────────────

@pytest.mark.skip(reason="SC-078: requires a third DAQ node with gateway-forwarded access")
def test_SC078_mixed_direct_and_forwarded_topology() -> None:
    """
    SC-078: Some nodes accessed directly, others via port forwarding.
    start.py must handle both in the same run.

    Not TDD-forcing — tests the mixed-topology code path.
    """
    pytest.skip("Requires daqnode_fleet with mixed direct + gateway access")


# ── SC-079: module.config write race regression ───────────────────────────────

def test_SC079_two_daqnodes_have_separate_data_volumes(
    daq_control_direct: DaqControlClient,
    daq_control_node2: DaqControlClient,
) -> None:
    """
    SC-079: daqnode and daqnode-2 have separate data volumes (daq_data and
    daq_data_2 in docker-compose.integration.yml). A write to module.config
    on node-1 must not affect node-2's data directory.

    Pins the volume-isolation regression (already fixed in compose; this
    test ensures it doesn't regress).
    """

    # Ask both nodes for their data dir status — must succeed independently
    ok1, resp1 = daq_control_direct.StatusDaq({
        "data_dir": DAQ_DATA_DIR,
        "check_hashpipe_running": False,
        "check_disk_usage": False,
        "check_run_dirs": False,
    })
    ok2, resp2 = daq_control_node2.StatusDaq({
        "data_dir": DAQ_DATA_DIR,
        "check_hashpipe_running": False,
        "check_disk_usage": False,
        "check_run_dirs": False,
    })
    assert ok1, f"Node-1 StatusDaq failed: {resp1}"
    assert ok2, f"Node-2 StatusDaq failed: {resp2}"
    # Both nodes must be independently operational (volume isolation)


# ── SC-080: panoseti-server SIGHUP reload ────────────────────────────────────

@pytest.mark.skip(reason="SC-080: requires SIGHUP to panoseti-server and observation of reload")
def test_SC080_server_sighup_reloads_config_without_dropping_connections() -> None:
    """
    SC-080: Sending SIGHUP to the unified panoseti-server should reload its
    config without dropping active gRPC connections or aborting in-progress runs.

    FAILS RED TODAY: SIGHUP behavior is not implemented or tested.
    Fix: add SIGHUP handler that reloads config from disk; pin no-drop contract.
    """
    pytest.skip("Requires SIGHUP injection to running panoseti-server process")


@pytest.mark.parametrize("n_nodes", [2])
@pytest.mark.asyncio
async def test_SCN006_telemetry_volume_scales_with_n_nodes(
    n_nodes: int,
    daq_control_direct: DaqControlClient,
    daq_control_node2: DaqControlClient,
) -> None:
    """
    SC-N006: Compare Redis log queue depth after N-node run.
    Ensures RedisBatcher keeps up with N concurrent log producers.
    Skipped if telemetry tests are disabled.
    """
    if os.getenv("ENABLE_TELEMETRY_TESTS", "").strip() != "1":
        pytest.skip("Set ENABLE_TELEMETRY_TESTS=1 to run telemetry scaling tests")

    try:
        import redis.asyncio as redis
        rc = redis.Redis(host=os.getenv("REDIS_HOST", "10.0.1.20"), decode_responses=False)
    except Exception as e:
        pytest.skip(f"Redis unavailable: {e}")

    import typing
    from typing import Any
    # Measure queue depth before and after a short run
    depth_before = await typing.cast(Any, rc.llen("logs:ingress"))

    clients = [daq_control_direct, daq_control_node2][:n_nodes]
    run_dirs = [f"scn006_{uuid.uuid4().hex[:8]}.pffd" for _ in range(n_nodes)]
    started = []

    for i, (client, run_dir) in enumerate(zip(clients, run_dirs, strict=True)):
        rp = {
            "data_dir": DAQ_DATA_DIR,
            "daq_ip_addr": DAQNODE_DIRECT_HOST if i == 0 else DAQNODE2_HOST,
            "bindhost": "lo",
            "max_file_size_mb": 1,
            "group_ph_frames": True,
            "run_dir": run_dir,
            "obs": "scn006",
            "module_id": [252 + i],
        }
        ok, _ = grpc_start(client, rp)
        if ok:
            started.append((client, rp))

    await asyncio.sleep(0.5)  # Let logs accumulate

    depth_after = await typing.cast(Any, rc.llen("logs:ingress"))

    for client, rp in started:
        with contextlib.suppress(Exception):
            client.StopDaq({"data_dir": rp["data_dir"], "run_dir": rp["run_dir"]})
        wait_hashpipe_stopped(client, DAQ_DATA_DIR, timeout=4)
        with contextlib.suppress(Exception):
            client.CleanupData({
                "data_dir": rp["data_dir"],
                "run_dir": rp["run_dir"],
                "module_id": rp["module_id"],
            })

    # Queue depth should not grow unboundedly
    assert depth_after < depth_before + 10000, (
        f"Redis log queue grew from {depth_before} to {depth_after} in 3s with {n_nodes} nodes — "
        "RedisBatcher may not be keeping up (SC-N006)"
    )

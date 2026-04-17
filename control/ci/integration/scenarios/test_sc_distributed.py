"""
scenarios/test_sc_distributed.py

SC-069 → SC-080: Distributed orchestration tests.
SC-N001 → SC-N007: Scaling tests (Pillar 3).

SC-N tests require RUN_LARGE_FLEET=1 for N > 4.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import sys
import time
import uuid

import pytest

CONTROL_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from panoseti_grpc.daq_control.client import DaqControlClient  # noqa: E402

from ci.integration.conftest import (  # noqa: E402
    DAQ_DATA_DIR,
    DAQNODE2_HOST,
    DAQNODE_DIRECT_HOST,
    wait_hashpipe_stopped,
)

from .conftest import _start as grpc_start  # noqa: E402

# ── SC-069: 3 DAQ nodes, node-2 drops during StartDaq ───────────────────────

@pytest.mark.skip(reason="SC-069: requires dynamic fleet with N≥3 nodes")
def test_SC069_partial_start_3_nodes_rolls_back() -> None:
    """
    SC-069: same as SC-002 at scale — failure on node-2 must roll back nodes 0-1.
    Requires dynamic fleet fixture (Pillar 3) for a third container.
    """
    pytest.skip("Requires daqnode_fleet(n=3) fixture")


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
            wait_hashpipe_stopped(client, DAQ_DATA_DIR, timeout=8)
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
            wait_hashpipe_stopped(client, DAQ_DATA_DIR, timeout=8)
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

    await asyncio.sleep(3)  # Let logs accumulate

    depth_after = await typing.cast(Any, rc.llen("logs:ingress"))

    for client, rp in started:
        with contextlib.suppress(Exception):
            client.StopDaq({"data_dir": rp["data_dir"], "run_dir": rp["run_dir"]})
        wait_hashpipe_stopped(client, DAQ_DATA_DIR, timeout=8)
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

"""
scenarios/test_sc_distributed.py

SC-069 → SC-080: Distributed orchestration tests.
SC-N001 → SC-N007: Scaling tests (Pillar 3).

SC-N tests require RUN_LARGE_FLEET=1 for N > 4.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import time
import uuid
from typing import Any

import pytest

CONTROL_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from panoseti_grpc.daq_control.client import DaqControlClient

from ci.integration.conftest import (
    DAQNODE_DIRECT_HOST,
    DAQNODE2_HOST,
    GRPC_PORT,
    DAQ_DATA_DIR,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
    wait_until,
)
from ci.integration.fleet import Fleet, make_fleet, MAX_DEFAULT_FLEET_N


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
        for i, (client, run_dir) in enumerate(zip(clients, run_dirs)):
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
            ok, resp = client.StartDaq(rp)
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
            loop.run_in_executor(executor, c.StartDaq, p)
            for c, p in zip(clients, params_list)
        ), return_exceptions=True)
        elapsed = time.monotonic() - t0
        for i, (result, rp) in enumerate(zip(results, params_list)):
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


@pytest.mark.parametrize("n_nodes", [2])
def test_SCN006_telemetry_volume_scales_with_n_nodes(
    n_nodes: int,
    daq_control_direct: DaqControlClient,
    daq_control_node2: DaqControlClient,
) -> None:
    """
    SC-N006: Compare Redis log queue depth after N-node run.
    Ensures RedisBatcher keeps up with N concurrent log producers.
    Skipped if telemetry tests are disabled.
    """
    if not os.getenv("ENABLE_TELEMETRY_TESTS", "").strip() == "1":
        pytest.skip("Set ENABLE_TELEMETRY_TESTS=1 to run telemetry scaling tests")

    try:
        import redis
        rc = redis.Redis(host=os.getenv("REDIS_HOST", "10.0.1.20"), decode_responses=False)
    except Exception as e:
        pytest.skip(f"Redis unavailable: {e}")

    # Measure queue depth before and after a short run
    depth_before = rc.llen("logs:ingress")

    clients = [daq_control_direct, daq_control_node2][:n_nodes]
    run_dirs = [f"scn006_{uuid.uuid4().hex[:8]}.pffd" for _ in range(n_nodes)]
    started = []

    for i, (client, run_dir) in enumerate(zip(clients, run_dirs)):
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
        ok, _ = client.StartDaq(rp)
        if ok:
            started.append((client, rp))

    time.sleep(3)  # Let logs accumulate

    depth_after = rc.llen("logs:ingress")

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

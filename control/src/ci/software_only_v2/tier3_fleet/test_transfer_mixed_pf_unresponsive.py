"""
tier3_fleet/test_transfer_mixed_pf_unresponsive.py — Edge cases for the transfer queue.
Covers mixed port-forwarding topologies and unresponsive node failures.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ci.fixtures.rsync_fixtures import RsyncMock
from ci.software_only_v2.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    get_mapped_client_factory,
    simulate_rsync_from_fleet,
)
from control.transfer.daemon import _process_job, run_daemon
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.utils import config_file
from control.utils.pydantic_config_models import PortForwarding
from control.utils.run_state import RunStateManager, RunStatus
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


@requires_docker
@pytest.mark.parametrize("num_nodes", [2, 4, 8])
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_transfer_queue_mixed_port_forwarding(
    pseti_workspace: Any,
    num_nodes: int,
    mock_rsync_transfer: RsyncMock,
    transfer_job_factory: Callable[..., TransferJob],
    transfer_queue: TransferQueue,
) -> None:
    """Verify transfer succeeds with a mix of direct and port-forwarded DAQ nodes."""
    head_data_dir = pseti_workspace.root / "head_data"
    head_data_dir.mkdir(parents=True, exist_ok=True)
    
    nodes = []
    for i in range(num_nodes):
        if i % 2 == 0:
            pf = None
        else:
            pf = PortForwarding(status=True, gw_ip=f"10.0.1.{10+i}", port=2200+i)
            
        nodes.append(
            config_file.DaqNode(
                ip_addr=f"192.168.100.{10+i}",
                username="panoseti",
                data_dir="/data",
                module_ids=[i+100],
                port_forwarding=pf
            )
        )
        
    run_name = "mixed_pf_test.pffd"
    tq = transfer_queue
    job = transfer_job_factory(
        run_name=run_name,
        head_data_dir=head_data_dir,
        daq_nodes=[
            TransferNodeSpec(
                ip_addr=str(n.ip_addr),
                username="panoseti",
                data_dir=n.data_dir,
                module_ids=n.module_ids,
                port_forwarding=n.port_forwarding
            )
            for n in nodes
        ]
    )
    tq.enqueue(job)
    active_job = tq.claim()
    
    # Mock Manifest
    async def mock_gen_manifest(*args, **kwargs):
        return {"success": True, "manifest_path": "/data/dp_manifest.node_test.algo_blake3.txt", "file_count": 1}

    # Verify that rsync uses the correct IP in the command
    rsync_ips = set()
    def rsync_side_effect(*args, **kwargs):
        cmd_str = " ".join(args)
        for n in nodes:
            expected_ip = str(n.port_forwarding.gw_ip) if n.port_forwarding and n.port_forwarding.status else str(n.ip_addr)
            if expected_ip in cmd_str:
                rsync_ips.add(expected_ip)

        (head_data_dir / run_name).mkdir(parents=True, exist_ok=True)
        (head_data_dir / run_name / "dp_manifest.node_test.algo_blake3.txt").touch()
        return None

    mock_rsync_transfer.side_effect = rsync_side_effect

    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient") as mock_client_cls, \
         patch("control.transfer.daemon.verify_manifest", return_value=(True, [])):
        
        mock_client = AsyncMock()
        mock_client.GenerateManifest.side_effect = mock_gen_manifest
        mock_client.CleanupData.return_value = {"success": True}
        mock_client.__aenter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        mgr = RunStateManager(base_dir=pseti_workspace.root / "state")
        success, err = await asyncio.wait_for(_process_job(active_job, asyncio.Event(), mgr), timeout=10.0)
        assert success, err
        
        # Verify that we rsynced exactly the expected IPs
        expected_ips = {str(n.port_forwarding.gw_ip) if n.port_forwarding else str(n.ip_addr) for n in nodes}
        assert rsync_ips == expected_ips


@requires_docker
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_transfer_queue_unresponsive_node_fails_transfer(
    session_fleet: Fleet,
    mock_rsync_transfer: RsyncMock,
    transfer_queue: TransferQueue,
) -> None:
    """Transfer must not complete if a node becomes unresponsive."""
    fleet = session_fleet
    daq_config = fleet.live_daq_config
    head_data_dir = Path(daq_config.head_node_data_dir)
    run_name = f"unresponsive_{uuid.uuid4().hex[:8]}.pffd"
    
    # 1. Valid start and stop condition
    await generate_mocked_run(fleet, daq_config, run_name)
    
    mgr = RunStateManager()
    tq = transfer_queue
    
    # Check that job is in pending
    assert len(list((tq._queue / "pending").glob("*.job.toml"))) == 1
    assert mgr.load_state(run_name).status == RunStatus.RECORDING_ENDED
    
    # 2. Pick a random node to be unresponsive
    unresponsive_node = random.choice(daq_config.daq_nodes)
    unresponsive_ip = str(unresponsive_node.ip_addr)
    
    # Override client factory to simulate unresponsiveness for that IP
    real_client_factory = get_mapped_client_factory(daq_config)
    def mocked_client_factory(host, port=50051):
        if host == unresponsive_ip or (unresponsive_node.port_forwarding and host == str(unresponsive_node.port_forwarding.gw_ip)):
            mock_client = AsyncMock()
            mock_client.GenerateManifest.side_effect = TimeoutError("Node unresponsive")
            mock_client.__aenter__.return_value = mock_client
            return mock_client
        return real_client_factory(host, port)
        
    def rsync_side_effect(*args, **kwargs):
        cmd_str = " ".join(args)
        if unresponsive_ip in cmd_str or (unresponsive_node.port_forwarding and str(unresponsive_node.port_forwarding.gw_ip) in cmd_str):
            raise TimeoutError("Rsync connection timed out")
            
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        return None

    mock_rsync_transfer.side_effect = rsync_side_effect

    # 3. Start transfer queue loop with fast retries
    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=mocked_client_factory), \
         patch("control.transfer.daemon.RETRY_DELAYS", [1, 1]):
         
        daemon_task = asyncio.create_task(run_daemon(poll_interval=0.5))
        
        # 4. Wait for job to eventually reach 'failed' state in queue
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < 30:
            ledger = mgr.load_state(run_name)
            if ledger and ledger.status == RunStatus.TRANSFER_FAILED and (tq._queue / "failed" / f"{run_name}.job.toml").exists():
                    break
            await asyncio.sleep(0.5)
        else:
            pytest.fail(f"Transfer did not fail as expected. Ledger status: {ledger.status if ledger else 'None'}")
            
        daemon_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await daemon_task
            
    # Verify the job is in the failed queue bucket
    assert (tq._queue / "failed" / f"{run_name}.job.toml").exists()
    assert not (tq._queue / "completed" / f"{run_name}.job.toml").exists()
    
    # Also verify that the partial data on head node doesn't mark it as run_complete
    run_dir_on_head = head_data_dir / run_name
    assert not (run_dir_on_head / "run_complete").exists()

"""
test_transfer_mixed_pf_unresponsive.py — Edge cases for the transfer queue.
Covers mixed port-forwarding topologies and unresponsive node failures.
"""

import asyncio
import random
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ci.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    get_mapped_client_factory,
    setup_isolated_transfer_env,
    simulate_rsync_from_fleet,
)
from control.transfer.daemon import _process_job, run_daemon
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.utils.config_file import DaqConfig, DaqNode
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import PortForwarding
from control.utils.run_state import RunStateManager


@pytest.mark.parametrize("num_nodes", [2, 4, 8])
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_transfer_queue_mixed_port_forwarding(
    num_nodes: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify transfer succeeds with a mix of direct and port-forwarded DAQ nodes."""
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    head_data_dir = tmp_path / "head_data"
    head_data_dir.mkdir(parents=True)
    monkeypatch.setenv("HEAD_DATA_DIR", str(head_data_dir))
    PanoPaths.ensure_state_dirs()
    
    nodes = []
    for i in range(num_nodes):
        if i % 2 == 0:
            # Direct connection
            pf = None
        else:
            # Port forwarded behind different routers
            pf = PortForwarding(status=True, gw_ip=f"10.0.1.{10+i}", port=2200+i)
            
        nodes.append(
            DaqNode(
                ip_addr=f"192.168.100.{10+i}",
                username="panoseti",
                data_dir="/data",
                module_ids=[i+100],
                port_forwarding=pf
            )
        )
        
    daq_config = DaqConfig(
        daq_nodes=nodes, 
        head_node_data_dir=str(head_data_dir),
        head_node_ip_addr="127.0.0.1"
    )
    
    run_name = "mixed_pf_test.pffd"
    tq = TransferQueue()
    job = TransferJob(
        run_name=run_name,
        head_data_dir=str(head_data_dir),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        daq_nodes=[
            TransferNodeSpec(
                ip_addr=n.ip_addr,
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
    
    # Mock Manifest and Rsync
    async def mock_gen_manifest(*args, **kwargs):
        return {"success": True, "manifest_path": "/data/dp_manifest.node_test.algo_blake3.txt", "file_count": 1}

    # Verify that rsync uses the correct IP in the command
    rsync_ips = set()
    async def mock_rsync(*args, **kwargs):
        # args is the cmd tuple: ("rsync", "-aP", ...)
        cmd_str = " ".join(args)
        for n in nodes:
            # Determine which IP we expect rsync to use for this node
            expected_ip = str(n.port_forwarding.gw_ip) if n.port_forwarding and n.port_forwarding.status else str(n.ip_addr)
            if expected_ip in cmd_str:
                rsync_ips.add(expected_ip)

        (head_data_dir / run_name).mkdir(parents=True, exist_ok=True)
        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b'', b''))
        proc.stdout.readline = AsyncMock(return_value=b'')
        proc.stderr.read = AsyncMock(return_value=b'')
        return proc

    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient") as mock_client_cls, \
         patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=mock_rsync), \
         patch("control.transfer.daemon.verify_manifest", return_value=(True, [])):
        
        mock_client = AsyncMock()
        mock_client.GenerateManifest.side_effect = mock_gen_manifest
        mock_client.CleanupData.return_value = {"success": True}
        mock_client.__aenter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        mgr = RunStateManager()
        success, err = await asyncio.wait_for(_process_job(active_job, asyncio.Event(), mgr), timeout=10.0)
        assert success, err
        assert mock_client.GenerateManifest.call_count == num_nodes
        
        # Verify that we rsynced exactly the expected IPs
        expected_ips = {str(n.port_forwarding.gw_ip) if n.port_forwarding else str(n.ip_addr) for n in nodes}
        assert rsync_ips == expected_ips


@pytest.mark.asyncio
@pytest.mark.timeout(60)  # Smaller timeout
async def test_transfer_queue_unresponsive_node_fails_transfer(
    session_fleet: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transfer must not complete if a node becomes unresponsive. It must return to pending and ultimately failed."""
    fleet, daq_cfg_dict = session_fleet
    head_data_dir, daq_config = setup_isolated_transfer_env(tmp_path, monkeypatch, daq_cfg_dict)
    run_name = f"unresponsive_{uuid.uuid4().hex[:8]}.pffd"
    
    # 1. Valid start and stop condition -> data generated and transfer enqueued
    expected_data = await generate_mocked_run(fleet, daq_config, run_name)
    
    mgr = RunStateManager()
    tq = TransferQueue()
    
    # Check that job is in pending
    assert len(list((tq._queue / "pending").glob("*.job.toml"))) == 1
    assert mgr.load_state().status == "RECORDING_ENDED"
    
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
        
    async def mocked_rsync(*args, **kwargs):
        cmd_str = " ".join(args)
        if unresponsive_ip in cmd_str or (unresponsive_node.port_forwarding and str(unresponsive_node.port_forwarding.gw_ip) in cmd_str):
            raise TimeoutError("Rsync connection timed out")
            
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b'', b''))
        proc.stdout.readline = AsyncMock(return_value=b'')
        proc.stderr.read = AsyncMock(return_value=b'')
        return proc

    # 3. Start transfer queue loop with fast retries
    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=mocked_rsync), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=mocked_client_factory), \
         patch("control.transfer.daemon.RETRY_DELAYS", [1, 1]):
         
        daemon_task = asyncio.create_task(run_daemon(poll_interval=0.5))
        
        # 4. Wait for job to eventually reach 'failed' state in queue
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < 30:
            ledger = mgr.load_state()
            if ledger and ledger.status == "TRANSFER_FAILED" and (tq._queue / "failed" / f"{run_name}.job.toml").exists():
                    break
            await asyncio.sleep(0.5)
        else:
            pytest.fail(f"Transfer did not fail as expected. Ledger status: {ledger.status if ledger else 'None'}")
            
        daemon_task.cancel()
        try:
            await daemon_task
        except asyncio.CancelledError:
            pass
            
    # Verify the job is in the failed queue bucket
    assert (tq._queue / "failed" / f"{run_name}.job.toml").exists()
    assert not (tq._queue / "completed" / f"{run_name}.job.toml").exists()
    
    # Also verify that the partial data on head node doesn't mark it as run_complete
    run_dir_on_head = head_data_dir / run_name
    assert not (run_dir_on_head / "run_complete").exists()

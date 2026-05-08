from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ci.fixtures.rsync_fixtures import RsyncMock
from ci.software_only_v2.orchestrator.fleet import Fleet
from ci.software_only_v2.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    simulate_rsync_from_fleet,
)
from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob, TransferStatus
from control.transfer.queue import TransferQueue
from control.utils.run_state import RunStateManager

# Mark tests as requiring docker
requires_docker = pytest.mark.requires_docker

@requires_docker
@pytest.mark.asyncio
async def test_when_pff_corrupted_after_transfer_then_verify_fails_and_cleanup_skipped(
    session_fleet: Fleet,
    mock_rsync_transfer: RsyncMock,
    transfer_job_factory: Callable[..., TransferJob],
    transfer_queue: TransferQueue,
) -> None:
    """Detection of data corruption on the head node.

    Scenario: files are transferred successfully.  A bit is flipped in one .pff
    file on the head node disk.
    Expectation: Manifest verification fails; job is moved to VERIFY_FAILED;
    CleanupData is NOT called on the DAQ node.
    """
    fleet = session_fleet
    daq_config = fleet.live_daq_config
    head_data_dir = Path(daq_config.head_node_data_dir)
    run_name = f"xfail_corrupt_{uuid.uuid4().hex[:8]}.pffd"

    await generate_mocked_run(fleet, daq_config, run_name)

    mgr = RunStateManager()
    tq = transfer_queue
    pending_path = tq._job_path(TransferStatus.PENDING, run_name)
    if pending_path.exists():
        pending_path.unlink()

    job = transfer_job_factory(run_name=run_name, head_data_dir=head_data_dir, daq_config=daq_config)
    assert tq.enqueue(job) is True

    cleanup_called = False

    def rsync_side_effect(*args: object, **kwargs: object) -> None:
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        # Flip a bit in one transferred file
        dest_run = head_data_dir / run_name
        pff_files = [f for f in dest_run.glob("**/*.pff") if f.stat().st_size > 0]
        assert len(pff_files) > 0
        data = bytearray(pff_files[0].read_bytes())
        data[0] ^= 0xFF
        pff_files[0].write_bytes(bytes(data))

    mock_rsync_transfer.side_effect = rsync_side_effect

    from panoseti_grpc.daq_control.client import AsyncDaqControlClient as RealADCC

    def wrapped_client_factory(host: str, port: int = 50051) -> object:
        mock_client = AsyncMock(spec=RealADCC)
        mock_client.__aenter__.return_value = mock_client
        
        # We need real manifest for verification to fail correctly on the corrupt file
        # But we want to spy on CleanupData.
        # Actually, let's just mock GenerateManifest and GetManifest to return
        # what they would normally return, but since we are in a fleet, we can
        # just use a real client for everything EXCEPT CleanupData.
        
        # Manually map to gateway if needed (avoiding recursion)
        target_host, target_port = host, port
        assert daq_config is not None
        for node in daq_config.daq_nodes:
            if str(node.ip_addr) == host:
                if node.port_forwarding and node.port_forwarding.status:
                    target_host = str(node.port_forwarding.gw_ip)
                    target_port = node.port_forwarding.grpc_port
                break
        
        real_client = RealADCC(host=target_host, port=target_port)
        
        mock_client.GenerateManifest.side_effect = real_client.GenerateManifest
        mock_client.GetManifest.side_effect = real_client.GetManifest

        async def spy_cleanup(*a: object, **kw: object) -> dict:
            nonlocal cleanup_called
            cleanup_called = True
            return {"success": True}
        mock_client.CleanupData.side_effect = spy_cleanup
        
        return mock_client

    with patch(
        "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
        side_effect=wrapped_client_factory,
    ):
        success, _err = await _process_job(job, asyncio.Event(), mgr)

    assert success is False, "Job should have failed due to data corruption"
    assert not cleanup_called, "Cleanup should not be called after verification failure"

    state = mgr.load_state()
    assert state.status == TransferStatus.VERIFY_FAILED


@requires_docker
@pytest.mark.asyncio
async def test_when_cleanup_dag_rejects_digest_then_job_fails(
    session_fleet: Fleet,
    mock_rsync_transfer: RsyncMock,
    transfer_job_factory: Callable[..., TransferJob],
    transfer_queue: TransferQueue,
) -> None:
    """Detection of CleanupData failure.

    Scenario: the DAQ node's CleanupData returns FAILED_PRECONDITION because
    the manifest_digest from the head node doesn't match the server's copy.
    Expectation: job fails; status is VERIFY_FAILED or similar.
    """
    fleet = session_fleet
    daq_config = fleet.live_daq_config
    assert daq_config is not None
    head_data_dir = Path(daq_config.head_node_data_dir)
    run_name = f"xfail_precond_{uuid.uuid4().hex[:8]}.pffd"

    await generate_mocked_run(fleet, daq_config, run_name)

    mgr = RunStateManager()
    tq = transfer_queue
    pending_path = tq._job_path(TransferStatus.PENDING, run_name)
    if pending_path.exists():
        pending_path.unlink()

    job = transfer_job_factory(run_name=run_name, head_data_dir=head_data_dir, daq_config=daq_config)
    assert tq.enqueue(job) is True

    def rsync_side_effect(*args: object, **kwargs: object) -> None:
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)

    mock_rsync_transfer.side_effect = rsync_side_effect

    from panoseti_grpc.daq_control.client import AsyncDaqControlClient as RealADCC

    def wrapped_client_factory(host: str, port: int = 50051) -> object:
        mock_client = AsyncMock(spec=RealADCC)
        mock_client.__aenter__.return_value = mock_client
        
        # Manually map to gateway if needed (avoiding recursion)
        target_host, target_port = host, port
        for node in daq_config.daq_nodes:
            if str(node.ip_addr) == host:
                if node.port_forwarding and node.port_forwarding.status:
                    target_host = str(node.port_forwarding.gw_ip)
                    target_port = node.port_forwarding.grpc_port
                break
        
        real_client = RealADCC(host=target_host, port=target_port)
        mock_client.GenerateManifest.side_effect = real_client.GenerateManifest
        mock_client.GetManifest.side_effect = real_client.GetManifest

        async def reject_cleanup(*c_args: object, **c_kwargs: object) -> dict:
            return {
                "success": False,
                "message": "FAILED_PRECONDITION: manifest_digest mismatch",
            }
        mock_client.CleanupData.side_effect = reject_cleanup
        
        return mock_client

    with patch(
        "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
        side_effect=wrapped_client_factory,
    ):
        success, err = await _process_job(job, asyncio.Event(), mgr)

    assert success is False, "Job should fail when cleanup is rejected"
    assert "FAILED_PRECONDITION" in (err or "")

    state = mgr.load_state()
    assert state is not None
    assert state.status == TransferStatus.CLEAN_FAILED

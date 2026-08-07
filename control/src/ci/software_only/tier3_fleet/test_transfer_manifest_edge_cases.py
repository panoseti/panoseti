from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ci.fixtures.rsync_fixtures import RsyncMock
from ci.software_only.orchestrator.fleet import Fleet
from ci.software_only.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    simulate_rsync_from_fleet,
)
from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob, TransferStatus
from control.transfer.queue import TransferQueue
from control.utils.pydantic_config_models import RunStatus
from control.utils.run_state import RunStateManager

# Mark tests as requiring docker
requires_docker = pytest.mark.requires_docker

@requires_docker
@pytest.mark.asyncio
@pytest.mark.timeout(120)
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
        # Manually map to gateway if needed
        target_host, target_port = host, port
        for node in daq_config.daq_nodes:
            if str(node.ip_addr) == host:
                if node.port_forwarding and node.port_forwarding.status:
                    target_host = str(node.port_forwarding.gw_ip)
                    target_port = node.port_forwarding.grpc_port
                break
        
        client = RealADCC(host=target_host, port=target_port)
        original_aenter = client.__aenter__

        async def mocked_aenter() -> object:
            res = await original_aenter()
            
            async def spy_cleanup(*a: object, **kw: object) -> dict:
                nonlocal cleanup_called
                cleanup_called = True
                # Call real cleanup via the stub which is now initialized
                v_params = type('v_params', (), kw) # dummy
                # Actually, just return success since we want to spy
                return {"success": True}

            res.CleanupData = spy_cleanup
            return res

        client.__aenter__ = mocked_aenter
        return client

    with patch(
        "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
        side_effect=wrapped_client_factory,
    ):
        success, _err = await _process_job(job, asyncio.Event(), mgr)

    assert success is False, "Job should have failed due to data corruption"
    assert not cleanup_called, "Cleanup should not be called after verification failure"

    state = mgr.load_state()
    assert state.status == RunStatus.VERIFY_FAILED


@requires_docker
@pytest.mark.asyncio
@pytest.mark.timeout(120)
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
        mock_client.__aexit__.return_value = None

        # Success for manifest generation (Stage 1)
        mock_client.GenerateManifest.return_value = {"success": True, "manifest_path": "mocked_path"}

        # Success for fetching manifest entries (Stage 1)
        async def mock_get_manifest(*a: object, **kw: object): # type: ignore
            # yield some minimal valid entries so Stage 1 finishes
            yield {
                "digest_hex": "mocked_digest",
                "size_bytes": 100,
                "mtime_ns": 12345,
                "relative_path": "mocked_file.pff",
            }
        mock_client.GetManifest.side_effect = mock_get_manifest

        # FAILURE for cleanup (Stage 4)
        mock_client.CleanupData.return_value = {
            "success": False,
            "message": "FAILED_PRECONDITION: manifest_digest mismatch",
        }
        return mock_client

    with patch(
        "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
        side_effect=wrapped_client_factory,
    ), patch("control.transfer.daemon.verify_manifest", return_value=(True, [])):
        success, err = await _process_job(job, asyncio.Event(), mgr)

    assert success is False, "Job should fail when cleanup is rejected"
    assert "FAILED_PRECONDITION" in (err or "")

    state = mgr.load_state()
    # When cleanup fails, daemon transitions to VERIFY_FAILED (or similar) to indicate
    # that the job did not complete successfully and needs attention.
    assert state.status == RunStatus.VERIFY_FAILED

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ci.fixtures.fleet import Fleet
from ci.fixtures.rsync_fixtures import RsyncMock
from ci.software_only.tier3_fleet.transfer_testing_utils import (
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
@pytest.mark.timeout(120)
async def test_when_poisoned_manifest_on_daq_node_then_grpc_content_wins(
    session_fleet: Fleet,
    mock_rsync_transfer: RsyncMock,
    transfer_job_factory: Callable[..., TransferJob],
    transfer_queue: TransferQueue,
) -> None:
    """Manifest on head node must reflect GetManifest RPC content, not rsync copy.

    Scenario: a "poisoned" manifest file is written to the DAQ node disk before
    the transfer starts.  GetManifest returns independently-generated content.
    Expectation: the head node's manifest file contains the RPC content and
    NOT the poisoned content.  (rsync already excludes manifest files, but
    this test confirms the RPC path is what the daemon trusts.)
    """
    fleet = session_fleet
    daq_config = fleet.live_daq_config
    head_data_dir = Path(daq_config.head_node_data_dir)
    run_name = f"manifest_test_{uuid.uuid4().hex[:8]}.pffd"

    await generate_mocked_run(fleet, daq_config, run_name)

    # Poison: write a fake manifest file on each DAQ node before the transfer.
    poison_content = "POISONED_MANIFEST_CONTENT"
    for i, node in enumerate(daq_config.daq_nodes):
        fleet.exec_in_node(
            i,
            f"sh -c 'echo \"{poison_content}\" "
            f"> {node.data_dir}/{run_name}/dp_manifest.node_poisoned.algo_blake3.txt'",
        )

    mgr = RunStateManager()
    tq = transfer_queue

    # stop_run auto-enqueued a job. We remove it so we can enqueue our
    # custom one with the right config.
    pending_path = tq._job_path(TransferStatus.PENDING, run_name)
    if pending_path.exists():
        pending_path.unlink()

    job = transfer_job_factory(run_name=run_name, head_data_dir=head_data_dir, daq_config=daq_config)
    assert tq.enqueue(job) is True

    def rsync_side_effect(*args: object, **kwargs: object) -> None:
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)

    mock_rsync_transfer.side_effect = rsync_side_effect

    # Intercept GetManifest to return a well-known secure value.
    _SECURE_CONTENT = "REAL_SECURE_MANIFEST_CONTENT"
    _SECURE_FILE = "secure_file.pff"

    from panoseti_grpc.daq_control.client import AsyncDaqControlClient as RealADCC

    def wrapped_client_factory(host: str, port: int = 50051) -> object:
        mock_client = AsyncMock(spec=RealADCC)
        mock_client.__aenter__.return_value = mock_client
        mock_client.GenerateManifest.return_value = {"success": True, "manifest_path": "mocked_path"}

        async def mock_get_manifest(*a: object, **kw: object):  # type: ignore
            yield {
                "digest_hex": _SECURE_CONTENT,
                "size_bytes": 123,
                "mtime_ns": 456,
                "relative_path": _SECURE_FILE,
            }
        mock_client.GetManifest.side_effect = mock_get_manifest
        return mock_client

    with patch(
        "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
        side_effect=wrapped_client_factory,
    ):
        await asyncio.wait_for(
            _process_job(job, asyncio.Event(), mgr),  # type: ignore[arg-type]
            timeout=30.0,
        )

    head_run_dir = head_data_dir / run_name
    manifest_files = list(head_run_dir.glob("dp_manifest.node_*.txt"))

    assert len(manifest_files) > 0, "No manifest files found on head node"
    for mf in manifest_files:
        content = mf.read_text().strip()
        assert poison_content not in content, (
            f"Manifest {mf.name} contains poisoned content — rsync exclusion failed"
        )
        assert _SECURE_CONTENT in content, (
            f"Manifest {mf.name} missing secure RPC content"
        )
        assert _SECURE_FILE in content, (
            f"Manifest {mf.name} missing entry from GetManifest stream"
        )

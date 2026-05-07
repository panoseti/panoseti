"""
tier3_fleet/test_transfer_manifest_edge_cases.py

'Expect-to-fail' scenarios for the manifest verification and selective
cleanup contracts.  Ported from software_only/tier3_fleet/.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from ci.fixtures.rsync_fixtures import RsyncMock
from ci.software_only_v2.orchestrator.fleet import Fleet
from ci.software_only_v2.tier3_fleet.conftest import requires_docker
from ci.software_only_v2.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    get_mapped_client_factory,
    simulate_rsync_from_fleet,
)
from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob
from control.utils.pydantic_config_models import RunStatus
from control.utils.run_state import RunStateManager

pytestmark = pytest.mark.tier3


@requires_docker
@pytest.mark.asyncio
async def test_when_pff_corrupted_after_transfer_then_verify_fails_and_cleanup_skipped(
    session_fleet: Fleet,
    mock_rsync_transfer: RsyncMock,
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    """VERIFYING stage detects a bit-flip and never calls CleanupData.

    Scenario: a .pff byte is flipped on the head node after rsync completes.
    Expectation: job transitions to VERIFY_FAILED; CleanupData is never invoked.
    """
    fleet = session_fleet
    daq_config = fleet.live_daq_config
    head_data_dir = Path(daq_config.head_node_data_dir)
    run_name = f"xfail_corrupt_{uuid.uuid4().hex[:8]}.pffd"

    await generate_mocked_run(fleet, daq_config, run_name)

    mgr = RunStateManager()
    job = transfer_job_factory(run_name=run_name, head_data_dir=head_data_dir)

    def rsync_side_effect(*args: object, **kwargs: object) -> None:
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        dest_run = head_data_dir / run_name
        pff_files = [f for f in dest_run.glob("**/*.pff") if f.stat().st_size > 0]
        assert len(pff_files) > 0
        data = bytearray(pff_files[0].read_bytes())
        data[0] ^= 0xFF
        pff_files[0].write_bytes(bytes(data))

    mock_rsync_transfer.side_effect = rsync_side_effect

    cleanup_called = False

    def wrapped_client_factory(*args: object, **kwargs: object) -> object:
        client = get_mapped_client_factory(daq_config)(*args, **kwargs)
        original_aenter = client.__aenter__

        async def mocked_aenter() -> object:
            stub = await original_aenter()
            original_cleanup = stub.CleanupData

            async def spy_cleanup(*c_args: object, **c_kwargs: object) -> object:
                nonlocal cleanup_called
                cleanup_called = True
                return await original_cleanup(*c_args, **c_kwargs)

            stub.CleanupData = spy_cleanup
            return stub

        client.__aenter__ = mocked_aenter
        return client

    with patch(
        "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
        side_effect=wrapped_client_factory,
    ):
        success, _err = await _process_job(job, asyncio.Event(), mgr)

    assert success is False, "Job should have failed due to data corruption"
    assert not cleanup_called, "CleanupData MUST NOT be called when verification fails"

    state = mgr.load_state()
    assert state is not None
    assert state.status == RunStatus.VERIFY_FAILED


@requires_docker
@pytest.mark.asyncio
async def test_when_cleanup_dag_rejects_digest_then_job_fails(
    session_fleet: Fleet,
    mock_rsync_transfer: RsyncMock,
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    """CleanupData rejection with FAILED_PRECONDITION causes the job to fail.

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
    job = transfer_job_factory(run_name=run_name, head_data_dir=head_data_dir)

    def rsync_side_effect(*args: object, **kwargs: object) -> None:
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)

    mock_rsync_transfer.side_effect = rsync_side_effect

    def wrapped_client_factory(*args: object, **kwargs: object) -> object:
        client = get_mapped_client_factory(daq_config)(*args, **kwargs)
        original_aenter = client.__aenter__

        async def mocked_aenter() -> object:
            stub = await original_aenter()

            async def reject_cleanup(*c_args: object, **c_kwargs: object) -> dict:
                return {
                    "success": False,
                    "message": "FAILED_PRECONDITION: manifest_digest mismatch",
                }

            stub.CleanupData = reject_cleanup
            return stub

        client.__aenter__ = mocked_aenter
        return client

    with patch(
        "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
        side_effect=wrapped_client_factory,
    ):
        success, err = await _process_job(job, asyncio.Event(), mgr)

    assert success is False, "Job should fail when cleanup is rejected"
    assert "FAILED_PRECONDITION" in (err or "")

    state = mgr.load_state()
    assert state is not None

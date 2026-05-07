"""
tier3_fleet/test_transfer_manifest_integrity.py

Verifies that the manifest on the head node is obtained via the GetManifest
RPC (not copied from the DAQ node disk via rsync).

Ported from software_only/tier3_fleet/test_transfer_manifest_integrity.py.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from collections.abc import Callable

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
from control.utils.run_state import RunStateManager

pytestmark = pytest.mark.tier3


@requires_docker
@pytest.mark.asyncio
async def test_when_poisoned_manifest_on_daq_node_then_grpc_content_wins(
    session_fleet: Fleet,
    mock_rsync_transfer: RsyncMock,
    transfer_job_factory: Callable[..., TransferJob],
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
    job = transfer_job_factory(run_name=run_name, head_data_dir=head_data_dir)

    def rsync_side_effect(*args: object, **kwargs: object) -> None:
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)

    mock_rsync_transfer.side_effect = rsync_side_effect

    # Intercept GetManifest to return a well-known secure value.
    _SECURE_CONTENT = "REAL_SECURE_MANIFEST_CONTENT"
    _SECURE_FILE = "secure_file.pff"

    def wrapped_client_factory(*args: object, **kwargs: object) -> object:
        client = get_mapped_client_factory(daq_config)(*args, **kwargs)
        original_aenter = client.__aenter__

        async def mocked_aenter() -> object:
            stub = await original_aenter()

            async def mock_generate_manifest(*a: object, **kw: object) -> dict:
                return {"success": True, "manifest_path": "mocked_path"}

            async def mock_get_manifest(*a: object, **kw: object):  # type: ignore[return]
                yield {
                    "digest_hex": _SECURE_CONTENT,
                    "size_bytes": 123,
                    "mtime_ns": 456,
                    "relative_path": _SECURE_FILE,
                }

            stub.GenerateManifest = mock_generate_manifest
            stub.GetManifest = mock_get_manifest
            return stub

        client.__aenter__ = mocked_aenter
        return client

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

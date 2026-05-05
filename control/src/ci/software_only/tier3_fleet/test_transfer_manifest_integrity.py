"""
test_transfer_manifest_integrity.py

Verifies that the manifest used for verification is NOT simply copied from the 
DAQ node via rsync, which would defeat the purpose of independent verification.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ci.software_only.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    get_mapped_client_factory,
    setup_isolated_transfer_env,
    simulate_rsync_from_fleet,
)
from control.transfer.daemon import _process_job
from control.utils.run_state import RunStateManager


@pytest.mark.asyncio
async def test_manifest_not_overwritten_by_rsync(
    session_fleet: Any,
    tmp_path: Path,
    ensure_clean_daq_state: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Scenario: A 'poisoned' manifest exists on the DAQ node disk.
    Expectation: The TransferDaemon should either ignore it or ensure the 
    manifest obtained via gRPC (source of truth) takes precedence and is NOT
    overwritten by rsync's recursive copy.
    """
    fleet, daq_cfg_dict = session_fleet
    head_data_dir, daq_config = setup_isolated_transfer_env(tmp_path, monkeypatch, daq_cfg_dict)
    run_name = f"manifest_test_{uuid.uuid4().hex[:8]}.pffd"
    
    # 1. Generate real run data
    await generate_mocked_run(fleet, daq_config, run_name)
    
    # 2. Poison the DAQ node with a fake manifest file
    # The expected filename is dp_manifest.node_<hostname>.algo_blake3.txt
    # We'll just poison all DAQ nodes for simplicity.
    poison_content = "POISONED_MANIFEST_CONTENT"
    
    for i, container in enumerate(fleet.containers):
        node = daq_config.daq_nodes[i]
        # Note: In the fleet test, hostname inside container might be the container ID or 'daqnode-...'
        # We'll use a glob to find it or just create one with a known name.
        # But wait, daemon.py globs for dp_manifest.node_*.txt
        
        # Let's find the real hostname of the container
        res = container.get_wrapped_container().exec_run("hostname")
        hostname = res.output.decode().strip()
        manifest_name = f"dp_manifest.node_{hostname}.algo_blake3.txt"
        
        # Create the poison file
        container.get_wrapped_container().exec_run(
            f"sh -c 'echo \"{poison_content}\" > {node.data_dir}/{run_name}/{manifest_name}'"
        )

    mgr = RunStateManager()
    from datetime import UTC, datetime

    from control.transfer.models import TransferJob, TransferNodeSpec
    
    job = TransferJob(
        run_name=run_name,
        head_data_dir=str(head_data_dir),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        daq_nodes=[
            TransferNodeSpec(
                ip_addr=n.ip_addr,
                username=n.username,
                data_dir=str(n.data_dir),
                module_ids=n.module_ids,
                port_forwarding=n.port_forwarding
            )
            for n in daq_config.daq_nodes
        ]
    )

    async def normal_rsync(*args, **kwargs):
        # This will copy everything, including our poison manifest!
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        proc = MagicMock(returncode=0)
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.stdout.readline = AsyncMock(return_value=b"")
        proc.stderr.read = AsyncMock(return_value=b"")
        return proc

    # Mock GenerateManifest and GetManifest to return something ELSE (the 'real' secure manifest)
    
    # We need a custom mock for the client that handles the stream
    class MockClient:
        def __init__(self, real_client):
            self.real_client = real_client
            
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        
        async def GenerateManifest(self, *args, **kwargs):
            return {"success": True, "manifest_path": "mocked_path"}
            
        async def GetManifest(self, *args, **kwargs):
            # Return a single entry that matches our 'secure' content
            yield {
                "digest_hex": "REAL_SECURE_MANIFEST_CONTENT",
                "size_bytes": 123,
                "mtime_ns": 456,
                "relative_path": "secure_file.pff"
            }
        
        # Delegate other calls to real client
        def __getattr__(self, name):
            return getattr(self.real_client, name)

    def mocked_client_factory(daq_config):
        real_factory = get_mapped_client_factory(daq_config)
        def _get_mocked(host, port=50051):
            return MockClient(real_factory(host, port))
        return _get_mocked

    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=normal_rsync), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=mocked_client_factory(daq_config)):
         
         # Execute the transfer job
         _success, _ = await asyncio.wait_for(_process_job(job, asyncio.Event(), mgr), timeout=30.0)
         
         # The head node should now have exactly one manifest file per node,
         # and its content should be what we returned via GetManifest.
         
         head_run_dir = head_data_dir / run_name
         manifest_files = list(head_run_dir.glob("dp_manifest.node_*.txt"))
         
         assert len(manifest_files) > 0, "No manifest files found on head node!"
         
         for mf in manifest_files:
             content = mf.read_text().strip()
             # If this fails, it proves the manifest WAS copied from the DAQ node disk (poisoned)
             # rather than being generated independently/securely.
             assert poison_content not in content, f"Manifest {mf.name} contains poisoned content!"
             assert "REAL_SECURE_MANIFEST_CONTENT" in content, f"Manifest {mf.name} missing secure RPC content."
             assert "secure_file.pff" in content, f"Manifest {mf.name} missing entry from GetManifest stream."

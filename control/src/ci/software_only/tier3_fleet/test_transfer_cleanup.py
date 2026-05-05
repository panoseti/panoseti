"""
test_transfer_cleanup.py

Verifies the CleanupData behavior of the TransferDaemon on DAQ nodes after
a successful transfer and verification.
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
from control.utils.pydantic_config_models import RunStatus
from control.utils.run_state import RunStateManager


@pytest.mark.asyncio
async def test_transfer_selective_cleanup(
    session_fleet: Any,
    tmp_path: Path,
    ensure_clean_daq_state: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Scenario: A successful run transfer occurs.
    Expectation: The TransferDaemon invokes CleanupData to delete .pff files
    on the DAQ nodes while preserving metadata like .json and .log files.
    """
    fleet, daq_cfg_dict = session_fleet
    head_data_dir, daq_config = setup_isolated_transfer_env(tmp_path, monkeypatch, daq_cfg_dict)
    run_name = f"cleanup_test_{uuid.uuid4().hex[:8]}.pffd"
    
    # 1. Generate real run data on the DAQ containers (.pff files and meta.json)
    await generate_mocked_run(fleet, daq_config, run_name)
    
    mgr = RunStateManager()
    from datetime import UTC, datetime

    from control.transfer.models import TransferJob, TransferNodeSpec
    
    job = TransferJob(
        run_name=run_name,
        head_data_dir=str(head_data_dir),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        no_cleanup=False,  # Important: enable cleanup!
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
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b'', b''))
        proc.stdout.readline = AsyncMock(return_value=b'')
        proc.stderr.read = AsyncMock(return_value=b'')
        return proc

    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=normal_rsync), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):
         
         # Allow some time for the full state machine to run including RPCs
         success, _err = await asyncio.wait_for(_process_job(job, asyncio.Event(), mgr), timeout=30.0)
         
         assert success is True, f"Job failed: {_err}"
         
         state = mgr.load_state()
         assert state is not None
         assert state.status == RunStatus.ARCHIVED

    # Verify that cleanup happened on the DAQ nodes
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        
        # Check root run dir metadata
        meta_file = host_root / run_name / "meta.json"
        assert meta_file.exists(), f"Metadata file {meta_file} should have been preserved on DAQ node"
        
        # Check module subdirs for .pff deletion
        for mid in spec.module_ids:
            host_mod_run_dir = host_root / f"module_{mid}" / run_name
            if host_mod_run_dir.exists():
                pff_files = list(host_mod_run_dir.glob("*.pff"))
                assert len(pff_files) == 0, f"Expected 0 .pff files in {host_mod_run_dir} after cleanup, found {len(pff_files)}"


@pytest.mark.asyncio
async def test_transfer_cleanup_isolation(
    session_fleet: Any,
    tmp_path: Path,
    ensure_clean_daq_state: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Scenario: Multiple runs exist on the DAQ node.
    Expectation: Cleanup for run A MUST NOT affect run B.
    """
    fleet, daq_cfg_dict = session_fleet
    head_data_dir, daq_config = setup_isolated_transfer_env(tmp_path, monkeypatch, daq_cfg_dict)
    
    run_to_clean = f"clean_me_{uuid.uuid4().hex[:8]}.pffd"
    run_to_keep = f"keep_me_{uuid.uuid4().hex[:8]}.pffd"
    
    # 1. Generate data for both runs
    await generate_mocked_run(fleet, daq_config, run_to_clean)
    await generate_mocked_run(fleet, daq_config, run_to_keep)
    
    mgr = RunStateManager()
    from datetime import UTC, datetime

    from control.transfer.models import TransferJob, TransferNodeSpec
    
    job = TransferJob(
        run_name=run_to_clean,
        head_data_dir=str(head_data_dir),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        no_cleanup=False,
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
        simulate_rsync_from_fleet(fleet, run_to_clean, head_data_dir / run_to_clean)
        proc = MagicMock(returncode=0)
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.stdout.readline = AsyncMock(return_value=b"")
        proc.stderr.read = AsyncMock(return_value=b"")
        return proc

    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=normal_rsync), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):
         
         success, _ = await asyncio.wait_for(_process_job(job, asyncio.Event(), mgr), timeout=30.0)
         assert success is True

    # Verify Isolation
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        
        for mid in spec.module_ids:
            # Run A should be cleaned
            clean_dir = host_root / f"module_{mid}" / run_to_clean
            if clean_dir.exists():
                assert len(list(clean_dir.glob("*.pff"))) == 0
            
            # Run B should be UNTOUCHED
            keep_dir = host_root / f"module_{mid}" / run_to_keep
            assert keep_dir.exists()
            assert len(list(keep_dir.glob("*.pff"))) > 0, f"Run {run_to_keep} was accidentally cleaned!"


@pytest.mark.asyncio
async def test_transfer_no_cleanup_on_verification_failure(
    session_fleet: Any,
    tmp_path: Path,
    ensure_clean_daq_state: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Scenario: Verification fails (data corruption).
    Expectation: Cleanup MUST NOT be performed; data preserved on DAQ for recovery.
    """
    fleet, daq_cfg_dict = session_fleet
    head_data_dir, daq_config = setup_isolated_transfer_env(tmp_path, monkeypatch, daq_cfg_dict)
    run_name = f"fail_cleanup_{uuid.uuid4().hex[:8]}.pffd"
    
    await generate_mocked_run(fleet, daq_config, run_name)
    
    mgr = RunStateManager()
    from datetime import UTC, datetime

    from control.transfer.models import TransferJob, TransferNodeSpec
    
    job = TransferJob(
        run_name=run_name,
        head_data_dir=str(head_data_dir),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        no_cleanup=False,
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

    async def corrupting_rsync(*args, **kwargs):
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        # Corrupt head-node copy to trigger verification failure
        dest_run = head_data_dir / run_name
        pff = next(dest_run.glob("*.pff"))
        pff.write_bytes(b"CORRUPTED DATA")
        
        proc = MagicMock(returncode=0)
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.stdout.readline = AsyncMock(return_value=b"")
        proc.stderr.read = AsyncMock(return_value=b"")
        return proc

    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=corrupting_rsync), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):
         
         success, _ = await asyncio.wait_for(_process_job(job, asyncio.Event(), mgr), timeout=30.0)
         assert success is False
         assert mgr.load_state().status == RunStatus.VERIFY_FAILED

    # Verify NO cleanup happened on DAQ
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        for mid in spec.module_ids:
            host_mod_run_dir = host_root / f"module_{mid}" / run_name
            if host_mod_run_dir.exists():
                assert len(list(host_mod_run_dir.glob("*.pff"))) > 0, "Data was cleaned despite verification failure!"

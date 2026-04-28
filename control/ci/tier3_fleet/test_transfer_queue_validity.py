"""
test_transfer_queue_validity.py — Happy path test for the transfer queue.
Verifies 100% byte accuracy of transferred .pff files from DAQ nodes to head node.
"""

import asyncio
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ci.fixtures.fleet import Fleet
from control.start import start_run
from control.stop import stop_run
from control.transfer.daemon import _process_job
from control.transfer.queue import TransferQueue
from control.utils import config_file
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager


def _prepare_simulated_data(fleet: Fleet, run_name: str) -> dict[str, bytes]:
    """
    Populate DAQ nodes with simulated data files.
    Returns a mapping of relative_path (on head node) -> expected_bytes.
    """
    expected_data = {}
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        container = fleet.containers[i].get_wrapped_container()
        
        # Log for debugging
        print(f"DEBUG: Node {i} host_root={host_root}")
        
        # 1. Ensure host root is writable
        os.system(f"chmod 777 {host_root}")
        
        # 2. Create the root run dir
        daq_run_dir = host_root / run_name
        daq_run_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Create module dirs
        for mid in spec.module_ids:
            mod_run_dir = host_root / f"module_{mid}" / run_name
            mod_run_dir.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Created {mod_run_dir}")
            
            for f_idx in range(2):
                filename = f"start_2026.dp_ph256.module_{mid}.seqno_{f_idx}.pff"
                content = os.urandom(1024) 
                f_path = mod_run_dir / filename
                f_path.write_bytes(content)
                expected_data[filename] = content
                
        # 4. CRITICAL: Force 777 recursively again after all files are written
        subprocess.run(["chmod", "-R", "777", str(host_root)], check=True)
        
        # 5. Debug check from inside the container
        for mid in spec.module_ids:
            container_path = f"/data/module_{mid}/{run_name}"
            res = container.exec_run(f"ls -ld {container_path}")
            print(f"DEBUG: Container path {container_path} check: exit={res.exit_code} output={res.output.decode().strip()}")
                
    return expected_data

@pytest.mark.asyncio
async def test_transfer_queue_validity_happy_path(
    session_fleet: Any,
    tmp_path: Path,
    ensure_clean_daq_state: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Happy path transfer test:
    1. Start run via pseti start logic (mocked HW).
    2. Populate DAQ nodes with randomized .pff files (2 per module).
    3. Stop run via pseti stop logic (enqueues job).
    4. Execute transfer state machine.
    5. Verify 100% byte accuracy on head node.
    6. Verify ledger and queue state transitions at each step.
    """
    fleet, daq_cfg_dict = session_fleet
    
    # --- Step 1: Isolation & Config ---
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    head_data_dir = tmp_path / "head_data"
    head_data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HEAD_DATA_DIR", str(head_data_dir))
    PanoPaths.ensure_state_dirs()
    
    # Update daq_config with isolated head_data_dir and persist to disk
    daq_config = config_file.DaqConfig.model_validate(daq_cfg_dict)
    daq_config.head_node_data_dir = str(head_data_dir)
    config_dir = Path(os.environ["PSETI_CONFIG"])
    (config_dir / "daq_config.json").write_text(daq_config.model_dump_json())
    
    obs_config = config_file.get_obs_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    net_config = config_file.get_network_config()
    
    run_name = f"transfer_happy_{uuid.uuid4().hex[:8]}.pffd"
    mgr = RunStateManager()
    tq = TransferQueue()

    # --- Step 2: Start Run ---
    def mocked_make_run_dirs(rn, oc, dc, quids, dtc, nc):
        # Create head node run dir locally
        run_dir = Path(dc.head_node_data_dir) / rn
        run_dir.mkdir(parents=True, exist_ok=True)
        # 1. We MUST create the directories on the DAQ nodes too.
        # Since we are in a fleet test without SSH, we use docker exec.
        for i, node in enumerate(dc.daq_nodes):
            container = fleet.containers[i].get_wrapped_container()
            # Root run dir
            container.exec_run(f"mkdir -p {node.data_dir}/{rn}")
            container.exec_run(f"chmod 777 {node.data_dir}/{rn}")
            for mid in node.module_ids:
                mpath = f"{node.data_dir}/module_{mid}/{rn}"
                container.exec_run(f"mkdir -p {mpath}")
                container.exec_run(f"chmod 777 {mpath}")

    with patch("control.start.ph_baseline_file_ok", return_value=True), \
         patch("control.start._check_daq_reachability"), \
         patch("control.start._check_quabo_reachability"), \
         patch("control.start.start_data_flow"), \
         patch("control.start.make_run_dirs", side_effect=mocked_make_run_dirs), \
         patch("control.start.util.start_hk_recorder"), \
         patch("control.start.util.write_run_name"):
         
         actual_run_name = await start_run(
             obs_config, daq_config, quabo_uids, data_config, net_config,
             run_name=run_name, no_hv=True, no_redis=True, no_data=False, no_check_daq=True
         )
         assert actual_run_name == run_name

    # Assert: Ledger is ACTIVE
    ledger = mgr.load_state()
    assert ledger.status == "ACTIVE"

    # --- Step 3: Populate Data ---
    expected_data = {}
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        for mid in spec.module_ids:
            # The directory should ALREADY exist thanks to mocked_make_run_dirs
            host_mod_run_dir = host_root / f"module_{mid}" / run_name
            for f_idx in range(2):
                filename = f"start_2026.dp_ph256.module_{mid}.seqno_{f_idx}.pff"
                content = os.urandom(1024) 
                f_path = host_mod_run_dir / filename
                f_path.write_bytes(content)
                expected_data[filename] = content
        
        # Ensure container can see them
        subprocess.run(["chmod", "-R", "777", str(host_root)], check=True)
    
    await asyncio.sleep(0.5)

    # --- Step 4: Stop Run (Enqueue) ---
    with patch("control.stop.util.stop_data_flow"), \
         patch("control.stop.util.kill_hk_recorder"):
        stop_ok = await stop_run(daq_config, net_config, quabo_uids, run=run_name)
        assert stop_ok

    # Assert: Ledger is RECORDING_ENDED
    ledger = mgr.load_state()
    assert ledger.status == "RECORDING_ENDED"

    # Assert: Job is in PENDING queue
    pending_jobs = list((tq._queue / "pending").glob("*.job.toml"))
    assert len(pending_jobs) == 1
    assert run_name in pending_jobs[0].name

    # --- Step 5: Process Job ---
    job = tq.claim()
    assert job is not None
    assert job.run_name == run_name
    
    # Assert: Job is in ACTIVE queue
    assert not list((tq._queue / "pending").glob("*.job.toml"))
    assert len(list((tq._queue / "active").glob("*.job.toml"))) == 1

    # Mock Rsync (simulates flattening)
    def simulate_rsync_from_fleet(fleet: Fleet, run_name: str, head_run_dir: Path) -> None:
        head_run_dir.mkdir(parents=True, exist_ok=True)
        for temp_dir in fleet._temp_dirs:
            host_root = Path(temp_dir)
            # 1. Root contents (hp_stdout, pss)
            daq_run_dir = host_root / run_name
            if daq_run_dir.is_dir():
                for f in daq_run_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, head_run_dir / f.name)
            # 2. Module contents (flattened as per build_rsync_cmd)
            for mod_root in host_root.glob("module_*"):
                src = mod_root / run_name
                if src.is_dir():
                    for f in src.iterdir():
                        if f.is_file():
                            shutil.copy2(f, head_run_dir / f.name)

    async def mocked_rsync(*args, **kwargs):
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b'', b''))
        proc.stdout.readline = AsyncMock(return_value=b'')
        proc.stderr.read = AsyncMock(return_value=b'')
        return proc

    from panoseti_grpc.daq_control.client import AsyncDaqControlClient
    def _get_mapped_client(host, port=50051):
        for node in daq_config.daq_nodes:
            if str(node.ip_addr) == host or (node.port_forwarding and str(node.port_forwarding.gw_ip) == host):
                return AsyncDaqControlClient(host=node.port_forwarding.gw_ip, port=node.port_forwarding.grpc_port)
        return AsyncDaqControlClient(host=host, port=port)

    # Note: We don't patch GenerateManifest because we want to test the real gRPC server logic in the containers.
    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=mocked_rsync), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        
        # Execute the state machine stages (Manifest -> Rsync -> Verify -> Cleanup -> Archive)
        success, err = await asyncio.wait_for(_process_job(job, asyncio.Event(), mgr), timeout=60.0)
        assert success, f"Transfer job failed: {err}"

    # Finalize job in queue
    tq.complete(job.run_name)

    # --- Step 6: Final Verification ---
    # Ledger should be ARCHIVED
    ledger = mgr.load_state()
    assert ledger.status == "ARCHIVED"

    # Queue should be COMPLETED
    assert not list((tq._queue / "active").glob("*.job.toml"))
    assert len(list((tq._queue / "completed").glob("*.job.toml"))) == 1

    # Verify byte accuracy on head node
    run_dir_on_head = head_data_dir / run_name
    assert run_dir_on_head.exists()
    assert (run_dir_on_head / "run_complete").exists()
    
    for filename, expected_bytes in expected_data.items():
        actual_path = run_dir_on_head / filename
        assert actual_path.exists(), f"File {filename} missing on head node"
        actual_bytes = actual_path.read_bytes()
        assert actual_bytes == expected_bytes, f"Byte accuracy failure for {filename}"

    # Verify Cleanup on DAQ nodes
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        for mod_root in host_root.glob("module_*"):
            daq_mod_run_dir = mod_root / run_name
            # .pff files should be deleted
            pff_files = list(daq_mod_run_dir.glob("*.pff"))
            assert not pff_files, f"Cleanup failed: .pff files still on DAQ node {i}"
            
            # manifest should remain (preserved pattern)
            assert (daq_mod_run_dir / "manifest.blake3").exists(), f"Manifest missing on DAQ {i}"
            
        # Root run dir metadata should remain
        assert (host_root / run_name / "meta.json").exists(), f"Metadata missing on DAQ {i}"

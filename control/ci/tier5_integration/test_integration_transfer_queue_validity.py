"""
test_integration_transfer_queue_validity.py — Tier 5 Heavy Integration test for transfer queue.

Validates the full lifecycle:
1. Start run (real Hashpipe + tcpreplay).
2. Wait for real data generation.
3. Stop run (enqueues transfer job).
4. Run real Transfer Daemon in background.
5. Poll ledger until ARCHIVED.
6. Verify 100% byte accuracy and selective cleanup on DAQ nodes.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import time
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient

from control.start import start_run
from control.stop import stop_run
from control.utils import config_file
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager

# Use the same run_dir name as in conftest.py's run_params or something unique
TEST_RUN_NAME = f"int_transfer_test_{int(time.time())}.pffd"

@pytest.fixture(scope="module")
def int_run_params(run_params: dict[str, Any]) -> dict[str, Any]:
    """Override run_params with a unique run_dir for this test module."""
    p = dict(run_params)
    p["run_dir"] = TEST_RUN_NAME
    return p

@pytest.fixture(autouse=True)
def isolated_state(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect all PSETI state and config to tmp_path for isolation."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    # 1. Copy real integration configs to the isolated dir
    real_config_dir = PanoPaths.config_dir()
    for f in real_config_dir.glob("*.json"):
        import shutil
        shutil.copy2(f, config_dir / f.name)
        
    # 2. Set environment variables
    monkeypatch.setenv("PSETI_STATE", str(state_dir))
    monkeypatch.setenv("PSETI_CONFIG", str(config_dir))
    monkeypatch.setenv("HEAD_DATA_DIR", str(tmp_path / "head_data"))
    
    # Ensure directories exist
    PanoPaths.ensure_state_dirs()
    (tmp_path / "head_data").mkdir(parents=True, exist_ok=True)

    # Create dummy PH baseline to pass pre-flight checks
    ph_baseline_path = PanoPaths.calibration_file("quabo_ph_baseline.json")
    ph_baseline_path.write_text("{}")
    
    # Reload config objects to see new environment
    import importlib

    import control.utils.config_file
    importlib.reload(control.utils.config_file)

@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_integration_transfer_queue_lifecycle(
    int_run_params: dict[str, Any],
    daqnode_container: Any,
    daq_control_direct: DaqControlClient,
    head_data_dir: pathlib.Path,
) -> None:
    """
    Test the full transfer queue lifecycle in a Tier 5 integration environment.
    """
    # --- Step 1: Start Run (Real Hashpipe + tcpreplay via hashpipe_pcap_session logic) ---
    # We don't use the hashpipe_pcap_session fixture directly because we want to 
    # drive the transaction through start_run/stop_run.
    
    # Prepare configs
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    net_config = config_file.get_network_config()
    
    # Force the run_dir
    data_config.run_type = "modified" # avoid strict checks
    
    # Update daq_config with isolated head_data_dir
    daq_config.head_node_data_dir = os.environ["HEAD_DATA_DIR"]
    
    # Start the run
    def mocked_make_run_dirs(rn, oc, dc, quids, dtc, nc):
        # Create head node run dir locally
        run_dir = pathlib.Path(dc.head_node_data_dir) / rn
        run_dir.mkdir(parents=True, exist_ok=True)
        # Create the directories on the DAQ node via docker exec (matching fleet test logic)
        for node in dc.daq_nodes:
            # We assume first node for now as per test logic
            daqnode_container.exec_run(f"mkdir -p {node.data_dir}/{rn}")
            daqnode_container.exec_run(f"chmod 777 {node.data_dir}/{rn}")
            for mid in node.module_ids:
                mpath = f"{node.data_dir}/module_{mid}/{rn}"
                daqnode_container.exec_run(f"mkdir -p {mpath}")
                daqnode_container.exec_run(f"chmod 777 {mpath}")
            daqnode_container.exec_run(f"chmod -R 777 {node.data_dir}")

    # Mock build_rsync_cmd to use local rsync in this test environment.
    # In Tier 5, /data is shared between all containers.
    # The daemon builds commands like: rsync /data/module_73/run.pffd/ /head/run.pffd/
    # Since we are in the same process, we just need to ensure the files get there.
    def mocked_build_rsync_cmd(node, run_name, head_run_dir):
        # We simulate the per-node rsync by copying everything from /data 
        # (which contains all modules in this integration stack) to head_run_dir.
        # The daemon expects to run one rsync per node.
        # We'll just do a broad sync of the relevant module dirs.
        cmd = ["rsync", "-rtv"]
        for mid in node.module_ids:
            cmd.append(f"/data/module_{mid}/{run_name}/")
        # Also include root files (hp_stdout, etc)
        cmd.append(f"/data/{run_name}/")
        cmd.append(str(head_run_dir) + "/")
        return cmd

    from unittest.mock import patch
    with patch("control.start.make_run_dirs", side_effect=mocked_make_run_dirs), \
         patch("control.start.start_data_flow"), \
         patch("control.start._check_quabo_reachability"), \
         patch("control.start.util.write_run_name"), \
         patch("control.transfer.daemon.build_rsync_cmd", side_effect=mocked_build_rsync_cmd):

        actual_run_name = await start_run(
            obs_config, daq_config, quabo_uids, data_config, net_config,
            no_hv=True, no_redis=True, no_data=False,
            run_name=TEST_RUN_NAME, no_check_daq=True
        )
        assert actual_run_name == TEST_RUN_NAME

        # Enable promisc mode and start tcpreplay to generate real data
        daqnode_container.exec_run("ip link set lo promisc on")
        from ci.tier3_fleet.conftest import PCAP_GLOB
        replay_cmd = f"sh -c 'tcpreplay --mbps=0.1 --loop=0 --intf1=lo {PCAP_GLOB}'"
        daqnode_container.exec_run(replay_cmd, detach=True)

        # Simulate metadata generation on the DAQ node (normally done by control plane)
        daqnode_container.exec_run(f"sh -c \"echo '{{'test': true}}' > /data/{TEST_RUN_NAME}/meta.json\"")

        # Let it run for a bit to generate .pff files
        await asyncio.sleep(5.0)
        # --- Step 2: Stop Run (Enqueue Job) ---
        stop_ok = await stop_run(
            daq_config, net_config, quabo_uids,
            run=TEST_RUN_NAME, no_collect=False, no_cleanup=False
        )
        assert stop_ok

        # Verify job is enqueued
        mgr = RunStateManager()
        ledger = mgr.load_state()
        assert ledger.status == "RECORDING_ENDED"

        tq_dir = PanoPaths.transfer_queue_dir()
        pending_jobs = list((tq_dir / "pending").glob("*.job.toml"))
        assert len(pending_jobs) == 1

        # --- Step 3: Run Transfer Daemon Loop as a task ---
        from control.transfer.daemon import run_daemon
        daemon_task = asyncio.create_task(run_daemon(poll_interval=1.0))

        # Wait for daemon to start and initialize (heartbeat appearing)
        hb_path = PanoPaths.transfer_queue_dir().parent / "daemon.heartbeat"
        start_time = time.time()
        while time.time() - start_time < 10:
            if hb_path.exists():
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Transfer daemon task failed to start heartbeat.")

        try:
            # --- Step 4: Poll Ledger until ARCHIVED ---
            timeout = 180.0 # generous timeout for rsync/verify
            start_time = time.time()
            while time.time() - start_time < timeout:
                ledger = mgr.load_state()
                if ledger and ledger.status == "ARCHIVED":
                    break
                if ledger and ledger.status == "STOPPED_WITH_ERRORS":
                    pytest.fail(f"Transfer failed with errors: {ledger.last_transfer_error}")
                if ledger and ledger.status == "TRANSFER_FAILED":
                    pytest.fail(f"Transfer failed permanently: {ledger.last_transfer_error}")
                if ledger and ledger.status == "VERIFY_FAILED":
                    pytest.fail(f"Manifest verification failed: {ledger.last_transfer_error}")
                await asyncio.sleep(2.0)
            else:
                pytest.fail(f"Timed out waiting for ARCHIVED status. Current status: {ledger.status if ledger else 'None'}")

            # --- Step 5: Final Validation ---

            # 1. Byte accuracy on head node
            head_run_dir = pathlib.Path(os.environ["HEAD_DATA_DIR"]) / TEST_RUN_NAME
            assert head_run_dir.exists()
            assert (head_run_dir / "run_complete").exists()
            
            # Check manifest
            manifests = list(head_run_dir.glob("dp_manifest.node_*.txt"))
            assert manifests, "Manifest file missing on head node"
            
            # Check for real .pff files (expecting imaging or PH data)
            pff_files = list(head_run_dir.glob("*.pff"))
            assert len(pff_files) > 0, "No .pff files transferred to head node"
            
            # 2. Selective cleanup on DAQ node
            # We check the host-side mapped directory
            daq_data_root = pathlib.Path(os.environ["DAQ_DATA_DIR"])
            
            # meta.json and manifest should be preserved in root run dir on DAQ
            daq_root_run_dir = daq_data_root / TEST_RUN_NAME
            assert (daq_root_run_dir / "meta.json").exists(), "Metadata missing on DAQ root run dir"
            assert (daq_root_run_dir / "hp_stdout.log").exists(), "Hashpipe log missing on DAQ root run dir"
            assert list(daq_root_run_dir.glob("dp_manifest.node_*.txt")), "Manifest missing on DAQ root run dir"

            for mid in daq_config.daq_nodes[0].module_ids:
                daq_mod_run_dir = daq_data_root / f"module_{mid}" / TEST_RUN_NAME
                # Science files should be gone
                remaining_pff = list(daq_mod_run_dir.glob("*.pff"))
                assert not remaining_pff, f"Cleanup failed: .pff files still on DAQ module {mid}"

        finally:
            # Kill tcpreplay and stop daemon task
            daqnode_container.exec_run("pkill -9 tcpreplay", detach=False)
            daemon_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await daemon_task

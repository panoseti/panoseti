"""Shared utilities and fixtures for Tier 5 integration transfer queue tests."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from control.start import start_run
from control.stop import stop_run
from control.utils import config_file
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import DaqConfig


def setup_isolated_integration_transfer_env(tmp_path: Path, monkeypatch: Any) -> tuple[Path, DaqConfig]:
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
    
    # In Docker CI (Tier 5), DAQ_DATA_DIR should always point to the shared /data volume
    if os.environ.get("IN_DOCKER_CI") == "1":
        monkeypatch.setenv("DAQ_DATA_DIR", "/data")
    
    # Ensure directories exist
    PanoPaths.ensure_state_dirs()
    head_data_dir = tmp_path / "head_data"
    head_data_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy PH baseline to pass pre-flight checks
    ph_baseline_path = PanoPaths.calibration_file("quabo_ph_baseline.json")
    ph_baseline_path.write_text("{}")
    
    # Reload config objects to see new environment
    import importlib

    import control.utils.config_file
    importlib.reload(control.utils.config_file)
    
    daq_config = config_file.get_daq_config()
    daq_config.head_node_data_dir = str(head_data_dir)
    return head_data_dir, daq_config

def mocked_make_run_dirs_factory(daqnode_container: Any):
    def mocked_make_run_dirs(rn, oc, dc, quids, dtc, nc):
        # Create head node run dir locally
        head_run_dir = Path(dc.head_node_data_dir) / rn
        head_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the directories on the DAQ node shared volume directly
        for node in dc.daq_nodes:
            # DAQ node data_dir is usually /data
            daq_data_path = Path(node.data_dir)
            
            # Root run dir
            (daq_data_path / rn).mkdir(parents=True, exist_ok=True)
            
            # Module subdirs
            for mid in node.module_ids:
                (daq_data_path / f"module_{mid}" / rn).mkdir(parents=True, exist_ok=True)
                
            # Ensure permissions allow both containers to read/write
            # (In Docker volumes, this is usually 777 or shared group)
            import subprocess
            subprocess.run(["chmod", "-R", "777", str(daq_data_path)], check=False)
            
    return mocked_make_run_dirs

def mocked_build_rsync_cmd(node, run_name, head_run_dir):
    # We simulate the per-node rsync by copying everything from /data 
    cmd = ["rsync", "-rtv"]
    for mid in node.module_ids:
        cmd.append(f"/data/module_{mid}/{run_name}/")
    cmd.append(f"/data/{run_name}/")
    cmd.append(str(head_run_dir) + "/")
    return cmd

async def generate_integration_run(run_name: str, daq_config: DaqConfig, daqnode_container: Any) -> None:
    """Start run, generate real data via tcpreplay, simulate metadata, and stop run."""
    obs_config = config_file.get_obs_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    net_config = config_file.get_network_config()
    
    data_config.run_type = "modified" # avoid strict checks

    hostname = getattr(daqnode_container, "name", "test-node")
    
    with patch("control.start.make_run_dirs", side_effect=mocked_make_run_dirs_factory(daqnode_container)), \
         patch("control.start.start_data_flow"), \
         patch("control.start._check_quabo_reachability"), \
         patch("control.start.util.write_run_name"), \
         patch("control.transfer.daemon.build_rsync_cmd", side_effect=mocked_build_rsync_cmd):

        actual_run_name = await start_run(
            obs_config, daq_config, quabo_uids, data_config, net_config,
            no_hv=True, no_redis=True, no_data=False,
            run_name=run_name, no_check_daq=True
        )
        assert actual_run_name == run_name

        # Enable promisc mode and start tcpreplay to generate real data
        daqnode_container.exec_run("ip link set lo promisc on")
        from ci.software_only.conftest import PCAP_GLOB
        replay_cmd = f"sh -c 'tcpreplay --mbps=0.1 --loop=0 --intf1=lo {PCAP_GLOB}'"
        daqnode_container.exec_run(replay_cmd, detach=True)

        # Simulate metadata and logs generation on the DAQ node.
        # Since /data is a shared volume between int-tester and daqnode, we can write directly.
        daq_run_path = Path("/data") / run_name
        daq_run_path.mkdir(parents=True, exist_ok=True)
        (daq_run_path / "meta.json").write_text('{"test": true}')
        (daq_run_path / f"hp_stdout_{hostname}.log").touch()
        (daq_run_path / "dp_manifest.node_test.txt").touch()

        # Let it run for a bit to generate .pff files
        await asyncio.sleep(5.0)

        # --- Step 2: Stop Run (Enqueue Job) ---
        stop_ok = await stop_run(
            daq_config, net_config, quabo_uids,
            run=run_name, no_collect=False, no_cleanup=False
        )
        assert stop_ok
        
        # Kill tcpreplay for this run
        daqnode_container.exec_run("pkill -9 tcpreplay", detach=False)

def verify_integration_transfer_accuracy(head_data_dir: Path, run_name: str, daq_config: DaqConfig) -> None:
    head_run_dir = head_data_dir / run_name
    assert head_run_dir.exists()
    assert (head_run_dir / "run_complete").exists()
    
    # Check manifest
    manifests = list(head_run_dir.glob("dp_manifest.node_*.txt"))
    assert manifests, f"Manifest file missing on head node for {run_name}"
    
    # Check for real .pff files
    pff_files = list(head_run_dir.glob("*.pff"))
    assert len(pff_files) > 0, f"No .pff files transferred to head node for {run_name}"
    
    # 2. Selective cleanup on DAQ node
    daq_data_root = Path(os.environ["DAQ_DATA_DIR"])
    daq_root_run_dir = daq_data_root / run_name
    assert (daq_root_run_dir / "meta.json").exists(), f"Metadata missing on DAQ root run dir for {run_name}"
    
    # Use glob for node-specific logs
    logs = list(daq_root_run_dir.glob("hp_stdout_*.log"))
    assert len(logs) > 0, f"Hashpipe log missing on DAQ root run dir for {run_name}"
    
    assert list(daq_root_run_dir.glob("dp_manifest.node_*.txt")), f"Manifest missing on DAQ root run dir for {run_name}"

    for mid in daq_config.daq_nodes[0].module_ids:
        daq_mod_run_dir = daq_data_root / f"module_{mid}" / run_name
        remaining_pff = list(daq_mod_run_dir.glob("*.pff"))
        assert not remaining_pff, f"Cleanup failed: .pff files still on DAQ module {mid} for {run_name}"

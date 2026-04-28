"""Shared utilities and fixtures for transfer queue tests."""
import asyncio
import os
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

from panoseti_grpc.daq_control.client import AsyncDaqControlClient

from ci.fixtures.fleet import Fleet
from control.start import start_run
from control.stop import stop_run
from control.utils import config_file
from control.utils.paths import PanoPaths


def setup_isolated_transfer_env(tmp_path: Path, monkeypatch: Any, daq_cfg_dict: dict[str, Any]) -> tuple[Path, config_file.DaqConfig]:
    """Isolates the PSETI state and config for a transfer test."""
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
    
    return head_data_dir, daq_config

def get_mapped_client_factory(daq_config: config_file.DaqConfig):
    """Returns a factory function for AsyncDaqControlClient that handles port forwarding."""
    def _get_mapped_client(host, port=50051):
        for node in daq_config.daq_nodes:
            # Match by internal IP (if non-forwarded) OR by gateway IP + port (if forwarded)
            if str(node.ip_addr) == host:
                 return AsyncDaqControlClient(host=node.port_forwarding.gw_ip, port=node.port_forwarding.grpc_port)
            if node.port_forwarding and str(node.port_forwarding.gw_ip) == host and node.port_forwarding.grpc_port == port:
                return AsyncDaqControlClient(host=node.port_forwarding.gw_ip, port=node.port_forwarding.grpc_port)
        return AsyncDaqControlClient(host=host, port=port)
    return _get_mapped_client

def simulate_rsync_from_fleet(fleet: Fleet, run_name: str, head_run_dir: Path) -> None:
    """Simulates the effect of rsync by copying files from the fleet's host paths to the head node."""
    head_run_dir.mkdir(parents=True, exist_ok=True)
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        # 1. Root contents (hp_stdout, pss, meta.json)
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

async def generate_mocked_run(fleet: Fleet, daq_config: config_file.DaqConfig, run_name: str) -> dict[str, bytes]:
    """Starts a run, generates fake .pff files and metadata on containers, and stops the run."""
    obs_config = config_file.get_obs_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    net_config = config_file.get_network_config()

    def mocked_make_run_dirs(rn, oc, dc, quids, dtc, nc):
        # Create head node run dir locally
        run_dir = Path(dc.head_node_data_dir) / rn
        run_dir.mkdir(parents=True, exist_ok=True)
        # 1. Create the directories on the DAQ nodes via docker exec
        for i, node in enumerate(dc.daq_nodes):
            container = fleet.containers[i].get_wrapped_container()
            # Root run dir on DAQ node
            container.exec_run(f"mkdir -p {node.data_dir}/{rn}")
            container.exec_run(f"chmod 777 {node.data_dir}/{rn}")
            # Module-specific dirs on DAQ node
            for mid in node.module_ids:
                mpath = f"{node.data_dir}/module_{mid}/{rn}"
                container.exec_run(f"mkdir -p {mpath}")
                container.exec_run(f"chmod 777 {mpath}")
                # Also ensure the parent module_mid dir exists
                container.exec_run(f"chmod 777 {node.data_dir}/module_{mid}")
            
            # Final broad chmod to be absolutely sure
            container.exec_run(f"chmod -R 777 {node.data_dir}")

    # Create dummy PH baseline to pass pre-flight checks
    ph_baseline_path = PanoPaths.calibration_file("quabo_ph_baseline.json")
    ph_baseline_path.write_text("{}")

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

    expected_data = {}
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        container = fleet.containers[i].get_wrapped_container()
        
        # 1. Add metadata to root run dir
        meta_file = host_root / run_name / "meta.json"
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text('{"test": true}')
        
        for mid in spec.module_ids:
            # The directory should ALREADY exist thanks to mocked_make_run_dirs
            host_mod_run_dir = host_root / f"module_{mid}" / run_name
            host_mod_run_dir.mkdir(parents=True, exist_ok=True)
            
            for f_idx in range(2):
                # Unique name across the whole fleet
                filename = f"start_2026.dp_ph256.module_{mid}.seqno_{f_idx}.pff"
                content = os.urandom(1024) 
                f_path = host_mod_run_dir / filename
                f_path.write_bytes(content)
                expected_data[filename] = content
                
                # Fix permissions from inside so gRPC server (root) can read them
                container.exec_run(f"chmod 666 /data/module_{mid}/{run_name}/{filename}")
        
        # Final safety chmod
        container.exec_run(f"chmod -R 777 /data/{run_name}")
        for mid in spec.module_ids:
            container.exec_run(f"chmod -R 777 /data/module_{mid}/{run_name}")
        
        # Explicit sync to flush caches inside the container
        container.exec_run("sync")
    
    # 1s settling period for mount propagation
    await asyncio.sleep(1.0)

    # --- Step 4: Stop Run (Enqueue) ---
    with patch("control.stop.util.stop_data_flow"), \
         patch("control.stop.util.kill_hk_recorder"):
        stop_ok = await stop_run(daq_config, net_config, quabo_uids, run=run_name)
        assert stop_ok
        
    return expected_data

def verify_head_node_accuracy(head_data_dir: Path, run_name: str, expected_data: dict[str, bytes]) -> None:
    """Verifies byte accuracy on the head node."""
    run_dir_on_head = head_data_dir / run_name
    assert run_dir_on_head.exists()
    assert (run_dir_on_head / "run_complete").exists()
    assert (run_dir_on_head / "meta.json").exists()
    
    for filename, expected_bytes in expected_data.items():
        actual_path = run_dir_on_head / filename
        assert actual_path.exists(), f"File {filename} missing on head node"
        actual_bytes = actual_path.read_bytes()
        assert actual_bytes == expected_bytes, f"Byte accuracy failure for {filename}"

"""Shared utilities and fixtures for transfer queue tests in v2."""
from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import patch

from panoseti_grpc.daq_control.client import AsyncDaqControlClient

from ci.software_only_v2.orchestrator.fleet import Fleet
from control.start import start_run
from control.stop import stop_run
from control.utils import config_file
from control.utils.paths import PanoPaths


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
    
    exclude_patterns = ["dp_manifest.node_*.txt", "manifest.*"]

    def _should_exclude(name: str) -> bool:
        import fnmatch
        return any(fnmatch.fnmatch(name, pat) for pat in exclude_patterns)

    # In v2, self._temp_dirs matches the order of self.daq_nodes
    for _i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        # 1. Root contents (hp_stdout, pss, meta.json)
        daq_run_dir = host_root / run_name
        if daq_run_dir.is_dir():
            for f in daq_run_dir.iterdir():
                if f.is_file() and not _should_exclude(f.name):
                    shutil.copy2(f, head_run_dir / f.name)
        # 2. Module contents (flattened as per build_rsync_cmd)
        for mod_root in host_root.glob("module_*"):
            src = mod_root / run_name
            if src.is_dir():
                for f in src.iterdir():
                    if f.is_file() and not _should_exclude(f.name):
                        shutil.copy2(f, head_run_dir / f.name)

async def generate_mocked_run(fleet: Fleet, daq_config: config_file.DaqConfig, run_name: str, file_size_kb: int = 1) -> dict[str, bytes]:
    """Starts a run, generates fake .pff files and metadata on containers, and stops the run."""
    obs_config = config_file.get_obs_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    net_config = config_file.get_network_config()

    # Create dummy PH baseline to pass pre-flight checks
    ph_baseline_path = PanoPaths.calibration_file("quabo_ph_baseline.json")
    import json
    dummy_data = {
        "date": "2024-01-01T00:00:00",
        "quabos": []
    }
    ph_baseline_path.write_text(json.dumps(dummy_data))

    with patch("control.start.ph_baseline_file_ok", return_value=True), \
         patch("control.start._check_daq_reachability"), \
         patch("control.start._check_quabo_reachability"), \
         patch("control.start.start_data_flow"), \
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

        # 1. Add metadata to root run dir
        meta_file = host_root / run_name / "meta.json"
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text('{"test": true}')

        node_spec = fleet.workspace.topology.daq.daq_nodes[i]
        for mid in node_spec.module_ids:
            host_mod_run_dir = host_root / f"module_{mid}" / run_name
            host_mod_run_dir.mkdir(parents=True, exist_ok=True)

            for f_idx in range(2):
                filename = f"{run_name}.dp_ph256.module_{mid}.seqno_{f_idx}.pff"
                content = os.urandom(file_size_kb * 1024)
                f_path = host_mod_run_dir / filename
                f_path.write_bytes(content)
                expected_data[filename] = content

                fleet.exec_in_node(i, f"chmod 666 /data/module_{mid}/{run_name}/{filename}")

        fleet.exec_in_node(i, "sync")
    
    await asyncio.sleep(0.5)

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

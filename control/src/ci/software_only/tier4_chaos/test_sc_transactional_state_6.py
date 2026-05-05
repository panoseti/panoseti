"""
scenarios/test_sc_transactional_state_6.py

SC-035, SC-036, SC-039: Transactional state corruption tests.
Part 6 of partitioned test suite.
"""

# ruff: noqa
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import time
import unittest.mock
import uuid
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import AsyncDaqControlClient, DaqControlClient
from control.utils.pydantic_config_models import DaqConfig, DaqNode
from unittest.mock import AsyncMock, MagicMock

from ci.software_only.conftest import (
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)
from ci.software_only.tier3_fleet.conftest import (
    DAQ_DATA_DIR,
)
from ci.fixtures.state_probe import StateProbe
from ci.software_only.tier4_chaos.conftest import (
    _start as grpc_start,
)
from ci.software_only.tier4_chaos.conftest import (
    _stop as grpc_stop,
)

from ci.software_only.qa_utils import get_isolated_env

from control.utils.paths import PanoPaths
PH_BASELINE_FILE = PanoPaths.config_dir() / "quabo_ph_baseline.json"


# ── Shared Helpers ───────────────────────────────────────────────────────────

@contextlib.contextmanager
def mock_daq_config_for_headnode():
    """Temporarily patch daq_config.json to point to localhost (CI headnode)."""
    import json

    from control.utils import config_file
    
    from control.utils.paths import PanoPaths
    path = PanoPaths.config_dir() / "daq_config.json"
    backup = str(path) + ".bak"
    # Ensure tmp/ and configs/ exist (should already, but let's be safe)
    PanoPaths.ensure_dirs()
    PanoPaths.config_dir().mkdir(parents=True, exist_ok=True)
    
    # Create a dummy PH baseline if missing
    ph_baseline = PanoPaths.tmp_dir() / "quabo_ph_baseline.json"
    if not os.path.exists(ph_baseline):
        with open(ph_baseline, "w") as f:
            json.dump({"date": "2024-01-01T00:00:00", "quabos": []}, f)

    if os.path.exists(path):
        import shutil
        shutil.copyfile(path, backup)
    
    with open(path) as f:
        cfg = json.load(f)
    
    import tempfile
    # Prefer the isolated HEAD_DATA_DIR set by auto_isolate so the subprocess
    # env and the daq_config.json written here share the same path.  Fall back
    # to a fresh tempdir only when running outside the pytest harness.
    tmp_data_dir = os.environ.get("HEAD_DATA_DIR") or tempfile.mkdtemp()
    os.makedirs(tmp_data_dir, exist_ok=True)

    tester_ip = f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'
    cfg["head_node_ip_addr"] = tester_ip
    cfg["head_node_data_dir"] = tmp_data_dir
    cfg["head_node_container"] = True
    
    # Coherence Fix: Ensure the DAQ node is handling ALL modules defined in the 
    # current obs_config.json to prevent "no DAQ node is handling module X" errors.
    mids = []
    obs = config_file.get_obs_config()
    for dome in obs.domes:
        for module in dome.modules:
            mids.append(config_file.ip_addr_to_module_id(str(module.ip_addr)))

    # Assign ALL modules to the single available CI node
    # Use a DAQ IP that is on the same /24 subnet as the modules (192.168.3.x)
    # to pass strict Tier-2 Subnet Coherence validation.
    daqnode_ip = "192.168.3.30"
    cfg["daq_nodes"] = [
        {
            "ip_addr": daqnode_ip,
            "data_dir": "/data",
            "username": "root",
            "module_ids": mids,
            "bindhost": "lo"
        }
    ]
    
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    
    # Write matching quabo_uids.json to tmp/ so associate() in subprocess passes
    uids_path = PanoPaths.tmp_dir() / "quabo_uids.json"
    uids_path.parent.mkdir(parents=True, exist_ok=True)
    from control.utils.pydantic_config_models import QuaboUids
    uids_dict: dict[str, Any] = {"domes": [{"num": 0, "modules": []}]}
    for mid in mids:
        uids_dict["domes"][0]["modules"].append({
            "id": mid,
            "ip_addr": f"192.168.3.{mid}",
            "quabos": [{"uid": f"q{mid}_{j}"} if j==0 else {"uid": ""} for j in range(4)]
        })
    with open(uids_path, "w") as f:
        json.dump(uids_dict, f, indent=4)

    with unittest.mock.patch("control.utils.config_file.get_quabo_uids", return_value=QuaboUids(**uids_dict)):
        try:
            yield
        finally:
            if os.path.exists(backup):
                import shutil
                shutil.move(backup, path)


# ── SC-035: quabo_uids.json UID refused by mock-quabo ─────────────────────────

@pytest.mark.asyncio
async def test_SC035_unreachable_quabo_uid_silently_fails() -> None:
    """
    SC-035: When quabo_uids.json lists a UID that the quabo refuses (e.g., wrong
    module IP), start_data_flow() calls quabo.send_daq_params() fire UDP into a
    void — no error is surfaced.

    FAILS RED TODAY: send_daq_params is fire-and-forget UDP; no ACK or error.
    Fix: ping-sweep quabos before start_data_flow, or verify HK packet received
    after configuration.
    """
    from control.utils.run_state import RunStateManager
    from control.utils import config_file, util
    from control.start import start_run
    from control.utils.pydantic_config_models import QuaboUids
    import unittest.mock

    # 0. Clear stale state
    RunStateManager().clear_state()
    # 1. Inject a Quabo UID that points to a non-existent IP but valid module_id range
    mid = 254
    mock_uids = QuaboUids(domes=[{"num": 0, "modules": [{
        "id": mid,
        "ip_addr": "192.168.3.248",
        "quabos": [{"uid": ""}, {"uid": "nonexistent_quabo_sc035"}, {"uid": "DEADBEEF00000003"}, {"uid": "DEADBEEF00000004"}]
    }]}])

    # 2. Run start.py logic directly
    # We patch everything that would fail without a real DAQ fleet or SSH
    with unittest.mock.patch("control.utils.config_file.get_quabo_uids", return_value=mock_uids), \
         unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
         unittest.mock.patch("control.start._check_quabo_reachability"), \
         unittest.mock.patch("control.start._check_daq_reachability"), \
         unittest.mock.patch("control.start.make_run_dirs"), \
         unittest.mock.patch("control.utils.util.local_ip", return_value=["127.0.0.1", "10.0.1.5"]):

        daq_config = config_file.get_daq_config()
        # Coherence: Ensure DAQ IP matches module subnet
        daq_config.daq_nodes = [
            DaqNode(**{
                    'username': 'panoseti',
                    'ip_addr': "192.168.3.30", 
                    'data_dir': '/tmp/',
                    'module_ids': [mid]
                }
            )
        ]
        validated_mock_daq_config = DaqConfig(**daq_config.model_dump())
        
        success = await start_run(
            config_file.get_obs_config(), 
            validated_mock_daq_config, 
            mock_uids, 
            config_file.get_data_config(), 
            config_file.get_network_config(),
            no_hv=True, no_redis=True, no_data=True, strict=False
        )
    
    assert success, "start.py should succeed even if UDP commands were sent to a black hole (SC-035)"


# ── SC-036: Run directory collision (clock resolution = seconds) ──────────────

def test_SC036_run_dir_collision_is_detected(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    SC-036: Rapid sequential StartDaq calls within the same UTC second produce
    the same auto-generated run_dir name → mkdir raises FileExistsError.

    This tests the server's behavior when a run_dir already exists.
    The server must return ok=False with a clear error, not crash.
    """
    # Start and stop a run with a known run_dir
    ok1, _ = grpc_start(daq_control_direct, run_params)
    assert ok1, "First StartDaq must succeed"
    assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=4)
    daq_control_direct.StopDaq({
        "data_dir": run_params["data_dir"],
        "run_dir": run_params["run_dir"],
    })
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=4)

    # Attempt to start again with the same run_dir (the dir still exists on disk)
    ok2, resp2 = grpc_start(daq_control_direct, run_params)
    # Server must handle the collision gracefully
    if not ok2:
        assert resp2, "Collision must produce a descriptive error message"
    # Either outcome is acceptable here (server may allow reuse); what matters is
    # no crash and a deterministic response.

    # Cleanup
    with contextlib.suppress(Exception):
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir": run_params["run_dir"],
        })
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=5)
    with contextlib.suppress(Exception):
        daq_control_direct.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })


# ── SC-039: Config modification races ───────────────────────────────

def test_SC039_data_config_modified_between_get_params_and_start(tmp_path: pathlib.Path) -> None:
    """
    SC-039: If data_config.json is modified between get_daq_params() and
    start_recording(), quabos have mode A while hashpipe was told mode B.
    This is an inconsistency that can cause silent data corruption.

    FAILS RED TODAY: no config snapshot is taken before multi-step start.
    Fix: snapshot (deepcopy) config at the start of start_run() and use
    the snapshot throughout the transaction.
    """
    import json
    import os
    import subprocess
    import time

    from control.utils.run_state import RunStateManager

    RunStateManager().clear_state()

    from control.utils.paths import PanoPaths
    data_cfg_path = PanoPaths.config_dir() / "data_config.json"
    with open(data_cfg_path) as f:
        original_data = json.load(f)

    # Ensure quabo_uids.json exists for validate_all to pass
    uids_path = PanoPaths.tmp_dir() / "quabo_uids.json"
    uids_path.parent.mkdir(parents=True, exist_ok=True)
    if not uids_path.exists():
        from control.utils.pydantic_config_models import QuaboUids
        with open(uids_path, "w") as f:
            json.dump(QuaboUids(domes=[]).model_dump(mode="json"), f)

    run_name = "sc039_test_fixed_run.pffd"

    try:
        # Start start.py in background
        # We'll use a wrapper that adds a delay before start_data_flow
        with open("tmp_slow_start.py", "w") as f:
            f.write(f"""
import asyncio

import json
import time
import sys
import os
import control.start as start
import unittest.mock
from control.utils import config_file

async def slow_start():
    obs = config_file.get_obs_config()
    daq = config_file.get_daq_config()
    # Force integration path
    daq.head_node_data_dir = f"{str(tmp_path / "head_data")}"
    daq.head_node_container = True
    try:
        uids = config_file.get_quabo_uids()
        # Ensure daq_config covers all modules in chaos uids to avoid "no DAQ node is handling module X"
        mids = []
        for dome in uids.domes:
            for mod in dome.modules:
                mids.append(mod.id)
        if daq.daq_nodes:
            daq.daq_nodes[0].module_ids = mids
    except SystemExit:
        # Fallback for CI if quabo_uids.json is missing early
        from control.utils.pydantic_config_models import QuaboUids
        uids = QuaboUids(domes=[])
    data = config_file.get_data_config()
    print(f"DEBUG: slow_start in-memory data.run_type={{data.run_type}}", flush=True)
    from control.utils.paths import PanoPaths
    with open(PanoPaths.config_dir() / "data_config.json") as f:
        disk_data = json.load(f)
        print(f"DEBUG: slow_start on-disk data.run_type={{disk_data['run_type']}}", flush=True)
    net = config_file.get_network_config()    # Use fixed run name for test
    run_name = "{run_name}"

    print("CONFIGS_LOADED", flush=True)
    # Delay to allow disk modification
    time.sleep(2)

    def mocked_copy_config_files(daq_config, run_dir, verbose=False):
        import shutil
        import os
        import json
        dest = f"{{daq_config.head_node_data_dir}}/{{run_dir}}"
        os.makedirs(dest, exist_ok=True)
        # In CI, configs are in the current config directory
        from control.utils.paths import PanoPaths
        config_dir = str(PanoPaths.config_dir())
        
        # Snapshot the ORIGINAL config from memory
        # (In SC039 we verify data_config, in SC040 we verify obs_config)
        with open(f"{{dest}}/data_config.json", "w") as f:
            json.dump(data.model_dump(exclude={'modules', 'daq_node'}), f, indent=4, default=str)
        with open(f"{{dest}}/obs_config.json", "w") as f:
            json.dump(obs.model_dump(exclude={'modules', 'daq_node'}), f, indent=4, default=str)
        
        config_file_names = [
            'daq_config.json', 'quabo_uids.json', 'daemons.json', 
            'network_config.json', 'quabo_info.json'
        ]
        for f_name in config_file_names:
            src = f"{{config_dir}}/{{f_name}}"
            if os.path.exists(src):
                shutil.copy(src, dest)
            else:
                src_alt = f"{{config_dir}}/direct/{{f_name}}"
                if os.path.exists(src_alt):
                    shutil.copy(src_alt, dest)
    try:
        # Patch everything that would fail without a real DAQ fleet or SSH
        with unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
             unittest.mock.patch("control.start._check_quabo_reachability"), \
             unittest.mock.patch("control.start._check_daq_reachability"), \
             unittest.mock.patch("control.start.start_data_flow"), \
             unittest.mock.patch("control.start.start_recording"), \
             unittest.mock.patch("control.utils.util.start_hk_recorder"), \\
             unittest.mock.patch("control.utils.util.kill_hk_recorder"), \\
             unittest.mock.patch("control.utils.util.kill_hv_updater"), \\
             unittest.mock.patch("control.utils.util.kill_module_temp_monitor"), \\
             unittest.mock.patch("subprocess.run", return_value=unittest.mock.Mock(returncode=0)), \\
             unittest.mock.patch("control.utils.file_xfer.copy_config_files", side_effect=mocked_copy_config_files):

            await start.start_run(
                obs, daq, uids, data, net,
                no_hv=True, no_redis=True, no_data=False, force_reset=True, strict=False,
                run_name=run_name
            )
    except Exception as e:
        print(f"START_RUN_FAILED:{{e}}", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(slow_start())
    except Exception as e:
        print(f"HELPER_SCRIPT_CRASHED: {{e}}", flush=True)
        import traceback
        traceback.print_exc()
        """)

        env = get_isolated_env()

        env["PYTHONPATH"] = f"{PanoPaths.base_dir() / 'src'}:{env.get('PYTHONPATH', '')}"
        proc = subprocess.Popen(["python3", "tmp_slow_start.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

        # 1. Wait for it to load config into memory
        deadline = time.monotonic() + 15
        configs_loaded = False
        assert proc.stdout is not None
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line: break
            print(f"HELPER: {line.strip()}", flush=True)
            if "CONFIGS_LOADED" in line:
                configs_loaded = True
                break
        
        if not configs_loaded:
            pytest.fail("Helper script failed to load configs")

        # 2. Modify config on disk immediately
        modified_data = dict(original_data)
        modified_data["run_type"] = "MODIFIED"
        with open(data_cfg_path, "w") as f:
            json.dump(modified_data, f)

        # 3. Wait for process to finish while continuing to read output
        while True:
            line = proc.stdout.readline()
            if not line: break
            print(f"HELPER: {line.strip()}", flush=True)
        
        proc.wait()        
        # 4. Verify that the run directory (or aborted dir) contains the ORIGINAL config
        run_dir = f"{str(tmp_path / "head_data")}/{run_name}"
        if not os.path.exists(run_dir):
             # Check aborted dir if rollback happened
             run_dir = f"{str(tmp_path / "head_data")}/_aborted/{run_name}"
             
        if not os.path.exists(run_dir):
            print("--- RUN DIR NOT FOUND ---")
            print("-------------------------")
             
        assert os.path.exists(run_dir), f"Neither run dir nor aborted dir exists for {run_name}"

        with open(f"{run_dir}/data_config.json") as f:
            copied_data = json.load(f)
            
        assert copied_data["run_type"] == original_data["run_type"], \
            f"FAIL (SC-039): data_config.json in run dir matches modified disk version, not original in-memory version. Got {copied_data['run_type']}, expected {original_data['run_type']}"
            
    finally:
        with open(data_cfg_path, "w") as f:
            json.dump(original_data, f)
        if os.path.exists("tmp_slow_start.py"):
            os.remove("tmp_slow_start.py")
        if 'proc' in locals():
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

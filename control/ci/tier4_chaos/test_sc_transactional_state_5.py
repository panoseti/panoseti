"""
scenarios/test_sc_transactional_state_5.py

SC-040, SC-015: Transactional state corruption tests.
Part 5 of partitioned test suite.
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
from unittest.mock import AsyncMock, MagicMock

from ci.tier3_fleet.conftest import (  # noqa: E402
    DAQ_DATA_DIR,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)
from ci.fixtures.state_probe import StateProbe  # noqa: E402

from ci.tier4_chaos.conftest import (  # noqa: E402
    _start as grpc_start,
)
from ci.tier4_chaos.conftest import (  # noqa: E402
    _stop as grpc_stop,
)

from ci.qa_utils import get_isolated_env

from control.utils.paths import PanoPaths
INTERLEAVE_PID_FILE = PanoPaths.tmp_dir() / "interleave.pid"
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
            json.dump({"quabos": []}, f)

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


# ── SC-040: Obs config timing mode change ────────────────────────────────────

@pytest.mark.xfail(reason="Known Bug: start.py reloads obs_config from disk instead of using session_start snapshot")
def test_SC040_obs_config_timing_mode_change_between_session_and_run() -> None:
    """
    SC-040: If timing_mode in obs_config.json changes between session_start and
    start.py, quabo 0 still runs the old WR firmware but start.py configures for
    GNSS (or vice versa). Result: inconsistent timing across modules.

    FAILS RED TODAY: start.py reloads obs_config fresh and does not compare
    against the session_start snapshot.
    Fix: write a session_config_snapshot.json at session_start; validate it
    matches current obs_config at start.py time.
    """
    import json
    import os
    import subprocess
    import time

    from control.utils.run_state import RunStateManager

    RunStateManager().clear_state()

    from control.utils.paths import PanoPaths
    obs_cfg_path = PanoPaths.config_dir() / "obs_config.json"
    with open(obs_cfg_path) as f:
        original_obs = json.load(f)
    
    run_name = "sc040_test_fixed_run.pffd"
    
    try:
        # Start start.py in background
        wrapper_name = "tmp_slow_start_obs.py"
        with open(wrapper_name, "w") as f:
            f.write(f"""
import asyncio

import json
import time
import sys
import os
import pathlib
import control.start as start
import unittest.mock
from control.utils import config_file

async def slow_start():
    obs = config_file.get_obs_config()
    daq = config_file.get_daq_config()
    # Path is tricky in Part 5, we use a fixed one for testing
    daq.head_node_data_dir = "/tmp/head_data"
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
        from control.utils.pydantic_config_models import QuaboUids
        uids = QuaboUids(domes=[])
    data = config_file.get_data_config()
    net = config_file.get_network_config()

    run_name = "{run_name}"

    print("CONFIGS_LOADED", flush=True)
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
    import unittest.mock
    try:
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
                no_hv=True, no_redis=True, no_data=False, force_reset=True, strict=False
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
        proc = subprocess.Popen(["python3", wrapper_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        
        # 1. Wait for marker
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

        # 2. Modify obs_config on disk (Timing mode change)
        modified_obs = dict(original_obs)
        modified_obs["domes"][0]["modules"][0]["timing_mode"] = "gnss"
        with open(obs_cfg_path, "w") as f:
            json.dump(modified_obs, f)
            
        while True:
            line = proc.stdout.readline()
            if not line: break
            print(f"HELPER: {line.strip()}", flush=True)
        
        proc.wait()
        
        # Verify that the run directory contains the ORIGINAL obs_config
        head_data_dir = pathlib.Path('/tmp') / "head_data"
        run_dir = head_data_dir / run_name
        if not os.path.exists(run_dir):
             run_dir = head_data_dir / "_aborted" / run_name
             
        if not os.path.exists(run_dir):
            print("--- RUN DIR NOT FOUND ---")
            print("-------------------------")

        assert os.path.exists(run_dir), f"Run dir not found for {run_name}"

        with open(f"{run_dir}/obs_config.json") as f:
            copied_obs = json.load(f)
            
        assert copied_obs["domes"][0]["modules"][0]["timing_mode"] == original_obs["domes"][0]["modules"][0]["timing_mode"], \
            "FAIL (SC-040): obs_config.json in run dir matches modified disk version, not original in-memory version."
            
    finally:
        with open(obs_cfg_path, "w") as f:
            json.dump(original_obs, f)
        if os.path.exists("tmp_slow_start_obs.py"):
            os.remove("tmp_slow_start_obs.py")
        if 'proc' in locals():
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass


# ── SC-015: Stale ledger self-heal ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_SC015_stale_ledger_self_heal(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    tmp_path: pathlib.Path,
) -> None:
    """
    SC-015: Ensure start.py can recover if a previous run crashed violently
    and left an ACTIVE state in the TOML ledger.
    """
    import os
    import socket
    from datetime import UTC, datetime

    import control.start as start
    from control.utils import config_file, util
    from control.utils.pydantic_config_models import RunStateLedger
    from control.utils.run_state import RunStateManager

    # 1. Clear state and inject a STALE ledger
    mgr = RunStateManager()
    mgr.clear_state()
    
    # Find a dead PID
    dead_pid = 99999
    while True:
        try:
            os.kill(dead_pid, 0)
            dead_pid -= 1
        except OSError:
            break

    stale_ledger = RunStateLedger(
        run_name="stale_run_to_be_archived.pffd",
        status="ACTIVE",
        start_time=datetime.now(UTC).isoformat(),
        pid=dead_pid,
        host=socket.gethostname()
    )
    mgr.save_state(stale_ledger)

    # 2. Run start.py (should self-heal and proceed)
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    # Mock head node IP to match this container
    from ipaddress import IPv4Address
    daq_config.head_node_ip_addr = IPv4Address(f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5')
    daq_config.head_node_data_dir = str(tmp_path / "head_data")
    
    # Filter for nodes that actually exist in integration
    reachable_ips = [IPv4Address("192.168.0.10"), IPv4Address("192.168.0.11")]
    daq_config.daq_nodes = [n for n in daq_config.daq_nodes if n.ip_addr in reachable_ips]

    from control.utils.pydantic_config_models import QuaboUids
    mock_uids = QuaboUids(domes=[])

    import unittest.mock
    with unittest.mock.patch("control.utils.config_file.get_quabo_uids", return_value=mock_uids), \
         unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
         unittest.mock.patch("control.start._check_quabo_reachability"), \
         unittest.mock.patch("control.start._check_daq_reachability"), \
         unittest.mock.patch("control.start.start_data_flow"), \
         unittest.mock.patch("control.start.start_recording"), \
         unittest.mock.patch("control.start.make_run_dirs"), \
         unittest.mock.patch("control.utils.config_file.associate"), \
         unittest.mock.patch("control.utils.config_file.show_daq_assignments"):

        # Reload to ensure mocks are used
        quabo_uids = config_file.get_quabo_uids()
        data_config = config_file.get_data_config()
        network_config = config_file.get_network_config()
        util.attach_daq_config(daq_config, network_config)

        # We expect this to succeed now because self-heal logic is in start.py
        success = await start.start_run(
                obs_config, daq_config, quabo_uids, data_config, network_config,
                no_hv=True, no_redis=True, no_data=False, force_reset=True, strict=False
            )
    
    assert success, "start.py failed to self-heal and start a new run (SC-015)"
    
    # 3. Verify archiving
    aborted_root = pathlib.Path(daq_config.head_node_data_dir) / "_aborted"
    archived_ledger = aborted_root / "stale_run_to_be_archived.pffd" / "stale_run_state.toml"
    assert archived_ledger.exists(), "Stale ledger was not archived to _aborted/"

    # Cleanup
    mgr.clear_state()
    if archived_ledger.parent.exists():
        import shutil
        shutil.rmtree(archived_ledger.parent)

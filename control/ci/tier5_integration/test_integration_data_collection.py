"""
test_integration_data_collection.py — Tier 5 Heavy Integration tests for data collection.

Transaction invariant:
    CleanupData on the DAQ node MUST only run after data has been
    successfully copied to the head node. If the copy fails, the data
    MUST be preserved on the DAQ node for retry.
"""
from __future__ import annotations

import os
import pathlib
import time

from ci.tier3_fleet.conftest import (
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)
from ci.tier4_chaos.conftest import (
    _cleanup as grpc_cleanup,
)
from ci.tier4_chaos.conftest import (
    _start as grpc_start,
)


def _prepare_host_dirs(params: dict, create_run_dir: bool = True) -> None:
    """
    Data Injection:
    Create dummy files on the host using DAQ_DATA_DIR.
    """
    host_data_root = os.environ.get("DAQ_DATA_DIR")
    if not host_data_root:
        return
    host_root = pathlib.Path(host_data_root)
    
    run_dir = params["run_dir"]
    for mid in params["module_id"]:
        mod_root = host_root / f"module_{mid}"
        mod_root.mkdir(parents=True, exist_ok=True)
        os.chmod(mod_root, 0o777)

        if create_run_dir:
            d = mod_root / run_dir
            d.mkdir(parents=True, exist_ok=True)
            dummy_file = d / "data.pff"
            dummy_file.write_bytes(b"synthetic data")

            # Recursive chmod 0o777
            for root, dirs, files in os.walk(mod_root):
                os.chmod(root, 0o777)
                for dr in dirs:
                    os.chmod(os.path.join(root, dr), 0o777)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o777)
            
    # Root run dir for validator
    main_dir = host_root / run_dir
    main_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(main_dir, 0o777)


def _wait_for_data(run_params: dict, timeout: float = 10.0) -> bool:
    """Wait for dummy data to be visible on the host."""
    host_data_root = os.environ.get("DAQ_DATA_DIR")
    if not host_data_root:
        return False
        
    src_root = pathlib.Path(host_data_root)
    run_dir  = run_params["run_dir"]
    
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(
            (src_root / f"module_{mid}" / run_dir).exists()
            for mid in run_params["module_id"]
        ):
            return True
        time.sleep(0.5)
    return False


class TestIntegrationDataCollection:
    """Happy-path and sad-path collection + cleanup scenarios requiring real environment."""

    def test_successful_copy_then_cleanup(
        self, daq_control_direct, run_params, head_data_dir
    ) -> None:
        """Standard sequence: Start -> Record -> Stop -> Copy -> Cleanup."""
        params = dict(run_params)
        params["data_dir"] = "/data" 
        
        # 1. Start recording
        ok, _ = grpc_start(daq_control_direct, params)
        assert ok
        assert wait_hashpipe_running(daq_control_direct, "/data")

        # 2. Simulate hashpipe creating data
        _prepare_host_dirs(params)
        assert _wait_for_data(params)

        # 3. Stop recording
        daq_control_direct.StopDaq({"data_dir": "/data", "run_dir": params["run_dir"]})
        assert wait_hashpipe_stopped(daq_control_direct, "/data")

        # 4. Simulate head-node copying data
        from ci.tier3_fleet.test_transfer_daemon_e2e import copy_run_dir
        assert copy_run_dir(params, pathlib.Path(head_data_dir))

        # 5. Cleanup MUST succeed now for all modules
        for mid in params["module_id"]:
            ok, msg = grpc_cleanup(daq_control_direct, {
                "data_dir":  "/data",
                "run_dir":   params["run_dir"],
                "module_id": [mid],
            })
            assert ok, f"Cleanup failed for module {mid}: {msg}"

        # 6. Verify data is gone from DAQ node (on host)
        host_root = pathlib.Path(os.environ["DAQ_DATA_DIR"])
        for mid in params["module_id"]:
            assert not (host_root / f"module_{mid}" / params["run_dir"]).exists()

    def test_cleanup_blocked_while_hashpipe_running(
        self, daq_control_direct, run_params
    ) -> None:
        """CleanupData MUST fail if hashpipe is still active for that run."""
        params = dict(run_params)
        params["data_dir"] = "/data"
        
        grpc_start(daq_control_direct, params)
        _prepare_host_dirs(params)
        assert wait_hashpipe_running(daq_control_direct, "/data")

        # Cleanup MUST fail
        try:
            for mid in params["module_id"]:
                ok, msg = grpc_cleanup(daq_control_direct, {
                    "data_dir":  "/data",
                    "run_dir":   params["run_dir"],
                    "module_id": [mid],
                })
                assert not ok
                assert "alive" in msg.lower()
        finally:
            daq_control_direct.StopDaq({"data_dir": "/data", "run_dir": params["run_dir"]})

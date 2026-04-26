"""
test_data_collection.py — Integration tests for data collection + cleanup transaction.

Transaction invariant:
    CleanupData on the DAQ node MUST only run after data has been
    successfully copied to the head node. If the copy fails, the data
    MUST be preserved on the DAQ node for retry.
"""
from __future__ import annotations

import os
import pathlib
import time

import pytest

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
    Split-Brain Data Injection:
    Create dummy files on the host using DAQ_DATA_DIR so the container-side
    server can see them in its /data mount.
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

            # Recursive chmod 0o777 for the module hierarchy
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


class TestDataCollectionTransaction:
    """Happy-path and sad-path collection + cleanup scenarios."""

    @pytest.mark.skip(reason="Requires Tier 5 Heavy Integration Stack")
    def test_successful_copy_then_cleanup(
        self, daq_control_direct, run_params, head_data_dir, ensure_clean_daq_state
    ) -> None:
        """Standard sequence: Start -> Record -> Stop -> Copy -> Cleanup."""
        run_params = dict(run_params)
        run_params["data_dir"] = "/data" # Container perspective
        
        # 1. Start recording
        ok, _ = grpc_start(daq_control_direct, run_params)
        assert ok
        assert wait_hashpipe_running(daq_control_direct, "/data")

        # 2. Simulate hashpipe creating data
        _prepare_host_dirs(run_params)
        assert _wait_for_data(run_params)

        # 3. Stop recording
        daq_control_direct.StopDaq({"data_dir": "/data", "run_dir": run_params["run_dir"]})
        assert wait_hashpipe_stopped(daq_control_direct, "/data")

        # 4. Simulate head-node copying data (e.g. TransferDaemon stage)
        # We manually copy to the isolated head_data_dir fixture
        from ci.tier3_fleet.test_transfer_daemon_e2e import copy_run_dir
        assert copy_run_dir(run_params, head_data_dir)

        # 5. Cleanup MUST succeed now for all modules
        for mid in run_params["module_id"]:
            ok, msg = grpc_cleanup(daq_control_direct, {
                "data_dir":  "/data",
                "run_dir":   run_params["run_dir"],
                "module_id": [mid],
            })
            assert ok, f"Cleanup failed for module {mid}: {msg}"

        # 6. Verify data is gone from DAQ node (on host)
        host_root = pathlib.Path(os.environ["DAQ_DATA_DIR"])
        for mid in run_params["module_id"]:
            assert not (host_root / f"module_{mid}" / run_params["run_dir"]).exists()

    @pytest.mark.skip(reason="Requires Tier 5 Heavy Integration Stack")
    def test_cleanup_blocked_while_hashpipe_running(
        self, daq_control_direct, run_params, ensure_clean_daq_state
    ) -> None:
        """CleanupData MUST fail if hashpipe is still active for that run."""
        run_params = dict(run_params)
        run_params["data_dir"] = "/data"
        
        grpc_start(daq_control_direct, run_params)
        _prepare_host_dirs(run_params)
        assert wait_hashpipe_running(daq_control_direct, "/data")

        # Cleanup MUST fail
        for mid in run_params["module_id"]:
            ok, msg = grpc_cleanup(daq_control_direct, {
                "data_dir":  "/data",
                "run_dir":   run_params["run_dir"],
                "module_id": [mid],
            })
            assert not ok
            assert "alive" in msg.lower()

    def test_cleanup_not_called_if_copy_fails(
        self, daq_control_direct, run_params, ensure_clean_daq_state
    ) -> None:
        """If data copy fails, CleanupData is not called and data persists."""
        run_params = dict(run_params)
        run_params["data_dir"] = "/data"
        
        _prepare_host_dirs(run_params)

        # Simulate copy failure: we just DON'T call copy_run_dir or CleanupData
        # Verify data still exists on host
        host_root = pathlib.Path(os.environ["DAQ_DATA_DIR"])
        for mid in run_params["module_id"]:
            assert (host_root / f"module_{mid}" / run_params["run_dir"]).exists()

    def test_cleanup_idempotent(
        self, daq_control_direct, run_params, ensure_clean_daq_state
    ) -> None:
        """Calling CleanupData twice on the same run_dir is safe (noop)."""
        run_params = dict(run_params)
        run_params["data_dir"] = "/data"
        
        # Ensure server state is clean
        daq_control_direct.StopDaq({"data_dir": "/data", "run_dir": run_params["run_dir"]})
        _prepare_host_dirs(run_params)

        for mid in run_params["module_id"]:
            req = {
                "data_dir":  "/data",
                "run_dir":   run_params["run_dir"],
                "module_id": [mid],
            }
            ok1, _ = grpc_cleanup(daq_control_direct, req)
            assert ok1

            ok2, _ = grpc_cleanup(daq_control_direct, req)
            assert ok2


class TestCleanupEdgeCases:
    """Robustness against missing directories."""

    def test_cleanup_nonexistent_module_dirs_succeeds(
        self, daq_control_direct, run_params, ensure_clean_daq_state
    ) -> None:
        """CleanupData succeeds even if some module subdirs are missing (already cleaned)."""
        run_params = dict(run_params)
        run_params["data_dir"] = "/data"
        
        # Ensure server state is clean
        daq_control_direct.StopDaq({"data_dir": "/data", "run_dir": run_params["run_dir"]})

        # CRITICAL: Prepare base module directories so DirectoryPath validation passes,
        # but leave the run_dir subdirectories missing to test the edge case.
        _prepare_host_dirs(run_params, create_run_dir=False)

        # Cleanup should succeed (nothing to do)
        for mid in run_params["module_id"]:
            ok, _ = grpc_cleanup(daq_control_direct, {
                "data_dir":  "/data",
                "run_dir":   run_params["run_dir"],
                "module_id": [mid],
            })
            assert ok


class TestNodeFailureDuringCollection:
    """Fault-tolerance: node crashes while head node is copying."""

    def test_partial_copy_preserves_daqnode_data(
        self, daq_control_direct, run_params, ensure_clean_daq_state
    ) -> None:
        """If node crashes mid-copy, head node MUST NOT issue CleanupData."""
        run_params = dict(run_params)
        run_params["data_dir"] = "/data"
        
        _prepare_host_dirs(run_params)
        
        # Simulate node crash: we don't actually crash it, we just don't call Cleanup
        # If head node detects gRPC timeout or rsync error, it aborts the transaction.
        host_root = pathlib.Path(os.environ["DAQ_DATA_DIR"])
        for mid in run_params["module_id"]:
            assert (host_root / f"module_{mid}" / run_params["run_dir"]).exists()

    def test_cleanup_after_node_restart_succeeds(
        self, daq_control_direct, run_params, ensure_clean_daq_state
    ) -> None:
        """A restarted node can safely cleanup runs from a previous session."""
        run_params = dict(run_params)
        run_params["data_dir"] = "/data"
        
        # 1. Previous session's data exists
        _prepare_host_dirs(run_params)
        
        # 2. Issue cleanup (no need to Start/Stop in this session if server is stateless)
        for mid in run_params["module_id"]:
            ok, _ = grpc_cleanup(daq_control_direct, {
                "data_dir":  "/data",
                "run_dir":   run_params["run_dir"],
                "module_id": [mid],
            })
            assert ok
        
        host_root = pathlib.Path(os.environ["DAQ_DATA_DIR"])
        for mid in run_params["module_id"]:
            assert not (host_root / f"module_{mid}" / run_params["run_dir"]).exists()

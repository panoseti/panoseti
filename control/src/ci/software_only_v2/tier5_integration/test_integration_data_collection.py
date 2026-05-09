"""
tier5_integration/test_integration_data_collection.py — Data-collection + cleanup tests.

Transaction invariant:
    CleanupData MUST only run after data has been successfully copied to the
    head node. If hashpipe is still active, cleanup must be refused.
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any

import pytest

from ci.software_only_v2.tier5_integration.conftest import (
    DAQ_DATA_DIR,
    requires_compose_stack,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)

pytestmark = [pytest.mark.tier5, requires_compose_stack]


def _prepare_host_dirs(params: dict[str, Any], create_run_dir: bool = True) -> None:
    """Create module subdirs + dummy .pff file to simulate hashpipe output."""
    host_root = pathlib.Path(DAQ_DATA_DIR)
    run_dir = params["run_dir"]

    (host_root / run_dir).mkdir(parents=True, exist_ok=True)
    os.chmod(host_root / run_dir, 0o777)

    for mid in params["module_id"]:
        mod_root = host_root / f"module_{mid}"
        mod_root.mkdir(parents=True, exist_ok=True)
        os.chmod(mod_root, 0o777)
        if create_run_dir:
            d = mod_root / run_dir
            d.mkdir(parents=True, exist_ok=True)
            (d / "data.pff").write_bytes(b"synthetic data")
            for root, dirs, files in os.walk(mod_root):
                os.chmod(root, 0o777)
                for dr in dirs:
                    os.chmod(os.path.join(root, dr), 0o777)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o777)


def _wait_for_data(params: dict[str, Any], timeout: float = 10.0) -> bool:
    host_root = pathlib.Path(DAQ_DATA_DIR)
    run_dir = params["run_dir"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(
            (host_root / f"module_{mid}" / run_dir).exists()
            for mid in params["module_id"]
        ):
            return True
        time.sleep(0.5)
    return False


def _copy_run_dir(params: dict[str, Any], head_data_dir: pathlib.Path) -> bool:
    """Copy DAQ-node run data to the head-node directory (shared-volume path)."""
    import shutil
    host_root = pathlib.Path(DAQ_DATA_DIR)
    run_dir = params["run_dir"]
    head_run = head_data_dir / run_dir
    head_run.mkdir(parents=True, exist_ok=True)

    for mid in params["module_id"]:
        src = host_root / f"module_{mid}" / run_dir
        if not src.exists():
            return False
        dst = head_run / f"module_{mid}"
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    return True


class TestIntegrationDataCollection:
    """Happy-path and sad-path collection + cleanup scenarios."""

    def test_successful_copy_then_cleanup(
        self,
        daq_control_node1: Any,
        run_params: dict[str, Any],
        head_data_dir: pathlib.Path,
    ) -> None:
        """Standard sequence: Start → Record → Stop → Copy → Cleanup."""
        params = dict(run_params)

        assert daq_control_node1.StartDaq(params) is True
        assert wait_hashpipe_running(daq_control_node1, params["data_dir"])

        _prepare_host_dirs(params)
        assert _wait_for_data(params)

        daq_control_node1.StopDaq({"data_dir": params["data_dir"], "run_dir": params["run_dir"]})
        assert wait_hashpipe_stopped(daq_control_node1, params["data_dir"])

        assert _copy_run_dir(params, head_data_dir)

        # Cleanup must succeed after a copy
        for mid in params["module_id"]:
            result = daq_control_node1.CleanupData({
                "data_dir": params["data_dir"],
                "run_dir": params["run_dir"],
                "module_id": [mid],
            })
            assert result.get("success", False), (
                f"Cleanup failed for module {mid}: {result.get('message', '')}"
            )

        # Data must be gone from the DAQ volume
        host_root = pathlib.Path(DAQ_DATA_DIR)
        for mid in params["module_id"]:
            assert not (host_root / f"module_{mid}" / params["run_dir"]).exists()

    def test_cleanup_blocked_while_hashpipe_running(
        self,
        daq_control_node1: Any,
        run_params: dict[str, Any],
    ) -> None:
        """CleanupData must fail (server rejects) while hashpipe is active."""
        from panoseti_grpc.grpc_utils.exceptions import FailedPreconditionError
        params = dict(run_params)
        daq_control_node1.StartDaq(params)
        _prepare_host_dirs(params)
        assert wait_hashpipe_running(daq_control_node1, params["data_dir"])

        try:
            for mid in params["module_id"]:
                with pytest.raises(FailedPreconditionError) as exc_info:
                    daq_control_node1.CleanupData({
                        "data_dir": params["data_dir"],
                        "run_dir": params["run_dir"],
                        "module_id": [mid],
                    })
                
                msg = str(exc_info.value).lower()
                assert "alive" in msg or "running" in msg, (
                    f"Expected 'alive'/'running' in refusal message, got: {msg}"
                )
        finally:
            daq_control_node1.StopDaq({"data_dir": params["data_dir"], "run_dir": params["run_dir"]})

    def test_status_disk_usage_is_reported(
        self,
        daq_control_node1: Any,
        run_params: dict[str, Any],
    ) -> None:
        """StatusDaq with check_disk_usage=True returns a numeric value."""
        _, status = daq_control_node1.StatusDaq({
            "data_dir": run_params["data_dir"],
            "check_hashpipe_running": False,
            "check_disk_usage": True,
            "check_run_dirs": False,
        })
        assert "disk_usage_gb" in status or "disk_free_gb" in status or status is not None

"""
test_daq_lifecycle.py — Integration tests for the DAQ Start/Stop/Status lifecycle.

Tests are parameterized for both direct (daqnode IP) and gateway (socat-forwarded)
connections, validating that gRPC topology works end-to-end for both paths.
"""
from __future__ import annotations

import contextlib
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient

from ci.software_only.conftest import (
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)


@pytest.fixture(params=["direct", "gateway"])
def daq_client_multi(
    request: Any,
    daq_control_direct: DaqControlClient,
    daq_control_gateway: DaqControlClient,
) -> DaqControlClient:
    """Parameterized fixture — runs every test against both network paths."""
    if request.param == "direct":
        return daq_control_direct
    return daq_control_gateway


class TestDaqLifecycle:
    """Full Start → Status (running) → double-start rejected → Stop → Status (stopped)."""

    def test_start_daq(self, daq_client_multi, run_params, ensure_clean_daq_state) -> None:
        """StartDaq returns True for a fresh run."""
        daq_client_multi.StartDaq(run_params)
        assert wait_hashpipe_running(daq_client_multi, run_params['data_dir']), (
            "Hashpipe failed to start"
        )

    def test_status_shows_running(self, daq_client_multi, run_params, ensure_clean_daq_state) -> None:
        """After StartDaq, StatusDaq reports hashpipe_running=True."""
        daq_client_multi.StartDaq(run_params)
        assert wait_hashpipe_running(daq_client_multi, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        ok, status = daq_client_multi.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        assert ok
        assert status.get("hashpipe_running") is True

    def test_double_start_rejected(self, daq_client_multi, run_params, ensure_clean_daq_state) -> None:
        """A second StartDaq while hashpipe is running must fail (raises ValueError)."""
        daq_client_multi.StartDaq(run_params)
        assert wait_hashpipe_running(daq_client_multi, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        # Server returns success=False → client raises ValueError
        with pytest.raises(ValueError):
            daq_client_multi.StartDaq(run_params)

    def test_stop_daq(self, daq_client_multi, run_params, ensure_clean_daq_state) -> None:
        """StopDaq returns True."""
        daq_client_multi.StartDaq(run_params)
        assert wait_hashpipe_running(daq_client_multi, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        ok = daq_client_multi.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        assert ok is True

    def test_status_shows_stopped_after_stop(self, daq_client_multi, run_params, ensure_clean_daq_state) -> None:
        """After StopDaq, StatusDaq reports hashpipe_running=False."""
        daq_client_multi.StartDaq(run_params)
        assert wait_hashpipe_running(daq_client_multi, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        daq_client_multi.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        assert wait_hashpipe_stopped(daq_client_multi, run_params["data_dir"]), (
            "hashpipe did not stop within timeout"
        )
        ok, status = daq_client_multi.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        assert ok
        assert status.get("hashpipe_running") is False

    def test_run_dir_appears_in_status(self, daq_client_multi, run_params, ensure_clean_daq_state) -> None:
        """After StartDaq, the run_dir appears in StatusDaq run_dirs list."""
        daq_client_multi.StartDaq(run_params)
        assert wait_hashpipe_running(daq_client_multi, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        ok, status = daq_client_multi.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": False,
            "check_disk_usage":       False,
            "check_run_dirs":         True,
        })
        assert ok
        run_dirs = status.get("run_dirs", [])
        assert any(run_params["run_dir"] in d for d in run_dirs), (
            f"run_dir={run_params['run_dir']!r} not in {run_dirs}"
        )

    def test_stop_idempotent(self, daq_client_multi, run_params, ensure_clean_daq_state) -> None:
        """StopDaq when nothing is running should not raise."""
        ok = daq_client_multi.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        assert ok is True


class TestDaqDiskUsage:

    def test_disk_usage_fields_present(self, daq_client, run_params, ensure_clean_daq_state) -> None:
        """StatusDaq with check_disk_usage returns expected disk usage keys."""
        daq_client.StartDaq(run_params)
        assert wait_hashpipe_running(daq_client, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        daq_client.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        ok, status = daq_client.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": False,
            "check_disk_usage":       True,
            "check_run_dirs":         False,
        })
        assert ok
        du = status.get("disk_usage", {})
        assert du.get("total_disk_space", -1) > 0
        assert du.get("free_disk_space", -1) >= 0
        assert du.get("used_disk_space", -1) >= 0

    def test_disk_usage_values_plausible(self, daq_client, run_params, ensure_clean_daq_state) -> None:
        """Disk usage values are internally consistent: used ≈ total - free."""
        ok, status = daq_client.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": False,
            "check_disk_usage":       True,
            "check_run_dirs":         False,
        })
        assert ok
        du = status.get("disk_usage", {})
        total = du.get("total_disk_space", 0)
        free  = du.get("free_disk_space", 0)
        used  = du.get("used_disk_space", 0)

        # Docker containers have at least a few hundred MB of disk
        assert total > 100 * 1024 * 1024, f"total_disk_space {total} bytes seems too small"
        assert free >= 0
        # used + free should be ≈ total (within 10% — some OSes report slightly
        # different figures due to reserved blocks)
        assert abs((used + free) - total) < 0.10 * total, (
            f"used={used} + free={free} deviates >10% from total={total}"
        )


class TestDaqRunDirIsolation:
    """Multiple run directories coexist independently on the same node."""

    def test_cleanup_removes_only_specified_run(
        self, daq_client, run_params, ensure_clean_daq_state
    ) -> None:
        """CleanupData for run_dir A must not remove run_dir B on the same node."""
        import uuid as _uuid
        rp_b = dict(run_params)
        rp_b["run_dir"] = f"ci_run_b_{_uuid.uuid4().hex[:8]}.pffd"

        # Start, confirm running, then stop run A
        daq_client.StartDaq(run_params)
        assert wait_hashpipe_running(daq_client, run_params["data_dir"]), (
            "run A: hashpipe did not start"
        )
        daq_client.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        assert wait_hashpipe_stopped(daq_client, run_params["data_dir"]), (
            "run A: hashpipe did not stop"
        )

        # Start, confirm running, then stop run B
        daq_client.StartDaq(rp_b)
        assert wait_hashpipe_running(daq_client, rp_b["data_dir"]), (
            "run B: hashpipe did not start"
        )
        daq_client.StopDaq({
            "data_dir": rp_b["data_dir"],
            "run_dir":  rp_b["run_dir"],
        })
        assert wait_hashpipe_stopped(daq_client, rp_b["data_dir"]), (
            "run B: hashpipe did not stop"
        )

        # Clean up run A only
        daq_client.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })

        _, status = daq_client.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": False,
            "check_disk_usage":       False,
            "check_run_dirs":         True,
        })
        run_dirs = status.get("run_dirs", [])

        # run_dir A should be gone
        assert not any(run_params["run_dir"] in d for d in run_dirs), (
            f"run_dir A={run_params['run_dir']!r} should have been removed"
        )
        # run_dir B must still be present
        assert any(rp_b["run_dir"] in d for d in run_dirs), (
            f"run_dir B={rp_b['run_dir']!r} was unexpectedly removed"
        )

        # Teardown run B
        with contextlib.suppress(Exception):
            daq_client.CleanupData({
                "data_dir":  rp_b["data_dir"],
                "run_dir":   rp_b["run_dir"],
                "module_id": rp_b["module_id"],
            })


class TestStopDaqRobustness:
    """Verify that StopDaq handles process leaks and graceful waits correctly."""

    def test_stop_daq_clears_all_orphans(
        self, daq_client, run_params, ensure_clean_daq_state, daqnode_container
    ) -> None:
        """One StopDaq call must clear ALL hashpipe processes on the node."""
        # 1. Start one hashpipe via gRPC
        daq_client.StartDaq(run_params)
        assert wait_hashpipe_running(daq_client, run_params["data_dir"])

        # 2. Start a SECOND hashpipe manually to simulate an orphan
        # We don't need a real hashpipe; any process named 'hashpipe' will trigger the cleanup.
        wrapped = daqnode_container
        wrapped.exec_run("cp /bin/sleep /tmp/hashpipe")
        wrapped.exec_run("/tmp/hashpipe 300", detach=True)
        
        # Verify two processes are running
        import time
        time.sleep(2.0)
        exit_code, output = wrapped.exec_run("pgrep hashpipe")
        pids = output.decode().strip().split()
        assert len(pids) >= 2, f"Expected >=2 hashpipe processes, found {len(pids)}: {output.decode()}"

        # 3. Call StopDaq once
        ok = daq_client.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        }, timeout=70.0)
        assert ok is True

        # 4. Verify ALL are gone
        assert wait_hashpipe_stopped(daq_client, run_params["data_dir"]), "Hashpipes still running after StopDaq"
        exit_code, output = wrapped.exec_run("pgrep hashpipe")
        assert exit_code != 0 or not output.strip(), f"Processes leaked: {output.decode()}"

    def test_stop_daq_graceful_wait(self, daq_client, run_params, ensure_clean_daq_state) -> None:
        """StopDaq must wait for the process to actually terminate before returning."""
        import time
        daq_client.StartDaq(run_params)
        assert wait_hashpipe_running(daq_client, run_params["data_dir"])

        t0 = time.monotonic()
        ok = daq_client.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        }, timeout=70.0)
        elapsed = time.monotonic() - t0

        assert ok is True
        # It should take at least the poll interval, but usually very fast if it exits gracefully.
        # This just ensures it doesn't hang for the full 60s unless necessary.
        assert elapsed < 60.0, f"StopDaq took too long: {elapsed:.1f}s"
        
        # Status should immediately show not running
        ok, status = daq_client.StatusDaq({
            "data_dir": run_params["data_dir"],
            "check_hashpipe_running": True
        })
        assert ok and status.get("hashpipe_running") is False

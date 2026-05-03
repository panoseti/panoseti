"""
test_concurrent_daq_operations.py — Concurrent and rapid-cycle DAQ operation tests.

Tests:
  - Only one of N concurrent StartDaq calls succeeds (server serializes via asyncio).
  - Concurrent StatusDaq calls during an active run all succeed.
  - CleanupData is blocked while hashpipe is running; succeeds after StopDaq.
  - Repeated Start→Stop cycles don't leave the server in a broken state.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient

from ci.software_only.conftest import wait_hashpipe_running, wait_hashpipe_stopped
from ci.software_only.tier3_fleet.conftest import BINDHOST, DAQ_DATA_DIR

# ---------------------------------------------------------------------------
# Extra run_params fixture for concurrent tests (distinct module + run_dir
# from the default, so the autouse ensure_clean_daq_state fixture doesn't
# interfere while these tests manage state themselves)
# ---------------------------------------------------------------------------

@pytest.fixture
def run_params_conc(session_fleet) -> dict[str, Any]:
    """Run parameters for concurrent tests — unique per test invocation."""
    fleet, _ = session_fleet
    
    return {
        "data_dir":         DAQ_DATA_DIR,
        "daq_ip_addr":      fleet.node_ip(0),
        "bindhost":         BINDHOST,
        "max_file_size_mb": 1,
        "group_ph_frames":  False,
        "run_dir":          f"ci_conc_{uuid.uuid4().hex[:8]}.pffd",
        "obs":              "citest",
        "module_id":        [200],
    }


@pytest.fixture
def ensure_clean_daq_state_conc(daq_control_direct: DaqControlClient, run_params_conc: dict[str, Any]) -> Iterator[None]:
    """Stop hashpipe and clean up if a concurrent test leaves it running."""
    yield
    with contextlib.suppress(Exception):
        daq_control_direct.StopDaq({
            "data_dir": run_params_conc["data_dir"],
            "run_dir":  run_params_conc["run_dir"],
        })
    wait_hashpipe_stopped(daq_control_direct, run_params_conc["data_dir"], timeout=8)
    with contextlib.suppress(Exception):
        daq_control_direct.CleanupData({
            "data_dir":  run_params_conc["data_dir"],
            "run_dir":   run_params_conc["run_dir"],
            "module_id": run_params_conc["module_id"],
        })

class TestConcurrentDaqOperations:
    """Server must serialise concurrent StartDaq requests (asyncio event loop)."""

    def test_concurrent_start_only_one_wins(
        self, daq_control_direct, run_params_conc, ensure_clean_daq_state_conc, session_fleet
    ) -> None:
        """Three simultaneous StartDaq calls: exactly one returns True, rest raise ValueError.

        The gRPC server uses an asyncio event loop, so concurrent client calls
        are processed sequentially on the server.  The first call sets
        hashpipe_pid; subsequent calls see pid > 0 and return success=False,
        which the client converts to ValueError.
        """
        from random import uniform
        import time
        rp = run_params_conc
        fleet, _ = session_fleet
        spec = fleet.specs[0]

        def attempt():
            # Each thread needs its own gRPC channel to actually be concurrent.
            # Use the fleet's dynamic host/port (not DAQNODE_DIRECT_HOST which
            # is only set in the Docker CI environment).
            client = DaqControlClient(host=spec.container_host_ip, port=spec.mapped_port)
            try:
                time.sleep(uniform(0.05, 0.75))
                return client.StartDaq(rp)   # True on success
            except ValueError:
                return False                  # Server rejected duplicate start

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(attempt) for _ in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        successes = [r for r in results if r is True]
        failures  = [r for r in results if r is False]

        assert len(successes) == 1, f"Expected exactly 1 success, got {successes}"
        assert len(failures) == 2, f"Expected exactly 2 failures, got {failures}"

    def test_concurrent_status_all_succeed(
        self, daq_control_direct, run_params_conc, ensure_clean_daq_state_conc
    ) -> None:
        """Ten concurrent StatusDaq calls while hashpipe is running → all succeed."""
        rp = run_params_conc
        daq_control_direct.StartDaq(rp)
        assert wait_hashpipe_running(daq_control_direct, rp["data_dir"]), (
            "hashpipe did not start within timeout"
        )

        status_params = {
            "data_dir":               rp["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        }

        def poll_status():
            ok, _status = daq_control_direct.StatusDaq(status_params)
            return ok

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(poll_status) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(results), f"Some concurrent StatusDaq calls failed: {results}"

    def test_cleanup_blocked_while_running_then_succeeds(
        self, daq_control_direct, run_params_conc, ensure_clean_daq_state_conc
    ) -> None:
        """CleanupData is blocked while hashpipe is running; succeeds after StopDaq."""
        rp = run_params_conc
        daq_control_direct.StartDaq(rp)
        assert wait_hashpipe_running(daq_control_direct, rp["data_dir"]), (
            "hashpipe did not start within timeout"
        )

        # CleanupData should be blocked while hashpipe is live
        cleanup_resp = daq_control_direct.CleanupData({
            "data_dir":  rp["data_dir"],
            "run_dir":   rp["run_dir"],
            "module_id": rp["module_id"],
        })
        assert cleanup_resp["success"] is False
        assert "HASHPIPE is still alive" in cleanup_resp["message"]

        # After stop, cleanup should succeed (or gracefully no-op)
        daq_control_direct.StopDaq({
            "data_dir": rp["data_dir"],
            "run_dir":  rp["run_dir"],
        })
        assert wait_hashpipe_stopped(daq_control_direct, rp["data_dir"]), (
            "hashpipe did not stop within timeout"
        )

        cleanup_resp2 = daq_control_direct.CleanupData({
            "data_dir":  rp["data_dir"],
            "run_dir":   rp["run_dir"],
            "module_id": rp["module_id"],
        })
        assert cleanup_resp2["success"] is True

    def test_rapid_start_stop_cycles(
        self, daq_control_direct, run_params_conc, ensure_clean_daq_state_conc
    ) -> None:
        """Five rapid Start→Stop cycles complete without server state corruption."""
        rp = run_params_conc
        for cycle in range(5):
            # Use a distinct run_dir per cycle so CleanupData doesn't conflict
            rp["run_dir"] = f"ci_rapid_{uuid.uuid4().hex[:8]}.pffd"

            ok = daq_control_direct.StartDaq(rp)
            assert ok is True, f"Cycle {cycle}: StartDaq returned {ok!r}"

            # Wait for hashpipe to be confirmed running before stopping
            assert wait_hashpipe_running(daq_control_direct, rp["data_dir"]), (
                f"Cycle {cycle}: hashpipe did not start within timeout"
            )

            stop_ok = daq_control_direct.StopDaq({
                "data_dir": rp["data_dir"],
                "run_dir":  rp["run_dir"],
            })
            assert stop_ok is True, f"Cycle {cycle}: StopDaq returned {stop_ok!r}"

            # Wait for hashpipe to fully exit before the next StartDaq.
            # Fixed-delay sleep is not sufficient: the process may still be in the
            # teardown path when the next StartDaq arrives, leaving a stale pid > 0
            # on the server that rejects the new start.
            assert wait_hashpipe_stopped(daq_control_direct, rp["data_dir"]), (
                f"Cycle {cycle}: hashpipe still running after StopDaq"
            )

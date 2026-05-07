"""
test_concurrent_daq_operations.py — Concurrent and rapid-cycle DAQ tests.

Ported from ci/software_only/tier3_fleet/test_concurrent_daq_operations.py.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import pytest

from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.orchestrator.fleet import Fleet
from ci.software_only_v2.tier3_fleet.conftest import make_startdaq_params, requires_docker

pytestmark = pytest.mark.tier3


def wait_until(
    condition: Any,
    timeout: float = 10.0,
    interval: float = 0.2,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if condition():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


@requires_docker
@pytest.mark.parametrize(
    "pseti_workspace_session",
    [FleetSpec.minimal_fleet()],
    indirect=True,
)
class TestConcurrentDaqOperations:
    """Server must serialize concurrent StartDaq requests."""

    def test_when_three_concurrent_starts_issued_then_exactly_one_wins(
        self, session_fleet: Fleet
    ) -> None:
        """Three simultaneous StartDaq calls: exactly one returns True."""
        fleet = session_fleet

        fleet.exec_in_node(0, "mkdir -p /data/conc_run && chmod 777 /data/conc_run")

        def attempt() -> bool:
            client = fleet.daq_control_client(0)
            try:
                time.sleep(0.1)
                return client.StartDaq(make_startdaq_params(fleet, 0, "conc_run"))
            except ValueError:
                return False
            finally:
                client.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(attempt) for _ in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        successes = [r for r in results if r is True]
        assert len(successes) == 1

    def test_when_ten_concurrent_status_calls_issued_then_all_succeed(
        self, session_fleet: Fleet
    ) -> None:
        """Ten concurrent StatusDaq calls while hashpipe is running → all succeed."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        fleet.exec_in_node(0, "mkdir -p /data/conc_status && chmod 777 /data/conc_status")
        client.StartDaq(make_startdaq_params(fleet, 0, "conc_status"))

        def poll_status() -> bool:
            c = fleet.daq_control_client(0)
            ok, _ = c.StatusDaq({"data_dir": "/data"})
            c.close()
            return ok

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(poll_status) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(results)
        client.close()

    def test_when_cleanup_called_while_running_then_blocked_then_succeeds_after_stop(
        self, session_fleet: Fleet
    ) -> None:
        """CleanupData is blocked while hashpipe runs; succeeds after StopDaq."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        fleet.exec_in_node(0, "mkdir -p /data/cleanup_block && chmod 777 /data/cleanup_block")
        client.StartDaq(make_startdaq_params(fleet, 0, "cleanup_block"))

        from panoseti_grpc.grpc_utils.exceptions import FailedPreconditionError
        node_cfg = fleet.live_daq_config.daq_nodes[0]
        cleanup_params = {
            "data_dir": "/data",
            "run_dir": "cleanup_block",
            "module_id": list(node_cfg.module_ids),
        }
        try:
            client.CleanupData(cleanup_params)
        except FailedPreconditionError as exc:
            fleet._last_cleanup_exc = exc  # type: ignore[attr-defined]

        from ci.software_only_v2.infra.parity import run_scenario
        run_scenario("cleanup_blocked_while_hashpipe_running", fleet=fleet, node_index=0)

        client.StopDaq({"data_dir": "/data", "run_dir": "cleanup_block"})

        def check_stopped() -> bool:
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return not bool(s["hashpipe_running"])

        assert wait_until(check_stopped)

        res = client.CleanupData(cleanup_params)
        assert res["success"] is True
        client.close()

    def test_when_rapid_start_stop_cycles_repeated_then_all_complete(
        self, session_fleet: Fleet
    ) -> None:
        """Three rapid Start→Stop cycles all complete cleanly."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        for i in range(3):
            run_dir = f"rapid_{i}"
            fleet.exec_in_node(0, f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}")

            assert client.StartDaq(make_startdaq_params(fleet, 0, run_dir)) is True

            def check_running() -> bool:
                _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
                return bool(s["hashpipe_running"])

            assert wait_until(check_running)

            assert client.StopDaq({"data_dir": "/data", "run_dir": run_dir}) is True

            def check_stopped() -> bool:
                _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
                return not bool(s["hashpipe_running"])

            assert wait_until(check_stopped)
        client.close()

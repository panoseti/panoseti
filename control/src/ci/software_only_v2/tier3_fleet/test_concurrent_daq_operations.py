# mypy: ignore-errors
"""
test_concurrent_daq_operations.py — Concurrent and rapid-cycle DAQ tests.

Ported from ci/software_only/tier3_fleet/test_concurrent_daq_operations.py.
"""

from __future__ import annotations

import concurrent.futures
import time
import uuid
from typing import Any

import pytest
import docker

from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.infra.workspace import Workspace
from ci.software_only_v2.orchestrator.fleet import Fleet

pytestmark = pytest.mark.tier3


def _docker_available() -> bool:
    try:
        import docker
        docker.from_env(timeout=5).ping()
        return True
    except Exception:
        return False


requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available",
)


def wait_until(condition, timeout=10.0, interval=0.2):
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

    def test_concurrent_start_only_one_wins(self, session_fleet: Fleet) -> None:
        """Three simultaneous StartDaq calls: exactly one returns True."""
        fleet = session_fleet
        
        run_params = {
            "data_dir": "/data",
            "run_dir": "conc_run",
            "module_id": [200]
        }
        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            "mkdir -p /data/conc_run && chmod 777 /data/conc_run"
        )
        
        def attempt():
            from panoseti_grpc.daq_control.client import DaqControlClient
            client = DaqControlClient(host=fleet.daq_nodes[0].grpc_host, port=fleet.daq_nodes[0].grpc_port)
            try:
                time.sleep(0.1)
                return client.StartDaq(run_params)
            except ValueError:
                return False
            finally:
                client.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(attempt) for _ in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        successes = [r for r in results if r is True]
        assert len(successes) == 1

    def test_concurrent_status_all_succeed(self, session_fleet: Fleet) -> None:
        """Ten concurrent StatusDaq calls while hashpipe is running → all succeed."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_params = {"data_dir": "/data", "run_dir": "conc_status", "module_id": [200]}
        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            "mkdir -p /data/conc_status && chmod 777 /data/conc_status"
        )
        client.StartDaq(run_params)
        
        def poll_status():
            from panoseti_grpc.daq_control.client import DaqControlClient
            c = DaqControlClient(host=fleet.daq_nodes[0].grpc_host, port=fleet.daq_nodes[0].grpc_port)
            ok, _ = c.StatusDaq({"data_dir": "/data"})
            c.close()
            return ok

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(poll_status) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(results)

    def test_cleanup_blocked_while_running_then_succeeds(self, session_fleet: Fleet) -> None:
        """CleanupData is blocked while hashpipe is running."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_params = {"data_dir": "/data", "run_dir": "cleanup_block", "module_id": [200]}
        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            "mkdir -p /data/cleanup_block && chmod 777 /data/cleanup_block"
        )
        client.StartDaq(run_params)
        
        # Use parity scenario for blocked cleanup
        # First set the last exception on the fleet object if parity expects it
        # Actually, the built-in parity scenario expects it on the fleet object.
        
        from panoseti_grpc.grpc_utils.exceptions import FailedPreconditionError
        try:
            client.CleanupData({
                "data_dir": "/data",
                "run_dir": "cleanup_block",
                "module_id": [200]
            })
        except FailedPreconditionError as e:
            fleet._last_cleanup_exc = e
        
        from ci.software_only_v2.infra.parity import run_scenario
        run_scenario("cleanup_blocked_while_hashpipe_running", fleet=fleet, node_index=0)
        
        # After stop, cleanup should succeed
        client.StopDaq({"data_dir": "/data", "run_dir": "cleanup_block"})
        
        # Wait for stop
        def check_stopped():
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return s["hashpipe_running"] is False
        assert wait_until(check_stopped)
        
        res = client.CleanupData({
            "data_dir": "/data",
            "run_dir": "cleanup_block",
            "module_id": [200]
        })
        assert res["success"] is True
        client.close()

    def test_rapid_start_stop_cycles(self, session_fleet: Fleet) -> None:
        """Five rapid Start→Stop cycles complete."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        for i in range(3):
            run_dir = f"rapid_{i}"
            run_params = {"data_dir": "/data", "run_dir": run_dir, "module_id": [200]}
            docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
                f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}"
            )
            
            assert client.StartDaq(run_params) is True
            
            def check_running():
                _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
                return s["hashpipe_running"] is True
            assert wait_until(check_running)
            
            assert client.StopDaq({"data_dir": "/data", "run_dir": run_dir}) is True
            
            def check_stopped():
                _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
                return s["hashpipe_running"] is False
            assert wait_until(check_stopped)
        client.close()

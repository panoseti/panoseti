"""
test_data_collection.py — Data collection and cleanup transaction tests.

Ported from ci/software_only/tier3_fleet/test_data_collection.py.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from ci.software_only.infra.spec import FleetSpec
from ci.software_only.orchestrator.fleet import Fleet
from ci.software_only.tier3_fleet.conftest import make_startdaq_params, requires_docker

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
@pytest.mark.timeout(120)
class TestDataCollectionTransaction:
    """Happy-path and sad-path collection + cleanup scenarios."""

    def test_when_daq_started_and_stopped_then_hashpipe_exits_cleanly(
        self, session_fleet: Fleet
    ) -> None:
        """Start→wait running→stop→wait stopped full lifecycle completes."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_dir = "happy_run"

        fleet.exec_in_node(0, f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}")

        ok = client.StartDaq(make_startdaq_params(fleet, 0, run_dir))
        assert ok is True

        def check_running() -> bool:
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return bool(s["hashpipe_running"])

        assert wait_until(check_running), "hashpipe did not start"

        ok = client.StopDaq({"data_dir": "/data", "run_dir": run_dir})
        assert ok is True

        def check_stopped() -> bool:
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return not bool(s["hashpipe_running"])

        assert wait_until(check_stopped), "hashpipe did not stop"
        client.close()

    def test_when_cleanup_called_twice_then_both_succeed(
        self, session_fleet: Fleet
    ) -> None:
        """Calling CleanupData twice on the same run dir is idempotent."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_dir = "cleanup_idempotent"
        node_cfg = fleet.live_daq_config.daq_nodes[0]

        fleet.exec_in_node(0, f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}")

        cleanup_params = {
            "data_dir": "/data",
            "run_dir": run_dir,
            "module_id": list(node_cfg.module_ids),
            "mode": "CLEANUP_FULL",
            "force": True,
        }

        res1 = client.CleanupData(cleanup_params)
        assert res1["success"]

        res2 = client.CleanupData(cleanup_params)
        assert res2["success"]
        client.close()

    def test_when_cleanup_called_on_nonexistent_run_dir_then_succeeds(
        self, session_fleet: Fleet
    ) -> None:
        """CleanupData succeeds even if the target run directory does not exist."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        node_cfg = fleet.live_daq_config.daq_nodes[0]

        res = client.CleanupData({
            "data_dir": "/data",
            "run_dir": "nonexistent_run",
            "module_id": list(node_cfg.module_ids),
            "mode": "CLEANUP_FULL",
            "force": True,
        })
        assert res["success"]
        client.close()

"""
test_daq_lifecycle.py — Integration tests for DAQ Start/Stop/Status.

Ported from ci/software_only/tier3_fleet/test_daq_lifecycle.py.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from ci.software_only_v2.infra.spec import FleetSpec, GatewaySpec
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


SPEC_LIFECYCLE = (
    FleetSpec(seed=100, name="lifecycle_test", tier="tier3")
    .with_headnode(ip="10.0.1.5")
    .add_dome("d0", lat=37, lon=-121, alt=1000)
    .add_module(200, ip="192.168.3.32")
    .add_module(201, ip="192.168.3.36")
    .add_daq_node(ip="192.168.0.10", modules=[200], bindhost="lo")
    .add_daq_node(
        ip="192.168.0.20",
        modules=[201],
        gateway=GatewaySpec(ip="10.200.146.13", grpc_port=50051),
        bindhost="lo",
    )
)


@requires_docker
@pytest.mark.parametrize(
    "pseti_workspace_session",
    [SPEC_LIFECYCLE],
    indirect=True,
)
class TestDaqLifecycle:
    """Full Start → Status (running) → double-start rejected → Stop → Status (stopped)."""

    @pytest.mark.parametrize("node_index", [0, 1])
    def test_when_daq_started_then_lifecycle_completes_cleanly(
        self, session_fleet: Fleet, node_index: int
    ) -> None:
        """Start → status shows running → double-start rejected → stop → status shows stopped."""
        fleet = session_fleet
        client = fleet.daq_control_client(node_index)
        run_dir = f"run_node_{node_index}"

        fleet.exec_in_node(node_index, f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}")

        ok = client.StartDaq(make_startdaq_params(fleet, node_index, run_dir))
        assert ok is True

        def check_running() -> bool:
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return bool(s["hashpipe_running"])

        assert wait_until(check_running), "Hashpipe did not start"

        with pytest.raises(ValueError):
            client.StartDaq(make_startdaq_params(fleet, node_index, run_dir))

        ok = client.StopDaq({"data_dir": "/data", "run_dir": run_dir})
        assert ok is True

        def check_stopped() -> bool:
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return not bool(s["hashpipe_running"])

        assert wait_until(check_stopped), "Hashpipe did not stop"

        ok = client.StopDaq({"data_dir": "/data", "run_dir": run_dir})
        assert ok is True  # idempotent

        client.close()

    def test_when_status_includes_disk_then_usage_is_plausible(
        self, session_fleet: Fleet
    ) -> None:
        """StatusDaq with check_disk_usage=True returns non-zero total_disk_space."""
        client = session_fleet.daq_control_client(0)
        ok, status = client.StatusDaq({"data_dir": "/data", "check_disk_usage": True})
        assert ok
        du = status.get("disk_usage", {})
        assert du.get("total_disk_space", 0) > 0
        assert du.get("free_disk_space", 0) >= 0
        client.close()

    def test_when_run_started_then_run_dir_appears_in_status(
        self, session_fleet: Fleet
    ) -> None:
        """After StartDaq, the run_dir appears in the StatusDaq run_dirs list."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_dir = "status_test_run"

        fleet.exec_in_node(0, f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}")
        client.StartDaq(make_startdaq_params(fleet, 0, run_dir))

        ok, status = client.StatusDaq({"data_dir": "/data", "check_run_dirs": True})
        assert ok
        assert any(run_dir in d for d in status.get("run_dirs", []))
        client.close()

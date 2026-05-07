# mypy: ignore-errors
"""
test_daq_lifecycle.py — Integration tests for DAQ Start/Stop/Status.

Ported from ci/software_only/tier3_fleet/test_daq_lifecycle.py.
"""

from __future__ import annotations

import time
import pytest
import docker

from ci.software_only_v2.infra.spec import FleetSpec, GatewaySpec
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
        bindhost="lo"
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
    def test_daq_lifecycle_full(self, session_fleet: Fleet, node_index: int) -> None:
        fleet = session_fleet
        topology = fleet.workspace.topology
        client = fleet.daq_control_client(node_index)

        run_params = {
            "data_dir": "/data",
            "run_dir": f"run_node_{node_index}",
            "module_id": list(topology.daq.daq_nodes[node_index].module_ids)
        }

        # Prepare dir in container
        docker.from_env().containers.get(fleet.daq_nodes[node_index].name).exec_run(
            f"mkdir -p /data/{run_params['run_dir']} && chmod 777 /data/{run_params['run_dir']}"
        )

        # 1. Start
        ok = client.StartDaq(run_params)
        assert ok is True

        # 2. Status shows running
        def check_running():
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return s["hashpipe_running"] is True
        assert wait_until(check_running), "Hashpipe did not start"

        # 3. Double-start rejected
        with pytest.raises(ValueError):
            client.StartDaq(run_params)

        # 4. Stop
        ok = client.StopDaq({"data_dir": "/data", "run_dir": run_params["run_dir"]})
        assert ok is True

        # 5. Status shows stopped
        def check_stopped():
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return s["hashpipe_running"] is False
        assert wait_until(check_stopped), "Hashpipe did not stop"

        # 6. Stop idempotent
        ok = client.StopDaq({"data_dir": "/data", "run_dir": run_params["run_dir"]})
        assert ok is True

        client.close()

    def test_daq_disk_usage(self, session_fleet: Fleet) -> None:
        """StatusDaq returns plausible disk usage."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        ok, status = client.StatusDaq({"data_dir": "/data", "check_disk_usage": True})
        assert ok
        du = status.get("disk_usage", {})
        assert du.get("total_disk_space", 0) > 0
        assert du.get("free_disk_space", 0) >= 0
        client.close()

    def test_run_dir_appears_in_status(self, session_fleet: Fleet) -> None:
        """After StartDaq, the run_dir appears in StatusDaq run_dirs list."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_dir = "status_test_run"
        run_params = {
            "data_dir": "/data",
            "run_dir": run_dir,
            "module_id": [200]
        }
        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}"
        )
        client.StartDaq(run_params)

        ok, status = client.StatusDaq({"data_dir": "/data", "check_run_dirs": True})
        assert ok
        assert any(run_dir in d for d in status.get("run_dirs", []))
        client.close()

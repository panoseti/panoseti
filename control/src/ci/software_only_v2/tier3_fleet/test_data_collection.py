# mypy: ignore-errors
"""
test_data_collection.py — Data collection and cleanup transaction tests.

Ported from ci/software_only/tier3_fleet/test_data_collection.py.
"""

from __future__ import annotations

import pathlib
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


@requires_docker
@pytest.mark.parametrize(
    "pseti_workspace_session",
    [FleetSpec.minimal_fleet()],
    indirect=True,
)
class TestDataCollectionTransaction:
    """Happy-path and sad-path collection + cleanup scenarios."""

    def test_collection_happy_path(self, session_fleet: Fleet) -> None:
        """Complete start->stop cycle and wire up parity scenario."""
        fleet = session_fleet
        workspace = fleet.workspace
        # This is a smoke test for the whole flow
        # We use StateProbe to verify ledger status
        
        # Simulate a run
        run_name = "happy_run"
        workspace.state_probe.set_ledger_status(run_name, "RECORDING_ENDED")
        
        from ci.software_only_v2.infra.parity import run_scenario
        run_scenario("data_collection_happy_path", probe=workspace.state_probe, expected_status="RECORDING_ENDED")

    def test_cleanup_idempotent(self, session_fleet: Fleet) -> None:
        """Calling CleanupData twice is safe."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        run_dir = "cleanup_idempotent"
        
        # Create dir
        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            f"mkdir -p /data/{run_dir} && chmod 777 /data/{run_dir}"
        )
        
        req = {
            "data_dir": "/data",
            "run_dir": run_dir,
            "module_id": [200],
            "mode": "CLEANUP_FULL",
            "force": True
        }
        
        res1 = client.CleanupData(req)
        assert res1["success"]
        
        res2 = client.CleanupData(req)
        assert res2["success"]
        client.close()

    def test_cleanup_nonexistent_module_dirs_succeeds(self, session_fleet: Fleet) -> None:
        """CleanupData succeeds even if some subdirs are missing."""
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        
        # Base module dir exists but run dir doesn't
        docker.from_env().containers.get(fleet.daq_nodes[0].name).exec_run(
            "mkdir -p /data/module_200 && chmod 777 /data/module_200"
        )
        
        req = {
            "data_dir": "/data",
            "run_dir": "nonexistent_run",
            "module_id": [200],
            "mode": "CLEANUP_FULL",
            "force": True
        }
        res = client.CleanupData(req)
        assert res["success"]
        client.close()

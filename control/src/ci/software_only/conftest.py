"""
conftest.py — v2 test suite root configuration.

Extends the shared fixtures from ci/conftest.py with v2-specific isolation.
Loads only v2 fixture modules; does NOT re-load v1 fixture modules (though
v1 modules are still available from the parent ci/conftest.py).
"""

from __future__ import annotations

import contextlib
import os
import uuid
from typing import Any

import pytest

pytest_plugins = [
    "ci.software_only.fixtures.workspace",
    "ci.software_only.fixtures.fleet",
    "ci.software_only.fixtures.chaos",
    "ci.software_only.fixtures.corpus",
    "ci.fixtures.rsync_fixtures",
    "ci.fixtures.transfer_fixtures",
]


def pytest_configure(config: Any) -> None:
    """
    Apply v2-specific environment isolation before any test collection.

    Sets PSETI_* env vars to safe test defaults if not already overridden.
    The per-test pseti_workspace fixture will override these with truly isolated
    per-test tmp_path values; this configure just prevents accidentally hitting
    production paths during collection.
    """
    # Give each xdist worker a unique TC_SESSION_ID to prevent container name
    # collisions across parallel workers.
    if hasattr(config, "workerinput"):
        worker_id = config.workerinput.get("workerid", "master")
        run_uuid = config.workerinput.get("tc_run_uuid", uuid.uuid4().hex[:8])
    else:
        worker_id = "solo"
        run_uuid = uuid.uuid4().hex[:8]

    os.environ.setdefault("TC_SESSION_ID", f"tc-v2-{worker_id}-{run_uuid}")

    # Isolation defaults (not production paths)
    os.environ.setdefault("PSETI_TMP", "/tmp/pseti_v2_test/tmp")
    os.environ.setdefault("PSETI_LOGS", "/tmp/pseti_v2_test/logs")
    os.environ.setdefault("PSETI_QUABOS", "/tmp/pseti_v2_test/quabos")

    for d in ["PSETI_TMP", "PSETI_LOGS", "PSETI_QUABOS"]:
        os.makedirs(os.environ[d], exist_ok=True)

    # Per-worker telemetry isolation (same pattern as v1)
    try:
        db_index = int("".join(filter(str.isdigit, worker_id))) if worker_id not in ("master", "solo") else 0
    except ValueError:
        db_index = 0
    os.environ.setdefault("REDIS_DB", str(db_index))
    os.environ.setdefault("LOKI_TENANT_ID", f"v2_test_tenant_{db_index}")


def pytest_unconfigure(config: Any) -> None:
    """Final cleanup after all tests finish."""
    # Only prune if we are the master process (or solo) and NOT inside a container
    if not hasattr(config, "workerinput") and not os.path.exists("/.dockerenv"):
        try:
            import docker
            client = docker.from_env()

            # 1. Aggressively kill and remove any pseti-v2 containers
            # This ensures that SharedNetwork removal doesn't fail with "active endpoints"
            # if a test crashed or a fixture failed to tear down.
            container_patterns = ["pseti-v2-"]
            for container in client.containers.list(all=True):
                if any(p in container.name for p in container_patterns):
                    with contextlib.suppress(Exception):
                        container.stop(timeout=2)
                        container.remove(force=True, v=True)

            # 2. Prune any pseti-v2 networks left behind
            network_patterns = ["pseti-v2-tc-", "pseti-v2-shared-net"]
            for network in client.networks.list():
                if any(p in network.name for p in network_patterns):
                    with contextlib.suppress(Exception):
                        network.remove()
        except Exception:
            # If docker is not available (e.g. inside a restricted container), just skip
            pass


@pytest.fixture(scope="session")
def worker_id(request: Any) -> str:
    """Returns the xdist worker ID, or 'master' for single-process runs."""
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]
    return "master"


@pytest.fixture(autouse=True)
def clear_shared_state(request: pytest.FixtureRequest) -> None:
    """Clear shared state between tests (ledger, transfer queue, and running processes)."""
    import shutil

    from control.utils.paths import PanoPaths
    from control.utils.run_state import RunStateManager

    # 1. Clear ledger
    with contextlib.suppress(Exception):
        RunStateManager().clear_state()

    # 2. Clear transfer queue
    with contextlib.suppress(Exception):
        q_dir = PanoPaths.transfer_queue_dir()
        if q_dir.exists():
            for d in q_dir.iterdir():
                if d.is_dir():
                    shutil.rmtree(d)

    # 3. Stop all DAQ processes if a fleet is active
    if "session_fleet" in request.fixturenames:
        with contextlib.suppress(Exception):
            fleet = request.getfixturevalue("session_fleet")
            for i in range(fleet.n_nodes):
                client = fleet.daq_control_client(i)
                # StopDaq on our server implementation kills all instances by name
                client.StopDaq({"data_dir": "/data", "run_dir": "reset"})
                client.close()
    
    if "chaos_fleet" in request.fixturenames:
        with contextlib.suppress(Exception):
            fleet = request.getfixturevalue("chaos_fleet")
            for i in range(fleet.n_nodes):
                client = fleet.daq_control_client(i)
                client.StopDaq({"data_dir": "/data", "run_dir": "reset"})
                client.close()



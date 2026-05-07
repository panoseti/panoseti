"""
conftest.py — v2 test suite root configuration.

Extends the shared fixtures from ci/conftest.py with v2-specific isolation.
Loads only v2 fixture modules; does NOT re-load v1 fixture modules (though
v1 modules are still available from the parent ci/conftest.py).
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

pytest_plugins = [
    "ci.software_only_v2.fixtures.workspace",
    "ci.software_only_v2.fixtures.fleet",
    "ci.software_only_v2.fixtures.chaos",
    # corpus, state_probe, clients — added in later phases
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


@pytest.fixture(scope="session")
def worker_id(request: Any) -> str:
    """Returns the xdist worker ID, or 'master' for single-process runs."""
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]
    return "master"

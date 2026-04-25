"""
ci/fixtures/conftest.py

Primary entry point for shared fixtures across all dependency tiers.
Includes the mandatory auto_isolate fixture.
"""

from __future__ import annotations

import os
import pathlib
import shutil
from typing import Any, Iterator

import pytest

from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager

@pytest.fixture(scope="session")
def worker_id(request: Any) -> str:
    """Returns the xdist worker ID or 'master' if not running in parallel."""
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]
    return "master"

@pytest.fixture(autouse=True)
def auto_isolate(
    tmp_path: pathlib.Path, 
    monkeypatch: pytest.MonkeyPatch,
    worker_id: str
) -> Iterator[pathlib.Path]:
    """
    Mandatory fixture that provides per-test isolation for configs, transient state,
    and telemetry databases.
    """
    # 1. Setup isolated directories inside tmp_path
    state_tmp = tmp_path / "state"
    ctl_tmp = tmp_path / "control"
    cfg_tmp = ctl_tmp / "configs"
    
    for d in [state_tmp, ctl_tmp, cfg_tmp]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Apply environment overrides
    monkeypatch.setenv("PSETI_STATE", str(state_tmp))
    monkeypatch.setenv("PSETI_CONTROL", str(ctl_tmp))
    monkeypatch.setenv("PSETI_CONFIG", str(cfg_tmp))
    
    # 3. Telemetry and Database Isolation
    # Assign unique Redis DBs and Loki Tenant IDs based on xdist worker_id
    try:
        db_index = int("".join(filter(str.isdigit, worker_id))) if worker_id != "master" else 0
    except ValueError:
        db_index = 0
        
    monkeypatch.setenv("REDIS_DB", str(db_index))
    monkeypatch.setenv("LOKI_TENANT_ID", f"test_tenant_{db_index}")
    
    # 4. Ensure role-segregated tree exists
    PanoPaths.ensure_state_dirs()
    RunStateManager().clear_state()
    
    yield tmp_path

# Import factories as fixtures
from .factories import make_transfer_job, simulate_daq_filesystem, make_mock_daq_config

@pytest.fixture
def transfer_job_factory():
    return make_transfer_job

@pytest.fixture
def daq_fs_simulator():
    return simulate_daq_filesystem

@pytest.fixture
def daq_config_factory():
    return make_mock_daq_config
